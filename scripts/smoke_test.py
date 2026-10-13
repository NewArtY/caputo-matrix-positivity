"""Smoke test: every experiment driver in a reduced configuration, in a working copy of the code.

Copies code/ and scripts/ to a working directory, replaces the configuration files by the reduced ones of
code/configs/quick/ and runs there every stage of every benchmark (references, short campaigns, single runs, the
timing passes), followed by the provenance check of the files written. results/raw of the archive is not touched.
The reduced configurations use tiny meshes and a few hundred optimiser iterations: the numbers are meaningless and
are not aggregated; the test shows that every driver runs to the end on the installed software.

Usage: python scripts/smoke_test.py [--workers 4] [--workdir DIR] [--keep] [--only NAME ...]
  NAME is one of: E1_E3 E5 E4 B3 B4 tail B1 B2 surrogate B6 B7 B5 timings provenance
Exit status 1 if a command fails. About 5 minutes with 4 worker processes.
"""
import argparse, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stages(w):
    """(benchmark, command) in execution order; commands run in <work>/code/experiments unless they start with 'scripts/'."""
    w = str(w)
    return [
        ('E1_E3', ['e1_e3_verification.py', 'all']),
        ('E5', ['l1_positivity.py']),
        ('E4', ['e4_soe.py', 'all']),
        ('B3', ['dirac_reference.py', 'B3']),
        ('B4', ['dirac_reference.py', 'B4']),
        ('B4', ['dirac_reference.py', 'B4check']),
        ('tail', ['dirac_tail_check.py', 'all']),
        ('B1', ['b1_floor.py']),
        ('B1', ['b1_ls_mesh.py']),
        ('B1', ['run_b1_campaign.py', 'all', w]),
        ('B1', ['b1_scalar.py', '--mode', 'forward', '--alpha', '0.5', '--input', 'singular', '--gate', 'pow', '--seed', '1']),
        ('B2', ['b2_prepare.py', w]),
        ('B2', ['run_b2_campaign.py', 'all', w]),
        ('B2', ['b2_density.py', '--alpha', '0.6', '--state', 'pure', '--map', 'gate2b', '--ic', 'hard', '--gate', 'pow', '--seed', '1']),
        ('B2', ['b2_timing.py', 'b2', '0.6', 'pure', '0']),
        ('surrogate', ['b2_surrogate.py', 'prepare', w]),
        ('surrogate', ['b2_surrogate.py', 'campaign', w]),
        ('surrogate', ['b2_timing.py', 'surrogate']),
        ('surrogate', ['b2_surrogate.py', 'train', '--map', 'lin', '--seed', '0', '--stage', 'timing']),
        ('B6', ['b6_prepare.py', w]),
        ('B6', ['b6_campaign.py', 'all', w]),
        ('B6', ['b6_positivity.py', '--system', 'qutrit', '--alpha', '0.5', '--map', 'lin', '--gate', 'pow', '--seed', '1']),
        ('B7', ['b7_inverse.py', 'prepare', w]),
        ('B7', ['b7_campaign.py', 'all', w]),
        ('B7', ['b7_inverse.py', 'pinn', '--design', 'xz', '--noise', '0.01', '--map', 'add', '--weight', '100', '--real', '5']),
        ('B7', ['b7_inverse.py', 'nls', '--design', 'xz', '--noise', '0.01', '--real', '5']),
        ('B5', ['b5_conformation.py', 'prepare']),
        ('B5', ['b5_conformation.py', 'campaign', '--workers', w]),
        ('B5', ['b5_conformation.py', 'campaign', '--workers', w, '--variants', 'sym-rel']),
        ('B5', ['b5_conformation.py', 'run', 'shear', '1', '0.6', 'rotated', 'logm', 'nat', '0', '--budget', '2', '--save']),
        ('B5', ['b5_inverse.py', 'nls']),
        ('B5', ['b5_inverse.py', 'campaign', '--workers', w]),
        ('B5', ['b5_fenep.py', 'prepare']),
        ('B5', ['b5_fenep.py', 'campaign', '--workers', w]),
        ('B5', ['b5_positivity.py']),
        ('B5', ['b5_fenep_step.py', '--quick', '--workers', w]),
        ('B5', ['b5_case_selection.py']),
        ('timings', ['scripts/rerun_timings.py', '--no-aggregate', '--max-load', '100']),
        ('provenance', ['scripts/check_provenance.py']),
    ]


def main():
    ap = argparse.ArgumentParser(description='Run every experiment driver in a reduced configuration in a working copy.')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--workdir'); ap.add_argument('--keep', action='store_true')
    ap.add_argument('--only', nargs='+')
    a = ap.parse_args()
    work = os.path.abspath(a.workdir) if a.workdir else tempfile.mkdtemp(prefix='fracsp_smoke_')
    if os.path.exists(os.path.join(work, 'code')):
        sys.exit(f'{work} already contains a copy of the code; choose an empty --workdir')
    ignore = shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc')
    shutil.copytree(os.path.join(ROOT, 'code'), os.path.join(work, 'code'), ignore=ignore)
    shutil.copytree(os.path.join(ROOT, 'scripts'), os.path.join(work, 'scripts'), ignore=ignore)
    cfg = os.path.join(work, 'code', 'configs')
    for f in sorted(os.listdir(os.path.join(cfg, 'quick'))):                 # the reduced configurations replace the full ones
        shutil.copy(os.path.join(cfg, 'quick', f), os.path.join(cfg, f))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               FRACSP_MAX_WORKERS=str(a.workers))
    print(f'working copy: {work}; {a.workers} worker processes', flush=True)
    rows = []; t_all = time.perf_counter(); os.makedirs(os.path.join(work, 'logs'), exist_ok=True)
    for k, (name, cmd) in enumerate(stages(a.workers)):
        if a.only and name not in a.only:
            continue
        script = os.path.join(work, cmd[0]) if cmd[0].startswith('scripts/') else os.path.join(work, 'code', 'experiments', cmd[0])
        cwd = work if cmd[0].startswith('scripts/') else os.path.join(work, 'code', 'experiments')
        log = os.path.join(work, 'logs', f'{k:02d}_{name}.log'); t0 = time.perf_counter()
        with open(log, 'w', encoding='utf-8') as f:
            rc = subprocess.run([sys.executable, script] + cmd[1:], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = time.perf_counter() - t0; rows.append((name, ' '.join(cmd), rc, dt))
        print(f"{'ok    ' if rc == 0 else 'FAILED'} {dt:6.1f} s  {name:10s} python {' '.join(cmd)}", flush=True)
        if rc:
            print(''.join(open(log, encoding='utf-8', errors='replace').readlines()[-25:]), flush=True)
    nres = sum(len(fs) for _, _, fs in os.walk(os.path.join(work, 'results'))) if os.path.isdir(os.path.join(work, 'results')) else 0
    nbad = sum(r[2] != 0 for r in rows)
    print(f'\n{len(rows)} commands, {nbad} failed, {nres} result files written, {time.perf_counter() - t_all:.0f} s in total')
    if a.keep or nbad:
        print(f'working copy kept: {work}')
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if nbad else 0


if __name__ == '__main__':
    sys.exit(main())

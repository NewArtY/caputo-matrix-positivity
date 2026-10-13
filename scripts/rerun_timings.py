"""Rerun the timing measurements only and update the tables and figures that depend on them.

To be run on an otherwise idle machine. Nothing but wall times is measured anew: errors, numbers of modes and trained
networks are recomputed by the same deterministic code and must equal the stored ones.

  stage      commands (in code/experiments)                                   raw results replaced                      about
  soe        e4_soe.py solver, residual, dirac; twice (run 1 and run 2)       results/raw/E4_soe/E4b_solver*.csv,       20 min
                                                                              E4c_residual*.csv, E4d_dirac_chunk*.csv
  b2         b2_timing.py b2 0.6 pure 0                                       results/raw/B2_factorial/timing/           3 min
  surrogate  b2_timing.py surrogate;                                          results/raw/B2_surrogate/timing.json,      5 min
             b2_surrogate.py train --map lin --seed 0 --stage timing          results/raw/B2_surrogate/timing/

Table entries that change: Table_SOE.csv (wall_*, setup_s, time_ratio_*), Table_SOE_matched.csv (wall_*, speedup_*),
Table_SOE_residual.csv and Table_SOE_dirac_chunk.csv (wall_*, time_ratio_*, microseconds_per_point_step),
Table_B2_timing.csv (wall_s, cpu_s, t_*), Table_surrogate.csv and Table_surrogate_breakeven.csv (inference_ms_*,
stepper_ms_per_query, break_even_*, train_wall_s of the rows 'idle repeat'), the corresponding rows of
results-manifest.csv, figures/data/FigDet_SOE_work_precision.csv, the figures FigDet_SOE and Neural_surrogate, and, of
the tables and figures of the article, tables/final/Table_S1_soe.tex (times and ratios) and panel (d) of
figures/final/Figure_05.pdf.
The wall and CPU times in the other tables (in the article: the last column of Table_07.tex and the training times in
Figure_05.pdf) were measured during the campaigns and change only if a campaign is rerun.

The files of the previous measurement are copied to a folder previous_timing_<UTC time> next to them before they are
replaced. Usage: python scripts/rerun_timings.py [--only soe|b2|surrogate ...] [--max-load 15] [--no-aggregate]
"""
import argparse, datetime, glob, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'code', 'experiments'); RAW = os.path.join(ROOT, 'results', 'raw')
sys.path.insert(0, EXP)
import det_common as dc  # noqa: E402

ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
SOE_FILES = ['E4b_solver.csv', 'E4c_residual.csv', 'E4d_dirac_chunk.csv', 'E4_solver.log', 'E4_residual.log', 'E4_dirac.log']


def run(*args):
    print('+ python ' + ' '.join(args), flush=True); t0 = time.perf_counter()
    rc = subprocess.run([sys.executable] + list(args), cwd=EXP, env=ENV).returncode
    if rc:
        sys.exit(f'FAILED (exit status {rc}): python ' + ' '.join(args))
    print(f'  done in {time.perf_counter() - t0:.0f} s', flush=True)


def keep_previous(folder, patterns):
    """Copy the files of the previous measurement to <folder>/previous_timing_<UTC time>/, keeping relative paths."""
    files = [f for p in patterns for f in sorted(glob.glob(os.path.join(folder, p))) if os.path.isfile(f)]
    if files:
        dst = os.path.join(folder, 'previous_timing_' + STAMP)
        for f in files:
            target = os.path.join(dst, os.path.relpath(f, folder)); os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(f, target)
        print(f'  previous measurement ({len(files)} files) kept in {dc.rel(dst)}', flush=True)


def with_suffix(name, suffix):
    stem, ext = os.path.splitext(name)
    return stem + suffix + ext


def soe():
    d = os.path.join(RAW, 'E4_soe')
    keep_previous(d, ['E4b_solver*.csv', 'E4c_residual*.csv', 'E4d_dirac_chunk*.csv', 'E4_solver*.log', 'E4_residual*.log', 'E4_dirac*.log'])
    for k in (1, 2):                                   # two complete, independent timing runs
        for stage in ('solver', 'residual', 'dirac'):
            run('e4_soe.py', stage)
        if k == 1:                                     # set the first run aside while the second is measured
            for f in SOE_FILES:
                os.replace(os.path.join(d, f), os.path.join(d, with_suffix(f, '_first')))
    for f in SOE_FILES:                                # second run -> *_run2, first run -> primary names
        os.replace(os.path.join(d, f), os.path.join(d, with_suffix(f, '_run2')))
        os.replace(os.path.join(d, with_suffix(f, '_first')), os.path.join(d, f))


def b2():
    keep_previous(os.path.join(RAW, 'B2_factorial'), ['timing/*'])
    run('b2_timing.py', 'b2', '0.6', 'pure', '0')


def surrogate():
    d = os.path.join(RAW, 'B2_surrogate')
    keep_previous(d, ['timing.json', 'timing/*.json'])
    run('b2_timing.py', 'surrogate')
    run('b2_surrogate.py', 'train', '--map', 'lin', '--seed', '0', '--stage', 'timing')


def main():
    ap = argparse.ArgumentParser(description='Rerun the timing measurements on an idle machine and update tables and figures.')
    ap.add_argument('--only', action='append', choices=['soe', 'b2', 'surrogate'])
    ap.add_argument('--max-load', type=float, default=15.0, help='refuse to start if the CPU load in per cent is above this value')
    ap.add_argument('--no-aggregate', action='store_true', help='do not rerun scripts/aggregate.py and scripts/make_figures.py afterwards')
    a = ap.parse_args()
    dc.disable_power_throttling()
    load = dc.system_load_percent()
    print(f'CPU load before the measurements: {load} %')
    if load is not None and load > a.max_load:
        sys.exit(f'the machine is not idle (CPU load {load} % > {a.max_load} %); close other jobs or pass a larger --max-load')
    for stage in a.only or ['soe', 'b2', 'surrogate']:
        print(f'=== {stage}', flush=True)
        dict(soe=soe, b2=b2, surrogate=surrogate)[stage]()
    print(f'CPU load after the measurements: {dc.system_load_percent()} %')
    if not a.no_aggregate:
        for script in ('aggregate.py', 'make_figures.py'):
            rc = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', script)], cwd=ROOT, env=ENV).returncode
            if rc:
                sys.exit(f'scripts/{script} FAILED (exit status {rc})')
    return 0


if __name__ == '__main__':
    sys.exit(main())

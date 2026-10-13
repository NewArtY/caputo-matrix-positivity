"""Entry point of the aggregation stage: every table and all figure data from the stored raw results.

Runs the aggregation scripts of the three groups of benchmarks one after another,
    deterministic   scripts/aggregate_deterministic.py   E1-E5, Dirac B3, B4 and tail    (results/raw/E*, DIRAC_v2)
    neural          scripts/aggregate_neural.py          B1, B2, B6, B7, surrogate       (results/raw/B1_*, B2_*, B6_*, B7_*)
    b5              scripts/aggregate_b5.py              conformation tensor B5          (results/raw/B5_conformation)
and then writes results-manifest.csv, one row per principal quantity of the summary tables:
    id            <table>:<row key>:<column>
    value         the table cell, verbatim
    table         file in tables/data
    source_file   raw result files under results/raw from which the table row is computed
    script        driver(s) in code/experiments -> aggregation script in scripts
and the tables of the article, tables/final/Table_*.tex, with scripts/make_tables.py --extra.
Nothing is computed here; the tables in tables/data are the complete record.

Usage: python scripts/aggregate.py [--only deterministic|neural|b5 ...] [--clean] [--check] [--rtol 0]
  --clean   remove the existing outputs of the selected groups first (cached figure data are then recomputed)
  --check   compare every regenerated file with the file that was present before the run and report
            identical / numerically different (largest relative difference) / missing; exit status 1 on a difference
            above --rtol. Side-car records (*.meta.json) are compared without their time stamp and environment.
The manifest and the tables of the article are written only when all groups are aggregated.
"""
import argparse, csv, fnmatch, io, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(ROOT, 'tables', 'data'); FD = os.path.join(ROOT, 'figures', 'data'); TF = os.path.join(ROOT, 'tables', 'final')
MANIFEST = os.path.join(ROOT, 'results-manifest.csv')
GROUPS = {   # group: (command, table patterns, figure-data patterns)
    'deterministic': (['aggregate_deterministic.py', 'all'], ['Table_E*', 'Table_SOE*', 'Table_Dirac_*', 'Table_L1_*'], ['FigDet_*']),
    'neural': (['aggregate_neural.py'], ['Table_B1_*', 'Table_B2_*', 'Table_B6_*', 'Table_B7_*', 'Table_surrogate*'], ['Neural_*']),
    'b5': (['aggregate_b5.py'], ['Table_B5_*'], ['B5_*']),
}
VOLATILE = ('timestamp_utc', 'command', 'python', 'numpy', 'scipy', 'torch', 'mpmath', 'fracsp', 'blas', 'platform', 'machine', 'cpu',
            'logical_cpus', 'ram_gb', 'blas_threads', 'source_sha256_16', 'files')
DET, NEU, B5 = 'aggregate_deterministic.py', 'aggregate_neural.py', 'aggregate_b5.py'
PAIR = ['n_pairs', 'wins_a', 'gmean_ratio_a_over_b', 'p_wilcoxon']
ACC = ['rel_fro_median', 'rel_fro_mean', 'rel_fro_sd', 'max_fro_median', 'n_runs_negative_eig', 'median_over_stepper']
EST = [f'{p}_{q}' for p in ('alpha', 'Gdown', 'Gphi') for q in ('bias', 'sd', 'rmse', 'crb_sd')]
# Quantities listed in results-manifest.csv: (table, key columns, value columns or '*' = all others, raw source under results/raw,
# driver in code/experiments, aggregation script in scripts, row filter {column: admitted values} or None). The manifest holds the
# principal quantities of the summary tables; convergence histories, checks of the error estimates, per-configuration and per-run
# values are complete in tables/data and figures/data.
QUANTITIES = [
    ('Table_E1_E3_summary', ['experiment', 'function', 'alpha', 'mesh', 'quantity'], ['value_at_N_max', 'order_last_pair', 'order_min_all_pairs', 'order_max_all_pairs'],
     'E1_E3/E1_l1_operator.csv; E1_E3/E1T_truncation.csv; E1_E3/E3_mittag_leffler.csv', 'e1_e3_verification.py', DET, None),
    ('Table_E2_alpha_derivative', ['mesh', 'grading_r', 'alpha', 'N'], ['rel_err_autograd_vs_mpmath', 'rel_err_closed_form_vs_mpmath', 'rel_err_fd_vs_mpmath'],
     'E1_E3/E2_alpha_gradient.csv', 'e1_e3_verification.py', DET, None),
    ('Table_Examples_continuous', ['example', 'quantity'], ['value'], 'none (closed forms)', 'none', DET, None),
    ('Table_L1_positivity', ['system', 'd', 'alpha', 'mesh'],
     ['n_trajectories', 'min_eigenvalue', 'max_trace_defect', 'min_choi_eig_solution_map', 'min_eigenvalue_soe', 'max_bloch_norm_minus_1'],
     'E5_positivity/E5_l1_positivity.csv; DIRAC_v2/B3_summary.csv; DIRAC_v2/B4_*.npz', 'l1_positivity.py; dirac_reference.py', DET, None),
    ('Table_SOE', ['solver', 'N'],
     ['M', 'soe_rel_kernel_err', 'wall_median_s', 'err_max_all_nodes', 'err_max_t_ge_T64', 'err_final_time', 'reldiff_to_full_same_mesh', 'peak_heap_MB',
      'time_ratio_full_over_this', 'wall_median_run2_s', 'time_ratio_full_over_this_run2'],
     'E4_soe/E4b_solver.csv; E4_soe/E4b_solver_run2.csv; E4_soe/E4b_memory.csv', 'e4_soe.py', DET, None),
    ('Table_SOE_matched', ['error_measure', 'reference', 'solver'], ['target_error', 'N_needed', 'wall_needed_s', 'speedup_vs_reference', 'speedup_vs_reference_best'],
     'E4_soe/E4b_solver.csv; E4_soe/E4b_solver_run2.csv', 'e4_soe.py', DET, None),
    ('Table_SOE_modes', ['alpha'], '*', 'E4_soe/E4a_kernel.csv', 'e4_soe.py', DET, None),
    ('Table_SOE_kernel', ['alpha', 'T', 'N', 'criterion'], ['M', 'max_rel_err', 'rel_err_over_tol'], 'E4_soe/E4a_kernel_windows.csv', 'e4_soe.py', DET, {'window': {'used'}}),
    ('Table_SOE_residual', ['N', 'mode'], ['M', 'wall_median_s', 'reldiff_to_full', 'time_ratio_full_over_soe'], 'E4_soe/E4c_residual.csv', 'e4_soe.py', DET, None),
    ('Table_SOE_dirac_chunk', ['solver', 'N'], ['M', 'wall_median_s', 'absdiff_final_bloch_to_full', 'time_ratio_full_over_soe'], 'E4_soe/E4d_dirac_chunk.csv',
     'e4_soe.py', DET, None),
    ('Table_Dirac_B3', ['a0', 'alpha'],
     ['nc_rep', 'err_est', 'nc_PI_richardson', 'abs_diff_L1_PI', 'order_L1_finest', 'order_PI_finest', 'nc_L1_N16384', 'max_abs_full_minus_soe',
      'soe_rel_kernel_err', 'max_coherence', 'min_purity', 'min_eigenvalue'], 'DIRAC_v2/B3_summary.csv', 'dirac_reference.py', DET, None),
    ('Table_Dirac_B4', ['zeta', 'alpha', 'kappa_c'],
     ['N_K_rep', 'N_Kp_rep', 'N_tot_rep', 'dN_rep', 'asym_rep', 'peak_Jx_band_rep', 'peak_Jy_band_rep', 't_peak_Jx_band', 'Jx_intra_at_peak', 'Jx_inter_at_peak',
      'peak_Jx_sea', 'peak_Jy_sea'], 'DIRAC_v2/B4_*.npz; DIRAC_v2/B4check_*.npz', 'dirac_reference.py', DET, {'kappa_c': {'3.0', '8.0', '32.0', 'inf'}}),
    ('Table_Dirac_B4_fits', ['zeta', 'alpha', 'quantity'],
     ['limit', 'limit_err', 'coefficient', 'remainder_at_kc3', 'remainder_at_kc8', 'B_fit_rms_over_t', 'B_closed_form_rms_over_t', 'B_closed_form_at_peak',
      'B_fit_minus_closed_form_rms', 'B_fit4_minus_closed_form_rms', 'B_fit_rel_diff_to_closed_form', 'B_fit4_rel_diff_to_closed_form'],
     'DIRAC_v2/B4_*.npz; DIRAC_v2/current_cutoff_coefficient.csv',
     'dirac_reference.py; dirac_cutoff_coefficient.py', DET, None),
    ('Table_Dirac_B4_runs', ['zeta', 'alpha'], '*', 'DIRAC_v2/B4_*.npz', 'dirac_reference.py', DET, None),
    ('Table_Dirac_tail', ['study', 'zeta', 'phi_deg', 'xi', 'alpha', 'case', 'K', 't_f', 'quantity'], ['value', 'rep', 'analytic', 'ratio_to_analytic'],
     'DIRAC_v2/tail_*.csv; DIRAC_v2/B4_*.npz', 'dirac_tail_check.py; dirac_reference.py', DET, None),
    ('Table_B2_factorial', ['stage', 'alpha', 'state', 'cell'], ACC, 'B2_factorial/{main,extras,long}/*.json; B2_factorial/reference/l1_stepper.csv',
     'run_b2_campaign.py; b2_prepare.py', NEU, None),
    ('Table_B2_marginal', ['alpha', 'map', 'ic_gate'], ['gmean', 'median'], 'B2_factorial/main/*.json', 'run_b2_campaign.py', NEU, None),
    ('Table_B2_paired', ['comparison', 'alpha', 'a', 'b'], PAIR, 'B2_factorial/{main,extras,long}/*.json', 'run_b2_campaign.py', NEU,
     {'metric': {'rel_fro'}, 'states': {'all'}}),
    ('Table_B2_budget', ['alpha', 'state', 'cell'], ['rel_fro_median_default', 'rel_fro_median_long', 'ratio_long_over_default', 'l1_stepper_rel_fro_N128',
                                                    'budget_factor'],
     'B2_factorial/{main,long}/*.json', 'run_b2_campaign.py', NEU, None),
    ('Table_B2_soft_ic_pilot', ['alpha', 'map', 'ic_weight'], ['gmean', 'seeds'], 'B2_factorial/pilot/*.json', 'run_b2_campaign.py', NEU, None),
    ('Table_B2_timing', ['cell', 'alpha', 'state', 'seed'], ['wall_s', 'cpu_s', 'identical_to_stored'], 'B2_factorial/timing/timing.csv', 'b2_timing.py', NEU, None),
    ('Table_B6_accuracy', ['stage', 'system', 'alpha', 'cell'], ACC + ['rel_fro_proj_median', 'adam_iters', 'lbfgs_iters'],
     'B6_positivity/{main,long}/*.json; B6_positivity/reference/l1_stepper.csv',
     'b6_campaign.py; b6_prepare.py', NEU, None),
    ('Table_B6_positivity', ['stage', 'system', 'alpha', 'cell'],
     ['n', 'runs_with_negative_eig', 'most_negative_eig', 'median_fraction_of_time_negative', 'rel_fro_median', 'rel_fro_after_projection_median',
      'exact_max_smallest_eig'], 'B6_positivity/{main,long}/*.json', 'b6_campaign.py', NEU, None),
    ('Table_B6_paired', ['stage', 'group', 'a', 'b'], PAIR, 'B6_positivity/{main,long}/*.json', 'b6_campaign.py', NEU, {'metric': {'rel_fro'}}),
    ('Table_B7_estimates', ['design', 'noise_sd', 'method'], ['n_ok'] + EST + ['n_alpha_at_lower_bound', 'n_agree_with_ls', 'wall_median_s', 'runs_with_negative_eig'],
     'B7_inverse/main/*.json; B7_inverse/reference/crb.csv', 'b7_campaign.py; b7_inverse.py', NEU, None),
    ('Table_B7_start_dependence', ['method', 'design', 'noise_sd'],
     ['n_start_alpha_below_0p5', 'max_start_alpha_below_0p5', 'n_at_lower_bound_start_below_0p5', 'n_start_alpha_above_0p5', 'min_start_alpha_above_0p5',
      'n_at_lower_bound_start_above_0p5', 'n_not_at_lower_bound', 'rmse_ratio_to_ls', 'n_below_half_true_alpha_not_at_lower_bound',
      'min_alpha_hat_of_these', 'max_alpha_hat_of_these'], 'B7_inverse/main/*.json', 'b7_campaign.py; b7_inverse.py', NEU, None),
    ('Table_B7_data_weight_pilot', ['design', 'noise', 'weight'], ['median', 'realizations'], 'B7_inverse/pilot/*.json', 'b7_campaign.py', NEU, None),
    ('Table_B1_design', ['stage', 'mode', 'alpha_true', 'input', 'gate', 'noise_sd', 'mesh_r'],
     ['n_ok', 'alpha_hat_mean', 'alpha_hat_sd', 'bias', 'rmse', 'discretisation_bias_ls', 'discretisation_bias_joint', 'optimisation_error_rms', 'crb_sd_alpha',
      'rel_l2_median', 'median_over_stepper', 'wall_median_s', 'final_objective_median', 'objective_joint_optimum'],
     'B1_design/{main,mesh}/*.json; B1_design/reference/*.csv', 'run_b1_campaign.py; b1_floor.py', NEU, None),
    ('Table_B1_l1_floor', ['alpha', 'mesh_r'], ['rel_l2', 'max_abs', 'wall_ms'], 'B1_design/reference/l1_floor.csv', 'b1_floor.py', NEU, None),
    ('Table_B1_ls_mesh', ['alpha_true', 'mesh_r', 'N', 'noise'], ['alpha_ls_mean', 'alpha_ls_sd', 'n', 'bias'], 'B1_design/reference/ls_mesh.csv',
     'b1_ls_mesh.py', NEU, None),
    ('Table_B1_discrete_optimum', ['alpha_true', 'mesh_r', 'noise', 'seed'], ['alpha_res', 'alpha_ls', 'alpha_joint'], 'B1_design/reference/discrete_optimum.csv',
     'b1_floor.py', NEU, None),
    ('Table_B1_paired', ['mode', 'metric', 'alpha_true', 'noise_sd', 'a', 'b'], PAIR, 'B1_design/main/*.json', 'run_b1_campaign.py', NEU, None),
    ('Table_surrogate', ['method', 'seed'],
     ['train_wall_s', 'final_loss', 'rel_fro_median', 'rel_fro_max', 'max_fro_max', 'min_eig_random_params', 'frac_random_params_with_negative_eig',
      'inference_ms_per_query', 'inference_ms_per_query_final_time', 'stepper_N_equal_accuracy', 'stepper_ms_per_query', 'break_even_queries',
      'break_even_queries_final_time_only', 'n_training_parameter_points', 'n_test_parameter_points', 'n_test_times', 'n_random_parameter_points'],
     'B2_surrogate/main/*.json; B2_surrogate/timing.json; B2_surrogate/reference/stepper.csv',
     'b2_surrogate.py; b2_timing.py', NEU, None),
    ('Table_surrogate_breakeven', ['method', 'seed', 'training_time', 'stepper_N'], ['train_wall_s', 'identical_to_campaign_run', 'break_even_queries',
                                                                                    'break_even_queries_final_time_only'],
     'B2_surrogate/{main,timing}/*.json; B2_surrogate/timing.json', 'b2_surrogate.py; b2_timing.py', NEU, None),
] + [(f'Table_B5_{t}', keys, '*', f'B5_conformation/{src}', drv, B5, None) for t, keys, src, drv in [
    ('forward_stepper', ['flow', 'Wi', 'alpha'], 'forward/stepper_*.json', 'b5_conformation.py'),
    ('forward_a0.6_aligned', ['output / residual norm'], 'forward/*.json', 'b5_conformation.py'),
    ('forward_a0.6_rotated', ['output / residual norm'], 'forward/*.json', 'b5_conformation.py'),
    ('forward_a0.9_aligned', ['output / residual norm'], 'forward/*.json', 'b5_conformation.py'),
    ('forward_a0.9_rotated', ['output / residual norm'], 'forward/*.json', 'b5_conformation.py'),
    ('forward_best_vs_stepper', ['flow', 'Wi', 'alpha', 'frame'], 'forward/*.json', 'b5_conformation.py'),
    ('forward_admissibility', ['output / residual norm'], 'forward/*_s*.json', 'b5_conformation.py'),
    ('forward_budget', ['flow', 'Wi', 'alpha', 'frame', 'output / residual norm', 'budget factor'], 'forward/*_budget*.json', 'b5_conformation.py'),
    ('inverse_summary', ['alpha true', 'noise, %', 'method'], 'inverse/*.json', 'b5_inverse.py'),
    ('inverse_paired', ['noise, %', 'method'], 'inverse/*.json', 'b5_inverse.py'),
    ('fenep_summary', ['Wi', 'alpha', 'method'], 'fenep/*.json', 'b5_fenep.py'),
    ('positivity_steppers', ['flow', 'Wi', 'alpha', 'mesh'], 'positivity/positivity.json', 'b5_positivity.py'),
    ('positivity_nonlinear', ['law', 'flow', 'Wi', 'parameter', 'alpha'], 'positivity/positivity.json', 'b5_positivity.py'),
    ('positivity_exact', ['flow', 'Wi', 'alpha'], 'positivity/positivity.json', 'b5_positivity.py'),
    ('fenep_step', ['flows', 'subset'], 'positivity/fenep_step.json', 'b5_fenep_step.py'),
    ('fenep_step_case', ['item', 'quantity'], 'positivity/fenep_step.json', 'b5_fenep_step.py'),
    ('fenep_step_case2', ['item', 'quantity'], 'positivity/fenep_step.json', 'b5_fenep_step.py'),
    ('fenep_stored_check', ['Wi', 'alpha', 'N (graded mesh)'], 'positivity/fenep_step.json; fenep/reference_*.npz', 'b5_fenep_step.py'),
]]


def outputs(groups):
    """Existing output files (absolute paths) of the given groups."""
    out = []
    for g in groups:
        for folder, pats in ((TD, GROUPS[g][1]), (FD, GROUPS[g][2])):
            if os.path.isdir(folder):
                out += [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if any(fnmatch.fnmatch(f, p) for p in pats)]
    return out


def write_manifest():
    rows = []
    for table, keys, values, raw, driver, agg, where in QUANTITIES:
        path = os.path.join(TD, table + '.csv')
        with open(path, encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                if where and any(r[c] not in ok for c, ok in where.items()):
                    continue
                key = ','.join(f'{k}={r[k]}' for k in keys if r[k] != '')
                for col in ([c for c in r if c not in keys] if values == '*' else values):
                    if r[col] != '':
                        rows.append((f'{table[6:]}:{key}:{col}', r[col], f'{table}.csv', raw, f'{driver} -> {agg}'))
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit('results-manifest: identifiers are not unique')
    with open(MANIFEST, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['id', 'value', 'table', 'source_file', 'script']); w.writerows(rows)
    print(f'wrote results-manifest.csv ({len(rows)} quantities from {len(QUANTITIES)} tables)')


def stable(path, data):
    """Content of an output file without the parts that legitimately change between runs."""
    if path.endswith('.meta.json'):
        d = json.loads(data.decode('utf-8'))
        return json.dumps({k: v for k, v in d.items() if k not in VOLATILE}, sort_keys=True).encode()
    return data.replace(b'\r\n', b'\n')


def numeric_difference(path, old, new):
    """Largest relative difference of the numeric cells of two versions of a CSV file or of the arrays of an NPZ file
    (inf if the files differ in structure or in a non-numeric cell)."""
    import numpy as np
    if path.endswith('.npz'):
        a, b = np.load(io.BytesIO(old)), np.load(io.BytesIO(new))
        if sorted(a.files) != sorted(b.files) or any(a[k].shape != b[k].shape for k in a.files):
            return float('inf')
        return max([float(np.nanmax(np.abs(a[k] - b[k]) / np.maximum(np.abs(a[k]), 1e-300))) for k in a.files
                    if a[k].dtype.kind == 'f' and a[k].size and not np.array_equal(a[k], b[k], equal_nan=True)] or [0.0])
    if not path.endswith('.csv'):
        return float('inf')
    ra, rb = (list(csv.reader(io.StringIO(x.decode('utf-8')))) for x in (old, new))
    if len(ra) != len(rb) or any(len(x) != len(y) for x, y in zip(ra, rb)):
        return float('inf')
    worst = 0.0
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            if u != v:
                try:
                    fu, fv = float(u), float(v)
                except ValueError:
                    return float('inf')
                if fu != fv and not (fu != fu and fv != fv):
                    worst = max(worst, abs(fu - fv) / max(abs(fu), 1e-300))
    return worst


def main():
    ap = argparse.ArgumentParser(description='All tables and figure data from the stored raw results.')
    ap.add_argument('--only', action='append', choices=sorted(GROUPS)); ap.add_argument('--clean', action='store_true')
    ap.add_argument('--check', action='store_true'); ap.add_argument('--rtol', type=float, default=0.0)
    a = ap.parse_args(); groups = a.only or ['deterministic', 'neural', 'b5']; full = len(groups) == len(GROUPS)

    def products():
        """Outputs of the selected groups; with all groups also the manifest and the tables of the article."""
        extra = [MANIFEST] + ([os.path.join(TF, f) for f in sorted(os.listdir(TF)) if f.endswith('.tex')] if os.path.isdir(TF) else []) if full else []
        return outputs(groups) + [p for p in extra if os.path.exists(p)]
    before = {p: open(p, 'rb').read() for p in products()} if a.check else {}
    if a.clean:
        for p in products():
            os.remove(p)
    # one BLAS thread: the order of the floating-point sums, and hence the last digit, does not depend on the machine
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for g in groups:
        t0 = time.perf_counter(); print(f'--- {g}: python scripts/' + ' '.join(GROUPS[g][0]), flush=True)
        rc = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', GROUPS[g][0][0])] + GROUPS[g][0][1:], cwd=ROOT, env=env).returncode
        if rc:
            sys.exit(f'aggregation of the group {g} FAILED (exit status {rc})')
        print(f'--- {g}: done in {time.perf_counter() - t0:.0f} s', flush=True)
    if full:
        write_manifest(); print('--- tables of the article: python scripts/make_tables.py --extra', flush=True)
        rc = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'make_tables.py'), '--extra'], cwd=ROOT, env=env).returncode
        if rc:
            sys.exit(f'scripts/make_tables.py FAILED (exit status {rc})')
    else:
        print('results-manifest.csv and tables/final are written only when all groups are aggregated')
    rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, '/')
    known = set(outputs(sorted(GROUPS)))
    other = [rel(os.path.join(d, f)) for d in (TD, FD) for f in sorted(os.listdir(d)) if os.path.join(d, f) not in known]
    if other:
        print(f'note: {len(other)} files in tables/data and figures/data are not outputs of this script: ' + ', '.join(other))
    if not a.check:
        return 0
    after = {p: open(p, 'rb').read() for p in products()}
    same = 0; bad = []; close = []; new = sorted(rel(p) for p in after if p not in before)
    for p in sorted(before):
        if p not in after:
            bad.append(f'{rel(p)}: not regenerated'); continue
        if stable(p, before[p]) == stable(p, after[p]):
            same += 1; continue
        d = numeric_difference(p, before[p], after[p])
        (bad if d > a.rtol else close).append(f'{rel(p)}: largest relative difference {d:.3e}')
    print(f'\ncheck: {len(before)} files existed before the run; {same} regenerated identically, {len(close)} within the tolerance, '
          f'{len(bad)} different; {len(new)} new files')
    for line in close:
        print('  within the tolerance:', line)
    for line in new:
        print('  new:', line)
    for line in bad:
        print('  DIFFERENT:', line)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())

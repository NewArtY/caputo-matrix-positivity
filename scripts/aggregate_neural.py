"""Summary tables and figure data of the neural benchmarks B1, B2, B6, B7 and of the parametric surrogate.

Reads results/raw/{B1_design, B2_factorial, B2_surrogate, B6_positivity, B7_inverse} and writes
  tables/data/Table_B2_*.csv, Table_B6_*.csv, Table_B7_*.csv, Table_B1_*.csv, Table_surrogate*.csv
  figures/data/Neural_*.csv
Every number in the tables is computed here from the per-run result files; nothing is typed in by hand.
Usage: python scripts/aggregate_neural.py [B1 B2 B6 B7 surrogate]      (default: all that are present)
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'results', 'raw'); TD = os.path.join(ROOT, 'tables', 'data'); FD = os.path.join(ROOT, 'figures', 'data')
for d in (TD, FD):
    os.makedirs(d, exist_ok=True)
TOL = 1e-12                                     # eigenvalues below -TOL count as negative


def load(pattern, drop=('curves', 'history', 'config', 'code_sha256', 'final_terms', 'fits', 'test_grid', 'random_params', 'theta_init', 'theta_starts')):
    """One row per result file (sorted by name, so that the output does not depend on the file system)."""
    recs = []
    for f in sorted(glob.glob(os.path.join(RAW, pattern))):
        r = json.load(open(f)); recs.append({k: v for k, v in r.items() if k not in drop}); recs[-1]['_file'] = os.path.relpath(f, ROOT).replace(os.sep, '/')
        opt = (r.get('config') or {}).get('optimizer') or {}                     # training budget of the run
        recs[-1]['adam_iters'] = opt.get('adam_iters'); recs[-1]['lbfgs_iters'] = opt.get('lbfgs_iters')
    return pd.DataFrame(recs)


def uniq(x):
    """Common value of a column within a group of runs (their budget); different values are joined."""
    v = sorted(set(int(i) for i in pd.Series(x).dropna()))
    return (v[0] if len(v) == 1 else ' '.join(str(i) for i in v)) if v else np.nan


def rmse(x):
    return float(np.sqrt((np.asarray(x, dtype=float) ** 2).mean())) if len(x) else np.nan


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def save(df, name, folder=TD):
    # fixed line terminator: the tables are byte-identical on every platform
    df.to_csv(os.path.join(folder, name), index=False, float_format='%.6g', lineterminator='\r\n'); print(f'wrote {name}: {len(df)} rows')


def paired(df, keys, cell, a, b, metric='rel_fro'):
    """Paired comparison of cells a and b over the runs that share `keys` (seed, state, ...): number of pairs, wins of a,
    geometric mean and median of metric(a)/metric(b), two-sided Wilcoxon signed-rank test on the log ratios."""
    A = df[df[cell] == a].set_index(keys)[metric]; B = df[df[cell] == b].set_index(keys)[metric]
    idx = A.index.intersection(B.index)
    if len(idx) == 0:
        return None
    lr = np.log(A.loc[idx].values / B.loc[idx].values)
    p = wilcoxon(lr).pvalue if np.any(lr != 0) and len(lr) >= 6 else float('nan')
    return dict(a=a, b=b, n_pairs=len(idx), wins_a=int((lr < 0).sum()), gmean_ratio_a_over_b=float(np.exp(lr.mean())),
                median_ratio_a_over_b=float(np.exp(np.median(lr))), min_ratio=float(np.exp(lr.min())), max_ratio=float(np.exp(lr.max())),
                p_wilcoxon=float(p))


def accuracy_summary(df, by):
    """Per-cell statistics over seeds of the forward benchmarks (B2, B6)."""
    rows = []
    for key, g in df.groupby(by, dropna=False, sort=True):
        ok = g[g.status == 'ok']; r = dict(zip(by, key if isinstance(key, tuple) else (key,))); e = ok.rel_fro
        r.update(n=len(g), n_ok=len(ok), seeds=' '.join(str(s) for s in sorted(g.seed)),
                 rel_fro_mean=e.mean(), rel_fro_sd=e.std(ddof=1), rel_fro_median=e.median(), rel_fro_min=e.min(), rel_fro_max=e.max(),
                 rel_fro_gmean=gmean(e) if len(e) else np.nan, max_fro_mean=ok.max_fro.mean(), max_fro_median=ok.max_fro.median(),
                 rel_inc_layer_median=ok.rel_inc_layer.median(), rel_fro_proj_median=ok.rel_fro_proj.median(),
                 ic_error_max=ok.ic_error.max(), min_eig_min=ok.min_eig.min(), min_eig_median=ok.min_eig.median(),
                 n_runs_negative_eig=int((ok.min_eig < -TOL).sum()), neg_time_frac_median=ok.neg_time_frac.median(),
                 neg_time_frac_max=ok.neg_time_frac.max(), neg_log_frac_median=ok.neg_log_frac.median(),
                 max_negativity_median=ok.max_negativity.median(), max_negativity_max=ok.max_negativity.max(),
                 max_trace_defect=ok.max_trace_defect.max(), max_herm_defect=ok.max_herm_defect.max(),
                 dist_to_l1_nodes_median=ok.node_rms_fro_to_l1.median(), wall_median_s=ok.wall.median(), cpu_median_s=ok.cpu.median(),
                 n_params=int(g.n_params.iloc[0]), adam_iters=uniq(g.adam_iters), lbfgs_iters=uniq(g.lbfgs_iters))
        rows.append(r)
    return pd.DataFrame(rows)


def curve_table(files, by_cols, df):
    """Median over seeds of the stored error and smallest-eigenvalue curves (log-spaced times) of each cell."""
    out = []
    for key, g in df.groupby(by_cols, dropna=False, sort=True):
        cs = [json.load(open(os.path.join(ROOT, f)))['curves'] for f in g._file]
        tau = np.array(cs[0]['tau']); err = np.array([c['err_fro'] for c in cs]); me = np.array([c['min_eig'] for c in cs])
        for k in range(len(tau)):
            out.append(dict(zip(by_cols, key), tau=tau[k], err_fro_median=np.median(err[:, k]), err_fro_q25=np.quantile(err[:, k], 0.25),
                            err_fro_q75=np.quantile(err[:, k], 0.75), min_eig_median=np.median(me[:, k]), min_eig_min=me[:, k].min(),
                            min_eig_exact=cs[0]['min_eig_exact'][k]))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- B2

def b2():
    df = pd.concat([load(f'B2_factorial/{s}/*.json') for s in ('main', 'extras', 'long')], ignore_index=True)
    if df.empty:
        return print('B2: no results')
    sel = json.load(open(os.path.join(RAW, 'B2_factorial', 'soft_ic_selected.json')))
    tuned = {(m, float(a)): w for m, d in sel['selected'].items() for a, w in d.items()}
    df['feature'] = df.feature.fillna('plain')

    def cell(r):
        if r.ic == 'soft':
            v = 'soft-tuned' if tuned[(r['map'], r.alpha)] == r.ic_weight else f'soft-w{r.ic_weight:g}'
        else:
            v = 'hard-' + r.gate
        return f"{r['map']}/{v}" + ('/singular' if r.feature == 'singular' else '')
    df['cell'] = df.apply(cell, axis=1); df['ic_gate'] = df.cell.str.split('/').str[1]
    st = pd.read_csv(os.path.join(RAW, 'B2_factorial', 'reference', 'l1_stepper.csv'))
    floor = st[st.N == 128].set_index(['alpha', 'state']).rel_fro
    by = ['stage', 'alpha', 'state', 'map', 'ic', 'ic_gate', 'gate', 'ic_weight', 'feature', 'cell']
    T = accuracy_summary(df, by)
    T['l1_stepper_rel_fro_N128'] = [floor[(a, s)] for a, s in zip(T.alpha, T.state)]
    T['median_over_stepper'] = T.rel_fro_median / T.l1_stepper_rel_fro_N128
    S = st.rename(columns={'wall_ms': 'wall_median_s'}); S['wall_median_s'] /= 1e3
    S = S.assign(stage='reference', map='l1_stepper', ic='exact', ic_gate='-', cell='l1_stepper/N' + S.N.astype(str), n=1, n_ok=1,
                 rel_fro_mean=S.rel_fro, rel_fro_median=S.rel_fro, max_fro_mean=S.max_fro, max_fro_median=S.max_fro,
                 rel_inc_layer_median=S.rel_inc_layer, min_eig_min=S.min_eig, neg_time_frac_max=S.neg_time_frac,
                 mesh_N=S.N).drop(columns=['N', 'rel_fro', 'max_fro', 'rel_inc_layer', 'min_eig', 'neg_time_frac', 'tau_max_fro',
                                           'rel_fro_u1000', 'node_max_fro', 'residual_J_phys', 'mesh_r'])
    save(pd.concat([T, S], ignore_index=True), 'Table_B2_factorial.csv')
    # geometric means over states and seeds: the factorial at a glance
    M = df[(df.stage == 'main') & (df.status == 'ok')].groupby(['alpha', 'map', 'ic_gate']).rel_fro.agg(gmean=gmean, median='median', n='count').reset_index()
    save(M, 'Table_B2_marginal.csv')
    # paired comparisons (same alpha, state, seed)
    keys = ['alpha', 'state', 'seed']; ok = df[df.status == 'ok']; main = ok[ok.stage == 'main']; rows = []

    def add(group, a, b, sub, label):
        # e_F against the exact solution, and the distance to the L1 solution at the nodes (optimisation error only)
        for metric in ('rel_fro', 'node_rms_fro_to_l1'):
            for al in [None] + sorted(sub.alpha.unique()):
                s = sub if al is None else sub[sub.alpha == al]
                p = paired(s, keys, 'cell', a, b, metric)
                if p:
                    rows.append(dict(comparison=group, metric=metric, alpha='all' if al is None else al, states=label, **p))
    for label, sub in [('all', main), ('pure', main[main.state == 'pure'])]:
        for m in ('unc', 'herm', 'lin', 'factor'):
            add('initial condition: hard(exp) vs soft(tuned)', f'{m}/hard-exp', f'{m}/soft-tuned', sub, label)
            add('initial condition: hard(pow) vs soft(tuned)', f'{m}/hard-pow', f'{m}/soft-tuned', sub, label)
        for m in ('unc', 'herm', 'lin', 'factor', 'add', 'gate2', 'gate2b'):
            add('gate: pow vs exp', f'{m}/hard-pow', f'{m}/hard-exp', sub, label)
        for g in ('hard-exp', 'hard-pow'):
            for m in ('herm', 'lin', 'factor', 'add'):
                add(f'map vs unconstrained ({g})', f'{m}/{g}', f'unc/{g}', sub, label)
            for a, b in (('factor', 'lin'), ('add', 'lin'), ('add', 'factor'), ('gate2', 'factor'), ('gate2b', 'factor'), ('gate2b', 'add'),
                         ('gate2b', 'lin')):
                add(f'map vs map ({g})', f'{a}/{g}', f'{b}/{g}', sub, label)
        add('first campaign: SP-fPINN vs Direct', 'factor/hard-exp', 'unc/soft-w1', sub, label)
        add('first campaign: SP-soft vs Direct', 'factor/soft-w1', 'unc/soft-w1', sub, label)
        add('soft IC: tuned vs weight 1', 'unc/soft-tuned', 'unc/soft-w1', sub, label)
    ex = ok[ok.stage.isin(['main', 'extras'])]
    add('minimal Bloch head vs projected linear head (hard-pow)', 'bloch/hard-pow', 'lin/hard-pow', ex, 'all')
    for m in ('lin', 'factor', 'add'):
        add('input: (tau, tau^alpha) vs tau (hard-pow)', f'{m}/hard-pow/singular', f'{m}/hard-pow', ex, 'all')
    lg = ok[ok.stage == 'long']
    for a, b in (('factor', 'lin'), ('add', 'lin'), ('lin', 'unc'), ('factor', 'unc'), ('add', 'factor')):
        add('long budget: map vs map (hard-pow)', f'{a}/hard-pow', f'{b}/hard-pow', lg, 'all')
    save(pd.DataFrame(rows), 'Table_B2_paired.csv')
    # effect of the budget (same cells, seeds 0-4): long over default
    rows = []
    for (a, s, c), g in lg.groupby(['alpha', 'state', 'cell']):
        d = main[(main.alpha == a) & (main.state == s) & (main.cell == c) & main.seed.isin(g.seed)]
        if len(d):
            rows.append(dict(alpha=a, state=s, cell=c, n=len(g), rel_fro_median_default=d.rel_fro.median(), rel_fro_median_long=g.rel_fro.median(),
                             ratio_long_over_default=g.rel_fro.median() / d.rel_fro.median(), l1_stepper_rel_fro_N128=floor[(a, s)],
                             cpu_median_default_s=d.cpu.median(), cpu_median_long_s=g.cpu.median(),
                             adam_iters_default=uniq(d.adam_iters), lbfgs_iters_default=uniq(d.lbfgs_iters),
                             adam_iters_long=uniq(g.adam_iters), lbfgs_iters_long=uniq(g.lbfgs_iters),
                             budget_factor=uniq(g.adam_iters) / uniq(d.adam_iters)))
    if rows:
        save(pd.DataFrame(rows), 'Table_B2_budget.csv')
    # figure data
    save(df[['stage', 'alpha', 'state', 'map', 'ic_gate', 'feature', 'cell', 'seed', 'status', 'rel_fro', 'max_fro', 'rel_inc_layer', 'rel_fro_proj',
             'min_eig', 'max_trace_defect', 'max_herm_defect', 'ic_error', 'node_rms_fro_to_l1', 'wall', 'cpu']], 'Neural_B2_runs.csv', FD)
    save(curve_table(None, ['alpha', 'state', 'cell'], main[main.ic == 'hard']), 'Neural_B2_error_curves.csv', FD)
    save(st, 'Neural_B2_stepper.csv', FD)
    pil = load('B2_factorial/pilot/*.json')
    pt = pil.groupby(['alpha', 'map', 'ic_weight']).rel_fro.agg(gmean=gmean, median='median', n='count').reset_index()
    pt['seeds'] = pil.groupby(['alpha', 'map', 'ic_weight']).seed.agg(lambda s: ' '.join(str(i) for i in sorted(set(s)))).values
    save(pt, 'Table_B2_soft_ic_pilot.csv')
    tm = os.path.join(RAW, 'B2_factorial', 'timing', 'timing.csv')          # sequential timing pass (b2_timing.py), if it was run
    if os.path.exists(tm):
        save(pd.read_csv(tm), 'Table_B2_timing.csv')


# ---------------------------------------------------------------------------------------------- B6

def b6():
    df = pd.concat([load(f'B6_positivity/{s}/*.json') for s in ('main', 'long')], ignore_index=True)
    if df.empty:
        return print('B6: no results')
    df['cell'] = df['map'] + '/hard-' + df.gate
    st = pd.read_csv(os.path.join(RAW, 'B6_positivity', 'reference', 'l1_stepper.csv'))
    floor = st[st.N == 128].set_index(['system', 'alpha']).rel_fro
    T = accuracy_summary(df, ['stage', 'system', 'alpha', 'map', 'gate', 'cell'])
    T['l1_stepper_rel_fro_N128'] = [floor[(s, a)] for s, a in zip(T.system, T.alpha)]
    T['median_over_stepper'] = T.rel_fro_median / T.l1_stepper_rel_fro_N128
    S = st.rename(columns={'wall_ms': 'wall_median_s'}); S['wall_median_s'] /= 1e3
    S = S.assign(stage='reference', map='l1_stepper', gate='-', cell='l1_stepper/N' + S.N.astype(str), n=1, n_ok=1, rel_fro_mean=S.rel_fro,
                 rel_fro_median=S.rel_fro, max_fro_mean=S.max_fro, max_fro_median=S.max_fro, rel_inc_layer_median=S.rel_inc_layer,
                 min_eig_min=S.min_eig, neg_time_frac_max=S.neg_time_frac, n_runs_negative_eig=(S.min_eig < -TOL).astype(int),
                 mesh_N=S.N).drop(columns=['N', 'rel_fro', 'max_fro', 'rel_inc_layer', 'min_eig', 'neg_time_frac', 'tau_max_fro',
                                           'rel_fro_u1000', 'node_max_fro', 'residual_J_phys', 'mesh_r'])
    save(pd.concat([T, S], ignore_index=True), 'Table_B6_accuracy.csv')
    # positivity: how often, by how much, for how long
    ok = df[df.status == 'ok']; rows = []
    ck = json.load(open(os.path.join(RAW, 'B6_positivity', 'reference', 'prepare.json')))['checks']
    for (stg, s, a, c), g in ok.groupby(['stage', 'system', 'alpha', 'cell']):
        neg = g[g.min_eig < -TOL]
        rows.append(dict(stage=stg, system=s, alpha=a, cell=c, n=len(g), runs_with_negative_eig=len(neg),
                         most_negative_eig=g.min_eig.min(), median_min_eig=g.min_eig.median(),
                         median_fraction_of_time_negative=g.neg_time_frac.median(), max_fraction_of_time_negative=g.neg_time_frac.max(),
                         median_fraction_of_log_time_negative=g.neg_log_frac.median(),
                         median_max_negativity=g.max_negativity.median(), median_integrated_negativity=g.int_negativity.median(),
                         rel_fro_median=g.rel_fro.median(), max_fro_median=g.max_fro.median(),
                         rel_fro_after_projection_median=g.rel_fro_proj.median(), max_trace_defect=g.max_trace_defect.max(),
                         max_herm_defect=g.max_herm_defect.max(), exact_max_smallest_eig=ck[f'{s}_a{a}']['max_smallest_eig_exact']))
    save(pd.DataFrame(rows), 'Table_B6_positivity.csv')
    keys = ['system', 'alpha', 'seed']; rows = []
    for stg in ('main', 'long'):
        sub = ok[ok.stage == stg]
        for grp, sel in [('all', sub)] + [(f'{s} alpha={a}', sub[(sub.system == s) & (sub.alpha == a)]) for s in sorted(sub.system.unique())
                                           for a in sorted(sub.alpha.unique())]:
            # e_F against the exact solution, and the distance to the L1 solution at the nodes (optimisation error only)
            for metric in ('rel_fro', 'node_rms_fro_to_l1'):
                for a, b in (('factor', 'lin'), ('add', 'lin'), ('gate2', 'lin'), ('gate2b', 'lin'), ('lin', 'unc'), ('herm', 'unc'),
                             ('factor', 'unc'), ('add', 'factor'), ('gate2', 'factor'), ('gate2b', 'factor')):
                    for g in ('pow', 'exp'):
                        p = paired(sel, keys, 'cell', f'{a}/hard-{g}', f'{b}/hard-{g}', metric)
                        if p:
                            rows.append(dict(stage=stg, group=grp, metric=metric, **p))
                for m in ('lin', 'factor', 'add'):
                    p = paired(sel, keys, 'cell', f'{m}/hard-pow', f'{m}/hard-exp', metric)
                    if p:
                        rows.append(dict(stage=stg, group=grp, metric=metric, **p))
    save(pd.DataFrame(rows), 'Table_B6_paired.csv')
    save(df[['stage', 'system', 'alpha', 'map', 'gate', 'cell', 'seed', 'status', 'rel_fro', 'max_fro', 'rel_fro_proj', 'min_eig', 'tau_min_eig',
             'neg_time_frac', 'neg_log_frac', 'max_negativity', 'int_negativity', 'max_trace_defect', 'max_herm_defect', 'wall', 'cpu']],
         'Neural_B6_runs.csv', FD)
    save(curve_table(None, ['system', 'alpha', 'cell'], ok[ok.stage == 'main']), 'Neural_B6_curves.csv', FD)
    save(st, 'Neural_B6_stepper.csv', FD)


# ---------------------------------------------------------------------------------------------- B7

def b7():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, 'B7_inverse', 'main', '*.json'))):
        r = json.load(open(f)); base = dict(design=r['design'], noise=r['noise'], real=r['real'])
        if r['method'] == 'pinn':
            rows.append(dict(base, method=f"PINN {r['map']}", family='pinn', status=r['status'], weight=r['weight'],
                             alpha_init=r['theta_init'][0], Gdown_init=r['theta_init'][1], Gphi_init=r['theta_init'][2],
                             **{k: r.get(k) for k in ('alpha_hat', 'Gdown_hat', 'Gphi_hat', 'rel_err_rms', 'wall', 'cpu', 'final_residual',
                                                      'final_data_misfit', 'state_rel_fro', 'state_max_fro', 'state_min_eig',
                                                      'state_max_trace_defect', 'state_max_herm_defect', 'state_neg_time_frac')}))
        else:
            for name, ft in r['fits'].items():
                rows.append(dict(base, method=f'NLS-L1 {name}', family='nls', status='ok', n_solves=ft['n_solves'],
                                 starts_reaching_best=ft['starts_reaching_best'], rms_misfit=ft['rms_misfit'],
                                 data_points_changed_by_projection=ft['data_points_changed_by_projection'], state_min_eig=ft['min_eig_nodes'],
                                 **{k: ft.get(k) for k in ('alpha_hat', 'Gdown_hat', 'Gphi_hat', 'rel_err_rms', 'wall', 'state_rel_fro',
                                                           'state_max_fro', 'state_max_trace_defect', 'state_max_herm_defect',
                                                           'state_neg_time_frac')}))
    df = pd.DataFrame(rows)
    if df.empty:
        return print('B7: no results')
    cfg = json.load(open(os.path.join(ROOT, 'code', 'configs', 'b7_inverse.json')))
    true = dict(alpha=cfg['alpha_true'], Gdown=cfg['phys_true']['Gdown'], Gphi=cfg['phys_true']['Gphi'])
    crb = pd.read_csv(os.path.join(RAW, 'B7_inverse', 'reference', 'crb.csv')).set_index(['design', 'noise_sd', 'parameter']).crb_sd
    # a run "agrees with least squares" if every parameter is within 3 Cramer-Rao s.d. + 2 % of the true value of the
    # least-squares estimate obtained from the same data on the mesh of the networks
    R = df[df.method == 'NLS-L1 N128_interp'].set_index(['design', 'noise', 'real'])
    agree = np.ones(len(df), dtype=bool)
    for p in ('alpha', 'Gdown', 'Gphi'):
        ref_p = np.array([R.loc[(d, n, r), p + '_hat'] for d, n, r in zip(df.design, df.noise, df.real)])
        tol = np.array([3 * crb[(d, n, p)] + 0.02 * true[p] for d, n in zip(df.design, df.noise)])
        agree &= (np.abs(df[p + '_hat'].values - ref_p) <= tol)
    df['agrees_with_ls'] = agree & (df.status == 'ok').values
    out = []
    for (d, n, m), g in df.groupby(['design', 'noise', 'method'], sort=True):
        ok = g[g.status == 'ok']; r = dict(design=d, noise_sd=n, method=m, n=len(g), n_ok=len(ok))
        for p in ('alpha', 'Gdown', 'Gphi'):
            x = ok[p + '_hat']
            r.update({f'{p}_true': true[p], f'{p}_mean': x.mean(), f'{p}_bias': x.mean() - true[p], f'{p}_sd': x.std(ddof=1),
                      f'{p}_rmse': float(np.sqrt(((x - true[p]) ** 2).mean())), f'{p}_crb_sd': crb[(d, n, p)]})
        # runs in which the trainable order ended at its lower bound (alpha_hat < 0.06; the bound is 0.05), runs
        # that agree with least squares, and the statistics of the latter
        conv = ok[ok.agrees_with_ls]; r.update(n_alpha_at_lower_bound=int((ok.alpha_hat < 0.06).sum()), n_agree_with_ls=len(conv))
        for p in ('alpha', 'Gdown', 'Gphi'):
            x = conv[p + '_hat']
            r.update({f'{p}_mean_conv': x.mean(), f'{p}_sd_conv': x.std(ddof=1), f'{p}_rmse_conv': float(np.sqrt(((x - true[p]) ** 2).mean()))})
        r.update(rel_err_rms_median=ok.rel_err_rms.median(), wall_median_s=ok.wall.median(), state_rel_fro_median=ok.state_rel_fro.median(),
                 state_min_eig_min=ok.state_min_eig.min(), runs_with_negative_eig=int((ok.state_min_eig < -TOL).sum()),
                 state_max_trace_defect=ok.state_max_trace_defect.max(), state_max_herm_defect=ok.state_max_herm_defect.max(),
                 n_solves_median=ok.n_solves.median() if 'n_solves' in ok else np.nan,
                 min_starts_reaching_best=ok.starts_reaching_best.min() if 'starts_reaching_best' in ok else np.nan,
                 data_points_changed_by_projection=ok.data_points_changed_by_projection.sum() if 'data_points_changed_by_projection' in ok else np.nan)
        out.append(r)
    save(pd.DataFrame(out), 'Table_B7_estimates.csv')
    # per-realisation distance of every method to the least-squares estimate on the mesh of the networks
    rows = []; ref_name = 'NLS-L1 N128_interp'
    for (d, n), g in df[df.status == 'ok'].groupby(['design', 'noise']):
        R = g[g.method == ref_name].set_index('real')
        for m, h in g.groupby('method'):
            h = h.set_index('real'); idx = h.index.intersection(R.index); r = dict(design=d, noise_sd=n, method=m, reference=ref_name, n=len(idx))
            for p in ('alpha', 'Gdown', 'Gphi'):
                dd = h.loc[idx, p + '_hat'] - R.loc[idx, p + '_hat']
                r.update({f'{p}_mean_diff': dd.mean(), f'{p}_rms_diff': float(np.sqrt((dd ** 2).mean())), f'{p}_crb_sd': crb[(d, n, p)]})
            rows.append(r)
    save(pd.DataFrame(rows), 'Table_B7_vs_least_squares.csv')
    # Dependence of the network fits on the initial guess of the order. The guesses of a realization are the same for all designs and
    # maps and fall into two groups (below and above 0.5). Counted are the runs that end at the lower bound of the order
    # (alpha_hat < 0.06; the bound is 0.05); the runs that do not are compared with least squares on the mesh of the networks for the
    # same data sets, and those of them that end below half the true order are counted separately.
    L = df[df.method == 'NLS-L1 N128'].set_index(['design', 'noise', 'real']).alpha_hat
    pin = df[(df.family == 'pinn') & (df.status == 'ok')]; rows = []
    sets = [(m, pin[pin.method == m]) for m in sorted(pin.method.unique())]
    sets.append(('PINN lin and PINN add', pin[pin.method.isin(['PINN lin', 'PINN add'])]))
    for m, hm in sets:
        for (d, n), g in [((d, n), g) for (d, n), g in hm.groupby(['design', 'noise'])] + [(('all', 'all'), hm)]:
            lo = g[g.alpha_init < 0.5]; hi = g[g.alpha_init >= 0.5]; nb = g[g.alpha_hat >= 0.06]; mid = nb[nb.alpha_hat < 0.5 * true['alpha']]
            ls = np.array([L[(a, b, c)] for a, b, c in zip(nb.design, nb.noise, nb.real)])
            e, el = rmse(nb.alpha_hat - true['alpha']), rmse(ls - true['alpha'])
            rows.append(dict(method=m, design=d, noise_sd=n, n=len(g), n_start_alpha_below_0p5=len(lo), max_start_alpha_below_0p5=lo.alpha_init.max(),
                             n_at_lower_bound_start_below_0p5=int((lo.alpha_hat < 0.06).sum()), n_start_alpha_above_0p5=len(hi),
                             min_start_alpha_above_0p5=hi.alpha_init.min(), n_at_lower_bound_start_above_0p5=int((hi.alpha_hat < 0.06).sum()),
                             n_not_at_lower_bound=len(nb), alpha_rmse_not_at_lower_bound=e, alpha_rmse_ls_N128_same_data=el,
                             rmse_ratio_to_ls=e / el if len(nb) else np.nan,
                             n_closer_to_truth_than_ls=int((np.abs(nb.alpha_hat.values - true['alpha']) < np.abs(ls - true['alpha'])).sum()),
                             n_below_half_true_alpha_not_at_lower_bound=len(mid),
                             min_alpha_hat_of_these=mid.alpha_hat.min() if len(mid) else np.nan,
                             max_alpha_hat_of_these=mid.alpha_hat.max() if len(mid) else np.nan))
    save(pd.DataFrame(rows), 'Table_B7_start_dependence.csv')
    save(df, 'Neural_B7_estimates.csv', FD)
    pil = load('B7_inverse/pilot/*.json')
    if not pil.empty:
        pt = pil.groupby(['design', 'noise', 'weight']).rel_err_rms.agg(median='median', max='max', n='count').reset_index()
        pt['realizations'] = pil.groupby(['design', 'noise', 'weight']).real.agg(lambda s: ' '.join(str(i) for i in sorted(set(s)))).values
        save(pt, 'Table_B7_data_weight_pilot.csv')


# ---------------------------------------------------------------------------------------------- B1

def b1():
    df = pd.concat([load(f'B1_design/{s}/*.json') for s in ('main', 'mesh')], ignore_index=True)
    if df.empty:
        return print('B1: no results')
    ref = os.path.join(RAW, 'B1_design', 'reference')
    opt = pd.read_csv(os.path.join(ref, 'discrete_optimum.csv')); fl = pd.read_csv(os.path.join(ref, 'l1_floor.csv'))
    fisher = pd.read_csv(os.path.join(ref, 'fisher.csv')).set_index('alpha_true')['sd_alpha_for_noise_0.01']
    df['mesh_r'] = df.mesh_r.round(6); opt['mesh_r'] = opt.mesh_r.round(6); fl['mesh_r'] = fl.mesh_r.round(6)
    floor = fl.set_index(['alpha', 'mesh_r']); rows = []
    # each inverse run next to the discrete optima of its own data (noise-free data: one optimum for all seeds)
    o = opt.set_index(['alpha_true', 'mesh_r', 'noise', 'seed'])

    def optimum(r, col):
        return o.loc[(r.alpha_true, r.mesh_r, r.noise, r.seed if r.noise > 0 else 0), col]
    inv = df[df['mode'] == 'inverse'].copy()
    for c in ('alpha_res', 'alpha_ls', 'alpha_joint', 'J_joint_min'):
        inv[c] = [optimum(r, c) for r in inv.itertuples()]
    inv['opt_error_joint'] = inv.alpha_hat - inv.alpha_joint; inv['opt_error_ls'] = inv.alpha_hat - inv.alpha_ls
    # training objective at the end of the run: mean squared residual + data weight * mean squared misfit
    inv['final_objective'] = inv.final_residual + float(opt.data_weight.iloc[0]) * inv.final_data_misfit
    for (stg, a, i, g, n, r), h in inv.groupby(['stage', 'alpha_true', 'input', 'gate', 'noise', 'mesh_r'], sort=True):
        ok = h[h.status == 'ok']
        rows.append(dict(stage=stg, mode='inverse', alpha_true=a, input=i, gate=g, noise_sd=n, mesh_r=r, n=len(h), n_ok=len(ok),
                         alpha_hat_mean=ok.alpha_hat.mean(), alpha_hat_sd=ok.alpha_hat.std(ddof=1), alpha_hat_median=ok.alpha_hat.median(),
                         bias=ok.alpha_hat.mean() - a, rmse=float(np.sqrt(((ok.alpha_hat - a) ** 2).mean())),
                         alpha_res=ok.alpha_res.mean(), alpha_ls_mean=ok.alpha_ls.mean(), alpha_ls_sd=ok.alpha_ls.std(ddof=1),
                         alpha_joint_mean=ok.alpha_joint.mean(), discretisation_bias_ls=ok.alpha_ls.mean() - a,
                         discretisation_bias_joint=ok.alpha_joint.mean() - a, discretisation_bias_res=ok.alpha_res.mean() - a,
                         optimisation_error_mean=ok.opt_error_joint.mean(), optimisation_error_sd=ok.opt_error_joint.std(ddof=1),
                         optimisation_error_rms=float(np.sqrt((ok.opt_error_joint ** 2).mean())),
                         rms_distance_to_ls=float(np.sqrt((ok.opt_error_ls ** 2).mean())), crb_sd_alpha=n / 0.01 * fisher[a],
                         rel_l2_median=ok.rel_l2.median(), wall_median_s=ok.wall.median(), cpu_median_s=ok.cpu.median(),
                         final_objective_median=ok.final_objective.median(), final_objective_min=ok.final_objective.min(),
                         final_objective_max=ok.final_objective.max(), objective_joint_optimum=ok.J_joint_min.mean()))
    fw = df[df['mode'] == 'forward']
    for (stg, a, i, g, r), h in fw.groupby(['stage', 'alpha_true', 'input', 'gate', 'mesh_r'], sort=True):
        ok = h[h.status == 'ok']; f = floor.loc[(a, r)]
        rows.append(dict(stage=stg, mode='forward', alpha_true=a, input=i, gate=g, noise_sd=0.0, mesh_r=r, n=len(h), n_ok=len(ok),
                         rel_l2_mean=ok.rel_l2.mean(), rel_l2_sd=ok.rel_l2.std(ddof=1), rel_l2_median=ok.rel_l2.median(),
                         max_abs_median=ok.max_abs.median(), l1_stepper_rel_l2=f.rel_l2, l1_stepper_max_abs=f.max_abs,
                         median_over_stepper=ok.rel_l2.median() / f.rel_l2, l1_stepper_wall_ms=f.wall_ms, wall_median_s=ok.wall.median(),
                         cpu_median_s=ok.cpu.median()))
    save(pd.DataFrame(rows), 'Table_B1_design.csv')
    save(opt, 'Table_B1_discrete_optimum.csv'); save(fl, 'Table_B1_l1_floor.csv')
    # least squares through the time stepper on the mesh of the networks and on finer meshes (b1_ls_mesh.py)
    lm = os.path.join(ref, 'ls_mesh.csv')
    if os.path.exists(lm):
        L = pd.read_csv(lm)
        T = L.groupby(['alpha_true', 'mesh_r', 'N', 'noise']).alpha_ls.agg(alpha_ls_mean='mean', alpha_ls_sd=lambda x: x.std(ddof=1) if len(x) > 1 else np.nan,
                                                                        n='count').reset_index()
        T['bias'] = T.alpha_ls_mean - T.alpha_true
        save(T, 'Table_B1_ls_mesh.csv')
    # paired effects of the two factors on |alpha_hat - alpha_true| and on the forward error
    rows = []; keys = ['alpha_true', 'noise', 'mesh_r', 'seed']; df['cell'] = df.input + '/' + df.gate; df['abs_err'] = (df.alpha_hat - df.alpha_true).abs()
    for mode, metric in (('inverse', 'abs_err'), ('forward', 'rel_l2')):
        sub = df[(df['mode'] == mode) & (df.stage == 'main') & (df.status == 'ok')]
        for a, b in (('plain/pow', 'plain/exp'), ('singular/pow', 'singular/exp'), ('singular/exp', 'plain/exp'), ('singular/pow', 'plain/pow'),
                     ('singular/pow', 'plain/exp')):
            for al in [None] + sorted(sub.alpha_true.unique()):
                for nz in sorted(sub.noise.unique()):
                    s = sub[(sub.noise == nz)] if al is None else sub[(sub.noise == nz) & (sub.alpha_true == al)]
                    p = paired(s, keys, 'cell', a, b, metric)
                    if p:
                        rows.append(dict(mode=mode, metric=metric, alpha_true='all' if al is None else al, noise_sd=nz, **p))
    save(pd.DataFrame(rows), 'Table_B1_paired.csv')
    if len(inv):
        save(inv[['stage', 'alpha_true', 'input', 'gate', 'noise', 'mesh_r', 'seed', 'status', 'alpha_init', 'alpha_hat', 'alpha_res', 'alpha_ls',
                  'alpha_joint', 'opt_error_joint', 'rel_l2', 'final_residual', 'final_data_misfit', 'wall']], 'Neural_B1_inverse_runs.csv', FD)
    save(fw[['stage', 'alpha_true', 'input', 'gate', 'mesh_r', 'seed', 'status', 'rel_l2', 'max_abs', 'rel_l2_u1000', 'wall']], 'Neural_B1_forward_runs.csv', FD)


# ---------------------------------------------------------------------------------------------- surrogate

def surrogate():
    files = sorted(glob.glob(os.path.join(RAW, 'B2_surrogate', 'main', '*.json')))
    if not files:
        return print('surrogate: no results')
    ref = os.path.join(RAW, 'B2_surrogate', 'reference'); st = pd.read_csv(os.path.join(ref, 'stepper.csv'))
    z = np.load(os.path.join(ref, 'exact.npz')); P = z['params']
    sm = st.groupby('N').agg(rel_fro_median=('rel_fro', 'median'), rel_fro_max=('rel_fro', 'max'), wall_ms=('wall_ms', 'median')).reset_index()
    # costs per query: from the single-process timing pass if it exists (b2_timing.py surrogate), else as measured during the campaign
    tp = os.path.join(RAW, 'B2_surrogate', 'timing.json'); timing = json.load(open(tp)) if os.path.exists(tp) else None
    if timing:
        sm['wall_ms'] = [timing['stepper'][str(N)]['ms_per_query_median'] for N in sm.N]
    rows = []; grid = []
    for f in files:
        r = json.load(open(f))
        if 'test_grid' not in r:
            rows.append(dict(method=f"surrogate {r['map']}", seed=r['seed'], status=r['status'])); continue
        e = r['rel_fro_median']; cheaper = sm[sm.rel_fro_median <= e]            # coarsest stepper mesh that is at least as accurate
        Neq = int(cheaper.N.min()) if len(cheaper) else int(sm.N.max()); ts = float(sm[sm.N == Neq].wall_ms.iloc[0])
        src = timing if timing else r; ti = src['inference_ms_per_query_batched']; tf = src['inference_ms_per_query_final_time_batched']
        rr = r['random_params']
        rows.append(dict(method=f"surrogate {r['map']}", seed=r['seed'], status=r['status'], n_params=r['n_params'], train_wall_s=r['wall'],
                         train_cpu_s=r['cpu'], final_loss=r['final_loss'], rel_fro_median=e, rel_fro_mean=r['rel_fro_mean'],
                         rel_fro_max=r['rel_fro_max'], max_fro_max=r['max_fro_max'], min_eig_test_grid=min(r['test_grid']['min_eig']),
                         min_eig_random_params=rr['min_eig'], frac_random_params_with_negative_eig=rr['frac_params_with_negative_eig'],
                         frac_random_points_with_negative_eig=rr['frac_points_with_negative_eig'],
                         max_negativity=rr['max_negativity'], max_trace_defect=rr['max_trace_defect'], max_herm_defect=rr['max_herm_defect'],
                         inference_ms_per_query=ti, inference_ms_per_query_single=src['inference_ms_per_query_single'],
                         inference_ms_per_query_final_time=tf, stepper_N_equal_accuracy=Neq, stepper_ms_per_query=ts,
                         stepper_rel_fro_median_at_N=float(sm[sm.N == Neq].rel_fro_median.iloc[0]),
                         break_even_queries=r['wall'] * 1e3 / (ts - ti) if ts > ti else float('inf'),
                         break_even_queries_final_time_only=r['wall'] * 1e3 / (ts - tf) if ts > tf else float('inf'),
                         cost_source='timing pass' if timing else 'campaign',
                         n_training_parameter_points=r['config']['train']['pool'], n_test_parameter_points=len(r['test_grid']['rel_fro']),
                         n_test_times=r['n_test_times'], n_random_parameter_points=rr['n'],
                         adam_iters=r['config']['train']['adam_iters'], lbfgs_iters=r['config']['train']['lbfgs_iters']))
        for k in range(len(P)):
            grid.append(dict(method=f"surrogate {r['map']}", seed=r['seed'], alpha=P[k, 0], Gdown=P[k, 1],
                             **{key: r['test_grid'][key][k] for key in ('rel_fro', 'max_fro', 'min_eig', 'neg_time_frac')}))
    T = pd.DataFrame(rows)
    # Sensitivity of the break-even: against every stepper mesh, with the training time of the campaign (loaded machine) and, for a
    # run that was repeated alone on the idle machine (B2_surrogate/timing/<run>.json from `b2_surrogate.py train --stage timing`),
    # with that time as well. break-even = training time / (stepper cost per query - surrogate cost per query).
    be = []
    for f in files:
        r = json.load(open(f))
        if 'test_grid' not in r:
            continue
        src = timing if timing else r; ti = src['inference_ms_per_query_batched']; tf = src['inference_ms_per_query_final_time_batched']
        walls = [('campaign', r['wall'], None)]; fi = os.path.join(RAW, 'B2_surrogate', 'timing', os.path.basename(f))
        if os.path.exists(fi):
            q = json.load(open(fi))
            walls.append(('idle repeat', q['wall'], bool(q['final_loss'] == r['final_loss'] and q['test_grid'] == r['test_grid'])))
        for label, w, same in walls:
            for N in sm.N:
                ts = float(sm[sm.N == N].wall_ms.iloc[0]); es = float(sm[sm.N == N].rel_fro_median.iloc[0])
                be.append(dict(method=f"surrogate {r['map']}", seed=r['seed'], training_time=label, train_wall_s=w,
                               identical_to_campaign_run=same, surrogate_rel_fro_median=r['rel_fro_median'], stepper_N=int(N),
                               stepper_rel_fro_median=es, stepper_at_least_as_accurate=bool(es <= r['rel_fro_median']),
                               stepper_ms_per_query=ts, inference_ms_per_query=ti, inference_ms_per_query_final_time=tf,
                               break_even_queries=w * 1e3 / (ts - ti) if ts > ti else float('inf'),
                               break_even_queries_final_time_only=w * 1e3 / (ts - tf) if ts > tf else float('inf'),
                               cost_source='timing pass' if timing else 'campaign'))
    save(pd.DataFrame(be), 'Table_surrogate_breakeven.csv')
    S = sm.assign(method='l1_stepper N=' + sm.N.astype(str), status='ok', stepper_ms_per_query=sm.wall_ms,
                  cost_source='timing pass' if timing else 'campaign').drop(columns=['N', 'wall_ms'])
    save(pd.concat([T, S], ignore_index=True), 'Table_surrogate.csv')
    save(pd.DataFrame(grid), 'Neural_surrogate_grid.csv', FD); save(st, 'Neural_surrogate_stepper.csv', FD)


if __name__ == '__main__':
    todo = sys.argv[1:] or ['B1', 'B2', 'B6', 'B7', 'surrogate']
    for name in todo:
        dict(B1=b1, B2=b2, B6=b6, B7=b7, surrogate=surrogate)[name]()

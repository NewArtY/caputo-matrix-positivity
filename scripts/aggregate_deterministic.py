"""Aggregation of the deterministic experiments (E1-E5, Dirac B3/B4/tail) into tables and figure data.

  python scripts/aggregate_deterministic.py [E|SOE|B3|B4|tail|positivity|examples|all]

Inputs : results/raw/E1_E3, results/raw/E4_soe, results/raw/E5_positivity, results/raw/DIRAC_v2
Outputs: tables/data/Table_E1_E3.csv, Table_SOE.csv, Table_Dirac_B3.csv, Table_Dirac_B4.csv, Table_Dirac_tail.csv,
         Table_L1_positivity.csv (+ companion tables Table_*_<part>.csv), figures/data/FigDet_*.csv|npz, and
         Table_Examples_continuous.csv (numbers of the article that follow from closed forms; no raw input).
Every output CSV is plain (no comment lines) and has a side-car <name>.meta.json with the provenance record of the
aggregation and the list of raw sources. Nothing is computed here except (i) observed orders, (ii) Richardson
extrapolation in the number of time steps, (iii) fits in the momentum cutoff, (iv) the analytical tail constants of
fracsp.dirac, and (v) rounding to the digits supported by the error estimates.

Conventions for the Dirac tables
  value columns hold full precision; `*_err` is the error estimate; `*_rep` is the value rounded to the digits that
  the error estimate supports, e.g. 0.17459(3).
  alpha < 1: values are Richardson extrapolated from the two finest L1 levels with the theoretical order 2 - alpha;
             err_time = |extrapolate(N2, N3) - extrapolate(N1, N2)|.
  alpha = 1: DOP853; err_time from the tolerance check.
  err_grid : momentum-quadrature error estimated from the runs with halved spacings.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'experiments')); sys.path.insert(0, os.path.join(ROOT, 'code'))
import det_common as dc          # first: pins BLAS to one thread before numpy is loaded (same sums on every machine)
import numpy as np
import pandas as pd
from fracsp import dirac, reference as ref

RAW = os.path.join(ROOT, 'results', 'raw'); TD = os.path.join(ROOT, 'tables', 'data'); FD = os.path.join(ROOT, 'figures', 'data')
os.makedirs(TD, exist_ok=True); os.makedirs(FD, exist_ok=True)
DIR = os.path.join(RAW, 'DIRAC_v2')
dcfg = json.load(open(os.path.join(ROOT, 'code', 'configs', 'dirac.json')))
OBS = ['nc', 'jx_band', 'jy_band', 'jx_intra', 'jy_intra', 'jx_init', 'jy_init']
fmt = dc.fmt_unc


def write_table(name, df, sources, note, folder=TD):
    path = os.path.join(folder, name)
    df.to_csv(path, index=False, lineterminator='\n')
    meta = dc.provenance(os.path.abspath(__file__), None, dict(raw_sources=sorted(set(sources)), note=note, rows=int(len(df)), columns=list(map(str, df.columns))))
    with open(os.path.splitext(path)[0] + '.meta.json', 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, indent=1); f.write('\n')
    print(f'wrote {dc.rel(path)} ({len(df)} rows)')


def pars(a0=None, zeta=0.0, **upd):
    p = dict(dcfg['pulse']); p.update(delta=dcfg['delta'], Gdown=dcfg['Gdown'], Gup=dcfg['Gup'], Gphi=dcfg['Gphi'], zeta=zeta)
    if a0 is not None:
        p['a0'] = a0
    p.update(upd)
    return p


def add_orders(df, keys, col):
    """Observed order log2(e(N/2)/e(N)) for every refinement pair (NaN for the coarsest mesh)."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby(keys, sort=False):
        g = g.sort_values('N'); e = g[col].values; N = g['N'].values
        o = np.full(len(g), np.nan)
        for i in range(1, len(g)):
            if N[i] == 2 * N[i - 1] and e[i] > 0 and e[i - 1] > 0:
                o[i] = math.log2(e[i - 1] / e[i])
        out[[df.index.get_loc(j) for j in g.index]] = o
    return out


# ====================================================================================================== E1-E3
def agg_E():
    E = os.path.join(RAW, 'E1_E3'); src = []
    e1 = dc.read_csv(os.path.join(E, 'E1_l1_operator.csv')); e3 = dc.read_csv(os.path.join(E, 'E3_mittag_leffler.csv'))
    et = dc.read_csv(os.path.join(E, 'E1T_truncation.csv')); e2 = dc.read_csv(os.path.join(E, 'E2_alpha_gradient.csv'))
    src += ['results/raw/E1_E3/' + x for x in ('E1_l1_operator.csv', 'E3_mittag_leffler.csv', 'E1T_truncation.csv', 'E2_alpha_gradient.csv')]
    rows = []

    def nominal(exp, q, a, r, fn=None):
        if exp == 'E1' and fn == 'smooth':
            return 2 - a
        if exp in ('E1', 'E1T'):                         # f ~ t^alpha (E1 weak) or u = E_alpha(-t^alpha) (E1T)
            if q.startswith('max_abs'):
                return 0.0
            if q.startswith('weighted'):
                return min(2 - a, r * a)
            return min(2 - a, r * (1 + a))                # fixed positive time
        if exp == 'E3':
            return min(2 - a, r * a) if q.startswith('max') else min(2 - a, r * 1.0) if r == 1.0 else 2 - a
        return float('nan')
    for exp, df, qs, extra in (('E1', e1, ['max_abs_err', 'final_abs_err', 'weighted_max_err'], 'function'),
                               ('E1T', et, ['max_abs_trunc', 'weighted_max_trunc', 'final_abs_trunc'], None),
                               ('E3', e3, ['max_abs_err', 'final_abs_err'], None)):
        keys = ['alpha', 'mesh'] + ([extra] if extra else [])
        for q in qs:
            d = df.copy(); d['order'] = add_orders(d, keys, q)
            for _, r in d.iterrows():
                fn = r[extra] if extra else ('u=E_alpha(-t^alpha)' if exp != 'E1' else '')
                rows.append(dict(experiment=exp, function={'smooth': 't^2', 'weak': 't^alpha'}.get(fn, fn), alpha=r.alpha, mesh=r.mesh, grading_r=round(r.grading_r, 4), quantity=q,
                                 N=int(r.N), value=r[q], observed_order=r.order, nominal_order=nominal(exp, q, r.alpha, r.grading_r, r[extra] if extra else None),
                                 argmax_node=int(r.argmax_node) if q.startswith('max_abs') else ''))
    T = pd.DataFrame(rows)
    write_table('Table_E1_E3.csv', T, src,
                'observed_order = log2(value(N/2)/value(N)) for EVERY refinement pair. E1: L1 operator on t^2 and t^alpha (T=1). E1T: truncation error of the L1 formula for '
                'u = E_alpha(-t^alpha) on [0,5]. E3: implicit L1 solver for D^alpha u = -u on [0,5]. weighted_* = max_n t_n^alpha |.|. nominal_order: 2-alpha (smooth); '
                '0 for the unweighted maximum of a weakly singular function; min(2-alpha, r alpha) for the weighted maximum and for the maximum solver error; '
                'min(2-alpha, r(1+alpha)) for the operator at a fixed positive time; 1 for the uniform-mesh solver at a fixed positive time')
    # summary: order of the last pair and range over all pairs
    S = []
    for (exp, fn, a, mesh, q), g in T.groupby(['experiment', 'function', 'alpha', 'mesh', 'quantity'], sort=False):
        g = g.sort_values('N'); o = g.observed_order.dropna().values
        S.append(dict(experiment=exp, function=fn, alpha=a, mesh=mesh, grading_r=g.grading_r.values[0], quantity=q, N_max=int(g.N.max()), value_at_N_max=g.value.values[-1],
                      order_last_pair=o[-1], order_min_all_pairs=o.min(), order_max_all_pairs=o.max(), order_last_three_pairs_mean=o[-3:].mean(), nominal_order=g.nominal_order.values[0]))
    write_table('Table_E1_E3_summary.csv', pd.DataFrame(S), src, 'one row per convergence series of Table_E1_E3.csv: observed order of the last pair and its range over all pairs')
    e2o = e2.copy()
    write_table('Table_E2_alpha_derivative.csv', e2o, src[-1:],
                'alpha-derivative of sum_n D_h^alpha f_n: autograd, closed form, central finite differences (step 1e-6) and the 60-digit reference (mpmath); uniform operator and '
                'nonuniform operator on graded meshes; last column: maximum relative error of the per-node derivative (graded only)')
    T[(T.experiment == 'E3')].to_csv(os.path.join(FD, 'FigDet_E3_convergence.csv'), index=False, lineterminator='\n')
    T[(T.experiment.isin(['E1T'])) | ((T.experiment == 'E1') & (T.function == 't^alpha'))].to_csv(os.path.join(FD, 'FigDet_E1_truncation.csv'), index=False, lineterminator='\n')
    return T


# ====================================================================================================== E4 (SOE)
def loglog_interp(x, xs, ys):
    """y(x) by linear interpolation of log y against log x (xs decreasing or increasing); nan outside the range."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    o = np.argsort(lx); lx, ly = lx[o], ly[o]
    if not (lx[0] <= math.log(x) <= lx[-1]):
        return float('nan')
    return float(np.exp(np.interp(math.log(x), lx, ly)))


def agg_SOE():
    E = os.path.join(RAW, 'E4_soe'); pre = 'results/raw/E4_soe/'
    b = dc.read_csv(os.path.join(E, 'E4b_solver.csv')); mem = dc.read_csv(os.path.join(E, 'E4b_memory.csv'))
    ker = dc.read_csv(os.path.join(E, 'E4a_kernel.csv')); win = dc.read_csv(os.path.join(E, 'E4a_kernel_windows.csv'))
    res = dc.read_csv(os.path.join(E, 'E4c_residual.csv')); dch = dc.read_csv(os.path.join(E, 'E4d_dirac_chunk.csv'))
    mb = dc.read_meta(os.path.join(E, 'E4b_solver.csv'))
    t = b.drop(columns=['walls_s']).merge(mem[['solver', 'N', 'peak_heap_MB']], on=['solver', 'N'], how='left')
    t['wall_spread_percent'] = 100 * (t.wall_max_s - t.wall_min_s) / t.wall_median_s
    # speed-up of SOE over the full-history solver with tabulated weights on the same mesh (row weights on graded meshes)
    base = {('uniform', r.N): r.wall_median_s for r in t[t.solver == 'full_uniform_tab'].itertuples()}
    base.update({('graded', r.N): r.wall_median_s for r in t[t.solver == 'full_graded_row'].itertuples()})
    t['time_ratio_full_over_this'] = [base.get((r.mesh, r.N), np.nan) / r.wall_median_s for r in t.itertuples()]
    for q in ('err_max_all_nodes', 'err_max_t_ge_T64', 'err_final_time'):
        t['order_' + q] = add_orders(t, ['solver'], q)
    # second, independent timing run, if present: reproducibility of times and ratios
    f2 = os.path.join(E, 'E4b_solver_run2.csv'); src2 = []
    if os.path.exists(f2):
        b2 = dc.read_csv(f2); m2 = dc.read_meta(f2); src2 = [pre + 'E4b_solver_run2.csv']
        base2 = {('uniform', r.N): r.wall_median_s for r in b2[b2.solver == 'full_uniform_tab'].itertuples()}
        base2.update({('graded', r.N): r.wall_median_s for r in b2[b2.solver == 'full_graded_row'].itertuples()})
        b2['time_ratio_full_over_this_run2'] = [base2.get((r.mesh, r.N), np.nan) / r.wall_median_s for r in b2.itertuples()]
        t = t.merge(b2[['solver', 'N', 'wall_median_s', 'wall_min_s', 'wall_max_s', 'time_ratio_full_over_this_run2']].rename(
            columns=dict(wall_median_s='wall_median_run2_s', wall_min_s='wall_min_run2_s', wall_max_s='wall_max_run2_s')), on=['solver', 'N'], how='left')
        t['wall_best_of_both_runs_s'] = np.fmin(t.wall_min_s, t.wall_min_run2_s)
        run2_note = (f" Columns *_run2: second complete timing run (system load before/after: {m2.get('system_load_percent_before')}% / "
                     f"{m2.get('system_load_percent_after')}%); wall_best_of_both_runs_s: smallest single time of both runs.")
    else:
        run2_note = ''
    write_table('Table_SOE.csv', t, [pre + 'E4b_solver.csv', pre + 'E4b_memory.csv'] + src2,
                'E4b: implicit L1 time stepping of 16 fractional Bloch systems (alpha=0.6, T=10). wall_*: median/min/max of n_repeats timed solves (s); '
                'err_*: absolute maximum-norm errors against the closed-form solution (all nodes n>=1 / nodes with t>=T/64 / t=T); absdiff/reldiff_to_full_same_mesh: '
                'max|x_SOE - x_full| over all nodes (absolute / divided by max|x_full|); soe_rel_kernel_err: REALISED relative kernel error on [delta,T]; '
                'peak_heap_MB: separate untimed tracemalloc run storing only the final state; time_ratio_full_over_this: wall(full history, tabulated weights on the '
                'uniform mesh, row weights on the graded mesh) / wall(this solver) at the same N. Timing conditions: ' + str(mb.get('timing_protocol', '')) +
                f" System load before/after the timing run: {mb.get('system_load_percent_before')}% / {mb.get('system_load_percent_after')}%." + run2_note)
    # matched accuracy: wall time each method needs to reach the accuracy of uniform SOE at its largest N
    rows = []
    refs = t[t.solver == 'soe_uniform'].sort_values('N')
    for Nref in (32768, int(refs.N.max())):
        rr = refs[refs.N == Nref].iloc[0]
        for q, label in (('err_max_all_nodes', 'maximum over all nodes'), ('err_max_t_ge_T64', 'maximum over nodes with t >= T/64'), ('err_final_time', 'final time')):
            target = rr[q]
            for s in ('soe_uniform', 'full_uniform_tab', 'full_graded_row', 'soe_graded'):
                g = t[t.solver == s].sort_values('N')
                w = loglog_interp(target, g[q].values, g.wall_median_s.values); Nn = loglog_interp(target, g[q].values, g.N.values)
                row = dict(error_measure=label, reference='soe_uniform N=%d' % Nref, target_error=target, solver=s, N_needed=Nn, wall_needed_s=w,
                           wall_reference_s=rr.wall_median_s, speedup_vs_reference=rr.wall_median_s / w if w == w else np.nan)
                if 'wall_best_of_both_runs_s' in t.columns:             # same analysis with the smallest single time of the two timing runs
                    wb = loglog_interp(target, g[q].values, g.wall_best_of_both_runs_s.values)
                    row.update(wall_needed_best_s=wb, wall_reference_best_s=rr.wall_best_of_both_runs_s, speedup_vs_reference_best=rr.wall_best_of_both_runs_s / wb if wb == wb else np.nan)
                rows.append(row)
    write_table('Table_SOE_matched.csv', pd.DataFrame(rows), [pre + 'E4b_solver.csv'],
                'matched accuracy: number of steps and wall time (log-log interpolation of the measured work-precision curves) that each solver needs to reach the error of '
                'the uniform-mesh SOE solver at N = 32768 and at its largest N; speedup_vs_reference = wall(reference)/wall(this solver); nan = target outside the measured range. '
                'wall_needed_s etc.: medians of timing run 1; *_best: smallest single time of both timing runs (present only if a second run is stored)')
    # kernel: number of modes and realised error
    k = ker.copy()

    def leading(d):
        # leading-order estimate of the article, (L / pi^2) [L / alpha + ln(T / delta)] with L = ln(1 / realised relative error)
        L = np.log(1.0 / d.max_rel_err.values)
        return L / math.pi ** 2 * (L / d.alpha.values + np.log(d.T_over_delta.values))
    kk = pd.concat([k.assign(window='generic'), win.assign(window='used', T_over_delta=win.N, M_estimate=np.nan, dy=np.nan)], ignore_index=True)
    rel = (kk.criterion == 'relative').values
    kk['M_leading_order'] = np.where(rel, leading(kk), np.nan)
    kk['M_over_leading_order'] = np.where(rel, kk.M.values / leading(kk), np.nan)
    write_table('Table_SOE_kernel.csv', kk,
                [pre + 'E4a_kernel.csv', pre + 'E4a_kernel_windows.csv'],
                'SOE kernel on [delta,T]: number of modes M and REALISED maximum relative error for the relative criterion of this version and for the legacy absolute '
                'criterion; window=used: the windows of B3/B4 (T=60) and E4b (T=10) with tol=1e-10. M_leading_order: (L/pi^2) [L/alpha + ln(T/delta)] with '
                'L = ln(1/max_rel_err), the leading term of the number of modes as the error tends to zero; M_over_leading_order: M divided by it (relative criterion)')
    # growth law of M: fit M = c0 + c1 L + c2 L^2 (L = ln(1/tol)) at fixed T/delta, and slope in ln(T/delta) at fixed tol
    rows = []
    kr = k[(k.criterion == 'relative') & (k['T'] == 1.0)]
    for a, g in kr.groupby('alpha'):
        g3 = g[g.T_over_delta == 1e3].sort_values('tol', ascending=False); L = np.log(1 / g3.tol.values)
        c = np.polyfit(L, g3.M.values, 2)
        g10 = g[g.tol == 1e-10].sort_values('T_over_delta'); sl = np.polyfit(np.log(g10.T_over_delta.values), g10.M.values, 1)[0]
        dy = g10.dy.values[0]
        ga = kk[rel & (kk.alpha == a) & (kk.window == 'generic')]
        rat = {t: ga[ga.tol == t].M_over_leading_order for t in sorted(ga.tol.unique(), reverse=True)}
        small = ga[ga.tol <= 1e-6].M_over_leading_order
        rows.append(dict(alpha=a, M_tol_1e4_to_1e12=' '.join(str(int(x)) for x in g3.M.values), second_differences=' '.join(str(int(x)) for x in np.diff(g3.M.values, 2)),
                         fit_c2_L2=c[0], fit_c1_L=c[1], fit_c0=c[2], predicted_c2_1_over_alpha_pi2=1 / (a * math.pi ** 2),
                         M_Tdelta_1e3_to_1e6_tol1e10=' '.join(str(int(x)) for x in g10.M.values), slope_dM_dln_Tdelta=sl, predicted_slope_1_over_dy=1 / dy,
                         M_over_leading_order_by_tol=' '.join(f'{rat[t].min():.2f}-{rat[t].max():.2f}' for t in rat),
                         M_over_leading_order_min_tol_le_1e6=small.min(), M_over_leading_order_max_tol_le_1e6=small.max()))
    write_table('Table_SOE_modes.csv', pd.DataFrame(rows), [pre + 'E4a_kernel.csv'],
                'growth of the number of modes (relative criterion, T=1): quadratic fit M = c2 L^2 + c1 L + c0 in L = ln(1/tol) at T/delta = 1e3 '
                '(leading coefficient 1/(alpha pi^2) expected), and slope dM/d ln(T/delta) at tol = 1e-10 (1/dy expected). M_over_leading_order: number of modes divided by '
                'the leading-order estimate (L/pi^2) [L/alpha + ln(T/delta)], L = ln(1/realised relative error), over T = 1, 10, 60 and T/delta = 1e3 ... 1e6: range for each '
                'tolerance 1e-4 ... 1e-12, and smallest and largest value for the tolerances 1e-6 ... 1e-12')
    write_table('Table_SOE_residual.csv', res.assign(time_ratio_full_over_soe=[np.nan if r['mode'] == 'full' else res[(res.N == r.N) & (res['mode'] == 'full')].wall_median_s.values[0] / r.wall_median_s
                                                                             for _, r in res.iterrows()]), [pre + 'E4c_residual.csv'],
                'E4c: forward+backward pass of the torch residual operator (8 channels, alpha=0.6): full causal convolution versus SOE recurrence; median/min/max of n_repeats')
    write_table('Table_SOE_dirac_chunk.csv', dch.assign(time_ratio_full_over_soe=[np.nan if r.solver.startswith('full') else dch[(dch.N == r.N) & (dch.solver == 'full_uniform_tab')].wall_median_s.values[0] / r.wall_median_s
                                                                               for r in dch.itertuples()]), [pre + 'E4d_dirac_chunk.csv'],
                'E4d: complete implicit L1 solves for 256 Dirac wave vectors on the uniform mesh of B4 (smooth solution): full history with tabulated weights versus SOE')
    t.to_csv(os.path.join(FD, 'FigDet_SOE_work_precision.csv'), index=False, lineterminator='\n')
    k.to_csv(os.path.join(FD, 'FigDet_SOE_kernel.csv'), index=False, lineterminator='\n')
    return t


# ====================================================================================================== B3
def agg_B3():
    p = os.path.join(DIR, 'B3_summary.csv'); b = dc.read_csv(p); src = ['results/raw/DIRAC_v2/B3_summary.csv']
    rows = []; conv = []
    for (a0, al), g in b.groupby(['a0', 'alpha'], sort=False):
        row = dict(a0=a0, alpha=al)
        if al == 1.0:
            d = g[g.method == 'DOP853'].copy(); d['rt'] = d.N_or_rtol.astype(float); d = d.sort_values('rt', ascending=False)
            r = g[g.method == 'Radau'].copy(); r['rt'] = r.N_or_rtol.astype(float); r = r.sort_values('rt', ascending=False)
            v = d.nc_final.values[-1]; err = max(abs(v - r.nc_final.values[-1]), abs(v - d.nc_final.values[-2]))
            eu = g[g.method == 'L1-euler'].copy(); eu['N'] = eu.N_or_rtol.astype(int); eu = eu.sort_values('N')
            row.update(nc_reference=v, err_est=err, nc_rep=fmt(v, max(err, 5e-11)), reference_method='DOP853 rtol 1e-12 (err: |DOP853-Radau| and tolerance change)',
                       nc_L1_richardson=np.nan, err_L1=np.nan, order_L1_finest=math.log2(abs(eu.nc_final.values[-3] - eu.nc_final.values[-2]) / abs(eu.nc_final.values[-2] - eu.nc_final.values[-1])),
                       nc_PI_richardson=np.nan, err_PI=np.nan, order_PI_finest=np.nan, abs_diff_L1_PI=np.nan,
                       nc_L1_N16384=eu[eu.N == 16384].nc_final.values[0], nc_L1_Nmax=eu.nc_final.values[-1], N_L1_max=int(eu.N.max()),
                       max_abs_full_minus_soe=np.nan, soe_M=0, soe_rel_kernel_err=0.0)
            for _, e in eu.iterrows():
                conv.append(dict(a0=a0, alpha=al, method='L1 (implicit Euler)', N=int(e.N), nc_final=e.nc_final, abs_err_vs_reference=abs(e.nc_final - v)))
            allrows = g
        else:
            s = g[g.method == 'L1-soe'].copy(); s['N'] = s.N_or_rtol.astype(int); s = s.sort_values('N')
            f = g[g.method == 'L1-full'].copy(); f['N'] = f.N_or_rtol.astype(int)
            q = g[g.method == 'PI-trapezoid'].copy(); q['N'] = q.N_or_rtol.astype(int); q = q.sort_values('N')
            vs, vq = s.nc_final.values, q.nc_final.values; pL = 2 - al
            RL = [ref.richardson(vs[i], vs[i + 1], pL) for i in range(len(vs) - 1)]
            RQ = [ref.richardson(vq[i], vq[i + 1], 2.0) for i in range(len(vq) - 1)]
            errL = abs(RL[-1] - RL[-2]); errQ = abs(RQ[-1] - RQ[-2])
            oL = math.log2(abs(vs[-3] - vs[-2]) / abs(vs[-2] - vs[-1])); oQ = math.log2(abs(vq[-3] - vq[-2]) / abs(vq[-2] - vq[-1]))
            dLQ = abs(RL[-1] - RQ[-1]); err = max(errL, dLQ)
            mfs = max(abs(f[f.N == n].nc_final.values[0] - s[s.N == n].nc_final.values[0]) for n in f.N.values)
            row.update(nc_reference=RL[-1], err_est=err, nc_rep=fmt(RL[-1], err), reference_method='L1-SOE, Richardson (order 2-alpha) from the two finest N; err = max(change of the extrapolate, |L1 - PI|)',
                       nc_L1_richardson=RL[-1], err_L1=errL, order_L1_finest=oL, nc_PI_richardson=RQ[-1], err_PI=errQ, order_PI_finest=oQ, abs_diff_L1_PI=dLQ,
                       nc_L1_N16384=s[s.N == 16384].nc_final.values[0], nc_L1_Nmax=vs[-1], N_L1_max=int(s.N.max()),
                       max_abs_full_minus_soe=mfs, soe_M=int(s.soe_M.values[-1]), soe_rel_kernel_err=s.soe_rel_kernel_err.max())
            # convergence table: errors against the most accurate value available (product-trapezoid extrapolate, error ~ err_PI)
            for i, e in enumerate(s.itertuples()):
                conv.append(dict(a0=a0, alpha=al, method='L1-soe', N=e.N, nc_final=e.nc_final, abs_err_vs_reference=abs(e.nc_final - RQ[-1]),
                                 richardson_with_previous=RL[i - 1] if i > 0 else np.nan, abs_err_of_richardson=abs(RL[i - 1] - RQ[-1]) if i > 0 else np.nan,
                                 observed_order=math.log2(abs(vs[i - 2] - vs[i - 1]) / abs(vs[i - 1] - vs[i])) if i > 1 else np.nan))
            for i, e in enumerate(q.itertuples()):
                conv.append(dict(a0=a0, alpha=al, method='PI-trapezoid', N=e.N, nc_final=e.nc_final, abs_err_vs_reference=abs(e.nc_final - RQ[-1]),
                                 richardson_with_previous=RQ[i - 1] if i > 0 else np.nan, abs_err_of_richardson=abs(RQ[i - 1] - RQ[-1]) if 0 < i < len(vq) - 1 else np.nan,
                                 observed_order=math.log2(abs(vq[i - 2] - vq[i - 1]) / abs(vq[i - 1] - vq[i])) if i > 1 else np.nan))
            allrows = g
        row.update(max_coherence=allrows.max_coherence.max(), min_purity=allrows.min_purity.min(), max_bloch_norm_minus_1=allrows.max_bloch_norm.max() - 1.0,
                   min_eigenvalue=allrows.min_eigenvalue.min())
        rows.append(row)
    T = pd.DataFrame(rows)
    write_table('Table_Dirac_B3.csv', T, src,
                'B3: final conduction population at kappa = 0 (valley xi=+1, zeta=0). nc_reference/err_est/nc_rep: recommended value, its error estimate and the rounded form. '
                'alpha<1: L1-SOE Richardson extrapolate (N = 65536, 131072; order 2-alpha), err_L1 = change against the extrapolate of the previous pair; '
                'nc_PI_richardson: independent product-trapezoidal rule (N = 16384, 32768; order 2). nc_L1_N16384: the unextrapolated value of the L1 scheme with '
                '16384 steps. max_coherence, min_purity, max_bloch_norm_minus_1, min_eigenvalue: extrema over all methods and all time steps')
    write_table('Table_Dirac_B3_convergence.csv', pd.DataFrame(conv), src,
                'B3: n_c(t_f) versus N for the implicit L1 scheme (SOE history) and the product-trapezoidal rule. abs_err_vs_reference: against the product-trapezoid Richardson '
                'extrapolate from N = 16384, 32768 (alpha < 1; its own error estimate is err_PI of Table_Dirac_B3.csv) or DOP853 (alpha = 1); richardson_with_previous: extrapolate '
                'from (N/2, N) with order 2-alpha (L1) or 2 (product trapezoid); abs_err_of_richardson: its error against the same reference; observed_order from three consecutive N')
    # trajectories for the figure
    tr = []
    for a0 in dcfg['B3']['a0_list']:
        for al in dcfg['B3']['alphas']:
            fp = os.path.join(DIR, f'B3_traj_a0{a0}_alpha{al}.npz')
            if os.path.exists(fp):
                z = np.load(fp); X = z['traj']
                tr.append(pd.DataFrame(dict(a0=a0, alpha=al, t=X[:, 0], nc=X[:, 4], coherence=X[:, 5], purity=X[:, 6])))
    if tr:
        pd.concat(tr).to_csv(os.path.join(FD, 'FigDet_B3_trajectories.csv'), index=False, lineterminator='\n')
    pd.DataFrame(conv).to_csv(os.path.join(FD, 'FigDet_B3_convergence.csv'), index=False, lineterminator='\n')
    return T


# ====================================================================================================== B4
def tag(alpha, zeta):
    return f'a{alpha}_z{zeta}'


def load_run(path):
    z = np.load(path, allow_pickle=False)
    return dict(S=z['S'], t=z['t'], kc=z['kappa_c'], kappa=z['kappa'], weight=z['weight'], level=z['level'], ncf=z['ncf'], meta=json.loads(str(z['meta'])))


def peak(t, y):
    """max |y(t)| with parabolic refinement between samples: (peak value, time, signed value)."""
    i = int(np.abs(y).argmax()); s = 1.0 if y[i] >= 0 else -1.0
    if 0 < i < len(y) - 1:
        y0, y1, y2 = s * y[i - 1], s * y[i], s * y[i + 1]; den = y0 - 2 * y1 + y2
        if den < 0:
            d = 0.5 * (y0 - y2) / den
            return y1 - 0.25 * (y0 - y2) * d, t[i] + d * (t[1] - t[0]), s
    return abs(y[i]), t[i], s


def cutoff_fit(kc, Y, kmin, terms):
    """Least-squares fit Y(kc) = c0 + c1/kc (+ c2/kc^2) over kc >= kmin along the last axis of Y: returns coefficients (terms, ...)."""
    m = kc >= kmin; x = 1.0 / kc[m]
    A = np.stack([x ** j for j in range(terms)], axis=1)
    Yv = np.moveaxis(Y[..., m], -1, 0).reshape(m.sum(), -1)
    c = np.linalg.lstsq(A, Yv, rcond=None)[0]
    return c.reshape((terms,) + Y.shape[:-1])


def b4_levels(alpha, zeta):
    """Dict N -> run for alpha < 1, {'ode': run} for alpha = 1, and the list of sources."""
    if alpha == 1.0:
        f = os.path.join(DIR, f'B4_{tag(1.0, zeta)}_ode.npz'); return {'ode': load_run(f)}, [dc.rel(f)]
    out = {}; src = []
    for N in dcfg['B4']['N_list']:
        f = os.path.join(DIR, f'B4_{tag(alpha, zeta)}_N{N}.npz'); out[N] = load_run(f); src.append(dc.rel(f))
    return out, src


def extrapolated(runs, alpha, key='S'):
    """(fine, coarse) Richardson extrapolates in N (order 2 - alpha) of the array `key`; for alpha = 1 the single ODE array twice."""
    if 'ode' in runs:
        return runs['ode'][key], runs['ode'][key]
    Ns = sorted(runs); p = 2 - alpha
    return (ref.richardson(runs[Ns[-2]][key], runs[Ns[-1]][key], p), ref.richardson(runs[Ns[-3]][key], runs[Ns[-2]][key], p))


def quantities(S, t, kc):
    """Derived quantities from the cumulative shell sums S (2, nt, nshell, nobs): dict of arrays over kappa_c."""
    q = {}
    q['N_K'] = S[0, -1, :, 0]; q['N_Kp'] = S[1, -1, :, 0]; q['N_tot'] = q['N_K'] + q['N_Kp']; q['dN'] = q['N_K'] - q['N_Kp']; q['asym'] = q['dN'] / q['N_tot']
    for name, ix, iy in (('band', 1, 2), ('intra', 3, 4), ('init', 5, 6)):
        q['Jx_' + name] = S[0, :, :, ix] + S[1, :, :, ix]; q['Jy_' + name] = S[0, :, :, iy] + S[1, :, :, iy]          # (nt, nshell)
    for c in ('x', 'y'):
        q[f'J{c}_inter'] = q[f'J{c}_band'] - q[f'J{c}_intra']
        for name in ('band', 'init', 'intra', 'inter'):
            pk = [peak(t, q[f'J{c}_{name}'][:, j]) for j in range(len(kc))]
            q[f'peak_J{c}_{name}'] = np.array([x[0] for x in pk]); q[f'tpeak_J{c}_{name}'] = np.array([x[1] for x in pk])
    return q


def grid_check():
    """Relative momentum-quadrature errors from the runs with halved spacings (same N or same ODE tolerance)."""
    rows = []; src = []
    for case in dcfg['B4check']['grid']['cases']:
        a, z = case['alpha'], case['zeta']
        ff = os.path.join(DIR, f'B4check_grid_{tag(a, z)}_' + ('ode' if a == 1.0 else f"N{dcfg['B4check']['grid']['N']}") + '.npz')
        fc = os.path.join(DIR, f'B4_{tag(a, z)}_' + ('ode' if a == 1.0 else f"N{dcfg['B4check']['grid']['N']}") + '.npz')
        if not (os.path.exists(ff) and os.path.exists(fc)):
            continue
        F, C = load_run(ff), load_run(fc); src += [dc.rel(ff), dc.rel(fc)]
        qf = quantities(F['S'], F['t'], F['kc']); qc = quantities(C['S'], C['t'], C['kc'])
        for j, k in enumerate(F['kc']):
            jc = list(C['kc']).index(k)
            for name in ('N_K', 'N_Kp', 'N_tot', 'dN', 'peak_Jx_band', 'peak_Jy_band', 'peak_Jx_init'):
                vf, vc = qf[name][j], qc[name][jc]
                if abs(vf) < 1e-8:                       # quantities that vanish by symmetry (rounding noise only)
                    continue
                rows.append(dict(alpha=a, zeta=z, levels_refined=case['levels'], kappa_c=k, quantity=name, value_spacing_h=vc, value_spacing_h_over_2=vf, abs_diff=abs(vc - vf),
                                 rel_diff=abs(vc - vf) / abs(vf)))
            dd = np.abs(qf['Jx_band'][:, j] - qc['Jx_band'][:, jc]); pkv = np.abs(qf['Jx_band'][:, j]).max(); tt = F['t']
            for lo, hi in ((-30.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 30.0)):
                m = (tt >= lo) & (tt <= hi)
                rows.append(dict(alpha=a, zeta=z, levels_refined=case['levels'], kappa_c=k, quantity=f'Jx_band(t), max over {lo:g}<=t<={hi:g}, relative to the peak',
                                 value_spacing_h=np.abs(qc['Jx_band'][m, jc]).max(), value_spacing_h_over_2=np.abs(qf['Jx_band'][m, j]).max(), abs_diff=dd[m].max(), rel_diff=dd[m].max() / pkv))
    return pd.DataFrame(rows), src


def time_check():
    """L1 + Richardson versus the product-trapezoidal rule on the coarse nested grid; DOP853 tolerance check."""
    ct = dcfg['B4check']['time']; rows = []; src = []
    for z in ct['zetas']:
        for a in ct['alphas']:
            try:
                L = {N: load_run(os.path.join(DIR, f'B4check_time_l1_{tag(a, z)}_N{N}.npz')) for N in ct['N_l1']}
                P = {N: load_run(os.path.join(DIR, f'B4check_time_pi_{tag(a, z)}_N{N}.npz')) for N in ct['N_pi']}
            except FileNotFoundError:
                continue
            src += [f'results/raw/DIRAC_v2/B4check_time_l1_{tag(a, z)}_N*.npz', f'results/raw/DIRAC_v2/B4check_time_pi_{tag(a, z)}_N*.npz']
            t = L[ct['N_l1'][0]]['t']; kc = L[ct['N_l1'][0]]['kc']; Nl = ct['N_l1']; Np = ct['N_pi']
            RL = {(Nl[i], Nl[i + 1]): ref.richardson(L[Nl[i]]['S'], L[Nl[i + 1]]['S'], 2 - a) for i in range(len(Nl) - 1)}
            RP = {(Np[i], Np[i + 1]): ref.richardson(P[Np[i]]['S'], P[Np[i + 1]]['S'], 2.0) for i in range(len(Np) - 1)}
            best = quantities(RP[(Np[-2], Np[-1])], t, kc); bestL = quantities(RL[(Nl[-2], Nl[-1])], t, kc)
            for j, k in enumerate(kc):
                for name in ('N_tot', 'dN', 'peak_Jx_band', 'peak_Jy_band'):
                    if abs(best[name][j]) < 1e-12:
                        continue
                    r = dict(alpha=a, zeta=z, kappa_c=k, quantity=name, PI_richardson=best[name][j], L1_richardson_finest=bestL[name][j])
                    for N in Nl:
                        r[f'L1_N{N}'] = quantities(L[N]['S'], t, kc)[name][j]
                    for (n1, n2), S in RL.items():
                        r[f'L1_R({n1},{n2})'] = quantities(S, t, kc)[name][j]
                    for N in Np:
                        r[f'PI_N{N}'] = quantities(P[N]['S'], t, kc)[name][j]
                    r['PI_R_change'] = abs(best[name][j] - quantities(RP[(Np[-3], Np[-2])], t, kc)[name][j])
                    main = (dcfg['B4']['N_list'][-2], dcfg['B4']['N_list'][-1]); prev = (dcfg['B4']['N_list'][-3], dcfg['B4']['N_list'][-2])
                    r['err_estimate_as_in_main_table'] = abs(r[f'L1_R({main[0]},{main[1]})'] - r[f'L1_R({prev[0]},{prev[1]})'])
                    r['true_err_of_main_extrapolate'] = abs(r[f'L1_R({main[0]},{main[1]})'] - best[name][j])
                    r['rel_true_err'] = r['true_err_of_main_extrapolate'] / abs(best[name][j])
                    rows.append(r)
    orow = []
    for z in ct['zetas']:
        try:
            O = {rt: load_run(os.path.join(DIR, f'B4check_ode_{tag(1.0, z)}_rtol{rt:g}.npz')) for rt in dcfg['B4check']['ode']['rtols']}
        except FileNotFoundError:
            continue
        src.append(f'results/raw/DIRAC_v2/B4check_ode_{tag(1.0, z)}_rtol*.npz')
        rts = sorted(O, reverse=True); t = O[rts[0]]['t']; kc = O[rts[0]]['kc']; bestq = quantities(O[rts[-1]]['S'], t, kc)
        for rt in rts[:-1]:
            q = quantities(O[rt]['S'], t, kc)
            for j, k in enumerate(kc):
                for name in ('N_tot', 'dN', 'peak_Jx_band', 'peak_Jy_band'):
                    if abs(bestq[name][j]) < 1e-8:
                        continue
                    orow.append(dict(zeta=z, rtol=rt, kappa_c=k, quantity=name, value=q[name][j], value_rtol_min=bestq[name][j], rel_diff=abs(q[name][j] - bestq[name][j]) / abs(bestq[name][j])))
    return pd.DataFrame(rows), pd.DataFrame(orow), src


def agg_B4():
    c = dcfg['B4']; kc = np.array(c['kappa_c_list']); src_all = []
    gc, gsrc = grid_check(); tc, oc, tsrc = time_check()
    if len(gc):
        write_table('Table_Dirac_B4_gridcheck.csv', gc, gsrc, 'momentum-quadrature check: main nested grid (spacing 0.125/0.25/0.5) versus all spacings halved, same time discretisation (N=2048 or DOP853)')
    if len(tc):
        write_table('Table_Dirac_B4_timecheck.csv', tc, tsrc,
                    'time-discretisation check on a coarse nested momentum grid (400 wave vectors per valley; quadrature not converged, identical sums): implicit L1 (SOE) with Richardson '
                    'extrapolation versus the independent product-trapezoidal rule (PI_richardson: N = 4096, 8192, order 2). err_estimate_as_in_main_table: |R(4096,8192) - R(2048,4096)|; '
                    'true_err_of_main_extrapolate: |R(4096,8192) - PI_richardson|')
    if len(oc):
        write_table('Table_Dirac_B4_odecheck.csv', oc, tsrc, 'alpha = 1: DOP853 tolerance check on the coarse nested grid (atol = 0.01 rtol)')

    def rel_grid(quantity_prefix, alpha_is_one, kcut):
        if not len(gc):
            return float('nan')
        # largest relative difference, over the checked cases of the same class (alpha = 1 / alpha < 1), at the same cutoff
        # (at the largest checked cutoff if kcut lies beyond it)
        g = gc[(gc.quantity.str.startswith(quantity_prefix)) & ((gc.alpha == 1.0) == alpha_is_one)]
        if not len(g):
            return float('nan')
        kmax_checked = g.kappa_c.max()
        out = g[g.kappa_c == min(kcut, kmax_checked)].rel_diff.max()
        if kcut > kmax_checked:                           # outer frames were refined only for one alpha < 1 case: use it as well
            h = gc[(gc.quantity.str.startswith(quantity_prefix)) & (gc.kappa_c == kcut)]
            if len(h):
                out = max(out, h.rel_diff.max())
        return float(out)

    def rel_ode(quantity):
        if not len(oc):
            return float('nan')
        g = oc[(oc.quantity == quantity) & (oc.rtol == 1e-10)]
        return float(g.rel_diff.max()) if len(g) else float('nan')
    rows = []; cur_rows = []; cut_rows = []; traces = []; fitrows = []
    fcl = os.path.join(DIR, 'current_cutoff_coefficient.csv')
    closed = dc.read_csv(fcl) if os.path.exists(fcl) else None
    fit_src = [dc.rel(fcl)] if closed is not None else []
    for zeta in c['zetas']:
        for alpha in c['alphas']:
            try:
                runs, src = b4_levels(alpha, zeta)
            except FileNotFoundError:
                print('missing B4 runs for', alpha, zeta); continue
            src_all += src
            r0 = runs[sorted(runs, key=str)[-1]] if 'ode' not in runs else runs['ode']
            t = r0['t']; Sf, Sc = extrapolated(runs, alpha)
            qf, qc = quantities(Sf, t, kc), quantities(Sc, t, kc)
            metas = [runs[k]['meta'] for k in runs]
            p = pars(None, zeta)
            a_t = dirac.pulse_A(t, p['a0'], p['sp'], zeta, p['chi'], p['cep'])
            # ---- cutoff extrapolation of the band-referenced current: J(kc) = J_inf + B/kc + C/kc^2
            ext = {}
            for comp in ('x', 'y'):
                Y = qf[f'J{comp}_band']; Yc = qc[f'J{comp}_band']
                c3 = cutoff_fit(kc, Y, 8.0, 3); c2 = cutoff_fit(kc, Y, 16.0, 2); c3b = cutoff_fit(kc, Y, 12.0, 3); c3c = cutoff_fit(kc, Yc, 8.0, 3)
                pk3 = peak(t, c3[0]); pk2 = peak(t, c2[0]); pk3b = peak(t, c3b[0]); pkc = peak(t, c3c[0])
                ext[comp] = dict(trace=c3[0], B=c3[1], C=c3[2], peak=pk3[0], tpeak=pk3[1], err_fit=max(abs(pk3[0] - pk2[0]), abs(pk3[0] - pk3b[0])), err_time=abs(pk3[0] - pkc[0]))
            for j, k in enumerate(kc):
                is1 = alpha == 1.0
                et = {n: (abs(qf[n][j] - qc[n][j]) if not is1 else abs(qf[n][j]) * (rel_ode(n if n in ('N_tot', 'dN') or n.startswith('peak') else 'N_tot') if True else 0.0))
                      for n in ('N_K', 'N_Kp', 'N_tot', 'dN', 'asym', 'peak_Jx_band', 'peak_Jy_band', 'peak_Jx_init', 'peak_Jy_init')}
                if is1:
                    et['asym'] = abs(qf['asym'][j]) * rel_ode('dN'); et['peak_Jx_init'] = abs(qf['peak_Jx_init'][j]) * rel_ode('peak_Jx_band'); et['peak_Jy_init'] = abs(qf['peak_Jy_init'][j]) * rel_ode('peak_Jy_band')
                eg = {n: abs(qf[n][j]) * rel_grid('N_' if n in ('N_K', 'N_Kp', 'N_tot') else 'dN' if n in ('dN', 'asym') else 'peak_J', is1, k) for n in et}
                row = dict(zeta=zeta, alpha=alpha, kappa_c=k)
                for n in ('N_K', 'N_Kp', 'N_tot', 'dN', 'asym', 'peak_Jx_band', 'peak_Jy_band'):
                    e = et[n] + (eg[n] if eg[n] == eg[n] else 0.0)
                    row[n] = qf[n][j]; row[n + '_err_time'] = et[n]; row[n + '_err_grid'] = eg[n]; row[n + '_err'] = e
                    row[n + '_rep'] = fmt(qf[n][j], e) if abs(qf[n][j]) > 1e-9 else '0'
                for comp in ('x', 'y'):
                    jt = int(np.argmin(np.abs(t - qf[f'tpeak_J{comp}_band'][j])))
                    row[f't_peak_J{comp}_band'] = qf[f'tpeak_J{comp}_band'][j]
                    row[f'J{comp}_intra_at_peak'] = qf[f'J{comp}_intra'][jt, j] * np.sign(qf[f'J{comp}_band'][jt, j]) if abs(qf[f'J{comp}_band'][jt, j]) > 0 else 0.0
                    row[f'J{comp}_inter_at_peak'] = qf[f'J{comp}_inter'][jt, j] * np.sign(qf[f'J{comp}_band'][jt, j]) if abs(qf[f'J{comp}_band'][jt, j]) > 0 else 0.0
                    row[f'peak_J{comp}_sea'] = peak(t, qf[f'J{comp}_init'][:, j] - qf[f'J{comp}_band'][:, j])[0]
                row['sea_analytic_coeff_times_kc_times_max_a'] = dirac.sea_current_square(k, np.abs(a_t[:, 0]).max())
                rows.append(row)
            # ---- row for the cutoff-extrapolated current
            row = dict(zeta=zeta, alpha=alpha, kappa_c=np.inf)
            for comp in ('x', 'y'):
                e = ext[comp]
                if alpha == 1.0:                                   # one DOP853 run: time error from the tolerance check
                    ro = rel_ode(f'peak_J{comp}_band'); e['err_time'] = e['peak'] * ro if ro == ro else float('nan')
                rg = rel_grid('peak_J', alpha == 1.0, 32.0); eq = e['peak'] * rg if rg == rg else 0.0
                tot = e['err_fit'] + (e['err_time'] if e['err_time'] == e['err_time'] else 0.0) + eq
                row[f'peak_J{comp}_band'] = e['peak']; row[f'peak_J{comp}_band_err_time'] = e['err_time']; row[f'peak_J{comp}_band_err_grid'] = eq
                row[f'peak_J{comp}_band_err_cutoff_fit'] = e['err_fit']
                row[f'peak_J{comp}_band_err'] = tot; row[f'peak_J{comp}_band_rep'] = fmt(e['peak'], tot) if e['peak'] > 1e-9 else '0'; row[f't_peak_J{comp}_band'] = e['tpeak']
            # valley contrast: dN(kc) = dN_inf + c2/kc^2 (fit over kc >= 6), density: logarithmic
            if abs(qf['dN'][-1]) > 1e-9:
                x = 1 / kc ** 2; m = kc >= 6
                cf = np.polyfit(x[m], qf['dN'][m], 1); cf2 = np.polyfit(x[kc >= 10], qf['dN'][kc >= 10], 1); cfc = np.polyfit(x[m], qc['dN'][m], 1)
                e = abs(cf[1] - cf2[1]) + abs(cf[1] - cfc[1])
                row.update(dN=cf[1], dN_err_time=abs(cf[1] - cfc[1]), dN_err_grid=abs(cf[1] - cf2[1]), dN_err=e, dN_rep=fmt(cf[1], e))
                fitrows.append(dict(zeta=zeta, alpha=alpha, quantity='dN', model='dN_inf + c/kc^2 (kc>=6)', limit=cf[1], limit_err=e, coefficient=cf[0], remainder_at_kc3=cf[0] / 9, remainder_at_kc8=cf[0] / 64))
            rows.append(row)
            for comp in ('x', 'y'):
                e = ext[comp]
                if e['peak'] < 1e-9:
                    continue
                jt = int(np.argmin(np.abs(t - e['tpeak'])))
                da = dirac.pulse_dA(t, p['a0'], p['sp'], zeta, p['chi'], p['cep']); ic = 0 if comp == 'x' else 1
                dda = np.gradient(da[:, ic], t)
                G1, G2, req = dirac.rates(p)
                Ban = math.sqrt(2) / (2 * math.pi ** 2) * (dda + G2 * da[:, ic])               # alpha = 1 prediction of the 1/kc coefficient
                frow = dict(zeta=zeta, alpha=alpha, quantity=f'J{comp}_band', model='J_inf + B/kc + C/kc^2 (kc>=8)', limit=e['peak'], limit_err=e['err_fit'] + e['err_time'],
                            coefficient=e['B'][jt] * np.sign(e['trace'][jt]), remainder_at_kc3=(qf[f'peak_J{comp}_band'][list(kc).index(3.0)] - e['peak']),
                            remainder_at_kc8=(qf[f'peak_J{comp}_band'][list(kc).index(8.0)] - e['peak']),
                            B_fit_rms_over_t=float(np.sqrt(np.mean(e['B'] ** 2))), B_alpha1_analytic_rms_over_t=float(np.sqrt(np.mean(Ban ** 2))),
                            B_fit_minus_analytic_rms=float(np.sqrt(np.mean((e['B'] - Ban) ** 2))))
                if closed is not None:                             # closed form of the coefficient for this alpha (fracsp.dirac_cutoff)
                    g = closed[(closed.zeta == zeta) & (closed.alpha == alpha)]
                    Bcf = np.interp(t, g.t.values, g[f'B_{comp}'].values)
                    B4t = cutoff_fit(kc, qf[f'J{comp}_band'], 6.0, 4)[1]             # alternative fit with the term 1/kc^3 over kc >= 6
                    rms_cf = float(np.sqrt(np.mean(Bcf ** 2)))
                    d3 = float(np.sqrt(np.mean((e['B'] - Bcf) ** 2))); d4 = float(np.sqrt(np.mean((B4t - Bcf) ** 2)))
                    frow.update(B_closed_form_rms_over_t=rms_cf, B_closed_form_at_peak=float(Bcf[jt] * np.sign(e['trace'][jt])),
                                B_fit_minus_closed_form_rms=d3, B_fit4_minus_closed_form_rms=d4,
                                B_fit_rel_diff_to_closed_form=d3 / rms_cf, B_fit4_rel_diff_to_closed_form=d4 / rms_cf)
                fitrows.append(frow)
            # ---- figure data: traces
            for j, k in enumerate(kc):
                if k in (3.0, 8.0, 32.0):
                    traces.append(pd.DataFrame(dict(zeta=zeta, alpha=alpha, kappa_c=k, t=t, Jx_band=qf['Jx_band'][:, j], Jy_band=qf['Jy_band'][:, j], Jx_intra=qf['Jx_intra'][:, j],
                                                    Jy_intra=qf['Jy_intra'][:, j], Jx_initref=qf['Jx_init'][:, j], Jy_initref=qf['Jy_init'][:, j],
                                                    Jx_band_time_err=np.abs(qf['Jx_band'][:, j] - qc['Jx_band'][:, j]), a_x=a_t[:, 0], a_y=a_t[:, 1])))
            traces.append(pd.DataFrame(dict(zeta=zeta, alpha=alpha, kappa_c=np.inf, t=t, Jx_band=ext['x']['trace'], Jy_band=ext['y']['trace'], Jx_intra=np.nan, Jy_intra=np.nan,
                                            Jx_initref=np.nan, Jy_initref=np.nan, Jx_band_time_err=np.nan, a_x=a_t[:, 0], a_y=a_t[:, 1])))
            # ---- cutoff dependence (long format) and logarithmic slopes
            for j, k in enumerate(kc):
                cut_rows.append(dict(zeta=zeta, alpha=alpha, kappa_c=k, N_K=qf['N_K'][j], N_Kp=qf['N_Kp'][j], N_tot=qf['N_tot'][j], N_tot_err_time=abs(qf['N_tot'][j] - qc['N_tot'][j]),
                                     dN=qf['dN'][j], asym=qf['asym'][j], peak_Jx_band=qf['peak_Jx_band'][j], peak_Jy_band=qf['peak_Jy_band'][j],
                                     peak_Jx_band_err_time=abs(qf['peak_Jx_band'][j] - qc['peak_Jx_band'][j]), peak_Jx_initref=qf['peak_Jx_init'][j], peak_Jy_initref=qf['peak_Jy_init'][j],
                                     slope_dNtot_dlnkc_from_previous=(qf['N_tot'][j] - qf['N_tot'][j - 1]) / math.log(kc[j] / kc[j - 1]) if j > 0 else np.nan,
                                     slope_err_time=abs((qf['N_tot'][j] - qf['N_tot'][j - 1]) - (qc['N_tot'][j] - qc['N_tot'][j - 1])) / math.log(kc[j] / kc[j - 1]) if j > 0 else np.nan))
            cur_rows.append(dict(zeta=zeta, alpha=alpha, n_points_per_valley=metas[0]['n_points_per_valley'], N_levels=' '.join(str(k) for k in runs),
                                 soe_M=' '.join(str(m.get('M', 0)) for m in metas), soe_rel_kernel_err_max=max(m.get('soe_rel_err', 0.0) for m in metas),
                                 max_bloch_norm_minus_1=max(max(m['max_bloch_norm']) for m in metas) - 1.0, min_purity=min(min(m['min_purity']) for m in metas),
                                 min_eigenvalue=min(min(m['min_eigenvalue']) for m in metas), wall_s=sum(m['wall_s'] for m in metas), cpu_s=sum(m['cpu_s'] for m in metas)))
    T = pd.DataFrame(rows)
    write_table('Table_Dirac_B4.csv', T, src_all + gsrc + tsrc,
                'B4: both valleys, a0=1, nested momentum grid (spacing 0.125 for |kappa|_inf<=8, 0.25 to 16, 0.5 to 32), densities at t_f within the square |kappa_x|,|kappa_y|<=kappa_c, '
                'units (omega0/v_F)^2 incl. g_s=2; dN = N_K - N_Kp; asym = dN/N_tot; peak_J*_band: maximum over time of the sheet current referred to the instantaneous filled valence band '
                '(units e omega0^2/v_F). alpha<1: Richardson extrapolation in N (4096, 8192; order 2-alpha), *_err_time = change against the extrapolate from (2048, 4096); alpha=1: DOP853 '
                '(rtol 1e-10), *_err_time from the tolerance check. *_err_grid: momentum-quadrature error from the halved-spacing runs (largest relative difference of the same kind of '
                'quantity). *_err = sum; *_rep = value rounded accordingly. kappa_c = inf: cutoff-extrapolated current (fit J_inf + B/kc + C/kc^2 over kc>=8; '
                '*_err_cutoff_fit = spread of two alternative fits, included in *_err) and valley contrast (fit dN_inf + c/kc^2; its fit spread is stored in dN_err_grid). peak_J*_sea: '
                'maximum over time of the difference between the current referred to the initial state, tr[j(rho - rho(t0))] (peak_J*_initref in Table_Dirac_B4_cutoff.csv), and the '
                'band-referenced current (current of the shifted filled band, linear in kappa_c); J*_intra/inter_at_peak: decomposition at the peak time')
    write_table('Table_Dirac_B4_cutoff.csv', pd.DataFrame(cut_rows), src_all, 'B4: all quantities versus the cutoff kappa_c (Richardson extrapolated for alpha<1), and the logarithmic slope of the total density between consecutive cutoffs')
    write_table('Table_Dirac_B4_fits.csv', pd.DataFrame(fitrows), src_all + fit_src,
                'cutoff extrapolations: limit, fitted coefficient of the leading remainder (current: B in J_inf + B/kc at the peak time, signed along the peak; valley contrast: c in c/kc^2), '
                'remainders at kc = 3 and 8. B_alpha1_analytic: sqrt(2)/(2 pi^2) (d^2a/dt^2 + G2 da/dt), the alpha = 1 prediction for the square patch (rms over the time window). '
                'B_closed_form: coefficient of 1/kc from the terms of order kappa^-3 of the large-momentum expansion for the alpha of the row (fracsp.dirac_cutoff, '
                'current_cutoff_coefficient.csv), rms over the time window and value at the peak time; B_fit_minus_closed_form_rms: rms over the time window of its difference '
                'to the fitted coefficient; B_fit4_minus_closed_form_rms: the same for the fit J_inf + B/kc + C/kc^2 + D/kc^3 over kc >= 6; '
                'B_fit_rel_diff_to_closed_form, B_fit4_rel_diff_to_closed_form: these two differences divided by B_closed_form_rms_over_t')
    write_table('Table_Dirac_B4_runs.csv', pd.DataFrame(cur_rows), src_all, 'B4 runs: resolution, realised SOE kernel error, extrema of the Bloch norm (positivity), cost')
    if traces:
        pd.concat(traces).to_csv(os.path.join(FD, 'FigDet_B4_current_traces.csv'), index=False, lineterminator='\n')
    pd.DataFrame(cut_rows).to_csv(os.path.join(FD, 'FigDet_B4_cutoff.csv'), index=False, lineterminator='\n')
    # maps for the figure (valley K, zeta = 1, level 0)
    maps = {}
    for alpha in (1.0, 0.8, 0.9):
        try:
            runs, src = b4_levels(alpha, 1.0)
        except FileNotFoundError:
            continue
        r0 = runs[list(runs)[0]]; nf, ncs = extrapolated(runs, alpha, 'ncf')
        m0 = r0['level'] == 0; kap = r0['kappa'][m0]; n0 = int(round(math.sqrt(m0.sum())))
        o = np.lexsort((kap[:, 1], kap[:, 0]))
        maps[f'nc_K_alpha{alpha}'] = nf[0][m0][o].reshape(n0, n0); maps[f'nc_Kp_alpha{alpha}'] = nf[1][m0][o].reshape(n0, n0)
        maps['kappa_axis'] = np.unique(kap[:, 0])
    if maps:
        np.savez_compressed(os.path.join(FD, 'FigDet_B4_maps_zeta1.npz'), **maps)
    return T


# ====================================================================================================== tail
def shell_profile(kappa, ncf_valley_mean, edges):
    r = np.hypot(kappa[:, 0], kappa[:, 1]); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        out.append((math.sqrt(lo * hi), ncf_valley_mean[m].mean() if m.any() else np.nan, int(m.sum())))
    return np.array(out)


def agg_tail():
    rows = []; src = []; t0, tf = dcfg['window']; taub = dcfg['taubar_c']

    def add(**kw):
        base = dict(study='', zeta=np.nan, phi_deg=np.nan, xi=np.nan, alpha=np.nan, case='', K=np.nan, t_f=tf, quantity='', value=np.nan, err=np.nan, rep='', analytic=np.nan,
                    ratio_to_analytic=np.nan, local_slope=np.nan, method='', note='')
        base.update(kw); rows.append(base)
    # ---- rays
    f = os.path.join(DIR, 'tail_rays.csv')
    if os.path.exists(f):
        d = dc.read_csv(f); src.append(dc.rel(f))
        for (z, ph, xi, al), g in d.groupby(['zeta', 'phi_deg', 'xi', 'alpha'], sort=False):
            C = dirac.tail_constant(al, pars(None, z), math.radians(ph), taub, (t0, tf))
            prevK = prevV = None
            for K, gk in g.groupby('K'):
                gp = gk[gk.method == 'PI-trapezoid'].sort_values('N'); gl = gk[gk.method == 'L1-soe'].sort_values('N')
                vp, vl = gp.nc_final.values, gl.nc_final.values
                RP = ref.richardson(vp[-2], vp[-1], 2.0); eP = abs(RP - ref.richardson(vp[-3], vp[-2], 2.0))
                rl = gl.N.values[-1] / gl.N.values[-2]
                RL = ref.richardson(vl[-2], vl[-1], 2 - al, rl); eL = abs(RL - ref.richardson(vl[-3], vl[-2], 2 - al, rl))
                err = max(eP, abs(RP - RL))
                sl = math.log(RP / prevV) / math.log(K / prevK) if prevK else np.nan
                lead = C > 1e-12
                add(study='rays', zeta=z, phi_deg=ph, xi=xi, alpha=al, K=K, quantity='K^2 n_c' if lead else 'K^4 n_c', value=(K ** 2 if lead else K ** 4) * RP, err=(K ** 2 if lead else K ** 4) * err,
                    rep=fmt((K ** 2 if lead else K ** 4) * RP, (K ** 2 if lead else K ** 4) * err), analytic=C if lead else np.nan, ratio_to_analytic=K ** 2 * RP / C if lead else np.nan,
                    local_slope=sl, method='PI-trapezoid Richardson (N=16384,32768); err = max(change of extrapolate, |PI - L1 Richardson|)',
                    note=f'n_c={RP:.6e}; L1-SOE Richardson (N=32768,131072): {RL:.6e}; raw L1 N=8192: {vl[0]:.6e}; raw PI N=8192: {vp[0]:.6e}')
                prevK, prevV = K, RP
    # ---- dissipation
    f = os.path.join(DIR, 'tail_dissipation.csv')
    if os.path.exists(f):
        d = dc.read_csv(f); src.append(dc.rel(f))
        for (case, al, K), g in d.groupby(['case', 'alpha', 'K'], sort=False):
            g = g.sort_values('N'); pw = int(g.power.values[0])
            if al < 1:
                v = g.nc_final.values; R = ref.richardson(v[-2], v[-1], 2.0); e = abs(R - v[-1])
            else:
                R = g.nc_final.values[-1]; e = 2e-5 * abs(R)
            add(study='dissipation', zeta=1.0, phi_deg=0.0, xi=1, alpha=al, case=case, K=K, quantity=f'K^{pw} n_c', value=K ** pw * R, err=K ** pw * e, rep=fmt(K ** pw * R, K ** pw * e),
                analytic=g.analytic_constant.values[0], ratio_to_analytic=K ** pw * R / g.analytic_constant.values[0] if g.analytic_constant.values[0] > 0 else np.nan,
                method='PI-trapezoid Richardson (N=16384,32768); err = |extrapolate - finest|' if al < 1 else 'DOP853 deviation form',
                note=f'Gdown={g.Gdown.values[0]}, Gphi={g.Gphi.values[0]}')
    # ---- final time
    f = os.path.join(DIR, 'tail_tf.csv'); f2 = os.path.join(DIR, 'tail_tf_constants.csv')
    if os.path.exists(f) and os.path.exists(f2):
        d = dc.read_csv(f); cst = dc.read_csv(f2); src += [dc.rel(f), dc.rel(f2)]
        for (al, tt, K), g in d.groupby(['alpha', 't', 'K'], sort=False):
            gp = g[g.method == 'PI-trapezoid'].sort_values('N'); gl = g[g.method == 'L1-soe'].sort_values('N')
            RP = ref.richardson(gp.nc.values[-2], gp.nc.values[-1], 2.0); vl = gl.nc.values
            RL = ref.richardson(vl[-2], vl[-1], 2 - al); err = max(abs(RP - RL), abs(RL - ref.richardson(vl[-3], vl[-2], 2 - al)))
            C = cst[(cst.alpha == al) & (cst.tf == tt)].C_alpha_analytic.values[0]
            add(study='final_time', zeta=1.0, phi_deg=0.0, xi=1, alpha=al, K=K, t_f=tt, quantity='K^2 n_c', value=K * K * RP, err=K * K * err, rep=fmt(K * K * RP, K * K * err), analytic=C,
                ratio_to_analytic=K * K * RP / C, method='PI-trapezoid Richardson; err = max(|PI - L1 Richardson|, change of the L1 extrapolate)', note=f'n_c={RP:.6e}')
        for r in cst.itertuples():
            add(study='final_time_constant', zeta=1.0, phi_deg=0.0, alpha=r.alpha, t_f=r.tf, quantity='C_alpha(t_f)', value=r.C_alpha_analytic, analytic=r.C_alpha_analytic, method='analytical')
    # ---- alpha = 1
    f = os.path.join(DIR, 'tail_alpha1.csv')
    if os.path.exists(f):
        d = dc.read_csv(f); src.append(dc.rel(f))
        for r in d.itertuples():
            lead = r.analytic_D > 1e-14
            add(study='alpha1', zeta=r.zeta, phi_deg=r.phi_deg, xi=r.xi, alpha=1.0, K=r.K, quantity='K^4 n_c' if lead else 'K^6 n_c', value=r.K4_nc if lead else r.K6_nc, err=2e-5 * abs(r.K4_nc if lead else r.K6_nc),
                rep=f'{(r.K4_nc if lead else r.K6_nc):.5g}', analytic=r.analytic_D if lead else np.nan, ratio_to_analytic=r.K4_nc / r.analytic_D if lead else np.nan, local_slope=r.local_slope,
                method='DOP853 deviation form (rtol 1e-12)', note=f'n_c={r.nc_final:.6e}')
    # ---- constants
    f = os.path.join(DIR, 'tail_constants.csv')
    if os.path.exists(f):
        d = dc.read_csv(f); src.append(dc.rel(f))
        for r in d.itertuples():
            if r.alpha < 1:
                for q, v in (('C_alpha(phi=0)', r.C_phi0), ('C_alpha(phi=90deg)', r.C_phi90), ('C_alpha angular mean', r.C_angular_mean), ('2<C>/pi = asymptotic dN_tot/dln(kappa_c)', r.slope_dNtot_dlnkc),
                             ('<C>/(1-alpha)', r.C_mean_over_1_minus_alpha), ('<C> without dissipation', r.C_mean_no_dissipation)):
                    if v == v:
                        add(study='constants', zeta=r.zeta, alpha=r.alpha, quantity=q, value=v, analytic=v, method='analytical (Gauss quadrature of the closed formula)')
            else:
                add(study='constants', zeta=r.zeta, alpha=1.0, quantity='D(phi=0) = lim K^4 n_c', value=r.C_phi0, analytic=r.C_phi0, method='analytical')
                add(study='constants', zeta=r.zeta, alpha=1.0, quantity='D(phi=90deg) = lim K^4 n_c', value=r.C_phi90, analytic=r.C_phi90, method='analytical')
    # ---- converged logarithmic slopes and shell-mean exponents from the B4 runs
    c = dcfg['B4']; kc = np.array(c['kappa_c_list']); prof = []
    for zeta in c['zetas']:
        for alpha in c['alphas']:
            try:
                runs, s = b4_levels(alpha, zeta)
            except FileNotFoundError:
                continue
            src += s
            r0 = runs[list(runs)[0]]; t = r0['t']; Sf, Sc = extrapolated(runs, alpha); qf, qc = quantities(Sf, t, kc), quantities(Sc, t, kc)
            if alpha < 1:
                Cm = np.mean([dirac.tail_constant(alpha, pars(None, zeta), ph, taub, (t0, tf)) for ph in np.linspace(0, math.pi, 24, endpoint=False)]); an = 2 * Cm / math.pi
                for (k1, k2) in ((7.0, 8.0), (8.0, 16.0), (16.0, 32.0), (24.0, 32.0)):
                    i1, i2 = list(kc).index(k1), list(kc).index(k2)
                    sl = (qf['N_tot'][i2] - qf['N_tot'][i1]) / math.log(k2 / k1); slc = (qc['N_tot'][i2] - qc['N_tot'][i1]) / math.log(k2 / k1)
                    add(study='log_slope', zeta=zeta, alpha=alpha, K=k2, quantity=f'dN_tot/dln(kappa_c) between {k1:g} and {k2:g}', value=sl, err=abs(sl - slc), rep=fmt(sl, abs(sl - slc)), analytic=an,
                        ratio_to_analytic=sl / an, method='B4 nested grid, Richardson in N', note='analytic: 2<C>/pi')
                    if (k1, k2) == (7.0, 8.0):
                        Ns = sorted(runs); raw = (runs[Ns[0]]['S'][:, -1, i2, 0].sum() - runs[Ns[0]]['S'][:, -1, i1, 0].sum()) / math.log(k2 / k1)
                        add(study='log_slope', zeta=zeta, alpha=alpha, K=k2, quantity=f'dN_tot/dln(kappa_c) between 7 and 8, unextrapolated N={Ns[0]}', value=raw, analytic=an, ratio_to_analytic=raw / an,
                            method=f'B4 nested grid, L1 N={Ns[0]} (resolution of the first manuscript version)')
            # valley contrast remainder
            if abs(qf['dN'][-1]) > 1e-9:
                for k in (3.0, 6.0, 8.0, 16.0, 32.0):
                    i = list(kc).index(k)
                    add(study='valley_contrast', zeta=zeta, alpha=alpha, K=k, quantity='dN(kappa_c) = N_K - N_Kp', value=qf['dN'][i], err=abs(qf['dN'][i] - qc['dN'][i]), rep=fmt(qf['dN'][i], abs(qf['dN'][i] - qc['dN'][i])),
                        method='B4 nested grid', note=f"asymmetry dN/N_tot = {qf['asym'][i]:.4f}")
            # shell-averaged, valley-averaged population versus |kappa|
            nf, ncs = extrapolated(runs, alpha, 'ncf'); vm = 0.5 * (nf[0] + nf[1]); kap = r0['kappa']
            edges = np.geomspace(1.0, 32.0, 31); P = shell_profile(kap, vm, edges)
            for row in P:
                prof.append(dict(zeta=zeta, alpha=alpha, K=row[0], nc_shell_mean=row[1], n_cells=int(row[2])))
            for lo, hi in ((3.0, 8.0), (8.0, 16.0), (16.0, 32.0)):
                r = np.hypot(kap[:, 0], kap[:, 1]); m = (r >= lo) & (r < hi) & (vm > 0)
                # bin first (equal weight per log-radius bin), then fit
                eb = np.geomspace(lo, hi, 9); Pb = shell_profile(kap, vm, eb); ok = np.isfinite(Pb[:, 1]) & (Pb[:, 1] > 0)
                ex = np.polyfit(np.log(Pb[ok, 0]), np.log(Pb[ok, 1]), 1)[0]
                add(study='shell_exponent', zeta=zeta, alpha=alpha, K=hi, quantity=f'd ln<n_c>/d ln|kappa| on [{lo:g},{hi:g})', value=ex, analytic=-2.0 if alpha < 1 else -4.0,
                    method='B4 nested grid; valley- and angle-averaged n_c(t_f) in 8 logarithmic radial bins', note='pre-asymptotic for small |kappa|')
    # analytical C_alpha(t_f) on a dense set of final times, with and without dissipation (figure data; cached)
    fp = os.path.join(FD, 'FigDet_tail_constant_vs_tf.csv')
    if not os.path.exists(fp):
        crow = []
        for al in dcfg['tail']['alphas']:
            for tfx in (30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0, 330.0, 450.0):
                crow.append(dict(alpha=al, tf=tfx, C_paper=dirac.tail_constant(al, pars(None, 1.0), 0.0, taub, (t0, tfx)),
                                 C_no_dissipation=dirac.tail_constant(al, pars(None, 1.0, Gdown=0.0, Gphi=0.0), 0.0, taub, (t0, tfx))))
        pd.DataFrame(crow).to_csv(fp, index=False, lineterminator='\n')
    T = pd.DataFrame(rows)
    write_table('Table_Dirac_tail.csv', T, src,
                'high-momentum tail of n_c(kappa, t_f). study=rays: K^2 n_c along rays (phi = polar angle of kappa; polarisation axis of the linear pulse: phi=0) against the analytical constant '
                'C_alpha(phi) (K^4 n_c where the leading term vanishes); local_slope = d ln n_c/d ln K from the previous K. dissipation: rates switched off. final_time: dependence on t_f. '
                'alpha1: K^4 n_c against D(phi). constants: analytical constants. log_slope: dN_tot/d ln kappa_c from the momentum-resolved runs against 2<C>/pi. valley_contrast: dN(kappa_c). '
                'shell_exponent: fitted exponent of the angle-averaged population')
    if prof:
        pd.DataFrame(prof).to_csv(os.path.join(FD, 'FigDet_tail_shell_profile.csv'), index=False, lineterminator='\n')
    T[T.study.isin(['rays', 'alpha1', 'dissipation', 'final_time'])].to_csv(os.path.join(FD, 'FigDet_tail_rays.csv'), index=False, lineterminator='\n')
    return T


# ====================================================================================================== positivity
def agg_positivity():
    f = os.path.join(RAW, 'E5_positivity', 'E5_l1_positivity.csv'); d = dc.read_csv(f); src = [dc.rel(f)]
    rows = []
    for (dd, al, mesh), g in d.groupby(['d', 'alpha', 'mesh'], sort=True):
        rows.append(dict(system='random time-dependent GKSL generators', d=dd, alpha=al, mesh=mesh, grading_r=g.grading_r.values[0], n_configurations=len(g), n_trajectories=int(g.n_states.sum()),
                         N=int(g.N.values[0]), min_eigenvalue=g.min_eig.min(), max_trace_defect=g.max_trace_defect.max(), max_hermiticity_defect=g.max_herm_defect.max(),
                         min_choi_eig_solution_map=g.min_choi_eig_solution_map.min(), min_choi_eig_one_step=g.min_choi_eig_one_step.min(), min_convex_weight=g.min_convex_weight.min(),
                         max_abs_weight_sum_minus_1=g.max_abs_weight_sum_minus_1.max(), min_eigenvalue_soe=g.min_eig_soe.min(), max_bloch_norm_minus_1=np.nan, note='kinds: ' + ' '.join(sorted(g.kind.unique())) +
                         '; strengths: ' + ' '.join(str(x) for x in sorted(g.strength.unique()))))
    # Dirac trajectories
    f3 = os.path.join(DIR, 'B3_summary.csv')
    if os.path.exists(f3):
        b = dc.read_csv(f3); src.append(dc.rel(f3))
        for (al, meth), g in b[b.method.isin(['L1-soe', 'L1-full', 'L1-euler'])].groupby(['alpha', 'method']):
            rows.append(dict(system='Dirac B3 (kappa=0), ' + meth, d=2, alpha=al, mesh='uniform', grading_r=1.0, n_configurations=len(g), n_trajectories=len(g), N=int(g.N_or_rtol.astype(int).max()),
                             min_eigenvalue=g.min_eigenvalue.min(), max_bloch_norm_minus_1=g.max_bloch_norm.max() - 1.0, note='all time steps; a0 = 0.2 and 1; pure initial state (min eigenvalue 0 at t0)'))
    for zeta in dcfg['B4']['zetas']:
        for alpha in dcfg['B4']['alphas']:
            try:
                runs, s = b4_levels(alpha, zeta)
            except FileNotFoundError:
                continue
            src += s; metas = [runs[k]['meta'] for k in runs]
            rows.append(dict(system=f'Dirac B4, zeta={zeta}, ' + ('DOP853 (output times only)' if alpha == 1.0 else 'L1-soe'), d=2, alpha=alpha, mesh='uniform', grading_r=1.0, n_configurations=len(metas),
                             n_trajectories=2 * metas[0]['n_points_per_valley'] * len(metas), N=max(m['N'] for m in metas), min_eigenvalue=min(min(m['min_eigenvalue']) for m in metas),
                             max_bloch_norm_minus_1=max(max(m['max_bloch_norm']) for m in metas) - 1.0, note='extrema over all wave vectors, both valleys and all time steps; pure initial states'))
    write_table('Table_L1_positivity.csv', pd.DataFrame(rows), src,
                'positivity of the implicit L1 step. Random generators: smallest eigenvalue / largest trace and Hermiticity defects over all time steps and initial states (4 pure, 2 mixed per '
                'configuration); Choi matrix of the discrete solution map and of the one-step resolvent; smallest convex weight theta_{n,m}; SOE variant on the same mesh. '
                'Dirac rows: smallest eigenvalue (1-|r|)/2 and max|r|-1 along the trajectories (the initial states are pure, so 0 is attained at t0)')
    return pd.DataFrame(rows)


def agg_examples():
    """Numbers of the article that follow from closed forms, without raw input.

    (b) Time-independent generator -(i/2) omega [sigma_z, .]: the coherence is multiplied by E_alpha(-i x),
        x = omega c (t - t_0)^alpha. For alpha = 0.8 its modulus has a local minimum followed by a maximum; between the
        two times the coherence grows by their ratio, so that the solution map is not divisible.
    (c) Pure dephasing with the rate gamma = 3 on [0, 1) and -0.05 on [1, 40] (c = 1, t_0 = 0): the coherence factor y
        solves D^alpha y = -gamma(t) y, y(0) = 1. By the variation-of-constants formula with the rate -0.05,
        y(t) = E_alpha(0.05 t^alpha) - 3.05 int_0^1 (t - s)^(alpha-1) E_{alpha,alpha}(0.05 (t - s)^alpha) E_alpha(-3 s^alpha) ds, t >= 1.
        With the rate switched off instead (gamma = 0 on [1, infinity)) the generator vanishes for t >= 1 and
        y(t) = 1 - (3 / Gamma(alpha)) int_0^1 (t - s)^(alpha-1) E_alpha(-3 s^alpha) ds, which tends to 1: the coherence
        returns to its initial value.
    (d) Nodes of the graded mesh tau_n = (n/N)^r, N = 128, r = 3, in the initial layer 0 < tau < 0.01."""
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    from fracsp import exact
    rows = []
    al = 0.8
    mod = lambda x: abs(exact.ml_complex(-1j * x, al))
    lo = minimize_scalar(mod, bounds=(6.5, 8.0), method='bounded', options=dict(xatol=1e-10))
    hi = minimize_scalar(lambda x: -mod(x), bounds=(8.0, 9.5), method='bounded', options=dict(xatol=1e-10))
    ex = 'coherence factor |E_alpha(-i x)|, alpha = 0.8 (non-divisibility)'
    rows += [dict(example=ex, quantity='x at the local minimum', value=float(lo.x)), dict(example=ex, quantity='modulus at the local minimum', value=float(lo.fun)),
             dict(example=ex, quantity='x at the next maximum', value=float(hi.x)), dict(example=ex, quantity='modulus at the next maximum', value=float(-hi.fun)),
             dict(example=ex, quantity='ratio maximum / minimum', value=float(-hi.fun / lo.fun))]
    for a in (0.5, 0.8):
        E = lambda z, b=1.0: exact.ml_complex(z, a, b).real
        f = lambda s: (40.0 - s) ** (a - 1.0) * E(0.05 * (40.0 - s) ** a, a) * E(-3.0 * s ** a)
        y40 = E(0.05 * 40.0 ** a) - 3.05 * quad(f, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
        rows.append(dict(example='dephasing with rate 3 on [0, 1) and -0.05 on [1, 40]: coherence factor y(40)', quantity=f'alpha = {a:g}', value=float(y40)))
    rows.append(dict(example='dephasing with rate 3 on [0, 1) and -0.05 on [1, 40]: coherence factor y(40)', quantity='alpha = 1 (exp(-1.05))',
                     value=math.exp(-1.05)))
    ex = 'dephasing with rate 3 on [0, 1), switched off afterwards: coherence factor'
    for a in (0.5, 0.8):
        E = lambda z: exact.ml_complex(z, a, 1.0).real
        rows.append(dict(example=ex, quantity=f'y(1), alpha = {a:g}', value=float(E(-3.0))))
        for t in (40.0, 400.0):
            f = lambda s: (t - s) ** (a - 1.0) * E(-3.0 * s ** a)
            rows.append(dict(example=ex, quantity=f'y({t:g}), alpha = {a:g}',
                             value=float(1.0 - 3.0 / math.gamma(a) * quad(f, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=400)[0])))
    rows.append(dict(example=ex, quantity='y(1), alpha = 1 (exp(-3)); constant afterwards', value=math.exp(-3.0)))
    N, r = 128, 3.0
    rows.append(dict(example='graded mesh tau_n = (n/N)^r, N = 128, r = 3', quantity='number of nodes with 0 < tau_n < 0.01',
                     value=float(sum(0.0 < (n / N) ** r < 0.01 for n in range(N + 1)))))
    write_table('Table_Examples_continuous.csv', pd.DataFrame(rows), [],
                'numbers of the article that follow from closed forms (no raw input): Mittag-Leffler functions by fracsp.exact.ml_complex, '
                'extrema by bounded scalar minimisation, the integral of example (c) by adaptive quadrature; see the docstring of agg_examples')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    todo = dict(E=[agg_E], SOE=[agg_SOE], B3=[agg_B3], B4=[agg_B4], tail=[agg_tail], positivity=[agg_positivity], examples=[agg_examples],
                all=[agg_E, agg_SOE, agg_B3, agg_B4, agg_tail, agg_positivity, agg_examples])[what]
    for f in todo:
        f()
    # provenance of the figure data (plain CSV/NPZ files): one manifest with the record of this aggregation and file hashes
    import hashlib
    files = sorted(x for x in os.listdir(FD) if x.startswith('FigDet_') and not x.endswith('.json'))
    man = dc.provenance(os.path.abspath(__file__), None, dict(
        note='figure data of the deterministic part, written by scripts/aggregate_deterministic.py from the tables and raw results named in tables/data/*.meta.json',
        files={x: dict(sha256_16=hashlib.sha256(open(os.path.join(FD, x), 'rb').read()).hexdigest()[:16], bytes=os.path.getsize(os.path.join(FD, x))) for x in files}))
    with open(os.path.join(FD, 'FigDet_manifest.meta.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(man, fh, indent=1); fh.write('\n')

"""Tables and figure data of the conformation-tensor benchmark B5 (forward, inverse, FENE-P, positivity).

Reads results/raw/B5_conformation/{forward, inverse, fenep, positivity} and writes
  tables/data/Table_B5_*.csv (each with a side-car <name>.meta.json: title, note, raw sources, provenance)
  figures/data/B5_*.csv
Every number is computed here from the per-run result files; table cells are formatted strings (percentages,
counts, three-part cells), as described in the title and note of each table.
Usage: python scripts/aggregate_b5.py [forward inverse fenep model positivity fenep_step]      (default: all)
"""
import csv
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code', 'experiments')); sys.path.insert(0, os.path.join(ROOT, 'code'))
import det_common as dc  # noqa: E402  (first: pins BLAS to one thread before numpy is loaded)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fracsp import spd  # noqa: E402

RES = os.path.join(ROOT, 'results', 'raw', 'B5_conformation')
TD = os.path.join(ROOT, 'tables', 'data'); FD = os.path.join(ROOT, 'figures', 'data')
PRE = 'results/raw/B5_conformation/'


def load(pattern):
    """Result records matching the pattern, sorted by file name; '_file' holds the base name."""
    out = []
    for fn in sorted(glob.glob(os.path.join(RES, pattern))):
        d = json.load(open(fn))
        if isinstance(d, dict):
            d['_file'] = os.path.basename(fn)
            out.append(d)
    return out


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}e}'


def pct(x, nd=2):
    return f'{100 * x:.{nd}f}'


def write_table(name, title, header, rows, sources, note=''):
    """tables/data/Table_B5_<name>.csv and its side-car with the title, the note and the raw sources."""
    os.makedirs(TD, exist_ok=True)
    path = os.path.join(TD, f'Table_B5_{name}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:          # csv module: '\r\n' on every platform
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    meta = dc.provenance(os.path.abspath(__file__), None, dict(title=title, note=note, raw_sources=sorted(set(sources)), rows=len(rows),
                                                             columns=[str(h) for h in header]))
    with open(os.path.splitext(path)[0] + '.meta.json', 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, indent=1); f.write('\n')
    print(f'wrote {dc.rel(path)} ({len(rows)} rows)')


def write_figdata(name, df):
    os.makedirs(FD, exist_ok=True)
    df.to_csv(os.path.join(FD, name), index=False, lineterminator='\n')          # floats: shortest round-trip representation
    print(f'wrote figures/data/{name} ({len(df)} rows)')


def indefinite(r, key_min='min_eig', key_scale='scale'):
    """Loss of positive definiteness beyond round-off: smallest eigenvalue below -1e-9 x (a-priori scale).
    (The exponential map can return a numerically singular matrix whose computed smallest eigenvalue is
    of the order of -1e-16 x largest eigenvalue; such runs are counted separately as non-positive.)"""
    return r[key_min] < -1e-9 * max(1.0, r.get(key_scale, 1.0))


VARIANT_ORDER = ['sym-fro', 'sym-comp', 'sym-rel', 'factor-fro', 'factor-nat', 'anchor-fro', 'logm-fro', 'logm-nat',
                 'bfactor-fro', 'bfactor-nat', 'blogm-nat']
VLABEL = {'sym-fro': 'unconstrained / Frobenius', 'sym-comp': 'unconstrained / component-scaled',
          'sym-rel': 'unconstrained / relative', 'factor-fro': 'factor T T^T / Frobenius',
          'factor-nat': 'factor T T^T / affine-invariant', 'anchor-fro': 'C0 + g B B^T / Frobenius',
          'logm-fro': 'exp(Psi) / Frobenius', 'logm-nat': 'exp(Psi) / affine-invariant',
          'bfactor-fro': 'bounded-trace factor / Frobenius', 'bfactor-nat': 'bounded-trace factor / affine-invariant',
          'blogm-nat': 'bounded-trace exp / affine-invariant'}


# ==============================================================================================
# forward
# ==============================================================================================
def forward():
    allruns = [r for r in load('forward/*_s*.json') if r.get('method') == 'network']
    budget_runs = [r for r in allruns if '_budget' in r['_file'] and not r.get('failed')]
    runs = [r for r in allruns if '_budget' not in r['_file']]
    steps = {(s['flow'], s['Wi'], s['alpha'], s['frame']): s for s in load('forward/stepper_*.json')}
    src_runs = [PRE + 'forward/<case>_<output>-<loss>_s<seed>.json']; src_step = [PRE + 'forward/stepper_<case>.json']
    if budget_runs:
        rows = []
        keys = sorted({(r['flow'], r['Wi'], r['alpha'], r['frame'], r['variant'], r['budget_factor']) for r in budget_runs})
        for (flow, Wi, al, fr, v, bf) in keys:
            rr = [r for r in budget_runs if (r['flow'], r['Wi'], r['alpha'], r['frame'], r['variant'], r['budget_factor']) == (flow, Wi, al, fr, v, bf)]
            base = [r for r in runs if not r.get('failed') and (r['flow'], r['Wi'], r['alpha'], r['frame'], r['variant']) == (flow, Wi, al, fr, v)
                    and r['seed'] in {q['seed'] for q in rr}]
            e = np.array([r['rel_fro'] for r in rr])
            rows.append([flow, f'{Wi:g}', al, fr, VLABEL[v], f'{bf:g}', len(rr), pct(float(np.median(e))), pct(float(e.min())),
                         pct(float(e.max())), sum(r['frac_time_not_pd'] > 0 for r in rr),
                         pct(float(np.median([r['rel_fro'] for r in base]))) if base else 'n/a',
                         f"{np.median([r['wall_s'] for r in rr]):.0f}"])
        write_table('forward_budget', 'Budget sensitivity (Adam and L-BFGS iterations multiplied by the budget factor)',
                    ['flow', 'Wi', 'alpha', 'frame', 'output / residual norm', 'budget factor', 'runs', 'median e_F, %',
                     'min e_F, %', 'max e_F, %', 'runs not PD', 'median e_F at budget 1 (same seeds), %', 'median wall, s'], rows,
                    [PRE + 'forward/<case>_<output>-<loss>_s<seed>_budget<factor>.json'] + src_runs)
    if not runs:
        print('no forward runs')
        return
    failed = [r for r in runs if r.get('failed')]
    runs = [r for r in runs if not r.get('failed')]
    cases = sorted({(r['flow'], r['Wi']) for r in runs}, key=lambda c: (c[0], c[1]))
    alphas = sorted({r['alpha'] for r in runs})
    frames = sorted({r['frame'] for r in runs})
    variants = [v for v in VARIANT_ORDER if any(r['variant'] == v for r in runs)]

    # --- stepper baseline table
    rows = []
    for (flow, Wi) in cases:
        for al in alphas:
            s = steps.get((flow, Wi, al, 'aligned'))
            if s is None:
                continue
            rows.append([flow, f'{Wi:g}', al, fmt(s['exact_max_norm']), fmt(s['exact_max_cond']),
                         f"{s['min_eig_exact']:.3f}", f"{s['exact_min_eig_of_C_minus_C0']:.3f}", fmt(s['rel_fro']),
                         fmt(s['nodal_rel_fro']), fmt(s['max_rel_fro']), fmt(s['log_euclid_rms']),
                         f"{s['min_eig']:.3f}", f"{1e3 * s['wall_s_median']:.1f}"])
    write_table('forward_stepper', 'Implicit L1 stepper on the collocation mesh (N = 128), linear law',
                ['flow', 'Wi', 'alpha', 'max Frobenius norm of C', 'max cond(C)', 'min eig C (exact)', 'min eig (C - I) (exact)',
                 'e_F test grid', 'e_F nodes', 'max pointwise rel. err.', 'log-Euclid rms', 'min eig (L1)', 'wall, ms'],
                rows, src_step, 'e_F: relative Frobenius error; the test-grid value uses the piecewise-linear interpolant of the nodal values.')

    # --- long summary
    long_rows = []
    summ = {}
    for (flow, Wi) in cases:
        for al in alphas:
            for fr in frames:
                for v in variants:
                    rr = [r for r in runs if (r['flow'], r['Wi'], r['alpha'], r['frame'], r['variant']) == (flow, Wi, al, fr, v)]
                    if not rr:
                        continue
                    e = np.array([r['rel_fro'] for r in rr])
                    notpd = [r for r in rr if r['frac_time_not_pd'] > 0]
                    le = [r['log_euclid_rms'] for r in rr if r['log_euclid_rms'] is not None]
                    d = dict(n=len(rr), med=float(np.median(e)), lo=float(e.min()), hi=float(e.max()),
                             n_indef=sum(indefinite(r) for r in rr), n_le=len(le),
                             med_max=float(np.median([r['max_rel_fro'] for r in rr])), n_notpd=len(notpd),
                             med_frac=float(np.median([r['frac_time_not_pd'] for r in rr])),
                             med_mineig=float(np.median([r['min_eig'] for r in rr])),
                             min_mineig=float(np.min([r['min_eig'] for r in rr])),
                             ex_mineig=rr[0]['min_eig_exact'], med_le=float(np.median(le)) if le else None,
                             med_opt=float(np.median([r['nodal_rel_diff_to_L1'] for r in rr])),
                             med_wall=float(np.median([r['wall_s'] for r in rr])),
                             nonfinite=int(sum(r['nonfinite_events'] for r in rr)))
                    summ[(flow, Wi, al, fr, v)] = d
                    long_rows.append([flow, f'{Wi:g}', al, fr, v, d['n'], fmt(d['med']), fmt(d['lo']), fmt(d['hi']),
                                      fmt(d['med_max']), d['n_indef'], d['n_notpd'], f"{d['med_frac']:.3f}", f"{d['med_mineig']:.3g}",
                                      f"{d['min_mineig']:.3g}", f"{d['ex_mineig']:.3g}", fmt(d['med_le']), fmt(d['med_opt']),
                                      f"{d['med_wall']:.1f}", d['nonfinite']])
    write_table('forward_summary_long', 'Forward benchmark, all configurations',
                ['flow', 'Wi', 'alpha', 'frame', 'variant', 'runs', 'median e_F', 'min e_F', 'max e_F',
                 'median max pointwise', 'runs indefinite (min eig < -1e-9 s)', 'runs with min eig <= 0',
                 'median fraction of time with min eig <= 0', 'median min eig', 'worst min eig',
                 'exact min eig', 'median log-Euclid rms (PD runs)', 'median nodal diff. to L1', 'median wall, s',
                 'non-finite loss events'], long_rows, src_runs)

    # --- compact tables: rows = variants, columns = cases
    for al in alphas:
        for fr in frames:
            header = ['output / residual norm'] + [f'{flow} Wi={Wi:g}' for (flow, Wi) in cases]
            rows = []
            srow = ['implicit L1, same mesh']
            for (flow, Wi) in cases:
                st = steps.get((flow, Wi, al, fr))
                srow.append(f"{pct(st['rel_fro'])} ; {st['log_euclid_rms']:.3f} ; 0" if st else 'n/a')
            rows.append(srow)
            for v in variants:
                row = [VLABEL[v]]
                for (flow, Wi) in cases:
                    d = summ.get((flow, Wi, al, fr, v))
                    if d is None:
                        row.append('n/a')
                        continue
                    le = 'n/a' if d['med_le'] is None or d['n_le'] < d['n'] else f"{d['med_le']:.3f}"
                    ef = pct(d['med']) if d['med'] < 10 else '>1000'
                    row.append(f"{ef} ; {le} ; {d['n_indef']}/{d['n']}")
                rows.append(row)
            write_table(f'forward_a{al:g}_{fr}', f'Forward benchmark, alpha = {al:g}, {fr} frame. Cell: median e_F in % ; '
                        f'median log-Euclidean rms error ; runs that lose positive definiteness / runs', header, rows, src_runs + src_step,
                        'e_F: relative Frobenius error on the test grid. Log-Euclidean error: rms over the test grid of |log X - log C|_F '
                        '(n/a if X is not positive definite in floating point at some test time in at least one run). Loss of positive '
                        'definiteness: smallest eigenvalue below -1e-9 s on the test grid.')

    # --- best network against the stepper, per configuration
    rows = []
    ratios = []
    nbetter = 0
    for (flow, Wi) in cases:
        for al in alphas:
            for fr in frames:
                st = steps.get((flow, Wi, al, fr))
                cand = [(summ[(flow, Wi, al, fr, v)]['med'], v) for v in variants if (flow, Wi, al, fr, v) in summ]
                if st is None or not cand:
                    continue
                best, vb = min(cand)
                ratio = best / st['rel_fro']
                ratios.append(ratio)
                nbetter += ratio < 1.0
                dbest = summ[(flow, Wi, al, fr, vb)]
                rows.append([flow, f'{Wi:g}', al, fr, pct(st['rel_fro'], 3), VLABEL[vb], pct(best, 3), f'{ratio:.2f}',
                             f"{dbest['med_wall'] / st['wall_s_median']:.0f}"])
    write_table('forward_best_vs_stepper', 'Forward benchmark: most accurate network variant (median over seeds) against the implicit L1 stepper on the same mesh',
                ['flow', 'Wi', 'alpha', 'frame', 'stepper e_F, %', 'best network variant', 'its median e_F, %',
                 'ratio network / stepper', 'wall-time ratio network / stepper'], rows, src_runs + src_step,
                f'Configurations: {len(ratios)}; best network median below the stepper error in {nbetter} of them; '
                f'ratio network / stepper: min {min(ratios):.2f}, median {float(np.median(ratios)):.2f}, max {max(ratios):.2f}.')

    # --- overall admissibility count
    rows = []
    for v in variants:
        rr = [r for r in runs if r['variant'] == v]
        nind = sum(indefinite(r) for r in rr)
        npd = sum(r['frac_time_not_pd'] > 0 for r in rr)
        acc = sum(r['rel_fro'] < 0.1 for r in rr)
        acc2 = sum(r['rel_fro'] < 0.1 and r['log_euclid_rms'] is not None and r['log_euclid_rms'] < 0.1 for r in rr)
        rows.append([VLABEL[v], len(rr), nind, f'{100 * nind / len(rr):.1f}', npd - nind, acc, f'{100 * acc / len(rr):.1f}',
                     acc2, f'{100 * acc2 / len(rr):.1f}', f"{np.median([r['wall_s'] for r in rr]):.1f}"])
    write_table('forward_admissibility', 'Forward benchmark: all runs pooled (6 cases x 2 orders x 2 frames x seeds)',
                ['output / residual norm', 'runs', 'runs indefinite', '% indefinite',
                 'runs singular to round-off (min eig <= 0, not indefinite)', 'runs with e_F < 10 %', '% with e_F < 10 %',
                 'runs with e_F < 10 % and log-Euclid rms < 0.1', '% with both', 'median wall, s'], rows, src_runs,
                f'indefinite: smallest eigenvalue below -1e-9 s at a test time. Failed (exception) runs: {len(failed)}.')

    # --- figure data: every run, the stepper rows, smallest-eigenvalue curves of two representative cases
    cols = ('flow', 'Wi', 'alpha', 'frame', 'variant', 'seed', 'rel_fro', 'max_rel_fro', 'min_eig', 'min_eig_exact', 'frac_time_not_pd',
            'log_euclid_rms', 'nodal_rel_diff_to_L1', 'scale', 'wall_s')
    write_figdata('B5_forward_runs.csv', pd.DataFrame([dict({k: r[k] for k in cols}, budget_factor=r.get('budget_factor', 1.0),
                                                            indefinite=bool(indefinite(r))) for r in runs + budget_runs]))
    scol = ('flow', 'Wi', 'alpha', 'frame', 'N', 'rel_fro', 'nodal_rel_fro', 'max_rel_fro', 'min_eig', 'min_eig_exact', 'log_euclid_rms',
            'scale', 'wall_s_median')
    write_figdata('B5_forward_stepper.csv', pd.DataFrame([{k: s[k] for k in scol} for s in steps.values()]))
    tq = spd.test_grid(); curves = []
    for (flow, Wi, al, fr) in [('shear', 100.0, 0.6, 'rotated'), ('uniaxial', 1.5, 0.6, 'rotated')]:
        ex = np.load(os.path.join(RES, 'forward', f'exact_{flow}_Wi{Wi:g}_a{al:g}_aligned.npy'))
        curves.append(pd.DataFrame(dict(flow=flow, Wi=Wi, alpha=al, frame=fr, variant='exact', seed=-1, tau=tq, min_eig=np.linalg.eigvalsh(ex)[:, 0])))
        for v in ('sym-fro', 'sym-comp', 'logm-nat'):
            for r in runs:
                if (r['flow'], r['Wi'], r['alpha'], r['frame'], r['variant']) == (flow, Wi, al, fr, v) and 'curve_min_eig' in r:
                    curves.append(pd.DataFrame(dict(flow=flow, Wi=Wi, alpha=al, frame=fr, variant=v, seed=r['seed'], tau=tq, min_eig=r['curve_min_eig'])))
    write_figdata('B5_forward_min_eig_curves.csv', pd.concat(curves, ignore_index=True))


# ==============================================================================================
# inverse
# ==============================================================================================
def inverse():
    nets = [r for r in load('inverse/net_*.json') if not r.get('failed')]
    nls = load('inverse/nls*_a*.json')
    fn = os.path.join(RES, 'inverse', 'cramer_rao.json')
    crb = {(c['alpha_true'], c['noise']): c for c in json.load(open(fn))} if os.path.exists(fn) else {}
    if not nets and not nls:
        print('no inverse runs')
        return
    src = [PRE + 'inverse/net_<output>-<loss>_a<alpha>_n<noise>_s<seed>.json', PRE + 'inverse/nls<N>_a<alpha>_n<noise>_s<seed>.json',
           PRE + 'inverse/cramer_rao.json']
    methods = [m for m in ['nls128', 'nls512'] if any(r['method'] == m for r in nls)]
    nvar = [v for v in VARIANT_ORDER if any(r['variant'] == v for r in nets)]
    alphas = sorted({r['alpha_true'] for r in nls + nets})
    noises = sorted({r['noise'] for r in nls + nets})
    rows = []
    for al in alphas:
        for nz in noises:
            c = crb.get((al, nz))
            for m in methods + nvar:
                rr = ([r for r in nls if (r['method'], r['alpha_true'], r['noise']) == (m, al, nz)] if m in methods else
                      [r for r in nets if (r['variant'], r['alpha_true'], r['noise']) == (m, al, nz)])
                if not rr:
                    continue
                a = np.array([r['alpha_hat'] for r in rr]) - al
                le = np.array([r['ell_hat'] / r['ell_true'] - 1.0 for r in rr])
                npd = sum(r['state_frac_time_not_pd'] > 0 for r in rr)
                d = dict(n=len(rr), a_bias=float(a.mean()), a_sd=float(a.std(ddof=1)) if len(rr) > 1 else float('nan'),
                         a_rmse=float(np.sqrt((a ** 2).mean())), l_bias=float(le.mean()),
                         l_sd=float(le.std(ddof=1)) if len(rr) > 1 else float('nan'), l_rmse=float(np.sqrt((le ** 2).mean())),
                         state=float(np.median([r['state_rel_fro'] for r in rr])), npd=npd,
                         wall=float(np.median([r['wall_s'] for r in rr])),
                         cost=float(np.median([r.get('n_forward_solves', r.get('n_loss_evals', float('nan'))) for r in rr])))
                label = {'nls128': 'NLS through implicit L1, N = 128', 'nls512': 'NLS through implicit L1, N = 512'}.get(m)
                label = label or ('network, ' + VLABEL[m])
                if m in methods:                # least squares: starts per fit, and fits in which all starts end at one minimum
                    st, = {len(r['starts']) for r in rr}
                    starts = [st, f"{sum(r['n_distinct_minima'] == 1 for r in rr)}/{d['n']}"]
                else:
                    starts = ['', '']
                rows.append([al, f'{100 * nz:g}', label, d['n'], f"{d['a_bias']:+.4f}", f"{d['a_sd']:.4f}", f"{d['a_rmse']:.4f}",
                             f"{100 * d['l_bias']:+.2f}", f"{100 * d['l_sd']:.2f}", f"{100 * d['l_rmse']:.2f}",
                             pct(d['state']), f"{npd}/{d['n']}", f"{d['wall']:.2f}", f"{d['cost']:.0f}"] + starts)
            if c is not None:
                rows.append([al, f'{100 * nz:g}', 'Cramer-Rao bound (sd)', '', '', f"{c['sd_alpha']:.4f}", '', '',
                             f"{100 * c['sd_log_ell']:.2f}", '', '', '', '', '', '', ''])
    write_table('inverse_summary', 'Identification of (alpha, lambda) from shear-stress and N1 growth data (true Wi = 5, T = 4 lambda, 30 times)',
                ['alpha true', 'noise, %', 'method', 'runs', 'bias alpha', 'sd alpha', 'rmse alpha', 'bias lambda, %',
                 'sd lambda, %', 'rmse lambda, %', 'median state e_F, %', 'reconstructions not PD', 'median wall, s',
                 'median forward solves / loss evaluations', 'starts per least-squares fit', 'fits with one distinct minimum'], rows, src,
                'bias = mean over noise realisations (and network seeds) of estimate minus truth; sd = sample standard deviation. '
                'fits with one distinct minimum: least-squares fits in which all starts end at the same (alpha, lambda) to four digits.')

    # paired comparison on the noise realisations common to all methods; pooled over the two orders
    common = None
    for m in methods + nvar:
        sd = ({r['seed'] for r in nls if r['method'] == m} if m in methods else {r['seed'] for r in nets if r['variant'] == m})
        common = sd if common is None else common & sd
    common = sorted(common or [])
    rows = []
    for nz in noises:
        for m in methods + nvar:
            rr = ([r for r in nls if r['method'] == m and r['noise'] == nz and r['seed'] in common] if m in methods else
                  [r for r in nets if r['variant'] == m and r['noise'] == nz and r['seed'] in common])
            if not rr:
                continue
            a = np.array([r['alpha_hat'] - r['alpha_true'] for r in rr])
            le = np.array([r['ell_hat'] / r['ell_true'] - 1.0 for r in rr])
            label = {'nls128': 'NLS through implicit L1, N = 128', 'nls512': 'NLS through implicit L1, N = 512'}.get(m)
            label = label or ('network, ' + VLABEL[m])
            rows.append([f'{100 * nz:g}', label, len(rr), f'{np.sqrt((a ** 2).mean()):.4f}', f'{np.abs(a).max():.4f}',
                         f'{100 * np.sqrt((le ** 2).mean()):.2f}', f'{100 * np.abs(le).max():.2f}',
                         pct(float(np.median([r['state_rel_fro'] for r in rr]))),
                         sum(r['state_frac_time_not_pd'] > 0 for r in rr), f"{np.median([r['wall_s'] for r in rr]):.1f}"])
    write_table('inverse_paired', f'Identification: all methods on the same noise realisations (seeds {common}), both orders pooled',
                ['noise, %', 'method', 'runs', 'rmse alpha', 'max abs error alpha', 'rmse lambda, %', 'max abs error lambda, %',
                 'median state e_F, %', 'reconstructions not PD', 'median wall, s'], rows, src)

    # --- figure data: every estimate, and the Cramer-Rao bounds
    cols = ('alpha_true', 'ell_true', 'noise', 'seed', 'alpha_hat', 'ell_hat', 'state_rel_fro', 'state_frac_time_not_pd', 'wall_s')
    est = [dict(method=r['method'], family='nls', **{k: r[k] for k in cols}) for r in nls]
    est += [dict(method=r['variant'], family='network', **{k: r[k] for k in cols}) for r in nets]
    write_figdata('B5_inverse_estimates.csv', pd.DataFrame(est))
    if crb:
        write_figdata('B5_inverse_crb.csv', pd.DataFrame([dict(alpha_true=c['alpha_true'], noise=c['noise'], sd_alpha=c['sd_alpha'],
                                                               sd_log_ell=c['sd_log_ell']) for c in crb.values()]))


# ==============================================================================================
# FENE-P
# ==============================================================================================
def fenep():
    runs = [r for r in load('fenep/fenep_*_s*.json') if r.get('method') == 'network' and not r.get('failed')]
    steps = {(s['Wi'], s['alpha']): s for s in load('fenep/stepper_*.json')}
    if not runs:
        print('no fenep runs')
        return
    variants = [v for v in VARIANT_ORDER if any(r['variant'] == v for r in runs)]
    cases = sorted({(r['Wi'], r['alpha']) for r in runs})
    rows = []
    for (Wi, al) in cases:
        s = steps.get((Wi, al))
        if s:
            rows.append([f'{Wi:g}', al, 'implicit L1 time stepper, same mesh', 1, pct(s['rel_fro']), pct(s['rel_fro']),
                         pct(s['rel_fro']), '0/1', '0.000', f"{s['min_eig']:.4f}", f"{s['min_eig_exact']:.4f}",
                         f"{s['max_trace']:.2f}", fmt(s['log_euclid_rms']), f"{s['wall_s_median']:.3f}",
                         fmt(s['reference_est_rel_diff_N_2N'])])
        for v in variants:
            rr = [r for r in runs if (r['Wi'], r['alpha'], r['variant']) == (Wi, al, v)]
            if not rr:
                continue
            e = np.array([r['rel_fro'] for r in rr])
            nin = sum(r['frac_time_inadmissible'] > 0 for r in rr)
            le = [r['log_euclid_rms'] for r in rr if r['log_euclid_rms'] is not None]
            rows.append([f'{Wi:g}', al, VLABEL[v], len(rr), pct(float(np.median(e))), pct(float(e.min())), pct(float(e.max())),
                         f'{nin}/{len(rr)}', f"{np.median([r['frac_time_inadmissible'] for r in rr]):.3f}",
                         f"{np.median([r['min_eig'] for r in rr]):.4f}", f"{rr[0]['min_eig_exact']:.4f}",
                         f"{np.median([r['max_trace'] for r in rr]):.2f}", fmt(float(np.median(le))) if le else 'n/a',
                         f"{np.median([r['wall_s'] for r in rr]):.1f}", ''])
    write_table('fenep_summary', 'Start-up of uniaxial extension, Caputo fractional FENE-P (b = 50, T = 2 lambda)',
                ['Wi', 'alpha', 'method', 'runs', 'median e_F, %', 'min e_F, %', 'max e_F, %', 'runs inadmissible',
                 'median fraction of time inadmissible', 'median min eig', 'reference min eig', 'median max trace',
                 'median log-Euclid rms (PD runs)', 'median wall, s', 'reference: e_F between N = 2048 and N = 4096'], rows,
                [PRE + 'fenep/fenep_uniaxial_<case>_<output>-<loss>_s<seed>.json', PRE + 'fenep/stepper_fenep_uniaxial_<case>.json'],
                'inadmissible: X not positive definite or tr X >= b at a test time. Time stepper and reference (N = 2048 and 4096, Richardson): implicit L1 with '
                "Newton's iteration; its iterates are admissible at every step and are therefore the admissible solution of the step "
                '(Table_B5_fenep_stored_check.csv). Last column: difference of the solutions with 2048 and 4096 steps on the test grid, '
                'relative, in the Frobenius norm summed over the test times (the measure e_F).')

    # --- figure data: trajectories of one case (all seeds of three variants) and the reference
    tq = spd.test_grid(); Wi, al = 5.0, 0.6; curves = []
    Th, = {r['Theta'] for r in runs if (r['Wi'], r['alpha']) == (Wi, al)}        # window in units of the relaxation time: t / lambda = Theta tau
    ex = np.load(os.path.join(RES, 'fenep', f'reference_fenep_uniaxial_Wi{Wi:g}_a{al:g}.npz'))['C']
    curves.append(pd.DataFrame(dict(Wi=Wi, alpha=al, variant='reference', seed=-1, tau=tq, t_over_lambda=Th * tq, C11=ex[:, 0, 0],
                                    min_eig=np.linalg.eigvalsh(ex)[:, 0])))
    for v in ('sym-rel', 'sym-comp', 'blogm-nat'):
        for r in runs:
            if (r['Wi'], r['alpha'], r['variant']) == (Wi, al, v):
                curves.append(pd.DataFrame(dict(Wi=Wi, alpha=al, variant=v, seed=r['seed'], tau=tq, t_over_lambda=Th * tq, C11=r['curve_C11'],
                                                min_eig=r['curve_min_eig'])))
    write_figdata('B5_fenep_curves.csv', pd.concat(curves, ignore_index=True))


# ==============================================================================================
# exact solutions of the linear law (illustration) and positivity of time stepping
# ==============================================================================================
def model():
    tq = np.logspace(-4, 0, 160); out = []
    for (flow, Wi, Th) in [('shear', 10.0, 4.0), ('shear', 100.0, 4.0), ('uniaxial', 1.5, 2.0)]:
        for al in (1.0, 0.9, 0.6):
            w = np.linalg.eigvalsh(spd.exact_ucm(flow, Wi, Th, al, tq))
            out.append(pd.DataFrame(dict(flow=flow, Wi=Wi, Theta=Th, alpha=al, tau=tq, eig_min=w[:, 0], eig_max=w[:, -1])))
    write_figdata('B5_model_eigenvalues.csv', pd.concat(out, ignore_index=True))


def positivity():
    fn = os.path.join(RES, 'positivity', 'positivity.json')
    if not os.path.exists(fn):
        return
    d = json.load(open(fn)); src = [PRE + 'positivity/positivity.json']
    rows = []
    keys = sorted({(r['flow'], r['Wi'], r['alpha'], r['mesh']) for r in d['stepper']})
    for (flow, Wi, al, mesh) in keys:
        rr = sorted([r for r in d['stepper'] if (r['flow'], r['Wi'], r['alpha'], r['mesh']) == (flow, Wi, al, mesh)],
                    key=lambda r: r['N'])
        imp_fail = [r['N'] for r in rr if not r['implicit_spd']]
        exp_fail = [r['N'] for r in rr if not r['explicit_spd']]
        hc = rr[0]['h_crit_implicit']
        rows.append([flow, f'{Wi:g}', al, mesh, 'none (unconditional)' if hc is None or not math.isfinite(hc) else f'{hc:.3g}',
                     ', '.join(map(str, imp_fail)) or 'none', all(r['implicit_spd'] == r['implicit_predicted_spd'] for r in rr),
                     f"{rr[0]['h1_crit_explicit_shear']:.3g}", ', '.join(map(str, exp_fail)) or 'none'])
    write_table('positivity_steppers', 'Positive definiteness of L1 time stepping (linear law), N in {4, 8, ..., 1024}',
                ['flow', 'Wi', 'alpha', 'mesh', 'implicit: predicted bound on max step', 'implicit: N with loss of PD',
                 'prediction confirmed', 'explicit: first-step bound (shear)', 'explicit: N with loss of PD'], rows, src)
    if 'nonlinear' in d:
        rows = []
        keys = []
        for r in d['nonlinear']:
            k = (r['law'], r['flow'], r['Wi'], r.get('aG'), r.get('b'), r['alpha'])
            if k not in keys:
                keys.append(k)
        bad = lambda law, b, lam, tr: lam <= 0 or (law == 'fenep' and tr >= b)
        for k in keys:
            rr = [r for r in d['nonlinear'] if (r['law'], r['flow'], r['Wi'], r.get('aG'), r.get('b'), r['alpha']) == k]
            okr = [r for r in rr if r['ok']]
            par = f"a = {k[3]:g}" if k[0] == 'giesekus' else f"b = {k[4]:g}"
            rows.append([k[0], k[1], f'{k[2]:g}', par, k[5], len(rr), sum(r['step_condition_lhs'] >= 1 for r in rr), len(rr) - len(okr),
                         sum(bad(k[0], k[4], r['min_eig'], r['max_trace']) for r in okr),
                         f"{min(r['min_eig'] for r in okr):.4g}" if okr else 'n/a',
                         f"{max(r['max_trace'] for r in okr):.4g}" if okr else 'n/a',
                         sum(bad(k[0], k[4], r['safeguarded_min_eig'], r['safeguarded_max_trace']) for r in rr),
                         f"{min(r['safeguarded_min_eig'] for r in rr):.4g}", f"{max(r['safeguarded_max_trace'] for r in rr):.4g}",
                         sum(r['safeguarded_root_steps'] > 0 for r in rr), fmt(max(r['safeguarded_max_residual'] for r in rr), 1)])
        write_table('positivity_nonlinear', "Implicit L1 for nonlinear laws: Newton's iteration without a test of the result and safeguarded solver, uniform and graded meshes, N in {16, 64, 256, 1024}",
                    ['law', 'flow', 'Wi', 'parameter', 'alpha', 'runs', 'runs above the step bound of the linear law', 'Newton: no convergence',
                     'Newton: inadmissible solutions', 'Newton: smallest eigenvalue over converged runs', 'Newton: largest trace over converged runs',
                     'safeguarded: inadmissible solutions', 'safeguarded: smallest eigenvalue', 'safeguarded: largest trace',
                     'safeguarded: runs that used the root solver', 'safeguarded: largest scaled residual'], rows, src,
                    'Newton: iteration started from the previous iterate, for FENE-P with a step halved as long as the new iterate would have tr C >= b, '
                    'no admissibility test of the result (fracsp.spd.solve_l1_newton). Safeguarded: the same iteration, accepted '
                    'only if it converged to an admissible tensor, otherwise the admissible solution of the step from the scalar equation (FENE-P) or the '
                    'Riccati equation (Giesekus) (fracsp.spd.solve_l1). Inadmissible: smallest eigenvalue <= 0, or tr C >= b for FENE-P. Step bound of the '
                    'linear law: Gamma(2 - alpha) (h_n / lambda)^alpha (2 Wi s - 1) < 1 with s the spectral abscissa of the unit-rate velocity gradient.')
    rows = [[e['flow'], f"{e['Wi']:g}", e['alpha'], f"{e['min_eig']:.4f}", f"{e['min_det']:.6f}", f"{e['min_eig_C_minus_I']:.4f}"]
            for e in d['exact']]
    write_table('positivity_exact', 'Exact solution on the dense test grid', ['flow', 'Wi', 'alpha', 'min eig C', 'min det C',
                                                                             'min eig (C - I)'], rows, src)


def fenep_step():
    """Tables of the driver b5_fenep_step.py: admissible solution of the FENE-P step against Newton's iteration."""
    fn = os.path.join(RES, 'positivity', 'fenep_step.json')
    if not os.path.exists(fn):
        return
    d = json.load(open(fn)); src = [PRE + 'positivity/fenep_step.json']

    def newton(rr, p):
        viol = [r for r in rr if r['violates_step_condition']]
        sat = [r for r in rr if not r['violates_step_condition']]
        return [sum(r[p + '_status'] == 'inadmissible' for r in rr), sum(r[p + '_status'] == 'no convergence' for r in rr),
                sum(r[p + '_status'] != 'admissible' for r in viol), sum(r[p + '_status'] != 'admissible' for r in sat)]

    def line(label, subset, rr):
        viol = [r for r in rr if r['violates_step_condition']]
        pole = sum(r['newton_full_first_inadmissible_kind'] is not None and 'tr C >= b' in r['newton_full_first_inadmissible_kind'] for r in rr)
        inad = sum(r['newton_full_status'] == 'inadmissible' and not r['violates_step_condition'] for r in rr)
        return [label, subset, len(rr), len(viol)] + newton(rr, 'newton_full') + [inad, pole] + newton(rr, 'newton') + [
                sum(r['root_admissible'] for r in rr),
                f"{min(r['root_min_eig'] for r in rr):.4g}", f"{max(r['root_max_tr_over_b'] for r in rr):.4f}",
                sum(r['root_trace_bound_holds'] for r in rr), fmt(max(r['root_max_residual'] for r in rr), 1),
                sum(r['safeguarded_admissible'] for r in rr), sum(r['safeguarded_root_steps'] > 0 for r in rr),
                fmt(max(r['safeguarded_rel_diff_to_root'] for r in rr), 1)]
    sw = d['sweep']['runs']; ge = d['general']['runs']
    uni = 'uniaxial extension'; oth = 'shear, planar extension, random traceless velocity gradients'
    rows = [line(uni, 'all', sw)]
    for key in ('Wi', 'alpha', 'b'):
        for v in sorted({r[key] for r in sw}):
            rows.append(line(uni, f'{key} = {v:g}', [r for r in sw if r[key] == v]))
    for v in sorted({r['mesh'] for r in sw}, reverse=True):
        rows.append(line(uni, f'{v} mesh', [r for r in sw if r['mesh'] == v]))
    rows.append(line(oth, 'all', ge))
    for v in sorted({r['flow'] for r in ge if not r['flow'].startswith('random')}, reverse=True):
        rows.append(line(oth, v, [r for r in ge if r['flow'] == v]))
    rows.append(line(oth, 'random', [r for r in ge if r['flow'].startswith('random')]))
    dsw = d['sweep']['design']
    nw = d.get('newton_iteration', {})
    write_table('fenep_step', "Implicit L1 step of the FENE-P law: admissible solution (root solver), Newton's iteration and safeguarded solver",
                ['flows', 'subset', 'runs', 'runs above the step bound of the linear law',
                 'Newton, full steps: inadmissible', 'Newton, full steps: no convergence', 'Newton, full steps: failures above the step bound',
                 'Newton, full steps: failures within the step bound', 'Newton, full steps: inadmissible within the step bound',
                 'Newton, full steps: first inadmissible iterate has tr C >= b',
                 'Newton, halved steps: inadmissible', 'Newton, halved steps: no convergence', 'Newton, halved steps: failures above the step bound',
                 'Newton, halved steps: failures within the step bound', 'root solver: admissible runs',
                 'root solver: smallest eigenvalue', 'root solver: largest tr C / b', 'root solver: runs with tr C <= theta_*',
                 'root solver: largest scaled residual', 'safeguarded: admissible runs', 'safeguarded: runs that used the root solver',
                 'safeguarded: largest relative difference to the root solver'], rows, src,
                f"Uniaxial extension: Wi in {dsw['Wi']}, alpha in {dsw['alpha']}, b in {dsw['b']}, uniform and graded (r = (2 - alpha)/alpha) meshes "
                f"with N in {dsw['N']} on [0, {dsw['Theta']:g} lambda], C^0 = I. A run is one mesh. Newton's iteration starts from the previous iterate "
                f"and stops when the Frobenius norm of the update is at most {nw.get('tol', 1e-12):g} (1 + |C|_F), after at most {nw.get('maxit', 60)} "
                'iterations; its result is not tested. Full steps: fracsp.spd.solve_l1(nonlinear="newton-full"); halved steps: a step is halved as long '
                'as the new iterate would have tr C >= b (nonlinear="newton"). A Newton run is inadmissible if every step converged and an iterate has '
                'a non-positive eigenvalue or tr C >= b; it has no convergence if the iteration of a step does not converge (in all such runs an '
                'earlier iterate is inadmissible); a failure is either of the two. Step bound of the linear law: '
                'Gamma(2 - alpha) (h_n / lambda)^alpha (2 Wi s - 1) < 1 for all n. theta_*: bound of the trace for the continuous problem. Scaled '
                'residual: fracsp.spd.step_residuals.')
    if 'case' in d:
        c = d['case']; rows = []
        par = c['parameters']
        for k in ('Wi', 'alpha', 'b', 'Theta', 'N'):
            rows.append(['parameters', k, f"{par[k]:g}"])
        for k in ('step_over_lambda', 'step_bound_linear_law', 'step_over_bound', 'step_condition_lhs', 'theta_star'):
            rows.append(['mesh', k, f"{c[k]:.6g}"])
        rows.append(['first step', 'w_11 in units of c / lambda', f"{c['first_step']['w']:.6g}"])
        rows.append(['first step', 'f_c', f"{c['first_step']['f_c']:.6g}"])
        for i, r in enumerate(c['first_step']['roots'], 1):
            for k in ('f', 'C11', 'C22', 'trace'):
                rows.append([f'first step, real solution {i}', k, f"{r[k]:.6g}"])
            rows.append([f'first step, real solution {i}', 'admissible', str(r['admissible'])])
        for name, label in (('newton', 'Newton iteration, halved steps'), ('root', 'admissible iterates (root solver)')):
            q = c[name]
            for k in ('C11_first_step', 'C22_first_step', 'min_eig', 'max_tr_over_b', 'C11_final', 'e_F', 'max_residual'):
                rows.append([label, k, f"{q[k]:.6g}"])
            for tt, v in zip(c['times_over_lambda'], q['C11_at_times']):
                rows.append([label, f'C11 at t / lambda = {tt:g}', f"{v:.6g}"])
        rows.append(['Newton iteration, halved steps', 'steps_with_negative_eigenvalue', str(c['newton']['steps_with_negative_eigenvalue'])])
        if 'newton_full' in c:
            q = c['newton_full']
            for k in ('C11_first_step', 'C22_first_step', 'min_eig', 'max_tr_over_b', 'C11_final', 'e_F', 'max_residual', 'max_abs_diff_to_limited_newton'):
                rows.append(['Newton iteration, full steps', k, f"{q[k]:.6g}"])
            rows.append(['Newton iteration, full steps', 'steps_with_negative_eigenvalue', str(q['steps_with_negative_eigenvalue'])])
        rows.append(['admissible iterates (root solver)', 'max_tr', f"{c['root']['max_tr']:.6g}"])
        rows.append(['safeguarded solver', 'steps solved by the root solver', ' '.join(map(str, c['safeguarded']['root_steps']))])
        rows.append(['safeguarded solver', 'rel_diff_to_root', fmt(c['safeguarded']['rel_diff_to_root'], 1)])
        q = c['reference']
        rows.append(['reference (graded meshes, root solver)', 'steps', ' and '.join(map(str, q['meshes']))])
        for k in ('rel_diff_N_2N', 'min_eig', 'max_tr'):
            rows.append(['reference (graded meshes, root solver)', k, f"{q[k]:.6g}"])
        for tt, v in zip(c['times_over_lambda'], q['C11_at_times']):
            rows.append(['reference (graded meshes, root solver)', f'C11 at t / lambda = {tt:g}', f"{v:.6g}"])
        write_table('fenep_step_case', 'FENE-P, uniaxial extension, Wi = 20, alpha = 0.6, b = 50, uniform mesh with 256 steps on [0, 2 lambda]',
                    ['item', 'quantity', 'value'], rows, src,
                    'The first step from C^0 = I has three real solutions (C diagonal, C22 = C33; f: Peterlin factor). e_F: relative Frobenius error on the '
                    'test grid of the conformation benchmark against the Richardson extrapolate of the two reference meshes. The reference values at '
                    'given times are those of the finer reference mesh. On this mesh no Newton step is halved: both variants of the iteration return '
                    'the same iterates (max_abs_diff_to_limited_newton).')
    if 'case2' in d:
        c = d['case2']; rows = []
        par = c['parameters']
        for k in ('Wi', 'alpha', 'b', 'Theta', 'N'):
            rows.append(['parameters', k, f"{par[k]:g}"])
        rows.append(['mesh', 'step_condition_lhs', f"{c['step_condition_lhs']:.6g}"])
        rows.append(['mesh', 'theta_star', f"{c['theta_star']:.6g}"])
        q = c['newton_full']
        rows.append(['Newton iteration, full steps', 'converged_at_every_step', str(q['converged_at_every_step'])])
        rows.append(['Newton iteration, full steps', 'first_inadmissible_step', str(q['first_inadmissible_step'])])
        rows.append(['Newton iteration, full steps', 'first_inadmissible_kind', str(q['first_inadmissible_kind'])])
        for name, label in (('previous_iterate', 'iterate before that step'), ('iterate', 'iterate at that step'),
                            ('admissible_solution_of_that_step', 'admissible solution of that step')):
            for k in ('C11', 'C22', 'trace'):
                rows.append(['Newton iteration, full steps', f'{label}: {k}', f"{q[name][k]:.6g}"])
        rows.append(['Newton iteration, full steps', 'rel_diff_to_root_before_that_step', fmt(q['max_rel_diff_to_root_before'], 1)])
        rows.append(['Newton iteration, full steps', 'inadmissible_steps', str(q['inadmissible_steps'])])
        rows.append(['Newton iteration, full steps', 'steps_computed', str(q['steps_computed'])])
        rows.append(['Newton iteration, full steps', 'max_residual', f"{q['max_residual']:.6g}"])
        rows.append(['Newton iteration, halved steps', 'admissible_at_every_step', str(c['newton']['admissible'])])
        rows.append(['Newton iteration, halved steps', 'rel_diff_to_root', fmt(c['newton']['rel_diff_to_root'], 1)])
        rows.append(['admissible iterates (root solver)', 'min_eig', f"{c['root']['min_eig']:.6g}"])
        rows.append(['admissible iterates (root solver)', 'max_tr', f"{c['root']['max_tr']:.6g}"])
        write_table('fenep_step_case2', 'FENE-P, uniaxial extension, Wi = 20, alpha = 0.9, b = 10, uniform mesh with 512 steps on [0, 2 lambda]',
                    ['item', 'quantity', 'value'], rows, src,
                    'A mesh that satisfies the step bound of the linear law (step_condition_lhs < 1). With full Newton steps an iterate passes the pole '
                    'tr C = b of the FENE-P term and the iteration converges to a solution of the step equation with tr C > b; with halved steps it '
                    'returns the admissible iterates.')
    if 'stored' in d:
        rows = []
        for r in d['stored']:
            for N, v in r['meshes'].items():
                rows.append([f"{r['Wi']:g}", r['alpha'], N, f"{v['step_condition_lhs']:.4f}", v['newton_admissible'], f"{v['newton_min_eig']:.5f}",
                             f"{v['newton_max_tr_over_b']:.5f}", fmt(v['newton_rel_diff_to_root'], 1), v['safeguarded_identical_to_newton'],
                             fmt(r['stored_reference_rel_diff_to_root'], 1) if 'stored_reference_rel_diff_to_root' in r else 'n/a'])
        write_table('fenep_stored_check', 'FENE-P benchmark: the Newton iterates behind the references, stepper baselines and a-priori scales are the admissible solution',
                    ['Wi', 'alpha', 'N (graded mesh)', 'left-hand side of the step bound of the linear law', 'Newton iterates admissible',
                     'smallest eigenvalue', 'largest tr C / b', 'relative difference Newton - root solver', 'safeguarded solver identical to Newton',
                     'stored reference file against the root solver (relative difference)'], rows,
                    src + [PRE + 'fenep/reference_fenep_uniaxial_<case>.npz'],
                    'N = 64: a-priori scales; N = 128: stepper baseline and collocation mesh; N = 2048 and 4096: reference. The last column compares the '
                    'stored reference with the Richardson extrapolate recomputed with the root solver and is the same for the four meshes of a case.')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['forward', 'inverse', 'fenep', 'model', 'positivity', 'fenep_step']
    for name in todo:
        dict(forward=forward, inverse=inverse, fenep=fenep, model=model, positivity=positivity, fenep_step=fenep_step)[name]()

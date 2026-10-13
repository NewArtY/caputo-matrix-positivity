"""B5 FENE-P benchmark: start-up of uniaxial extension of a Caputo time-fractional FENE-P fluid,

    D_tau^alpha C = Theta^alpha [ Wi (Ghat C + C Ghat^T) - ( f(tr C) C - I ) ],   f = (b - 3)/(b - tr C),

whose right-hand side is defined only on the admissible set {C > 0, tr C < b}. Output maps:
unconstrained symmetric (sym), matrix exponential (logm; SPD, trace unbounded) and the
bounded-trace maps X = b S/(1 + tr S) with S = T T^T (bfactor) or S = expm(Psi) (blogm).
Reference: implicit L1 time stepper on graded meshes with 2048/4096 steps and Richardson extrapolation. The time
stepper (fracsp.spd.solve_l1) solves each step by Newton's iteration and accepts the result only if it is admissible;
the step equation has exactly one admissible solution. On the meshes of this benchmark the Newton iterates are
admissible at every step (checked by b5_fenep_step.py against the root solver, Table_B5_fenep_stored_check.csv).

Usage
-----
  python b5_fenep.py prepare
  python b5_fenep.py run <Wi> <alpha> <output> <loss> <seed>
  python b5_fenep.py campaign [--workers 3] [--seeds 0-4]

Writes results/raw/B5_conformation/fenep/: one JSON file per run, reference_*.npz, stepper_*.json, prepare.json.
Campaigns skip existing result files, record failed runs and exit with status 1 if a run failed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
from fracsp import spd, pinn, reference as ref, operators as op  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = os.path.join(ROOT, 'results', 'raw', 'B5_conformation', 'fenep')
CFG = os.path.join(HERE, '..', 'configs', 'b5_fenep.json')


def load_cfg(path=None):
    return json.load(open(CFG if path is None else path))


def provenance(cfg, seed=None, cfg_path=None):
    """Provenance record stored under the key 'provenance' of every result file written by this driver."""
    return pinn.run_metadata(seed=seed, config=cfg, driver=__file__, sources=[CFG if cfg_path is None else cfg_path])


def cid(Wi, alpha):
    return f'fenep_uniaxial_Wi{Wi:g}_a{alpha:g}'


def test_points(cfg):
    t = cfg['test']
    return spd.test_grid(t['n_uniform'], t['n_layer'], tuple(t['layer']))


def build_model(cfg, Wi):
    return spd.Model('fenep', cfg['flow'], Wi, cfg['Theta'], b=cfg['b'])


def reference(cfg, Wi, alpha):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f'reference_{cid(Wi, alpha)}.npz')
    if os.path.exists(fn):
        d = np.load(fn)
        return d['C'], float(d['est'])
    _, Cr, est = spd.reference_solution(build_model(cfg, Wi), alpha, test_points(cfg), N=cfg['reference']['N'])
    np.savez(fn, C=Cr, est=est)
    return Cr, est


def mesh(cfg, alpha):
    return ref.graded_mesh(1.0, cfg['mesh']['N'], (2.0 - alpha) / alpha)


def adm_metrics(X, ex, b):
    m = spd.metrics(X, ex)
    tr = np.trace(X, axis1=1, axis2=2)
    m['max_trace'] = float(tr.max())
    m['frac_time_trace_ge_b'] = float((tr >= b).mean())
    m['frac_time_inadmissible'] = float(((tr >= b) | (np.linalg.eigvalsh(0.5 * (X + np.swapaxes(X, -1, -2)))[:, 0] <= 0)).mean())
    return m


def run_stepper(cfg, Wi, alpha, repeats=5):
    model = build_model(cfg, Wi)
    t = mesh(cfg, alpha)
    tq = test_points(cfg)
    ex, est = reference(cfg, Wi, alpha)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C, info = spd.solve_l1(model, t, alpha, return_info=True)
        times.append(time.perf_counter() - t0)
    X = spd.interp_linear(t, C, tq)
    res = dict(method='implicit_L1_newton', Wi=Wi, alpha=alpha, N=cfg['mesh']['N'], b=cfg['b'], Theta=cfg['Theta'],
               wall_s_median=float(np.median(times)), newton_iters=info['newton_iters'],
               reference_est_rel_diff_N_2N=est, scale=spd.apriori_scale(model, alpha, cfg['scale']['N']),
               exact_max_cond=float((np.linalg.eigvalsh(ex)[:, -1] / np.linalg.eigvalsh(ex)[:, 0]).max()),
               reference_max_trace=float(np.trace(ex, axis1=1, axis2=2).max()), **adm_metrics(X, ex, cfg['b']),
               provenance=provenance(cfg))
    json.dump(res, open(os.path.join(OUT, f'stepper_{cid(Wi, alpha)}.json'), 'w'), indent=1)
    return res


def run_network(cfg, Wi, alpha, output, loss, seed):
    o = cfg['optimizer']
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg, Wi)
    scale = spd.apriori_scale(model, alpha, cfg['scale']['N'])
    cs = spd.apriori_comp_scale(model, alpha, cfg['scale']['N'])
    t = mesh(cfg, alpha)
    tcol = torch.tensor(t)
    W = op.l1_nonuniform_matrix(tcol, alpha)
    # bounded-trace maps act on S = C/(b - tr C); its a-priori scale comes from the same coarse solve
    if output in ('bfactor', 'blogm'):
        tc = ref.graded_mesh(1.0, cfg['scale']['N'], (2.0 - alpha) / alpha)
        Cc = spd.solve_l1(model, tc, alpha)
        Sc = Cc / (cfg['b'] - np.trace(Cc, axis1=1, axis2=2))[:, None, None]
        net_scale = float(np.sqrt(((Sc - Sc[0]) ** 2).sum((-1, -2))).max())
    else:
        net_scale = scale
    net = spd.SPDNet(output, np.eye(spd.D), alpha, net_scale, cfg['network']['width'], cfg['network']['depth'],
                     cfg['hard_ic']['beta'], comp_scale=cs if loss == 'comp' else None, b=cfg['b'])
    cst = torch.tensor(cs)
    loss_fn = lambda: spd.residual_loss(net, model, tcol, W, alpha, scale, kind=loss, comp_scale=cst)
    t0 = time.perf_counter()
    info = spd.train(net, loss_fn, list(net.parameters()), o['adam_iters'], o['lbfgs_iters'], o['lr'])
    wall = time.perf_counter() - t0
    tq = test_points(cfg)
    ex, _ = reference(cfg, Wi, alpha)
    with torch.no_grad():
        X = net(torch.tensor(tq)).numpy()
        Xc = net(tcol).numpy()
    CL1 = spd.solve_l1(model, t, alpha)
    trc = np.trace(Xc, axis1=1, axis2=2)
    res = dict(method='network', output=output, loss=loss, variant=f'{output}-{loss}', Wi=Wi, alpha=alpha, seed=seed,
               b=cfg['b'], Theta=cfg['Theta'], scale=scale, net_scale=net_scale, N=cfg['mesh']['N'], wall_s=wall,
               nodal_rel_diff_to_L1=float(np.sqrt(((Xc - CL1) ** 2).sum() / (CL1 ** 2).sum())),
               nodal_max_trace=float(trc.max()), finite=bool(np.isfinite(X).all()),
               **adm_metrics(X, ex, cfg['b']), **info)
    Xs = 0.5 * (X + np.swapaxes(X, -1, -2))
    res['curve_min_eig'] = np.linalg.eigvalsh(Xs)[:, 0].tolist()
    res['curve_trace'] = np.trace(X, axis1=1, axis2=2).tolist()
    res['curve_C11'] = X[:, 0, 0].tolist()
    return res


def out_name(Wi, alpha, output, loss, seed):
    return os.path.join(OUT, f'{cid(Wi, alpha)}_{output}-{loss}_s{seed}.json')


def _job(args):
    cfg_path, Wi, alpha, output, loss, seed = args
    torch.set_num_threads(1)
    cfg = load_cfg(cfg_path)
    fn = out_name(Wi, alpha, output, loss, seed)
    if os.path.exists(fn):
        return fn, 'skip'
    try:
        res = run_network(cfg, Wi, alpha, output, loss, seed)
        res['provenance'] = provenance(cfg, seed, cfg_path)
        json.dump(res, open(fn, 'w'), indent=1)
        return fn, (f"ok rel_fro={res['rel_fro']:.3e} min_eig={res['min_eig']:.3e} max_tr={res['max_trace']:.2f} "
                    f"nonfinite={res['nonfinite_events']} wall={res['wall_s']:.1f}s")
    except Exception as e:
        json.dump(dict(method='network', output=output, loss=loss, variant=f'{output}-{loss}', Wi=Wi, alpha=alpha,
                       seed=seed, failed=True, error=repr(e), provenance=provenance(cfg, seed, cfg_path)), open(fn, 'w'), indent=1)
        return fn, 'FAILED ' + repr(e)


def parse_seeds(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('rest', nargs='*')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'prepare':
        for Wi in cfg['Wi']:
            for alpha in cfg['alphas']:
                r = run_stepper(cfg, Wi, alpha)
                print(cid(Wi, alpha), {k: float('%.4g' % r[k]) for k in (
                    'rel_fro', 'min_eig', 'min_eig_exact', 'max_trace', 'reference_max_trace', 'exact_max_cond',
                    'reference_est_rel_diff_N_2N', 'scale', 'wall_s_median')}, 'newton', r['newton_iters'], flush=True)
        json.dump(provenance(cfg, cfg_path=a.config), open(os.path.join(OUT, 'prepare.json'), 'w'), indent=1)      # covers reference_*.npz
    elif a.mode == 'run':
        Wi, alpha, output, loss, seed = float(a.rest[0]), float(a.rest[1]), a.rest[2], a.rest[3], int(a.rest[4])
        res = run_network(cfg, Wi, alpha, output, loss, seed)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith('curve') and k != 'history'}))
    elif a.mode == 'campaign':
        import multiprocessing as mp
        seeds = cfg['seeds'] if a.seeds is None else parse_seeds(a.seeds)
        jobs = [(a.config, Wi, al, v[0], v[1], s) for s in seeds for Wi in cfg['Wi'] for al in cfg['alphas']
                for v in cfg['variants']]
        t0 = time.perf_counter()
        nfail = 0
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for k, (fn, msg) in enumerate(pool.imap_unordered(_job, jobs)):
                nfail += msg.startswith('FAILED')
                print(f'[{k + 1}/{len(jobs)} {time.perf_counter() - t0:7.0f}s] {os.path.basename(fn)} {msg}', flush=True)
        print(f'campaign finished: {len(jobs)} jobs, {nfail} failed', flush=True)
        sys.exit(1 if nfail else 0)
    else:
        raise SystemExit('unknown mode')


if __name__ == '__main__':
    main()

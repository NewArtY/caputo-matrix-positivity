"""B5 forward benchmark: start-up of shear / uniaxial extension of a Caputo time-fractional
upper-convected Maxwell (Hookean dumbbell) fluid. Output maps (unconstrained symmetric, factor,
C0-anchored, matrix exponential) x residual norms (Frobenius, component-scaled, relative, affine-invariant),
all with a hard initial condition, identical network, budget and mesh; implicit L1 stepper on the
same mesh in the same metric. Each flow is posed in a flow-aligned and in a rotated frame.

Usage
-----
  python b5_conformation.py prepare                       exact references + stepper baseline (all cases)
  python b5_conformation.py run <flow> <Wi> <alpha> <frame> <output> <loss> <seed> [--budget 3] [--save]
  python b5_conformation.py campaign [--workers 3] [--seeds 0-4] [--variants sym-rel] [--config <file>]

Writes results/raw/B5_conformation/forward/: one JSON file per run, exact_*.npy, stepper_*.json, prepare.json.
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

OUT = os.path.join(ROOT, 'results', 'raw', 'B5_conformation', 'forward')
CFG = os.path.join(HERE, '..', 'configs', 'b5_forward.json')


def load_cfg(path=None):
    return json.load(open(CFG if path is None else path))


def provenance(cfg, seed=None, cfg_path=None):
    """Provenance record stored under the key 'provenance' of every result file written by this driver."""
    return pinn.run_metadata(seed=seed, config=cfg, driver=__file__, sources=[CFG if cfg_path is None else cfg_path])


def cid(flow, Wi, alpha, frame):
    return f'{flow}_Wi{Wi:g}_a{alpha:g}_{frame}'


def get_case(cfg, flow, Wi):
    for c in cfg['cases']:
        if c['flow'] == flow and abs(c['Wi'] - Wi) < 1e-12:
            return c
    raise KeyError((flow, Wi))


def test_points(cfg):
    t = cfg['test']
    return spd.test_grid(t['n_uniform'], t['n_layer'], tuple(t['layer']))


def frame_Q(cfg, frame):
    return spd.rotation_matrix(cfg['frames'][frame])


def build_model(cfg, case, frame):
    Q = frame_Q(cfg, frame)
    return spd.Model(cfg['law'], case['flow'], case['Wi'], case['Theta'], Ghat=Q @ spd.ghat(case['flow']) @ Q.T)


def exact_on_test(cfg, case, alpha, frame):
    """Exact Mittag-Leffler solution on the test grid, rotated into the frame (cached)."""
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"exact_{cid(case['flow'], case['Wi'], alpha, 'aligned')}.npy")
    if os.path.exists(fn):
        C = np.load(fn)
    else:
        C = spd.exact_ucm(case['flow'], case['Wi'], case['Theta'], alpha, test_points(cfg))
        np.save(fn, C)
    Q = frame_Q(cfg, frame)
    return Q @ C @ Q.T


def mesh(cfg, alpha):
    return ref.graded_mesh(1.0, cfg['mesh']['N'], (2.0 - alpha) / alpha)


def run_stepper(cfg, case, alpha, frame, repeats=7):
    """Implicit L1 on the collocation mesh, evaluated on the test grid (piecewise-linear interpolant)."""
    model = build_model(cfg, case, frame)
    t = mesh(cfg, alpha)
    tq = test_points(cfg)
    ex = exact_on_test(cfg, case, alpha, frame)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = spd.solve_l1(model, t, alpha)
        times.append(time.perf_counter() - t0)
    X = spd.interp_linear(t, C, tq)
    Q = frame_Q(cfg, frame)
    exn = Q @ spd.exact_ucm(case['flow'], case['Wi'], case['Theta'], alpha, t[1:]) @ Q.T      # exact at the nodes
    nodal = float(np.sqrt(((C[1:] - exn) ** 2).sum() / (exn ** 2).sum()))
    wex = np.linalg.eigvalsh(ex)
    res = dict(method='implicit_L1', flow=case['flow'], Wi=case['Wi'], Theta=case['Theta'], alpha=alpha, frame=frame,
               N=cfg['mesh']['N'], wall_s_median=float(np.median(times)), wall_s_min=float(np.min(times)),
               nodal_min_eig=float(np.linalg.eigvalsh(C)[:, 0].min()), nodal_rel_fro=nodal,
               scale=spd.apriori_scale(model, alpha, cfg['scale']['N']),
               comp_scale=spd.apriori_comp_scale(model, alpha, cfg['scale']['N']).tolist(),
               exact_max_norm=float(np.sqrt((ex ** 2).sum((-1, -2))).max()),
               exact_max_cond=float((wex[:, -1] / wex[:, 0]).max()),
               exact_min_eig_of_C_minus_C0=float(np.linalg.eigvalsh(ex - np.eye(spd.D))[:, 0].min()),
               **spd.metrics(X, ex, cfg['pd_tolerance']), provenance=provenance(cfg))
    json.dump(res, open(os.path.join(OUT, f"stepper_{cid(case['flow'], case['Wi'], alpha, frame)}.json"), 'w'), indent=1)
    return res


def run_network(cfg, case, alpha, frame, output, loss, seed, save_curves=True, overrides=None):
    o = dict(cfg['optimizer'])
    if overrides:
        o.update(overrides)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg, case, frame)
    scale = spd.apriori_scale(model, alpha, cfg['scale']['N'])
    cs = spd.apriori_comp_scale(model, alpha, cfg['scale']['N'])
    t = mesh(cfg, alpha)
    tcol = torch.tensor(t)
    W = op.l1_nonuniform_matrix(tcol, alpha)
    net = spd.SPDNet(output, np.eye(spd.D), alpha, scale, cfg['network']['width'], cfg['network']['depth'],
                     cfg['hard_ic']['beta'], comp_scale=cs if loss == 'comp' else None)
    cst = torch.tensor(cs)
    loss_fn = lambda: spd.residual_loss(net, model, tcol, W, alpha, scale, kind=loss, comp_scale=cst)
    t0 = time.perf_counter()
    info = spd.train(net, loss_fn, list(net.parameters()), o['adam_iters'], o['lbfgs_iters'], o['lr'])
    wall = time.perf_counter() - t0
    tq = test_points(cfg)
    ex = exact_on_test(cfg, case, alpha, frame)
    with torch.no_grad():
        X = net(torch.tensor(tq)).numpy()
        Xc = net(tcol).numpy()
        fro_res = float(spd.residual_loss(net, model, tcol, W, alpha, scale, kind='fro'))
    m = spd.metrics(X, ex, cfg['pd_tolerance'])
    CL1 = spd.solve_l1(model, t, alpha)                                # discrete solution on the same mesh
    res = dict(method='network', output=output, loss=loss, variant=f'{output}-{loss}', flow=case['flow'],
               Wi=case['Wi'], Theta=case['Theta'], alpha=alpha, frame=frame, seed=seed, scale=scale,
               N=cfg['mesh']['N'], adam_iters=o['adam_iters'], lbfgs_iters=o['lbfgs_iters'],
               n_params=sum(p.numel() for p in net.parameters()), wall_s=wall,
               ic_error=float(np.abs(Xc[0] - np.eye(spd.D)).max()),
               nodal_min_eig=float(np.linalg.eigvalsh(0.5 * (Xc + np.swapaxes(Xc, -1, -2)))[:, 0].min()),
               nodal_rel_diff_to_L1=float(np.sqrt(((Xc - CL1) ** 2).sum() / (CL1 ** 2).sum())),
               final_fro_residual=fro_res, finite=bool(np.isfinite(X).all()), **m, **info)
    if save_curves:
        Xs = 0.5 * (X + np.swapaxes(X, -1, -2))
        res['curve_min_eig'] = np.linalg.eigvalsh(Xs)[:, 0].tolist()
        res['curve_rel_err'] = (np.sqrt(((X - ex) ** 2).sum((-1, -2))) / np.sqrt((ex ** 2).sum((-1, -2)))).tolist()
    return res


def out_name(case, alpha, frame, output, loss, seed, tag=''):
    return os.path.join(OUT, f"{cid(case['flow'], case['Wi'], alpha, frame)}_{output}-{loss}_s{seed}{tag}.json")


def _job(args):
    cfg_path, flow, Wi, alpha, frame, output, loss, seed = args
    torch.set_num_threads(1)
    cfg = load_cfg(cfg_path)
    case = get_case(cfg, flow, Wi)
    fn = out_name(case, alpha, frame, output, loss, seed)
    if os.path.exists(fn):
        return fn, 'skip'
    try:
        res = run_network(cfg, case, alpha, frame, output, loss, seed)
        res['provenance'] = provenance(cfg, seed, cfg_path)
        json.dump(res, open(fn, 'w'), indent=1)
        return fn, f"ok rel_fro={res['rel_fro']:.3e} min_eig={res['min_eig']:.3e} wall={res['wall_s']:.1f}s"
    except Exception as e:                                         # a failed run is recorded, never dropped
        json.dump(dict(method='network', output=output, loss=loss, variant=f'{output}-{loss}', flow=flow, Wi=Wi,
                       alpha=alpha, frame=frame, seed=seed, failed=True, error=repr(e),
                       provenance=provenance(cfg, seed, cfg_path)), open(fn, 'w'), indent=1)
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
    ap.add_argument('--variants', default=None, help='subset or extra variants, e.g. sym-rel,logm-nat')
    ap.add_argument('--budget', type=float, default=1.0, help="multiplies adam_iters and lbfgs_iters ('run' mode)")
    ap.add_argument('--save', action='store_true', help="'run' mode: write <id>_budget<x>.json next to the campaign results")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    frames = [f for f in cfg['frames'] if f != 'definition']
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'prepare':
        for case in cfg['cases']:
            for alpha in cfg['alphas']:
                for frame in frames:
                    r = run_stepper(cfg, case, alpha, frame)
                    print(cid(case['flow'], case['Wi'], alpha, frame),
                          {k: float('%.4g' % r[k]) for k in ('rel_fro', 'nodal_rel_fro', 'max_rel_fro', 'min_eig',
                                                              'min_eig_exact', 'scale', 'exact_max_cond',
                                                              'exact_min_eig_of_C_minus_C0', 'wall_s_median')}, flush=True)
        json.dump(provenance(cfg, cfg_path=a.config), open(os.path.join(OUT, 'prepare.json'), 'w'), indent=1)      # covers exact_*.npy
    elif a.mode == 'run':
        flow, Wi, alpha, frame, output, loss, seed = (a.rest[0], float(a.rest[1]), float(a.rest[2]), a.rest[3],
                                                      a.rest[4], a.rest[5], int(a.rest[6]))
        case = get_case(cfg, flow, Wi)
        ov = None
        if a.budget != 1.0:
            ov = dict(adam_iters=int(round(a.budget * cfg['optimizer']['adam_iters'])),
                      lbfgs_iters=int(round(a.budget * cfg['optimizer']['lbfgs_iters'])))
        res = run_network(cfg, case, alpha, frame, output, loss, seed, overrides=ov)
        res['budget_factor'] = a.budget
        res['provenance'] = provenance(cfg, seed, a.config)
        if a.save:
            json.dump(res, open(out_name(case, alpha, frame, output, loss, seed, f'_budget{a.budget:g}'), 'w'), indent=1)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith('curve') and k not in ('history', 'provenance')}))
    elif a.mode == 'campaign':
        import multiprocessing as mp
        seeds = cfg['seeds'] if a.seeds is None else parse_seeds(a.seeds)
        variants = cfg['variants'] if a.variants is None else [v.split('-') for v in a.variants.split(',')]
        jobs = [(a.config, c['flow'], c['Wi'], al, fr, v[0], v[1], s) for s in seeds for c in cfg['cases']
                for al in cfg['alphas'] for fr in frames for v in variants]
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

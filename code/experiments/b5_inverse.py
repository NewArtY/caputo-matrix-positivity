"""B5 inverse benchmark: identify (alpha, lambda) of the Caputo time-fractional UCM model from noisy
shear-stress and first-normal-stress-difference data in start-up of steady shear.

Methods
-------
  network : L1-residual network with trainable (alpha, lambda); output maps / residual norms as in
            the forward benchmark
  nls     : classical nonlinear least squares through the implicit L1 stepper
            (scipy.optimize.least_squares, multi-start)
  crb     : Cramer-Rao bound of the estimation problem (exact Mittag-Leffler sensitivities)

Usage
-----
  python b5_inverse.py nls                              all (alpha, noise, seed), both meshes, + CRB
  python b5_inverse.py run <alpha> <noise> <seed> <output> <loss> [--data-weight w]
  python b5_inverse.py campaign [--workers 3] [--seeds 0-9] [--variants factor-fro,sym-fro]

Writes results/raw/B5_conformation/inverse/: one JSON file per fit, cramer_rao.json, clean_*.npy, exact_state_*.npy, prepare.json.
Campaigns skip existing result files, record failed runs and exit with status 1 if a run failed.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
from fracsp import spd, pinn, reference as ref, operators as op  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = os.path.join(ROOT, 'results', 'raw', 'B5_conformation', 'inverse')
CFG = os.path.join(HERE, '..', 'configs', 'b5_inverse.json')
A_LO, A_W = 0.3, 0.7                                   # alpha = A_LO + A_W sigmoid(a)


def load_cfg(path=None):
    return json.load(open(CFG if path is None else path))


def provenance(cfg, seed=None, cfg_path=None):
    """Provenance record stored under the key 'provenance' of every result file written by this driver."""
    return pinn.run_metadata(seed=seed, config=cfg, driver=__file__, sources=[CFG if cfg_path is None else cfg_path])


def obs_times(cfg):
    return np.logspace(-2.0, 0.0, cfg['observations']['n_times'])


def exact_obs(cfg, alpha, ell, tau):
    """Exact observables (C12, C11 - C22) for the parameters (alpha, ell = lambda/T)."""
    C = spd.exact_ucm('shear', cfg['strain_total'] * ell, 1.0 / ell, alpha, tau)
    return np.stack([C[:, 0, 1], C[:, 0, 0] - C[:, 1, 1]])


def make_data(cfg, alpha_true, noise, seed):
    """Clean data from the exact Mittag-Leffler solution (cached) plus additive Gaussian noise."""
    os.makedirs(OUT, exist_ok=True)
    tau = obs_times(cfg)
    fn = os.path.join(OUT, f'clean_a{alpha_true:g}.npy')
    if os.path.exists(fn):
        clean = np.load(fn)
    else:
        clean = exact_obs(cfg, alpha_true, cfg['ell_true'], tau)
        np.save(fn, clean)
    rms = np.sqrt((clean ** 2).mean(1))
    rng = np.random.default_rng(cfg['observations']['noise_seed_offset'] + seed)
    y = clean + noise * rms[:, None] * rng.standard_normal(clean.shape)
    return dict(tau=tau, y=y, clean=clean, sd=noise * rms, sc=np.sqrt((y ** 2).mean(1)))


def initial_guess(cfg, seed, k=0):
    rng = np.random.default_rng(cfg['init']['seed_offset'] + 1000 * k + seed)
    a0 = rng.uniform(*cfg['init']['alpha_range'])
    l0 = math.exp(rng.uniform(math.log(cfg['init']['ell_range'][0]), math.log(cfg['init']['ell_range'][1])))
    return a0, l0


def data_scales(data):
    y = data['y']
    s = float(np.sqrt(y[1] ** 2 + 2 * y[0] ** 2).max())
    cs = np.ones((spd.D, spd.D))
    cs[0, 0] = max(np.abs(y[1]).max(), 1.0)
    cs[0, 1] = cs[1, 0] = max(np.abs(y[0]).max(), 1.0)
    return s, cs


def state_metrics(cfg, alpha_true, X, tq):
    fn = os.path.join(OUT, f'exact_state_a{alpha_true:g}.npy')
    if os.path.exists(fn):
        ex = np.load(fn)
    else:
        ex = spd.exact_ucm('shear', cfg['strain_total'] * cfg['ell_true'], 1.0 / cfg['ell_true'], alpha_true, tq)
        np.save(fn, ex)
    m = spd.metrics(X, ex)
    return {('state_' + k): v for k, v in m.items()}


# ----------------------------------------------------------------------------------------------
# classical nonlinear least squares through the implicit L1 stepper
# ----------------------------------------------------------------------------------------------
def stepper_obs(cfg, alpha, ell, tau, t):
    model = spd.Model('ucm', 'shear', cfg['strain_total'] * ell, 1.0 / ell)
    C = spd.solve_l1(model, t, alpha)
    Co = spd.interp_linear(t, C, tau)
    return np.stack([Co[:, 0, 1], Co[:, 0, 0] - Co[:, 1, 1]]), C


def run_nls(cfg, alpha_true, noise, seed, N):
    data = make_data(cfg, alpha_true, noise, seed)
    t = ref.graded_mesh(1.0, N, cfg['mesh']['grading_r'])
    nfev = [0]

    def fun(x):
        nfev[0] += 1
        yh, _ = stepper_obs(cfg, x[0], math.exp(x[1]), data['tau'], t)
        return ((yh - data['y']) / data['sc'][:, None]).reshape(-1) / math.sqrt(data['y'].size)
    lb = [cfg['nls']['bounds_alpha'][0], math.log(cfg['nls']['bounds_ell'][0])]
    ub = [cfg['nls']['bounds_alpha'][1], math.log(cfg['nls']['bounds_ell'][1])]
    t0 = time.perf_counter()
    best, starts = None, []
    for k in range(cfg['nls']['n_starts']):
        a0, l0 = initial_guess(cfg, seed, k)
        sol = least_squares(fun, [a0, math.log(l0)], bounds=(lb, ub), method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12)
        starts.append(dict(alpha0=a0, ell0=l0, alpha=float(sol.x[0]), ell=float(math.exp(sol.x[1])),
                           cost=float(2 * sol.cost), status=int(sol.status)))
        if best is None or sol.cost < best.cost:
            best = sol
    wall = time.perf_counter() - t0
    ahat, lhat = float(best.x[0]), float(math.exp(best.x[1]))
    tq = spd.test_grid(cfg['test']['n_uniform'], cfg['test']['n_layer'], tuple(cfg['test']['layer']))
    _, C = stepper_obs(cfg, ahat, lhat, data['tau'], t)
    X = spd.interp_linear(t, C, tq)
    res = dict(method=f'nls{N}', alpha_true=alpha_true, ell_true=cfg['ell_true'], noise=noise, seed=seed, N=N,
               alpha_hat=ahat, ell_hat=lhat, data_misfit=float(2 * best.cost), wall_s=wall, n_forward_solves=nfev[0],
               n_distinct_minima=len({(round(s['alpha'], 4), round(s['ell'], 4)) for s in starts}), starts=starts,
               **state_metrics(cfg, alpha_true, X, tq), provenance=provenance(cfg, seed))
    json.dump(res, open(os.path.join(OUT, f'nls{N}_a{alpha_true:g}_n{noise:g}_s{seed}.json'), 'w'), indent=1)
    return res


def cramer_rao(cfg, alpha_true, noise):
    """Cramer-Rao bound for (alpha, log ell) from exact sensitivities (central differences)."""
    tau = obs_times(cfg)
    e = 1e-5
    l0 = cfg['ell_true']
    Ja = (exact_obs(cfg, alpha_true + e, l0, tau) - exact_obs(cfg, alpha_true - e, l0, tau)) / (2 * e)
    Jl = (exact_obs(cfg, alpha_true, l0 * math.exp(e), tau) - exact_obs(cfg, alpha_true, l0 * math.exp(-e), tau)) / (2 * e)
    clean = exact_obs(cfg, alpha_true, l0, tau)
    rms = np.sqrt((clean ** 2).mean(1))
    out = dict(alpha_true=alpha_true, noise=noise)
    if noise > 0:
        sd = noise * rms
        J = np.stack([(Ja / sd[:, None]).reshape(-1), (Jl / sd[:, None]).reshape(-1)], 1)
        cov = np.linalg.inv(J.T @ J)
        out.update(sd_alpha=float(math.sqrt(cov[0, 0])), sd_log_ell=float(math.sqrt(cov[1, 1])),
                   corr=float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])))
    else:
        out.update(sd_alpha=0.0, sd_log_ell=0.0, corr=None)
    return out


# ----------------------------------------------------------------------------------------------
# network with trainable (alpha, lambda)
# ----------------------------------------------------------------------------------------------
def run_network(cfg, alpha_true, noise, seed, output, loss, data_weight=None, overrides=None):
    wd = cfg['data_weight'] if data_weight is None else data_weight
    o = dict(cfg['optimizer'])
    if overrides:
        o.update(overrides)
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = make_data(cfg, alpha_true, noise, seed)
    s, cs = data_scales(data)
    a0, l0 = initial_guess(cfg, seed, 0)
    araw = torch.tensor(math.log((a0 - A_LO) / A_W / (1 - (a0 - A_LO) / A_W)), requires_grad=True)
    p = torch.tensor(math.log(l0), requires_grad=True)
    Gam = cfg['strain_total']

    def get():
        alpha = A_LO + A_W * torch.sigmoid(araw)
        ell = torch.exp(p)
        return alpha, ell

    model = spd.Model('ucm', 'shear', 1.0, 1.0)             # Wi and Theta are passed as tensors
    tcol = torch.tensor(ref.graded_mesh(1.0, cfg['mesh']['N'], cfg['mesh']['grading_r']))
    net = spd.SPDNet(output, np.eye(spd.D), None, s, cfg['network']['width'], cfg['network']['depth'],
                     cfg['hard_ic']['beta'], comp_scale=cs if loss == 'comp' else None)
    cst = torch.tensor(cs)
    tob = torch.tensor(data['tau'])
    yob = torch.tensor(data['y'])
    sc = torch.tensor(data['sc'])

    def terms():
        alpha, ell = get()
        W = op.l1_nonuniform_matrix(tcol, alpha)
        Lr = spd.residual_loss(net, model, tcol, W, alpha, s, Wi=Gam * ell, Theta=1.0 / ell, kind=loss, comp_scale=cst)
        Xo = net(tob, alpha)
        yh = torch.stack([Xo[:, 0, 1], Xo[:, 0, 0] - Xo[:, 1, 1]])
        Ld = (((yh - yob) / sc[:, None]) ** 2).mean()
        return Lr, Ld

    def loss_fn():
        Lr, Ld = terms()
        return Lr + wd * Ld

    params = list(net.parameters()) + [araw, p]
    t0 = time.perf_counter()
    info = spd.train(net, loss_fn, params, o['adam_iters'], o['lbfgs_iters'], o['lr'])
    wall = time.perf_counter() - t0
    with torch.no_grad():
        alpha, ell = get()
        try:
            Lr, Ld = [float(x) for x in terms()]
        except RuntimeError:
            Lr, Ld = float('nan'), float('nan')
        tq = spd.test_grid(cfg['test']['n_uniform'], cfg['test']['n_layer'], tuple(cfg['test']['layer']))
        X = net(torch.tensor(tq), alpha).numpy()
    res = dict(method='network', output=output, loss=loss, variant=f'{output}-{loss}', alpha_true=alpha_true,
               ell_true=cfg['ell_true'], noise=noise, seed=seed, data_weight=wd, alpha0=a0, ell0=l0,
               alpha_hat=float(alpha), ell_hat=float(ell), residual_loss=Lr, data_misfit=Ld, wall_s=wall,
               n_loss_evals=o['adam_iters'] + info['lbfgs_evals'], finite=bool(np.isfinite(X).all()),
               **state_metrics(cfg, alpha_true, X, tq), **info)
    return res


def out_name(alpha_true, noise, seed, output, loss, tag=''):
    return os.path.join(OUT, f'net_{output}-{loss}_a{alpha_true:g}_n{noise:g}_s{seed}{tag}.json')


def _job(args):
    cfg_path, alpha_true, noise, seed, output, loss = args
    torch.set_num_threads(1)
    cfg = load_cfg(cfg_path)
    fn = out_name(alpha_true, noise, seed, output, loss)
    if os.path.exists(fn):
        return fn, 'skip'
    try:
        res = run_network(cfg, alpha_true, noise, seed, output, loss)
        res['provenance'] = provenance(cfg, seed, cfg_path)
        json.dump(res, open(fn, 'w'), indent=1)
        return fn, (f"ok alpha={res['alpha_hat']:.4f} ell={res['ell_hat']:.4f} state_err={res['state_rel_fro']:.2e} "
                    f"wall={res['wall_s']:.1f}s")
    except Exception as e:
        json.dump(dict(method='network', output=output, loss=loss, variant=f'{output}-{loss}', alpha_true=alpha_true,
                       noise=noise, seed=seed, failed=True, error=repr(e), provenance=provenance(cfg, seed, cfg_path)),
                  open(fn, 'w'), indent=1)
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
    ap.add_argument('--data-weight', type=float, default=None)
    ap.add_argument('--variants', default=None, help='e.g. factor-fro,sym-fro')
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    os.makedirs(OUT, exist_ok=True)
    seeds = cfg['seeds'] if a.seeds is None else parse_seeds(a.seeds)
    noises = cfg['observations']['noise_levels']
    if a.mode == 'nls':
        crb = [cramer_rao(cfg, al, nz) for al in cfg['alphas_true'] for nz in noises]
        json.dump(crb, open(os.path.join(OUT, 'cramer_rao.json'), 'w'), indent=1)
        json.dump(provenance(cfg, cfg_path=a.config), open(os.path.join(OUT, 'prepare.json'), 'w'), indent=1)      # covers cramer_rao.json, *.npy
        print('CRB', crb, flush=True)
        for N in cfg['nls']['meshes_N']:
            for al in cfg['alphas_true']:
                for nz in noises:
                    for s in seeds:
                        r = run_nls(cfg, al, nz, s, N)
                        print(f"nls{N} a={al} noise={nz} s={s}: alpha={r['alpha_hat']:.5f} ell={r['ell_hat']:.5f} "
                              f"misfit={r['data_misfit']:.3e} solves={r['n_forward_solves']} wall={r['wall_s']:.2f}s "
                              f"minima={r['n_distinct_minima']}", flush=True)
    elif a.mode == 'run':
        al, nz, s, output, loss = float(a.rest[0]), float(a.rest[1]), int(a.rest[2]), a.rest[3], a.rest[4]
        res = run_network(cfg, al, nz, s, output, loss, a.data_weight)
        print(json.dumps({k: v for k, v in res.items() if k != 'history'}))
    elif a.mode == 'campaign':
        import multiprocessing as mp
        variants = cfg['variants'] if a.variants is None else [v.split('-') for v in a.variants.split(',')]
        jobs = [(a.config, al, nz, s, v[0], v[1]) for v in variants for s in seeds for al in cfg['alphas_true'] for nz in noises]
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

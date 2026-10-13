"""B7: identification of (alpha, Gamma_down, Gamma_phi) of the driven two-level system from noisy partial observations.

  pinn : L1-residual network with trainable parameters, loss = mean ||residual||_F^2 + w_data * mean (misfit)^2
  nls  : classical nonlinear least squares through the implicit L1 time stepper (scipy.optimize.least_squares,
         multi-start), on the mesh of the networks and on meshes that contain the observation times, with and without
         projection of the noisy data on the admissible set

Usage: python b7_inverse.py prepare
       python b7_inverse.py pinn --design xz --noise 0.01 --map add --weight 100 --real 0 [--stage manual]
       python b7_inverse.py nls  --design xz --noise 0.01 --real 0 [--stage manual]
Writes results/raw/B7_inverse/reference/* (exact reference, data, Cramer-Rao bounds) and
       results/raw/B7_inverse/<stage>/<name>.json
"""
import argparse, functools, json, math, os, sys, time
import numpy as np, torch
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import operators as op, reference as ref, pinn
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
CFG = os.path.join(HERE, '..', 'configs', 'b7_inverse.json'); cfg = json.load(open(CFG))
RAW = os.path.join(ROOT, 'results', 'raw', 'B7_inverse'); REF = os.path.join(RAW, 'reference')
CD = torch.complex128
THETA = np.array([cfg['alpha_true'], cfg['phys_true']['Gdown'], cfg['phys_true']['Gphi']])
OBS = {'sx': lambda X: 2 * X[..., 0, 1].real, 'sy': lambda X: -2 * X[..., 0, 1].imag,
       'sz': lambda X: (X[..., 0, 0] - X[..., 1, 1]).real, 'pe': lambda X: X[..., 0, 0].real}


def parts():
    """Superoperators with L(theta) = S_H + Gdown S_d + Gphi S_p (the generator of fracsp.pinn.liouvillian)."""
    H = 0.5 * cfg['phys_true']['Omega'] * pinn.sx.numpy(); Z = np.zeros((2, 2))
    return pinn.gksl_superop(H, []), pinn.gksl_superop(Z, [pinn.sm.numpy()]), pinn.gksl_superop(Z, [pinn.sz.numpy() / math.sqrt(2)])


S_H, S_D, S_P = parts()


def generator(theta):
    return S_H + theta[1] * S_D + theta[2] * S_P


def rho0_of(design):
    return pinn.bloch_to_rho(torch.tensor([float(x) for x in cfg['designs'][design]['rho0_bloch']])).numpy()


def observe(X, design):
    """Observables of a batch of states (numpy or torch), shape (..., n_observables)."""
    vals = [OBS[o](X) for o in cfg['designs'][design]['observables']]
    return torch.stack(vals, -1) if torch.is_tensor(X) else np.stack(vals, -1)


def obs_times():
    return np.sort(np.random.default_rng(cfg['obs_seed']).uniform(0.02, 1.0, cfg['n_obs']))


def grids():
    T = cfg['test']; t, iu, il = pinn.test_grid(T['n_uniform'], T['log_min'], T['per_decade'])
    return ref.graded_mesh(1.0, cfg['mesh']['N'], cfg['mesh']['r']), t, iu, il


@functools.lru_cache(maxsize=None)
def truth(design):
    """Exact states of the true model on the test grid and at the nodes, exact observables at the observation times."""
    tcol, t, iu, il = grids(); R = pinn.load_reference(os.path.join(REF, f'exact_a{cfg["alpha_true"]}.npz'))
    assert np.array_equal(R['t'], t) and np.array_equal(R['tcol'], tcol), 'stored reference does not match the configuration'
    Xt, Xn = pinn.reference_states(R, generator(THETA), rho0_of(design))
    z = np.load(os.path.join(REF, 'data.npz'))
    return dict(t=t, is_uniform=iu, is_log=il, ref=Xt, ref_nodes=Xn, tob=z['tob'], y_exact=z[f'y_{design}'])


def data(design, noise, real):
    T = truth(design)
    return T['y_exact'] + noise * np.random.default_rng(2000 + real).standard_normal(T['y_exact'].shape)


def project_data(y, design):
    """Projection of noisy data on the admissible set: (sx, sz) on the unit disc, a population on [0, 1]."""
    if design == 'xz':
        return y / np.maximum(1.0, np.linalg.norm(y, axis=1, keepdims=True))
    return np.clip(y, 0.0, 1.0)


def init_guesses(real, n):
    I = cfg['init']; rng = np.random.default_rng(3000 + real); out = []
    for _ in range(n):
        out.append([rng.uniform(*I['alpha']), math.exp(rng.uniform(*np.log(I['Gdown_log_uniform']))),
                    math.exp(rng.uniform(*np.log(I['Gphi_log_uniform'])))])
    return np.array(out)


class Stepper:
    """Observables of the implicit L1 solution as a function of theta = (alpha, Gdown, Gphi)."""
    def __init__(s, design, N, obs_nodes):
        s.design = design; s.rho0 = rho0_of(design).reshape(-1); s.tob = obs_times(); s.nsolve = 0
        tg = ref.graded_mesh(1.0, N, cfg['mesh']['r'])
        s.t = np.unique(np.concatenate([tg, s.tob])) if obs_nodes else tg
        s.idx = np.searchsorted(s.t, s.tob) if obs_nodes else None

    def states(s, theta):
        s.nsolve += 1
        return pinn.l1_stepper(generator(theta), s.rho0, s.t, theta[0]).reshape(-1, 2, 2)

    def __call__(s, theta):
        X = s.states(theta)
        Xo = X[s.idx] if s.idx is not None else pinn.interp_nodes(s.tob, s.t, X.reshape(len(s.t), -1)).reshape(-1, 2, 2)
        return observe(Xo, s.design)


def summary(theta):
    rel = (np.asarray(theta) - THETA) / THETA
    return dict(alpha_hat=float(theta[0]), Gdown_hat=float(theta[1]), Gphi_hat=float(theta[2]),
                rel_err_rms=float(np.sqrt(np.mean(rel ** 2))))


# ---------------------------------------------------------------------------------------------- reference

def prepare(workers=1):
    os.makedirs(REF, exist_ok=True); a = cfg['alpha_true']; tcol, t, iu, il = grids(); S = generator(THETA)
    assert np.abs(S - pinn.gksl_superop(*pinn.two_level_ops(**cfg['phys_true']))).max() < 1e-14
    path = os.path.join(REF, f'exact_a{a}.npz'); jobs = pinn.reference_jobs(path, S, tcol, t, a, max(workers, 1))
    if pinn.run_campaign(pinn.ml_table_job, jobs, workers, os.path.join(REF, 'prepare.log'), 'B7 prepare'):
        sys.exit('B7 prepare: tabulation of the Mittag-Leffler function FAILED')
    checks = pinn.assemble_reference(path, S, tcol, t, iu, il, a, jobs); tob = obs_times(); out = dict(tob=tob); crb = {}
    rows = ['design,noise_sd,parameter,true_value,crb_sd,crb_rel_sd']
    for dname, d in cfg['designs'].items():
        Xo, _ = pinn.ml_exact(S, rho0_of(dname).reshape(-1), tob, a)
        out[f'y_{dname}'] = observe(Xo.reshape(-1, 2, 2), dname)
        fwd = Stepper(dname, cfg['crb']['N'], True); J = []
        for k in range(3):
            e = 1e-5; tp = THETA.copy(); tm = THETA.copy(); tp[k] += e; tm[k] -= e
            J.append(((fwd(tp) - fwd(tm)) / (2 * e)).ravel())
        J = np.array(J).T; C = np.linalg.inv(J.T @ J); sd = np.sqrt(np.diag(C))
        crb[dname] = dict(unit_noise_sd=sd.tolist(), correlation=(C / np.outer(sd, sd)).tolist(), cond_jacobian=float(np.linalg.cond(J)),
                          model_error_at_obs=float(np.abs(fwd(THETA) - out[f'y_{dname}']).max()))
        for s in d['noise_sd']:
            rows += [f'{dname},{s},{p},{THETA[k]},{s * sd[k]:.6e},{s * sd[k] / THETA[k]:.6e}' for k, p in enumerate(cfg['unknowns'])]
    np.savez(os.path.join(REF, 'data.npz'), **out)
    open(os.path.join(REF, 'crb.csv'), 'w').write('\n'.join(rows) + '\n'); print('\n'.join(rows))
    pinn.write_json(os.path.join(REF, 'prepare.json'), dict(checks=checks, crb=crb, **pinn.run_metadata(config=cfg, driver=__file__, sources=[CFG])))


# ---------------------------------------------------------------------------------------------- jobs

def make_job(stage, method, design, noise, real, map=None, weight=None, adam_iters=None, lbfgs_iters=None):
    o = cfg['pinn']['optimizer']
    j = dict(stage=stage, method=method, design=design, noise=noise, real=real)
    if method == 'pinn':
        j.update(map=map, weight=weight, adam_iters=o['adam_iters'] if adam_iters is None else adam_iters,
                 lbfgs_iters=o['lbfgs_iters'] if lbfgs_iters is None else lbfgs_iters)
        j['name'] = f'pinn_{design}_n{noise:g}_{map}_w{weight:g}_r{real}'
    else:
        j['name'] = f'nls_{design}_n{noise:g}_r{real}'
    j['out'] = os.path.join(RAW, stage, j['name'] + '.json')
    return j


def finish(j, res, extra_conf):
    conf = dict(job={k: v for k, v in j.items() if k != 'out'}, alpha_true=cfg['alpha_true'], phys_true=cfg['phys_true'],
                design=cfg['designs'][j['design']], n_obs=cfg['n_obs'], obs_seed=cfg['obs_seed'], init=cfg['init'], mesh=cfg['mesh'],
                test=cfg['test'], config_file='configs/b7_inverse.json', **extra_conf)
    res.update(benchmark='B7', peak_rss_mb=pinn.peak_rss_mb(), **{k: j[k] for k in ('stage', 'name', 'method', 'design', 'noise', 'real')})
    res.update(pinn.run_metadata(seed=j['real'], config=conf, driver=__file__, sources=[CFG]))
    os.makedirs(os.path.dirname(j['out']), exist_ok=True); pinn.write_json(j['out'], res)


def run_pinn(j):
    design, real = j['design'], j['real']; T = truth(design); P = cfg['pinn']; net = P['network']
    torch.manual_seed(real); np.random.seed(real)
    x0 = init_guesses(real, 1)[0]; y = torch.tensor(data(design, j['noise'], real)); rho0 = rho0_of(design)
    model = pinn.DensityNet(torch.tensor(rho0), map=j['map'], ic='hard', gate=P['gate'], alpha=float(x0[0]), beta=net['beta'],
                            width=net['width'], depth=net['depth'])
    p0 = (x0[0] - 0.05) / 0.95
    araw = torch.tensor(math.log(p0 / (1 - p0)), requires_grad=True); u = torch.tensor(np.log(x0[1:]), requires_grad=True)
    tcol = torch.tensor(grids()[0]); tob = torch.tensor(T['tob']); SH, SD, SP = (torch.tensor(S) for S in (S_H, S_D, S_P))

    def theta():
        return 0.05 + 0.95 * torch.sigmoid(araw), torch.exp(u)

    def loss():
        a, g = theta()
        R = pinn.l1_residual(model(tcol, alpha=a), op.l1_nonuniform_matrix(tcol, a).to(CD), SH + g[0] * SD + g[1] * SP)
        mis = observe(model(tob, alpha=a), design) - y
        return {'phys': (R.abs() ** 2).sum((-1, -2)).mean(), 'data': j['weight'] * (mis ** 2).mean()}
    info = pinn.fit(list(model.parameters()) + [araw, u], loss, j['adam_iters'], j['lbfgs_iters'], P['optimizer']['lr'])
    with torch.no_grad():
        a, g = theta(); th = np.array([float(a), float(g[0]), float(g[1])])
        X = model(torch.tensor(T['t']), alpha=a).numpy()
    res = dict(map=j['map'], weight=j['weight'], theta_init=x0.tolist(), **summary(th), **info,
               final_residual=info['final_terms']['phys'], final_data_misfit=info['final_terms']['data'] / j['weight'],
               n_params=sum(p.numel() for p in model.parameters()) + 3)
    if np.isfinite(X).all():
        res.update({'state_' + k: v for k, v in pinn.state_metrics(X, T['ref'], T['t'], T['is_uniform'], T['is_log'], rho0,
                                                                    cfg['eig_tol']).items()})
    elif res['status'] == 'ok':
        res['status'] = 'non-finite network output'
    finish(j, res, dict(pinn=P))
    return f"{res['status']} alpha={th[0]:.4f} Gd={th[1]:.4f} Gp={th[2]:.4f} wall={info['wall']:.0f}s"


def run_nls(j):
    design, real = j['design'], j['real']; T = truth(design); Nc = cfg['nls']; rho0 = rho0_of(design)
    y = data(design, j['noise'], real); starts = init_guesses(real, Nc['starts']); B = Nc['bounds']
    lb = [B[k][0] for k in cfg['unknowns']]; ub = [B[k][1] for k in cfg['unknowns']]; fits = {}
    for mname, m in Nc['meshes'].items():
        for proj in (False, True):
            fwd = Stepper(design, m['N'], m['obs_nodes']); yy = project_data(y, design) if proj else y
            if proj and np.array_equal(yy, y):          # the projection leaves the data unchanged: same fit
                fits[mname + '_proj'] = dict(fits[mname], identical_to_unprojected=True); continue
            t0 = time.perf_counter(); sols = []
            for x0 in starts:
                sols.append(least_squares(lambda th: (fwd(th) - yy).ravel(), x0, jac='3-point', bounds=(lb, ub), x_scale='jac',
                                          xtol=1e-12, ftol=1e-12, gtol=1e-12))
            wall = time.perf_counter() - t0; best = min(sols, key=lambda r: r.cost); th = best.x
            same = sum(bool(np.all(np.abs(r.x - th) < 1e-5 * np.abs(th))) for r in sols)
            Xm = fwd.states(th); Xt = pinn.interp_nodes(T['t'], fwd.t, Xm.reshape(len(fwd.t), -1)).reshape(-1, 2, 2)
            dof = max(yy.size - 3, 1); C = np.linalg.inv(best.jac.T @ best.jac) * 2 * best.cost / dof
            f = dict(**summary(th), cost=float(best.cost), rms_misfit=float(np.sqrt(2 * best.cost / yy.size)), wall=wall,
                     n_solves=fwd.nsolve, n_starts=len(starts), starts_reaching_best=int(same),
                     all_converged=bool(all(r.success for r in sols)), sd_estimate=np.sqrt(np.diag(C)).tolist(),
                     n_nodes=len(fwd.t) - 1, data_points_changed_by_projection=int((np.abs(yy - y).max(axis=1) > 0).sum()),
                     min_eig_nodes=float(np.linalg.eigvalsh(0.5 * (Xm + Xm.conj().transpose(0, 2, 1)))[:, 0].min()))
            f.update({'state_' + k: v for k, v in pinn.state_metrics(Xt, T['ref'], T['t'], T['is_uniform'], T['is_log'], rho0,
                                                                      cfg['eig_tol']).items()})
            fits[mname + ('_proj' if proj else '')] = f
    res = dict(fits=fits, theta_starts=starts.tolist(), status='ok')
    finish(j, res, dict(nls=Nc))
    b = fits['N128']
    return f"alpha={b['alpha_hat']:.4f} Gd={b['Gdown_hat']:.4f} Gp={b['Gphi_hat']:.4f} wall={b['wall']:.2f}s"


def run_job(j):
    return run_pinn(j) if j['method'] == 'pinn' else run_nls(j)


def main():
    if sys.argv[1] == 'prepare':
        return prepare(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    ap = argparse.ArgumentParser(); ap.add_argument('method'); ap.add_argument('--design', required=True)
    ap.add_argument('--noise', type=float, required=True); ap.add_argument('--real', type=int, required=True)
    ap.add_argument('--map', default='add'); ap.add_argument('--weight', type=float, default=100.0)
    ap.add_argument('--adam', type=int); ap.add_argument('--lbfgs', type=int); ap.add_argument('--stage', default='manual')
    a = ap.parse_args()
    j = make_job(a.stage, a.method, a.design, a.noise, a.real, a.map, a.weight, a.adam, a.lbfgs)
    print(j['name'], run_job(j)); print('written', j['out'])


if __name__ == '__main__':
    main()

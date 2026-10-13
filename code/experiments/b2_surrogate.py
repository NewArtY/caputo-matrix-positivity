"""Use-case check: a parametric surrogate rho(tau; alpha, Gamma_down) of the driven two-level system, trained once
by the L1 residual on a pool of parameter points (no data), against the implicit L1 time stepper.

Usage: python b2_surrogate.py prepare [workers]    exact solution and time stepper on the test parameter grid
       python b2_surrogate.py train --map add --seed 0 [--stage manual]
       python b2_surrogate.py campaign [workers]   all maps and seeds of configs/b2_surrogate.json
Writes results/raw/B2_surrogate/reference/{exact.npz, stepper.csv, prepare.json} and
       results/raw/B2_surrogate/<stage>/<name>.json
A query is one trajectory, i.e. the state at all test times for one parameter point.
"""
import argparse, functools, json, os, sys, time
import numpy as np, torch
from scipy.stats import qmc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import operators as op, reference as ref, pinn
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
CFG = os.path.join(HERE, '..', 'configs', 'b2_surrogate.json'); cfg = json.load(open(CFG))
RAW = os.path.join(ROOT, 'results', 'raw', 'B2_surrogate'); REF = os.path.join(RAW, 'reference')
CD = torch.complex128; (A0, A1), (G0, G1) = cfg['ranges']['alpha'], cfg['ranges']['Gdown']
RHO0 = pinn.bloch_to_rho(torch.tensor([float(x) for x in cfg['rho0_bloch']])).numpy()
Z2 = np.zeros((2, 2))
S_H = pinn.gksl_superop(0.5 * cfg['phys']['Omega'] * pinn.sx.numpy(), [np.sqrt(cfg['phys']['Gphi'] / 2) * pinn.sz.numpy()])
S_D = pinn.gksl_superop(Z2, [pinn.sm.numpy()])


def generator(gd):
    return S_H + gd * S_D


def scale(alpha, gd):
    """Network inputs: the parameters scaled to [-1, 1]."""
    return torch.stack([(2 * alpha - A0 - A1) / (A1 - A0), (2 * gd - G0 - G1) / (G1 - G0)], -1)


def test_params():
    T = cfg['test']
    return np.array([(a, g) for a in np.linspace(A0, A1, T['n_alpha']) for g in np.linspace(G0, G1, T['n_Gdown'])])


def test_times():
    T = cfg['test']
    return pinn.test_grid(T['n_uniform'], T['log_min'], T['per_decade'])


def mesh(alpha, N):
    return ref.graded_mesh(1.0, N, (2 - alpha) / alpha)


# ---------------------------------------------------------------------------------------------- reference

def prepare(workers=1):
    os.makedirs(REF, exist_ok=True); P = test_params(); t, iu, il = test_times(); jobs = []
    for k, (a, g) in enumerate(P):
        jobs.append(pinn.reference_jobs(os.path.join(REF, f'_exact_{k}.npz'), generator(g), np.zeros(0), t, float(a), 1))
    if pinn.run_campaign(pinn.ml_table_job, sum(jobs, []), workers, os.path.join(REF, 'prepare.log'), 'surrogate prepare'):
        sys.exit('surrogate prepare: tabulation of the Mittag-Leffler function FAILED')
    X = np.zeros((len(P), len(t), 2, 2), dtype=complex); cond = []
    rows = ['alpha,Gdown,N,wall_ms,rel_fro,max_fro,min_eig,max_trace_defect']
    for k, (a, g) in enumerate(P):
        path = os.path.join(REF, f'_exact_{k}.npz'); S = generator(g)
        cond.append(pinn.assemble_reference(path, S, np.zeros(0), t, iu, il, float(a), jobs[k])['cond_V'])
        R = pinn.load_reference(path); os.remove(path)
        X[k] = pinn.ml_exact(S, RHO0.reshape(-1), t, None, (R['lam'], R['V']), R['E_test'])[0].reshape(-1, 2, 2)
        for N in cfg['stepper']['N']:
            tn = mesh(a, N); ws = []
            for _ in range(cfg['stepper']['timing_repeats']):
                t0 = time.perf_counter(); Xn = pinn.l1_stepper(S, RHO0.reshape(-1), tn, a); Xi = pinn.interp_nodes(t, tn, Xn)
                ws.append(time.perf_counter() - t0)
            m = pinn.state_metrics(Xi.reshape(-1, 2, 2), X[k], t, iu, il, RHO0, cfg['eig_tol'])
            rows.append(f'{a:.6f},{g:.6f},{N},{np.median(ws) * 1e3:.4f},{m["rel_fro"]:.6e},{m["max_fro"]:.6e},{m["min_eig"]:.3e},'
                        f'{m["max_trace_defect"]:.3e}')
    np.savez(os.path.join(REF, 'exact.npz'), params=P, t=t, is_uniform=iu, is_log=il, X=X)
    open(os.path.join(REF, 'stepper.csv'), 'w').write('\n'.join(rows) + '\n')
    ev = np.linalg.eigvalsh(X)
    pinn.write_json(os.path.join(REF, 'prepare.json'), dict(
        checks=dict(max_cond_V=float(max(cond)), min_eig_exact=float(ev[..., 0].min()),
                    max_trace_defect_exact=float(np.abs(np.trace(X, axis1=2, axis2=3) - 1).max())),
        **pinn.run_metadata(config=cfg, driver=__file__, sources=[CFG])))
    print('stepper rows:', len(rows) - 1, 'max cond(V):', max(cond))


# ---------------------------------------------------------------------------------------------- training

def make_job(stage, map, seed, adam_iters=None, lbfgs_iters=None):
    T = cfg['train']
    j = dict(stage=stage, map=map, seed=seed, adam_iters=T['adam_iters'] if adam_iters is None else adam_iters,
             lbfgs_iters=T['lbfgs_iters'] if lbfgs_iters is None else lbfgs_iters)
    j['name'] = f'surrogate_{map}_s{seed}'; j['out'] = os.path.join(RAW, stage, j['name'] + '.json')
    return j


@functools.lru_cache(maxsize=None)
def pool():
    """Training pool: parameter points, their meshes, L1 weight matrices and generators."""
    T = cfg['train']; u = qmc.Sobol(d=2, scramble=True, seed=T['pool_seed']).random(T['pool'])
    al = A0 + (A1 - A0) * u[:, 0]; gd = G0 + (G1 - G0) * u[:, 1]
    tau = torch.tensor(np.stack([mesh(a, T['N']) for a in al]))
    W = torch.stack([op.l1_nonuniform_matrix(tau[k], float(al[k])) for k in range(len(al))])
    S = torch.tensor(np.stack([generator(g) for g in gd]))
    return torch.tensor(al), torch.tensor(gd), tau, W, S


def residual_loss(model, idx):
    """Mean squared Frobenius norm of the L1 residual over the nodes of the parameter points idx of the pool."""
    al, gd, tau, W, S = pool(); tb = tau[idx]; nb, n1 = tb.shape
    a = al[idx][:, None].expand(nb, n1); par = scale(a, gd[idx][:, None].expand(nb, n1))
    X = model(tb, alpha=a, par=par)                                        # (nb, N+1, 2, 2)
    dX = (X[:, 1:] - X[:, :-1]).reshape(nb, n1 - 1, 4)
    D = torch.complex(W[idx] @ dX.real, W[idx] @ dX.imag).reshape(nb, n1 - 1, 2, 2)
    R = D - pinn.apply_superop(S[idx][:, None], X[:, 1:])
    return (R.abs() ** 2).sum((-1, -2)).mean()


def evaluate(model, params, t):
    """States of the surrogate for all parameter points and times: (len(params), len(t), 2, 2)."""
    a = torch.tensor(params[:, 0])[:, None].expand(len(params), len(t)); g = torch.tensor(params[:, 1])[:, None].expand(len(params), len(t))
    with torch.no_grad():
        return model(torch.tensor(t)[None, :].expand(len(params), len(t)), alpha=a, par=scale(a, g)).numpy()


def run_job(j):
    T = cfg['train']; net = cfg['network']; seed = j['seed']
    torch.manual_seed(seed); np.random.seed(seed)
    model = pinn.DensityNet(torch.tensor(RHO0), map=j['map'], ic='hard', gate=cfg['gate'], alpha=0.5 * (A0 + A1), beta=net['beta'],
                            width=net['width'], depth=net['depth'], n_par=2)
    gen = torch.Generator().manual_seed(seed); npool = T['pool']; hist = []; status = 'ok'
    opt = torch.optim.Adam(model.parameters(), lr=T['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, j['adam_iters'], eta_min=T['lr'] * 0.05)
    t0 = time.perf_counter(); c0 = time.process_time()
    for it in range(j['adam_iters']):
        opt.zero_grad(); loss = residual_loss(model, torch.randperm(npool, generator=gen)[:T['batch']])
        if not torch.isfinite(loss):
            status = f'non-finite loss at Adam iteration {it}'; break
        loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0:
            hist.append((it, loss.item()))
    t_adam = time.perf_counter() - t0; allidx = torch.arange(npool); nev = [0]; t1 = time.perf_counter()
    if j['lbfgs_iters'] > 0 and status == 'ok':
        opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=j['lbfgs_iters'], history_size=50, tolerance_grad=1e-12,
                                 tolerance_change=1e-15, line_search_fn='strong_wolfe')

        def closure():
            opt2.zero_grad(); loss = residual_loss(model, allidx); loss.backward(); nev[0] += 1
            return loss
        opt2.step(closure)
    t_lbfgs = time.perf_counter() - t1; wall = t_adam + t_lbfgs; cpu = time.process_time() - c0
    with torch.no_grad():
        Jfin = residual_loss(model, allidx).item()
    # accuracy on the test parameter grid against the exact solution
    z = np.load(os.path.join(REF, 'exact.npz')); P, t, iu, il, Xex = z['params'], z['t'], z['is_uniform'], z['is_log'], z['X']
    assert np.array_equal(P, test_params()) and np.array_equal(t, test_times()[0]), 'stored reference does not match the configuration'
    X = evaluate(model, P, t); res = dict(map=j['map'], seed=seed, status=status, final_loss=Jfin, history=hist, t_adam=t_adam,
                                          t_lbfgs=t_lbfgs, lbfgs_evals=nev[0], wall=wall, cpu=cpu,
                                          n_params=sum(p.numel() for p in model.parameters()))
    if not np.isfinite(X).all():
        res['status'] = status if status != 'ok' else 'non-finite network output'
    else:
        ms = [pinn.state_metrics(X[k], Xex[k], t, iu, il, RHO0, cfg['eig_tol']) for k in range(len(P))]
        res['test_grid'] = {key: [m[key] for m in ms] for key in ('rel_fro', 'max_fro', 'min_eig', 'neg_time_frac', 'max_trace_defect',
                                                                  'max_herm_defect', 'rel_fro_proj')}
        e = np.array(res['test_grid']['rel_fro'])
        res.update(rel_fro_median=float(np.median(e)), rel_fro_mean=float(e.mean()), rel_fro_max=float(e.max()),
                   max_fro_max=float(max(res['test_grid']['max_fro'])))
        # admissibility over random parameter points
        rng = np.random.default_rng(cfg['test']['random_seed']); nr = cfg['test']['random_params']
        Pr = np.c_[rng.uniform(A0, A1, nr), rng.uniform(G0, G1, nr)]; Xr = torch.tensor(evaluate(model, Pr, t))
        adm = pinn.L.admissibility(Xr)
        res['random_params'] = dict(n=nr, min_eig=float(adm['min_eig'].min()),
                                    frac_params_with_negative_eig=float((adm['min_eig'].min(1).values < -cfg['eig_tol']).double().mean()),
                                    frac_points_with_negative_eig=float((adm['min_eig'] < -cfg['eig_tol']).double().mean()),
                                    max_negativity=float(adm['negativity'].max()), max_trace_defect=float(adm['trace_defect'].max()),
                                    max_herm_defect=float(adm['herm_defect'].max()))
        # cost of one query (one trajectory at the test times), batched and one at a time
        nb = cfg['test']['inference_batch']; Pb = Pr[:nb]; ts = []
        for _ in range(5):
            t0 = time.perf_counter(); evaluate(model, Pb, t); ts.append(time.perf_counter() - t0)
        t1s = []
        for k in range(50):
            t0 = time.perf_counter(); evaluate(model, Pb[k:k + 1], t); t1s.append(time.perf_counter() - t0)
        tf = []                                                            # one output time (tau = 1) per query
        for _ in range(5):
            t0 = time.perf_counter(); evaluate(model, Pb, t[-1:]); tf.append(time.perf_counter() - t0)
        res.update(inference_ms_per_query_batched=float(np.median(ts)) / nb * 1e3, inference_ms_per_query_single=float(np.median(t1s)) * 1e3,
                   inference_ms_per_query_final_time_batched=float(np.median(tf)) / nb * 1e3, n_test_times=len(t))
    conf = dict(job={k: v for k, v in j.items() if k != 'out'}, **{k: cfg[k] for k in ('phys', 'rho0_bloch', 'ranges', 'network', 'gate', 'test')},
                train=dict(T, adam_iters=j['adam_iters'], lbfgs_iters=j['lbfgs_iters']), config_file='configs/b2_surrogate.json')
    res.update(benchmark='surrogate', stage=j['stage'], name=j['name'], peak_rss_mb=pinn.peak_rss_mb())
    res.update(pinn.run_metadata(seed=seed, config=conf, driver=__file__, sources=[CFG]))
    os.makedirs(os.path.dirname(j['out']), exist_ok=True); pinn.write_json(j['out'], res)
    return f"{res['status']} J={Jfin:.3e} median rel_fro={res.get('rel_fro_median', float('nan')):.3e} wall={wall:.0f}s"


def main():
    cmd = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    if cmd == 'prepare':
        return prepare(w)
    if cmd == 'campaign':
        jobs = [make_job('main', m, s) for m in cfg['maps'] for s in cfg['seeds']]
        nbad = pinn.run_campaign(run_job, jobs, w, os.path.join(RAW, 'main', 'campaign.log'), 'surrogate')
        if nbad:
            sys.exit(f'surrogate: {nbad} of {len(jobs)} jobs FAILED')
        return print(f'surrogate: {len(jobs)} result files')
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--map', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--adam', type=int); ap.add_argument('--lbfgs', type=int); ap.add_argument('--stage', default='manual')
    a = ap.parse_args(); j = make_job(a.stage, a.map, a.seed, a.adam, a.lbfgs)
    print(j['name'], run_job(j)); print('written', j['out'])


if __name__ == '__main__':
    main()

"""B1 references: exact solution on the test grid, the implicit L1 time stepper in the metric of the networks
(discretisation floor of the forward problem) and the discrete optima of the inverse problem, which separate the
discretisation bias of an estimate of alpha from the optimisation error of the network.

Usage: python b1_floor.py
Writes results/raw/B1_design/reference/exact.npz, l1_floor.csv, discrete_optimum.csv, fisher.csv, prepare.json
"""
import json, os, sys, time
import numpy as np, torch
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn, reference as ref, mittag as ml
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
CFG = os.path.join(HERE, '..', 'configs', 'b1_scalar.json'); cfg = json.load(open(CFG))
OUT = os.path.join(ROOT, 'results', 'raw', 'B1_design', 'reference'); lam = cfg['lambda']
MESH_R = [cfg['mesh']['r']] + cfg['mesh_sensitivity']['r']


def grid():
    T = cfg['test']
    return pinn.test_grid(T['n_uniform'], T['log_min'], T['per_decade'])


def exact_u(t, alpha):
    return ml.ml_vec(-lam * np.asarray(t) ** alpha, alpha)


def observations(alpha, noise, seed):
    """Observation times (fixed by obs_seed), noisy data and exact values; the noise realisation depends on seed."""
    I = cfg['inverse']; rng = np.random.default_rng(I['obs_seed'])
    tob = np.sort(rng.uniform(0.02, 1.0, I['n_obs']))
    u = exact_u(tob, alpha)
    return tob, u + noise * np.random.default_rng(1000 + seed).standard_normal(u.shape), u


def stepper(t, alpha):
    """Implicit L1 solution of D^alpha u = -lam u, u(0) = 1, at the nodes of t."""
    return pinn.l1_stepper(np.array([[-lam]]), np.array([1.0]), t, alpha)[:, 0]


def interp_matrix(tq, t):
    """Matrix of the piecewise-linear interpolation from the nodes t to the times tq."""
    j = np.clip(np.searchsorted(t, tq, side='right') - 1, 0, len(t) - 2); s = (tq - t[j]) / (t[j + 1] - t[j])
    P = np.zeros((len(tq), len(t))); P[np.arange(len(tq)), j] = 1 - s; P[np.arange(len(tq)), j + 1] = s
    return P


def minimise(J):
    """Global minimum of a scalar function of alpha on the admitted interval: coarse scan, then bounded refinement."""
    lo, hi = cfg['discrete_optimum']['alpha_bounds']; g = np.linspace(lo, hi, 95); v = [J(a) for a in g]; k = int(np.argmin(v))
    r = minimize_scalar(J, bounds=(g[max(k - 1, 0)], g[min(k + 1, len(g) - 1)]), method='bounded', options=dict(xatol=1e-10))
    return float(r.x), float(r.fun)


def alpha_res(alpha_true, t):
    """argmin of the mean squared L1 residual of the exact solution on the mesh t."""
    u = exact_u(t, alpha_true)
    return minimise(lambda a: float(np.mean((pinn.l1_weight_matrix(t, a) @ np.diff(u) + lam * u[1:]) ** 2)))


def alpha_ls(tob, y, t):
    """Classical least squares through the L1 time stepper (piecewise-linear interpolation to the data)."""
    P = interp_matrix(tob, t)
    return minimise(lambda a: float(np.mean((P @ stepper(t, a) - y) ** 2)))


def alpha_joint(tob, y, t, weight):
    """argmin over alpha and all nodal values of mean(residual^2) + weight * mean(misfit^2), u(0) = 1."""
    N = len(t) - 1; P = interp_matrix(tob, t); D = np.diff(np.eye(N + 1), axis=0); sel = np.eye(N + 1)[1:]

    def J(a):
        M = pinn.l1_weight_matrix(t, a) @ D + lam * sel
        A = np.vstack([M[:, 1:] / np.sqrt(N), np.sqrt(weight / len(y)) * P[:, 1:]])
        b = -np.concatenate([M[:, 0] / np.sqrt(N), np.sqrt(weight / len(y)) * (P[:, 0] - y)])
        u = np.linalg.lstsq(A, b, rcond=None)[0]
        return float(np.sum((A @ u - b) ** 2))
    return minimise(J)


def main():
    os.makedirs(OUT, exist_ok=True); N = cfg['mesh']['N']; I = cfg['inverse']; t, iu, il = grid(); w = pinn.trapz_weights(t)
    ex = {a: exact_u(t, a) for a in cfg['alphas_true']}
    np.savez(os.path.join(OUT, 'exact.npz'), t=t, is_uniform=iu, is_log=il, **{f'u_a{a}': ex[a] for a in cfg['alphas_true']})
    floor = ['alpha,mesh_r,N,wall_ms,rel_l2,max_abs,rel_l2_u1000,max_nodal_err']
    opt = ['alpha_true,mesh_r,N,noise,seed,alpha_res,alpha_ls,alpha_joint,J_res_min,J_ls_min,J_joint_min,data_weight']
    fisher = ['alpha_true,dudalpha_norm2,sd_alpha_for_noise_0.01']
    for a in cfg['alphas_true']:
        e = 1e-4; tob, _, _ = observations(a, 0.0, 0)
        s2 = float((((exact_u(tob, a + e) - exact_u(tob, a - e)) / (2 * e)) ** 2).sum())
        fisher.append(f'{a},{s2:.6e},{0.01 / np.sqrt(s2):.6e}')
        for r in MESH_R:
            tn = ref.graded_mesh(1.0, N, r); ws = []
            for _ in range(20):
                t0 = time.perf_counter(); u = stepper(tn, a); ws.append(time.perf_counter() - t0)
            ui = np.interp(t, tn, u); d = ui - ex[a]
            floor.append(f'{a},{r:.6f},{N},{np.median(ws) * 1e3:.4f},{np.sqrt((w * d ** 2).sum() / (w * ex[a] ** 2).sum()):.6e},'
                         f'{np.abs(d).max():.6e},{np.linalg.norm(d[iu]) / np.linalg.norm(ex[a][iu]):.6e},{np.abs(u - exact_u(tn, a)).max():.6e}')
            ar, Jr = alpha_res(a, tn)
            for noise in I['noise_sd']:
                for seed in (cfg['seeds'] if noise > 0 else [0]):
                    if noise > 0 and r != cfg['mesh']['r']:
                        continue
                    tob, y, _ = observations(a, noise, seed)
                    al, Jl = alpha_ls(tob, y, tn); aj, Jj = alpha_joint(tob, y, tn, I['data_weight'])
                    opt.append(f'{a},{r:.6f},{N},{noise},{seed},{ar:.8f},{al:.8f},{aj:.8f},{Jr:.6e},{Jl:.6e},{Jj:.6e},{I["data_weight"]}')
                    print(opt[-1], flush=True)
            print(floor[-1], flush=True)
    for name, rows in (('l1_floor.csv', floor), ('discrete_optimum.csv', opt), ('fisher.csv', fisher)):
        open(os.path.join(OUT, name), 'w').write('\n'.join(rows) + '\n')
    pinn.write_json(os.path.join(OUT, 'prepare.json'), pinn.run_metadata(config=cfg, driver=__file__, sources=[CFG]))


if __name__ == '__main__':
    main()

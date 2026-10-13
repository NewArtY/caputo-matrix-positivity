"""E5: the implicit L1 step with a GKSL generator maps density matrices to density matrices.

The scheme for  D^alpha rho = L_t rho  on an arbitrary mesh reads
    rho^n = (I - L_{t_n}/w_nn)^{-1} sum_{m<n} theta_{n,m} rho^m ,   theta_{n,m} >= 0,  sum_m theta_{n,m} = 1,
(fracsp.reference.l1_convex_weights), and the resolvent (I - s L)^{-1}, s > 0, of a GKSL generator is completely
positive and trace preserving. This driver checks the statement numerically for random TIME-DEPENDENT generators
(d = 2, 3, 4; smooth, switched and instantaneous-eigenbasis generators; weak to strong driving), on uniform and
graded meshes, for the full-history and the SOE variants:
    * smallest eigenvalue, trace defect and Hermiticity defect along trajectories started from pure and mixed states;
    * smallest eigenvalue of the Choi matrix of the discrete solution map (complete positivity);
    * smallest convex weight and the deviation of their sum from one.
  python l1_positivity.py
Output: results/raw/E5_positivity/E5_l1_positivity.csv"""
import det_common as dc
import json, math, os, sys, time
import numpy as np

HERE = dc.HERE
OUT = os.path.join(dc.RAW, 'E5_positivity')
CFGP = os.path.join(HERE, '..', 'configs', 'verification.json')
cfg = json.load(open(CFGP))['E5_positivity']


def rand_herm(rng, d):
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (X + X.conj().T) / 2


def make_generator(rng, d, strength, kind):
    """Random time-dependent GKSL generator t -> superoperator matrix (d^2 x d^2)."""
    from fracsp import reference as ref
    H0, H1, H2 = rand_herm(rng, d), rand_herm(rng, d), rand_herm(rng, d)
    Ls0 = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(3)]
    w1, w2, w3 = rng.uniform(1, 25, 3)
    tsw = np.sort(rng.uniform(0, 1, 4))

    def gen(t):
        if kind == 'smooth':
            H = strength * (H0 + math.cos(w1 * t) * H1 + math.sin(w2 * t) * H2)
            Ls = [math.sqrt(abs(math.sin(w3 * t + k))) * L for k, L in enumerate(Ls0)]
        elif kind == 'switch':          # piecewise constant: damping phases alternate with strong rotations
            ph = int(np.searchsorted(tsw, t))
            if ph % 2 == 0:
                H = 0.1 * H0; Ls = [strength * 0.3 * L for L in Ls0[:1]]
            else:
                H = strength * 4 * H1; Ls = []
        elif kind == 'eigen':           # jump operators in the instantaneous eigenbasis of H(t), as in the Dirac model
            H = strength * (H0 + math.cos(w1 * t) * H1)
            w, V = np.linalg.eigh(H)
            Ls = [math.sqrt(0.5 * strength) * V[:, [0]] @ V[:, [-1]].conj().T,
                  math.sqrt(0.2 * strength) * (V[:, [-1]] @ V[:, [-1]].conj().T - V[:, [0]] @ V[:, [0]].conj().T)]
        else:
            raise ValueError(kind)
        return ref.gksl_superoperator(H, Ls)
    return gen


def _task(args):
    from fracsp import reference as ref, operators as op
    kind, d, alpha, strength, mesh, seed = args
    rng = np.random.default_rng(seed)
    gen = make_generator(rng, d, strength, kind)
    r = 1.0 if mesh == 'uniform' else ((2 - alpha) / alpha if alpha < 1 else 3.0)
    t = ref.graded_mesh(cfg['T'], cfg['N'], r)
    # initial states: pure (boundary of the state space) and full-rank mixed
    psi = rng.standard_normal((cfg['n_pure'], d)) + 1j * rng.standard_normal((cfg['n_pure'], d)); psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    rho0 = [np.outer(p, p.conj()) for p in psi]
    for _ in range(cfg['n_mixed']):
        X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); M = X @ X.conj().T; rho0.append(M / np.trace(M).real)
    rho0 = np.array(rho0)
    t1 = time.perf_counter()
    tt, R = ref.solve_l1_gksl(gen, rho0, t, alpha)
    ev, trd, hd = ref.density_defects(R)
    # complete positivity of the discrete solution map: propagate the matrix units E_ij
    Eij = np.eye(d * d, dtype=complex).reshape(d * d, d, d)
    tt, RE = ref.solve_l1_gksl(gen, Eij, t, alpha)
    choi_min = min(np.linalg.eigvalsh(0.5 * (C + C.conj().T)).min()
                   for C in (ref.choi_matrix(RE[k].reshape(d * d, d * d).T, d) for k in range(0, len(tt), 5)))
    # one-step resolvent (I - L/w_nn)^{-1} at a few steps
    choi_step = 0.0; th_min = np.inf; th_sum = 0.0
    for n in (1, 2, cfg['N'] // 2, cfg['N']):
        th, wnn = ref.l1_convex_weights(t, n, alpha)
        th_min = min(th_min, th.min()); th_sum = max(th_sum, abs(th.sum() - 1))
        P = np.linalg.inv(np.eye(d * d) - gen(t[n]) / wnn)
        C = ref.choi_matrix(P, d); choi_step = min(choi_step, np.linalg.eigvalsh(0.5 * (C + C.conj().T)).min())
    # SOE history (same mesh)
    if alpha < 1:
        om, la, ea, er = op.soe_kernel(alpha, ref.soe_delta_for_mesh(t), t[-1] - t[0], cfg['soe_tol'])
        tt, Rs = ref.solve_l1_gksl(gen, rho0, t, alpha, soe=(om, la))
        evs, trs, hds = ref.density_defects(Rs); dsoe = float(np.abs(Rs - R).max())
    else:
        evs, trs, dsoe = float('nan'), float('nan'), float('nan')
    return (f'{kind},{d},{alpha},{strength:g},{mesh},{r:.4f},{cfg["N"]},{len(rho0)},{ev:.3e},{trd:.3e},{hd:.3e},{choi_min:.3e},{choi_step:.3e},'
            f'{th_min:.3e},{th_sum:.1e},{evs:.3e},{trs:.3e},{dsoe:.3e},{time.perf_counter() - t1:.2f}')


if __name__ == '__main__':
    dc.disable_power_throttling(); os.makedirs(OUT, exist_ok=True)
    log = dc.Log(os.path.join(OUT, 'E5_l1_positivity.log'))
    ss = np.random.SeedSequence(cfg['seed'])
    combos = [(k, d, a, s, m) for k in cfg['kinds'] for d in cfg['dims'] for a in cfg['alphas'] for s in cfg['strengths'] for m in ('uniform', 'graded')]
    seeds = [int(x.generate_state(1)[0]) for x in ss.spawn(len(combos))]
    rows = dc.pool_map(_task, [c + (s,) for c, s in zip(combos, seeds)], log=log, label='E5')
    dc.write_csv(os.path.join(OUT, 'E5_l1_positivity.csv'),
                 'kind,d,alpha,strength,mesh,grading_r,N,n_states,min_eig,max_trace_defect,max_herm_defect,min_choi_eig_solution_map,min_choi_eig_one_step,'
                 'min_convex_weight,max_abs_weight_sum_minus_1,min_eig_soe,max_trace_defect_soe,max_abs_diff_soe_full,wall_s', rows,
                 dc.provenance(os.path.abspath(__file__), CFGP, dict(
                     note='implicit L1 scheme for D^alpha rho = L_t rho with random time-dependent GKSL generators; min_eig etc. are extrema over all time steps and '
                          'initial states (pure and mixed); min_choi_eig_solution_map: smallest eigenvalue of the Choi matrix of the discrete solution map at every '
                          'fifth step; min_choi_eig_one_step: Choi matrix of the resolvent (I - L/w_nn)^{-1} at four steps; *_soe: SOE history on the same mesh')))
    r = dc.read_csv(os.path.join(OUT, 'E5_l1_positivity.csv'))
    log(f'{len(r)} configurations: worst min_eig = {r.min_eig.min():.3e}; worst Choi (solution map) = {r.min_choi_eig_solution_map.min():.3e}; '
        f'worst Choi (one step) = {r.min_choi_eig_one_step.min():.3e}; worst min_eig SOE = {r.min_eig_soe.min():.3e}; smallest convex weight = {r.min_convex_weight.min():.3e}; '
        f'max trace defect = {r.max_trace_defect.max():.2e}')
    log('done'); log.close()

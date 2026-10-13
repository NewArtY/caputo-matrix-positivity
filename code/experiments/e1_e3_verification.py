"""E1-E3: deterministic verification of the discrete Caputo operators and of the implicit L1 reference solver.

  python e1_e3_verification.py [E1|E1T|E2|E3|all]

  E1   L1 operator applied to f = t^2 and f = t^alpha on uniform and graded meshes: maximum nodal error, final-node
       error and the weighted maximum max_n t_n^alpha |R^n|. For f = t^alpha the unweighted maximum does NOT converge
       on any mesh (the defect at the first node is |Gamma(1+alpha) - 1/Gamma(2-alpha)|).
  E1T  truncation error R^n = D_h u(t_n) - D^alpha u(t_n) for the solution u = E_alpha(-lambda t^alpha) of E3:
       max_n |R^n| (O(1)) and max_n t_n^alpha |R^n| (order min(2 - alpha, r alpha)), the quantity that a correct
       discrete stability bound uses.
  E2   alpha-derivative of sum_n D_h^alpha f_n: automatic differentiation, closed form and central finite differences
       against a 60-digit reference (mpmath), for the uniform operator AND for the nonuniform operator on graded
       meshes (the operator differentiated in the inverse problem).
  E3   implicit L1 solver for D^alpha u = -lambda u against the Mittag-Leffler solution: maximum and final error.
Observed orders are formed for ALL refinement pairs by scripts/aggregate_deterministic.py.
Outputs: results/raw/E1_E3/*.csv"""
import det_common as dc
import json, math, os, sys, time
import numpy as np
from scipy.special import gamma

HERE = dc.HERE
OUT = os.path.join(dc.RAW, 'E1_E3')
CFGP = os.path.join(HERE, '..', 'configs', 'verification.json')
cfg = json.load(open(CFGP))


def meta(extra=None):
    return dc.provenance(os.path.abspath(__file__), CFGP, extra)


def grading(mesh, a):
    return 1.0 if mesh == 'uniform' else (2 - a) / a


# ---------------------------------------------------------------- E1
def _e1_task(args):
    import torch
    from fracsp import operators as op, reference as ref
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    a, fname, mesh, N, T = args
    r = grading(mesh, a); t = ref.graded_mesh(T, N, r)
    if fname == 'smooth':
        f = t ** 2; ex = 2 * t[1:] ** (2 - a) / gamma(3 - a)
    else:
        f = t ** a; ex = np.full(N, gamma(1 + a))
    W = op.l1_nonuniform_matrix(torch.tensor(t), a)
    Dh = op.l1_nonuniform(torch.tensor(f), W).numpy()
    err = np.abs(Dh - ex)
    return f'{a},{fname},{mesh},{r:.6f},{N},{err.max():.6e},{int(err.argmax()) + 1},{err[-1]:.6e},{(t[1:] ** a * err).max():.6e}'


def exp_e1(log):
    c = cfg['E1_l1_operator']
    tasks = [(a, fn, mesh, N, c['T']) for a in c['alphas'] for fn in ('smooth', 'weak') for mesh in ('uniform', 'graded') for N in c['N_list']]
    rows = dc.pool_map(_e1_task, tasks)
    dc.write_csv(os.path.join(OUT, 'E1_l1_operator.csv'), 'alpha,function,mesh,grading_r,N,max_abs_err,argmax_node,final_abs_err,weighted_max_err', rows,
                 meta(dict(note='errors of D_h f against the exact Caputo derivative at the nodes n = 1..N; weighted_max_err = max_n t_n^alpha |error_n|; '
                                'function weak = t^alpha (first-node defect |Gamma(1+alpha) - 1/Gamma(2-alpha)| on every mesh)')))
    log('E1:', len(rows), 'rows')


# ---------------------------------------------------------------- E1T and E3 (share the Mittag-Leffler values)
def _e3_task(args):
    from fracsp import reference as ref, mittag as ml
    a, mesh, N, T, lam = args
    r = grading(mesh, a); t = ref.graded_mesh(T, N, r)
    ex = ml.ml_vec(-lam * t ** a, a)
    t0 = time.perf_counter(); tt, u = ref.solve_scalar_l1(lam, t, a); w = time.perf_counter() - t0
    err = np.abs(u - ex)
    # truncation error of the L1 formula applied to the exact solution: R^n = D_h u(t_n) - D^alpha u(t_n) = D_h u(t_n) + lam u(t_n)
    du = np.diff(ex)
    Rn = np.array([ref.l1_row_weights(t, n, a) @ du[:n] for n in range(1, N + 1)]) + lam * ex[1:]
    row3 = f'{a},{mesh},{r:.6f},{N},{err.max():.6e},{int(err.argmax())},{err[-1]:.6e},{w:.4f}'
    rowt = f'{a},{mesh},{r:.6f},{N},{np.abs(Rn).max():.6e},{int(np.abs(Rn).argmax()) + 1},{(t[1:] ** a * np.abs(Rn)).max():.6e},{abs(Rn[-1]):.6e}'
    traj = np.c_[t, u, ex] if (N == 256 and mesh == 'graded') else None
    return row3, rowt, traj


def exp_e3(log):
    c = cfg['E3_mittag_leffler']
    tasks = [(a, mesh, N, c['T'], c['lambda']) for a in c['alphas'] for mesh in ('uniform', 'graded') for N in c['N_list']]
    tasks_sorted = sorted(range(len(tasks)), key=lambda i: -tasks[i][2])
    res_s = dc.pool_map(_e3_task, [tasks[i] for i in tasks_sorted], log=log, label='E3')
    res = [None] * len(tasks)
    for i, r in zip(tasks_sorted, res_s):
        res[i] = r
    dc.write_csv(os.path.join(OUT, 'E3_mittag_leffler.csv'), 'alpha,mesh,grading_r,N,max_abs_err,argmax_node,final_abs_err,wall_s', [r[0] for r in res],
                 meta(dict(note='implicit L1 solver for D^alpha u = -lambda u, u(0) = 1, on [0, T] against E_alpha(-lambda t^alpha) (mpmath series); '
                                'max over all nodes (argmax_node: node index of the maximum) and final node')))
    dc.write_csv(os.path.join(OUT, 'E1T_truncation.csv'), 'alpha,mesh,grading_r,N,max_abs_trunc,argmax_node,weighted_max_trunc,final_abs_trunc', [r[1] for r in res],
                 meta(dict(note='truncation error R^n = D_h u(t_n) + lambda u(t_n) of the L1 formula for u = E_alpha(-lambda t^alpha) on [0, T]; '
                                'weighted_max_trunc = max_n t_n^alpha |R^n| (enters the discrete stability bound with the factor Gamma(1-alpha)); '
                                'max_abs_trunc does not tend to zero')))
    for t, r in zip(tasks, res):
        if r[2] is not None:
            dc.write_csv(os.path.join(OUT, f'E3_traj_alpha{t[0]}_graded_N256.csv'), 't,u_L1,u_exact', [f'{x[0]:.17e},{x[1]:.17e},{x[2]:.17e}' for x in r[2]], meta())
    log('E3/E1T:', len(res), 'rows')


# ---------------------------------------------------------------- E2
def _mp_sum_uniform(f, h, a_mp, mp):
    """S(alpha) = sum_n D_h f_n = h^-a/Gamma(2-a) sum_m df_m (N-m+1)^(1-a)  (telescoped), multiprecision."""
    N = len(f) - 1
    fm = [mp.mpf(float(x)) for x in f]
    s = mp.mpf(0)
    for m in range(1, N + 1):
        s += (fm[m] - fm[m - 1]) * mp.mpf(N - m + 1) ** (1 - a_mp)
    return mp.mpf(float(h)) ** (-a_mp) / mp.gamma(2 - a_mp) * s


def _mp_nodes_nonuniform(f, t, a_mp, mp):
    """D_h f_n, n = 1..N, of the nonuniform L1 formula in multiprecision (list)."""
    N = len(t) - 1
    tm = [mp.mpf(float(x)) for x in t]; fm = [mp.mpf(float(x)) for x in f]
    g = mp.gamma(2 - a_mp); out = []
    for n in range(1, N + 1):
        s = mp.mpf(0); prev = (tm[n] - tm[0]) ** (1 - a_mp)
        for m in range(1, n + 1):
            cur = (tm[n] - tm[m]) ** (1 - a_mp) if m < n else mp.mpf(0)
            s += (fm[m] - fm[m - 1]) / (tm[m] - tm[m - 1]) * (prev - cur)
            prev = cur
        out.append(s / g)
    return out


def _e2_task(args):
    import torch, mpmath as mp
    from fracsp import operators as op, reference as ref
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    kind, a0, N, rspec = args
    c = cfg['E2_alpha_gradient']; e = c['fd_step']; dps = c['nonuniform']['mp_dps']; em = c['nonuniform']['mp_step']
    if kind == 'uniform':
        h = 1.0 / N; t = torch.arange(N + 1) * h; f = t ** 1.5 + torch.sin(3 * t)
        a = torch.tensor(a0, requires_grad=True)
        op.l1_uniform(f, a, h).sum().backward(); ga = a.grad.item()
        gfd = ((op.l1_uniform(f, a0 + e, h).sum() - op.l1_uniform(f, a0 - e, h).sum()) / (2 * e)).item()
        df = np.diff(f.numpy()); b = op.l1_weights(a0, N).numpy(); db = op.dweights_dalpha(a0, N)
        conv_b = np.array([b[:n][::-1] @ df[:n] for n in range(1, N + 1)]); conv_db = np.array([db[:n][::-1] @ df[:n] for n in range(1, N + 1)])
        gan = (op.dcoef_dalpha(a0, h) * conv_b + h ** (-a0) / gamma(2 - a0) * conv_db).sum()
        with mp.workdps(dps):
            am = mp.mpf(a0); em_ = mp.mpf(em)
            gmp = float((_mp_sum_uniform(f.numpy(), h, am + em_, mp) - _mp_sum_uniform(f.numpy(), h, am - em_, mp)) / (2 * em_))
        node = float('nan'); r = 1.0; mesh = 'uniform'
    else:
        r = (2 - a0) / a0 if rspec == '(2-alpha)/alpha' else float(rspec); mesh = f'graded'
        tn = ref.graded_mesh(1.0, N, r); t = torch.tensor(tn); f = t ** 1.5 + torch.sin(3 * t)
        a = torch.tensor(a0, requires_grad=True)
        op.l1_nonuniform(f, op.l1_nonuniform_matrix(t, a)).sum().backward(); ga = a.grad.item()
        S = lambda x: op.l1_nonuniform(f, op.l1_nonuniform_matrix(t, x)).sum().item()
        gfd = (S(a0 + e) - S(a0 - e)) / (2 * e)
        dW = op.dl1_nonuniform_matrix_dalpha(tn, a0); gan = float((dW @ np.diff(f.numpy())).sum())
        # per-node derivative by automatic differentiation (Jacobian with respect to alpha)
        Jn = torch.autograd.functional.jacobian(lambda x: op.l1_nonuniform(f, op.l1_nonuniform_matrix(t, x)), torch.tensor(a0)).numpy()
        with mp.workdps(dps):
            am = mp.mpf(a0); em_ = mp.mpf(em)
            up = _mp_nodes_nonuniform(f.numpy(), tn, am + em_, mp); lo = _mp_nodes_nonuniform(f.numpy(), tn, am - em_, mp)
            dn = [(u - l) / (2 * em_) for u, l in zip(up, lo)]
            gmp = float(sum(dn)); dnf = np.array([float(x) for x in dn])
        node = float(np.max(np.abs(Jn - dnf) / np.maximum(np.abs(dnf), 1e-300)))
    rel = lambda x, y: abs(x - y) / abs(y)
    return (f'{mesh},{r:.6f},{a0},{N},{ga:.15e},{gan:.15e},{gfd:.15e},{gmp:.15e},{rel(ga, gmp):.3e},{rel(gan, gmp):.3e},{rel(gfd, gmp):.3e},'
            f'{rel(ga, gfd):.3e},{rel(gan, ga):.3e},{node:.3e}')


def exp_e2(log):
    c = cfg['E2_alpha_gradient']
    tasks = [('uniform', a, N, None) for a in c['alphas'] for N in c['N_list']]
    tasks += [('graded', a, N, rs) for a in c['alphas'] for rs in c['nonuniform']['gradings'] for N in c['nonuniform']['N_list']]
    rows = dc.pool_map(_e2_task, tasks, log=log, label='E2')
    dc.write_csv(os.path.join(OUT, 'E2_alpha_gradient.csv'),
                 'mesh,grading_r,alpha,N,grad_autograd,grad_closed_form,grad_fd,grad_mpmath,rel_err_autograd_vs_mpmath,rel_err_closed_form_vs_mpmath,'
                 'rel_err_fd_vs_mpmath,rel_err_autograd_vs_fd,rel_err_closed_form_vs_autograd,max_rel_err_nodes_autograd_vs_mpmath', rows,
                 meta(dict(note='d/dalpha of sum_n D_h^alpha f_n, f = t^1.5 + sin 3t on [0,1]. grad_mpmath: central difference (step 1e-20) of the discrete sum in 60-digit '
                                'arithmetic on the same double-precision nodal values. uniform: l1_uniform (closed form Eq. 4.27); graded: l1_nonuniform_matrix '
                                '(closed form dl1_nonuniform_matrix_dalpha). Last column (graded only): maximum over the '
                                'nodes of the relative error of the per-node derivative')))
    for r in rows:
        log('E2', r)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    dc.disable_power_throttling(); os.makedirs(OUT, exist_ok=True)
    log = dc.Log(os.path.join(OUT, f'E1_E3_{what}.log'))
    todo = dict(E1=[exp_e1], E1T=[exp_e3], E3=[exp_e3], E2=[exp_e2], all=[exp_e1, exp_e2, exp_e3])[what]
    for f in todo:
        f(log)
    log('done'); log.close()

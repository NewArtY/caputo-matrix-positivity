"""B5: the implicit L1 step of the FENE-P law, admissible solution against Newton's iteration.

The equation of one step of the implicit L1 scheme for the FENE-P law has several solutions. Exactly one of them
is admissible (positive definite with tr C < b), for every step size and every velocity gradient; it is the zero
of a scalar increasing function of the Peterlin factor (fracsp.spd.fenep_step_root). Newton's iteration started
from the previous iterate can converge to another solution. Two variants are run, both without a test of the
result (fracsp.spd.solve_l1, tolerance 1e-12 on the update, at most 60 iterations):

  full     full Newton steps (nonlinear='newton-full'), the textbook iteration;
  limited  a Newton step is halved as long as the new iterate would have tr C >= b (nonlinear='newton').

The step bound of the linear law, Gamma(2 - alpha) (h_n / lambda)^alpha (2 Wi - 1) < 1 in uniaxial extension, does
not delimit the failures: the full iteration also fails on meshes that satisfy it, because an iterate can pass
the pole tr C = b. This driver documents

  case     Wi = 20, alpha = 0.6, b = 50, uniform mesh with 256 steps on [0, 2 lambda] (step 2.9 times the bound):
           all real solutions of the first step, the tensors returned by Newton's iteration, the admissible
           iterates and their distance to a reference on graded meshes with 2048 and 4096 steps;
  case2    Wi = 20, alpha = 0.9, b = 10, uniform mesh with 512 steps (left-hand side of the step bound 0.25): the
           step at which the full iteration passes the pole, its result and the admissible solution of that step;
  sweep    uniaxial extension, Wi x alpha x b x (uniform, graded) x N: both Newton variants, root solver and
           safeguarded solver (Newton accepted only if admissible); smallest eigenvalue, largest tr C / b, the
           bound theta_* of the trace, residual of the scheme;
  general  shear, planar extension and random traceless velocity gradients: the same;
  stored   the references, stepper baselines and a-priori scales of b5_fenep.py, which were computed with the
           limited Newton iteration: comparison with the root solver and with the stored reference files.

Usage: python b5_fenep_step.py [--workers 6] [--quick]
Writes results/raw/B5_conformation/positivity/fenep_step.json and prints a summary. Deterministic; the result does
not depend on the number of workers. --quick: reduced sweep for the smoke test (writes fenep_step_quick.json).
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')          # one BLAS thread per process; must precede the first import of numpy

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
from fracsp import spd, pinn, reference as ref  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'raw', 'B5_conformation', 'positivity')
FENEP = os.path.join(ROOT, 'results', 'raw', 'B5_conformation', 'fenep')
CFG_FENEP = os.path.join(HERE, '..', 'configs', 'b5_fenep.json')
SWEEP = dict(Wi=[2.0, 5.0, 20.0, 100.0], alpha=[0.3, 0.6, 0.9], b=[10.0, 50.0, 200.0], mesh=['uniform', 'graded'],
             N=[4, 8, 16, 32, 64, 128, 256, 512, 1024], Theta=2.0)
QUICK = dict(Wi=[5.0, 20.0], alpha=[0.6], b=[50.0], mesh=['uniform', 'graded'], N=[8, 32], Theta=2.0)
GENERAL = dict(n_random=40, seed=20261012, alpha=[0.3, 0.6, 0.9], b=[20.0, 100.0], N=[8, 64], Theta=2.0,
               fixed=[['shear', 100.0], ['planar', 10.0]])
CASE = dict(Wi=20.0, alpha=0.6, b=50.0, Theta=2.0, N=256, N_ref=2048)
CASE2 = dict(Wi=20.0, alpha=0.9, b=10.0, Theta=2.0, N=512)
NEWTON = dict(tol=1e-12, maxit=60, start='previous iterate', stopping_rule='|update|_F <= tol (1 + |C|_F)',
              variants={'newton': 'a step is halved as long as the new iterate would have tr C >= b',
                        'newton_full': 'full steps'})


def mesh_of(name, N, alpha):
    return np.linspace(0.0, 1.0, N + 1) if name == 'uniform' else ref.graded_mesh(1.0, N, (2.0 - alpha) / alpha)


def admissibility(C, b):
    """Smallest eigenvalue and largest tr C / b over the iterates n >= 1."""
    lam = np.linalg.eigvalsh(C[1:])[:, 0]
    tr = np.trace(C[1:], axis1=1, axis2=2)
    return float(lam.min()), float(tr.max() / b)


def first_inadmissible(C, b):
    """Index of the first inadmissible iterate of C^0 ... and how it is inadmissible, or (None, None)."""
    if len(C) < 2:
        return None, None
    lam = np.linalg.eigvalsh(C[1:])[:, 0]
    tr = np.trace(C[1:], axis1=1, axis2=2)
    bad = np.where(~((lam > 0.0) & (tr < b)))[0]
    if not len(bad):
        return None, None
    k = int(bad[0])
    kind = 'negative eigenvalue and tr C >= b' if (not lam[k] > 0.0 and not tr[k] < b) else \
        ('tr C >= b' if not tr[k] < b else 'negative eigenvalue')
    return k + 1, kind


def newton_record(m, t, alpha, W, b, Cr, full, prefix):
    """One run of Newton's iteration without a test of the result (full=False: steps halved so that tr C < b)."""
    try:
        Cn = spd.solve_l1_newton(m, t, alpha, full=full, W=W)
        lam, trb = admissibility(Cn, b)
        ok = lam > 0.0 and trb < 1.0
        step, kind = first_inadmissible(Cn, b)
        return {prefix + '_status': 'admissible' if ok else 'inadmissible', prefix + '_min_eig': lam, prefix + '_max_tr_over_b': trb,
                prefix + '_max_residual': float(spd.step_residuals(m, t, alpha, Cn, W).max()),
                prefix + '_rel_diff_to_root': float(np.abs(Cn - Cr).max() / np.abs(Cr).max()) if ok else None,
                prefix + '_first_inadmissible_step': step, prefix + '_first_inadmissible_kind': kind, prefix + '_failing_step': None}
    except RuntimeError as e:
        step, kind = first_inadmissible(e.iterates, b)
        return {prefix + '_status': 'no convergence', prefix + '_min_eig': None, prefix + '_max_tr_over_b': None,
                prefix + '_max_residual': None, prefix + '_rel_diff_to_root': None, prefix + '_error': str(e),
                prefix + '_first_inadmissible_step': step, prefix + '_first_inadmissible_kind': kind, prefix + '_failing_step': e.step}


def run_one(args):
    """One mesh: root solver, both Newton variants, safeguarded solver. Returns a flat record."""
    Ghat, flow, Wi, alpha, b, meshname, N, Theta = args
    m = spd.Model('fenep', flow, Wi, Theta, b=b, Ghat=Ghat)
    t = mesh_of(meshname, N, alpha)
    W = spd.l1_weight_rows(t, alpha)
    lhs = spd.step_condition_lhs(m, t, alpha)
    row = dict(flow=flow, Wi=Wi, alpha=alpha, b=b, mesh=meshname, N=N, Theta=Theta, step_condition_lhs=lhs,
               violates_step_condition=bool(lhs >= 1.0), theta_star=spd.fenep_trace_bound(m))
    # root solver
    Cr = spd.solve_l1(m, t, alpha, W=W, nonlinear='root')
    lam, trb = admissibility(Cr, b)
    row.update(root_min_eig=lam, root_max_tr_over_b=trb, root_admissible=bool(lam > 0.0 and trb < 1.0),
               root_max_tr=trb * b, root_trace_bound_holds=bool(trb * b <= row['theta_star'] * (1.0 + 1e-12)),
               root_max_residual=float(spd.step_residuals(m, t, alpha, Cr, W).max()))
    # Newton's iteration from the previous iterate, steps halved so that tr C < b
    row.update(newton_record(m, t, alpha, W, b, Cr, False, 'newton'))
    # Newton's iteration from the previous iterate, full steps
    row.update(newton_record(m, t, alpha, W, b, Cr, True, 'newton_full'))
    # safeguarded solver (the default of fracsp.spd.solve_l1)
    Cs, info = spd.solve_l1(m, t, alpha, W=W, return_info=True)
    lam, trb = admissibility(Cs, b)
    row.update(safeguarded_min_eig=lam, safeguarded_max_tr_over_b=trb, safeguarded_admissible=bool(lam > 0.0 and trb < 1.0),
               safeguarded_root_steps=len(info['root_steps']),
               safeguarded_rel_diff_to_root=float(np.abs(Cs - Cr).max() / np.abs(Cr).max()))
    return row


def first_step_roots_uniaxial(Wi, alpha, b, Theta, h):
    """All real solutions of the first step from C^0 = I in uniaxial extension (C diagonal, C22 = C33):
        (w - 2 Wi + f) C11 = w + 1,   (w + Wi + f) C22 = w + 1,   f (b - C11 - 2 C22) = b - 3,
    in units of c / lambda, w = h^-alpha / (Gamma(2 - alpha) Theta^alpha): a cubic in f."""
    w = h ** (-alpha) / math.gamma(2.0 - alpha) / Theta ** alpha
    p1 = np.poly1d([1.0, w - 2.0 * Wi]); p2 = np.poly1d([1.0, w + Wi]); pf = np.poly1d([1.0, 0.0])
    cubic = pf * (b * p1 * p2 - (w + 1.0) * p2 - 2.0 * (w + 1.0) * p1) - (b - 3.0) * p1 * p2
    out = []
    for f in sorted(cubic.roots, key=lambda z: z.real):
        if abs(f.imag) > 1e-10 * (1.0 + abs(f)):
            continue
        f = float(f.real)
        for _ in range(3):                                     # Newton polish of the cubic root
            f -= cubic(f) / cubic.deriv()(f)
        c11 = (w + 1.0) / (w - 2.0 * Wi + f); c22 = (w + 1.0) / (w + Wi + f)
        out.append(dict(f=f, C11=c11, C22=c22, trace=c11 + 2.0 * c22,
                        admissible=bool(c11 > 0.0 and c22 > 0.0 and c11 + 2.0 * c22 < b),
                        defect=abs(f * (b - c11 - 2.0 * c22) - (b - 3.0)) / (b - 3.0)))
    return dict(w=w, f_c=2.0 * Wi - w, roots=out)


def case_study():
    c = CASE
    m = spd.Model('fenep', 'uniaxial', c['Wi'], c['Theta'], b=c['b'])
    al = c['alpha']
    t = mesh_of('uniform', c['N'], al)
    W = spd.l1_weight_rows(t, al)
    h_lam = c['Theta'] / c['N']
    h_bound = (math.gamma(2.0 - al) * (2.0 * c['Wi'] - 1.0)) ** (-1.0 / al)
    Cn, infn = spd.solve_l1(m, t, al, W=W, nonlinear='newton', return_info=True)      # raises if a step does not converge
    Cf, inff = spd.solve_l1(m, t, al, W=W, nonlinear='newton-full', return_info=True)
    Cr = spd.solve_l1(m, t, al, W=W, nonlinear='root')
    Cs, infs = spd.solve_l1(m, t, al, W=W, return_info=True)
    tq = spd.test_grid()
    refs = []
    for Nr in (c['N_ref'], 2 * c['N_ref']):
        tr_ = mesh_of('graded', Nr, al)
        refs.append(spd.interp_linear(tr_, spd.solve_l1(m, tr_, al, nonlinear='root'), tq))
    rich = refs[1] + (refs[1] - refs[0]) / (2.0 ** (2.0 - al) - 1.0)
    times = [1 / 128, 1 / 64, 1 / 16, 1 / 4, 1.0, 2.0]                    # t / lambda
    at = lambda tt, CC: [float(x) for x in spd.interp_linear(tt, CC, np.array(times) / c['Theta'])[:, 0, 0]]
    tref = mesh_of('graded', 2 * c['N_ref'], al)
    Cref = spd.solve_l1(m, tref, al, nonlinear='root')
    eF = lambda X: float(np.sqrt(((X - rich) ** 2).sum() / (rich ** 2).sum()))
    lam_n, trb_n = admissibility(Cn, c['b']); lam_r, trb_r = admissibility(Cr, c['b']); lam_f, trb_f = admissibility(Cf, c['b'])
    return dict(parameters=c, step_over_lambda=h_lam, step_bound_linear_law=h_bound, step_over_bound=h_lam / h_bound,
                step_condition_lhs=spd.step_condition_lhs(m, t, al), theta_star=spd.fenep_trace_bound(m),
                first_step=first_step_roots_uniaxial(c['Wi'], al, c['b'], c['Theta'], 1.0 / c['N']),
                newton=dict(converged_at_every_step=True, newton_iters=infn['newton_iters'], min_eig=lam_n, max_tr_over_b=trb_n,
                            C11_first_step=float(Cn[1, 0, 0]), C22_first_step=float(Cn[1, 1, 1]), C11_final=float(Cn[-1, 0, 0]),
                            steps_with_negative_eigenvalue=int((np.linalg.eigvalsh(Cn[1:])[:, 0] <= 0.0).sum()),
                            max_residual=float(spd.step_residuals(m, t, al, Cn, W).max()),
                            e_F=eF(spd.interp_linear(t, Cn, tq)), C11_at_times=at(t, Cn)),
                newton_full=dict(converged_at_every_step=True, newton_iters=inff['newton_iters'], min_eig=lam_f, max_tr_over_b=trb_f,
                                 C11_first_step=float(Cf[1, 0, 0]), C22_first_step=float(Cf[1, 1, 1]), C11_final=float(Cf[-1, 0, 0]),
                                 steps_with_negative_eigenvalue=int((np.linalg.eigvalsh(Cf[1:])[:, 0] <= 0.0).sum()),
                                 max_residual=float(spd.step_residuals(m, t, al, Cf, W).max()),
                                 e_F=eF(spd.interp_linear(t, Cf, tq)),
                                 max_abs_diff_to_limited_newton=float(np.abs(Cf - Cn).max())),
                root=dict(min_eig=lam_r, max_tr_over_b=trb_r, max_tr=trb_r * c['b'], C11_first_step=float(Cr[1, 0, 0]),
                          C22_first_step=float(Cr[1, 1, 1]), C11_final=float(Cr[-1, 0, 0]),
                          max_residual=float(spd.step_residuals(m, t, al, Cr, W).max()),
                          e_F=eF(spd.interp_linear(t, Cr, tq)), C11_at_times=at(t, Cr)),
                safeguarded=dict(root_steps=infs['root_steps'], rel_diff_to_root=float(np.abs(Cs - Cr).max() / np.abs(Cr).max())),
                reference=dict(meshes=[c['N_ref'], 2 * c['N_ref']], rel_diff_N_2N=float(np.sqrt(((refs[1] - refs[0]) ** 2).sum() / (refs[1] ** 2).sum())),
                               min_eig=float(np.linalg.eigvalsh(Cref)[:, 0].min()), max_tr=float(np.trace(Cref, axis1=1, axis2=2).max()),
                               C11_at_times=at(tref, Cref)),
                times_over_lambda=times)


def case_study_pole():
    """A mesh that satisfies the step bound of the linear law on which the full Newton iteration passes the pole."""
    c = CASE2
    m = spd.Model('fenep', 'uniaxial', c['Wi'], c['Theta'], b=c['b'])
    al = c['alpha']
    t = mesh_of('uniform', c['N'], al)
    W = spd.l1_weight_rows(t, al)
    Cr = spd.solve_l1(m, t, al, W=W, nonlinear='root')
    Cl = spd.solve_l1(m, t, al, W=W, nonlinear='newton')
    out = dict(parameters=c, step_condition_lhs=spd.step_condition_lhs(m, t, al), theta_star=spd.fenep_trace_bound(m),
               newton=dict(admissible=bool(first_inadmissible(Cl, c['b'])[0] is None),
                           rel_diff_to_root=float(np.abs(Cl - Cr).max() / np.abs(Cr).max())))
    lam_r, trb_r = admissibility(Cr, c['b'])
    out['root'] = dict(min_eig=lam_r, max_tr=trb_r * c['b'])
    try:
        Cf = spd.solve_l1(m, t, al, W=W, nonlinear='newton-full')
        failing = None
    except RuntimeError as e:
        Cf, failing = e.iterates, e.step
    k, kind = first_inadmissible(Cf, c['b'])
    trf = np.trace(Cf[1:], axis1=1, axis2=2); lamf = np.linalg.eigvalsh(Cf[1:])[:, 0]
    full = dict(converged_at_every_step=failing is None, failing_step=failing, first_inadmissible_step=k, first_inadmissible_kind=kind,
                inadmissible_steps=int((~((lamf > 0.0) & (trf < c['b']))).sum()), steps_computed=int(len(Cf) - 1),
                max_residual=float(spd.step_residuals(m, t[:len(Cf)], al, Cf, W[:len(Cf) - 1, :len(Cf) - 1]).max()))
    if k is not None:
        full.update(previous_iterate=dict(C11=float(Cf[k - 1, 0, 0]), C22=float(Cf[k - 1, 1, 1]), trace=float(np.trace(Cf[k - 1]))),
                    iterate=dict(C11=float(Cf[k, 0, 0]), C22=float(Cf[k, 1, 1]), trace=float(np.trace(Cf[k]))),
                    admissible_solution_of_that_step=dict(C11=float(Cr[k, 0, 0]), C22=float(Cr[k, 1, 1]), trace=float(np.trace(Cr[k]))),
                    max_rel_diff_to_root_before=float(np.abs(Cf[:k] - Cr[:k]).max() / np.abs(Cr[:k]).max()))
    out['newton_full'] = full
    return out


def stored_check():
    """References, stepper baselines and a-priori scales of b5_fenep.py: Newton's iteration against the root solver."""
    cfg = json.load(open(CFG_FENEP))
    tq = spd.test_grid(cfg['test']['n_uniform'], cfg['test']['n_layer'], tuple(cfg['test']['layer']))
    rows = []
    for Wi in cfg['Wi']:
        for al in cfg['alphas']:
            m = spd.Model('fenep', cfg['flow'], Wi, cfg['Theta'], b=cfg['b'])
            row = dict(Wi=Wi, alpha=al, b=cfg['b'], meshes={})
            sols = []
            for N in (cfg['scale']['N'], cfg['mesh']['N'], cfg['reference']['N'], 2 * cfg['reference']['N']):
                t = mesh_of('graded', N, al)
                W = spd.l1_weight_rows(t, al)
                Cn = spd.solve_l1_newton(m, t, al, W=W)
                Cr = spd.solve_l1(m, t, al, W=W, nonlinear='root')
                Cs, info = spd.solve_l1(m, t, al, W=W, return_info=True)
                lam, trb = admissibility(Cn, cfg['b'])
                row['meshes'][str(N)] = dict(step_condition_lhs=spd.step_condition_lhs(m, t, al), newton_min_eig=lam,
                                             newton_max_tr_over_b=trb, newton_admissible=bool(lam > 0.0 and trb < 1.0),
                                             newton_rel_diff_to_root=float(np.abs(Cn - Cr).max() / np.abs(Cr).max()),
                                             safeguarded_identical_to_newton=bool(np.array_equal(Cs, Cn)),
                                             safeguarded_root_steps=len(info['root_steps']))
                if N >= cfg['reference']['N']:
                    sols.append(spd.interp_linear(t, Cr, tq))
            rich = sols[1] + (sols[1] - sols[0]) / (2.0 ** (2.0 - al) - 1.0)
            fn = os.path.join(FENEP, f'reference_fenep_uniaxial_Wi{Wi:g}_a{al:g}.npz')
            if os.path.exists(fn):
                stored = np.load(fn)['C']
                row['stored_reference_rel_diff_to_root'] = float(np.abs(stored - rich).max() / np.abs(rich).max())
                row['stored_reference_min_eig'] = float(np.linalg.eigvalsh(stored)[:, 0].min())
                row['stored_reference_max_tr_over_b'] = float(np.trace(stored, axis1=1, axis2=2).max() / cfg['b'])
            rows.append(row)
    return rows


def newton_summary(rr, p):
    """Counts of one Newton variant (prefix p) over the runs rr."""
    viol = [r for r in rr if r['violates_step_condition']]
    sat = [r for r in rr if not r['violates_step_condition']]
    bad = [r for r in rr if r[p + '_status'] != 'admissible']
    okn = [r for r in rr if r[p + '_status'] == 'admissible']
    return {p + '_inadmissible': sum(r[p + '_status'] == 'inadmissible' for r in rr),
            p + '_no_convergence': sum(r[p + '_status'] == 'no convergence' for r in rr),
            p + '_failures_on_meshes_violating_step_condition': sum(r[p + '_status'] != 'admissible' for r in viol),
            p + '_failures_on_meshes_satisfying_step_condition': sum(r[p + '_status'] != 'admissible' for r in sat),
            p + '_inadmissible_on_meshes_satisfying_step_condition': sum(r[p + '_status'] == 'inadmissible' for r in sat),
            p + '_no_convergence_on_meshes_satisfying_step_condition': sum(r[p + '_status'] == 'no convergence' for r in sat),
            p + '_admissible_on_meshes_violating_step_condition': sum(r[p + '_status'] == 'admissible' for r in viol),
            p + '_no_convergence_after_an_inadmissible_iterate':
                sum(r[p + '_status'] == 'no convergence' and r[p + '_first_inadmissible_step'] is not None for r in rr),
            p + '_first_inadmissible_iterate_has_tr_ge_b':
                sum(r[p + '_first_inadmissible_kind'] is not None and 'tr C >= b' in r[p + '_first_inadmissible_kind'] for r in bad),
            p + '_smallest_eigenvalue': min([r[p + '_min_eig'] for r in rr if r[p + '_min_eig'] is not None], default=None),
            p + '_largest_tr_over_b': max([r[p + '_max_tr_over_b'] for r in rr if r[p + '_max_tr_over_b'] is not None], default=None),
            p + '_max_rel_diff_to_root_when_admissible': max([r[p + '_rel_diff_to_root'] for r in okn], default=None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sw = QUICK if a.quick else SWEEP
    gen = dict(GENERAL, n_random=4, N=[8]) if a.quick else GENERAL
    t0 = time.perf_counter()
    jobs = [(None, 'uniaxial', Wi, al, b, mn, N, sw['Theta']) for Wi in sw['Wi'] for al in sw['alpha'] for b in sw['b']
            for mn in sw['mesh'] for N in sw['N']]
    rng = np.random.default_rng(gen['seed'])
    gjobs = []
    for k in range(gen['n_random']):
        G = rng.standard_normal((3, 3))
        G -= np.trace(G) / 3.0 * np.eye(3)
        G /= np.linalg.norm(G, 2)                                         # unit spectral norm; Wi is the rate
        Wi = float(10.0 ** rng.uniform(0.0, 1.7))
        for al in gen['alpha']:
            for b in gen['b']:
                for N in gen['N']:
                    gjobs.append((G, f'random{k}', Wi, al, b, 'uniform', N, gen['Theta']))
    for flow, Wi in gen['fixed']:
        for al in gen['alpha']:
            for b in gen['b']:
                for N in gen['N']:
                    gjobs.append((spd.ghat(flow), flow, Wi, al, b, 'uniform', N, gen['Theta']))
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.workers) as pool:
            rows = pool.map(run_one, jobs, chunksize=4)
            grows = pool.map(run_one, gjobs, chunksize=4)
    else:
        rows = [run_one(j) for j in jobs]
        grows = [run_one(j) for j in gjobs]

    def summary(rr):
        viol = [r for r in rr if r['violates_step_condition']]
        out = dict(runs=len(rr), runs_violating_step_condition=len(viol))
        out.update(newton_summary(rr, 'newton'))
        out.update(newton_summary(rr, 'newton_full'))
        out.update(root_admissible=sum(r['root_admissible'] for r in rr), root_smallest_eigenvalue=min(r['root_min_eig'] for r in rr),
                   root_largest_tr_over_b=max(r['root_max_tr_over_b'] for r in rr),
                   root_trace_bound_holds=sum(r['root_trace_bound_holds'] for r in rr),
                   root_max_residual=max(r['root_max_residual'] for r in rr),
                   safeguarded_admissible=sum(r['safeguarded_admissible'] for r in rr),
                   safeguarded_runs_using_root_solver=sum(r['safeguarded_root_steps'] > 0 for r in rr),
                   safeguarded_max_rel_diff_to_root=max(r['safeguarded_rel_diff_to_root'] for r in rr))
        return out
    for r, j in zip(grows, gjobs):                                         # velocity gradient of unit rate, row by row
        r['Ghat'] = np.asarray(j[0]).reshape(-1).tolist()
    res = dict(newton_iteration=NEWTON,
               sweep=dict(design=sw, summary=summary(rows), runs=rows),
               general=dict(design=gen, summary=summary(grows), runs=grows),
               provenance=pinn.run_metadata(seed=gen['seed'], config=dict(sweep=sw, general=gen, case=CASE, case2=CASE2, newton=NEWTON),
                                            driver=__file__, sources=[CFG_FENEP]))
    if not a.quick:
        res['case'] = case_study()
        res['case2'] = case_study_pole()
        res['stored'] = stored_check()
    fn = os.path.join(OUT, 'fenep_step_quick.json' if a.quick else 'fenep_step.json')
    pinn.write_json(fn, res)
    print(f"{len(rows)} + {len(grows)} runs in {time.perf_counter() - t0:.0f} s; wrote {os.path.relpath(fn, ROOT).replace(os.sep, '/')}")
    for name in ('sweep', 'general'):
        s = res[name]['summary']
        print(f"{name}: {s['runs']} runs, {s['runs_violating_step_condition']} with steps above the bound of the linear law")
        for p, label in (('newton_full', 'Newton, full steps'), ('newton', 'Newton, steps halved so that tr C < b')):
            print(f"  {label}: inadmissible {s[p + '_inadmissible']}, no convergence {s[p + '_no_convergence']}; failures on meshes "
                  f"above the bound {s[p + '_failures_on_meshes_violating_step_condition']}, within the bound "
                  f"{s[p + '_failures_on_meshes_satisfying_step_condition']}; admissible although above the bound: "
                  f"{s[p + '_admissible_on_meshes_violating_step_condition']}; first inadmissible iterate with tr C >= b in "
                  f"{s[p + '_first_inadmissible_iterate_has_tr_ge_b']} runs")
        print(f"  root solver: admissible {s['root_admissible']}/{s['runs']}, smallest eigenvalue {s['root_smallest_eigenvalue']:.4g}, "
              f"largest tr C / b {s['root_largest_tr_over_b']:.6f}, tr C <= theta_* in {s['root_trace_bound_holds']}, "
              f"largest residual {s['root_max_residual']:.1e}")
        print(f"  safeguarded: admissible {s['safeguarded_admissible']}/{s['runs']}, root solver used in "
              f"{s['safeguarded_runs_using_root_solver']} runs, largest difference to the root solver {s['safeguarded_max_rel_diff_to_root']:.1e}")
    if not a.quick:
        cs = res['case']
        print(f"case: step / bound = {cs['step_over_bound']:.2f}; first step, real solutions (C11, C22, tr C, admissible):")
        for r in cs['first_step']['roots']:
            print(f"    f = {r['f']:.4f}: {r['C11']:.4f}, {r['C22']:.5f}, {r['trace']:.4f}, {r['admissible']}")
        print(f"  Newton (steps halved): C11 = {cs['newton']['C11_first_step']:.4f} at the first step, smallest eigenvalue "
              f"{cs['newton']['min_eig']:.4f}, e_F = {cs['newton']['e_F']:.3f}; full steps: "
              f"{cs['newton_full']['steps_with_negative_eigenvalue']} iterates with a negative eigenvalue, largest difference to the "
              f"halved variant {cs['newton_full']['max_abs_diff_to_limited_newton']:.1e}; admissible iterates: smallest eigenvalue "
              f"{cs['root']['min_eig']:.4f}, largest trace {cs['root']['max_tr']:.4f} (theta_* = {cs['theta_star']:.4f}), "
              f"e_F = {cs['root']['e_F']:.2e}")
        c2 = res['case2']; f2 = c2['newton_full']
        print(f"case2: left-hand side of the step bound {c2['step_condition_lhs']:.4f}; Newton with full steps: first inadmissible "
              f"iterate at step {f2['first_inadmissible_step']} ({f2['first_inadmissible_kind']}), C11 = {f2['iterate']['C11']:.5f}, "
              f"C22 = {f2['iterate']['C22']:.5f}, tr C = {f2['iterate']['trace']:.5f}; admissible solution of that step: "
              f"{f2['admissible_solution_of_that_step']['C11']:.5f}, {f2['admissible_solution_of_that_step']['C22']:.5f}, "
              f"tr C = {f2['admissible_solution_of_that_step']['trace']:.5f}; {f2['inadmissible_steps']} of {f2['steps_computed']} "
              f"iterates inadmissible; Newton with halved steps admissible: {c2['newton']['admissible']}")
        for r in res['stored']:
            worst = max(v['newton_rel_diff_to_root'] for v in r['meshes'].values())
            print(f"stored Wi={r['Wi']:g} alpha={r['alpha']}: Newton admissible on all meshes "
                  f"{all(v['newton_admissible'] for v in r['meshes'].values())}, largest difference to the root solver {worst:.1e}, "
                  f"stored reference against root solver {r.get('stored_reference_rel_diff_to_root', float('nan')):.1e}")


if __name__ == '__main__':
    main()

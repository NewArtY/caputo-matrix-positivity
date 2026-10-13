"""B5: positivity of the exact solution and of L1 time stepping for the Caputo conformation-tensor equation.

Checks, by direct computation, the statements of the positivity analysis:
  (P1) exact solution: C(t) > 0 and det C(t) >= 1 (subordination of the congruence semigroup), and a Monte
       Carlo average of the classical solution over the random operational time (subordination formula);
  (P2) implicit L1: the step operator (w_nn - Theta^alpha L)^-1 is a positive map iff
       w_nn > 2 Theta^alpha s(K),  K = Wi Ghat - I/2  (s = spectral abscissa):
       unconditional in shear (s(K) = -1/2), step restriction
       Gamma(2-alpha) (h_n Theta)^alpha (2 Wi - 1) < 1 in uniaxial extension with Wi > 1/2;
  (P3) explicit L1 loses positive definiteness in the first step of start-up of shear unless
       Gamma(2-alpha) (h_1 Theta)^alpha Wi < 1.

Nonlinear laws (Giesekus, FENE-P): the equation of a step has several solutions, exactly one of which is admissible,
for every step size. Each mesh is solved twice: by Newton's iteration started from the previous iterate, for
FENE-P with a step halved as long as the new iterate would have tr C >= b and without a test of the result
(fracsp.spd.solve_l1_newton; keys ok, min_eig, max_trace, newton_iters, error), which can return an inadmissible
solution or fail to converge, and by the safeguarded solver (fracsp.spd.solve_l1; keys safeguarded_*), which
returns the admissible solution of every step. Newton's iteration with full steps is studied in b5_fenep_step.py.

Usage: python b5_positivity.py
Writes results/raw/B5_conformation/positivity/positivity.json and subordination_monte_carlo.json and prints a table.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
from fracsp import spd, pinn, reference as ref  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'raw', 'B5_conformation', 'positivity')
MC_SEED, MC_SAMPLES = 12345, 2_000_000


def min_eig(C):
    return float(np.linalg.eigvalsh(C)[:, 0].min())


def subordination_monte_carlo(Wi=3.0, z=1.3):
    """C_alpha(t) = E[C_1(S_t)] with the operational time S_t = z L^(-alpha), z = (t/lambda)^alpha, L a standard
    positive alpha-stable variable (Kanter's representation): Monte Carlo mean and standard error of C12 and of
    N1 = C11 - C22 in start-up of shear against the exact Mittag-Leffler solution."""
    out = []
    for alpha in (0.6, 0.9):
        rng = np.random.default_rng(MC_SEED); n = MC_SAMPLES
        U = rng.uniform(0.0, math.pi, n)
        E = rng.exponential(1.0, n)
        Lst = np.sin(alpha * U) / np.sin(U) ** (1 / alpha) * (np.sin((1 - alpha) * U) / E) ** ((1 - alpha) / alpha)
        s = z * Lst ** (-alpha)
        c12 = Wi * (1 - np.exp(-s))
        n1 = 2 * Wi ** 2 * (1 - np.exp(-s) - s * np.exp(-s))
        C = spd.exact_ucm('shear', Wi, 1.0, alpha, np.array([z ** (1 / alpha)]))[0]
        out.append(dict(alpha=alpha, Wi=Wi, z=z, n=n, C12_mc=float(c12.mean()), C12_se=float(c12.std() / math.sqrt(n)),
                        C12_exact=float(C[0, 1]), N1_mc=float(n1.mean()), N1_se=float(n1.std() / math.sqrt(n)),
                        N1_exact=float(C[0, 0] - C[1, 1])))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    prov = pinn.run_metadata(seed=MC_SEED, config=dict(monte_carlo_samples=MC_SAMPLES), driver=__file__)
    rows = []
    cases = [('shear', 10.0, 4.0), ('shear', 100.0, 4.0), ('uniaxial', 1.5, 2.0), ('uniaxial', 3.0, 2.0)]
    Ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for flow, Wi, Th in cases:
        for alpha in (0.6, 0.9):
            m = spd.Model('ucm', flow, Wi, Th)
            g2 = math.gamma(2 - alpha)
            if flow == 'uniaxial':
                h_imp = (g2 * Th ** alpha * (2 * Wi - 1)) ** (-1 / alpha) if Wi > 0.5 else float('inf')
            else:
                h_imp = float('inf')
            h_exp = (g2 * Th ** alpha * Wi) ** (-1 / alpha)        # first-step threshold of explicit L1 (shear)
            for meshname in ('uniform', 'graded'):
                for N in Ns:
                    t = np.linspace(0, 1, N + 1) if meshname == 'uniform' else ref.graded_mesh(1.0, N, (2 - alpha) / alpha)
                    hmax = float(np.diff(t).max())
                    h1 = float(t[1] - t[0])
                    Ci = spd.solve_l1(m, t, alpha)
                    with np.errstate(over='ignore', invalid='ignore'):
                        Ce = spd.solve_l1(m, t, alpha, explicit=True)
                    ok_e = bool(np.isfinite(Ce).all())
                    rows.append(dict(flow=flow, Wi=Wi, Theta=Th, alpha=alpha, mesh=meshname, N=N, h_max=hmax, h_1=h1,
                                     h_crit_implicit=h_imp, h1_crit_explicit_shear=h_exp,
                                     implicit_min_eig=min_eig(Ci), implicit_spd=bool(min_eig(Ci) > 0),
                                     implicit_predicted_spd=bool(hmax < h_imp),
                                     explicit_min_eig=min_eig(Ce) if ok_e else None,
                                     explicit_spd=bool(ok_e and min_eig(Ce) > 0)))
    # exact solution: SPD and det >= 1 on a dense grid, for all forward cases
    tq = spd.test_grid(400, 100)
    exact = []
    for flow, Wi, Th in [('shear', 1.0, 4.0), ('shear', 10.0, 4.0), ('shear', 100.0, 4.0),
                         ('uniaxial', 0.3, 2.0), ('uniaxial', 1.0, 2.0), ('uniaxial', 1.5, 2.0)]:
        for alpha in (0.6, 0.9):
            C = spd.exact_ucm(flow, Wi, Th, alpha, tq)
            exact.append(dict(flow=flow, Wi=Wi, Theta=Th, alpha=alpha, min_eig=min_eig(C),
                              min_det=float(np.linalg.det(C).min()),
                              min_eig_C_minus_I=float(np.linalg.eigvalsh(C - np.eye(3))[:, 0].min())))
    # nonlinear laws: positive definiteness and, for FENE-P, tr C < b; Newton's iteration without a test and safeguarded solver
    nonlin = []
    for law, flow, Wi, Th, kw in [('giesekus', 'shear', 10.0, 4.0, dict(aG=0.2)), ('giesekus', 'shear', 100.0, 4.0, dict(aG=0.2)),
                                  ('giesekus', 'shear', 100.0, 4.0, dict(aG=0.5)), ('giesekus', 'uniaxial', 5.0, 2.0, dict(aG=0.2)),
                                  ('fenep', 'uniaxial', 2.0, 2.0, dict(b=50.0)), ('fenep', 'uniaxial', 5.0, 2.0, dict(b=50.0)),
                                  ('fenep', 'uniaxial', 20.0, 2.0, dict(b=50.0)), ('fenep', 'shear', 100.0, 4.0, dict(b=50.0))]:
        for alpha in (0.6, 0.9):
            m = spd.Model(law, flow, Wi, Th, **kw)
            for meshname in ('uniform', 'graded'):
                for N in (16, 64, 256, 1024):
                    t = np.linspace(0, 1, N + 1) if meshname == 'uniform' else ref.graded_mesh(1.0, N, (2 - alpha) / alpha)
                    try:
                        C, info = spd.solve_l1_newton(m, t, alpha, return_info=True)
                        row = dict(ok=True, min_eig=min_eig(C), max_trace=float(np.trace(C, axis1=1, axis2=2).max()),
                                   newton_iters=info['newton_iters'])
                    except Exception as e:
                        row = dict(ok=False, error=repr(e))
                    Cs, infs = spd.solve_l1(m, t, alpha, return_info=True)
                    row.update(step_condition_lhs=spd.step_condition_lhs(m, t, alpha), safeguarded_min_eig=min_eig(Cs),
                               safeguarded_max_trace=float(np.trace(Cs, axis1=1, axis2=2).max()),
                               safeguarded_root_steps=len(infs['root_steps']),
                               safeguarded_max_residual=float(spd.step_residuals(m, t, alpha, Cs).max()))
                    nonlin.append(dict(law=law, flow=flow, Wi=Wi, Theta=Th, alpha=alpha, mesh=meshname, N=N, **kw, **row))
    json.dump(dict(stepper=rows, exact=exact, nonlinear=nonlin, provenance=prov), open(os.path.join(OUT, 'positivity.json'), 'w'), indent=1)
    mc = subordination_monte_carlo()
    json.dump(mc, open(os.path.join(OUT, 'subordination_monte_carlo.json'), 'w'), indent=1)
    json.dump(prov, open(os.path.join(OUT, 'subordination_monte_carlo.provenance.json'), 'w'), indent=1)
    nfail = sum(not r['ok'] for r in nonlin)
    nbad = sum(r['ok'] and (r['min_eig'] <= 0 or (r['law'] == 'fenep' and r['max_trace'] >= r['b'])) for r in nonlin)
    nsafe = sum(r['safeguarded_min_eig'] <= 0 or (r['law'] == 'fenep' and r['safeguarded_max_trace'] >= r['b']) for r in nonlin)
    print(f'nonlinear laws, Newton iteration without a test: {len(nonlin)} runs, {nfail} Newton failures, {nbad} inadmissible solutions '
          f'(on meshes within the step condition of the linear law: '
          f"{sum((not r['ok'] or r['min_eig'] <= 0) and r['step_condition_lhs'] < 1 for r in nonlin)})")
    print(f"nonlinear laws, safeguarded solver: {nsafe} inadmissible solutions, root solver used in "
          f"{sum(r['safeguarded_root_steps'] > 0 for r in nonlin)} runs, smallest eigenvalue "
          f"{min(r['safeguarded_min_eig'] for r in nonlin):.4g}")
    for r in nonlin:
        if r['N'] in (16, 1024) and r['mesh'] == 'graded':
            print({k: (float('%.4g' % v) if isinstance(v, float) else v) for k, v in r.items()})
    agree = sum(r['implicit_spd'] == r['implicit_predicted_spd'] for r in rows)
    print(f'implicit L1: prediction "SPD iff h_max < h_crit" agrees with computation in {agree}/{len(rows)} cases')
    print('flow Wi alpha mesh | N: implicit SPD? / explicit SPD?   (h_crit implicit, h1_crit explicit)')
    for flow, Wi, Th in cases:
        for alpha in (0.6, 0.9):
            for meshname in ('uniform', 'graded'):
                rr = [r for r in rows if (r['flow'], r['Wi'], r['alpha'], r['mesh']) == (flow, Wi, alpha, meshname)]
                s = ' '.join(f"{r['N']}:{'Y' if r['implicit_spd'] else 'n'}/{'Y' if r['explicit_spd'] else 'n'}" for r in rr)
                print(f"{flow} {Wi:g} {alpha} {meshname:8s} | {s}   ({rr[0]['h_crit_implicit']:.3g}, {rr[0]['h1_crit_explicit_shear']:.3g})")
    print('exact solution: min eig, min det, min eig(C - I)')
    for e in exact:
        print(f"{e['flow']} Wi={e['Wi']:g} a={e['alpha']}: {e['min_eig']:.4f} {e['min_det']:.6f} {e['min_eig_C_minus_I']:.4f}")
    print('subordination, Monte Carlo against exact (C12; N1):')
    for m in mc:
        print(f"alpha={m['alpha']}: {m['C12_mc']:.5f} +- {m['C12_se']:.5f} vs {m['C12_exact']:.5f}; "
              f"{m['N1_mc']:.4f} +- {m['N1_se']:.4f} vs {m['N1_exact']:.4f}")


if __name__ == '__main__':
    main()

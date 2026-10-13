"""B6 references (d = 3 and d = 4, pure initial states): exact trajectories from the matrix Mittag-Leffler function,
their independent checks, and the implicit L1 time stepper in the metric of the networks.

Usage: python b6_prepare.py [workers]
Writes results/raw/B6_positivity/reference/exact_<system>_a<alpha>.npz, exact_eigs_<system>_a<alpha>.csv (spectrum of the
exact state along the trajectory), l1_stepper.csv and prepare.json (checks: conditioning of the eigenbasis, Richardson
extrapolation of the L1 scheme, closed form for alpha = 1/2, admissibility of the reference).
"""
import json, math, os, sys
import numpy as np, torch
from scipy.special import wofz
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn, reference as ref
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
CFG = os.path.join(HERE, '..', 'configs', 'b6_positivity.json'); cfg = json.load(open(CFG))
OUT = os.path.join(ROOT, 'results', 'raw', 'B6_positivity', 'reference')


def unit(d, i, j):
    E = np.zeros((d, d), dtype=complex); E[i, j] = 1
    return E


def system(name):
    """Hamiltonian, jump operators (rates absorbed) and pure initial state of a B6 system."""
    p = cfg['systems'][name]; d = p['d']
    if name == 'qutrit':
        H = 0.5 * p['O1'] * (unit(3, 0, 1) + unit(3, 1, 0)) + 0.5 * p['O2'] * (unit(3, 1, 2) + unit(3, 2, 1)) + p['D2'] * unit(3, 2, 2)
        Ls = [math.sqrt(p['g10']) * unit(3, 0, 1), math.sqrt(p['g21']) * unit(3, 1, 2),
              math.sqrt(p['gphi']) * (unit(3, 1, 1) + 2 * unit(3, 2, 2))]
    elif name == 'two_qubit':
        sm = unit(2, 1, 0); sp = sm.T; sz = np.diag([1.0, -1.0]).astype(complex); I = np.eye(2)
        H = p['J'] * (np.kron(sp, sm) + np.kron(sm, sp)) + 0.5 * p['delta'] * (np.kron(sz, I) - np.kron(I, sz))
        Ls = [math.sqrt(p['g1']) * np.kron(sm, I), math.sqrt(p['g2']) * np.kron(I, sm), math.sqrt(p['gphi'] / 2) * np.kron(sz, I)]
    else:
        raise ValueError(name)
    return H, Ls, unit(d, p['initial_level'], p['initial_level'])


def problem(name, alpha):
    H, Ls, rho0 = system(name); S = pinn.gksl_superop(H, Ls)
    tcol = ref.graded_mesh(1.0, cfg['mesh']['N'], (2 - alpha) / alpha)
    T = cfg['test']; t, iu, il = pinn.test_grid(T['n_uniform'], T['log_min'], T['per_decade'])
    return S, rho0, tcol, t, iu, il


def ref_path(name, alpha):
    return os.path.join(OUT, f'exact_{name}_a{alpha}.npz')


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(OUT, exist_ok=True); checks = {}
    cases = [(n, a) for n in cfg['systems'] for a in cfg['alphas']]
    jobs = {c: pinn.reference_jobs(ref_path(*c), problem(*c)[0], problem(*c)[2], problem(*c)[3], c[1], 2 * workers) for c in cases}
    if pinn.run_campaign(pinn.ml_table_job, sum(jobs.values(), []), workers, os.path.join(OUT, 'prepare.log'), 'B6 prepare'):
        sys.exit('B6 prepare: tabulation of the Mittag-Leffler function FAILED')
    lines = [','.join(('system', 'alpha') + pinn.STEPPER_COLUMNS)]
    for name, a in cases:
        S, rho0, tcol, t, iu, il = problem(name, a); d = rho0.shape[0]; x0 = rho0.reshape(-1)
        ck = pinn.assemble_reference(ref_path(name, a), S, tcol, t, iu, il, a, jobs[(name, a)])
        R = pinn.load_reference(ref_path(name, a)); Xt, Xn = pinn.reference_states(R, S, rho0)
        ev = np.linalg.eigvalsh(Xt)
        ck.update(min_eig_exact=float(ev[:, 0].min()), max_smallest_eig_exact=float(ev[:, 0].max()),
                  max_second_eig_exact=float(ev[:, 1].max()),
                  max_trace_defect_exact=float(np.abs(np.trace(Xt, axis1=1, axis2=2) - 1).max()),
                  max_herm_defect_exact=float(np.abs(Xt - Xt.conj().transpose(0, 2, 1)).max()),
                  max_abs_eig_generator=float(np.abs(R['lam']).max()), norm_L_rho0=float(np.linalg.norm(S @ x0)))
        np.savetxt(os.path.join(OUT, f'exact_eigs_{name}_a{a}.csv'), np.c_[t, ev], delimiter=',', comments='',
                   header='tau,' + ','.join(f'eig{k}' for k in range(d)))
        # independent check 1: Richardson extrapolation of the L1 scheme on nested graded meshes (order 2 - alpha)
        N1, N2 = cfg['stepper']['richardson_N']; r = (2 - a) / a; sub = np.arange(0, N1 + 1, N1 // 16)[1:]
        X1 = pinn.l1_stepper(S, x0, ref.graded_mesh(1.0, N1, r), a)[sub]
        X2 = pinn.l1_stepper(S, x0, ref.graded_mesh(1.0, N2, r), a)[2 * sub]
        ts = ref.graded_mesh(1.0, N1, r)[sub]; Xe, _ = pinn.ml_exact(S, x0, ts, a, (R['lam'], R['V']))
        XR = X2 + (X2 - X1) / (2 ** (2 - a) - 1)
        e1, e2, eR = (float(np.abs(X - Xe).max()) for X in (X1, X2, XR))
        ck.update(richardson=dict(N=[N1, N2], max_err_N1=e1, max_err_N2=e2, max_err_extrapolated=eR,
                                  observed_order=math.log2(e1 / e2), n_times=len(sub)))
        # independent check 2: E_{1/2}(z) = w(-i z) (Faddeeva function)
        if a == 0.5:
            Xw = (wofz(-1j * np.sqrt(ts)[:, None] * R['lam'][None, :]) * np.linalg.solve(R['V'], x0)[None, :]) @ R['V'].T
            ck['max_diff_to_faddeeva'] = float(np.abs(Xe - Xw).max())
        checks[f'{name}_a{a}'] = ck
        print(name, a, json.dumps(ck), flush=True)
        for row in pinn.stepper_rows(S, rho0, a, R, cfg['stepper']['N'], cfg['stepper']['timing_repeats'], cfg['eig_tol']):
            lines.append(f'{name},{a},' + ','.join(f'{row[k]:.6e}' if isinstance(row[k], float) else str(row[k])
                                                   for k in pinn.STEPPER_COLUMNS))
            print(lines[-1], flush=True)
    open(os.path.join(OUT, 'l1_stepper.csv'), 'w').write('\n'.join(lines) + '\n')
    pinn.write_json(os.path.join(OUT, 'prepare.json'), dict(checks=checks, **pinn.run_metadata(config=cfg, driver=__file__, sources=[CFG])))


if __name__ == '__main__':
    main()

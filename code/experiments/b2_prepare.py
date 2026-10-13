"""B2 references: exact trajectories (matrix Mittag-Leffler function) on the test grid and at the collocation nodes
for every alpha of configs/b2_density.json, and the implicit L1 time stepper evaluated in the metric of the networks.

Usage: python b2_prepare.py [workers]
Writes results/raw/B2_factorial/reference/exact_a<alpha>.npz  (test grid, collocation mesh, eigen-decomposition of
       the generator, tables E_alpha(lam_k tau^alpha))
       results/raw/B2_factorial/reference/l1_stepper.csv      (classical reference rows)
       results/raw/B2_factorial/reference/prepare.json        (checks and provenance)
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn, reference as ref, exact
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
CFG = os.path.join(HERE, '..', 'configs', 'b2_density.json'); cfg = json.load(open(CFG))
OUT = os.path.join(ROOT, 'results', 'raw', 'B2_factorial', 'reference')


def problem(alpha):
    """Generator, collocation mesh and test grid of B2 for one alpha."""
    H, Ls = pinn.two_level_ops(**cfg['phys']); S = pinn.gksl_superop(H, Ls)
    tcol = ref.graded_mesh(1.0, cfg['mesh']['N'], (2 - alpha) / alpha)
    T = cfg['test']; t, iu, il = pinn.test_grid(T['n_uniform'], T['log_min'], T['per_decade'])
    return S, tcol, t, iu, il


def rho0_of(state):
    return pinn.bloch_to_rho(torch.tensor(cfg['initial_states_bloch'][state])).numpy()


def ref_path(alpha):
    return os.path.join(OUT, f'exact_a{alpha}.npz')


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(OUT, exist_ok=True); checks = {}
    # 1. tables of the scalar Mittag-Leffler function, in chunks of times (the series is expensive for small alpha)
    jobs = {a: pinn.reference_jobs(ref_path(a), problem(a)[0], problem(a)[1], problem(a)[2], a, (4 if a < 0.5 else 1) * workers)
            for a in cfg['alphas']}
    if pinn.run_campaign(pinn.ml_table_job, sum(jobs.values(), []), workers, os.path.join(OUT, 'prepare.log'), 'B2 prepare'):
        sys.exit('B2 prepare: tabulation of the Mittag-Leffler function FAILED')
    ph = cfg['phys']; G2 = ph['Gdown'] / 2 + ph['Gphi']
    A = np.array([[-G2, 0, 0], [0, -G2, -ph['Omega']], [0, ph['Omega'], -ph['Gdown']]]); b = np.array([0, 0, -ph['Gdown']])
    for a in cfg['alphas']:
        S, tcol, t, iu, il = problem(a)
        checks[f'a{a}'] = pinn.assemble_reference(ref_path(a), S, tcol, t, iu, il, a, jobs[a])
        # checks: the independent Bloch-vector formula fracsp.exact.bloch_exact, and admissibility of the reference
        R = pinn.load_reference(ref_path(a)); ts = np.array([1e-9, 1e-4, 3e-2, 0.4, 1.0]); worst = 0.0
        for st, r0 in cfg['initial_states_bloch'].items():
            Xs, _ = pinn.ml_exact(S, rho0_of(st).reshape(-1), ts, a, (R['lam'], R['V']))
            rb = exact.bloch_exact(A, b, np.array(r0), ts, a)
            worst = max(worst, float(np.abs(pinn.rho_to_bloch(torch.tensor(Xs.reshape(-1, 2, 2))).numpy() - rb).max()))
            Xt, _ = pinn.reference_states(R, S, rho0_of(st))
            checks[f'a{a}_{st}'] = dict(min_eig_exact=float(np.linalg.eigvalsh(Xt)[:, 0].min()),
                                        max_trace_defect_exact=float(np.abs(np.trace(Xt, axis1=1, axis2=2) - 1).max()),
                                        max_herm_defect_exact=float(np.abs(Xt - Xt.conj().transpose(0, 2, 1)).max()))
        checks[f'a{a}'].update(max_diff_to_bloch_exact=worst, mesh_r=(2 - a) / a, tau_1=float(tcol[1]))
        print(f'alpha={a}:', checks[f'a{a}'], flush=True)
    # 2. implicit L1 time stepper in the metric of the networks
    lines = [','.join(('alpha', 'state') + pinn.STEPPER_COLUMNS)]
    for a in cfg['alphas']:
        S = problem(a)[0]; R = pinn.load_reference(ref_path(a))
        for st in cfg['initial_states_bloch']:
            for row in pinn.stepper_rows(S, rho0_of(st), a, R, cfg['stepper']['N'], cfg['stepper']['timing_repeats'], cfg['eig_tol']):
                lines.append(f'{a},{st},' + ','.join(f'{row[k]:.6e}' if isinstance(row[k], float) else str(row[k])
                                                     for k in pinn.STEPPER_COLUMNS))
                print(lines[-1], flush=True)
    open(os.path.join(OUT, 'l1_stepper.csv'), 'w').write('\n'.join(lines) + '\n')
    pinn.write_json(os.path.join(OUT, 'prepare.json'), dict(checks=checks, **pinn.run_metadata(config=cfg, driver=__file__, sources=[CFG])))


if __name__ == '__main__':
    main()

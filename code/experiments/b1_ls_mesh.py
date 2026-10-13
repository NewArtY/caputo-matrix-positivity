"""B1, identification of the order: least squares through the implicit L1 time stepper on finer meshes.

The networks of B1 and the least-squares estimate of b1_floor.py use the mesh tau_n = (n/N)^r with N = 128, r = 3.
The estimate on that mesh carries the discretisation bias of the scheme. This driver repeats the least-squares fit of
b1_floor.py (same data, same objective, same minimisation) on the meshes with N, 4 N and 16 N steps and the same
grading, so that this bias can be told apart from the error of the estimator.

  noise-free data: all three meshes;
  noisy data (the realisations of the campaign, seeds of the configuration): N and 4 N.

Usage: python b1_ls_mesh.py
Writes results/raw/B1_design/reference/ls_mesh.csv and its provenance record ls_mesh.provenance.json. The rows with
the coarsest mesh repeat the column alpha_ls of discrete_optimum.csv.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn, reference as ref
import b1_floor as F

FACTORS = (1, 4, 16)             # numbers of steps relative to the mesh of the networks
NOISY_UP_TO = 4                  # noisy data are fitted on the meshes up to this factor


def ls_rows(cfg=None, factors=FACTORS, noisy_up_to=NOISY_UP_TO):
    """One row per order, mesh, noise level and noise realisation: (alpha_true, mesh_r, N, noise, seed, alpha_ls, J_ls_min)."""
    cfg = F.cfg if cfg is None else cfg
    N0, r, rows = cfg['mesh']['N'], cfg['mesh']['r'], []
    for a in cfg['alphas_true']:
        for k in factors:
            t = ref.graded_mesh(1.0, k * N0, r)
            for noise in cfg['inverse']['noise_sd']:
                if noise > 0 and k > noisy_up_to:
                    continue
                for seed in (cfg['seeds'] if noise > 0 else [0]):
                    tob, y, _ = F.observations(a, noise, seed)
                    al, J = F.alpha_ls(tob, y, t)
                    rows.append((a, r, k * N0, noise, seed, al, J))
    return rows


def main():
    os.makedirs(F.OUT, exist_ok=True)
    out = ['alpha_true,mesh_r,N,noise,seed,alpha_ls,J_ls_min']
    for a, r, N, noise, seed, al, J in ls_rows():
        out.append(f'{a},{r:.6f},{N},{noise},{seed},{al:.8f},{J:.6e}')
        print(out[-1], flush=True)
    open(os.path.join(F.OUT, 'ls_mesh.csv'), 'w').write('\n'.join(out) + '\n')
    pinn.write_json(os.path.join(F.OUT, 'ls_mesh.provenance.json'),
                    pinn.run_metadata(config=dict(F.cfg, ls_mesh=dict(factors=list(FACTORS), noisy_up_to=NOISY_UP_TO)), driver=__file__,
                                      sources=[F.CFG, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'b1_floor.py')]))


if __name__ == '__main__':
    main()

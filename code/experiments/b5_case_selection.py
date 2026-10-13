"""B5: accuracy of the implicit L1 stepper for candidate start-up cases on meshes with 128, 512 and 2048 steps
(deterministic; used to choose the cases of the forward benchmark).

Usage: python b5_case_selection.py
Writes results/raw/B5_conformation/positivity/case_selection.json and prints a table.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
from fracsp import spd, pinn, reference as ref  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'raw', 'B5_conformation', 'positivity')

tq = spd.test_grid(200, 50)
rows = []
print('flow Wi Theta alpha | max|C|_F max cond | e_F of implicit L1 for N = 128, 512, 2048')
for flow, Wi, Th in [('uniaxial', 1.0, 2.0), ('uniaxial', 1.5, 2.0), ('uniaxial', 2.0, 2.0), ('uniaxial', 3.0, 1.0),
                     ('shear', 10.0, 4.0), ('shear', 30.0, 4.0), ('shear', 100.0, 4.0)]:
    for alpha in (0.6, 0.9):
        ex = spd.exact_ucm(flow, Wi, Th, alpha, tq)
        m = spd.Model('ucm', flow, Wi, Th)
        errs = {}
        for N in (128, 512, 2048):
            t = ref.graded_mesh(1.0, N, (2 - alpha) / alpha)
            errs[N] = spd.metrics(spd.interp_linear(t, spd.solve_l1(m, t, alpha), tq), ex)['rel_fro']
        w = np.linalg.eigvalsh(ex)
        row = dict(flow=flow, Wi=Wi, Theta=Th, alpha=alpha, max_norm=float(np.sqrt((ex ** 2).sum((-1, -2))).max()),
                   max_cond=float((w[:, -1] / w[:, 0]).max()), e_F={str(k): v for k, v in errs.items()})
        rows.append(row)
        print(flow, Wi, Th, alpha, '| %.3g %.3g |' % (row['max_norm'], row['max_cond']), ' '.join('%.2e' % e for e in errs.values()), flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(rows, open(os.path.join(OUT, 'case_selection.json'), 'w'), indent=1)
json.dump(pinn.run_metadata(driver=__file__), open(os.path.join(OUT, 'case_selection.provenance.json'), 'w'), indent=1)

"""Coefficient of the term 1/kappa_c of the band-referenced sheet current of the laser-driven Dirac model.

The current within the square |kappa_x|, |kappa_y| <= kappa_c converges like J(t; kappa_c) = J_inf(t) + B(t)/kappa_c + ...
The coefficient B(t) follows from the terms of order kappa^-3 of the large-momentum expansion
(fracsp.dirac_cutoff.current_cutoff_coefficient); for alpha = 1 it is (sqrt(2) g_s/(4 pi^2)) (a'' + G2 a'). This driver
evaluates it for the cases of the benchmark B4. The comparison with the coefficient fitted to the cutoff sweep of B4
is made by scripts/aggregate_deterministic.py (Table_Dirac_B4_fits.csv).

  python dirac_cutoff_coefficient.py
Output: results/raw/DIRAC_v2/current_cutoff_coefficient.csv (zeta, alpha, t, B_x, B_y on a uniform time grid; the
change of B under halving of the step of the quadrature in time is recorded in the header line 'note')."""
import det_common as dc
import math, os, time
import numpy as np
import dirac_reference as D
from fracsp import dirac, dirac_cutoff

OUT = D.OUT
cfg = D.cfg
STEP = 0.025                      # step of the time grid of the convolutions (the finer of the two grids is STEP / 2)


def main():
    t0, tf = cfg['window']; c4 = cfg['B4']
    t_out = np.linspace(t0, tf, c4['n_output'] + 1)
    rows = []; notes = []; t_start = time.perf_counter()
    for zeta in c4['zetas']:
        for alpha in c4['alphas']:
            p = D.pars(None, zeta)
            t, B, parts = dirac_cutoff.current_cutoff_coefficient(alpha, p, cfg['taubar_c'], (t0, tf), STEP)
            t2, B2, parts2 = dirac_cutoff.current_cutoff_coefficient(alpha, p, cfg['taubar_c'], (t0, tf), STEP / 2)
            conv = float(np.abs(B2[::2] - B).max() / np.abs(B2).max())
            line = f'zeta={zeta:g} alpha={alpha:g}: max|B|={np.abs(B2).max():.6f}, relative change under halving of the step {conv:.1e}'
            if parts2:
                cref = dirac.tail_constant(alpha, p, math.pi / 2, cfg['taubar_c'], (t0, tf))
                line += f", C_alpha(t_f; 90 deg)={parts2['C_xx'][-1]:.7f} (fracsp.dirac.tail_constant: {cref:.7f})"
            print(line, flush=True); notes.append(line)
            bx = np.interp(t_out, t2, B2[:, 0]); by = np.interp(t_out, t2, B2[:, 1])
            for ti, x, y in zip(t_out, bx, by):
                rows.append(f'{zeta},{alpha},{ti:.10f},{x:.12e},{y:.12e}')
    dc.write_csv(os.path.join(OUT, 'current_cutoff_coefficient.csv'), 'zeta,alpha,t,B_x,B_y', rows,
                 D.meta(dict(note='coefficient B(t) of the term 1/kappa_c of the band-referenced sheet current within the square |kappa_x|, |kappa_y| <= kappa_c, '
                                  f'both valleys, g_s = 2 (closed form, fracsp.dirac_cutoff); time step {STEP / 2:g}, interpolated linearly to the listed times. '
                                  + '; '.join(notes))))
    print(f'wrote current_cutoff_coefficient.csv ({len(rows)} rows) in {time.perf_counter() - t_start:.0f} s')


if __name__ == '__main__':
    main()

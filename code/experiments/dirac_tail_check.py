"""High-momentum tail of the final conduction population n_c(kappa, t_f): numerical values (two independent time
discretisations, Richardson extrapolated) against the analytical asymptotic constants

    alpha < 1:  n_c = C_alpha(phi) / |kappa|^2 [1 + O(|kappa|^-2)],   C_alpha from fracsp.dirac.tail_constant
    alpha = 1:  n_c = D(phi) / |kappa|^4 [1 + ...],                    D from fracsp.dirac.tail_constant_alpha1

  python dirac_tail_check.py [rays|dissipation|tf|alpha1|constants|all]

  rays         kappa^2 n_c along rays for circular and linear polarisation, both valleys (anisotropy: the leading
               term vanishes along the polarisation axis of a linearly polarised pulse, where n_c ~ kappa^-4)
  dissipation  the tail with the dissipator switched off (and with dephasing / relaxation removed separately)
  tf           dependence on the final time t_f (same step size)
  alpha1       alpha = 1: kappa^4 n_c (kappa^6 n_c along the polarisation axis), DOP853 in deviation form
  constants    C_alpha versus alpha, angular averages and the asymptotic slope dN_tot/d ln kappa_c = 2 <C>/pi
Outputs: results/raw/DIRAC_v2/tail_*.csv"""
import det_common as dc
import json, math, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp
import dirac_reference as D
from fracsp import dirac

OUT = D.OUT
cfg = D.cfg; ct = cfg['tail']


def ray_points(phi_deg, Ks):
    ph = math.radians(phi_deg); Ks = np.asarray(Ks, dtype=float)
    c, s = math.cos(ph), math.sin(ph)
    c = 0.0 if abs(c) < 1e-15 else c; s = 0.0 if abs(s) < 1e-15 else s
    return np.c_[Ks * c, Ks * s]


def nc_final(kap, xi, p, alpha, N, mode, window=None, n_output=1):
    res = D.simulate(kap, xi, p, alpha, N, mode, n_output, window=window)
    tf = res['t'][-1]
    nc = dirac.observables(res['R'][-1:], kap, xi, [tf], p)[0][0]
    return nc, res


def nc_alpha1(kap, xi, p, rtol, window=None):
    """alpha = 1 by DOP853 in the deviation form s = r + n (no cancellation in 1 + n.r); each wave vector is
    integrated separately. The absolute tolerance is tied to the size of the deviation, |s| ~ a0/(2 K^2): a
    tolerance below the rounding noise of the largest component (about 1e-16 |s|) can never be met by the
    components that are small by symmetry, and the step-size control then stalls."""
    t0, tf = window if window is not None else cfg['window']
    out = np.empty(len(kap))
    for i in range(len(kap)):
        k1 = kap[i:i + 1]
        scale = p['a0'] / (2.0 * max(1.0, float(np.hypot(k1[0, 0], k1[0, 1]))) ** 2)
        sol = solve_ivp(lambda t, y: dirac.deviation_rhs(k1, xi, t, y[None, :], p)[0], (t0, tf), np.zeros(3), method='DOP853', rtol=rtol, atol=1e-13 * scale)
        if not sol.success:
            raise RuntimeError(sol.message)
        out[i] = 0.5 * float(dirac.axis(k1, xi, tf, p)[1][0] @ sol.y[:, -1])
    return out


def _ray_task(a):
    p = D.pars(None, a['zeta'], **a.get('upd', {}))
    kap = ray_points(a['phi_deg'], a['Ks'])
    t1 = time.perf_counter()
    if a['mode'] == 'ode':
        nc = nc_alpha1(kap, a['xi'], p, a['rtol']); info = {}
    else:
        nc, res = nc_final(kap, a['xi'], p, a['alpha'], a['N'], a['mode'])
        info = res['info']; info['max_bloch_norm'] = math.sqrt(res['r2max'])
    return dict(nc=nc, wall=time.perf_counter() - t1, info=info)


def exp_rays(log):
    tasks = []
    for ray in ct['rays']:
        for xi in (1, -1):
            for alpha in ct['alphas']:
                for N in ct['N_pi']:
                    tasks.append(dict(zeta=ray['zeta'], phi_deg=ray['phi_deg'], xi=xi, alpha=alpha, mode='pi', N=N, Ks=ct['K_list']))
                for N in ct['N_l1']:
                    tasks.append(dict(zeta=ray['zeta'], phi_deg=ray['phi_deg'], xi=xi, alpha=alpha, mode='soe', N=N, Ks=ct['K_list']))
    tasks.sort(key=lambda a: -(a['N'] ** (2 if a['mode'] == 'pi' else 1) / (4 if a['mode'] == 'pi' else 1)))
    res = dc.pool_map(_ray_task, tasks, log=log, label='rays')
    rows = []
    for a, r in zip(tasks, res):
        for K, v in zip(a['Ks'], r['nc']):
            rows.append(f"{a['zeta']},{a['phi_deg']},{a['xi']},{a['alpha']},{'PI-trapezoid' if a['mode'] == 'pi' else 'L1-soe'},{a['N']},{K},{v:.12e},{K * K * v:.10e},"
                        f"{r['info'].get('max_bloch_norm', float('nan')):.15f},{r['info'].get('soe_rel_err', 0.0):.3e}")
    rows.sort()
    dc.write_csv(os.path.join(OUT, 'tail_rays.csv'), 'zeta,phi_deg,xi,alpha,method,N,K,nc_final,K2_nc,max_bloch_norm,soe_rel_kernel_err', rows,
                 D.meta(dict(note='n_c(t_f) at kappa = K (cos phi, sin phi); polarisation axis of the linear pulse (zeta = 0): phi = 0')))
    log('rays:', len(rows), 'rows')


def exp_dissipation(log):
    Ks = [8.0, 16.0, 32.0, 64.0]; tasks = []
    for case, upd in ct['dissipation_cases'].items():
        for alpha in ct['alphas']:
            for N in ct['N_pi'][-2:]:
                tasks.append(dict(case=case, upd=upd, zeta=1.0, phi_deg=0.0, xi=1, alpha=alpha, mode='pi', N=N, Ks=Ks))
        tasks.append(dict(case=case, upd=upd, zeta=1.0, phi_deg=0.0, xi=1, alpha=1.0, mode='ode', N=0, rtol=ct['alpha1_rtol'], Ks=Ks))
    res = dc.pool_map(_ray_task, tasks, log=log, label='dissipation')
    rows = []
    for a, r in zip(tasks, res):
        p = D.pars(None, 1.0, **a['upd'])
        const = dirac.tail_constant(a['alpha'], p, 0.0, cfg['taubar_c'], tuple(cfg['window'])) if a['alpha'] < 1 else dirac.tail_constant_alpha1(p, 0.0, tuple(cfg['window']))
        for K, v in zip(a['Ks'], r['nc']):
            pw = 2 if a['alpha'] < 1 else 4
            rows.append(f"{a['case']},{p['Gdown']},{p['Gphi']},{a['alpha']},{'PI-trapezoid' if a['mode'] == 'pi' else 'DOP853-deviation'},{a['N']},{K},{v:.12e},{pw},{K ** pw * v:.10e},{const:.10e}")
    dc.write_csv(os.path.join(OUT, 'tail_dissipation.csv'), 'case,Gdown,Gphi,alpha,method,N,K,nc_final,power,Kpow_nc,analytic_constant', rows,
                 D.meta(dict(note='zeta = 1, kappa || x, valley K. power = 2 (alpha < 1; analytic_constant = C_alpha) or 4 (alpha = 1; analytic_constant = D)')))
    for r in rows:
        log(r)


def _tf_task(a):
    p = D.pars(None, 1.0); kap = ray_points(0.0, a['Ks']); t0 = cfg['window'][0]
    res = D.simulate(kap, 1, p, a['alpha'], a['N'], a['mode'], a['n_output'], window=(t0, a['tf_max']))
    nc = dirac.observables(res['R'], kap, 1, res['t'], p)[0]
    return dict(t=res['t'], nc=nc)


def exp_tf(log):
    Ks = [8.0, 16.0, 32.0]; t0 = cfg['window'][0]; tfm = max(ct['tf_list']); T = tfm - t0
    N1 = int(round(T / ct['tf_step'])); tasks = []
    for alpha in ct['alphas']:
        for mode, Ns in (('pi', [N1 // 2, N1]), ('soe', [N1, 2 * N1, 4 * N1])):
            for N in Ns:
                tasks.append(dict(alpha=alpha, mode=mode, N=N, Ks=Ks, tf_max=tfm, n_output=int(round(T / 15.0))))
    res = dc.pool_map(_tf_task, tasks, log=log, label='tf')
    rows = []
    for a, r in zip(tasks, res):
        for i, t in enumerate(r['t']):
            if any(abs(t - x) < 1e-9 for x in ct['tf_list']):
                for j, K in enumerate(Ks):
                    rows.append(f"{a['alpha']},{'PI-trapezoid' if a['mode'] == 'pi' else 'L1-soe'},{a['N']},{T / a['N']:.10g},{t:g},{K},{r['nc'][i, j]:.12e},{K * K * r['nc'][i, j]:.10e}")
    crow = []
    p = D.pars(None, 1.0)
    for alpha in ct['alphas']:
        for tf in ct['tf_list'] + [240.0, 480.0]:
            crow.append(f"{alpha},{tf:g},{dirac.tail_constant(alpha, p, 0.0, cfg['taubar_c'], (t0, tf)):.10e}")
    dc.write_csv(os.path.join(OUT, 'tail_tf.csv'), 'alpha,method,N,h,t,K,nc,K2_nc', rows,
                 D.meta(dict(note=f'zeta = 1, kappa || x, valley K; one run on [{t0}, {tfm}] per (method, N); n_c in the instantaneous basis at time t (the field is negligible for t >= 30)')))
    dc.write_csv(os.path.join(OUT, 'tail_tf_constants.csv'), 'alpha,tf,C_alpha_analytic', crow, D.meta(dict(note='analytical constant for the window [t0, tf], zeta = 1')))
    for r in crow:
        log('C(tf):', r)


def exp_alpha1(log):
    tasks = []
    for ray in ct['rays']:
        for xi in (1, -1):
            tasks.append(dict(zeta=ray['zeta'], phi_deg=ray['phi_deg'], xi=xi, alpha=1.0, mode='ode', N=0, rtol=ct['alpha1_rtol'], Ks=ct['alpha1_K_list']))
    res = dc.pool_map(_ray_task, tasks, log=log, label='alpha1')
    rows = []
    for a, r in zip(tasks, res):
        p = D.pars(None, a['zeta']); Dc = dirac.tail_constant_alpha1(p, math.radians(a['phi_deg']), tuple(cfg['window']))
        Ks = a['Ks']
        for i, (K, v) in enumerate(zip(Ks, r['nc'])):
            sl = math.log(r['nc'][i] / r['nc'][i - 1]) / math.log(Ks[i] / Ks[i - 1]) if i > 0 and r['nc'][i] > 0 and r['nc'][i - 1] > 0 else float('nan')
            rows.append(f"{a['zeta']},{a['phi_deg']},{a['xi']},{K},{v:.12e},{K ** 4 * v:.10e},{K ** 6 * v:.10e},{sl:.4f},{Dc:.10e}")
    dc.write_csv(os.path.join(OUT, 'tail_alpha1.csv'), 'zeta,phi_deg,xi,K,nc_final,K4_nc,K6_nc,local_slope,analytic_D', rows,
                 D.meta(dict(note=f'alpha = 1, DOP853 in deviation form, rtol = {ct["alpha1_rtol"]:g}; local_slope = d ln n_c / d ln K from the previous K')))
    for r in rows:
        log(r)


def exp_constants(log):
    rows = []; t0, tf = cfg['window']
    phis = np.linspace(0.0, math.pi, 24, endpoint=False)
    for zeta in (1.0, 0.0):
        p = D.pars(None, zeta)
        for alpha in ct['constant_alphas']:
            Cs = np.array([dirac.tail_constant(alpha, p, ph, cfg['taubar_c'], (t0, tf)) for ph in phis])
            C0 = Cs[0]; C90 = Cs[12]; Cm = Cs.mean()
            pn = dict(p, Gdown=0.0, Gphi=0.0)
            Cn = np.mean([dirac.tail_constant(alpha, pn, ph, cfg['taubar_c'], (t0, tf)) for ph in phis[::6]]) if zeta == 1.0 else float('nan')
            rows.append(f'{zeta},{alpha},{C0:.10e},{C90:.10e},{Cm:.10e},{Cs.min():.10e},{Cs.max():.10e},{2 * Cm / math.pi:.10e},{Cm / (1 - alpha):.8f},{Cn:.10e}')
            log('constants', rows[-1])
        Dm = np.mean([dirac.tail_constant_alpha1(p, ph, (t0, tf)) for ph in phis])
        rows.append(f'{zeta},1.0,{dirac.tail_constant_alpha1(p, 0.0, (t0, tf)):.10e},{dirac.tail_constant_alpha1(p, math.pi / 2, (t0, tf)):.10e},{Dm:.10e},nan,nan,nan,nan,nan')
    dc.write_csv(os.path.join(OUT, 'tail_constants.csv'),
                 'zeta,alpha,C_phi0,C_phi90,C_angular_mean,C_min,C_max,slope_dNtot_dlnkc,C_mean_over_1_minus_alpha,C_mean_no_dissipation', rows,
                 D.meta(dict(note='analytical constants (Gauss quadrature of the closed formula): alpha < 1: C_alpha(phi) = lim K^2 n_c; alpha = 1 row: D(phi) = lim K^4 n_c. '
                                  'slope_dNtot_dlnkc = 2 <C>/pi (g_s = 2, two valleys). Angular mean over 24 directions')))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    dc.disable_power_throttling(); os.makedirs(OUT, exist_ok=True)
    log = dc.Log(os.path.join(OUT, f'tail_{what}.log'))
    todo = dict(rays=[exp_rays], dissipation=[exp_dissipation], tf=[exp_tf], alpha1=[exp_alpha1], constants=[exp_constants],
                all=[exp_constants, exp_alpha1, exp_dissipation, exp_tf, exp_rays])[what]
    for f in todo:
        f(log)
    log('done'); log.close()

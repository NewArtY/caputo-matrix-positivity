"""B3 (fixed wave vector) and B4 (momentum-resolved) reference computations for the fractional laser-driven
massive-Dirac model in Bloch form.

  python dirac_reference.py B3        single wave vector: DOP853/Radau (alpha = 1), implicit L1 with full and SOE
                                      history and the independent product-trapezoidal rule for alpha < 1, N refinement
  python dirac_reference.py B4        both valleys on the nested momentum grid of configs/dirac.json:
                                      L1-SOE at every N of B4.N_list (alpha < 1), DOP853 (alpha = 1); densities, the
                                      sheet current and their dependence on the cutoff kappa_c as functions of time
  python dirac_reference.py B4check   momentum-quadrature check (halved spacings), time-discretisation check
                                      (L1 + Richardson versus product-trapezoidal rule), full versus SOE history,
                                      DOP853 tolerance check
Existing output files are kept (delete a file to recompute it). Outputs: results/raw/DIRAC_v2/.

Momentum integration. The patch |kappa_x|, |kappa_y| <= kappa_max is covered by nested square frames with
midpoint cells (spacing 0.125 up to 8, 0.25 up to 16, 0.5 up to 32 in the default configuration). All sums
are accumulated per shell kappa_c[j-1] < max(|kappa_x|, |kappa_y|) < kappa_c[j], so that one run gives every
quantity as a function of the cutoff kappa_c.

Sheet current (units e omega_0^2 / v_F, spin degeneracy g_s = 2 included, valleys stored separately):
    'band' : g_s int d^2kappa/(2 pi)^2 tr[ j (rho - P_-(kappa, t)) ]   reference: instantaneous filled valence band
    'intra': part carried by the conduction/valence populations,  'inter' = 'band' - 'intra'
    'init' : g_s int d^2kappa/(2 pi)^2 tr[ j (rho - rho(t0)) ]         definition of the first manuscript version;
             contains the current of the shifted filled band (grows linearly with the cutoff)
"""
import det_common as dc
import json, math, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp
from fracsp import dirac, reference as ref, operators as op

HERE = dc.HERE
OUT = os.path.join(dc.RAW, 'DIRAC_v2')
CFGP = os.path.join(HERE, '..', 'configs', 'dirac.json')
cfg = json.load(open(CFGP))
GS = 2.0
OBS = ['nc', 'jx_band', 'jy_band', 'jx_intra', 'jy_intra', 'jx_init', 'jy_init']


def meta(extra=None):
    return dc.provenance(os.path.abspath(__file__), CFGP, extra)


def pars(a0=None, zeta=0.0, **upd):
    p = dict(cfg['pulse']); p.update(delta=cfg['delta'], Gdown=cfg['Gdown'], Gup=cfg['Gup'], Gphi=cfg['Gphi'], zeta=zeta)
    if a0 is not None:
        p['a0'] = a0
    p.update(upd)
    return p


_SOE_CACHE = {}


def soe_for(alpha, h, T, tol=None):
    """SOE kernel for the window [h, T] with its REALISED errors: (omega, lam, info)."""
    tol = cfg['soe_tol'] if tol is None else tol
    key = (float(alpha), float(h), float(T), float(tol))
    if key not in _SOE_CACHE:
        om, la, ea, er = op.soe_kernel(alpha, h, T, tol)
        _SOE_CACHE[key] = (om, la, dict(M=len(om), soe_rel_err=er, soe_abs_err=ea, soe_tol=tol))
    return _SOE_CACHE[key]


def simulate(kappa, xi, p, alpha, N, mode='soe', n_output=None, window=None, rtol=None, atol=None):
    """Bloch trajectories for the wave vectors kappa (K,2) in valley xi.

    mode: 'soe'  implicit L1, uniform mesh, SOE history          'full' implicit L1, full history (tabulated weights)
          'pi'   implicit product-trapezoidal rule (2nd order)    'ode'  DOP853 (alpha = 1 only)
          'euler' implicit Euler = L1 at alpha = 1
    Returns dict(t, R (n_out+1, K, 3), r2max, r2min, wall, info); r2max/r2min are the extrema of |r|^2 over ALL
    time steps and wave vectors (at the output times only for mode 'ode'); the smallest eigenvalue of rho along the
    trajectories is (1 - sqrt(r2max))/2."""
    t0, tf = window if window is not None else cfg['window']
    T = tf - t0; h = T / N
    n_output = N if n_output is None else n_output
    se = max(1, N // n_output)
    pref = cfg['taubar_c'] ** (1 - alpha)
    K = kappa.shape[0]
    r0 = dirac.ground_state(kappa, xi, p, t0)
    ext = [1.0, 1.0]

    def mon(n, tn, x):
        r2 = np.einsum('ki,ki->k', x, x)
        ext[0] = max(ext[0], float(r2.max())); ext[1] = min(ext[1], float(r2.min()))
    if pref == 1.0:
        gen = lambda tn: dirac.bloch_generator(kappa, xi, t0 + tn, p)
    else:
        def gen(tn):
            A, b, n, e = dirac.bloch_generator(kappa, xi, t0 + tn, p)
            return pref * A, pref * b
    info = {}
    t1 = time.perf_counter()
    if mode == 'ode':
        if alpha != 1.0:
            raise ValueError("mode 'ode' requires alpha = 1")
        rtol = cfg['B4']['ode_rtol'] if rtol is None else rtol
        atol = (cfg['B4']['ode_atol'] if rtol is None else rtol * 1e-2) if atol is None else atol
        te = t0 + np.arange(0, N + 1, se) * h
        sol = solve_ivp(lambda t, y: dirac.bloch_rhs(kappa, xi, t, y.reshape(K, 3), p).ravel(), (t0, tf), r0.ravel(),
                        method='DOP853', rtol=rtol, atol=atol, t_eval=te)
        if not sol.success:
            raise RuntimeError('DOP853 failed: ' + sol.message)
        tt = sol.t; R = sol.y.T.reshape(len(sol.t), K, 3)
        r2 = (R ** 2).sum(-1); ext = [max(1.0, float(r2.max())), float(r2.min())]
        info = dict(ode_rtol=rtol, ode_atol=atol, nfev=int(sol.nfev))
    elif mode == 'full':
        tt, R = ref.solve_l1(gen, None, r0, np.arange(N + 1) * h, alpha, store_every=se, weights='uniform', callback=mon)
        tt = tt + t0
    elif mode == 'pi':
        gen_abs = (lambda t: dirac.bloch_generator(kappa, xi, t, p)) if pref == 1.0 else (lambda t: gen(t - t0))
        tt, R = ref.solve_pi_trapezoid(gen_abs, None, r0, t0, T, N, alpha, store_every=se, callback=mon)
    elif mode in ('soe', 'euler'):
        if alpha == 1.0 or mode == 'euler':
            om = la = np.zeros(0); info = dict(M=0, soe_rel_err=0.0, soe_abs_err=0.0)          # no history: implicit Euler
        else:
            om, la, info = soe_for(alpha, h, T)
        tt, R = ref.solve_l1_soe(gen, None, r0, h, N, alpha, om, la, store_every=se, callback=mon)
        tt = tt + t0
    else:
        raise ValueError(mode)
    return dict(t=tt, R=R, r2max=ext[0], r2min=ext[1], wall=time.perf_counter() - t1, info=dict(info))


# ------------------------------------------------------------------------------------------------------------------
# nested momentum grid and shell-resolved sums
# ------------------------------------------------------------------------------------------------------------------
_GRID_CACHE = {}


def nested_grid(levels, kappa_c_list):
    """Cells of the nested square frames: dict(kappa (K,2), weight (K,), level (K,), shell (K,), kc, order).
    'shell' is the index j of the smallest cutoff kappa_c[j] whose square contains the cell; 'order' sorts the
    cells by (level, Chebyshev radius)."""
    key = json.dumps([levels, list(kappa_c_list)])
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    pts, wts, lev = [], [], []
    inner = 0.0
    kc = np.array(kappa_c_list, dtype=float)
    for li, L in enumerate(levels):
        km, s = float(L['kappa_max']), float(L['spacing']); n = int(round(2 * km / s))
        if abs(n * s - 2 * km) > 1e-9 or abs(round(inner / s) * s - inner) > 1e-9:
            raise ValueError('kappa_max of every level and of the enclosed level must be multiples of the spacing')
        for c in kc[(kc > inner) & (kc <= km)]:
            if abs(round(c / s) * s - c) > 1e-9:
                raise ValueError(f'cutoff {c} is not aligned with the spacing {s} of its level')
        g = (np.arange(n) + 0.5) * s - km
        KX, KY = np.meshgrid(g, g, indexing='ij'); k = np.c_[KX.ravel(), KY.ravel()]
        keep = np.abs(k).max(1) > inner
        pts.append(k[keep]); wts.append(np.full(int(keep.sum()), s * s)); lev.append(np.full(int(keep.sum()), li))
        inner = km
    kappa = np.concatenate(pts); weight = np.concatenate(wts); level = np.concatenate(lev)
    rho = np.abs(kappa).max(1)
    shell = np.searchsorted(kc, rho, side='right')
    if shell.max() >= len(kc):
        raise ValueError('largest cutoff is smaller than the patch')
    order = np.lexsort((rho, level))
    G = dict(kappa=kappa, weight=weight, level=level, shell=shell, kc=kc, order=order)
    _GRID_CACHE[key] = G
    return G


def shell_sums(tt, R, kappa, xi, p, wshell):
    """Weighted shell sums of the per-wave-vector observables OBS at the output times:
    S[t, shell, obs] = sum_k wshell[shell, k] obs[t, k]  (no degeneracy factor)."""
    a = dirac.pulse_A(np.asarray(tt, dtype=float), p['a0'], p['sp'], p.get('zeta', 0.0), p.get('chi', 0.0), p.get('cep', 0.0))
    hx = xi * (kappa[None, :, 0] + a[:, None, 0]); hy = kappa[None, :, 1] + a[:, None, 1]
    eps = np.sqrt(hx ** 2 + hy ** 2 + p['delta'] ** 2)
    nx = hx / eps; ny = hy / eps; nz = p['delta'] / eps
    sx = R[..., 0] + nx; sy = R[..., 1] + ny; sz = R[..., 2] + nz
    twonc = nx * sx + ny * sy + nz * sz                                   # 1 + n.r = 2 n_c
    obs = np.stack([0.5 * twonc, -xi * sx, -sy, -twonc * xi * nx, -twonc * ny,
                    -xi * (R[..., 0] - R[0, :, 0][None]), -(R[..., 1] - R[0, :, 1][None])], axis=-1)
    return np.einsum('sk,tko->tso', wshell, obs), 0.5 * twonc[-1]


def _b4_task(a):
    """One chunk of wave vectors in one valley (worker process)."""
    G = nested_grid(a['levels'], a['kc'])
    idx = G['order'][a['i0']:a['i1']]
    kap = G['kappa'][idx]; p = pars(None, a['zeta'])
    res = simulate(kap, a['xi'], p, a['alpha'], a['N'], a['mode'], a['n_output'], rtol=a.get('rtol'))
    wshell = np.zeros((len(G['kc']), len(idx))); wshell[G['shell'][idx], np.arange(len(idx))] = G['weight'][idx]
    S, ncf = shell_sums(res['t'], res['R'], kap, a['xi'], p, wshell)
    return dict(S=S * GS / (2 * math.pi) ** 2, ncf=ncf, t=res['t'], r2max=res['r2max'], r2min=res['r2min'], wall=res['wall'], info=res['info'])


def momentum_run(alpha, zeta, N, mode, levels, kc, path, log, n_output=None, chunk=None, rtol=None, note=''):
    """All chunks and both valleys for one (alpha, zeta, N); writes one npz file (skipped if it exists)."""
    if os.path.exists(path):
        log('exists, skipped:', dc.rel(path)); return
    c = cfg['B4']; n_output = c['n_output'] if n_output is None else n_output; chunk = c['chunk'] if chunk is None else chunk
    G = nested_grid(levels, kc); K = len(G['kappa'])
    tasks = [dict(alpha=alpha, zeta=zeta, xi=xi, N=N, mode=mode, levels=levels, kc=list(kc), i0=i0, i1=min(K, i0 + chunk), n_output=n_output, rtol=rtol)
             for xi in (1, -1) for i0 in range(0, K, chunk)]
    t1 = time.perf_counter()
    res = dc.pool_map(_b4_task, tasks)
    nt = len(res[0]['t']); S = np.zeros((2, nt, len(kc), len(OBS))); ncf = np.zeros((2, K))
    r2max = [1.0, 1.0]; r2min = [1.0, 1.0]; cpu = 0.0
    for tk, r in zip(tasks, res):
        v = 0 if tk['xi'] == 1 else 1
        S[v] += r['S']; ncf[v, G['order'][tk['i0']:tk['i1']]] = r['ncf']
        r2max[v] = max(r2max[v], r['r2max']); r2min[v] = min(r2min[v], r['r2min']); cpu += r['wall']
    S = np.cumsum(S, axis=2)                                      # sums within kappa_c
    info = res[0]['info']
    extra = dict(alpha=alpha, zeta=zeta, N=N, mode=mode, n_points_per_valley=K, wall_s=round(time.perf_counter() - t1, 1), cpu_s=round(cpu, 1),
                 max_bloch_norm=[math.sqrt(x) for x in r2max], min_purity=[0.5 * (1 + x) for x in r2min],
                 min_eigenvalue=[0.5 * (1 - math.sqrt(x)) for x in r2max], note=note, **info)
    dc.write_npz(path, meta(extra), t=res[0]['t'], kappa_c=np.array(kc), S=S, obs=np.array(OBS), valleys=np.array([1, -1]),
                 kappa=G['kappa'], weight=G['weight'], level=G['level'], ncf=ncf)
    Nf = S[:, -1, :, 0]; i3 = list(kc).index(3.0) if 3.0 in list(kc) else 0
    log(f'{os.path.basename(path)}: N_K({kc[i3]:g})={Nf[0, i3]:.6f} N_Kp={Nf[1, i3]:.6f} max|Jx_band|({kc[i3]:g})={np.abs(S[:, :, i3, 1].sum(0)).max():.5f} '
        f'max|r|-1={max(extra["max_bloch_norm"]) - 1:.1e} {info} wall={extra["wall_s"]}s cpu={extra["cpu_s"]}s')


def tag(alpha, zeta):
    return f'a{alpha}_z{zeta}'


def B4(log):
    c = cfg['B4']
    for zeta in c['zetas']:
        for alpha in c['alphas']:
            if alpha == 1.0:
                momentum_run(1.0, zeta, c['n_output'], 'ode', c['levels'], c['kappa_c_list'], os.path.join(OUT, f'B4_{tag(1.0, zeta)}_ode.npz'), log)
            else:
                for N in c['N_list']:
                    momentum_run(alpha, zeta, N, 'soe', c['levels'], c['kappa_c_list'], os.path.join(OUT, f'B4_{tag(alpha, zeta)}_N{N}.npz'), log)


def halved(levels):
    return [dict(kappa_max=L['kappa_max'], spacing=L['spacing'] / 2) for L in levels]


def _soe_check_task(a):
    alpha = a['alpha']; cs = cfg['B4check']['soe']; n = cs['grid_n']; km = cs['kappa_max']
    g = (np.arange(n) + 0.5) / n * 2 * km - km; KX, KY = np.meshgrid(g, g, indexing='ij'); kap = np.c_[KX.ravel(), KY.ravel()]
    p = pars(None, 1.0)
    rs = simulate(kap, 1, p, alpha, cs['N'], 'soe', 256); rf = simulate(kap, 1, p, alpha, cs['N'], 'full', 256)
    ncs = dirac.observables(rs['R'][-1:], kap, 1, [cfg['window'][1]], p)[0][0]; ncf = dirac.observables(rf['R'][-1:], kap, 1, [cfg['window'][1]], p)[0][0]
    return (f"{alpha},{cs['N']},{len(kap)},{rs['info']['M']},{rs['info']['soe_rel_err']:.4e},{np.abs(rs['R'] - rf['R']).max():.4e},"
            f"{np.abs(ncs - ncf).max():.4e},{(np.abs(ncs - ncf) / ncf).max():.4e},{rf['wall']:.2f},{rs['wall']:.2f}")


def B4check(log):
    c = cfg['B4']; ck = cfg['B4check']
    # (1) momentum quadrature: halved spacings
    for case in ck['grid']['cases']:
        lv = halved(c['levels'] if case['levels'] == 'all' else c['levels'][:1])
        kc = [x for x in c['kappa_c_list'] if x <= lv[-1]['kappa_max']]
        a, z = case['alpha'], case['zeta']
        if a == 1.0:
            momentum_run(a, z, c['n_output'], 'ode', lv, kc, os.path.join(OUT, f'B4check_grid_{tag(a, z)}_ode.npz'), log, note='halved spacings')
        else:
            momentum_run(a, z, ck['grid']['N'], 'soe', lv, kc, os.path.join(OUT, f'B4check_grid_{tag(a, z)}_N{ck["grid"]["N"]}.npz'), log, note='halved spacings')
    # (2) time discretisation on a coarse nested grid: L1 + Richardson versus product-trapezoidal rule
    ct = ck['time']
    for z in ct['zetas']:
        for a in ct['alphas']:
            for N in ct['N_l1']:
                momentum_run(a, z, N, 'soe', ct['levels'], ct['kappa_c_list'], os.path.join(OUT, f'B4check_time_l1_{tag(a, z)}_N{N}.npz'), log, chunk=64, note='coarse grid')
            for N in ct['N_pi']:
                momentum_run(a, z, N, 'pi', ct['levels'], ct['kappa_c_list'], os.path.join(OUT, f'B4check_time_pi_{tag(a, z)}_N{N}.npz'), log, chunk=64, note='coarse grid')
        # (4) DOP853 tolerance on the same coarse grid
        for rt in ck['ode']['rtols']:
            momentum_run(1.0, z, c['n_output'], 'ode', ct['levels'], ct['kappa_c_list'], os.path.join(OUT, f'B4check_ode_{tag(1.0, z)}_rtol{rt:g}.npz'), log,
                         chunk=64, rtol=rt, note='coarse grid, tolerance check')
    # (3) full versus SOE history
    path = os.path.join(OUT, 'B4check_soe.csv')
    if not os.path.exists(path):
        rows = dc.pool_map(_soe_check_task, [dict(alpha=a) for a in ck['soe']['alphas']])
        dc.write_csv(path, 'alpha,N,n_wave_vectors,M,soe_rel_kernel_err,max_abs_diff_bloch,max_abs_diff_nc_final,max_rel_diff_nc_final,wall_full_s,wall_soe_s', rows,
                     meta(dict(note='full-history (tabulated weights) versus SOE implicit L1 on an 8 x 8 midpoint grid, kappa_max = 3, valley K, zeta = 1; '
                                    'differences over 257 output times')))
        for r in rows:
            log('full vs SOE:', r)


# ------------------------------------------------------------------------------------------------------------------
# B3
# ------------------------------------------------------------------------------------------------------------------

def _b3_task(a):
    c = cfg['B3']; kappa = np.array([c['kappa']], dtype=float); xi = c['xi']; p = pars(a['a0'], c['zeta'])
    t0, tf = cfg['window']
    if a['method'] in ('DOP853', 'Radau'):
        r0 = dirac.ground_state(kappa, xi, p, t0)[0]
        t1 = time.perf_counter()
        s = solve_ivp(lambda t, r: dirac.bloch_rhs(kappa, xi, t, r[None, :], p)[0], (t0, tf), r0, method=a['method'], rtol=a['rtol'], atol=a['rtol'] * 1e-2, dense_output=True)
        if not s.success:
            raise RuntimeError(a['method'] + ' failed: ' + s.message)
        w = time.perf_counter() - t1
        tt = np.linspace(t0, tf, c['n_output'] + 1); R = s.sol(tt).T[:, None, :]
        r2 = (R ** 2).sum(-1); r2max, r2min = float(r2.max()), float(r2.min()); info = {}; Nlab = f"{a['rtol']:g}"
    else:
        res = simulate(kappa, xi, p, a['alpha'], a['N'], a['mode'], c['n_output'])
        tt, R, r2max, r2min, w, info = res['t'], res['R'], res['r2max'], res['r2min'], res['wall'], res['info']; Nlab = str(a['N'])
    nc, coh, pur = dirac.observables(R, kappa, xi, tt, p)
    row = (f"{a['a0']},{a['alpha']},{a['method']},{Nlab},{nc[-1, 0]:.12e},{coh.max():.10e},{0.5 * (1 + r2min):.10e},{math.sqrt(max(r2max, 1.0)):.15f},"
           f"{0.5 * (1 - math.sqrt(max(r2max, 1.0))):.3e},{info.get('M', 0)},{info.get('soe_rel_err', 0.0):.3e},{w:.3f}")
    traj = np.c_[tt, R[:, 0], nc[:, 0], coh[:, 0], pur[:, 0]] if a.get('keep') else None
    return row, traj


def B3(log):
    c = cfg['B3']; tasks = []
    for a0 in c['a0_list']:
        for rt in c['ode_rtols']:
            for m in ('DOP853', 'Radau'):
                tasks.append(dict(a0=a0, alpha=1.0, method=m, rtol=rt, keep=(m == 'DOP853' and rt == c['ode_rtols'][-1])))
        for N in c['N_euler']:
            tasks.append(dict(a0=a0, alpha=1.0, method='L1-euler', mode='euler', N=N))
        for alpha in c['alphas']:
            if alpha == 1.0:
                continue
            for N in c['N_soe']:
                tasks.append(dict(a0=a0, alpha=alpha, method='L1-soe', mode='soe', N=N))
            for N in c['N_full']:
                tasks.append(dict(a0=a0, alpha=alpha, method='L1-full', mode='full', N=N))
            for N in c['N_pi']:
                tasks.append(dict(a0=a0, alpha=alpha, method='PI-trapezoid', mode='pi', N=N, keep=(N == c['N_pi'][-1])))
    tasks.sort(key=lambda a: -(a.get('N', 0) ** (2 if a.get('mode') in ('pi', 'full') else 1)))        # long tasks first
    res = dc.pool_map(_b3_task, tasks, log=log, label='B3')
    order = sorted(range(len(tasks)), key=lambda i: (tasks[i]['a0'], -tasks[i]['alpha'], tasks[i]['method'], tasks[i].get('N', 0), tasks[i].get('rtol', 0)))
    rows = [res[i][0] for i in order]
    dc.write_csv(os.path.join(OUT, 'B3_summary.csv'),
                 'a0,alpha,method,N_or_rtol,nc_final,max_coherence,min_purity,max_bloch_norm,min_eigenvalue,soe_M,soe_rel_kernel_err,wall_s', rows,
                 meta(dict(note='kappa = 0, valley xi = +1, zeta = 0. min_purity / max_bloch_norm / min_eigenvalue = (1 - max|r|)/2 are extrema over ALL time steps '
                                '(output times for DOP853/Radau); max_coherence over 1025 output times. L1-full uses tabulated uniform weights. '
                                'soe_rel_kernel_err is the REALISED relative sup-error of the SOE kernel on [h, T]')))
    for i in order:
        if res[i][1] is not None:
            a = tasks[i]
            dc.write_npz(os.path.join(OUT, f"B3_traj_a0{a['a0']}_alpha{a['alpha']}.npz"),
                         meta(dict(a0=a['a0'], alpha=a['alpha'], method=a['method'], N_or_rtol=a.get('N', a.get('rtol')))),
                         columns=np.array(['t', 'rx', 'ry', 'rz', 'nc', 'coh', 'purity']), traj=res[i][1])
    for r in rows:
        log(r)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else ''
    if what not in ('B3', 'B4', 'B4check'):
        print(__doc__); sys.exit(1)
    dc.disable_power_throttling()
    os.makedirs(OUT, exist_ok=True)
    log = dc.Log(os.path.join(OUT, f'{what}.log'))
    {'B3': B3, 'B4': B4, 'B4check': B4check}[what](log)
    log('done'); log.close()

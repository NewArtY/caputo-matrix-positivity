"""Dimensionless laser-driven massive-Dirac two-band model in Bloch form.

rho = (I + r . sigma)/2 ; time unit 1/omega_0, energies in hbar*omega_0, kappa = v_F k / omega_0.
    ^C D^alpha r = taubar^{1-alpha} [ 2 h x r - G1 (n.r - r_eq) n - G2 (r - (n.r) n) ]
with h = (xi kx_pi, ky_pi, delta), n = h/|h|, G1 = Gdown + Gup, r_eq = (Gup - Gdown)/G1,
G2 = G1/2 + Gphi (manuscript Eqs. for the GKSL closure with instantaneous projectors).

Observables (per wave vector; the momentum integration is done by the experiment drivers):
    n_c = (1 + n.r)/2                          conduction population (instantaneous basis)
    current, referred to the INSTANTANEOUS filled valence band P_- = (I - n.sigma)/2:
        tr[ j (rho - P_-) ] = -(xi (r_x + n_x), r_y + n_y)      in units e v_F
    The projector P_-(kappa, t) is taken at the kinetic wave vector kappa + a(t). Subtracting the INITIAL state
    instead (rho - rho(t0)) leaves the current of the adiabatically shifted filled band, which for a patch fixed
    in the canonical wave vector grows linearly with the cutoff (sea_current_square).
"""
import math
import numpy as np


def pulse_A(tbar, a0, sp, zeta=0.0, chi=0.0, cep=0.0):
    """Normalized vector potential a0*g(tbar); tbar = omega_0 (t - t_c). Scalar tbar -> array (2,);
    array tbar of shape S -> array S + (2,)."""
    f = np.exp(-tbar ** 2 / (2 * sp ** 2))
    ph = tbar + cep
    e1 = np.array([np.cos(chi), np.sin(chi)])
    e2 = np.array([-np.sin(chi), np.cos(chi)])
    if np.ndim(tbar) == 0:
        return a0 * f / np.sqrt(1 + zeta ** 2) * (e1 * np.cos(ph) + zeta * e2 * np.sin(ph))
    pre = (a0 * f / np.sqrt(1 + zeta ** 2))[..., None]
    return pre * (e1 * np.cos(ph)[..., None] + zeta * e2 * np.sin(ph)[..., None])


def pulse_dA(tbar, a0, sp, zeta=0.0, chi=0.0, cep=0.0):
    """Analytic derivative d(a0 g)/d tbar of the vector potential (same shape conventions as pulse_A)."""
    tb = np.asarray(tbar, dtype=float)
    f = np.exp(-tb ** 2 / (2 * sp ** 2)); ph = tb + cep
    e1 = np.array([np.cos(chi), np.sin(chi)])
    e2 = np.array([-np.sin(chi), np.cos(chi)])
    dc = -tb / sp ** 2 * np.cos(ph) - np.sin(ph)          # d/dt [f cos]/f
    ds = -tb / sp ** 2 * np.sin(ph) + np.cos(ph)          # d/dt [f sin]/f
    pre = (a0 * f / np.sqrt(1 + zeta ** 2))[..., None]
    return pre * (e1 * dc[..., None] + zeta * e2 * ds[..., None])


def pulse_E(tbar, a0, sp, zeta=0.0, chi=0.0, cep=0.0, eps=None):
    """Dimensionless field  E_bar = -d(a0 g)/d tbar (analytic; the argument eps of the former
    finite-difference version is ignored)."""
    return -pulse_dA(tbar, a0, sp, zeta, chi, cep)


def _pulse_args(pars):
    return pars['a0'], pars['sp'], pars.get('zeta', 0.0), pars.get('chi', 0.0), pars.get('cep', 0.0)


def axis(kappa, xi, tbar, pars):
    """Hamiltonian vector h (K,3), instantaneous axis n = h/|h| (K,3) and band energy eps (K,)."""
    a = pulse_A(tbar, *_pulse_args(pars))
    K = kappa.shape[0]
    h = np.empty((K, 3))
    h[:, 0] = xi * (kappa[:, 0] + a[0]); h[:, 1] = kappa[:, 1] + a[1]; h[:, 2] = pars['delta']
    eps = np.sqrt(h[:, 0] ** 2 + h[:, 1] ** 2 + h[:, 2] ** 2)
    return h, h / eps[:, None], eps


def rates(pars):
    """(G1, G2, r_eq) of the GKSL closure."""
    G1 = pars['Gdown'] + pars.get('Gup', 0.0)
    G2 = G1 / 2 + pars['Gphi']
    req = (pars.get('Gup', 0.0) - pars['Gdown']) / G1 if G1 > 0 else 0.0
    return G1, G2, req


def bloch_generator(kappa, xi, tbar, pars):
    """Return (A, b, n, eps): A (K,3,3), b (K,3) for the right-hand side A r + b (without memory prefactor)."""
    a = pulse_A(tbar, pars['a0'], pars['sp'], pars.get('zeta', 0.0), pars.get('chi', 0.0), pars.get('cep', 0.0))
    kx = kappa[:, 0] + a[0]
    ky = kappa[:, 1] + a[1]
    K = kappa.shape[0]
    h = np.stack([xi * kx, ky, np.full(K, pars['delta'])], axis=1)
    eps = np.linalg.norm(h, axis=1)
    n = h / eps[:, None]
    A = np.zeros((K, 3, 3))
    # 2 h x r
    A[:, 0, 1] = -2 * h[:, 2]; A[:, 0, 2] = 2 * h[:, 1]
    A[:, 1, 0] = 2 * h[:, 2];  A[:, 1, 2] = -2 * h[:, 0]
    A[:, 2, 0] = -2 * h[:, 1]; A[:, 2, 1] = 2 * h[:, 0]
    G1 = pars['Gdown'] + pars.get('Gup', 0.0)
    G2 = G1 / 2 + pars['Gphi']
    req = (pars.get('Gup', 0.0) - pars['Gdown']) / G1 if G1 > 0 else 0.0
    nn = n[:, :, None] * n[:, None, :]
    A += -G1 * nn - G2 * (np.eye(3)[None] - nn)
    b = G1 * req * n
    return A, b, n, eps


def bloch_rhs(kappa, xi, tbar, r, pars):
    """Right-hand side A(t) r + b(t) evaluated without forming A (alpha = 1 integrators); r: (K,3)."""
    h, n, eps = axis(kappa, xi, tbar, pars)
    G1, G2, req = rates(pars)
    par = (n * r).sum(1, keepdims=True)
    out = np.empty_like(r)
    out[:, 0] = 2 * (h[:, 1] * r[:, 2] - h[:, 2] * r[:, 1])
    out[:, 1] = 2 * (h[:, 2] * r[:, 0] - h[:, 0] * r[:, 2])
    out[:, 2] = 2 * (h[:, 0] * r[:, 1] - h[:, 1] * r[:, 0])
    out -= G1 * (par - req) * n + G2 * (r - par * n)
    return out


def deviation_rhs(kappa, xi, tbar, s, pars):
    """alpha = 1 right-hand side for the deviation s = r + n from the instantaneous valence state:
        ds/dt = A(t) s + G1 (1 + r_eq) n + dn/dt ,   s(t0) = 0,   n_c = n.s/2   (no cancellation for tiny n_c).
    dn/dt = (dh/dt - n (n.dh/dt))/eps with dh/dt = (xi da_x/dt, da_y/dt, 0)."""
    h, n, eps = axis(kappa, xi, tbar, pars)
    G1, G2, req = rates(pars)
    da = pulse_dA(tbar, *_pulse_args(pars))
    hd = np.zeros_like(h); hd[:, 0] = xi * da[0]; hd[:, 1] = da[1]
    nd = (hd - n * (n * hd).sum(1, keepdims=True)) / eps[:, None]
    par = (n * s).sum(1, keepdims=True)
    out = np.empty_like(s)
    out[:, 0] = 2 * (h[:, 1] * s[:, 2] - h[:, 2] * s[:, 1])
    out[:, 1] = 2 * (h[:, 2] * s[:, 0] - h[:, 0] * s[:, 2])
    out[:, 2] = 2 * (h[:, 0] * s[:, 1] - h[:, 1] * s[:, 0])
    out += -G1 * par * n - G2 * (s - par * n) + G1 * (1 + req) * n + nd
    return out


def ground_state(kappa, xi, pars, tbar0):
    _, _, n, _ = bloch_generator(kappa, xi, tbar0, pars)
    return -n   # lower-band pure state, |r| = 1


def observables(R, kappa, xi, tgrid, pars):
    """Conduction population, invariant coherence magnitude and purity from Bloch vectors R (Nt,K,3)."""
    nc = np.zeros(R.shape[:2]); coh = np.zeros(R.shape[:2])
    for i, tb in enumerate(tgrid):
        _, n, _ = axis(kappa, xi, tb, pars)
        par = (R[i] * n).sum(1)
        nc[i] = 0.5 * (1 + par)
        perp = R[i] - par[:, None] * n
        coh[i] = 0.5 * np.linalg.norm(perp, axis=1)
    purity = 0.5 * (1 + (R ** 2).sum(-1))
    return nc, coh, purity


def current_integrands(r, n, xi, r0=None):
    """Per-wave-vector current integrands in units e v_F (to be multiplied by g_s d^2kappa/(2 pi)^2 and summed
    over wave vectors and valleys). r, n: (K,3) at the same time.

    Returns a dict of (K,2) arrays:
      'band'  : tr[j (rho - P_-(t))] = -(xi (r_x + n_x), r_y + n_y)     (cutoff-convergent definition)
      'intra' : part carried by the populations, -2 n_c (xi n_x, n_y) = -2 n_c v_c/v_F
      'inter' : 'band' - 'intra' (interband polarisation current)
      'init'  : tr[j (rho - rho(t0))] = -(xi (r_x - r0_x), r_y - r0_y)  (only if r0 is given; contains the
                current of the shifted filled band and is not cutoff convergent)."""
    s = r + n
    band = -np.stack([xi * s[:, 0], s[:, 1]], axis=1)
    twonc = (n * s).sum(1)                              # = 1 + n.r = 2 n_c
    intra = -twonc[:, None] * np.stack([xi * n[:, 0], n[:, 1]], axis=1)
    out = dict(band=band, intra=intra, inter=band - intra)
    if r0 is not None:
        out['init'] = -np.stack([xi * (r[:, 0] - r0[:, 0]), r[:, 1] - r0[:, 1]], axis=1)
    return out


def sea_current_square(kappa_max, a, gs=2.0, nvalley=2):
    """Leading term (kappa_max >> |a|, delta) of the current of the adiabatically shifted filled valence band over
    the square patch |kappa_x|, |kappa_y| <= kappa_max, i.e. of  J[rho - rho(t0)] - J[rho - P_-(t)] :
        J_sea = gs nvalley/(2 pi)^2 * 4 arsinh(1) kappa_max a(t) = (4 arsinh(1)/pi^2) kappa_max a(t)
    (units e omega_0^2/v_F; a = a0 g(t)). It is linear in the cutoff: a regularisation artefact."""
    return gs * nvalley / (2 * math.pi) ** 2 * 4 * math.asinh(1.0) * kappa_max * np.asarray(a)


# ----------------------------------------------------------------------------------------------
# Large-|kappa| asymptotics of the final conduction population
# ----------------------------------------------------------------------------------------------

def _a_perp(t, pars, phi):
    """Component of the vector potential a0 g(t) perpendicular to the wave-vector direction phi."""
    a = pulse_A(np.asarray(t, dtype=float), *_pulse_args(pars))
    return a[..., 1] * math.cos(phi) - a[..., 0] * math.sin(phi)


def _ml_small(z, a, b):
    """E_{a,b}(-z) for 0 <= z <~ 3 by the power series in double precision (array z)."""
    from scipy.special import gammaln
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z); k = 0
    while True:
        term = (-1.0) ** k * np.exp(k * np.log(np.maximum(z, 1e-300)) - gammaln(a * k + b)) if k > 0 else np.full_like(z, math.exp(-math.lgamma(b)))
        out += term
        if np.abs(term).max() < 1e-17 and k > 3:
            return out
        k += 1
        if k > 400:
            raise RuntimeError('_ml_small did not converge (argument too large)')


def tail_source(t, pars, phi, alpha, t0, panel=0.5, ng=12):
    """Q(t) = v D^alpha v - D^alpha(v^2)/2 for v = a_perp (a0 included), evaluated from the exact identity
        Q(t) = [ (v(t) - v(t0))^2 (t - t0)^{-alpha} + alpha int_0^{t-t0} x^{1-alpha} g(x) dx ] / (2 Gamma(1-alpha)),
        g(x) = ((v(t) - v(t-x))/x)^2   (analytic, g(0) = v'(t)^2),
    by Gauss-Jacobi quadrature (weight x^{1-alpha}) on [0, panel] and Gauss-Legendre panels beyond.
    Q >= 0; it is the source of the |kappa|^-2 population tail, n_c ~ sigma/(2 kappa^2), D^alpha sigma + c G1 sigma = Q."""
    from scipy.special import roots_jacobi, roots_legendre
    xj, wj = roots_jacobi(ng, 0.0, 1.0 - alpha)
    xl, wl = roots_legendre(ng)
    t = np.atleast_1d(np.asarray(t, dtype=float)); out = np.empty(t.shape)
    g1 = math.gamma(1.0 - alpha)
    for i, ti in enumerate(t):
        X = ti - t0
        if X <= 0.0:
            out[i] = 0.0; continue
        vt = _a_perp(ti, pars, phi)
        d = min(panel, X)
        u = 0.5 * d * (1.0 + xj)                                   # nodes in [0, d]
        gq = ((vt - _a_perp(ti - u, pars, phi)) / u) ** 2
        integ = (0.5 * d) ** (2.0 - alpha) * np.dot(wj, gq)
        if X > d:
            npan = int(math.ceil((X - d) / panel)); edges = np.linspace(d, X, npan + 1)
            mid = 0.5 * (edges[1:] + edges[:-1]); half = 0.5 * (edges[1:] - edges[:-1])
            x = (mid[:, None] + half[:, None] * xl[None, :]).ravel(); w = (half[:, None] * wl[None, :]).ravel()
            integ += np.dot(w, x ** (-1.0 - alpha) * (vt - _a_perp(ti - x, pars, phi)) ** 2)
        out[i] = ((vt - _a_perp(t0, pars, phi)) ** 2 * X ** (-alpha) + alpha * integ) / (2.0 * g1)
    return out


def tail_constant(alpha, pars, phi, taubar=1.0, window=(-30.0, 30.0), panel=0.5, ng=12, nseries=8):
    """Analytical constant of the high-momentum tail for 0 < alpha < 1,
        n_c(kappa, t_f) = C_alpha(phi) / |kappa|^2 [1 + o(1)],
        C_alpha = (1/2) int_{t0}^{tf} (tf - s)^{alpha-1} E_{alpha,alpha}(-c G1 (tf - s)^alpha) Q(s) ds,  c = taubar^{1-alpha},
    with Q from tail_source and phi the polar angle of kappa. C_alpha is independent of the valley, of the gap and
    of Gphi, does not vanish for G1 = Gphi = 0, and is proportional to the squared field component perpendicular
    to kappa. The outer integral uses Gauss-Jacobi rules for the powers tau^{k alpha - 1} of the kernel on the
    first panel and Gauss-Legendre panels elsewhere."""
    from scipy.special import roots_jacobi, roots_legendre
    t0, tf = window; T = tf - t0
    G1, G2, req = rates(pars)
    lam = taubar ** (1.0 - alpha) * G1
    xl, wl = roots_legendre(ng)
    d = min(panel, T)
    total = 0.0
    # first panel [0, d] in tau = tf - s: kernel = sum_k (-lam)^k tau^{(k+1) alpha - 1}/Gamma((k+1) alpha)
    for k in range(nseries):
        p = (k + 1) * alpha - 1.0
        xj, wj = roots_jacobi(ng, 0.0, p)
        tau = 0.5 * d * (1.0 + xj)
        Qv = tail_source(tf - tau, pars, phi, alpha, t0, panel, ng)
        total += (-lam) ** k / math.gamma((k + 1) * alpha) * (0.5 * d) ** (p + 1.0) * np.dot(wj, Qv)
        if lam == 0.0:
            break
    if T > d:
        npan = int(math.ceil((T - d) / panel)); edges = np.linspace(d, T, npan + 1)
        mid = 0.5 * (edges[1:] + edges[:-1]); half = 0.5 * (edges[1:] - edges[:-1])
        tau = (mid[:, None] + half[:, None] * xl[None, :]).ravel(); w = (half[:, None] * wl[None, :]).ravel()
        kern = tau ** (alpha - 1.0) * _ml_small(lam * tau ** alpha, alpha, alpha)
        total += np.dot(w, kern * tail_source(tf - tau, pars, phi, alpha, t0, panel, ng))
    return 0.5 * total


def tail_constant_alpha1(pars, phi, window=(-30.0, 30.0), n=200001):
    """alpha = 1: n_c(kappa, t_f) = D(phi)/|kappa|^4 [1 + o(1)] with
        D = (Gphi/8) int |d a_perp/dt|^2 exp(-G1 (tf - t)) dt
    (only pure dephasing excites irreversibly; the reversible virtual population |dn/dt|^2/(16 eps^2) vanishes
    after the pulse)."""
    t0, tf = window
    t = np.linspace(t0, tf, n)
    da = pulse_dA(t, *_pulse_args(pars))
    dv = da[:, 1] * math.cos(phi) - da[:, 0] * math.sin(phi)
    G1, G2, req = rates(pars)
    f = dv ** 2 * np.exp(-G1 * (tf - t))
    h = t[1] - t[0]
    integ = h * (f.sum() - 0.5 * (f[0] + f[-1]))                  # trapezoidal rule (integrand vanishes at both ends)
    return pars['Gphi'] / 8.0 * integ

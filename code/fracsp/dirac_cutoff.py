"""Next order of the large-momentum expansion of the laser-driven Dirac model and the dependence of the
band-referenced sheet current on the momentum cutoff.

Notation (dimensionless, as in fracsp.dirac): kappa = |kappa| (cos phi, sin phi), a_perp = a_y cos phi - a_x sin phi,
a_par = a_x cos phi + a_y sin phi, c = taubar^(1-alpha), D = Caputo derivative of order alpha with lower terminal t0,
e_mu(t) = t^(alpha-1) E_{alpha,alpha}(-mu t^alpha), (e * f)(t) = int_{t0}^{t} e(t - s) f(s) ds, G1 = Gdown (Gup = 0),
G2 = G1/2 + Gphi. In the frame b1 = (xi cos phi, sin phi, 0), b2 = (-sin phi, xi cos phi, 0), b3 = (0, 0, 1) the deviation
v = r + n of the Bloch vector from the instantaneous valence state has the components

    v_1 = 2 C/kappa^2 + [2 C3 - (xi delta/(2c)) D a_perp]/kappa^3 + O(kappa^-4),
    v_2 = [xi Y + (delta/(2c)) D a_par]/kappa^3 + O(kappa^-4),        v_3 = xi D a_perp/(2c kappa^2) + O(kappa^-3),

    C  = (1/2) e_{c G1} * Q[a_perp],            Q[f] = f D f - D(f^2)/2                   (fracsp.dirac.tail_constant),
    C3 = (1/2) e_{c G1} * T[a_perp, a_par],     T[p, q] = D(p^2 q) - p D(p q) - p q D p,
    Y  = (G2/(4c)) D a_perp + 2 a_perp C + D D a_perp/(4 c^2),

so that n_c = C/kappa^2 + C3/kappa^3 + O(kappa^-4). The terms of order kappa^-2 are odd under kappa -> -kappa in the
current and cancel in a centrally symmetric region; the terms of order kappa^-3, summed over the valleys, give

    J_x(t; kappa_c) = J_x,inf(t) + B_x(t)/kappa_c + ...,   B_x = (g_s/(2 pi^2)) int_0^{2 pi} w(phi) [2 cos(phi) C3 - sin(phi) Y] dphi,
    J_y(t; kappa_c) = J_y,inf(t) + B_y(t)/kappa_c + ...,   B_y = (g_s/(2 pi^2)) int_0^{2 pi} w(phi) [2 sin(phi) C3 + cos(phi) Y] dphi,

with w = max(|cos phi|, |sin phi|) for the square |kappa_x|, |kappa_y| <= kappa_c (w = 1 for a disc). For alpha = 1:
C = C3 = 0, Y = (G2 a_perp' + a_perp'')/4 and B = (sqrt(2) g_s/(4 pi^2)) (a'' + G2 a'), so that the term 1/kappa_c is
not an effect of the memory. The expansion is formal (no bound of the remainder); current_cutoff_coefficient is
compared with the cutoff sweep of the benchmark B4 in Table_Dirac_B4_fits.csv.

The Caputo derivatives use the identity Gamma(1 - alpha) D f(s) = (f(s) - f(t0))/(s - t0)^alpha
+ alpha int (f(s) - f(u))/(s - u)^(1 + alpha) du, which gives
    2 Gamma(1 - alpha) Q[f] = Bil[f, f],    Gamma(1 - alpha) T[p, q] = -Bil[p, p q],
    Bil[f, g](s) = (f(s) - f(t0))(g(s) - g(t0))/(s - t0)^alpha + alpha int_{t0}^{s} (f(s) - f(u))(g(s) - g(u))/(s - u)^(1 + alpha) du.
"""
import math

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre

from . import dirac


def pulse_d2A(tbar, a0, sp, zeta=0.0, chi=0.0, cep=0.0):
    """Second derivative of the vector potential fracsp.dirac.pulse_A (array tbar of shape S -> array S + (2,))."""
    tb = np.asarray(tbar, dtype=float)
    f = np.exp(-tb ** 2 / (2 * sp ** 2)); ph = tb + cep
    e1 = np.array([np.cos(chi), np.sin(chi)])
    e2 = np.array([-np.sin(chi), np.cos(chi)])
    q = tb ** 2 / sp ** 4 - 1.0 / sp ** 2 - 1.0
    ddc = q * np.cos(ph) + 2 * tb / sp ** 2 * np.sin(ph)
    dds = q * np.sin(ph) - 2 * tb / sp ** 2 * np.cos(ph)
    pre = (a0 * f / np.sqrt(1 + zeta ** 2))[..., None]
    return pre * (e1 * ddc[..., None] + zeta * e2 * dds[..., None])


class _Quad:
    """Nodes and weights for int_0^X x^p G(x) dx with smooth G: Gauss-Jacobi on [0, min(panel, X)], Gauss-Legendre
    panels beyond; the weights include x^p."""

    def __init__(self, power, panel=0.5, ng=14):
        self.p, self.panel = power, panel
        self.xj, self.wj = roots_jacobi(ng, 0.0, power)
        self.xl, self.wl = roots_legendre(ng)

    def nodes(self, X):
        d = min(self.panel, X)
        x = [0.5 * d * (1 + self.xj)]; w = [(0.5 * d) ** (self.p + 1) * self.wj]
        if X > d:
            npan = int(math.ceil((X - d) / self.panel)); e = np.linspace(d, X, npan + 1)
            mid = 0.5 * (e[1:] + e[:-1]); half = 0.5 * (e[1:] - e[:-1])
            xx = (mid[:, None] + half[:, None] * self.xl[None, :]).ravel()
            x.append(xx); w.append((half[:, None] * self.wl[None, :]).ravel() * xx ** self.p)
        return np.concatenate(x), np.concatenate(w)


def bilinear_form(f, g, s, alpha, t0, quad=None):
    """Bil[f, g](s) for callables f, g and an array s of times (zero at s = t0)."""
    quad = _Quad(1.0 - alpha) if quad is None else quad
    out = np.zeros(len(s))
    for i, si in enumerate(s):
        X = si - t0
        if X <= 0.0:
            continue
        x, w = quad.nodes(X)
        fs, gs = f(si), g(si)
        out[i] = (fs - f(t0)) * (gs - g(t0)) * X ** (-alpha) + alpha * np.dot(w, (fs - f(si - x)) / x * (gs - g(si - x)) / x)
    return out


def caputo_smooth(df, s, alpha, t0, quad=None):
    """Caputo derivative of order alpha < 1 of a smooth function with derivative df, at the times s."""
    quad = _Quad(-alpha) if quad is None else quad
    out = np.zeros(len(s))
    for i, si in enumerate(s):
        X = si - t0
        if X > 0.0:
            x, w = quad.nodes(X)
            out[i] = np.dot(w, df(si - x)) / gamma(1.0 - alpha)
    return out


def fractional_integral(f, s, beta, t0, quad=None):
    """Riemann-Liouville integral of order beta > 0 of a smooth function f, at the times s."""
    quad = _Quad(beta - 1.0) if quad is None else quad
    out = np.zeros(len(s))
    for i, si in enumerate(s):
        X = si - t0
        if X > 0.0:
            x, w = quad.nodes(X)
            out[i] = np.dot(w, f(si - x)) / gamma(beta)
    return out


def kernel_convolution(S, h, alpha, mu, nseries=80):
    """(e_mu * S)(t0 + n h), n = 0..len(S)-1, for samples S on the uniform grid: S is interpolated linearly and the
    kernel is integrated exactly on every cell through its power series (requires mu (n h)^alpha of order one)."""
    n = len(S)
    x = h * np.arange(n)
    if mu * x[-1] ** alpha > 8.0:
        raise ValueError('kernel_convolution: mu T^alpha too large for the power series')
    I0 = np.zeros(n); I1 = np.zeros(n)                        # int_0^x e_mu and int_0^x tau e_mu
    for k in range(nseries):
        e = alpha * (k + 1)
        I0[1:] += (-mu) ** k * x[1:] ** e / gamma(e + 1.0)
        I1[1:] += (-mu) ** k * x[1:] ** (e + 1.0) / (gamma(e) * (e + 1.0))
        if mu == 0.0:
            break
    m0 = np.diff(I0); m1 = np.diff(I1)
    j = np.arange(n - 1)
    wa = ((j + 1) * h * m0 - m1) / h                          # weight of S(t_n - j h)
    wb = (m1 - j * h * m0) / h                                # weight of S(t_n - (j + 1) h)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = np.dot(wa[:i], S[i::-1][:i]) + np.dot(wb[:i], S[i - 1::-1][:i])
    return out


def current_cutoff_coefficient(alpha, pars, taubar=1.0, window=(-30.0, 30.0), h=0.025, gs=2.0, shape='square', nphi=12):
    """Coefficient B(t) of the term 1/kappa_c of the band-referenced sheet current (both valleys, Gup = 0).

    Returns (t, B, parts): t = t0 + h n; B of shape (n, 2) with the components (B_x, B_y); parts: dict with the functions
    of time from which the angular dependence is assembled (C_xx, C_yy, C_xy: the quadratic form of the tail coefficient,
    C(phi) = cos^2 C_yy + sin^2 C_xx - 2 sin cos C_xy; empty for alpha = 1)."""
    t0, tf = window
    args = dirac._pulse_args(pars)
    G1, G2, _ = dirac.rates(pars)
    if pars.get('Gup', 0.0) != 0.0:
        raise ValueError('current_cutoff_coefficient assumes Gup = 0')
    c = taubar ** (1.0 - alpha)
    n = int(round((tf - t0) / h)) + 1
    t = t0 + h * np.arange(n)
    ax = lambda s: dirac.pulse_A(np.asarray(s, dtype=float), *args)[..., 0]
    ay = lambda s: dirac.pulse_A(np.asarray(s, dtype=float), *args)[..., 1]
    dax = lambda s: dirac.pulse_dA(s, *args)[..., 0]
    day = lambda s: dirac.pulse_dA(s, *args)[..., 1]
    ddax = lambda s: pulse_d2A(s, *args)[..., 0]
    dday = lambda s: pulse_d2A(s, *args)[..., 1]
    xl, wl = roots_legendre(nphi)
    phis = []; wphi = []
    for k in range(8):                                        # w(phi) has kinks at the odd multiples of pi/4
        lo, hi = k * math.pi / 4, (k + 1) * math.pi / 4
        phis.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * xl); wphi.append(0.5 * (hi - lo) * wl)
    phis = np.concatenate(phis); wphi = np.concatenate(wphi)
    cs, sn = np.cos(phis)[:, None], np.sin(phis)[:, None]
    wsh = np.maximum(np.abs(cs), np.abs(sn)) if shape == 'square' else np.ones_like(cs)
    parts = {}
    if alpha == 1.0:
        C3 = np.zeros((len(phis), n))
        Y = (cs * (G2 * day(t) + dday(t))[None] - sn * (G2 * dax(t) + ddax(t))[None]) / 4.0
    else:
        q2 = _Quad(1.0 - alpha); g1m = gamma(1.0 - alpha); mu = c * G1
        conv = lambda S: 0.5 * kernel_convolution(S, h, alpha, mu)
        Cxx = conv(bilinear_form(ax, ax, t, alpha, t0, q2) / (2 * g1m))
        Cyy = conv(bilinear_form(ay, ay, t, alpha, t0, q2) / (2 * g1m))
        Cxy = conv(bilinear_form(ax, ay, t, alpha, t0, q2) / (2 * g1m))
        parts = dict(C_xx=Cxx, C_yy=Cyy, C_xy=Cxy)
        # a_perp a_par = (cos^2 - sin^2) a_x a_y + cos sin (a_y^2 - a_x^2)
        pxy = lambda s: ax(s) * ay(s)
        pdd = lambda s: ay(s) ** 2 - ax(s) ** 2
        E = [conv(-bilinear_form(f, g, t, alpha, t0, q2) / g1m) for f, g in ((ay, pxy), (ay, pdd), (ax, pxy), (ax, pdd))]
        C3 = cs * (cs ** 2 - sn ** 2) * E[0][None] + cs ** 2 * sn * E[1][None] - sn * (cs ** 2 - sn ** 2) * E[2][None] - sn ** 2 * cs * E[3][None]
        Dax = caputo_smooth(dax, t, alpha, t0); Day = caputo_smooth(day, t, alpha, t0)
        sing = np.zeros(n); sing[1:] = (t[1:] - t0) ** (1.0 - 2.0 * alpha) / gamma(2.0 - 2.0 * alpha)       # not evaluated at t0
        DDax = fractional_integral(ddax, t, 2.0 - 2.0 * alpha, t0) + float(dax(t0)) * sing
        DDay = fractional_integral(dday, t, 2.0 - 2.0 * alpha, t0) + float(day(t0)) * sing
        Calpha = cs ** 2 * Cyy[None] + sn ** 2 * Cxx[None] - 2.0 * cs * sn * Cxy[None]
        aperp = cs * ay(t)[None] - sn * ax(t)[None]
        Y = G2 / (4.0 * c) * (cs * Day[None] - sn * Dax[None]) + 2.0 * aperp * Calpha + (cs * DDay[None] - sn * DDax[None]) / (4.0 * c ** 2)
    pref = gs / (2.0 * math.pi ** 2)
    wq = wphi[:, None] * wsh
    Bx = pref * (wq * (2.0 * cs * C3 - sn * Y)).sum(0)
    By = pref * (wq * (2.0 * sn * C3 + cs * Y)).sum(0)
    return t, np.stack([Bx, By], axis=1), parts

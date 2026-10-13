"""Discrete Caputo operators used by the neural residual (torch, differentiable).

All routines act on the first axis (time) of a tensor F with shape (N+1, ...).
Double precision is used throughout.
"""
import math
import numpy as np
import torch
import torch.nn.functional as tF

DT = torch.float64


def l1_weights(alpha, n):
    """b_j = (j+1)^{1-a} - j^{1-a}, j=0..n-1, evaluated with expm1 for stability near a=1.
    alpha may be a python float or a 0-d torch tensor (differentiable)."""
    a = torch.as_tensor(alpha, dtype=DT)
    j = torch.arange(1, n, dtype=DT)
    bj = j ** (1 - a) * torch.expm1((1 - a) * torch.log1p(1.0 / j))
    return torch.cat([torch.ones(1, dtype=DT) + 0 * a, bj])


def l1_uniform(F, alpha, h):
    """Full-history L1 approximation on a uniform grid t_n = n h.

    Returns D[n] for n = 1..N (shape (N, ...)). Implemented as a causal 1-D convolution of the
    increments with the weights b_j (batched triangular convolution); the (N x N) history matrix
    is never materialised. Complex input is treated componentwise (the weights are real); for real
    input the arithmetic is exactly that of the real-only implementation.
    """
    if torch.is_complex(F):
        return torch.complex(l1_uniform(F.real, alpha, h), l1_uniform(F.imag, alpha, h))
    a = torch.as_tensor(alpha, dtype=DT)
    N = F.shape[0] - 1
    dF = F[1:] - F[:-1]                      # (N, ...)
    shp = dF.shape[1:]
    x = dF.reshape(N, -1).T.unsqueeze(1)     # (C, 1, N)
    b = l1_weights(a, N)                     # (N,)
    kernel = torch.flip(b, [0]).view(1, 1, N)
    y = tF.conv1d(tF.pad(x, (N - 1, 0)), kernel)   # (C,1,N): y[n] = sum_j b_j dF[n-j]
    c = h ** (-a) / torch.exp(torch.lgamma(2 - a))
    return (c * y.squeeze(1).T).reshape((N,) + shp)


def l1_nonuniform_matrix(t, alpha):
    """Dense (N x N) weight matrix W with D[n] = sum_{m=1}^{n} W[n-1,m-1] (f_m - f_{m-1}), Eq. (nonuniform L1).
    Used for graded meshes (moderate N). The mesh must be strictly increasing (ValueError otherwise).

    For m < n the difference a^{1-alpha} - (a-h)^{1-alpha}, a = t_n - t_{m-1}, h = t_m - t_{m-1}, is evaluated
    as -a^{1-alpha} expm1((1-alpha) log1p(-h/a)), which has no cancellation for h << a (strongly graded meshes);
    0 < h/a < 1 on every strictly increasing mesh. The local interval (m = n, a = h) is evaluated separately so
    that the alpha-derivative stays finite at alpha = 1. Entries with m > n are exactly zero."""
    a = torch.as_tensor(alpha, dtype=DT)
    t = torch.as_tensor(t, dtype=DT)
    N = t.numel() - 1
    if N < 1 or not bool(torch.all(t[1:] > t[:-1])):
        raise ValueError('l1_nonuniform_matrix: the mesh must be strictly increasing and have at least two nodes')
    tn = t[1:].view(N, 1)
    tm = t[1:].view(1, N)
    tm1 = t[:-1].view(1, N)
    hm = (tm - tm1)
    mask = (tm <= tn + 0.0)  # m <= n
    diag = torch.eye(N, dtype=torch.bool)
    lower = mask & ~diag     # m < n: the only entries on which the off-diagonal formula is used
    # placeholders (A = 1, ratio = 1/2) on the unused entries keep every intermediate and its alpha-derivative finite
    A = torch.where(lower, tn - tm1, torch.ones_like(hm))
    ratio = torch.where(lower, hm / A, torch.full_like(A, 0.5))
    Woff = -(A ** (1 - a)) * torch.expm1((1 - a) * torch.log1p(-ratio)) / hm
    Wdiag = hm ** (-a)
    W = torch.where(diag, Wdiag, Woff) / torch.exp(torch.lgamma(2 - a))
    return torch.where(mask, W, torch.zeros_like(W))


def l1_nonuniform(F, W):
    N = F.shape[0] - 1
    dF = (F[1:] - F[:-1]).reshape(N, -1)
    if torch.is_complex(dF) and not torch.is_complex(W):
        W = W.to(dF.dtype)
    return (W @ dF).reshape((N,) + F.shape[1:])


def dcoef_dalpha(alpha, h):
    """Analytical d c_alpha / d alpha with c_alpha = h^{-a}/Gamma(2-a)."""
    from scipy.special import gamma, digamma
    c = h ** (-alpha) / gamma(2 - alpha)
    return c * (-math.log(h) + digamma(2 - alpha))


def dweights_dalpha(alpha, n):
    j = np.arange(n, dtype=float)
    out = -(j + 1) ** (1 - alpha) * np.log(j + 1)
    out[1:] += j[1:] ** (1 - alpha) * np.log(j[1:])
    return out


def dl1_nonuniform_matrix_dalpha(t, alpha):
    """Closed-form d W / d alpha of the nonuniform L1 weight matrix (numpy, float alpha):
        d w_nm / d alpha = psi(2-alpha) w_nm + B_nm / (Gamma(2-alpha) h_m),
        B_nm = d/dalpha [ a^{1-alpha} - (a-h)^{1-alpha} ] = a^{1-alpha} [ log(a) expm1(c) + exp(c) L ]   (m < n),
        L = log1p(-h/a), c = (1-alpha) L,  a = t_n - t_{m-1}, h = t_m - t_{m-1};   B_nn = -h^{1-alpha} log h.
    Both terms of B_nm are O(h/a), so the expression has no cancellation for h << a (graded meshes).
    Counterpart of Eq. (4.25)/(4.27) for the nonuniform operator; used for verification (experiment E2)."""
    from scipy.special import gamma, digamma
    t = np.asarray(t, dtype=float)
    N = len(t) - 1
    tn = t[1:][:, None]; tm = t[1:][None, :]; tm1 = t[:-1][None, :]
    hm = tm - tm1
    low = tm < tn; dia = np.eye(N, dtype=bool)
    A = np.where(low, tn - tm1, 1.0)
    L = np.log1p(-np.where(low, hm / A, 0.5)); c = (1 - alpha) * L
    W = l1_nonuniform_matrix(torch.as_tensor(t), float(alpha)).numpy()
    br = np.where(low, A ** (1 - alpha) * (np.log(A) * np.expm1(c) + np.exp(c) * L), 0.0)
    br = np.where(dia, -hm ** (1 - alpha) * np.log(hm), br)
    return digamma(2 - alpha) * W + np.where(low | dia, br / (gamma(2 - alpha) * hm), 0.0)


# ----------------------------------------------------------------------------------------------
# Sum-of-exponentials (SOE) compression of K(s) = s^{-a}/Gamma(1-a) on [delta, T]
# ----------------------------------------------------------------------------------------------

def soe_design(alpha, delta, T, tol=1e-10, criterion='relative'):
    """Step and truncation limits (dy, ylo, yhi) of the trapezoidal rule in y for
        K(s) = s^{-a}/Gamma(1-a) = sin(pi a)/pi * int_R exp(-s e^y + a y) dy,     delta <= s <= T.

    criterion='relative' (default): the three error contributions are bounded RELATIVE to K(s), uniformly on
    [delta, T], with the budget tol = 0.6 tol (trapezoidal rule) + 0.2 tol (lower tail) + 0.2 tol (upper tail):
      * trapezoidal rule (Poisson summation; independent of s):  2 |Gamma(a + 2 pi i/dy)| / Gamma(a) <= 0.6 tol;
      * lower tail  y < ylo :  int exp(a y) dy = e^{a ylo}/a  <=  0.2 tol K(T) pi/sin(pi a), i.e.
            ylo = log(0.2 tol Gamma(1+a)) / a - log T          (the worst point is s = T);
      * upper tail  y > yhi :  Gamma(a, delta e^{yhi}) / Gamma(a) <= 0.2 tol   (the worst point is s = delta).
    criterion='absolute' reproduces the construction of the first version of this package, whose lower limit
    bounds the ABSOLUTE error by tol/10 and therefore loses the relative target when Gamma(1-a) T^a > 1.
    Returns (dy, ylo, yhi)."""
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError('soe_design requires 0 < alpha < 1')
    if not (0.0 < delta < T):
        raise ValueError('soe_design requires 0 < delta < T')
    pref = math.sin(math.pi * a) / math.pi
    if criterion == 'absolute':
        dy = math.pi ** 2 / math.log(10.0 / tol)
        ylo = math.log(tol * a / (pref * 10.0)) / a
        yhi = math.log((math.log(10.0 / tol) + 10.0) / delta) + 0.5
        return dy, ylo, yhi
    if criterion != 'relative':
        raise ValueError("criterion must be 'relative' or 'absolute'")
    from scipy.special import gammainccinv, loggamma
    from scipy.optimize import brentq
    lg_a = float(loggamma(a).real)

    def trap_err_log(tau):          # log( 2 |Gamma(a + i tau)| / Gamma(a) ), decreasing in tau > 0
        return math.log(2.0) + float(loggamma(complex(a, tau)).real) - lg_a
    target = math.log(0.6 * tol)
    hi = 4.0
    while trap_err_log(hi) > target:
        hi *= 2.0
    lo = 1e-3
    tau = brentq(lambda x: trap_err_log(x) - target, lo, hi, xtol=1e-12) if trap_err_log(lo) > target else lo
    dy = 2.0 * math.pi / tau
    ylo = math.log(0.2 * tol * math.gamma(1.0 + a)) / a - math.log(T)
    yhi = math.log(float(gammainccinv(a, 0.2 * tol)) / delta)
    return dy, ylo, yhi


def soe_mode_estimate(alpha, delta, T, tol=1e-10):
    """Leading-order estimate of the number of modes of soe_kernel(..., criterion='relative'):
        M ~ (L / pi^2) [ log(T/delta) + L / a + log L + c_a ],   L = log(1/tol),
    i.e. QUADRATIC in log(1/tol) and linear in log(T/delta). Returned for documentation; the actual M is
    len(omega)."""
    dy, ylo, yhi = soe_design(alpha, delta, T, tol, 'relative')
    return (yhi - ylo) / dy + 1.0


def soe_kernel(alpha, delta, T, tol=1e-10, step=None, criterion='relative', n_check=4000):
    """SOE approximation  K(s) ~ sum_l omega_l exp(-lam_l s)  on [delta, T] by the truncated trapezoidal rule
    in y applied to the representation given in soe_design.

    tol is the target RELATIVE sup-error on [delta, T] (criterion='relative', default; see soe_design for the
    error budget). Returns (omega, lam, measured_abs_err, measured_rel_err): the realised errors are measured
    on n_check logarithmically spaced points of [delta, T] including both end points and must be stored by the
    caller next to the results; tol itself is only the design target. criterion='absolute' gives the legacy
    construction (absolute lower-tail criterion, independent of T)."""
    a = float(alpha)
    pref = math.sin(math.pi * a) / math.pi
    dy, ylo, yhi = soe_design(a, delta, T, tol, criterion)
    if step is not None:
        dy = float(step)
    if criterion == 'absolute':
        y = np.arange(ylo, yhi + dy, dy)
    else:
        y = ylo + dy * np.arange(int(math.ceil((yhi - ylo) / dy)) + 1)
    lam = np.exp(y)
    omega = pref * dy * np.exp(a * y)
    s = np.logspace(math.log10(delta), math.log10(T), n_check)
    s[0] = delta; s[-1] = T
    from scipy.special import gamma
    K = s ** (-a) / gamma(1 - a)
    Ka = (omega[None, :] * np.exp(-np.outer(s, lam))).sum(1)
    err = np.abs(K - Ka)
    return omega, lam, float(err.max()), float((err / K).max())


def l1_soe_uniform(F, alpha, h, omega, lam):
    """Fast L1 operator, Eqs. (fast recurrence): local interval in L1 form, older history in SOE form.
    Differentiable through F for fixed (omega, lam); alpha is used as a constant (no gradient with respect to
    alpha, because omega and lam depend on it). Real or complex F. Returns D[n], n=1..N."""
    a = float(alpha)
    N = F.shape[0] - 1
    om = torch.as_tensor(omega, dtype=DT)
    la = torch.as_tensor(lam, dtype=DT)
    q = torch.exp(-la * h)
    g = q * (-torch.expm1(-la * h)) / (la * h)          # q(1-q)/(lambda h)
    shp = F.shape[1:]
    Fl = F.reshape(N + 1, -1)
    Hm = torch.zeros(la.numel(), Fl.shape[1], dtype=DT)
    c0 = 1.0 / (math.gamma(2 - a) * h ** a)
    out = []
    for n in range(1, N + 1):
        if n >= 2:
            Hm = q[:, None] * Hm + g[:, None] * (Fl[n - 1] - Fl[n - 2])[None, :]
        out.append(c0 * (Fl[n] - Fl[n - 1]) + (om[:, None] * Hm).sum(0))
    return torch.stack(out).reshape((N,) + shp)

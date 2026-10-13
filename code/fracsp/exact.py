"""Exact solutions used as references."""
import math
import numpy as np
import mpmath as mp


def ml_complex(z, a, b=1.0, dps=30):
    """Series evaluation of E_{a,b}(z) for complex z with adaptive precision."""
    extra = int(abs(complex(z)) ** (1.0 / a) / 2.3) + 20
    with mp.workdps(dps + extra):
        z = mp.mpc(complex(z)); a_ = mp.mpf(a); b_ = mp.mpf(b)
        s = mp.mpc(0); m = 0
        while True:
            term = z ** m / mp.gamma(a_ * m + b_)
            s += term
            if m > 10 and abs(term) < mp.mpf(10) ** (-dps) * max(1, abs(s)):
                break
            m += 1
        return complex(s)


def ml_complex_many(zs, a, b=1.0, dps=30):
    """E_{a,b}(z) for an array of complex arguments: same power series and precision rule as ml_complex, but the
    coefficients 1/Gamma(a m + b) are computed once and every value is a Horner sum. Intended for the many
    arguments lambda t_n^a of a time mesh (the cost of ml_complex is dominated by the gamma functions).
    Returns a complex ndarray of the shape of zs."""
    zs = np.asarray(zs, dtype=complex)
    flat = zs.reshape(-1)
    out = np.empty(flat.shape, dtype=complex)
    if flat.size == 0:
        return out.reshape(zs.shape)
    absz = np.abs(flat)
    zmax = float(absz.max())
    lg = lambda x: math.lgamma(x)
    ln_tol = -(dps + 4) * math.log(10.0)

    def n_terms(az):
        # smallest M > 10 such that all terms m >= M are below 10^-(dps+4) (the terms decrease once
        # a*log(a m + b) > log|z|, which holds beyond the largest term)
        if az == 0.0:
            return 11
        m = 11
        while True:
            if m * math.log(az) - lg(a * m + b) < ln_tol and a * math.log(a * m + b) > math.log(az) + 0.1:
                return m
            m += max(1, m // 20)
    Mmax = n_terms(zmax)
    with mp.workdps(dps + int(zmax ** (1.0 / a) / 2.3) + 20):
        a_ = mp.mpf(a); b_ = mp.mpf(b)
        coef = [mp.rgamma(a_ * m + b_) for m in range(Mmax + 1)]
    real_arg = bool(np.all(flat.imag == 0.0))
    for i, z in enumerate(flat):
        az = float(absz[i])
        M = min(Mmax, n_terms(az))
        with mp.workdps(dps + int(az ** (1.0 / a) / 2.3) + 20):
            zz = mp.mpf(z.real) if real_arg else mp.mpc(z)
            s = +coef[M]
            for m in range(M - 1, -1, -1):
                s = s * zz + coef[m]
            out[i] = complex(s)
    return out.reshape(zs.shape)


def bloch_exact(A, b, r0, t, alpha):
    """Exact solution of D^alpha r = A r + b, r(0)=r0, for constant diagonalisable A:
    r(t) = r* + V E_alpha(Lambda t^alpha) V^{-1} (r0 - r*),  r* = -A^{-1} b."""
    rs = -np.linalg.solve(A, b)
    lam, V = np.linalg.eig(A)
    c = np.linalg.solve(V, (r0 - rs).astype(complex))
    out = np.zeros((len(t), len(r0)))
    for i, ti in enumerate(t):
        e = np.array([ml_complex(l * ti ** alpha, alpha) for l in lam])
        out[i] = (rs + V @ (e * c)).real
    return out


def bloch_exact_many(A, b, R0, t, alpha, dps=30):
    """Same closed form as bloch_exact for K initial states R0 (K, m) at all times t at once; the Mittag-Leffler
    values are shared between the states and evaluated with ml_complex_many (complex-conjugate eigenvalues are
    evaluated once). Returns an array (len(t), K, m)."""
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float); R0 = np.asarray(R0, dtype=float)
    t = np.asarray(t, dtype=float)
    rs = -np.linalg.solve(A, b)
    lam, V = np.linalg.eig(A)
    C = np.linalg.solve(V, (R0 - rs).T.astype(complex))            # (m, K)
    E = np.empty((len(t), len(lam)), dtype=complex)
    done = {}
    for j, l in enumerate(lam):
        key = (round(l.real, 12), round(abs(l.imag), 12))
        if key in done and abs(l - np.conj(lam[done[key]])) < 1e-9 * max(1.0, abs(l)):
            E[:, j] = np.conj(E[:, done[key]])
        elif key in done and abs(l - lam[done[key]]) < 1e-9 * max(1.0, abs(l)):
            E[:, j] = E[:, done[key]]
        else:
            E[:, j] = ml_complex_many(l * t ** alpha, alpha, 1.0, dps); done[key] = j
    out = np.einsum('ij,tj,jk->tki', V, E, C)
    return rs[None, None, :] + out.real

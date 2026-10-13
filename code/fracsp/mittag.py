"""Mittag-Leffler function E_{a,b}(z) for real z <= 0 via mpmath.

Two independent evaluations are provided:
  * ml_series : power series with adaptive working precision (exact up to the chosen dps)
  * ml_laplace: numerical inversion of the Laplace transform s^{a-b}/(s^a - z) (Talbot)
They are cross-checked in tests/test_all.py (test_mittag_cross_check). Complex arguments, which are needed
for the matrix solution with oscillatory eigenvalues, are handled by fracsp.exact.ml_complex and
fracsp.exact.ml_complex_many (tests: test_ml_complex_closed_forms, test_ml_complex_many_matches_series).
"""
import mpmath as mp
import numpy as np


def ml_series(z, a, b=1.0, tol=1e-30):
    z = mp.mpf(z)
    # working precision: series terms grow like |z|^m / Gamma(a m + b); cancellation ~ exp(|z|^(1/a))
    extra = int(float(abs(z)) ** (1.0 / float(a)) / 2.3) + 30
    with mp.workdps(30 + extra):
        a = mp.mpf(a); b = mp.mpf(b)   # a*m+b must be formed in multiprecision (cancellation)
        s = mp.mpf(0)
        m = 0
        while True:
            term = z ** m / mp.gamma(a * m + b)
            s += term
            if m > 10 and abs(term) < tol * max(1, abs(s)):
                break
            m += 1
            if m > 20000:
                raise RuntimeError("ML series did not converge")
        return s


def ml_laplace(z, a, b=1.0, t=1.0):
    """E_{a,b}(z) = t^{1-b} L^{-1}[ s^{a-b}/(s^a - z/t^a) ](t) evaluated at t=1."""
    z = mp.mpf(z)
    with mp.workdps(40):
        a = mp.mpf(a); b = mp.mpf(b)
        F = lambda s: s ** (a - b) / (s ** a - z)
        return mp.invertlaplace(F, 1, method='talbot')


def ml_vec(z_array, a, b=1.0):
    return np.array([float(ml_series(z, a, b)) for z in np.atleast_1d(z_array)])

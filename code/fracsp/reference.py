"""Independent implicit L1 reference solvers (numpy).

These solvers do NOT reuse the batched torch history routine of the neural residual: history sums
are evaluated by explicit per-step loops with weights computed from closed-form expressions.

Problem class (affine, batched over K independent trajectories):
    ^C D^alpha_{t0,t} x(t) = A(t) x(t) + b(t),   x(t0) = x0,   x in R^m (or C^m)
The prefactor tau_c^{1-alpha} of the manuscript must be included in A and b by the caller.

Generator callbacks. Every solver takes (Afun, bfun) with Afun(t) -> (K, m, m) and bfun(t) -> (K, m).
If bfun is None, Afun(t) must return the pair (A, b) (further return values are ignored), so that a
generator whose A and b share intermediate quantities is evaluated once per step.

Contents
    graded_mesh, l1_row_weights, l1_uniform_weights, l1_weight_table, l1_convex_weights
    solve_l1                  implicit L1, arbitrary mesh, full history (row / tabulated weights)
    solve_l1_soe              implicit L1, uniform mesh, SOE-compressed history
    solve_l1_soe_nonuniform   implicit L1, arbitrary mesh, SOE-compressed history
    solve_pi_trapezoid        implicit product-trapezoidal rule (second order; independent check)
    gksl_superoperator, solve_l1_gksl, density_defects, choi_matrix   (positivity of the L1 step)
    richardson, observed_order
"""
import math
import numpy as np
from scipy.special import gamma


def graded_mesh(T, N, r):
    n = np.arange(N + 1)
    return T * (n / N) ** r


def l1_row_weights(t, n, alpha):
    """w_{n,m}, m=1..n, for the nonuniform L1 formula at t_n (explicit closed form)."""
    tn = t[n]
    tm = t[1:n + 1]
    tm1 = t[0:n]
    # a^{1-alpha} - (a-h)^{1-alpha} = -a^{1-alpha} expm1((1-alpha) log1p(-h/a)):  avoids the catastrophic
    # cancellation that occurs on strongly graded meshes when h_m << t_n - t_{m-1}
    a = tn - tm1; hm = tm - tm1
    w = np.empty(n)
    w[:-1] = -(a[:-1] ** (1 - alpha)) * np.expm1((1 - alpha) * np.log1p(-hm[:-1] / a[:-1])) / hm[:-1]
    w[-1] = hm[-1] ** (-alpha)            # local interval (a = h): evaluated separately (finite for alpha = 1)
    return w / gamma(2 - alpha)


def l1_uniform_weights(alpha, N, h):
    """Tabulated uniform-mesh weights: w[k] multiplies the increment k steps back, k = 0..N-1,
    w[k] = b_k h^{-alpha}/Gamma(2-alpha), b_k = (k+1)^{1-alpha} - k^{1-alpha} (expm1/log1p form)."""
    j = np.arange(1, N, dtype=float)
    b = np.r_[1.0, j ** (1 - alpha) * np.expm1((1 - alpha) * np.log1p(1.0 / j))]
    return b * h ** (-alpha) / gamma(2 - alpha)


def l1_weight_table(t, alpha):
    """Dense table W[n-1, m-1] = w_{n,m} (m <= n, zero above the diagonal) of the nonuniform L1 formula."""
    t = np.asarray(t, dtype=float); N = len(t) - 1
    W = np.zeros((N, N))
    for n in range(1, N + 1):
        W[n - 1, :n] = l1_row_weights(t, n, alpha)
    return W


def l1_convex_weights(t, n, alpha):
    """Coefficients theta_{n,m}, m = 0..n-1, of the implicit L1 step written as
        (I - A_n / w_nn) x^n = sum_{m=0}^{n-1} theta_{n,m} x^m + b_n / w_nn ,
    theta_{n,0} = w_{n,1}/w_nn,  theta_{n,m} = (w_{n,m+1} - w_{n,m})/w_nn  (1 <= m <= n-1).
    Because w_{n,m} is the mean of (t_n - s)^{-alpha}/Gamma(1-alpha) over [t_{m-1}, t_m], the weights are positive
    and increasing in m on EVERY mesh; hence theta >= 0 and sum_m theta_{n,m} = 1 (convex combination).
    Returns (theta, w_nn)."""
    w = l1_row_weights(np.asarray(t, dtype=float), n, alpha)
    theta = np.empty(n)
    theta[0] = w[0]
    theta[1:] = np.diff(w)
    return theta / w[-1], w[-1]


def _generator(Afun, bfun):
    if bfun is None:
        def gen(tn):
            out = Afun(tn)
            return out[0], out[1]
        return gen
    return lambda tn: (Afun(tn), bfun(tn))


def soe_delta_for_mesh(t):
    """Smallest argument at which the history kernel is evaluated by an SOE solver on the mesh t:
    the history of step n covers [0, t_{n-1}], i.e. kernel arguments in [t_n - t_{n-1}, t_n], n >= 2."""
    h = np.diff(np.asarray(t, dtype=float))
    return float(h[1:].min()) if len(h) > 1 else float(h[0])


def solve_l1(Afun, bfun, x0, t, alpha, store_every=1, weights='row', callback=None):
    """Implicit (backward) L1 scheme on an arbitrary mesh t (full history).

    Afun(tn) -> (K, m, m) array, bfun(tn) -> (K, m) array (or bfun=None and Afun -> (A, b)). Returns
    (t_out, X_out) with X_out of shape (len(t_out), K, m). For alpha == 1 the scheme reduces to implicit Euler.

    weights: 'row'     -- the n weights of row n are recomputed at every step (default; valid on every mesh;
                          one power, log1p and expm1 per history entry);
             'uniform' -- uniform mesh only: w_{n,m} = c b_{n-m} is tabulated once (same scheme, the history
                          sum then costs one dot product per step);
             'table'   -- the full lower-triangular table is precomputed (N <= 8192; 8 N^2 bytes).
    callback(n, tn, x) is called after every step if given (monitoring; x must not be modified).
    """
    x0 = np.asarray(x0)
    K, m = x0.shape
    t = np.asarray(t, dtype=float)
    N = len(t) - 1
    gen = _generator(Afun, bfun)
    dX = np.zeros((N, K, m), dtype=x0.dtype)
    x = x0.copy()
    out_t, out_x = [t[0]], [x0.copy()]
    I = np.eye(m)
    wu = wrev = Wtab = None
    if weights == 'uniform':
        h = (t[-1] - t[0]) / N
        if np.abs(np.diff(t) - h).max() > 1e-12 * abs(h):
            raise ValueError("weights='uniform' requires a uniform mesh")
        wu = l1_uniform_weights(alpha, N, h); wrev = wu[::-1].copy()
    elif weights == 'table':
        if N > 8192:
            raise ValueError("weights='table' is limited to N <= 8192")
        Wtab = l1_weight_table(t, alpha)
    elif weights != 'row':
        raise ValueError("weights must be 'row', 'uniform' or 'table'")
    for n in range(1, N + 1):
        if wu is not None:
            wnn = wu[0]
            hist = np.tensordot(wrev[N - n:N - 1], dX[:n - 1], axes=(0, 0)) if n > 1 else 0.0
        else:
            w = Wtab[n - 1, :n] if Wtab is not None else l1_row_weights(t, n, alpha)            # (n,)
            wnn = w[-1]
            hist = np.tensordot(w[:-1], dX[:n - 1], axes=(0, 0)) if n > 1 else 0.0
        A, b = gen(t[n])
        M = wnn * I[None] - A
        rhs = b + wnn * x - hist
        xn = np.linalg.solve(M, rhs[..., None])[..., 0]
        dX[n - 1] = xn - x
        x = xn
        if callback is not None:
            callback(n, t[n], x)
        if n % store_every == 0 or n == N:
            out_t.append(t[n]); out_x.append(x.copy())
    return np.array(out_t), np.array(out_x)


def solve_l1_soe(Afun, bfun, x0, h, N, alpha, omega, lam, store_every=1, callback=None):
    """Implicit L1 scheme on the uniform grid t_n = n h with SOE-compressed history.

    The local interval (t_{n-1}, t_n) is treated exactly in L1 form; the history t in (0, t_{n-1})
    is represented by M exponential modes (recurrence of the fast L1 operator). Memory: O(M K m)
    for the history, independent of N. The kernel must be accurate on [h, N h] (operators.soe_kernel).
    """
    x0 = np.asarray(x0)
    K, m = x0.shape
    gen = _generator(Afun, bfun)
    q = np.exp(-lam * h)[:, None, None]
    g = (np.exp(-lam * h) * (-np.expm1(-lam * h)) / (lam * h))[:, None, None]
    Hm = np.zeros((len(lam), K, m), dtype=x0.dtype)
    tmp = np.empty_like(Hm)
    c0 = 1.0 / (gamma(2 - alpha) * h ** alpha)
    x = x0.copy(); xprev = None
    I = np.eye(m)
    out_t, out_x = [0.0], [x0.copy()]
    for n in range(1, N + 1):
        if n >= 2:
            Hm *= q                                         # Hm = q Hm + g (x^{n-1} - x^{n-2})
            np.multiply(g, (x - xprev)[None], out=tmp)
            Hm += tmp
        hist = np.tensordot(omega, Hm, axes=(0, 0))
        tn = n * h
        A, b = gen(tn)
        M = c0 * I[None] - A
        rhs = b + c0 * x - hist
        xn = np.linalg.solve(M, rhs[..., None])[..., 0]
        xprev, x = x, xn
        if callback is not None:
            callback(n, tn, x)
        if n % store_every == 0 or n == N:
            out_t.append(tn); out_x.append(x.copy())
    return np.array(out_t), np.array(out_x)


def solve_l1_soe_nonuniform(Afun, bfun, x0, t, alpha, omega, lam, store_every=1, callback=None):
    """Implicit L1 scheme on an ARBITRARY mesh with SOE-compressed history.

    With h_n = t_n - t_{n-1} and H_{l,n} = int_0^{t_{n-1}} exp(-lam_l (t_n - s)) x'(s) ds for the piecewise-linear
    interpolant,
        H_{l,n} = exp(-lam_l h_n) [ H_{l,n-1} + phi(lam_l h_{n-1}) (x^{n-1} - x^{n-2}) ],  phi(z) = (1 - e^{-z})/z,
        H_{l,1} = 0,   D_h x^n = h_n^{-alpha}/Gamma(2-alpha) (x^n - x^{n-1}) + sum_l omega_l H_{l,n},
    which reduces to the uniform recurrence for h_n = h. The kernel (omega, lam) must be accurate on
    [soe_delta_for_mesh(t), t_N - t_0]. Cost per step: 2 M exponentials and O(M K m) updates."""
    x0 = np.asarray(x0)
    K, m = x0.shape
    t = np.asarray(t, dtype=float); N = len(t) - 1
    gen = _generator(Afun, bfun)
    hs = np.diff(t)
    Hm = np.zeros((len(lam), K, m), dtype=x0.dtype)
    tmp = np.empty_like(Hm)
    cg = 1.0 / gamma(2 - alpha)
    x = x0.copy(); xprev = None
    I = np.eye(m)
    out_t, out_x = [t[0]], [x0.copy()]
    for n in range(1, N + 1):
        hn = hs[n - 1]
        if n >= 2:
            z = lam * hs[n - 2]
            phi = (-np.expm1(-z) / z)[:, None, None]
            np.multiply(phi, (x - xprev)[None], out=tmp)
            Hm += tmp
            Hm *= np.exp(-lam * hn)[:, None, None]
        hist = np.tensordot(omega, Hm, axes=(0, 0))
        c0 = cg * hn ** (-alpha)
        A, b = gen(t[n])
        M = c0 * I[None] - A
        rhs = b + c0 * x - hist
        xn = np.linalg.solve(M, rhs[..., None])[..., 0]
        xprev, x = x, xn
        if callback is not None:
            callback(n, t[n], x)
        if n % store_every == 0 or n == N:
            out_t.append(t[n]); out_x.append(x.copy())
    return np.array(out_t), np.array(out_x)


def solve_scalar_l1(lmbda, t, alpha, u0=1.0):
    """Scalar relaxation D^alpha u = -lambda u (full history, arbitrary mesh)."""
    A = lambda tn: np.full((1, 1, 1), -lmbda)
    b = lambda tn: np.zeros((1, 1))
    tt, X = solve_l1(A, b, np.array([[u0]]), t, alpha)
    return tt, X[:, 0, 0]


# ----------------------------------------------------------------------------------------------
# Independent second-order scheme (not an L1 scheme): implicit product-trapezoidal rule
# ----------------------------------------------------------------------------------------------
_PI_WEIGHTS = {}


def pi_trapezoid_weights(alpha, N):
    """Weights of the product-trapezoidal rule for the Riemann-Liouville integral of order alpha on a uniform grid
    (Diethelm-Ford-Freed):  c[k] = (k+1)^{a+1} + (k-1)^{a+1} - 2 k^{a+1} (k >= 1), c[0] = 1, and
    a0[n] = n^{a+1} - (n - a)(n+1)^a. Both are second differences that cancel to relative order k^-2 in double
    precision; they are evaluated with 40-digit arithmetic (mpmath) and cached."""
    key = (float(alpha), int(N))
    if key not in _PI_WEIGHTS:
        import mpmath as mp
        with mp.workdps(40):
            a = mp.mpf(float(alpha))
            pw = [mp.mpf(k) ** (a + 1) for k in range(N + 2)]
            c = np.array([1.0] + [float(pw[k + 1] + pw[k - 1] - 2 * pw[k]) for k in range(1, N + 1)])
            a0 = np.array([float(pw[n] - (n - a) * mp.mpf(n + 1) ** a) for n in range(N)])
        _PI_WEIGHTS[key] = (c, a0)
    return _PI_WEIGHTS[key]


def solve_pi_trapezoid(Afun, bfun, x0, t0, T, N, alpha, store_every=1, callback=None):
    """Implicit product-trapezoidal rule for the Volterra form of D^alpha x = A(t) x + b(t) on the uniform grid
    t_n = t0 + n h, h = T/N (full history):
        x_{n+1} = x_0 + h^a/Gamma(a+2) [ f_{n+1} + a0[n] f_0 + sum_{j=1}^{n} c[n-j+1] f_j ],   f_j = A_j x_j + b_j.
    Second order for smooth solutions (for alpha = 1: the trapezoidal rule). The scheme is unrelated to the L1
    formula and serves as an independent check of the L1 reference values; it is NOT positivity preserving in
    general. Returns (t_out, X_out)."""
    x0 = np.asarray(x0)
    K, m = x0.shape
    gen = _generator(Afun, bfun)
    h = T / N
    c, a0 = pi_trapezoid_weights(alpha, N)
    cc = h ** alpha / gamma(alpha + 2)
    F = np.zeros((N + 1, K, m), dtype=x0.dtype)
    A, b = gen(t0)
    F[0] = np.einsum('kij,kj->ki', A, x0) + b
    I = np.eye(m); x = x0.copy()
    out_t, out_x = [t0], [x0.copy()]
    for n in range(N):                              # computes x_{n+1}
        hist = a0[n] * F[0]
        if n >= 1:
            hist = hist + np.tensordot(c[1:n + 1][::-1], F[1:n + 1], axes=(0, 0))
        tn1 = t0 + (n + 1) * h
        A, b = gen(tn1)
        x = np.linalg.solve(I[None] - cc * A, (x0 + cc * (hist + b))[..., None])[..., 0]
        F[n + 1] = np.einsum('kij,kj->ki', A, x) + b
        if callback is not None:
            callback(n + 1, tn1, x)
        if (n + 1) % store_every == 0 or n + 1 == N:
            out_t.append(tn1); out_x.append(x.copy())
    return np.array(out_t), np.array(out_x)


# ----------------------------------------------------------------------------------------------
# Matrix-valued (GKSL) form: positivity of the implicit L1 step
# ----------------------------------------------------------------------------------------------

def gksl_superoperator(H, Ls):
    """Matrix of  rho -> -i[H, rho] + sum_k (L_k rho L_k^+ - 1/2 {L_k^+ L_k, rho})  acting on vec(rho)
    (row-major vectorisation, vec(A rho B) = (A kron B^T) vec(rho))."""
    H = np.asarray(H, dtype=complex); d = H.shape[0]; I = np.eye(d)
    S = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for L in Ls:
        L = np.asarray(L, dtype=complex); LdL = L.conj().T @ L
        S = S + np.kron(L, L.conj()) - 0.5 * (np.kron(LdL, I) + np.kron(I, LdL.T))
    return S


def solve_l1_gksl(Sfun, rho0, t, alpha, store_every=1, soe=None):
    """Implicit L1 scheme for  D^alpha rho = L_t rho  with a (time-dependent) GKSL generator given as the
    d^2 x d^2 superoperator matrix Sfun(t) (see gksl_superoperator; any prefactor tau_c^{1-alpha} included).
    rho0: (B, d, d) batch of initial matrices. Returns (t_out, rho) with rho of shape (len(t_out), B, d, d).

    Each step is  rho^n = (I - L_n/w_nn)^{-1} sum_{m<n} theta_{n,m} rho^m  with the convex weights of
    l1_convex_weights; the resolvent (I - s L)^{-1} = int_0^inf e^{-u} e^{u s L} du of a GKSL generator is
    completely positive and trace preserving for every s > 0, so the step maps density matrices to density
    matrices on every mesh and for every step size. soe=(omega, lam) selects the SOE history (any mesh)."""
    rho0 = np.asarray(rho0, dtype=complex)
    B, d, _ = rho0.shape
    x0 = rho0.reshape(B, d * d)
    Af = lambda tn: np.broadcast_to(np.asarray(Sfun(tn), dtype=complex), (B, d * d, d * d))
    bf = lambda tn: np.zeros((B, d * d), dtype=complex)
    if soe is None:
        tt, X = solve_l1(Af, bf, x0, t, alpha, store_every=store_every)
    else:
        tt, X = solve_l1_soe_nonuniform(Af, bf, x0, t, alpha, soe[0], soe[1], store_every=store_every)
    return tt, X.reshape(X.shape[0], B, d, d)


def density_defects(rho):
    """(min eigenvalue of the Hermitian part, max |tr rho - 1|, max ||rho - rho^+||_F) over a batch (..., d, d)."""
    rho = np.asarray(rho)
    Hh = 0.5 * (rho + np.conj(np.swapaxes(rho, -1, -2)))
    ev = np.linalg.eigvalsh(Hh)
    tr = np.trace(rho, axis1=-2, axis2=-1)
    herm = np.linalg.norm(rho - np.conj(np.swapaxes(rho, -1, -2)), axis=(-2, -1))
    return float(ev.min()), float(np.abs(tr - 1).max()), float(herm.max())


def choi_matrix(Phi, d):
    """Choi matrix sum_ij E_ij (x) Phi(E_ij) of a linear map given as a d^2 x d^2 matrix acting on the row-major
    vec(rho). The map is completely positive iff its Choi matrix is positive semidefinite."""
    return np.asarray(Phi).reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)


# ----------------------------------------------------------------------------------------------
# Extrapolation helpers
# ----------------------------------------------------------------------------------------------

def richardson(q_coarse, q_fine, order, ratio=2.0):
    """Richardson extrapolate of two values computed with steps h and h/ratio for an error c h^order."""
    return q_fine + (q_fine - q_coarse) / (ratio ** order - 1.0)


def observed_order(q1, q2, q3, ratio=2.0):
    """Observed order from three values on meshes refined by the factor `ratio` each time (nan if undefined)."""
    d1 = np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)
    d2 = np.asarray(q2, dtype=float) - np.asarray(q3, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.abs(d1 / d2)) / math.log(ratio)

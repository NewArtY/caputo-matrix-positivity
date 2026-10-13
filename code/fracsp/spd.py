"""B5: Caputo time-fractional conformation-tensor models with a symmetric positive definite (SPD) state.

Model class (dimensionless time tau = t/T in [0, 1], Theta = T/lambda, tau_c = lambda):

    D_tau^alpha C = Theta^alpha * [ Wi (Ghat(tau) C + C Ghat(tau)^T) - R(C) ],      C(0) = C0,

    R(C) = C - I                           upper-convected Maxwell / Oldroyd-B (Hookean dumbbell)
    R(C) = (C - I) + a_G (C - I)^2         Giesekus
    R(C) = f(tr C) C - I,  f = (b-3)/(b - tr C)      FENE-P (normalised so that C_eq = I)

which is the manuscript's class D_t^alpha X = tau_c^{1-alpha} F(t, X) with
F = G C + C G^T - R(C)/lambda, G = (Wi/lambda) Ghat.

Contents
--------
* flows, constitutive models (numpy and torch versions of the right-hand side, analytic Jacobians)
* exact Mittag-Leffler solutions for the linear law (start-up of shear and of uniaxial extension)
  and a general matrix Mittag-Leffler series for an arbitrary constant velocity gradient
* implicit L1 time stepper for the matrix problem. The step of a nonlinear law has several solutions, exactly
  one of which is admissible, for every step size: fenep_step_root and giesekus_step_root return it; solve_l1
  uses Newton's iteration and accepts its result only if it is admissible (nonlinear='safeguarded', the default);
  solve_l1_newton is Newton's iteration without this test, which can return an inadmissible solution, either
  with steps halved so that tr C < b (nonlinear='newton') or with full steps (nonlinear='newton-full')
* SPD output maps with a hard initial condition (torch) and the L1-residual training loop
* metrics
"""
import math
import time

import numpy as np
import torch
import mpmath as mp

from . import operators as op                 # L1 weights and operators (torch)
from . import reference as ref                # graded mesh, L1 row weights (numpy)
from . import mittag as ml                    # Mittag-Leffler series (mpmath)
from . import layers as L                     # factor of a state, hard initial condition at the factor level
from .pinn import MLP                         # same backbone as in B2

DT = torch.float64
D = 3                                         # tensors are 3 x 3 throughout
I3 = np.eye(D)


# ----------------------------------------------------------------------------------------------
# flows (dimensionless velocity gradient of unit rate)
# ----------------------------------------------------------------------------------------------
def ghat(flow):
    G = np.zeros((D, D))
    if flow == 'shear':                # u = (gdot y, 0, 0)
        G[0, 1] = 1.0
    elif flow == 'uniaxial':           # uniaxial extension along x
        G[0, 0], G[1, 1], G[2, 2] = 1.0, -0.5, -0.5
    elif flow == 'planar':             # planar extension
        G[0, 0], G[1, 1] = 1.0, -1.0
    elif flow == 'rest':
        pass
    else:
        raise ValueError(flow)
    return G


class Model:
    """Constitutive model with a (possibly time-dependent) homogeneous velocity gradient.

    law   : 'ucm' | 'giesekus' | 'fenep'
    Wi    : Weissenberg number lambda * rate
    Theta : T / lambda (number of relaxation times covered by tau in [0, 1])
    amp   : callable tau -> scalar amplitude multiplying Ghat (default 1: start-up of steady flow)
    """

    def __init__(self, law='ucm', flow='shear', Wi=1.0, Theta=1.0, aG=0.0, b=100.0, amp=None, Ghat=None):
        self.law, self.flow = law, flow
        self.Wi, self.Theta = float(Wi), float(Theta)
        self.aG, self.b = float(aG), float(b)
        self.Gh = ghat(flow) if Ghat is None else np.asarray(Ghat, dtype=float)
        self.amp = amp

    # ---- numpy -------------------------------------------------------------------------------
    def _amp(self, tau):
        return 1.0 if self.amp is None else float(self.amp(tau))

    def relax(self, C):
        E = C - I3
        if self.law == 'ucm':
            return E
        if self.law == 'giesekus':
            return E + self.aG * E @ E
        if self.law == 'fenep':
            f = (self.b - D) / (self.b - np.trace(C))
            return f * C - I3
        raise ValueError(self.law)

    def F(self, tau, C, alpha):
        """Right-hand side of the dimensionless Caputo equation (includes Theta^alpha)."""
        G = self.Wi * self._amp(tau) * self.Gh
        return self.Theta ** alpha * (G @ C + C @ G.T - self.relax(C))

    def dF(self, tau, C, alpha):
        """Jacobian d vec(F) / d vec(C) (row-major vec, 9 x 9)."""
        G = self.Wi * self._amp(tau) * self.Gh
        J = np.kron(G, I3) + np.kron(I3, G)
        E = C - I3
        if self.law == 'ucm':
            J = J - np.eye(D * D)
        elif self.law == 'giesekus':
            J = J - np.eye(D * D) - self.aG * (np.kron(I3, E.T) + np.kron(E, I3))
        elif self.law == 'fenep':
            f = (self.b - D) / (self.b - np.trace(C))
            fp = (self.b - D) / (self.b - np.trace(C)) ** 2
            J = J - f * np.eye(D * D) - fp * np.outer(C.reshape(-1), I3.reshape(-1))
        return self.Theta ** alpha * J

    # ---- torch (batched over time; parameters may be tensors) ---------------------------------
    def F_torch(self, tau, C, alpha, Wi=None, Theta=None):
        Wi = self.Wi if Wi is None else Wi
        Theta = self.Theta if Theta is None else Theta
        Gh = torch.as_tensor(self.Gh, dtype=DT)
        if self.amp is None:
            G = Wi * Gh
            conv = G @ C + C @ G.T
        else:
            a = torch.as_tensor([self._amp(float(s)) for s in tau], dtype=DT)
            G = Wi * a[:, None, None] * Gh
            conv = G @ C + C @ G.transpose(-1, -2)
        I = torch.eye(D, dtype=DT)
        E = C - I
        if self.law == 'ucm':
            R = E
        elif self.law == 'giesekus':
            R = E + self.aG * E @ E
        elif self.law == 'fenep':
            tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
            R = ((self.b - D) / (self.b - tr))[:, None, None] * C - I
        else:
            raise ValueError(self.law)
        Th = torch.as_tensor(Theta, dtype=DT)
        return Th ** alpha * (conv - R)


# ----------------------------------------------------------------------------------------------
# exact solutions (linear law, constant velocity gradient)
# ----------------------------------------------------------------------------------------------
def _E(z, a, b=1.0):
    return float(ml.ml_series(z, a, b))


def exact_ucm(flow, Wi, Theta, alpha, tau, C0=None):
    """Closed-form Mittag-Leffler solution for start-up from equilibrium (C0 = I).

    shear   : C12 = Wi (1 - E_a(-z)),  C11 = 1 + 2 Wi^2 [1 - E_a(-z) - (z/a) E_{a,a}(-z)],  C22 = C33 = 1
    uniaxial: C11 = [2 Wi E_a((2Wi-1) z) - 1]/(2Wi-1),  C22 = C33 = [1 + Wi E_a(-(1+Wi) z)]/(1+Wi)
    with z = (Theta tau)^alpha = (t/lambda)^alpha.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    out = np.zeros((tau.size, D, D))
    for i, s in enumerate(tau):
        z = (Theta * s) ** alpha
        C = np.eye(D)
        if flow == 'shear':
            e = _E(-z, alpha)
            ea = _E(-z, alpha, alpha)
            C[0, 1] = C[1, 0] = Wi * (1.0 - e)
            C[0, 0] = 1.0 + 2.0 * Wi ** 2 * (1.0 - e - z / alpha * ea)
        elif flow == 'uniaxial':
            a1 = 2.0 * Wi - 1.0
            if abs(a1) < 1e-12:
                C[0, 0] = 1.0 + z / math.gamma(1.0 + alpha)
            else:
                C[0, 0] = (2.0 * Wi * _E(a1 * z, alpha) - 1.0) / a1
            c = (1.0 + Wi * _E(-(1.0 + Wi) * z, alpha)) / (1.0 + Wi)
            C[1, 1] = C[2, 2] = c
        elif flow == 'planar':
            a1 = 2.0 * Wi - 1.0
            C[0, 0] = 1.0 + z / math.gamma(1.0 + alpha) if abs(a1) < 1e-12 else (2.0 * Wi * _E(a1 * z, alpha) - 1.0) / a1
            C[1, 1] = (1.0 + 2.0 * Wi * _E(-(1.0 + 2.0 * Wi) * z, alpha)) / (1.0 + 2.0 * Wi)
        elif flow == 'rest':
            pass
        else:
            raise ValueError(flow)
        out[i] = C
    return out


def lyapunov_matrix(K):
    """Matrix of X -> K X + X K^T acting on the row-major vec(X)."""
    n = K.shape[0]
    return np.kron(K, np.eye(n)) + np.kron(np.eye(n), K)


def exact_linear_general(Ghat_, Wi, Theta, alpha, tau, C0=None, dps=40):
    """Matrix Mittag-Leffler solution for an arbitrary constant velocity gradient (mpmath series):
        c(tau) = E_a(A tau^a) c0 + tau^a E_{a,a+1}(A tau^a) b,
    A = Theta^a (Wi * Lyap(Ghat) - I), b = Theta^a vec(I). Slow; used for cross-checks only."""
    C0 = np.eye(D) if C0 is None else C0
    A = Theta ** alpha * (Wi * lyapunov_matrix(np.asarray(Ghat_, dtype=float)) - np.eye(D * D))
    bvec = Theta ** alpha * np.eye(D).reshape(-1)
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    out = np.zeros((tau.size, D, D))
    nrmA = np.abs(A).sum(1).max()
    for i, s in enumerate(tau):
        zn = nrmA * s ** alpha
        with mp.workdps(dps + int(zn ** (1.0 / alpha) / 2.3) + 10):
            a = mp.mpf(alpha)
            Z = mp.matrix(A.tolist()) * mp.mpf(s) ** a
            c0 = mp.matrix(C0.reshape(-1).tolist())
            bb = mp.matrix(bvec.tolist()) * mp.mpf(s) ** a
            P1 = c0.copy()          # Z^k c0
            P2 = bb.copy()          # Z^k b tau^a
            acc = mp.matrix([0] * (D * D))
            k = 0
            while True:
                t1 = P1 / mp.gamma(a * k + 1)
                t2 = P2 / mp.gamma(a * k + a + 1)
                acc += t1 + t2
                sz = max(abs(x) for x in (t1 + t2))
                if k > 10 and sz < mp.mpf(10) ** (-dps) * max(1, max(abs(x) for x in acc)):
                    break
                P1 = Z * P1
                P2 = Z * P2
                k += 1
                if k > 20000:
                    raise RuntimeError('matrix ML series did not converge')
            out[i] = np.array([float(x) for x in acc]).reshape(D, D)
    return out


# ----------------------------------------------------------------------------------------------
# implicit L1 stepper for matrix-valued problems
# ----------------------------------------------------------------------------------------------
def l1_weight_rows(t, alpha):
    """Lower-triangular (N, N) array W with D_h[n] = sum_{m<=n} W[n-1, m-1] (C^m - C^{m-1})."""
    N = len(t) - 1
    W = np.zeros((N, N))
    for n in range(1, N + 1):
        W[n - 1, :n] = ref.l1_row_weights(t, n, alpha)
    return W


def _sym_basis():
    """(9, 6) array whose columns are an orthonormal basis of the symmetric 3 x 3 matrices (row-major vec)."""
    B = np.zeros((D * D, D * (D + 1) // 2))
    k = 0
    for i in range(D):
        for j in range(i, D):
            E = np.zeros((D, D))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = math.sqrt(0.5)
            B[:, k] = E.reshape(-1)
            k += 1
    return B


_SYM = _sym_basis()
NONLINEAR_SOLVERS = ('safeguarded', 'newton', 'newton-full', 'root')


def _newton_failure(n, C):
    """RuntimeError for a step at which Newton's iteration of solve_l1 did not converge (nonlinear='newton' or
    'newton-full'). Attributes: step, the index n of the step; iterates, the array (n, 3, 3) with C^0 ... C^{n-1}."""
    e = RuntimeError(f'Newton did not converge at step {n}')
    e.step = n
    e.iterates = C[:n].copy()
    return e


def is_admissible(model, C):
    """True if the symmetric matrix C is positive definite and, for the FENE-P law, tr C < b."""
    if not np.isfinite(C).all():
        return False
    if not np.linalg.eigvalsh(0.5 * (C + C.T))[0] > 0.0:
        return False
    return model.law != 'fenep' or bool(np.trace(C) < model.b)


def fenep_step_root(wnn, R, G, kappa, b, f0=None, return_info=False):
    """The admissible solution of one implicit L1 step of the FENE-P law.

    Step n of the scheme for D^alpha C = kappa [G C + C G^T - (f C - I)], f = (b - 3)/(b - tr C), reads

        (wnn + kappa f) C - kappa (G C + C G^T) = R,        R = wnn sum_{m<n} q_nm C^m + kappa I,

    where R is symmetric positive definite if the earlier iterates are positive semidefinite. For a fixed value of
    the Peterlin factor f this is a Lyapunov equation. Its solution C(f) is positive definite exactly for
    f > f_c = (2 kappa s(G) - wnn)/kappa, s(G) the largest real part of the eigenvalues of G, and tr C(f) decreases
    strictly from infinity to zero on (f_c, infinity). Therefore

        g(f) = f (b - tr C(f)) - (b - 3)

    is negative as long as C(f) is not in {C > 0, tr C < b}, increases strictly afterwards and has exactly one zero
    f_* there, and C(f_*) is the only solution of the step equation that is positive definite with tr C < b, for
    every step size and every velocity gradient. All other solutions have a negative eigenvalue or tr C > b.
    The zero is bracketed and computed by Brent's method; g is given a negative value where C(f) is not admissible,
    which keeps the bracket on the admissible side of f_c without computing f_c.

    wnn : diagonal L1 weight; R : (3, 3) right-hand side; G : (3, 3) velocity gradient; kappa : prefactor of the
    relaxation term (Theta^alpha in the dimensionless form of this module); f0 : optional starting value of f.
    Returns C, or (C, info) with the Peterlin factor, the number of evaluations of g and the relative defect
    |f - (b - 3)/(b - tr C)| / f. Raises RuntimeError if R is not positive definite or no admissible root is found.
    """
    from scipy.optimize import brentq
    R = np.asarray(R, dtype=float)
    R = 0.5 * (R + R.T)
    if not (np.isfinite(R).all() and np.linalg.eigvalsh(R)[0] > 0.0):
        raise RuntimeError('FENE-P step: the right-hand side is not positive definite (inadmissible history)')
    L6 = kappa * (_SYM.T @ lyapunov_matrix(np.asarray(G, dtype=float)) @ _SYM)
    r6 = _SYM.T @ R.reshape(-1)
    I6 = np.eye(L6.shape[0])
    cache = {}

    def C_of(f):
        return (_SYM @ np.linalg.solve((wnn + kappa * f) * I6 - L6, r6)).reshape(D, D)

    def g(f):
        if f not in cache:
            try:
                C = C_of(f)
            except np.linalg.LinAlgError:
                C = None
            if C is None or not np.isfinite(C).all() or not np.linalg.eigvalsh(0.5 * (C + C.T))[0] > 0.0:
                cache[f] = -(b - D)                           # f <= f_c
            else:
                cache[f] = f * (b - np.trace(C)) - (b - D)    # negative if tr C >= b
        return cache[f]

    # bracket: g < 0 on (0, f_*) and g > 0 on (f_*, infinity); g(lo0) < 0 because tr C > 0 for an admissible C
    lo0 = (b - D) / b
    start = float(f0) if f0 is not None and f0 > lo0 else 2.0 * lo0
    fac = 1.125
    if g(start) < 0.0:
        lo = start
        while True:
            hi = lo * fac
            if not math.isfinite(hi):
                raise RuntimeError('FENE-P step: no bracket for the admissible root')
            if g(hi) >= 0.0:
                break
            lo, fac = hi, min(fac * fac, 4.0)
    else:
        hi = start
        while True:
            lo = max(hi / fac, lo0)
            if lo == lo0 or g(lo) < 0.0:
                break
            hi, fac = lo, min(fac * fac, 4.0)
    f = hi if g(hi) == 0.0 else brentq(g, lo, hi, xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=300)
    C = C_of(f)
    C = 0.5 * (C + C.T)
    tr = float(np.trace(C))
    if not (np.linalg.eigvalsh(C)[0] > 0.0 and tr < b):
        raise RuntimeError('FENE-P step: the computed root is not admissible')
    defect = abs(f - (b - D) / (b - tr)) / f
    if defect > 1e-6:
        raise RuntimeError(f'FENE-P step: defect {defect:.1e} of the scalar equation')
    # Newton refinement in the six entries of C. The scalar equation fixes f to round-off, but C(f) loses digits
    # when f is close to f_c (large Wi); a step is accepted only if it stays admissible and lowers the residual.
    t6 = _SYM.T @ I3.reshape(-1)                              # t6 . c6 = tr C

    def res6(c6):
        return ((wnn + kappa * (b - D) / (b - t6 @ c6)) * I6 - L6) @ c6 - r6
    c6 = _SYM.T @ C.reshape(-1)
    r = res6(c6)
    polish = 0
    for _ in range(4):
        tr6 = t6 @ c6
        J = (wnn + kappa * (b - D) / (b - tr6)) * I6 - L6 + kappa * (b - D) / (b - tr6) ** 2 * np.outer(c6, t6)
        try:
            c_new = c6 - np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            break
        C_new = (_SYM @ c_new).reshape(D, D)
        if not (np.isfinite(c_new).all() and t6 @ c_new < b and np.linalg.eigvalsh(0.5 * (C_new + C_new.T))[0] > 0.0):
            break
        r_new = res6(c_new)
        if not np.abs(r_new).max() < np.abs(r).max():
            break
        c6, r, polish = c_new, r_new, polish + 1
    C = (_SYM @ c6).reshape(D, D)
    C = 0.5 * (C + C.T)
    if return_info:
        return C, dict(f=float((b - D) / (b - np.trace(C))), evals=len(cache), defect=float(defect), newton_refinements=polish)
    return C


def giesekus_step_root(wnn, R, G, kappa, aG, tol=1e-13, maxit=20, return_info=False):
    """The positive definite solution of one implicit L1 step of the Giesekus law, 0 < aG <= 1.

    With (C - I) + aG (C - I)^2 = aG C^2 + (1 - 2 aG) C - (1 - aG) I, step n of the scheme reads

        kappa aG C^2 + (wnn + kappa (1 - 2 aG)) C - kappa (G C + C G^T) = R,    R = wnn sum_{m<n} q_nm C^m + kappa (1 - aG) I,

    an algebraic Riccati equation A^T C + C A - C (kappa aG I) C + R = 0 with A = (kappa G - (wnn + kappa (1 - 2 aG))/2 I)^T.
    Since kappa aG I and R are positive definite, the pair is controllable and observable, and the equation has
    exactly one positive semidefinite solution, the stabilising one, which is positive definite; all other
    solutions have a negative eigenvalue. It is computed with scipy.linalg.solve_continuous_are and refined by
    Newton's iteration on the step equation, whose Jacobian is nonsingular at this solution.
    Returns C, or (C, info). Raises RuntimeError if R is not positive definite or the result is not positive definite.
    """
    from scipy.linalg import solve_continuous_are
    if not 0.0 < aG <= 1.0:
        raise ValueError('giesekus_step_root requires 0 < aG <= 1')
    R = np.asarray(R, dtype=float)
    R = 0.5 * (R + R.T)
    if not (np.isfinite(R).all() and np.linalg.eigvalsh(R)[0] > 0.0):
        raise RuntimeError('Giesekus step: the right-hand side is not positive definite (inadmissible history)')
    G = np.asarray(G, dtype=float)
    sig = wnn + kappa * (1.0 - 2.0 * aG)
    C = solve_continuous_are((kappa * G - 0.5 * sig * I3).T, math.sqrt(kappa * aG) * I3, R, I3)
    C = 0.5 * (C + C.T)

    def resid(X):
        return kappa * aG * X @ X + sig * X - kappa * (G @ X + X @ G.T) - R
    its = 0
    for its in range(1, maxit + 1):
        M = kappa * aG * C + 0.5 * sig * I3 - kappa * G           # Jacobian: X -> M X + X M^T
        dC_ = np.linalg.solve(lyapunov_matrix(M), -resid(C).reshape(-1)).reshape(D, D)
        C = C + 0.5 * (dC_ + dC_.T)
        if np.abs(dC_).max() <= tol * (1.0 + np.abs(C).max()):
            break
    if not np.linalg.eigvalsh(C)[0] > 0.0:
        raise RuntimeError('Giesekus step: the computed root is not positive definite')
    if return_info:
        return C, dict(newton_iters=its, residual=float(np.abs(resid(C)).max()))
    return C


def step_root(model, wnn, R0, tn, alpha, Cprev=None):
    """Admissible solution of one implicit L1 step of a nonlinear law; R0 = wnn sum_{m<n} q_nm C^m (3 x 3)."""
    kappa = model.Theta ** alpha
    G = model.Wi * model._amp(tn) * model.Gh
    if model.law == 'fenep':
        f0 = None
        if Cprev is not None and np.trace(Cprev) < model.b:
            f0 = (model.b - D) / (model.b - np.trace(Cprev))
        return fenep_step_root(wnn, R0 + kappa * I3, G, kappa, model.b, f0)
    if model.law == 'giesekus':
        return giesekus_step_root(wnn, R0 + kappa * (1.0 - model.aG) * I3, G, kappa, model.aG)
    raise ValueError(model.law)


def solve_l1(model, t, alpha, C0=None, W=None, tol=1e-12, maxit=60, explicit=False, return_info=False,
             nonlinear='safeguarded'):
    """Implicit L1 scheme for D^alpha C = F(tau, C) on the mesh t (full history).

    Linear law: one linear solve per step (9 x 9); the iterates are positive definite if and only if
    w_nn > 2 Theta^alpha s(Wi Ghat - I/2) at every step (the step condition of the article).

    Nonlinear laws (FENE-P, Giesekus with 0 < aG <= 1): the equation of a step has several solutions, exactly one of
    which is admissible (positive definite, tr C < b for FENE-P), for every step size. Newton's iteration starts
    from the previous iterate and stops when the Frobenius norm of the update is at most tol (1 + |C|_F), after at
    most maxit iterations. The argument ``nonlinear`` selects how the step is solved:
      'safeguarded' (default) Newton's iteration; for FENE-P a Newton step is halved as long as the new iterate
                    would have tr C >= b. The result is accepted only if the iteration converged to an admissible
                    tensor, which is then the admissible solution because that solution is unique. Otherwise the
                    step is solved by fenep_step_root or giesekus_step_root.
      'root'        every step is solved by fenep_step_root or giesekus_step_root.
      'newton'      the same iteration as in 'safeguarded' (steps halved so that tr C < b), without the test of the
                    result. It can converge to a solution with a negative eigenvalue without any indication, or
                    fail to converge (RuntimeError with the attributes step and iterates); see solve_l1_newton.
      'newton-full' Newton's iteration with full steps (the textbook iteration, no halving, no test). For FENE-P an
                    iterate can pass the pole tr C = b, so that this variant can also converge to a solution with
                    tr C > b; it fails on more meshes than 'newton', also on meshes that satisfy the step condition
                    of the linear law (experiment b5_fenep_step.py). For the Giesekus law it equals 'newton'.
    explicit=True gives the explicit (forward) L1 scheme, used only to illustrate loss of positivity.

    return_info=True also returns a dictionary with the number of Newton iterations (linear solves for the linear
    law) and, for nonlinear laws, the list 'root_steps' of the steps that were solved by the root solver.
    """
    if nonlinear not in NONLINEAR_SOLVERS:
        raise ValueError(f'nonlinear must be one of {NONLINEAR_SOLVERS}')
    C0 = np.eye(D) if C0 is None else np.asarray(C0, dtype=float)
    N = len(t) - 1
    W = l1_weight_rows(t, alpha) if W is None else W
    C = np.zeros((N + 1, D, D))
    C[0] = C0
    dC = np.zeros((N, D * D))
    nit_tot = 0
    root_steps = []
    Id = np.eye(D * D)
    for n in range(1, N + 1):
        w = W[n - 1, :n]
        hist = w[:-1] @ dC[:n - 1] if n > 1 else np.zeros(D * D)
        wnn = w[-1]
        xprev = C[n - 1].reshape(-1)
        if explicit:
            x = xprev + (model.F(t[n - 1], C[n - 1], alpha).reshape(-1) - hist) / wnn
        elif model.law == 'ucm':
            J = model.dF(t[n], C[n - 1], alpha)
            f0 = model.F(t[n], np.zeros((D, D)), alpha).reshape(-1)     # affine part
            x = np.linalg.solve(wnn * Id - J, wnn * xprev - hist + f0)
            nit_tot += 1
        else:
            converged = False
            if nonlinear != 'root':
                x = xprev.copy()
                limit = model.law == 'fenep' and nonlinear != 'newton-full'
                try:
                    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                        for it in range(maxit):
                            Cx = x.reshape(D, D)
                            res = wnn * (x - xprev) + hist - model.F(t[n], Cx, alpha).reshape(-1)
                            J = wnn * Id - model.dF(t[n], Cx, alpha)
                            dx = np.linalg.solve(J, -res)
                            step = 1.0
                            if limit:                           # stay inside tr C < b
                                while np.trace((x + step * dx).reshape(D, D)) >= model.b and step > 1e-6:
                                    step *= 0.5
                            x = x + step * dx
                            nit_tot += 1
                            if not np.isfinite(x).all():        # overflow or division by zero at the pole: no convergence
                                break
                            if np.linalg.norm(step * dx) <= tol * (1.0 + np.linalg.norm(x)):
                                converged = True
                                break
                except np.linalg.LinAlgError:
                    if nonlinear in ('newton', 'newton-full'):
                        raise _newton_failure(n, C) from None
                if nonlinear in ('newton', 'newton-full'):
                    if not converged:
                        raise _newton_failure(n, C)
                elif converged:                             # accept only an admissible solution of the step equation
                    Cx = x.reshape(D, D)
                    converged = is_admissible(model, Cx)
                    if converged:
                        Fx = model.F(t[n], Cx, alpha).reshape(-1)
                        res = wnn * (x - xprev) + hist - Fx
                        scale = wnn * (1.0 + np.abs(x).max()) + np.abs(Fx).max() + np.abs(hist).max()
                        converged = bool(np.abs(res).max() <= 1e-8 * scale)
            if not converged:
                x = step_root(model, wnn, (wnn * xprev - hist).reshape(D, D), t[n], alpha, C[n - 1]).reshape(-1)
                root_steps.append(n)
        Cn = x.reshape(D, D)
        Cn = 0.5 * (Cn + Cn.T)
        C[n] = Cn
        dC[n - 1] = Cn.reshape(-1) - xprev
    if return_info:
        return C, dict(newton_iters=nit_tot, root_steps=root_steps)
    return C


def solve_l1_newton(model, t, alpha, full=False, **kw):
    """Implicit L1 scheme with Newton's iteration for nonlinear laws, without the admissibility test of the result.

    Kept to document the behaviour of an unsafeguarded solver. Started from the previous iterate, the iteration can
    converge to an inadmissible solution of the step equation without any indication.
    full=False (nonlinear='newton'): for FENE-P a step is halved as long as the new iterate would have tr C >= b, so
    that an inadmissible result has a negative eigenvalue; in the experiment b5_fenep_step.py this variant fails
    only on meshes whose steps exceed the bound of the linear law.
    full=True (nonlinear='newton-full'): full Newton steps; an iterate can pass the pole tr C = b, and the variant
    also fails on meshes that satisfy the step condition of the linear law.
    Raises RuntimeError if the iteration does not converge at a step; the error has the attributes ``step`` (index
    of the step) and ``iterates`` (the iterates computed before it)."""
    return solve_l1(model, t, alpha, nonlinear='newton-full' if full else 'newton', **kw)


def step_residuals(model, t, alpha, C, W=None):
    """Scaled residual of the scheme at every step: max |D_h C^n - F(t_n, C^n)| divided by
    w_nn (1 + max |C^n|) + max |F(t_n, C^n)| + max |history sum| (the scale of the acceptance test of solve_l1)."""
    N = len(t) - 1
    W = l1_weight_rows(t, alpha) if W is None else W
    dC = np.diff(C.reshape(N + 1, -1), axis=0)
    out = np.zeros(N)
    for n in range(1, N + 1):
        w = W[n - 1, :n]
        hist = w[:-1] @ dC[:n - 1] if n > 1 else np.zeros(D * D)
        Fx = model.F(t[n], C[n], alpha).reshape(-1)
        res = w[-1] * dC[n - 1] + hist - Fx
        out[n - 1] = np.abs(res).max() / (w[-1] * (1.0 + np.abs(C[n]).max()) + np.abs(Fx).max() + np.abs(hist).max())
    return out


def fenep_trace_bound(model):
    """Bound theta_* of tr C for the FENE-P law: positive root of
    2 Wi |Ghat|_2 th^2 + b (1 - 2 Wi |Ghat|_2) th - 3 b = 0 (continuous solution and admissible iterates, C0 = I)."""
    g = 2.0 * model.Wi * np.linalg.norm(model.Gh, 2)
    b = model.b
    if g == 0.0:
        return float(D)
    return float((-(b * (1.0 - g)) + math.sqrt((b * (1.0 - g)) ** 2 + 4.0 * g * D * b)) / (2.0 * g))


def step_condition_lhs(model, t, alpha):
    """Largest value over the steps of Gamma(2 - alpha) (Theta h_n)^alpha (2 Wi s(Ghat) - 1); the iterates of the
    linear law are positive definite if and only if it is below one (in uniaxial extension s(Ghat) = 1)."""
    s = float(np.linalg.eigvals(model.Gh).real.max())
    return float(math.gamma(2.0 - alpha) * (model.Theta * np.diff(t).max()) ** alpha * (2.0 * model.Wi * s - 1.0))


def interp_linear(t, C, tq):
    """Piecewise-linear interpolation in time of nodal matrices (the interpolant underlying L1)."""
    tq = np.atleast_1d(tq)
    j = np.clip(np.searchsorted(t, tq, side='right') - 1, 0, len(t) - 2)
    s = ((tq - t[j]) / (t[j + 1] - t[j]))[:, None, None]
    return (1 - s) * C[j] + s * C[j + 1]


def reference_solution(model, alpha, tq, N=2048, r=None, C0=None):
    """High-accuracy reference by the implicit L1 scheme on graded meshes with N and 2N steps and
    one Richardson step of order min(2 - alpha, r alpha) ... returns (C_2N, C_rich, est_err)."""
    r = (2.0 - alpha) / alpha if r is None else r
    sols = []
    for n in (N, 2 * N):
        t = ref.graded_mesh(1.0, n, r)
        sols.append(interp_linear(t, solve_l1(model, t, alpha, C0), tq))
    p = 2.0 - alpha
    rich = sols[1] + (sols[1] - sols[0]) / (2.0 ** p - 1.0)
    num = np.sqrt(((sols[1] - sols[0]) ** 2).sum())
    den = np.sqrt((sols[1] ** 2).sum())
    return sols[1], rich, float(num / den)


# ----------------------------------------------------------------------------------------------
# SPD output maps with a hard initial condition (torch)
# ----------------------------------------------------------------------------------------------
VARIANTS = ('sym', 'factor', 'anchor', 'logm', 'bfactor', 'blogm')


def safe_pow(tau, alpha):
    """tau^alpha with a finite alpha-derivative at tau = 0."""
    return torch.clamp(tau, min=1e-300) ** alpha


def sym_logm_np(C):
    w, V = np.linalg.eigh(C)
    return (V * np.log(w)) @ V.T


class SPDNet(torch.nn.Module):
    """Network tau -> X(tau) in R^{3x3}, X(0) = C0 exactly for every parameter vector.

    All variants share the MLP (inputs (tau, tau^alpha), 9 outputs reshaped to a raw matrix N) and
    the gate g(tau) = 1 - exp(-beta tau^alpha):

      sym    : X = C0 + s g Sym(N)                        unconstrained symmetric output
      factor : T = T0 + sqrt(s) g N,  X = T T^T           factor map of the manuscript (no trace norm.)
      anchor : X = C0 + s g N N^T                         C0-anchored PSD increment (X >= C0)
      logm   : X = expm( log C0 + log(1+s) g Sym(N) )     log-conformation output

      bfactor: S = T T^T, T = T0 + sqrt(s) g N,  X = b S / (1 + tr S)      SPD and tr X < b (FENE-P)
      blogm  : S = expm(Psi0 + log(1+s) g Sym(N)), X = b S / (1 + tr S)   SPD and tr X < b (FENE-P)

    The last two generalise the trace-normalised factor map of the manuscript (rho = T T^+/tr T T^+)
    to the admissible set {X > 0, tr X < b} of finitely extensible models; S0 = C0/(b - tr C0).

    s is an a-priori scale of |C - C0| (the same number for all variants). For the unconstrained
    output an optional component-wise scale matrix ``comp_scale`` (3 x 3, symmetric) replaces s:
    X_ij = C0_ij + s_ij g Sym(N)_ij  (best-practice scaling of an unconstrained output).
    """

    def __init__(self, variant, C0, alpha, scale=1.0, width=32, depth=3, beta=5.0, comp_scale=None, b=None):
        super().__init__()
        assert variant in VARIANTS
        self.variant, self.beta = variant, beta
        self.alpha = alpha
        self.scale = float(scale)
        self.comp_scale = None if comp_scale is None else torch.as_tensor(comp_scale, dtype=DT)
        self.b = b
        self.net = MLP(width, depth, D * D, nin=2)
        C0 = torch.as_tensor(C0, dtype=DT)
        self.C0 = C0
        if variant in ('bfactor', 'blogm'):
            assert b is not None and float(torch.trace(C0)) < b
            C0 = C0 / (b - torch.trace(C0))                 # S0 with b S0/(1 + tr S0) = C0
        self.T0 = L.factor_of_state(C0)                     # T0 T0^T = C0 (eigen-decomposition)
        self.Psi0 = torch.as_tensor(sym_logm_np(C0.numpy()), dtype=DT)

    def forward(self, tau, alpha=None):
        a = self.alpha if alpha is None else alpha
        ta = safe_pow(tau, a)
        N = self.net(torch.stack([tau, ta], 1)).reshape(-1, D, D)
        g = -torch.expm1(-self.beta * ta)
        S = 0.5 * (N + N.transpose(-1, -2))
        s = self.scale
        if self.variant == 'sym':
            if self.comp_scale is not None:
                return self.C0 + g[:, None, None] * (self.comp_scale * S)
            return self.C0 + (s * g)[:, None, None] * S
        if self.variant == 'factor':
            # hard initial condition at the factor level: fracsp.layers.hard_ic_factor with tau -> tau^alpha
            T = L.hard_ic_factor(self.T0, math.sqrt(s) * N, ta, self.beta)
            return T @ T.transpose(-1, -2)
        if self.variant == 'anchor':
            return self.C0 + (s * g)[:, None, None] * (N @ N.transpose(-1, -2))
        if self.variant == 'logm':
            Psi = self.Psi0 + (math.log1p(s) * g)[:, None, None] * S
            return torch.linalg.matrix_exp(Psi)
        if self.variant in ('bfactor', 'blogm'):
            if self.variant == 'bfactor':
                T = L.hard_ic_factor(self.T0, math.sqrt(s) * N, ta, self.beta)
                Sm = T @ T.transpose(-1, -2)
            else:
                Sm = torch.linalg.matrix_exp(self.Psi0 + (math.log1p(s) * g)[:, None, None] * S)
            tr = torch.diagonal(Sm, dim1=-2, dim2=-1).sum(-1)
            return self.b * Sm / (1.0 + tr)[:, None, None]
        raise ValueError(self.variant)


LOSSES = ('fro', 'comp', 'rel', 'nat')


def residual_loss(net, model, tcol, W, alpha, scale, Wi=None, Theta=None, kind='fro', comp_scale=None):
    """Mean squared L1 residual at the collocation nodes n = 1..N.

    kind = 'fro'  : |R|_F^2 / s^2                    (one global a-priori scale s)
    kind = 'comp' : sum_ij (R_ij / s_ij)^2           (component-wise a-priori scales; frame dependent)
    kind = 'rel'  : |R|_F^2 / |X|_F^2                (scalar relative residual; defined for any X != 0,
                    balances times but not eigen-directions; the scale-free option for an
                    unconstrained output)
    kind = 'nat'  : tr(X^-1 R X^-1 R) = |X^-1/2 R X^-1/2|_F^2
                    (affine-invariant norm of the residual at the current state X; needs X SPD,
                     i.e. an SPD output map; frame indifferent and free of a-priori scales)
    """
    X = net(tcol, alpha)
    Dh = op.l1_nonuniform(X, W)                                   # (N, 3, 3) discrete Caputo derivative
    R = Dh - model.F_torch(tcol[1:], X[1:], alpha, Wi, Theta)
    if kind == 'fro':
        return (R ** 2).sum((-1, -2)).mean() / scale ** 2
    if kind == 'comp':
        return ((R / comp_scale) ** 2).sum((-1, -2)).mean()
    if kind == 'rel':
        return ((R ** 2).sum((-1, -2)) / (X[1:] ** 2).sum((-1, -2))).mean()
    if kind == 'nat':
        Y = torch.linalg.solve(X[1:], R)                          # X^-1 R
        return (Y * Y.transpose(-1, -2)).sum((-1, -2)).mean()
    raise ValueError(kind)


def rotation_matrix(angles_deg):
    """Proper rotation Q = Rz(a) Ry(b) Rx(c) (degrees); used to pose a flow in a frame that is not
    aligned with its principal directions: Ghat' = Q Ghat Q^T, C' = Q C Q^T."""
    a, b, c = [math.radians(x) for x in angles_deg]
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(c), -math.sin(c)], [0, math.sin(c), math.cos(c)]])
    return Rz @ Ry @ Rx


def apriori_comp_scale(model, alpha, N=64, r=None, floor=1.0):
    """Component-wise scales max_n |C_ij^n - C0_ij| (floored) from a coarse implicit L1 solve."""
    r = (2.0 - alpha) / alpha if r is None else r
    t = ref.graded_mesh(1.0, N, r)
    C = solve_l1(model, t, alpha)
    return np.maximum(np.abs(C - C[0]).max(0), floor)


def train(net, loss_fn, params, adam_iters=4000, lbfgs_iters=500, lr=2e-3, log_every=500):
    """Adam (cosine schedule to 5 %) followed by L-BFGS, as in pinn.train of the package."""
    hist = []
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters, eta_min=lr * 0.05)
    t0 = time.perf_counter()
    nonfinite = 0
    for it in range(adam_iters):
        opt.zero_grad()
        try:
            loss = loss_fn()
        except RuntimeError:                                 # e.g. singular X in the affine-invariant norm
            loss = torch.tensor(float('nan'))
        if not torch.isfinite(loss):
            nonfinite += 1
            sched.step()
            continue
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0:
            hist.append((it, float(loss.detach())))
    t_adam = time.perf_counter() - t0
    nev = [0]
    t1 = time.perf_counter()
    if lbfgs_iters > 0:
        opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters, history_size=50,
                                 tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn='strong_wolfe')
        snapshot = [p.detach().clone() for p in params]

        def closure():
            opt2.zero_grad()
            loss = loss_fn()
            loss.backward()
            nev[0] += 1
            return loss
        try:
            opt2.step(closure)
        except Exception:                                    # pragma: no cover (defensive)
            nonfinite += 1

        def safe_loss():
            with torch.no_grad():
                try:
                    return float(loss_fn())
                except RuntimeError:
                    return float('nan')
        if not math.isfinite(safe_loss()):                   # L-BFGS diverged: restore the Adam iterate
            nonfinite += 1
            with torch.no_grad():
                for p, q in zip(params, snapshot):
                    p.copy_(q)
    t_lbfgs = time.perf_counter() - t1
    with torch.no_grad():
        try:
            final = float(loss_fn())
        except RuntimeError:
            final = float('nan')
    return dict(history=hist, final_loss=final, t_adam=t_adam, t_lbfgs=t_lbfgs, lbfgs_evals=nev[0],
                nonfinite_events=nonfinite)


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def test_grid(n_uniform=400, n_layer=100, layer=(1e-6, 1e-2)):
    """Dense test grid: uniform midpoints on (0, 1) plus logarithmically spaced points in the
    initial layer. Disjoint from the collocation nodes."""
    tu = (np.arange(1, n_uniform + 1) - 0.5) / n_uniform
    tl = np.logspace(math.log10(layer[0]), math.log10(layer[1]), n_layer)
    return np.sort(np.concatenate([tl, tu]))


def metrics(X, Cex, pd_tol=0.0):
    """X, Cex: (M, 3, 3). Relative Frobenius error (global and worst pointwise), smallest
    eigenvalue of the symmetric part, fraction of test times at which X is not positive definite,
    log-Euclidean error (only where X is SPD)."""
    Xs = 0.5 * (X + np.swapaxes(X, -1, -2))
    dif = X - Cex
    nd = np.sqrt((dif ** 2).sum((-1, -2)))
    nc = np.sqrt((Cex ** 2).sum((-1, -2)))
    w = np.linalg.eigvalsh(Xs)
    wex = np.linalg.eigvalsh(Cex)
    lmin = w[:, 0]
    bad = lmin <= pd_tol
    out = dict(rel_fro=float(np.sqrt((nd ** 2).sum() / (nc ** 2).sum())),
               max_rel_fro=float((nd / nc).max()),
               min_eig=float(lmin.min()), min_eig_exact=float(wex[:, 0].min()),
               max_rel_err_min_eig=float((np.abs(lmin - wex[:, 0]) / wex[:, 0]).max()),
               frac_time_not_pd=float(bad.mean()),
               min_det=float(np.linalg.det(Xs).min()), min_det_exact=float(np.linalg.det(Cex).min()))
    if not bad.any():
        le = np.array([np.linalg.norm(sym_logm_np(a) - sym_logm_np(b)) for a, b in zip(Xs, Cex)])
        out['log_euclid_rms'] = float(np.sqrt((le ** 2).mean()))
        out['log_euclid_max'] = float(le.max())
    else:
        out['log_euclid_rms'] = None
        out['log_euclid_max'] = None
    return out


def apriori_scale(model, alpha, N=64, r=None):
    """Scale of |C - C0|_F from a coarse implicit L1 solve (available without the exact solution)."""
    r = (2.0 - alpha) / alpha if r is None else r
    t = ref.graded_mesh(1.0, N, r)
    C = solve_l1(model, t, alpha)
    return float(np.sqrt(((C - C[0]) ** 2).sum((-1, -2))).max())

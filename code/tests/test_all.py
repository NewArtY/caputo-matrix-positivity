"""Unit and analytical tests (pytest). Run: python -m pytest -q tests

Coverage (deterministic part):
  L1 weights (uniform and graded) against 50-digit arithmetic; exactness for linear functions; torch versus numpy
  alpha-derivatives of the uniform AND of the nonuniform operator against finite differences and 50-digit references
  alpha -> 1 limit of operators and solvers (implicit Euler)
  SOE kernel: realised relative error <= target on [delta, T] (checked in 40-digit arithmetic), operator bound,
              uniform and nonuniform SOE solvers against the full-history solver
  Mittag-Leffler function for real and complex arguments (closed forms, Laplace inversion); matrix solution
  Dirac Bloch generator against a direct Lindblad construction; band-referenced current; analytical tail constant
  positivity of the implicit L1 step for GKSL generators (d = 2..4, uniform and graded meshes), Bloch-ball invariance
  second-order product-trapezoidal rule (independent scheme)
"""
import math, sys, os
import numpy as np
import torch
import pytest
import mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fracsp import operators as op, reference as ref, layers as L, mittag as ml, exact, dirac
from scipy.special import gamma
torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------------------------------------ L1 operators
@pytest.mark.parametrize('alpha', [0.3, 0.5, 0.8, 1.0])
def test_l1_linear_exact(alpha):
    # L1 is exact for piecewise-linear functions: D^a t = t^{1-a}/Gamma(2-a)
    N, T = 64, 1.0; h = T / N
    t = torch.arange(N + 1) * h
    D = op.l1_uniform(t, alpha, h)
    ex = t[1:] ** (1 - alpha) / math.gamma(2 - alpha)
    assert torch.max(torch.abs(D - ex)) < 1e-12


def test_l1_uniform_vs_nonuniform():
    N = 50; h = 1.0 / N; t = torch.arange(N + 1) * h
    f = torch.sin(3 * t) + t ** 2
    W = op.l1_nonuniform_matrix(t, 0.6)
    assert torch.max(torch.abs(op.l1_uniform(f, 0.6, h) - op.l1_nonuniform(f, W))) < 1e-11


def test_l1_torch_vs_reference_weights():
    # torch conv implementation vs independent numpy row weights
    N = 40; t = np.linspace(0, 1, N + 1); f = np.cos(2 * t)
    Dt = op.l1_uniform(torch.tensor(f), 0.4, 1 / N).numpy()
    Dn = np.array([ref.l1_row_weights(t, n, 0.4) @ np.diff(f)[:n] for n in range(1, N + 1)])
    assert np.max(np.abs(Dt - Dn)) < 1e-11


@pytest.mark.parametrize('alpha', [0.05, 0.3, 0.6, 0.9, 0.999999])
def test_l1_uniform_weights_vs_mpmath(alpha):
    # b_j = (j+1)^{1-a} - j^{1-a} in 50-digit arithmetic; the expm1/log1p form keeps full relative accuracy near a = 1
    n = 600
    b = op.l1_weights(alpha, n).numpy()
    with mp.workdps(50):
        a = mp.mpf(alpha)
        refb = np.array([float(mp.mpf(j + 1) ** (1 - a) - mp.mpf(j) ** (1 - a)) for j in range(n)])
    assert np.max(np.abs(b - refb) / np.abs(refb)) < 5e-15
    assert np.max(np.abs(ref.l1_uniform_weights(alpha, n, 1.0) * gamma(2 - alpha) - refb) / np.abs(refb)) < 5e-15


@pytest.mark.parametrize('alpha,r,N', [(0.3, 17 / 3, 48), (0.6, 7 / 3, 64), (0.8, 3.0, 64), (1.0, 3.0, 32)])
def test_l1_graded_weights_vs_mpmath(alpha, r, N):
    # nonuniform weights on a strongly GRADED mesh (where the naive difference of powers cancels) against 50 digits
    t = ref.graded_mesh(1.0, N, r)
    W = op.l1_nonuniform_matrix(torch.tensor(t), alpha).numpy()
    with mp.workdps(50):
        a = mp.mpf(alpha); tm = [mp.mpf(float(x)) for x in t]; g = mp.gamma(2 - a)
        Wr = np.zeros((N, N))
        for n in range(1, N + 1):
            for m in range(1, n + 1):
                upper = (tm[n] - tm[m]) ** (1 - a) if m < n else mp.mpf(0)          # (t_n - t_n)^{1-a} = 0 also at a = 1
                Wr[n - 1, m - 1] = float(((tm[n] - tm[m - 1]) ** (1 - a) - upper) / (g * (tm[m] - tm[m - 1])))
    low = np.tril(np.ones((N, N), dtype=bool))
    if alpha < 1:
        assert np.max(np.abs(W - Wr)[low] / np.abs(Wr)[low]) < 5e-14
    else:                       # alpha = 1: off-diagonal weights vanish, diagonal = 1/h_n
        assert np.max(np.abs(W - Wr)) < 1e-9 * np.abs(Wr).max()
    assert np.all(W[~low] == 0.0)
    Wn = np.zeros((N, N))
    for n in range(1, N + 1):
        Wn[n - 1, :n] = ref.l1_row_weights(t, n, alpha)
    assert np.max(np.abs(Wn - W)) <= 1e-13 * np.abs(W).max()
    if alpha == 0.3:            # the naive double-precision difference of powers is visibly wrong on this mesh
        tn = t[1:][:, None]; tmm = t[1:][None, :]; tm1 = t[:-1][None, :]
        naive = (np.abs(tn - tm1) ** (1 - alpha) - np.clip(tn - tmm, 0, None) ** (1 - alpha)) / (gamma(2 - alpha) * (tmm - tm1))
        assert np.max(np.abs(naive - Wr)[low] / np.abs(Wr)[low]) > 1e-9


def test_l1_nonuniform_rejects_invalid_mesh():
    for bad in ([0.0, 0.1, 0.1, 0.3], [0.0, 0.2, 0.1], [0.0]):
        with pytest.raises(ValueError):
            op.l1_nonuniform_matrix(torch.tensor(bad), 0.5)


def test_l1_uniform_complex_input():
    g = torch.Generator().manual_seed(3)
    F = torch.complex(torch.randn(40, 2, 2, generator=g), torch.randn(40, 2, 2, generator=g))
    D = op.l1_uniform(F, 0.6, 0.025)
    assert torch.equal(D.real, op.l1_uniform(F.real, 0.6, 0.025)) and torch.equal(D.imag, op.l1_uniform(F.imag, 0.6, 0.025))
    t = torch.arange(40) * 0.025
    Dn = op.l1_nonuniform(F, op.l1_nonuniform_matrix(t, 0.6))           # real weight matrix, complex data
    assert torch.max(torch.abs(D - Dn)) < 1e-10


# ------------------------------------------------------------------------------------------------ alpha -> 1
def test_alpha_one_limit():
    N = 20; h = 0.05; f = torch.arange(N + 1) * h; f = f ** 2
    D = op.l1_uniform(f, 1.0, h)
    assert torch.max(torch.abs(D - (f[1:] - f[:-1]) / h)) < 1e-12


def test_alpha_one_limit_operators_and_solvers():
    # operators: backward difference at alpha = 1 on a graded mesh, deviation O(1 - alpha) nearby
    t = ref.graded_mesh(1.0, 60, 2.0); tt = torch.tensor(t); f = torch.sin(3 * tt) + tt ** 2
    bd = (f[1:] - f[:-1]) / (tt[1:] - tt[:-1])
    assert torch.max(torch.abs(op.l1_nonuniform(f, op.l1_nonuniform_matrix(tt, 1.0)) - bd)) < 1e-11
    d2 = torch.max(torch.abs(op.l1_nonuniform(f, op.l1_nonuniform_matrix(tt, 0.99)) - bd)).item()
    d3 = torch.max(torch.abs(op.l1_nonuniform(f, op.l1_nonuniform_matrix(tt, 0.999)) - bd)).item()
    assert 5 < d2 / d3 < 20                                             # linear in 1 - alpha
    # solvers: implicit Euler, written out by hand, for a time-dependent 2x2 system on uniform and graded meshes
    A = lambda s: np.array([[[-1.0, 2.0 * math.cos(s)], [-2.0 * math.cos(s), -0.5]]]); b = lambda s: np.array([[0.3, math.sin(s)]])
    x0 = np.array([[1.0, -0.4]])
    for r in (1.0, 2.5):
        t = ref.graded_mesh(2.0, 80, r)
        x = x0.copy(); xs = [x0.copy()]
        for n in range(1, len(t)):
            hn = t[n] - t[n - 1]
            x = np.linalg.solve(np.eye(2)[None] / hn - A(t[n]), (b(t[n]) + x / hn)[..., None])[..., 0]; xs.append(x.copy())
        xs = np.array(xs)
        assert np.abs(ref.solve_l1(A, b, x0, t, 1.0)[1] - xs).max() < 1e-12
        assert np.abs(ref.solve_l1_soe_nonuniform(A, b, x0, t, 1.0, np.zeros(0), np.zeros(0))[1] - xs).max() < 1e-12
        if r == 1.0:
            assert np.abs(ref.solve_l1_soe(A, b, x0, 2.0 / 80, 80, 1.0, np.zeros(0), np.zeros(0))[1] - xs).max() < 1e-12
            assert np.abs(ref.solve_l1(A, b, x0, t, 1.0, weights='uniform')[1] - xs).max() < 1e-12
    # the fractional solution approaches the alpha = 1 solution linearly in 1 - alpha
    t = np.linspace(0, 2.0, 81); x1 = ref.solve_l1(A, b, x0, t, 1.0)[1]
    e2 = np.abs(ref.solve_l1(A, b, x0, t, 0.99)[1] - x1).max(); e3 = np.abs(ref.solve_l1(A, b, x0, t, 0.999)[1] - x1).max()
    assert 5 < e2 / e3 < 20


# ------------------------------------------------------------------------------------------------ alpha-derivatives
def test_alpha_gradient():
    N = 32; h = 1 / N; t = torch.arange(N + 1) * h; f = t ** 1.5
    for a0 in [0.3, 0.6, 0.9]:
        a = torch.tensor(a0, requires_grad=True)
        s = op.l1_uniform(f, a, h).sum(); s.backward()
        e = 1e-6
        fd = (op.l1_uniform(f, a0 + e, h).sum() - op.l1_uniform(f, a0 - e, h).sum()) / (2 * e)
        assert abs(a.grad.item() - fd.item()) / abs(fd.item()) < 1e-7


@pytest.mark.parametrize('a0', [0.2, 0.5, 0.8, 0.95])
def test_alpha_gradient_uniform_vs_mpmath(a0):
    # sum_n D_h f_n = h^-a/Gamma(2-a) sum_m df_m (N-m+1)^(1-a): derivative from 50-digit central differences
    N = 96; h = 1 / N; t = torch.arange(N + 1) * h; f = t ** 1.5 + torch.sin(3 * t) + 0.2
    a = torch.tensor(a0, requires_grad=True); op.l1_uniform(f, a, h).sum().backward()
    fn = f.numpy()
    with mp.workdps(50):
        def S(x):
            return mp.mpf(h) ** (-x) / mp.gamma(2 - x) * sum((mp.mpf(float(fn[m])) - mp.mpf(float(fn[m - 1]))) * mp.mpf(N - m + 1) ** (1 - x) for m in range(1, N + 1))
        e = mp.mpf('1e-18'); gref = float((S(mp.mpf(a0) + e) - S(mp.mpf(a0) - e)) / (2 * e))
    assert abs(a.grad.item() - gref) / abs(gref) < 1e-11
    # closed form of Eq. (4.25)/(4.27)
    df = np.diff(fn); b = op.l1_weights(a0, N).numpy(); db = op.dweights_dalpha(a0, N)
    conv_b = np.array([b[:n][::-1] @ df[:n] for n in range(1, N + 1)]); conv_db = np.array([db[:n][::-1] @ df[:n] for n in range(1, N + 1)])
    gan = (op.dcoef_dalpha(a0, h) * conv_b + h ** (-a0) / gamma(2 - a0) * conv_db).sum()
    assert abs(gan - gref) / abs(gref) < 1e-11


@pytest.mark.parametrize('a0,r', [(0.3, 17 / 3), (0.5, 3.0), (0.8, 3.0), (0.95, 3.0)])
def test_alpha_gradient_nonuniform_vs_mpmath(a0, r):
    # the operator differentiated in the inverse problem: nonuniform L1 matrix on a GRADED mesh, per-node derivative
    N = 40; tn = ref.graded_mesh(1.0, N, r); t = torch.tensor(tn); f = t ** 1.5 + torch.sin(3 * t) + 0.2
    J = torch.autograd.functional.jacobian(lambda x: op.l1_nonuniform(f, op.l1_nonuniform_matrix(t, x)), torch.tensor(a0)).numpy()
    fn = f.numpy()
    with mp.workdps(50):
        tm = [mp.mpf(float(x)) for x in tn]; fm = [mp.mpf(float(x)) for x in fn]

        def Dn(x):
            g = mp.gamma(2 - x); out = []
            for n in range(1, N + 1):
                out.append(sum((fm[m] - fm[m - 1]) / (tm[m] - tm[m - 1]) * ((tm[n] - tm[m - 1]) ** (1 - x) - ((tm[n] - tm[m]) ** (1 - x) if m < n else 0))
                               for m in range(1, n + 1)) / g)
            return out
        e = mp.mpf('1e-18'); up = Dn(mp.mpf(a0) + e); lo = Dn(mp.mpf(a0) - e)
        dref = np.array([float((u - l) / (2 * e)) for u, l in zip(up, lo)])
    assert np.max(np.abs(J - dref) / np.abs(dref)) < 1e-10
    # closed-form matrix derivative and central finite differences of the sum
    dW = op.dl1_nonuniform_matrix_dalpha(tn, a0)
    assert np.max(np.abs(dW @ np.diff(fn) - dref) / np.abs(dref)) < 1e-10
    S = lambda x: op.l1_nonuniform(f, op.l1_nonuniform_matrix(t, x)).sum().item()
    assert abs((S(a0 + 1e-6) - S(a0 - 1e-6)) / 2e-6 - dref.sum()) / abs(dref.sum()) < 1e-6


def test_alpha_gradient_finite_at_one():
    t = torch.tensor(ref.graded_mesh(1.0, 30, 3.0)); a = torch.tensor(1.0, requires_grad=True)
    op.l1_nonuniform_matrix(t, a).sum().backward()
    assert math.isfinite(a.grad.item())


# ------------------------------------------------------------------------------------------------ structure layers
def test_factor_map_structure():
    g = torch.Generator().manual_seed(0)
    T = torch.complex(torch.randn(1000, 3, 3, generator=g), torch.randn(1000, 3, 3, generator=g))
    T[:10, :, 1:] = 0  # rank-one (pure) states
    rho = L.normalized_factor_map(T)
    d = L.state_diagnostics(rho)
    assert d['herm_defect'] < 1e-14 and d['trace_defect'] < 1e-14 and d['min_eig'] > -1e-14
    pur = torch.einsum('bij,bji->b', rho[:10], rho[:10]).real
    assert torch.max(torch.abs(pur - 1)) < 1e-13


def test_hard_ic_exact():
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)
    T0 = L.factor_of_state(rho0)
    N = torch.complex(torch.randn(5, 2, 2), torch.randn(5, 2, 2))
    tau = torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0])
    rho = L.normalized_factor_map(L.hard_ic_factor(T0, N, tau))
    assert torch.max(torch.abs(rho[0] - rho0)) == 0.0


def test_rapidity_identities():
    p = torch.randn(10000, 2) * torch.logspace(-8, 3, 10000)[:, None]
    eta, eps, v = L.rapidity_deterministic(p, 1.0)
    disp = torch.abs(eps ** 2 - 1.0 - (p ** 2).sum(-1)) / eps ** 2
    assert disp.max() < 1e-13
    assert torch.linalg.norm(v, dim=-1).max() <= 1.0


# ------------------------------------------------------------------------------------------------ Mittag-Leffler
def test_mittag_cross_check():
    # series versus Talbot inversion; for alpha = 0.3 the series at z = -10 needs about 1000 digits (cancellation
    # ~ exp(|z|^(1/alpha))), so the largest argument is -6 there
    for a, zs in [(0.3, [-0.5, -3.0, -6.0]), (0.6, [-0.5, -3.0, -10.0]), (0.9, [-0.5, -3.0, -10.0])]:
        for z in zs:
            s = ml.ml_series(z, a); l = ml.ml_laplace(z, a)
            assert abs(float(s - l)) < 1e-12


def test_ml_exponential_limit():
    assert abs(float(ml.ml_series(-2.0, 1.0)) - math.exp(-2.0)) < 1e-14


def test_ml_complex_closed_forms():
    # E_1(z) = exp(z), E_{1/2}(z) = exp(z^2) erfc(-z), E_2(-z^2) = cos z, for COMPLEX arguments
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 8, 12) * np.exp(1j * rng.uniform(-math.pi, math.pi, 12))
    for x in z:
        assert abs(exact.ml_complex(x, 1.0) - np.exp(x)) < 1e-13 * max(1.0, abs(np.exp(x)))
        r12 = complex(mp.exp(mp.mpc(x) ** 2) * mp.erfc(-mp.mpc(x)))
        assert abs(exact.ml_complex(x, 0.5) - r12) < 1e-13 * max(1.0, abs(r12))
        assert abs(exact.ml_complex(-x ** 2, 2.0) - np.cos(x)) < 1e-13 * max(1.0, abs(np.cos(x)))
    # two-parameter function: E_{1,2}(z) = (exp(z) - 1)/z
    assert abs(exact.ml_complex(1.5 - 2.0j, 1.0, 2.0) - (np.exp(1.5 - 2.0j) - 1) / (1.5 - 2.0j)) < 1e-14


def test_ml_complex_many_matches_series():
    rng = np.random.default_rng(2)
    for a, zmax in [(0.3, 4.0), (0.6, 20.0), (0.9, 30.0)]:
        z = rng.uniform(0, zmax, 12) * np.exp(1j * rng.uniform(-math.pi, math.pi, 12))
        e1 = np.array([exact.ml_complex(x, a) for x in z]); e2 = exact.ml_complex_many(z, a)
        assert np.max(np.abs(e1 - e2) / np.abs(e1)) < 1e-14
    x = -rng.uniform(0, 8, 20)
    assert np.max(np.abs(exact.ml_complex_many(x, 0.5).real - ml.ml_vec(x, 0.5))) < 1e-15


def test_ml_complex_vs_laplace_inversion():
    # oscillatory eigenvalue of the fractional Bloch system: E_a(lambda t^a) from the series versus Talbot inversion
    a = 0.6; lam = complex(-1.0, 2 * math.pi)
    # (mpmath's Talbot rule assumes F(conj s) = conj F(s); the real and imaginary parts are therefore inverted
    # separately from the two reflection-symmetric combinations of the transforms for lambda and conj(lambda))
    for t in (0.05, 0.7, 3.0):
        with mp.workdps(30):
            am = mp.mpf(a); lm = mp.mpc(lam); lc = mp.conj(lm)
            Fre = lambda s: (s ** (am - 1) / (s ** am - lm) + s ** (am - 1) / (s ** am - lc)) / 2
            Fim = lambda s: (s ** (am - 1) / (s ** am - lm) - s ** (am - 1) / (s ** am - lc)) / 2j
            val = complex(float(mp.re(mp.invertlaplace(Fre, t, method='talbot'))), float(mp.re(mp.invertlaplace(Fim, t, method='talbot'))))
        assert abs(exact.ml_complex(lam * t ** a, a) - val) < 1e-12


def test_bloch_exact_many_and_solver():
    Om = 2 * math.pi; A = np.array([[-1.0, 0, 0], [0, -1.0, -Om], [0, Om, -1.0]]); b = np.array([0, 0, -1.0])
    R0 = np.array([[0.3, 0.2, 0.4], [0.0, 0.0, 1.0]]); t = np.array([0.0, 0.01, 0.3, 1.0])
    E1 = np.stack([exact.bloch_exact(A, b, r0, t, 0.6) for r0 in R0], 1); E2 = exact.bloch_exact_many(A, b, R0, t, 0.6)
    assert np.abs(E1 - E2).max() < 1e-14
    assert np.abs(E2[0] - R0).max() < 1e-15
    # implicit L1 on a graded mesh converges to the closed form (order 2 - alpha = 1.4 in the maximum over the nodes)
    errs = []
    for N in (64, 128):
        tm = ref.graded_mesh(1.0, N, 7 / 3)
        X = ref.solve_l1(lambda s: np.broadcast_to(A, (2, 3, 3)), lambda s: np.broadcast_to(b, (2, 3)), R0, tm, 0.6)[1]
        errs.append(np.abs(X - exact.bloch_exact_many(A, b, R0, tm, 0.6)).max())
    assert errs[1] < 5e-3 and 1.1 < math.log2(errs[0] / errs[1]) < 1.7


def test_reference_solver_ml():
    a = 0.7; N = 2000; t = np.linspace(0, 1, N + 1)
    tt, u = ref.solve_scalar_l1(1.0, t, a)
    ex = ml.ml_vec(-t[-1:] ** a, a)
    assert abs(u[-1] - ex[0]) < 1e-3


# ------------------------------------------------------------------------------------------------ SOE
def test_soe_operator_matches_full():
    N = 400; h = 1 / N; t = torch.arange(N + 1) * h; f = torch.sin(4 * t) + t ** 0.7
    om, la, ea, er = op.soe_kernel(0.5, h, 1.0, tol=1e-10)
    Ds = op.l1_soe_uniform(f, 0.5, h, om, la); Df = op.l1_uniform(f, 0.5, h)
    assert torch.max(torch.abs(Ds - Df)) / torch.max(torch.abs(Df)) < 1e-8


@pytest.mark.parametrize('alpha,delta,T,tol', [(0.3, 1e-3, 1.0, 1e-8), (0.6, 10 / 32768, 10.0, 1e-10), (0.8, 60 / 4096, 60.0, 1e-10),
                                             (0.95, 60 / 16384, 60.0, 1e-10), (0.9, 1e-7, 150.0, 1e-6), (0.5, 1e-4, 1.0, 1e-12)])
def test_soe_relative_error_bound(alpha, delta, T, tol):
    # the REALISED relative error on [delta, T] meets the target (relative criterion, any window length) ...
    om, la, ea, er = op.soe_kernel(alpha, delta, T, tol)
    assert er <= tol and np.all(om > 0) and np.all(la > 0)
    # ... and the double-precision measurement returned by soe_kernel is confirmed in 40-digit arithmetic
    with mp.workdps(40):
        omm = [mp.mpf(float(x)) for x in om]; lam_ = [mp.mpf(float(x)) for x in la]
        worst = 0.0
        for s in list(np.geomspace(delta, T, 9)) + [0.37 * T]:
            K = mp.mpf(s) ** (-mp.mpf(alpha)) / mp.gamma(1 - mp.mpf(alpha))
            Ka = sum(w * mp.exp(-l * mp.mpf(s)) for w, l in zip(omm, lam_))
            worst = max(worst, float(abs(K - Ka) / K))
    assert worst <= tol and worst <= 1.05 * er + 1e-15
    # the number of modes follows the documented estimate
    assert abs(len(om) - op.soe_mode_estimate(alpha, delta, T, tol)) <= 1.5


def test_soe_legacy_criterion_is_absolute():
    # the first version bounded the ABSOLUTE lower-tail error: on the Dirac window the relative target is missed
    om, la, ea, er = op.soe_kernel(0.9, 60 / 4096, 60.0, 1e-10, criterion='absolute')
    assert er > 1e-9
    assert op.soe_kernel(0.9, 60 / 4096, 60.0, 1e-10)[3] <= 1e-10


def test_soe_mode_count_growth():
    # M grows quadratically in log(1/tol) (second differences > 0) and linearly in log(T/delta)
    Ms = [len(op.soe_kernel(0.8, 1e-3, 1.0, tol)[0]) for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)]
    assert np.all(np.diff(Ms, 2) > 0)
    Md = [len(op.soe_kernel(0.8, 10.0 ** (-k), 1.0, 1e-10)[0]) for k in (3, 4, 5, 6)]
    assert np.max(np.abs(np.diff(Md, 2))) <= 1


def test_soe_operator_error_bound():
    # |D_SOE f_n - D_L1 f_n| <= eps * sum_{m<n} w_nm |f_m - f_{m-1}|  (eps: realised relative kernel error), Eq. (5.30)
    N = 256; h = 1.0 / N; t = np.arange(N + 1) * h; f = np.sin(5 * t) + t ** 0.6
    for a in (0.4, 0.8):
        om, la, ea, er = op.soe_kernel(a, h, 1.0, 1e-6)
        Ds = op.l1_soe_uniform(torch.tensor(f), a, h, om, la).numpy(); Df = op.l1_uniform(torch.tensor(f), a, h).numpy()
        w = ref.l1_uniform_weights(a, N, h); adf = np.abs(np.diff(f))
        bound = np.array([w[1:n][::-1] @ adf[:n - 1] if n > 1 else 0.0 for n in range(1, N + 1)])
        assert np.all(np.abs(Ds - Df) <= er * bound + 1e-13)
        assert np.abs(Ds - Df).max() > 1e-10                     # the bound is not vacuous at this tolerance


def test_soe_solvers_match_full_history():
    Om = 2 * math.pi; A0 = np.array([[-1.0, 0, 0], [0, -1.0, -Om], [0, Om, -1.0]]); b0 = np.array([0, 0, -1.0])
    X0 = np.array([[0.3, 0.2, 0.4], [0.0, 0.0, 1.0]]); gen = lambda s: (np.broadcast_to(A0, (2, 3, 3)), np.broadcast_to(b0, (2, 3)))
    a = 0.6; N = 256; T = 2.0; h = T / N
    om, la, ea, er = op.soe_kernel(a, h, T, 1e-10)
    Xf = ref.solve_l1(gen, None, X0, np.arange(N + 1) * h, a, weights='uniform')[1]
    assert np.abs(ref.solve_l1(gen, None, X0, np.arange(N + 1) * h, a)[1] - Xf).max() < 1e-12       # tabulated = row weights
    assert np.abs(ref.solve_l1_soe(gen, None, X0, h, N, a, om, la)[1] - Xf).max() < 1e-9
    assert np.abs(ref.solve_l1_soe_nonuniform(gen, None, X0, np.arange(N + 1) * h, a, om, la)[1] - Xf).max() < 1e-9
    t = ref.graded_mesh(T, N, 7 / 3)                                                               # graded mesh
    om, la, ea, er = op.soe_kernel(a, ref.soe_delta_for_mesh(t), T, 1e-10)
    Xg = ref.solve_l1(gen, None, X0, t, a)[1]
    assert np.abs(ref.solve_l1_soe_nonuniform(gen, None, X0, t, a, om, la)[1] - Xg).max() < 1e-9
    assert np.abs(ref.solve_l1(gen, None, X0, t, a, weights='table')[1] - Xg).max() == 0.0


# ------------------------------------------------------------------------------------------------ independent scheme
def test_product_trapezoid_second_order():
    # D^a u = -u + f with u = t^2: errors decrease with order 2; agreement with Richardson-extrapolated L1
    a = 0.8; errs = []
    for N in (64, 128, 256):
        Af = lambda s: -np.ones((1, 1, 1)); bf = lambda s: np.array([[2 * s ** (2 - a) / gamma(3 - a) + s ** 2]])
        errs.append(abs(ref.solve_pi_trapezoid(Af, bf, np.zeros((1, 1)), 0.0, 1.0, N, a)[1][-1, 0, 0] - 1.0))
    assert 1.85 < math.log2(errs[0] / errs[1]) < 2.1 and 1.85 < math.log2(errs[1] / errs[2]) < 2.1
    u = [ref.solve_l1(Af, bf, np.zeros((1, 1)), np.linspace(0, 1, N + 1), a)[1][-1, 0, 0] for N in (256, 512)]
    assert abs(ref.richardson(u[0], u[1], 2 - a) - 1.0) < 2e-5 < abs(u[1] - 1.0)


# ------------------------------------------------------------------------------------------------ Dirac model
PARS = dict(a0=1.0, sp=6.0, cep=0.0, chi=0.0, delta=0.5, Gdown=0.02, Gup=0.0, Gphi=0.05, zeta=1.0)


def test_dirac_generator_vs_lindblad():
    # literal construction: H = xi kx sx + ky sy + delta sz at the kinetic wave vector, jump operators built from the
    # instantaneous eigenvectors (Eqs. 3.42, 3.45-3.47); Bloch components of the GKSL right-hand side versus A r + b
    sx = np.array([[0, 1], [1, 0]], dtype=complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], dtype=complex)
    rng = np.random.default_rng(5); worst = 0.0
    for _ in range(60):
        p = dict(a0=rng.uniform(0, 2), sp=rng.uniform(2, 8), cep=rng.uniform(-3, 3), chi=rng.uniform(-3, 3), delta=rng.uniform(0.05, 2),
                 Gdown=rng.uniform(0, 0.5), Gup=rng.uniform(0.01, 0.5), Gphi=rng.uniform(0, 0.5), zeta=rng.uniform(-1, 1))
        xi = int(rng.choice([-1, 1])); tb = rng.uniform(-10, 10); kap = rng.normal(size=(1, 2)) * 2
        a = dirac.pulse_A(tb, p['a0'], p['sp'], p['zeta'], p['chi'], p['cep'])
        H = xi * (kap[0, 0] + a[0]) * sx + (kap[0, 1] + a[1]) * sy + p['delta'] * sz
        w, V = np.linalg.eigh(H); um, up = V[:, [0]], V[:, [1]]                      # lower, upper band
        Pp = up @ up.conj().T; Pm = um @ um.conj().T
        Ls = [math.sqrt(p['Gdown']) * um @ up.conj().T, math.sqrt(p['Gup']) * up @ um.conj().T, math.sqrt(p['Gphi'] / 2) * (Pp - Pm)]
        r = rng.normal(size=3) * 0.5; rho = 0.5 * (np.eye(2) + r[0] * sx + r[1] * sy + r[2] * sz)
        out = -1j * (H @ rho - rho @ H)
        for Lk in Ls:
            LdL = Lk.conj().T @ Lk; out = out + Lk @ rho @ Lk.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
        rdot = np.array([np.trace(out @ s).real for s in (sx, sy, sz)])
        A, b, n, eps = dirac.bloch_generator(kap, xi, tb, p)
        worst = max(worst, np.abs(A[0] @ r + b[0] - rdot).max(), np.abs(dirac.bloch_rhs(kap, xi, tb, r[None], p)[0] - rdot).max())
        assert abs(np.trace(out)) < 1e-13                                             # trace annihilation
        assert abs(eps[0] - w[1]) < 1e-12 and np.abs(Pm - 0.5 * (np.eye(2) - n[0, 0] * sx - n[0, 1] * sy - n[0, 2] * sz)).max() < 1e-12
    assert worst < 1e-12


def test_dirac_pulse_derivative_and_deviation_form():
    ts = np.linspace(-9, 9, 11); args = (1.0, 6.0, 0.7, 0.4, 0.3)
    fd = np.array([(dirac.pulse_A(x + 1e-6, *args) - dirac.pulse_A(x - 1e-6, *args)) / 2e-6 for x in ts])
    assert np.abs(dirac.pulse_dA(ts, *args) - fd).max() < 1e-8 and np.abs(dirac.pulse_E(ts, *args) + fd).max() < 1e-8
    rng = np.random.default_rng(6); kap = rng.normal(size=(20, 2)) * 3; s = rng.normal(size=(20, 3)) * 0.1; tb = 1.3
    n = dirac.axis(kap, -1, tb, PARS)[1]
    nd = (dirac.axis(kap, -1, tb + 1e-6, PARS)[1] - dirac.axis(kap, -1, tb - 1e-6, PARS)[1]) / 2e-6
    assert np.abs(dirac.deviation_rhs(kap, -1, tb, s, PARS) - dirac.bloch_rhs(kap, -1, tb, s - n, PARS) - nd).max() < 1e-8


def test_dirac_current_reference_band():
    # a valence band that follows the field adiabatically (r = -n(t)) carries NO current relative to the instantaneous
    # filled band, whereas the current relative to the initial state grows linearly with the cutoff
    p = dict(PARS, zeta=0.0); t0, tb = -30.0, 0.0
    a = dirac.pulse_A(tb, p['a0'], p['sp'], p['zeta'], p['chi'], p['cep'])
    vals = []
    for km in (8.0, 16.0):
        nk = int(km / 0.25) * 2; g = (np.arange(nk) + 0.5) / nk * 2 * km - km
        KX, KY = np.meshgrid(g, g, indexing='ij'); kap = np.c_[KX.ravel(), KY.ravel()]; J = np.zeros(2)
        for xi in (1, -1):
            n = dirac.axis(kap, xi, tb, p)[1]; r0 = dirac.ground_state(kap, xi, p, t0)
            c = dirac.current_integrands(-n, n, xi, r0)
            assert np.abs(c['band']).max() == 0.0 and np.abs(c['intra']).max() == 0.0
            J += 2.0 * c['init'].sum(0) * (2 * km / nk) ** 2 / (2 * math.pi) ** 2
        vals.append(J[0]); assert abs(J[1]) < 1e-12
        assert abs(J[0] - dirac.sea_current_square(km, a[0])) < 0.03 * abs(J[0])
    assert 1.9 < vals[1] / vals[0] < 2.1
    # decomposition band = intra + inter and n_c from the same quantities
    rng = np.random.default_rng(8); kap = rng.normal(size=(30, 2)); r = rng.normal(size=(30, 3)) * 0.4; n = dirac.axis(kap, -1, 0.3, p)[1]
    c = dirac.current_integrands(r, n, -1)
    assert np.abs(c['band'] - c['intra'] - c['inter']).max() < 1e-15
    assert np.abs(c['band'] + np.stack([-(r[:, 0] + n[:, 0]), r[:, 1] + n[:, 1]], 1)).max() < 1e-15


def test_dirac_bloch_ball_and_soe():
    # the implicit L1 trajectories of the Dirac model stay in the Bloch ball (full history and SOE)
    kap = np.array([[0.0, 0.0], [0.4, -0.3], [2.0, 0.0], [8.0, 3.0]]); t0, T, N, a = -30.0, 60.0, 600, 0.8; h = T / N
    gen = lambda s: dirac.bloch_generator(kap, 1, t0 + s, PARS); r0 = dirac.ground_state(kap, 1, PARS, t0)
    mx = [0.0]
    def cb(n, tn, x): mx[0] = max(mx[0], float(np.linalg.norm(x, axis=1).max()))
    Xf = ref.solve_l1(gen, None, r0, np.arange(N + 1) * h, a, weights='uniform', callback=cb)[1]
    om, la, ea, er = op.soe_kernel(a, h, T, 1e-10)
    Xs = ref.solve_l1_soe(gen, None, r0, h, N, a, om, la, callback=cb)[1]
    assert mx[0] <= 1.0 + 1e-13 and np.abs(Xf - Xs).max() < 1e-9
    nc = dirac.observables(Xf[-1:], kap, 1, [t0 + T], PARS)[0][0]
    assert np.all(nc > 0) and np.all(nc < 1)


def test_dirac_tail_constant():
    # analytical constants: C_alpha (alpha < 1; valley independent, sin^2 anisotropy for linear polarisation) and D (alpha = 1)
    C = dirac.tail_constant(0.8, PARS, 0.0)
    assert abs(C - 0.1231789) < 2e-7 and abs(dirac.tail_constant(0.8, PARS, 0.0, panel=0.25, ng=16) - C) < 1e-10
    p0 = dict(PARS, zeta=0.0)
    Cp = dirac.tail_constant(0.8, p0, math.pi / 2)
    assert dirac.tail_constant(0.8, p0, 0.0) == 0.0 and abs(dirac.tail_constant(0.8, p0, math.pi / 4) - Cp / 2) < 1e-10 and abs(Cp - 2 * C) < 2e-4
    assert dirac.tail_constant(0.8, dict(PARS, Gdown=0.0, Gphi=0.0), 0.0) > C                 # larger without dissipation
    assert abs(dirac.tail_constant_alpha1(PARS, 0.0) - 0.00928) < 1e-5
    # numerical confirmation at |kappa| = 64 with the independent second-order scheme
    kap = np.array([[64.0, 0.0]]); t0, T, N, a = -30.0, 60.0, 4096, 0.8
    X = ref.solve_pi_trapezoid(lambda s: dirac.bloch_generator(kap, 1, s, PARS), None, dirac.ground_state(kap, 1, PARS, t0), t0, T, N, a, store_every=N)[1]
    nc = dirac.observables(X[-1:], kap, 1, [t0 + T], PARS)[0][0, 0]
    assert abs(64.0 ** 2 * nc / C - 1) < 5e-3


# ------------------------------------------------------------------------------------------------ positivity of the L1 step
def _random_gksl(rng, d, strength):
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); H0 = strength * (X + X.conj().T) / 2
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); H1 = strength * (X + X.conj().T) / 2
    Ls = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(2)]
    return lambda t: ref.gksl_superoperator(H0 + math.cos(9 * t) * H1, [math.sqrt(abs(math.sin(5 * t + k))) * Lk for k, Lk in enumerate(Ls)])


@pytest.mark.parametrize('d', [2, 3, 4])
def test_l1_step_preserves_density_matrices(d):
    rng = np.random.default_rng(100 + d)
    psi = rng.standard_normal((5, d)) + 1j * rng.standard_normal((5, d)); psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    rho0 = np.einsum('bi,bj->bij', psi, psi.conj())                                    # pure states: boundary of the state space
    for alpha in (0.3, 0.8, 1.0):
        for r in (1.0, 3.0, (2 - alpha) / alpha):
            for strength in (1.0, 30.0):
                gen = _random_gksl(rng, d, strength); t = ref.graded_mesh(1.0, 60, r)
                tt, R = ref.solve_l1_gksl(gen, rho0, t, alpha)
                ev, trd, hd = ref.density_defects(R)
                assert ev > -1e-12 and trd < 1e-11 and hd < 1e-11
                # convex weights and complete positivity of the one-step map (Choi matrix of the resolvent)
                for n in (1, 2, 30, 60):
                    th, wnn = ref.l1_convex_weights(t, n, alpha)
                    assert th.min() >= 0.0 and abs(th.sum() - 1.0) < 1e-12
                    C = ref.choi_matrix(np.linalg.inv(np.eye(d * d) - gen(t[n]) / wnn), d)
                    assert np.linalg.eigvalsh(0.5 * (C + C.conj().T)).min() > -1e-12
    # SOE history on a graded mesh
    gen = _random_gksl(rng, d, 10.0); t = ref.graded_mesh(1.0, 60, 3.0)
    om, la, ea, er = op.soe_kernel(0.5, ref.soe_delta_for_mesh(t), 1.0, 1e-10)
    tt, Rs = ref.solve_l1_gksl(gen, rho0, t, 0.5, soe=(om, la))
    assert ref.density_defects(Rs)[0] > -1e-12 and np.abs(Rs - ref.solve_l1_gksl(gen, rho0, t, 0.5)[1]).max() < 1e-8


def test_gksl_superoperator_matches_bloch_matrix():
    # the B2 generator: superoperator form versus the Bloch matrix used by the reference solvers
    Om, Gd, Gp = 2 * math.pi, 1.0, 0.5
    sx = np.array([[0, 1], [1, 0]], dtype=complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sm = np.array([[0, 0], [1, 0]], dtype=complex)
    S = ref.gksl_superoperator(0.5 * Om * sx, [math.sqrt(Gd) * sm, math.sqrt(Gp / 2) * sz])
    A = np.array([[-(Gd / 2 + Gp), 0, 0], [0, -(Gd / 2 + Gp), -Om], [0, Om, -Gd]]); b = np.array([0, 0, -Gd])
    rng = np.random.default_rng(9)
    for _ in range(10):
        r = rng.normal(size=3) * 0.4; rho = 0.5 * (np.eye(2) + r[0] * sx + r[1] * sy + r[2] * sz)
        out = (S @ rho.reshape(-1)).reshape(2, 2)
        assert np.abs(np.array([np.trace(out @ s).real for s in (sx, sy, sz)]) - (A @ r + b)).max() < 1e-13


def test_current_cutoff_coefficient():
    # coefficient of the term 1/kappa_c of the band-referenced sheet current (fracsp.dirac_cutoff)
    from fracsp import dirac_cutoff as dcut
    args = dirac._pulse_args(PARS); G2 = dirac.rates(PARS)[1]
    ts = np.linspace(-12.0, 12.0, 25)
    fd = (dirac.pulse_dA(ts + 1e-6, *args) - dirac.pulse_dA(ts - 1e-6, *args)) / 2e-6
    assert np.abs(dcut.pulse_d2A(ts, *args) - fd).max() < 1e-8
    # alpha = 1: B = sqrt(2) g_s/(4 pi^2) (a'' + G2 a') for the square and g_s/(8 pi) (a'' + G2 a') for a disc
    t, B, parts = dcut.current_cutoff_coefficient(1.0, PARS, h=0.5)
    src = dcut.pulse_d2A(t, *args) + G2 * dirac.pulse_dA(t, *args)
    assert parts == {} and np.abs(B - math.sqrt(2) * 2.0 / (4 * math.pi ** 2) * src).max() < 1e-14
    Bd = dcut.current_cutoff_coefficient(1.0, PARS, h=0.5, shape='disc')[1]
    assert np.abs(Bd - 2.0 / (8 * math.pi) * src).max() < 1e-14
    # alpha < 1: the quadratic form of the tail coefficient reproduces fracsp.dirac.tail_constant; refinement of the step
    a = 0.8
    t1, B1, p1 = dcut.current_cutoff_coefficient(a, PARS, h=0.1)
    t2, B2, p2 = dcut.current_cutoff_coefficient(a, PARS, h=0.05)
    C = dirac.tail_constant(a, PARS, math.pi / 2)
    assert abs(p2['C_xx'][-1] - C) < 1e-5 * C and abs(p2['C_yy'][-1] - dirac.tail_constant(a, PARS, 0.0)) < 1e-5 * C
    assert abs(p2['C_xy'][-1]) < 1e-3 * C                              # circular polarisation: isotropic coefficient
    assert np.abs(B2[::2] - B1).max() < 5e-4 * np.abs(B2).max()
    assert 0.04 < np.abs(B2[:, 0]).max() < 0.06 and 0.04 < np.abs(B2[:, 1]).max() < 0.06
    # linear polarisation along x: no y-component; without memory terms the coefficient differs (alpha = 1 formula)
    B3 = dcut.current_cutoff_coefficient(a, dict(PARS, zeta=0.0), h=0.1)[1]
    assert np.abs(B3[:, 1]).max() < 1e-12 * np.abs(B3[:, 0]).max()
    # identity behind the source of C3: T[p, q] = D(p^2 q) - p D(p q) - p q D p = -Bil[p, p q] / Gamma(1 - alpha)
    f = lambda s: np.exp(-0.3 * np.asarray(s, dtype=float)) * np.cos(np.asarray(s, dtype=float))
    g = lambda s: np.sin(0.7 * np.asarray(s, dtype=float)) + 0.2 * np.asarray(s, dtype=float)
    df = lambda s: -np.exp(-0.3 * s) * (0.3 * np.cos(s) + np.sin(s)); dg = lambda s: 0.7 * np.cos(0.7 * s) + 0.2
    s = np.array([0.4, 1.3, 2.9]); t0 = 0.0
    D = lambda dfun: dcut.caputo_smooth(dfun, s, a, t0)
    T = D(lambda u: 2 * f(u) * df(u) * g(u) + f(u) ** 2 * dg(u)) - f(s) * D(lambda u: df(u) * g(u) + f(u) * dg(u)) - f(s) * g(s) * D(df)
    assert np.abs(T + dcut.bilinear_form(f, lambda u: f(u) * g(u), s, a, t0) / gamma(1 - a)).max() < 1e-10
    Q = f(s) * D(df) - 0.5 * D(lambda u: 2 * f(u) * df(u))
    assert np.abs(Q - dcut.bilinear_form(f, f, s, a, t0) / (2 * gamma(1 - a))).max() < 1e-10
    # kernel convolution: e_mu * 1 = t^alpha E_{alpha, alpha+1}(-mu t^alpha), and I^(1-alpha) of a polynomial
    h = 0.01; n = 301; mu = 0.3
    conv = dcut.kernel_convolution(np.ones(n), h, a, mu)
    assert abs(conv[-1] - 3.0 ** a * float(ml.ml_series(-mu * 3.0 ** a, a, a + 1))) < 1e-10
    assert abs(dcut.fractional_integral(lambda u: u ** 2, np.array([2.0]), 0.4, 0.0)[0] - 2.0 ** 2.4 * gamma(3.0) / gamma(3.4)) < 1e-11


def test_package_init_has_no_code():
    # fracsp/__init__.py consists of a docstring and the version string; its fingerprint is recorded in the result files,
    # and scripts/check_provenance.py treats a state of the file that differs only in the docstring as equivalent
    import ast
    import fracsp
    tree = ast.parse(open(fracsp.__file__, encoding='utf-8').read())
    assert len(tree.body) == 2 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[1], ast.Assign) and tree.body[1].targets[0].id == '__version__' and fracsp.__version__ == '0.1.0'


def test_l1_stability_bound_family():
    # D_h E^n = L_n[E^n] + Z^n with GKSL generators L_n on an arbitrary mesh implies, for 0 <= sigma <= 1,
    #   |E^n|_1 <= |E^0|_1 + Gamma(1+sigma-alpha)/Gamma(1+sigma) (t_n-t_0)^sigma max_{j<=n} (t_j-t_0)^(alpha-sigma) |Z^j|_1
    # (sigma = 0: constant Gamma(1-alpha); for sigma > 0 the constant stays bounded as alpha -> 1)
    rng = np.random.default_rng(21)
    tn = lambda X: np.abs(np.linalg.eigvalsh(X)).sum()                       # trace norm of a Hermitian matrix
    worst = {s: 0.0 for s in ('0', 'alpha/2', 'alpha', '1')}
    for case in range(60):
        d = int(rng.integers(2, 4)); alpha = float(rng.uniform(0.05, 0.999)); N = int(rng.integers(3, 30))
        t = np.r_[0.0, np.cumsum(10 ** rng.uniform(-4, 0, N))]                # random steps over four decades
        herm = lambda: (lambda A: A + A.conj().T)(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
        gens = [_random_gksl(rng, d, 10 ** rng.uniform(-1, 1.5)) for _ in range(3)]
        E = [herm()]; Z = [None]
        for n in range(1, N + 1):
            w = ref.l1_row_weights(t, n, alpha)
            S = gens[n % 3](t[n])
            Zn = herm() * 10 ** rng.uniform(-2, 1)
            hist = sum(w[m - 1] * (E[m] - E[m - 1]) for m in range(1, n)) if n > 1 else 0 * Zn
            rhs = w[-1] * E[n - 1] - hist + Zn
            En = np.linalg.solve(w[-1] * np.eye(d * d) - S, rhs.reshape(-1)).reshape(d, d)
            E.append(0.5 * (En + En.conj().T)); Z.append(Zn)
        for name, sigma in (('0', 0.0), ('alpha/2', alpha / 2), ('alpha', alpha), ('1', 1.0)):
            kap = gamma(1 + sigma - alpha) / gamma(1 + sigma)
            for n in range(1, N + 1):
                bound = tn(E[0]) + kap * t[n] ** sigma * max(t[j] ** (alpha - sigma) * tn(Z[j]) for j in range(1, n + 1))
                worst[name] = max(worst[name], tn(E[n]) / bound)
    assert max(worst.values()) <= 1.0 + 1e-10
    # the constants: Gamma(1-alpha) grows like 1/(1-alpha); sigma = alpha/2 gives at most 2
    assert gamma(1 - 0.999) > 900 and all(gamma(1 - a / 2) / gamma(1 + a / 2) <= 2.0 + 1e-12 for a in np.linspace(0.01, 1.0, 100))
    # D_h (t^sigma)^n >= Caputo derivative of t^sigma at t_n on every mesh (concavity), the inequality behind the bound
    for _ in range(200):
        alpha = float(rng.uniform(0.05, 0.999)); sigma = float(rng.uniform(0.01, 1.0)); N = int(rng.integers(1, 20))
        t = np.r_[0.0, np.cumsum(10 ** rng.uniform(-4, 0, N))]
        w = ref.l1_row_weights(t, N, alpha)
        Dh = float(w @ np.diff(t ** sigma))
        assert Dh >= gamma(1 + sigma) / gamma(1 + sigma - alpha) * t[-1] ** (sigma - alpha) * (1 - 1e-12)


def test_dirac_alpha1_tail_needs_a_field_that_vanishes_at_the_ends():
    # alpha = 1 on the window [-3, 3], at whose ends a_perp' is not zero: kappa^4 n_c(t_f) is not the dephasing integral D
    # of tail_constant_alpha1 alone. With d0 = a_perp'(t0), d1 = a_perp'(tf), G1 = gamma_parallel, G2 = gamma_perp,
    #   kappa^4 n_c(tf) = D + d1^2/16 + exp(-G1 (tf - t0)) d0^2/16 - d1 d0 exp(-G2 (tf - t0)) cos(Theta)/8 + O(1/kappa),
    # Theta = 2 int |h| dt: boundary terms of the integration by parts and the rotating solution excited by v(t0) = 0.
    from scipy.integrate import solve_ivp, quad
    t0, tf, phi, xi = -3.0, 3.0, 0.6, 1
    G1, G2, _ = dirac.rates(PARS)
    args = (PARS['a0'], PARS['sp'], PARS['zeta'], PARS['chi'], PARS['cep'])
    dperp = lambda t: float(dirac.pulse_dA(np.array([t]), *args)[0] @ np.array([-math.sin(phi), math.cos(phi)]))
    D = PARS['Gphi'] / 8.0 * quad(lambda s: math.exp(-G1 * (tf - s)) * dperp(s) ** 2, t0, tf, epsabs=1e-13, epsrel=1e-12)[0]
    d0, d1 = dperp(t0), dperp(tf)
    assert abs(d0) > 0.4 and abs(d1) > 0.4
    for kappa in (40.0, 40.26):
        kap = np.array([[kappa * math.cos(phi), kappa * math.sin(phi)]])
        sol = solve_ivp(lambda t, y: dirac.deviation_rhs(kap, xi, t, y[None, :], PARS)[0], (t0, tf), np.zeros(3), method='DOP853',
                        rtol=1e-11, atol=1e-13 * PARS['a0'] / (2.0 * kappa ** 2))
        assert sol.success
        nc = 0.5 * float(dirac.axis(kap, xi, tf, PARS)[1][0] @ sol.y[:, -1])

        def habs(s):
            A = dirac.pulse_A(np.array([s]), *args)[0]
            return math.sqrt((kap[0, 0] + A[0]) ** 2 + (kap[0, 1] + A[1]) ** 2 + PARS['delta'] ** 2)
        theta = 2.0 * quad(habs, t0, tf, limit=2000, epsabs=1e-12, epsrel=1e-13)[0]
        full = D + d1 ** 2 / 16.0 + math.exp(-G1 * (tf - t0)) * d0 ** 2 / 16.0 - d1 * d0 * math.exp(-G2 * (tf - t0)) * math.cos(theta) / 8.0
        assert abs(kappa ** 4 * nc - full) < 0.05 * full           # remainder O(1/kappa)
        assert kappa ** 4 * nc > 1.5 * D                           # the dephasing integral alone is far off


def test_closed_form_examples_of_the_continuous_problem():
    # coherence factor E_alpha(-i x) of a Hamiltonian generator, alpha = 0.8: a local minimum followed by a larger maximum
    m = lambda x: abs(exact.ml_complex(-1j * x, 0.8))
    xs = np.linspace(6.8, 9.0, 221)
    v = np.array([m(x) for x in xs])
    i0 = int(np.argmin(v[:80])); i1 = 80 + int(np.argmax(v[80:]))
    assert abs(xs[i0] - 7.27) < 0.02 and abs(v[i0] - 0.01666) < 2e-5
    assert abs(xs[i1] - 8.54) < 0.02 and abs(v[i1] - 0.02865) < 2e-5 and abs(v[i1] / v[i0] - 1.72) < 5e-3
    # dephasing with the rate 3 on [0, 1) and -0.05 on [1, 40]: closed form of y(40) against the implicit L1 scheme
    from scipy.integrate import quad
    for a, y40 in ((0.5, 1.3444), (0.8, 1.6967)):
        E = lambda z, b=1.0: exact.ml_complex(z, a, b).real
        f = lambda s: (40.0 - s) ** (a - 1.0) * E(0.05 * (40.0 - s) ** a, a) * E(-3.0 * s ** a)
        y = E(0.05 * 40.0 ** a) - 3.05 * quad(f, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
        assert abs(y - y40) < 1e-4 and y > 1.0                      # the coherence exceeds its initial modulus
        t = np.concatenate([ref.graded_mesh(1.0, 400, (2 - a) / a), 1.0 + ref.graded_mesh(39.0, 1600, (2 - a) / a)[1:]])
        u = np.ones(len(t))
        for n in range(1, len(t)):                                  # implicit L1 with the rate of the right end point of the step
            w = ref.l1_row_weights(t, n, a)
            g = 3.0 if t[n] <= 1.0 else -0.05
            u[n] = (w[-1] * u[n - 1] - w[:-1] @ np.diff(u[:n])) / (w[-1] + g)
        assert abs(u[-1] - y) < 5e-3

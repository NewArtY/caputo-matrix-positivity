"""Unit and analytical tests of the conformation-tensor benchmark B5 (fracsp.spd): exact solutions, implicit L1
stepper, Jacobians, output maps, positivity statements, FENE-P. Run: python -m pytest -q tests/test_b5.py"""
import math
import os
import sys

import numpy as np
import pytest
import torch
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import spd, reference as ref, operators as op  # noqa: E402

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- exact solutions
@pytest.mark.parametrize('flow,Wi', [('shear', 3.0), ('uniaxial', 0.3), ('uniaxial', 1.5), ('planar', 0.8)])
@pytest.mark.parametrize('alpha', [0.6, 0.9])
def test_closed_form_vs_matrix_series(flow, Wi, alpha):
    tau = np.array([1e-4, 0.03, 0.4, 1.0])
    a = spd.exact_ucm(flow, Wi, 2.0, alpha, tau)
    b = spd.exact_linear_general(spd.ghat(flow), Wi, 2.0, alpha, tau)
    assert np.max(np.abs(a - b) / (1 + np.abs(b))) < 1e-11


def test_alpha_one_limit_is_ucm():
    # alpha = 1: classical start-up of shear, C12 = Wi (1 - e^{-t}), N1 = 2 Wi^2 (1 - e^{-t} - t e^{-t})
    Wi, Th = 2.0, 3.0
    tau = np.array([0.1, 0.5, 1.0])
    C = spd.exact_ucm('shear', Wi, Th, 1.0, tau)
    t = Th * tau
    assert np.allclose(C[:, 0, 1], Wi * (1 - np.exp(-t)), atol=1e-13)
    assert np.allclose(C[:, 0, 0] - C[:, 1, 1], 2 * Wi ** 2 * (1 - np.exp(-t) - t * np.exp(-t)), atol=1e-12)


@pytest.mark.parametrize('flow,Wi', [('shear', 5.0), ('uniaxial', 1.0)])
def test_stepper_converges_to_exact(flow, Wi):
    alpha, Th = 0.6, 2.0
    m = spd.Model('ucm', flow, Wi, Th)
    tq = spd.test_grid(50, 20)
    ex = spd.exact_ucm(flow, Wi, Th, alpha, tq)
    errs = []
    for N in (64, 128, 256):
        t = ref.graded_mesh(1.0, N, (2 - alpha) / alpha)
        C = spd.interp_linear(t, spd.solve_l1(m, t, alpha), tq)
        errs.append(spd.metrics(C, ex)['rel_fro'])
    order = math.log2(errs[1] / errs[2])
    assert errs[2] < 2e-3 and order > 1.0


def test_stepper_matches_package_reference_solver():
    # independent implementation in fracsp.reference (affine, vectorised)
    alpha, Wi, Th = 0.7, 2.0, 1.5
    m = spd.Model('ucm', 'shear', Wi, Th)
    t = ref.graded_mesh(1.0, 60, 2.0)
    C = spd.solve_l1(m, t, alpha)
    A = Th ** alpha * (Wi * spd.lyapunov_matrix(spd.ghat('shear')) - np.eye(9))
    b = Th ** alpha * np.eye(3).reshape(-1)
    _, X = ref.solve_l1(lambda tn: A[None], lambda tn: b[None], np.eye(3).reshape(1, 9), t, alpha)
    assert np.max(np.abs(X[:, 0].reshape(-1, 3, 3) - C)) < 1e-11


# ---------------------------------------------------------------- Jacobians, torch right-hand side
@pytest.mark.parametrize('law', ['ucm', 'giesekus', 'fenep'])
def test_jacobian_and_torch_rhs(law):
    rng = np.random.default_rng(0)
    m = spd.Model(law, 'shear', 2.5, 1.7, aG=0.3, b=60.0)
    A = rng.standard_normal((3, 3))
    C = np.eye(3) + 0.3 * A @ A.T
    J = m.dF(0.4, C, 0.6)
    Jfd = np.zeros((9, 9))
    for k in range(9):
        e = np.zeros(9); e[k] = 1e-6
        Jfd[:, k] = (m.F(0.4, C + e.reshape(3, 3), 0.6) - m.F(0.4, C - e.reshape(3, 3), 0.6)).reshape(-1) / 2e-6
    assert np.max(np.abs(J - Jfd)) < 1e-7
    Ft = m.F_torch(torch.tensor([0.4]), torch.tensor(C)[None], 0.6)[0].numpy()
    assert np.max(np.abs(Ft - m.F(0.4, C, 0.6))) < 1e-12


def test_torch_residual_vanishes_on_stepper_solution():
    # the network residual and the time stepper discretise the same equations
    alpha = 0.6
    for m in (spd.Model('ucm', 'uniaxial', 1.0, 2.0), spd.Model('giesekus', 'shear', 5.0, 2.0, aG=0.2)):
        t = ref.graded_mesh(1.0, 48, (2 - alpha) / alpha)
        C = torch.tensor(spd.solve_l1(m, t, alpha))
        W = op.l1_nonuniform_matrix(torch.tensor(t), alpha)
        R = op.l1_nonuniform(C, W) - m.F_torch(torch.tensor(t[1:]), C[1:], alpha)
        assert R.abs().max().item() < 1e-8 * C.abs().max().item()


# ---------------------------------------------------------------- output maps
@pytest.mark.parametrize('variant', ['sym', 'factor', 'anchor', 'logm'])
def test_hard_ic_and_structure(variant):
    torch.manual_seed(0)
    C0 = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    net = spd.SPDNet(variant, C0, 0.6, scale=50.0)
    for p in net.net.parameters():            # large random weights: structure must not depend on training
        torch.nn.init.normal_(p, std=2.0)
    tau = torch.cat([torch.zeros(1), torch.rand(200)])
    X = net(tau).detach()
    assert torch.max(torch.abs(X[0] - torch.tensor(C0))).item() < 1e-13
    assert torch.max(torch.abs(X - X.transpose(-1, -2))).item() < 1e-9 * X.abs().max().item()
    lmin = torch.linalg.eigvalsh(0.5 * (X + X.transpose(-1, -2)))[:, 0]
    if variant in ('factor', 'logm', 'anchor'):
        assert lmin.min().item() > -1e-9 * X.abs().max().item()
    else:
        assert lmin.min().item() < 0          # the unconstrained map is not sign-definite


def test_alpha_gradient_finite_at_tau_zero():
    net = spd.SPDNet('factor', np.eye(3), 0.6, scale=10.0)
    a = torch.tensor(0.6, requires_grad=True)
    X = net(torch.tensor([0.0, 0.1, 1.0]), a)
    X.sum().backward()
    assert torch.isfinite(a.grad)


# ---------------------------------------------------------------- positivity analysis
def test_subordination_formula_alpha_half():
    # C_alpha(t) = int_0^inf M_alpha(theta) C_1(theta z) d theta, z = (t/lambda)^alpha;
    # for alpha = 1/2 the M-Wright density is exp(-theta^2/4)/sqrt(pi).
    Wi, z = 3.0, 1.3
    c12 = lambda s: Wi * (1 - math.exp(-s))
    n1 = lambda s: 2 * Wi ** 2 * (1 - math.exp(-s) - s * math.exp(-s))
    M = lambda th: math.exp(-th ** 2 / 4) / math.sqrt(math.pi)
    q12 = integrate.quad(lambda th: M(th) * c12(th * z), 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    q11 = integrate.quad(lambda th: M(th) * n1(th * z), 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    C = spd.exact_ucm('shear', Wi, 1.0, 0.5, np.array([z ** 2]))[0]      # Theta = 1, tau^0.5 = z
    assert abs(C[0, 1] - q12) < 1e-10 and abs(C[0, 0] - C[1, 1] - q11) < 1e-9


@pytest.mark.parametrize('flow,Wi', [('shear', 20.0), ('uniaxial', 0.45), ('uniaxial', 2.0)])
def test_exact_solution_spd_and_det_ge_one(flow, Wi):
    tq = spd.test_grid(60, 30)
    for alpha in (0.6, 0.9):
        C = spd.exact_ucm(flow, Wi, 2.0, alpha, tq)
        assert np.linalg.eigvalsh(C)[:, 0].min() > 0
        assert np.linalg.det(C).min() > 1 - 1e-9


def test_implicit_l1_positivity_threshold_in_extension():
    # resolvent of the Lyapunov operator is positive iff w_nn > Theta^alpha (2 Wi - 1)
    alpha, Wi, Th = 0.6, 2.0, 2.0
    m = spd.Model('ucm', 'uniaxial', Wi, Th)
    hcrit = (math.gamma(2 - alpha) * Th ** alpha * (2 * Wi - 1)) ** (-1 / alpha)
    for fac, ok in ((0.8, True), (1.25, False)):
        N = int(round(1.0 / (fac * hcrit)))
        t = np.linspace(0, 1, N + 1)
        C = spd.solve_l1(m, t, alpha)
        assert (np.linalg.eigvalsh(C)[:, 0].min() > 0) == ok


def test_implicit_l1_unconditionally_spd_in_shear():
    for N in (1, 2, 4, 8):
        for alpha in (0.6, 0.9):
            C = spd.solve_l1(spd.Model('ucm', 'shear', 50.0, 5.0), np.linspace(0, 1, N + 1), alpha)
            assert np.linalg.eigvalsh(C)[:, 0].min() > 0


# ---------------------------------------------------------------- finitely extensible model (FENE-P)
@pytest.mark.parametrize('variant', ['bfactor', 'blogm'])
def test_bounded_trace_maps(variant):
    torch.manual_seed(1)
    b = 50.0
    C0 = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    net = spd.SPDNet(variant, C0, 0.6, scale=20.0, b=b)
    for p in net.net.parameters():
        torch.nn.init.normal_(p, std=2.0)
    tau = torch.cat([torch.zeros(1), torch.rand(300)])
    X = net(tau).detach()
    assert torch.max(torch.abs(X[0] - torch.tensor(C0))).item() < 1e-12
    tr = torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)
    assert tr.max().item() <= b * (1 + 1e-12)     # tr X < b in exact arithmetic; = b (1 + eps) at overflow scale
    assert torch.linalg.eigvalsh(X)[:, 0].min().item() > -1e-9
    torch.manual_seed(2)
    net2 = spd.SPDNet(variant, C0, 0.6, scale=20.0, b=b)        # default initialisation: strictly inside
    X2 = net2(tau).detach()
    assert torch.diagonal(X2, dim1=-2, dim2=-1).sum(-1).max().item() < b
    assert torch.linalg.eigvalsh(X2)[:, 0].min().item() > 0


def test_fenep_stepper_admissible_and_convergent():
    alpha = 0.6
    m = spd.Model('fenep', 'uniaxial', 3.0, 2.0, b=50.0)
    tq = spd.test_grid(40, 10)
    sols = []
    for N in (256, 512, 1024):
        t = ref.graded_mesh(1.0, N, (2 - alpha) / alpha)
        C = spd.solve_l1(m, t, alpha)
        assert np.linalg.eigvalsh(C)[:, 0].min() > 0 and np.trace(C, axis1=1, axis2=2).max() < m.b
        sols.append(spd.interp_linear(t, C, tq))
    d1 = np.linalg.norm(sols[1] - sols[0]); d2 = np.linalg.norm(sols[2] - sols[1])
    assert d2 < d1 and math.log2(d1 / d2) > 1.0
    # the stretch approaches the steady state of the FENE-P model from below
    from scipy.optimize import fsolve

    def steady(x):
        f = (m.b - 3) / (m.b - x[0] - 2 * x[1])
        return [2 * m.Wi * x[0] - (f * x[0] - 1), -m.Wi * x[1] - (f * x[1] - 1)]
    c11, c22 = fsolve(steady, [40.0, 0.1])
    assert 0.95 * c11 < sols[2][-1][0, 0] < c11


# ---------------------------------------------------------------- subordination: further independent checks
def test_subordination_growth_alpha_half():
    # extension above the coil-stretch transition: C11 of the classical model grows like exp((2Wi-1)s)
    Wi, z = 1.5, 0.9
    a1 = 2 * Wi - 1
    c22 = lambda s: (1 + Wi * math.exp(-(1 + Wi) * s)) / (1 + Wi)
    M = lambda th: math.exp(-th ** 2 / 4) / math.sqrt(math.pi)
    # M(th) * C11_1(th z) with the two exponentials combined (no overflow at large th)
    f11 = lambda th: (2 * Wi * math.exp(-th ** 2 / 4 + a1 * th * z) - math.exp(-th ** 2 / 4)) / (a1 * math.sqrt(math.pi))
    q11 = integrate.quad(f11, 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    q22 = integrate.quad(lambda th: M(th) * c22(th * z), 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    C = spd.exact_ucm('uniaxial', Wi, 1.0, 0.5, np.array([z ** 2]))[0]
    assert abs(C[0, 0] - q11) < 1e-9 * q11 and abs(C[1, 1] - q22) < 1e-10


@pytest.mark.parametrize('alpha', [0.6, 0.9])
def test_subordination_monte_carlo(alpha):
    # C_alpha(t) = E[ C_1(S_t) ], S_t = (t/lambda)^alpha L^(-alpha) in units of lambda, with L a standard
    # positive alpha-stable variable (Kanter's representation): the second moment of a Hookean dumbbell
    # whose Langevin dynamics is run in the operational time of an inverse stable subordinator.
    rng = np.random.default_rng(12345)
    n = 2_000_000
    U = rng.uniform(0.0, math.pi, n)
    E = rng.exponential(1.0, n)
    Lst = np.sin(alpha * U) / np.sin(U) ** (1 / alpha) * (np.sin((1 - alpha) * U) / E) ** ((1 - alpha) / alpha)
    Wi, z = 3.0, 1.3
    s = z * Lst ** (-alpha)
    c12 = Wi * (1 - np.exp(-s))
    n1 = 2 * Wi ** 2 * (1 - np.exp(-s) - s * np.exp(-s))
    C = spd.exact_ucm('shear', Wi, 1.0, alpha, np.array([z ** (1 / alpha)]))[0]
    for mc, ex in ((c12, C[0, 1]), (n1, C[0, 0] - C[1, 1])):
        assert abs(mc.mean() - ex) < 4 * mc.std() / math.sqrt(n)


# ---------------------------------------------------------------- implicit L1 step of the nonlinear laws: the admissible root
def _first_step_roots_uniaxial(Wi, alpha, b, Theta, h):
    """All real solutions (f, C11, C22) of the first FENE-P step from C^0 = I in uniaxial extension (cubic in f)."""
    w = h ** (-alpha) / math.gamma(2 - alpha) / Theta ** alpha
    p1 = np.poly1d([1.0, w - 2 * Wi]); p2 = np.poly1d([1.0, w + Wi]); pf = np.poly1d([1.0, 0.0])
    cubic = pf * (b * p1 * p2 - (w + 1) * p2 - 2 * (w + 1) * p1) - (b - 3) * p1 * p2
    return [(f.real, (w + 1) / (w - 2 * Wi + f.real), (w + 1) / (w + Wi + f.real)) for f in cubic.roots if abs(f.imag) < 1e-9]


def test_fenep_step_of_section_5_5_three_solutions_one_admissible():
    # Wi = 20, alpha = 0.6, b = 50, uniform mesh with 256 steps on [0, 2 lambda]: the step is 2.87 times the bound of the linear law
    Wi, alpha, b, Th, N = 20.0, 0.6, 50.0, 2.0, 256
    roots = sorted(_first_step_roots_uniaxial(Wi, alpha, b, Th, 1.0 / N))
    assert len(roots) == 3
    adm = [r for r in roots if r[1] > 0 and r[2] > 0 and r[1] + 2 * r[2] < b]
    assert len(adm) == 1 and abs(adm[0][1] - 46.9018) < 1e-4 and abs(adm[0][2] - 0.35914) < 1e-5
    m = spd.Model('fenep', 'uniaxial', Wi, Th, b=b)
    t = np.linspace(0, 1, N + 1)
    assert 2.86 < (Th / N) / (math.gamma(2 - alpha) * (2 * Wi - 1)) ** (-1 / alpha) < 2.88 and spd.step_condition_lhs(m, t, alpha) > 1
    Cn = spd.solve_l1_newton(m, t, alpha)                       # converges at every step, to an indefinite tensor
    assert abs(Cn[1, 0, 0] + 1.1835) < 1e-4 and abs(Cn[1, 1, 1] - 0.52133) < 1e-5
    assert abs(np.linalg.eigvalsh(Cn)[:, 0].min() + 1.1835) < 1e-4
    assert any(abs(r[1] - Cn[1, 0, 0]) < 1e-8 and abs(r[2] - Cn[1, 1, 1]) < 1e-8 for r in roots)      # it is a solution of the step
    for mode in ('safeguarded', 'root'):
        C, info = spd.solve_l1(m, t, alpha, nonlinear=mode, return_info=True)
        assert abs(C[1, 0, 0] - adm[0][1]) < 1e-9 and abs(C[1, 1, 1] - adm[0][2]) < 1e-10
        lam = np.linalg.eigvalsh(C)[:, 0].min(); tr = np.trace(C, axis1=1, axis2=2).max()
        assert abs(lam - 0.021668) < 1e-5 and tr < spd.fenep_trace_bound(m) < b and abs(C[-1, 0, 0] - 48.7736) < 1e-4
        assert spd.step_residuals(m, t, alpha, C).max() < 1e-12
    assert info['root_steps'] == list(range(1, N + 1))
    assert spd.solve_l1(m, t, alpha, return_info=True)[1]['root_steps'] == [1]      # Newton fails at the first step only


def test_fenep_step_root_is_the_only_admissible_solution_random_flows():
    # g(f) = f (b - tr C(f)) - (b - 3) on f > max(0, f_c): negative, then increasing with exactly one zero
    rng = np.random.default_rng(3)
    for k in range(40):
        G = rng.standard_normal((3, 3)) * 10 ** rng.uniform(-1, 1.5)
        if k % 2:
            G -= np.trace(G) / 3 * np.eye(3)
        Y = rng.standard_normal((3, 3))
        w = 10 ** rng.uniform(-2, 3); kap = 10 ** rng.uniform(-1, 1); b = 3 + 10 ** rng.uniform(-1, 3)
        R = w * (Y @ Y.T * 10 ** rng.uniform(-2, 1) + 1e-3 * np.eye(3)) + kap * np.eye(3)
        C, info = spd.fenep_step_root(w, R, G, kap, b, return_info=True)
        f = (b - 3) / (b - np.trace(C))
        assert np.linalg.eigvalsh(C)[0] > 0 and np.trace(C) < b
        res = (w + kap * f) * C - kap * (G @ C + C @ G.T) - R
        assert np.abs(res).max() < 1e-10 * (np.abs(R).max() + (w + kap * f) * np.abs(C).max())
        fc = (2 * kap * np.linalg.eigvals(G).real.max() - w) / kap
        assert f > max(fc, 0.0)
        L9 = kap * spd.lyapunov_matrix(G)
        fs = max(fc, 0.0) + (f - max(fc, 0.0)) * np.logspace(-6, 6, 241)       # grid on (max(0, f_c), infinity) through the root
        g = []
        for x in fs:
            Cx = np.linalg.solve((w + kap * x) * np.eye(9) - L9, R.reshape(-1)).reshape(3, 3)
            assert np.linalg.eigvalsh(0.5 * (Cx + Cx.T))[0] > 0                  # C(f) is positive definite for f > f_c
            g.append(x * (b - np.trace(Cx)) - (b - 3))
        g = np.array(g)
        assert (g[fs < f * (1 - 1e-9)] < 0).all() and (g[fs > f * (1 + 1e-9)] > 0).all()
        assert (np.diff(g[fs > f]) > 0).all()
        # at and below f_c the Lyapunov equation has no positive definite solution
        for x in fc - (np.abs(fc) + 1.0) * np.array([1e-3, 0.3, 3.0]):
            try:
                Cx = np.linalg.solve((w + kap * x) * np.eye(9) - L9, R.reshape(-1)).reshape(3, 3)
            except np.linalg.LinAlgError:
                continue
            assert not np.linalg.eigvalsh(0.5 * (Cx + Cx.T))[0] > 0


def test_fenep_uniaxial_all_real_solutions_classified():
    # every real solution of the first step other than the one returned by the root solver is inadmissible
    rng = np.random.default_rng(5)
    for _ in range(200):
        Wi = 10 ** rng.uniform(-0.5, 2.3); alpha = rng.uniform(0.1, 1.0); b = 3 + 10 ** rng.uniform(-0.5, 2.5)
        Th = 2.0; h = 10 ** rng.uniform(-6, 0)
        roots = _first_step_roots_uniaxial(Wi, alpha, b, Th, h)
        adm = [r for r in roots if r[1] > 0 and r[2] > 0 and r[1] + 2 * r[2] < b]
        assert len(adm) == 1
        kap = Th ** alpha; w = h ** (-alpha) / math.gamma(2 - alpha)
        C = spd.fenep_step_root(w, (w + kap) * np.eye(3), Wi * spd.ghat('uniaxial'), kap, b)
        assert abs(C[0, 0] - adm[0][1]) < 1e-8 * adm[0][1] and abs(C[1, 1] - adm[0][2]) < 1e-8 * adm[0][2]


@pytest.mark.parametrize('flow,Wi', [('uniaxial', 2.0), ('uniaxial', 20.0), ('shear', 10.0), ('planar', 5.0)])
def test_fenep_admissible_iterates_satisfy_the_trace_bound(flow, Wi):
    for alpha in (0.3, 0.6, 0.9, 1.0):
        for N in (4, 32):
            m = spd.Model('fenep', flow, Wi, 2.0, b=50.0)
            t = np.linspace(0, 1, N + 1)
            for mode in ('safeguarded', 'root'):
                C = spd.solve_l1(m, t, alpha, nonlinear=mode)
                assert np.linalg.eigvalsh(C)[:, 0].min() > 0
                assert np.trace(C, axis1=1, axis2=2).max() <= spd.fenep_trace_bound(m) * (1 + 1e-12) < m.b
                assert spd.step_residuals(m, t, alpha, C).max() < 1e-11


def test_safeguarded_solver_reproduces_newton_where_newton_is_admissible():
    # graded meshes of the FENE-P benchmark: the Newton iterates are admissible, hence the admissible solution
    for Wi, alpha, N in [(2.0, 0.6, 128), (5.0, 0.6, 64), (5.0, 0.9, 128)]:
        m = spd.Model('fenep', 'uniaxial', Wi, 2.0, b=50.0)
        t = ref.graded_mesh(1.0, N, (2 - alpha) / alpha)
        Cn = spd.solve_l1_newton(m, t, alpha)
        Cs, info = spd.solve_l1(m, t, alpha, return_info=True)
        Cr = spd.solve_l1(m, t, alpha, nonlinear='root')
        assert np.array_equal(Cn, Cs) and info['root_steps'] == []
        assert np.abs(Cr - Cn).max() < 1e-11 * np.abs(Cr).max()
    with pytest.raises(ValueError):
        spd.solve_l1(m, t, alpha, nonlinear='other')


def test_giesekus_step_has_one_positive_definite_solution():
    # uniaxial extension: every diagonal entry solves a scalar quadratic with exactly one positive root
    m = spd.Model('giesekus', 'uniaxial', 5.0, 2.0, aG=0.2)
    for alpha, newton_fails in ((0.6, False), (0.9, True)):
        t = np.linspace(0, 1, 17)                               # violates the step condition of the linear law
        if newton_fails:
            with pytest.raises(RuntimeError):
                spd.solve_l1_newton(m, t, alpha)
        else:
            assert np.linalg.eigvalsh(spd.solve_l1_newton(m, t, alpha))[:, 0].min() < 0
        C, info = spd.solve_l1(m, t, alpha, return_info=True)
        assert np.linalg.eigvalsh(C)[:, 0].min() > 0 and info['root_steps'] == [1]
        assert spd.step_residuals(m, t, alpha, C).max() < 1e-12
        kap = m.Theta ** alpha; w = t[1] ** (-alpha) / math.gamma(2 - alpha)
        for i, gi in enumerate(np.diag(m.Wi * m.Gh)):
            a = kap * m.aG; bq = w + kap * (1 - 2 * m.aG) - 2 * kap * gi; cq = -(w + kap * (1 - m.aG))
            assert abs(C[1, i, i] - (-bq + math.sqrt(bq * bq - 4 * a * cq)) / (2 * a)) < 1e-11 * C[1, i, i]
    # general velocity gradient: residual of the Riccati equation and positive definiteness
    rng = np.random.default_rng(11)
    for _ in range(20):
        G = rng.standard_normal((3, 3)) * 10 ** rng.uniform(-1, 1.5); Y = rng.standard_normal((3, 3))
        w = 10 ** rng.uniform(-2, 2); kap = 10 ** rng.uniform(-1, 1); aG = rng.uniform(0.05, 1.0)
        R = w * (Y @ Y.T + 1e-3 * np.eye(3)) + kap * (1 - aG) * np.eye(3)
        C = spd.giesekus_step_root(w, R, G, kap, aG)
        res = kap * aG * C @ C + (w + kap * (1 - 2 * aG)) * C - kap * (G @ C + C @ G.T) - R
        assert np.linalg.eigvalsh(C)[0] > 0 and np.abs(res).max() < 1e-10 * (np.abs(R).max() + np.abs(C).max() ** 2)


def test_linear_law_step_condition_is_necessary_and_sufficient_for_general_flows():
    # iterates positive definite  <=>  w_nn > 2 Theta^alpha s(Wi Ghat - I/2) at every step (cases close to the bound excluded)
    rng = np.random.default_rng(17)
    checked = 0
    for _ in range(300):
        G = rng.standard_normal((3, 3)); G -= np.trace(G) / 3 * np.eye(3)
        Wi = 10 ** rng.uniform(-0.5, 1.5); Th = 2.0; alpha = rng.uniform(0.2, 1.0); N = int(rng.integers(2, 9))
        m = spd.Model('ucm', 'rest', Wi, Th, Ghat=G)
        t = np.linspace(0, 1, N + 1) ** rng.uniform(1.0, 3.0)
        lhs = [math.gamma(2 - alpha) * (Th * h) ** alpha * (2 * Wi * np.linalg.eigvals(G).real.max() - 1) for h in np.diff(t)]
        if min(abs(x - 1) for x in lhs) < 0.05:
            continue
        C = spd.solve_l1(m, t, alpha)
        lam = np.linalg.eigvalsh(C)[:, 0]
        assert (lam[1:].min() > 0) == (max(lhs) < 1)
        if max(lhs) > 1:                                        # the first iterate that is not positive definite
            first = next(n for n, x in enumerate(lhs, 1) if x > 1)
            assert lam[1:first].min(initial=1.0) > 0 and lam[first] <= 0
        checked += 1
    assert checked > 200


def test_bounded_trace_maps_return_the_initial_value_for_C0_identity():
    # the maps use S0 = C0 / (b - tr C0); with C0 in its place they would return b C0 / (1 + tr C0) = 12.5 I at tau = 0
    b = 50.0
    for variant in ('bfactor', 'blogm'):
        net = spd.SPDNet(variant, np.eye(3), 0.6, scale=20.0, b=b)
        X0 = net(torch.zeros(1)).detach().numpy()[0]
        assert np.abs(X0 - np.eye(3)).max() < 1e-13
        S0 = (net.T0 @ net.T0.T).numpy() if variant == 'bfactor' else torch.linalg.matrix_exp(net.Psi0).numpy()
        assert np.abs(S0 - np.eye(3) / (b - 3.0)).max() < 1e-15


def test_newton_with_full_steps_fails_on_meshes_within_the_step_bound_of_the_linear_law():
    # Wi = 20, alpha = 0.9, b = 10, 512 uniform steps on [0, 2 lambda]: left-hand side of the step bound 0.252
    m = spd.Model('fenep', 'uniaxial', 20.0, 2.0, b=10.0)
    alpha = 0.9
    t = np.linspace(0, 1, 513)
    assert 0.25 < spd.step_condition_lhs(m, t, alpha) < 0.26
    Cr = spd.solve_l1(m, t, alpha, nonlinear='root')
    Cf = spd.solve_l1_newton(m, t, alpha, full=True)            # converges at every step
    tr = np.trace(Cf, axis1=1, axis2=2)
    assert np.abs(Cf[:9] - Cr[:9]).max() < 1e-12 * np.abs(Cr[:9]).max()            # the admissible iterates up to step 8
    assert abs(tr[9] - 11.58034) < 1e-4 and abs(np.trace(Cr[9]) - 9.62322) < 1e-4  # step 9: an iterate passes tr C = b
    assert abs(Cf[9, 0, 0] - 10.75702) < 1e-4 and abs(Cr[9, 0, 0] - 8.89797) < 1e-4
    assert spd.step_residuals(m, t, alpha, Cf).max() < 1e-12                        # a solution of every step equation
    Cl = spd.solve_l1_newton(m, t, alpha)                       # steps halved so that tr C < b: the admissible iterates
    assert np.array_equal(Cl, spd.solve_l1(m, t, alpha)) and np.abs(Cl - Cr).max() < 1e-12 * np.abs(Cr).max()
    # shear: the step condition of the linear law holds on every mesh, and the full iteration still leaves tr C < b
    ms = spd.Model('fenep', 'shear', 100.0, 2.0, b=20.0)
    ts = np.linspace(0, 1, 9)
    assert spd.step_condition_lhs(ms, ts, 0.6) < 0
    Cfs = spd.solve_l1_newton(ms, ts, 0.6, full=True)
    assert np.trace(Cfs, axis1=1, axis2=2).max() > ms.b
    Cls = spd.solve_l1_newton(ms, ts, 0.6)
    assert np.linalg.eigvalsh(Cls)[:, 0].min() > 0 and np.trace(Cls, axis1=1, axis2=2).max() < ms.b


def test_failed_newton_iteration_reports_step_and_iterates():
    m = spd.Model('giesekus', 'uniaxial', 5.0, 2.0, aG=0.2)
    t = np.linspace(0, 1, 17)
    with pytest.raises(RuntimeError) as err:
        spd.solve_l1_newton(m, t, 0.9)
    e = err.value
    assert str(e) == f'Newton did not converge at step {e.step}'
    assert e.step >= 1 and e.iterates.shape == (e.step, 3, 3) and np.array_equal(e.iterates[0], np.eye(3))
    with pytest.raises(RuntimeError) as err2:                   # no trace limit for the Giesekus law: the variants coincide
        spd.solve_l1_newton(m, t, 0.9, full=True)
    assert err2.value.step == e.step and np.array_equal(err2.value.iterates, e.iterates)
    assert 'newton-full' in spd.NONLINEAR_SOLVERS

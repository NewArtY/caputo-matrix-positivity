"""Tests of the output maps, gates, generators, time stepper, metrics and campaign tools added for the factorial
benchmarks (B1, B2, B6, B7, surrogate). Run: python -m pytest -q tests/test_pinn_variants.py"""
import json, math, os, sys
import numpy as np
import pytest
import torch
from scipy.special import wofz
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn, layers as L, operators as op, reference as ref, exact
torch.set_default_dtype(torch.float64)
CD = torch.complex128
TOL = 1e-13


def rand_state(d, kind, seed=0):
    """Random density matrix: 'full' (full rank), 'pure' (rank one), 'rank2' (rank two, d >= 3)."""
    g = torch.Generator().manual_seed(seed)
    k = {'full': d, 'pure': 1, 'rank2': 2}[kind]
    A = torch.complex(torch.randn(d, k, generator=g), torch.randn(d, k, generator=g))
    S = A @ A.conj().T
    return S / torch.diagonal(S).real.sum()


def rand_raw(n, d, seed=1, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.complex(torch.randn(n, d, d, generator=g), torch.randn(n, d, d, generator=g))


def rand_gksl(d, seed=0, nl=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); H = (X + X.conj().T) / 2
    Ls = [0.5 * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) for _ in range(nl)]
    return H, Ls


TAU = torch.tensor([0.0, 1e-9, 1e-4, 0.03, 0.4, 1.0])
CASES = [(d, kind) for d in (2, 3, 4) for kind in ('full', 'pure', 'rank2') if not (d == 2 and kind == 'rank2')]


def apply_hard(name, M, tau, rho0, kind='pow', alpha=0.6):
    fr = L.state_frame(rho0); g = L.gate(tau, 5.0, kind, alpha); sg = L.sqrt_gate(tau, 5.0, kind, alpha)
    if name == 'bloch':
        M = L.bloch_head(M.real.reshape(M.shape[0], -1)[:, :fr['d'] ** 2 - 1], L.su_basis(fr['d']))
    return L.map_hard(name, M, g, fr, sg)


# ------------------------------------------------------------------ linear heads and bases

@pytest.mark.parametrize('d', [2, 3, 4])
def test_su_basis(d):
    G = L.su_basis(d)
    assert G.shape == (d * d - 1, d, d)
    assert torch.max(torch.abs(G - G.conj().transpose(-1, -2))) == 0
    assert torch.max(torch.abs(torch.einsum('kii->k', G))) < TOL
    gram = torch.einsum('aij,bji->ab', G, G)
    assert torch.max(torch.abs(gram - 2 * torch.eye(d * d - 1))) < TOL
    if d == 2:
        assert torch.equal(G, torch.stack([pinn.sx, pinn.sy, pinn.sz]))


def test_bloch_head_is_bloch_vector():
    r = torch.tensor([[0.3, -0.2, 0.5]])
    assert torch.max(torch.abs(pinn.I2 / 2 + L.bloch_head(r, L.su_basis(2)) - pinn.bloch_to_rho(r))) < TOL


# ------------------------------------------------------------------ admissibility of the maps

@pytest.mark.parametrize('d,kind', CASES)
@pytest.mark.parametrize('gate', ['exp', 'pow'])
@pytest.mark.parametrize('name', L.HARD_MAPS)
def test_hard_maps_admissible(name, gate, d, kind):
    rho0 = rand_state(d, kind); X = apply_hard(name, rand_raw(len(TAU), d), TAU, rho0, gate)
    a = L.admissibility(X)
    if name != 'unc':
        assert a['herm_defect'].max() < TOL
    if name in ('lin', 'bloch') + L.PSD_MAPS:
        assert a['trace_defect'].max() < TOL
    if name in L.PSD_MAPS:
        assert a['min_eig'].min() > -TOL and a['negativity'].max() < TOL


@pytest.mark.parametrize('d', [2, 3, 4])
@pytest.mark.parametrize('name', L.SOFT_MAPS)
def test_soft_maps_admissible(name, d):
    M = rand_raw(7, d)
    if name == 'bloch':
        M = L.bloch_head(M.real.reshape(7, -1)[:, :d * d - 1], L.su_basis(d))
    a = L.admissibility(L.map_soft(name, M))
    if name != 'unc':
        assert a['herm_defect'].max() < TOL
    if name in ('lin', 'bloch', 'factor'):
        assert a['trace_defect'].max() < TOL
    if name == 'factor':
        assert a['min_eig'].min() > -TOL


def test_linear_head_leaves_the_psd_cone():
    """The linear maps are Hermitian and of unit trace but not positive: a large output gives a negative eigenvalue."""
    rho0 = rand_state(3, 'pure'); X = apply_hard('lin', rand_raw(1, 3, scale=50.0), torch.tensor([0.5]), rho0)
    assert L.admissibility(X)['min_eig'].item() < -1e-3


# ------------------------------------------------------------------ exactness of the hard initial condition

@pytest.mark.parametrize('d,kind', CASES)
@pytest.mark.parametrize('gate', ['exp', 'pow'])
@pytest.mark.parametrize('name', L.HARD_MAPS)
def test_hard_ic_exact(name, gate, d, kind):
    rho0 = rand_state(d, kind); X0 = apply_hard(name, rand_raw(1, d), torch.zeros(1), rho0, gate)[0]
    if name in ('unc', 'herm', 'lin', 'bloch', 'add'):
        assert torch.equal(X0, rho0.to(CD))                       # bitwise
    else:
        assert torch.max(torch.abs(X0 - rho0)) < 1e-15            # T0 T0^+ = rho0 up to round-off


def test_hard_ic_exact_canonical_pure_state():
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=CD)
    for name in L.HARD_MAPS:
        assert torch.equal(apply_hard(name, rand_raw(1, 2), torch.zeros(1), rho0)[0], rho0)


@pytest.mark.parametrize('d', [2, 3, 4])
def test_two_gate_maps_reduce_to_factor_for_full_rank(d):
    rho0 = rand_state(d, 'full'); M = rand_raw(len(TAU), d)
    F = apply_hard('factor', M, TAU, rho0)
    for name in ('gate2', 'gate2b'):
        assert torch.max(torch.abs(apply_hard(name, M, TAU, rho0) - F)) < 1e-13


@pytest.mark.parametrize('d', [2, 3])
def test_growth_out_of_a_pure_state(d):
    """Kernel population ~ g^2 (factor) or ~ g (add, gate2, gate2b); range-kernel coherence ~ g, except ~ sqrt(g)
    for gate2. These are the orders stated in the docstring of fracsp.layers."""
    rho0 = torch.zeros(d, d, dtype=CD); rho0[0, 0] = 1                 # range = e_0, kernel = e_1, ...
    M = rand_raw(1, d, seed=3, scale=1.0).repeat(2, 1, 1); tau = torch.tensor([1e-6, 1e-8])
    g = L.gate(tau, 5.0, 'exp')
    for name, p_pop, p_coh in [('factor', 2.0, 1.0), ('add', 1.0, 1.0), ('gate2', 1.0, 0.5), ('gate2b', 1.0, 1.0)]:
        X = apply_hard(name, M, tau, rho0, 'exp')
        pop = X[:, 1, 1].real; coh = X[:, 0, 1].abs()
        assert abs(math.log(pop[0] / pop[1]) / math.log(g[0] / g[1]) - p_pop) < 1e-3, name
        assert abs(math.log(coh[0] / coh[1]) / math.log(g[0] / g[1]) - p_coh) < 1e-3, name


# ------------------------------------------------------------------ gates and gradients

def test_gate_values_and_alpha_gradient():
    a = torch.tensor(0.6, requires_grad=True); tau = torch.tensor([0.0, 1e-8, 0.2, 1.0])
    for fn in (L.gate, L.sqrt_gate):
        v = fn(tau, 5.0, 'pow', a)
        assert v[0] == 0 and torch.all(v[1:] > 0) and torch.all(v < 1)
        J = torch.autograd.functional.jacobian(lambda x: fn(tau, 5.0, 'pow', x), a)
        assert torch.isfinite(J).all() and J[0] == 0
        e = 1e-6
        fd = (fn(tau, 5.0, 'pow', torch.tensor(0.6 + e)) - fn(tau, 5.0, 'pow', torch.tensor(0.6 - e))) / (2 * e)
        assert torch.max(torch.abs(J - fd)) < 1e-8
    assert torch.max(torch.abs(L.sqrt_gate(tau, 5.0, 'exp') ** 2 - L.gate(tau, 5.0, 'exp'))) < 1e-15


@pytest.mark.parametrize('d,kind', [(2, 'full'), (2, 'pure'), (3, 'pure'), (3, 'rank2')])
@pytest.mark.parametrize('name', L.HARD_MAPS)
def test_gradcheck_hard_maps(name, d, kind):
    rho0 = rand_state(d, kind); fr = L.state_frame(rho0); tau = torch.tensor([1e-3, 0.3, 1.0])
    g = L.gate(tau, 5.0, 'pow', 0.6); sg = L.sqrt_gate(tau, 5.0, 'pow', 0.6); basis = L.su_basis(d)

    def f(raw):
        M = L.bloch_head(raw, basis) if name == 'bloch' else L.factor_from_raw(raw, d)
        X = L.map_hard(name, M, g, fr, sg)
        return torch.stack([X.real, X.imag])
    raw = torch.randn(3, d * d - 1 if name == 'bloch' else 2 * d * d, generator=torch.Generator().manual_seed(5),
                      requires_grad=True)
    assert torch.autograd.gradcheck(f, (raw,), eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('name', ['lin', 'factor', 'add', 'gate2', 'gate2b'])
def test_loss_gradient_with_respect_to_alpha(name):
    """d J / d alpha through the L1 weights, the pow gate and the singular feature against central differences."""
    torch.manual_seed(0)
    H, Ls = pinn.two_level_ops(2 * math.pi, 1.0, 0.5); S = torch.tensor(pinn.gksl_superop(H, Ls))
    model = pinn.DensityNet(pinn.bloch_to_rho(torch.tensor([0.0, 0.0, 1.0])), map=name, ic='hard', gate='pow', alpha=0.6,
                            feature='singular', width=8, depth=2)
    tcol = torch.tensor(ref.graded_mesh(1.0, 24, 2.0))

    def J(a):
        X = model(tcol, alpha=a)
        R = pinn.l1_residual(X, op.l1_nonuniform_matrix(tcol, a).to(CD), S)
        return (R.abs() ** 2).sum((-1, -2)).mean()
    a = torch.tensor(0.6, requires_grad=True); Ja = J(a); Ja.backward()
    e = 1e-6
    with torch.no_grad():
        fd = (J(torch.tensor(0.6 + e)) - J(torch.tensor(0.6 - e))) / (2 * e)
    assert torch.isfinite(a.grad) and abs(a.grad - fd) / abs(fd) < 1e-6


# ------------------------------------------------------------------ projection on the set of density matrices

@pytest.mark.parametrize('d', [2, 3, 4])
def test_project_to_density(d):
    X = rand_raw(50, d, scale=0.7); P = L.project_to_density(X); a = L.admissibility(P)
    assert a['herm_defect'].max() < TOL and a['trace_defect'].max() < TOL and a['min_eig'].min() > -TOL
    assert torch.max(torch.abs(L.project_to_density(P) - P)) < TOL                       # idempotent
    sig = torch.stack([rand_state(d, 'full', seed=s) for s in range(50)])
    assert torch.max(torch.abs(L.project_to_density(sig) - sig)) < TOL                   # identity on the set
    n = torch.linalg.matrix_norm
    assert torch.all(n(P - sig) <= n(X - sig) + TOL)                                     # non-expansive


def test_project_to_density_bloch_ball():
    r = torch.tensor([[0.0, 0.0, 1.7], [0.3, 0.1, -0.2], [2.0, -1.0, 0.5]])
    rb = pinn.rho_to_bloch(L.project_to_density(pinn.bloch_to_rho(r)))
    ex = r / torch.clamp(torch.linalg.norm(r, dim=-1, keepdim=True), min=1.0)
    assert torch.max(torch.abs(rb - ex)) < TOL


# ------------------------------------------------------------------ generators

def test_superop_equals_two_level_liouvillian():
    phys = dict(Omega=2 * math.pi, Gdown=1.0, Gphi=0.5); X = rand_raw(9, 2)
    S = torch.tensor(pinn.gksl_superop(*pinn.two_level_ops(**phys)))
    assert torch.max(torch.abs(pinn.apply_superop(S, X) - pinn.liouvillian(X, **phys))) < TOL


@pytest.mark.parametrize('d', [3, 4])
def test_superop_general_d(d):
    H, Ls = rand_gksl(d); S = torch.tensor(pinn.gksl_superop(H, Ls)); X = rand_raw(5, d)
    Ht = torch.tensor(H); out = -1j * (Ht @ X - X @ Ht)
    for Lk in Ls:
        Lt = torch.tensor(Lk); LdL = Lt.conj().T @ Lt
        out = out + Lt @ X @ Lt.conj().T - 0.5 * (LdL @ X + X @ LdL)
    Y = pinn.apply_superop(S, X)
    assert torch.max(torch.abs(Y - out)) < 1e-12
    assert torch.max(torch.abs(L.trace(Y))) < 1e-12                                      # trace preserving
    Xh = L.herm(X); Yh = pinn.apply_superop(S, Xh)
    assert torch.max(torch.abs(Yh - Yh.conj().transpose(-1, -2))) < 1e-12                # Hermiticity preserving
    Sb = S.expand(5, d * d, d * d)                                                       # batched generators
    assert torch.max(torch.abs(pinn.apply_superop(Sb, X) - Y)) < 1e-12


# ------------------------------------------------------------------ time stepper and exact solution

@pytest.mark.parametrize('alpha', [0.5, 0.8])
def test_stepper_is_the_zero_of_the_training_residual(alpha):
    d = 3; H, Ls = rand_gksl(d, seed=2); S = pinn.gksl_superop(H, Ls)
    rho0 = rand_state(d, 'pure').numpy(); t = ref.graded_mesh(1.0, 48, (2 - alpha) / alpha)
    X = pinn.l1_stepper(S, rho0.reshape(-1), t, alpha)
    # independent solver of the package (explicit history loops, own weights)
    _, Xr = ref.solve_l1(lambda tn: S[None], lambda tn: np.zeros((1, d * d)), rho0.reshape(1, -1), t, alpha)
    assert np.abs(X - Xr[:, 0]).max() < 1e-11
    R = pinn.l1_residual(torch.tensor(X.reshape(-1, d, d)), torch.tensor(pinn.l1_weight_matrix(t, alpha)).to(CD), torch.tensor(S))
    assert R.abs().max() < 1e-9
    a = L.admissibility(torch.tensor(X.reshape(-1, d, d)))                               # the scheme preserves the state space
    assert a['min_eig'].min() > -1e-13 and a['trace_defect'].max() < 1e-12 and a['herm_defect'].max() < 1e-12


def test_ml_exact_closed_form_alpha_half():
    """E_{1/2}(z) = exp(z^2) erfc(-z) = w(-i z): the matrix solution against the Faddeeva function."""
    d = 3; H, Ls = rand_gksl(d, seed=4); S = pinn.gksl_superop(H, Ls); rho0 = rand_state(d, 'pure').numpy().reshape(-1)
    t = np.array([0.0, 1e-6, 0.01, 0.3, 1.0])
    X, info = pinn.ml_exact(S, rho0, t, 0.5)
    lam, V = info['lam'], info['V']
    Xw = (wofz(-1j * np.sqrt(t)[:, None] * lam[None, :]) * np.linalg.solve(V, rho0)[None, :]) @ V.T
    assert info['cond_V'] < 1e3 and info['eig_residual'] < 1e-12
    assert np.abs(X - Xw).max() < 1e-11
    assert np.abs(X[0] - rho0).max() < 1e-13


def test_ml_exact_equals_bloch_exact_and_stepper_converges():
    alpha = 0.6; ph = dict(Omega=2 * math.pi, Gdown=1.0, Gphi=0.5)
    S = pinn.gksl_superop(*pinn.two_level_ops(**ph)); r0 = np.array([0.3, 0.2, 0.4])
    rho0 = pinn.bloch_to_rho(torch.tensor(r0)).numpy().reshape(-1); t = np.array([1e-5, 0.02, 0.5, 1.0])
    X, _ = pinn.ml_exact(S, rho0, t, alpha)
    G2 = ph['Gdown'] / 2 + ph['Gphi']
    A = np.array([[-G2, 0, 0], [0, -G2, -ph['Omega']], [0, ph['Omega'], -ph['Gdown']]]); b = np.array([0, 0, -ph['Gdown']])
    rb = exact.bloch_exact(A, b, r0, t, alpha)
    assert np.abs(pinn.rho_to_bloch(torch.tensor(X.reshape(-1, 2, 2))).numpy() - rb).max() < 1e-13
    errs = []
    for N in (32, 128):
        tn = ref.graded_mesh(1.0, N, (2 - alpha) / alpha)
        errs.append(np.abs(pinn.interp_nodes(t[1:], tn, pinn.l1_stepper(S, rho0, tn, alpha)) - X[1:]).max())
    assert errs[1] < errs[0] / 4 and errs[1] < 2e-3


def test_interp_nodes():
    t = np.array([0.0, 0.1, 0.5, 1.0]); X = np.stack([2 * t, 1 - t], 1)
    tq = np.array([0.0, 0.05, 0.1, 0.7, 1.0])
    assert np.abs(pinn.interp_nodes(tq, t, X) - np.stack([2 * tq, 1 - tq], 1)).max() < 1e-15


# ------------------------------------------------------------------ metrics

def test_grid_and_metrics():
    t, iu, il = pinn.test_grid(1000, -12, 40)
    assert t[0] == 0 and t[-1] == 1 and np.all(np.diff(t) > 0) and iu.sum() == 1000 and il.sum() == 481
    assert abs(pinn.trapz_weights(t).sum() - 1) < 1e-14
    rho0 = np.diag([1.0, 0.0]).astype(complex)
    p = 0.5 * t ** 0.6                                                        # population leaving the initial state
    Xref = np.zeros((len(t), 2, 2), complex); Xref[:, 0, 0] = 1 - p; Xref[:, 1, 1] = p
    m = pinn.state_metrics(Xref, Xref, t, iu, il, rho0)
    assert m['rel_fro'] == 0 and m['max_fro'] == 0 and m['min_eig'] == 0 and m['neg_time_frac'] == 0
    X = Xref.copy(); X[:, 0, 0] += 0.01; X[:, 1, 1] -= 0.02                    # negative population at early times
    m = pinn.state_metrics(X, Xref, t, iu, il, rho0)
    assert abs(m['max_fro'] - math.sqrt(5e-4)) < 1e-12 and abs(m['max_trace_defect'] - 0.01) < 1e-12
    assert abs(m['min_eig'] + 0.02) < 1e-12 and m['tau_min_eig'] == 0
    tneg = (0.02 / 0.5) ** (1 / 0.6)                                           # p(t) < 0.02 for t < tneg
    assert abs(m['neg_time_frac'] - tneg) < 2e-3 and m['neg_log_frac'] > 0.8
    assert m['rel_fro_proj'] < m['rel_fro'] and m['max_herm_defect'] == 0


# ------------------------------------------------------------------ models, training and provenance

def test_density_net_argument_checks():
    rho0 = rand_state(2, 'full')
    with pytest.raises(ValueError):
        pinn.DensityNet(rho0, map='add', ic='soft')
    with pytest.raises(ValueError):
        pinn.DensityNet(rho0, map='factor', ic='hard', gate='pow')            # alpha missing
    assert sum(p.numel() for p in pinn.DensityNet(rho0, map='factor').parameters()) == 2440
    assert sum(p.numel() for p in pinn.DensityNet(rho0, map='bloch').parameters()) == 2275


def test_first_campaign_variants_are_members_of_the_family():
    """Model('sp') of the first campaign is DensityNet(map='factor', ic='hard', gate='exp') with the same weights."""
    rho0 = pinn.bloch_to_rho(torch.tensor([0.0, 0.0, 1.0])); tau = torch.linspace(0, 1, 11)
    for old, kw in [('sp', dict(map='factor', ic='hard', gate='exp')), ('sp_soft', dict(map='factor', ic='soft')),
                    ('direct', dict(map='unc', ic='soft'))]:
        torch.manual_seed(3); m_old = pinn.Model(old, rho0)
        torch.manual_seed(3); m_new = pinn.DensityNet(rho0, **kw)
        assert torch.max(torch.abs(m_old(tau) - m_new(tau))) < 1e-14


def _tiny_run(seed, map='add'):
    alpha = 0.6; S = pinn.gksl_superop(*pinn.two_level_ops(2 * math.pi, 1.0, 0.5))
    rho0 = np.diag([1.0, 0.0]).astype(complex); tcol = ref.graded_mesh(1.0, 16, (2 - alpha) / alpha)
    t, iu, il = pinn.test_grid(20, -6, 4)
    Xl = pinn.l1_stepper(S, rho0.reshape(-1), tcol, alpha)
    test = dict(t=t, is_uniform=iu, is_log=il, ref=pinn.interp_nodes(t, tcol, Xl).reshape(-1, 2, 2),
                ref_nodes=Xl.reshape(-1, 2, 2), l1_nodes=Xl.reshape(-1, 2, 2))
    res, X = pinn.forward_run(S, rho0, alpha, tcol, dict(map=map, ic='hard', gate='pow'), dict(width=8, depth=2, beta=5.0),
                              dict(adam_iters=30, lbfgs_iters=5, lr=2e-3), seed, test)
    return res, X


def test_forward_run_is_deterministic_and_admissible():
    r1, X1 = _tiny_run(7); r2, X2 = _tiny_run(7); r3, X3 = _tiny_run(8)
    assert np.array_equal(X1, X2) and r1['rel_fro'] == r2['rel_fro'] and not np.array_equal(X1, X3)
    assert r1['status'] == 'ok' and r1['ic_error'] == 0 and r1['min_eig'] > -TOL and r1['max_trace_defect'] < TOL
    json.dumps(r1)                                                             # the record is serialisable


def test_fit_reports_non_finite_loss():
    p = torch.nn.Parameter(torch.tensor(1.0))
    info = pinn.fit([p], lambda: {'phys': p * float('nan')}, 5, 5, 1e-3)
    assert info['status'].startswith('non-finite')


def test_run_metadata_and_memory():
    m = pinn.run_metadata(seed=3, config=dict(a=1), driver=__file__)
    for k in ('timestamp_utc', 'python', 'torch', 'numpy', 'scipy', 'platform', 'seed', 'config', 'code_sha256'):
        assert m[k] is not None
    assert m['seed'] == 3 and m['driver'] == os.path.basename(__file__)
    assert {'pinn.py', 'layers.py', 'operators.py', os.path.basename(__file__)} <= set(m['code_sha256'])
    assert all(len(v) == 16 for v in m['code_sha256'].values())
    rss = pinn.peak_rss_mb()
    assert rss is None or 10 < rss < 1e5


def _good_job(job):
    pinn.write_json(job['out'], dict(value=job['x'] ** 2))
    return 'ok'


def _bad_job(job):
    if job['x'] == 2:
        raise RuntimeError('deliberate failure')
    if job['x'] == 3:
        return 'no result file written'
    pinn.write_json(job['out'], dict(value=job['x']))


def test_run_campaign_fails_loudly(tmp_path):
    jobs = [dict(name=f'j{x}', out=str(tmp_path / f'j{x}.json'), x=x) for x in range(4)]
    log = str(tmp_path / 'campaign.log')
    assert pinn.run_campaign(_good_job, jobs[:2], 2, log, 'test') == 0
    assert json.load(open(jobs[1]['out']))['value'] == 1
    nbad = pinn.run_campaign(_bad_job, jobs, 2, log, 'test')                   # j0, j1 are skipped (done), j2 raises, j3 writes nothing
    assert nbad == 2
    assert 'deliberate failure' in open(jobs[2]['out'] + '.err.txt').read()
    txt = open(log).read()
    assert 'FAILED j2' in txt and 'MISSING result file of j3' in txt and '2 skipped' in txt


# ------------------------------------------------------------------ experiment drivers (B1, B7, surrogate)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))


def test_b1_trial_functions_and_discrete_optima():
    import b1_floor as F, b1_scalar as B1
    t = torch.tensor([0.0, 1e-6, 0.3, 1.0]); a = torch.tensor(0.5)
    for inp in ('plain', 'singular'):
        for gate in ('exp', 'pow'):
            torch.manual_seed(0); u = B1.Net(5.0, inp, gate)(t, a)
            assert u[0] == 1.0 and torch.isfinite(u).all()                      # hard initial condition u(0) = 1
    # data generated by the scheme itself are fitted exactly: both discrete optima return the generating order
    tn = ref.graded_mesh(1.0, 48, 3.0); tob = np.array([0.05, 0.2, 0.33, 0.61, 0.9]); a0 = 0.55
    y = F.interp_matrix(tob, tn) @ F.stepper(tn, a0)
    als, Jls = F.alpha_ls(tob, y, tn); ajo, Jjo = F.alpha_joint(tob, y, tn, 100.0)
    assert abs(als - a0) < 1e-7 and Jls < 1e-14 and abs(ajo - a0) < 1e-6 and Jjo < 1e-12
    assert np.abs(F.interp_matrix(tob, tn) @ tn - tob).max() < 1e-15


def test_b1_least_squares_on_finer_meshes():
    import b1_floor as F, b1_ls_mesh as M
    cfg = dict(F.cfg, alphas_true=[0.5], seeds=[0], mesh=dict(N=16, r=3.0), inverse=dict(F.cfg['inverse'], noise_sd=[0.0]))
    rows = M.ls_rows(cfg, factors=(1, 4), noisy_up_to=1)
    assert [r[2] for r in rows] == [16, 64] and all(r[3] == 0.0 and r[4] == 0 for r in rows)
    e = [abs(r[5] - 0.5) for r in rows]                 # distance of the noise-free estimate from the true order
    assert e[1] < 0.5 * e[0] < 0.02                     # the discretisation bias decreases under refinement


def test_b7_generator_and_least_squares_recovery():
    from scipy.optimize import least_squares
    import b7_inverse as B7
    th = B7.THETA
    S = pinn.gksl_superop(*pinn.two_level_ops(B7.cfg['phys_true']['Omega'], th[1], th[2]))
    assert np.abs(B7.generator(th) - S).max() < 1e-14
    y = B7.project_data(np.array([[3.0, 4.0], [0.3, 0.4]]), 'xz')
    assert np.allclose(y, [[0.6, 0.8], [0.3, 0.4]]) and np.allclose(B7.project_data(np.array([[1.2], [-0.1], [0.4]]), 'pop'), [[1.0], [0.0], [0.4]])
    X = pinn.bloch_to_rho(torch.tensor([[0.1, -0.2, 0.3]])).numpy()
    assert np.allclose(B7.observe(X, 'xz'), [[0.1, 0.3]]) and np.allclose(B7.observe(X, 'pop'), [[0.65]])
    assert np.allclose(B7.OBS['sy'](X), [-0.2])
    fwd = B7.Stepper('xz', 32, True); y = fwd(th)                             # noise-free data from the scheme itself
    r = least_squares(lambda x: (fwd(x) - y).ravel(), [0.8, 0.5, 1.0], jac='3-point', bounds=([0.05, 1e-3, 1e-3], [1.0, 50, 50]),
                      x_scale='jac', xtol=1e-12, ftol=1e-12, gtol=1e-12)
    assert np.abs(r.x - th).max() < 1e-6
    a = pinn.L.admissibility(torch.tensor(fwd.states(th)))                     # the fitted trajectory is a trajectory of states
    assert a['min_eig'].min() > -1e-13 and a['trace_defect'].max() < 1e-12


def test_surrogate_batched_residual():
    import b2_surrogate as SG
    torch.manual_seed(1)
    model = pinn.DensityNet(torch.tensor(SG.RHO0), map='add', ic='hard', gate='pow', alpha=0.6, width=8, depth=2, n_par=2)
    al, gd, tau, W, S = SG.pool(); idx = torch.tensor([3, 17, 200])
    J = SG.residual_loss(model, idx); terms = []
    for k in idx:
        a = al[k].expand(tau.shape[1]); par = SG.scale(a, gd[k].expand(tau.shape[1]))
        X = model(tau[k], alpha=a, par=par)
        R = pinn.l1_residual(X, op.l1_nonuniform_matrix(tau[k], float(al[k])).to(CD), S[k])
        terms.append((R.abs() ** 2).sum((-1, -2)).mean())
    assert abs(J - torch.stack(terms).mean()) < 1e-12 * abs(J)
    P = SG.test_params(); assert P.shape == (64, 2) and SG.scale(torch.tensor(P[:, 0]), torch.tensor(P[:, 1])).abs().max() <= 1 + 1e-12


def test_b6_systems_stay_on_or_near_the_boundary():
    import b6_prepare as B6
    for name, d in (('qutrit', 3), ('two_qubit', 4)):
        H, Ls, rho0 = B6.system(name); S = pinn.gksl_superop(H, Ls)
        assert np.abs(H - H.conj().T).max() == 0 and abs(np.trace(rho0) - 1) == 0 and np.linalg.matrix_rank(rho0) == 1
        assert np.abs(np.eye(d).reshape(-1) @ S).max() < 1e-14                  # trace preserving: vec(I)^T S = 0
        t = ref.graded_mesh(1.0, 64, 3.0); X = pinn.l1_stepper(S, rho0.reshape(-1), t, 0.5).reshape(-1, d, d)
        ev = np.linalg.eigvalsh(0.5 * (X + X.conj().transpose(0, 2, 1)))
        assert ev.min() > -1e-14 and np.abs(np.trace(X, axis1=1, axis2=2) - 1).max() < 1e-13
        if name == 'two_qubit':                                                # |ee> is never populated: an exact zero eigenvalue
            assert np.abs(X[:, 0, :]).max() == 0 and np.abs(X[:, :, 0]).max() == 0 and np.abs(ev[:, 0]).max() < 1e-15
        else:                                                                  # the third level is weakly populated
            assert 0 < ev[-1, 0] < 0.03 and X[-1, 2, 2].real < 0.1

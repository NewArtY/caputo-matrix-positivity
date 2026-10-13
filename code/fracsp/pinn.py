"""Fractional PINN variants for matrix-valued Caputo problems  D^alpha rho = L[rho]  (dimensionless time, the
prefactor tau_c^{1-alpha} is absorbed in the generator).

The residual uses the nonuniform L1 operator on the collocation mesh (discrete Caputo history).

Contents
  MLP, liouvillian, bloch_to_rho, rho_to_bloch      building blocks of the two-level benchmark
  Model, loss_terms, train                          the four variants of the first campaign (direct, penalty, sp_soft,
                                                    sp), kept so that the superseded results can be regenerated
  gksl_superop, apply_superop, two_level_ops        GKSL generators for arbitrary dimension d (superoperator form)
  DensityNet, forward_loss, fit, forward_run        factorial family: output map x initial condition x gate
  l1_weight_matrix, l1_stepper, stepper_rows        implicit L1 time stepper on the collocation mesh (classical reference)
  ml_table, ml_exact, reference_jobs, ...           exact solution E_alpha(t^alpha L) rho0 for a constant generator
  test_grid, trapz_weights, state_metrics           error and admissibility metrics on a grid that resolves the initial layer
  run_metadata, peak_rss_mb, run_campaign           provenance record and campaign launcher that fails loudly
"""
import datetime, hashlib, json, math, os, platform, sys, time, traceback
import numpy as np
import torch
from . import operators as op
from . import layers as L
from . import __version__

DT = torch.float64
CD = torch.complex128
sx = torch.tensor([[0, 1], [1, 0]], dtype=CD)
sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CD)
sz = torch.tensor([[1, 0], [0, -1]], dtype=CD)
I2 = torch.eye(2, dtype=CD)
sm = torch.tensor([[0, 0], [1, 0]], dtype=CD)   # |g><e| with |e>=(1,0): lowering


class MLP(torch.nn.Module):
    def __init__(self, width=32, depth=3, nout=8, nin=1):
        super().__init__()
        layers = []; d = nin
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width, dtype=DT), torch.nn.Tanh()]; d = width
        layers += [torch.nn.Linear(d, nout, dtype=DT)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def liouvillian(rho, Omega, Gdown, Gphi):
    """Matrix-form GKSL generator: H = (Omega/2) sx, L1 = sqrt(Gdown) sm, Lphi = sqrt(Gphi/2) sz."""
    H = 0.5 * Omega * sx
    out = -1j * (H @ rho - rho @ H)
    for Lr, g in [(sm, Gdown), (sz, Gphi / 2)]:
        LdL = Lr.conj().T @ Lr
        out = out + g * (Lr @ rho @ Lr.conj().T - 0.5 * (LdL @ rho + rho @ LdL))
    return out


def bloch_to_rho(r):
    r = torch.as_tensor(r, dtype=DT)
    return 0.5 * (I2 + r[..., 0, None, None] * sx + r[..., 1, None, None] * sy + r[..., 2, None, None] * sz)


def rho_to_bloch(rho):
    return torch.stack([torch.einsum('...ij,ji->...', rho, s).real for s in (sx, sy, sz)], -1)


# ----------------------------------------------------------------------------------------------
# Variants of the first campaign (superseded by DensityNet; kept for traceability)
# ----------------------------------------------------------------------------------------------

class Model(torch.nn.Module):
    """direct: unconstrained output, soft IC; penalty: direct + penalties; sp_soft: factor map, soft IC;
    sp: factor map with hard IC at the factor level, gate 1 - exp(-beta tau). In the factorial family these are
    DensityNet(map='unc', ic='soft'), -, DensityNet(map='factor', ic='soft'), DensityNet(map='factor', ic='hard')."""
    def __init__(self, variant, rho0, width=32, depth=3, beta=5.0):
        super().__init__()
        self.variant = variant; self.net = MLP(width, depth, 8)
        self.rho0 = rho0; self.T0 = L.factor_of_state(rho0); self.beta = beta

    def forward(self, tau):
        raw = self.net(tau[:, None])
        M = L.factor_from_raw(raw, 2)
        if self.variant in ('direct', 'penalty'):
            return M
        if self.variant == 'sp_soft':
            return L.normalized_factor_map(M)
        if self.variant == 'sp':
            return L.normalized_factor_map(L.hard_ic_factor(self.T0, M, tau, self.beta))
        raise ValueError(self.variant)


def loss_terms(model, tcol, W, alpha, phys, pen_weight=0.0):
    X = model(tcol)                                    # (N+1,2,2)
    D = op.l1_nonuniform(X, W.to(CD))                  # (N,2,2)
    R = D - liouvillian(X[1:], **phys)
    J = {'phys': (R.abs() ** 2).sum((-1, -2)).mean()}
    if model.variant != 'sp':
        J['ic'] = ((X[0] - model.rho0).abs() ** 2).sum()
    if model.variant == 'penalty':
        herm = ((X - X.conj().transpose(-1, -2)).abs() ** 2).sum((-1, -2)).mean()
        tr = (torch.diagonal(X, dim1=-2, dim2=-1).sum(-1) - 1).abs().pow(2).mean()
        Hm = 0.5 * (X + X.conj().transpose(-1, -2))
        lmin = torch.linalg.eigvalsh(Hm)[:, 0]
        neg = torch.relu(-lmin).pow(2).mean()
        J['pen'] = pen_weight * (herm + tr + neg)
    return J


def train(model, tcol, W, alpha, phys, adam_iters, lbfgs_iters, lr, pen_weight=0.0, log_every=500):
    return fit(model.parameters(), lambda: loss_terms(model, tcol, W, alpha, phys, pen_weight), adam_iters, lbfgs_iters,
               lr, log_every)


# ----------------------------------------------------------------------------------------------
# GKSL generators for arbitrary d (superoperator on the row-major vectorisation)
# ----------------------------------------------------------------------------------------------

def gksl_superop(H, Ls):
    """Matrix S with vec(L[rho]) = S vec(rho), vec = row-major, for
    L[rho] = -i[H, rho] + sum_k (L_k rho L_k^+ - {L_k^+ L_k, rho}/2); rates are absorbed in the jump operators."""
    H = np.asarray(H, dtype=complex); d = H.shape[0]; I = np.eye(d)
    S = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for Lk in Ls:
        Lk = np.asarray(Lk, dtype=complex); LdL = Lk.conj().T @ Lk
        S = S + np.kron(Lk, Lk.conj()) - 0.5 * (np.kron(LdL, I) + np.kron(I, LdL.T))
    return S


def apply_superop(S, X):
    """L[X] for a batch X (..., d, d) and a superoperator S (d^2, d^2), or a batch of them broadcastable to X."""
    d = X.shape[-1]
    v = X.reshape(X.shape[:-2] + (d * d,))
    return (S @ v[..., None])[..., 0].reshape(X.shape) if S.dim() > 2 else (v @ S.T).reshape(X.shape)


def two_level_ops(Omega, Gdown, Gphi):
    """H and jump operators of the two-level benchmark (same generator as `liouvillian`)."""
    return 0.5 * Omega * sx.numpy(), [math.sqrt(Gdown) * sm.numpy(), math.sqrt(Gphi / 2) * sz.numpy()]


# ----------------------------------------------------------------------------------------------
# Factorial family: output map x initial condition x gate
# ----------------------------------------------------------------------------------------------

class DensityNet(torch.nn.Module):
    """X_theta(tau) = map(M_theta(features)) for a d x d state; the maps are defined in fracsp.layers.

    ic='hard'   : X(0) = rho0 for every parameter vector through the gate g (kind 'exp' or 'pow');
    ic='soft'   : no gate, the initial condition is a loss term (see forward_loss);
    feature     : 'plain' (network input tau) or 'singular' (inputs tau and tau^alpha);
    n_par       : number of additional inputs (parametric surrogate);
    alpha       : order used by the 'pow' gate and the singular feature; it can be overridden in forward(), e.g. by a
                  trainable tensor in inverse problems or by one value per point in a parametric surrogate.
    All maps use the same backbone and the same 2 d^2 real outputs, except 'bloch' (d^2 - 1 outputs)."""
    def __init__(self, rho0, map='factor', ic='hard', gate='exp', alpha=None, beta=5.0, width=32, depth=3,
                 feature='plain', n_par=0):
        super().__init__()
        if map not in (L.HARD_MAPS if ic == 'hard' else L.SOFT_MAPS):
            raise ValueError(f'map {map!r} is not defined for ic={ic!r}')
        if (gate == 'pow' and ic == 'hard' or feature == 'singular') and alpha is None:
            raise ValueError('alpha is required by the pow gate and by the singular feature')
        self.map, self.ic, self.gate_kind, self.alpha, self.beta, self.feature = map, ic, gate, alpha, beta, feature
        self.fr = L.state_frame(torch.as_tensor(rho0)); self.d = d = self.fr['d']
        self.basis = L.su_basis(d) if map == 'bloch' else None
        self.net = MLP(width, depth, d * d - 1 if map == 'bloch' else 2 * d * d,
                       nin=(2 if feature == 'singular' else 1) + n_par)

    def raw(self, tau, alpha=None, par=None):
        a = self.alpha if alpha is None else alpha
        cols = [tau, tau ** a] if self.feature == 'singular' else [tau]
        x = torch.stack(cols, -1)
        if par is not None:
            x = torch.cat([x, par], -1)
        out = self.net(x)
        return L.bloch_head(out, self.basis) if self.map == 'bloch' else L.factor_from_raw(out, self.d)

    def forward(self, tau, alpha=None, par=None):
        a = self.alpha if alpha is None else alpha
        M = self.raw(tau, a, par)
        if self.ic == 'soft':
            return L.map_soft(self.map, M)
        g = L.gate(tau, self.beta, self.gate_kind, a)
        sg = L.sqrt_gate(tau, self.beta, self.gate_kind, a) if self.map in ('gate2', 'gate2b') else None
        return L.map_hard(self.map, M, g, self.fr, sg)


def l1_residual(X, W, S):
    """Discrete residual R_n = (D_h^alpha X)_n - L[X_n], n = 1..N, of a trajectory X (N+1, d, d) on the collocation
    mesh; W is the complex (N, N) L1 weight matrix, S the superoperator of the generator."""
    return op.l1_nonuniform(X, W) - apply_superop(S, X[1:])


def forward_loss(model, tcol, W, S, ic_weight=1.0):
    """Mean squared Frobenius norm of the L1 residual over the collocation nodes, plus (soft IC only) the term
    ic_weight * ||X(0) - rho0||_F^2."""
    X = model(tcol)
    R = l1_residual(X, W, S)
    J = {'phys': (R.abs() ** 2).sum((-1, -2)).mean()}
    if model.ic == 'soft':
        J['ic'] = ic_weight * ((X[0] - model.fr['rho0']).abs() ** 2).sum()
    return J


def fit(params, loss_fn, adam_iters, lbfgs_iters, lr, log_every=500):
    """Adam (cosine schedule to 5 % of lr) followed by L-BFGS (strong Wolfe). loss_fn() returns a dict of terms.
    A non-finite loss is not hidden: training stops and the returned status says where it happened."""
    params = list(params); hist = []; status = 'ok'
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters, eta_min=lr * 0.05)
    t0 = time.perf_counter(); c0 = time.process_time()
    for it in range(adam_iters):
        opt.zero_grad()
        loss = sum(loss_fn().values())
        if not torch.isfinite(loss):
            status = f'non-finite loss at Adam iteration {it}'; break
        loss.backward(); opt.step(); sched.step()
        if it % log_every == 0:
            hist.append((it, loss.item()))
    t_adam = time.perf_counter() - t0
    nev = [0]; t1 = time.perf_counter()
    if lbfgs_iters > 0 and status == 'ok':
        opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters, history_size=50, tolerance_grad=1e-12,
                                 tolerance_change=1e-15, line_search_fn='strong_wolfe')

        def closure():
            opt2.zero_grad()
            loss = sum(loss_fn().values()); loss.backward(); nev[0] += 1
            return loss
        opt2.step(closure)
    t_lbfgs = time.perf_counter() - t1
    with torch.no_grad():
        final = {k: v.item() for k, v in loss_fn().items()}
    if status == 'ok' and not all(math.isfinite(v) for v in final.values()):
        status = 'non-finite loss after L-BFGS'
    return dict(status=status, history=hist, final_terms=final, t_adam=t_adam, t_lbfgs=t_lbfgs, lbfgs_evals=nev[0],
                wall=t_adam + t_lbfgs, cpu=time.process_time() - c0)


# ----------------------------------------------------------------------------------------------
# Classical reference: implicit L1 time stepper on the collocation mesh, and the exact solution
# ----------------------------------------------------------------------------------------------

def l1_weight_matrix(t, alpha):
    """(N, N) weight matrix of the nonuniform L1 formula (numpy): the operator of the training residual."""
    return op.l1_nonuniform_matrix(torch.as_tensor(np.asarray(t), dtype=DT), float(alpha)).numpy()


def l1_stepper(S, x0, t, alpha, W=None):
    """Implicit L1 scheme for D^alpha x = S x, x(0) = x0, with a constant matrix S on the mesh t; the weights are
    tabulated once. Returns the nodal values (N+1, m). For a density matrix, x = vec(rho) and S = gksl_superop(...);
    the result is the unique zero of the discrete residual that the networks minimise (l1_residual)."""
    S = np.asarray(S); x0 = np.asarray(x0).reshape(-1); N = len(t) - 1; m = x0.size
    if W is None:
        W = l1_weight_matrix(t, alpha)
    X = np.zeros((N + 1, m), dtype=np.result_type(S.dtype, x0.dtype, float)); X[0] = x0
    dX = np.zeros_like(X[1:])
    Minv = np.linalg.inv(np.diagonal(W)[:, None, None] * np.eye(m) - S)
    for n in range(1, N + 1):
        rhs = W[n - 1, n - 1] * X[n - 1]
        if n > 1:
            rhs = rhs - W[n - 1, :n - 1] @ dX[:n - 1]
        X[n] = Minv[n - 1] @ rhs
        dX[n - 1] = X[n] - X[n - 1]
    return X


def interp_nodes(tq, t, X):
    """Piecewise-linear interpolation in time of nodal values X (N+1, m) (the interpolant underlying the L1 formula)."""
    j = np.clip(np.searchsorted(t, tq, side='right') - 1, 0, len(t) - 2)
    s = ((tq - t[j]) / (t[j + 1] - t[j]))[:, None]
    return (1 - s) * X[j] + s * X[j + 1]


def ml_table(lams, t, alpha):
    """E_alpha(lam_k t_j^alpha), array (len(t), len(lams)), from the multiprecision series fracsp.exact.ml_complex.
    Complex-conjugate arguments are evaluated once."""
    from . import exact
    lams = np.asarray(lams, dtype=complex); t = np.asarray(t, dtype=float)
    E = np.zeros((len(t), len(lams)), dtype=complex); done = {}
    for k, lam in enumerate(lams):
        key = (round(lam.real, 12), round(abs(lam.imag), 12))
        if key in done:
            kk, conj = done[key], (lam.imag * lams[done[key]].imag < 0)
            E[:, k] = E[:, kk].conj() if conj else E[:, kk]
            continue
        done[key] = k
        E[:, k] = [1.0 if tj == 0 else exact.ml_complex(lam * tj ** alpha, alpha) for tj in t]
    return E


def ml_exact(S, x0, t, alpha, eig=None, table=None):
    """Exact solution x(t) = E_alpha(t^alpha S) x0 of D^alpha x = S x for a constant diagonalisable S, through the
    eigen-decomposition S = V diag(lam) V^{-1}. Returns (X, info) with X of shape (len(t), m); info holds lam, V,
    the table E_alpha(lam_k t_j^alpha), the condition number of V and the residual of the decomposition, which
    bound the round-off amplification and must be checked by the caller."""
    S = np.asarray(S, dtype=complex)
    lam, V = eig if eig is not None else np.linalg.eig(S)
    if table is None:
        table = ml_table(lam, t, alpha)
    c = np.linalg.solve(V, np.asarray(x0, dtype=complex).reshape(-1))
    X = (table * c[None, :]) @ V.T
    info = dict(lam=lam, V=V, table=table, cond_V=float(np.linalg.cond(V)),
                eig_residual=float(np.abs(V @ np.diag(lam) - S @ V).max()))
    return X, info


def reference_jobs(path, S, tcol, t, alpha, nchunks):
    """Jobs (for run_campaign with ml_table_job) that tabulate E_alpha(lam_k tau^alpha) on the test grid t and at
    the nodes tcol in nchunks chunks of times; assemble_reference(path, ...) collects them into the file `path`."""
    lam = np.linalg.eig(np.asarray(S, dtype=complex))[0]
    tall = np.concatenate([t, tcol]); order = np.argsort(-tall)             # the expensive late times are spread evenly
    base = os.path.splitext(path)[0]
    return [dict(name=f'{os.path.basename(base)}_chunk{c}', out=f'{base}_chunk{c}.json', alpha=alpha, chunk=c, nchunks=nchunks,
                 lam_re=lam.real.tolist(), lam_im=lam.imag.tolist(), t=tall[order[c::nchunks]].tolist(),
                 index=order[c::nchunks].tolist()) for c in range(nchunks)]


def ml_table_job(job):
    E = ml_table(np.array(job['lam_re']) + 1j * np.array(job['lam_im']), np.array(job['t']), job['alpha'])
    np.save(job['out'][:-5] + '.npy', E)
    write_json(job['out'], dict(n=len(job['t']), alpha=job['alpha']))


def assemble_reference(path, S, tcol, t, is_uniform, is_log, alpha, jobs):
    """Collect the chunks into `path` (npz: test grid, mesh, eigen-decomposition, tables) and remove them."""
    lam, V = np.linalg.eig(np.asarray(S, dtype=complex))
    assert np.array_equal(lam, np.array(jobs[0]['lam_re']) + 1j * np.array(jobs[0]['lam_im']))
    E = np.zeros((len(t) + len(tcol), len(lam)), dtype=complex)
    for j in jobs:
        E[j['index']] = np.load(j['out'][:-5] + '.npy')
    np.savez(path, t=t, is_uniform=is_uniform, is_log=is_log, tcol=tcol, alpha=alpha, lam=lam, V=V, E_test=E[:len(t)],
             E_nodes=E[len(t):])
    for j in jobs:
        os.remove(j['out'][:-5] + '.npy'); os.remove(j['out'])
    return dict(cond_V=float(np.linalg.cond(V)), eig_residual=float(np.abs(V @ np.diag(lam) - S @ V).max()))


def load_reference(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def reference_states(R, S, rho0):
    """Exact states (len, d, d) on the test grid and at the nodes from a stored reference R (load_reference)."""
    d = rho0.shape[0]; eig = (R['lam'], R['V'])
    Xt, _ = ml_exact(S, rho0.reshape(-1), R['t'], None, eig, R['E_test'])
    Xn, _ = ml_exact(S, rho0.reshape(-1), R['tcol'], None, eig, R['E_nodes'])
    return Xt.reshape(-1, d, d), Xn.reshape(-1, d, d)


STEPPER_COLUMNS = ('N', 'mesh_r', 'wall_ms', 'rel_fro', 'max_fro', 'tau_max_fro', 'rel_fro_u1000', 'rel_inc_layer',
                   'node_max_fro', 'min_eig', 'max_trace_defect', 'max_herm_defect', 'neg_time_frac', 'residual_J_phys')


def stepper_rows(S, rho0, alpha, R, Ns, repeats, eig_tol=1e-12):
    """The implicit L1 time stepper on the graded meshes tau_n = (n/N)^r, r = (2 - alpha)/alpha, in the metric of
    the networks (state_metrics on the test grid, piecewise-linear interpolation between the nodes). The wall time is
    the median of `repeats` solves including the tabulation of the weights. Returns one dict per N."""
    d = rho0.shape[0]; r = (2 - alpha) / alpha; Xt, Xn = reference_states(R, S, rho0); rows = []
    for N in Ns:
        tn = (np.arange(N + 1) / N) ** r; ws = []
        for _ in range(repeats):
            t0 = time.perf_counter(); X = l1_stepper(S, rho0.reshape(-1), tn, alpha); ws.append(time.perf_counter() - t0)
        m = state_metrics(interp_nodes(R['t'], tn, X).reshape(-1, d, d), Xt, R['t'], R['is_uniform'], R['is_log'], rho0, eig_tol)
        row = dict(N=N, mesh_r=r, wall_ms=float(np.median(ws)) * 1e3, node_max_fro=float('nan'), residual_J_phys=float('nan'),
                   **{k: m[k] for k in STEPPER_COLUMNS if k in m})
        if len(tn) == len(R['tcol']) and np.array_equal(tn, R['tcol']):
            Xm = X.reshape(-1, d, d)
            row['node_max_fro'] = float(np.sqrt((np.abs(Xm - Xn) ** 2).sum((-1, -2))).max())
            Rs = l1_residual(torch.as_tensor(Xm), torch.as_tensor(l1_weight_matrix(tn, alpha)).to(CD), torch.as_tensor(S, dtype=CD))
            row['residual_J_phys'] = float((Rs.abs() ** 2).sum((-1, -2)).mean())
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------
# Test grid and metrics
# ----------------------------------------------------------------------------------------------

def test_grid(n_uniform=1000, log_min=-12, per_decade=40):
    """Test times on [0, 1] that resolve the initial layer: tau = 0, the midpoints (j - 1/2)/n_uniform (the test
    points of the first campaign) and per_decade logarithmically spaced points per decade from 10^log_min to 1.
    Returns (t, is_uniform, is_log), t sorted."""
    tu = (np.arange(1, n_uniform + 1) - 0.5) / n_uniform
    tl = 10.0 ** (log_min + np.arange(-log_min * per_decade + 1) / per_decade)
    t = np.unique(np.concatenate([[0.0], tu, tl]))
    return t, np.isin(t, tu), np.isin(t, tl)


def trapz_weights(t):
    w = np.zeros_like(t); h = np.diff(t)
    w[1:] += 0.5 * h; w[:-1] += 0.5 * h
    return w


def state_metrics(X, Xref, t, is_uniform, is_log, rho0, tol=1e-12, layer_tau=1e-2):
    """Errors and admissibility defects of a trajectory X (M, d, d) against the reference Xref on the test grid t.

    rel_fro        [int ||X - Xref||_F^2 dtau / int ||Xref||_F^2 dtau]^(1/2) over [0, 1], trapezoidal rule on t
    max_fro        max_j ||X(t_j) - Xref(t_j)||_F over the whole grid (it reaches tau = 10^log_min), at tau_max_fro
    rel_fro_u1000  [sum ||X - Xref||_F^2 / sum ||Xref||_F^2]^(1/2) on the uniform midpoints only (metric of the first campaign)
    rel_inc_layer  [sum ||X - Xref||_F^2 / sum ||Xref - rho0||_F^2]^(1/2) over the log-spaced points with tau <= layer_tau:
                   relative error of the increment rho - rho0 in the initial layer, uniform in log(tau)
    rel_fro_proj, max_fro_proj   the first two after the Frobenius projection of X on the set of density matrices
    min_eig        smallest eigenvalue of the Hermitian part over the grid (at tau_min_eig)
    neg_time_frac  fraction of [0, 1] (trapezoidal measure) on which the smallest eigenvalue is below -tol
    neg_log_frac   fraction of the log-spaced points on which it is below -tol
    max_negativity, int_negativity   maximum and time integral of the sum of the moduli of the negative eigenvalues
    max_trace_defect, max_herm_defect   max_j |tr X - 1| and max_j ||X - X^+||_F
    """
    Xt = torch.as_tensor(X); Xr = torch.as_tensor(Xref); w = trapz_weights(t)
    e2 = ((Xt - Xr).abs() ** 2).sum((-1, -2)).numpy(); n2 = (Xr.abs() ** 2).sum((-1, -2)).numpy()
    inc2 = ((Xr - torch.as_tensor(rho0)).abs() ** 2).sum((-1, -2)).numpy()
    lay = is_log & (t <= layer_tau)
    P = L.project_to_density(Xt); p2 = ((P - Xr).abs() ** 2).sum((-1, -2)).numpy()
    adm = {k: v.numpy() for k, v in L.admissibility(Xt).items()}
    neg = adm['min_eig'] < -tol; jm = int(np.argmax(e2)); je = int(np.argmin(adm['min_eig']))
    return dict(rel_fro=float(np.sqrt((w * e2).sum() / (w * n2).sum())), max_fro=float(np.sqrt(e2[jm])),
                tau_max_fro=float(t[jm]),
                rel_fro_u1000=float(np.sqrt(e2[is_uniform].sum() / n2[is_uniform].sum())),
                rel_inc_layer=float(np.sqrt(e2[lay].sum() / inc2[lay].sum())),
                rel_fro_proj=float(np.sqrt((w * p2).sum() / (w * n2).sum())), max_fro_proj=float(np.sqrt(p2.max())),
                min_eig=float(adm['min_eig'][je]), tau_min_eig=float(t[je]),
                neg_time_frac=float((w * neg).sum() / w.sum()), neg_log_frac=float(neg[is_log].mean()),
                max_negativity=float(adm['negativity'].max()), int_negativity=float((w * adm['negativity']).sum()),
                max_trace_defect=float(adm['trace_defect'].max()), max_herm_defect=float(adm['herm_defect'].max()))


def curve(x, sig=5):
    return [float(f'{v:.{sig - 1}e}') for v in np.asarray(x, dtype=float)]


def forward_run(S, rho0, alpha, tcol, variant, net, opt, seed, test):
    """Train one member of the factorial family on D^alpha rho = S rho, rho(0) = rho0, and evaluate it.

    S (d^2, d^2) complex numpy superoperator; rho0 (d, d); tcol collocation mesh (numpy, tcol[0] = 0);
    variant = dict(map, ic, gate, feature='plain', ic_weight=1.0); net = dict(width, depth, beta);
    opt = dict(adam_iters, lbfgs_iters, lr); test = dict(t, is_uniform, is_log, ref, ref_nodes, l1_nodes) with the
    exact states on the test grid and at the nodes and the L1 time-stepper states at the nodes.
    Returns (result dict, X on the test grid). The seed fixes the network initialisation; everything else is
    deterministic."""
    torch.manual_seed(seed); np.random.seed(seed)
    d = rho0.shape[0]
    model = DensityNet(torch.as_tensor(rho0, dtype=CD), map=variant['map'], ic=variant['ic'], gate=variant.get('gate', 'exp'),
                       alpha=alpha, beta=net['beta'], width=net['width'], depth=net['depth'],
                       feature=variant.get('feature', 'plain'))
    tc = torch.as_tensor(tcol, dtype=DT); Wc = op.l1_nonuniform_matrix(tc, alpha).to(CD); St = torch.as_tensor(S, dtype=CD)
    icw = float(variant.get('ic_weight', 1.0))
    info = fit(model.parameters(), lambda: forward_loss(model, tc, Wc, St, icw), opt['adam_iters'], opt['lbfgs_iters'],
               opt['lr'])
    tt = torch.as_tensor(test['t'], dtype=DT)
    with torch.no_grad():
        t1 = time.perf_counter(); X = model(tt); t_inf = time.perf_counter() - t1
        Xc = model(tc)
    X = X.numpy(); Xc = Xc.numpy()
    res = dict(d=d, alpha=alpha, seed=seed, n_params=sum(p.numel() for p in model.parameters()),
               N=len(tcol) - 1, n_test=len(test['t']), t_inference=t_inf, **{k: variant.get(k) for k in
               ('map', 'ic', 'gate', 'feature', 'ic_weight')}, **info)
    if not np.isfinite(X).all() or not np.isfinite(Xc).all():
        res['status'] = info['status'] if info['status'] != 'ok' else 'non-finite network output'
        return res, X
    res.update(state_metrics(X, test['ref'], test['t'], test['is_uniform'], test['is_log'], rho0))
    en = np.sqrt((np.abs(Xc - test['ref_nodes']) ** 2).sum((-1, -2)))
    dn = np.sqrt((np.abs(Xc - test['l1_nodes']) ** 2).sum((-1, -2)))
    res.update(ic_error=float(en[0]), node_max_fro=float(en.max()), node_max_fro_to_l1=float(dn.max()),
               node_rms_fro_to_l1=float(np.sqrt((dn[1:] ** 2).mean())),
               min_eig_nodes=float(np.linalg.eigvalsh(0.5 * (Xc + Xc.conj().transpose(0, 2, 1)))[:, 0].min()))
    il = np.flatnonzero(test['is_log'])[::4]; iu = np.flatnonzero(test['is_uniform'])[::20]
    adm = L.admissibility(torch.as_tensor(X[il])); err = np.sqrt((np.abs(X - test['ref']) ** 2).sum((-1, -2)))
    res['curves'] = dict(tau=curve(test['t'][il], 8), err_fro=curve(err[il]), min_eig=curve(adm['min_eig'].numpy()),
                         min_eig_exact=curve(np.linalg.eigvalsh(test['ref'][il])[:, 0]),
                         tau_uniform=curve(test['t'][iu], 8), err_fro_uniform=curve(err[iu]))
    return res, X


# ----------------------------------------------------------------------------------------------
# Provenance and campaign launcher
# ----------------------------------------------------------------------------------------------

_FINGERPRINTS = {}


def code_fingerprint(extra_files=()):
    """SHA-256 (first 16 hex digits, line endings normalised) of every source file of the package and of the given
    driver files. Evaluated once per process, so that it describes the code that the process has loaded."""
    key = tuple(sorted(os.path.abspath(f) for f in extra_files))
    if key not in _FINGERPRINTS:
        here = os.path.dirname(os.path.abspath(__file__))
        files = sorted(os.path.join(here, f) for f in os.listdir(here) if f.endswith('.py')) + list(key)
        _FINGERPRINTS[key] = {os.path.basename(f): hashlib.sha256(open(f, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:16]
                              for f in files}
    return _FINGERPRINTS[key]


def peak_rss_mb():
    """Peak resident set size of this process in MB (Windows, Linux, macOS); None if it cannot be measured."""
    if sys.platform == 'win32':
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [(n, ctypes.c_size_t) for n in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL('kernel32'); psapi = ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.PeakWorkingSetSize / 2 ** 20
    try:
        import resource
    except ImportError:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS, kilobytes on Linux
    return r / 2 ** 20 if sys.platform == 'darwin' else r / 2 ** 10


def run_metadata(seed=None, config=None, driver=None, sources=()):
    """Provenance record written into every result file: time stamp, versions, platform, seed, configuration and
    the fingerprints of the package sources, of the driver and of further source or configuration files.
    No host or user name is recorded."""
    import scipy
    import mpmath
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, mpmath=mpmath.__version__, fracsp=__version__,
                platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(), dtype='float64',
                driver=os.path.basename(driver) if driver else None,
                code_sha256=code_fingerprint(([driver] if driver else []) + list(sources)), seed=seed, config=config)


def write_json(path, obj):
    """Atomic write (a partially written result is never mistaken for a finished run)."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _unthrottle():
    """Windows 11 on hybrid CPUs confines background console processes to the efficiency cores (execution-speed
    throttling, 'EcoQoS'). The worker opts out for itself; priority, affinity and files are not touched. Returns
    True if the request was accepted, False otherwise (other platforms: None)."""
    if sys.platform != 'win32':
        return None
    import ctypes
    import ctypes.wintypes as wt

    class PPTS(ctypes.Structure):
        _fields_ = [('Version', wt.ULONG), ('ControlMask', wt.ULONG), ('StateMask', wt.ULONG)]
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    st = PPTS(1, 0x1, 0x0)          # control EXECUTION_SPEED, state 0: throttling off
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))


def _init_worker():
    torch.set_num_threads(1); torch.set_default_dtype(DT)
    _unthrottle()


def _call(func, job):
    t0 = time.perf_counter()
    try:
        out = func(job)
        return dict(name=job['name'], rc=0, wall=time.perf_counter() - t0, info=out)
    except Exception:
        return dict(name=job['name'], rc=1, wall=time.perf_counter() - t0, info=traceback.format_exc())


def run_campaign(func, jobs, workers, log_path, label):
    """Run func(job) for every job (dicts with the keys 'name' and 'out') in `workers` single-thread processes.

    func must write the result file job['out'] and may return a short string for the log. A job is skipped only if
    its result file already exists and parses as JSON. Failures are not swallowed: the traceback of every failed job
    goes to the campaign log and to <out>.err.txt, the other jobs are still run, and the function returns the number
    of failed jobs (jobs that raised, or whose result file is missing afterwards); the callers exit with status 1 if
    it is not zero."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
    todo, skipped = [], 0
    for j in jobs:
        try:
            json.load(open(j['out'])); skipped += 1
        except (OSError, ValueError):
            todo.append(j)
    t0 = time.perf_counter(); failed = []; done = 0
    with open(log_path, 'a') as log:
        def say(s):
            line = f'{datetime.datetime.now().isoformat(timespec="seconds")} {s}'
            log.write(line + '\n'); log.flush(); print(line, flush=True)
        say(f'[{label}] {len(jobs)} jobs, {skipped} already done, {len(todo)} to run, {workers} workers')
        if todo:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
                futs = [ex.submit(_call, func, j) for j in todo]
                for fut in as_completed(futs):
                    r = fut.result(); done += 1
                    if r['rc'] != 0:
                        failed.append(r['name'])
                        out = next(j['out'] for j in todo if j['name'] == r['name'])
                        open(out + '.err.txt', 'w').write(r['info'])
                        say(f'[{label}] FAILED {r["name"]} after {r["wall"]:.1f} s\n{r["info"]}')
                    else:
                        say(f'[{label}] {done}/{len(todo)} {r["name"]} {r["wall"]:.1f} s {r["info"] or ""}')
        missing = [j['name'] for j in jobs if not os.path.exists(j['out']) and j['name'] not in failed]
        for n in missing:
            say(f'[{label}] MISSING result file of {n}')
        nbad = len(failed) + len(missing)
        say(f'[{label}] finished: {len(todo) - len(failed)} run, {skipped} skipped, {nbad} FAILED, '
            f'wall {time.perf_counter() - t0:.1f} s')
    return nbad

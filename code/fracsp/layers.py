"""Structure layers (torch, float64).

Output maps for density matrices (all act on the last two axes of a complex tensor M of shape (..., d, d) that is
obtained from the raw network output with `factor_from_raw`):

  name      hard initial condition  X(0) = rho0                      soft initial condition      exact constraints
  unc       rho0 + g M                                               M                           none
  herm      rho0 + g (M + M^+)/2                                     (M + M^+)/2                 Hermitian
  lin       rho0 + g tl((M + M^+)/2)                                 I/d + tl((M + M^+)/2)       Hermitian, tr = 1 (linear)
  bloch     rho0 + g sum_k c_k G_k / 2   (d^2-1 real outputs c)      I/d + sum_k c_k G_k / 2     Hermitian, tr = 1 (linear)
  factor    N(T0 + g M),  N(T) = T T^+ / tr(T T^+)                   N(M)                        Hermitian, tr = 1, PSD
  add       (rho0 + g M M^+) / (1 + g tr(M M^+))                     -                           Hermitian, tr = 1, PSD
  gate2     N(T0 + (g P0 + sqrt(g) Q0) M)                            -                           Hermitian, tr = 1, PSD
  gate2b    N(V (sqrt(w) + G o V^+ M)), G_ij = sqrt(g) if i,j in ker -                           Hermitian, tr = 1, PSD

tl(.) removes the trace, T0 = V sqrt(w) is a factor of rho0 = V w V^+, P0 projects on range(rho0), Q0 = I - P0 and
g = g(tau) is a gate with g(0) = 0 (see `gate`).  For a rank-deficient rho0 the eigenvalues that vanish at tau = 0 grow
like g^2 under `factor`, like g under `add`, `gate2` and `gate2b`; the coherences between range(rho0) and its kernel grow
like g under `factor`, `add` and `gate2b` and like sqrt(g) under `gate2`.
"""
import math
import torch

DT = torch.float64
CD = torch.complex128

HARD_MAPS = ('unc', 'herm', 'lin', 'bloch', 'factor', 'add', 'gate2', 'gate2b')
SOFT_MAPS = ('unc', 'herm', 'lin', 'bloch', 'factor')
PSD_MAPS = ('factor', 'add', 'gate2', 'gate2b')


def factor_from_raw(raw, d):
    """Map raw network outputs (..., 2*d*d) to a complex factor T (..., d, d) (full factor)."""
    re, im = raw[..., : d * d], raw[..., d * d:]
    return torch.complex(re, im).reshape(raw.shape[:-1] + (d, d))


def normalized_factor_map(T, eps=0.0):
    """rho = T T^dagger / tr(T T^dagger). eps (optional) is a guard added to the denominator only;
    eps=0 is used in all reported experiments so that boundary (rank-deficient) states are admissible.
    With eps > 0 the trace is tr/(tr + eps) < 1, i.e. the output is no longer of unit trace."""
    S = T @ T.conj().transpose(-1, -2)
    tr = torch.diagonal(S, dim1=-2, dim2=-1).real.sum(-1)
    return S / (tr + eps)[..., None, None]


def factor_of_state(rho0):
    """A factor T0 with T0 T0^dagger = rho0 (eigen-decomposition; valid for rank-deficient rho0)."""
    w, V = torch.linalg.eigh(rho0)
    w = torch.clamp(w, min=0.0)
    return V @ torch.diag_embed(torch.sqrt(w).to(V.dtype))


def hard_ic_factor(T0, Nout, tau, beta=5.0):
    """T(tau) = T0 + g(tau) N(tau), g(tau) = 1 - exp(-beta tau); exactly T(0) = T0."""
    g = -torch.expm1(-beta * tau)
    return T0 + g[..., None, None] * Nout


# ----------------------------------------------------------------------------------------------
# Gates, linear heads and admissible maps with an exact initial condition
# ----------------------------------------------------------------------------------------------

def herm(M):
    return 0.5 * (M + M.conj().transpose(-1, -2))


def trace(M):
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def traceless(M):
    d = M.shape[-1]
    return M - (trace(M) / d)[..., None, None] * torch.eye(d, dtype=M.dtype)


def su_basis(d):
    """Generalised Gell-Mann matrices G_k, k = 1..d^2-1: traceless, Hermitian, tr(G_j G_k) = 2 delta_jk.
    Order: symmetric, antisymmetric, diagonal; for d = 2 these are the Pauli matrices (sx, sy, sz)."""
    sym, asym, diag = [], [], []
    for j in range(d):
        for k in range(j + 1, d):
            S = torch.zeros(d, d, dtype=CD); S[j, k] = 1; S[k, j] = 1; sym.append(S)
            A = torch.zeros(d, d, dtype=CD); A[j, k] = -1j; A[k, j] = 1j; asym.append(A)
    for l in range(1, d):
        D = torch.zeros(d, d, dtype=CD)
        for j in range(l):
            D[j, j] = 1
        D[l, l] = -l
        diag.append(math.sqrt(2.0 / (l * (l + 1))) * D)
    return torch.stack(sym + asym + diag)


def gate(tau, beta=5.0, kind='exp', alpha=None):
    """Gate of the hard initial condition, g(0) = 0 and 0 <= g < 1.
    kind='exp': g = 1 - exp(-beta tau)        (smooth, g ~ beta tau);
    kind='pow': g = 1 - exp(-beta tau^alpha)  (g ~ beta tau^alpha, the leading behaviour of a Caputo solution)."""
    if kind == 'exp':
        return -torch.expm1(-beta * tau)
    if kind == 'pow':
        return -torch.expm1(-beta * tau ** alpha)
    raise ValueError(kind)


def sqrt_gate(tau, beta=5.0, kind='exp', alpha=None):
    """sqrt(g(tau)), written so that its derivative with respect to alpha is finite (zero) at tau = 0."""
    pos = tau > 0
    ts = torch.where(pos, tau, torch.ones_like(tau))
    return torch.where(pos, torch.sqrt(gate(ts, beta, kind, alpha)), torch.zeros_like(tau))


def state_frame(rho0, rank_tol=1e-12):
    """Spectral data of the initial state used by the hard-IC maps: rho0 = V diag(w) V^+, the factor
    T0 = V diag(sqrt(w)), the projector P0 on range(rho0), Q0 = I - P0 and the kernel mask (w <= rank_tol max w)."""
    rho0 = rho0.to(CD)
    d = rho0.shape[-1]
    w, V = torch.linalg.eigh(rho0)
    w = torch.clamp(w, min=0.0)
    ker = w <= rank_tol * w.max()
    w = torch.where(ker, torch.zeros_like(w), w)
    Vk = V[:, ker]
    Q0 = Vk @ Vk.conj().T
    return dict(rho0=rho0, d=d, V=V, w=w, sqrtw=torch.diag_embed(torch.sqrt(w)).to(CD), ker=ker,
                T0=V @ torch.diag_embed(torch.sqrt(w)).to(CD), P0=torch.eye(d, dtype=CD) - Q0, Q0=Q0)


def bloch_head(c, basis):
    """Hermitian traceless matrix sum_k c_k G_k / 2 from d^2-1 real coefficients."""
    return 0.5 * torch.einsum('...k,kij->...ij', c.to(CD), basis)


def additive_psd_map(rho0, B, g):
    """rho = (rho0 + g B B^+) / (1 + g tr(B B^+)): Hermitian, PSD, unit trace, rho = rho0 where g = 0.
    Equivalent to the rectangular factor T = [T0, sqrt(g) B]; every block of rho - rho0 is linear in g."""
    C = B @ B.conj().transpose(-1, -2)
    trC = torch.diagonal(C, dim1=-2, dim2=-1).real.sum(-1)
    return (rho0 + g[..., None, None] * C) / (1.0 + g * trC)[..., None, None]


def two_gate_factor(fr, M, g, sg):
    """T = T0 + (g P0 + sqrt(g) Q0) M: rows in range(rho0) are gated by g, rows in the kernel by sqrt(g)."""
    G = g[..., None, None] * fr['P0'] + sg[..., None, None] * fr['Q0']
    return fr['T0'] + G @ M


def block_gate_factor(fr, M, g, sg):
    """T = V (sqrt(w) + G o (V^+ M)) with G_ij = sqrt(g) if both i and j are kernel indices of rho0 and g otherwise,
    so that the kernel block of T T^+ and the range-kernel coherences are both linear in g. For a full-rank rho0
    this is T0 + g M, i.e. the map 'factor'."""
    kk = (fr['ker'][:, None] & fr['ker'][None, :]).to(DT)
    G = g[..., None, None] * (1.0 - kk) + sg[..., None, None] * kk
    return fr['V'] @ (fr['sqrtw'] + G * (fr['V'].conj().T @ M))


def map_hard(name, M, g, fr, sg=None):
    """Output map with exact initial condition X = rho0 where g = 0 (see the module docstring).
    M: (..., d, d) complex; g, sg: gate and its square root, shape (...)."""
    gg = g[..., None, None]
    if name == 'unc':
        return fr['rho0'] + gg * M
    if name == 'herm':
        return fr['rho0'] + gg * herm(M)
    if name == 'lin':
        return fr['rho0'] + gg * traceless(herm(M))
    if name == 'bloch':                      # M is already Hermitian and traceless (see bloch_head)
        return fr['rho0'] + gg * M
    if name == 'factor':
        return normalized_factor_map(fr['T0'] + gg * M)
    if name == 'add':
        return additive_psd_map(fr['rho0'], M, g)
    if name == 'gate2':
        return normalized_factor_map(two_gate_factor(fr, M, g, sg))
    if name == 'gate2b':
        return normalized_factor_map(block_gate_factor(fr, M, g, sg))
    raise ValueError(name)


def map_soft(name, M):
    """Output map without built-in initial condition (the initial condition enters the loss)."""
    d = M.shape[-1]
    if name == 'unc':
        return M
    if name == 'herm':
        return herm(M)
    if name == 'lin':
        return torch.eye(d, dtype=M.dtype) / d + traceless(herm(M))
    if name == 'bloch':
        return torch.eye(d, dtype=M.dtype) / d + M
    if name == 'factor':
        return normalized_factor_map(M)
    raise ValueError(name)


def project_to_density(X):
    """Frobenius-nearest density matrix of X: Hermitian part, eigenvalues projected on the probability simplex.
    The projection is non-expansive, so it never increases the Frobenius distance to an admissible state."""
    w, V = torch.linalg.eigh(herm(X))                       # ascending
    d = w.shape[-1]
    u = torch.flip(w, [-1])                                 # descending
    css = torch.cumsum(u, -1) - 1.0
    k = torch.arange(1, d + 1, dtype=w.dtype)
    nact = ((u - css / k) > 0).sum(-1, keepdim=True)        # number of eigenvalues kept positive (>= 1)
    theta = torch.gather(css, -1, nact - 1) / nact.to(w.dtype)
    wp = torch.clamp(w - theta, min=0.0)
    return (V * wp[..., None, :].to(V.dtype)) @ V.conj().transpose(-1, -2)


def admissibility(X):
    """Pointwise defects of a batch of matrices X (..., d, d): smallest eigenvalue of the Hermitian part,
    negativity (sum of the moduli of the negative eigenvalues), |tr X - 1| and ||X - X^+||_F."""
    ev = torch.linalg.eigvalsh(herm(X))
    return dict(min_eig=ev[..., 0], negativity=torch.clamp(-ev, min=0.0).sum(-1),
                trace_defect=(trace(X) - 1).abs(), herm_defect=torch.linalg.matrix_norm(X - X.conj().transpose(-1, -2)))


# ----------------------------------------------------------------------------------------------
# Rapidity maps (Dirac kinematics)
# ----------------------------------------------------------------------------------------------

def rapidity_deterministic(p, Delta, vF=1.0):
    """eta = arsinh(vF |p| / Delta); returns (eps, v) from the rapidity representation."""
    pn = torch.linalg.norm(p, dim=-1)
    eta = torch.asinh(vF * pn / Delta)
    eps = Delta * torch.cosh(eta)
    phat = p / torch.clamp(pn, min=1e-300)[..., None]
    v = vF * torch.tanh(eta)[..., None] * phat
    return eta, eps, v


def rapidity_latent(eta_raw, phi_raw, Delta, vF=1.0):
    """Latent mode: eta = softplus(raw) >= 0, direction (cos phi, sin phi)."""
    eta = torch.nn.functional.softplus(eta_raw)
    d = torch.stack([torch.cos(phi_raw), torch.sin(phi_raw)], -1)
    p = (Delta / vF) * torch.sinh(eta)[..., None] * d
    eps = Delta * torch.cosh(eta)
    v = vF * torch.tanh(eta)[..., None] * d
    return eta, p, eps, v


def state_diagnostics(rho):
    herm_d = torch.linalg.matrix_norm(rho - rho.conj().transpose(-1, -2)).max().item()
    tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)
    trdef = (tr - 1).abs().max().item()
    H = 0.5 * (rho + rho.conj().transpose(-1, -2))
    lmin = torch.linalg.eigvalsh(H).min().item()
    return dict(herm_defect=herm_d, trace_defect=trdef, min_eig=lmin)

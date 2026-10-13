"""B1: forward and inverse scalar fractional relaxation with an L1-residual fPINN, 2 x 2 design
{network input: plain | singular} x {gate of the hard initial condition: exp | pow}:
    u(tau) = 1 + g(tau) N(x),   x = tau  or  (tau, tau^alpha),   g = 1 - exp(-beta tau)  or  1 - exp(-beta tau^alpha).
In the inverse problem alpha is trainable and the current value is used in the residual, the gate and the feature.
Requires the references written by b1_floor.py.

Usage: python b1_scalar.py --mode forward|inverse --alpha 0.5 --input plain --gate pow --seed 0
       [--noise 0.01] [--r 3.0] [--N 128] [--adam 4000] [--lbfgs 500] [--stage manual]
The variants of the first campaign are 'plain' = plain/exp and 'singular' = singular/pow.
Writes results/raw/B1_design/<stage>/<name>.json
"""
import argparse, functools, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import operators as op, reference as ref, layers as L, pinn
import b1_floor as P
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
cfg = P.cfg; lam = cfg['lambda']; RAW = os.path.join(P.ROOT, 'results', 'raw', 'B1_design')


class Net(torch.nn.Module):
    """u = 1 + g(tau) N(x): hard initial condition u(0) = 1."""
    def __init__(s, beta, inp, gate):
        super().__init__(); s.inp = inp; s.gate = gate; s.beta = beta
        s.net = pinn.MLP(cfg['network']['width'], cfg['network']['depth'], 1, nin=1 if inp == 'plain' else 2)

    def forward(s, t, a):
        x = t[:, None] if s.inp == 'plain' else torch.stack([t, t ** a], 1)
        return 1.0 + L.gate(t, s.beta, s.gate, a) * s.net(x)[:, 0]


def make_job(stage, mode, alpha, inp, gate, seed, noise=0.0, r=None, N=None, adam_iters=None, lbfgs_iters=None):
    o = cfg['optimizer']
    j = dict(stage=stage, mode=mode, alpha=alpha, input=inp, gate=gate, seed=seed, noise=noise if mode == 'inverse' else 0.0,
             r=cfg['mesh']['r'] if r is None else r, N=cfg['mesh']['N'] if N is None else N,
             adam_iters=o['adam_iters'] if adam_iters is None else adam_iters,
             lbfgs_iters=o['lbfgs_iters'] if lbfgs_iters is None else lbfgs_iters)
    j['name'] = f"{mode}_a{alpha}_in-{inp}_g-{gate}_n{j['noise']:g}_r{j['r']:.3g}_s{seed}"
    j['out'] = os.path.join(RAW, stage, j['name'] + '.json')
    return j


@functools.lru_cache(maxsize=None)
def exact_reference():
    z = np.load(os.path.join(P.OUT, 'exact.npz')); t, iu, il = P.grid()
    assert np.array_equal(z['t'], t), 'stored reference does not match the configuration'
    return t, iu, {a: z[f'u_a{a}'] for a in cfg['alphas_true']}


def run_job(j):
    mode, alpha, seed = j['mode'], j['alpha'], j['seed']; I = cfg['inverse']
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(cfg['network']['beta'], j['input'], j['gate'])
    tcol = torch.tensor(ref.graded_mesh(1.0, j['N'], j['r'])); params = list(net.parameters())
    if mode == 'inverse':
        lo, hi = I['alpha_init_range']; a0 = np.random.default_rng(seed).uniform(lo, hi)
        araw = torch.tensor(math.log((a0 - 0.05) / 0.95 / (1 - (a0 - 0.05) / 0.95)), requires_grad=True)
        params = params + [araw]
        tob, uob, _ = P.observations(alpha, j['noise'], seed); tob_t = torch.tensor(tob); uob_t = torch.tensor(uob)
    else:
        a0 = alpha; Wfix = op.l1_nonuniform_matrix(tcol, alpha)

    def get_alpha():
        return 0.05 + 0.95 * torch.sigmoid(araw) if mode == 'inverse' else torch.tensor(alpha)

    def loss():
        a = get_alpha(); W = op.l1_nonuniform_matrix(tcol, a) if mode == 'inverse' else Wfix
        u = net(tcol, a)
        R = op.l1_nonuniform(u[:, None], W)[:, 0] + lam * u[1:]
        J = {'phys': (R ** 2).mean()}
        if mode == 'inverse':
            J['data'] = I['data_weight'] * ((net(tob_t, a) - uob_t) ** 2).mean()
        return J
    info = pinn.fit(params, loss, j['adam_iters'], j['lbfgs_iters'], cfg['optimizer']['lr'])
    t, iu, ex = exact_reference(); ex = ex[alpha]; w = pinn.trapz_weights(t)
    with torch.no_grad():
        ahat = float(get_alpha()); u = net(torch.tensor(t), get_alpha()).numpy()
    d = u - ex
    res = dict(benchmark='B1', stage=j['stage'], name=j['name'], mode=mode, input=j['input'], gate=j['gate'], alpha_true=alpha,
               seed=seed, noise=j['noise'], mesh_r=j['r'], N=j['N'], alpha_init=float(a0), alpha_hat=ahat, alpha_err=ahat - alpha,
               rel_l2=float(np.sqrt((w * d ** 2).sum() / (w * ex ** 2).sum())), max_abs=float(np.abs(d).max()),
               tau_max_abs=float(t[np.abs(d).argmax()]), rel_l2_u1000=float(np.linalg.norm(d[iu]) / np.linalg.norm(ex[iu])),
               final_residual=info['final_terms']['phys'],
               final_data_misfit=info['final_terms'].get('data', 0.0) / I['data_weight'], n_params=sum(p.numel() for p in net.parameters()),
               peak_rss_mb=pinn.peak_rss_mb(), **info)
    conf = dict(job={k: v for k, v in j.items() if k != 'out'}, **{k: cfg[k] for k in ('lambda', 'network', 'inverse', 'test')},
                optimizer=dict(cfg['optimizer'], adam_iters=j['adam_iters'], lbfgs_iters=j['lbfgs_iters']), config_file='configs/b1_scalar.json')
    res.update(pinn.run_metadata(seed=seed, config=conf, driver=__file__, sources=[P.__file__, P.CFG]))
    os.makedirs(os.path.dirname(j['out']), exist_ok=True); pinn.write_json(j['out'], res)
    return f"{res['status']} alpha_hat={ahat:.5f} rel_l2={res['rel_l2']:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True); ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--input', default='plain'); ap.add_argument('--gate', default='exp'); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--noise', type=float, default=0.0); ap.add_argument('--r', type=float); ap.add_argument('--N', type=int)
    ap.add_argument('--adam', type=int); ap.add_argument('--lbfgs', type=int); ap.add_argument('--stage', default='manual')
    a = ap.parse_args()
    j = make_job(a.stage, a.mode, a.alpha, a.input, a.gate, a.seed, a.noise, a.r, a.N, a.adam, a.lbfgs)
    print(j['name'], run_job(j)); print('written', j['out'])


if __name__ == '__main__':
    main()

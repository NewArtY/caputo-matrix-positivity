"""B2: one training run of the factorial family (output map x initial condition x gate) on the driven two-level
system with the exact matrix Mittag-Leffler reference. Requires the references written by b2_prepare.py.

Usage: python b2_density.py --alpha 0.6 --state pure --map factor --ic hard --gate pow --seed 0
       [--feature singular] [--icw 10] [--adam 4000] [--lbfgs 500] [--stage manual] [--save_traj 1]
maps: unc, herm, lin, bloch, factor, add, gate2, gate2b (fracsp.layers); ic: hard | soft; gate: exp | pow.
The variants of the first campaign are: Direct = unc/soft/icw 1, SP-soft = factor/soft/icw 1, SP-fPINN = factor/hard/exp.
Writes results/raw/B2_factorial/<stage>/<name>.json (and <name>_test_rho.npy with --save_traj 1).
"""
import argparse, functools, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn
import b2_prepare as P
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
cfg = P.cfg; RAW = os.path.join(P.ROOT, 'results', 'raw', 'B2_factorial')


def job_name(j):
    v = f"{j['map']}_{j['ic']}-" + (j['gate'] if j['ic'] == 'hard' else f"w{j['ic_weight']:g}")
    if j.get('feature', 'plain') != 'plain':
        v += '_' + j['feature']
    return f"a{j['alpha']}_{j['state']}_{v}_s{j['seed']}"


def make_job(stage, alpha, state, map, ic, seed, gate='exp', ic_weight=1.0, feature='plain', adam_iters=None,
             lbfgs_iters=None, save_traj=False):
    o = cfg['optimizer']
    j = dict(stage=stage, alpha=alpha, state=state, map=map, ic=ic, gate=gate if ic == 'hard' else None,
             ic_weight=ic_weight if ic == 'soft' else None, feature=feature, seed=seed,
             adam_iters=o['adam_iters'] if adam_iters is None else adam_iters,
             lbfgs_iters=o['lbfgs_iters'] if lbfgs_iters is None else lbfgs_iters, save_traj=save_traj)
    j['name'] = job_name(j); j['out'] = os.path.join(RAW, stage, j['name'] + '.json')
    return j


@functools.lru_cache(maxsize=None)
def setup(alpha, state):
    """Generator, mesh, test grid, exact states and L1 time-stepper states of one (alpha, initial state)."""
    S, tcol, t, iu, il = P.problem(alpha); R = pinn.load_reference(P.ref_path(alpha)); rho0 = P.rho0_of(state)
    assert np.array_equal(R['t'], t) and np.array_equal(R['tcol'], tcol), 'stored reference does not match the configuration'
    Xt, Xn = pinn.reference_states(R, S, rho0)
    Xl = pinn.l1_stepper(S, rho0.reshape(-1), tcol, alpha).reshape(-1, 2, 2)
    return S, rho0, tcol, dict(t=t, is_uniform=iu, is_log=il, ref=Xt, ref_nodes=Xn, l1_nodes=Xl)


def run_job(j):
    S, rho0, tcol, test = setup(j['alpha'], j['state'])
    variant = dict(map=j['map'], ic=j['ic'], gate=j['gate'] or 'exp', feature=j['feature'], ic_weight=j['ic_weight'] or 1.0)
    net = cfg['network']; opt = dict(adam_iters=j['adam_iters'], lbfgs_iters=j['lbfgs_iters'], lr=cfg['optimizer']['lr'])
    res, X = pinn.forward_run(S, rho0, j['alpha'], tcol, variant, net, opt, j['seed'], test)
    res.update(benchmark='B2', state=j['state'], stage=j['stage'], name=j['name'], gate=j['gate'], ic_weight=j['ic_weight'],
               peak_rss_mb=pinn.peak_rss_mb())
    conf = dict(job={k: v for k, v in j.items() if k != 'out'}, phys=cfg['phys'], mesh=cfg['mesh'], network=net,
                optimizer=dict(cfg['optimizer'], **opt), test=cfg['test'], eig_tol=cfg['eig_tol'], config_file='configs/b2_density.json')
    res.update(pinn.run_metadata(seed=j['seed'], config=conf, driver=__file__, sources=[P.__file__, P.CFG]))
    os.makedirs(os.path.dirname(j['out']), exist_ok=True)
    if j.get('save_traj'):
        np.save(j['out'][:-5] + '_test_rho.npy', X)
    pinn.write_json(j['out'], res)
    return f"{res['status']} rel_fro={res.get('rel_fro', float('nan')):.3e} max_fro={res.get('max_fro', float('nan')):.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True); ap.add_argument('--state', required=True)
    ap.add_argument('--map', required=True); ap.add_argument('--ic', default='hard'); ap.add_argument('--gate', default='exp')
    ap.add_argument('--icw', type=float, default=1.0); ap.add_argument('--feature', default='plain')
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--adam', type=int); ap.add_argument('--lbfgs', type=int)
    ap.add_argument('--stage', default='manual'); ap.add_argument('--save_traj', type=int, default=0)
    a = ap.parse_args()
    j = make_job(a.stage, a.alpha, a.state, a.map, a.ic, a.seed, a.gate, a.icw, a.feature, a.adam, a.lbfgs, bool(a.save_traj))
    print(j['name'], run_job(j)); print('written', j['out'])


if __name__ == '__main__':
    main()

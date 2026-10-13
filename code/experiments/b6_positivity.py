"""B6: one training run on a d = 3 or d = 4 problem whose exact state stays on or near the boundary of the state
space (pure initial state). Requires the references written by b6_prepare.py.

Usage: python b6_positivity.py --system qutrit --alpha 0.5 --map lin --gate pow --seed 0
       [--adam 4000] [--lbfgs 500] [--stage manual] [--save_traj 1]
systems: qutrit (d = 3), two_qubit (d = 4); maps: unc, herm, lin, bloch, factor, add, gate2, gate2b; hard IC throughout.
Writes results/raw/B6_positivity/<stage>/<name>.json
"""
import argparse, functools, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn
import b6_prepare as P
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
cfg = P.cfg; RAW = os.path.join(P.ROOT, 'results', 'raw', 'B6_positivity')


def make_job(stage, system, alpha, map, gate, seed, adam_iters=None, lbfgs_iters=None, save_traj=False):
    o = cfg['optimizer']
    j = dict(stage=stage, system=system, alpha=alpha, map=map, ic='hard', gate=gate, seed=seed,
             adam_iters=o['adam_iters'] if adam_iters is None else adam_iters,
             lbfgs_iters=o['lbfgs_iters'] if lbfgs_iters is None else lbfgs_iters, save_traj=save_traj)
    j['name'] = f'{system}_a{alpha}_{map}_hard-{gate}_s{seed}'; j['out'] = os.path.join(RAW, stage, j['name'] + '.json')
    return j


@functools.lru_cache(maxsize=None)
def setup(system, alpha):
    S, rho0, tcol, t, iu, il = P.problem(system, alpha); R = pinn.load_reference(P.ref_path(system, alpha))
    assert np.array_equal(R['t'], t) and np.array_equal(R['tcol'], tcol), 'stored reference does not match the configuration'
    Xt, Xn = pinn.reference_states(R, S, rho0); d = rho0.shape[0]
    Xl = pinn.l1_stepper(S, rho0.reshape(-1), tcol, alpha).reshape(-1, d, d)
    return S, rho0, tcol, dict(t=t, is_uniform=iu, is_log=il, ref=Xt, ref_nodes=Xn, l1_nodes=Xl)


def run_job(j):
    S, rho0, tcol, test = setup(j['system'], j['alpha'])
    variant = dict(map=j['map'], ic='hard', gate=j['gate'], feature='plain')
    net = cfg['network']; opt = dict(adam_iters=j['adam_iters'], lbfgs_iters=j['lbfgs_iters'], lr=cfg['optimizer']['lr'])
    res, X = pinn.forward_run(S, rho0, j['alpha'], tcol, variant, net, opt, j['seed'], test)
    res.update(benchmark='B6', system=j['system'], stage=j['stage'], name=j['name'], peak_rss_mb=pinn.peak_rss_mb())
    conf = dict(job={k: v for k, v in j.items() if k != 'out'}, system=cfg['systems'][j['system']], mesh=cfg['mesh'], network=net,
                optimizer=dict(cfg['optimizer'], **opt), test=cfg['test'], eig_tol=cfg['eig_tol'], config_file='configs/b6_positivity.json')
    res.update(pinn.run_metadata(seed=j['seed'], config=conf, driver=__file__, sources=[P.__file__, P.CFG]))
    os.makedirs(os.path.dirname(j['out']), exist_ok=True)
    if j.get('save_traj'):
        np.save(j['out'][:-5] + '_test_rho.npy', X)
    pinn.write_json(j['out'], res)
    return (f"{res['status']} rel_fro={res.get('rel_fro', float('nan')):.3e} min_eig={res.get('min_eig', float('nan')):+.2e} "
            f"neg_time={res.get('neg_time_frac', float('nan')):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True); ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--map', required=True); ap.add_argument('--gate', default='pow'); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--adam', type=int); ap.add_argument('--lbfgs', type=int)
    ap.add_argument('--stage', default='manual'); ap.add_argument('--save_traj', type=int, default=0)
    a = ap.parse_args()
    j = make_job(a.stage, a.system, a.alpha, a.map, a.gate, a.seed, a.adam, a.lbfgs, bool(a.save_traj))
    print(j['name'], run_job(j)); print('written', j['out'])


if __name__ == '__main__':
    main()

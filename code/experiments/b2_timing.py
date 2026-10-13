"""Timing and reproducibility pass, run one job after another in a single process (nothing else of these campaigns
running), so that costs can be compared.

Usage: python b2_timing.py b2 [alpha] [state] [seed]     (default: 0.6 pure 0)
           repeats the runs of the B2 main campaign for one alpha, state and seed, compares every repeated result bit
           for bit with the stored one, and times the implicit L1 time stepper on the same mesh;
           writes results/raw/B2_factorial/timing/{<name>.json, timing.csv, timing.json}
       python b2_timing.py surrogate
           cost of one query of the parametric surrogate (inference does not depend on the trained weights, so the
           architecture is timed) and of the L1 time stepper on every mesh of configs/b2_surrogate.json;
           writes results/raw/B2_surrogate/timing.json
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn


def b2(alpha=0.6, state='pure', seed=0):
    import b2_density as B
    cfg = B.cfg; sel = json.load(open(os.path.join(B.RAW, 'soft_ic_selected.json')))['selected']; M = cfg['maps']
    cells = [(m, 'hard', g, None) for m in M['hard'] + (M['hard_pure_only'] if state == 'pure' else []) for g in cfg['gates']]
    cells += [(m, 'soft', None, sel[m][str(alpha)]) for m in M['soft']]
    out = os.path.join(B.RAW, 'timing'); os.makedirs(out, exist_ok=True)
    rows = ['cell,alpha,state,seed,wall_s,cpu_s,t_adam_s,t_lbfgs_s,lbfgs_evals,rel_fro,rel_fro_stored,identical_to_stored']
    for m, ic, g, w in cells:
        j = B.make_job('timing', alpha, state, m, ic, seed, gate=g or 'exp', ic_weight=w or 1.0)
        if os.path.exists(j['out']):
            os.remove(j['out'])
        B.run_job(j); r = json.load(open(j['out']))
        s = json.load(open(B.make_job('main', alpha, state, m, ic, seed, gate=g or 'exp', ic_weight=w or 1.0)['out']))
        same = all(r[k] == s[k] for k in ('rel_fro', 'max_fro', 'min_eig', 'rel_inc_layer')) and r['final_terms'] == s['final_terms']
        rows.append(f"{m}/{ic}-{g or 'w%g' % w},{alpha},{state},{seed},{r['wall']:.3f},{r['cpu']:.3f},{r['t_adam']:.3f},{r['t_lbfgs']:.3f},"
                    f"{r['lbfgs_evals']},{r['rel_fro']!r},{s['rel_fro']!r},{same}")
        print(rows[-1], flush=True)
    S, rho0, tcol, test = B.setup(alpha, state); ws = []
    for _ in range(200):
        t0 = time.perf_counter(); pinn.l1_stepper(S, rho0.reshape(-1), tcol, alpha); ws.append(time.perf_counter() - t0)
    rows.append(f'l1_stepper/N{len(tcol) - 1},{alpha},{state},,{np.median(ws):.6f},,,,,,,')
    print(rows[-1])
    open(os.path.join(out, 'timing.csv'), 'w').write('\n'.join(rows) + '\n')
    pinn.write_json(os.path.join(out, 'timing.json'), dict(stepper_wall_s_median=float(np.median(ws)), stepper_wall_s_min=float(np.min(ws)),
                                                          n_identical=sum(r.endswith('True') for r in rows), n_runs=len(cells),
                                                          **pinn.run_metadata(seed=seed, config=dict(alpha=alpha, state=state), driver=__file__)))


def surrogate():
    import b2_surrogate as G
    cfg = G.cfg; P = G.test_params(); t = G.test_times()[0]; net = cfg['network']; nb = cfg['test']['inference_batch']; step = {}
    for N in cfg['stepper']['N']:
        ws = []
        for a, g in P:
            t0 = time.perf_counter(); tn = G.mesh(a, N); Xn = pinn.l1_stepper(G.generator(g), G.RHO0.reshape(-1), tn, a)
            pinn.interp_nodes(t, tn, Xn); ws.append(time.perf_counter() - t0)
        step[str(N)] = dict(ms_per_query_median=float(np.median(ws)) * 1e3, ms_per_query_min=float(np.min(ws)) * 1e3)
    model = pinn.DensityNet(torch.tensor(G.RHO0), map='add', ic='hard', gate=cfg['gate'], alpha=0.5 * (G.A0 + G.A1), beta=net['beta'],
                            width=net['width'], depth=net['depth'], n_par=2)
    rng = np.random.default_rng(cfg['test']['random_seed']); Pb = np.c_[rng.uniform(G.A0, G.A1, nb), rng.uniform(G.G0, G.G1, nb)]
    tb, t1, tf = [], [], []
    for _ in range(7):
        t0 = time.perf_counter(); G.evaluate(model, Pb, t); tb.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); G.evaluate(model, Pb, t[-1:]); tf.append(time.perf_counter() - t0)
    for k in range(100):
        t0 = time.perf_counter(); G.evaluate(model, Pb[k:k + 1], t); t1.append(time.perf_counter() - t0)
    res = dict(stepper=step, n_test_times=len(t), inference_ms_per_query_batched=float(np.median(tb)) / nb * 1e3,
               inference_ms_per_query_single=float(np.median(t1)) * 1e3, inference_ms_per_query_final_time_batched=float(np.median(tf)) / nb * 1e3,
               **pinn.run_metadata(config=cfg, driver=__file__, sources=[G.__file__, G.CFG]))
    pinn.write_json(os.path.join(G.RAW, 'timing.json'), res)
    print(json.dumps({k: res[k] for k in ('stepper', 'inference_ms_per_query_batched', 'inference_ms_per_query_single',
                                          'inference_ms_per_query_final_time_batched')}, indent=1))


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    if sys.argv[1] == 'surrogate':
        surrogate()
    else:
        b2(float(sys.argv[2]) if len(sys.argv) > 2 else 0.6, sys.argv[3] if len(sys.argv) > 3 else 'pure', int(sys.argv[4]) if len(sys.argv) > 4 else 0)

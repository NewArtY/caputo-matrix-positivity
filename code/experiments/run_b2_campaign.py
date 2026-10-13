"""B2 factorial campaign (configs/b2_density.json): output map x initial condition x gate, 10 seeds, three alphas.

Usage: python run_b2_campaign.py <stage> [workers]      (references first: python b2_prepare.py [workers])
  pilot   soft-IC weight grid on the pilot seeds; ends with the selection of one weight per (map, alpha),
          written to results/raw/B2_factorial/soft_ic_selected.json
  main    hard IC: maps x gates; soft IC: maps with the selected weight (and weight 1 as the untuned reference)
  extras  minimal Bloch head; singular input feature (tau, tau^alpha)
  long    four-fold optimiser budget for the hard-IC maps with the pow gate
  all     pilot, main, extras, long
The launcher exits with status 1 if any job failed or left no result file. Runs whose training ended with a
non-finite loss are not failures of the launcher: their result file records status != 'ok' and they are counted in
the tables.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn
import b2_density as B
cfg = B.cfg; SEL = os.path.join(B.RAW, 'soft_ic_selected.json')
STATES = list(cfg['initial_states_bloch'])


def jobs_pilot():
    s = cfg['soft_ic']
    return [B.make_job('pilot', a, st, m, 'soft', sd, ic_weight=w) for a in cfg['alphas'] for st in STATES
            for m in cfg['maps']['soft'] for w in s['weights'] for sd in s['pilot_seeds']]


def select_weights():
    """One soft-IC weight per (map, alpha): smallest geometric mean of rel_fro over states and pilot seeds."""
    s = cfg['soft_ic']; table = {}; sel = {}
    for m in cfg['maps']['soft']:
        sel[m] = {}
        for a in cfg['alphas']:
            gm = {}
            for w in s['weights']:
                e = [json.load(open(B.make_job('pilot', a, st, m, 'soft', sd, ic_weight=w)['out'])) for st in STATES
                     for sd in s['pilot_seeds']]
                bad = [r['name'] for r in e if r['status'] != 'ok']
                gm[w] = float('inf') if bad else float(np.exp(np.mean(np.log([r['rel_fro'] for r in e]))))
            table[f'{m}_a{a}'] = {f'{w:g}': v for w, v in gm.items()}
            sel[m][str(a)] = min(gm, key=gm.get)
    pinn.write_json(SEL, dict(selected=sel, geometric_mean_rel_fro=table, rule=s['selection'],
                              **pinn.run_metadata(config=s, driver=__file__)))
    print('selected soft-IC weights:', json.dumps(sel))


def jobs_main():
    sel = json.load(open(SEL))['selected']; M = cfg['maps']; jobs = []
    for a in cfg['alphas']:
        for st in STATES:
            for sd in cfg['seeds']:
                keep = sd == 0 and a == 0.6
                for g in cfg['gates']:
                    for m in M['hard'] + (M['hard_pure_only'] if st == 'pure' else []):
                        jobs.append(B.make_job('main', a, st, m, 'hard', sd, gate=g, save_traj=keep))
                for m in M['soft']:
                    jobs.append(B.make_job('main', a, st, m, 'soft', sd, ic_weight=sel[m][str(a)], save_traj=keep))
                u = cfg['soft_ic']['untuned_reference']
                for m in u['maps']:
                    if a in u['alphas'] and sel[m][str(a)] != u['weight']:
                        jobs.append(B.make_job('main', a, st, m, 'soft', sd, ic_weight=u['weight'], save_traj=keep))
    return jobs


def jobs_extras():
    e = cfg['extras']; jobs = []; b = e['bloch_head']; f = e['singular_feature']
    for st in STATES:
        for sd in cfg['seeds']:
            jobs += [B.make_job('extras', a, st, b['map'], b['ic'], sd, gate=b['gate']) for a in cfg['alphas']]
            jobs += [B.make_job('extras', a, st, m, f['ic'], sd, gate=f['gate'], feature='singular') for a in f['alphas']
                     for m in f['maps']]
    return jobs


def jobs_long():
    l = cfg['extras']['long_budget']
    return [B.make_job('long', a, st, m, l['ic'], sd, gate=l['gate'], adam_iters=l['adam_iters'], lbfgs_iters=l['lbfgs_iters'])
            for a in l['alphas'] for st in STATES for m in l['maps'] + (cfg['maps']['hard_pure_only'] if st == 'pure' else [])
            for sd in l['seeds']]


def main():
    stage = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for st in (['pilot', 'main', 'extras', 'long'] if stage == 'all' else [stage]):
        jobs = dict(pilot=jobs_pilot, main=jobs_main, extras=jobs_extras, long=jobs_long)[st]()
        nbad = pinn.run_campaign(B.run_job, jobs, workers, os.path.join(B.RAW, st, 'campaign.log'), f'B2 {st}')
        if nbad:
            sys.exit(f'B2 {st}: {nbad} of {len(jobs)} jobs FAILED (see {os.path.join(B.RAW, st, "campaign.log")})')
        if st == 'pilot':
            select_weights()
        nok = sum(json.load(open(j['out']))['status'] == 'ok' for j in jobs)
        print(f'B2 {st}: {len(jobs)} result files, {len(jobs) - nok} runs with status != ok')


if __name__ == '__main__':
    main()

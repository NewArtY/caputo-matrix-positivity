"""B6 campaign (configs/b6_positivity.json): hard-IC maps on a qutrit (d = 3) and on two coupled qubits (d = 4),
alpha in {0.5, 0.8}, 10 seeds, equal network, mesh and optimiser budget.

Usage: python b6_campaign.py <stage> [workers]      (references first: python b6_prepare.py [workers])
  main   all maps with the pow gate; lin and factor also with the exp gate
  long   four-fold optimiser budget for lin, factor, add, gate2b (pow gate), 3 seeds
  all    main, long
Exits with status 1 if any job failed or left no result file.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn
import b6_positivity as B
cfg = B.cfg


def jobs_main():
    v = cfg['variants']
    return [B.make_job('main', s, a, m, g, sd, save_traj=(sd == 0)) for s in cfg['systems'] for a in cfg['alphas']
            for sd in cfg['seeds'] for g in ('pow', 'exp') for m in v[g]]


def jobs_long():
    l = cfg['long_budget']
    return [B.make_job('long', s, a, m, l['gate'], sd, l['adam_iters'], l['lbfgs_iters']) for s in cfg['systems']
            for a in cfg['alphas'] for sd in l['seeds'] for m in l['maps']]


def main():
    stage = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for st in (['main', 'long'] if stage == 'all' else [stage]):
        jobs = dict(main=jobs_main, long=jobs_long)[st]()
        nbad = pinn.run_campaign(B.run_job, jobs, workers, os.path.join(B.RAW, st, 'campaign.log'), f'B6 {st}')
        if nbad:
            sys.exit(f'B6 {st}: {nbad} of {len(jobs)} jobs FAILED (see {os.path.join(B.RAW, st, "campaign.log")})')
        nok = sum(json.load(open(j['out']))['status'] == 'ok' for j in jobs)
        print(f'B6 {st}: {len(jobs)} result files, {len(jobs) - nok} runs with status != ok')


if __name__ == '__main__':
    main()

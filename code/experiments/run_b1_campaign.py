"""B1 campaign (configs/b1_scalar.json): 2 x 2 design {input} x {gate}, alpha_true in {0.3, 0.5, 0.8}, 10 seeds.

Usage: python run_b1_campaign.py <stage> [workers]      (references first: python b1_floor.py)
  main   forward problem and inverse problem (noise 0 and 0.01) on the common mesh r = 3
  mesh   inverse problem, noise-free, for the gradings r = 1.5, 7/3, 5 (5 seeds)
  all    main, mesh
Exits with status 1 if any job failed or left no result file.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn
import b1_scalar as B
cfg = B.cfg
CELLS = [(i, g) for i in cfg['inputs'] for g in cfg['gates']]


def jobs_main():
    jobs = [B.make_job('main', 'forward', a, i, g, s) for a in cfg['alphas_true'] for s in cfg['seeds'] for i, g in CELLS]
    return jobs + [B.make_job('main', 'inverse', a, i, g, s, noise=n) for a in cfg['alphas_true'] for n in cfg['inverse']['noise_sd']
                   for s in cfg['seeds'] for i, g in CELLS]


def jobs_mesh():
    m = cfg['mesh_sensitivity']
    return [B.make_job('mesh', 'inverse', a, i, g, s, noise=0.0, r=r) for r in m['r'] for a in cfg['alphas_true']
            for s in m['seeds'] for i, g in CELLS]


def main():
    stage = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for st in (['main', 'mesh'] if stage == 'all' else [stage]):
        jobs = dict(main=jobs_main, mesh=jobs_mesh)[st]()
        nbad = pinn.run_campaign(B.run_job, jobs, workers, os.path.join(B.RAW, st, 'campaign.log'), f'B1 {st}')
        if nbad:
            sys.exit(f'B1 {st}: {nbad} of {len(jobs)} jobs FAILED (see {os.path.join(B.RAW, st, "campaign.log")})')
        nok = sum(json.load(open(j['out']))['status'] == 'ok' for j in jobs)
        print(f'B1 {st}: {len(jobs)} result files, {len(jobs) - nok} runs with status != ok')


if __name__ == '__main__':
    main()

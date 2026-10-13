"""B7 campaign (configs/b7_inverse.json): PINN variants against classical nonlinear least squares through the implicit
L1 time stepper, 10 noise realisations per design and noise level.

Usage: python b7_campaign.py <stage> [workers]      (reference first: python b7_inverse.py prepare [workers])
  pilot  data-weight grid of the PINN on the pilot realisations; ends with the selection of one weight per
         (design, noise level), written to results/raw/B7_inverse/data_weight_selected.json
  main   PINN maps with the selected weight, and the least-squares fits, on the reported realisations
  all    pilot, main
Exits with status 1 if any job failed or left no result file.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fracsp import pinn
import b7_inverse as B
cfg = B.cfg; SEL = os.path.join(B.RAW, 'data_weight_selected.json')
CASES = [(d, n) for d in cfg['designs'] for n in cfg['designs'][d]['noise_sd']]


def jobs_pilot():
    P = cfg['pinn']
    return [B.make_job('pilot', 'pinn', d, n, r, P['pilot']['map'], w) for d, n in CASES for w in P['data_weights']
            for r in P['pilot']['realisations']]


def select_weights():
    P = cfg['pinn']; sel = {}; table = {}
    for d, n in CASES:
        med = {}
        for w in P['data_weights']:
            e = [json.load(open(B.make_job('pilot', 'pinn', d, n, r, P['pilot']['map'], w)['out'])) for r in P['pilot']['realisations']]
            med[w] = float('inf') if any(r['status'] != 'ok' for r in e) else float(np.median([r['rel_err_rms'] for r in e]))
        table[f'{d}_n{n:g}'] = {f'{w:g}': v for w, v in med.items()}; sel[f'{d}_n{n:g}'] = min(med, key=med.get)
    pinn.write_json(SEL, dict(selected=sel, median_rel_err_rms=table, rule=P['pilot']['selection'],
                              **pinn.run_metadata(config=P, driver=__file__)))
    print('selected data weights:', json.dumps(sel))


def jobs_main():
    sel = json.load(open(SEL))['selected']; jobs = []
    for d, n in CASES:
        for r in cfg['realisations']:
            jobs.append(B.make_job('main', 'nls', d, n, r))
            jobs += [B.make_job('main', 'pinn', d, n, r, m, sel[f'{d}_n{n:g}']) for m in cfg['pinn']['maps']]
    return jobs


def main():
    stage = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for st in (['pilot', 'main'] if stage == 'all' else [stage]):
        jobs = dict(pilot=jobs_pilot, main=jobs_main)[st]()
        nbad = pinn.run_campaign(B.run_job, jobs, workers, os.path.join(B.RAW, st, 'campaign.log'), f'B7 {st}')
        if nbad:
            sys.exit(f'B7 {st}: {nbad} of {len(jobs)} jobs FAILED (see {os.path.join(B.RAW, st, "campaign.log")})')
        if st == 'pilot':
            select_weights()
        nok = sum(json.load(open(j['out']))['status'] == 'ok' for j in jobs)
        print(f'B7 {st}: {len(jobs)} result files, {len(jobs) - nok} runs with status != ok')


if __name__ == '__main__':
    main()

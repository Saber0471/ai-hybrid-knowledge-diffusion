"""Stage-5.1 supplementary battery: beta and x0 sensitivity (reviewer-request anticipation)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_abm_mesa import run_monte_carlo, analyze, BASELINE_PARAMS
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation_data')

def main():
    out = {}
    # beta sensitivity at two (q, alpha) points x 3 topologies
    for topo in ['RR', 'WS', 'BA']:
        rows = []
        for q, al, tag in ((0.5, 0.0, 'coop'), (0.5, 0.5, 'defect'), (0.3, 0.25, 'mixed')):
            for beta in (5.0, 10.0, 20.0):
                p = dict(BASELINE_PARAMS); p.update({'q': q, 'alpha': al, 'beta': beta})
                a = analyze(run_monte_carlo(topo, n_replicates=16, params=p))
                a.update({'q': q, 'alpha': al, 'beta': beta, 'tag': tag})
                rows.append(a)
        out[f'beta_{topo}'] = rows
        print(f'[beta] {topo} done', flush=True)
    # x0 sensitivity at boundary region
    for topo in ['RR', 'WS', 'BA']:
        rows = []
        for q, al in ((0.5, 0.375), (0.5, 0.25), (0.5, 0.5)):
            for x0 in (0.1, 0.3, 0.5, 0.7, 0.9):
                p = dict(BASELINE_PARAMS); p.update({'q': q, 'alpha': al, 'x0': x0})
                a = analyze(run_monte_carlo(topo, n_replicates=16, params=p))
                a.update({'q': q, 'alpha': al, 'x0': x0})
                rows.append(a)
        out[f'x0_{topo}'] = rows
        print(f'[x0] {topo} done', flush=True)
    obj = {'design': 'beta x (basin,topology) 16 reps; x0 x (q=.5, alpha) x topology 16 reps',
           '_generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'data': out}
    with open(os.path.join(OUT, 'sensitivity_beta_x0.json'), 'w') as f:
        json.dump(obj, f, indent=1)
    print('[saved] sensitivity_beta_x0.json', flush=True)

if __name__ == '__main__':
    main()

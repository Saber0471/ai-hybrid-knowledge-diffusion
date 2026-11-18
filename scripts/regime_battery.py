"""
Regime-validation battery: does the simulated phase boundary track the analytic
prediction alpha* = [R - (1-rho)c]/b - 1 as the payoff ratio varies?
Runs after the main battery. Output: simulation_data/regime_scan.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_abm_mesa import run_monte_carlo, analyze, BASELINE_PARAMS
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation_data')

def main():
    out = {}
    for topo in ['RR', 'WS', 'BA']:
        rows = []
        for b in (0.8, 1.0, 1.2):
            alpha_star = (BASELINE_PARAMS['R'] - (1 - BASELINE_PARAMS['rho']) * BASELINE_PARAMS['c']) / b - 1
            for alpha in np.round(np.linspace(0.0, 1.0, 21), 3):
                p = dict(BASELINE_PARAMS)
                p.update({'q': 0.5, 'b': float(b), 'alpha': float(alpha), 'h': 0.0})
                a = analyze(run_monte_carlo(topo, n_replicates=16, params=p))
                a.update({'b': float(b), 'alpha': float(alpha), 'alpha_star': float(alpha_star)})
                rows.append(a)
            print(f"[regime] {topo} b={b} done (alpha*={alpha_star:.3f})", flush=True)
        out[topo] = rows
    obj = {'design': 'R/b regime validation at q=.5, h=0, 16 reps; analytic alpha* per row',
           '_meta': {k: BASELINE_PARAMS[k] for k in ('N', 'c', 'R', 'beta', 'K', 'p_ws', 'm_ba', 'rho')},
           '_generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'data': out}
    with open(os.path.join(OUT, 'regime_scan.json'), 'w') as f:
        json.dump(obj, f, indent=1)
    print('[saved] regime_scan.json', flush=True)

if __name__ == '__main__':
    main()

"""
Full experiment battery for ai_hybrid_knowledge_diffusion.
All outputs -> simulation_data/*.json (traceable per anti-fabrication rule #8).
Run: python3 scripts/run_experiments.py  (background recommended, ~1h)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Canonical implementation: Mesa 2.4 port (constitution trap #5: standardized
# ABM platform required for SCI/SSCI submission). Cross-validated against the
# hand-rolled implementation (18/18 x_eq CI overlap, flow field exact).
from hybrid_abm_mesa import (HybridDiffusionModel, run_monte_carlo, analyze,
                             BASELINE_PARAMS)
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation_data')
os.makedirs(OUT, exist_ok=True)
META = {k: BASELINE_PARAMS[k] for k in
        ('N', 'c', 'R', 'b', 'delta', 'beta', 'K', 'p_ws', 'm_ba',
         'x0', 'T_max', 'epsilon', 'window', 'rho', 'lam', 'gamma')}
TOPOS = ['RR', 'WS', 'BA']

def save(name, obj):
    obj['_meta'] = META
    obj['_generated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=1)
    print(f"[saved] {name}", flush=True)

def grid_scan():
    """Experiment 1: (q, alpha) phase diagram, primary battery."""
    qs = np.round(np.linspace(0.0, 1.0, 17), 3)
    alphas = np.round(np.linspace(0.0, 2.0, 17), 3)
    reps = 16  # raised from 12: bistable cells estimate basin proportions
    out = {}
    for topo in TOPOS:
        rows = []
        for q in qs:
            for al in alphas:
                p = dict(BASELINE_PARAMS); p['q'] = float(q); p['alpha'] = float(al)
                p['h'] = 0.0  # isolate incentive structure from noise
                a = analyze(run_monte_carlo(topo, n_replicates=reps, params=p))
                a['q'] = float(q); a['alpha'] = float(al)
                rows.append(a)
            print(f"[grid] {topo} q={q:.2f} done ({len(rows)}/{len(qs)*len(alphas)})", flush=True)
        out[topo] = rows
    save('grid_scan.json', {'design': 'q x alpha, h=0, 12 reps', 'data': out})

def h_scan():
    """Experiment 2: noise robustness (H3) — DUAL-BASIN design (v0.3).

    Basin A (q=.5, alpha=.5): substitution-dominated defection basin —
    noise taxes extraction -> predicted cooperation restoration.
    Basin B (q=.5, alpha=0): complementarity-dominated cooperation basin —
    noise erodes quality -> tests the original H3 prediction.
    """
    hs = np.round(np.linspace(0.0, 0.5, 11), 3)
    reps = 20
    out = {}
    for label, al in (('basinA_defection', 0.5), ('basinB_cooperation', 0.0)):
        for topo in TOPOS:
            rows = []
            for h in hs:
                p = dict(BASELINE_PARAMS); p['q'] = 0.5; p['alpha'] = al; p['h'] = float(h)
                a = analyze(run_monte_carlo(topo, n_replicates=reps, params=p))
                a['h'] = float(h)
                rows.append(a)
            out[f'{label}_{topo}'] = rows
            print(f"[h] {label} {topo} done", flush=True)
    save('h_scan.json', {'design': 'dual-basin h x topology, 20 reps', 'data': out})

def assignment_scan():
    """Experiment 3: AI assignment mode comparison."""
    modes = ['random', 'degree', 'homophily']
    points = [(0.3, 0.25), (0.3, 0.5), (0.6, 0.25), (0.6, 0.5)]
    reps = 15
    out = {}
    for topo in TOPOS:
        rows = []
        for mode in modes:
            for q, al in points:
                p = dict(BASELINE_PARAMS); p['q'] = q; p['alpha'] = al
                p['assignment'] = mode
                a = analyze(run_monte_carlo(topo, n_replicates=reps, params=p))
                a['mode'] = mode; a['q'] = q; a['alpha'] = al
                rows.append(a)
        out[topo] = rows
        print(f"[assign] {topo} done", flush=True)
    save('assignment_scan.json', {'design': 'mode x (q,alpha), 15 reps', 'data': out})

def finite_size():
    """Experiment 4: finite-size check."""
    Ns = [250, 500, 1000]
    reps = 10
    out = {}
    for topo in TOPOS:
        rows = []
        for N in Ns:
            p = dict(BASELINE_PARAMS); p['N'] = N
            a = analyze(run_monte_carlo(topo, n_replicates=reps, params=p))
            a['N'] = N
            rows.append(a)
        out[topo] = rows
        print(f"[N] {topo} done", flush=True)
    save('finite_size.json', {'design': 'N x topology at baseline, 10 reps', 'data': out})

def lambda_scan():
    """Experiment 5: robustness of corruption severity."""
    out = {}
    for lam in (0.3, 0.7):
        for topo in TOPOS:
            p = dict(BASELINE_PARAMS); p['q'] = 0.5; p['alpha'] = 0.5
            p['h'] = 0.2; p['lam'] = lam
            a = analyze(run_monte_carlo(topo, n_replicates=10, params=p))
            out[f'lam{lam}_{topo}'] = a
    print("[lambda] done", flush=True)
    save('lambda_scan.json', {'design': 'lambda robustness, 10 reps', 'data': out})

def flow_sensitivity():
    """Experiment 6 (v0.3): flow-layer parameter sensitivity."""
    out = {}
    for topo in TOPOS:
        rows = []
        for L in (2, 3, 4):
            for mu in (0.3, 0.5, 0.7):
                for q, al in ((0.3, 0.25), (0.6, 0.25)):
                    p = dict(BASELINE_PARAMS); p['q'] = q; p['alpha'] = al
                    p['L'] = L; p['mu'] = mu
                    a = analyze(run_monte_carlo(topo, n_replicates=10, params=p))
                    a.update({'L': L, 'mu': mu, 'q': q, 'alpha': al})
                    rows.append(a)
        out[topo] = rows
        print(f"[flow] {topo} done", flush=True)
    save('flow_sensitivity.json', {'design': 'L x mu x (q,alpha), 10 reps', 'data': out})

if __name__ == '__main__':
    t0 = time.time()
    print("=== experiment battery v2 (flow layer) start ===", flush=True)
    h_scan(); assignment_scan(); finite_size(); lambda_scan(); flow_sensitivity()
    grid_scan()
    print(f"=== all done in {(time.time()-t0)/60:.1f} min ===", flush=True)

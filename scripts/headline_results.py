"""
Headline-results extractor for sec5 writing.
Reads all archived JSONs and prints the key numbers (with source file + cell
coordinates) so every manuscript statistic has a traceable origin.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation_data')

def load(n):
    with open(os.path.join(DATA, n)) as f:
        return json.load(f)

def cell(rows, **kw):
    return next(r for r in rows if all(abs(r[k] - v) < 1e-6 for k, v in kw.items() if isinstance(v, float)) and all(r[k] == v for k, v in kw.items() if not isinstance(v, float)))

print("=== E1 grid_scan (h=0) ===")
g = load('grid_scan.json')['data']
for topo in ['RR', 'WS', 'BA']:
    rows = g[topo]
    conv = [r for r in rows if 'x' in r]
    print(f"[{topo}] cells={len(rows)} converged={len(conv)} rate={len(conv)/len(rows):.2f}")
    # column alpha=0 and alpha=0.5
    for al in (0.0, 0.125, 0.25, 0.5):
        r = cell(rows, q=0.5, alpha=al) if any(abs(r['alpha']-al)<1e-6 and r['q']==0.5 for r in rows) else None
        if r and 'x' in r:
            print(f"   q=0.5 alpha={al:.3f}: x={r['x']['mean']:.3f}±{r['x']['se']:.3f} "
                  f"Gini={r['gini_stock']['mean']:.3f} corr={r['corr_deg_stock']['mean']:.3f} "
                  f"Q={r['Q']['mean']:.1f} clean={r['clean_ratio']['mean']:.3f} conv={r['converged_rate']:.2f}")
    # q=0 column
    for al in (0.0, 0.5):
        r = next((x for x in rows if x['q'] == 0.0 and abs(x['alpha']-al) < 1e-6), None)
        if r and 'x' in r:
            print(f"   q=0.0 alpha={al:.3f}: x={r['x']['mean']:.3f} conv={r['converged_rate']:.2f}")
    # boundary: alpha where x crosses 0.5 at q=0.5
    xs = sorted([(r['alpha'], r['x']['mean']) for r in rows if r['q'] == 0.5 and 'x' in r])
    cross = [a for a, x in xs if 0.3 <= x <= 0.7]
    if cross:
        print(f"   boundary alphas (0.3<=x<=0.7) at q=0.5: {[round(a,3) for a in cross]}")

print("\n=== E2 h_scan dual basin ===")
h = load('h_scan.json')['data']
for label in ('basinA_defection', 'basinB_cooperation'):
    for topo in ['RR', 'WS', 'BA']:
        rows = sorted(h[f'{label}_{topo}'], key=lambda r: r['h'])
        s = ' | '.join(f"h={r['h']:.2f}:x={r['x']['mean']:.2f},cl={r['clean_ratio']['mean']:.2f},cv={r['converged_rate']:.2f}" for r in rows)
        print(f"[{label} {topo}] {s}")

print("\n=== E3 assignment ===")
a = load('assignment_scan.json')['data']
for topo in ['RR', 'WS', 'BA']:
    for m in ('random', 'degree', 'homophily'):
        r = next(x for x in a[topo] if x['mode'] == m and x['q'] == 0.6 and x['alpha'] == 0.25)
        print(f"[{topo} {m}] x={r['x']['mean']:.2f} Gini={r['gini_stock']['mean']:.2f} "
              f"corr={r['corr_deg_stock']['mean']:.2f} aiTop={r['x_ai_share_top']['mean']:.2f}")

print("\n=== E4 finite size ===")
f = load('finite_size.json')['data']
for topo in ['RR', 'WS', 'BA']:
    for r in f[topo]:
        print(f"[{topo} N={r['N']}] x={r['x']['mean']:.2f} Gini={r['gini_stock']['mean']:.2f} conv={r['converged_rate']:.2f}")

print("\n=== E5a lambda ===")
for k, v in load('lambda_scan.json')['data'].items():
    print(f"[{k}] x={v['x']['mean']:.2f} clean={v['clean_ratio']['mean']:.2f}")

print("\n=== E5b flow sensitivity ===")
fl = load('flow_sensitivity.json')['data']
for topo in ['RR', 'WS', 'BA']:
    base = [r for r in fl[topo] if r['L'] == 3 and r['mu'] == 0.5]
    for r in base:
        print(f"[{topo} base L=3 mu=0.5 q={r['q']} a={r['alpha']}] x={r['x']['mean']:.2f} Gini={r['gini_stock']['mean']:.2f}")
    for L in (2, 4):
        for mu in (0.3, 0.7):
            r = next(x for x in fl[topo] if x['L'] == L and x['mu'] == mu and x['q'] == 0.6 and x['alpha'] == 0.25)
            print(f"[{topo} L={L} mu={mu} q=0.6 a=0.25] x={r['x']['mean']:.2f} Gini={r['gini_stock']['mean']:.2f}")

print("\n=== E6 regime validation ===")
try:
    rg = load('regime_scan.json')['data']
    for topo in ['RR', 'WS', 'BA']:
        for b in (0.8, 1.0, 1.2):
            rows = sorted([r for r in rg[topo] if r['b'] == b and 'x' in r], key=lambda r: r['alpha'])
            xs = [(r['alpha'], r['x']['mean']) for r in rows]
            cross = [a for a, x in xs if x <= 0.5]
            sim_boundary = max(cross) if cross else float('nan')
            astar = rows[0]['alpha_star']
            print(f"[{topo} b={b}] analytic alpha*={astar:.3f}  sim boundary (last x<=0.5)={sim_boundary:.3f}")
except FileNotFoundError:
    print("regime_scan.json not yet generated")

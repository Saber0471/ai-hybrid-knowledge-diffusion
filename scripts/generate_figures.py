"""
Generate all manuscript figures from archived simulation outputs.
Data source: simulation_data/*.json ONLY (no hardcoded curves; anti-fabrication rule #8).
Style: ColorBrewer Set2, 300 DPI, PNG+PDF, no in-figure titles, white-box panel labels.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation_data')

SET2 = plt.get_cmap('Set2').colors
TOPO_STYLE = {'RR': SET2[0], 'WS': SET2[1], 'BA': SET2[2]}
TOPO_LABEL = {'RR': 'Regular (RR)', 'WS': 'Small-world (WS)', 'BA': 'Scale-free (BA)'}

def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {name}.png/pdf')

def panel_label(ax, s):
    ax.text(0.02, 0.98, s, transform=ax.transAxes, va='top', ha='left',
            fontsize=11, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.25'))

# ---------------- Fig 1: schematic + payoff table ----------------
def fig1():
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    G = nx.barabasi_albert_graph(26, 2, seed=7)
    deg = dict(G.degree())
    hubs = sorted(deg, key=deg.get, reverse=True)[:6]
    rng = np.random.RandomState(3)
    ai = set(hubs[:3]) | set(rng.choice([n for n in G if n not in hubs], 4, replace=False))
    pos = nx.spring_layout(G, seed=11)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#cccccc', width=1)
    h_nodes = [n for n in G if n not in ai]
    a_nodes = [n for n in G if n in ai]
    nx.draw_networkx_nodes(G, pos, nodelist=h_nodes, node_color=SET2[0], node_size=180, ax=ax, label='Human-only')
    nx.draw_networkx_nodes(G, pos, nodelist=a_nodes, node_color=SET2[2], node_size=300, edgecolors='black', ax=ax, label='AI-augmented')
    ax.legend(loc='lower left', frameon=True, fontsize=9)
    ax.set_title('(a) Human–AI hybrid knowledge network', fontsize=10)
    ax.axis('off')
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis('off')
    rows = ['C vs C', 'C vs F', 'F vs C', 'F vs F']
    cols = ['Human sender', 'AI sender']
    data = [['R − c', 'R − (1−ρ)c'],
            ['−c', '−(1−ρ)c'],
            ['b', '(1+α)b·disc'],
            ['δ', 'δ']]
    t2 = ax2.table(cellText=data, rowLabels=rows, colLabels=cols, loc='center', cellLoc='center')
    t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1, 1.6)
    ax2.set_title('(b) Per-interaction payoffs (focal agent first)', fontsize=10)
    save(fig, 'fig1_schematic')

# ---------------- Fig 2: (q, alpha) phase diagram ----------------
def fig2():
    g = load('grid_scan.json')['data']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, topo in zip(axes, ['RR', 'WS', 'BA']):
        rows = g[topo]
        qs = sorted(set(r['q'] for r in rows))
        als = sorted(set(r['alpha'] for r in rows))
        Z = np.full((len(als), len(qs)), np.nan)
        for r in rows:
            if 'x' in r:
                Z[als.index(r['alpha']), qs.index(r['q'])] = r['x']['mean']
        pcm = ax.pcolormesh(qs, als, Z, cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
        ax.contour(qs, als, Z, levels=[0.5], colors='black', linewidths=1.6)
        ax.axhline(0.35, color='white', ls='--', lw=1.4)
        ax.text(0.02, 0.375, r'analytic $\alpha^*$=0.35', color='white', fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_xlabel('AI penetration q'); ax.set_title(TOPO_LABEL[topo], fontsize=10)
        panel_label(ax, {'RR':'a','WS':'b','BA':'c'}[topo])
    axes[0].set_ylabel(r'extraction amplification $\alpha$')
    fig.colorbar(pcm, ax=axes, shrink=0.85, label='equilibrium contribution $x_{eq}$')
    save(fig, 'fig2_phase_diagram')

# ---------------- Fig 3: stratification ----------------
def fig3():
    g = load('grid_scan.json')['data']
    a = load('assignment_scan.json')['data']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for topo in ['RR', 'WS', 'BA']:
        rows = [r for r in g[topo] if abs(r['alpha'] - 0.25) < 1e-6 and 'gini_stock' in r]
        rows.sort(key=lambda r: r['q'])
        ax.plot([r['q'] for r in rows], [r['gini_stock']['mean'] for r in rows],
                'o-', color=TOPO_STYLE[topo], label=TOPO_LABEL[topo], ms=4)
    ax.set_xlabel('AI penetration q'); ax.set_ylabel('Gini of knowledge stock')
    ax.legend(fontsize=9); panel_label(ax, 'a')
    ax = axes[1]
    width = 0.27
    modes = ['random', 'degree', 'homophily']
    ml = {'random': 'random', 'degree': 'degree-biased', 'homophily': 'homophilic'}
    for i, topo in enumerate(['RR', 'WS', 'BA']):
        vals = []
        for m in modes:
            r = next((x for x in a[topo] if x['mode'] == m and x['q'] == 0.6 and x['alpha'] == 0.25), None)
            vals.append(r['x_ai_share_top']['mean'] if r else np.nan)
        xs = np.arange(3) + (i - 1) * width
        ax.bar(xs, vals, width=width, color=TOPO_STYLE[topo], label=TOPO_LABEL[topo])
    ax.axhline(0.6, color='gray', ls=':', lw=1)
    ax.text(-0.38, 0.615, 'q (random baseline)', fontsize=8, color='gray')
    ax.set_xticks(range(3)); ax.set_xticklabels([ml[m] for m in modes])
    ax.set_ylabel('AI share of top-10% degree nodes')
    ax.set_ylim(0, 1.15); ax.legend(fontsize=8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.14)); panel_label(ax, 'b')
    save(fig, 'fig3_stratification')

# ---------------- Fig 4: dual-basin noise ----------------
def fig4():
    h = load('h_scan.json')['data']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for topo in ['RR', 'WS', 'BA']:
        rows = sorted(h[f'basinA_defection_{topo}'], key=lambda r: r['h'])
        ax.plot([r['h'] for r in rows], [r['x']['mean'] for r in rows],
                'o-', color=TOPO_STYLE[topo], label=TOPO_LABEL[topo], ms=4)
    ax.set_xlabel('corruption rate h'); ax.set_ylabel(r'contribution $x_{eq}$')
    ax.set_ylim(-0.03, 1.05); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(r'(a) defection basin ($\alpha$=0.5): noise taxes extraction', fontsize=10)
    ax.legend(fontsize=9); panel_label(ax, 'a')
    ax = axes[1]
    for topo in ['RR', 'WS', 'BA']:
        rows = sorted(h[f'basinB_cooperation_{topo}'], key=lambda r: r['h'])
        ax.plot([r['h'] for r in rows], [r['clean_ratio']['mean'] for r in rows],
                'o-', color=TOPO_STYLE[topo], label=TOPO_LABEL[topo], ms=4)
    ax.set_xlabel('corruption rate h'); ax.set_ylabel('quality ratio (clean share)')
    ax.set_ylim(0.7, 1.02); ax.set_yticks(np.arange(0.7, 1.01, 0.05))
    ax.set_title(r'(b) cooperation basin ($\alpha$=0): noise erodes quality', fontsize=10)
    ax.legend(fontsize=9); panel_label(ax, 'b')
    save(fig, 'fig4_noise_dual_basin')

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print('all figures generated')

"""
ABM Simulation for Knowledge Diffusion in Human-AI Hybrid Networks
==================================================================
Mesa 2.x port of scripts/hybrid_abm.py (hand-written numpy+networkx
implementation). The two files are twin implementations of the SAME model
(design doc: doc/model_design.md v0.2, audited); every payoff term maps 1:1
to Section 3 of that document. Semantics are identical; only the random
number streams differ (this version routes all stochasticity through
self.random, Mesa's per-model Random instance — required by ScholarForge
Constitution trap #5: SCI/SSCI submissions must use a standard ABM platform).

Differences vs the hand-written version (RNG only, no semantic change):
- All random draws use model.random (Mersenne Twister, seeded per replicate)
  instead of the global numpy RandomState. NetworkX graph seeds are derived
  from model.random. Distribution-level equivalence is verified against
  hybrid_abm.py (payoff cells, flow-layer analytics, Monte Carlo CIs).
- Network/topology, type assignment, Fermi updates, item corruption draws,
  update order: same distributions, different streams.

Author: single author
Date: 2026-11-17
"""

import numpy as np
import networkx as nx
import mesa
import json
import os
import time
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MODEL PARAMETERS
# ============================================================
BASELINE_PARAMS = {
    # inherited from parent v2
    'N': 500,
    # NOTE (2026-11-17, design decision after audit): parent v2 params (R=1.0,
    # b=1.5) make F strictly dominant for any c>0 (b > R-c and delta > -c),
    # so the cooperation basin is empty and no phase structure exists. The
    # minimal non-degenerate public-goods regime requires R - c > b:
    'c': 0.3, 'R': 1.5, 'b': 1.0, 'delta': 0.1,
    'beta': 10.0,
    'K': 4, 'p_ws': 0.3, 'm_ba': 2,
    'x0': 0.5,
    'T_max': 1000, 'epsilon': 1e-4, 'window': 100,
    # new: AI access asymmetry
    'q': 0.3,          # fraction of AI-augmented agents
    'alpha': 0.5,      # absorption amplification for AI-augmented
    'rho': 0.5,        # production cost reduction for AI-augmented
    'h': 0.1,          # corruption probability of AI-augmented transmission
    'lam': 0.5,        # quality loss per corrupted transmission
    'gamma': 0.02,     # geometric decay of knowledge stock
    'L': 3,            # flow horizon (max hops, v0.3 flow layer)
    'mu': 0.5,         # per-hop attenuation of knowledge flow (v0.3)
    'assignment': 'random',   # random | degree | homophily
}

# ============================================================
# NETWORK CREATION (inherited)
# ============================================================

def create_network(topology, N, K, p_ws=0.3, m_ba=2, seed=None):
    if topology == 'RR':
        G = nx.random_regular_graph(d=K, n=N, seed=seed)
    elif topology == 'WS':
        G = nx.watts_strogatz_graph(n=N, k=K, p=p_ws, seed=seed)
    elif topology == 'BA':
        G = nx.barabasi_albert_graph(n=N, m=m_ba, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return G

# ============================================================
# TYPE ASSIGNMENT (new; design doc P1-4)
# ============================================================

def assign_types(G, q, mode, rng):
    """Return boolean array is_ai of length N. `rng` is model.random."""
    N = G.number_of_nodes()
    n_ai = int(round(N * q))
    nodes = np.arange(N)
    degrees = np.array([d for _, d in G.degree()], dtype=float)

    if mode == 'random':
        chosen = np.array(sorted(rng.sample(range(N), n_ai)))
    elif mode == 'degree':
        # weighted sampling without replacement (sequential, renormalized):
        # distributionally equivalent to np.random.choice(..., replace=False, p)
        w = degrees.copy()
        chosen_list = []
        for _ in range(n_ai):
            total = w.sum()
            r = rng.random() * total
            cum = np.cumsum(w)
            idx = int(np.searchsorted(cum, r, side='right'))
            idx = min(idx, N - 1)
            chosen_list.append(idx)
            w[idx] = 0.0
        chosen = np.array(sorted(chosen_list))
    elif mode == 'homophily':
        # greedy: seed with highest-degree node, grow along adjacency frontier
        adj = {n: set(G.neighbors(n)) for n in nodes}
        chosen_set = set()
        frontier = []
        cand = int(nodes[np.argmax(degrees)])
        while len(chosen_set) < n_ai:
            if cand not in chosen_set:
                chosen_set.add(cand)
                frontier.extend(adj[cand] - chosen_set)
            if frontier:
                # pick frontier node with highest degree
                cand = max(frontier, key=lambda n: degrees[n])
                frontier.remove(cand)
            else:
                remaining = [n for n in nodes if n not in chosen_set]
                cand = max(remaining, key=lambda n: degrees[n])
        chosen = np.array(sorted(chosen_set))
    else:
        raise ValueError(f"Unknown assignment mode: {mode}")
    is_ai = np.zeros(N, dtype=bool)
    if n_ai > 0:
        is_ai[np.asarray(chosen, dtype=int)] = True
    return is_ai

# ============================================================
# AGENT AND MODEL (Mesa 2.x)
# ============================================================

class HybridKnowledgeAgent(mesa.Agent):
    """Agent with fixed type (H/A) and evolving strategy (C/F)."""

    def __init__(self, unique_id, model, agent_type, strategy):
        super().__init__(unique_id, model)
        self.agent_type = agent_type      # 'H' or 'A', fixed (exogenous)
        self.strategy = strategy          # 'C' or 'F'
        self.payoff = 0.0
        self.stock = 0.0                  # total knowledge stock K_i
        self.clean_stock = 0.0            # clean knowledge stock S_i
        self.item_corrupted = False       # v0.3: this step's item draw (sender side)

    # --- per-neighbor payoff (16-cell matrix, design doc Section 3, P1-5) ---
    # Asymmetric amplification: AI augments extraction (b) but not mutual
    # exchange (R); AI reduces production cost. This keeps the substitution
    # channel mathematically live: for high alpha, F dominates within AI type.
    # Noise (design doc Section 3b): receiver-side discount uses the sender's
    # ITEM-LEVEL corruption flag (one draw per item at generation).
    def pair_payoff(self, neighbor):
        """Payoff from one unordered interaction with a neighbor."""
        p = self.model.params
        extract_mult = (1 + p['alpha']) if self.agent_type == 'A' else 1.0
        cost = (1 - p['rho']) * p['c'] if self.agent_type == 'A' else p['c']

        if self.strategy == 'C' and neighbor.strategy == 'C':
            return p['R'] - cost
        elif self.strategy == 'C' and neighbor.strategy == 'F':
            return -cost
        elif self.strategy == 'F' and neighbor.strategy == 'C':
            # knowledge flows neighbor -> self; discount if neighbor's item corrupted
            disc = (1 - p['lam']) if neighbor.item_corrupted else 1.0
            return extract_mult * p['b'] * disc
        else:  # F-F
            return p['delta']

    def calculate_payoff(self):
        neighbors = list(self.model.G.neighbors(self.unique_id))
        if not neighbors:
            self.payoff = self.model.params['delta']
            return
        total = sum(self.pair_payoff(self.model.id_map[n]) for n in neighbors)
        self.payoff = total / len(neighbors)

    def decay_and_snapshot(self):
        p = self.model.params
        self.stock = (1 - p['gamma']) * self.stock
        self.clean_stock = (1 - p['gamma']) * self.clean_stock

    def update_strategy(self):
        """Fermi update: copy strategy (never type) from a random neighbor."""
        beta = self.model.params['beta']
        neighbors = list(self.model.G.neighbors(self.unique_id))
        if not neighbors:
            return
        nid = self.model.random.choice(neighbors)
        neighbor = self.model.id_map[nid]
        delta_payoff = neighbor.payoff - self.payoff
        prob = 1.0 / (1.0 + np.exp(-beta * delta_payoff))
        if self.model.random.random() < prob:
            self.strategy = neighbor.strategy


class HybridDiffusionModel(mesa.Model):
    def __init__(self, topology='RR', params=None, seed=None):
        super().__init__(seed=seed)
        self.params = dict(BASELINE_PARAMS if params is None else params)
        self.topology = topology
        self.seed = seed

        nx_seed = self.random.randrange(2**31 - 1)
        self.G = create_network(topology, self.params['N'], self.params['K'],
                                self.params['p_ws'], self.params['m_ba'], seed=nx_seed)
        self.degrees = np.array([d for _, d in self.G.degree()], dtype=float)
        # edge list for the flow layer (v0.3): bincount-based hop propagation.
        # NOTE: dense matmul (A @ w) is NOT used — this machine's BLAS returns
        # garbage for 500x500 float64 matmul (spurious 'divide by zero in
        # matmul' + wrong values, numpy+Accelerate bug). bincount avoids BLAS.
        self.edge_src, self.edge_dst = zip(*self.G.edges()) if self.G.number_of_edges() else ([], [])
        self.edge_src = np.array(self.edge_src, dtype=int)
        self.edge_dst = np.array(self.edge_dst, dtype=int)

        N = self.params['N']
        self.is_ai = assign_types(self.G, self.params['q'],
                                  self.params['assignment'], self.random)

        n_c = int(N * self.params['x0'])
        strategies = ['C'] * n_c + ['F'] * (N - n_c)
        self.random.shuffle(strategies)

        self.agent_list = []
        self.id_map = {}
        for i in range(N):
            atype = 'A' if self.is_ai[i] else 'H'
            a = HybridKnowledgeAgent(i, self, atype, strategies[i])
            self.agent_list.append(a)
            self.id_map[i] = a

        self.n_corrupt_events = 0
        self.series = {'x': [], 'x_A': [], 'x_H': []}
        self.steady = {k: [] for k in
                       ['x', 'x_A', 'x_H', 'gini_stock', 'corr_deg_stock',
                        'Q', 'clean_ratio', 'x_ai_share_top', 'corrupt_per_step']}
        self.converged = False
        self.convergence_time = None

    # --- item-level corruption draw (design doc Section 3b) ---
    def generate_items(self):
        """Each C agent produces one item; AI-augmented items corrupt w.p. h.
        ONE draw per item; the flag governs both payoff discount (1-hop
        exchange) and flow-layer clean-stock accounting (multi-hop)."""
        p = self.params
        for a in self.agent_list:
            a.item_corrupted = False
            if a.strategy == 'C' and a.agent_type == 'A':
                if self.random.random() < p['h']:
                    a.item_corrupted = True
                    self.n_corrupt_events += 1

    # --- flow layer (v0.3): L-hop attenuated diffusion of this step's items ---
    def flow_inflows(self):
        """Return (inflow_total, inflow_clean) vectors per agent.

        inflow_i = absorb_i * sum_{s in C, 1<=dist<=L} mu^dist * weight(s)
        weight = 1 for total stock; 0 for corrupted items in clean field.
        """
        p = self.params
        n = len(self.agent_list)
        w_total = np.array([1.0 if a.strategy == 'C' else 0.0 for a in self.agent_list])
        w_clean = np.array([1.0 if (a.strategy == 'C' and not a.item_corrupted)
                            else 0.0 for a in self.agent_list])
        inflow_total = np.zeros(n)
        inflow_clean = np.zeros(n)
        es, ed = self.edge_src, self.edge_dst
        for d in range(1, p['L'] + 1):
            # one undirected hop: values flow along edges in both directions
            w_total = (np.bincount(ed, weights=w_total[es], minlength=n)
                       + np.bincount(es, weights=w_total[ed], minlength=n))
            w_clean = (np.bincount(ed, weights=w_clean[es], minlength=n)
                       + np.bincount(es, weights=w_clean[ed], minlength=n))
            inflow_total += p['mu'] ** d * w_total
            inflow_clean += p['mu'] ** d * w_clean
        absorb = np.where(self.is_ai, 1 + p['alpha'], 1.0)
        return absorb * inflow_total, absorb * inflow_clean

    def step(self):
        # 1. item generation + single corruption draw per item
        self.generate_items()

        # 2. payoffs (game layer, 1-hop, uses item flags)
        for a in self.agent_list:
            a.calculate_payoff()

        # 3. flow layer: multi-hop attenuated diffusion -> stock inflow
        inflow_total, inflow_clean = self.flow_inflows()
        for idx, a in enumerate(self.agent_list):
            a.stock += inflow_total[idx]
            a.clean_stock += inflow_clean[idx]

        # 4. strategy updating (Fermi; copies strategy, never type)
        order = self.random.sample(range(len(self.agent_list)), len(self.agent_list))
        for idx in order:
            self.agent_list[idx].update_strategy()

        # 5. stock decay
        for a in self.agent_list:
            a.decay_and_snapshot()

        # record series
        N = self.params['N']
        x = sum(1 for a in self.agent_list if a.strategy == 'C') / N
        ai_ids = np.where(self.is_ai)[0]
        hu_ids = np.where(~self.is_ai)[0]
        x_A = sum(1 for i in ai_ids if self.id_map[int(i)].strategy == 'C') / max(len(ai_ids), 1)
        x_H = sum(1 for i in hu_ids if self.id_map[int(i)].strategy == 'C') / max(len(hu_ids), 1)
        self.series['x'].append(x)
        self.series['x_A'].append(x_A)
        self.series['x_H'].append(x_H)

        # steady-window snapshots (P1-1: metrics as steady-state means over the
        # final convergence window; taken every step once window length reached)
        w = self.params['window']
        if len(self.series['x']) >= w:
            self._snapshot_steady(x, x_A, x_H)

    def _snapshot_steady(self, x, x_A, x_H):
        s = self.steady
        s['x'].append(x); s['x_A'].append(x_A); s['x_H'].append(x_H)
        stocks = np.array([a.stock for a in self.agent_list])
        clean = np.array([a.clean_stock for a in self.agent_list])
        s['gini_stock'].append(gini(stocks))
        s['corr_deg_stock'].append(safe_pearson(self.degrees, stocks))
        s['Q'].append(clean.mean())
        # quality ratio: aggregate clean / total stock (robust to volume effects)
        tot = stocks.sum()
        s['clean_ratio'].append(float(clean.sum() / tot) if tot > 1e-12 else 1.0)
        s['corrupt_per_step'].append(self.n_corrupt_events)
        self.n_corrupt_events = 0
        # share of AI agents among top-10% degree nodes
        k90 = np.percentile(self.degrees, 90)
        top = np.where(self.degrees >= k90)[0]
        s['x_ai_share_top'].append(float(self.is_ai[top].mean()))

    def run_simulation(self):
        p = self.params
        T_max, eps, w = p['T_max'], p['epsilon'], p['window']
        for t in range(T_max):
            self.step()
            if t >= w:
                conv = all(max(self.series[k][-w:]) - min(self.series[k][-w:]) < eps
                           for k in ('x', 'x_A', 'x_H'))
                if conv:
                    self.converged = True
                    self.convergence_time = t
                    break
        if not self.converged:
            self.convergence_time = T_max

        steady_mean = {k: float(np.mean(v)) if v else float('nan')
                       for k, v in self.steady.items()}
        return {
            'topology': self.topology,
            'converged': self.converged,
            't_eq': self.convergence_time,
            'params': {k: p[k] for k in ('q', 'alpha', 'rho', 'h', 'lam', 'gamma', 'assignment')},
            'steady': steady_mean,
            'seed': self.seed,
        }

# ============================================================
# METRICS HELPERS
# ============================================================

def gini(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))

def safe_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# ============================================================
# MONTE CARLO + ANALYSIS
# ============================================================

def run_monte_carlo(topology, n_replicates=20, params=None, seed_offset=0):
    results = []
    for rep in range(n_replicates):
        seed = seed_offset + rep
        model = HybridDiffusionModel(topology=topology, params=params, seed=seed)
        r = model.run_simulation()
        r['replicate'] = rep
        results.append(r)
    return results

def analyze(results):
    conv = [r for r in results if r['converged']]
    out = {'n_total': len(results), 'n_converged': len(conv),
           'converged_rate': len(conv) / len(results)}
    if not conv:
        return out
    for key in ('x', 'x_A', 'x_H', 'gini_stock', 'corr_deg_stock', 'Q',
                'clean_ratio', 'x_ai_share_top', 'corrupt_per_step'):
        vals = [r['steady'][key] for r in conv]
        out[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'se': float(np.std(vals) / np.sqrt(len(vals))),
            'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        }
    tvals = [r['t_eq'] for r in conv]
    out['t_eq'] = {'mean': float(np.mean(tvals)), 'std': float(np.std(tvals))}
    return out

# ============================================================
# TIMING / FEASIBILITY (decision: small-scale timing first)
# ============================================================

def timing_test():
    print("=" * 60)
    print("Feasibility timing test (Mesa 2.x port)")
    print("=" * 60)
    configs = [('RR', BASELINE_PARAMS),
               ('WS', BASELINE_PARAMS),
               ('BA', BASELINE_PARAMS)]
    per_sim = {}
    for topo, params in configs:
        t0 = time.time()
        model = HybridDiffusionModel(topology=topo, params=params, seed=0)
        r = model.run_simulation()
        dt = time.time() - t0
        per_sim[topo] = dt
        print(f"  {topo}: {dt:.2f}s/sim  converged={r['converged']} "
              f"t_eq={r['t_eq']} x={r['steady']['x']:.3f} "
              f"Gini={r['steady']['gini_stock']:.3f} corr={r['steady']['corr_deg_stock']:.3f} "
              f"Q={r['steady']['Q']:.3f}")
    # grid estimate: n_topo(3) x n_grid^2 x reps
    for grid, reps in [(12, 10), (16, 15), (20, 20)]:
        worst = max(per_sim.values())
        est_hours = 3 * grid * grid * reps * worst / 3600
        print(f"  grid {grid}x{grid} x {reps} reps x 3 topologies: ~{est_hours:.1f} h (worst-topology bound)")
    return per_sim

if __name__ == '__main__':
    timing_test()

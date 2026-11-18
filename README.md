# Knowledge Stratification in Human-AI Hybrid Networks — Replication Package

Replication package for the paper *"Knowledge Stratification in Human-AI Hybrid Networks: An Evolutionary Game and Agent-Based Study of AI Access Asymmetry in Knowledge Diffusion"* (JASSS submission).

## Model

Two-type (human-only / AI-augmented) evolutionary social-dilemma game on complex networks (RR/WS/BA), coupled to a multi-hop attenuated knowledge-flow layer with item-level corruption. Canonical implementation: **Mesa 2.4** (`scripts/hybrid_abm_mesa.py`); an independently developed reference implementation was cross-validated against it (payoff matrix 16/16 cells; distributional equivalence on benchmark configurations).

## Reproduce

```bash
pip install mesa networkx numpy   # Mesa 2.4.x
python3 scripts/run_experiments.py      # full battery (~2 h): grid, h dual-basin,
                                        # assignment, finite-size, lambda, flow sensitivity
python3 scripts/regime_battery.py       # R/b regime validation (~25 min)
python3 scripts/sensitivity_battery.py  # beta / x0 sensitivity (~10 min)
python3 scripts/headline_results.py     # every statistic quoted in the paper
python3 scripts/generate_figures.py     # Figures 1-4 (PNG+PDF)
```

All runs are seed-fixed (seed = replicate index); every statistic in the paper is recomputable from `simulation_data/*.json`, which are the exact archived outputs of the runs above.

## Data

`simulation_data/` contains the complete output of all experiment batteries (Monte Carlo means, SEs, 95% percentile intervals, convergence rates per condition). No empirical data are used; the study is theory + simulation.

## License

Code: MIT. Data (JSON outputs): CC BY 4.0. Paper: all rights reserved by the author until publication.

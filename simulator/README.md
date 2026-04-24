# Paper-aligned simulator for GIDM and SC-GIDM

This package is a clean rebuild aimed at the project's actual mechanism, rather than the earlier exploratory simulator.

## What it implements

- **GIDM-style allocation and payment core** on the **optimal allocation tree / dominator tree** representation.
- **SC-GIDM** with:
  - dominator clustering,
  - macro-cluster DAG construction,
  - SCM-style random shortest-path tree sampling on the cluster graph,
  - macro GIDM,
  - lifting back to identities where a winning cluster gives the item to its highest bidder and an intermediary cluster routes reward to its root.
- **Split attacks** where one real buyer creates fake descendants.
- Metrics for:
  - seller revenue,
  - attacker utility,
  - attacker reward mass,
  - real welfare,
  - fake winners,
  - reward mass to fake nodes,
  - reward mass to real nodes.

## Important scope note

This code is the **best project-relevant simulator** for the current work. It is much closer to the paper pseudocode than the earlier exploratory code because it explicitly models:

- the optimal allocation tree / dominator-tree backbone,
- DFS/LIFO-style GIDM processing,
- displacement bookkeeping through `GetFrom`,
- welfare-counterfactual payments,
- SCM-style tree sampling in SC-GIDM.

However, this is still a **research implementation**, not an official reproduction package from the original authors. The intended use is:

1. reproduce the project's running examples,
2. compare GIDM vs SC-GIDM on random graph families,
3. inspect how attacks change revenue and reward distribution.

## Files

- `src/gidm_tree.py` — paper-aligned GIDM core.
- `src/sc_gidm.py` — SC-GIDM wrapper.
- `src/graph_utils.py` — dominators, clusters, shortest-path tree sampling.
- `src/attacks.py` — split attacks.
- `src/metrics.py` — welfare/revenue/reward accounting.
- `src/examples.py` — your presentation example graphs.
- `run_examples.py` — runs the user/presentation example and writes `results/user_examples.json`.
- `run_random_study.py` — random benchmarks on a Price-style DAG family.
- `plot_results.py` — generate plots from benchmark summaries.

## Usage

Run the presentation/user example:

```bash
python run_examples.py
```

Run random comparisons:

```bash
python run_random_study.py --out_dir results --seeds 200 --n 100 --m 2 --K 5 --qmax 4
```

Generate plots:

```bash
python plot_results.py --results_dir results
```

## Output

- `results/user_examples.json`
- `results/random_study.csv`
- `results/random_summary.csv`
- one plot per metric

## Recommended sanity checks

Before trusting large experiments, first inspect `results/user_examples.json` and check:

- GIDM on the truthful user graph,
- GIDM under the fake-child attack,
- SC-GIDM on the attacked graph,
- the change in the **attacker's reward mass** and **attacker utility**.

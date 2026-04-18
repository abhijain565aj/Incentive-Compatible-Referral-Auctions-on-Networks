
# Sybil-Robust Multi-Item Diffusion Auctions

This folder contains the simulator used in the project report and presentation.

## What is implemented

- `local_k_vickrey`: local homogeneous-item auction among seller neighbours only.
- `gidm`: adapted implementation of the baseline GIDM mechanism for multiple homogeneous items.
- `sc_gidm`: our proposed cluster-reduction mechanism:
  1. contract reported identities into Sybil clusters,
  2. keep only the certified non-Sybil root bid for each cluster,
  3. run GIDM on the cluster graph.

The baseline GIDM code is adapted from an MIT-licensed public implementation and repackaged here with attribution.

## Structure

- `src/gidm_adapted.py` - baseline GIDM and local baseline.
- `src/cluster_gidm.py` - SC-GIDM.
- `src/attacks.py` - star/chain Sybil attack generators.
- `src/sim_utils.py` - graph generators and attacker selection.
- `src/utils.py` - welfare, utility, fake-winner metrics.
- `run_experiments.py` - regenerate all tables/plots used in the report.

## Reproducing the main results

```bash
cd code
python run_experiments.py --out_dir results --seeds 120
```

This writes:

- `results/canonical_family.csv`
- `results/random_experiments.csv`
- `results/random_summary.csv`
- `results/*.png` plots used in the report

## Important modelling assumptions

Our theorem-level claims are for:

- homogeneous items,
- single-unit demand,
- no collusion across real buyers,
- a perfect Sybil-cluster oracle with a certified non-Sybil root per cluster.

The simulations use the ground-truth owner map as the oracle.

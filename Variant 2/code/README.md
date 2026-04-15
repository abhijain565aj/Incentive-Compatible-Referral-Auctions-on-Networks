# Externality-aware referral auction simulator

This code accompanies the final project report. It implements a small but extensible simulator for single-item referral auctions on rooted trees under non-i.i.d. valuations and public/estimated negative externalities.

## Mechanisms included

- `Local-Vickrey`: only seller's direct neighbors participate.
- `Referral-SP`: transformed referral second price; this is the tree-level IDM-style benchmark used in the report.
- `LbLEV`: prior-tuned exponent mechanism inspired by Bhattacharyya et al.
- `EA-Referral-SP`: our externality-adjusted threshold mechanism.
- `EA-Virtual-RA`: our externality-regularized Myerson/referral mechanism for uniform-max branch priors.

## Run

```bash
cd code
python -m refauc.experiments --out ../results --n 20 40 --eta 0 0.5 1 2 3 --runs 100
python -m refauc.truthfulness_checks --max-value 30
python -m refauc.mean_externality_routing
```

Outputs are written to `results/runs.csv`, `results/summary.csv`, and PNG plots.

## Extension points

- Replace `branch_externality` in `refauc/mechanisms.py` with another public externality model.
- Add graph-to-tree extraction before mechanisms in `refauc/network.py`.
- Replace uniform-max virtual values in `externality_adjusted_virtual` with empirical virtual values learned from samples.

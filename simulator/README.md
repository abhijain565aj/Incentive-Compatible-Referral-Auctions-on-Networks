# Referral Auction Simulator (Config-Driven)

This simulator runs referral-auction experiments and plotting from one entry point.

## Mechanisms included

- `central_vickrey`
- `local_vickrey`
- `network_vcg`
- `idm`
- `param_referral`
- `sybil_resistant_referral`

For the Li et al. (AAAI-17) paper comparison, use:
- `network_vcg` (the paper's network extension of VCG / modified VCG)
- `idm` (Information Diffusion Mechanism)

## Main workflow

`run_experiments.py` does all of this, in order:

1. Reads all parameters from `config.ini`
2. Runs simulations for every configured grid combination
3. Saves run-level and summary CSV files
4. Automatically generates clean plots in the configured plot folder

## Quick start

```bash
cd simulator
venv
python -m pip install -r requirements.txt
python run_experiments.py
```

Use a custom config file:

```bash
python run_experiments.py --config my_config.ini
```

## Config file (`config.ini`)

All simulation settings are editable in `config.ini`.

### `[paths]`
- `raw_results_csv`: merged run-level CSV
- `summary_results_csv`: grouped summary CSV
- `per_run_dir`: one CSV per `(valuation_mode, diffusion_strategy)`
- `plot_dir`: folder where all plots are saved

### `[simulation]`
- `seed_start`, `seed_count`: seeds are `range(seed_start, seed_start + seed_count)`
- `sizes`: comma-separated node counts
- `topologies`: comma-separated list from `line,star,tree,er,ba`
- `valuation_modes`: comma-separated list from `uniform,exponential,depth_biased,community,lognormal`
- `diffusion_strategies`: comma-separated list from `full,none,probabilistic`
- `mechanisms`: comma-separated mechanism names to run and compare
- `invite_prob`: used only for `probabilistic` diffusion

Available mechanism names:
- `modified_vcg` (alias of `network_vcg`), `network_vcg`, `idm`, `central_vickrey`, `local_vickrey`, `param_referral`, `sybil_resistant_referral`

Default in `config.ini` is now:
- `mechanisms = modified_vcg,idm`

### `[topology]`
- `tree_branching`
- `ba_m`
- `er_p_max`
- `er_p_scale`

## Example bigger run

Set in `config.ini`:

```ini
[simulation]
seed_start = 0
seed_count = 40
sizes = 30,60,90,120
topologies = line,star,tree,er,ba
valuation_modes = uniform,lognormal,depth_biased,exponential
diffusion_strategies = full,probabilistic
invite_prob = 0.65
```

Then run:

```bash
python run_experiments.py
```

## Outputs

- merged run-level CSV at `raw_results_csv`
- summary CSV at `summary_results_csv`
- per-grid CSV files in `per_run_dir`
- clean plots in `plot_dir`

Additional welfare metrics are computed per run and plotted:
- `welfare_sum_utilities` = sum of utilities of participants
- `welfare_product_utilities` = product of utilities of participants
- `welfare_log_product_utilities` = log product of utilities (defined when all participant utilities are positive)

## Accuracy + plotting notes

- Probabilistic diffusion now correctly uses `invite_prob` from config.
- Topology generation parameters are fully configurable.
- Plots are saved with constrained layouts, rotated labels, and moved legends to avoid overlap.

## Run tests

```bash
venv
PYTHONPATH=$PWD pytest -q
```

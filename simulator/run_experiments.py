from __future__ import annotations

import argparse
import configparser
import csv
from pathlib import Path
from typing import List

from plot_results import main as plot_main
from refauc.experiments import summarize, sweep


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_from_config(config_path: str = "config.ini") -> None:
    cfg = configparser.ConfigParser()
    loaded = cfg.read(config_path)
    if not loaded:
        raise FileNotFoundError(f"Could not read config file: {config_path}")

    paths = cfg["paths"]
    sim = cfg["simulation"]
    topo = cfg["topology"]

    raw_results = Path(paths.get("raw_results_csv", "results/sample_results.csv"))
    summary_results = Path(paths.get("summary_results_csv", "results/summary_results.csv"))
    per_run_dir = Path(paths.get("per_run_dir", "results/per_run"))
    plot_dir = Path(paths.get("plot_dir", "results/plots"))

    seed_start = sim.getint("seed_start", 0)
    seed_count = sim.getint("seed_count", 12)
    seeds = range(seed_start, seed_start + seed_count)
    sizes = _parse_int_list(sim.get("sizes", "20,40"))
    valuation_modes = _parse_csv_list(sim.get("valuation_modes", "uniform,lognormal,depth_biased"))
    diffusion_strategies = _parse_csv_list(sim.get("diffusion_strategies", "full,probabilistic"))
    topologies = _parse_csv_list(sim.get("topologies", "line,star,tree,er,ba"))
    mechanisms = _parse_csv_list(sim.get("mechanisms", "network_vcg,idm"))
    invite_prob = sim.getfloat("invite_prob", 0.7)

    tree_branching = topo.getint("tree_branching", 2)
    ba_m = topo.getint("ba_m", 2)
    er_p_max = topo.getfloat("er_p_max", 0.25)
    er_p_scale = topo.getfloat("er_p_scale", 3.0)

    all_rows = []
    per_run_dir.mkdir(parents=True, exist_ok=True)
    for valuation_mode in valuation_modes:
        for diffusion_strategy in diffusion_strategies:
            per_run_path = per_run_dir / f"{valuation_mode}_{diffusion_strategy}.csv"
            effective_invite_prob = invite_prob if diffusion_strategy == "probabilistic" else 1.0
            rows = sweep(
                out_csv=str(per_run_path),
                seeds=seeds,
                sizes=sizes,
                topologies=topologies,
                valuation_mode=valuation_mode,
                diffusion_strategy=diffusion_strategy,
                mechanisms=mechanisms,
                invite_prob=effective_invite_prob,
                tree_branching=tree_branching,
                ba_m=ba_m,
                er_p_max=er_p_max,
                er_p_scale=er_p_scale,
            )
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No simulation rows were generated. Check config settings.")

    raw_results.parent.mkdir(parents=True, exist_ok=True)
    with raw_results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = summarize(all_rows)
    summary_results.parent.mkdir(parents=True, exist_ok=True)
    with summary_results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {len(all_rows)} rows to {raw_results}")
    print(f"wrote {len(summary_rows)} summary rows to {summary_results}")

    # Always plot after simulation run.
    plot_main(path=str(raw_results), out_dir=str(plot_dir), config_path=config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run referral-auction simulations from config.ini and generate plots.")
    parser.add_argument("--config", default="config.ini", help="Path to config.ini")
    args = parser.parse_args()
    run_from_config(args.config)

from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional
import csv
import statistics
from pathlib import Path
from .instance import AuctionInstance
from .generators import erdos_renyi_instance, barabasi_instance, balanced_tree_instance, line_instance, star_instance
from .mechanisms import central_vickrey, local_vickrey, network_vcg, information_diffusion_mechanism, parametric_referral_auction

MECHS = [central_vickrey, local_vickrey, network_vcg, information_diffusion_mechanism, parametric_referral_auction]


def run_one(inst: AuctionInstance, seed: int = 0, diffusion_strategy: str = "full", **kwargs) -> List[Dict[str, object]]:
    rows = []
    for mech in MECHS:
        res = mech(inst, diffusion_strategy=diffusion_strategy, seed=seed, **kwargs)
        row = res.as_row(inst)
        row["instance"] = inst.name
        row["seed"] = seed
        rows.append(row)
    return rows


def sweep(
    out_csv: str = "results.csv",
    seeds: Iterable[int] = range(20),
    sizes: Iterable[int] = (20, 50, 100),
    topologies: Iterable[str] = ("line", "star", "tree", "er", "ba"),
    valuation_mode: str = "uniform",
    diffusion_strategy: str = "full",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for n in sizes:
        for seed in seeds:
            for topo in topologies:
                if topo == "line":
                    inst = line_instance(n, seed=seed, valuation_mode=valuation_mode)
                elif topo == "star":
                    inst = star_instance(n, seed=seed, valuation_mode=valuation_mode)
                elif topo == "tree":
                    inst = balanced_tree_instance(2, max(1, int((n + 1).bit_length() - 1)), seed=seed, valuation_mode=valuation_mode)
                elif topo == "er":
                    inst = erdos_renyi_instance(n, p=min(0.25, 3 / max(n, 1)), seed=seed, valuation_mode=valuation_mode)
                elif topo == "ba":
                    inst = barabasi_instance(n, m=2 if n > 3 else 1, seed=seed, valuation_mode=valuation_mode)
                else:
                    raise ValueError(topo)
                for row in run_one(inst, seed=seed, diffusion_strategy=diffusion_strategy):
                    row["topology"] = topo
                    row["n"] = n
                    row["valuation_mode"] = valuation_mode
                    rows.append(row)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows

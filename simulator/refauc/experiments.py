from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional
import csv
import statistics
from pathlib import Path
from .instance import AuctionInstance
from .generators import erdos_renyi_instance, barabasi_instance, balanced_tree_instance, line_instance, star_instance
from .mechanisms import (
    central_vickrey,
    local_vickrey,
    network_vcg,
    information_diffusion_mechanism,
    parametric_referral_auction,
    sybil_resistant_referral_auction,
)

MECHS = [
    central_vickrey,
    local_vickrey,
    network_vcg,
    information_diffusion_mechanism,
    parametric_referral_auction,
    sybil_resistant_referral_auction,
]


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
    summary_csv: Optional[str] = None,
    seeds: Iterable[int] = range(20),
    sizes: Iterable[int] = (20, 50, 100),
    topologies: Iterable[str] = ("line", "star", "tree", "er", "ba"),
    valuation_mode: str = "uniform",
    diffusion_strategy: str = "full",
    invite_prob: float = 1.0,
    tree_branching: int = 2,
    ba_m: int = 2,
    er_p_max: float = 0.25,
    er_p_scale: float = 3.0,
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
                    inst = balanced_tree_instance(tree_branching, max(1, int((n + 1).bit_length() - 1)), seed=seed, valuation_mode=valuation_mode)
                elif topo == "er":
                    inst = erdos_renyi_instance(
                        n,
                        p=min(er_p_max, er_p_scale / max(n, 1)),
                        seed=seed,
                        valuation_mode=valuation_mode,
                    )
                elif topo == "ba":
                    inst = barabasi_instance(n, m=ba_m if n > 3 else 1, seed=seed, valuation_mode=valuation_mode)
                else:
                    raise ValueError(topo)
                for row in run_one(
                    inst,
                    seed=seed,
                    diffusion_strategy=diffusion_strategy,
                    invite_prob=invite_prob,
                ):
                    row["topology"] = topo
                    row["n"] = n
                    row["valuation_mode"] = valuation_mode
                    row["diffusion_strategy"] = diffusion_strategy
                    row["invite_prob"] = invite_prob
                    rows.append(row)
    if rows:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if summary_csv:
            summary_rows = summarize(rows)
            Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            row.get("mechanism"),
            row.get("topology"),
            row.get("n"),
            row.get("valuation_mode"),
            row.get("diffusion_strategy"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, grp in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rev = [float(r["revenue"]) for r in grp]
        wel = [float(r["welfare"]) for r in grp]
        npart = [float(r["n_participants"]) for r in grp]
        deficits = sum(1 for r in rev if r < -1e-9)
        out.append(
            {
                "mechanism": key[0],
                "topology": key[1],
                "n": key[2],
                "valuation_mode": key[3],
                "diffusion_strategy": key[4],
                "runs": len(grp),
                "revenue_mean": statistics.fmean(rev),
                "revenue_std": statistics.pstdev(rev) if len(rev) > 1 else 0.0,
                "welfare_mean": statistics.fmean(wel),
                "welfare_std": statistics.pstdev(wel) if len(wel) > 1 else 0.0,
                "participants_mean": statistics.fmean(npart),
                "deficit_rate": deficits / len(grp),
                "budget_balance_rate": 1.0 - deficits / len(grp),
            }
        )
    return out

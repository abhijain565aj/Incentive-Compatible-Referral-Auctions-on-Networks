from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Iterable, Any
import math

from .generators import make_instance
from .mechanisms import run_all_mechanisms


def run_experiment(n_values: List[int], eta_values: List[float], sigma: float, runs: int,
                   distribution: str, penalty_lambda: float, lblev_lambda: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed = 12345
    for n in n_values:
        for eta in eta_values:
            for r in range(runs):
                inst = make_instance(n=n, seed=seed, sigma=sigma, distribution=distribution)
                seed += 1
                for res in run_all_mechanisms(inst.tree, inst.values, inst.means, inst.sensitivity,
                                              inst.prior_upper, eta=eta,
                                              penalty_lambda=penalty_lambda, lblev_lambda=lblev_lambda):
                    row: Dict[str, Any] = {
                        "n": n,
                        "eta": eta,
                        "run": r,
                        "mechanism": res.name,
                        "winner": res.winner if res.winner is not None else -1,
                        "branch": res.winning_branch if res.winning_branch is not None else -1,
                        "revenue": res.revenue,
                        "raw_welfare": res.raw_welfare,
                        "externality": res.externality,
                        "adjusted_welfare": res.adjusted_welfare,
                        "depth": res.depth,
                    }
                    for k, v in res.details.items():
                        row[f"detail_{k}"] = v
                    rows.append(row)
    return rows


def _mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else 0.0


def _se(xs: Iterable[float]) -> float:
    vals = list(xs)
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["n"], row["eta"], row["mechanism"])
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (n, eta, mechanism), g in sorted(groups.items()):
        out.append({
            "n": n,
            "eta": eta,
            "mechanism": mechanism,
            "revenue_mean": _mean(float(x["revenue"]) for x in g),
            "revenue_se": _se(float(x["revenue"]) for x in g),
            "adj_welfare_mean": _mean(float(x["adjusted_welfare"]) for x in g),
            "raw_welfare_mean": _mean(float(x["raw_welfare"]) for x in g),
            "externality_mean": _mean(float(x["externality"]) for x in g),
            "depth_mean": _mean(float(x["depth"]) for x in g),
        })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    # Put common keys first.
    common = ["n", "eta", "run", "mechanism", "winner", "branch", "revenue", "raw_welfare",
              "externality", "adjusted_welfare", "depth"]
    keys = [k for k in common if k in keys] + [k for k in keys if k not in common]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary: List[Dict[str, Any]], out_dir: Path, metric: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return
    ns = sorted({row["n"] for row in summary})
    for n in ns:
        sub = [row for row in summary if row["n"] == n]
        mechs = sorted({row["mechanism"] for row in sub})
        plt.figure(figsize=(8, 5))
        for mech in mechs:
            m = sorted([row for row in sub if row["mechanism"] == mech], key=lambda x: x["eta"])
            plt.plot([row["eta"] for row in m], [row[metric] for row in m], marker="o", label=mech)
        plt.xlabel("externality scale eta")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs externality scale (n={n})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_n{n}.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run referral auction simulations.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--n", nargs="+", type=int, default=[30])
    parser.add_argument("--eta", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--distribution", choices=["normal", "uniform"], default="normal")
    parser.add_argument("--penalty-lambda", type=float, default=1.0)
    parser.add_argument("--lblev-lambda", type=float, default=0.7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = run_experiment(args.n, args.eta, args.sigma, args.runs, args.distribution,
                          args.penalty_lambda, args.lblev_lambda)
    summ = summarize(rows)
    write_csv(args.out / "runs.csv", rows)
    write_csv(args.out / "summary.csv", summ)
    plot_metric(summ, args.out, "revenue_mean", "mean seller revenue")
    plot_metric(summ, args.out, "adj_welfare_mean", "mean externality-adjusted welfare")
    plot_metric(summ, args.out, "externality_mean", "mean realized externality")
    for row in summ:
        print(row)


if __name__ == "__main__":
    main()

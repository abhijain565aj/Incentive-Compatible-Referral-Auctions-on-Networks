from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any
import statistics

from .generators import make_instance, path_tree, Instance, add_sybil_attack
from .mechanisms import (
    all_single_item_mechanisms, idm, participation_cost_idm, network_vcg, sybil_tax_idm, taxed_network_vcg,
    multi_unit_vcg, quality_weighted_multi_item, multi_seller_partition,
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    preferred = ["scenario", "topology", "n", "run", "param", "mechanism", "revenue", "welfare", "net_welfare", "winner"]
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r.get("scenario"), r.get("topology"), r.get("n"), r.get("param"), r.get("mechanism"))
        groups.setdefault(key, []).append(r)
    out: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        scenario, topology, n, param, mechanism = key
        rev = [float(x["revenue"]) for x in vals]
        wel = [float(x["welfare"]) for x in vals]
        net = [float(x["net_welfare"]) for x in vals]
        out.append({
            "scenario": scenario, "topology": topology, "n": n, "param": param, "mechanism": mechanism,
            "revenue_mean": statistics.mean(rev),
            "revenue_se": statistics.stdev(rev) / (len(rev) ** 0.5) if len(rev) > 1 else 0.0,
            "welfare_mean": statistics.mean(wel),
            "net_welfare_mean": statistics.mean(net),
        })
    return out


def run_baseline_and_cost(n_values: List[int], topologies: List[str], runs: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed = 100
    for topology in topologies:
        for n in n_values:
            for r in range(runs):
                inst = make_instance(n=n, seed=seed, topology=topology, cost_mode="participation")
                seed += 1
                for res in all_single_item_mechanisms(inst.tree, inst.values, inst.costs, sybil_tax=1.0):
                    rows.append({
                        "scenario": "single_item_cost", "topology": topology, "n": n, "run": r, "param": "base",
                        "mechanism": res.name, "revenue": res.revenue, "welfare": res.welfare,
                        "net_welfare": res.net_welfare, "winner": res.winner if res.winner is not None else -1,
                    })
    return rows


def run_sybil(n: int, max_sybils: int, runs: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in range(runs):
        inst = make_instance(n=n, seed=2000 + r, topology="path", cost_mode="zero", sigma=2.0)
        # force last node to be the high-value winner and earlier nodes to zero-ish.
        for u in inst.tree.buyers:
            inst.values[u] = 0.0
        inst.values[n] = 100.0
        for k in range(max_sybils + 1):
            attacked = add_sybil_attack(inst, target_child=n, sybils=k, sybil_value=0.0)
            for res in [network_vcg(attacked.tree, attacked.values, attacked.costs), taxed_network_vcg(attacked.tree, attacked.values, attacked.costs, tax=2.0), idm(attacked.tree, attacked.values, attacked.costs)]:
                rows.append({
                    "scenario": "sybil", "topology": "path", "n": n, "run": r, "param": str(k),
                    "mechanism": res.name, "revenue": res.revenue, "welfare": res.welfare,
                    "net_welfare": res.net_welfare, "winner": res.winner if res.winner is not None else -1,
                })
    return rows


def run_multi_item(n_values: List[int], runs: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    qualities = [1.0, 0.8, 0.55]
    for n in n_values:
        for r in range(runs):
            inst = make_instance(n=n, seed=3000 + 17 * n + r, topology="random", cost_mode="zero")
            mechanisms = [multi_unit_vcg(inst.tree, inst.values, k=3), quality_weighted_multi_item(inst.tree, inst.values, qualities)]
            for res in mechanisms:
                rows.append({
                    "scenario": "multi_item", "topology": "random", "n": n, "run": r, "param": "K=3",
                    "mechanism": res.name, "revenue": res.revenue, "welfare": res.welfare,
                    "net_welfare": res.net_welfare, "winner": res.winner if res.winner is not None else -1,
                })
    return rows


def run_multi_seller(n_values: List[int], runs: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in n_values:
        for r in range(runs):
            inst = make_instance(n=n, seed=4000 + 31 * n + r, topology="binary", cost_mode="participation")
            seller_roots = inst.tree.first_level()[:2]
            res_list = multi_seller_partition(inst.tree, inst.values, seller_roots, inst.costs)
            rows.append({
                "scenario": "multi_seller", "topology": "binary", "n": n, "run": r, "param": "2 sellers",
                "mechanism": "Partition-PC-IDM", "revenue": sum(x.revenue for x in res_list),
                "welfare": sum(x.welfare for x in res_list), "net_welfare": sum(x.net_welfare for x in res_list),
                "winner": len([x for x in res_list if x.winner is not None]),
            })
    return rows


def _svg_line_plot(path: Path, series: Dict[str, List[tuple]], xlabel: str, ylabel: str, title: str) -> None:
    width, height = 900, 560
    ml, mr, mt, mb = 90, 190, 60, 80
    xs = [x for pts in series.values() for x, _ in pts]
    ys = [y for pts in series.values() for _, y in pts]
    if not xs or not ys:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(0.0, min(ys)), max(ys)
    if abs(xmax - xmin) < 1e-9:
        xmax += 1.0
    if abs(ymax - ymin) < 1e-9:
        ymax += 1.0
    def sx(x: float) -> float:
        return ml + (x - xmin) * (width - ml - mr) / (xmax - xmin)
    def sy(y: float) -> float:
        return height - mb - (y - ymin) * (height - mt - mb) / (ymax - ymin)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{title}</text>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#222" stroke-width="2"/>')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#222" stroke-width="2"/>')
    for t in range(6):
        xv = xmin + t*(xmax-xmin)/5
        yv = ymin + t*(ymax-ymin)/5
        parts.append(f'<line x1="{sx(xv):.1f}" y1="{height-mb}" x2="{sx(xv):.1f}" y2="{height-mb+6}" stroke="#222"/>')
        parts.append(f'<text x="{sx(xv):.1f}" y="{height-mb+24}" text-anchor="middle" font-family="Arial" font-size="13">{xv:.0f}</text>')
        parts.append(f'<line x1="{ml-6}" y1="{sy(yv):.1f}" x2="{ml}" y2="{sy(yv):.1f}" stroke="#222"/>')
        parts.append(f'<text x="{ml-10}" y="{sy(yv)+4:.1f}" text-anchor="end" font-family="Arial" font-size="13">{yv:.1f}</text>')
        parts.append(f'<line x1="{ml}" y1="{sy(yv):.1f}" x2="{width-mr}" y2="{sy(yv):.1f}" stroke="#eee"/>')
    parts.append(f'<text x="{(ml+width-mr)/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="16">{xlabel}</text>')
    parts.append(f'<text transform="translate(24,{(mt+height-mb)/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="16">{ylabel}</text>')
    for idx, (name, pts) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        pts = sorted(pts)
        d = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        ly = mt + 25 + idx*24
        parts.append(f'<line x1="{width-mr+25}" y1="{ly}" x2="{width-mr+55}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-mr+62}" y="{ly+5}" font-family="Arial" font-size="13">{name}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))
    # SVG is kept as the lightweight plot artifact; the LaTeX report uses pgfplots.


def plot_summary(summary: List[Dict[str, Any]], out_dir: Path) -> None:
    for metric in ["revenue_mean", "net_welfare_mean"]:
        sub = [r for r in summary if r["scenario"] == "single_item_cost" and r["topology"] == "random" and r["param"] == "base"]
        series: Dict[str, List[tuple]] = {}
        for mech in sorted(set(r["mechanism"] for r in sub)):
            series[mech] = [(float(r["n"]), float(r[metric])) for r in sub if r["mechanism"] == mech]
        _svg_line_plot(out_dir / f"single_item_cost_{metric}.svg", series, "number of buyers", metric.replace("_", " "), f"Single item with participation costs: {metric}")
    sub = [r for r in summary if r["scenario"] == "sybil"]
    series = {}
    for mech in sorted(set(r["mechanism"] for r in sub)):
        series[mech] = [(float(r["param"]), float(r["revenue_mean"])) for r in sub if r["mechanism"] == mech]
    _svg_line_plot(out_dir / "sybil_revenue.svg", series, "inserted sybil identities", "seller revenue mean", "Sybil stress test on a path")
    sub = [r for r in summary if r["scenario"] == "multi_item"]
    for metric in ["revenue_mean", "welfare_mean"]:
        series = {}
        for mech in sorted(set(r["mechanism"] for r in sub)):
            series[mech] = [(float(r["n"]), float(r[metric])) for r in sub if r["mechanism"] == mech]
        _svg_line_plot(out_dir / f"multi_item_{metric}.svg", series, "number of buyers", metric.replace("_", " "), f"Multiple/different items: {metric}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("../results"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--n", nargs="+", type=int, default=[20, 40, 60])
    parser.add_argument("--max-sybils", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    rows += run_baseline_and_cost(args.n, ["random", "path", "binary"], args.runs)
    rows += run_sybil(n=12, max_sybils=args.max_sybils, runs=max(10, args.runs // 4))
    rows += run_multi_item(args.n, args.runs)
    rows += run_multi_seller(args.n, args.runs)
    summary = summarize(rows)
    write_csv(args.out / "runs.csv", rows)
    write_csv(args.out / "summary.csv", summary)
    plot_summary(summary, args.out)
    print(f"Wrote {len(rows)} run rows and {len(summary)} summary rows to {args.out}")


if __name__ == "__main__":
    main()

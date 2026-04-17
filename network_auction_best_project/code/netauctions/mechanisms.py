from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import math

from .network import RootedTree

EPS = 1e-9


@dataclass
class Result:
    name: str
    winner: Optional[int]
    revenue: float
    welfare: float
    net_welfare: float
    seller: int = 0
    payments: Dict[int, float] | None = None
    notes: Dict[str, float] | None = None

    def utility(self, values: Dict[int, float], costs: Dict[int, float], i: int) -> float:
        p = 0.0 if self.payments is None else self.payments.get(i, 0.0)
        alloc = 1.0 if self.winner == i else 0.0
        return alloc * values.get(i, 0.0) - p - costs.get(i, 0.0)


def max_value(nodes: List[int], score: Dict[int, float]) -> float:
    if not nodes:
        return 0.0
    return max(score.get(u, 0.0) for u in nodes)


def argmax_value(nodes: List[int], score: Dict[int, float]) -> Optional[int]:
    if not nodes:
        return None
    return max(nodes, key=lambda u: (score.get(u, 0.0), -u))


def local_vickrey(tree: RootedTree, values: Dict[int, float], costs: Dict[int, float] | None = None) -> Result:
    costs = costs or {u: 0.0 for u in tree.buyers}
    bidders = tree.first_level()
    if not bidders:
        return Result("Local-Vickrey", None, 0.0, 0.0, 0.0, payments={})
    ordered = sorted(bidders, key=lambda u: values[u], reverse=True)
    w = ordered[0]
    price = values[ordered[1]] if len(ordered) > 1 else 0.0
    return Result("Local-Vickrey", w, price, values[w], values[w] - costs.get(w, 0.0), payments={w: price})


def network_vcg(tree: RootedTree, values: Dict[int, float], costs: Dict[int, float] | None = None) -> Result:
    costs = costs or {u: 0.0 for u in tree.buyers}
    w = argmax_value(tree.buyers, values)
    if w is None:
        return Result("Network-VCG", None, 0.0, 0.0, 0.0, payments={})
    W = values[w]
    payments: Dict[int, float] = {}
    for i in tree.buyers:
        without_di = max_value(tree.outside_subtree(i), values)
        payments[i] = without_di - (W - (values[i] if i == w else 0.0))
    revenue = sum(payments.values())
    return Result("Network-VCG", w, revenue, values[w], values[w] - costs.get(w, 0.0), payments=payments)


def idm_with_score(tree: RootedTree, values: Dict[int, float], score: Dict[int, float], name: str,
                   costs: Dict[int, float] | None = None) -> Result:
    """IDM on any nonnegative single-parameter score.

    If score_i=value_i this is the baseline IDM. If score_i=max(value_i-cost_i,0), it is the
    participation-cost-adjusted IDM studied in the report.
    """
    costs = costs or {u: 0.0 for u in tree.buyers}
    # no sale if every transformed value is zero or negative
    m = argmax_value(tree.buyers, score)
    if m is None or score[m] <= EPS:
        return Result(name, None, 0.0, 0.0, 0.0, payments={})
    C = tree.path(m, include_root=False)
    # Winner is first critical node i on C such that i is the max outside next subtree.
    w = m
    for idx, i in enumerate(C[:-1]):
        nxt = C[idx + 1]
        outside_next = tree.outside_subtree(nxt)
        if abs(score[i] - max_value(outside_next, score)) <= EPS or score[i] >= max_value(outside_next, score) - EPS:
            w = i
            break
    Cw = tree.path(w, include_root=False)
    payments: Dict[int, float] = {i: 0.0 for i in tree.buyers}
    for idx, i in enumerate(Cw[:-1]):
        nxt = Cw[idx + 1]
        payments[i] = max_value(tree.outside_subtree(i), score) - max_value(tree.outside_subtree(nxt), score)
    payments[w] = max_value(tree.outside_subtree(w), score)
    revenue = sum(payments.values())
    welfare = values[w]
    net_welfare = values[w] - costs.get(w, 0.0)
    return Result(name, w, revenue, welfare, net_welfare, payments=payments,
                  notes={"score_w": score[w], "path_len": float(len(Cw))})


def idm(tree: RootedTree, values: Dict[int, float], costs: Dict[int, float] | None = None) -> Result:
    return idm_with_score(tree, values, values, "IDM", costs=costs)


def participation_cost_idm(tree: RootedTree, values: Dict[int, float], costs: Dict[int, float]) -> Result:
    score = {u: max(0.0, values[u] - costs.get(u, 0.0)) for u in tree.buyers}
    return idm_with_score(tree, values, score, "PC-IDM", costs=costs)


def sybil_tax_idm(tree: RootedTree, values: Dict[int, float], costs: Dict[int, float] | None = None,
                  tax: float = 1.0) -> Result:
    base = idm(tree, values, costs=costs)
    payments = dict(base.payments or {})
    # Registration tax/submission fee for every participating identity. This is a simple
    # simulation proxy for sybil-tax mechanisms, not claimed to be the full STM/SCM mechanism.
    for i in tree.buyers:
        payments[i] = payments.get(i, 0.0) + tax
    return Result("Tax-IDM", base.winner, sum(payments.values()), base.welfare, base.net_welfare,
                  payments=payments, notes={"tax": tax})


def multi_unit_vcg(tree: RootedTree, values: Dict[int, float], k: int) -> Result:
    """VCG for k identical indivisible units among reachable bidders, ignoring diffusion rewards."""
    ordered = sorted(tree.buyers, key=lambda u: values[u], reverse=True)
    winners = ordered[:k]
    price = values[ordered[k]] if len(ordered) > k else 0.0
    payments = {u: (price if u in winners else 0.0) for u in tree.buyers}
    welfare = sum(values[u] for u in winners)
    return Result(f"MultiUnit-VCG-{k}", winners[0] if winners else None, sum(payments.values()), welfare, welfare,
                  payments=payments, notes={"units": float(k), "winners": float(len(winners))})


def divisible_discretized_vcg(tree: RootedTree, values: Dict[int, float], units: int) -> Result:
    """Approximate a unit mass divisible good by `units` equal micro-units.

    With linear values, efficient allocation gives all micro-units to the highest value bidder.
    This looks trivial, but it is useful as a convergence baseline and to expose the same
    VCG-vs-diffusion deficit issue under finer discretization.
    """
    return multi_unit_vcg(tree, values, k=max(1, units))


def quality_weighted_multi_item(tree: RootedTree, values: Dict[int, float], qualities: List[float]) -> Result:
    """VCG-like allocation for public-quality heterogeneous items and scalar private values.

    Buyer i's value for item q is q*v_i and each buyer gets at most one item. The efficient
    allocation sorts qualities and values in descending order. Payments are computed by externality.
    This is a classical VCG benchmark over the reached set, not a full diffusion mechanism.
    """
    qs = sorted(qualities, reverse=True)
    bidders = sorted(tree.buyers, key=lambda u: values[u], reverse=True)
    winners = bidders[:len(qs)]
    welfare = sum(qs[t] * values[winners[t]] for t in range(len(winners)))
    payments: Dict[int, float] = {u: 0.0 for u in tree.buyers}
    for idx, i in enumerate(winners):
        others = [u for u in tree.buyers if u != i]
        others_sorted = sorted(others, key=lambda u: values[u], reverse=True)
        welfare_without_i = sum(qs[t] * values[others_sorted[t]] for t in range(min(len(qs), len(others_sorted))))
        welfare_others_in_chosen = welfare - qs[idx] * values[i]
        payments[i] = welfare_without_i - welfare_others_in_chosen
    return Result("Heterogeneous-VCG", winners[0] if winners else None, sum(payments.values()), welfare, welfare,
                  payments=payments, notes={"items": float(len(qs))})


def multi_seller_partition(tree: RootedTree, values: Dict[int, float], seller_roots: List[int], costs: Dict[int, float]) -> List[Result]:
    """Heuristic multi-seller scenario: partition by closest seller-root and run PC-IDM in each region.

    This is explicitly an experimental baseline, not a global DSIC mechanism.
    """
    results: List[Result] = []
    for sr in seller_roots:
        nodes = tree.subtree(sr)
        if not nodes:
            continue
        parent = {0: None}
        # Re-root seller sr to artificial seller 0; preserve descendants.
        mapping = {sr: 0}
        next_id = 1
        for u in nodes:
            if u == sr:
                continue
            mapping[u] = next_id
            next_id += 1
        for u in nodes:
            if u == sr:
                continue
            p = tree.parent[u]
            if p in mapping:
                parent[mapping[u]] = mapping[p]  # type: ignore[index]
        local_tree = RootedTree(parent)
        local_values = {mapping[u]: values[u] for u in nodes if u != sr}
        local_costs = {mapping[u]: costs.get(u, 0.0) for u in nodes if u != sr}
        if local_tree.buyers:
            res = participation_cost_idm(local_tree, local_values, local_costs)
            res.name = f"SellerRegion-PC-IDM"
            res.seller = sr
            results.append(res)
    return results


def all_single_item_mechanisms(tree: RootedTree, values: Dict[int, float], costs: Dict[int, float], sybil_tax: float = 1.0) -> List[Result]:
    return [
        local_vickrey(tree, values, costs),
        network_vcg(tree, values, costs),
        idm(tree, values, costs),
        participation_cost_idm(tree, values, costs),
        sybil_tax_idm(tree, values, costs, tax=sybil_tax),
    ]


def taxed_network_vcg(tree: RootedTree, values: Dict[int, float], costs: Dict[int, float] | None = None,
                      tax: float = 1.0) -> Result:
    base = network_vcg(tree, values, costs=costs)
    payments = dict(base.payments or {})
    for i in tree.buyers:
        payments[i] = payments.get(i, 0.0) + tax
    return Result("Tax-Network-VCG", base.winner, sum(payments.values()), base.welfare, base.net_welfare,
                  payments=payments, notes={"tax": tax})

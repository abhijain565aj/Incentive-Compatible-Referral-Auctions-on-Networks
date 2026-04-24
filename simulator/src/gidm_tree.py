from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import networkx as nx

from .graph_utils import (
    ancestors_exclusive,
    build_digraph,
    immediate_dominators_tree,
    induced_tree,
    minimal_subtree_nodes,
    path_in_tree,
    reachable_subgraph,
    subtree_nodes,
    top_k_nodes,
    tree_children,
)

Node = str


@dataclass
class GIDMOutcome:
    winners: List[Node]
    payments: Dict[Node, float]
    get_from: Dict[Node, Node]
    n_opt: List[Node]
    active_holders: List[Node]
    parent: Dict[Node, Node | None]
    children: Dict[Node, List[Node]]
    weights: Dict[Node, int]
    optimal_tree_nodes: Set[Node]


def _sorted_by_bid(nodes: Iterable[Node], bids: Mapping[Node, float]) -> List[Node]:
    return sorted(nodes, key=lambda x: (-bids.get(x, float('-inf')), str(x)))


def _compute_weights(nodes: Iterable[Node], active: Set[Node], children: Mapping[Node, Sequence[Node]], root: Node) -> Dict[Node, int]:
    weights: Dict[Node, int] = {}

    def dfs(u: Node) -> int:
        total = 1 if u in active else 0
        for c in children.get(u, []):
            total += dfs(c)
        weights[u] = total
        return total

    dfs(root)
    for u in nodes:
        weights.setdefault(u, 0)
    return weights


def _top_k_child_subtrees(i: Node, children: Mapping[Node, Sequence[Node]], bids: Mapping[Node, float], K: int) -> Set[Node]:
    kids = list(children.get(i, []))
    if len(kids) <= K:
        top = kids
    else:
        top = sorted(kids, key=lambda x: (-bids.get(x, float('-inf')), str(x)))[:K]
    removed: Set[Node] = set()
    for c in top:
        removed |= subtree_nodes(c, children)
    return removed


def _allocation_under_removed(
    nodes: Set[Node],
    bids: Mapping[Node, float],
    K: int,
    removed: Set[Node],
    fixed: Set[Node],
    root: Node,
) -> List[Node]:
    avail = [u for u in nodes if u != root and u not in removed]
    fixed2 = [u for u in fixed if u in avail]
    rem = K - len(fixed2)
    others = [u for u in avail if u not in fixed2]
    chosen = _sorted_by_bid(others, bids)[: max(0, rem)]
    out = list(dict.fromkeys(fixed2 + chosen))
    return out[:K]


def _winners_on_path(winners_so_far: Set[Node], i: Node, parent: Mapping[Node, Node | None], root: Node) -> Set[Node]:
    return set(ancestors_exclusive(i, parent, root)) & winners_so_far


def _nodes_on_any_winner_path(winners: Iterable[Node], parent: Mapping[Node, Node | None], root: Node) -> Set[Node]:
    out: Set[Node] = set()
    for w in winners:
        out |= set(ancestors_exclusive(w, parent, root))
    return out


def _feasible_allocation_for_node(
    i: Node,
    root: Node,
    nodes: Set[Node],
    bids: Mapping[Node, float],
    K: int,
    parent: Mapping[Node, Node | None],
    children: Mapping[Node, Sequence[Node]],
    n_opt: Set[Node],
    winners_so_far: Set[Node],
    get_from: Mapping[Node, Node],
) -> List[Node]:
    removed = _top_k_child_subtrees(i, children, bids, K)
    received = _winners_on_path(winners_so_far, i, parent, root)
    out_nodes = {get_from[w] for w in received if get_from[w] not in received}
    fixed = received | ((n_opt - removed) - out_nodes)
    return _allocation_under_removed(nodes, bids, K, removed, fixed, root)


def _sw_minus_set(
    removed: Set[Node],
    i: Node,
    root: Node,
    nodes: Set[Node],
    bids: Mapping[Node, float],
    K: int,
    parent: Mapping[Node, Node | None],
    n_opt: Set[Node],
    winners: Set[Node],
    get_from: Mapping[Node, Node],
) -> float:
    received = set(ancestors_exclusive(i, parent, root)) & winners
    out_nodes = {get_from[w] for w in received if get_from[w] not in received}
    fixed = received | ((n_opt - removed) - out_nodes)
    alloc = _allocation_under_removed(nodes, bids, K, removed, fixed, root)
    return sum(bids[u] for u in alloc)


def gidm_on_tree(root: Node, reports: Mapping[Node, Sequence[Node]], bids: Mapping[Node, float], K: int) -> GIDMOutcome:
    g = reachable_subgraph(build_digraph(root, reports), root)
    nodes = set(g.nodes())
    all_buyers = [u for u in nodes if u != root]

    # Use immediate dominator tree on the reachable graph / tree input.
    parent_full = immediate_dominators_tree(g, root)
    n_opt = top_k_nodes(all_buyers, bids, K)
    keep = minimal_subtree_nodes(root, n_opt, parent_full)
    parent = induced_tree(parent_full, keep, root)
    children = tree_children(parent)

    active: Set[Node] = set(n_opt)
    weights = _compute_weights(keep, active, children, root)

    Q: List[Node] = []
    for c in sorted(children.get(root, []), reverse=True):
        if weights[c] > 0:
            Q.append(c)

    winners: List[Node] = []
    get_from: Dict[Node, Node] = {}

    while Q:
        i = Q.pop()
        alloc_i = _feasible_allocation_for_node(i, root, keep, bids, K, parent, children, set(n_opt), set(winners), get_from)
        if i in alloc_i:
            winners.append(i)
            if i in active:
                get_from[i] = i
            else:
                k_i = max(1, weights.get(i, 0))
                cand = [u for u in active if u in subtree_nodes(i, children)]
                if not cand:
                    get_from[i] = i
                    active.add(i)
                else:
                    cand_sorted = _sorted_by_bid(cand, bids)
                    idx = min(k_i - 1, len(cand_sorted) - 1)
                    out = cand_sorted[idx]
                    active.remove(out)
                    active.add(i)
                    get_from[i] = out
            weights = _compute_weights(keep, active, children, root)
        for c in sorted(children.get(i, []), reverse=True):
            if weights.get(c, 0) > 0:
                Q.append(c)

    path_nodes = _nodes_on_any_winner_path(winners, parent, root)
    payments: Dict[Node, float] = {u: 0.0 for u in keep if u != root}
    for i in [u for u in keep if u != root]:
        D_i = subtree_nodes(i, children)
        C_iK = _top_k_child_subtrees(i, children, bids, K)
        sw_D = _sw_minus_set(D_i, i, root, keep, bids, K, parent, set(n_opt), set(winners), get_from)
        sw_C = _sw_minus_set(C_iK, i, root, keep, bids, K, parent, set(n_opt), set(winners), get_from)
        if i in winners:
            payments[i] = sw_D - (sw_C - bids[i])
        elif i in path_nodes:
            payments[i] = sw_D - sw_C
        else:
            payments[i] = 0.0

    return GIDMOutcome(
        winners=sorted(winners, key=lambda x: (-bids[x], str(x))),
        payments=payments,
        get_from=get_from,
        n_opt=n_opt,
        active_holders=sorted(active, key=lambda x: (-bids[x], str(x))),
        parent=parent,
        children=children,
        weights=weights,
        optimal_tree_nodes=keep,
    )


def gidm_from_graph(root: Node, reports: Mapping[Node, Sequence[Node]], bids: Mapping[Node, float], K: int) -> GIDMOutcome:
    return gidm_on_tree(root, reports, bids, K)


def seller_revenue(payments: Mapping[Node, float]) -> float:
    return float(sum(payments.values()))

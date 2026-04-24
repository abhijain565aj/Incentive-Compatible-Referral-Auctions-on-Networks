from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence
import random

from .gidm_tree import GIDMOutcome, gidm_on_tree
from .graph_utils import ClusterInfo, dominator_clusters, tree_reports_from_nx

Node = str


@dataclass
class SCGIDMOutcome:
    macro: GIDMOutcome
    cluster_info: ClusterInfo
    winners: List[Node]
    payments: Dict[Node, float]


def sc_gidm(
    seller: Node,
    reports: Mapping[Node, Sequence[Node]],
    bids: Mapping[Node, float],
    K: int,
    rng: random.Random | None = None,
) -> SCGIDMOutcome:
    rng = rng or random.Random(0)
    info = dominator_clusters(seller, reports, bids, rng)
    tree_reports = tree_reports_from_nx(info.cluster_tree)
    macro = gidm_on_tree(seller, tree_reports, {seller: 0.0, **info.macro_bids}, K)

    winners: List[Node] = []
    payments: Dict[Node, float] = {u: 0.0 for u in bids}
    winning_clusters = set(macro.winners)
    path_clusters = set()
    for w in winning_clusters:
        cur = macro.parent[w]
        while cur is not None and cur != seller:
            path_clusters.add(cur)
            cur = macro.parent[cur]

    for c in winning_clusters:
        eta = info.eta[c]
        winners.append(eta)
        payments[eta] += macro.payments.get(c, 0.0)

    for c in path_clusters - winning_clusters:
        rho = info.cluster_roots[c]
        payments[rho] += macro.payments.get(c, 0.0)

    return SCGIDMOutcome(macro=macro, cluster_info=info, winners=winners, payments=payments)

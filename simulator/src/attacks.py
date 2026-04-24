from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

Node = str


def truthful_owner(reports: Mapping[Node, Sequence[Node]]) -> Dict[Node, Node]:
    return {u: u for u in reports}


def apply_split_child_attack(
    seller: Node,
    reports: Mapping[Node, Sequence[Node]],
    bids: Mapping[Node, float],
    owner: Mapping[Node, Node],
    attacker: Node,
    q: int = 1,
    sybil_bid: float | None = None,
) -> Tuple[Dict[Node, List[Node]], Dict[Node, float], Dict[Node, Node]]:
    """Insert q child sybils as an outward chain from attacker and move all outgoing edges to the last sybil."""
    new_reports = {u: list(vs) for u, vs in reports.items()}
    new_bids = dict(bids)
    new_owner = dict(owner)

    outs = list(new_reports.get(attacker, []))
    if q <= 0:
        return new_reports, new_bids, new_owner

    prev = attacker
    chain = []
    for t in range(1, q + 1):
        s = f"{attacker}_syb{t}"
        chain.append(s)
        new_reports[s] = []
        new_owner[s] = owner[attacker]
        new_bids[s] = bids[attacker] if sybil_bid is None else sybil_bid
        new_reports[prev].append(s)
        prev = s

    new_reports[attacker] = [x for x in new_reports[attacker] if x not in outs]
    new_reports[chain[-1]].extend(outs)
    return new_reports, new_bids, new_owner


def apply_split_star_attack(
    seller: Node,
    reports: Mapping[Node, Sequence[Node]],
    bids: Mapping[Node, float],
    owner: Mapping[Node, Node],
    attacker: Node,
    q: int = 1,
    sybil_bid: float | None = None,
) -> Tuple[Dict[Node, List[Node]], Dict[Node, float], Dict[Node, Node]]:
    """Create q sybil children, each attached to one slice of outgoing neighbors."""
    new_reports = {u: list(vs) for u, vs in reports.items()}
    new_bids = dict(bids)
    new_owner = dict(owner)
    outs = list(new_reports.get(attacker, []))
    if q <= 0 or not outs:
        return new_reports, new_bids, new_owner

    new_reports[attacker] = []
    buckets = [[] for _ in range(q)]
    for idx, v in enumerate(outs):
        buckets[idx % q].append(v)

    for t in range(1, q + 1):
        s = f"{attacker}_syb{t}"
        new_owner[s] = owner[attacker]
        new_bids[s] = bids[attacker] if sybil_bid is None else sybil_bid
        new_reports[attacker].append(s)
        new_reports[s] = buckets[t - 1]
    return new_reports, new_bids, new_owner

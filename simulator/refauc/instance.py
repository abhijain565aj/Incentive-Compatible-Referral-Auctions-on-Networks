from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
import random
import math
import networkx as nx

Node = int


@dataclass
class AuctionInstance:
    """A single-item auction instance on a social network.

    The seller is a distinguished node, default 0. All other nodes are buyers.
    `graph` is the true social graph. Diffusion creates a directed invitation
    graph rooted at the seller.
    """

    graph: nx.Graph
    valuations: Dict[Node, float]
    seller: Node = 0
    reserve: float = 0.0
    name: str = "instance"

    def buyers(self) -> List[Node]:
        return sorted([v for v in self.graph.nodes if v != self.seller])

    def seller_neighbors(self) -> List[Node]:
        return sorted([v for v in self.graph.neighbors(self.seller) if v != self.seller])

    def value(self, node: Node) -> float:
        return float(self.valuations.get(node, 0.0))


@dataclass
class AuctionResult:
    mechanism: str
    winner: Optional[Node]
    allocation_value: float
    payments: Dict[Node, float]
    participants: Set[Node]
    invited_edges: Set[Tuple[Node, Node]] = field(default_factory=set)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def revenue(self) -> float:
        return float(sum(self.payments.values()))

    @property
    def social_welfare(self) -> float:
        return float(self.allocation_value)

    def utility(self, inst: AuctionInstance, node: Node) -> float:
        return (inst.value(node) if self.winner == node else 0.0) - self.payments.get(node, 0.0)

    def as_row(self, inst: AuctionInstance) -> Dict[str, object]:
        utilities = [self.utility(inst, i) for i in sorted(self.participants)]
        sum_utilities = float(sum(utilities))
        product_utilities = float(math.prod(utilities)) if utilities else 0.0
        if utilities and all(u > 0 for u in utilities):
            log_product_utilities = float(sum(math.log(u) for u in utilities))
        elif utilities:
            log_product_utilities = float("-inf")
        else:
            log_product_utilities = 0.0
        return {
            "mechanism": self.mechanism,
            "winner": self.winner,
            "winner_value": self.allocation_value,
            "revenue": self.revenue,
            "welfare": self.social_welfare,
            "welfare_sum_utilities": sum_utilities,
            "welfare_product_utilities": product_utilities,
            "welfare_log_product_utilities": log_product_utilities,
            "n_participants": len(self.participants),
            "diffusion_depth": self.meta.get("diffusion_depth", None),
            "negative_payments": sum(1 for x in self.payments.values() if x < -1e-9),
        }


def simulate_diffusion(
    inst: AuctionInstance,
    strategy: str = "full",
    invite_prob: float = 1.0,
    seed: Optional[int] = None,
    custom_invites: Optional[Dict[Node, Iterable[Node]]] = None,
) -> Tuple[Set[Node], Set[Tuple[Node, Node]], Dict[Node, int]]:
    """Create a feasible directed invitation graph.

    Strategies:
    - full: every reached node invites all unreached neighbours.
    - none: only seller neighbours are reached.
    - probabilistic: reached buyers invite each neighbour independently.
    - custom: use custom_invites[node] as invited neighbour set.

    Returns (participants, directed_invitation_edges, depth_from_seller).
    """
    rng = random.Random(seed)
    seller = inst.seller
    reached: Set[Node] = set([seller])
    participants: Set[Node] = set()
    edges: Set[Tuple[Node, Node]] = set()
    depth: Dict[Node, int] = {seller: 0}

    # seller always announces to all direct neighbours
    q: List[Node] = []
    for nb in sorted(inst.graph.neighbors(seller)):
        if nb == seller:
            continue
        reached.add(nb)
        participants.add(nb)
        edges.add((seller, nb))
        depth[nb] = 1
        q.append(nb)

    if strategy == "none":
        return participants, edges, depth

    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        if strategy == "full":
            invitees = list(inst.graph.neighbors(u))
        elif strategy == "probabilistic":
            invitees = [v for v in inst.graph.neighbors(u) if rng.random() <= invite_prob]
        elif strategy == "custom":
            invitees = list(custom_invites.get(u, [])) if custom_invites else []
        else:
            raise ValueError(f"unknown diffusion strategy: {strategy}")

        for v in sorted(invitees):
            if v == seller or v == u:
                continue
            edges.add((u, v))
            if v not in reached:
                reached.add(v)
                participants.add(v)
                depth[v] = depth[u] + 1
                q.append(v)
    return participants, edges, depth


def invitation_digraph(inst: AuctionInstance, invited_edges: Set[Tuple[Node, Node]]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(inst.graph.nodes)
    G.add_edges_from(invited_edges)
    return G


def reachable_buyers(inst: AuctionInstance, DG: nx.DiGraph, banned: Optional[Set[Node]] = None) -> Set[Node]:
    banned = set() if banned is None else set(banned)
    if inst.seller in banned:
        return set()
    H = DG.copy()
    H.remove_nodes_from(banned)
    if inst.seller not in H:
        return set()
    return set(nx.descendants(H, inst.seller)) - {inst.seller}


def critical_descendant_sets(inst: AuctionInstance, DG: nx.DiGraph, participants: Set[Node]) -> Dict[Node, Set[Node]]:
    """Compute d_i = {i} plus buyers for whom i is a diffusion critical node."""
    ans: Dict[Node, Set[Node]] = {}
    base = set(participants)
    for i in participants:
        without_i = reachable_buyers(inst, DG, banned={i})
        lost = base - without_i
        ans[i] = set(lost) | {i}
    return ans


def diffusion_critical_sequence(
    inst: AuctionInstance,
    DG: nx.DiGraph,
    participants: Set[Node],
    target: Node,
    dsets: Optional[Dict[Node, Set[Node]]] = None,
) -> List[Node]:
    """Return critical nodes of target ordered from seller side to target.

    A node i is critical for target if target is not reachable after removing i.
    The target itself is always the final element.
    """
    if target not in participants:
        return []
    dsets = dsets or critical_descendant_sets(inst, DG, participants)
    crit = [i for i in participants if target in dsets.get(i, set()) and i != target]
    # In directed diffusion graphs, critical nodes form a dominator chain.
    # Sort by shortest distance in the invitation graph; tie by size of d_i and id.
    dist = nx.single_source_shortest_path_length(DG, inst.seller)
    crit.sort(key=lambda x: (dist.get(x, 10**9), -len(dsets.get(x, set())), x))
    return crit + [target]


def max_bidder(inst: AuctionInstance, candidates: Iterable[Node]) -> Optional[Node]:
    cand = list(candidates)
    if not cand:
        return None
    return max(cand, key=lambda x: (inst.value(x), -x))


def max_value(inst: AuctionInstance, candidates: Iterable[Node]) -> float:
    b = max_bidder(inst, candidates)
    return inst.reserve if b is None else max(inst.reserve, inst.value(b))


def second_value(inst: AuctionInstance, candidates: Iterable[Node]) -> float:
    vals = sorted([inst.value(i) for i in candidates], reverse=True)
    if len(vals) < 2:
        return inst.reserve
    return max(inst.reserve, vals[1])

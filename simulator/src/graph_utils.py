from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import random

import networkx as nx


Node = str
Adj = Mapping[Node, Sequence[Node]]


def build_digraph(seller: Node, reports: Adj) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(seller)
    for u, nbrs in reports.items():
        g.add_node(u)
        for v in nbrs:
            g.add_edge(u, v)
    return g


def reachable_subgraph(g: nx.DiGraph, root: Node) -> nx.DiGraph:
    reach = {root} | nx.descendants(g, root)
    return g.subgraph(reach).copy()


def immediate_dominators_tree(g: nx.DiGraph, root: Node) -> Dict[Node, Node | None]:
    idom = nx.immediate_dominators(g, root)
    parent: Dict[Node, Node | None] = {root: None}
    for v, p in idom.items():
        if v == root:
            continue
        parent[v] = p
    return parent


def tree_children(parent: Mapping[Node, Node | None]) -> Dict[Node, List[Node]]:
    children: Dict[Node, List[Node]] = {u: [] for u in parent}
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)
    for u in children:
        children[u].sort()
    return children


def descendants_in_tree(root: Node, children: Mapping[Node, Sequence[Node]]) -> Dict[Node, Set[Node]]:
    memo: Dict[Node, Set[Node]] = {}

    def dfs(u: Node) -> Set[Node]:
        if u in memo:
            return memo[u]
        out = {u}
        for c in children.get(u, []):
            out |= dfs(c)
        memo[u] = out
        return out

    dfs(root)
    return memo


def subtree_nodes(u: Node, children: Mapping[Node, Sequence[Node]]) -> Set[Node]:
    out = {u}
    stack = [u]
    while stack:
        x = stack.pop()
        for c in children.get(x, []):
            out.add(c)
            stack.append(c)
    return out


def path_in_tree(u: Node, parent: Mapping[Node, Node | None]) -> List[Node]:
    path: List[Node] = []
    cur: Node | None = u
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def ancestors_exclusive(u: Node, parent: Mapping[Node, Node | None], root: Node) -> List[Node]:
    p = path_in_tree(u, parent)
    if p and p[0] == root:
        p = p[1:]
    if p and p[-1] == u:
        p = p[:-1]
    return p


def top_k_nodes(nodes: Iterable[Node], bids: Mapping[Node, float], k: int) -> List[Node]:
    cand = sorted(nodes, key=lambda x: (-bids.get(x, float('-inf')), str(x)))
    return cand[:k]


def minimal_subtree_nodes(root: Node, winners: Iterable[Node], parent: Mapping[Node, Node | None]) -> Set[Node]:
    keep = {root}
    for w in winners:
        cur: Node | None = w
        while cur is not None:
            keep.add(cur)
            cur = parent[cur]
    return keep


def induced_tree(parent: Mapping[Node, Node | None], keep: Set[Node], root: Node) -> Dict[Node, Node | None]:
    out: Dict[Node, Node | None] = {root: None}
    for v in keep:
        if v == root:
            continue
        cur = parent[v]
        while cur is not None and cur not in keep:
            cur = parent[cur]
        out[v] = cur
    return out


@dataclass
class ClusterInfo:
    seller: Node
    clusters: Dict[Node, Set[Node]]
    cluster_roots: Dict[Node, Node]
    node_to_cluster: Dict[Node, Node]
    cluster_graph: nx.DiGraph
    cluster_tree: nx.DiGraph
    eta: Dict[Node, Node]
    macro_bids: Dict[Node, float]


def dominator_clusters(seller: Node, reports: Adj, bids: Mapping[Node, float], rng: random.Random) -> ClusterInfo:
    g = reachable_subgraph(build_digraph(seller, reports), seller)
    parent = immediate_dominators_tree(g, seller)
    children = tree_children(parent)
    desc = descendants_in_tree(seller, children)

    cluster_roots = {v: v for v, p in parent.items() if v != seller and p == seller}
    clusters: Dict[Node, Set[Node]] = {r: set(desc[r]) for r in cluster_roots}
    node_to_cluster: Dict[Node, Node] = {}
    for r, nodes in clusters.items():
        for u in nodes:
            node_to_cluster[u] = r

    H = nx.DiGraph()
    H.add_node(seller)
    for r in clusters:
        H.add_node(r)
        H.add_edge(seller, r)
    for u, nbrs in reports.items():
        if u not in node_to_cluster:
            continue
        cu = node_to_cluster[u]
        for v in nbrs:
            if v not in node_to_cluster:
                continue
            cv = node_to_cluster[v]
            if cu != cv:
                H.add_edge(cu, cv)

    T = random_shortest_path_tree(H, seller, rng)

    eta: Dict[Node, Node] = {}
    macro_bids: Dict[Node, float] = {}
    for r, nodes in clusters.items():
        rep = min(nodes, key=lambda x: (-bids[x], str(x)))
        eta[r] = rep
        macro_bids[r] = bids[rep]

    return ClusterInfo(
        seller=seller,
        clusters=clusters,
        cluster_roots=cluster_roots,
        node_to_cluster=node_to_cluster,
        cluster_graph=H,
        cluster_tree=T,
        eta=eta,
        macro_bids=macro_bids,
    )


def random_shortest_path_tree(g: nx.DiGraph, root: Node, rng: random.Random) -> nx.DiGraph:
    dist = nx.single_source_shortest_path_length(g, root)
    T = nx.DiGraph()
    for v in dist:
        T.add_node(v)
    for v in dist:
        if v == root:
            continue
        candidates = [u for u in g.predecessors(v) if u in dist and dist[u] + 1 == dist[v]]
        if not candidates:
            raise ValueError(f"No shortest-path parent for node {v}")
        p = rng.choice(sorted(candidates))
        T.add_edge(p, v)
    return T


def tree_to_parent(T: nx.DiGraph, root: Node) -> Dict[Node, Node | None]:
    parent: Dict[Node, Node | None] = {root: None}
    for u, v in nx.bfs_edges(T, root):
        parent[v] = u
    return parent


def tree_reports_from_nx(T: nx.DiGraph) -> Dict[Node, List[Node]]:
    out: Dict[Node, List[Node]] = {}
    for u in T.nodes:
        out[u] = sorted(T.successors(u))
    return out

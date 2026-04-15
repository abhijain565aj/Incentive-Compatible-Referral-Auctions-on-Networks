from __future__ import annotations
from typing import Dict, Optional
import random
import math
import networkx as nx
from .instance import AuctionInstance


def sample_valuations(nodes, mode: str = "uniform", seed: Optional[int] = None, **kwargs) -> Dict[int, float]:
    rng = random.Random(seed)
    vals: Dict[int, float] = {}
    for node in nodes:
        if node == 0:
            continue
        if mode == "uniform":
            lo, hi = kwargs.get("lo", 0.0), kwargs.get("hi", 1.0)
            vals[node] = rng.uniform(lo, hi)
        elif mode == "exponential":
            lam = kwargs.get("lam", 1.0)
            vals[node] = rng.expovariate(lam)
        elif mode == "depth_biased":
            # Non-i.i.d.: deeper agents have stochastically higher values.
            depth = kwargs.get("depth", {}).get(node, 1)
            vals[node] = rng.uniform(0, 1) + kwargs.get("slope", 0.15) * depth
        elif mode == "community":
            # Non-i.i.d.: nodes in a chosen community have higher mean.
            high = set(kwargs.get("high_nodes", []))
            if node in high:
                vals[node] = rng.uniform(0.6, 1.6)
            else:
                vals[node] = rng.uniform(0.0, 0.8)
        elif mode == "lognormal":
            vals[node] = rng.lognormvariate(kwargs.get("mu", 0.0), kwargs.get("sigma", 1.0))
        else:
            raise ValueError(f"unknown valuation mode: {mode}")
    return vals


def line_instance(n: int, seed: int = 0, valuation_mode: str = "uniform") -> AuctionInstance:
    G = nx.path_graph(n + 1)  # seller 0, buyers 1..n
    vals = sample_valuations(G.nodes, valuation_mode, seed=seed)
    return AuctionInstance(G, vals, name=f"line_{n}")


def star_instance(n: int, seed: int = 0, valuation_mode: str = "uniform") -> AuctionInstance:
    G = nx.star_graph(n)  # center 0 seller
    vals = sample_valuations(G.nodes, valuation_mode, seed=seed)
    return AuctionInstance(G, vals, name=f"star_{n}")


def erdos_renyi_instance(n: int, p: float, seed: int = 0, valuation_mode: str = "uniform") -> AuctionInstance:
    rng = random.Random(seed)
    # Generate buyers then connect seller to at least one node.
    G = nx.erdos_renyi_graph(n + 1, p, seed=seed)
    if 0 not in G:
        G.add_node(0)
    for i in range(1, n + 1):
        if i not in G:
            G.add_node(i)
    if G.degree(0) == 0:
        G.add_edge(0, rng.randint(1, n))
    # Keep the seller component only and relabel compactly if needed.
    comp = nx.node_connected_component(G, 0)
    G = G.subgraph(comp).copy()
    vals = sample_valuations(G.nodes, valuation_mode, seed=seed)
    return AuctionInstance(G, vals, name=f"er_{n}_{p}")


def barabasi_instance(n: int, m: int = 2, seed: int = 0, valuation_mode: str = "uniform") -> AuctionInstance:
    G = nx.barabasi_albert_graph(n + 1, m, seed=seed)
    # Treat node 0 as seller.
    vals = sample_valuations(G.nodes, valuation_mode, seed=seed)
    return AuctionInstance(G, vals, name=f"ba_{n}_{m}")


def balanced_tree_instance(branching: int, height: int, seed: int = 0, valuation_mode: str = "uniform") -> AuctionInstance:
    G = nx.balanced_tree(branching, height)
    # root 0 is seller.
    vals = sample_valuations(G.nodes, valuation_mode, seed=seed)
    return AuctionInstance(G, vals, name=f"tree_{branching}_{height}")

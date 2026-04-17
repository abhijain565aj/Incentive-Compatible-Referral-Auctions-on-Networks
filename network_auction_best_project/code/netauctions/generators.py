from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import random

from .network import RootedTree


@dataclass
class Instance:
    tree: RootedTree
    values: Dict[int, float]
    costs: Dict[int, float]
    means: Dict[int, float]
    sybil_owner: Dict[int, int]


def random_rooted_tree(n: int, rng: random.Random, max_children: Optional[int] = None) -> RootedTree:
    if n < 1:
        raise ValueError("n must be >= 1")
    max_children = max_children or max(2, int(n ** 0.5) + 1)
    parent = {0: None}
    remaining = list(range(1, n + 1))
    frontier = [0]
    while remaining:
        u = frontier.pop(0)
        kmax = min(max_children, len(remaining))
        k = rng.randint(1, kmax)
        if u == 0 and len(remaining) > 2:
            k = max(2, k)
        chosen = remaining[:k]
        remaining = remaining[k:]
        for v in chosen:
            parent[v] = u
            frontier.append(v)
    tree = RootedTree(parent)
    assert tree.is_valid()
    return tree


def path_tree(n: int) -> RootedTree:
    parent = {0: None}
    for i in range(1, n + 1):
        parent[i] = i - 1
    return RootedTree(parent)


def balanced_binary_tree(n: int) -> RootedTree:
    parent = {0: None}
    for i in range(1, n + 1):
        parent[i] = (i - 1) // 2
    return RootedTree(parent)


def noniid_means(n: int, rng: random.Random) -> Dict[int, float]:
    buyers = list(range(1, n + 1))
    rng.shuffle(buyers)
    means: Dict[int, float] = {}
    for idx, u in enumerate(buyers):
        if idx < max(1, n // 8):
            means[u] = 100.0
        elif idx < max(2, n // 2):
            means[u] = 70.0
        else:
            means[u] = 45.0
    return means


def sample_values(means: Dict[int, float], rng: random.Random, sigma: float = 15.0) -> Dict[int, float]:
    return {u: max(0.0, rng.gauss(mu, sigma)) for u, mu in means.items()}


def sample_costs(n: int, rng: random.Random, mode: str = "participation") -> Dict[int, float]:
    if mode == "zero":
        return {u: 0.0 for u in range(1, n + 1)}
    if mode == "participation":
        return {u: rng.gammavariate(2.0, 3.0) for u in range(1, n + 1)}
    if mode == "high":
        return {u: rng.gammavariate(3.0, 6.0) for u in range(1, n + 1)}
    raise ValueError("unknown cost mode")


def make_instance(n: int, seed: int, topology: str = "random", sigma: float = 15.0,
                  cost_mode: str = "participation") -> Instance:
    rng = random.Random(seed)
    if topology == "random":
        tree = random_rooted_tree(n, rng)
    elif topology == "path":
        tree = path_tree(n)
    elif topology == "binary":
        tree = balanced_binary_tree(n)
    else:
        raise ValueError("topology must be random, path, or binary")
    means = noniid_means(n, rng)
    values = sample_values(means, rng, sigma=sigma)
    costs = sample_costs(n, rng, mode=cost_mode)
    return Instance(tree=tree, values=values, costs=costs, means=means, sybil_owner={})


def add_sybil_attack(inst: Instance, target_child: int, sybils: int, sybil_value: float = 0.0) -> Instance:
    parent = inst.tree.parent[target_child]
    if parent is None:
        raise ValueError("target_child cannot be the root")
    start = max(inst.tree.nodes) + 1
    new_tree = inst.tree.insert_sybil_chain((parent, target_child), sybils, start)
    values = dict(inst.values)
    costs = dict(inst.costs)
    means = dict(inst.means)
    owner: Dict[int, int] = dict(inst.sybil_owner)
    for z in range(start, start + sybils):
        values[z] = sybil_value
        costs[z] = 0.0
        means[z] = sybil_value
        owner[z] = parent
    return Instance(new_tree, values, costs, means, owner)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .network import RootedTree


@dataclass
class Instance:
    tree: RootedTree
    values: Dict[int, float]
    means: Dict[int, float]
    sensitivity: Dict[int, float]
    prior_upper: Dict[int, float]


def random_rooted_tree(n: int, rng: np.random.Generator, max_children: Optional[int] = None) -> RootedTree:
    """Generate a connected rooted tree with n buyers plus seller 0."""
    if n < 1:
        raise ValueError("n must be >= 1")
    max_children = max_children or max(2, int(np.sqrt(n)) + 1)
    parent = {0: None}
    remaining = list(range(1, n + 1))
    frontier = [0]
    while remaining:
        u = frontier.pop(0)
        # Ensure progress but avoid consuming all nodes at the root too often.
        k = int(rng.integers(1, min(max_children, len(remaining)) + 1))
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


def assign_means(n: int, rng: np.random.Generator) -> Dict[int, float]:
    """Non-i.i.d. bidder classes: one high, roughly half medium, rest low."""
    buyers = list(range(1, n + 1))
    rng.shuffle(buyers)
    means: Dict[int, float] = {}
    if buyers:
        means[buyers[0]] = 100.0
    for u in buyers[1: 1 + max(0, (n - 1) // 2)]:
        means[u] = 70.0
    for u in buyers:
        means.setdefault(u, 50.0)
    return means


def sample_values(means: Dict[int, float], rng: np.random.Generator, sigma: float = 10.0,
                  distribution: str = "normal") -> Dict[int, float]:
    values: Dict[int, float] = {}
    for u, mu in means.items():
        if distribution == "normal":
            values[u] = max(0.0, float(rng.normal(mu, sigma)))
        elif distribution == "uniform":
            # Support depends on the class mean, giving non-identical priors.
            values[u] = float(rng.uniform(0.0, 2.0 * mu))
        else:
            raise ValueError("distribution must be 'normal' or 'uniform'")
    return values


def prior_upper_bounds(means: Dict[int, float], sigma: float, distribution: str) -> Dict[int, float]:
    if distribution == "uniform":
        return {u: 2.0 * mu for u, mu in means.items()}
    # Conservative 4-sigma truncation for virtual-value simulations.
    return {u: max(1.0, mu + 4.0 * sigma) for u, mu in means.items()}


def sample_sensitivities(n: int, rng: np.random.Generator) -> Dict[int, float]:
    # Private discomfort / competitive externality weights are not reported in our mechanism.
    return {u: float(rng.beta(2.0, 5.0)) for u in range(1, n + 1)}


def make_instance(n: int, seed: int, sigma: float = 10.0, distribution: str = "normal",
                  max_children: Optional[int] = None) -> Instance:
    rng = np.random.default_rng(seed)
    tree = random_rooted_tree(n, rng, max_children=max_children)
    means = assign_means(n, rng)
    values = sample_values(means, rng, sigma=sigma, distribution=distribution)
    sensitivity = sample_sensitivities(n, rng)
    upper = prior_upper_bounds(means, sigma=sigma, distribution=distribution)
    return Instance(tree=tree, values=values, means=means, sensitivity=sensitivity, prior_upper=upper)

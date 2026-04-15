from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import math

from .network import RootedTree


@dataclass
class MechanismResult:
    name: str
    winner: Optional[int]
    winning_branch: Optional[int]
    revenue: float
    raw_welfare: float
    externality: float
    adjusted_welfare: float
    depth: int
    details: Dict[str, float]


def _branch_data(tree: RootedTree, values: Dict[int, float]) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, float]]:
    branches = tree.branch_subtrees()
    rep: Dict[int, int] = {}
    rho: Dict[int, float] = {}
    for b, nodes in branches.items():
        best = max(nodes, key=lambda u: values[u])
        rep[b] = best
        rho[b] = values[best]
    return branches, rep, rho


def branch_externality(tree: RootedTree, branch: int, sensitivities: Dict[int, float], eta: float = 1.0) -> float:
    """Public/estimated branch-level negative externality if this branch wins.

    This is deliberately structural, not a function of reported values, so that threshold monotonicity
    is preserved. It can be interpreted as expected competitive harm to agents outside the winning branch.
    """
    branch_nodes = set(tree.subtree(branch))
    outside = [u for u in tree.buyers if u not in branch_nodes]
    if not outside:
        return 0.0
    branch_size = len(branch_nodes)
    avg_sensitivity = sum(sensitivities[u] for u in outside) / len(outside)
    # Large branches create more market power; far branches are mildly discounted.
    depth_discount = 1.0 / (1.0 + 0.15 * tree.depth(branch))
    return eta * avg_sensitivity * math.sqrt(branch_size) * len(outside) * depth_discount


def all_branch_externalities(tree: RootedTree, sensitivities: Dict[int, float], eta: float) -> Dict[int, float]:
    return {b: branch_externality(tree, b, sensitivities, eta=eta) for b in tree.first_level()}


def _result(name: str, tree: RootedTree, values: Dict[int, float], winner: Optional[int], branch: Optional[int],
            revenue: float, ext: float, details: Dict[str, float]) -> MechanismResult:
    raw = 0.0 if winner is None else values[winner]
    return MechanismResult(
        name=name,
        winner=winner,
        winning_branch=branch,
        revenue=max(0.0, float(revenue)),
        raw_welfare=raw,
        externality=ext if winner is not None else 0.0,
        adjusted_welfare=raw - (ext if winner is not None else 0.0),
        depth=0 if winner is None else tree.depth(winner),
        details=details,
    )


def local_vickrey(tree: RootedTree, values: Dict[int, float]) -> MechanismResult:
    bidders = tree.first_level()
    if not bidders:
        return _result("Local-Vickrey", tree, values, None, None, 0.0, 0.0, {})
    ordered = sorted(bidders, key=lambda u: values[u], reverse=True)
    winner = ordered[0]
    revenue = values[ordered[1]] if len(ordered) > 1 else 0.0
    return _result("Local-Vickrey", tree, values, winner, winner, revenue, 0.0, {"second": revenue})


def referral_second_price(tree: RootedTree, values: Dict[int, float], sensitivities: Dict[int, float], eta: float = 0.0) -> MechanismResult:
    """Tree-level referral second price / IDM-style benchmark at first-level transformed auction.

    The seller sees each first-level child as representing the maximum valuation in its subtree.
    This is the transformed-auction view used in referral-auction simulations.
    """
    _, rep, rho = _branch_data(tree, values)
    if not rho:
        return _result("Referral-SP", tree, values, None, None, 0.0, 0.0, {})
    ordered = sorted(rho, key=lambda b: rho[b], reverse=True)
    b = ordered[0]
    second = rho[ordered[1]] if len(ordered) > 1 else 0.0
    ext = branch_externality(tree, b, sensitivities, eta=eta)
    return _result("Referral-SP", tree, values, rep[b], b, second, ext, {"rho_w": rho[b], "threshold": second})


def lblev(tree: RootedTree, values: Dict[int, float], sensitivities: Dict[int, float], eta: float,
          exponents: Dict[int, float]) -> MechanismResult:
    _, rep, rho = _branch_data(tree, values)
    if not rho:
        return _result("LbLEV", tree, values, None, None, 0.0, 0.0, {})
    score = {b: rho[b] ** exponents.get(b, 1.0) for b in rho}
    ordered = sorted(score, key=lambda b: score[b], reverse=True)
    b = ordered[0]
    second_score = score[ordered[1]] if len(ordered) > 1 else 0.0
    threshold = second_score ** (1.0 / exponents.get(b, 1.0)) if second_score > 0 else 0.0
    ext = branch_externality(tree, b, sensitivities, eta=eta)
    return _result("LbLEV", tree, values, rep[b], b, threshold, ext,
                   {"rho_w": rho[b], "score_w": score[b], "threshold": threshold})


def learn_lblev_exponents(tree: RootedTree, means: Dict[int, float], lam: float = 0.5) -> Dict[int, float]:
    """Simple prior-based exponents inspired by Bhattacharyya et al.'s LbLEV experiment.

    Compute the expected representative value of each first-level subtree as max mean in that subtree.
    Increase the expected runner-up branch exponent toward log(winner)/log(runner-up).
    """
    branches = tree.branch_subtrees()
    if len(branches) <= 1:
        return {b: 1.0 for b in branches}
    expected = {b: max(means[u] for u in nodes) for b, nodes in branches.items()}
    ordered = sorted(expected, key=lambda b: expected[b], reverse=True)
    winner, runner = ordered[0], ordered[1]
    exponents = {b: 1.0 for b in branches}
    if expected[runner] > 1.0 and expected[winner] > expected[runner]:
        target = math.log(expected[winner]) / math.log(expected[runner])
        exponents[runner] = (1.0 - lam) * 1.0 + lam * target
    return exponents


def externality_adjusted_sp(tree: RootedTree, values: Dict[int, float], sensitivities: Dict[int, float],
                            eta: float = 1.0, penalty_lambda: float = 1.0) -> MechanismResult:
    """Externality-adjusted referral second price.

    Selects branch maximizing rho_b - lambda * E_b and charges the threshold in rho-space.
    This is DSIC for the scalar transformed branch value because allocation is monotone in rho_b.
    """
    _, rep, rho = _branch_data(tree, values)
    if not rho:
        return _result("EA-Referral-SP", tree, values, None, None, 0.0, 0.0, {})
    E = all_branch_externalities(tree, sensitivities, eta=eta)
    score = {b: rho[b] - penalty_lambda * E[b] for b in rho}
    b = max(score, key=lambda x: score[x])
    if score[b] < 0:
        return _result("EA-Referral-SP", tree, values, None, None, 0.0, 0.0,
                       {"best_score": score[b]})
    second_score = max([0.0] + [score[k] for k in rho if k != b])
    threshold = second_score + penalty_lambda * E[b]
    ext = E[b]
    return _result("EA-Referral-SP", tree, values, rep[b], b, threshold, ext,
                   {"rho_w": rho[b], "score_w": score[b], "threshold": threshold, "E_w": E[b]})


def _uniform_max_virtual(z: float, upper: float, m: int) -> float:
    z = max(1e-9, min(z, upper - 1e-9))
    return z - (upper ** m - z ** m) / (m * z ** (m - 1))


def _uniform_max_virtual_inv(target: float, upper: float, m: int) -> float:
    lo, hi = 0.0, upper
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _uniform_max_virtual(mid, upper, m) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def externality_adjusted_virtual(tree: RootedTree, values: Dict[int, float], sensitivities: Dict[int, float],
                                 prior_upper: Dict[int, float], eta: float = 1.0,
                                 penalty_lambda: float = 1.0) -> MechanismResult:
    """Regularized Myerson-style transformed referral auction.

    Assumes each bidder in a branch has independent uniform priors [0, upper_i]; we approximate the
    branch representative prior by a uniform-max with upper=max upper_i and m=branch size.
    The objective is expected revenue minus lambda times public branch externality.
    """
    branches, rep, rho = _branch_data(tree, values)
    if not rho:
        return _result("EA-Virtual-RA", tree, values, None, None, 0.0, 0.0, {})
    E = all_branch_externalities(tree, sensitivities, eta=eta)
    params = {b: (max(prior_upper[u] for u in nodes), len(nodes)) for b, nodes in branches.items()}
    virt = {b: _uniform_max_virtual(rho[b], params[b][0], params[b][1]) for b in rho}
    score = {b: virt[b] - penalty_lambda * E[b] for b in rho}
    b = max(score, key=lambda x: score[x])
    if score[b] < 0:
        return _result("EA-Virtual-RA", tree, values, None, None, 0.0, 0.0, {"best_score": score[b]})
    second = max([0.0] + [score[k] for k in rho if k != b])
    target_virtual = second + penalty_lambda * E[b]
    threshold = _uniform_max_virtual_inv(target_virtual, params[b][0], params[b][1])
    ext = E[b]
    return _result("EA-Virtual-RA", tree, values, rep[b], b, threshold, ext,
                   {"rho_w": rho[b], "virt_w": virt[b], "score_w": score[b],
                    "threshold": threshold, "E_w": E[b]})


def run_all_mechanisms(tree: RootedTree, values: Dict[int, float], means: Dict[int, float],
                       sensitivities: Dict[int, float], prior_upper: Dict[int, float], eta: float,
                       penalty_lambda: float, lblev_lambda: float) -> List[MechanismResult]:
    exponents = learn_lblev_exponents(tree, means, lam=lblev_lambda)
    return [
        local_vickrey(tree, values),
        referral_second_price(tree, values, sensitivities, eta=eta),
        lblev(tree, values, sensitivities, eta=eta, exponents=exponents),
        externality_adjusted_sp(tree, values, sensitivities, eta=eta, penalty_lambda=penalty_lambda),
        externality_adjusted_virtual(tree, values, sensitivities, prior_upper=prior_upper,
                                     eta=eta, penalty_lambda=penalty_lambda),
    ]

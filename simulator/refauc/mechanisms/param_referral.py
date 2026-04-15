from __future__ import annotations
from typing import Dict, Optional
from ..instance import (
    AuctionInstance, AuctionResult, invitation_digraph, simulate_diffusion,
    critical_descendant_sets, diffusion_critical_sequence, max_bidder, max_value, second_value
)


def parametric_referral_auction(
    inst: AuctionInstance,
    diffusion_strategy: str = "full",
    invite_prob: float = 1.0,
    seed=None,
    reserve_by_node: Optional[Dict[int, float]] = None,
    referral_share: float = 0.25,
    **kwargs,
) -> AuctionResult:
    """A simple tunable referral auction for experiments.

    This is not claimed as a theorem-level mechanism. It is a research sandbox
    inspired by referral-auction and MLM mechanisms: choose the highest bidder
    above a node-specific reserve, charge a second-price/reserve threshold, then
    redistribute a fraction of the winner payment along the critical sequence.

    It is useful for stress-testing non-i.i.d. priors and topology sensitivity.
    """
    reserve_by_node = reserve_by_node or {}
    participants, edges, depth = simulate_diffusion(inst, diffusion_strategy, invite_prob, seed=seed)
    DG = invitation_digraph(inst, edges)
    dsets = critical_descendant_sets(inst, DG, participants)

    eligible = [i for i in participants if inst.value(i) >= reserve_by_node.get(i, inst.reserve)]
    winner = max_bidder(inst, eligible)
    payments = {i: 0.0 for i in participants}
    if winner is None:
        return AuctionResult("param_referral", None, 0.0, payments, participants, edges, {"diffusion_depth": max(depth.values(), default=0)})

    # Payment is max of own reserve and second highest eligible value.
    base_payment = max(reserve_by_node.get(winner, inst.reserve), second_value(inst, eligible))
    C = diffusion_critical_sequence(inst, DG, participants, winner, dsets)
    ancestors = C[:-1]
    total_reward = referral_share * base_payment if ancestors else 0.0
    # Geometric reward towards nearer ancestors, still sum <= base_payment.
    if ancestors:
        weights = [2 ** k for k in range(len(ancestors))]
        Z = sum(weights)
        for node, wt in zip(ancestors, weights):
            payments[node] = -total_reward * wt / Z
    payments[winner] = base_payment
    return AuctionResult("param_referral", winner, inst.value(winner), payments, participants, edges, {
        "base_payment": base_payment,
        "referral_share": referral_share,
        "critical_sequence_winner": C,
        "dsets": dsets,
        "diffusion_depth": max([depth.get(i, 0) for i in participants], default=0),
    })

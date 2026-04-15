from __future__ import annotations
from ..instance import (
    AuctionInstance, AuctionResult, invitation_digraph, simulate_diffusion,
    critical_descendant_sets, max_bidder, max_value
)


def network_vcg(inst: AuctionInstance, diffusion_strategy: str = "full", invite_prob: float = 1.0, seed=None, **kwargs) -> AuctionResult:
    participants, edges, depth = simulate_diffusion(inst, diffusion_strategy, invite_prob, seed=seed)
    DG = invitation_digraph(inst, edges)
    dsets = critical_descendant_sets(inst, DG, participants)
    winner = max_bidder(inst, participants)
    payments = {i: 0.0 for i in participants}
    if winner is None or inst.value(winner) < inst.reserve:
        return AuctionResult("network_vcg", None, 0.0, payments, participants, edges, {"diffusion_depth": max(depth.values(), default=0)})
    W = inst.value(winner)
    for i in participants:
        outside_di = participants - dsets[i]
        W_without_di = max_value(inst, outside_di)
        payments[i] = W_without_di - (W - (inst.value(i) if i == winner else 0.0))
    return AuctionResult("network_vcg", winner, W, payments, participants, edges, {
        "dsets": dsets,
        "diffusion_depth": max([depth.get(i, 0) for i in participants], default=0),
    })

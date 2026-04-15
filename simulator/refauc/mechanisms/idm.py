from __future__ import annotations
from ..instance import (
    AuctionInstance, AuctionResult, invitation_digraph, simulate_diffusion,
    critical_descendant_sets, diffusion_critical_sequence, max_bidder, max_value
)

EPS = 1e-9


def information_diffusion_mechanism(inst: AuctionInstance, diffusion_strategy: str = "full", invite_prob: float = 1.0, seed=None, **kwargs) -> AuctionResult:
    participants, edges, depth = simulate_diffusion(inst, diffusion_strategy, invite_prob, seed=seed)
    DG = invitation_digraph(inst, edges)
    dsets = critical_descendant_sets(inst, DG, participants)
    m = max_bidder(inst, participants)
    payments = {i: 0.0 for i in participants}
    if m is None or inst.value(m) < inst.reserve:
        return AuctionResult("idm", None, 0.0, payments, participants, edges, {"diffusion_depth": max(depth.values(), default=0)})

    C = diffusion_critical_sequence(inst, DG, participants, m, dsets)
    w = m
    # First critical ancestor that is exactly the best outside the next descendant set.
    # With real-valued random bids, equality is tested numerically.
    for idx, i in enumerate(C[:-1]):
        nxt = C[idx + 1]
        outside_next = participants - dsets[nxt]
        threshold = max_value(inst, outside_next)
        if abs(inst.value(i) - threshold) <= EPS or inst.value(i) >= threshold - EPS:
            w = i
            break

    Cw = diffusion_critical_sequence(inst, DG, participants, w, dsets)
    for idx, i in enumerate(Cw[:-1]):
        nxt = Cw[idx + 1]
        payments[i] = max_value(inst, participants - dsets[i]) - max_value(inst, participants - dsets[nxt])
    payments[w] = max_value(inst, participants - dsets[w])
    return AuctionResult("idm", w, inst.value(w), payments, participants, edges, {
        "highest_bidder": m,
        "critical_sequence_highest": C,
        "critical_sequence_winner": Cw,
        "dsets": dsets,
        "diffusion_depth": max([depth.get(i, 0) for i in participants], default=0),
    })

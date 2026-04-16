from __future__ import annotations

from typing import Dict, List

from ..instance import (
    AuctionInstance,
    AuctionResult,
    critical_descendant_sets,
    diffusion_critical_sequence,
    invitation_digraph,
    max_bidder,
    max_value,
    second_value,
    simulate_diffusion,
)


def sybil_resistant_referral_auction(
    inst: AuctionInstance,
    diffusion_strategy: str = "full",
    invite_prob: float = 1.0,
    seed=None,
    reward_budget: float = 0.35,
    depth_penalty: float = 0.03,
    diffusion_bonus: float = 0.10,
    **kwargs,
) -> AuctionResult:
    """A practical, simulation-oriented, Sybil-aware referral mechanism.

    Design goals:
    1) Keep revenue nonnegative by capping referral rewards to a fraction
       of winner payment.
    2) Prefer candidates with stronger local bids, but discount deep chains
       to reduce reward-extraction opportunities from fake-depth (Sybil) nodes.
    3) Reward only true critical ancestors of the selected winner.

    Notes:
    - This mechanism is experimental and not claimed to be theorem-optimal.
    - Payments are buyer-to-seller; negative values are rewards.
    """
    reward_budget = min(max(reward_budget, 0.0), 0.9)
    participants, edges, depth = simulate_diffusion(inst, diffusion_strategy, invite_prob, seed=seed)
    DG = invitation_digraph(inst, edges)
    dsets = critical_descendant_sets(inst, DG, participants)
    payments = {i: 0.0 for i in participants}

    m = max_bidder(inst, participants)
    if m is None or inst.value(m) < inst.reserve:
        return AuctionResult(
            "sybil_resistant_referral",
            None,
            0.0,
            payments,
            participants,
            edges,
            {"diffusion_depth": max(depth.values(), default=0)},
        )

    C = diffusion_critical_sequence(inst, DG, participants, m, dsets)

    # Candidate scoring over the critical chain of the highest bidder.
    # A candidate must clear a local threshold, then maximizes:
    #   (bid - threshold) + diffusion_bonus * coverage - depth_penalty * depth
    best_score = float("-inf")
    winner = m
    threshold_map: Dict[int, float] = {}

    for idx, i in enumerate(C):
        if idx < len(C) - 1:
            nxt = C[idx + 1]
            threshold = max_value(inst, participants - dsets[nxt])
        else:
            threshold = max_value(inst, participants - dsets[i])
        threshold_map[i] = threshold

        if inst.value(i) + 1e-9 < threshold:
            continue

        coverage = len(dsets.get(i, set())) / max(1, len(participants))
        score = (inst.value(i) - threshold) + diffusion_bonus * coverage - depth_penalty * depth.get(i, 0)
        if score > best_score + 1e-12:
            best_score = score
            winner = i

    base_second = second_value(inst, participants)
    winner_threshold = threshold_map.get(winner, max_value(inst, participants - dsets[winner]))
    winner_payment = max(inst.reserve, base_second, winner_threshold)

    # Reward only critical ancestors of winner, with depth-discounted marginal impact.
    Cw = diffusion_critical_sequence(inst, DG, participants, winner, dsets)
    ancestors: List[int] = Cw[:-1]
    if ancestors:
        reward_pool = reward_budget * winner_payment
        weights: List[float] = []
        for idx, anc in enumerate(ancestors):
            nxt = Cw[idx + 1]
            marginal = max(1, len(dsets[anc]) - len(dsets[nxt]))
            w = marginal / (1.0 + depth.get(anc, 0))
            weights.append(w)
        Z = sum(weights)
        if Z > 0:
            for anc, w in zip(ancestors, weights):
                payments[anc] = -reward_pool * (w / Z)

    payments[winner] = winner_payment
    return AuctionResult(
        "sybil_resistant_referral",
        winner,
        inst.value(winner),
        payments,
        participants,
        edges,
        {
            "highest_bidder": m,
            "critical_sequence_highest": C,
            "critical_sequence_winner": Cw,
            "winner_threshold": winner_threshold,
            "base_second_value": base_second,
            "reward_budget": reward_budget,
            "depth_penalty": depth_penalty,
            "diffusion_bonus": diffusion_bonus,
            "dsets": dsets,
            "diffusion_depth": max([depth.get(i, 0) for i in participants], default=0),
        },
    )

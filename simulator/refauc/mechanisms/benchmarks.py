from __future__ import annotations
from typing import Optional, Set, Tuple
from ..instance import AuctionInstance, AuctionResult, max_bidder, second_value


def central_vickrey(inst: AuctionInstance, **kwargs) -> AuctionResult:
    participants = set(inst.buyers())
    winner = max_bidder(inst, participants)
    payments = {i: 0.0 for i in participants}
    if winner is not None and inst.value(winner) >= inst.reserve:
        payments[winner] = second_value(inst, participants)
        val = inst.value(winner)
    else:
        winner, val = None, 0.0
    return AuctionResult("central_vickrey", winner, val, payments, participants, meta={"diffusion_depth": None})


def local_vickrey(inst: AuctionInstance, **kwargs) -> AuctionResult:
    participants = set(inst.seller_neighbors())
    winner = max_bidder(inst, participants)
    payments = {i: 0.0 for i in participants}
    if winner is not None and inst.value(winner) >= inst.reserve:
        payments[winner] = second_value(inst, participants)
        val = inst.value(winner)
    else:
        winner, val = None, 0.0
    return AuctionResult("local_vickrey", winner, val, payments, participants, meta={"diffusion_depth": 1})

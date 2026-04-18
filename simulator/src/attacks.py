
"""
Sybil attack generators used in the experiments.

The no-collusion Sybil model follows the baseline Sybil paper:
fake identities can connect internally or to the creator's neighbours,
but cannot obtain incoming edges from outside except through the creator.
"""

from .utils import all_nodes


def apply_chain_attack(seller_net, reports, bids, attacker, q, syb_bid=None):
    reports = {u: list(vs) for u, vs in reports.items()}
    bids2 = dict(bids)
    owner = {n: n for n in all_nodes(seller_net, reports)}
    original_children = reports.get(attacker, [])
    if syb_bid is None:
        syb_bid = bids[attacker]

    if not original_children or q <= 0:
        return seller_net, reports, bids2, owner

    reports[attacker] = []
    previous = attacker
    for t in range(q):
        syb = f"syb{attacker}_{t}"
        owner[syb] = attacker
        bids2[syb] = syb_bid
        reports.setdefault(previous, [])
        reports[previous].append(syb)
        previous = syb
    reports[previous] = list(original_children)
    return seller_net, reports, bids2, owner


def apply_star_attack(seller_net, reports, bids, attacker, q, syb_bid=None):
    reports = {u: list(vs) for u, vs in reports.items()}
    bids2 = dict(bids)
    owner = {n: n for n in all_nodes(seller_net, reports)}
    original_children = reports.get(attacker, [])
    if syb_bid is None:
        syb_bid = bids[attacker]

    if not original_children or q <= 0:
        return seller_net, reports, bids2, owner

    reports[attacker] = []
    for t in range(q):
        syb = f"syb{attacker}_{t}"
        owner[syb] = attacker
        bids2[syb] = syb_bid
        reports[attacker].append(syb)
        reports[syb] = list(original_children)
    return seller_net, reports, bids2, owner

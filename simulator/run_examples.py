from __future__ import annotations

import json
import random
from pathlib import Path

from src.attacks import truthful_owner, apply_split_child_attack
from src.examples import user_gidm_truthful, user_gidm_attacked
from src.gidm_tree import gidm_from_graph
from src.metrics import attacker_reward_mass, real_utility, real_welfare, seller_revenue, fake_winners
from src.sc_gidm import sc_gidm


def summarize(name, winners, payments, owner, bids, attacker):
    util = real_utility(winners, payments, owner, bids)
    return {
        'name': name,
        'winners': list(winners),
        'payments': dict(sorted(payments.items())),
        'seller_revenue': seller_revenue(payments),
        'attacker_utility': util[attacker],
        'attacker_reward_mass': attacker_reward_mass(payments, owner, attacker),
        'real_welfare': real_welfare(winners, owner, bids),
        'fake_winners': fake_winners(winners, owner),
    }


def main():
    out = Path('results')
    out.mkdir(exist_ok=True)

    seller, reports, bids, K, attacker = user_gidm_truthful()
    owner = truthful_owner({u: v for u, v in reports.items() if u != seller})
    gidm_truth = gidm_from_graph(seller, reports, bids, K)

    seller2, reports2, bids2, K2, attacker2 = user_gidm_attacked()
    owner2 = truthful_owner({u: v for u, v in reports2.items() if u != seller2})
    owner2["C'"] = attacker2
    gidm_attack = gidm_from_graph(seller2, reports2, bids2, K2)
    sc_attack = sc_gidm(seller2, reports2, bids2, K2, random.Random(7))

    data = {
        'truthful_gidm': summarize('truthful_gidm', gidm_truth.winners, gidm_truth.payments, owner, bids, attacker),
        'attacked_gidm': summarize('attacked_gidm', gidm_attack.winners, gidm_attack.payments, owner2, bids2, attacker2),
        'attacked_sc_gidm': summarize('attacked_sc_gidm', sc_attack.winners, sc_attack.payments, owner2, bids2, attacker2),
    }
    data['deltas'] = {
        'gidm_attacker_reward_change': data['attacked_gidm']['attacker_reward_mass'] - data['truthful_gidm']['attacker_reward_mass'],
        'gidm_attacker_utility_change': data['attacked_gidm']['attacker_utility'] - data['truthful_gidm']['attacker_utility'],
        'sc_vs_attacked_reward_change': data['attacked_sc_gidm']['attacker_reward_mass'] - data['attacked_gidm']['attacker_reward_mass'],
    }
    (out / 'user_examples.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()

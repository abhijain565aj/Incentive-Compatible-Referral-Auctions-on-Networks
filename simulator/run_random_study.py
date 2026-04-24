from __future__ import annotations

import argparse
import random
from pathlib import Path

import networkx as nx
import pandas as pd

from src.attacks import truthful_owner, apply_split_child_attack
from src.gidm_tree import gidm_from_graph
from src.metrics import attacker_reward_mass, real_utility, real_welfare, seller_revenue, fake_winners, reward_mass_to_fake_nodes, reward_mass_to_real_nodes
from src.sc_gidm import sc_gidm


def price_dag(n: int, m: int, seed: int):
    rng = random.Random(seed)
    seller = 'S'
    nodes = [seller]
    reports = {seller: []}
    bids = {seller: 0.0}
    for t in range(n):
        u = f'b{t}'
        reports[u] = []
        bids[u] = rng.random()
        # choose m parents from existing nodes with preferential attachment over outdegree+1
        weights = [len(reports[x]) + 1 for x in nodes]
        parents = set()
        while len(parents) < min(m, len(nodes)):
            parents.add(rng.choices(nodes, weights=weights, k=1)[0])
        for p in parents:
            reports[p].append(u)
        nodes.append(u)
    return seller, reports, bids


def choose_attacker(reports, bids):
    cands = [u for u, vs in reports.items() if u != 'S' and len(vs) > 0]
    return max(cands, key=lambda x: (len(reports[x]), bids[x], x)) if cands else None


def row_for(mech, winners, payments, owner, bids, attacker):
    util = real_utility(winners, payments, owner, bids)
    return {
        'mechanism': mech,
        'seller_revenue': seller_revenue(payments),
        'attacker_utility': util.get(attacker, 0.0),
        'attacker_reward_mass': attacker_reward_mass(payments, owner, attacker),
        'real_welfare': real_welfare(winners, owner, bids),
        'fake_winners': fake_winners(winners, owner),
        'fake_reward_mass': reward_mass_to_fake_nodes(payments, owner),
        'real_reward_mass': reward_mass_to_real_nodes(payments, owner),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results')
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--m', type=int, default=2)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--qmax', type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        seller, reports, bids = price_dag(args.n, args.m, seed)
        attacker = choose_attacker(reports, bids)
        if attacker is None:
            continue
        owner0 = truthful_owner({u: vs for u, vs in reports.items() if u != seller})
        base = gidm_from_graph(seller, reports, bids, args.K)
        base_row = row_for('GIDM_base', base.winners, base.payments, owner0, bids, attacker)
        base_util = base_row['attacker_utility']
        base_reward = base_row['attacker_reward_mass']
        for q in range(args.qmax + 1):
            if q == 0:
                reports_a, bids_a, owner = reports, bids, owner0
            else:
                reports_a, bids_a, owner = apply_split_child_attack(seller, reports, bids, owner0, attacker, q=q, sybil_bid=bids[attacker])
            gidm = gidm_from_graph(seller, reports_a, bids_a, args.K)
            sc = sc_gidm(seller, reports_a, bids_a, args.K, random.Random(seed + 17 * q + 3))
            for mech, winners, payments in [
                ('GIDM', gidm.winners, gidm.payments),
                ('SC-GIDM', sc.winners, sc.payments),
            ]:
                r = row_for(mech, winners, payments, owner, bids_a, attacker)
                r.update({'seed': seed, 'q': q, 'attacker': attacker, 'attacker_value': bids[attacker]})
                r['attacker_gain'] = r['attacker_utility'] - base_util
                r['attacker_reward_gain'] = r['attacker_reward_mass'] - base_reward
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'random_study.csv', index=False)
    summary = df.groupby(['mechanism', 'q'])[[
        'seller_revenue','attacker_utility','attacker_gain','attacker_reward_mass','attacker_reward_gain','real_welfare','fake_winners','fake_reward_mass','real_reward_mass'
    ]].mean().reset_index()
    summary.to_csv(out / 'random_summary.csv', index=False)
    print(summary)


if __name__ == '__main__':
    main()

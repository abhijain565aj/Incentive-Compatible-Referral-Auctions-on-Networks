
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from code.src.attacks import apply_chain_attack, apply_star_attack
from code.src.cluster_gidm import sc_gidm
from code.src.gidm_adapted import gidm, local_k_vickrey, seller_revenue
from code.src.sim_utils import generate_graph, pick_attacker
from code.src.utils import truthful_owner, utility_single_unit, real_welfare, fake_winner_count


def mechanism_metrics(alloc, pay, owner, true_values, attacker):
    utility = utility_single_unit(alloc, pay, owner, true_values)
    return {
        "revenue": seller_revenue(pay),
        "welfare": real_welfare(alloc, owner, true_values),
        "attacker_utility": utility[attacker],
        "fake_winners": fake_winner_count(alloc),
    }


def run_single(seed, model, q, k, n, seller_deg, attack_mode, syb_bid_mode):
    seller_net, reports, bids, g = generate_graph(n=n, model=model, seed=seed, seller_deg=seller_deg)
    attacker = pick_attacker(g)

    alloc0, pay0 = gidm(k, seller_net, reports, bids)
    owner0 = truthful_owner(seller_net, reports)
    true_values = {i: bids[i] for i in owner0}
    base = mechanism_metrics(alloc0, pay0, owner0, true_values, attacker)

    if syb_bid_mode == "same":
        syb_bid = bids[attacker]
    elif syb_bid_mode == "high":
        syb_bid = max(100, bids[attacker] + 20)
    elif syb_bid_mode == "low":
        syb_bid = max(1, bids[attacker] // 4)
    else:
        syb_bid = bids[attacker]

    builder = apply_chain_attack if attack_mode == "chain" else apply_star_attack
    seller_net_a, reports_a, bids_a, owner_a = builder(seller_net, reports, bids, attacker, q, syb_bid=syb_bid)

    alloc_g, pay_g = gidm(k, seller_net_a, reports_a, bids_a)
    m_g = mechanism_metrics(alloc_g, pay_g, owner_a, bids, attacker)
    m_g["attacker_gain"] = m_g["attacker_utility"] - base["attacker_utility"]

    alloc_sc, pay_sc = sc_gidm(k, seller_net_a, reports_a, bids_a, owner_a)
    m_sc = mechanism_metrics(alloc_sc, pay_sc, owner_a, bids, attacker)
    m_sc["attacker_gain"] = m_sc["attacker_utility"] - base["attacker_utility"]

    alloc_lv, pay_lv = local_k_vickrey(k, seller_net, bids)
    m_lv = mechanism_metrics(alloc_lv, pay_lv, owner0, true_values, attacker)

    row = {
        "seed": seed,
        "model": model,
        "q": q,
        "attacker": attacker,
        "attacker_value": bids[attacker],
    }
    for prefix, data in [("base_gidm", base), ("attacked_gidm", m_g), ("sc_gidm", m_sc), ("local_k_vickrey", m_lv)]:
        for k2, v in data.items():
            row[f"{prefix}_{k2}"] = v
    return row


def canonical_family():
    seller_net = ["b1", "b3"]
    reports = {
        "b0": ["b2", "b4", "b6"],
        "b1": ["b3", "b7"],
        "b2": ["b0", "b7"],
        "b3": ["b1"],
        "b4": ["b0", "b5"],
        "b5": ["b4"],
        "b6": ["b0"],
        "b7": ["b1", "b2"],
    }
    bids = {"b0": 98, "b1": 34, "b2": 5, "b3": 1, "b4": 19, "b5": 85, "b6": 76, "b7": 61}
    return seller_net, reports, bids, "b0", 3


def run_canonical(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seller_net, reports, bids, attacker, k = canonical_family()
    base_alloc, base_pay = gidm(k, seller_net, reports, bids)
    base_owner = truthful_owner(seller_net, reports)
    base_utility = utility_single_unit(base_alloc, base_pay, base_owner, bids)[attacker]

    rows = []
    for q in range(0, 6):
        if q == 0:
            owner = base_owner
            alloc_g, pay_g = base_alloc, base_pay
            alloc_sc, pay_sc = base_alloc, base_pay
        else:
            seller_net_a, reports_a, bids_a, owner = apply_chain_attack(
                seller_net, reports, bids, attacker, q, syb_bid=bids[attacker]
            )
            alloc_g, pay_g = gidm(k, seller_net_a, reports_a, bids_a)
            alloc_sc, pay_sc = sc_gidm(k, seller_net_a, reports_a, bids_a, owner)

        for mech, alloc, pay in [("GIDM", alloc_g, pay_g), ("SC-GIDM", alloc_sc, pay_sc)]:
            utility = utility_single_unit(alloc, pay, owner, bids)[attacker]
            rows.append(
                {
                    "q": q,
                    "mechanism": mech,
                    "attacker_utility": utility,
                    "attacker_gain": utility - base_utility,
                    "fake_winners": fake_winner_count(alloc),
                    "real_welfare": real_welfare(alloc, owner, bids),
                    "revenue": seller_revenue(pay),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "canonical_family.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mech in ["GIDM", "SC-GIDM"]:
        sub = df[df["mechanism"] == mech]
        ax.plot(sub["q"], sub["attacker_gain"], marker="o", label=mech)
    ax.set_xlabel("Number of Sybils q")
    ax.set_ylabel("Attacker gain")
    ax.set_title("Canonical family: attacker gain under a chain Sybil attack")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "canonical_attacker_gain.png", dpi=200)
    plt.close(fig)


def run_random_experiments(out_dir, seeds=100, n=20, k=4, seller_deg=3, attack_mode="chain", syb_bid_mode="same"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ["tree", "er"]:
        for q in range(0, 5):
            for seed in range(seeds):
                row = run_single(seed, model, q, k, n, seller_deg, attack_mode, syb_bid_mode)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "random_experiments.csv", index=False)

    summary = (
        df.groupby(["model", "q"])[
            [
                "attacked_gidm_welfare",
                "sc_gidm_welfare",
                "attacked_gidm_revenue",
                "sc_gidm_revenue",
                "attacked_gidm_fake_winners",
                "sc_gidm_fake_winners",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(out_dir / "random_summary.csv", index=False)

    for model in ["tree", "er"]:
        sub = summary[summary["model"] == model]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["q"], sub["attacked_gidm_welfare"], marker="o", label="Attacked GIDM")
        ax.plot(sub["q"], sub["sc_gidm_welfare"], marker="s", label="SC-GIDM")
        ax.set_xlabel("Number of Sybils q")
        ax.set_ylabel("Average real welfare")
        ax.set_title(f"Real welfare under Sybil attacks ({model})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"welfare_{model}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["q"], sub["attacked_gidm_fake_winners"], marker="o", label="Attacked GIDM")
        ax.plot(sub["q"], sub["sc_gidm_fake_winners"], marker="s", label="SC-GIDM")
        ax.set_xlabel("Number of Sybils q")
        ax.set_ylabel("Average number of fake winners")
        ax.set_title(f"Fake allocations under Sybil attacks ({model})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"fake_winners_{model}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["q"], sub["attacked_gidm_revenue"], marker="o", label="Attacked GIDM")
        ax.plot(sub["q"], sub["sc_gidm_revenue"], marker="s", label="SC-GIDM")
        ax.set_xlabel("Number of Sybils q")
        ax.set_ylabel("Average seller revenue")
        ax.set_title(f"Seller revenue under Sybil attacks ({model})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"revenue_{model}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    run_canonical(out_dir)
    run_random_experiments(out_dir, seeds=args.seeds)


if __name__ == "__main__":
    main()

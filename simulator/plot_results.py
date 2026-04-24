from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ('seller_revenue', 'Seller revenue'),
    ('attacker_gain', 'Attacker utility gain'),
    ('attacker_reward_gain', 'Attacker reward gain'),
    ('real_welfare', 'Real welfare'),
    ('fake_winners', 'Fake winners'),
    ('fake_reward_mass', 'Reward mass to fake nodes'),
    ('real_reward_mass', 'Reward mass to real nodes'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results')
    args = ap.parse_args()
    res = Path(args.results_dir)
    df = pd.read_csv(res / 'random_summary.csv')
    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(7,4))
        for mech in sorted(df['mechanism'].unique()):
            sub = df[df['mechanism'] == mech]
            ax.plot(sub['q'], sub[metric], marker='o', label=mech)
        ax.set_xlabel('Sybil budget q')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(res / f'{metric}.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main(path="results/sample_results.csv"):
    df = pd.read_csv(path)
    out = Path("results/plots")
    out.mkdir(parents=True, exist_ok=True)
    for metric in ["revenue", "welfare", "n_participants"]:
        avg = df.groupby(["topology", "mechanism"], as_index=False)[metric].mean()
        for topo, sub in avg.groupby("topology"):
            plt.figure(figsize=(8, 4))
            plt.bar(sub["mechanism"], sub[metric])
            plt.title(f"Average {metric} on {topo}")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(out / f"{metric}_{topo}.png", dpi=160)
            plt.close()
    print(f"plots written under {out}")

if __name__ == "__main__":
    main()

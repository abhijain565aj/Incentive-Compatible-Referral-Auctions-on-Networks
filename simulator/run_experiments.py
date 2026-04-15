from refauc.experiments import sweep

if __name__ == "__main__":
    rows = sweep(
        out_csv="results/sample_results.csv",
        seeds=range(5),
        sizes=(15, 30),
        topologies=("line", "tree", "er", "ba"),
        valuation_mode="uniform",
        diffusion_strategy="full",
    )
    print(f"wrote {len(rows)} rows to results/sample_results.csv")

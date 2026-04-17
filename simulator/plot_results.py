from pathlib import Path
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _load_default_paths(config_path: str = "config.ini") -> tuple[str, str]:
    cfg = configparser.ConfigParser()
    loaded = cfg.read(config_path)
    if not loaded or "paths" not in cfg:
        return "results/sample_results.csv", "results/plots"
    return (
        cfg["paths"].get("raw_results_csv", "results/sample_results.csv"),
        cfg["paths"].get("plot_dir", "results/plots"),
    )


def _save(fig_or_grid, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig_or_grid, "savefig"):
        fig_or_grid.savefig(path, dpi=240, bbox_inches="tight")
    else:
        plt.savefig(path, dpi=240, bbox_inches="tight")


def main(path: str | None = None, out_dir: str | None = None, config_path: str = "config.ini"):
    default_path, default_out = _load_default_paths(config_path)
    path = path or default_path
    out_dir = out_dir or default_out

    df = pd.read_csv(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = "DejaVu Serif"

    # Keep mechanism ordering and labels stable in every chart.
    mech_order = [
        "central_vickrey",
        "local_vickrey",
        "network_vcg",
        "idm",
        "param_referral",
        "sybil_resistant_referral",
    ]
    mech_label = {
        "central_vickrey": "Central-Vickrey",
        "local_vickrey": "Local-Vickrey",
        "network_vcg": "Network-VCG",
        "idm": "IDM",
        "param_referral": "Param-Referral",
        "sybil_resistant_referral": "Sybil-Resistant",
    }
    topo_label = {
        "line": "Line",
        "star": "Star",
        "tree": "Tree",
        "er": "Erdos-Renyi",
        "ba": "Scale-Free",
    }

    df["mechanism_label"] = df["mechanism"].map(mech_label).fillna(df["mechanism"])
    if "topology" in df.columns:
        df["topology_label"] = df["topology"].map(topo_label).fillna(df["topology"])
    else:
        df["topology_label"] = "All"

    for col in [
        "revenue",
        "welfare",
        "welfare_sum_utilities",
        "welfare_product_utilities",
        "welfare_log_product_utilities",
        "n_participants",
        "diffusion_depth",
        "winner_value",
        "negative_payments",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean non-finite values for plotting stability.
    df = df.replace([np.inf, -np.inf], np.nan)

    existing_order = [m for m in mech_order if m in set(df["mechanism"])]
    existing_labels = [mech_label[m] for m in existing_order]

    # 1) Revenue distribution by mechanism (violin + quartiles)
    plt.figure(figsize=(12, 5.6), constrained_layout=True)
    sns.violinplot(
        data=df,
        x="mechanism_label",
        y="revenue",
        hue="mechanism_label",
        order=existing_labels,
        hue_order=existing_labels,
        inner="quartile",
        cut=0,
        linewidth=1,
        palette="Set2",
        legend=False,
    )
    plt.title("Seller Revenue Distribution Across Mechanisms")
    plt.xlabel("Mechanism")
    plt.ylabel("Revenue")
    plt.xticks(rotation=25, ha="right")
    _save(plt, out / "01_revenue_violin_mechanisms.png")
    plt.close()

    # 2) Welfare distribution by mechanism
    plt.figure(figsize=(12, 5.6), constrained_layout=True)
    sns.violinplot(
        data=df,
        x="mechanism_label",
        y="welfare",
        hue="mechanism_label",
        order=existing_labels,
        hue_order=existing_labels,
        inner="quartile",
        cut=0,
        linewidth=1,
        palette="Set3",
        legend=False,
    )
    plt.title("Social Welfare Distribution Across Mechanisms")
    plt.xlabel("Mechanism")
    plt.ylabel("Welfare")
    plt.xticks(rotation=25, ha="right")
    _save(plt, out / "02_welfare_violin_mechanisms.png")
    plt.close()

    # 3) Participants distribution (boxplot)
    plt.figure(figsize=(12, 5.6), constrained_layout=True)
    sns.boxplot(
        data=df,
        x="mechanism_label",
        y="n_participants",
        hue="mechanism_label",
        order=existing_labels,
        hue_order=existing_labels,
        palette="pastel",
        fliersize=2,
        legend=False,
    )
    plt.title("Participant Reach Across Mechanisms")
    plt.xlabel("Mechanism")
    plt.ylabel("Number of Participants")
    plt.xticks(rotation=25, ha="right")
    _save(plt, out / "03_participants_box_mechanisms.png")
    plt.close()

    # 4) Mean revenue with 95% CI by topology
    g = sns.catplot(
        data=df,
        kind="bar",
        x="topology_label",
        y="revenue",
        hue="mechanism_label",
        hue_order=existing_labels,
        errorbar=("ci", 95),
        capsize=0.08,
        height=5.1,
        aspect=1.25,
        palette="tab10",
    )
    g.set_titles("")
    g.set_axis_labels("Topology", "Mean Revenue")
    g.fig.suptitle("Revenue Comparison by Topology", y=1.03)
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    _save(g, out / "04_revenue_bar_topology.png")
    plt.close("all")

    # 5) Mean welfare with 95% CI by topology
    g = sns.catplot(
        data=df,
        kind="bar",
        x="topology_label",
        y="welfare",
        hue="mechanism_label",
        hue_order=existing_labels,
        errorbar=("ci", 95),
        capsize=0.08,
        height=5.1,
        aspect=1.25,
        palette="viridis",
    )
    g.set_titles("")
    g.set_axis_labels("Topology", "Mean Welfare")
    g.fig.suptitle("Welfare Comparison by Topology", y=1.03)
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    _save(g, out / "05_welfare_bar_topology.png")
    plt.close("all")

    # 6) Revenue by valuation mode (if available)
    if "valuation_mode" in df.columns:
        g = sns.catplot(
            data=df,
            kind="bar",
            x="valuation_mode",
            y="revenue",
            hue="mechanism_label",
            hue_order=existing_labels,
            errorbar=("ci", 95),
            capsize=0.08,
            height=5.1,
            aspect=1.25,
            palette="Set1",
        )
        g.set_axis_labels("Valuation Mode", "Mean Revenue")
        g.fig.suptitle("Revenue Comparison by Valuation Mode", y=1.03)
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        _save(g, out / "06_revenue_bar_valuation_mode.png")
        plt.close("all")

    # 7) Welfare vs revenue Pareto-style scatter
    agg = (
        df.groupby(["mechanism", "topology"], as_index=False)
        .agg(
            revenue_mean=("revenue", "mean"),
            welfare_mean=("welfare", "mean"),
            participants_mean=("n_participants", "mean"),
        )
    )
    agg["mechanism_label"] = agg["mechanism"].map(mech_label).fillna(agg["mechanism"])
    agg["topology_label"] = agg["topology"].map(topo_label).fillna(agg["topology"])
    plt.figure(figsize=(11, 6.6), constrained_layout=True)
    sns.scatterplot(
        data=agg,
        x="welfare_mean",
        y="revenue_mean",
        hue="mechanism_label",
        hue_order=existing_labels,
        style="topology_label",
        size="participants_mean",
        sizes=(50, 220),
        alpha=0.9,
        palette="Dark2",
    )
    plt.title("Revenue-Welfare Frontier (mean over runs)")
    plt.xlabel("Mean Welfare")
    plt.ylabel("Mean Revenue")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _save(plt, out / "07_revenue_welfare_frontier.png")
    plt.close()

    # 8) Budget-balance heatmap (share of runs with nonnegative revenue)
    bb = (
        df.assign(nonnegative_revenue=(df["revenue"] >= 0).astype(float))
        .groupby(["mechanism", "topology"], as_index=False)["nonnegative_revenue"]
        .mean()
    )
    heat = bb.pivot(index="mechanism", columns="topology", values="nonnegative_revenue")
    heat = heat.reindex(existing_order)
    heat.index = [mech_label.get(m, m) for m in heat.index]
    heat = heat.rename(columns=topo_label)
    plt.figure(figsize=(10, 5.5))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title("Budget Balance Rate by Mechanism and Topology")
    plt.xlabel("Topology")
    plt.ylabel("Mechanism")
    _save(plt, out / "08_budget_balance_heatmap.png")
    plt.close()

    # 9) Mean revenue heatmap
    rev_h = (
        df.groupby(["mechanism", "topology"], as_index=False)["revenue"]
        .mean()
        .pivot(index="mechanism", columns="topology", values="revenue")
        .reindex(existing_order)
    )
    rev_h.index = [mech_label.get(m, m) for m in rev_h.index]
    rev_h = rev_h.rename(columns=topo_label)
    plt.figure(figsize=(10, 5.5))
    sns.heatmap(rev_h, annot=True, fmt=".3f", cmap="rocket_r")
    plt.title("Mean Revenue Heatmap by Mechanism and Topology")
    plt.xlabel("Topology")
    plt.ylabel("Mechanism")
    _save(plt, out / "09_mean_revenue_heatmap.png")
    plt.close()

    # 10) Diffusion depth / negative payments comparison
    if "diffusion_depth" in df.columns and df["diffusion_depth"].notna().any():
        ddf = df.dropna(subset=["diffusion_depth"]).copy()
        plt.figure(figsize=(12, 5.6), constrained_layout=True)
        sns.boxplot(
            data=ddf,
            x="mechanism_label",
            y="diffusion_depth",
            hue="mechanism_label",
            order=existing_labels,
            hue_order=existing_labels,
            palette="coolwarm",
            fliersize=2,
            legend=False,
        )
        plt.title("Diffusion Depth Distribution Across Mechanisms")
        plt.xlabel("Mechanism")
        plt.ylabel("Diffusion Depth")
        plt.xticks(rotation=25, ha="right")
        _save(plt, out / "10_diffusion_depth_box_mechanisms.png")
        plt.close()
    else:
        npay = (
            df.groupby("mechanism_label", as_index=False)["negative_payments"]
            .mean()
            .set_index("mechanism_label")
            .reindex(existing_labels)
            .reset_index()
        )
        plt.figure(figsize=(12, 5.6), constrained_layout=True)
        sns.barplot(
            data=npay,
            x="mechanism_label",
            y="negative_payments",
            hue="mechanism_label",
            palette="magma",
            legend=False,
        )
        plt.title("Mean Number of Negative Payments (Referral Rewards)")
        plt.xlabel("Mechanism")
        plt.ylabel("Mean Negative Payments")
        plt.xticks(rotation=25, ha="right")
        _save(plt, out / "10_negative_payments_bar_mechanisms.png")
        plt.close()

    # Optional: prominent uplift chart if IDM and novel mechanism both exist.
    if {"idm", "sybil_resistant_referral"}.issubset(set(df["mechanism"])):
        uplift = (
            df[df["mechanism"].isin(["idm", "sybil_resistant_referral"])]
            .groupby(["topology_label", "mechanism"], as_index=False)["revenue"]
            .mean()
            .pivot(index="topology_label", columns="mechanism", values="revenue")
            .reset_index()
        )
        uplift["uplift"] = uplift["sybil_resistant_referral"] - uplift["idm"]
        plt.figure(figsize=(10, 5.5), constrained_layout=True)
        sns.barplot(
            data=uplift,
            x="topology_label",
            y="uplift",
            hue="topology_label",
            palette="RdYlGn",
            legend=False,
        )
        plt.axhline(0.0, color="black", linewidth=1)
        plt.title("Revenue Uplift of Sybil-Resistant vs IDM")
        plt.xlabel("Topology")
        plt.ylabel("Mean Revenue Difference")
        _save(plt, out / "11_novel_uplift_vs_idm.png")
        plt.close()

    # 12) Welfare as sum of participant utilities
    if "welfare_sum_utilities" in df.columns:
        g = sns.catplot(
            data=df,
            kind="bar",
            x="topology_label",
            y="welfare_sum_utilities",
            hue="mechanism_label",
            hue_order=existing_labels,
            errorbar=("ci", 95),
            capsize=0.08,
            height=5.1,
            aspect=1.25,
            palette="Set2",
        )
        g.set_titles("")
        g.set_axis_labels("Topology", "Sum of Utilities")
        g.fig.suptitle("Welfare (Sum of Participant Utilities)", y=1.03)
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        _save(g, out / "12_welfare_sum_utilities_bar_topology.png")
        plt.close("all")

    # 13) Welfare as product of participant utilities
    if "welfare_product_utilities" in df.columns:
        g = sns.catplot(
            data=df,
            kind="bar",
            x="topology_label",
            y="welfare_product_utilities",
            hue="mechanism_label",
            hue_order=existing_labels,
            errorbar=("ci", 95),
            capsize=0.08,
            height=5.1,
            aspect=1.25,
            palette="Set3",
        )
        g.set_titles("")
        g.set_axis_labels("Topology", "Product of Utilities")
        g.fig.suptitle("Welfare (Product of Participant Utilities)", y=1.03)
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        _save(g, out / "13_welfare_product_utilities_bar_topology.png")
        plt.close("all")

    # 14) Welfare as log-product of participant utilities
    if "welfare_log_product_utilities" in df.columns:
        dflog = df.dropna(subset=["welfare_log_product_utilities"]).copy()
        if not dflog.empty:
            g = sns.catplot(
                data=dflog,
                kind="bar",
                x="topology_label",
                y="welfare_log_product_utilities",
                hue="mechanism_label",
                hue_order=existing_labels,
                errorbar=("ci", 95),
                capsize=0.08,
                height=5.1,
                aspect=1.25,
                palette="muted",
            )
            g.set_titles("")
            g.set_axis_labels("Topology", "Log Product of Utilities")
            g.fig.suptitle("Welfare (Log Product of Participant Utilities)", y=1.03)
            sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
            _save(g, out / "14_welfare_log_product_utilities_bar_topology.png")
            plt.close("all")

    print(f"plots written under {out}")

if __name__ == "__main__":
    main()

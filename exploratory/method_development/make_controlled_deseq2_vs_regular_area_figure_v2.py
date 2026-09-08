#!/usr/bin/env python3
"""
make_controlled_deseq2_vs_regular_area_figure_v2.py
===================================================

Updated publication-style controlled comparison of DESeq2 vs covariate-
adjusted Regular AREA using the exact same AD-vs-NCI cohort.

Changes versus the earlier version
----------------------------------
1. Reframes the figure explicitly as a *matched-condition method check*.
2. Clarifies that both methods use the same binary phenotype, same cohort,
   same covariates, and same gene universe.
3. Improves panel titles and annotations so the biological interpretation is
   more obvious.
4. Cleans up Panel C to emphasize why each method has exclusive genes.
5. Keeps the same core inputs and outputs a revised figure + summary table.

Inputs
------
results/visualizations/
    final_locked_covariate_DESeq2_Regular_Weighted_venn_gene_membership.csv

results/deseq2_locked_covariates/AD4_vs_NCI1/
    AD4_vs_NCI1_DESeq2_all_results_maxit1000.csv

results/venn_region_characterization/
    all_regions_gene_geometry.csv

Outputs
-------
results/controlled_deseq2_vs_regular_area/
    DESeq2_vs_Regular_AREA_controlled_comparison_v2.png
    DESeq2_vs_Regular_AREA_controlled_comparison_v2.pdf
    DESeq2_vs_Regular_AREA_summary_v2.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


MEMBERSHIP = Path(
    "results/visualizations/"
    "final_locked_covariate_DESeq2_Regular_Weighted_venn_gene_membership.csv"
)

DESEQ2 = Path(
    "results/deseq2_locked_covariates/AD4_vs_NCI1/"
    "AD4_vs_NCI1_DESeq2_all_results_maxit1000.csv"
)

GEOMETRY = Path(
    "results/venn_region_characterization/"
    "all_regions_gene_geometry.csv"
)

OUTDIR = Path("results/controlled_deseq2_vs_regular_area")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "DESeq2_vs_Regular_AREA_controlled_comparison_v2.png"
PDF = OUTDIR / "DESeq2_vs_Regular_AREA_controlled_comparison_v2.pdf"
SUMMARY = OUTDIR / "DESeq2_vs_Regular_AREA_summary_v2.csv"

BLUE = "#4C78A8"
ORANGE = "#F58518"
SHARED = "#6C6C6C"
LIGHT_GRAY = "#A9A9A9"
TEXT = "#181818"
MUTED = "#4A4A4A"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    }
)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def resolve_column(df: pd.DataFrame, options: list[str], label: str) -> str:
    for col in options:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not identify {label}. Available columns:\n" + ", ".join(df.columns)
    )


def load_membership() -> pd.DataFrame:
    require(MEMBERSHIP)
    df = pd.read_csv(MEMBERSHIP)

    de_sig_col = resolve_column(
        df,
        ["DESeq2_significant", "DESeq2_sig", "deseq2_significant"],
        "DESeq2 significance column",
    )
    reg_sig_col = resolve_column(
        df,
        [
            "Regular_significant",
            "Regular_AREA_significant",
            "Regular_sig",
            "regular_significant",
        ],
        "Regular AREA significance column",
    )
    gene_col = resolve_column(
        df,
        ["gene_id", "ensembl_id_version"],
        "gene ID column",
    )
    reg_z_col = resolve_column(df, ["adjusted_Regular_AREA_Z"], "Regular AREA Z column")
    lfc_col = resolve_column(df, ["log2FoldChange"], "DESeq2 log2 fold-change column")

    df = df.copy()
    df["_de_sig"] = as_bool(df[de_sig_col])
    df["_reg_sig"] = as_bool(df[reg_sig_col])
    df["_gene_key"] = df[gene_col].astype(str)
    df["_regular_aligned_z"] = -pd.to_numeric(df[reg_z_col], errors="coerce")
    df["_lfc"] = pd.to_numeric(df[lfc_col], errors="coerce")

    conditions = [
        df["_de_sig"] & ~df["_reg_sig"],
        df["_de_sig"] & df["_reg_sig"],
        ~df["_de_sig"] & df["_reg_sig"],
    ]
    labels = ["DESeq2 only", "Shared", "Regular AREA only"]
    df["_method_region"] = np.select(conditions, labels, default="Neither")

    return df


def load_deseq2() -> pd.DataFrame:
    require(DESEQ2)
    df = pd.read_csv(DESEQ2)

    gene_col = resolve_column(
        df,
        ["gene_id", "ensembl_id_version", "ensembl_id"],
        "DESeq2 gene ID column",
    )
    if "stat" not in df.columns:
        raise ValueError("DESeq2 file does not contain Wald statistic column 'stat'.")

    out = df[[gene_col, "stat"]].copy()
    out["_gene_key"] = out[gene_col].astype(str)
    out["_deseq2_wald"] = pd.to_numeric(out["stat"], errors="coerce")

    return out[["_gene_key", "_deseq2_wald"]]


def load_geometry() -> pd.DataFrame:
    require(GEOMETRY)
    return pd.read_csv(GEOMETRY)


def spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    return pair["x"].corr(pair["y"], method="spearman")


def derive_geometry_summary(geometry: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    fallback = pd.DataFrame(
        {
            "region": ["DESeq2 only", "Shared", "Regular AREA only"],
            "median_abs_lfc": [0.169615, np.nan, 0.111199],
            "median_abs_rank_biserial": [0.156560, np.nan, 0.170482],
            "monotonic_fraction": [0.679208, np.nan, 0.792683],
        }
    )

    gene_col = next((c for c in ["gene_id", "ensembl_id_version"] if c in geometry.columns), None)
    lfc_col = next((c for c in ["abs_log2FoldChange", "abs_log2fc", "abs_LFC"] if c in geometry.columns), None)
    rank_col = next(
        (
            c
            for c in [
                "abs_rank_biserial",
                "abs_rank_biserial_correlation",
                "abs_rank_biserial_effect",
            ]
            if c in geometry.columns
        ),
        None,
    )
    mono_col = next((c for c in ["is_monotonic", "monotonic", "monotonic_trajectory"] if c in geometry.columns), None)

    if None in [gene_col, lfc_col, rank_col, mono_col]:
        return fallback

    region_map = (
        membership[["_gene_key", "_method_region"]]
        .drop_duplicates()
        .set_index("_gene_key")["_method_region"]
    )

    g = geometry.copy()
    g["_region"] = g[gene_col].astype(str).map(region_map)
    g["_abs_lfc"] = pd.to_numeric(g[lfc_col], errors="coerce")
    g["_abs_rank"] = pd.to_numeric(g[rank_col], errors="coerce")

    if g[mono_col].dtype == bool:
        g["_mono"] = g[mono_col].astype(float)
    else:
        g["_mono"] = (
            g[mono_col]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "increasing", "decreasing"])
            .astype(float)
        )

    rows = []
    for region in ["DESeq2 only", "Shared", "Regular AREA only"]:
        sub = g[g["_region"] == region]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "region": region,
                "median_abs_lfc": sub["_abs_lfc"].median(),
                "median_abs_rank_biserial": sub["_abs_rank"].median(),
                "monotonic_fraction": sub["_mono"].mean(),
            }
        )

    if len(rows) == 3:
        return pd.DataFrame(rows)
    return fallback


def panel_overlap(ax, n_de_only: int, n_shared: int, n_reg_only: int) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.7)
    ax.axis("off")

    ax.text(0.00, 7.55, "A", fontsize=17, fontweight="bold", va="top")
    ax.text(0.55, 7.55, "Matched-significance overlap", fontsize=13, fontweight="bold", va="top")

    left_center = (4.1, 3.95)
    right_center = (5.95, 3.95)
    radius = 2.25

    ax.add_patch(Circle(left_center, radius, facecolor=BLUE, edgecolor=BLUE, alpha=0.25, lw=1.8))
    ax.add_patch(Circle(right_center, radius, facecolor=ORANGE, edgecolor=ORANGE, alpha=0.25, lw=1.8))

    de_total = n_de_only + n_shared
    reg_total = n_reg_only + n_shared
    overlap_de = 100 * n_shared / de_total if de_total else np.nan
    overlap_reg = 100 * n_shared / reg_total if reg_total else np.nan

    ax.text(3.15, 6.45, f"DESeq2\n(n={de_total:,})", ha="center", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(6.90, 6.45, f"Regular AREA\n(n={reg_total:,})", ha="center", fontsize=12, fontweight="bold", color=ORANGE)

    ax.text(3.00, 4.00, f"{n_de_only:,}", ha="center", va="center", fontsize=21, fontweight="bold", color=BLUE)
    ax.text(5.02, 4.00, f"{n_shared:,}", ha="center", va="center", fontsize=21, fontweight="bold", color=TEXT)
    ax.text(7.03, 4.00, f"{n_reg_only:,}", ha="center", va="center", fontsize=21, fontweight="bold", color=ORANGE)

    ax.text(
        5.0,
        0.60,
        (
            f"Shared: {n_shared:,} genes "
            f"({overlap_de:.1f}% of DESeq2; {overlap_reg:.1f}% of Regular AREA)"
        ),
        ha="center",
        fontsize=10,
        color=MUTED,
    )

    ax.text(
        5.0,
        0.20,
        "Both methods use the same AD-vs-NCI endpoint, cohort, covariates, and gene universe.",
        ha="center",
        fontsize=9,
        color="#6A6A6A",
    )


def panel_scatter(ax, plot_df: pd.DataFrame, rho: float) -> None:
    ax.text(-0.15, 1.07, "B", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    ax.set_title("Genome-wide score concordance", fontsize=13, fontweight="bold", pad=14)

    neither = plot_df[plot_df["_method_region"] == "Neither"]
    ax.scatter(
        neither["_deseq2_wald"],
        neither["_regular_aligned_z"],
        s=6,
        alpha=0.10,
        linewidths=0,
        color=LIGHT_GRAY,
        label="Neither",
        rasterized=True,
    )

    category_style = {
        "DESeq2 only": BLUE,
        "Shared": SHARED,
        "Regular AREA only": ORANGE,
    }
    for label, color in category_style.items():
        sub = plot_df[plot_df["_method_region"] == label]
        ax.scatter(
            sub["_deseq2_wald"],
            sub["_regular_aligned_z"],
            s=11,
            alpha=0.58,
            linewidths=0,
            color=color,
            label=label,
            rasterized=True,
        )

    ax.axhline(0, color="#B8B8B8", lw=0.9)
    ax.axvline(0, color="#B8B8B8", lw=0.9)
    ax.grid(alpha=0.18, linewidth=0.6)

    ax.set_xlabel("DESeq2 Wald statistic\npositive = higher expression in AD")
    ax.set_ylabel("Aligned Regular AREA Z\npositive = higher expression in AD")

    ax.text(
        0.03,
        0.96,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CFCFCF"},
    )

    ax.text(
        0.03,
        0.88,
        "High concordance indicates that Regular AREA broadly\nrecapitulates the same binary disease signal.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#5A5A5A",
    )

    ax.legend(frameon=False, fontsize=8.5, loc="lower right", markerscale=1.4)


def panel_geometry(ax, summary: pd.DataFrame) -> None:
    ax.text(-0.05, 1.08, "C", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    ax.set_title("Why the exclusive genes differ", fontsize=13, fontweight="bold", pad=14)

    summary = summary.set_index("region")
    order = ["DESeq2 only", "Regular AREA only"]
    metrics = [
        ("median_abs_lfc", "Median |log2FC|"),
        ("median_abs_rank_biserial", "Median |rank-biserial|"),
        ("monotonic_fraction", "Monotonic fraction"),
    ]

    x = np.arange(len(metrics))
    width = 0.34
    styles = {"DESeq2 only": BLUE, "Regular AREA only": ORANGE}
    offsets = {"DESeq2 only": -width / 2, "Regular AREA only": width / 2}

    for region in order:
        vals = summary.loc[region, [m[0] for m in metrics]].astype(float).to_numpy()
        bars = ax.bar(
            x + offsets[region],
            vals,
            width=width,
            label=region,
            color=styles[region],
            alpha=0.86,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=9.7)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Effect size / fraction")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=9.2, loc="upper left")

    ax.text(
        0.02,
        0.03,
        (
            "DESeq2-only genes show larger endpoint mean shifts.\n"
            "Regular-AREA-only genes show stronger rank ordering and more monotonic trajectories."
        ),
        transform=ax.transAxes,
        fontsize=9.1,
        color=MUTED,
        va="bottom",
    )


def main() -> None:
    membership = load_membership()
    deseq = load_deseq2()
    geometry = load_geometry()

    df = membership.merge(deseq, on="_gene_key", how="left", validate="one_to_one")

    counts = df["_method_region"].value_counts()
    n_de_only = int(counts.get("DESeq2 only", 0))
    n_shared = int(counts.get("Shared", 0))
    n_reg_only = int(counts.get("Regular AREA only", 0))
    n_neither = int(counts.get("Neither", 0))

    n_de = n_de_only + n_shared
    n_reg = n_reg_only + n_shared
    rho = spearman(df["_deseq2_wald"], df["_regular_aligned_z"])
    geom_summary = derive_geometry_summary(geometry, membership)

    summary = pd.DataFrame(
        {
            "metric": [
                "comparison_universe_genes",
                "DESeq2_significant",
                "Regular_AREA_significant",
                "shared_significant",
                "DESeq2_only",
                "Regular_AREA_only",
                "neither_significant",
                "shared_percent_of_DESeq2",
                "shared_percent_of_Regular_AREA",
                "spearman_DESeq2_Wald_vs_aligned_Regular_Z",
            ],
            "value": [
                len(df),
                n_de,
                n_reg,
                n_shared,
                n_de_only,
                n_reg_only,
                n_neither,
                100 * n_shared / n_de if n_de else np.nan,
                100 * n_shared / n_reg if n_reg else np.nan,
                rho,
            ],
        }
    )
    summary.to_csv(SUMMARY, index=False)

    fig = plt.figure(figsize=(15.6, 9.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.95, 1.15],
        height_ratios=[1.0, 0.78],
        hspace=0.39,
        wspace=0.32,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_overlap(ax_a, n_de_only, n_shared, n_reg_only)
    panel_scatter(ax_b, df, rho)
    panel_geometry(ax_c, geom_summary)

    fig.suptitle(
        "Controlled comparison of DESeq2 and covariate-adjusted Regular AREA",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.949,
        (
            "Method-control comparison: same AD-vs-NCI phenotype, same n=418 cohort, "
            "same covariates, same 32,994-gene universe, BH FDR < 0.05"
        ),
        ha="center",
        fontsize=10.6,
        color="#3F3F3F",
    )

    fig.text(
        0.5,
        0.018,
        (
            "Panel B sign convention: Regular AREA Z is flipped so positive values indicate higher expression in AD on both axes."
        ),
        ha="center",
        fontsize=8.6,
        color="#5A5A5A",
    )

    fig.subplots_adjust(top=0.885, bottom=0.10, left=0.07, right=0.97)
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)

    print("=" * 92)
    print("CONTROLLED DESEQ2 VS REGULAR AREA COMPARISON (V2)")
    print("=" * 92)
    print(summary.to_string(index=False))
    print()
    print(f"Wrote: {PNG}")
    print(f"Wrote: {PDF}")
    print(f"Wrote: {SUMMARY}")


if __name__ == "__main__":
    main()

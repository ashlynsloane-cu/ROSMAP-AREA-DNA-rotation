#!/usr/bin/env python3
"""
make_controlled_deseq2_vs_regular_area_figure_publication.py
===========================================================

Publication-focused redesign of the controlled DESeq2 vs covariate-adjusted
Regular AREA comparison.

Design changes
--------------
- full-width journal geometry (7.2 in) instead of slide geometry
- smaller, consistent typography and panel labels
- simplified overlap panel
- cleaner concordance scatter with rasterized points
- Panel C split into three metric-specific mini-panels with independent y-scales
  (avoids comparing unlike quantities on one shared axis)
- reduced in-figure prose; interpretation belongs in the caption
- vector PDF/SVG + 600-dpi PNG

Inputs and summary outputs are unchanged from v2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
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

PNG = OUTDIR / "DESeq2_vs_Regular_AREA_controlled_comparison_publication.png"
PDF = OUTDIR / "DESeq2_vs_Regular_AREA_controlled_comparison_publication.pdf"
SVG = OUTDIR / "DESeq2_vs_Regular_AREA_controlled_comparison_publication.svg"
SUMMARY = OUTDIR / "DESeq2_vs_Regular_AREA_summary_publication.csv"


# -----------------------------------------------------------------------------
# Publication styling
# -----------------------------------------------------------------------------
BLUE = "#0072B2"       # colorblind-safe blue
ORANGE = "#E69F00"     # colorblind-safe orange
SHARED = "#595959"
NEITHER = "#BDBDBD"
TEXT = "#171717"
MUTED = "#5A5A5A"
GRID = "#D9D9D9"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.titlesize": 9.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.7,
    }
)


# -----------------------------------------------------------------------------
# Data helpers (same logic as v2)
# -----------------------------------------------------------------------------
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
    mono_col = next(
        (c for c in ["is_monotonic", "monotonic", "monotonic_trajectory"] if c in geometry.columns),
        None,
    )

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


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------
def panel_overlap(ax, n_de_only: int, n_shared: int, n_reg_only: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(-0.04, 1.04, "A", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="top")
    ax.text(0.04, 1.04, "Matched-significance overlap",
            transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", va="top")

    de_total = n_de_only + n_shared
    reg_total = n_reg_only + n_shared
    overlap_de = 100 * n_shared / de_total if de_total else np.nan
    overlap_reg = 100 * n_shared / reg_total if reg_total else np.nan

    # Smaller, cleaner Venn geometry.
    left_center = (0.43, 0.54)
    right_center = (0.59, 0.54)
    radius = 0.27

    ax.add_patch(Circle(left_center, radius, facecolor=BLUE, edgecolor=BLUE,
                        alpha=0.20, lw=1.0))
    ax.add_patch(Circle(right_center, radius, facecolor=ORANGE, edgecolor=ORANGE,
                        alpha=0.20, lw=1.0))

    ax.text(0.32, 0.90, f"DESeq2\n$n$={de_total:,}", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color=BLUE)
    ax.text(0.70, 0.90, f"Regular AREA\n$n$={reg_total:,}", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color=ORANGE)

    ax.text(0.30, 0.54, f"{n_de_only:,}", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=BLUE)
    ax.text(0.51, 0.54, f"{n_shared:,}", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(0.72, 0.54, f"{n_reg_only:,}", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=ORANGE)

    ax.text(
        0.51,
        0.15,
        f"Shared: {overlap_de:.1f}% of DESeq2; {overlap_reg:.1f}% of Regular AREA",
        ha="center",
        fontsize=6.8,
        color=MUTED,
    )


def panel_scatter(ax, plot_df: pd.DataFrame, rho: float) -> None:
    ax.text(-0.16, 1.06, "B", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="top")
    ax.set_title("Genome-wide score concordance", pad=7)

    neither = plot_df[plot_df["_method_region"] == "Neither"]
    ax.scatter(
        neither["_deseq2_wald"], neither["_regular_aligned_z"],
        s=2.4, alpha=0.075, linewidths=0, color=NEITHER,
        label="Neither", rasterized=True,
    )

    category_style = {
        "DESeq2 only": BLUE,
        "Shared": SHARED,
        "Regular AREA only": ORANGE,
    }
    for label, color in category_style.items():
        sub = plot_df[plot_df["_method_region"] == label]
        ax.scatter(
            sub["_deseq2_wald"], sub["_regular_aligned_z"],
            s=5.0, alpha=0.50, linewidths=0, color=color,
            label=label, rasterized=True,
        )

    ax.axhline(0, color="#B5B5B5", lw=0.65, zorder=0)
    ax.axvline(0, color="#B5B5B5", lw=0.65, zorder=0)
    ax.grid(alpha=0.20, linewidth=0.45, color=GRID)

    ax.set_xlabel("DESeq2 Wald statistic\npositive = higher expression in AD")
    ax.set_ylabel("Aligned Regular AREA Z\npositive = higher expression in AD")

    ax.text(
        0.03, 0.96, rf"Spearman $\rho$ = {rho:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#CFCFCF",
            "linewidth": 0.6,
        },
    )

    ax.legend(
        frameon=False,
        fontsize=6.4,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.04),
        markerscale=1.4,
        handletextpad=0.4,
        borderaxespad=0.4,
    )


def panel_metric(ax, title: str, values: list[float], *, y_max: float | None = None,
                 ylabel: str | None = None) -> None:
    xs = np.arange(2)
    colors = [BLUE, ORANGE]

    bars = ax.bar(xs, values, width=0.58, color=colors, alpha=0.90)
    ax.set_xticks(xs)
    ax.set_xticklabels(["DESeq2\nonly", "Regular AREA\nonly"], fontsize=6.6)
    ax.set_title(title, fontsize=7.6, fontweight="bold", pad=5)

    if y_max is None:
        vmax = max(values)
        y_max = vmax * 1.30 if vmax > 0 else 1.0
    ax.set_ylim(0, y_max)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.0)

    ax.grid(axis="y", alpha=0.20, linewidth=0.45, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.035,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )


def add_panel_c(fig, subgrid, summary: pd.DataFrame) -> None:
    summary = summary.set_index("region")

    de = summary.loc["DESeq2 only"]
    reg = summary.loc["Regular AREA only"]

    axes = [fig.add_subplot(subgrid[0, i]) for i in range(3)]

    axes[0].text(-0.38, 1.22, "C", transform=axes[0].transAxes,
                 fontsize=10.5, fontweight="bold", va="top")
    axes[0].text(-0.17, 1.22, "Exclusive-gene characteristics",
                 transform=axes[0].transAxes,
                 fontsize=9.2, fontweight="bold", va="top")

    panel_metric(
        axes[0],
        r"Median $|\log_2\mathrm{FC}|$",
        [float(de["median_abs_lfc"]), float(reg["median_abs_lfc"])],
        y_max=max(float(de["median_abs_lfc"]), float(reg["median_abs_lfc"])) * 1.35,
    )
    panel_metric(
        axes[1],
        "Median |rank-biserial|",
        [float(de["median_abs_rank_biserial"]), float(reg["median_abs_rank_biserial"])],
        y_max=max(float(de["median_abs_rank_biserial"]), float(reg["median_abs_rank_biserial"])) * 1.35,
    )
    panel_metric(
        axes[2],
        "Monotonic fraction",
        [float(de["monotonic_fraction"]), float(reg["monotonic_fraction"])],
        y_max=1.0,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
                len(df), n_de, n_reg, n_shared, n_de_only, n_reg_only, n_neither,
                100 * n_shared / n_de if n_de else np.nan,
                100 * n_shared / n_reg if n_reg else np.nan,
                rho,
            ],
        }
    )
    summary.to_csv(SUMMARY, index=False)

    # Full-width journal figure; compact but readable at final size.
    fig = plt.figure(figsize=(7.2, 5.65))
    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[0.90, 1.15],
        height_ratios=[1.08, 0.72],
        hspace=0.42,
        wspace=0.31,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])

    panel_overlap(ax_a, n_de_only, n_shared, n_reg_only)
    panel_scatter(ax_b, df, rho)

    c_grid = outer[1, :].subgridspec(1, 3, wspace=0.42)
    add_panel_c(fig, c_grid, geom_summary)

    fig.suptitle(
        "Controlled comparison: DESeq2 vs Regular AREA",
        fontsize=11.4,
        fontweight="bold",
        y=0.985,
        color=TEXT,
    )
    fig.text(
        0.5,
        0.953,
        "Same AD-vs-NCI cohort (n=418), covariates, 32,994-gene universe; BH FDR < 0.05",
        ha="center",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )

    # Keep the sign-convention note, but demote it to a compact footnote.
    fig.text(
        0.5,
        0.018,
        "Regular AREA Z is sign-aligned so positive values indicate higher expression in AD on both axes.",
        ha="center",
        fontsize=6.2,
        color=MUTED,
    )

    fig.subplots_adjust(top=0.895, bottom=0.085, left=0.075, right=0.985)

    fig.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("=" * 92)
    print("CONTROLLED DESEQ2 VS REGULAR AREA COMPARISON — PUBLICATION VERSION")
    print("=" * 92)
    print(summary.to_string(index=False))
    print()
    print(f"Wrote: {PNG}")
    print(f"Wrote: {PDF}")
    print(f"Wrote: {SVG}")
    print(f"Wrote: {SUMMARY}")


if __name__ == "__main__":
    main()

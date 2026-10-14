#!/usr/bin/env python3
"""
Figure 1 — Cross-phenotype recurrence of molecular signals.

Canonical formatting revision of the validated cross-phenotype recurrence figure.

Target publication canvas
-------------------------
    Width:  170 mm
    Height: 140 mm
    DPI:    600 for PNG preview

The height is chosen to match the compact full-width scale used by the
Hsp60/10 publication figures (Figure 1 = 170 x 140 mm). The manuscript
pipeline uses 170 mm full-width figures and a 225 mm maximum height.

Scientific question
-------------------
Do the same molecular features recur across independent clinical and
neuropathological measures of Alzheimer's disease more often than expected
by chance?

Panel logic
-----------
A. Three independent phenotype axes and the definition of recurrence.
B. Primary quantitative result: observed recurrence versus empirical null,
   partitioned into exactly 2 axes and all 3 axes.
C. Directional coherence of recurrent features.
D. >=2-axis recurrence enrichment for AREA and conventional analysis.

Important publication-style choices
-----------------------------------
* No figure-level title/subtitle: the caption carries the figure title.
* Compact panel headings and panel letters.
* Manuscript-scale typography.
* Minimal classic axes; no top/right spines.
* Restrained AREA/conventional palette.
* No decorative hatching.
* Fixed physical canvas; never use bbox_inches='tight'.
* PDF/SVG are publication outputs; PNG is a 600-dpi preview.

The script reads the frozen Stage 7 empirical-null table and does not
recompute the analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch


# ---------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------

FIGURE_WIDTH_MM = 170
FIGURE_HEIGHT_MM = 140
PNG_DPI = 600

INK = "#22272B"
MID = "#69747D"
LIGHT = "#D5DADF"
GRID = "#E7EAED"

AREA_BLUE = "#7A8FA8"
AREA_DARK = "#4D6278"
CONVENTIONAL_GRAY = "#B8C0CA"
NULL_FILL = "#F3F5F6"

FONT = "DejaVu Sans"

MODALITIES = ["monocyte", "dlpfc", "tmt"]
MODALITY_LABELS = {
    "monocyte": "Monocyte RNA",
    "dlpfc": "DLPFC RNA",
    "tmt": "DLPFC TMT",
}

METRICS = {
    "ge2_axes",
    "all3_axes",
    "ge2_axes_same_direction",
}


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_stage7(repo_root: Path) -> pd.DataFrame:
    path = (
        repo_root
        / "results"
        / "stage7_pathology_integration"
        / "empirical_null"
        / "stage7_pathology_integration_empirical_null.tsv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Stage 7 table not found: {path}")

    df = pd.read_csv(path, sep="\t")

    required = {
        "modality",
        "framework",
        "top_k",
        "metric",
        "observed",
        "null_mean",
        "empirical_fdr",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Stage 7 table is missing columns: {sorted(missing)}"
        )

    df = df[
        (df["top_k"] == 500)
        & df["metric"].isin(METRICS)
        & df["modality"].isin(MODALITIES)
        & df["framework"].isin({"AREA", "CONVENTIONAL"})
    ].copy()

    expected = {
        (modality, framework, metric)
        for modality in MODALITIES
        for framework in ("AREA", "CONVENTIONAL")
        for metric in METRICS
    }
    observed = {
        (row.modality, row.framework, row.metric)
        for row in df.itertuples(index=False)
    }

    if len(df) != 18 or observed != expected:
        raise ValueError(
            "Expected exactly 18 top-500 modality/framework/metric "
            "combinations in the Stage 7 empirical-null table."
        )

    return df


def get_row(
    df: pd.DataFrame,
    modality: str,
    framework: str,
    metric: str,
) -> pd.Series:
    match = df[
        (df["modality"] == modality)
        & (df["framework"] == framework)
        & (df["metric"] == metric)
    ]

    if len(match) != 1:
        raise ValueError(
            f"Expected one row for {modality}/{framework}/{metric}; "
            f"found {len(match)}."
        )

    return match.iloc[0]


# ---------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------


def style_axis(ax: plt.Axes) -> None:
    """Compact manuscript-style axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.tick_params(
        axis="both",
        which="both",
        length=2.5,
        width=0.55,
        color=INK,
        labelcolor=INK,
        pad=2,
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.075,
        1.02,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def panel_heading(
    ax: plt.Axes,
    title: str,
    subtitle: str | None = None,
) -> None:
    ax.text(
        0.0,
        1.015,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )

    if subtitle:
        ax.text(
            0.0,
            0.965,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            color=MID,
            clip_on=False,
        )


# ---------------------------------------------------------------------
# Panel A — conceptual setup
# ---------------------------------------------------------------------


def panel_a(ax: plt.Axes) -> None:
    ax.axis("off")
    panel_letter(ax, "A")

    panel_heading(
        ax,
        "Phenotype axes",
    )

    boxes = [
        (0.03, "Clinical\ndiagnosis", "cogdx"),
        (0.37, "Tau\npathology", "Braak"),
        (0.71, "Amyloid\npathology", "CERAD"),
    ]

    for x, title, code in boxes:
        box = FancyBboxPatch(
            (x, 0.55),
            0.24,
            0.26,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor="white",
            edgecolor=MID,
            linewidth=0.60,
            transform=ax.transAxes,
        )
        ax.add_patch(box)

        ax.text(
            x + 0.12,
            0.690,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=INK,
            linespacing=0.92,
        )
        ax.text(
            x + 0.12,
            0.595,
            code,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.0,
            color=MID,
        )

    # Subtle connectors: the axes are independent, but feed the same
    # recurrence definition.
    for x1, x2 in [(0.27, 0.37), (0.61, 0.71)]:
        ax.plot(
            [x1, x2],
            [0.68, 0.68],
            color=LIGHT,
            linewidth=0.75,
            transform=ax.transAxes,
        )

    ax.text(
        0.50,
        0.34,
        "Recurrence = same feature detected on ≥2 axes",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.50,
        0.23,
        "All 3 axes = full phenotype set",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.0,
        color=MID,
    )


# ---------------------------------------------------------------------
# Panel B — primary recurrence result
# ---------------------------------------------------------------------


def draw_recurrence_bar(
    ax: plt.Axes,
    y: float,
    exact_two: float,
    all_three: float,
    total: float,
    observed: bool,
) -> None:
    """Draw exactly-2 and all-3 recurrence as a compact stacked bar."""

    if observed:
        first_color = AREA_BLUE
        second_color = AREA_DARK
        edge = AREA_DARK
        alpha = 1.0
    else:
        first_color = NULL_FILL
        second_color = NULL_FILL
        edge = CONVENTIONAL_GRAY
        alpha = 1.0

    ax.barh(
        y,
        exact_two,
        left=0,
        height=0.14,
        color=first_color,
        edgecolor=edge,
        linewidth=0.45,
        alpha=alpha,
    )

    ax.barh(
        y,
        all_three,
        left=exact_two,
        height=0.14,
        color=second_color,
        edgecolor=edge,
        linewidth=0.45,
        alpha=alpha,
    )

    ax.text(
        total + 7,
        y,
        f"{int(round(total))}",
        ha="left",
        va="center",
        fontsize=4.7,
        color=INK if observed else MID,
    )

    # Label the stricter all-3 subset only on observed bars. The null
    # is intentionally represented by a single total to keep the panel
    # readable at manuscript scale.
    if observed and all_three >= 30:
        ax.text(
            exact_two + all_three / 2,
            y,
            f"{int(round(all_three))}",
            ha="center", va="center", fontsize=4.5,
            color="white", zorder=4,
        )
    elif observed and all_three > 0:
        # Narrow segments cannot legibly accommodate an interior label.
        ax.annotate(
            f"{int(round(all_three))}",
            xy=(exact_two + all_three / 2, y + 0.07),
            xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=4.6, color=AREA_DARK,
            clip_on=False,
        )


def panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    panel_letter(ax, "B")

    panel_heading(
        ax,
        "Cross-phenotype recurrence",
    )

    y_positions = [1.65, 0.825, 0.0]

    for modality, y in zip(MODALITIES, y_positions):
        ge2 = get_row(df, modality, "AREA", "ge2_axes")
        all3 = get_row(df, modality, "AREA", "all3_axes")

        observed_exact_two = (
            ge2["observed"] - all3["observed"]
        )
        null_exact_two = (
            ge2["null_mean"] - all3["null_mean"]
        )

        draw_recurrence_bar(
            ax,
            y + 0.16,
            observed_exact_two,
            all3["observed"],
            ge2["observed"],
            observed=True,
        )
        draw_recurrence_bar(
            ax,
            y - 0.16,
            null_exact_two,
            all3["null_mean"],
            ge2["null_mean"],
            observed=False,
        )

        ax.text(
            -0.012,
            y + 0.16,
            "Observed",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=4.5,
            color=INK,
        )
        ax.text(
            -0.012,
            y - 0.16,
            "Null",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=4.5,
            color=MID,
        )
        ax.text(
            -0.012,
            y,
            MODALITY_LABELS[modality],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=INK,
        )

        # The empirical-null total is labeled at the bar end.
        # Suppress tiny segment labels that overlap bar endpoints.

    ax.set_xlim(0, 610)
    ax.set_ylim(-0.32, 2.18)
    ax.set_yticks([])
    ax.set_xlabel(
        "Number of top-500 features",
        fontsize=6.8,
        fontweight="bold",
        labelpad=3,
    )
    ax.tick_params(axis="x", labelsize=6.2)

    for y in [0.5, 1.5]:
        ax.axhline(y, color=GRID, linewidth=0.55, zorder=0)

    # Compact in-panel key.
    ax.text(
        0.995,
        0.985,
        "exactly 2 axes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.7,
        color=AREA_BLUE,
        fontweight="bold",
    )
    ax.text(
        0.995,
        0.952,
        "all 3 axes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.7,
        color=AREA_DARK,
        fontweight="bold",
    )


    style_axis(ax)


# ---------------------------------------------------------------------
# Panel C — directional coherence
# ---------------------------------------------------------------------


def panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    panel_letter(ax, "C")

    panel_heading(
        ax,
        "Directional coherence",
    )

    y_positions = [1.35, 0.675, 0.0]

    for modality, y in zip(MODALITIES, y_positions):
        ge2 = get_row(df, modality, "AREA", "ge2_axes")
        same = get_row(
            df,
            modality,
            "AREA",
            "ge2_axes_same_direction",
        )

        observed_pct = (
            100 * same["observed"] / ge2["observed"]
        )
        null_pct = (
            100 * same["null_mean"] / ge2["null_mean"]
        )

        ax.plot(
            [null_pct, observed_pct],
            [y, y],
            color=MID,
            linewidth=0.9,
            zorder=1,
        )
        ax.scatter(
            null_pct,
            y,
            marker="x",
            s=29,
            linewidths=1.15,
            color=CONVENTIONAL_GRAY,
            zorder=3,
        )
        ax.scatter(
            observed_pct,
            y,
            marker="o",
            s=28,
            color=AREA_BLUE,
            zorder=3,
        )

        ax.text(
            -0.015,
            y,
            MODALITY_LABELS[modality],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.5,
            color=INK,
        )

        ax.text(
            null_pct,
            y + 0.085,
            f"{null_pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=4.7,
            color=MID,
        )
        ax.text(
            observed_pct - 0.7,
            y - 0.085,
            f"{observed_pct:.0f}%",
            ha="right",
            va="top",
            fontsize=4.9,
            color=INK,
        )

    ax.set_xlim(42, 108)
    ax.set_ylim(-0.28, 1.62)
    ax.set_yticks([])
    ax.set_xlabel(
        "Same disease-oriented direction (%)",
        fontsize=5.9,
        fontweight="bold",
        labelpad=3,
    )
    ax.tick_params(axis="x", labelsize=6.1)

    ax.text(
        0.995,
        0.955,
        "● observed   × null",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.8,
        color=MID,
    )


    # Light horizontal guides, matching the restrained manuscript style.
    for y in y_positions:
        ax.axhline(y, color=GRID, linewidth=0.45, zorder=0)

    style_axis(ax)


# ---------------------------------------------------------------------
# Panel D — method comparison
# ---------------------------------------------------------------------


def panel_d(ax: plt.Axes, df: pd.DataFrame) -> None:
    panel_letter(ax, "D")

    panel_heading(
        ax,
        "Method comparison",
    )

    y_positions = [1.35, 0.675, 0.0]

    for modality, y in zip(MODALITIES, y_positions):
        area = get_row(
            df,
            modality,
            "AREA",
            "ge2_axes",
        )
        conventional = get_row(
            df,
            modality,
            "CONVENTIONAL",
            "ge2_axes",
        )

        area_fold = (
            area["observed"] / area["null_mean"]
        )
        conventional_fold = (
            conventional["observed"]
            / conventional["null_mean"]
        )

        ax.plot(
            [area_fold, conventional_fold],
            [y, y],
            color=LIGHT,
            linewidth=0.8,
            zorder=1,
        )

        ax.text(
            -0.015,
            y,
            MODALITY_LABELS[modality],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.5,
            color=INK,
        )

        # When the two methods are identical, separate the markers vertically
        # rather than pretending there are two distinct x positions.
        if abs(area_fold - conventional_fold) < 0.04:
            ax.scatter(
                area_fold,
                y + 0.055,
                marker="o",
                s=29,
                color=AREA_BLUE,
                zorder=3,
            )
            ax.scatter(
                conventional_fold,
                y - 0.055,
                marker="s",
                s=27,
                facecolor="white",
                edgecolor=CONVENTIONAL_GRAY,
                linewidth=0.9,
                zorder=3,
            )
            ax.text(
                area_fold,
                y + 0.17,
                f"{area_fold:.1f}× both",
                ha="center",
                va="bottom",
                fontsize=4.7,
                color=INK,
            )
        else:
            ax.scatter(
                area_fold,
                y,
                marker="o",
                s=29,
                color=AREA_BLUE,
                zorder=3,
            )
            ax.scatter(
                conventional_fold,
                y,
                marker="s",
                s=27,
                facecolor="white",
                edgecolor=CONVENTIONAL_GRAY,
                linewidth=0.9,
                zorder=3,
            )
            ax.text(
                area_fold,
                y + 0.14,
                f"{area_fold:.1f}×",
                ha="center",
                va="bottom",
                fontsize=4.7,
                color=INK,
            )
            ax.text(
                conventional_fold,
                y - 0.14,
                f"{conventional_fold:.1f}×",
                ha="center",
                va="top",
                fontsize=4.7,
                color=INK,
            )


    ax.axvline(
        1,
        color=MID,
        linewidth=0.65,
        linestyle="--",
        zorder=0,
    )
    ax.set_xlim(0.72, 4.40)
    ax.set_ylim(-0.28, 1.62)
    ax.set_yticks([])
    ax.set_xlabel(
        "Observed / empirical-null mean",
        fontsize=5.9,
        fontweight="bold",
        labelpad=3,
    )
    ax.tick_params(axis="x", labelsize=6.1)

    ax.text(
        0.995,
        0.955,
        "● AREA   □ DESeq2 / limma",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.8,
        color=MID,
    )


    for y in y_positions:
        ax.axhline(y, color=GRID, linewidth=0.45, zorder=0)

    style_axis(ax)


# ---------------------------------------------------------------------
# Caption / audit
# ---------------------------------------------------------------------


def make_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for modality in MODALITIES:
        for framework in ["AREA", "CONVENTIONAL"]:
            ge2 = get_row(df, modality, framework, "ge2_axes")
            all3 = get_row(df, modality, framework, "all3_axes")
            same = get_row(
                df,
                modality,
                framework,
                "ge2_axes_same_direction",
            )

            rows.append(
                {
                    "modality": modality,
                    "framework": framework,
                    "observed_exactly_2_axes": (
                        ge2["observed"] - all3["observed"]
                    ),
                    "null_exactly_2_axes": (
                        ge2["null_mean"] - all3["null_mean"]
                    ),
                    "observed_all_3_axes": all3["observed"],
                    "null_all_3_axes": all3["null_mean"],
                    "observed_same_direction_fraction": (
                        same["observed"] / ge2["observed"]
                    ),
                    "null_same_direction_fraction": (
                        same["null_mean"] / ge2["null_mean"]
                    ),
                    "ge2_fold_enrichment": (
                        ge2["observed"] / ge2["null_mean"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def write_caption(output_dir: Path) -> None:
    caption = """Figure 1. Molecular signals recur across independent measures
of Alzheimer's disease. (A) Molecular features were ranked separately across
clinical diagnosis, tau pathology, and amyloid pathology. Cross-phenotype
recurrence was defined as the same top-500 feature appearing on at least two
phenotype axes; recurrence across all three axes represents the stricter
intersection. (B) Observed recurrence is compared with the empirical
permutation-null mean. Bars partition ≥2-axis recurrence into features
appearing on exactly two axes and features appearing on all three. (C) Among
≥2-axis recurrent features, the proportion showing the same disease-oriented
direction is compared with the permutation-null expectation. (D) ≥2-axis
recurrence enrichment is shown for AREA and conventional differential
analysis frameworks. Values are based on the frozen Stage 7 empirical-null
results and were not recomputed for figure generation.
"""
    (output_dir / "Figure1_caption.txt").write_text(caption)


# ---------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------


def make_figure(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(
        figsize=(
            FIGURE_WIDTH_MM / 25.4,
            FIGURE_HEIGHT_MM / 25.4,
        ),
        facecolor="white",
    )

    # Fixed manuscript canvas. Do not use bbox_inches='tight'.
    grid = fig.add_gridspec(
        2,
        2,
        left=0.135,
        right=0.975,
        top=0.935,
        bottom=0.11,
        wspace=0.49,
        hspace=0.35,
        width_ratios=[0.95, 1.05],
        height_ratios=[0.88, 1.0],
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    panel_a(ax_a)
    panel_b(ax_b, df)
    panel_c(ax_c, df)
    panel_d(ax_d, df)

    return fig


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            repo_root
            / "results"
            / "figures"
            / "figure1_phenotype_integration"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_stage7(repo_root)

    fig = make_figure(df)

    stem = "Figure1_cross_phenotype_recurrence_manuscript"

    # Publication exports. No bbox_inches='tight':
    # physical canvas remains exactly 170 x 140 mm.
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches=None,
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches=None,
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=PNG_DPI,
        bbox_inches=None,
        facecolor="white",
    )

    audit = make_audit(df)
    audit.to_csv(
        output_dir / f"{stem}_audit.csv",
        index=False,
    )
    write_caption(output_dir)

    plt.close(fig)

    print(
        f"Figure canvas: "
        f"{FIGURE_WIDTH_MM} x {FIGURE_HEIGHT_MM} mm"
    )
    print(
        f"PDF: {output_dir / (stem + '.pdf')}"
    )
    print(
        f"SVG: {output_dir / (stem + '.svg')}"
    )
    print(
        f"PNG: {output_dir / (stem + '.png')}"
    )
    print(
        f"Audit: {output_dir / (stem + '_audit.csv')}"
    )
    print(
        f"Caption: {output_dir / 'Figure1_caption.txt'}"
    )


if __name__ == "__main__":
    main()

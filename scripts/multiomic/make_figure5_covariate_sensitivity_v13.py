#!/usr/bin/env python3
"""
Figure 5 — Covariate-adjustment sensitivity and attribution.

Scientific question
-------------------
Does covariate adjustment affect AREA more than it affects matched
conventional differential-expression methods, and which covariates drive the
AREA ranking transformation?

Figure design
-------------
A. Method comparison:
   Direct AREA-versus-conventional comparison of adjustment-induced rank
   reorganization. Each point is one phenotype contrast. The 1:1 line means
   identical sensitivity; distance from the origin shows the magnitude of
   ranking reorganization.

B. Covariate attribution:
   Mean exact Shapley-allocated normalized rank distance across six phenotype
   contrasts. Segment length is the absolute contribution of each covariate
   to the B-to-A ranking transformation; total bar length is the mean total
   ranking reorganization.

C. Practical feature turnover:
   Fraction of the unadjusted top 500 retained after adjustment, shown for
   AREA and the matched conventional method.

Frozen sources
--------------
Stage 11 conventional comparison:
    results/stage11_conventional_covariate_sensitivity/comparison/
        AREA_VS_CONVENTIONAL_ADJUSTMENT_SENSITIVITY.tsv

Stage 11 covariate attribution:
    results/stage11_covariate_attribution_analysis/
        ABSOLUTE_SHAPLEY_MODALITY_SUMMARY.tsv

The script does not recompute statistical analyses. It validates the frozen
inputs and then renders the manuscript figure.

Interpretation guardrail
------------------------
Shapley allocations describe contributions to the statistical-model
transformation from intercept-only AREA to fully adjusted AREA. They are not
causal effects, percentages of biological signal, or percentages of causal
confounding.

Output
------
170 × 140 mm
PDF, SVG, and 600-dpi PNG
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIG_WIDTH_MM = 170
FIG_HEIGHT_MM = 140
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FIG_HEIGHT_IN = FIG_HEIGHT_MM / 25.4
PNG_DPI = 600

INK = "#283038"
MID = "#68747F"
GRID = "#E2E6EA"
AREA = "#718DAA"
CONVENTIONAL = "#B9C1C9"

COVARIATE_COLORS = {
    "sequencing_batch": "#5B718A",
    "batch": "#5B718A",
    "age": "#9C7A56",
    "rin": "#7E9B73",
    "sex": "#9A829D",
    "pmi": "#9B9B9B",
}

COVARIATE_LABELS = {
    "sequencing_batch": "Batch",
    "batch": "Batch",
    "age": "Age",
    "rin": "RIN",
    "sex": "Sex",
    "pmi": "PMI",
}

CONTRAST_ORDER = [
    "cogdx_NCI_vs_AD",
    "cogdx_NCI_vs_MCI",
    "cogdx_MCI_vs_AD",
    "Braak_0-2_vs_3-4",
    "Braak_3-4_vs_5-6",
    "CERAD_3-4_vs_1-2",
]

CONTRAST_LABELS = {
    "cogdx_NCI_vs_AD": "NCI–AD",
    "cogdx_NCI_vs_MCI": "NCI–MCI",
    "cogdx_MCI_vs_AD": "MCI–AD",
    "Braak_0-2_vs_3-4": "Braak 0–2\nvs 3–4",
    "Braak_3-4_vs_5-6": "Braak 3–4\nvs 5–6",
    "CERAD_3-4_vs_1-2": "CERAD 3–4\nvs 1–2",
}

MODALITY_ORDER = ["dlpfc", "tmt"]
MODALITY_LABELS = {
    "dlpfc": "DLPFC RNA",
    "tmt": "DLPFC TMT",
}
EXPECTED_METHODS = {
    "dlpfc": "DESeq2",
    "tmt": "limma",
}

EXPECTED_MEDIANS = {
    "dlpfc": {
        "AREA_median_normalized_rank_shift": 0.1674,
        "CONVENTIONAL_median_normalized_rank_shift": 0.1669,
        "AREA_TOP500_overlap_fraction": 0.506,
        "CONVENTIONAL_TOP500_overlap_fraction": 0.472,
    },
    "tmt": {
        "AREA_median_normalized_rank_shift": 0.0792,
        "CONVENTIONAL_median_normalized_rank_shift": 0.0882,
        "AREA_TOP500_overlap_fraction": 0.786,
        "CONVENTIONAL_TOP500_overlap_fraction": 0.739,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render manuscript Figure 5 from frozen Stage 11 outputs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="ROSMAP-AREA-DNA-rotation repository root.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Optional explicit Stage 11 conventional-comparison directory.",
    )
    parser.add_argument(
        "--attribution-dir",
        type=Path,
        default=None,
        help="Optional explicit Stage 11 covariate-attribution directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory.",
    )
    return parser.parse_args()


def locate_dir(
    repo_root: Path,
    explicit: Path | None,
    relative_candidates: list[Path],
    label: str,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    candidates = [
        repo_root / relative_path for relative_path in relative_candidates
    ]
    candidates.append(
        repo_root / "data/frozen_inputs" / relative_candidates[0]
    )

    for path in candidates:
        path = path.expanduser().resolve()
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Could not locate the frozen Stage 11 {label} directory. "
        "Tried:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def load_conventional(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {
        "task_id",
        "contrast_id",
        "modality",
        "method",
        "AREA_median_normalized_rank_shift",
        "CONVENTIONAL_median_normalized_rank_shift",
        "AREA_TOP500_overlap_fraction",
        "CONVENTIONAL_TOP500_overlap_fraction",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Conventional comparison file is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df[df["modality"].isin(MODALITY_ORDER)].copy()

    if len(df) != 12:
        raise ValueError(
            "Expected exactly 12 matched rows (6 DLPFC + 6 TMT); "
            f"found {len(df)}."
        )

    if df[["modality", "contrast_id"]].duplicated().any():
        raise ValueError(
            "Duplicate modality/contrast rows found in the frozen "
            "conventional comparison."
        )

    if set(df["contrast_id"]) != set(CONTRAST_ORDER):
        raise ValueError(
            "Frozen conventional comparison does not contain exactly the "
            "six expected phenotype contrasts."
        )

    for modality, expected_method in EXPECTED_METHODS.items():
        observed = set(df.loc[df["modality"] == modality, "method"])
        if observed != {expected_method}:
            raise ValueError(
                f"Expected {expected_method} for {modality}; "
                f"found {sorted(observed)}."
            )

    return df


def load_attribution(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {
        "modality",
        "group",
        "mean_shapley",
        "mean_absolute_contribution",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Shapley modality summary is missing required columns: "
            f"{sorted(missing)}"
        )

    expected_modalities = {"dlpfc", "monocyte", "tmt"}
    observed_modalities = set(df["modality"])
    if observed_modalities != expected_modalities:
        raise ValueError(
            "Expected DLPFC, monocyte, and TMT attribution summaries; "
            f"found {sorted(observed_modalities)}."
        )

    return df.copy()


def validate_frozen_values(df: pd.DataFrame) -> None:
    for modality in MODALITY_ORDER:
        subset = df[df["modality"] == modality]
        area_median = subset["AREA_median_normalized_rank_shift"].median()
        conventional_median = subset[
            "CONVENTIONAL_median_normalized_rank_shift"
        ].median()
        area_top500 = subset["AREA_TOP500_overlap_fraction"].median()
        conventional_top500 = subset[
            "CONVENTIONAL_TOP500_overlap_fraction"
        ].median()

        expected = EXPECTED_MEDIANS[modality]
        checks = {
            "AREA rank shift": (
                area_median,
                expected["AREA_median_normalized_rank_shift"],
            ),
            "conventional rank shift": (
                conventional_median,
                expected["CONVENTIONAL_median_normalized_rank_shift"],
            ),
            "AREA top-500": (
                area_top500,
                expected["AREA_TOP500_overlap_fraction"],
            ),
            "conventional top-500": (
                conventional_top500,
                expected["CONVENTIONAL_TOP500_overlap_fraction"],
            ),
        }

        for label, (observed, target) in checks.items():
            if not np.isclose(observed, target, atol=5e-4):
                raise ValueError(
                    f"Frozen-value guardrail failed for {modality} "
                    f"{label}: observed {observed}, expected {target}."
                )


def style_axis(ax: plt.Axes) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(
        axis="x",
        labelsize=5.9,
        length=2,
        width=0.45,
        color=INK,
        labelcolor=INK,
        pad=3,
    )
    ax.tick_params(
        axis="y",
        labelsize=6.1,
        length=2,
        width=0.45,
        color=INK,
        labelcolor=INK,
        pad=3,
    )


def draw_method_agreement(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    """Show whether AREA and conventional methods reorganize rankings similarly."""

    d = df.copy()
    x = d["CONVENTIONAL_median_normalized_rank_shift"].to_numpy()
    y = d["AREA_median_normalized_rank_shift"].to_numpy()

    r = np.corrcoef(x, y)[0, 1]
    r_squared = r**2

    # Same scale on both axes: distance from the origin is the magnitude of
    # adjustment-induced rank reorganization; proximity to the 1:1 line is
    # method agreement.
    lim = 0.26
    ax.plot(
        [0, lim],
        [0, lim],
        linestyle="--",
        linewidth=0.8,
        color=MID,
        zorder=1,
    )

    modality_specs = [
        ("dlpfc", "DLPFC RNA", "o"),
        ("tmt", "DLPFC TMT", "s"),
    ]

    for modality, label, marker in modality_specs:
        sub = d[d["modality"] == modality]
        ax.scatter(
            sub["CONVENTIONAL_median_normalized_rank_shift"],
            sub["AREA_median_normalized_rank_shift"],
            s=27,
            marker=marker,
            color=AREA if modality == "dlpfc" else CONVENTIONAL,
            edgecolor=INK,
            linewidth=0.4,
            zorder=3,
            label=label,
        )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("auto")
    ticks = np.arange(0, 0.251, 0.05)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{100 * t:.0f}" for t in ticks])
    ax.set_yticklabels([f"{100 * t:.0f}" for t in ticks])
    ax.set_xlabel(
        "Conventional rank reorganization (%)",
        fontsize=6.4,
        fontweight="bold",
    )
    ax.set_ylabel(
        "AREA rank reorganization (%)",
        fontsize=6.4,
        fontweight="bold",
    )
    ax.grid(color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    style_axis(ax)

    ax.text(
        0.02,
        0.96,
        f"1:1 = identical sensitivity   R² = {r_squared:.3f}  (n = {len(d)} matched contrasts)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color=INK,
    )

    ax.legend(
        frameon=False,
        fontsize=5.6,
        loc="lower right",
        handletextpad=0.35,
        borderpad=0.1,
    )


def draw_shapley(
    ax: plt.Axes,
    attribution: pd.DataFrame,
) -> None:
    modalities = ["dlpfc", "monocyte", "tmt"]
    labels = ["DLPFC RNA", "Monocyte RNA", "DLPFC TMT"]
    y = np.arange(len(modalities))
    left = np.zeros(len(modalities))

    covariates = [
        "sequencing_batch",
        "batch",
        "age",
        "rin",
        "sex",
        "pmi",
    ]

    for covariate in covariates:
        values = np.array(
            [
                attribution.loc[
                    (attribution["modality"] == modality)
                    & (attribution["group"] == covariate),
                    "mean_absolute_contribution",
                ].sum()
                for modality in modalities
            ]
        )

        ax.barh(
            y,
            values,
            left=left,
            height=0.46,
            color=COVARIATE_COLORS[covariate],
            edgecolor="white",
            linewidth=0.5,
        )

        for i, value in enumerate(values):
            if value >= 0.020:
                ax.text(
                    left[i] + value / 2,
                    y[i],
                    COVARIATE_LABELS[covariate],
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    fontweight="bold",
                    color="white",
                )

        left += values

    for i, modality in enumerate(modalities):
        total = left[i]
        ax.text(
            total + 0.004,
            y[i],
            f"{100 * total:.1f}%",
            va="center",
            ha="left",
            fontsize=5.9,
            color=INK,
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.255)
    b_ticks = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
    ax.set_xticks(b_ticks)
    ax.set_xticklabels([f"{100 * t:.0f}" for t in b_ticks])
    ax.set_xlabel(
        "Mean Shapley-allocated rank reorganization (%)",
        fontsize=6.1,
        fontweight="bold",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    style_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COVARIATE_COLORS[key],
            markeredgecolor=COVARIATE_COLORS[key],
            markersize=4.5,
            label=label,
        )
        for key, label in [
            ("sequencing_batch", "Batch"),
            ("age", "Age"),
            ("rin", "RIN"),
            ("sex", "Sex"),
            ("pmi", "PMI"),
        ]
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=5.3,
        loc="upper left",
        bbox_to_anchor=(0, -0.20),
        ncol=3,
        handletextpad=0.25,
        columnspacing=0.7,
    )


def draw_top500(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    modalities = ["dlpfc", "tmt"]
    labels = ["DLPFC RNA", "DLPFC TMT"]
    x = np.arange(2)

    for i, modality in enumerate(modalities):
        subset = df[df["modality"] == modality]
        area = subset["AREA_TOP500_overlap_fraction"].median() * 100
        conventional = (
            subset["CONVENTIONAL_TOP500_overlap_fraction"].median() * 100
        )

        ax.plot(
            [i, i],
            [conventional, area],
            color="#CDD3D8",
            linewidth=1.1,
            zorder=1,
        )
        ax.scatter(
            i,
            conventional,
            s=24,
            color=CONVENTIONAL,
            edgecolor=INK,
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            i,
            area,
            s=24,
            color=AREA,
            edgecolor=AREA,
            linewidth=0.35,
            zorder=4,
        )

        ax.text(
            i + 0.06,
            area,
            f"{area:.1f}%",
            va="center",
            ha="left",
            fontsize=5.8,
            color=INK,
        )
        ax.text(
            i + 0.06,
            conventional,
            f"{conventional:.1f}%",
            va="center",
            ha="left",
            fontsize=5.8,
            color=INK,
        )

    ax.set_xlim(-0.35, 1.45)
    ax.set_xticks(x, labels, fontsize=6.1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(
        "Top-500 retained (%)",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    style_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=AREA,
            markeredgecolor=AREA,
            markersize=5,
            label="AREA",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CONVENTIONAL,
            markeredgecolor=INK,
            markersize=5,
            label="Matched conventional",
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=5.5,
        loc="upper left",
        bbox_to_anchor=(0, -0.20),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def make_figure(
    conventional: pd.DataFrame,
    attribution: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        facecolor="white",
    )

    # Precise manuscript layout: Panel A remains dominant, while B/C receive
    # substantially more vertical plotting area. A dedicated title band keeps
    # the lower-panel headings clear of Panel A's x-axis label.
    ax_a = fig.add_axes([0.105, 0.55, 0.860, 0.25])
    ax_b = fig.add_axes([0.105, 0.16, 0.3675, 0.28])
    ax_c = fig.add_axes([0.5975, 0.16, 0.3675, 0.28])

    draw_method_agreement(ax_a, conventional)
    draw_shapley(ax_b, attribution)
    draw_top500(ax_c, conventional)

    # Dedicated title band for the lower panels.
    fig.text(
        0.060,
        0.468,
        "B",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    fig.text(
        0.105,
        0.468,
        "Which covariates drive rank reorganization?",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        va="bottom",
    )

    fig.text(
        0.552,
        0.468,
        "C",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    fig.text(
        0.5975,
        0.468,
        "What changes in the top-ranked features?",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        va="bottom",
    )

    # Panel A heading is deliberately explicit: the reader should know the
    # question before interpreting any points.
    ax_a.text(
        -0.055,
        1.17,
        "A",
        transform=ax_a.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    ax_a.text(
        0.0,
        1.17,
        "AREA and conventional methods show similar adjustment sensitivity",
        transform=ax_a.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    ax_a.text(
        0.0,
        1.08,
        "Each point = one phenotype contrast; rank reorganization = median absolute rank shift / Nfeatures.",
        transform=ax_a.transAxes,
        fontsize=5.8,
        color=MID,
        va="bottom",
    )

    # Keep the primary comparison numerically visible but subordinate to the
    # geometry of the plot.
    fig.text(
        0.06,
        0.955,
        "Figure 5. Covariate adjustment affects AREA and conventional rankings similarly",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        ha="left",
    )
    fig.text(
        0.06,
        0.925,
        "Adjustment sensitivity is compared on the same normalized rank-displacement scale across matched analyses",
        fontsize=6.6,
        color=MID,
        ha="left",
    )
    fig.text(
        0.06,
        0.045,
        "Rank reorganization = median(|rankadjusted − rankunadjusted|) / Nfeatures. "
        "Shapley allocations describe contributions to the statistical-model transformation",
        fontsize=4.45,
        color=MID,
        ha="left",
    )
    fig.text(
        0.06,
        0.030,
        "from intercept-only to fully adjusted AREA; they are not causal or biological fractions.",
        fontsize=4.45,
        color=MID,
        ha="left",
    )

    return fig


def write_caption(output_dir: Path) -> None:
    caption = """Figure 5. Covariate adjustment affects AREA and conventional rankings similarly. (A) Adjustment-induced rank reorganization for AREA plotted against the matched conventional method for six pre-specified phenotype contrasts in DLPFC RNA (DESeq2) and DLPFC TMT (limma). Each point is one phenotype contrast; the 1:1 line indicates identical adjustment sensitivity, and distance from the origin indicates the magnitude of ranking reorganization. Rank reorganization is median absolute rank displacement divided by Nfeatures and is displayed as a percentage; thus 10% corresponds to a median displacement of 0.10 × Nfeatures. R² = 0.989 across the 12 matched contrasts. (B) Exact Shapley attribution of the AREA rank transformation from intercept-only to fully adjusted models, shown as mean Shapley-allocated normalized rank distance across six phenotype contrasts. Segment length represents the absolute contribution of each covariate to the observed ranking transformation; total bar length is the mean total rank reorganization. Sequencing batch is the largest contributor in DLPFC and monocyte RNA, whereas age is the largest contributor in TMT proteomics. (C) Fraction of the 500 most significant features in the unadjusted analysis retained among the top 500 after adjustment. Shapley allocations describe contributions to a statistical-model transformation and are not causal or biological fractions.
"""
    (output_dir / "Figure5_caption.txt").write_text(caption)


def write_plot_data(
    output_dir: Path,
    conventional: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    conventional[
        [
            "task_id",
            "contrast_id",
            "modality",
            "method",
            "AREA_median_normalized_rank_shift",
            "CONVENTIONAL_median_normalized_rank_shift",
            "AREA_TOP500_overlap_fraction",
            "CONVENTIONAL_TOP500_overlap_fraction",
        ]
    ].to_csv(
        output_dir / "Figure5_method_comparison_plot_data.tsv",
        sep="\t",
        index=False,
    )

    attribution[
        [
            "modality",
            "group",
            "mean_shapley",
            "mean_absolute_contribution",
        ]
    ].to_csv(
        output_dir / "Figure5_covariate_attribution_plot_data.tsv",
        sep="\t",
        index=False,
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()

    comparison_dir = locate_dir(
        repo_root,
        args.comparison_dir,
        [
            Path("results")
            / "stage11_conventional_covariate_sensitivity"
            / "comparison"
        ],
        "conventional-comparison",
    )
    attribution_dir = locate_dir(
        repo_root,
        args.attribution_dir,
        [Path("results") / "stage11_covariate_attribution_analysis"],
        "covariate-attribution",
    )

    comparison_path = (
        comparison_dir
        / "AREA_VS_CONVENTIONAL_ADJUSTMENT_SENSITIVITY.tsv"
    )
    attribution_path = (
        attribution_dir
        / "ABSOLUTE_SHAPLEY_MODALITY_SUMMARY.tsv"
    )

    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)
    if not attribution_path.exists():
        raise FileNotFoundError(attribution_path)

    conventional = load_conventional(comparison_path)
    attribution = load_attribution(attribution_path)
    validate_frozen_values(conventional)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            repo_root
            / "results"
            / "figures"
            / "figure_5_covariate_sensitivity"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure(conventional, attribution)

    stem = "Figure5_covariate_sensitivity"
    fig.savefig(
        output_dir / f"{stem}.pdf",
        facecolor="white",
        bbox_inches=None,
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        facecolor="white",
        bbox_inches=None,
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=PNG_DPI,
        facecolor="white",
        bbox_inches=None,
    )
    plt.close(fig)

    write_caption(output_dir)
    write_plot_data(output_dir, conventional, attribution)

    print()
    print("=" * 88)
    print("FIGURE 5 COMPLETE")
    print("=" * 88)
    print(f"Stage 11 conventional source: {comparison_path}")
    print(f"Stage 11 attribution source:  {attribution_path}")
    print(f"Canvas: {FIG_WIDTH_MM} × {FIG_HEIGHT_MM} mm")
    print(f"PDF: {output_dir / (stem + '.pdf')}")
    print(f"SVG: {output_dir / (stem + '.svg')}")
    print(f"PNG: {output_dir / (stem + '.png')}")
    print(f"Caption: {output_dir / 'Figure5_caption.txt'}")


if __name__ == "__main__":
    main()

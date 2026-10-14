#!/usr/bin/env python3
"""
Figure 3 — AREA signal architecture.

Scientific question
-------------------
What distinguishes features preferentially ranked by AREA from those
preferentially ranked by conventional mean-based methods?

Panels
------
A. Participant-level NCI/AD distribution overlap in method-preferential
   feature sets.
B. Conditional odds ratios for the primary PURE NCI→AD analysis.
C. Phenotype-generalization summary across the 45 pre-specified models.

Publication format
------------------
170 × 140 mm exact canvas, following the Hsp60/10 publication-figure
formatting system: DejaVu Sans, compact typography, restrained palette,
minimal axes, no tight bounding-box export, vector PDF/SVG, 600-dpi PNG.

Data provenance
---------------
Panel A uses the nine frozen Stage 9 NCI-vs-AD overlap medians. The parser
first looks for explicit median-overlap values in the Stage 9 interpretation
text. The current final interpretation summarizes the result but does not
contain those nine numeric medians, so a validated frozen fallback table is
kept below in one auditable location.

Panels B/C read:
    stage9_generalization_adjusted_models.tsv

The script does not recompute statistics.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd


# ---------------------------------------------------------------------
# Hsp60/10 publication style
# ---------------------------------------------------------------------

FIG_WIDTH_MM = 170
FIG_HEIGHT_MM = 140
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FIG_HEIGHT_IN = FIG_HEIGHT_MM / 25.4
PNG_DPI = 600

INK = "#283038"
MID = "#66727C"
GRID = "#E5E9EC"
LIGHT = "#D5DADF"

AREA = "#7A8FA8"
CONV = "#B8C0CA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": PNG_DPI,
    }
)

MODALITIES = ["Monocyte RNA", "DLPFC RNA", "DLPFC TMT"]
THRESHOLDS = ["TOP250", "TOP500", "TOP1000"]

# Frozen Stage 9 primary NCI-vs-AD participant-distribution overlap medians.
# These are the values used for Panel A when the final interpretation text
# contains the qualitative result but not the nine numeric medians.
# Tuple order: (AREA, conventional).
FROZEN_NCI_AD_OVERLAP_MEDIANS = {
    "Monocyte RNA": {
        "TOP250": (0.815, 0.848),
        "TOP500": (0.827, 0.852),
        "TOP1000": (0.832, 0.858),
    },
    "DLPFC RNA": {
        "TOP250": (0.787, 0.798),
        "TOP500": (0.793, 0.819),
        "TOP1000": (0.815, 0.837),
    },
    "DLPFC TMT": {
        "TOP250": (0.735, 0.794),
        "TOP500": (0.763, 0.813),
        "TOP1000": (0.796, 0.839),
    },
}


GENERALIZATION_CONTRASTS = [
    "cogdx_NCI_vs_MCI",
    "cogdx_MCI_vs_AD",
    "Braak_0-2_vs_3-4",
    "Braak_3-4_vs_5-6",
    "CERAD_1-2_vs_3-4",
]

GENERALIZATION_LABELS = [
    "NCI → MCI",
    "MCI → AD",
    "Early Braak",
    "Late Braak",
    "CERAD",
]

MODEL_MODALITY_MAP = {
    "monocyte": "Monocyte RNA",
    "dlpfc": "DLPFC RNA",
    "tmt": "DLPFC TMT",
}


# ---------------------------------------------------------------------
# Arguments and input discovery
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=None,
        help="Optional stage9_generalization_adjusted_models.tsv path.",
    )
    parser.add_argument(
        "--overlap-source",
        type=Path,
        default=None,
        help="Optional Stage 9 final interpretation text containing "
        "the frozen overlap medians.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def first_existing(candidates: list[Path], description: str) -> Path:
    for path in candidates:
        path = path.expanduser().resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not locate {description}. Tried:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def locate_model_source(
    repo_root: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    return first_existing(
        [
            (
                repo_root
                / "results"
                / "stage9_phenotype_generalization"
                / "stage9_generalization_adjusted_models.tsv"
            ),
            (
                Path.home()
                / "rosmap_multiomic_slurm"
                / "results"
                / "stage9_phenotype_generalization"
                / "stage9_generalization_adjusted_models.tsv"
            ),
            (
                Path.home()
                / "Developer"
                / "stage11_offline_2026-09-28"
                / "results"
                / "stage9_phenotype_generalization"
                / "stage9_generalization_adjusted_models.tsv"
            ),
        ],
        "stage9_generalization_adjusted_models.tsv",
    )


def locate_overlap_source(
    repo_root: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    return first_existing(
        [
            (
                repo_root
                / "results"
                / "multiomic_concordance"
                / "STAGE9_SIGNAL_ARCHITECTURE_FINAL_INTERPRETATION_UPDATED.txt"
            ),
            (
                repo_root
                / "results"
                / "multiomic_concordance"
                / "STAGE9_SIGNAL_ARCHITECTURE_FINAL_INTERPRETATION.txt"
            ),
            (
                Path.home()
                / "rosmap_multiomic_slurm"
                / "results"
                / "multiomic_concordance"
                / "STAGE9_SIGNAL_ARCHITECTURE_FINAL_INTERPRETATION_UPDATED.txt"
            ),
            (
                Path.home()
                / "rosmap_multiomic_slurm"
                / "results"
                / "multiomic_concordance"
                / "STAGE9_SIGNAL_ARCHITECTURE_FINAL_INTERPRETATION.txt"
            ),
        ],
        "Stage 9 final interpretation text",
    )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def parse_overlap_medians(
    path: Path,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Load the nine frozen NCI-vs-AD overlap medians for Panel A.

    Older Stage 9 interpretation files contained explicit lines such as
    "AREA median overlap: ..." and "DESeq2 median overlap: ...". The current
    FINAL_INTERPRETATION file retains the biological conclusion but not those
    numeric medians. We therefore:

    1. parse explicit medians when they are present; otherwise
    2. verify that this is the expected Stage 9 NCI-vs-AD interpretation file
       and return the validated frozen Stage 9 medians defined above.

    This keeps the plotting code deterministic while making the provenance
    visible and auditable rather than silently inferring values from ORs.
    """
    text = path.read_text()

    # Normalize whitespace enough to tolerate tabs, non-breaking spaces,
    # Markdown-style headings, and variable blank-line formatting.
    normalized = text.replace("\u00a0", " ")

    modality_aliases = {
        "Monocyte RNA": [r"MONOCYTE\s+RNA", r"MONOCYTE"],
        "DLPFC RNA": [r"DLPFC\s+RNA", r"DLPFC"],
        "DLPFC TMT": [r"TMT\s+PROTEOMICS", r"DLPFC\s+TMT", r"TMT"],
    }

    parsed: dict[str, dict[str, tuple[float, float]]] = {
        modality: {} for modality in MODALITIES
    }

    # Flexible explicit-value parser. Search locally around each TOP threshold
    # for AREA + conventional median-overlap values.
    for modality in MODALITIES:
        conventional_name = (
            r"limma" if modality == "DLPFC TMT" else r"DESeq2"
        )

        # Identify a modality section if possible, but do not depend on exact
        # heading capitalization or blank-line structure.
        section = None
        for alias in modality_aliases[modality]:
            match = re.search(alias, normalized, flags=re.I)
            if match is not None:
                section = normalized[match.start():]
                break

        if section is None:
            continue

        for threshold in THRESHOLDS:
            # Restrict the search to a moderate window after each TOP label so
            # values from another modality/threshold cannot be mixed together.
            threshold_match = re.search(
                rf"\b{re.escape(threshold)}\b",
                section,
                flags=re.I,
            )
            if threshold_match is None:
                continue

            window = section[
                threshold_match.start():
                threshold_match.start() + 500
            ]

            area_match = re.search(
                r"AREA\s+median\s+overlap\s*[:=]\s*([0-9]*\.?[0-9]+)",
                window,
                flags=re.I,
            )
            conv_match = re.search(
                rf"{conventional_name}\s+median\s+overlap\s*[:=]\s*"
                r"([0-9]*\.?[0-9]+)",
                window,
                flags=re.I,
            )

            if area_match is not None and conv_match is not None:
                parsed[modality][threshold] = (
                    float(area_match.group(1)),
                    float(conv_match.group(1)),
                )

    complete = all(
        threshold in parsed[modality]
        for modality in MODALITIES
        for threshold in THRESHOLDS
    )
    if complete:
        print(
            "Panel A overlap source: parsed nine explicit median-overlap "
            f"values from {path}"
        )
        return parsed

    # The current FINAL_INTERPRETATION file does not contain the nine medians.
    # Verify that it is the expected frozen Stage 9 document before using the
    # validated fallback values.
    required_markers = [
        "STAGE 9 SIGNAL ARCHITECTURE",
        "PRIMARY NCI-vs-AD RESULT",
        "AREA-predominant features had lower",
        "participant-distribution overlap",
    ]
    missing_markers = [
        marker
        for marker in required_markers
        if marker.lower() not in normalized.lower()
    ]
    if missing_markers:
        raise ValueError(
            "Could not parse nine overlap medians from "
            f"{path}, and the file did not contain the expected Stage 9 "
            "primary-result markers. Missing markers: "
            + ", ".join(missing_markers)
        )

    print(
        "Panel A overlap source: current Stage 9 final interpretation text "
        "does not contain the nine numeric medians; using the validated "
        "frozen Stage 9 NCI-vs-AD overlap table encoded in this script."
    )

    return {
        modality: {
            threshold: tuple(
                FROZEN_NCI_AD_OVERLAP_MEDIANS[modality][threshold]
            )
            for threshold in THRESHOLDS
        }
        for modality in MODALITIES
    }


def load_models(path: Path) -> pd.DataFrame:
    models = pd.read_csv(path, sep="\t")

    required = {
        "contrast_id",
        "modality",
        "threshold",
        "odds_ratio_per_1SD",
        "OR_95CI_low",
        "OR_95CI_high",
        "focal_overlap_fdr_generalization45",
    }
    missing = required - set(models.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if "term" in models.columns:
        models = models[models["term"] == "bayestestR_overlap"].copy()

    models = models[
        models["modality"].isin(MODEL_MODALITY_MAP)
        & models["threshold"].isin(THRESHOLDS)
        & models["contrast_id"].isin(
            ["cogdx_NCI_vs_AD"] + GENERALIZATION_CONTRASTS
        )
    ].copy()

    expected = 54
    if len(models) != expected:
        raise ValueError(
            f"Expected 54 focal overlap models including NCI→AD; "
            f"found {len(models)}."
        )

    models["modality_label"] = models["modality"].map(
        MODEL_MODALITY_MAP
    )
    models["generalization_significant"] = (
        models["focal_overlap_fdr_generalization45"] < 0.05
    )

    return models


# ---------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------

def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)

    ax.tick_params(
        axis="x",
        labelsize=5.7,
        length=2.2,
        width=0.45,
        color=INK,
        labelcolor=INK,
        pad=2,
    )
    ax.tick_params(
        axis="y",
        labelsize=5.4,
        length=0,
        color=INK,
        labelcolor=INK,
        pad=2,
    )

    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.45,
    )
    ax.set_axisbelow(True)


def panel_heading(
    ax: plt.Axes,
    letter: str,
    title: str,
    subtitle: str,
    letter_x: float = -0.12,
) -> None:
    ax.text(
        letter_x,
        1.115,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        0.0,
        1.115,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        0.0,
        1.025,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.55,
        color=MID,
        linespacing=1.15,
        clip_on=False,
    )


# ---------------------------------------------------------------------
# Panel A — descriptive distribution overlap
# ---------------------------------------------------------------------

def draw_panel_a(
    ax: plt.Axes,
    overlap: dict[str, dict[str, tuple[float, float]]],
) -> None:
    y_values = np.arange(9)
    rows = [
        (modality, threshold)
        for modality in MODALITIES
        for threshold in THRESHOLDS
    ]

    for y, (modality, threshold) in zip(y_values, rows):
        area_value, conv_value = overlap[modality][threshold]

        ax.plot(
            [area_value, conv_value],
            [y, y],
            color=MID,
            linewidth=0.75,
            zorder=1,
        )
        ax.scatter(
            area_value,
            y,
            s=24,
            color=AREA,
            zorder=3,
        )
        ax.scatter(
            conv_value,
            y,
            s=22,
            marker="s",
            facecolor="white",
            edgecolor=CONV,
            linewidth=0.9,
            zorder=3,
        )

    for separator in [2.5, 5.5]:
        ax.axhline(
            separator,
            color=LIGHT,
            linewidth=0.55,
            zorder=0,
        )

    ax.set_yticks(
        y_values,
        [
            f"{modality} · {threshold}"
            for modality, threshold in rows
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0.70, 0.87)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85])
    ax.set_xlabel(
        "NCI–AD participant-distribution overlap",
        fontsize=6.8,
        fontweight="bold",
        labelpad=4,
    )

    panel_heading(
        ax,
        "A",
        "AREA features have lower overlap",
        "Median NCI–AD density overlap in method-preferential sets;\nlower = more separation",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=AREA,
            markeredgecolor=AREA,
            markersize=4,
            label="AREA",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor=CONV,
            markersize=4,
            label="Conventional",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.155),
        frameon=False,
        ncol=2,
        fontsize=5.1,
        handletextpad=0.22,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    style_axis(ax)


# ---------------------------------------------------------------------
# Panel B — primary conditional NCI→AD models
# ---------------------------------------------------------------------

def draw_panel_b(
    ax: plt.Axes,
    models: pd.DataFrame,
) -> None:
    primary = models[
        models["contrast_id"] == "cogdx_NCI_vs_AD"
    ].copy()

    rows = [
        (modality, threshold)
        for modality in MODALITIES
        for threshold in THRESHOLDS
    ]
    y_values = np.arange(len(rows))

    for y, (modality, threshold) in zip(y_values, rows):
        row = primary[
            (primary["modality_label"] == modality)
            & (primary["threshold"] == threshold)
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one NCI→AD model for "
                f"{modality} / {threshold}."
            )

        record = row.iloc[0]
        odds_ratio = float(record["odds_ratio_per_1SD"])
        ci_low = float(record["OR_95CI_low"])
        ci_high = float(record["OR_95CI_high"])

        ax.errorbar(
            odds_ratio,
            y,
            xerr=[
                [odds_ratio - ci_low],
                [ci_high - odds_ratio],
            ],
            fmt="o",
            color=AREA,
            ecolor=AREA,
            markersize=3.7,
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            zorder=3,
        )

        ax.annotate(
            f"{odds_ratio:.2f}",
            xy=(odds_ratio, y),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=4.7,
            color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.10),
            zorder=4,
        )

    for separator in [2.5, 5.5]:
        ax.axhline(
            separator,
            color=LIGHT,
            linewidth=0.55,
            zorder=0,
        )

    ax.axvline(
        1.0,
        color=MID,
        linestyle="--",
        linewidth=0.7,
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlim(0.008, 1.7)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.set_yticks(
        y_values,
        [
            f"{modality} · {threshold}"
            for modality, threshold in rows
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Odds ratio per 1-SD increase in overlap",
        fontsize=6.8,
        fontweight="bold",
        labelpad=2,
    )

    panel_heading(
        ax,
        "B",
        "Lower overlap predicts AREA",
        "Adjusted NCI→AD models;\nOR < 1 = lower overlap",
    )

    ax.text(
        0.99,
        1.005,
        "1×",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )
    style_axis(ax)


# ---------------------------------------------------------------------
# Panel C — phenotype generalization
# ---------------------------------------------------------------------

def draw_panel_c(
    ax: plt.Axes,
    models: pd.DataFrame,
) -> None:
    general = models[
        models["contrast_id"].isin(GENERALIZATION_CONTRASTS)
    ].copy()

    x0 = 0.205
    width = 0.675
    cell_width = width / 5

    ax.axis("off")

    # Panel C uses a custom heading block because it spans the full figure width.
    ax.text(
        0.015,
        0.995,
        "C",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        0.075,
        0.995,
        "Phenotype generalization",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        0.075,
        0.915,
        "FDR-significant conditional overlap models across TOP250 / TOP500 / TOP1000",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.55,
        color=MID,
        clip_on=False,
    )
    ax.text(
        x0 + width + 0.025,
        0.660,
        "FDR sig.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.2,
        color=MID,
    )

    for j, label in enumerate(GENERALIZATION_LABELS):
        ax.text(
            x0 + (j + 0.5) * cell_width,
            0.635,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.7,
            fontweight="bold",
            color=INK,
        )

    for i, modality in enumerate(MODALITIES):
        y = 0.430 - i * 0.205

        ax.text(
            x0 - 0.018,
            y,
            modality,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.2,
            color=INK,
        )

        for j, contrast_id in enumerate(GENERALIZATION_CONTRASTS):
            cell = general[
                (general["modality_label"] == modality)
                & (general["contrast_id"] == contrast_id)
            ]

            if len(cell) != 3:
                raise ValueError(
                    f"Expected 3 thresholds for "
                    f"{modality} / {contrast_id}."
                )

            significant = int(cell["generalization_significant"].sum())
            x = x0 + j * cell_width

            # 0/3 = very light; 3/3 = strongest AREA slate.
            alpha = 0.10 + 0.27 * significant

            ax.add_patch(
                Rectangle(
                    (x + 0.006, y - 0.060),
                    cell_width - 0.012,
                    0.120,
                    transform=ax.transAxes,
                    facecolor=AREA,
                    alpha=alpha,
                    edgecolor=LIGHT,
                    linewidth=0.45,
                )
            )

            ax.text(
                x + cell_width / 2,
                y,
                f"{significant}/3",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.3,
                fontweight="bold" if significant >= 2 else "normal",
                color=INK,
            )

        total_significant = int(
            general.loc[
                general["modality_label"] == modality,
                "generalization_significant",
            ].sum()
        )

        ax.text(
            x0 + width + 0.025,
            y,
            f"{total_significant}/15",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=INK,
        )

    total_significant = int(
        general["generalization_significant"].sum()
    )
    total_lower_overlap = int(
        (general["odds_ratio_per_1SD"] < 1.0).sum()
    )

    ax.text(
        0.985,
        0.915,
        f"{total_significant}/45 FDR-significant  · "
        f"{total_lower_overlap}/45 OR < 1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        fontweight="bold",
        color=INK,
    )



# ---------------------------------------------------------------------
# Figure assembly and outputs
# ---------------------------------------------------------------------

def make_figure(
    models: pd.DataFrame,
    overlap: dict[str, dict[str, tuple[float, float]]],
) -> plt.Figure:
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        facecolor="white",
    )

    grid = fig.add_gridspec(
        2,
        2,
        left=0.19,
        right=0.985,
        top=0.855,
        bottom=0.105,
        hspace=0.42,
        wspace=0.43,
        height_ratios=[1.0, 0.76],
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    # Move panel A left while preserving panel B and the overall figure width.
    pos_a = ax_a.get_position()
    ax_a.set_position([
        pos_a.x0 - 0.028,
        pos_a.y0,
        pos_a.width,
        pos_a.height,
    ])

    draw_panel_a(ax_a, overlap)
    draw_panel_b(ax_b, models)
    draw_panel_c(ax_c, models)

    return fig


def write_caption(output_dir: Path) -> None:
    caption = """Figure 3. AREA preferentially ranks features with lower
participant-distribution overlap. (A) Median NCI–AD density overlap among
features preferentially ranked by AREA versus conventional mean-based
methods (DESeq2 for RNA; limma for TMT) at TOP250, TOP500, and TOP1000.
Lower overlap indicates less shared density between participant-level NCI
and AD distributions. (B) Conditional logistic models for the PURE NCI→AD
comparison. The outcome is AREA-predominant versus conventional-predominant
feature membership, and the predictor is standardized NCI–AD distribution
overlap; OR < 1 indicates that higher overlap is associated with lower odds
of AREA-predominant membership. (C) Number of TOP250/TOP500/TOP1000 models
with FDR < 0.05 for the same overlap association across five pre-specified
clinical and neuropathological contrasts. The 45 models are correlated and
are not independent biological replications. NCI→AD is the primary anchor
and is not included in the 45-model generalization family.
"""
    (output_dir / "Figure3_caption.txt").write_text(caption)


def write_audit(
    output_dir: Path,
    models: pd.DataFrame,
    overlap: dict[str, dict[str, tuple[float, float]]],
    model_source: Path,
    overlap_source: Path,
) -> None:
    overlap_rows = []
    for modality in MODALITIES:
        for threshold in THRESHOLDS:
            area_value, conv_value = overlap[modality][threshold]
            overlap_rows.append(
                {
                    "modality": modality,
                    "threshold": threshold,
                    "AREA_median_overlap": area_value,
                    "conventional_median_overlap": conv_value,
                    "AREA_minus_conventional": area_value - conv_value,
                }
            )

    pd.DataFrame(overlap_rows).to_csv(
        output_dir / "Figure3_distribution_overlap_values_audit.csv",
        index=False,
    )

    audit_models = models[
        [
            "contrast_id",
            "modality",
            "threshold",
            "odds_ratio_per_1SD",
            "OR_95CI_low",
            "OR_95CI_high",
            "focal_overlap_fdr_generalization45",
            "generalization_significant",
        ]
    ].copy()

    audit_models.to_csv(
        output_dir / "Figure3_generalization_models_audit.csv",
        index=False,
    )

    (output_dir / "Figure3_model_source.txt").write_text(
        f"{model_source}\n"
    )
    (output_dir / "Figure3_overlap_source.txt").write_text(
        f"{overlap_source}\n"
    )


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.expanduser().resolve()

    model_source = locate_model_source(
        repo_root,
        args.model_source,
    )
    overlap_source = locate_overlap_source(
        repo_root,
        args.overlap_source,
    )

    models = load_models(model_source)
    overlap = parse_overlap_medians(overlap_source)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            repo_root
            / "results"
            / "figures"
            / "figure_3_signal_architecture"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("FIGURE 3 INPUTS")
    print(f"  Model source:   {model_source}")
    print(f"  Overlap source: {overlap_source}")

    print("\nFIGURE 3 PANEL A VALUES")
    for modality in MODALITIES:
        for threshold in THRESHOLDS:
            area_value, conv_value = overlap[modality][threshold]
            print(
                f"  {modality:12s} {threshold:7s} "
                f"AREA={area_value:.5f} "
                f"CONV={conv_value:.5f} "
                f"DELTA={area_value - conv_value:+.5f}"
            )

    print("\nFIGURE 3 PANEL B VALUES")
    primary = models[models["contrast_id"] == "cogdx_NCI_vs_AD"]
    for _, row in primary.sort_values(
        ["modality", "threshold"]
    ).iterrows():
        print(
            f"  {row['modality']:9s} {row['threshold']:7s} "
            f"OR={row['odds_ratio_per_1SD']:.6f} "
            f"CI=({row['OR_95CI_low']:.6f}, "
            f"{row['OR_95CI_high']:.6f})"
        )

    general = models[
        models["contrast_id"].isin(GENERALIZATION_CONTRASTS)
    ]
    print("\nFIGURE 3 PANEL C SUMMARY")
    print(
        f"  OR < 1: {(general['odds_ratio_per_1SD'] < 1).sum()}/45"
    )
    print(
        f"  FDR < 0.05: "
        f"{general['generalization_significant'].sum()}/45"
    )

    fig = make_figure(models, overlap)

    stem = "Figure3_signal_architecture_manuscript_v14"
    fig.savefig(
        output_dir / f"{stem}.pdf",
        dpi=PNG_DPI,
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
    plt.close(fig)

    write_caption(output_dir)
    write_audit(
        output_dir,
        models,
        overlap,
        model_source,
        overlap_source,
    )

    print("\nFIGURE 3 OUTPUT")
    print(f"  Canvas: {FIG_WIDTH_MM} × {FIG_HEIGHT_MM} mm")
    print(f"  PDF: {output_dir / (stem + '.pdf')}")
    print(f"  SVG: {output_dir / (stem + '.svg')}")
    print(f"  PNG: {output_dir / (stem + '.png')}")
    print(
        "  Audit: "
        f"{output_dir / 'Figure3_distribution_overlap_values_audit.csv'}"
    )


if __name__ == "__main__":
    main()

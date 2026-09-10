#!/usr/bin/env python3
"""
make_weighted_area_biology_summary.py
=====================================

Create a clean, publication-style summary figure for the main biological
findings from the refined Weighted AREA module analysis, and export a focused
6–8 module follow-up set for detailed biological interpretation.

This script does NOT rerun GSEA. It uses the final refined module outputs.

Expected inputs
---------------
results/weighted_area_biology_final_v2/
    final_priority_modules.csv
    final_module_driver_genes.csv
    final_module_representative_pathways.csv

Outputs
-------
results/weighted_area_biology_final_v2/

Figure:
    weighted_area_key_biological_themes.png
    weighted_area_key_biological_themes.pdf

Tables:
    detailed_followup_modules.csv
    detailed_followup_module_drivers.csv
    detailed_followup_representative_pathways.csv

The figure is intentionally simple:
    1. Shared neuronal / energetic decline
    2. Shared remodeling / stress activation
    3. Amyloid-enhanced maintenance failure

The follow-up module set is selected to capture the strongest, most distinct,
and most biologically interpretable programs without becoming redundant.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_ROOT = Path("results/weighted_area_biology_final_v2")

FOLLOWUP_MODULES = [
    "Mitochondrial energetics",
    "Synaptic transmission",
    "Inflammatory / NF-kB signaling",
    "TGF / ECM remodeling",
    "Mitochondrial quality control",
    "Protein folding / proteostasis",
    "RHO / cytoskeletal signaling",
    "Chromatin / transcriptional regulation",
]

THEME_DEFINITIONS = [
    {
        "title": "1. Shared neuronal / energetic decline",
        "accent": "#4C78A8",
        "fill": "#F4F8FC",
        "arrow": "down",
        "bullets": [
            "Synaptic transmission decreases across cognition, amyloid, and tau.",
            "Mitochondrial energetics and mitochondrial quality-control pathways are broadly reduced.",
            "Translation, proteostasis, vesicle transport, and RNA-processing programs are also reduced.",
        ],
    },
    {
        "title": "2. Shared remodeling / stress activation",
        "accent": "#D94F3D",
        "fill": "#FDF7F5",
        "arrow": "up",
        "bullets": [
            "Inflammatory / NF-kB, TGF / ECM, vascular, and RHO-cytoskeletal pathways increase with disease severity.",
            "Chromatin regulation, apoptosis, lipid metabolism, and cell-cycle / senescence programs are also elevated.",
            "These activated programs are broadly consistent across cognition, amyloid, and tau.",
        ],
    },
    {
        "title": "3. Amyloid-enhanced maintenance failure",
        "accent": "#6F4AA8",
        "fill": "#F8F5FC",
        "arrow": "down",
        "arrow_label": "strongest in amyloid",
        "bullets": [
            "Amyloid shows the strongest reduction in mitochondrial energetics.",
            "The same amyloid-enhanced decline is seen for mitochondrial quality control, proteostasis, and vesicle / intracellular transport.",
            "This pattern suggests stronger coupling of amyloid burden to loss of core cellular-maintenance programs.",
        ],
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Directory containing final refined biology outputs.",
    )
    return p.parse_args()


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def load_inputs(root: Path):
    priority_path = root / "final_priority_modules.csv"
    drivers_path = root / "final_module_driver_genes.csv"
    reps_path = root / "final_module_representative_pathways.csv"

    require(priority_path)
    require(drivers_path)
    require(reps_path)

    priority = pd.read_csv(priority_path)
    drivers = pd.read_csv(drivers_path)
    reps = pd.read_csv(reps_path)

    return priority, drivers, reps


def select_followup(priority: pd.DataFrame) -> pd.DataFrame:
    available = set(priority["module"].astype(str))
    missing = [m for m in FOLLOWUP_MODULES if m not in available]

    if missing:
        raise ValueError(
            "The following requested follow-up modules were not found in "
            f"final_priority_modules.csv: {missing}"
        )

    followup = (
        priority
        .set_index("module")
        .loc[FOLLOWUP_MODULES]
        .reset_index()
        .copy()
    )

    followup["followup_order"] = np.arange(1, len(followup) + 1)

    return followup


def export_followup_tables(
    root: Path,
    followup: pd.DataFrame,
    drivers: pd.DataFrame,
    reps: pd.DataFrame,
):
    followup.to_csv(
        root / "detailed_followup_modules.csv",
        index=False,
    )

    driver_subset = (
        drivers[
            drivers["module"].isin(FOLLOWUP_MODULES)
        ]
        .copy()
    )

    # Keep only strongest, module-specific driver genes.
    if "passes_driver_filter" in driver_subset.columns:
        driver_subset = driver_subset[
            driver_subset["passes_driver_filter"] == True
        ]

    driver_subset = (
        driver_subset
        .sort_values(
            ["module", "driver_score"],
            ascending=[True, False],
        )
    )

    driver_subset.to_csv(
        root / "detailed_followup_module_drivers.csv",
        index=False,
    )

    rep_subset = (
        reps[
            reps["module"].isin(FOLLOWUP_MODULES)
        ]
        .copy()
    )

    sort_cols = [
        c for c in
        [
            "module",
            "n_significant_phenotypes",
            "mean_abs_NES",
        ]
        if c in rep_subset.columns
    ]

    if sort_cols:
        ascending = [
            True if c == "module" else False
            for c in sort_cols
        ]
        rep_subset = rep_subset.sort_values(
            sort_cols,
            ascending=ascending,
        )

    rep_subset.to_csv(
        root / "detailed_followup_representative_pathways.csv",
        index=False,
    )


# =============================================================================
# FIGURE
# =============================================================================

def draw_panel(
    ax,
    y0,
    height,
    title,
    bullets,
    accent,
    fill,
    arrow,
    arrow_label=None,
):
    x0 = 0.04
    width = 0.92

    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor=accent,
        facecolor=fill,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)

    # Header
    ax.text(
        x0 + 0.025,
        y0 + height - 0.055,
        title,
        transform=ax.transAxes,
        fontsize=12.2,
        fontweight="bold",
        color=accent,
        va="top",
    )

    # Divider before arrow area
    divider_x = x0 + width - 0.15
    ax.plot(
        [divider_x, divider_x],
        [y0 + 0.04, y0 + height - 0.10],
        transform=ax.transAxes,
        color=accent,
        alpha=0.35,
        linewidth=0.8,
    )

    # Bullets
    start_y = y0 + height - 0.115
    line_gap = 0.060

    for i, bullet in enumerate(bullets):
        yy = start_y - i * line_gap

        ax.text(
            x0 + 0.026,
            yy,
            u"\u2022",
            transform=ax.transAxes,
            fontsize=13,
            color=accent,
            va="top",
        )

        ax.text(
            x0 + 0.058,
            yy,
            bullet,
            transform=ax.transAxes,
            fontsize=8.4,
            color="#222222",
            va="top",
            wrap=True,
        )

    # Arrow
    arrow_x = divider_x + 0.072
    if arrow == "down":
        y_start = y0 + height - 0.13
        y_end = y0 + 0.09
    else:
        y_start = y0 + 0.09
        y_end = y0 + height - 0.13

    arrow_patch = FancyArrowPatch(
        (arrow_x, y_start),
        (arrow_x, y_end),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=28,
        linewidth=2.1,
        color=accent,
    )
    ax.add_patch(arrow_patch)

    if arrow_label:
        ax.text(
            arrow_x,
            y0 + 0.042,
            arrow_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=accent,
        )


def make_summary_figure(root: Path):
    fig, ax = plt.subplots(
        figsize=(7.2, 8.4)
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Weighted AREA: key biological themes",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        0.5,
        0.945,
        (
            "Simplified summary of the dominant cross-phenotype patterns "
            "across cognition, amyloid, and tau"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#444444",
    )

    panel_height = 0.245
    panel_ys = [0.655, 0.375, 0.095]

    for y0, theme in zip(panel_ys, THEME_DEFINITIONS):
        draw_panel(
            ax=ax,
            y0=y0,
            height=panel_height,
            title=theme["title"],
            bullets=theme["bullets"],
            accent=theme["accent"],
            fill=theme["fill"],
            arrow=theme["arrow"],
            arrow_label=theme.get("arrow_label"),
        )

    ax.text(
        0.5,
        0.025,
        (
            "Summary derived from prioritized Weighted AREA biological modules; "
            "direction reflects increasing phenotype severity / pathology."
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#666666",
    )

    fig.savefig(
        root / "weighted_area_key_biological_themes.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        root / "weighted_area_key_biological_themes.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    root = Path(args.root)

    priority, drivers, reps = load_inputs(root)

    followup = select_followup(priority)

    export_followup_tables(
        root=root,
        followup=followup,
        drivers=drivers,
        reps=reps,
    )

    make_summary_figure(root)

    print("=" * 88)
    print("WEIGHTED AREA BIOLOGY SUMMARY + DETAILED FOLLOW-UP SET")
    print("=" * 88)
    print()
    print("Selected detailed follow-up modules:")
    print()

    for _, row in followup.iterrows():
        module = row["module"]
        module_class = row.get("module_class", "")
        cognition = row.get("cognition_mean_NES", np.nan)
        amyloid = row.get("amyloid_mean_NES", np.nan)
        tau = row.get("tau_mean_NES", np.nan)

        print(
            f"{int(row['followup_order'])}. {module}"
            f" | {module_class}"
            f" | cognition={cognition:.2f}"
            f" | amyloid={amyloid:.2f}"
            f" | tau={tau:.2f}"
        )

    print()
    print("Outputs:")
    print(f"  {root / 'weighted_area_key_biological_themes.png'}")
    print(f"  {root / 'weighted_area_key_biological_themes.pdf'}")
    print(f"  {root / 'detailed_followup_modules.csv'}")
    print(f"  {root / 'detailed_followup_module_drivers.csv'}")
    print(f"  {root / 'detailed_followup_representative_pathways.csv'}")


if __name__ == "__main__":
    main()

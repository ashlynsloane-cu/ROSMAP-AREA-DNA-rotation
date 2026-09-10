#!/usr/bin/env python3
"""
make_weighted_area_biology_theme_heatmaps_v7.py
================================================

Final layout refinement for the Weighted AREA biological-theme summary.

Changes from v3
---------------
- manuscript dimensions retained (7.2 in wide)
- dedicated header band above each heatmap:
    section title
    short subtitle
    cognition / amyloid / tau labels
  so these elements never overlap
- phenotype labels shown only on the bottom panel
- panel 3 amyloid emphasis retained as a thin outline only
- removed the "strongest decline" annotation from inside the plotting area
- tighter, more even vertical spacing
- cleaner colorbar and footer placement

The script automatically finds the final module phenotype profile table.

Outputs
-------
weighted_area_biology_themes_heatmap_v7.png
weighted_area_biology_themes_heatmap_v7.pdf
weighted_area_biology_themes_heatmap_v7_modules_used.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


# =============================================================================
# FIGURE SPECS
# =============================================================================

FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 7.65

PHENOTYPE_ORDER = [
    "Cognitive_stage",
    "cogn_global_impairment",
    "mmse_impairment",
    "CERAD_equal",
    "CERAD_amyloid_calibrated",
    "amyloid_continuous",
    "Braak_equal",
    "Braak_tangle_calibrated",
    "tangle_continuous",
]

PHENOTYPE_LABELS = {
    "Cognitive_stage": "Cognitive\nstage",
    "cogn_global_impairment": "Global\ncognition",
    "mmse_impairment": "MMSE",
    "CERAD_equal": "CERAD\nequal",
    "CERAD_amyloid_calibrated": "CERAD\ncalibrated",
    "amyloid_continuous": "Amyloid\ncontinuous",
    "Braak_equal": "Braak\nequal",
    "Braak_tangle_calibrated": "Braak\ncalibrated",
    "tangle_continuous": "Tangles\ncontinuous",
}

THEMES = [
    {
        "title": "Shared neuronal / energetic decline",
        "subtitle": "Shared decline across all phenotype axes",
        "modules": [
            "Synaptic transmission",
            "Mitochondrial energetics",
            "Translation / tRNA biology",
            "Mitochondrial quality control",
            "Protein folding / proteostasis",
            "Vesicle / intracellular transport",
            "RNA processing / stability",
        ],
    },
    {
        "title": "Shared remodeling / stress activation",
        "subtitle": "Shared activation across all phenotype axes",
        "modules": [
            "Inflammatory / NF-kB signaling",
            "TGF / ECM remodeling",
            "RHO / cytoskeletal signaling",
            "Vascular / endothelial remodeling",
            "Chromatin / transcriptional regulation",
            "Cell death / apoptosis",
            "Cell cycle / senescence",
            "Lipid metabolism",
        ],
    },
    {
        "title": "Amyloid-enhanced maintenance failure",
        "subtitle": "Maintenance decline is strongest along the amyloid axis",
        "modules": [
            "Mitochondrial energetics",
            "Mitochondrial quality control",
            "Protein folding / proteostasis",
            "Vesicle / intracellular transport",
        ],
    },
]


# =============================================================================
# INPUT DISCOVERY
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", default=None)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def find_profiles(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(
                f"Explicit profile table does not exist: {p}"
            )
        return p

    candidates = [
        Path(
            "results/weighted_area_biology_final_v2/"
            "final_module_phenotype_profiles.csv"
        ),
        Path(
            "results/weighted_area_biology_final/"
            "final_module_phenotype_profiles.csv"
        ),
        Path(
            "results/weighted_area_biology_refined/"
            "module_phenotype_profiles.csv"
        ),
    ]

    for p in candidates:
        if p.exists():
            return p

    root = Path("results")
    if root.exists():
        matches = sorted(
            root.rglob("final_module_phenotype_profiles.csv")
        )
        if matches:
            v2 = [p for p in matches if "final_v2" in str(p)]
            if v2:
                return v2[0]
            final = [p for p in matches if "biology_final" in str(p)]
            if final:
                return final[0]
            return matches[0]

        refined = sorted(
            root.rglob("module_phenotype_profiles.csv")
        )
        if refined:
            return refined[0]

    raise FileNotFoundError(
        "Could not locate final_module_phenotype_profiles.csv.\n"
        "Try:\n"
        "  find results -name 'final_module_phenotype_profiles.csv'"
    )


# =============================================================================
# LOAD TABLE
# =============================================================================

def resolve_column(
    df: pd.DataFrame,
    candidates: List[str],
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    raise ValueError(
        f"Could not identify {label}. "
        f"Available columns: {', '.join(df.columns)}"
    )


def load_profiles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    module_col = resolve_column(
        df,
        ["module", "module_name"],
        "module column",
    )

    phenotype_col = resolve_column(
        df,
        ["phenotype", "trait"],
        "phenotype column",
    )

    nes_col = resolve_column(
        df,
        [
            "median_NES",
            "median_nes",
            "mean_NES",
            "mean_nes",
            "NES",
        ],
        "module NES column",
    )

    sig_col = None
    for candidate in [
        "significant_fraction",
        "sig_fraction",
        "fraction_significant",
    ]:
        if candidate in df.columns:
            sig_col = candidate
            break

    out = pd.DataFrame(
        {
            "module": df[module_col].astype(str).str.strip(),
            "phenotype": df[phenotype_col].astype(str).str.strip(),
            "NES": pd.to_numeric(df[nes_col], errors="coerce"),
        }
    )

    if sig_col is not None:
        out["sig_fraction"] = pd.to_numeric(
            df[sig_col],
            errors="coerce",
        )
    else:
        out["sig_fraction"] = np.nan

    return out.dropna(
        subset=["module", "phenotype", "NES"]
    )


# =============================================================================
# MATRICES
# =============================================================================

def build_matrix(
    profiles: pd.DataFrame,
    modules: List[str],
):
    available = set(profiles["module"])

    present = [
        module for module in modules
        if module in available
    ]

    missing = [
        module for module in modules
        if module not in available
    ]

    if missing:
        print(
            "  Warning: missing modules: "
            + ", ".join(missing)
        )

    sub = profiles[
        profiles["module"].isin(present)
    ].copy()

    nes = (
        sub.pivot_table(
            index="module",
            columns="phenotype",
            values="NES",
            aggfunc="median",
        )
        .reindex(
            index=present,
            columns=PHENOTYPE_ORDER,
        )
    )

    sig = (
        sub.pivot_table(
            index="module",
            columns="phenotype",
            values="sig_fraction",
            aggfunc="median",
        )
        .reindex(
            index=present,
            columns=PHENOTYPE_ORDER,
        )
    )

    return nes, sig


# =============================================================================
# PLOTTING
# =============================================================================

def draw_panel(
    ax,
    nes: pd.DataFrame,
    sig: pd.DataFrame,
    title: str,
    subtitle: str,
    norm: TwoSlopeNorm,
    show_xlabels: bool,
    highlight_amyloid: bool,
):
    image = ax.imshow(
        nes.to_numpy(),
        cmap="coolwarm",
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    # -------------------------------------------------------------------------
    # Dedicated text bands ABOVE the axes.
    # No title/group-label overlap.
    # -------------------------------------------------------------------------
    ax.text(
        0.0,
        1.235,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.7,
        fontweight="bold",
        clip_on=False,
    )

    ax.text(
        0.0,
        1.135,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.15,
        color="#555555",
        clip_on=False,
    )

    # Phenotype-axis group labels.
    # x is in data coordinates; y is in axes coordinates.
    group_labels = [
        ("COGNITION", 1.0),
        ("AMYLOID", 4.0),
        ("TAU", 7.0),
    ]

    for label, x in group_labels:
        ax.text(
            x,
            1.035,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.9,
            fontweight="bold",
            clip_on=False,
        )

    # -------------------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------------------
    ax.set_yticks(
        np.arange(len(nes.index))
    )
    ax.set_yticklabels(
        nes.index,
        fontsize=5.9,
    )

    ax.set_xticks(
        np.arange(len(PHENOTYPE_ORDER))
    )

    if show_xlabels:
        ax.set_xticklabels(
            [
                PHENOTYPE_LABELS[p]
                for p in PHENOTYPE_ORDER
            ],
            rotation=36,
            ha="right",
            fontsize=5.35,
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(
            axis="x",
            length=0,
        )

    # -------------------------------------------------------------------------
    # Group separators
    # -------------------------------------------------------------------------
    for x in [2.5, 5.5]:
        ax.axvline(
            x,
            linewidth=0.6,
            color="#3478C5",
            alpha=0.65,
        )

    # -------------------------------------------------------------------------
    # Significance dots
    # -------------------------------------------------------------------------
    for i in range(len(nes.index)):
        for j in range(len(PHENOTYPE_ORDER)):
            frac = sig.iloc[i, j]

            if (
                pd.notna(frac)
                and frac >= 0.50
            ):
                ax.text(
                    j,
                    i,
                    "•",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="black",
                )

    # -------------------------------------------------------------------------
    # Subtle cell boundaries
    # -------------------------------------------------------------------------
    ax.set_xticks(
        np.arange(-0.5, len(PHENOTYPE_ORDER), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(nes.index), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=0.22,
        alpha=0.22,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # -------------------------------------------------------------------------
    # Panel 3 amyloid emphasis:
    # thin outline only. Subtitle already explains the interpretation.
    # -------------------------------------------------------------------------
    if highlight_amyloid:
        rect = Rectangle(
            (2.5, -0.5),
            3.0,
            len(nes.index),
            fill=False,
            edgecolor="#C56700",
            linewidth=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)

    return image


def make_figure(
    profiles: pd.DataFrame,
    outdir: Path,
):
    matrices = []
    all_values = []
    used_rows = []

    for theme in THEMES:
        nes, sig = build_matrix(
            profiles,
            theme["modules"],
        )

        matrices.append(
            (theme, nes, sig)
        )

        if len(nes):
            all_values.extend(
                nes.to_numpy()
                .ravel()
                .tolist()
            )

        for module in nes.index:
            used_rows.append(
                {
                    "theme": theme["title"],
                    "module": module,
                }
            )

    finite = np.array(
        [
            value
            for value in all_values
            if pd.notna(value)
        ]
    )

    if len(finite) == 0:
        raise ValueError(
            "No requested modules were found in the profile table."
        )

    vmax = max(
        2.0,
        float(
            np.nanpercentile(
                np.abs(finite),
                98,
            )
        ),
    )

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    fig = plt.figure(
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        )
    )

    row_counts = [
        max(len(nes), 1)
        for _, nes, _ in matrices
    ]

    # The extra constant gives each panel a consistent header band.
    height_ratios = [
        row_counts[0] + 1.4,
        row_counts[1] + 1.4,
        row_counts[2] + 2.2,
    ]

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=height_ratios,
        left=0.30,
        right=0.89,
        top=0.860,
        bottom=0.13,
        hspace=0.78,
    )

    axes = [
        fig.add_subplot(gs[i, 0])
        for i in range(3)
    ]

    images = []

    for i, (
        ax,
        (
            theme,
            nes,
            sig,
        ),
    ) in enumerate(
        zip(
            axes,
            matrices,
        )
    ):
        images.append(
            draw_panel(
                ax=ax,
                nes=nes,
                sig=sig,
                title=theme["title"],
                subtitle=theme["subtitle"],
                norm=norm,
                show_xlabels=(i == 2),
                highlight_amyloid=(i == 2),
            )
        )

    # -------------------------------------------------------------------------
    # Overall title block
    # -------------------------------------------------------------------------
    fig.suptitle(
        "Weighted AREA: key biological themes",
        fontsize=12.7,
        fontweight="bold",
        y=0.972,
    )

    fig.text(
        0.5,
        0.932,
        (
            "Prioritized modules grouped into the dominant "
            "cross-phenotype biological patterns"
        ),
        ha="center",
        fontsize=6.3,
        color="#444444",
    )

    # -------------------------------------------------------------------------
    # One shared colorbar
    # -------------------------------------------------------------------------
    cax = fig.add_axes(
        [
            0.915,
            0.355,
            0.015,
            0.28,
        ]
    )

    cbar = fig.colorbar(
        images[-1],
        cax=cax,
    )

    cbar.set_label(
        "Median module NES",
        fontsize=5.8,
    )

    cbar.ax.tick_params(
        labelsize=5.0,
    )

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    fig.text(
        0.5,
        0.035,
        (
            "Blue = lower expression with greater severity/pathology; "
            "red = higher expression. Dot = ≥50% of representative pathways significant."
        ),
        ha="center",
        fontsize=5.25,
        color="#555555",
    )

    png = (
        outdir
        / "weighted_area_biology_themes_heatmap_v7.png"
    )

    pdf = (
        outdir
        / "weighted_area_biology_themes_heatmap_v7.pdf"
    )

    used_csv = (
        outdir
        / "weighted_area_biology_themes_heatmap_v7_modules_used.csv"
    )

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    pd.DataFrame(
        used_rows
    ).drop_duplicates().to_csv(
        used_csv,
        index=False,
    )

    return png, pdf, used_csv


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print("=" * 100)
    print("WEIGHTED AREA BIOLOGY THEME HEATMAPS V6")
    print("=" * 100)
    print()

    profiles_path = find_profiles(
        args.profiles
    )

    outdir = (
        Path(args.outdir)
        if args.outdir
        else profiles_path.parent
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Profiles found: {profiles_path}"
    )
    print(
        f"Output directory: {outdir}"
    )
    print()

    profiles = load_profiles(
        profiles_path
    )

    print(
        f"Loaded {len(profiles):,} module × phenotype rows"
    )

    png, pdf, used_csv = make_figure(
        profiles,
        outdir,
    )

    print()
    print("DONE")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"Modules used: {used_csv}")


if __name__ == "__main__":
    main()

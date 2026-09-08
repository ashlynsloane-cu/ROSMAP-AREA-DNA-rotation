#!/usr/bin/env python3
"""
make_curated_weighted_area_gsea_figure_v5.py
============================================

Final alignment-cleanup version of the curated Weighted AREA GSEA heatmap.

Changes from v3
---------------
- replaces "–" with explicit "No" in DESeq2 / Regular AREA columns
- adds group header: "Endpoint AD vs NCI"
- keeps title above COGNITION / AMYLOID / TAU
- slightly reduces title size and adds more vertical spacing
- shortens the footnote
- preserves all NES values and pathway selections

Why one Regular AREA row is "No"
--------------------------------
The "Prefoldin → CCT/TRiC" pathway is NOT significant in Regular AREA GSEA
(FDR >= 0.05), even though it is significant across all 9 Weighted AREA
phenotypes. Therefore that row should correctly display "No" for Regular AREA.

Input:
    results/curated_weighted_area_gsea_figure/curated_pathway_panel.csv

Output:
    results/curated_weighted_area_gsea_figure_v5/
        curated_pathway_heatmap_v5.png
        curated_pathway_heatmap_v5.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHENOTYPES = [
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

DISPLAY_NAMES = [
    "Cognitive\nstage",
    "Global\ncognition",
    "MMSE",
    "CERAD\nequal",
    "CERAD\ncalibrated",
    "Amyloid\ncontinuous",
    "Braak\nequal",
    "Braak\ncalibrated",
    "Tangles\ncontinuous",
]

ROW_LABEL_OVERRIDES = {
    "Complex IV assembly": "Complex IV assembly",
    "Respiratory complex I assembly": "Complex I assembly",
    "PINK1–PRKN mitophagy": "PINK1–PRKN mitophagy",
    "Glutamate release cycle": "Glutamate release",
    "GABAergic synaptic transmission": "GABAergic transmission",
    "Post-NMDA receptor signaling": "Post-NMDA signaling",
    "tRNA aminoacylation": "tRNA aminoacylation",
    "Prefoldin → CCT/TRiC folding": "Prefoldin → CCT/TRiC",
    "Epigenetic regulation of gene expression": "Epigenetic regulation",
    "Chromatin remodeling": "Chromatin remodeling",
    "Integrin signaling": "Integrin signaling",
    "Regulation of apoptosis": "Apoptosis regulation",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "results/curated_weighted_area_gsea_figure/"
            "curated_pathway_panel.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        default="results/curated_weighted_area_gsea_figure_v5",
    )

    return parser.parse_args()


def as_bool(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def main():
    args = parse_args()

    infile = Path(args.input)

    if not infile.exists():
        raise FileNotFoundError(
            f"Missing curated pathway panel: {infile}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(infile)

    required = [
        "display_name",
        "curated_theme",
        "DESeq2_sig",
        "Regular_AREA_sig",
    ]

    for phenotype in PHENOTYPES:
        required.append(
            f"{phenotype}_NES"
        )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    for phenotype in PHENOTYPES:
        df[
            f"{phenotype}_NES"
        ] = pd.to_numeric(
            df[
                f"{phenotype}_NES"
            ],
            errors="coerce",
        )

    df["DESeq2_sig"] = as_bool(
        df["DESeq2_sig"]
    )

    df["Regular_AREA_sig"] = as_bool(
        df["Regular_AREA_sig"]
    )

    values = (
        df[
            [
                f"{phenotype}_NES"
                for phenotype in PHENOTYPES
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    n_rows = len(df)

    # ------------------------------------------------------------
    # Layout:
    # row 0 = title
    # row 1 = family + endpoint headers
    # row 2 = heatmap
    # ------------------------------------------------------------
    fig = plt.figure(
        figsize=(15.5, 9.6)
    )

    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            0.075,
            0.09,
            0.835,
        ],
        hspace=0.0,
    )

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    ax_title = fig.add_subplot(
        gs[0]
    )

    ax_title.axis(
        "off"
    )

    ax_title.text(
        0.5,
        0.55,
        (
            "Recurrent pathway associations across cognitive "
            "and neuropathologic phenotypes"
        ),
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        transform=ax_title.transAxes,
    )

    # ------------------------------------------------------------
    # Group headers
    # ------------------------------------------------------------
    ax_header = fig.add_subplot(
        gs[1]
    )

    ax_header.set_xlim(
        -0.5,
        10.8,
    )

    ax_header.set_ylim(
        0,
        1,
    )

    ax_header.axis(
        "off"
    )

    ax_header.text(
        1.0,
        0.40,
        "COGNITION",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax_header.text(
        4.0,
        0.40,
        "AMYLOID",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax_header.text(
        7.0,
        0.40,
        "TAU",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    # Endpoint comparison block.
    ax_header.text(
        9.65,
        0.83,
        "Endpoint AD vs NCI",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )

    ax_header.text(
        9.175,
        0.27,
        "DESeq2",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    ax_header.text(
        10.325,
        0.27,
        "Regular AREA",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    # ------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------
    ax = fig.add_subplot(
        gs[2]
    )

    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            len(PHENOTYPES)
        )
    )

    ax.set_xticklabels(
        DISPLAY_NAMES,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    row_labels = []

    for _, row in df.iterrows():
        display = ROW_LABEL_OVERRIDES.get(
            row["display_name"],
            row["display_name"],
        )

        row_labels.append(
            f"{row['curated_theme']}  |  {display}"
        )

    ax.set_yticks(
        np.arange(
            n_rows
        )
    )

    ax.set_yticklabels(
        row_labels,
        fontsize=9,
    )

    # Family separators.
    for xpos in [
        2.5,
        5.5,
    ]:
        ax.axvline(
            xpos,
            linewidth=1.3,
        )

    # Theme separators.
    themes = df[
        "curated_theme"
    ].tolist()

    for index in range(
        1,
        len(themes),
    ):
        if (
            themes[index]
            != themes[index - 1]
        ):
            ax.axhline(
                index - 0.5,
                linewidth=0.9,
            )

    # Endpoint comparison block.
    # Exact centers of the two method-summary columns:
    # DESeq2 spans x=8.5 to 9.85  -> center = 9.175
    # Regular AREA spans x=9.85 to 10.8 -> center = 10.325
    x_deseq = 9.175
    x_regular = 10.325

    ax.axvline(
        8.5,
        linewidth=1.3,
    )

    ax.axvline(
        9.85,
        linewidth=0.6,
    )

    for y in np.arange(
        -0.5,
        n_rows + 0.5,
        1.0,
    ):
        ax.plot(
            [
                8.5,
                10.8,
            ],
            [
                y,
                y,
            ],
            linewidth=0.35,
        )

    for index, row in df.iterrows():
        ax.text(
            x_deseq,
            index,
            (
                "Yes"
                if row["DESeq2_sig"]
                else "No"
            ),
            ha="center",
            va="center",
            fontsize=8.5,
        )

        ax.text(
            x_regular,
            index,
            (
                "Yes"
                if row[
                    "Regular_AREA_sig"
                ]
                else "No"
            ),
            ha="center",
            va="center",
            fontsize=8.5,
        )

    ax.set_xlim(
        -0.5,
        10.8,
    )

    # ------------------------------------------------------------
    # Color bar
    # ------------------------------------------------------------
    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.032,
        pad=0.025,
    )

    colorbar.set_label(
        "Normalized enrichment score (NES)",
        fontsize=10,
    )

    # ------------------------------------------------------------
    # Concise footnote
    # ------------------------------------------------------------
    fig.text(
        0.5,
        0.012,
        (
            "Cell color = Weighted AREA GSEA NES. "
            "Positive NES indicates higher expression with greater "
            "severity/pathology. Right columns show endpoint AD vs NCI "
            "GSEA significance."
        ),
        ha="center",
        fontsize=8.5,
    )

    fig.subplots_adjust(
        left=0.31,
        right=0.92,
        top=0.975,
        bottom=0.17,
    )

    png_path = (
        outdir
        / "curated_pathway_heatmap_v5.png"
    )

    pdf_path = (
        outdir
        / "curated_pathway_heatmap_v5.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # Also write a small audit table for the right-side method columns.
    audit = df[
        [
            "display_name",
            "DESeq2_sig",
            "Regular_AREA_sig",
        ]
    ].copy()

    audit.to_csv(
        outdir
        / "endpoint_method_significance_audit.csv",
        index=False,
    )

    print(
        f"Wrote: {png_path}"
    )

    print(
        f"Wrote: {pdf_path}"
    )

    print(
        f"Wrote: "
        f"{outdir / 'endpoint_method_significance_audit.csv'}"
    )

    # Explicitly flag any curated row not significant in Regular AREA.
    regular_no = audit.loc[
        ~audit[
            "Regular_AREA_sig"
        ]
    ]

    if not regular_no.empty:
        print(
            "\nCurated pathway(s) NOT significant in Regular AREA GSEA:"
        )
        print(
            regular_no[
                [
                    "display_name",
                    "Regular_AREA_sig",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()

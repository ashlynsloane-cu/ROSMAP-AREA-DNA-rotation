#!/usr/bin/env python3
"""
make_final_gsea_figures_v8.py
=============================

Generate TWO figures:

1) Fixed curated pathway heatmap
   - fixes header alignment by drawing ALL group/method headers in the SAME
     coordinate system as the heatmap itself
   - COGNITION / AMYLOID / TAU are centered exactly over their 3 columns
   - DESeq2 / Regular AREA are centered exactly over their summary columns
   - title remains above all headers

2) Phenotype-resolution comparison
   - compares equal / calibrated / continuous pathology encodings
   - separately for amyloid and tau
   - shows FDR-significant pathway counts for Hallmark, GO BP, Reactome

Inputs
------
Heatmap:
    results/curated_weighted_area_gsea_figure/curated_pathway_panel.csv

Resolution counts:
    results/weighted_area_full_phenotype_gsea_synthesis/
        gsea_count_summary_pivot.csv

Outputs
-------
results/final_gsea_figures_v8/
    curated_pathway_heatmap_v8.png
    curated_pathway_heatmap_v8.pdf
    pathology_resolution_pathway_counts.png
    pathology_resolution_pathway_counts.pdf
    pathology_resolution_plot_data.csv
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

LIBRARY_DISPLAY = {
    "Hallmark_2020": "Hallmark",
    "GO_Biological_Process_2025": "GO BP",
    "Reactome_Pathways_2024": "Reactome",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--heatmap-input",
        default=(
            "results/curated_weighted_area_gsea_figure/"
            "curated_pathway_panel.csv"
        ),
    )

    parser.add_argument(
        "--count-input",
        default=(
            "results/weighted_area_full_phenotype_gsea_synthesis/"
            "gsea_count_summary_pivot.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        default="results/final_gsea_figures_v8",
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


def make_heatmap(infile, outdir):
    infile = Path(infile)

    if not infile.exists():
        raise FileNotFoundError(
            f"Missing curated pathway panel: {infile}"
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
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Heatmap input missing columns: {missing}"
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

    values = df[
        [
            f"{phenotype}_NES"
            for phenotype in PHENOTYPES
        ]
    ].to_numpy(
        dtype=float
    )

    n_rows = len(df)

    fig, ax = plt.subplots(
        figsize=(15.5, 9.4)
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

    # ------------------------------------------------------------------
    # IMPORTANT ALIGNMENT FIX:
    # All group/method headers are drawn on THIS SAME AXIS.
    # Therefore they are centered in the exact same x coordinate system
    # as the cells/summary columns beneath them.
    # ------------------------------------------------------------------

    # Extend x-range for method-summary columns.
    method_left = 8.5
    method_split = 9.5
    method_right = 10.5

    ax.set_xlim(
        -0.5,
        method_right,
    )

    # Phenotype-family boundaries.
    for xpos in [
        2.5,
        5.5,
    ]:
        ax.axvline(
            xpos,
            linewidth=1.3,
        )

    # Boundary between heatmap and endpoint comparison.
    ax.axvline(
        method_left,
        linewidth=1.3,
    )

    ax.axvline(
        method_split,
        linewidth=0.6,
    )

    # Theme boundaries.
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

    # Method-summary row boundaries.
    for y in np.arange(
        -0.5,
        n_rows + 0.5,
        1.0,
    ):
        ax.plot(
            [
                method_left,
                method_right,
            ],
            [
                y,
                y,
            ],
            linewidth=0.35,
        )

    # Exact centers of the equal-width summary columns.
    x_deseq = (
        method_left
        + method_split
    ) / 2

    x_regular = (
        method_split
        + method_right
    ) / 2

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

    # ------------------------------------------------------------------
    # Headers use x in DATA coordinates and y in AXES coordinates.
    # This guarantees perfect centering over the columns below.
    # ------------------------------------------------------------------
    xaxis_transform = ax.get_xaxis_transform()

    ax.text(
        1.0,
        1.055,
        "COGNITION",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )

    ax.text(
        4.0,
        1.055,
        "AMYLOID",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )

    ax.text(
        7.0,
        1.055,
        "TAU",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )

    endpoint_center = (
        method_left
        + method_right
    ) / 2

    ax.text(
        endpoint_center,
        1.105,
        "Endpoint AD vs NCI",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )

    ax.text(
        x_deseq,
        1.055,
        "DESeq2",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )

    ax.text(
        x_regular,
        1.055,
        "Regular AREA",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )

    fig.suptitle(
        (
            "Recurrent pathway associations across cognitive "
            "and neuropathologic phenotypes"
        ),
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

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

    fig.text(
        0.5,
        0.018,
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
        top=0.84,
        bottom=0.17,
    )

    png = (
        outdir
        / "curated_pathway_heatmap_v8.png"
    )

    pdf = (
        outdir
        / "curated_pathway_heatmap_v8.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote: {png}"
    )
    print(
        f"Wrote: {pdf}"
    )


def make_resolution_figure(infile, outdir):
    infile = Path(infile)

    if not infile.exists():
        raise FileNotFoundError(
            f"Missing pathway-count table: {infile}"
        )

    df = pd.read_csv(infile)

    required = [
        "phenotype",
        "family",
        "Hallmark_2020",
        "GO_Biological_Process_2025",
        "Reactome_Pathways_2024",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Count input missing columns: {missing}"
        )

    phenotype_order = [
        "CERAD_equal",
        "CERAD_amyloid_calibrated",
        "amyloid_continuous",
        "Braak_equal",
        "Braak_tangle_calibrated",
        "tangle_continuous",
    ]

    encoding_label = {
        "CERAD_equal": "Equal",
        "CERAD_amyloid_calibrated": "Calibrated",
        "amyloid_continuous": "Continuous",
        "Braak_equal": "Equal",
        "Braak_tangle_calibrated": "Calibrated",
        "tangle_continuous": "Continuous",
    }

    family_label = {
        "CERAD_equal": "Amyloid",
        "CERAD_amyloid_calibrated": "Amyloid",
        "amyloid_continuous": "Amyloid",
        "Braak_equal": "Tau",
        "Braak_tangle_calibrated": "Tau",
        "tangle_continuous": "Tau",
    }

    plot_df = (
        df.loc[
            df[
                "phenotype"
            ].isin(
                phenotype_order
            )
        ]
        .copy()
        .set_index(
            "phenotype"
        )
        .loc[
            phenotype_order
        ]
        .reset_index()
    )

    plot_df["encoding"] = (
        plot_df[
            "phenotype"
        ].map(
            encoding_label
        )
    )

    plot_df["pathology_axis"] = (
        plot_df[
            "phenotype"
        ].map(
            family_label
        )
    )

    plot_df.to_csv(
        outdir
        / "pathology_resolution_plot_data.csv",
        index=False,
    )

    libraries = [
        "Hallmark_2020",
        "GO_Biological_Process_2025",
        "Reactome_Pathways_2024",
    ]

    # x positions:
    # Amyloid: 0,1,2
    # Tau: 4,5,6
    x = np.array(
        [
            0,
            1,
            2,
            4,
            5,
            6,
        ],
        dtype=float,
    )

    width = 0.22

    fig, ax = plt.subplots(
        figsize=(11.5, 7)
    )

    offsets = [
        -width,
        0.0,
        width,
    ]

    for library, offset in zip(
        libraries,
        offsets,
    ):
        values = pd.to_numeric(
            plot_df[
                library
            ],
            errors="coerce",
        ).to_numpy()

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=LIBRARY_DISPLAY[
                library
            ],
        )

        # Add exact pathway counts above each bar.
        for bar, value in zip(
            bars,
            values,
        ):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 6,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Equal",
            "Calibrated",
            "Continuous",
            "Equal",
            "Calibrated",
            "Continuous",
        ]
    )

    # These group headers are positioned in DATA coordinates on the SAME axis,
    # so they are centered exactly over their three encoding groups.
    transform = ax.get_xaxis_transform()

    ax.text(
        1.0,
        1.025,
        "AMYLOID",
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )

    ax.text(
        5.0,
        1.025,
        "TAU",
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )

    ax.set_ylabel(
        "FDR-significant pathways"
    )

    # Leave headroom for numeric labels above the tallest bars.
    ymax = np.nanmax(
        plot_df[
            libraries
        ].to_numpy(
            dtype=float
        )
    )

    ax.set_ylim(
        0,
        ymax * 1.12,
    )

    ax.set_title(
        (
            "Continuous pathology phenotypes reveal broader "
            "pathway-level associations"
        ),
        pad=34,
        fontsize=13,
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.text(
        0.5,
        0.02,
        (
            "Counts are genome-wide preranked GSEA pathways with FDR < 0.05. "
            "Equal and calibrated are ordinal pathology encodings; continuous "
            "uses quantitative amyloid or tangle burden."
        ),
        ha="center",
        fontsize=8.5,
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.16,
        left=0.11,
        right=0.97,
    )

    png = (
        outdir
        / "pathology_resolution_pathway_counts.png"
    )

    pdf = (
        outdir
        / "pathology_resolution_pathway_counts.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote: {png}"
    )
    print(
        f"Wrote: {pdf}"
    )


def main():
    args = parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_heatmap(
        args.heatmap_input,
        outdir,
    )

    make_resolution_figure(
        args.count_input,
        outdir,
    )


if __name__ == "__main__":
    main()

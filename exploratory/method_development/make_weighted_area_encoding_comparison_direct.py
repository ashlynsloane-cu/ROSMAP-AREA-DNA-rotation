#!/usr/bin/env python3
"""
make_weighted_area_encoding_comparison_direct.py
================================================

Direct, matched comparison of Weighted AREA phenotype encodings.

Compares Equal vs Calibrated vs Continuous separately for:
    - AMYLOID
    - TAU

This figure is intended to answer a methodological question BEFORE
interpreting biology:

    "How much does the Weighted AREA result change when we change
     only the phenotype encoding?"

Panels
------
Rows:
    AMYLOID
    TAU

Columns:
    1. Genome-wide gene-level Z-score concordance
       Spearman correlation across all genes.

    2. Significant-gene overlap
       Pairwise Jaccard similarity among FDR < 0.05 gene sets.

    3. Pathway-level discovery
       Number of FDR < 0.05 pathways in Hallmark, GO BP, and Reactome.

Expected Weighted AREA inputs
-----------------------------
results/weighted_area_full_phenotype_suite/
    CERAD_equal/results.csv
    CERAD_amyloid_calibrated/results.csv
    amyloid_continuous/results.csv
    Braak_equal/results.csv
    Braak_tangle_calibrated/results.csv
    tangle_continuous/results.csv

Expected pathway-count input
----------------------------
results/weighted_area_full_phenotype_gsea_synthesis/
    gsea_count_summary_pivot.csv

Outputs
-------
results/weighted_area_encoding_comparison/
    weighted_area_encoding_comparison_direct.png
    weighted_area_encoding_comparison_direct.pdf
    weighted_area_encoding_pairwise_metrics.csv
    weighted_area_encoding_gene_counts.csv
    weighted_area_encoding_pathway_counts.csv

Notes
-----
- Uses adjusted_AREA_Z for gene-level concordance.
- Significant genes are defined as adjusted_padj_BH < 0.05.
- Jaccard = intersection / union.
- This is a direct encoding comparison within Weighted AREA.
- It does NOT compare Weighted AREA against DESeq2 or Regular AREA.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
WEIGHTED_DIR = Path(
    "results/weighted_area_full_phenotype_suite"
)

GSEA_COUNTS = Path(
    "results/weighted_area_full_phenotype_gsea_synthesis/"
    "gsea_count_summary_pivot.csv"
)

OUTDIR = Path(
    "results/weighted_area_encoding_comparison"
)
OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG = OUTDIR / "weighted_area_encoding_comparison_direct.png"
PDF = OUTDIR / "weighted_area_encoding_comparison_direct.pdf"

PAIRWISE_CSV = OUTDIR / "weighted_area_encoding_pairwise_metrics.csv"
GENE_COUNTS_CSV = OUTDIR / "weighted_area_encoding_gene_counts.csv"
PATHWAY_COUNTS_CSV = OUTDIR / "weighted_area_encoding_pathway_counts.csv"


# -----------------------------------------------------------------------------
# Phenotype definitions
# -----------------------------------------------------------------------------
FAMILIES = {
    "Amyloid": {
        "Equal": "CERAD_equal",
        "Calibrated": "CERAD_amyloid_calibrated",
        "Continuous": "amyloid_continuous",
    },
    "Tau": {
        "Equal": "Braak_equal",
        "Calibrated": "Braak_tangle_calibrated",
        "Continuous": "tangle_continuous",
    },
}

ENCODING_ORDER = [
    "Equal",
    "Calibrated",
    "Continuous",
]

LIBRARIES = [
    "Hallmark_2020",
    "GO_Biological_Process_2025",
    "Reactome_Pathways_2024",
]

LIBRARY_LABELS = {
    "Hallmark_2020": "Hallmark",
    "GO_Biological_Process_2025": "GO BP",
    "Reactome_Pathways_2024": "Reactome",
}


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
TEXT = "#171717"
MUTED = "#555555"
GRID = "#D9D9D9"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------
def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


def resolve_column(
    df: pd.DataFrame,
    options: list[str],
    label: str,
) -> str:
    for col in options:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify {label}. "
        f"Available columns:\n{', '.join(df.columns)}"
    )


def load_weighted_result(
    phenotype: str,
) -> pd.DataFrame:
    path = (
        WEIGHTED_DIR
        / phenotype
        / "results.csv"
    )

    require(
        path
    )

    df = pd.read_csv(
        path
    )

    gene_col = resolve_column(
        df,
        [
            "gene_id",
            "ensembl_id_version",
            "ensembl_id",
        ],
        "gene ID column",
    )

    z_col = resolve_column(
        df,
        [
            "adjusted_AREA_Z",
        ],
        "Weighted AREA adjusted Z column",
    )

    q_col = resolve_column(
        df,
        [
            "adjusted_padj_BH",
        ],
        "Weighted AREA BH-FDR column",
    )

    out = df[
        [
            gene_col,
            z_col,
            q_col,
        ]
    ].copy()

    out.columns = [
        "gene_id",
        "z",
        "q",
    ]

    out["gene_id"] = (
        out["gene_id"]
        .astype(str)
    )

    out["z"] = pd.to_numeric(
        out["z"],
        errors="coerce",
    )

    out["q"] = pd.to_numeric(
        out["q"],
        errors="coerce",
    )

    out = (
        out.drop_duplicates(
            subset=[
                "gene_id",
            ]
        )
        .set_index(
            "gene_id"
        )
    )

    return out


def load_all_weighted():
    data = {}

    for family, mapping in FAMILIES.items():
        data[
            family
        ] = {}

        for encoding, phenotype in mapping.items():
            data[
                family
            ][
                encoding
            ] = load_weighted_result(
                phenotype
            )

    return data


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def spearman_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[float, int]:
    joined = (
        left[
            [
                "z",
            ]
        ]
        .rename(
            columns={
                "z": "left",
            }
        )
        .join(
            right[
                [
                    "z",
                ]
            ]
            .rename(
                columns={
                    "z": "right",
                }
            ),
            how="inner",
        )
        .dropna()
    )

    rho = (
        joined[
            "left"
        ]
        .corr(
            joined[
                "right"
            ],
            method="spearman",
        )
    )

    return (
        float(
            rho
        ),
        int(
            len(
                joined
            )
        ),
    )


def sig_set(
    df: pd.DataFrame,
) -> set[str]:
    return set(
        df.index[
            df[
                "q"
            ]
            < 0.05
        ]
    )


def jaccard(
    a: set[str],
    b: set[str],
) -> tuple[float, int, int]:
    intersection = (
        a
        & b
    )

    union = (
        a
        | b
    )

    if len(
        union
    ) == 0:
        score = np.nan
    else:
        score = (
            len(
                intersection
            )
            / len(
                union
            )
        )

    return (
        float(
            score
        ),
        int(
            len(
                intersection
            )
        ),
        int(
            len(
                union
            )
        ),
    )


def direction_concordance(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[float, int]:
    joined = (
        left[
            [
                "z",
            ]
        ]
        .rename(
            columns={
                "z": "left",
            }
        )
        .join(
            right[
                [
                    "z",
                ]
            ]
            .rename(
                columns={
                    "z": "right",
                }
            ),
            how="inner",
        )
        .dropna()
    )

    nonzero = joined[
        (
            joined[
                "left"
            ]
            != 0
        )
        & (
            joined[
                "right"
            ]
            != 0
        )
    ]

    if len(
        nonzero
    ) == 0:
        return (
            np.nan,
            0,
        )

    same = (
        np.sign(
            nonzero[
                "left"
            ]
        )
        == np.sign(
            nonzero[
                "right"
            ]
        )
    )

    return (
        float(
            same.mean()
        ),
        int(
            len(
                nonzero
            )
        ),
    )


def make_pairwise_metrics(
    data,
) -> pd.DataFrame:
    rows = []

    for family in FAMILIES:
        for i, left_name in enumerate(
            ENCODING_ORDER
        ):
            for right_name in ENCODING_ORDER[
                i
                + 1:
            ]:
                left = data[
                    family
                ][
                    left_name
                ]

                right = data[
                    family
                ][
                    right_name
                ]

                rho, n_common = spearman_pair(
                    left,
                    right,
                )

                left_sig = sig_set(
                    left
                )

                right_sig = sig_set(
                    right
                )

                jac, intersection, union = jaccard(
                    left_sig,
                    right_sig,
                )

                concordance, n_direction = direction_concordance(
                    left,
                    right,
                )

                rows.append(
                    {
                        "family": family,
                        "encoding_1": left_name,
                        "encoding_2": right_name,
                        "n_common_genes": n_common,
                        "spearman_Z": rho,
                        "significant_intersection": intersection,
                        "significant_union": union,
                        "significant_jaccard": jac,
                        "direction_concordance": concordance,
                        "n_direction_compared": n_direction,
                    }
                )

    return pd.DataFrame(
        rows
    )


def make_gene_counts(
    data,
) -> pd.DataFrame:
    rows = []

    for family in FAMILIES:
        for encoding in ENCODING_ORDER:
            df = data[
                family
            ][
                encoding
            ]

            rows.append(
                {
                    "family": family,
                    "encoding": encoding,
                    "n_genes": len(
                        df
                    ),
                    "n_fdr_significant": int(
                        (
                            df[
                                "q"
                            ]
                            < 0.05
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# -----------------------------------------------------------------------------
# GSEA pathway counts
# -----------------------------------------------------------------------------
def load_pathway_counts() -> pd.DataFrame:
    require(
        GSEA_COUNTS
    )

    df = pd.read_csv(
        GSEA_COUNTS
    )

    required = [
        "phenotype",
        *LIBRARIES,
    ]

    missing = [
        col
        for col in required
        if col
        not in df.columns
    ]

    if missing:
        raise ValueError(
            "GSEA count table is missing columns: "
            + ", ".join(
                missing
            )
        )

    phenotype_to_family_encoding = {}

    for family, mapping in FAMILIES.items():
        for encoding, phenotype in mapping.items():
            phenotype_to_family_encoding[
                phenotype
            ] = (
                family,
                encoding,
            )

    rows = []

    for _, row in df.iterrows():
        phenotype = str(
            row[
                "phenotype"
            ]
        )

        if phenotype not in phenotype_to_family_encoding:
            continue

        family, encoding = phenotype_to_family_encoding[
            phenotype
        ]

        for library in LIBRARIES:
            rows.append(
                {
                    "family": family,
                    "encoding": encoding,
                    "library": library,
                    "library_label": LIBRARY_LABELS[
                        library
                    ],
                    "n_fdr_significant": pd.to_numeric(
                        row[
                            library
                        ],
                        errors="coerce",
                    ),
                }
            )

    out = pd.DataFrame(
        rows
    )

    if len(
        out
    ) == 0:
        raise ValueError(
            "No amyloid/tau encoding rows were found "
            "in the GSEA count table."
        )

    return out


# -----------------------------------------------------------------------------
# Matrix helpers
# -----------------------------------------------------------------------------
def correlation_matrix(
    data,
    family: str,
) -> np.ndarray:
    matrix = np.eye(
        len(
            ENCODING_ORDER
        )
    )

    for i, left in enumerate(
        ENCODING_ORDER
    ):
        for j, right in enumerate(
            ENCODING_ORDER
        ):
            if i == j:
                continue

            rho, _ = spearman_pair(
                data[
                    family
                ][
                    left
                ],
                data[
                    family
                ][
                    right
                ],
            )

            matrix[
                i,
                j,
            ] = rho

    return matrix


def jaccard_matrix(
    data,
    family: str,
) -> np.ndarray:
    matrix = np.eye(
        len(
            ENCODING_ORDER
        )
    )

    sig = {
        encoding: sig_set(
            data[
                family
            ][
                encoding
            ]
        )
        for encoding in ENCODING_ORDER
    }

    for i, left in enumerate(
        ENCODING_ORDER
    ):
        for j, right in enumerate(
            ENCODING_ORDER
        ):
            if i == j:
                continue

            score, _, _ = jaccard(
                sig[
                    left
                ],
                sig[
                    right
                ],
            )

            matrix[
                i,
                j,
            ] = score

    return matrix


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_heatmap(
    ax,
    matrix,
    title,
    vmin,
    vmax,
    fmt,
    panel_letter=None,
):
    image = ax.imshow(
        matrix,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            len(
                ENCODING_ORDER
            )
        )
    )

    ax.set_yticks(
        np.arange(
            len(
                ENCODING_ORDER
            )
        )
    )

    ax.set_xticklabels(
        ENCODING_ORDER,
        rotation=35,
        ha="right",
    )

    ax.set_yticklabels(
        ENCODING_ORDER,
    )

    ax.set_title(
        title,
        pad=10,
    )

    for i in range(
        matrix.shape[
            0
        ]
    ):
        for j in range(
            matrix.shape[
                1
            ]
        ):
            value = matrix[
                i,
                j
            ]

            if np.isnan(
                value
            ):
                label = "NA"
            else:
                label = format(
                    value,
                    fmt,
                )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight=(
                    "bold"
                    if i
                    != j
                    else "normal"
                ),
            )

    if panel_letter:
        ax.text(
            -0.26,
            1.12,
            panel_letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

    return image


def plot_pathway_bars(
    ax,
    pathway_counts: pd.DataFrame,
    family: str,
    panel_letter=None,
):
    sub = pathway_counts[
        pathway_counts[
            "family"
        ]
        == family
    ].copy()

    x = np.arange(
        len(
            ENCODING_ORDER
        )
    )

    width = 0.22

    offsets = [
        -width,
        0.0,
        width,
    ]

    for library, offset in zip(
        LIBRARIES,
        offsets,
    ):
        values = []

        for encoding in ENCODING_ORDER:
            row = sub[
                (
                    sub[
                        "encoding"
                    ]
                    == encoding
                )
                & (
                    sub[
                        "library"
                    ]
                    == library
                )
            ]

            if len(
                row
            ) != 1:
                values.append(
                    np.nan
                )
            else:
                values.append(
                    float(
                        row.iloc[
                            0
                        ][
                            "n_fdr_significant"
                        ]
                    )
                )

        bars = ax.bar(
            x
            + offset,
            values,
            width=width,
            label=LIBRARY_LABELS[
                library
            ],
        )

        for bar, value in zip(
            bars,
            values,
        ):
            if np.isfinite(
                value
            ):
                ax.text(
                    bar.get_x()
                    + bar.get_width()
                    / 2,
                    bar.get_height()
                    + max(
                        values
                        + [
                            1,
                        ]
                    )
                    * 0.02,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        ENCODING_ORDER,
    )

    ax.set_ylabel(
        "FDR-significant pathways"
    )

    ax.set_title(
        "Pathway-level discovery",
        pad=10,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    if panel_letter:
        ax.text(
            -0.16,
            1.12,
            panel_letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )


def main():
    data = load_all_weighted()

    pairwise = make_pairwise_metrics(
        data
    )

    gene_counts = make_gene_counts(
        data
    )

    pathway_counts = load_pathway_counts()

    pairwise.to_csv(
        PAIRWISE_CSV,
        index=False,
    )

    gene_counts.to_csv(
        GENE_COUNTS_CSV,
        index=False,
    )

    pathway_counts.to_csv(
        PATHWAY_COUNTS_CSV,
        index=False,
    )

    fig = plt.figure(
        figsize=(
            15.5,
            9.2,
        )
    )

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[
            1.0,
            1.0,
            1.35,
        ],
        height_ratios=[
            1.0,
            1.0,
        ],
        hspace=0.42,
        wspace=0.38,
    )

    family_order = [
        "Amyloid",
        "Tau",
    ]

    letters = [
        [
            "A",
            "B",
            "C",
        ],
        [
            "D",
            "E",
            "F",
        ],
    ]

    corr_images = []
    jac_images = []

    for row_idx, family in enumerate(
        family_order
    ):
        ax_corr = fig.add_subplot(
            gs[
                row_idx,
                0,
            ]
        )

        ax_jac = fig.add_subplot(
            gs[
                row_idx,
                1,
            ]
        )

        ax_bar = fig.add_subplot(
            gs[
                row_idx,
                2,
            ]
        )

        corr = correlation_matrix(
            data,
            family,
        )

        jac = jaccard_matrix(
            data,
            family,
        )

        corr_image = plot_heatmap(
            ax_corr,
            corr,
            title="Genome-wide Z-score concordance",
            vmin=0.0,
            vmax=1.0,
            fmt=".3f",
            panel_letter=letters[
                row_idx
            ][
                0
            ],
        )

        jac_image = plot_heatmap(
            ax_jac,
            jac,
            title="FDR-significant gene-set Jaccard",
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            panel_letter=letters[
                row_idx
            ][
                1
            ],
        )

        plot_pathway_bars(
            ax_bar,
            pathway_counts,
            family,
            panel_letter=letters[
                row_idx
            ][
                2
            ],
        )

        corr_images.append(
            corr_image
        )

        jac_images.append(
            jac_image
        )

        # Row label placed outside axes for clear family grouping.
        fig.text(
            0.015,
            0.707
            if row_idx
            == 0
            else 0.295,
            family.upper(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    # Shared colorbars for the two matrix columns.
    cbar_ax1 = fig.add_axes(
        [
            0.304,
            0.075,
            0.015,
            0.17,
        ]
    )

    cb1 = fig.colorbar(
        corr_images[
            0
        ],
        cax=cbar_ax1,
    )

    cb1.set_label(
        "Spearman ρ",
        fontsize=9,
    )

    cbar_ax2 = fig.add_axes(
        [
            0.559,
            0.075,
            0.015,
            0.17,
        ]
    )

    cb2 = fig.colorbar(
        jac_images[
            0
        ],
        cax=cbar_ax2,
    )

    cb2.set_label(
        "Jaccard similarity",
        fontsize=9,
    )

    # Only one legend for pathway libraries.
    bar_axes = [
        ax
        for ax in fig.axes
        if ax.get_ylabel()
        == "FDR-significant pathways"
    ]

    if bar_axes:
        handles, labels = bar_axes[
            0
        ].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.82,
                0.928,
            ),
            ncol=3,
            frameon=False,
            fontsize=9.5,
        )

    fig.suptitle(
        "Direct comparison of Weighted AREA phenotype encodings",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.951,
        (
            "Equal vs calibrated vs continuous encodings, "
            "evaluated separately for amyloid and tau"
        ),
        ha="center",
        fontsize=10.5,
        color="#3F3F3F",
    )

    fig.text(
        0.5,
        0.018,
        (
            "Gene-level concordance uses adjusted Weighted AREA Z scores; "
            "significant-gene overlap uses BH FDR < 0.05. "
            "Pathway counts are preranked GSEA terms with FDR < 0.05."
        ),
        ha="center",
        fontsize=8.6,
        color=MUTED,
    )

    fig.subplots_adjust(
        top=0.89,
        bottom=0.12,
        left=0.075,
        right=0.97,
    )

    fig.savefig(
        PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "="
        * 96
    )
    print(
        "DIRECT WEIGHTED AREA ENCODING COMPARISON"
    )
    print(
        "="
        * 96
    )
    print()
    print(
        "PAIRWISE METRICS"
    )
    print(
        pairwise.to_string(
            index=False
        )
    )
    print()
    print(
        "GENE COUNTS"
    )
    print(
        gene_counts.to_string(
            index=False
        )
    )
    print()
    print(
        f"Wrote: {PNG}"
    )
    print(
        f"Wrote: {PDF}"
    )
    print(
        f"Wrote: {PAIRWISE_CSV}"
    )
    print(
        f"Wrote: {GENE_COUNTS_CSV}"
    )
    print(
        f"Wrote: {PATHWAY_COUNTS_CSV}"
    )


if __name__ == "__main__":
    main()

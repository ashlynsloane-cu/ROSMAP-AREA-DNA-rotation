#!/usr/bin/env python3
"""
make_weighted_area_encoding_comparison_direct_paper_v3.py
=========================================================

Publication-sized direct comparison of Weighted AREA phenotype encodings.

Design goals
------------
- Two-column manuscript width (~7.2 inches)
- No manually overlaid colorbars
- No legends overlapping panel titles
- Consistent 2 x 3 panel geometry
- Clear AMYLOID and TAU row labels
- Compact typography suitable for paper figures
- Same data and metrics as the prior direct-comparison figure

Panels
------
A/D: genome-wide adjusted-Z Spearman concordance
B/E: FDR-significant gene-set Jaccard similarity
C/F: number of FDR-significant GSEA pathways

Expected inputs
---------------
results/weighted_area_full_phenotype_suite/
    CERAD_equal/results.csv
    CERAD_amyloid_calibrated/results.csv
    amyloid_continuous/results.csv
    Braak_equal/results.csv
    Braak_tangle_calibrated/results.csv
    tangle_continuous/results.csv

results/weighted_area_full_phenotype_gsea_synthesis/
    gsea_count_summary_pivot.csv

Outputs
-------
results/weighted_area_encoding_comparison/
    weighted_area_encoding_comparison_direct_paper_v3.png
    weighted_area_encoding_comparison_direct_paper_v3.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WEIGHTED_DIR = Path("results/weighted_area_full_phenotype_suite")
GSEA_COUNTS = Path(
    "results/weighted_area_full_phenotype_gsea_synthesis/"
    "gsea_count_summary_pivot.csv"
)

OUTDIR = Path("results/weighted_area_encoding_comparison")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "weighted_area_encoding_comparison_direct_paper_publication.png"
PDF = OUTDIR / "weighted_area_encoding_comparison_direct_paper_publication.pdf"
SVG = OUTDIR / "weighted_area_encoding_comparison_direct_paper_publication.svg"

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

ENCODINGS = ["Equal", "Calibrated", "Continuous"]

LIBRARIES = [
    "Hallmark_2020",
    "GO_Biological_Process_2025",
    "Reactome_Pathways_2024",
]
LIB_LABELS = {
    "Hallmark_2020": "Hallmark",
    "GO_Biological_Process_2025": "GO BP",
    "Reactome_Pathways_2024": "Reactome",
}

TEXT = "#181818"
MUTED = "#5A5A5A"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.titlesize": 7.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.7,
    }
)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def resolve_column(df, options, label):
    for col in options:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not identify {label}. Available columns:\n"
        + ", ".join(df.columns)
    )


def load_weighted_result(phenotype: str) -> pd.DataFrame:
    path = WEIGHTED_DIR / phenotype / "results.csv"
    require(path)

    df = pd.read_csv(path)

    gene_col = resolve_column(
        df,
        ["gene_id", "ensembl_id_version", "ensembl_id"],
        "gene ID column",
    )
    z_col = resolve_column(df, ["adjusted_AREA_Z"], "adjusted AREA Z column")
    q_col = resolve_column(df, ["adjusted_padj_BH"], "BH FDR column")

    out = df[[gene_col, z_col, q_col]].copy()
    out.columns = ["gene_id", "z", "q"]

    out["gene_id"] = out["gene_id"].astype(str)
    out["z"] = pd.to_numeric(out["z"], errors="coerce")
    out["q"] = pd.to_numeric(out["q"], errors="coerce")

    return out.drop_duplicates("gene_id").set_index("gene_id")


def load_all_weighted():
    data = {}
    for family, mapping in FAMILIES.items():
        data[family] = {}
        for encoding, phenotype in mapping.items():
            data[family][encoding] = load_weighted_result(phenotype)
    return data


def spearman_pair(left: pd.DataFrame, right: pd.DataFrame) -> float:
    joined = (
        left[["z"]]
        .rename(columns={"z": "left"})
        .join(
            right[["z"]].rename(columns={"z": "right"}),
            how="inner",
        )
        .dropna()
    )
    return float(joined["left"].corr(joined["right"], method="spearman"))


def sig_set(df: pd.DataFrame) -> set[str]:
    return set(df.index[df["q"] < 0.05])


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def correlation_matrix(data, family: str) -> np.ndarray:
    mat = np.eye(3)
    for i, left in enumerate(ENCODINGS):
        for j, right in enumerate(ENCODINGS):
            if i == j:
                continue
            mat[i, j] = spearman_pair(
                data[family][left],
                data[family][right],
            )
    return mat


def jaccard_matrix(data, family: str) -> np.ndarray:
    sig = {
        encoding: sig_set(data[family][encoding])
        for encoding in ENCODINGS
    }

    mat = np.eye(3)
    for i, left in enumerate(ENCODINGS):
        for j, right in enumerate(ENCODINGS):
            if i == j:
                continue
            mat[i, j] = jaccard(sig[left], sig[right])
    return mat


def load_pathway_counts() -> pd.DataFrame:
    require(GSEA_COUNTS)
    df = pd.read_csv(GSEA_COUNTS)

    needed = ["phenotype", *LIBRARIES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns in GSEA count table: "
            + ", ".join(missing)
        )

    lookup = {}
    for family, mapping in FAMILIES.items():
        for encoding, phenotype in mapping.items():
            lookup[phenotype] = (family, encoding)

    rows = []

    for _, row in df.iterrows():
        phenotype = str(row["phenotype"])
        if phenotype not in lookup:
            continue

        family, encoding = lookup[phenotype]

        for library in LIBRARIES:
            rows.append(
                {
                    "family": family,
                    "encoding": encoding,
                    "library": library,
                    "count": pd.to_numeric(
                        row[library],
                        errors="coerce",
                    ),
                }
            )

    return pd.DataFrame(rows)


def annotate_matrix(ax, matrix, decimals: int):
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            fmt = f"{{:.{decimals}f}}"
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=6.35,
                fontweight="bold" if i != j else "normal",
                color="black",
            )


def plot_heatmap(ax, matrix, title, decimals, panel_letter):
    ax.imshow(
        matrix,
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(ENCODINGS, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ENCODINGS)

    ax.set_title(title, pad=6)

    annotate_matrix(
        ax,
        matrix,
        decimals=decimals,
    )

    ax.text(
        -0.18,
        1.065,
        panel_letter,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_pathway_bars(
    ax,
    pathway_counts,
    family,
    panel_letter,
    show_legend=False,
):
    sub = pathway_counts[
        pathway_counts["family"] == family
    ]

    x = np.arange(3)
    width = 0.22

    for k, library in enumerate(LIBRARIES):
        values = []

        for encoding in ENCODINGS:
            row = sub[
                (sub["encoding"] == encoding)
                & (sub["library"] == library)
            ]

            if len(row) == 1:
                values.append(float(row.iloc[0]["count"]))
            else:
                values.append(np.nan)

        offset = (k - 1) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=LIB_LABELS[library],
        )

        ymax = np.nanmax(values) if np.any(np.isfinite(values)) else 1

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(2, ymax * 0.018),
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=5.25,
                )

    ax.set_title("Pathway-level discovery", pad=6)
    ax.set_ylabel("FDR-significant pathways")

    ax.set_xticks(x)
    ax.set_xticklabels(ENCODINGS)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.margins(y=0.12)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    ax.text(
        -0.16,
        1.065,
        panel_letter,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
    )

    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=3,
            columnspacing=0.65,
            handlelength=1.35,
            handletextpad=0.35,
            borderaxespad=0.0,
        )


def main():
    data = load_all_weighted()
    pathway_counts = load_pathway_counts()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 6.15),
        gridspec_kw={
            "width_ratios": [1.0, 1.0, 1.16],
            "wspace": 0.36,
            "hspace": 0.66,
        },
    )

    family_order = ["Amyloid", "Tau"]
    letters = [["A", "B", "C"], ["D", "E", "F"]]

    for row, family in enumerate(family_order):
        corr = correlation_matrix(data, family)
        jac = jaccard_matrix(data, family)

        plot_heatmap(
            axes[row, 0],
            corr,
            "Genome-wide Z-score concordance",
            decimals=3,
            panel_letter=letters[row][0],
        )

        plot_heatmap(
            axes[row, 1],
            jac,
            "FDR-significant gene-set Jaccard",
            decimals=2,
            panel_letter=letters[row][1],
        )

        plot_pathway_bars(
            axes[row, 2],
            pathway_counts,
            family,
            panel_letter=letters[row][2],
            show_legend=(row == 0),
        )

    # Row labels outside the axes.
    fig.text(
        0.026,
        0.655,
        "AMYLOID",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )

    fig.text(
        0.022,
        0.300,
        "TAU",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )

    fig.suptitle(
        "Direct comparison of Weighted AREA phenotype encodings",
        fontsize=10.2,
        fontweight="bold",
        y=0.992,
    )

    fig.text(
        0.5,
        0.948,
        (
            "Equal vs calibrated vs continuous encodings, "
            "evaluated separately for amyloid and tau"
        ),
        ha="center",
        fontsize=6.15,
        color="#3F3F3F",
    )

    fig.text(
        0.5,
        0.022,
        (
            "Heatmaps are scaled from 0 (low similarity) to 1 (identical). "
            "Gene-level concordance uses adjusted Weighted AREA Z scores; significant-gene overlap uses BH FDR < 0.05. "
            "Pathway counts are preranked GSEA terms with FDR < 0.05."
        ),
        ha="center",
        fontsize=5.0,
        color=MUTED,
    )

    fig.subplots_adjust(
        top=0.825,
        bottom=0.165,
        left=0.105,
        right=0.985,
    )

    fig.savefig(
        PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        SVG,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote: {PNG}")
    print(f"Wrote: {PDF}")
    print(f"Wrote: {SVG}")


if __name__ == "__main__":
    main()

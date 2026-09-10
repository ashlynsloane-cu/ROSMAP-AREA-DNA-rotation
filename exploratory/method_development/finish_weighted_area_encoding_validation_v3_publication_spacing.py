#!/usr/bin/env python3
"""
finish_weighted_area_encoding_validation_v3.py
==============================================

Final Weighted AREA encoding-validation script using the ACTUAL GSEA layout
in this repository.

This fixes the failed auto-discovery in v2 by reading GSEA files directly from:

results/weighted_area_full_phenotype_gsea/<LIBRARY>/Weighted_<PHENOTYPE>/gsea_all_results.csv

Libraries:
    Hallmark_2020
    GO_Biological_Process_2025
    Reactome_Pathways_2024

Phenotypes:
    CERAD_equal
    CERAD_amyloid_calibrated
    amyloid_continuous
    Braak_equal
    Braak_tangle_calibrated
    tangle_continuous

The script produces:
1. cleaned gene-level encoding-validation figure
2. pathway NES-stability figure
3. gene-level summary tables
4. pathway-level summary tables
5. compact combined validation table

Outputs:
results/weighted_area_encoding_validation/
    weighted_area_encoding_validation_summary_v3.png
    weighted_area_encoding_validation_summary_v3.pdf
    weighted_area_pathway_NES_stability_v3.png
    weighted_area_pathway_NES_stability_v3.pdf
    encoding_gene_counts_v3.csv
    encoding_gene_pairwise_metrics_v3.csv
    encoding_pathway_significant_counts_v3.csv
    encoding_pathway_pairwise_metrics_v3.csv
    encoding_validation_compact_table_v3.csv
    encoding_validation_manifest_v3.txt
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

WEIGHTED_DIR = Path("results/weighted_area_full_phenotype_suite")
GSEA_DIR = Path("results/weighted_area_full_phenotype_gsea")

OUTDIR = Path("results/weighted_area_encoding_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

GENE_FIG_PNG = OUTDIR / "weighted_area_encoding_validation_summary_v3.png"
GENE_FIG_PDF = OUTDIR / "weighted_area_encoding_validation_summary_v3.pdf"

PATHWAY_FIG_PNG = OUTDIR / "weighted_area_pathway_NES_stability_v3.png"
PATHWAY_FIG_PDF = OUTDIR / "weighted_area_pathway_NES_stability_v3.pdf"

GENE_COUNTS_CSV = OUTDIR / "encoding_gene_counts_v3.csv"
GENE_PAIRWISE_CSV = OUTDIR / "encoding_gene_pairwise_metrics_v3.csv"
PATHWAY_COUNTS_CSV = OUTDIR / "encoding_pathway_significant_counts_v3.csv"
PATHWAY_PAIRWISE_CSV = OUTDIR / "encoding_pathway_pairwise_metrics_v3.csv"
COMPACT_CSV = OUTDIR / "encoding_validation_compact_table_v3.csv"
MANIFEST_TXT = OUTDIR / "encoding_validation_manifest_v3.txt"


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

PAIR_ORDER = [
    ("Equal", "Calibrated"),
    ("Equal", "Continuous"),
    ("Calibrated", "Continuous"),
]

LIBRARIES = {
    "Hallmark": "Hallmark_2020",
    "GO BP": "GO_Biological_Process_2025",
    "Reactome": "Reactome_Pathways_2024",
}

TEXT = "#171717"
MUTED = "#5A5A5A"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.titlesize": 7.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.3,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.4,
        "axes.linewidth": 0.7,
    }
)


# =============================================================================
# HELPERS
# =============================================================================

def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def resolve_column(df: pd.DataFrame, options: list[str], label: str) -> str:
    for col in options:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify {label}. Available columns:\n"
        + ", ".join(df.columns)
    )


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(pair) < 3:
        return np.nan

    return float(pair["x"].corr(pair["y"], method="spearman"))


def jaccard(a: set[str], b: set[str]) -> tuple[float, int, int]:
    inter = a & b
    union = a | b

    if not union:
        return np.nan, 0, 0

    return len(inter) / len(union), len(inter), len(union)


# =============================================================================
# GENE LEVEL
# =============================================================================

def load_weighted_result(phenotype: str) -> pd.DataFrame:
    path = WEIGHTED_DIR / phenotype / "results.csv"
    require(path)

    df = pd.read_csv(path)

    gene_col = resolve_column(
        df,
        ["gene_id", "ensembl_id_version", "ensembl_id"],
        "gene ID column",
    )

    z_col = resolve_column(
        df,
        ["adjusted_AREA_Z"],
        "Weighted AREA adjusted Z",
    )

    q_col = resolve_column(
        df,
        ["adjusted_padj_BH"],
        "Weighted AREA BH-FDR",
    )

    out = df[[gene_col, z_col, q_col]].copy()
    out.columns = ["gene_id", "z", "q"]

    out["gene_id"] = out["gene_id"].astype(str)
    out["z"] = pd.to_numeric(out["z"], errors="coerce")
    out["q"] = pd.to_numeric(out["q"], errors="coerce")

    return (
        out.drop_duplicates("gene_id")
        .set_index("gene_id")
        .sort_index()
    )


def load_all_weighted() -> dict:
    data = {}

    for family, mapping in FAMILIES.items():
        data[family] = {}

        for encoding, phenotype in mapping.items():
            data[family][encoding] = load_weighted_result(phenotype)

    return data


def significant_gene_set(df: pd.DataFrame) -> set[str]:
    return set(df.index[df["q"] < 0.05])


def direction_concordance(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[float, int]:
    joined = (
        left[["z"]]
        .rename(columns={"z": "z1"})
        .join(
            right[["z"]].rename(columns={"z": "z2"}),
            how="inner",
        )
        .dropna()
    )

    joined = joined[
        (joined["z1"] != 0)
        & (joined["z2"] != 0)
    ]

    if len(joined) == 0:
        return np.nan, 0

    same = np.sign(joined["z1"]) == np.sign(joined["z2"])

    return float(same.mean()), int(len(joined))


def make_gene_counts(data: dict) -> pd.DataFrame:
    rows = []

    for family in FAMILIES:
        for encoding in ENCODINGS:
            df = data[family][encoding]

            rows.append(
                {
                    "family": family,
                    "encoding": encoding,
                    "n_genes_tested": int(df["z"].notna().sum()),
                    "n_fdr_significant": int((df["q"] < 0.05).sum()),
                }
            )

    return pd.DataFrame(rows)


def make_gene_pairwise(data: dict) -> pd.DataFrame:
    rows = []

    for family in FAMILIES:
        for enc1, enc2 in PAIR_ORDER:
            left = data[family][enc1]
            right = data[family][enc2]

            joined = (
                left[["z"]]
                .rename(columns={"z": "z1"})
                .join(
                    right[["z"]].rename(columns={"z": "z2"}),
                    how="inner",
                )
                .dropna()
            )

            sig1 = significant_gene_set(left)
            sig2 = significant_gene_set(right)

            jac, inter, union = jaccard(sig1, sig2)
            direction, n_direction = direction_concordance(left, right)

            rows.append(
                {
                    "family": family,
                    "encoding_1": enc1,
                    "encoding_2": enc2,
                    "n_common_genes": int(len(joined)),
                    "spearman_Z": safe_spearman(joined["z1"], joined["z2"]),
                    "direction_concordance": direction,
                    "n_direction_compared": n_direction,
                    "sig_genes_1": len(sig1),
                    "sig_genes_2": len(sig2),
                    "sig_gene_intersection": inter,
                    "sig_gene_union": union,
                    "sig_gene_jaccard": jac,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# PATHWAY LEVEL - DIRECT PATHS
# =============================================================================

def gsea_path(
    phenotype: str,
    library_dir: str,
) -> Path:
    return (
        GSEA_DIR
        / library_dir
        / f"Weighted_{phenotype}"
        / "gsea_all_results.csv"
    )


def load_gsea_table(
    phenotype: str,
    library_dir: str,
) -> pd.DataFrame:
    path = gsea_path(phenotype, library_dir)
    require(path)

    df = pd.read_csv(path)

    term_col = resolve_column(
        df,
        [
            "Term",
            "term",
            "Pathway",
            "pathway",
            "Name",
            "name",
            "Gene_set",
            "gene_set",
        ],
        "GSEA pathway/term column",
    )

    nes_col = resolve_column(
        df,
        [
            "NES",
            "nes",
            "Normalized Enrichment Score",
            "normalized_enrichment_score",
        ],
        "GSEA NES column",
    )

    q_col = resolve_column(
        df,
        [
            "FDR_qvalue",
            "FDR q-val",
            "FDR q-value",
            "FDR",
            "fdr",
            "qvalue",
            "q_value",
            "padj",
            "adjusted_pvalue",
        ],
        "GSEA FDR/q-value column",
    )

    out = df[[term_col, nes_col, q_col]].copy()
    out.columns = ["term", "NES", "q"]

    out["term"] = out["term"].astype(str).str.strip()
    out["NES"] = pd.to_numeric(out["NES"], errors="coerce")
    out["q"] = pd.to_numeric(out["q"], errors="coerce")

    return (
        out.dropna(subset=["term", "NES"])
        .drop_duplicates("term")
        .set_index("term")
        .sort_index()
    )


def load_all_gsea() -> dict:
    gsea = {}

    for family, mapping in FAMILIES.items():
        gsea[family] = {}

        for encoding, phenotype in mapping.items():
            gsea[family][encoding] = {}

            for library_label, library_dir in LIBRARIES.items():
                gsea[family][encoding][library_label] = load_gsea_table(
                    phenotype,
                    library_dir,
                )

    return gsea


def significant_pathway_set(df: pd.DataFrame) -> set[str]:
    return set(df.index[df["q"] < 0.05])


def make_pathway_counts(gsea: dict) -> pd.DataFrame:
    rows = []

    for family in FAMILIES:
        for encoding in ENCODINGS:
            for library in LIBRARIES:
                df = gsea[family][encoding][library]

                rows.append(
                    {
                        "family": family,
                        "encoding": encoding,
                        "library": library,
                        "n_pathways_tested": int(df["NES"].notna().sum()),
                        "n_fdr_significant": int((df["q"] < 0.05).sum()),
                    }
                )

    return pd.DataFrame(rows)


def make_pathway_pairwise(gsea: dict) -> pd.DataFrame:
    rows = []

    for family in FAMILIES:
        for library in LIBRARIES:
            for enc1, enc2 in PAIR_ORDER:
                left = gsea[family][enc1][library]
                right = gsea[family][enc2][library]

                joined = (
                    left[["NES"]]
                    .rename(columns={"NES": "NES1"})
                    .join(
                        right[["NES"]].rename(columns={"NES": "NES2"}),
                        how="inner",
                    )
                    .dropna()
                )

                sig1 = significant_pathway_set(left)
                sig2 = significant_pathway_set(right)

                jac, inter, union = jaccard(sig1, sig2)

                rows.append(
                    {
                        "family": family,
                        "library": library,
                        "encoding_1": enc1,
                        "encoding_2": enc2,
                        "n_common_pathways": int(len(joined)),
                        "spearman_NES": safe_spearman(
                            joined["NES1"],
                            joined["NES2"],
                        ),
                        "sig_pathways_1": len(sig1),
                        "sig_pathways_2": len(sig2),
                        "sig_pathway_intersection": inter,
                        "sig_pathway_union": union,
                        "sig_pathway_jaccard": jac,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# MATRICES
# =============================================================================

def pair_matrix(
    pair_df: pd.DataFrame,
    family: str,
    value_col: str,
    library: str | None = None,
) -> np.ndarray:
    mat = np.eye(3)

    sub = pair_df[
        pair_df["family"] == family
    ].copy()

    if library is not None:
        sub = sub[
            sub["library"] == library
        ]

    lookup = {
        encoding: i
        for i, encoding in enumerate(ENCODINGS)
    }

    for _, row in sub.iterrows():
        i = lookup[row["encoding_1"]]
        j = lookup[row["encoding_2"]]
        value = row[value_col]

        mat[i, j] = value
        mat[j, i] = value

    return mat


def annotate_matrix(
    ax,
    mat,
    decimals: int,
) -> None:
    for i in range(3):
        for j in range(3):
            value = mat[i, j]

            label = (
                "NA"
                if np.isnan(value)
                else f"{value:.{decimals}f}"
            )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="bold" if i != j else "normal",
            )


def plot_matrix(
    ax,
    mat,
    title: str,
    panel_letter: str,
    decimals: int,
) -> None:
    ax.imshow(
        mat,
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))

    ax.set_xticklabels(
        ENCODINGS,
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_yticklabels(ENCODINGS)
    ax.tick_params(axis="x", pad=2.0)
    ax.tick_params(axis="y", pad=2.5)

    ax.set_title(
        title,
        pad=6,
    )

    annotate_matrix(
        ax,
        mat,
        decimals,
    )

    ax.text(
        -0.19,
        1.07,
        panel_letter,
        transform=ax.transAxes,
        fontsize=8.7,
        fontweight="bold",
        va="top",
    )


# =============================================================================
# GENE FIGURE
# =============================================================================

def plot_gene_counts_panel(
    ax,
    gene_counts: pd.DataFrame,
    family: str,
    letter: str,
) -> None:
    sub = (
        gene_counts[
            gene_counts["family"] == family
        ]
        .set_index("encoding")
        .reindex(ENCODINGS)
    )

    values = sub["n_fdr_significant"].to_numpy()
    x = np.arange(3)

    bars = ax.bar(
        x,
        values,
        width=0.58,
    )

    ymax = max(values)
    ax.set_ylim(0, ymax * 1.12)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.018,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=5.1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ENCODINGS)
    ax.tick_params(axis="x", pad=2.0)
    ax.set_ylabel("FDR-significant genes")
    ax.set_title("Gene-level discovery", pad=6)

    ax.grid(
        axis="y",
        alpha=0.15,
        linewidth=0.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        -0.15,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=8.7,
        fontweight="bold",
        va="top",
    )


def make_gene_figure(
    gene_counts: pd.DataFrame,
    gene_pairs: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 5.45),
        gridspec_kw={
            "width_ratios": [1.0, 1.0, 1.08],
            "wspace": 0.34,
            "hspace": 0.47,
        },
    )

    families = ["Amyloid", "Tau"]
    letters = [["A", "B", "C"], ["D", "E", "F"]]

    for row, family in enumerate(families):
        zmat = pair_matrix(
            gene_pairs,
            family,
            "spearman_Z",
        )

        dmat = pair_matrix(
            gene_pairs,
            family,
            "direction_concordance",
        )

        plot_matrix(
            axes[row, 0],
            zmat,
            "Genome-wide Z-score concordance",
            letters[row][0],
            decimals=3,
        )

        plot_matrix(
            axes[row, 1],
            dmat,
            "Direction concordance",
            letters[row][1],
            decimals=2,
        )

        plot_gene_counts_panel(
            axes[row, 2],
            gene_counts,
            family,
            letters[row][2],
        )

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
        0.026,
        0.304,
        "TAU",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )

    fig.suptitle(
        "Weighted AREA encoding validation",
        fontsize=10.2,
        fontweight="bold",
        y=0.987,
    )

    fig.text(
        0.5,
        0.948,
        (
            "Equal vs calibrated vs continuous encodings, "
            "evaluated before biological interpretation"
        ),
        ha="center",
        fontsize=6.0,
        color="#3F3F3F",
    )

    fig.text(
        0.5,
        0.018,
        (
            "Heatmaps: 0 = low agreement, 1 = identical. "
            "Gene discovery uses BH FDR < 0.05."
        ),
        ha="center",
        fontsize=5.2,
        color=MUTED,
    )

    fig.subplots_adjust(
        top=0.865,
        bottom=0.135,
        left=0.108,
        right=0.985,
    )

    fig.savefig(
        GENE_FIG_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        GENE_FIG_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# PATHWAY FIGURE
# =============================================================================

def make_pathway_figure(
    pathway_pairs: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 5.05),
        gridspec_kw={
            "wspace": 0.32,
            "hspace": 0.46,
        },
    )

    families = ["Amyloid", "Tau"]
    letters = [["A", "B", "C"], ["D", "E", "F"]]

    for row, family in enumerate(families):
        for col, library in enumerate(LIBRARIES):
            mat = pair_matrix(
                pathway_pairs,
                family,
                "spearman_NES",
                library=library,
            )

            plot_matrix(
                axes[row, col],
                mat,
                f"{library} NES concordance",
                letters[row][col],
                decimals=3,
            )

    fig.text(
        0.026,
        0.652,
        "AMYLOID",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )

    fig.text(
        0.026,
        0.304,
        "TAU",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )

    fig.suptitle(
        "Pathway NES stability across Weighted AREA encodings",
        fontsize=10.0,
        fontweight="bold",
        y=0.987,
    )

    fig.text(
        0.5,
        0.948,
        (
            "Spearman correlation of preranked GSEA NES values across "
            "Equal, Calibrated, and Continuous encodings"
        ),
        ha="center",
        fontsize=5.9,
        color="#3F3F3F",
    )

    fig.text(
        0.5,
        0.018,
        "Heatmaps: 0 = low pathway-score agreement, 1 = identical pathway ranking.",
        ha="center",
        fontsize=5.2,
        color=MUTED,
    )

    fig.subplots_adjust(
        top=0.858,
        bottom=0.145,
        left=0.108,
        right=0.985,
    )

    fig.savefig(
        PATHWAY_FIG_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        PATHWAY_FIG_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# COMPACT TABLE
# =============================================================================

def make_compact_table(
    gene_counts: pd.DataFrame,
    gene_pairs: pd.DataFrame,
    pathway_pairs: pd.DataFrame,
) -> pd.DataFrame:
    compact = gene_pairs.copy()

    count_lookup = {
        (row["family"], row["encoding"]): int(row["n_fdr_significant"])
        for _, row in gene_counts.iterrows()
    }

    compact["n_sig_genes_1"] = compact.apply(
        lambda r: count_lookup[(r["family"], r["encoding_1"])],
        axis=1,
    )

    compact["n_sig_genes_2"] = compact.apply(
        lambda r: count_lookup[(r["family"], r["encoding_2"])],
        axis=1,
    )

    pathway_summary = (
        pathway_pairs
        .groupby(
            ["family", "encoding_1", "encoding_2"],
            as_index=False,
        )
        .agg(
            mean_pathway_NES_spearman=("spearman_NES", "mean"),
            min_pathway_NES_spearman=("spearman_NES", "min"),
            max_pathway_NES_spearman=("spearman_NES", "max"),
            n_pathway_libraries=("library", "nunique"),
        )
    )

    return compact.merge(
        pathway_summary,
        on=["family", "encoding_1", "encoding_2"],
        how="left",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("WEIGHTED AREA ENCODING VALIDATION V3")
    print("=" * 100)
    print()

    # Gene-level
    gene_data = load_all_weighted()

    gene_counts = make_gene_counts(gene_data)
    gene_pairs = make_gene_pairwise(gene_data)

    gene_counts.to_csv(
        GENE_COUNTS_CSV,
        index=False,
    )

    gene_pairs.to_csv(
        GENE_PAIRWISE_CSV,
        index=False,
    )

    make_gene_figure(
        gene_counts,
        gene_pairs,
    )

    # Pathway-level
    gsea = load_all_gsea()

    pathway_counts = make_pathway_counts(gsea)
    pathway_pairs = make_pathway_pairwise(gsea)

    pathway_counts.to_csv(
        PATHWAY_COUNTS_CSV,
        index=False,
    )

    pathway_pairs.to_csv(
        PATHWAY_PAIRWISE_CSV,
        index=False,
    )

    make_pathway_figure(
        pathway_pairs,
    )

    # Compact combined
    compact = make_compact_table(
        gene_counts,
        gene_pairs,
        pathway_pairs,
    )

    compact.to_csv(
        COMPACT_CSV,
        index=False,
    )

    # Manifest
    manifest_lines = [
        "Weighted AREA encoding validation v3",
        "",
        "Exact GSEA files used:",
    ]

    for family, mapping in FAMILIES.items():
        for encoding, phenotype in mapping.items():
            for library_label, library_dir in LIBRARIES.items():
                manifest_lines.append(
                    f"{family} | {encoding} | {library_label} | "
                    f"{gsea_path(phenotype, library_dir)}"
                )

    manifest_lines.extend(
        [
            "",
            "Outputs:",
            str(GENE_FIG_PNG),
            str(GENE_FIG_PDF),
            str(PATHWAY_FIG_PNG),
            str(PATHWAY_FIG_PDF),
            str(GENE_COUNTS_CSV),
            str(GENE_PAIRWISE_CSV),
            str(PATHWAY_COUNTS_CSV),
            str(PATHWAY_PAIRWISE_CSV),
            str(COMPACT_CSV),
        ]
    )

    MANIFEST_TXT.write_text(
        "\n".join(manifest_lines) + "\n"
    )

    print("GENE PAIRWISE METRICS")
    print()
    print(gene_pairs.to_string(index=False))
    print()

    print("PATHWAY PAIRWISE METRICS")
    print()
    print(pathway_pairs.to_string(index=False))
    print()

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print()
    print(f"Wrote: {GENE_FIG_PNG}")
    print(f"Wrote: {GENE_FIG_PDF}")
    print(f"Wrote: {PATHWAY_FIG_PNG}")
    print(f"Wrote: {PATHWAY_FIG_PDF}")
    print(f"Wrote: {GENE_COUNTS_CSV}")
    print(f"Wrote: {GENE_PAIRWISE_CSV}")
    print(f"Wrote: {PATHWAY_COUNTS_CSV}")
    print(f"Wrote: {PATHWAY_PAIRWISE_CSV}")
    print(f"Wrote: {COMPACT_CSV}")
    print(f"Wrote: {MANIFEST_TXT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_weighted_area_biology.py
================================

Large-scale biological synthesis of the full Weighted AREA GSEA results.

This script is intended to begin the biological-interpretation phase AFTER
method validation is complete.

It analyzes all 9 Weighted AREA phenotypes across:
    - Hallmark
    - GO Biological Process
    - Reactome

and asks:

1. Which pathways recur across cognition, amyloid, and tau?
2. Which pathways are phenotype-axis specific?
3. Which pathways are significant across multiple encodings of the same axis?
4. How similar are pathway NES profiles across phenotypes?
5. Which leading-edge genes repeatedly drive significant pathways?
6. Which apparently redundant pathways can be collapsed into data-driven
   pathway modules based on leading-edge gene overlap?
7. Which representative pathways should be prioritized for biological review?

IMPORTANT
---------
This script does NOT assign hand-written biological labels such as
"mitochondrial", "synaptic", etc. It first performs objective recurrence,
specificity, leading-edge, and redundancy analyses. Those outputs can then be
used to build the final biological story without cherry-picking.

Expected GSEA layout
--------------------
results/weighted_area_full_phenotype_gsea/
    Hallmark_2020/
        Weighted_<phenotype>/
            gsea_all_results.csv
    GO_Biological_Process_2025/
        Weighted_<phenotype>/
            gsea_all_results.csv
    Reactome_Pathways_2024/
        Weighted_<phenotype>/
            gsea_all_results.csv

Expected columns
----------------
Known current format:
    Name
    pathway
    ES
    NES
    NOM_pvalue
    FDR_qvalue
    FWER_pvalue
    Tag_percent
    Gene_percent
    leading_edge_genes
    method
    library
    fdr_significant

Outputs
-------
results/weighted_area_biology/

Tables:
    pathway_long_all.csv
    pathway_recurrence_summary.csv
    pathway_axis_specificity.csv
    pathway_encoding_stability.csv
    pathway_NES_correlation_matrix.csv
    leading_edge_gene_recurrence.csv
    redundancy_clusters.csv
    representative_pathways.csv
    top_shared_pathways.csv
    top_axis_specific_pathways.csv

Figures:
    biology_recurrent_pathways_heatmap.png/.pdf
    biology_pathway_axis_summary.png/.pdf
    biology_NES_correlation_heatmap.png/.pdf
    biology_leading_edge_gene_recurrence.png/.pdf

The script is designed to be a large first-pass synthesis. It does not replace
manual biological interpretation of the highest-priority modules.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

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

PHENOTYPE_LABELS = {
    "Cognitive_stage": "Cognitive stage",
    "cogn_global_impairment": "Global cognition",
    "mmse_impairment": "MMSE",
    "CERAD_equal": "CERAD equal",
    "CERAD_amyloid_calibrated": "CERAD calibrated",
    "amyloid_continuous": "Amyloid continuous",
    "Braak_equal": "Braak equal",
    "Braak_tangle_calibrated": "Braak calibrated",
    "tangle_continuous": "Tangles continuous",
}

AXIS_MAP = {
    "Cognitive_stage": "Cognition",
    "cogn_global_impairment": "Cognition",
    "mmse_impairment": "Cognition",
    "CERAD_equal": "Amyloid",
    "CERAD_amyloid_calibrated": "Amyloid",
    "amyloid_continuous": "Amyloid",
    "Braak_equal": "Tau",
    "Braak_tangle_calibrated": "Tau",
    "tangle_continuous": "Tau",
}

ENCODING_MAP = {
    "Cognitive_stage": "Ordinal",
    "cogn_global_impairment": "Continuous",
    "mmse_impairment": "Continuous",
    "CERAD_equal": "Equal",
    "CERAD_amyloid_calibrated": "Calibrated",
    "amyloid_continuous": "Continuous",
    "Braak_equal": "Equal",
    "Braak_tangle_calibrated": "Calibrated",
    "tangle_continuous": "Continuous",
}

LIBRARIES = {
    "Hallmark": "Hallmark_2020",
    "GO BP": "GO_Biological_Process_2025",
    "Reactome": "Reactome_Pathways_2024",
}

AXES = ["Cognition", "Amyloid", "Tau"]

TEXT = "#171717"
MUTED = "#5A5A5A"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 7.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.2,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.4,
    }
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--gsea-root",
        default="results/weighted_area_full_phenotype_gsea",
    )

    p.add_argument(
        "--outdir",
        default="results/weighted_area_biology",
    )

    p.add_argument(
        "--fdr",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--top-recurrent",
        type=int,
        default=30,
        help="Number of recurrent pathways to show in heatmap.",
    )

    p.add_argument(
        "--top-leading-edge-genes",
        type=int,
        default=30,
    )

    p.add_argument(
        "--redundancy-jaccard",
        type=float,
        default=0.50,
        help=(
            "Leading-edge Jaccard threshold for greedy pathway clustering "
            "within each library and NES direction."
        ),
    )

    p.add_argument(
        "--min-significant-phenotypes",
        type=int,
        default=2,
        help=(
            "Minimum number of significant phenotypes for a pathway to be "
            "eligible for redundancy clustering."
        ),
    )

    return p.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

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


def parse_leading_edge(value) -> set[str]:
    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    # Known GSEA exports can use ;, comma, slash, or spaces.
    # Prefer semicolon/comma/slash splitting first.
    parts = re.split(r"[;,/|]+", text)

    if len(parts) == 1:
        # Fall back to whitespace only if it appears to be a simple symbol list.
        parts = text.split()

    genes = {
        p.strip()
        for p in parts
        if p.strip()
        and p.strip().lower() not in {"nan", "none", "na"}
    }

    return genes


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def shorten_term(term: str, max_chars: int = 62) -> str:
    term = str(term).replace("_", " ")
    term = re.sub(r"\s+", " ", term).strip()

    if len(term) <= max_chars:
        return term

    return term[: max_chars - 1].rstrip() + "…"


# =============================================================================
# LOAD GSEA RESULTS
# =============================================================================

def gsea_path(root: Path, library_dir: str, phenotype: str) -> Path:
    return (
        root
        / library_dir
        / f"Weighted_{phenotype}"
        / "gsea_all_results.csv"
    )


def load_one_gsea(
    root: Path,
    library_label: str,
    library_dir: str,
    phenotype: str,
    fdr: float,
) -> pd.DataFrame:

    path = gsea_path(
        root,
        library_dir,
        phenotype,
    )

    require(path)

    df = pd.read_csv(path)

    pathway_col = resolve_column(
        df,
        ["pathway", "Pathway", "Term", "term", "Name"],
        "pathway column",
    )

    nes_col = resolve_column(
        df,
        ["NES", "nes"],
        "NES column",
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
        ],
        "FDR/q-value column",
    )

    lead_col = None

    for candidate in [
        "leading_edge_genes",
        "Lead_genes",
        "lead_genes",
        "Leading_edge_genes",
    ]:
        if candidate in df.columns:
            lead_col = candidate
            break

    out = pd.DataFrame(
        {
            "library": library_label,
            "phenotype": phenotype,
            "phenotype_label": PHENOTYPE_LABELS[phenotype],
            "axis": AXIS_MAP[phenotype],
            "encoding": ENCODING_MAP[phenotype],
            "pathway": df[pathway_col].astype(str).str.strip(),
            "NES": pd.to_numeric(df[nes_col], errors="coerce"),
            "FDR": pd.to_numeric(df[q_col], errors="coerce"),
        }
    )

    if lead_col is not None:
        out["leading_edge_genes"] = df[lead_col].astype(str)
    else:
        out["leading_edge_genes"] = ""

    out["significant"] = out["FDR"] < fdr
    out["direction"] = np.where(
        out["NES"] >= 0,
        "Positive",
        "Negative",
    )

    out = out.dropna(
        subset=["pathway", "NES"]
    )

    # Sanity check against accidental constant "Name" selection.
    if out["pathway"].nunique() < 5 and len(out) >= 5:
        raise ValueError(
            f"Pathway column appears invalid for {path}: "
            f"only {out['pathway'].nunique()} unique terms."
        )

    return out.drop_duplicates(
        subset=["library", "phenotype", "pathway"]
    )


def load_all_gsea(args) -> pd.DataFrame:
    root = Path(args.gsea_root)

    frames = []

    for library_label, library_dir in LIBRARIES.items():
        for phenotype in PHENOTYPES:
            frames.append(
                load_one_gsea(
                    root=root,
                    library_label=library_label,
                    library_dir=library_dir,
                    phenotype=phenotype,
                    fdr=args.fdr,
                )
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =============================================================================
# PATHWAY RECURRENCE / SPECIFICITY
# =============================================================================

def make_recurrence_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (library, pathway), sub in long_df.groupby(
        ["library", "pathway"]
    ):
        sig = sub[sub["significant"]]

        axes_sig = sorted(
            sig["axis"].unique().tolist()
        )

        phenotypes_sig = sorted(
            sig["phenotype"].unique().tolist()
        )

        axis_counts = (
            sig.groupby("axis")["phenotype"]
            .nunique()
            .to_dict()
        )

        nes_by_axis = (
            sub.groupby("axis")["NES"]
            .mean()
            .to_dict()
        )

        # Stable direction across all significant phenotypes?
        sig_dirs = sig["direction"].dropna().unique().tolist()

        if len(sig_dirs) == 1:
            direction_consistency = sig_dirs[0]
        elif len(sig_dirs) == 0:
            direction_consistency = "Not significant"
        else:
            direction_consistency = "Mixed"

        rows.append(
            {
                "library": library,
                "pathway": pathway,
                "n_significant_phenotypes": int(
                    sig["phenotype"].nunique()
                ),
                "n_significant_axes": int(len(axes_sig)),
                "significant_axes": ";".join(axes_sig),
                "significant_phenotypes": ";".join(phenotypes_sig),
                "cognition_sig_count": int(axis_counts.get("Cognition", 0)),
                "amyloid_sig_count": int(axis_counts.get("Amyloid", 0)),
                "tau_sig_count": int(axis_counts.get("Tau", 0)),
                "mean_NES_cognition": nes_by_axis.get("Cognition", np.nan),
                "mean_NES_amyloid": nes_by_axis.get("Amyloid", np.nan),
                "mean_NES_tau": nes_by_axis.get("Tau", np.nan),
                "significant_direction_consistency": direction_consistency,
                "mean_abs_NES": float(sub["NES"].abs().mean()),
                "max_abs_NES": float(sub["NES"].abs().max()),
            }
        )

    out = pd.DataFrame(rows)

    return out.sort_values(
        [
            "n_significant_axes",
            "n_significant_phenotypes",
            "mean_abs_NES",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )


def classify_specificity(row) -> str:
    axes = {
        axis
        for axis in str(row["significant_axes"]).split(";")
        if axis
    }

    if len(axes) == 3:
        return "Shared across cognition + amyloid + tau"

    if len(axes) == 2:
        return "Shared across 2 axes"

    if axes == {"Cognition"}:
        return "Cognition-specific"

    if axes == {"Amyloid"}:
        return "Amyloid-specific"

    if axes == {"Tau"}:
        return "Tau-specific"

    return "Not significant"


def make_axis_specificity(
    recurrence: pd.DataFrame,
) -> pd.DataFrame:
    out = recurrence.copy()

    out["specificity_class"] = out.apply(
        classify_specificity,
        axis=1,
    )

    return out.sort_values(
        [
            "specificity_class",
            "n_significant_phenotypes",
            "mean_abs_NES",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )


# =============================================================================
# ENCODING STABILITY
# =============================================================================

def make_encoding_stability(
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for axis in ["Amyloid", "Tau"]:
        sub_axis = long_df[
            long_df["axis"] == axis
        ]

        for (library, pathway), sub in sub_axis.groupby(
            ["library", "pathway"]
        ):
            sig_by_encoding = (
                sub.set_index("encoding")["significant"]
                .to_dict()
            )

            nes_by_encoding = (
                sub.set_index("encoding")["NES"]
                .to_dict()
            )

            n_sig = sum(
                bool(sig_by_encoding.get(enc, False))
                for enc in ["Equal", "Calibrated", "Continuous"]
            )

            signs = [
                np.sign(nes_by_encoding[enc])
                for enc in ["Equal", "Calibrated", "Continuous"]
                if enc in nes_by_encoding
                and pd.notna(nes_by_encoding[enc])
                and nes_by_encoding[enc] != 0
            ]

            direction_stable = (
                len(set(signs)) <= 1
                if signs
                else np.nan
            )

            rows.append(
                {
                    "axis": axis,
                    "library": library,
                    "pathway": pathway,
                    "n_significant_encodings": int(n_sig),
                    "significant_equal": bool(
                        sig_by_encoding.get("Equal", False)
                    ),
                    "significant_calibrated": bool(
                        sig_by_encoding.get("Calibrated", False)
                    ),
                    "significant_continuous": bool(
                        sig_by_encoding.get("Continuous", False)
                    ),
                    "NES_equal": nes_by_encoding.get("Equal", np.nan),
                    "NES_calibrated": nes_by_encoding.get(
                        "Calibrated",
                        np.nan,
                    ),
                    "NES_continuous": nes_by_encoding.get(
                        "Continuous",
                        np.nan,
                    ),
                    "direction_stable_across_encodings": direction_stable,
                }
            )

    return pd.DataFrame(rows).sort_values(
        [
            "axis",
            "n_significant_encodings",
        ],
        ascending=[
            True,
            False,
        ],
    )


# =============================================================================
# PHENOTYPE NES CORRELATION
# =============================================================================

def make_nes_correlation(long_df: pd.DataFrame) -> pd.DataFrame:
    matrices = []

    for library in LIBRARIES:
        sub = long_df[
            long_df["library"] == library
        ]

        pivot = sub.pivot_table(
            index="pathway",
            columns="phenotype",
            values="NES",
            aggfunc="first",
        )

        pivot = pivot.reindex(
            columns=PHENOTYPES
        )

        corr = pivot.corr(
            method="spearman"
        )

        corr.insert(
            0,
            "phenotype_1",
            corr.index,
        )

        long_corr = corr.melt(
            id_vars="phenotype_1",
            var_name="phenotype_2",
            value_name="spearman_NES",
        )

        long_corr["library"] = library

        matrices.append(long_corr)

    return pd.concat(
        matrices,
        ignore_index=True,
    )


# =============================================================================
# LEADING-EDGE GENES
# =============================================================================

def make_leading_edge_gene_recurrence(
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    sig = long_df[
        long_df["significant"]
    ].copy()

    for _, row in sig.iterrows():
        genes = parse_leading_edge(
            row["leading_edge_genes"]
        )

        for gene in genes:
            rows.append(
                {
                    "gene": gene,
                    "library": row["library"],
                    "pathway": row["pathway"],
                    "phenotype": row["phenotype"],
                    "axis": row["axis"],
                    "NES": row["NES"],
                    "direction": row["direction"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "gene",
                "n_significant_pathway_instances",
                "n_unique_pathways",
                "n_phenotypes",
                "n_axes",
                "axes",
                "mean_NES_context",
            ]
        )

    exploded = pd.DataFrame(rows)

    summary = (
        exploded
        .groupby("gene")
        .agg(
            n_significant_pathway_instances=("pathway", "size"),
            n_unique_pathways=("pathway", "nunique"),
            n_phenotypes=("phenotype", "nunique"),
            n_axes=("axis", "nunique"),
            mean_NES_context=("NES", "mean"),
        )
        .reset_index()
    )

    axes = (
        exploded
        .groupby("gene")["axis"]
        .apply(
            lambda x: ";".join(
                sorted(set(x))
            )
        )
        .rename("axes")
        .reset_index()
    )

    summary = summary.merge(
        axes,
        on="gene",
        how="left",
    )

    return summary.sort_values(
        [
            "n_axes",
            "n_phenotypes",
            "n_unique_pathways",
            "n_significant_pathway_instances",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )


# =============================================================================
# REDUNDANCY REDUCTION
# =============================================================================

def build_pathway_leading_edge_sets(
    long_df: pd.DataFrame,
) -> dict:
    """
    Build pathway-level union of leading-edge genes across significant
    phenotypes, separately by library.
    """
    output = {}

    sig = long_df[
        long_df["significant"]
    ]

    for (library, pathway), sub in sig.groupby(
        ["library", "pathway"]
    ):
        genes = set()

        for value in sub["leading_edge_genes"]:
            genes |= parse_leading_edge(value)

        output[(library, pathway)] = genes

    return output


def greedy_redundancy_clusters(
    recurrence: pd.DataFrame,
    long_df: pd.DataFrame,
    threshold: float,
    min_sig_phenotypes: int,
) -> pd.DataFrame:
    """
    Greedy clustering using leading-edge Jaccard.

    Clustering is done within library AND mean NES sign to avoid grouping
    pathways that have overlapping genes but opposite biological direction.
    """
    lead_sets = build_pathway_leading_edge_sets(
        long_df
    )

    candidates = recurrence[
        recurrence["n_significant_phenotypes"]
        >= min_sig_phenotypes
    ].copy()

    rows = []
    cluster_id = 0

    for library in LIBRARIES:
        lib = candidates[
            candidates["library"] == library
        ].copy()

        # Use average NES across all phenotypes to define broad sign.
        lib["broad_direction"] = np.where(
            (
                lib[
                    [
                        "mean_NES_cognition",
                        "mean_NES_amyloid",
                        "mean_NES_tau",
                    ]
                ]
                .mean(axis=1)
                >= 0
            ),
            "Positive",
            "Negative",
        )

        for direction in ["Positive", "Negative"]:
            group = (
                lib[
                    lib["broad_direction"] == direction
                ]
                .sort_values(
                    [
                        "n_significant_axes",
                        "n_significant_phenotypes",
                        "mean_abs_NES",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                )
            )

            assigned = set()

            for _, seed in group.iterrows():
                seed_path = seed["pathway"]

                if seed_path in assigned:
                    continue

                cluster_id += 1
                members = [seed_path]
                assigned.add(seed_path)

                seed_genes = lead_sets.get(
                    (library, seed_path),
                    set(),
                )

                for _, candidate in group.iterrows():
                    candidate_path = candidate["pathway"]

                    if candidate_path in assigned:
                        continue

                    candidate_genes = lead_sets.get(
                        (library, candidate_path),
                        set(),
                    )

                    sim = jaccard(
                        seed_genes,
                        candidate_genes,
                    )

                    if sim >= threshold:
                        members.append(candidate_path)
                        assigned.add(candidate_path)

                for member in members:
                    member_row = group[
                        group["pathway"] == member
                    ].iloc[0]

                    similarity_to_seed = jaccard(
                        seed_genes,
                        lead_sets.get(
                            (library, member),
                            set(),
                        ),
                    )

                    rows.append(
                        {
                            "cluster_id": cluster_id,
                            "library": library,
                            "direction": direction,
                            "representative_pathway": seed_path,
                            "member_pathway": member,
                            "jaccard_to_representative": similarity_to_seed,
                            "cluster_size": len(members),
                            "member_n_significant_axes": member_row[
                                "n_significant_axes"
                            ],
                            "member_n_significant_phenotypes": member_row[
                                "n_significant_phenotypes"
                            ],
                            "member_mean_abs_NES": member_row[
                                "mean_abs_NES"
                            ],
                        }
                    )

    return pd.DataFrame(rows)


def make_representative_pathways(
    clusters: pd.DataFrame,
    recurrence: pd.DataFrame,
) -> pd.DataFrame:
    if len(clusters) == 0:
        return pd.DataFrame()

    reps = (
        clusters[
            [
                "cluster_id",
                "library",
                "direction",
                "representative_pathway",
                "cluster_size",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "representative_pathway": "pathway",
            }
        )
    )

    reps = reps.merge(
        recurrence,
        on=[
            "library",
            "pathway",
        ],
        how="left",
    )

    return reps.sort_values(
        [
            "n_significant_axes",
            "n_significant_phenotypes",
            "cluster_size",
            "mean_abs_NES",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )


# =============================================================================
# PRIORITY TABLES
# =============================================================================

def make_top_shared(
    recurrence: pd.DataFrame,
) -> pd.DataFrame:
    return (
        recurrence[
            recurrence["n_significant_axes"] == 3
        ]
        .sort_values(
            [
                "n_significant_phenotypes",
                "mean_abs_NES",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .copy()
    )


def make_top_axis_specific(
    specificity: pd.DataFrame,
) -> pd.DataFrame:
    keep = specificity[
        specificity["specificity_class"].isin(
            [
                "Cognition-specific",
                "Amyloid-specific",
                "Tau-specific",
            ]
        )
    ].copy()

    return keep.sort_values(
        [
            "specificity_class",
            "n_significant_phenotypes",
            "mean_abs_NES",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )


# =============================================================================
# FIGURES
# =============================================================================

def make_recurrent_heatmap(
    long_df: pd.DataFrame,
    recurrence: pd.DataFrame,
    outdir: Path,
    top_n: int,
):
    """
    Use top recurrent pathways across all libraries. Libraries are appended
    to row labels to preserve uniqueness.
    """
    top = (
        recurrence[
            recurrence["n_significant_phenotypes"] >= 2
        ]
        .head(top_n)
        .copy()
    )

    if len(top) == 0:
        return

    keys = set(
        zip(
            top["library"],
            top["pathway"],
        )
    )

    sub = long_df[
        long_df.apply(
            lambda r: (r["library"], r["pathway"]) in keys,
            axis=1,
        )
    ].copy()

    sub["row_label"] = (
        sub["library"]
        + " | "
        + sub["pathway"].map(shorten_term)
    )

    order = [
        f"{row.library} | {shorten_term(row.pathway)}"
        for row in top.itertuples()
    ]

    pivot = sub.pivot_table(
        index="row_label",
        columns="phenotype",
        values="NES",
        aggfunc="first",
    )

    pivot = pivot.reindex(
        index=order,
        columns=PHENOTYPES,
    )

    sig_pivot = sub.pivot_table(
        index="row_label",
        columns="phenotype",
        values="significant",
        aggfunc="first",
    ).reindex(
        index=order,
        columns=PHENOTYPES,
    )

    fig_h = max(
        5.2,
        0.23 * len(pivot) + 2.0,
    )

    fig, ax = plt.subplots(
        figsize=(7.2, fig_h)
    )

    vmax = np.nanmax(
        np.abs(pivot.to_numpy())
    )

    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(len(PHENOTYPES))
    )
    ax.set_xticklabels(
        [
            PHENOTYPE_LABELS[p]
            for p in PHENOTYPES
        ],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(pivot))
    )
    ax.set_yticklabels(
        pivot.index,
        fontsize=5.1,
    )

    for i in range(len(pivot)):
        for j in range(len(PHENOTYPES)):
            if bool(sig_pivot.iloc[i, j]):
                ax.text(
                    j,
                    i,
                    "•",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="black",
                )

    ax.set_title(
        "Recurrent Weighted AREA pathway associations",
        fontsize=9.2,
        fontweight="bold",
        pad=10,
    )

    ax.text(
        0.5,
        1.015,
        "Cell color = NES; dot = GSEA FDR < 0.05",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.5,
        color=MUTED,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(
        "NES",
        fontsize=5.8,
    )
    cbar.ax.tick_params(
        labelsize=5.0,
    )

    fig.subplots_adjust(
        left=0.37,
        right=0.94,
        top=0.91,
        bottom=0.18,
    )

    fig.savefig(
        outdir / "biology_recurrent_pathways_heatmap.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "biology_recurrent_pathways_heatmap.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def make_axis_summary_figure(
    specificity: pd.DataFrame,
    outdir: Path,
):
    classes = [
        "Shared across cognition + amyloid + tau",
        "Shared across 2 axes",
        "Cognition-specific",
        "Amyloid-specific",
        "Tau-specific",
    ]

    libraries = list(LIBRARIES.keys())

    counts = (
        specificity[
            specificity["n_significant_phenotypes"] > 0
        ]
        .groupby(
            [
                "library",
                "specificity_class",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=libraries,
            columns=classes,
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 3.4)
    )

    x = np.arange(len(libraries))
    bottom = np.zeros(
        len(libraries)
    )

    for cls in classes:
        vals = counts[cls].to_numpy()

        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.62,
            label=cls,
        )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(libraries)
    ax.set_ylabel(
        "Significant pathways"
    )
    ax.set_title(
        "Shared and phenotype-axis-specific pathway associations",
        fontsize=9.0,
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        fontsize=5.4,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.72,
        top=0.86,
        bottom=0.18,
    )

    fig.savefig(
        outdir / "biology_pathway_axis_summary.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "biology_pathway_axis_summary.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def make_nes_corr_figure(
    corr_long: pd.DataFrame,
    outdir: Path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.9),
        gridspec_kw={
            "wspace": 0.38,
        },
    )

    for ax, library in zip(
        axes,
        LIBRARIES,
    ):
        sub = corr_long[
            corr_long["library"] == library
        ]

        pivot = sub.pivot(
            index="phenotype_1",
            columns="phenotype_2",
            values="spearman_NES",
        ).reindex(
            index=PHENOTYPES,
            columns=PHENOTYPES,
        )

        ax.imshow(
            pivot.to_numpy(),
            vmin=0,
            vmax=1,
            cmap="viridis",
            aspect="equal",
        )

        ax.set_title(
            library,
            pad=5,
        )

        ax.set_xticks(
            np.arange(len(PHENOTYPES))
        )
        ax.set_yticks(
            np.arange(len(PHENOTYPES))
        )

        short = [
            "Cog",
            "Global",
            "MMSE",
            "CERAD-E",
            "CERAD-C",
            "Amyloid",
            "Braak-E",
            "Braak-C",
            "Tangles",
        ]

        ax.set_xticklabels(
            short,
            rotation=55,
            ha="right",
            fontsize=4.0,
        )

        ax.set_yticklabels(
            short,
            fontsize=4.0,
        )

    fig.suptitle(
        "Pathway NES similarity across phenotypes",
        fontsize=8.8,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.02,
        "Spearman correlation of pathway NES profiles",
        ha="center",
        fontsize=5.2,
        color=MUTED,
    )

    fig.subplots_adjust(
        top=0.83,
        bottom=0.28,
        left=0.08,
        right=0.98,
    )

    fig.savefig(
        outdir / "biology_NES_correlation_heatmap.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "biology_NES_correlation_heatmap.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def make_leading_edge_figure(
    gene_summary: pd.DataFrame,
    outdir: Path,
    top_n: int,
):
    if len(gene_summary) == 0:
        return

    top = gene_summary.head(
        top_n
    ).copy()

    top = top.sort_values(
        "n_unique_pathways",
        ascending=True,
    )

    fig_h = max(
        4.0,
        0.18 * len(top) + 1.6,
    )

    fig, ax = plt.subplots(
        figsize=(5.6, fig_h)
    )

    ax.barh(
        top["gene"],
        top["n_unique_pathways"],
    )

    ax.set_xlabel(
        "Number of significant pathways containing gene"
    )

    ax.set_title(
        "Recurrent leading-edge genes",
        fontsize=8.8,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.15,
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "biology_leading_edge_gene_recurrence.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "biology_leading_edge_gene_recurrence.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("WEIGHTED AREA BIOLOGICAL SYNTHESIS")
    print("=" * 100)
    print()

    long_df = load_all_gsea(args)

    long_df.to_csv(
        outdir / "pathway_long_all.csv",
        index=False,
    )

    recurrence = make_recurrence_summary(
        long_df
    )

    recurrence.to_csv(
        outdir / "pathway_recurrence_summary.csv",
        index=False,
    )

    specificity = make_axis_specificity(
        recurrence
    )

    specificity.to_csv(
        outdir / "pathway_axis_specificity.csv",
        index=False,
    )

    encoding_stability = make_encoding_stability(
        long_df
    )

    encoding_stability.to_csv(
        outdir / "pathway_encoding_stability.csv",
        index=False,
    )

    nes_corr = make_nes_correlation(
        long_df
    )

    nes_corr.to_csv(
        outdir / "pathway_NES_correlation_matrix.csv",
        index=False,
    )

    leading_edge = make_leading_edge_gene_recurrence(
        long_df
    )

    leading_edge.to_csv(
        outdir / "leading_edge_gene_recurrence.csv",
        index=False,
    )

    clusters = greedy_redundancy_clusters(
        recurrence=recurrence,
        long_df=long_df,
        threshold=args.redundancy_jaccard,
        min_sig_phenotypes=args.min_significant_phenotypes,
    )

    clusters.to_csv(
        outdir / "redundancy_clusters.csv",
        index=False,
    )

    representatives = make_representative_pathways(
        clusters,
        recurrence,
    )

    representatives.to_csv(
        outdir / "representative_pathways.csv",
        index=False,
    )

    shared = make_top_shared(
        recurrence
    )

    shared.to_csv(
        outdir / "top_shared_pathways.csv",
        index=False,
    )

    specific = make_top_axis_specific(
        specificity
    )

    specific.to_csv(
        outdir / "top_axis_specific_pathways.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    make_recurrent_heatmap(
        long_df=long_df,
        recurrence=recurrence,
        outdir=outdir,
        top_n=args.top_recurrent,
    )

    make_axis_summary_figure(
        specificity=specificity,
        outdir=outdir,
    )

    make_nes_corr_figure(
        corr_long=nes_corr,
        outdir=outdir,
    )

    make_leading_edge_figure(
        gene_summary=leading_edge,
        outdir=outdir,
        top_n=args.top_leading_edge_genes,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    n_shared = int(
        (
            recurrence["n_significant_axes"] == 3
        ).sum()
    )

    specificity_counts = (
        specificity[
            specificity["n_significant_phenotypes"] > 0
        ]["specificity_class"]
        .value_counts()
    )

    print(f"Total GSEA rows loaded: {len(long_df):,}")
    print(
        f"Unique library/pathway combinations: "
        f"{len(recurrence):,}"
    )
    print(
        f"Pathways significant across all 3 phenotype axes: "
        f"{n_shared:,}"
    )
    print()

    print("Specificity counts:")
    print(
        specificity_counts.to_string()
    )
    print()

    print("Top recurrent pathways:")
    print(
        recurrence[
            [
                "library",
                "pathway",
                "n_significant_axes",
                "n_significant_phenotypes",
                "significant_direction_consistency",
                "mean_abs_NES",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print()

    print("Top recurrent leading-edge genes:")
    if len(leading_edge) > 0:
        print(
            leading_edge.head(25).to_string(
                index=False
            )
        )
    else:
        print(
            "No leading-edge genes could be parsed."
        )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print()
    print(f"Outputs: {outdir}")


if __name__ == "__main__":
    main()

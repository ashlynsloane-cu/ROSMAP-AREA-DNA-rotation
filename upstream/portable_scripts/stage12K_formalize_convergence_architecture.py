#!/usr/bin/env python3
"""
Stage 12K — formalize cross-omic convergence architecture.

Purpose
-------
Put the three existing levels of cross-omic evidence side by side without
inventing a composite score or a new arbitrary biological threshold:

    gene level    -> formal conjunction / both-individual-FDR
    pathway level -> full-rank GSEA + shared RNA/protein leading edge
    rank level    -> RRHO / whole-profile RNA-protein concordance

Primary contrasts
-----------------
    cogdx_NCI_vs_AD
    cogdx_MCI_vs_AD
    Braak_3-4_vs_5-6
    CERAD_3-4_vs_1-2

Methods
-------
    AREA
    CONVENTIONAL  (DESeq2 RNA + limma protein)

This stage is DESCRIPTIVE. It does not assign labels such as
"gene-concentrated" or "distributed" by a hand-built score.

Inputs
------
Stage 12A:
    ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv

Stage 12B:
    ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv

Stage 12I:
    CROSS_MODAL_GSEA_COMPARISON.tsv

Stage 12J:
    RRHO_PAIR_SUMMARY.tsv
    RRHO_QUADRANT_PEAKS.tsv

Outputs
-------
results/stage12_brain_convergence/stage12K_convergence_architecture/

    CONVERGENCE_ARCHITECTURE_SUMMARY.tsv
        One row per contrast x method with gene-, pathway-, and rank-level
        metrics.

    CROSS_METHOD_ARCHITECTURE_COMPARISON.tsv
        AREA vs conventional overlap / consistency for formal genes,
        concordant GSEA pathways, leading-edge genes, and RRHO summaries.

    FORMAL_VS_BOTH_INDIVIDUAL_GENE_AUDIT.tsv
        Exact relationship between formal conjunction and the set of genes
        individually FDR<.05 in both modalities.

    SHARED_LEADING_EDGE_GENE_RECURRENCE.tsv
        Gene-level recurrence across concordant dual-FDR pathways.
        Counts are annotation-level and therefore can reflect GO/Reactome
        redundancy; source-specific counts are retained.

    CONTRAST_METHOD_SHARED_LEADING_EDGE_GENES.tsv
        One row per contrast x method x gene for any gene appearing in a
        shared RNA/protein leading edge.

    README_STAGE12K.txt

Interpretive principle
----------------------
A contrast may have:
    few formal genes + many concordant pathways + strong RRHO
without contradiction.

That pattern means distributed cross-omic evidence rather than absence of
cross-omic evidence.

Conversely:
    many formal genes
means substantial evidence is concentrated into individually strong
same-gene RNA-protein associations.

Neither architecture is defined as "better".
"""

from __future__ import annotations
import os

from pathlib import Path
import re

import numpy as np
import pandas as pd
ROOT = Path(os.environ["ROSMAP_OFFLINE_ROOT"]).expanduser().resolve()
AREA_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
    / "ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv"
)

CONVENTIONAL_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12B_deseq2_limma_conjunction"
    / "ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv"
)

GSEA_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12I_full_rank_gsea"
    / "CROSS_MODAL_GSEA_COMPARISON.tsv"
)

RRHO_SUMMARY_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12J_rrho"
    / "RRHO_PAIR_SUMMARY.tsv"
)

RRHO_PEAKS_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12J_rrho"
    / "RRHO_QUADRANT_PEAKS.tsv"
)

OUTDIR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12K_convergence_architecture"
)

CONTRASTS = [
    "cogdx_NCI_vs_AD",
    "cogdx_MCI_vs_AD",
    "Braak_3-4_vs_5-6",
    "CERAD_3-4_vs_1-2",
]

METHODS = [
    "AREA",
    "CONVENTIONAL",
]

EXPECTED_UNIVERSE = 4903


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def require_columns(
    df: pd.DataFrame,
    columns: set[str],
    label: str,
):
    missing = columns - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )


def parse_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        x.strip()
        for x in re.split(r"[;,]", text)
        if x.strip()
    ]


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def load_inputs():
    for path in [
        AREA_PATH,
        CONVENTIONAL_PATH,
        GSEA_PATH,
        RRHO_SUMMARY_PATH,
        RRHO_PEAKS_PATH,
    ]:
        require(path)

    area = pd.read_csv(
        AREA_PATH,
        sep="\t",
        low_memory=False,
    )

    conventional = pd.read_csv(
        CONVENTIONAL_PATH,
        sep="\t",
        low_memory=False,
    )

    gsea = pd.read_csv(
        GSEA_PATH,
        sep="\t",
        low_memory=False,
    )

    rrho_summary = pd.read_csv(
        RRHO_SUMMARY_PATH,
        sep="\t",
        low_memory=False,
    )

    rrho_peaks = pd.read_csv(
        RRHO_PEAKS_PATH,
        sep="\t",
        low_memory=False,
    )

    require_columns(
        area,
        {
            "contrast_id",
            "gene_symbol",
            "direction_concordant",
            "both_individual_fdr05",
            "crossomic_fdr05_bh",
        },
        "Stage 12A",
    )

    require_columns(
        conventional,
        {
            "contrast_id",
            "gene_symbol",
            "direction_concordant",
            "both_individual_fdr05",
            "crossomic_fdr05_bh",
        },
        "Stage 12B",
    )

    require_columns(
        gsea,
        {
            "contrast_id",
            "method",
            "gene_set_source",
            "term_name",
            "rna_nes",
            "protein_nes",
            "rna_fdr_qvalue",
            "protein_fdr_qvalue",
            "same_nes_direction",
            "both_fdr05",
            "concordant_both_fdr05",
            "shared_leading_edge_n",
            "shared_leading_edge_genes",
        },
        "Stage 12I",
    )

    require_columns(
        rrho_summary,
        {
            "contrast_id",
            "method",
            "strongest_concordant_quadrant",
            "strongest_concordant_neglog10_BY",
            "strongest_concordant_overlap_n",
            "spearman_rho",
            "same_direction_fraction",
        },
        "Stage 12J pair summary",
    )

    require_columns(
        rrho_peaks,
        {
            "contrast_id",
            "method",
            "quadrant",
            "quadrant_class",
            "peak_rna_tail_n",
            "peak_protein_tail_n",
            "peak_overlap_n",
            "peak_enrichment_ratio",
            "peak_by_fdr",
            "peak_neglog10_by_fdr",
        },
        "Stage 12J quadrant peaks",
    )

    return (
        area,
        conventional,
        gsea,
        rrho_summary,
        rrho_peaks,
    )


def gene_level_metrics(
    table: pd.DataFrame,
    contrast: str,
    method: str,
):
    d = table.loc[
        table["contrast_id"].astype(str).eq(contrast)
    ].copy()

    if len(d) != EXPECTED_UNIVERSE:
        raise RuntimeError(
            f"{contrast} {method}: expected {EXPECTED_UNIVERSE} "
            f"genes in conjunction table; found {len(d)}."
        )

    if d["gene_symbol"].duplicated().any():
        raise RuntimeError(
            f"{contrast} {method}: duplicate gene symbols."
        )

    direction = d["direction_concordant"].astype(bool)
    both = d["both_individual_fdr05"].astype(bool)
    formal = d["crossomic_fdr05_bh"].astype(bool)

    overlap = both & formal

    return {
        "mapped_genes": len(d),
        "direction_concordant_n": int(direction.sum()),
        "direction_concordant_fraction":
            float(direction.mean()),
        "both_individual_fdr05_n": int(both.sum()),
        "both_individual_fdr05_fraction":
            float(both.mean()),
        "formal_conjunction_fdr05_n": int(formal.sum()),
        "formal_conjunction_fdr05_fraction":
            float(formal.mean()),
        "formal_and_both_individual_n":
            int(overlap.sum()),
        "formal_not_both_individual_n":
            int((formal & ~both).sum()),
        "both_individual_not_formal_n":
            int((both & ~formal).sum()),
        "formal_and_both_over_both_individual_fraction":
            (
                float(overlap.sum() / both.sum())
                if both.sum() > 0
                else np.nan
            ),
    }


def gsea_level_metrics(
    gsea: pd.DataFrame,
    contrast: str,
    method: str,
):
    d = gsea.loc[
        gsea["contrast_id"].astype(str).eq(contrast)
        & gsea["method"].astype(str).eq(method)
    ].copy()

    if d.empty:
        raise RuntimeError(
            f"No Stage 12I GSEA rows for {contrast} {method}."
        )

    same = d["same_nes_direction"].astype(bool)
    both = d["both_fdr05"].astype(bool)
    concordant = d["concordant_both_fdr05"].astype(bool)

    discordant_both = both & ~same

    sig = d.loc[concordant].copy()

    if not sig.empty:
        sig["min_abs_nes"] = np.minimum(
            pd.to_numeric(
                sig["rna_nes"],
                errors="coerce",
            ).abs(),
            pd.to_numeric(
                sig["protein_nes"],
                errors="coerce",
            ).abs(),
        )

        shared_genes = set()

        for value in sig["shared_leading_edge_genes"]:
            shared_genes.update(
                parse_genes(value)
            )

        shared_memberships = int(
            pd.to_numeric(
                sig["shared_leading_edge_n"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        median_min_abs_nes = float(
            sig["min_abs_nes"].median()
        )

        max_min_abs_nes = float(
            sig["min_abs_nes"].max()
        )

    else:
        shared_genes = set()
        shared_memberships = 0
        median_min_abs_nes = np.nan
        max_min_abs_nes = np.nan

    return {
        "gsea_matched_pathways_n": len(d),
        "gsea_same_nes_direction_n": int(same.sum()),
        "gsea_same_nes_direction_fraction":
            float(same.mean()),
        "gsea_both_fdr05_n": int(both.sum()),
        "gsea_concordant_both_fdr05_n":
            int(concordant.sum()),
        "gsea_concordant_both_fdr05_fraction":
            float(concordant.mean()),
        "gsea_discordant_both_fdr05_n":
            int(discordant_both.sum()),
        "gsea_median_min_abs_nes_concordant":
            median_min_abs_nes,
        "gsea_max_min_abs_nes_concordant":
            max_min_abs_nes,
        "gsea_shared_leading_edge_memberships_n":
            shared_memberships,
        "gsea_unique_shared_leading_edge_genes_n":
            len(shared_genes),
    }


def strongest_rrho_peak(
    rrho_summary: pd.DataFrame,
    rrho_peaks: pd.DataFrame,
    contrast: str,
    method: str,
):
    summary = rrho_summary.loc[
        rrho_summary["contrast_id"].astype(str).eq(contrast)
        & rrho_summary["method"].astype(str).eq(method)
    ].copy()

    if len(summary) != 1:
        raise RuntimeError(
            f"Expected one RRHO summary row for {contrast} {method}; "
            f"found {len(summary)}."
        )

    s = summary.iloc[0]

    quadrant = str(
        s["strongest_concordant_quadrant"]
    )

    peak = rrho_peaks.loc[
        rrho_peaks["contrast_id"].astype(str).eq(contrast)
        & rrho_peaks["method"].astype(str).eq(method)
        & rrho_peaks["quadrant"].astype(str).eq(quadrant)
    ].copy()

    if len(peak) != 1:
        raise RuntimeError(
            f"Expected one RRHO peak for {contrast} {method} "
            f"{quadrant}; found {len(peak)}."
        )

    p = peak.iloc[0]

    return {
        "rrho_spearman_rho":
            float(s["spearman_rho"]),
        "rrho_same_direction_fraction":
            float(s["same_direction_fraction"]),
        "rrho_strongest_concordant_quadrant":
            quadrant,
        "rrho_strongest_concordant_neglog10_BY":
            float(
                s[
                    "strongest_concordant_neglog10_BY"
                ]
            ),
        "rrho_peak_rna_tail_n":
            int(p["peak_rna_tail_n"]),
        "rrho_peak_protein_tail_n":
            int(p["peak_protein_tail_n"]),
        "rrho_peak_overlap_n":
            int(p["peak_overlap_n"]),
        "rrho_peak_enrichment_ratio":
            float(p["peak_enrichment_ratio"]),
        "rrho_peak_by_fdr":
            float(p["peak_by_fdr"]),
    }


def architecture_summary(
    area: pd.DataFrame,
    conventional: pd.DataFrame,
    gsea: pd.DataFrame,
    rrho_summary: pd.DataFrame,
    rrho_peaks: pd.DataFrame,
):
    rows = []

    tables = {
        "AREA": area,
        "CONVENTIONAL": conventional,
    }

    for contrast in CONTRASTS:
        for method in METHODS:
            row = {
                "contrast_id": contrast,
                "method": method,
            }

            row.update(
                gene_level_metrics(
                    tables[method],
                    contrast,
                    method,
                )
            )

            row.update(
                gsea_level_metrics(
                    gsea,
                    contrast,
                    method,
                )
            )

            row.update(
                strongest_rrho_peak(
                    rrho_summary,
                    rrho_peaks,
                    contrast,
                    method,
                )
            )

            rows.append(row)

    return pd.DataFrame(rows)


def formal_vs_both_audit(
    area: pd.DataFrame,
    conventional: pd.DataFrame,
):
    rows = []

    for method, table in [
        ("AREA", area),
        ("CONVENTIONAL", conventional),
    ]:
        for contrast in CONTRASTS:
            d = table.loc[
                table["contrast_id"].astype(str).eq(contrast)
            ].copy()

            both = d["both_individual_fdr05"].astype(bool)
            formal = d["crossomic_fdr05_bh"].astype(bool)

            for category, mask in [
                (
                    "formal_and_both_individual",
                    formal & both,
                ),
                (
                    "formal_only_not_both_individual",
                    formal & ~both,
                ),
                (
                    "both_individual_only_not_formal",
                    both & ~formal,
                ),
            ]:
                genes = sorted(
                    d.loc[
                        mask,
                        "gene_symbol",
                    ]
                    .astype(str)
                    .tolist()
                )

                rows.append(
                    {
                        "contrast_id": contrast,
                        "method": method,
                        "category": category,
                        "n_genes": len(genes),
                        "genes": ",".join(genes),
                    }
                )

    return pd.DataFrame(rows)


def shared_leading_edge_recurrence(
    gsea: pd.DataFrame,
):
    membership_rows = []

    sig = gsea.loc[
        gsea["concordant_both_fdr05"].astype(bool)
    ].copy()

    for r in sig.itertuples(index=False):
        genes = parse_genes(
            r.shared_leading_edge_genes
        )

        for gene in sorted(set(genes)):
            membership_rows.append(
                {
                    "contrast_id":
                        r.contrast_id,
                    "method": r.method,
                    "gene_set_source":
                        r.gene_set_source,
                    "term_name": r.term_name,
                    "gene_symbol": gene,
                    "rna_nes": r.rna_nes,
                    "protein_nes":
                        r.protein_nes,
                    "rna_fdr_qvalue":
                        r.rna_fdr_qvalue,
                    "protein_fdr_qvalue":
                        r.protein_fdr_qvalue,
                }
            )

    membership = pd.DataFrame(
        membership_rows
    )

    if membership.empty:
        recurrence = pd.DataFrame(
            columns=[
                "gene_symbol",
                "n_contrast_method_combinations",
                "n_contrasts",
                "n_methods",
                "n_shared_leading_edge_pathway_annotations",
                "n_reactome_annotations",
                "n_go_bp_annotations",
                "n_hallmark_annotations",
                "contrasts",
                "methods",
            ]
        )
        return membership, recurrence

    rows = []

    for gene, g in membership.groupby(
        "gene_symbol",
        sort=False,
    ):
        combos = (
            g[
                [
                    "contrast_id",
                    "method",
                ]
            ]
            .drop_duplicates()
        )

        source_counts = (
            g["gene_set_source"]
            .astype(str)
            .value_counts()
        )

        rows.append(
            {
                "gene_symbol": gene,
                "n_contrast_method_combinations":
                    len(combos),
                "n_contrasts":
                    g["contrast_id"].nunique(),
                "n_methods":
                    g["method"].nunique(),
                "n_shared_leading_edge_pathway_annotations":
                    len(g),
                "n_reactome_annotations":
                    int(
                        source_counts.get(
                            "REAC",
                            0,
                        )
                    ),
                "n_go_bp_annotations":
                    int(
                        source_counts.get(
                            "GO_BP",
                            0,
                        )
                    ),
                "n_hallmark_annotations":
                    int(
                        source_counts.get(
                            "HALLMARK",
                            0,
                        )
                    ),
                "contrasts":
                    ",".join(
                        sorted(
                            g[
                                "contrast_id"
                            ]
                            .astype(str)
                            .unique()
                        )
                    ),
                "methods":
                    ",".join(
                        sorted(
                            g[
                                "method"
                            ]
                            .astype(str)
                            .unique()
                        )
                    ),
            }
        )

    recurrence = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "n_contrast_method_combinations",
                "n_contrasts",
                "n_shared_leading_edge_pathway_annotations",
                "gene_symbol",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return membership, recurrence


def cross_method_comparison(
    area: pd.DataFrame,
    conventional: pd.DataFrame,
    gsea: pd.DataFrame,
    architecture: pd.DataFrame,
    membership: pd.DataFrame,
):
    rows = []

    for contrast in CONTRASTS:
        a_gene = set(
            area.loc[
                area["contrast_id"].astype(str).eq(contrast)
                & area["crossomic_fdr05_bh"].astype(bool),
                "gene_symbol",
            ].astype(str)
        )

        c_gene = set(
            conventional.loc[
                conventional["contrast_id"].astype(str).eq(contrast)
                & conventional["crossomic_fdr05_bh"].astype(bool),
                "gene_symbol",
            ].astype(str)
        )

        a_path = set(
            gsea.loc[
                gsea["contrast_id"].astype(str).eq(contrast)
                & gsea["method"].astype(str).eq("AREA")
                & gsea["concordant_both_fdr05"].astype(bool),
                ["gene_set_source", "term_name"],
            ]
            .astype(str)
            .agg("||".join, axis=1)
        )

        c_path = set(
            gsea.loc[
                gsea["contrast_id"].astype(str).eq(contrast)
                & gsea["method"].astype(str).eq("CONVENTIONAL")
                & gsea["concordant_both_fdr05"].astype(bool),
                ["gene_set_source", "term_name"],
            ]
            .astype(str)
            .agg("||".join, axis=1)
        )

        if membership.empty:
            a_le = set()
            c_le = set()
        else:
            a_le = set(
                membership.loc[
                    membership["contrast_id"].astype(str).eq(contrast)
                    & membership["method"].astype(str).eq("AREA"),
                    "gene_symbol",
                ].astype(str)
            )

            c_le = set(
                membership.loc[
                    membership["contrast_id"].astype(str).eq(contrast)
                    & membership["method"].astype(str).eq("CONVENTIONAL"),
                    "gene_symbol",
                ].astype(str)
            )

        a_arch = architecture.loc[
            architecture["contrast_id"].eq(contrast)
            & architecture["method"].eq("AREA")
        ].iloc[0]

        c_arch = architecture.loc[
            architecture["contrast_id"].eq(contrast)
            & architecture["method"].eq("CONVENTIONAL")
        ].iloc[0]

        rows.append(
            {
                "contrast_id": contrast,

                "area_formal_genes_n":
                    len(a_gene),
                "conventional_formal_genes_n":
                    len(c_gene),
                "formal_gene_overlap_n":
                    len(a_gene & c_gene),
                "formal_gene_union_n":
                    len(a_gene | c_gene),
                "formal_gene_jaccard":
                    jaccard(
                        a_gene,
                        c_gene,
                    ),

                "area_concordant_gsea_pathways_n":
                    len(a_path),
                "conventional_concordant_gsea_pathways_n":
                    len(c_path),
                "concordant_gsea_pathway_overlap_n":
                    len(a_path & c_path),
                "concordant_gsea_pathway_union_n":
                    len(a_path | c_path),
                "concordant_gsea_pathway_jaccard":
                    jaccard(
                        a_path,
                        c_path,
                    ),

                "area_unique_shared_leading_edge_genes_n":
                    len(a_le),
                "conventional_unique_shared_leading_edge_genes_n":
                    len(c_le),
                "shared_leading_edge_gene_overlap_n":
                    len(a_le & c_le),
                "shared_leading_edge_gene_union_n":
                    len(a_le | c_le),
                "shared_leading_edge_gene_jaccard":
                    jaccard(
                        a_le,
                        c_le,
                    ),

                "area_rrho_spearman_rho":
                    a_arch["rrho_spearman_rho"],
                "conventional_rrho_spearman_rho":
                    c_arch["rrho_spearman_rho"],
                "rrho_spearman_rho_difference_area_minus_conventional":
                    (
                        a_arch["rrho_spearman_rho"]
                        - c_arch["rrho_spearman_rho"]
                    ),

                "area_rrho_same_direction_fraction":
                    a_arch[
                        "rrho_same_direction_fraction"
                    ],
                "conventional_rrho_same_direction_fraction":
                    c_arch[
                        "rrho_same_direction_fraction"
                    ],
                "rrho_same_direction_fraction_difference_area_minus_conventional":
                    (
                        a_arch[
                            "rrho_same_direction_fraction"
                        ]
                        - c_arch[
                            "rrho_same_direction_fraction"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows)


def main():
    (
        area,
        conventional,
        gsea,
        rrho_summary,
        rrho_peaks,
    ) = load_inputs()

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    architecture = architecture_summary(
        area,
        conventional,
        gsea,
        rrho_summary,
        rrho_peaks,
    )

    gene_audit = formal_vs_both_audit(
        area,
        conventional,
    )

    membership, recurrence = (
        shared_leading_edge_recurrence(
            gsea
        )
    )

    cross_method = cross_method_comparison(
        area,
        conventional,
        gsea,
        architecture,
        membership,
    )

    architecture_path = (
        OUTDIR
        / "CONVERGENCE_ARCHITECTURE_SUMMARY.tsv"
    )

    cross_method_path = (
        OUTDIR
        / "CROSS_METHOD_ARCHITECTURE_COMPARISON.tsv"
    )

    gene_audit_path = (
        OUTDIR
        / "FORMAL_VS_BOTH_INDIVIDUAL_GENE_AUDIT.tsv"
    )

    membership_path = (
        OUTDIR
        / "CONTRAST_METHOD_SHARED_LEADING_EDGE_GENES.tsv"
    )

    recurrence_path = (
        OUTDIR
        / "SHARED_LEADING_EDGE_GENE_RECURRENCE.tsv"
    )

    readme_path = (
        OUTDIR
        / "README_STAGE12K.txt"
    )

    architecture.to_csv(
        architecture_path,
        sep="\t",
        index=False,
    )

    cross_method.to_csv(
        cross_method_path,
        sep="\t",
        index=False,
    )

    gene_audit.to_csv(
        gene_audit_path,
        sep="\t",
        index=False,
    )

    membership.to_csv(
        membership_path,
        sep="\t",
        index=False,
    )

    recurrence.to_csv(
        recurrence_path,
        sep="\t",
        index=False,
    )

    readme = """STAGE 12K — CONVERGENCE ARCHITECTURE
====================================

Purpose
-------
Describe gene-, pathway-, and rank-level RNA-protein convergence side by side.

No composite architecture score is constructed.
No new biological cutoff is introduced.

Gene-level dimensions
---------------------
formal_conjunction_fdr05_n
    Number of genes passing same-direction max-p conjunction + BH FDR < .05.

both_individual_fdr05_n
    Genes individually FDR < .05 in RNA and protein with concordant direction.

formal_and_both_individual_n
formal_not_both_individual_n
both_individual_not_formal_n
    Explicitly show the relationship between those two evidence definitions.

Pathway-level dimensions
------------------------
gsea_concordant_both_fdr05_n
    Annotation terms significant at FDR < .05 in both RNA and protein with
    the same NES direction.

IMPORTANT:
    GO / Reactome terms are redundant. This count is an annotation-level
    descriptor, not a count of independent biological mechanisms.

gsea_unique_shared_leading_edge_genes_n
    Number of unique genes occurring in the RNA AND protein leading edge of at
    least one concordant dual-FDR pathway.

SHARED_LEADING_EDGE_GENE_RECURRENCE.tsv
    Counts repeated shared-leading-edge membership.
    Large counts can partly reflect annotation redundancy, so Reactome / GO /
    Hallmark counts are retained separately.

Rank-level dimensions
---------------------
RRHO Spearman rho
same-direction fraction
strongest concordant RRHO quadrant
peak tail sizes
peak enrichment ratio

RRHO peak tails can be broad. Peak membership is NOT treated as specific
driver-gene evidence.

Cross-method comparison
-----------------------
AREA vs conventional:
    formal-gene overlap / Jaccard
    concordant pathway overlap / Jaccard
    shared-leading-edge-gene overlap / Jaccard
    RRHO correlation differences

Interpretation
--------------
The intended test is descriptive:

    Does NCI->AD show much more gene-level conjunction than MCI->AD while
    MCI->AD retains strong pathway-level and rank-level convergence?

If yes, that supports the existence of different cross-omic convergence
architectures without requiring an arbitrary classifier.
"""

    readme_path.write_text(
        readme
    )

    print("=" * 120)
    print(
        "STAGE 12K — CONVERGENCE ARCHITECTURE"
    )
    print("=" * 120)

    show_cols = [
        "contrast_id",
        "method",
        "both_individual_fdr05_n",
        "formal_conjunction_fdr05_n",
        "formal_and_both_individual_n",
        "both_individual_not_formal_n",
        "gsea_concordant_both_fdr05_n",
        "gsea_unique_shared_leading_edge_genes_n",
        "rrho_spearman_rho",
        "rrho_same_direction_fraction",
        "rrho_strongest_concordant_quadrant",
        "rrho_peak_rna_tail_n",
        "rrho_peak_protein_tail_n",
        "rrho_peak_enrichment_ratio",
    ]

    print(
        architecture[
            show_cols
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print(
        "AREA VS CONVENTIONAL ARCHITECTURE CONSISTENCY"
    )
    print("=" * 120)

    print(
        cross_method
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print(
        "TOP RECURRENT SHARED RNA/PROTEIN GSEA LEADING-EDGE GENES"
    )
    print("=" * 120)

    if recurrence.empty:
        print("No recurrent genes.")
    else:
        print(
            recurrence
            .head(75)
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in [
        architecture_path,
        cross_method_path,
        gene_audit_path,
        membership_path,
        recurrence_path,
        readme_path,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()

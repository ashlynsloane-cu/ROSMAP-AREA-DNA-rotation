#!/usr/bin/env python3
"""
Stage 12D — assemble biologically interpretable brain candidate sets.

This script does NO new statistical testing and uses NO arbitrary thresholds.

It organizes existing Stage 12 results into two inferentially defined tracks:

TRACK 1 — FORMAL CROSS-OMIC DISCOVERIES
    conjunction BH FDR < 0.05

Categories:
    shared
        formal discovery by both AREA and DESeq2/limma
    AREA_only
        formal discovery by AREA only
    conventional_only
        formal discovery by DESeq2/limma only

TRACK 2 — NEAR-CONVERGENCE
For MCI->AD and late Braak only:
    same-direction RNA/protein signal
    AND RNA individual FDR < 0.05
    AND protein individual FDR < 0.05
    BUT conjunction BH FDR >= 0.05

Categories:
    shared_near
    AREA_only_near
    conventional_only_near

This explicitly distinguishes:
    "no formal conjunction discovery"
from
    "no RNA-protein evidence."

Monocyte evidence from Stage 12C is attached to formal brain candidates as
peripheral follow-up only.

Outputs
-------
results/stage12_brain_convergence/stage12D_biological_candidate_sets/
    FORMAL_BRAIN_DISCOVERY_CATEGORIES.tsv
    SHARED_FORMAL_DISCOVERIES.tsv
    AREA_ONLY_FORMAL_DISCOVERIES.tsv
    CONVENTIONAL_ONLY_FORMAL_DISCOVERIES.tsv
    MCI_AD_LATE_BRAAK_NEAR_CONVERGENCE.tsv
    CANDIDATE_SET_SUMMARY.tsv
    README_STAGE12D.txt
"""

from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(os.environ["ROSMAP_OFFLINE_ROOT"]).expanduser().resolve()
AREA_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
    / "ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv"
)

CONV_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12B_deseq2_limma_conjunction"
    / "ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv"
)

MONOCYTE_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12C_monocyte_followup"
    / "BRAIN_CANDIDATE_MONOCYTE_EVIDENCE.tsv"
)

OUTDIR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12D_biological_candidate_sets"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


FORMAL_CONTRASTS = [
    "cogdx_NCI_vs_MCI",
    "cogdx_MCI_vs_AD",
    "cogdx_NCI_vs_AD",
    "Braak_0-2_vs_3-4",
    "Braak_3-4_vs_5-6",
    "CERAD_3-4_vs_1-2",
]

NEAR_CONTRASTS = [
    "cogdx_MCI_vs_AD",
    "Braak_3-4_vs_5-6",
]


def load_inputs():
    for path in [AREA_PATH, CONV_PATH, MONOCYTE_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    area = pd.read_csv(
        AREA_PATH,
        sep="\t",
        low_memory=False,
    )

    conv = pd.read_csv(
        CONV_PATH,
        sep="\t",
        low_memory=False,
    )

    mono = pd.read_csv(
        MONOCYTE_PATH,
        sep="\t",
        low_memory=False,
    )

    return area, conv, mono


def require_columns(df, columns, label):
    missing = set(columns) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {sorted(missing)}"
        )


def area_formal_table(area):
    require_columns(
        area,
        [
            "contrast_id",
            "gene_symbol",
            "crossomic_fdr05_bh",
            "conjunction_pvalue",
            "conjunction_fdr_bh",
            "rna_pvalue",
            "rna_padj",
            "rna_direction",
            "protein_pvalue",
            "protein_padj",
            "protein_direction",
            "both_individual_fdr05",
            "direction_concordant",
        ],
        "AREA",
    )

    x = area.loc[
        area["crossomic_fdr05_bh"].astype(bool)
    ].copy()

    return x


def conv_formal_table(conv):
    require_columns(
        conv,
        [
            "contrast_id",
            "gene_symbol",
            "crossomic_fdr05_bh",
            "conjunction_pvalue",
            "conjunction_fdr_bh",
            "rna_pvalue",
            "rna_padj",
            "rna_effect",
            "protein_pvalue",
            "protein_padj",
            "protein_effect",
            "both_individual_fdr05",
            "direction_concordant",
        ],
        "conventional",
    )

    x = conv.loc[
        conv["crossomic_fdr05_bh"].astype(bool)
    ].copy()

    return x


def build_formal_categories(area, conv, mono):
    a = area_formal_table(area)
    c = conv_formal_table(conv)

    keys = sorted(
        set(
            zip(
                a["contrast_id"].astype(str),
                a["gene_symbol"].astype(str),
            )
        )
        | set(
            zip(
                c["contrast_id"].astype(str),
                c["gene_symbol"].astype(str),
            )
        )
    )

    rows = []

    area_index = {
        (str(r.contrast_id), str(r.gene_symbol)): r
        for r in a.itertuples(index=False)
    }

    conv_index = {
        (str(r.contrast_id), str(r.gene_symbol)): r
        for r in c.itertuples(index=False)
    }

    for key in keys:
        contrast_id, gene = key

        ar = area_index.get(key)
        cr = conv_index.get(key)

        if ar is not None and cr is not None:
            category = "shared"
        elif ar is not None:
            category = "AREA_only"
        else:
            category = "conventional_only"

        row = {
            "contrast_id": contrast_id,
            "gene_symbol": gene,
            "brain_category": category,
            "AREA_formal_BH": ar is not None,
            "conventional_formal_BH": cr is not None,
        }

        if ar is not None:
            row.update(
                {
                    "AREA_conjunction_pvalue":
                        ar.conjunction_pvalue,
                    "AREA_conjunction_fdr_bh":
                        ar.conjunction_fdr_bh,
                    "AREA_rna_pvalue":
                        ar.rna_pvalue,
                    "AREA_rna_padj":
                        ar.rna_padj,
                    "AREA_rna_direction":
                        ar.rna_direction,
                    "AREA_protein_pvalue":
                        ar.protein_pvalue,
                    "AREA_protein_padj":
                        ar.protein_padj,
                    "AREA_protein_direction":
                        ar.protein_direction,
                }
            )

        if cr is not None:
            row.update(
                {
                    "conventional_conjunction_pvalue":
                        cr.conjunction_pvalue,
                    "conventional_conjunction_fdr_bh":
                        cr.conjunction_fdr_bh,
                    "DESeq2_pvalue":
                        cr.rna_pvalue,
                    "DESeq2_padj":
                        cr.rna_padj,
                    "DESeq2_log2fc":
                        cr.rna_effect,
                    "limma_pvalue":
                        cr.protein_pvalue,
                    "limma_padj":
                        cr.protein_padj,
                    "limma_logfc":
                        cr.protein_effect,
                }
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    # Attach the Stage 12C targeted peripheral evidence.
    mono_keep = [
        c
        for c in [
            "contrast_id",
            "gene_symbol",
            "monocyte_area_feature_id",
            "area_pvalue",
            "area_padj",
            "area_direction",
            "monocyte_deseq2_feature_id",
            "deseq2_log2fc_stage11_orientation",
            "deseq2_pvalue",
            "deseq2_padj",
            "monocyte_area_fdr05",
            "monocyte_deseq2_fdr05",
            "monocyte_methods_direction_agree",
            "monocyte_area_matches_brain_area_direction",
            "monocyte_deseq2_matches_brain_conventional_direction",
        ]
        if c in mono.columns
    ]

    mono_small = mono[mono_keep].copy()

    if mono_small.duplicated(
        ["contrast_id", "gene_symbol"]
    ).any():
        raise RuntimeError(
            "Duplicate monocyte rows for a formal brain candidate."
        )

    out = out.merge(
        mono_small,
        on=["contrast_id", "gene_symbol"],
        how="left",
        validate="one_to_one",
    )

    out["monocyte_testable_AREA"] = (
        out["area_pvalue"].notna()
        if "area_pvalue" in out.columns
        else False
    )

    out["monocyte_testable_DESeq2"] = (
        out["deseq2_pvalue"].notna()
        if "deseq2_pvalue" in out.columns
        else False
    )

    category_order = {
        "shared": 0,
        "AREA_only": 1,
        "conventional_only": 2,
    }

    out["_category_order"] = out[
        "brain_category"
    ].map(category_order)

    out = (
        out
        .sort_values(
            [
                "contrast_id",
                "_category_order",
                "gene_symbol",
            ],
            kind="mergesort",
        )
        .drop(columns="_category_order")
        .reset_index(drop=True)
    )

    return out


def build_near_convergence(area, conv):
    """
    Near-convergence is strictly inferential:
      both modality-specific padj < .05 + same direction,
      but formal conjunction BH FDR does not pass.
    """
    rows = []

    for contrast_id in NEAR_CONTRASTS:
        a = area.loc[
            area["contrast_id"].astype(str).eq(contrast_id)
            & area["both_individual_fdr05"].astype(bool)
            & ~area["crossomic_fdr05_bh"].astype(bool)
        ].copy()

        c = conv.loc[
            conv["contrast_id"].astype(str).eq(contrast_id)
            & conv["both_individual_fdr05"].astype(bool)
            & ~conv["crossomic_fdr05_bh"].astype(bool)
        ].copy()

        a_index = {
            str(r.gene_symbol): r
            for r in a.itertuples(index=False)
        }

        c_index = {
            str(r.gene_symbol): r
            for r in c.itertuples(index=False)
        }

        genes = sorted(
            set(a_index) | set(c_index)
        )

        for gene in genes:
            ar = a_index.get(gene)
            cr = c_index.get(gene)

            if ar is not None and cr is not None:
                category = "shared_near"
            elif ar is not None:
                category = "AREA_only_near"
            else:
                category = "conventional_only_near"

            row = {
                "contrast_id": contrast_id,
                "gene_symbol": gene,
                "near_category": category,
                "AREA_both_individual_FDR05":
                    ar is not None,
                "conventional_both_individual_FDR05":
                    cr is not None,
            }

            if ar is not None:
                row.update(
                    {
                        "AREA_conjunction_pvalue":
                            ar.conjunction_pvalue,
                        "AREA_conjunction_fdr_bh":
                            ar.conjunction_fdr_bh,
                        "AREA_rna_pvalue":
                            ar.rna_pvalue,
                        "AREA_rna_padj":
                            ar.rna_padj,
                        "AREA_rna_direction":
                            ar.rna_direction,
                        "AREA_protein_pvalue":
                            ar.protein_pvalue,
                        "AREA_protein_padj":
                            ar.protein_padj,
                        "AREA_protein_direction":
                            ar.protein_direction,
                    }
                )

            if cr is not None:
                row.update(
                    {
                        "conventional_conjunction_pvalue":
                            cr.conjunction_pvalue,
                        "conventional_conjunction_fdr_bh":
                            cr.conjunction_fdr_bh,
                        "DESeq2_pvalue":
                            cr.rna_pvalue,
                        "DESeq2_padj":
                            cr.rna_padj,
                        "DESeq2_log2fc":
                            cr.rna_effect,
                        "limma_pvalue":
                            cr.protein_pvalue,
                        "limma_padj":
                            cr.protein_padj,
                        "limma_logfc":
                            cr.protein_effect,
                    }
                )

            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    category_order = {
        "shared_near": 0,
        "AREA_only_near": 1,
        "conventional_only_near": 2,
    }

    out["_category_order"] = out[
        "near_category"
    ].map(category_order)

    return (
        out
        .sort_values(
            [
                "contrast_id",
                "_category_order",
                "gene_symbol",
            ],
            kind="mergesort",
        )
        .drop(columns="_category_order")
        .reset_index(drop=True)
    )


def build_summary(formal, near):
    rows = []

    for (contrast_id, category), g in formal.groupby(
        ["contrast_id", "brain_category"],
        sort=False,
    ):
        rows.append(
            {
                "analysis_track": "formal_conjunction_BH",
                "contrast_id": contrast_id,
                "category": category,
                "n_genes": len(g),
            }
        )

    if not near.empty:
        for (contrast_id, category), g in near.groupby(
            ["contrast_id", "near_category"],
            sort=False,
        ):
            rows.append(
                {
                    "analysis_track":
                        "dual_individual_FDR05_no_conjunction",
                    "contrast_id": contrast_id,
                    "category": category,
                    "n_genes": len(g),
                }
            )

    return pd.DataFrame(rows)


def main():
    area, conv, mono = load_inputs()

    formal = build_formal_categories(
        area,
        conv,
        mono,
    )

    near = build_near_convergence(
        area,
        conv,
    )

    summary = build_summary(
        formal,
        near,
    )

    formal_path = (
        OUTDIR / "FORMAL_BRAIN_DISCOVERY_CATEGORIES.tsv"
    )
    shared_path = (
        OUTDIR / "SHARED_FORMAL_DISCOVERIES.tsv"
    )
    area_only_path = (
        OUTDIR / "AREA_ONLY_FORMAL_DISCOVERIES.tsv"
    )
    conv_only_path = (
        OUTDIR / "CONVENTIONAL_ONLY_FORMAL_DISCOVERIES.tsv"
    )
    near_path = (
        OUTDIR / "MCI_AD_LATE_BRAAK_NEAR_CONVERGENCE.tsv"
    )
    summary_path = (
        OUTDIR / "CANDIDATE_SET_SUMMARY.tsv"
    )
    readme_path = (
        OUTDIR / "README_STAGE12D.txt"
    )

    formal.to_csv(
        formal_path,
        sep="\t",
        index=False,
    )

    formal.loc[
        formal["brain_category"].eq("shared")
    ].to_csv(
        shared_path,
        sep="\t",
        index=False,
    )

    formal.loc[
        formal["brain_category"].eq("AREA_only")
    ].to_csv(
        area_only_path,
        sep="\t",
        index=False,
    )

    formal.loc[
        formal["brain_category"].eq("conventional_only")
    ].to_csv(
        conv_only_path,
        sep="\t",
        index=False,
    )

    near.to_csv(
        near_path,
        sep="\t",
        index=False,
    )

    summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
    )

    readme = """STAGE 12D BIOLOGICAL CANDIDATE SETS
========================================

No new statistical tests or arbitrary cutoffs are introduced here.

TRACK 1: FORMAL CROSS-OMIC DISCOVERIES
--------------------------------------
Definition:
    conjunction BH FDR < 0.05

Categories:
    shared
        formal in both AREA and DESeq2/limma
    AREA_only
        formal in AREA only
    conventional_only
        formal in DESeq2/limma only

These are the primary gene sets for biological interpretation.

TRACK 2: MCI->AD / LATE BRAAK NEAR-CONVERGENCE
-----------------------------------------------
Definition:
    RNA individual padj < 0.05
    AND protein individual padj < 0.05
    AND same RNA/protein direction
    AND conjunction BH FDR >= 0.05

These genes are NOT formal conjunction discoveries.
They are retained to understand why the two contrasts have substantial
within-modality evidence but no genes that survive the stricter formal
cross-omic conjunction correction.

MONOCYTE
--------
Monocyte values are attached only to the upstream brain-derived formal
candidate sets. Monocyte is a peripheral follow-up, not an independent
brain replication and not a new candidate discovery screen.

Biological deep-dive priority:
    1. shared formal discoveries
    2. AREA-only formal discoveries
    3. conventional-only formal discoveries as comparator
    4. shared and method-specific near-convergence in MCI->AD / late Braak
"""

    readme_path.write_text(readme)

    print("=" * 115)
    print("STAGE 12D — BIOLOGICAL CANDIDATE SET ASSEMBLY")
    print("=" * 115)

    print("\nFORMAL DISCOVERY SETS")
    print("-" * 115)

    formal_summary = (
        formal
        .groupby(
            ["contrast_id", "brain_category"],
            sort=False,
        )
        .size()
        .rename("n_genes")
        .reset_index()
    )

    print(
        formal_summary.to_string(index=False)
    )

    print("\nMCI->AD / LATE BRAAK NEAR-CONVERGENCE")
    print("-" * 115)

    if near.empty:
        print("No dual-individual-FDR near-convergence genes found.")
    else:
        near_summary = (
            near
            .groupby(
                ["contrast_id", "near_category"],
                sort=False,
            )
            .size()
            .rename("n_genes")
            .reset_index()
        )

        print(
            near_summary.to_string(index=False)
        )

        print("\nGENES")
        print(
            near[
                [
                    "contrast_id",
                    "gene_symbol",
                    "near_category",
                    "AREA_conjunction_fdr_bh",
                    "conventional_conjunction_fdr_bh",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )

    print("\nFORMAL SHARED GENES")
    print("-" * 115)

    shared = formal.loc[
        formal["brain_category"].eq("shared")
    ]

    print(
        shared[
            [
                "contrast_id",
                "gene_symbol",
                "AREA_conjunction_fdr_bh",
                "conventional_conjunction_fdr_bh",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nFORMAL AREA-ONLY GENES")
    print("-" * 115)

    area_only = formal.loc[
        formal["brain_category"].eq("AREA_only")
    ]

    if area_only.empty:
        print("None.")
    else:
        print(
            area_only[
                [
                    "contrast_id",
                    "gene_symbol",
                    "AREA_conjunction_fdr_bh",
                    "conventional_conjunction_fdr_bh",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in [
        formal_path,
        shared_path,
        area_only_path,
        conv_only_path,
        near_path,
        summary_path,
        readme_path,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()

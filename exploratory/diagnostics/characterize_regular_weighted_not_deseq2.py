#!/usr/bin/env python3
"""
characterize_regular_weighted_not_deseq2.py
===========================================

Characterize the key Venn region:

    Adjusted Regular AREA significant
    AND Weighted AREA significant
    AND DESeq2 non-significant

Expected size in the locked ROSMAP comparison: 468 genes.

Outputs:
- full 468-gene table
- top 50 by joint AREA support
- gene-symbol list for enrichment
- Ensembl/gene_id list
- summary metrics
- sign-concordant subset
- sign-discordant subset
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = (
    "results/visualizations/"
    "final_locked_covariate_DESeq2_Regular_Weighted_venn_gene_membership.csv"
)

OUTDIR = Path("results/characterization_468")

EXPECTED_N = 468


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    required = [
        "gene_id",
        "ensembl_id_version",
        "gene_symbol",
        "Regular_comparison_padj_BH",
        "adjusted_Regular_AREA_Z",
        "Weighted_comparison_padj_BH",
        "adjusted_AREA_Z",
        "DESeq2_comparison_padj_BH",
        "DESeq2_pvalue",
        "log2FoldChange",
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
        "overlap_region",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}"
        )

    # Target region:
    # Regular AREA significant + Weighted AREA significant + DESeq2 non-significant.
    sub = df.loc[
        (~df["DESeq2_significant"].astype(bool))
        & (df["Regular_AREA_significant"].astype(bool))
        & (df["Weighted_AREA_significant"].astype(bool))
    ].copy()

    if len(sub) != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N} genes in target region, found {len(sub)}."
        )

    # Numeric coercion.
    numeric_cols = [
        "Regular_comparison_padj_BH",
        "adjusted_Regular_AREA_Z",
        "Weighted_comparison_padj_BH",
        "adjusted_AREA_Z",
        "DESeq2_comparison_padj_BH",
        "DESeq2_pvalue",
        "log2FoldChange",
    ]

    for col in numeric_cols:
        sub[col] = pd.to_numeric(
            sub[col],
            errors="coerce",
        )

    # Direction/sign concordance across the two AREA methods.
    sub["abs_regular_z"] = sub[
        "adjusted_Regular_AREA_Z"
    ].abs()

    sub["abs_weighted_z"] = sub[
        "adjusted_AREA_Z"
    ].abs()

    sub["regular_sign"] = np.sign(
        sub["adjusted_Regular_AREA_Z"]
    )

    sub["weighted_sign"] = np.sign(
        sub["adjusted_AREA_Z"]
    )

    sub["sign_concordant"] = (
        sub["regular_sign"]
        == sub["weighted_sign"]
    )

    # Joint evidence across the two AREA methods.
    regular_q = sub[
        "Regular_comparison_padj_BH"
    ].clip(lower=1e-300)

    weighted_q = sub[
        "Weighted_comparison_padj_BH"
    ].clip(lower=1e-300)

    sub["joint_strength_score"] = (
        -np.log10(regular_q)
        + -np.log10(weighted_q)
    )

    # Conservative support measure: worst FDR of the two AREA methods.
    sub["max_BH_across_AREA_methods"] = sub[
        [
            "Regular_comparison_padj_BH",
            "Weighted_comparison_padj_BH",
        ]
    ].max(axis=1)

    # Rank strongest joint support first.
    sub = sub.sort_values(
        by=[
            "joint_strength_score",
            "max_BH_across_AREA_methods",
            "abs_regular_z",
            "abs_weighted_z",
        ],
        ascending=[
            False,
            True,
            False,
            False,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    preferred_cols = [
        "gene_id",
        "ensembl_id_version",
        "gene_symbol",
        "Regular_comparison_padj_BH",
        "adjusted_Regular_AREA_Z",
        "Weighted_comparison_padj_BH",
        "adjusted_AREA_Z",
        "DESeq2_comparison_padj_BH",
        "DESeq2_pvalue",
        "log2FoldChange",
        "sign_concordant",
        "joint_strength_score",
        "max_BH_across_AREA_methods",
        "overlap_region",
    ]

    other_cols = [
        c for c in sub.columns
        if c not in preferred_cols
    ]

    sub = sub[
        preferred_cols + other_cols
    ]

    # Full table.
    full_out = (
        OUTDIR
        / "regular_weighted_only_not_deseq2_468_genes.csv"
    )
    sub.to_csv(
        full_out,
        index=False,
    )

    # Top 50.
    top50_out = (
        OUTDIR
        / "regular_weighted_only_not_deseq2_top50.csv"
    )
    sub.head(50).to_csv(
        top50_out,
        index=False,
    )

    # Gene-symbol list for enrichment.
    symbols = (
        sub["gene_symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    symbols = symbols[
        (symbols != "")
        & (symbols.str.lower() != "nan")
    ]

    symbols = (
        symbols
        .drop_duplicates()
        .sort_values()
    )

    symbols_txt = (
        OUTDIR
        / "regular_weighted_only_not_deseq2_symbols.txt"
    )

    symbols.to_csv(
        symbols_txt,
        index=False,
        header=False,
    )

    # Ensembl / gene_id list.
    gene_ids = (
        sub["gene_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
    )

    gene_ids_txt = (
        OUTDIR
        / "regular_weighted_only_not_deseq2_gene_ids.txt"
    )

    gene_ids.to_csv(
        gene_ids_txt,
        index=False,
        header=False,
    )

    # Summary metrics.
    n_concordant = int(
        sub["sign_concordant"].sum()
    )

    n_discordant = int(
        (~sub["sign_concordant"]).sum()
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "n_genes",
                "n_unique_gene_symbols",
                "n_sign_concordant",
                "n_sign_discordant",
                "fraction_sign_concordant",
                "median_regular_BH",
                "median_weighted_BH",
                "median_DESeq2_BH",
                "median_abs_regular_z",
                "median_abs_weighted_z",
                "median_abs_DESeq2_log2FC",
            ],
            "value": [
                len(sub),
                symbols.nunique(),
                n_concordant,
                n_discordant,
                float(
                    sub["sign_concordant"].mean()
                ),
                float(
                    sub[
                        "Regular_comparison_padj_BH"
                    ].median()
                ),
                float(
                    sub[
                        "Weighted_comparison_padj_BH"
                    ].median()
                ),
                float(
                    sub[
                        "DESeq2_comparison_padj_BH"
                    ].median()
                ),
                float(
                    sub[
                        "abs_regular_z"
                    ].median()
                ),
                float(
                    sub[
                        "abs_weighted_z"
                    ].median()
                ),
                float(
                    sub[
                        "log2FoldChange"
                    ].abs().median()
                ),
            ],
        }
    )

    summary_out = (
        OUTDIR
        / "regular_weighted_only_not_deseq2_summary.csv"
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    # Concordance subsets.
    concordant_out = (
        OUTDIR
        / "regular_weighted_only_not_deseq2_sign_concordant.csv"
    )

    discordant_out = (
        OUTDIR
        / "regular_weighted_only_not_deseq2_sign_discordant.csv"
    )

    sub.loc[
        sub["sign_concordant"]
    ].to_csv(
        concordant_out,
        index=False,
    )

    sub.loc[
        ~sub["sign_concordant"]
    ].to_csv(
        discordant_out,
        index=False,
    )

    print("=" * 80)
    print("468-GENE CHARACTERIZATION")
    print("=" * 80)
    print(f"Input: {INPUT}")
    print(f"Genes in target region: {len(sub)}")
    print(f"Unique gene symbols: {symbols.nunique()}")
    print(f"Sign-concordant: {n_concordant}")
    print(f"Sign-discordant: {n_discordant}")
    print(
        "Fraction sign-concordant: "
        f"{sub['sign_concordant'].mean():.3f}"
    )

    print("\nTop 15 genes by joint AREA support:")

    print(
        sub[
            [
                "gene_symbol",
                "gene_id",
                "Regular_comparison_padj_BH",
                "Weighted_comparison_padj_BH",
                "DESeq2_comparison_padj_BH",
                "adjusted_Regular_AREA_Z",
                "adjusted_AREA_Z",
                "log2FoldChange",
                "joint_strength_score",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nWrote:")
    print(f"  {full_out}")
    print(f"  {top50_out}")
    print(f"  {symbols_txt}")
    print(f"  {gene_ids_txt}")
    print(f"  {summary_out}")
    print(f"  {concordant_out}")
    print(f"  {discordant_out}")


if __name__ == "__main__":
    main()

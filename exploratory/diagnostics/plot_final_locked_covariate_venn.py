#!/usr/bin/env python3
"""
plot_final_locked_covariate_venn.py
===================================

Final three-method Venn for the locked ROSMAP comparison:

1. DESeq2: AD4 vs NCI1, covariate-adjusted, n=418
2. Adjusted Regular AREA: AD4 vs NCI1, same endpoint participants/covariates
3. Weighted AREA: Cognitive_stage (NCI -> MCI -> AD), n=619

The 12 DESeq2 genes with persistent betaConv=FALSE are excluded from ALL THREE
methods for this cross-method comparison only. BH FDR is then recomputed from
the raw p-values on the identical post-exclusion gene universe.

This preserves:
- the standalone DESeq2 and AREA analyses exactly as originally run;
- a clean common universe for the Venn comparison.

Expected common universe:
    33,006 - 12 = 32,994 genes
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--deseq2-file", required=True)
    p.add_argument("--regular-area-file", required=True)
    p.add_argument("--weighted-area-file", required=True)

    p.add_argument("--deseq2-manifest", required=True)
    p.add_argument("--regular-manifest", required=True)

    p.add_argument("--exclude-gene-file", required=True)

    p.add_argument("--expected-deseq2-n", type=int, default=418)
    p.add_argument("--expected-regular-n", type=int, default=418)
    p.add_argument("--expected-weighted-n", type=int, default=619)
    p.add_argument("--expected-excluded-genes", type=int, default=12)
    p.add_argument("--expected-common-genes", type=int, default=32994)

    p.add_argument("--cutoff", type=float, default=0.05)

    p.add_argument(
        "--title",
        default=(
            "Covariate-Adjusted Mean- and Rank-Based "
            "Gene Signatures Across Cognitive AD"
        ),
    )

    p.add_argument("--output", required=True)

    return p.parse_args()


def require_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def bh_adjust_with_nan(values):
    p = np.asarray(values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)

    valid = np.isfinite(p)
    pv = p[valid]

    if len(pv) == 0:
        return out

    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]

    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    tmp = np.empty(len(pv), dtype=float)
    tmp[order] = q
    out[valid] = tmp

    return out


def manifest_ids(df, expected_n, label):
    require_columns(df, ["sample_id"], label)

    ids = (
        df["sample_id"]
        .astype(str)
        .str.strip()
    )

    if ids.duplicated().any():
        dup = ids[ids.duplicated()].unique().tolist()[:10]
        raise ValueError(
            f"{label} contains duplicate sample IDs: {dup}"
        )

    n = ids.nunique()

    if n != expected_n:
        raise ValueError(
            f"{label} sample count is {n}, expected {expected_n}."
        )

    return set(ids), n


def weighted_n(df, expected_n):
    require_columns(
        df,
        ["gene_id", "phenotype", "n_samples", "adjusted_pvalue"],
        "Weighted AREA",
    )

    ns = (
        pd.to_numeric(df["n_samples"], errors="coerce")
        .dropna()
        .unique()
    )

    if len(ns) != 1:
        raise ValueError(
            f"Weighted AREA has {len(ns)} distinct n_samples values."
        )

    n = int(ns[0])

    if n != expected_n:
        raise ValueError(
            f"Weighted AREA n={n}, expected {expected_n}."
        )

    phenos = (
        df["phenotype"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if phenos != ["Cognitive_stage"]:
        raise ValueError(
            "Weighted AREA file is not a pure Cognitive_stage result: "
            f"{phenos}"
        )

    return n


def region_name(de, reg, wgt):
    if de and reg and wgt:
        return "All three"
    if de and reg:
        return "DESeq2 + Adjusted Regular AREA only"
    if de and wgt:
        return "DESeq2 + Weighted AREA only"
    if reg and wgt:
        return "Adjusted Regular AREA + Weighted AREA only"
    if de:
        return "DESeq2 only"
    if reg:
        return "Adjusted Regular AREA only"
    if wgt:
        return "Weighted AREA only"
    return "Not significant"


def venn_counts(A, B, C):
    return {
        "DESeq2 only": len(A - (B | C)),
        "Adjusted Regular AREA only": len(B - (A | C)),
        "Weighted AREA only": len(C - (A | B)),
        "DESeq2 + Adjusted Regular AREA only": len((A & B) - C),
        "DESeq2 + Weighted AREA only": len((A & C) - B),
        "Adjusted Regular AREA + Weighted AREA only": len((B & C) - A),
        "All three": len(A & B & C),
        "Total unique": len(A | B | C),
    }


def draw_venn(
    A,
    B,
    C,
    counts,
    title,
    output,
    deseq_n,
    regular_n,
    weighted_n_samples,
    cutoff,
):
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.set_aspect("equal")
    ax.axis("off")

    left = (0.38, 0.57)
    right = (0.62, 0.57)
    top = (0.50, 0.38)
    r = 0.27

    circles = [
        Circle(left, r, alpha=0.22, linewidth=2.0),
        Circle(right, r, alpha=0.22, linewidth=2.0),
        Circle(top, r, alpha=0.22, linewidth=2.0),
    ]

    for c in circles:
        ax.add_patch(c)

    # Region counts: fixed-circle diagram; geometry is not area-proportional.
    positions = {
        "DESeq2 only": (0.20, 0.61),
        "Adjusted Regular AREA only": (0.80, 0.61),
        "Weighted AREA only": (0.50, 0.18),
        "DESeq2 + Adjusted Regular AREA only": (0.50, 0.69),
        "DESeq2 + Weighted AREA only": (0.36, 0.42),
        "Adjusted Regular AREA + Weighted AREA only": (0.64, 0.42),
        "All three": (0.50, 0.51),
    }

    for key, (x, y) in positions.items():
        ax.text(
            x,
            y,
            f"{counts[key]:,}",
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
        )

    ax.text(
        0.17,
        0.88,
        f"DESeq2\nAD4 vs NCI1\n(n={deseq_n:,})",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.83,
        0.88,
        f"Adjusted Regular AREA\nAD4 vs NCI1\n(n={regular_n:,})",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.50,
        0.03,
        f"Weighted AREA\nNCI → MCI → AD\n(n={weighted_n_samples:,})",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.50,
        1.02,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        0.50,
        0.95,
        (
            f"Common comparison universe: 32,994 genes | "
            f"BH FDR < {cutoff:g}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
    )

    ax.text(
        0.50,
        -0.09,
        (
            "All three analyses control age_death, sex, RIN, PMI, and "
            "sequencing batch. DESeq2 and adjusted Regular AREA use the "
            "same 418 AD4-vs-NCI1 participants; Weighted AREA uses the "
            "broader graded NCI→MCI→AD cohort (n=619). "
            "Twelve persistent DESeq2 non-converged genes were excluded "
            "from all methods before BH recomputation. Circle areas are "
            "not proportional to region counts."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        wrap=True,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    pdf_path = output.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return str(pdf_path)


def main():
    args = parse_args()

    deseq = pd.read_csv(args.deseq2_file)
    regular = pd.read_csv(args.regular_area_file)
    weighted = pd.read_csv(args.weighted_area_file)

    deseq_manifest = pd.read_csv(args.deseq2_manifest)
    regular_manifest = pd.read_csv(args.regular_manifest)

    excluded = pd.read_csv(args.exclude_gene_file)

    require_columns(
        deseq,
        [
            "ensembl_id_version",
            "pvalue",
            "log2FoldChange",
        ],
        "DESeq2",
    )

    require_columns(
        regular,
        [
            "gene_id",
            "adjusted_pvalue",
            "adjusted_Regular_AREA_Z",
        ],
        "Adjusted Regular AREA",
    )

    require_columns(
        weighted,
        [
            "gene_id",
            "phenotype",
            "n_samples",
            "adjusted_pvalue",
            "adjusted_AREA_Z",
        ],
        "Weighted AREA",
    )

    require_columns(
        excluded,
        ["ensembl_id_version"],
        "Excluded-gene file",
    )

    deseq_ids, deseq_n = manifest_ids(
        deseq_manifest,
        args.expected_deseq2_n,
        "DESeq2 manifest",
    )

    regular_ids, regular_n = manifest_ids(
        regular_manifest,
        args.expected_regular_n,
        "Adjusted Regular AREA manifest",
    )

    if deseq_ids != regular_ids:
        raise ValueError(
            "DESeq2 and adjusted Regular AREA do not use the exact "
            "same endpoint participants."
        )

    w_n = weighted_n(
        weighted,
        args.expected_weighted_n,
    )

    print("=" * 80)
    print("FINAL LOCKED-COVARIATE THREE-METHOD VENN")
    print("=" * 80)

    print("\nSAMPLE SCOPE — VERIFIED")
    print(f"  DESeq2 AD4 vs NCI1:            n={deseq_n:,}")
    print(f"  Adjusted Regular AREA:         n={regular_n:,}")
    print("  Endpoint sample IDs:           EXACT MATCH")
    print(f"  Weighted AREA Cognitive_stage: n={w_n:,}")

    excluded_ids = set(
        excluded["ensembl_id_version"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if len(excluded_ids) != args.expected_excluded_genes:
        raise ValueError(
            f"Excluded-gene count is {len(excluded_ids)}, "
            f"expected {args.expected_excluded_genes}."
        )

    # Verify the supplied IDs actually exist in all three universes.
    raw_deseq_ids = set(
        deseq["ensembl_id_version"]
        .dropna()
        .astype(str)
    )

    raw_regular_ids = set(
        regular["gene_id"]
        .dropna()
        .astype(str)
    )

    raw_weighted_ids = set(
        weighted["gene_id"]
        .dropna()
        .astype(str)
    )

    for label, universe in [
        ("DESeq2", raw_deseq_ids),
        ("Adjusted Regular AREA", raw_regular_ids),
        ("Weighted AREA", raw_weighted_ids),
    ]:
        missing = excluded_ids - universe
        if missing:
            raise ValueError(
                f"{label} is missing {len(missing)} excluded genes."
            )

    print("\nPERSISTENT DESEQ2 NON-CONVERGENCE")
    print(f"  Genes removed from comparison universe: {len(excluded_ids):,}")

    deseq = deseq.loc[
        ~deseq["ensembl_id_version"].astype(str).isin(excluded_ids)
    ].copy()

    regular = regular.loc[
        ~regular["gene_id"].astype(str).isin(excluded_ids)
    ].copy()

    weighted = weighted.loc[
        ~weighted["gene_id"].astype(str).isin(excluded_ids)
    ].copy()

    # Enforce uniqueness after exclusion.
    if deseq["ensembl_id_version"].duplicated().any():
        raise ValueError("DESeq2 has duplicate gene IDs.")

    if regular["gene_id"].duplicated().any():
        raise ValueError("Adjusted Regular AREA has duplicate gene IDs.")

    if weighted["gene_id"].duplicated().any():
        raise ValueError("Weighted AREA has duplicate gene IDs.")

    common_deseq = set(
        deseq["ensembl_id_version"].astype(str)
    )

    common_regular = set(
        regular["gene_id"].astype(str)
    )

    common_weighted = set(
        weighted["gene_id"].astype(str)
    )

    if not (
        common_deseq
        == common_regular
        == common_weighted
    ):
        raise ValueError(
            "Gene universes differ after removing the 12 "
            "non-converged DESeq2 genes."
        )

    if len(common_deseq) != args.expected_common_genes:
        raise ValueError(
            f"Final common universe is {len(common_deseq):,}, "
            f"expected {args.expected_common_genes:,}."
        )

    print(
        f"  Final common comparison universe: "
        f"{len(common_deseq):,} genes"
    )

    # Recompute BH on the exact same 32,994-gene comparison universe.
    deseq["pvalue"] = pd.to_numeric(
        deseq["pvalue"],
        errors="coerce",
    )

    regular["adjusted_pvalue"] = pd.to_numeric(
        regular["adjusted_pvalue"],
        errors="coerce",
    )

    weighted["adjusted_pvalue"] = pd.to_numeric(
        weighted["adjusted_pvalue"],
        errors="coerce",
    )

    deseq["comparison_padj_BH"] = bh_adjust_with_nan(
        deseq["pvalue"]
    )

    regular["comparison_padj_BH"] = bh_adjust_with_nan(
        regular["adjusted_pvalue"]
    )

    weighted["comparison_padj_BH"] = bh_adjust_with_nan(
        weighted["adjusted_pvalue"]
    )

    cutoff = args.cutoff

    A = set(
        deseq.loc[
            deseq["comparison_padj_BH"].notna()
            & (deseq["comparison_padj_BH"] < cutoff),
            "ensembl_id_version",
        ].astype(str)
    )

    B = set(
        regular.loc[
            regular["comparison_padj_BH"].notna()
            & (regular["comparison_padj_BH"] < cutoff),
            "gene_id",
        ].astype(str)
    )

    C = set(
        weighted.loc[
            weighted["comparison_padj_BH"].notna()
            & (weighted["comparison_padj_BH"] < cutoff),
            "gene_id",
        ].astype(str)
    )

    print("\nSIGNIFICANT GENES — COMMON UNIVERSE")
    print(f"  DESeq2:                {len(A):,}")
    print(f"  Adjusted Regular AREA: {len(B):,}")
    print(f"  Weighted AREA:         {len(C):,}")

    counts = venn_counts(A, B, C)

    order = [
        "DESeq2 only",
        "Adjusted Regular AREA only",
        "Weighted AREA only",
        "DESeq2 + Adjusted Regular AREA only",
        "DESeq2 + Weighted AREA only",
        "Adjusted Regular AREA + Weighted AREA only",
        "All three",
    ]

    print("\nEXACT VENN REGIONS")
    for key in order:
        print(f"  {key:<44} {counts[key]:,}")

    print(
        f"  {'Total unique':<44} "
        f"{counts['Total unique']:,}"
    )

    key_region = counts[
        "Adjusted Regular AREA + Weighted AREA only"
    ]

    print("\nKEY PROJECT REGION")
    print(
        "  Adjusted Regular AREA ∩ Weighted AREA "
        f"excluding DESeq2 = {key_region:,} genes"
    )

    # Gene-level comparison table.
    deseq_cmp = deseq[
        [
            "ensembl_id_version",
            "pvalue",
            "comparison_padj_BH",
            "log2FoldChange",
        ]
        + (
            ["gene_symbol"]
            if "gene_symbol" in deseq.columns
            else []
        )
    ].copy()

    deseq_cmp = deseq_cmp.rename(
        columns={
            "comparison_padj_BH":
                "DESeq2_comparison_padj_BH",
            "pvalue":
                "DESeq2_pvalue",
        }
    )

    regular_cmp = regular[
        [
            "gene_id",
            "adjusted_pvalue",
            "comparison_padj_BH",
            "adjusted_Regular_AREA_Z",
        ]
    ].copy()

    regular_cmp = regular_cmp.rename(
        columns={
            "adjusted_pvalue":
                "Regular_adjusted_pvalue",
            "comparison_padj_BH":
                "Regular_comparison_padj_BH",
        }
    )

    weighted_cmp = weighted[
        [
            "gene_id",
            "adjusted_pvalue",
            "comparison_padj_BH",
            "adjusted_AREA_Z",
        ]
    ].copy()

    weighted_cmp = weighted_cmp.rename(
        columns={
            "adjusted_pvalue":
                "Weighted_adjusted_pvalue",
            "comparison_padj_BH":
                "Weighted_comparison_padj_BH",
        }
    )

    comparison = (
        regular_cmp
        .merge(
            weighted_cmp,
            on="gene_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            deseq_cmp,
            left_on="gene_id",
            right_on="ensembl_id_version",
            how="inner",
            validate="one_to_one",
        )
    )

    comparison["DESeq2_significant"] = (
        comparison["DESeq2_comparison_padj_BH"].notna()
        & (
            comparison["DESeq2_comparison_padj_BH"]
            < cutoff
        )
    )

    comparison["Regular_AREA_significant"] = (
        comparison["Regular_comparison_padj_BH"].notna()
        & (
            comparison["Regular_comparison_padj_BH"]
            < cutoff
        )
    )

    comparison["Weighted_AREA_significant"] = (
        comparison["Weighted_comparison_padj_BH"].notna()
        & (
            comparison["Weighted_comparison_padj_BH"]
            < cutoff
        )
    )

    comparison["n_methods_significant"] = (
        comparison[
            [
                "DESeq2_significant",
                "Regular_AREA_significant",
                "Weighted_AREA_significant",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )

    comparison["overlap_region"] = [
        region_name(de, reg, wgt)
        for de, reg, wgt in zip(
            comparison["DESeq2_significant"],
            comparison["Regular_AREA_significant"],
            comparison["Weighted_AREA_significant"],
        )
    ]

    root, _ = os.path.splitext(args.output)

    membership_file = (
        root + "_gene_membership.csv"
    )

    counts_file = (
        root + "_overlap_counts.csv"
    )

    metadata_file = (
        root + "_run_metadata.csv"
    )

    excluded_file = (
        root + "_excluded_nonconverged_genes.csv"
    )

    comparison.to_csv(
        membership_file,
        index=False,
    )

    pd.DataFrame(
        {
            "region": order,
            "n_genes": [
                counts[x]
                for x in order
            ],
        }
    ).to_csv(
        counts_file,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "method": "DESeq2",
                "phenotype_scope": "AD4 vs NCI1",
                "n_samples": deseq_n,
                "covariates": (
                    "age_death + sex + rin_numeric + "
                    "pmi_numeric + sequencing_batch"
                ),
                "comparison_gene_universe":
                    len(common_deseq),
                "n_significant_genes": len(A),
            },
            {
                "method": "Adjusted Regular AREA",
                "phenotype_scope": "AD4 vs NCI1",
                "n_samples": regular_n,
                "covariates": (
                    "age_death + sex + rin_numeric + "
                    "pmi_numeric + sequencing_batch"
                ),
                "comparison_gene_universe":
                    len(common_deseq),
                "n_significant_genes": len(B),
            },
            {
                "method": "Weighted AREA",
                "phenotype_scope":
                    "Cognitive_stage: NCI -> MCI -> AD",
                "n_samples": w_n,
                "covariates": (
                    "age_death + sex + rin_numeric + "
                    "pmi_numeric + sequencing_batch"
                ),
                "comparison_gene_universe":
                    len(common_deseq),
                "n_significant_genes": len(C),
            },
        ]
    ).to_csv(
        metadata_file,
        index=False,
    )

    excluded.to_csv(
        excluded_file,
        index=False,
    )

    pdf_path = draw_venn(
        A=A,
        B=B,
        C=C,
        counts=counts,
        title=args.title,
        output=args.output,
        deseq_n=deseq_n,
        regular_n=regular_n,
        weighted_n_samples=w_n,
        cutoff=cutoff,
    )

    print("\nOUTPUTS")
    print(f"  PNG:        {args.output}")
    print(f"  PDF:        {pdf_path}")
    print(f"  Membership: {membership_file}")
    print(f"  Counts:     {counts_file}")
    print(f"  Metadata:   {metadata_file}")
    print(f"  Excluded:   {excluded_file}")


if __name__ == "__main__":
    main()

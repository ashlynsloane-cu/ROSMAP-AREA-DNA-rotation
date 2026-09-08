#!/usr/bin/env python3
"""
plot_final_deseq_regular_weighted_venn.py
========================================

Final 3-set Venn for:

1. DESeq2: AD4 vs NCI1
2. Regular AREA: AD4 vs NCI1
3. Weighted AREA: Cognitive_stage (NCI -> MCI -> AD)

IMPORTANT:
The first two methods use the binary AD4-vs-NCI1 endpoint comparison.
Weighted AREA uses the broader graded cognitive-stage cohort.

To prevent sample-size mistakes, this script:
- reads DESeq2 and Regular AREA sample counts from their own manifests;
- reads the Weighted AREA sample count directly from the production result file;
- verifies that Weighted AREA has exactly one n_samples value;
- verifies DESeq2 and Regular AREA use the exact same 420 participant IDs;
- fails if any expected sample count is wrong;
- prints the sample-scope difference prominently;
- writes the sample sizes into the figure itself.

Significance definitions:
    DESeq2:        padj < cutoff
    Regular AREA:  Regular_FDR < cutoff
    Weighted AREA: adjusted_padj_BH < cutoff

Gene matching:
    Regular/Weighted AREA gene_id <-> DESeq2 ensembl_id_version

Outputs:
    PNG
    PDF
    gene-level membership CSV
    exact Venn-region count CSV
    run metadata CSV
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle
import pandas as pd


PALETTE = {
    "DESeq2": "#1f77b4",
    "Regular": "#ff7f0e",
    "Weighted": "#2ca02c",
    "Shared_AREA": "#d62728",
}

SHADOW_EFFECT = [
    path_effects.withStroke(linewidth=2.5, foreground="white")
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--regular-area-file",
        required=True,
        help=(
            "Corrected matched AD4-vs-NCI1 AREA CSV containing "
            "gene_id and Regular_FDR."
        ),
    )
    p.add_argument(
        "--deseq2-file",
        required=True,
        help=(
            "Corrected AD4-vs-NCI1 DESeq2 all-results CSV containing "
            "ensembl_id_version, gene_symbol, and padj."
        ),
    )
    p.add_argument(
        "--weighted-area-file",
        required=True,
        help=(
            "Production Cognitive_stage Weighted AREA results.csv containing "
            "gene_id, n_samples, phenotype, and adjusted_padj_BH."
        ),
    )
    p.add_argument(
        "--regular-manifest",
        required=True,
        help="Sample manifest for the matched AD4-vs-NCI1 Regular AREA run.",
    )
    p.add_argument(
        "--deseq2-manifest",
        required=True,
        help="Sample manifest for the AD4-vs-NCI1 DESeq2 run.",
    )
    p.add_argument(
        "--expected-regular-n",
        type=int,
        default=420,
        help="Expected Regular AREA sample size. Default: 420.",
    )
    p.add_argument(
        "--expected-deseq2-n",
        type=int,
        default=420,
        help="Expected DESeq2 sample size. Default: 420.",
    )
    p.add_argument(
        "--expected-weighted-n",
        type=int,
        default=619,
        help="Expected production Cognitive_stage sample size. Default: 619.",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    p.add_argument(
        "--title",
        default="Mean- and Rank-Based Gene Signatures Across Cognitive AD",
    )

    return p.parse_args()


def require_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def get_manifest_sample_ids(manifest, expected_n, label):
    if len(manifest) == 0:
        raise ValueError(f"{label} sample manifest is empty.")

    if "sample_id" not in manifest.columns:
        raise ValueError(
            f"{label} manifest lacks required sample_id column."
        )

    ids = manifest["sample_id"].astype(str).str.strip()

    if ids.duplicated().any():
        dup = ids[ids.duplicated()].unique().tolist()[:10]
        raise ValueError(
            f"{label} manifest contains duplicate sample IDs: {dup}"
        )

    n = int(ids.nunique())

    if expected_n is not None and n != expected_n:
        raise ValueError(
            f"{label} sample count mismatch: manifest says n={n}, "
            f"expected n={expected_n}. Do not draw the Venn until resolved."
        )

    return set(ids), n

def get_weighted_n(weighted, expected):
    require_columns(
        weighted,
        ["n_samples", "adjusted_padj_BH", "gene_id"],
        "Weighted AREA file",
    )

    n_values = (
        pd.to_numeric(weighted["n_samples"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )

    if len(n_values) != 1:
        raise ValueError(
            "Weighted AREA file must contain exactly one unique n_samples "
            f"value; found {n_values.tolist()}."
        )

    n = int(n_values[0])

    if expected is not None and n != expected:
        raise ValueError(
            "Weighted Cognitive_stage sample count mismatch: "
            f"file says n={n}, expected n={expected}. "
            "Do not draw the Venn until this is resolved."
        )

    # If phenotype is present, verify that this really is Cognitive_stage.
    if "phenotype" in weighted.columns:
        phenos = weighted["phenotype"].dropna().astype(str).unique().tolist()
        if phenos != ["Cognitive_stage"]:
            raise ValueError(
                "Weighted AREA file is not a pure Cognitive_stage result: "
                f"phenotype values = {phenos}"
            )

    return n


def classify_region(row):
    de = bool(row["DESeq2_significant"])
    reg = bool(row["Regular_AREA_significant"])
    wgt = bool(row["Weighted_AREA_significant"])

    if de and reg and wgt:
        return "All three"
    if de and reg:
        return "DESeq2 + Regular AREA only"
    if de and wgt:
        return "DESeq2 + Weighted AREA only"
    if reg and wgt:
        return "Regular AREA + Weighted AREA only"
    if de:
        return "DESeq2 only"
    if reg:
        return "Regular AREA only"
    if wgt:
        return "Weighted AREA only"
    return "Not significant"


def draw_fixed_venn(
    deseq_sig,
    regular_sig,
    weighted_sig,
    cutoff,
    title,
    out_path,
    deseq2_n,
    regular_n,
    weighted_n,
):
    A = deseq_sig
    B = regular_sig
    C = weighted_sig

    v_100 = len(A - (B | C))
    v_010 = len(B - (A | C))
    v_001 = len(C - (A | B))
    v_110 = len((A & B) - C)
    v_101 = len((A & C) - B)
    v_011 = len((B & C) - A)
    v_111 = len(A & B & C)

    total_unique = len(A | B | C)

    fig, ax = plt.subplots(figsize=(7.0, 6.3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-5.5, 5.5)
    ax.axis("off")

    y_shift = -0.50
    c1_center = (-1.1, -0.6 + y_shift)
    c2_center = (1.1, -0.6 + y_shift)
    c3_center = (0.0, 1.1 + y_shift)
    radius = 2.45

    circles = [
        Circle(
            c1_center,
            radius,
            facecolor=PALETTE["DESeq2"],
            alpha=0.18,
            edgecolor=PALETTE["DESeq2"],
            linewidth=2,
        ),
        Circle(
            c2_center,
            radius,
            facecolor=PALETTE["Regular"],
            alpha=0.18,
            edgecolor=PALETTE["Regular"],
            linewidth=2,
        ),
        Circle(
            c3_center,
            radius,
            facecolor=PALETTE["Weighted"],
            alpha=0.18,
            edgecolor=PALETTE["Weighted"],
            linewidth=2,
        ),
    ]

    for circle in circles:
        ax.add_patch(circle)

    def plot_count(x, y, value, color="#111111", size=11):
        ax.text(
            x,
            y,
            f"{value:,}",
            fontsize=size,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            path_effects=SHADOW_EFFECT,
        )

    plot_count(-2.2, -1.2 + y_shift, v_100, PALETTE["DESeq2"], 11.5)
    plot_count(2.2, -1.2 + y_shift, v_010, PALETTE["Regular"], 11.5)
    plot_count(0.0, 2.3 + y_shift, v_001, PALETTE["Weighted"], 11.5)
    plot_count(0.0, -1.6 + y_shift, v_110, "#333333", 10)
    plot_count(-1.1, 0.5 + y_shift, v_101, "#333333", 10)
    plot_count(1.1, 0.5 + y_shift, v_011, PALETTE["Shared_AREA"], 10)
    plot_count(0.0, -0.3 + y_shift, v_111, "#111111", 13)

    ax.text(
        -3.45,
        -2.15 + y_shift,
        f"DESeq2\nAD4 vs NCI1\n(n={deseq2_n:,})",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["DESeq2"],
        ha="center",
    )

    ax.text(
        3.45,
        -2.15 + y_shift,
        f"Regular AREA\nAD4 vs NCI1\n(n={regular_n:,})",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["Regular"],
        ha="center",
    )

    ax.text(
        0.0,
        3.60 + y_shift,
        f"Weighted AREA\nNCI → MCI → AD\n(n={weighted_n:,})",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["Weighted"],
        ha="center",
    )

    ax.text(
        0.0,
        4.85,
        title,
        fontsize=11.5,
        fontweight="bold",
        color="#111111",
        ha="center",
        va="center",
    )

    ax.text(
        0.0,
        4.45,
        (
            f"Total unique significant genes: {total_unique:,}   |   "
            f"FDR < {cutoff:g}"
        ),
        fontsize=8.5,
        color="#555555",
        ha="center",
        va="center",
    )

    # This scope note is intentionally inside the figure so the 619-person
    # Weighted analysis cannot be mistaken for the 423-person endpoint contrast.
    ax.text(
        0.0,
        -5.0,
        (
            "Sample scope differs by design: DESeq2 and Regular AREA use the "
            f"AD4-vs-NCI1 endpoints (DESeq2 n={deseq2_n:,}; Regular AREA n={regular_n:,}); Weighted AREA uses the "
            f"graded NCI→MCI→AD cohort after covariate QC (n={weighted_n:,})."
        ),
        fontsize=7.4,
        color="#666666",
        ha="center",
        va="center",
        wrap=True,
    )

    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    root, _ = os.path.splitext(out_path)
    pdf_path = root + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "DESeq2 only": v_100,
        "Regular AREA only": v_010,
        "Weighted AREA only": v_001,
        "DESeq2 + Regular AREA only": v_110,
        "DESeq2 + Weighted AREA only": v_101,
        "Regular AREA + Weighted AREA only": v_011,
        "All three": v_111,
        "Total unique": total_unique,
        "pdf_path": pdf_path,
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("FINAL DESeq2 / REGULAR AREA / WEIGHTED AREA VENN")
    print("=" * 80)

    regular = pd.read_csv(args.regular_area_file)
    deseq = pd.read_csv(args.deseq2_file)
    weighted = pd.read_csv(args.weighted_area_file)
    regular_manifest = pd.read_csv(args.regular_manifest)
    deseq2_manifest = pd.read_csv(args.deseq2_manifest)

    require_columns(
        regular,
        ["gene_id", "Regular_FDR"],
        "Regular AREA file",
    )
    require_columns(
        deseq,
        ["ensembl_id_version", "gene_symbol", "padj"],
        "DESeq2 file",
    )

    regular_ids, regular_n = get_manifest_sample_ids(
        regular_manifest,
        args.expected_regular_n,
        "Regular AREA",
    )
    deseq2_ids, deseq2_n = get_manifest_sample_ids(
        deseq2_manifest,
        args.expected_deseq2_n,
        "DESeq2",
    )
    weighted_n = get_weighted_n(
        weighted,
        args.expected_weighted_n,
    )

    if regular_ids != deseq2_ids:
        raise ValueError(
            "DESeq2 and Regular AREA both have the expected sample counts, "
            "but they do NOT contain the exact same participants. "
            f"Regular-only samples: {len(regular_ids - deseq2_ids)}; "
            f"DESeq2-only samples: {len(deseq2_ids - regular_ids)}. "
            "Do not draw the endpoint-method Venn until this is resolved."
        )

    print("\nSAMPLE SCOPE — VERIFIED")
    print(f"  DESeq2 AD4 vs NCI1:       n={deseq2_n:,}")
    print(f"  Regular AREA AD4 vs NCI1: n={regular_n:,}")
    print("  Binary endpoint sample IDs are an EXACT MATCH.")
    print(f"  Weighted Cognitive_stage: n={weighted_n:,}")
    print(
        "  NOTE: Weighted AREA intentionally includes the intermediate MCI "
        "state and therefore has a larger sample set."
    )

    regular["Regular_FDR"] = pd.to_numeric(
        regular["Regular_FDR"],
        errors="coerce",
    )
    deseq["padj"] = pd.to_numeric(
        deseq["padj"],
        errors="coerce",
    )
    weighted["adjusted_padj_BH"] = pd.to_numeric(
        weighted["adjusted_padj_BH"],
        errors="coerce",
    )

    if regular["gene_id"].duplicated().any():
        raise ValueError("Regular AREA contains duplicate gene_id values.")
    if weighted["gene_id"].duplicated().any():
        raise ValueError("Weighted AREA contains duplicate gene_id values.")
    if deseq["ensembl_id_version"].duplicated().any():
        raise ValueError("DESeq2 contains duplicate ensembl_id_version values.")

    regular_ids = set(regular["gene_id"].dropna().astype(str))
    weighted_ids = set(weighted["gene_id"].dropna().astype(str))
    deseq_ids = set(deseq["ensembl_id_version"].dropna().astype(str))

    if not (regular_ids == weighted_ids == deseq_ids):
        raise ValueError(
            "The three gene universes do not match exactly.\n"
            f"Regular genes:  {len(regular_ids):,}\n"
            f"Weighted genes: {len(weighted_ids):,}\n"
            f"DESeq2 genes:   {len(deseq_ids):,}\n"
            "Do not draw the Venn from mismatched tested-gene universes."
        )

    cutoff = args.cutoff

    deseq_sig = set(
        deseq.loc[
            deseq["padj"].notna()
            & (deseq["padj"] < cutoff),
            "ensembl_id_version",
        ].astype(str)
    )

    regular_sig = set(
        regular.loc[
            regular["Regular_FDR"].notna()
            & (regular["Regular_FDR"] < cutoff),
            "gene_id",
        ].astype(str)
    )

    weighted_sig = set(
        weighted.loc[
            weighted["adjusted_padj_BH"].notna()
            & (weighted["adjusted_padj_BH"] < cutoff),
            "gene_id",
        ].astype(str)
    )

    print("\nSIGNIFICANT GENES")
    print(f"  DESeq2:        {len(deseq_sig):,}")
    print(f"  Regular AREA:  {len(regular_sig):,}")
    print(f"  Weighted AREA: {len(weighted_sig):,}")

    comparison = (
        regular[["gene_id", "Regular_FDR"]]
        .merge(
            weighted[
                [
                    "gene_id",
                    "n_samples",
                    "adjusted_AREA_Z",
                    "adjusted_pvalue",
                    "adjusted_padj_BH",
                ]
            ],
            on="gene_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            deseq[
                [
                    "ensembl_id_version",
                    "gene_symbol",
                    "padj",
                    "log2FoldChange",
                ]
            ],
            left_on="gene_id",
            right_on="ensembl_id_version",
            how="inner",
            validate="one_to_one",
        )
    )

    comparison["DESeq2_significant"] = (
        comparison["padj"].notna()
        & (comparison["padj"] < cutoff)
    )
    comparison["Regular_AREA_significant"] = (
        comparison["Regular_FDR"].notna()
        & (comparison["Regular_FDR"] < cutoff)
    )
    comparison["Weighted_AREA_significant"] = (
        comparison["adjusted_padj_BH"].notna()
        & (comparison["adjusted_padj_BH"] < cutoff)
    )

    comparison["n_methods_significant"] = comparison[
        [
            "DESeq2_significant",
            "Regular_AREA_significant",
            "Weighted_AREA_significant",
        ]
    ].astype(int).sum(axis=1)

    comparison["overlap_region"] = comparison.apply(
        classify_region,
        axis=1,
    )

    root, _ = os.path.splitext(args.output)
    membership_file = root + "_gene_membership.csv"
    counts_file = root + "_overlap_counts.csv"
    metadata_file = root + "_run_metadata.csv"

    comparison.to_csv(
        membership_file,
        index=False,
    )

    counts = draw_fixed_venn(
        deseq_sig=deseq_sig,
        regular_sig=regular_sig,
        weighted_sig=weighted_sig,
        cutoff=cutoff,
        title=args.title,
        out_path=args.output,
        deseq2_n=deseq2_n,
        regular_n=regular_n,
        weighted_n=weighted_n,
    )

    count_order = [
        "DESeq2 only",
        "Regular AREA only",
        "Weighted AREA only",
        "DESeq2 + Regular AREA only",
        "DESeq2 + Weighted AREA only",
        "Regular AREA + Weighted AREA only",
        "All three",
    ]

    pd.DataFrame(
        {
            "region": count_order,
            "n_genes": [
                counts[key]
                for key in count_order
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
                "n_samples": deseq2_n,
                "significance_column": "padj",
                "fdr_cutoff": cutoff,
                "n_significant_genes": len(deseq_sig),
            },
            {
                "method": "Regular AREA",
                "phenotype_scope": "AD4 vs NCI1",
                "n_samples": regular_n,
                "significance_column": "Regular_FDR",
                "fdr_cutoff": cutoff,
                "n_significant_genes": len(regular_sig),
            },
            {
                "method": "Weighted AREA",
                "phenotype_scope": "Cognitive_stage: NCI -> MCI -> AD",
                "n_samples": weighted_n,
                "significance_column": "adjusted_padj_BH",
                "fdr_cutoff": cutoff,
                "n_significant_genes": len(weighted_sig),
            },
        ]
    ).to_csv(
        metadata_file,
        index=False,
    )

    print("\nEXACT VENN REGIONS")
    for key in count_order:
        print(f"  {key:<38} {counts[key]:,}")

    print(f"  {'Total unique':<38} {counts['Total unique']:,}")

    key_region = counts["Regular AREA + Weighted AREA only"]
    print(
        "\nKEY PROJECT REGION:"
        "\n  Regular AREA ∩ Weighted AREA, excluding DESeq2 = "
        f"{key_region:,} genes"
    )

    print("\nOUTPUTS")
    print(f"  PNG:        {args.output}")
    print(f"  PDF:        {counts['pdf_path']}")
    print(f"  Membership: {membership_file}")
    print(f"  Counts:     {counts_file}")
    print(f"  Metadata:   {metadata_file}")


if __name__ == "__main__":
    main()

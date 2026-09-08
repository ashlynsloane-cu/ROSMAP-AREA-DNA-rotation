#!/usr/bin/env python3

"""
Compare corrected DESeq2, Regular AREA, and Weighted AREA results
for matched AD4 vs NCI1, and draw a fixed-geometry 3-set Venn diagram
matching the original publication-style figure.

Significance definitions:
    DESeq2:        padj < cutoff
    Regular AREA:  Regular_FDR < cutoff
    Weighted AREA: Weighted_FDR < cutoff

Gene matching:
    AREA gene_id <-> DESeq2 ensembl_id_version

Outputs:
    1. Publication-style PNG
    2. Matching PDF
    3. Gene-level membership CSV
    4. Exact overlap-count CSV
"""

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
    parser = argparse.ArgumentParser(
        description=(
            "Compare corrected DESeq2, Regular AREA, and Weighted AREA "
            "results using a fixed-geometry publication-style Venn diagram."
        )
    )
    parser.add_argument(
        "--area-file",
        required=True,
        help="Corrected AMS-AREA results CSV.",
    )
    parser.add_argument(
        "--deseq2-file",
        required=True,
        help="Corrected DESeq2 all-results CSV.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.05,
        help="FDR significance threshold. Default: 0.05",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Shared and Method-Specific Gene Signatures in Cognitive AD",
        help="Figure title.",
    )
    return parser.parse_args()


def validate_columns(df, required, name):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def draw_fixed_venn(deseq_sig, regular_sig, weighted_sig, cutoff, title, out_path):
    """Draw equal-sized, fixed-position circles like the original figure."""

    A = deseq_sig
    B = regular_sig
    C = weighted_sig

    # Exact mutually exclusive Venn regions.
    v_100 = len(A - (B | C))
    v_010 = len(B - (A | C))
    v_001 = len(C - (A | B))
    v_110 = len((A & B) - C)
    v_101 = len((A & C) - B)
    v_011 = len((B & C) - A)
    v_111 = len(A & B & C)

    total_unique = len(A | B | C)

    # Original publication dimensions: 170 mm x 150 mm.
    fig, ax = plt.subplots(figsize=(6.69, 5.9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.axis("off")

    # Original fixed geometry: three equal circles.
    y_shift = -0.45
    c1_center = (-1.1, -0.6 + y_shift)  # DESeq2, lower left
    c2_center = (1.1, -0.6 + y_shift)   # Regular AREA, lower right
    c3_center = (0.0, 1.1 + y_shift)    # Weighted AREA, top
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

    # Exact region placements from the original figure.
    plot_count(-2.2, -1.2 + y_shift, v_100, PALETTE["DESeq2"], 11.5)
    plot_count(2.2, -1.2 + y_shift, v_010, PALETTE["Regular"], 11.5)
    plot_count(0.0, 2.3 + y_shift, v_001, PALETTE["Weighted"], 11.5)
    plot_count(0.0, -1.6 + y_shift, v_110, "#333333", 10)
    plot_count(-1.1, 0.5 + y_shift, v_101, "#333333", 10)
    plot_count(1.1, 0.5 + y_shift, v_011, PALETTE["Shared_AREA"], 10)
    plot_count(0.0, -0.3 + y_shift, v_111, "#111111", 13)

    # Method labels, matching the original wording and positions.
    ax.text(
        -3.4,
        -2.1 + y_shift,
        "DESeq2\n(Group-Mean Shifts)",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["DESeq2"],
        ha="center",
    )
    ax.text(
        3.4,
        -2.1 + y_shift,
        "Regular AREA\n(State-Transition Switches)",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["Regular"],
        ha="center",
    )
    ax.text(
        0.0,
        3.8 + y_shift,
        "Weighted AREA\n(Continuous Dosage Model)",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["Weighted"],
        ha="center",
    )

    # Title/subtitle styled like the original figure.
    ax.text(
        0.0,
        4.7,
        title,
        fontsize=11.5,
        fontweight="bold",
        color="#111111",
        ha="center",
        va="center",
    )
    ax.text(
        0.0,
        4.3,
        (
            f"Total Unique Significant Genes: {total_unique:,}  |  "
            f"Significance Cutoff: FDR < {cutoff:g}"
        ),
        fontsize=8.5,
        color="#555555",
        ha="center",
        va="center",
    )

    plt.tight_layout()
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


def main():
    args = parse_args()

    print("=" * 72)
    print("CORRECTED DESeq2 / AREA OVERLAP")
    print("=" * 72)

    print(f"\nLoading AREA results:\n  {args.area_file}")
    area = pd.read_csv(args.area_file)

    print(f"\nLoading DESeq2 results:\n  {args.deseq2_file}")
    deseq = pd.read_csv(args.deseq2_file)

    validate_columns(
        area,
        ["gene_id", "Regular_FDR", "Weighted_FDR"],
        "AREA file",
    )
    validate_columns(
        deseq,
        ["ensembl_id_version", "gene_symbol", "padj", "log2FoldChange"],
        "DESeq2 file",
    )

    # Explicit numeric conversion protects against string-formatted columns.
    area["Regular_FDR"] = pd.to_numeric(area["Regular_FDR"], errors="coerce")
    area["Weighted_FDR"] = pd.to_numeric(area["Weighted_FDR"], errors="coerce")
    deseq["padj"] = pd.to_numeric(deseq["padj"], errors="coerce")

    area_ids = set(area["gene_id"].dropna().astype(str))
    deseq_ids = set(deseq["ensembl_id_version"].dropna().astype(str))
    overlap_ids = area_ids & deseq_ids

    print(f"\nAREA rows:   {len(area):,}")
    print(f"DESeq2 rows: {len(deseq):,}")
    print(
        "Gene ID overlap "
        "(AREA gene_id <-> DESeq2 ensembl_id_version): "
        f"{len(overlap_ids):,}"
    )

    # Fail loudly rather than silently drawing a Venn from mismatched IDs.
    if area_ids != deseq_ids:
        raise ValueError(
            "AREA and DESeq2 gene universes do not match exactly. "
            f"AREA-only IDs: {len(area_ids - deseq_ids):,}; "
            f"DESeq2-only IDs: {len(deseq_ids - area_ids):,}."
        )

    cutoff = args.cutoff

    deseq_sig = set(
        deseq.loc[
            deseq["padj"].notna() & (deseq["padj"] < cutoff),
            "ensembl_id_version",
        ].astype(str)
    )
    regular_sig = set(
        area.loc[
            area["Regular_FDR"].notna() & (area["Regular_FDR"] < cutoff),
            "gene_id",
        ].astype(str)
    )
    weighted_sig = set(
        area.loc[
            area["Weighted_FDR"].notna() & (area["Weighted_FDR"] < cutoff),
            "gene_id",
        ].astype(str)
    )

    print(f"\nSignificance cutoff: FDR < {cutoff:g}")
    print(f"DESeq2 significant:        {len(deseq_sig):,}")
    print(f"Regular AREA significant:  {len(regular_sig):,}")
    print(f"Weighted AREA significant: {len(weighted_sig):,}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    root, _ = os.path.splitext(args.output)
    membership_file = root + "_gene_membership.csv"
    counts_file = root + "_overlap_counts.csv"

    # Gene-level audit table.
    comparison = area[
        ["gene_id", "Regular_FDR", "Weighted_FDR"]
    ].merge(
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

    comparison["DESeq2_significant"] = (
        comparison["padj"].notna() & (comparison["padj"] < cutoff)
    )
    comparison["Regular_AREA_significant"] = (
        comparison["Regular_FDR"].notna()
        & (comparison["Regular_FDR"] < cutoff)
    )
    comparison["Weighted_AREA_significant"] = (
        comparison["Weighted_FDR"].notna()
        & (comparison["Weighted_FDR"] < cutoff)
    )
    comparison["n_methods_significant"] = comparison[
        [
            "DESeq2_significant",
            "Regular_AREA_significant",
            "Weighted_AREA_significant",
        ]
    ].astype(int).sum(axis=1)
    comparison["overlap_region"] = comparison.apply(classify_region, axis=1)
    comparison.to_csv(membership_file, index=False)

    counts = draw_fixed_venn(
        deseq_sig=deseq_sig,
        regular_sig=regular_sig,
        weighted_sig=weighted_sig,
        cutoff=cutoff,
        title=args.title,
        out_path=args.output,
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
            "n_genes": [counts[key] for key in count_order],
        }
    ).to_csv(counts_file, index=False)

    print("\nExact Venn regions:")
    for key in count_order:
        print(f"  {key:<36} {counts[key]:,}")
    print(f"  {'Total unique':<36} {counts['Total unique']:,}")

    print("\nOutputs:")
    print(f"  PNG:             {args.output}")
    print(f"  PDF:             {counts['pdf_path']}")
    print(f"  Gene membership: {membership_file}")
    print(f"  Overlap counts:  {counts_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()

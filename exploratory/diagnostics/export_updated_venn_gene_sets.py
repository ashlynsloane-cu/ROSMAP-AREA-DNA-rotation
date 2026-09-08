#!/usr/bin/env python3
"""
Export the seven mutually exclusive gene regions from the corrected
DESeq2 / Regular AREA / Weighted AREA Venn membership table.

Expected input columns:
    gene_symbol
    DESeq2_significant
    Regular_AREA_significant
    Weighted_AREA_significant

Outputs:
    <outdir>/DESeq2_only_genes.txt
    <outdir>/Regular_AREA_only_genes.txt
    <outdir>/Weighted_AREA_only_genes.txt
    <outdir>/DESeq2_Regular_only_genes.txt
    <outdir>/DESeq2_Weighted_only_genes.txt
    <outdir>/Regular_Weighted_only_genes.txt
    <outdir>/All_three_genes.txt
    <outdir>/venn_region_summary.csv
    <outdir>/venn_region_membership.csv
"""

import argparse
import os
import pandas as pd


REGIONS = [
    "DESeq2_only",
    "Regular_AREA_only",
    "Weighted_AREA_only",
    "DESeq2_Regular_only",
    "DESeq2_Weighted_only",
    "Regular_Weighted_only",
    "All_three",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export seven mutually exclusive gene sets from corrected Venn membership."
    )
    parser.add_argument(
        "--membership",
        required=True,
        help="Path to corrected *_gene_membership.csv.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for exported gene-set files.",
    )
    return parser.parse_args()


def as_bool(series):
    """Robustly convert bool / 0-1 / TRUE-FALSE strings to booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.fillna(0).astype(int).astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "nan": False,
                "none": False,
                "": False,
            }
        )
    )

    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"Could not interpret significance values as boolean: {bad}")

    return normalized.astype(bool)


def clean_gene_symbols(series):
    genes = series.astype("string").str.strip()
    invalid = genes.isna() | genes.eq("") | genes.str.lower().isin(["nan", "none"])
    return genes.mask(invalid)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.membership)

    required = [
        "gene_symbol",
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Membership file is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["gene_symbol"] = clean_gene_symbols(df["gene_symbol"])

    de = as_bool(df["DESeq2_significant"])
    reg = as_bool(df["Regular_AREA_significant"])
    wgt = as_bool(df["Weighted_AREA_significant"])

    masks = {
        "DESeq2_only": de & ~reg & ~wgt,
        "Regular_AREA_only": ~de & reg & ~wgt,
        "Weighted_AREA_only": ~de & ~reg & wgt,
        "DESeq2_Regular_only": de & reg & ~wgt,
        "DESeq2_Weighted_only": de & ~reg & wgt,
        "Regular_Weighted_only": ~de & reg & wgt,
        "All_three": de & reg & wgt,
    }

    region_membership = []
    summary_rows = []

    print("=" * 72)
    print("CORRECTED VENN GENE-SET EXPORT")
    print("=" * 72)

    for region in REGIONS:
        mask = masks[region]
        sub = df.loc[mask, ["gene_symbol"]].copy()
        sub = sub.dropna(subset=["gene_symbol"]).drop_duplicates()
        sub = sub.sort_values("gene_symbol")
        genes = sub["gene_symbol"]

        out_path = os.path.join(args.outdir, f"{region}_genes.txt")
        genes.to_csv(out_path, index=False, header=False)

        summary_rows.append(
            {
                "region": region,
                "n_rows_in_region": int(mask.sum()),
                "n_unique_gene_symbols": int(genes.nunique()),
                "gene_list_file": out_path,
            }
        )

        tmp = sub.copy()
        tmp["region"] = region
        region_membership.append(tmp)

        print(
            f"{region:26s} "
            f"{int(mask.sum()):6,d} Venn rows | "
            f"{int(genes.nunique()):6,d} unique symbols"
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "venn_region_summary.csv")
    summary.to_csv(summary_path, index=False)

    membership_out = pd.concat(region_membership, ignore_index=True)
    membership_path = os.path.join(args.outdir, "venn_region_membership.csv")
    membership_out.to_csv(membership_path, index=False)

    classified = de | reg | wgt
    print("-" * 72)
    print(f"Genes significant in >=1 method: {int(classified.sum()):,}")
    print(f"Genes significant in no methods: {int((~classified).sum()):,}")
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {membership_path}")
    print(f"Wrote seven gene-list TXT files to: {args.outdir}")


if __name__ == "__main__":
    main()

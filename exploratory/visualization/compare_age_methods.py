#!/usr/bin/env python3
"""
Compare preliminary ROSMAP age results from DESeq2, adjusted Regular AREA,
and adjusted Weighted AREA.

Expected design
---------------
DESeq2:        Oldest quartile (Q4) vs youngest quartile (Q1), covariate adjusted.
Regular AREA:  Same binary Older vs Younger comparison on the exact same cohort.
Weighted AREA: Continuous age_death within that same extreme-age cohort.

Significance is FDR < cutoff for all three methods. The script writes:
  - publication-ready 3-set Venn (PNG + PDF)
  - gene-level method membership table
  - exact Venn-region counts
  - method summary and pairwise overlap statistics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from matplotlib_venn import venn3
except ImportError as exc:
    raise SystemExit(
        "matplotlib-venn is required. Install it with: python3 -m pip install matplotlib-venn"
    ) from exc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--deseq2",
        default="results/age_prelim/deseq2/age_Q4_vs_Q1_DESeq2_all_results.csv",
    )
    p.add_argument(
        "--regular",
        default="results/age_prelim/area/regular_area_age_results.csv",
    )
    p.add_argument(
        "--weighted",
        default="results/age_prelim/area/weighted_area_age_results.csv",
    )
    p.add_argument(
        "--deseq2-manifest",
        default="results/age_prelim/deseq2/age_Q4_vs_Q1_sample_manifest.csv",
    )
    p.add_argument(
        "--area-manifest",
        default="results/age_prelim/area/matched_sample_manifest.csv",
    )
    p.add_argument("--cutoff", type=float, default=0.05)
    p.add_argument(
        "--outdir",
        default="results/age_prelim/comparison",
    )
    p.add_argument(
        "--title",
        default="Age-associated transcriptomic signatures",
    )
    return p.parse_args()


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns {missing}. Available: {df.columns.tolist()}"
        )


def infer_deseq_gene_col(df):
    for c in [
        "ensembl_id_version",
        "gene_id",
        "ensembl_id",
        "Geneid",
        "gene",
    ]:
        if c in df.columns:
            return c
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        if s.str.startswith("ENSG").mean() > 0.5:
            return c
    raise ValueError("Could not identify DESeq2 gene-ID column.")


def clean_ids(s):
    return s.astype(str).str.strip()


def region_name(de, reg, wgt):
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


def pairwise_stats(a_name, a, b_name, b):
    inter = len(a & b)
    union = len(a | b)
    return {
        "method_A": a_name,
        "method_B": b_name,
        "n_sig_A": len(a),
        "n_sig_B": len(b),
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else np.nan,
        "fraction_A_recovered_by_B": inter / len(a) if a else np.nan,
        "fraction_B_recovered_by_A": inter / len(b) if b else np.nan,
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    de = pd.read_csv(args.deseq2)
    reg = pd.read_csv(args.regular)
    wgt = pd.read_csv(args.weighted)
    de_manifest = pd.read_csv(args.deseq2_manifest)
    area_manifest = pd.read_csv(args.area_manifest)

    require_columns(de, ["padj", "log2FoldChange"], "DESeq2")
    require_columns(reg, ["gene_id", "adjusted_padj_BH", "adjusted_AREA_Z"], "Regular AREA")
    require_columns(wgt, ["gene_id", "adjusted_padj_BH", "adjusted_AREA_Z"], "Weighted AREA")
    require_columns(de_manifest, ["sample_id"], "DESeq2 manifest")
    require_columns(area_manifest, ["sample_id"], "AREA manifest")

    de_ids = set(clean_ids(de_manifest["sample_id"]))
    area_ids = set(clean_ids(area_manifest["sample_id"]))
    if de_ids != area_ids:
        only_de = sorted(de_ids - area_ids)[:10]
        only_area = sorted(area_ids - de_ids)[:10]
        raise ValueError(
            "DESeq2 and AREA sample cohorts are not identical. "
            f"DESeq2 n={len(de_ids)}, AREA n={len(area_ids)}; "
            f"DESeq2-only examples={only_de}; AREA-only examples={only_area}."
        )

    gene_col = infer_deseq_gene_col(de)
    de = de.copy()
    de["gene_id"] = clean_ids(de[gene_col])
    reg = reg.copy()
    wgt = wgt.copy()
    reg["gene_id"] = clean_ids(reg["gene_id"])
    wgt["gene_id"] = clean_ids(wgt["gene_id"])

    for df, label in [(de, "DESeq2"), (reg, "Regular AREA"), (wgt, "Weighted AREA")]:
        if df["gene_id"].duplicated().any():
            dup = df.loc[df["gene_id"].duplicated(), "gene_id"].unique().tolist()[:10]
            raise ValueError(f"{label} contains duplicate gene IDs; examples: {dup}")

    # Restrict to the strict common gene universe before counting significance.
    common = set(de["gene_id"]) & set(reg["gene_id"]) & set(wgt["gene_id"])
    if not common:
        raise ValueError("No common genes across the three result files.")

    de = de[de["gene_id"].isin(common)].copy()
    reg = reg[reg["gene_id"].isin(common)].copy()
    wgt = wgt[wgt["gene_id"].isin(common)].copy()

    de_sig = set(
        de.loc[pd.to_numeric(de["padj"], errors="coerce") < args.cutoff, "gene_id"]
    )
    reg_sig = set(
        reg.loc[pd.to_numeric(reg["adjusted_padj_BH"], errors="coerce") < args.cutoff, "gene_id"]
    )
    wgt_sig = set(
        wgt.loc[pd.to_numeric(wgt["adjusted_padj_BH"], errors="coerce") < args.cutoff, "gene_id"]
    )

    comparison = (
        pd.DataFrame({"gene_id": sorted(common)})
        .merge(
            de[[
                "gene_id",
                *(["gene_symbol"] if "gene_symbol" in de.columns else []),
                "padj",
                "pvalue",
                "log2FoldChange",
                *(["stat"] if "stat" in de.columns else []),
            ]],
            on="gene_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            reg[["gene_id", "adjusted_padj_BH", "adjusted_pvalue", "adjusted_AREA_Z"]]
            .rename(columns={
                "adjusted_padj_BH": "Regular_AREA_FDR",
                "adjusted_pvalue": "Regular_AREA_pvalue",
                "adjusted_AREA_Z": "Regular_AREA_Z",
            }),
            on="gene_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            wgt[["gene_id", "adjusted_padj_BH", "adjusted_pvalue", "adjusted_AREA_Z"]]
            .rename(columns={
                "adjusted_padj_BH": "Weighted_AREA_FDR",
                "adjusted_pvalue": "Weighted_AREA_pvalue",
                "adjusted_AREA_Z": "Weighted_AREA_Z",
            }),
            on="gene_id",
            how="left",
            validate="one_to_one",
        )
    )

    comparison["DESeq2_significant"] = comparison["gene_id"].isin(de_sig)
    comparison["Regular_AREA_significant"] = comparison["gene_id"].isin(reg_sig)
    comparison["Weighted_AREA_significant"] = comparison["gene_id"].isin(wgt_sig)
    comparison["n_methods_significant"] = comparison[[
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
    ]].sum(axis=1).astype(int)
    comparison["overlap_region"] = [
        region_name(a, b, c)
        for a, b, c in zip(
            comparison["DESeq2_significant"],
            comparison["Regular_AREA_significant"],
            comparison["Weighted_AREA_significant"],
        )
    ]
    comparison.to_csv(outdir / "age_method_gene_membership.csv", index=False)

    order = [
        "DESeq2 only",
        "Regular AREA only",
        "Weighted AREA only",
        "DESeq2 + Regular AREA only",
        "DESeq2 + Weighted AREA only",
        "Regular AREA + Weighted AREA only",
        "All three",
    ]
    region_counts = (
        comparison.loc[comparison["overlap_region"].ne("Not significant"), "overlap_region"]
        .value_counts()
        .reindex(order, fill_value=0)
    )
    pd.DataFrame({"region": order, "n_genes": [int(region_counts[x]) for x in order]}).to_csv(
        outdir / "age_method_overlap_counts.csv", index=False
    )

    method_summary = pd.DataFrame([
        {
            "method": "DESeq2",
            "phenotype_representation": "Oldest quartile (Q4) vs youngest quartile (Q1)",
            "n_samples": len(de_ids),
            "n_common_genes_tested": len(common),
            "n_fdr_significant": len(de_sig),
        },
        {
            "method": "Adjusted Regular AREA",
            "phenotype_representation": "Oldest quartile (Q4) vs youngest quartile (Q1)",
            "n_samples": len(area_ids),
            "n_common_genes_tested": len(common),
            "n_fdr_significant": len(reg_sig),
        },
        {
            "method": "Adjusted Weighted AREA",
            "phenotype_representation": "Continuous age_death within same extreme-age cohort",
            "n_samples": len(area_ids),
            "n_common_genes_tested": len(common),
            "n_fdr_significant": len(wgt_sig),
        },
    ])
    method_summary.to_csv(outdir / "age_method_summary.csv", index=False)

    pairwise = pd.DataFrame([
        pairwise_stats("DESeq2", de_sig, "Regular AREA", reg_sig),
        pairwise_stats("DESeq2", de_sig, "Weighted AREA", wgt_sig),
        pairwise_stats("Regular AREA", reg_sig, "Weighted AREA", wgt_sig),
    ])
    pairwise.to_csv(outdir / "age_method_pairwise_overlap.csv", index=False)

    # Venn: use exact sets; figsize sized to a single-column/small two-column figure.
    fig, ax = plt.subplots(figsize=(6.5, 5.25))
    v = venn3(
        [de_sig, reg_sig, wgt_sig],
        set_labels=("DESeq2", "Regular AREA", "Weighted AREA"),
        ax=ax,
    )
    if v is not None:
        for t in (v.set_labels or []):
            if t is not None:
                t.set_fontsize(10)
        for t in (v.subset_labels or []):
            if t is not None:
                t.set_fontsize(9)

    ax.set_title(args.title, fontsize=12.5, pad=12)
    fig.text(
        0.5,
        0.035,
        (
            f"Matched cohort n={len(area_ids)} | Youngest quartile (Q1) vs oldest quartile (Q4) | "
            f"FDR < {args.cutoff:g}\n"
            "Weighted AREA retains continuous age within the same participants"
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.14, top=0.88, left=0.06, right=0.94)
    png = outdir / "age_DESeq2_Regular_Weighted_venn.png"
    pdf = outdir / "age_DESeq2_Regular_Weighted_venn.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nAGE METHOD COMPARISON")
    print(f"  Matched samples: {len(area_ids):,}")
    print(f"  Common genes:    {len(common):,}")
    print(f"  DESeq2 FDR<{args.cutoff:g}:        {len(de_sig):,}")
    print(f"  Regular AREA FDR<{args.cutoff:g}:  {len(reg_sig):,}")
    print(f"  Weighted AREA FDR<{args.cutoff:g}: {len(wgt_sig):,}")
    print("\nExact Venn regions:")
    for x in order:
        print(f"  {x:<36} {int(region_counts[x]):,}")
    print(f"\nOutputs: {outdir}")
    print(f"  PNG: {png}")
    print(f"  PDF: {pdf}")


if __name__ == "__main__":
    main()

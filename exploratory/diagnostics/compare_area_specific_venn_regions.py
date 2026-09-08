#!/usr/bin/env python3
"""
compare_area_specific_venn_regions.py
=====================================

Focused comparison of the Venn regions most relevant to "what does AREA add?"

Primary contrasts:
1) Weighted AREA only vs DESeq2 only
   -> asks what graded NCI -> MCI -> AD association adds beyond endpoint DESeq2.

2) Adjusted Regular AREA only vs DESeq2 only
   -> asks what binary rank association adds beyond endpoint mean-based DESeq2.

Inputs:
- all_regions_gene_geometry.csv from characterize_all_venn_regions.py
- final locked Venn membership table
- local Enrichr text libraries:
    MSigDB_Hallmark_2020
    GO_Biological_Process_2025
    Reactome_Pathways_2024

Outputs:
- descriptive comparison summaries
- representative genes for each region
- pathway ORA for DESeq2-only, Regular-only, and Weighted-only
- side-by-side enrichment comparison tables
- compact comparison figures

Statistics:
- ORA: one-sided Fisher exact test ("greater")
- BH correction separately within each region x library
- background: unique annotated symbols in the locked common Venn universe
- GO/Reactome pathway size filter: 10 <= K <= 500 after background intersection

This script intentionally does NOT test whether the descriptive geometry metrics
differ "significantly" between Venn regions, because the regions were themselves
defined by significance thresholds. The geometry comparison is descriptive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


REGIONS = {
    "deseq2_only": "DESeq2 only",
    "regular_only": "Adjusted Regular AREA only",
    "weighted_only": "Weighted AREA only",
}

DISPLAY = {
    "deseq2_only": "DESeq2 only",
    "regular_only": "Regular AREA only",
    "weighted_only": "Weighted AREA only",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--geometry",
        default=(
            "results/venn_region_characterization/"
            "all_regions_gene_geometry.csv"
        ),
    )

    p.add_argument(
        "--membership",
        default=(
            "results/visualizations/"
            "final_locked_covariate_DESeq2_Regular_Weighted_venn_gene_membership.csv"
        ),
    )

    p.add_argument(
        "--hallmark-file",
        default="data/gene_sets/MSigDB_Hallmark_2020.txt",
    )

    p.add_argument(
        "--go-file",
        default="data/gene_sets/GO_Biological_Process_2025.txt",
    )

    p.add_argument(
        "--reactome-file",
        default="data/gene_sets/Reactome_Pathways_2024.txt",
    )

    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--representatives-per-region", type=int, default=10)
    p.add_argument("--min-pathway-size", type=int, default=10)
    p.add_argument("--max-pathway-size", type=int, default=500)

    p.add_argument(
        "--outdir",
        default="results/area_specific_venn_comparison",
    )

    return p.parse_args()


def clean_symbols(series):
    s = series.dropna().astype(str).str.strip()
    bad = {"", "nan", "none", "na", "n/a", "<na>"}
    return set(s[~s.str.lower().isin(bad)])


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]

    q = ranked * len(p) / np.arange(1, len(p) + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(len(p), dtype=float)
    out[order] = q
    return out


def load_enrichr_library(path, label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} library not found: {path}"
        )

    pathways = {}

    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue

        parts = raw.rstrip("\n").split("\t")

        if len(parts) < 3:
            raise ValueError(
                f"{label}: malformed line {line_no}."
            )

        term = parts[0].strip()
        genes = {x.strip() for x in parts[2:] if x.strip()}

        if not term or not genes:
            raise ValueError(
                f"{label}: bad term/genes on line {line_no}."
            )

        pathways[term] = genes

    if len(pathways) < 20:
        raise ValueError(
            f"{label}: parsed only {len(pathways)} terms."
        )

    return pathways


def run_ora(
    query,
    background,
    pathways,
    fdr,
    min_size=None,
    max_size=None,
):
    N = len(background)
    n = len(query)
    rows = []

    for term, genes in pathways.items():
        pathway_bg = genes & background
        K = len(pathway_bg)

        if K == 0:
            continue

        if min_size is not None and K < min_size:
            continue

        if max_size is not None and K > max_size:
            continue

        overlap = query & pathway_bg
        k = len(overlap)

        a = k
        b = n - k
        c = K - k
        d = N - K - n + k

        if min(a, b, c, d) < 0:
            raise RuntimeError(
                f"Invalid contingency table for {term}."
            )

        odds_ratio, pvalue = fisher_exact(
            [[a, b], [c, d]],
            alternative="greater",
        )

        fold_enrichment = (
            (k / n) / (K / N)
            if K > 0 and n > 0
            else np.nan
        )

        rows.append(
            {
                "pathway": term,
                "background_size_N": N,
                "query_size_n": n,
                "pathway_size_in_background_K": K,
                "overlap_k": k,
                "fold_enrichment": fold_enrichment,
                "odds_ratio": odds_ratio,
                "pvalue": pvalue,
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )

    res = pd.DataFrame(rows)

    if res.empty:
        raise RuntimeError(
            "No pathways remained for ORA."
        )

    res["padj_BH"] = bh_adjust(res["pvalue"].to_numpy())
    res["fdr_significant"] = res["padj_BH"] < fdr

    return res.sort_values(
        ["padj_BH", "pvalue", "fold_enrichment"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_region(sub):
    return {
        "n_genes": len(sub),
        "n_annotated_symbols": sub["gene_symbol"].notna().sum(),
        "median_DESeq2_BH": sub["DESeq2_BH"].median(),
        "fraction_DESeq2_BH_lt_0.10": (sub["DESeq2_BH"] < 0.10).mean(),
        "fraction_DESeq2_BH_lt_0.25": (sub["DESeq2_BH"] < 0.25).mean(),
        "median_abs_DESeq2_log2FC": sub["abs_DESeq2_log2FC"].median(),
        "median_abs_Cohens_d": sub["abs_cohen_d"].median(),
        "median_abs_rank_biserial": sub["abs_rank_biserial"].median(),
        "median_abs_spearman_rho": sub["abs_spearman_rho"].median(),
        "fraction_monotonic": sub["is_monotonic"].mean(),
        "fraction_monotonic_increasing": (
            sub["median_trajectory"] == "monotonic_increasing"
        ).mean(),
        "fraction_monotonic_decreasing": (
            sub["median_trajectory"] == "monotonic_decreasing"
        ).mean(),
        "fraction_non_monotonic": (
            sub["median_trajectory"] == "non_monotonic"
        ).mean(),
        "median_AD_NCI_variance_ratio": sub[
            "AD_to_NCI_variance_ratio"
        ].median(),
    }


def choose_representatives(sub, region_key, n):
    x = sub.loc[sub["gene_symbol"].notna()].copy()

    if region_key == "deseq2_only":
        x["priority"] = -np.log10(
            x["DESeq2_BH"].clip(lower=1e-300)
        )

    elif region_key == "regular_only":
        x["priority"] = -np.log10(
            x["Regular_BH"].clip(lower=1e-300)
        )

    elif region_key == "weighted_only":
        x["priority"] = -np.log10(
            x["Weighted_BH"].clip(lower=1e-300)
        )

    x["trajectory_bonus"] = x["is_monotonic"].astype(int)

    return (
        x.sort_values(
            [
                "priority",
                "trajectory_bonus",
                "abs_spearman_rho",
                "abs_rank_biserial",
            ],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
        .head(n)
        .copy()
    )


def save_top_term_plot(res, title, outpath, top_n):
    sig = res.loc[res["fdr_significant"]].copy()
    plot_df = (sig if not sig.empty else res).head(top_n).iloc[::-1].copy()

    plot_df["minus_log10_BH"] = -np.log10(
        plot_df["padj_BH"].clip(lower=1e-300)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        plot_df["pathway"],
        plot_df["minus_log10_BH"],
    )

    ax.axvline(
        -np.log10(0.05),
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel("-log10(BH-adjusted p-value)")
    ax.set_ylabel("")

    status = (
        f"{len(sig)} FDR-significant"
        if len(sig) > 0
        else "none passed FDR"
    )

    ax.set_title(
        f"{title}\n{status}"
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    geometry = pd.read_csv(args.geometry)
    membership = pd.read_csv(args.membership)

    required_geometry = [
        "gene_id",
        "gene_symbol",
        "region",
        "DESeq2_BH",
        "DESeq2_log2FoldChange",
        "Regular_BH",
        "Weighted_BH",
        "cohen_d_AD_minus_NCI",
        "rank_biserial_AD_minus_NCI",
        "AD_to_NCI_variance_ratio",
        "spearman_severity_vs_expression",
        "median_trajectory",
        "abs_DESeq2_log2FC",
        "abs_cohen_d",
        "abs_rank_biserial",
        "abs_spearman_rho",
        "is_monotonic",
    ]

    missing = [
        c for c in required_geometry
        if c not in geometry.columns
    ]

    if missing:
        raise ValueError(
            f"Geometry table missing columns: {missing}"
        )

    # Exact background from final common-universe membership table.
    if "gene_symbol" not in membership.columns:
        raise ValueError(
            "Membership table lacks gene_symbol."
        )

    background = clean_symbols(
        membership["gene_symbol"]
    )

    subsets = {}

    for key, region_name in REGIONS.items():
        sub = geometry.loc[
            geometry["region"] == region_name
        ].copy()

        if sub.empty:
            raise ValueError(
                f"No genes found for region: {region_name}"
            )

        subsets[key] = sub

    print("=" * 92)
    print("AREA-SPECIFIC VENN REGION COMPARISON")
    print("=" * 92)

    summary_rows = []

    for key, sub in subsets.items():
        row = summarize_region(sub)
        row["region_key"] = key
        row["region"] = REGIONS[key]
        row["display"] = DISPLAY[key]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_out = outdir / "area_specific_region_summary.csv"
    summary.to_csv(summary_out, index=False)

    print("\nDESCRIPTIVE SUMMARY")
    print(
        summary[
            [
                "display",
                "n_genes",
                "median_DESeq2_BH",
                "fraction_DESeq2_BH_lt_0.10",
                "median_abs_DESeq2_log2FC",
                "median_abs_Cohens_d",
                "median_abs_rank_biserial",
                "median_abs_spearman_rho",
                "fraction_monotonic",
            ]
        ].to_string(index=False)
    )

    # Representatives.
    reps = []

    for key, sub in subsets.items():
        top = choose_representatives(
            sub,
            key,
            args.representatives_per_region,
        )

        top["region_key"] = key
        top["display"] = DISPLAY[key]
        reps.append(top)

    reps = pd.concat(
        reps,
        ignore_index=True,
    )

    reps_out = outdir / "representative_genes_area_specific_regions.csv"
    reps.to_csv(reps_out, index=False)

    print("\nREPRESENTATIVE GENES")
    print(
        reps[
            [
                "display",
                "gene_symbol",
                "gene_id",
                "DESeq2_BH",
                "Regular_BH",
                "Weighted_BH",
                "DESeq2_log2FoldChange",
                "cohen_d_AD_minus_NCI",
                "rank_biserial_AD_minus_NCI",
                "spearman_severity_vs_expression",
                "median_trajectory",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # Compact geometry plots
    # ------------------------------------------------------------
    plot_order = [
        "deseq2_only",
        "regular_only",
        "weighted_only",
    ]

    labels = [
        DISPLAY[k]
        for k in plot_order
    ]

    def boxplot(metric, ylabel, title, filename):
        data = [
            subsets[k][metric].dropna().to_numpy()
            for k in plot_order
        ]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=False,
        )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()

        fig.savefig(
            outdir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    boxplot(
        "abs_DESeq2_log2FC",
        "|DESeq2 log2FC|",
        "Endpoint mean-shift magnitude",
        "compare_three_regions_abs_log2FC.png",
    )

    boxplot(
        "abs_rank_biserial",
        "|Rank-biserial effect|",
        "Endpoint rank-order association",
        "compare_three_regions_rank_effect.png",
    )

    boxplot(
        "abs_spearman_rho",
        "|Spearman severity-expression rho|",
        "Graded NCI → MCI → AD association",
        "compare_three_regions_graded_effect.png",
    )

    mono_values = [
        summary.loc[
            summary["region_key"] == key,
            "fraction_monotonic",
        ].iloc[0]
        for key in plot_order
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, mono_values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction monotonic NCI → MCI → AD")
    ax.set_title("Monotonic graded trajectories")
    fig.tight_layout()
    fig.savefig(
        outdir / "compare_three_regions_monotonicity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # ORA
    # ------------------------------------------------------------
    libraries = {
        "Hallmark_2020": {
            "pathways": load_enrichr_library(
                args.hallmark_file,
                "Hallmark 2020",
            ),
            "min_size": None,
            "max_size": None,
        },
        "GO_Biological_Process_2025": {
            "pathways": load_enrichr_library(
                args.go_file,
                "GO Biological Process 2025",
            ),
            "min_size": args.min_pathway_size,
            "max_size": args.max_pathway_size,
        },
        "Reactome_Pathways_2024": {
            "pathways": load_enrichr_library(
                args.reactome_file,
                "Reactome Pathways 2024",
            ),
            "min_size": args.min_pathway_size,
            "max_size": args.max_pathway_size,
        },
    }

    enrichment_summary_rows = []
    enrichment_results = {}

    for lib_name, lib_info in libraries.items():
        print(f"\n{lib_name}")

        enrichment_results[lib_name] = {}

        for key in plot_order:
            query = (
                clean_symbols(
                    subsets[key]["gene_symbol"]
                )
                & background
            )

            res = run_ora(
                query=query,
                background=background,
                pathways=lib_info["pathways"],
                fdr=args.fdr,
                min_size=lib_info["min_size"],
                max_size=lib_info["max_size"],
            )

            enrichment_results[lib_name][key] = res

            region_dir = (
                outdir
                / "enrichment"
                / key
                / lib_name
            )

            region_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            res.to_csv(
                region_dir / "ora_all_results.csv",
                index=False,
            )

            sig = res.loc[
                res["fdr_significant"]
            ].copy()

            sig.to_csv(
                region_dir / "ora_FDR0.05.csv",
                index=False,
            )

            save_top_term_plot(
                res,
                f"{DISPLAY[key]} — {lib_name}",
                region_dir / "ora_top_terms.png",
                args.top_n,
            )

            enrichment_summary_rows.append(
                {
                    "library": lib_name,
                    "region_key": key,
                    "region": DISPLAY[key],
                    "query_symbols_n": len(query),
                    "background_symbols_n": len(background),
                    "terms_tested_n": len(res),
                    "fdr_significant_terms_n": len(sig),
                }
            )

            print(
                f"  {DISPLAY[key]:<20} "
                f"query={len(query):4d} | "
                f"tested={len(res):4d} | "
                f"FDR<0.05={len(sig):3d}"
            )

            print(
                res[
                    [
                        "pathway",
                        "overlap_k",
                        "fold_enrichment",
                        "pvalue",
                        "padj_BH",
                    ]
                ]
                .head(5)
                .to_string(index=False)
                .replace("\n", "\n    ")
            )

        # Direct comparison table across all three regions for each library.
        base = None

        for key in plot_order:
            res = enrichment_results[lib_name][key].copy()

            keep = res[
                [
                    "pathway",
                    "pathway_size_in_background_K",
                    "overlap_k",
                    "fold_enrichment",
                    "odds_ratio",
                    "pvalue",
                    "padj_BH",
                    "fdr_significant",
                    "overlap_genes",
                ]
            ].copy()

            rename = {
                c: f"{key}_{c}"
                for c in keep.columns
                if c not in {
                    "pathway",
                    "pathway_size_in_background_K",
                }
            }

            keep = keep.rename(
                columns=rename
            )

            if base is None:
                base = keep

            else:
                base = base.merge(
                    keep,
                    on=[
                        "pathway",
                        "pathway_size_in_background_K",
                    ],
                    how="outer",
                    validate="one_to_one",
                )

        base.to_csv(
            outdir / f"{lib_name}_three_region_comparison.csv",
            index=False,
        )

    enrichment_summary = pd.DataFrame(
        enrichment_summary_rows
    )

    enrichment_summary_out = (
        outdir
        / "enrichment_summary.csv"
    )

    enrichment_summary.to_csv(
        enrichment_summary_out,
        index=False,
    )

    print("\nENRICHMENT SUMMARY")
    print(
        enrichment_summary.to_string(
            index=False
        )
    )

    (outdir / "analysis_design.txt").write_text(
        "Focused Venn-region comparison\n\n"
        "Regions:\n"
        "  DESeq2 only\n"
        "  Adjusted Regular AREA only\n"
        "  Weighted AREA only\n\n"
        "Primary contrasts:\n"
        "  Weighted-only vs DESeq2-only: graded severity sensitivity\n"
        "  Regular-only vs DESeq2-only: binary rank sensitivity\n\n"
        "Geometry metrics are descriptive; no formal region-vs-region "
        "hypothesis tests are performed because region membership is itself "
        "defined by significance thresholds.\n\n"
        "ORA:\n"
        "  one-sided Fisher exact test (greater)\n"
        "  BH correction separately within each region x library\n"
        "  background = unique annotated symbols in locked common universe\n"
        f"  GO/Reactome size filter: "
        f"{args.min_pathway_size} <= K <= {args.max_pathway_size}\n"
    )

    print("\nWROTE")
    print(f"  {summary_out}")
    print(f"  {reps_out}")
    print(f"  {enrichment_summary_out}")
    print(f"  {outdir / 'Hallmark_2020_three_region_comparison.csv'}")
    print(f"  {outdir / 'GO_Biological_Process_2025_three_region_comparison.csv'}")
    print(f"  {outdir / 'Reactome_Pathways_2024_three_region_comparison.csv'}")


if __name__ == "__main__":
    main()

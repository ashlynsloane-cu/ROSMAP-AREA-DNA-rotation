#!/usr/bin/env python3
"""
compare_468_vs_allthree_enrichment.py
=====================================

Compare over-representation analysis (ORA) for two locked Venn regions:

1) Rank-based-only set:
   Adjusted Regular AREA significant
   AND Weighted AREA significant
   AND DESeq2 non-significant
   Expected Ensembl-level size: 468

2) All-three set:
   DESeq2 significant
   AND Adjusted Regular AREA significant
   AND Weighted AREA significant
   Expected Ensembl-level size: 2,123

Libraries are read from LOCAL Enrichr text-format files:
- GO Biological Process 2025
- Reactome Pathways 2024

Background:
Unique annotated gene symbols in the locked 32,994-gene common comparison
universe.

Statistics:
- one-sided Fisher exact test ("greater")
- BH correction separately for each query set × library
- pathway size filter applied AFTER intersecting each pathway with the
  ROSMAP background; defaults: 10 <= K <= 500

Outputs include full/significant ORA tables, top-term plots, per-library
comparison tables, and a summary across both query sets.
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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--membership",
        default=(
            "results/visualizations/"
            "final_locked_covariate_DESeq2_Regular_Weighted_venn_gene_membership.csv"
        ),
    )

    p.add_argument(
        "--go-file",
        default="data/gene_sets/GO_Biological_Process_2025.txt",
    )

    p.add_argument(
        "--reactome-file",
        default="data/gene_sets/Reactome_Pathways_2024.txt",
    )

    p.add_argument("--expected-universe-n", type=int, default=32994)
    p.add_argument("--expected-rank-only-n", type=int, default=468)
    p.add_argument("--expected-all-three-n", type=int, default=2123)

    p.add_argument("--min-pathway-size", type=int, default=10)
    p.add_argument("--max-pathway-size", type=int, default=500)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=20)

    p.add_argument(
        "--outdir",
        default="results/characterization_468/go_reactome_comparison",
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
            f"{label} library file not found: {path}"
        )

    pathways = {}

    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue

        parts = raw.rstrip("\n").split("\t")

        if len(parts) < 3:
            raise ValueError(
                f"{label}: malformed line {line_no}; expected at least "
                "3 tab-delimited fields."
            )

        term = parts[0].strip()
        genes = {x.strip() for x in parts[2:] if x.strip()}

        if not term:
            raise ValueError(f"{label}: empty term on line {line_no}.")

        if not genes:
            raise ValueError(
                f"{label}: no genes parsed for '{term}' on line {line_no}."
            )

        if term in pathways:
            raise ValueError(
                f"{label}: duplicate term '{term}'."
            )

        pathways[term] = genes

    if len(pathways) < 20:
        raise ValueError(
            f"{label}: parsed only {len(pathways)} terms; "
            "the library file may be wrong or malformed."
        )

    return pathways


def run_ora(
    query_symbols,
    background_symbols,
    pathways,
    min_size,
    max_size,
    fdr,
):
    N = len(background_symbols)
    n = len(query_symbols)

    rows = []

    for term, genes in pathways.items():
        pathway_bg = genes & background_symbols
        K = len(pathway_bg)

        if K < min_size or K > max_size:
            continue

        overlap = query_symbols & pathway_bg
        k = len(overlap)

        a = k
        b = n - k
        c = K - k
        d = N - K - n + k

        if min(a, b, c, d) < 0:
            raise RuntimeError(
                f"Invalid contingency table for '{term}': "
                f"{a}, {b}, {c}, {d}"
            )

        odds_ratio, pvalue = fisher_exact(
            [[a, b], [c, d]],
            alternative="greater",
        )

        query_fraction = k / n
        background_fraction = K / N
        fold_enrichment = (
            query_fraction / background_fraction
            if background_fraction > 0
            else np.nan
        )

        rows.append(
            {
                "pathway": term,
                "background_size_N": N,
                "query_size_n": n,
                "pathway_size_in_background_K": K,
                "overlap_k": k,
                "query_fraction": query_fraction,
                "background_fraction": background_fraction,
                "fold_enrichment": fold_enrichment,
                "odds_ratio": odds_ratio,
                "pvalue": pvalue,
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )

    res = pd.DataFrame(rows)

    if res.empty:
        raise RuntimeError(
            "No pathways remained after background intersection and "
            "pathway-size filtering."
        )

    res["padj_BH"] = bh_adjust(res["pvalue"].to_numpy())
    res["fdr_significant"] = res["padj_BH"] < fdr

    return res.sort_values(
        ["padj_BH", "pvalue", "fold_enrichment"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def save_plot(res, query_label, library_label, outpath, top_n):
    sig = res.loc[res["fdr_significant"]].copy()
    plot_df = (sig if not sig.empty else res).head(top_n).iloc[::-1].copy()

    plot_df["minus_log10_BH"] = -np.log10(
        plot_df["padj_BH"].clip(lower=1e-300)
    )

    fig, ax = plt.subplots(figsize=(11, 8))

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
        f"{len(sig)} FDR-significant terms"
        if len(sig) > 0
        else "no terms passed FDR"
    )

    ax.set_title(
        f"{library_label} enrichment — {query_label}\n{status}"
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.membership)

    required = [
        "gene_id",
        "gene_symbol",
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Membership file is missing required columns: {missing}"
        )

    if len(df) != args.expected_universe_n:
        raise ValueError(
            f"Expected {args.expected_universe_n:,} universe rows, "
            f"found {len(df):,}."
        )

    rank_only_mask = (
        (~df["DESeq2_significant"].astype(bool))
        & df["Regular_AREA_significant"].astype(bool)
        & df["Weighted_AREA_significant"].astype(bool)
    )

    all_three_mask = (
        df["DESeq2_significant"].astype(bool)
        & df["Regular_AREA_significant"].astype(bool)
        & df["Weighted_AREA_significant"].astype(bool)
    )

    rank_only = df.loc[rank_only_mask].copy()
    all_three = df.loc[all_three_mask].copy()

    if len(rank_only) != args.expected_rank_only_n:
        raise ValueError(
            f"Expected {args.expected_rank_only_n} rank-only genes, "
            f"found {len(rank_only)}."
        )

    if len(all_three) != args.expected_all_three_n:
        raise ValueError(
            f"Expected {args.expected_all_three_n} all-three genes, "
            f"found {len(all_three)}."
        )

    background = clean_symbols(df["gene_symbol"])
    rank_only_symbols = clean_symbols(rank_only["gene_symbol"]) & background
    all_three_symbols = clean_symbols(all_three["gene_symbol"]) & background

    print("=" * 88)
    print("GO BP + REACTOME ORA: 468 RANK-ONLY VS 2,123 ALL-THREE GENES")
    print("=" * 88)
    print(f"Common Ensembl universe:        {len(df):,}")
    print(f"Annotated background symbols:   {len(background):,}")
    print(f"Rank-only Ensembl genes:         {len(rank_only):,}")
    print(f"Rank-only annotated symbols:     {len(rank_only_symbols):,}")
    print(f"All-three Ensembl genes:         {len(all_three):,}")
    print(f"All-three annotated symbols:     {len(all_three_symbols):,}")
    print(
        f"Pathway size filter:            "
        f"{args.min_pathway_size} <= K <= {args.max_pathway_size}"
    )

    libraries = {
        "GO_Biological_Process_2025": load_enrichr_library(
            args.go_file,
            "GO Biological Process 2025",
        ),
        "Reactome_Pathways_2024": load_enrichr_library(
            args.reactome_file,
            "Reactome Pathways 2024",
        ),
    }

    queries = {
        "rank_only_468": {
            "label": "Regular + Weighted AREA, not DESeq2",
            "symbols": rank_only_symbols,
        },
        "all_three_2123": {
            "label": "Significant in all three methods",
            "symbols": all_three_symbols,
        },
    }

    summary_rows = []

    for library_name, pathways in libraries.items():
        print(
            f"\n{library_name}: parsed {len(pathways):,} raw terms"
        )

        lib_results = {}

        for query_name, query_info in queries.items():
            query_symbols = query_info["symbols"]

            res = run_ora(
                query_symbols=query_symbols,
                background_symbols=background,
                pathways=pathways,
                min_size=args.min_pathway_size,
                max_size=args.max_pathway_size,
                fdr=args.fdr,
            )

            lib_results[query_name] = res

            query_dir = outdir / query_name / library_name
            query_dir.mkdir(parents=True, exist_ok=True)

            all_out = query_dir / "ora_all_results.csv"
            sig_out = query_dir / "ora_FDR0.05.csv"
            plot_out = query_dir / "ora_top_terms.png"
            symbols_out = query_dir / "query_symbols.txt"

            res.to_csv(all_out, index=False)

            sig = res.loc[res["fdr_significant"]].copy()
            sig.to_csv(sig_out, index=False)

            symbols_out.write_text(
                "\n".join(sorted(query_symbols)) + "\n"
            )

            save_plot(
                res=res,
                query_label=query_info["label"],
                library_label=library_name,
                outpath=plot_out,
                top_n=args.top_n,
            )

            summary_rows.append(
                {
                    "library": library_name,
                    "query": query_name,
                    "ensembl_set_size": (
                        len(rank_only)
                        if query_name == "rank_only_468"
                        else len(all_three)
                    ),
                    "annotated_query_symbols_n": len(query_symbols),
                    "annotated_background_symbols_n": len(background),
                    "raw_library_terms_n": len(pathways),
                    "terms_tested_after_size_filter_n": len(res),
                    "fdr_threshold": args.fdr,
                    "fdr_significant_terms_n": len(sig),
                    "min_pathway_size": args.min_pathway_size,
                    "max_pathway_size": args.max_pathway_size,
                }
            )

            print(
                f"  {query_name}: tested {len(res):,} terms; "
                f"FDR<0.05 = {len(sig):,}"
            )

            print("    Top 10:")
            print(
                res[
                    [
                        "pathway",
                        "overlap_k",
                        "pathway_size_in_background_K",
                        "fold_enrichment",
                        "odds_ratio",
                        "pvalue",
                        "padj_BH",
                    ]
                ]
                .head(10)
                .to_string(index=False)
                .replace("\n", "\n    ")
            )

        # Side-by-side comparison for terms tested in both query sets.
        rank_res = lib_results["rank_only_468"].copy()
        all_res = lib_results["all_three_2123"].copy()

        rank_res = rank_res.rename(
            columns={
                "overlap_k": "rank_only_overlap_k",
                "fold_enrichment": "rank_only_fold_enrichment",
                "odds_ratio": "rank_only_odds_ratio",
                "pvalue": "rank_only_pvalue",
                "padj_BH": "rank_only_padj_BH",
                "fdr_significant": "rank_only_fdr_significant",
                "overlap_genes": "rank_only_overlap_genes",
            }
        )

        all_res = all_res.rename(
            columns={
                "overlap_k": "all_three_overlap_k",
                "fold_enrichment": "all_three_fold_enrichment",
                "odds_ratio": "all_three_odds_ratio",
                "pvalue": "all_three_pvalue",
                "padj_BH": "all_three_padj_BH",
                "fdr_significant": "all_three_fdr_significant",
                "overlap_genes": "all_three_overlap_genes",
            }
        )

        comparison = rank_res[
            [
                "pathway",
                "pathway_size_in_background_K",
                "rank_only_overlap_k",
                "rank_only_fold_enrichment",
                "rank_only_odds_ratio",
                "rank_only_pvalue",
                "rank_only_padj_BH",
                "rank_only_fdr_significant",
                "rank_only_overlap_genes",
            ]
        ].merge(
            all_res[
                [
                    "pathway",
                    "all_three_overlap_k",
                    "all_three_fold_enrichment",
                    "all_three_odds_ratio",
                    "all_three_pvalue",
                    "all_three_padj_BH",
                    "all_three_fdr_significant",
                    "all_three_overlap_genes",
                ]
            ],
            on="pathway",
            how="inner",
            validate="one_to_one",
        )

        comparison["significance_pattern"] = np.select(
            [
                comparison["rank_only_fdr_significant"]
                & comparison["all_three_fdr_significant"],
                comparison["rank_only_fdr_significant"]
                & ~comparison["all_three_fdr_significant"],
                ~comparison["rank_only_fdr_significant"]
                & comparison["all_three_fdr_significant"],
            ],
            [
                "significant_both",
                "rank_only_specific",
                "all_three_specific",
            ],
            default="significant_neither",
        )

        comparison["fold_enrichment_difference_rank_minus_all"] = (
            comparison["rank_only_fold_enrichment"]
            - comparison["all_three_fold_enrichment"]
        )

        comparison = comparison.sort_values(
            [
                "rank_only_padj_BH",
                "all_three_padj_BH",
            ],
            ascending=[True, True],
            kind="mergesort",
        )

        comparison.to_csv(
            outdir / f"{library_name}_rank_only_vs_all_three.csv",
            index=False,
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        outdir / "enrichment_summary.csv",
        index=False,
    )

    (outdir / "analysis_design.txt").write_text(
        "Queries:\n"
        "  rank_only_468 = Regular AREA significant AND Weighted AREA "
        "significant AND DESeq2 non-significant\n"
        "  all_three_2123 = significant in DESeq2, Regular AREA, "
        "and Weighted AREA\n\n"
        "Background:\n"
        "  unique annotated gene symbols in the locked 32,994-gene "
        "common comparison universe\n\n"
        "Libraries:\n"
        f"  GO: {args.go_file}\n"
        f"  Reactome: {args.reactome_file}\n\n"
        "Statistics:\n"
        "  one-sided Fisher exact test (greater)\n"
        "  BH correction separately within each query x library\n"
        f"  pathway size filter after ROSMAP-background intersection: "
        f"{args.min_pathway_size} <= K <= {args.max_pathway_size}\n"
    )

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"Outputs: {outdir}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

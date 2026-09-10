#!/usr/bin/env python3
"""
run_age_preranked_gsea.py
=========================

Genome-wide preranked GSEA for the age-method comparison using the exact
33,006-gene membership table.

Ranking orientation is aligned across methods:
  positive score = expression increases with age
  negative score = expression decreases with age

  DESeq2:        Wald statistic
  Regular AREA:  -Regular_AREA_Z
  Weighted AREA: -Weighted_AREA_Z

Gene symbols are collapsed within each method by retaining the Ensembl entry
with the largest absolute ranking statistic. GSEA uses local Enrichr-format
Hallmark 2020, GO Biological Process 2025, and Reactome 2024 libraries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


LIBRARIES = {
    "Hallmark_2020": "data/gene_sets/MSigDB_Hallmark_2020.txt",
    "GO_Biological_Process_2025": "data/gene_sets/GO_Biological_Process_2025.txt",
    "Reactome_Pathways_2024": "data/gene_sets/Reactome_Pathways_2024.txt",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--membership",
        default="results/age_prelim/comparison/age_method_gene_membership.csv",
    )
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--min-size", type=int, default=10)
    p.add_argument("--max-size", type=int, default=500)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--expected-common-genes", type=int, default=33006)
    p.add_argument(
        "--outdir",
        default="results/age_prelim/enrichment/preranked_gsea",
    )
    p.add_argument(
        "--hallmark-file", default=LIBRARIES["Hallmark_2020"]
    )
    p.add_argument(
        "--go-file", default=LIBRARIES["GO_Biological_Process_2025"]
    )
    p.add_argument(
        "--reactome-file", default=LIBRARIES["Reactome_Pathways_2024"]
    )
    return p.parse_args()


def require_gseapy():
    try:
        import gseapy as gp
        return gp
    except ImportError:
        print(
            "\nERROR: gseapy is not installed. Install it with:\n\n"
            "    python3 -m pip install gseapy\n",
            file=sys.stderr,
        )
        raise


def clean_symbol_series(series):
    s = series.astype("string").str.strip()
    bad = s.isna() | (s == "") | s.str.lower().isin(
        ["nan", "none", "na", "n/a", "<na>"]
    )
    return s.mask(bad)


def load_enrichr_library(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} library not found: {path}")
    pathways = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < 3:
            raise ValueError(f"{label}: malformed line {line_no}")
        term = parts[0].strip()
        genes = {g.strip() for g in parts[2:] if g.strip()}
        if not term or not genes:
            raise ValueError(f"{label}: empty term/gene set on line {line_no}")
        pathways[term] = genes
    if len(pathways) < 20:
        raise ValueError(f"{label}: parsed only {len(pathways)} terms")
    return pathways


def collapse_to_symbols(df, score_col, method):
    x = df[["gene_id", "gene_symbol", score_col]].copy()
    x[score_col] = pd.to_numeric(x[score_col], errors="coerce")
    x["gene_symbol"] = clean_symbol_series(x["gene_symbol"])
    x = x.loc[
        x["gene_symbol"].notna()
        & x[score_col].notna()
        & np.isfinite(x[score_col])
    ].copy()
    x["abs_score"] = x[score_col].abs()
    x = (
        x.sort_values(
            ["gene_symbol", "abs_score", "gene_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("gene_symbol", keep="first")
        .rename(columns={score_col: "score"})
    )
    x["method"] = method
    return x[["gene_symbol", "score", "gene_id", "method"]]


def standardize_gseapy_results(res2d):
    x = res2d.copy()
    if "Term" not in x.columns:
        x = x.reset_index()
    rename_candidates = {
        "Term": "pathway",
        "ES": "ES",
        "NES": "NES",
        "NOM p-val": "NOM_pvalue",
        "NOM p-val ": "NOM_pvalue",
        "FDR q-val": "FDR_qvalue",
        "FWER p-val": "FWER_pvalue",
        "Tag %": "Tag_percent",
        "Gene %": "Gene_percent",
        "Lead_genes": "leading_edge_genes",
        "Lead_genes ": "leading_edge_genes",
    }
    x = x.rename(
        columns={k: v for k, v in rename_candidates.items() if k in x.columns}
    )
    if "pathway" not in x.columns:
        raise ValueError(
            f"Could not identify pathway column. Columns: {x.columns.tolist()}"
        )
    for col in ["ES", "NES", "NOM_pvalue", "FDR_qvalue", "FWER_pvalue"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x


def main():
    args = parse_args()
    gp = require_gseapy()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rank_dir = outdir / "ranking_tables"
    rank_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.membership)
    required = {
        "gene_id", "gene_symbol", "stat", "Regular_AREA_Z", "Weighted_AREA_Z"
    }
    missing = sorted(required - set(m.columns))
    if missing:
        raise ValueError(f"Membership table missing columns: {missing}")
    if len(m) != args.expected_common_genes:
        raise ValueError(
            f"Membership universe has {len(m):,}; expected {args.expected_common_genes:,}."
        )

    m["gene_id"] = m["gene_id"].astype(str).str.strip()
    m["gene_symbol"] = clean_symbol_series(m["gene_symbol"])
    m["DESeq2_score"] = pd.to_numeric(m["stat"], errors="coerce")
    m["Regular_score"] = -pd.to_numeric(m["Regular_AREA_Z"], errors="coerce")
    m["Weighted_score"] = -pd.to_numeric(m["Weighted_AREA_Z"], errors="coerce")

    ranks = {
        "DESeq2": collapse_to_symbols(m, "DESeq2_score", "DESeq2"),
        "Regular_AREA": collapse_to_symbols(m, "Regular_score", "Regular_AREA"),
        "Weighted_AREA": collapse_to_symbols(m, "Weighted_score", "Weighted_AREA"),
    }

    print("=" * 96)
    print("AGE GENOME-WIDE PRERANKED GSEA")
    print("=" * 96)
    print(f"Common Ensembl universe: {len(m):,}")
    print("Rank orientation: positive = expression increases with age")
    for method, rank_df in ranks.items():
        print(f"  {method:<15} {len(rank_df):,} unique annotated symbols")
        rank_df.to_csv(rank_dir / f"{method}_ranking.csv", index=False)

    library_paths = {
        "Hallmark_2020": args.hallmark_file,
        "GO_Biological_Process_2025": args.go_file,
        "Reactome_Pathways_2024": args.reactome_file,
    }
    libraries = {
        name: load_enrichr_library(path, name)
        for name, path in library_paths.items()
    }

    summary_rows = []
    all_results = {}

    for library_name, gene_sets in libraries.items():
        print(f"\n{library_name}: {len(gene_sets):,} raw gene sets")
        lib_dir = outdir / library_name
        lib_dir.mkdir(parents=True, exist_ok=True)
        all_results[library_name] = {}

        for method, rank_df in ranks.items():
            print(f"  Running {method}...")
            rnk = (
                rank_df[["gene_symbol", "score"]]
                .sort_values("score", ascending=False, kind="mergesort")
                .copy()
            )
            pre = gp.prerank(
                rnk=rnk,
                gene_sets=gene_sets,
                min_size=args.min_size,
                max_size=args.max_size,
                permutation_num=args.permutations,
                weight=1.0,
                ascending=False,
                seed=args.seed,
                threads=args.threads,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            res = standardize_gseapy_results(pre.res2d)
            res["method"] = method
            res["library"] = library_name
            if "FDR_qvalue" not in res.columns:
                raise ValueError(
                    f"gseapy result lacks FDR_qvalue; columns={res.columns.tolist()}"
                )
            res["fdr_significant"] = res["FDR_qvalue"] < args.fdr
            all_results[library_name][method] = res

            method_dir = lib_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            res.to_csv(method_dir / "gsea_all_results.csv", index=False)
            res.loc[res["fdr_significant"]].to_csv(
                method_dir / "gsea_FDR0.05.csv", index=False
            )
            summary_rows.append({
                "library": library_name,
                "method": method,
                "unique_ranked_symbols": len(rank_df),
                "pathways_tested": len(res),
                "fdr_significant_pathways": int(res["fdr_significant"].sum()),
                "positive_NES_significant": int(
                    (res["fdr_significant"] & (res["NES"] > 0)).sum()
                ),
                "negative_NES_significant": int(
                    (res["fdr_significant"] & (res["NES"] < 0)).sum()
                ),
            })

        # Wide comparison table by pathway.
        method_frames = []
        for method, res in all_results[library_name].items():
            cols = ["pathway", "NES", "FDR_qvalue", "fdr_significant"]
            x = res[cols].copy().rename(columns={
                "NES": f"{method}_NES",
                "FDR_qvalue": f"{method}_FDR",
                "fdr_significant": f"{method}_significant",
            })
            method_frames.append(x)

        comp = method_frames[0]
        for x in method_frames[1:]:
            comp = comp.merge(x, on="pathway", how="outer", validate="one_to_one")

        comp["AREA_both_significant_DESeq2_not"] = (
            comp["Regular_AREA_significant"].fillna(False).astype(bool)
            & comp["Weighted_AREA_significant"].fillna(False).astype(bool)
            & ~comp["DESeq2_significant"].fillna(False).astype(bool)
        )
        comp["Weighted_significant_DESeq2_not"] = (
            comp["Weighted_AREA_significant"].fillna(False).astype(bool)
            & ~comp["DESeq2_significant"].fillna(False).astype(bool)
        )
        comp.to_csv(
            outdir / f"{library_name}_method_comparison.csv", index=False
        )
        comp.loc[comp["AREA_both_significant_DESeq2_not"]].to_csv(
            outdir / f"{library_name}_both_AREA_not_DESeq2.csv", index=False
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "gsea_summary.csv", index=False)

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)
    print(f"Outputs: {outdir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

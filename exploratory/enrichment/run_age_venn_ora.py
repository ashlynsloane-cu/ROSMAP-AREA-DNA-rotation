#!/usr/bin/env python3
"""
run_age_venn_ora.py
===================

Over-representation analysis (ORA) for the age-method comparison.

Primary Venn regions:
  1) Regular + Weighted AREA, not DESeq2
  2) Weighted AREA only
  3) All three methods
  4) DESeq2 only

Each region is tested three ways:
  - all significant genes in the region
  - genes increasing with age
  - genes decreasing with age

Libraries (local Enrichr text format):
  - MSigDB Hallmark 2020
  - GO Biological Process 2025
  - Reactome Pathways 2024

Statistics:
  - background: unique annotated gene symbols in the exact age comparison universe
  - one-sided Fisher exact test (greater)
  - BH correction separately within each region x direction x library
  - pathway-size filter after background intersection: 10 <= K <= 500

Age-direction convention:
  - DESeq2: positive Wald/log2FC = higher expression in older participants
  - AREA: negative Z = higher expression with increasing age, so age_score = -AREA_Z
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
    "area_shared_not_deseq2": "Regular AREA + Weighted AREA only",
    "weighted_only": "Weighted AREA only",
    "all_three": "All three",
    "deseq2_only": "DESeq2 only",
}

DISPLAY = {
    "area_shared_not_deseq2": "Regular + Weighted AREA, not DESeq2",
    "weighted_only": "Weighted AREA only",
    "all_three": "All three methods",
    "deseq2_only": "DESeq2 only",
}

EXPECTED_COUNTS = {
    "area_shared_not_deseq2": 121,
    "weighted_only": 218,
    "all_three": 173,
    "deseq2_only": 106,
}

LIBRARIES = {
    "Hallmark_2020": {
        "default": "data/gene_sets/MSigDB_Hallmark_2020.txt",
        "use_size_filter": False,
    },
    "GO_Biological_Process_2025": {
        "default": "data/gene_sets/GO_Biological_Process_2025.txt",
        "use_size_filter": True,
    },
    "Reactome_Pathways_2024": {
        "default": "data/gene_sets/Reactome_Pathways_2024.txt",
        "use_size_filter": True,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--membership",
        default="results/age_prelim/comparison/age_method_gene_membership.csv",
    )
    p.add_argument(
        "--hallmark-file",
        default=LIBRARIES["Hallmark_2020"]["default"],
    )
    p.add_argument(
        "--go-file",
        default=LIBRARIES["GO_Biological_Process_2025"]["default"],
    )
    p.add_argument(
        "--reactome-file",
        default=LIBRARIES["Reactome_Pathways_2024"]["default"],
    )
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--min-pathway-size", type=int, default=10)
    p.add_argument("--max-pathway-size", type=int, default=500)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--expected-universe-n", type=int, default=33006)
    p.add_argument(
        "--skip-expected-region-check",
        action="store_true",
        help="Do not enforce the current 121/218/173/106 Venn counts.",
    )
    p.add_argument(
        "--outdir",
        default="results/age_prelim/enrichment/venn_ora",
    )
    return p.parse_args()


def clean_symbol_series(series):
    s = series.astype("string").str.strip()
    bad = s.isna() | (s == "") | s.str.lower().isin(
        ["nan", "none", "na", "n/a", "<na>"]
    )
    return s.mask(bad)


def clean_symbols(series):
    return set(clean_symbol_series(series).dropna().astype(str))


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if len(p) == 0:
        return np.array([], dtype=float)
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


def run_ora(query, background, pathways, fdr, min_size=None, max_size=None):
    query = set(query) & set(background)
    background = set(background)
    N = len(background)
    n = len(query)

    if n == 0:
        return pd.DataFrame()

    rows = []
    for term, genes in pathways.items():
        pathway_bg = set(genes) & background
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
            raise RuntimeError(f"Invalid contingency table for {term}")

        odds_ratio, pvalue = fisher_exact(
            [[a, b], [c, d]], alternative="greater"
        )
        fold_enrichment = (k / n) / (K / N) if K and n else np.nan
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
        return res
    res["padj_BH"] = bh_adjust(res["pvalue"].to_numpy())
    res["fdr_significant"] = res["padj_BH"] < fdr
    return res.sort_values(
        ["padj_BH", "pvalue", "fold_enrichment"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def assign_age_score(df):
    """Positive = expression increases with age; negative = decreases."""
    out = df.copy()
    out["age_score"] = np.nan
    out["age_score_source"] = ""

    weighted_mask = out["Weighted_AREA_significant"].astype(bool)
    out.loc[weighted_mask, "age_score"] = -pd.to_numeric(
        out.loc[weighted_mask, "Weighted_AREA_Z"], errors="coerce"
    )
    out.loc[weighted_mask, "age_score_source"] = "-Weighted_AREA_Z"

    regular_mask = (
        ~weighted_mask & out["Regular_AREA_significant"].astype(bool)
    )
    out.loc[regular_mask, "age_score"] = -pd.to_numeric(
        out.loc[regular_mask, "Regular_AREA_Z"], errors="coerce"
    )
    out.loc[regular_mask, "age_score_source"] = "-Regular_AREA_Z"

    deseq_mask = (
        ~weighted_mask
        & ~regular_mask
        & out["DESeq2_significant"].astype(bool)
    )
    # Wald statistic is preferred for sign; log2FC is equivalent for direction.
    out.loc[deseq_mask, "age_score"] = pd.to_numeric(
        out.loc[deseq_mask, "stat"], errors="coerce"
    )
    out.loc[deseq_mask, "age_score_source"] = "DESeq2_Wald_stat"

    out["age_direction"] = np.select(
        [out["age_score"] > 0, out["age_score"] < 0],
        ["increases_with_age", "decreases_with_age"],
        default="undetermined",
    )
    return out


def save_top_plot(res, title, outpath, top_n, fdr):
    if res.empty:
        return
    sig = res.loc[res["fdr_significant"]].copy()
    plot_df = (sig if not sig.empty else res).head(top_n).iloc[::-1].copy()
    plot_df["minus_log10_BH"] = -np.log10(
        plot_df["padj_BH"].clip(lower=1e-300)
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(plot_df["pathway"], plot_df["minus_log10_BH"])
    ax.axvline(-np.log10(fdr), linestyle="--", linewidth=1.0)
    ax.set_xlabel("-log10(BH-adjusted p-value)")
    ax.set_ylabel("")
    status = f"{len(sig)} FDR-significant" if len(sig) else "none passed FDR"
    ax.set_title(f"{title}\n{status}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.membership)
    required = {
        "gene_id", "gene_symbol", "stat", "log2FoldChange",
        "Regular_AREA_Z", "Weighted_AREA_Z",
        "DESeq2_significant", "Regular_AREA_significant",
        "Weighted_AREA_significant", "overlap_region",
    }
    missing = sorted(required - set(m.columns))
    if missing:
        raise ValueError(f"Membership table missing columns: {missing}")

    if len(m) != args.expected_universe_n:
        raise ValueError(
            f"Membership universe has {len(m):,} genes; "
            f"expected {args.expected_universe_n:,}."
        )

    m["gene_symbol"] = clean_symbol_series(m["gene_symbol"])
    m = assign_age_score(m)
    background = clean_symbols(m["gene_symbol"])

    subsets = {}
    for key, region in REGIONS.items():
        sub = m.loc[m["overlap_region"] == region].copy()
        if sub.empty:
            raise ValueError(f"No genes found for region: {region}")
        if not args.skip_expected_region_check:
            expected = EXPECTED_COUNTS[key]
            if len(sub) != expected:
                raise ValueError(
                    f"{region}: found {len(sub)} genes; expected {expected}."
                )
        subsets[key] = sub

    library_paths = {
        "Hallmark_2020": args.hallmark_file,
        "GO_Biological_Process_2025": args.go_file,
        "Reactome_Pathways_2024": args.reactome_file,
    }
    libraries = {
        name: load_enrichr_library(path, name)
        for name, path in library_paths.items()
    }

    print("=" * 96)
    print("AGE VENN ENRICHMENT: ORA")
    print("=" * 96)
    print(f"Common Ensembl universe:       {len(m):,}")
    print(f"Annotated background symbols: {len(background):,}")
    print("Direction convention: positive age_score = expression increases with age")

    region_summary = []
    for key, sub in subsets.items():
        ann = sub["gene_symbol"].notna().sum()
        up = int((sub["age_direction"] == "increases_with_age").sum())
        down = int((sub["age_direction"] == "decreases_with_age").sum())
        region_summary.append({
            "region_key": key,
            "region": REGIONS[key],
            "display": DISPLAY[key],
            "n_ensembl_genes": len(sub),
            "n_annotated_rows": int(ann),
            "n_unique_annotated_symbols": len(clean_symbols(sub["gene_symbol"])),
            "n_increases_with_age": up,
            "n_decreases_with_age": down,
        })
        print(
            f"  {DISPLAY[key]:<38} n={len(sub):>3}  "
            f"up={up:>3}  down={down:>3}"
        )

    pd.DataFrame(region_summary).to_csv(
        outdir / "age_venn_region_direction_summary.csv", index=False
    )

    all_rows = []
    summary_rows = []

    for lib_name, pathways in libraries.items():
        use_size_filter = LIBRARIES[lib_name]["use_size_filter"]
        min_size = args.min_pathway_size if use_size_filter else None
        max_size = args.max_pathway_size if use_size_filter else None
        print(f"\n{lib_name}: {len(pathways):,} raw terms")

        for key, sub in subsets.items():
            query_specs = {
                "all": sub,
                "increases_with_age": sub.loc[
                    sub["age_direction"] == "increases_with_age"
                ],
                "decreases_with_age": sub.loc[
                    sub["age_direction"] == "decreases_with_age"
                ],
            }

            for direction, qdf in query_specs.items():
                query = clean_symbols(qdf["gene_symbol"]) & background
                res = run_ora(
                    query=query,
                    background=background,
                    pathways=pathways,
                    fdr=args.fdr,
                    min_size=min_size,
                    max_size=max_size,
                )

                if res.empty:
                    n_sig = 0
                    n_tested = 0
                else:
                    res.insert(0, "library", lib_name)
                    res.insert(0, "direction", direction)
                    res.insert(0, "region", REGIONS[key])
                    res.insert(0, "region_key", key)
                    all_rows.append(res)
                    n_sig = int(res["fdr_significant"].sum())
                    n_tested = len(res)

                    stem = f"{lib_name}__{key}__{direction}"
                    res.to_csv(outdir / f"{stem}_all.csv", index=False)
                    res.loc[res["fdr_significant"]].to_csv(
                        outdir / f"{stem}_FDR0.05.csv", index=False
                    )
                    save_top_plot(
                        res,
                        title=f"{DISPLAY[key]} | {direction} | {lib_name}",
                        outpath=outdir / f"{stem}_top.png",
                        top_n=args.top_n,
                        fdr=args.fdr,
                    )

                summary_rows.append({
                    "library": lib_name,
                    "region_key": key,
                    "region": REGIONS[key],
                    "direction": direction,
                    "query_unique_annotated_symbols": len(query),
                    "terms_tested": n_tested,
                    "fdr_significant_terms": n_sig,
                })
                print(
                    f"  {key:<24} {direction:<20} "
                    f"query={len(query):>3}  FDR<0.05={n_sig:>3}"
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "enrichment_summary.csv", index=False)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(outdir / "all_ora_results.csv", index=False)

        # A compact discovery table: significant terms in the two AREA-focused sets.
        priority = combined.loc[
            combined["fdr_significant"]
            & combined["region_key"].isin(
                ["area_shared_not_deseq2", "weighted_only"]
            )
        ].copy()
        priority = priority.sort_values(
            ["region_key", "direction", "padj_BH", "fold_enrichment"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        priority.to_csv(outdir / "AREA_priority_significant_pathways.csv", index=False)

        # Cross-region recurrence summary for significant pathways.
        sig = combined.loc[combined["fdr_significant"]].copy()
        if not sig.empty:
            recurrence = (
                sig.groupby(["library", "pathway"], as_index=False)
                .agg(
                    n_region_direction_tests_significant=("region_key", "size"),
                    n_regions_significant=("region_key", "nunique"),
                    regions=("region_key", lambda x: ";".join(sorted(set(x)))),
                    directions=("direction", lambda x: ";".join(sorted(set(x)))),
                    min_padj_BH=("padj_BH", "min"),
                    max_fold_enrichment=("fold_enrichment", "max"),
                )
                .sort_values(
                    ["n_regions_significant", "min_padj_BH"],
                    ascending=[False, True],
                )
            )
            recurrence.to_csv(
                outdir / "significant_pathway_recurrence.csv", index=False
            )

    (outdir / "analysis_design.txt").write_text(
        "Age Venn ORA\n\n"
        "Primary regions:\n"
        "  area_shared_not_deseq2 = Regular AREA + Weighted AREA significant; DESeq2 non-significant\n"
        "  weighted_only = Weighted AREA significant only\n"
        "  all_three = significant in all three methods\n"
        "  deseq2_only = DESeq2 significant only\n\n"
        "Directions:\n"
        "  all\n"
        "  increases_with_age\n"
        "  decreases_with_age\n\n"
        "Background:\n"
        "  unique annotated gene symbols in the exact 33,006-gene age comparison universe\n\n"
        "Statistics:\n"
        "  one-sided Fisher exact test (greater)\n"
        "  BH correction separately within each region x direction x library\n"
        f"  GO/Reactome pathway size filter after background intersection: {args.min_pathway_size} <= K <= {args.max_pathway_size}\n"
        "  Hallmark: no additional size filter\n"
    )

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)
    print(f"Outputs: {outdir}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

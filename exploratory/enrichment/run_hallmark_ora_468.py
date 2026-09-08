#!/usr/bin/env python3
"""
Hallmark ORA for the 468 genes significant in adjusted Regular AREA and
Weighted AREA but not DESeq2.

Uses:
- query: annotated symbols among the 468 genes
- background: all annotated symbols in the locked 32,994-gene comparison universe
- gene sets: Enrichr MSigDB_Hallmark_2020
- test: one-sided Fisher exact test
- multiple testing: Benjamini-Hochberg across all Hallmark terms
"""

from pathlib import Path
import argparse
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

LIBRARY = "MSigDB_Hallmark_2020"
LIBRARY_URL = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary"
    "?mode=text&libraryName=" + urllib.parse.quote(LIBRARY)
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--membership",
        default=(
            "results/visualizations/"
            "final_locked_covariate_DESeq2_Regular_Weighted_venn_gene_membership.csv"
        ),
    )
    p.add_argument("--expected-target-n", type=int, default=468)
    p.add_argument("--expected-universe-n", type=int, default=32994)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument(
        "--outdir",
        default="results/characterization_468/hallmark_ora",
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


def download_library():
    req = urllib.request.Request(
        LIBRARY_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8")

    pathways = {}
    for raw in text.splitlines():
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        term = parts[0].strip()
        genes = {g.strip() for g in parts[2:] if g.strip()}
        if genes:
            pathways[term] = genes

    if not pathways:
        raise RuntimeError("Downloaded Hallmark library was empty.")

    return pathways


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
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != args.expected_universe_n:
        raise ValueError(
            f"Expected {args.expected_universe_n} universe genes, found {len(df)}."
        )

    target = df.loc[
        (~df["DESeq2_significant"].astype(bool))
        & df["Regular_AREA_significant"].astype(bool)
        & df["Weighted_AREA_significant"].astype(bool)
    ].copy()

    if len(target) != args.expected_target_n:
        raise ValueError(
            f"Expected {args.expected_target_n} target genes, found {len(target)}."
        )

    query = clean_symbols(target["gene_symbol"])
    background = clean_symbols(df["gene_symbol"])
    query &= background

    print("=" * 80)
    print("HALLMARK ORA — 468-GENE REGULAR + WEIGHTED / NOT DESEQ2 SET")
    print("=" * 80)
    print(f"Ensembl-level comparison universe: {len(df):,}")
    print(f"Ensembl-level target set:          {len(target):,}")
    print(f"Annotated query symbols:           {len(query):,}")
    print(f"Annotated background symbols:      {len(background):,}")

    (outdir / "hallmark_query_symbols.txt").write_text(
        "\n".join(sorted(query)) + "\n"
    )
    (outdir / "hallmark_background_symbols.txt").write_text(
        "\n".join(sorted(background)) + "\n"
    )
    (outdir / "hallmark_library_source.txt").write_text(
        f"Library: {LIBRARY}\n"
        f"Source: {LIBRARY_URL}\n"
        "Test: one-sided Fisher exact test (greater)\n"
        "Background: unique annotated symbols in the locked ROSMAP "
        "32,994-gene comparison universe\n"
    )

    print(f"\nDownloading {LIBRARY}...")
    pathways = download_library()
    print(f"Downloaded {len(pathways)} Hallmark terms.")

    N = len(background)
    n = len(query)
    rows = []

    for term, genes in pathways.items():
        pathway_bg = genes & background
        K = len(pathway_bg)
        if K == 0:
            continue

        overlap = query & pathway_bg
        k = len(overlap)

        a = k
        b = n - k
        c = K - k
        d = N - K - n + k

        odds_ratio, pvalue = fisher_exact(
            [[a, b], [c, d]],
            alternative="greater",
        )

        bg_fraction = K / N
        query_fraction = k / n
        fold_enrichment = (
            query_fraction / bg_fraction if bg_fraction > 0 else np.nan
        )

        rows.append(
            {
                "pathway": term,
                "background_size_N": N,
                "query_size_n": n,
                "pathway_size_in_background_K": K,
                "overlap_k": k,
                "query_fraction": query_fraction,
                "background_fraction": bg_fraction,
                "fold_enrichment": fold_enrichment,
                "odds_ratio": odds_ratio,
                "pvalue": pvalue,
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No Hallmark terms overlapped the background.")

    res["padj_BH"] = bh_adjust(res["pvalue"].to_numpy())
    res["fdr_significant"] = res["padj_BH"] < args.fdr
    res = res.sort_values(
        ["padj_BH", "pvalue", "fold_enrichment"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    all_out = outdir / "hallmark_ora_all_results.csv"
    sig_out = outdir / "hallmark_ora_FDR0.05.csv"
    summary_out = outdir / "hallmark_ora_summary.csv"

    res.to_csv(all_out, index=False)
    sig = res.loc[res["fdr_significant"]].copy()
    sig.to_csv(sig_out, index=False)

    pd.DataFrame(
        [{
            "common_ensembl_universe_n": len(df),
            "target_ensembl_n": len(target),
            "annotated_query_symbols_n": len(query),
            "annotated_background_symbols_n": len(background),
            "hallmark_terms_tested_n": len(res),
            "fdr_threshold": args.fdr,
            "fdr_significant_terms_n": len(sig),
        }]
    ).to_csv(summary_out, index=False)

    plot_df = (sig if not sig.empty else res).head(args.top_n).iloc[::-1].copy()
    plot_df["minus_log10_BH"] = -np.log10(
        plot_df["padj_BH"].clip(lower=1e-300)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["pathway"], plot_df["minus_log10_BH"])
    ax.set_xlabel("-log10(BH-adjusted p-value)")
    ax.set_ylabel("")
    ax.set_title(
        "Hallmark enrichment of Regular + Weighted AREA genes\n"
        "not significant by DESeq2"
    )
    fig.tight_layout()

    plot_out = outdir / "hallmark_ora_top_terms.png"
    fig.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nRESULT")
    print(f"Hallmark terms tested: {len(res):,}")
    print(f"FDR < {args.fdr:g}: {len(sig):,} pathways")
    print("\nTop Hallmark terms:")
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
        ].head(15).to_string(index=False)
    )

    print("\nWrote:")
    for path in [
        all_out,
        sig_out,
        summary_out,
        outdir / "hallmark_query_symbols.txt",
        outdir / "hallmark_background_symbols.txt",
        outdir / "hallmark_library_source.txt",
        plot_out,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

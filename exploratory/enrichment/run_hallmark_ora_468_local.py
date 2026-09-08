#!/usr/bin/env python3
"""
run_hallmark_ora_468_local.py
=============================

Hallmark over-representation analysis for the 468 genes significant in
covariate-adjusted Regular AREA and Weighted AREA but not DESeq2.

This version reads a LOCAL Enrichr Hallmark library file rather than
downloading it from Python, avoiding local Python SSL certificate issues.

Query:
    Regular AREA significant AND Weighted AREA significant AND
    DESeq2 non-significant

Background:
    Unique annotated gene symbols in the locked 32,994-gene common
    cross-method comparison universe.

Statistics:
    One-sided Fisher exact test ("greater")
    Benjamini-Hochberg correction across all Hallmark terms tested.
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


EXPECTED_HALLMARK_TERMS = 50


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
        "--library-file",
        default="data/gene_sets/MSigDB_Hallmark_2020.txt",
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
    s = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    bad = {
        "",
        "nan",
        "none",
        "na",
        "n/a",
        "<na>",
    }

    s = s[
        ~s.str.lower().isin(bad)
    ]

    return set(s)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)

    order = np.argsort(
        p,
        kind="mergesort",
    )

    ranked = p[order]

    q = (
        ranked
        * len(p)
        / np.arange(
            1,
            len(p) + 1,
            dtype=float,
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.minimum(
        q,
        1.0,
    )

    out = np.empty(
        len(p),
        dtype=float,
    )

    out[order] = q

    return out


def load_hallmark_library(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Hallmark library file not found: {path}"
        )

    raw_text = path.read_text()

    pathways = {}

    for line_number, raw_line in enumerate(
        raw_text.splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        parts = raw_line.rstrip("\n").split("\t")

        # Enrichr text format:
        # term <TAB> description(empty) <TAB> gene1 <TAB> gene2 ...
        if len(parts) < 3:
            raise ValueError(
                f"Malformed Hallmark line {line_number}: "
                f"expected at least 3 tab-delimited fields."
            )

        term = parts[0].strip()

        genes = {
            gene.strip()
            for gene in parts[2:]
            if gene.strip()
        }

        if not term:
            raise ValueError(
                f"Empty pathway name on line {line_number}."
            )

        if not genes:
            raise ValueError(
                f"No genes parsed for pathway '{term}'."
            )

        if term in pathways:
            raise ValueError(
                f"Duplicate Hallmark pathway: {term}"
            )

        pathways[term] = genes

    if len(pathways) != EXPECTED_HALLMARK_TERMS:
        raise ValueError(
            f"Parsed {len(pathways)} Hallmark pathways, expected "
            f"{EXPECTED_HALLMARK_TERMS}. The library file may be malformed."
        )

    # Spot-check separators in the downloaded Enrichr library. These catch
    # accidental gene concatenation such as CXCL1IER3 or NSDHLLSS.
    spot_checks = {
        "TNF-alpha Signaling via NF-kB": {
            "CXCL1",
            "IER3",
            "ZFP36",
            "ICAM1",
        },
        "Hypoxia": {
            "ALDOC",
            "GPI",
            "VEGFA",
            "BNIP3L",
        },
        "Cholesterol Homeostasis": {
            "NSDHL",
            "LSS",
            "HMGCR",
        },
    }

    for term, expected_genes in spot_checks.items():
        if term not in pathways:
            raise ValueError(
                f"Expected Hallmark pathway missing: {term}"
            )

        missing = (
            expected_genes
            - pathways[term]
        )

        if missing:
            raise ValueError(
                f"Hallmark file failed delimiter validation for '{term}'. "
                f"Missing expected genes: {sorted(missing)}. "
                "Do not run ORA with this file."
            )

    return pathways


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.membership
    )

    required = [
        "gene_id",
        "gene_symbol",
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Membership table is missing columns: {missing}"
        )

    if len(df) != args.expected_universe_n:
        raise ValueError(
            f"Expected {args.expected_universe_n:,} rows in the "
            f"comparison universe, found {len(df):,}."
        )

    target = df.loc[
        (~df["DESeq2_significant"].astype(bool))
        & df["Regular_AREA_significant"].astype(bool)
        & df["Weighted_AREA_significant"].astype(bool)
    ].copy()

    if len(target) != args.expected_target_n:
        raise ValueError(
            f"Expected {args.expected_target_n} target genes, "
            f"found {len(target)}."
        )

    query_symbols = clean_symbols(
        target["gene_symbol"]
    )

    background_symbols = clean_symbols(
        df["gene_symbol"]
    )

    query_symbols &= background_symbols

    if not query_symbols:
        raise RuntimeError(
            "No usable query symbols."
        )

    print("=" * 80)
    print("HALLMARK ORA — 468-GENE REGULAR + WEIGHTED / NOT DESEQ2 SET")
    print("=" * 80)

    print(
        f"Ensembl-level comparison universe: "
        f"{len(df):,}"
    )

    print(
        f"Ensembl-level target set:          "
        f"{len(target):,}"
    )

    print(
        f"Annotated query symbols:           "
        f"{len(query_symbols):,}"
    )

    print(
        f"Annotated background symbols:      "
        f"{len(background_symbols):,}"
    )

    pathways = load_hallmark_library(
        args.library_file
    )

    print(
        f"Hallmark pathways parsed:           "
        f"{len(pathways):,}"
    )

    print(
        "Hallmark delimiter spot-checks:     PASSED"
    )

    query_file = (
        outdir
        / "hallmark_query_symbols.txt"
    )

    background_file = (
        outdir
        / "hallmark_background_symbols.txt"
    )

    query_file.write_text(
        "\n".join(
            sorted(query_symbols)
        )
        + "\n"
    )

    background_file.write_text(
        "\n".join(
            sorted(background_symbols)
        )
        + "\n"
    )

    source_file = (
        outdir
        / "hallmark_library_source.txt"
    )

    source_file.write_text(
        "Library: MSigDB_Hallmark_2020\n"
        f"Local file: {args.library_file}\n"
        "Source endpoint: Enrichr geneSetLibrary text endpoint\n"
        "Test: one-sided Fisher exact test (greater)\n"
        "Multiple testing: Benjamini-Hochberg across Hallmark terms\n"
        "Background: unique annotated gene symbols in the locked "
        "32,994-gene ROSMAP cross-method comparison universe\n"
    )

    N = len(
        background_symbols
    )

    n = len(
        query_symbols
    )

    rows = []

    for term, genes in pathways.items():
        pathway_in_background = (
            genes
            & background_symbols
        )

        K = len(
            pathway_in_background
        )

        if K == 0:
            continue

        overlap_genes = (
            query_symbols
            & pathway_in_background
        )

        k = len(
            overlap_genes
        )

        a = k
        b = n - k
        c = K - k
        d = N - K - n + k

        if min(
            a,
            b,
            c,
            d,
        ) < 0:
            raise RuntimeError(
                f"Invalid contingency table for {term}: "
                f"{a}, {b}, {c}, {d}"
            )

        odds_ratio, pvalue = fisher_exact(
            [
                [a, b],
                [c, d],
            ],
            alternative="greater",
        )

        query_fraction = (
            k / n
        )

        background_fraction = (
            K / N
        )

        fold_enrichment = (
            query_fraction
            / background_fraction
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
                "overlap_genes": ";".join(
                    sorted(
                        overlap_genes
                    )
                ),
            }
        )

    res = pd.DataFrame(
        rows
    )

    if res.empty:
        raise RuntimeError(
            "No Hallmark pathways overlapped the ROSMAP background."
        )

    res[
        "padj_BH"
    ] = bh_adjust(
        res["pvalue"].to_numpy()
    )

    res[
        "fdr_significant"
    ] = (
        res["padj_BH"]
        < args.fdr
    )

    res = res.sort_values(
        [
            "padj_BH",
            "pvalue",
            "fold_enrichment",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    all_out = (
        outdir
        / "hallmark_ora_all_results.csv"
    )

    sig_out = (
        outdir
        / "hallmark_ora_FDR0.05.csv"
    )

    summary_out = (
        outdir
        / "hallmark_ora_summary.csv"
    )

    res.to_csv(
        all_out,
        index=False,
    )

    sig = res.loc[
        res[
            "fdr_significant"
        ]
    ].copy()

    sig.to_csv(
        sig_out,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "common_ensembl_universe_n":
                    len(df),
                "target_ensembl_n":
                    len(target),
                "annotated_query_symbols_n":
                    len(query_symbols),
                "annotated_background_symbols_n":
                    len(background_symbols),
                "hallmark_terms_tested_n":
                    len(res),
                "fdr_threshold":
                    args.fdr,
                "fdr_significant_terms_n":
                    len(sig),
            }
        ]
    ).to_csv(
        summary_out,
        index=False,
    )

    plot_df = (
        sig
        if not sig.empty
        else res
    ).head(
        args.top_n
    ).iloc[::-1].copy()

    plot_df[
        "minus_log10_BH"
    ] = (
        -np.log10(
            plot_df[
                "padj_BH"
            ].clip(
                lower=1e-300
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        plot_df["pathway"],
        plot_df[
            "minus_log10_BH"
        ],
    )

    ax.set_xlabel(
        "-log10(BH-adjusted p-value)"
    )

    ax.set_ylabel(
        ""
    )

    title_suffix = (
        "FDR-significant terms"
        if not sig.empty
        else "top terms; none passed FDR"
    )

    ax.set_title(
        "Hallmark enrichment of Regular + Weighted AREA genes\n"
        f"not significant by DESeq2 ({title_suffix})"
    )

    fig.tight_layout()

    plot_out = (
        outdir
        / "hallmark_ora_top_terms.png"
    )

    fig.savefig(
        plot_out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print("\nRESULT")

    print(
        f"Hallmark terms tested: "
        f"{len(res):,}"
    )

    print(
        f"FDR < {args.fdr:g}: "
        f"{len(sig):,} pathways"
    )

    print(
        "\nTop Hallmark terms:"
    )

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
        .head(15)
        .to_string(
            index=False
        )
    )

    print("\nWrote:")

    for path in [
        all_out,
        sig_out,
        summary_out,
        query_file,
        background_file,
        source_file,
        plot_out,
    ]:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()

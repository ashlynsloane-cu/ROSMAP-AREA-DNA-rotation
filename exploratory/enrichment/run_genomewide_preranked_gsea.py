#!/usr/bin/env python3
"""
run_genomewide_preranked_gsea.py
================================

Genome-wide preranked GSEA comparison across:

1) DESeq2
2) Covariate-adjusted Regular AREA
3) Weighted AREA (Cognitive_stage)

Purpose
-------
Move beyond thresholded Venn slices and ask whether coordinated biological
programs are enriched across the FULL ranked gene list for each method.

Common comparison universe:
    32,994 Ensembl genes
    (the 12 persistent DESeq2 non-converged genes are excluded)

Ranking orientation:
    Positive score = higher expression with greater disease severity / AD
    Negative score = lower expression with greater disease severity / AD

    DESeq2:
        Wald statistic from AD vs NCI contrast
        positive = AD higher expression

    Regular AREA:
        - adjusted_Regular_AREA_Z
        (project convention: positive AREA Z = AD lower expression)

    Weighted AREA:
        - adjusted_AREA_Z
        (aligned so positive = higher expression with greater severity)

Gene-set libraries:
    Hallmark 2020
    GO Biological Process 2025
    Reactome Pathways 2024

Gene symbols:
    GSEA is performed at the gene-symbol level.
    Duplicate symbols are collapsed WITHIN EACH METHOD by keeping the
    Ensembl entry with the largest absolute ranking statistic.

Implementation:
    gseapy.prerank()

Outputs
-------
results/genomewide_preranked_gsea/
    ranking_tables/
    Hallmark_2020/
    GO_Biological_Process_2025/
    Reactome_Pathways_2024/
    gsea_summary.csv
    <library>_method_comparison.csv

Important
---------
This is a genome-wide ranked pathway analysis. It does NOT depend on the
FDR<0.05 gene threshold used to make the Venn diagram.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


LIBRARIES = {
    "Hallmark_2020": "data/gene_sets/MSigDB_Hallmark_2020.txt",
    "GO_Biological_Process_2025": (
        "data/gene_sets/GO_Biological_Process_2025.txt"
    ),
    "Reactome_Pathways_2024": (
        "data/gene_sets/Reactome_Pathways_2024.txt"
    ),
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--deseq2-file",
        default=(
            "results/deseq2_locked_covariates/AD4_vs_NCI1/"
            "AD4_vs_NCI1_DESeq2_all_results_maxit1000.csv"
        ),
    )

    p.add_argument(
        "--regular-file",
        default=(
            "results/regular_area_covariate_adjusted/"
            "AD4_vs_NCI1/results.csv"
        ),
    )

    p.add_argument(
        "--weighted-file",
        default=(
            "results/weighted_area_full_phenotype_suite/"
            "Cognitive_stage/results.csv"
        ),
    )

    p.add_argument(
        "--membership",
        default=(
            "results/visualizations/"
            "final_locked_covariate_DESeq2_Regular_Weighted_"
            "venn_gene_membership.csv"
        ),
    )

    p.add_argument(
        "--exclude-gene-file",
        default=(
            "results/deseq2_locked_covariates/AD4_vs_NCI1/"
            "AD4_vs_NCI1_nonconverged_genes.csv"
        ),
    )

    p.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Gene-set permutations per GSEA run. Default: 1000.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260907,
    )

    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--min-size",
        type=int,
        default=10,
    )

    p.add_argument(
        "--max-size",
        type=int,
        default=500,
    )

    p.add_argument(
        "--fdr",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--expected-common-genes",
        type=int,
        default=32994,
    )

    p.add_argument(
        "--outdir",
        default="results/genomewide_preranked_gsea",
    )

    return p.parse_args()


def require_gseapy():
    try:
        import gseapy as gp
        return gp
    except ImportError:
        print(
            "\nERROR: gseapy is not installed.\n\n"
            "Install it with:\n\n"
            "    python3 -m pip install gseapy\n\n"
            "Then rerun this script.\n",
            file=sys.stderr,
        )
        raise


def clean_symbol_series(series):
    s = (
        series
        .astype("string")
        .str.strip()
    )

    bad = (
        s.isna()
        | (s == "")
        | s.str.lower().isin(
            ["nan", "none", "na", "n/a", "<na>"]
        )
    )

    return s.mask(bad)


def load_enrichr_library(path, label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} library not found: {path}"
        )

    pathways = {}

    for line_no, raw in enumerate(
        path.read_text().splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue

        parts = raw.rstrip("\n").split("\t")

        if len(parts) < 3:
            raise ValueError(
                f"{label}: malformed line {line_no}."
            )

        term = parts[0].strip()

        genes = {
            g.strip()
            for g in parts[2:]
            if g.strip()
        }

        if not term or not genes:
            raise ValueError(
                f"{label}: empty term/gene set on line {line_no}."
            )

        pathways[term] = genes

    if len(pathways) < 20:
        raise ValueError(
            f"{label}: parsed only {len(pathways)} terms."
        )

    return pathways


def collapse_to_symbols(df, score_col, method):
    """
    Keep one Ensembl entry per symbol: the entry with largest |score|.
    """
    x = df[
        [
            "ensembl_id_version",
            "gene_symbol",
            score_col,
        ]
    ].copy()

    x[score_col] = pd.to_numeric(
        x[score_col],
        errors="coerce",
    )

    x["gene_symbol"] = clean_symbol_series(
        x["gene_symbol"]
    )

    x = x.loc[
        x["gene_symbol"].notna()
        & x[score_col].notna()
        & np.isfinite(x[score_col])
    ].copy()

    x["abs_score"] = x[score_col].abs()

    x = (
        x.sort_values(
            [
                "gene_symbol",
                "abs_score",
                "ensembl_id_version",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            "gene_symbol",
            keep="first",
        )
        .copy()
    )

    x = x.rename(
        columns={
            score_col: "score",
        }
    )

    x["method"] = method

    return x[
        [
            "gene_symbol",
            "score",
            "ensembl_id_version",
            "method",
        ]
    ]


def standardize_gseapy_results(res2d):
    """
    Normalize gseapy column naming across versions.
    """
    x = res2d.copy()

    # Term is sometimes index, sometimes a column.
    if "Term" not in x.columns:
        x = x.reset_index()

    rename_candidates = {
        "Name": "Name",
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

    usable = {
        old: new
        for old, new in rename_candidates.items()
        if old in x.columns
    }

    x = x.rename(
        columns=usable
    )

    if "pathway" not in x.columns:
        raise ValueError(
            "Could not identify pathway column in gseapy result. "
            f"Columns: {x.columns.tolist()}"
        )

    for col in [
        "ES",
        "NES",
        "NOM_pvalue",
        "FDR_qvalue",
        "FWER_pvalue",
    ]:
        if col in x.columns:
            x[col] = pd.to_numeric(
                x[col],
                errors="coerce",
            )

    return x


def main():
    args = parse_args()
    gp = require_gseapy()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rank_dir = outdir / "ranking_tables"
    rank_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Locked common universe + annotation
    # ------------------------------------------------------------
    membership = pd.read_csv(
        args.membership
    )

    required_membership = [
        "gene_id",
        "gene_symbol",
    ]

    missing = [
        c
        for c in required_membership
        if c not in membership.columns
    ]

    if missing:
        raise ValueError(
            f"Membership table missing columns: {missing}"
        )

    membership["gene_id"] = (
        membership["gene_id"]
        .astype(str)
        .str.strip()
    )

    if len(membership) != args.expected_common_genes:
        raise ValueError(
            f"Membership universe has {len(membership):,} genes; "
            f"expected {args.expected_common_genes:,}."
        )

    common_ids = set(
        membership["gene_id"]
    )

    annotation = (
        membership[
            [
                "gene_id",
                "gene_symbol",
            ]
        ]
        .rename(
            columns={
                "gene_id": "ensembl_id_version",
            }
        )
        .copy()
    )

    # ------------------------------------------------------------
    # Load method results
    # ------------------------------------------------------------
    deseq = pd.read_csv(
        args.deseq2_file
    )

    regular = pd.read_csv(
        args.regular_file
    )

    weighted = pd.read_csv(
        args.weighted_file
    )

    # DESeq2
    if "ensembl_id_version" not in deseq.columns:
        raise ValueError(
            "DESeq2 file lacks ensembl_id_version."
        )

    if "stat" not in deseq.columns:
        raise ValueError(
            "DESeq2 file lacks Wald statistic column 'stat'."
        )

    deseq["ensembl_id_version"] = (
        deseq["ensembl_id_version"]
        .astype(str)
        .str.strip()
    )

    deseq = deseq.loc[
        deseq["ensembl_id_version"].isin(
            common_ids
        )
    ].copy()

    if "gene_symbol" not in deseq.columns:
        deseq = deseq.merge(
            annotation,
            on="ensembl_id_version",
            how="left",
            validate="one_to_one",
        )

    # Regular AREA
    if not {
        "gene_id",
        "adjusted_Regular_AREA_Z",
    }.issubset(
        regular.columns
    ):
        raise ValueError(
            "Regular AREA file lacks gene_id and/or "
            "adjusted_Regular_AREA_Z."
        )

    regular["gene_id"] = (
        regular["gene_id"]
        .astype(str)
        .str.strip()
    )

    regular = regular.loc[
        regular["gene_id"].isin(
            common_ids
        )
    ].copy()

    regular = regular.rename(
        columns={
            "gene_id": "ensembl_id_version",
        }
    )

    regular = regular.merge(
        annotation,
        on="ensembl_id_version",
        how="left",
        validate="one_to_one",
    )

    # Weighted AREA
    if not {
        "gene_id",
        "adjusted_AREA_Z",
    }.issubset(
        weighted.columns
    ):
        raise ValueError(
            "Weighted AREA file lacks gene_id and/or adjusted_AREA_Z."
        )

    weighted["gene_id"] = (
        weighted["gene_id"]
        .astype(str)
        .str.strip()
    )

    weighted = weighted.loc[
        weighted["gene_id"].isin(
            common_ids
        )
    ].copy()

    weighted = weighted.rename(
        columns={
            "gene_id": "ensembl_id_version",
        }
    )

    weighted = weighted.merge(
        annotation,
        on="ensembl_id_version",
        how="left",
        validate="one_to_one",
    )

    # Exact universe checks before symbol collapse.
    for label, frame in [
        ("DESeq2", deseq),
        ("Regular AREA", regular),
        ("Weighted AREA", weighted),
    ]:
        ids = set(
            frame["ensembl_id_version"]
            .astype(str)
        )

        if ids != common_ids:
            raise ValueError(
                f"{label} does not match the exact "
                "32,994-gene common universe."
            )

    # ------------------------------------------------------------
    # Align rank orientation:
    # positive = expression increases with disease severity / AD.
    # ------------------------------------------------------------
    deseq["gsea_score"] = pd.to_numeric(
        deseq["stat"],
        errors="coerce",
    )

    regular["gsea_score"] = -pd.to_numeric(
        regular["adjusted_Regular_AREA_Z"],
        errors="coerce",
    )

    weighted["gsea_score"] = -pd.to_numeric(
        weighted["adjusted_AREA_Z"],
        errors="coerce",
    )

    ranks = {
        "DESeq2": collapse_to_symbols(
            deseq,
            "gsea_score",
            "DESeq2",
        ),
        "Regular_AREA": collapse_to_symbols(
            regular,
            "gsea_score",
            "Regular_AREA",
        ),
        "Weighted_AREA": collapse_to_symbols(
            weighted,
            "gsea_score",
            "Weighted_AREA",
        ),
    }

    print("=" * 92)
    print("GENOME-WIDE PRERANKED GSEA")
    print("=" * 92)
    print(
        f"Common Ensembl universe: "
        f"{len(common_ids):,}"
    )
    print(
        "Rank orientation: positive = higher expression "
        "with greater disease severity / AD"
    )

    for method, rank_df in ranks.items():
        print(
            f"  {method:<15} "
            f"{len(rank_df):,} unique annotated symbols"
        )

        rank_df.to_csv(
            rank_dir / f"{method}_ranking.csv",
            index=False,
        )

    # ------------------------------------------------------------
    # Libraries
    # ------------------------------------------------------------
    libraries = {
        name: load_enrichr_library(
            path,
            name,
        )
        for name, path in LIBRARIES.items()
    }

    summary_rows = []

    # library -> method -> standardized result
    all_results = {}

    for library_name, gene_sets in libraries.items():
        print(
            f"\n{'-' * 92}\n"
            f"{library_name}: "
            f"{len(gene_sets):,} raw gene sets"
        )

        library_dir = (
            outdir
            / library_name
        )

        library_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_results[
            library_name
        ] = {}

        for method, rank_df in ranks.items():
            print(
                f"\n  Running {method}..."
            )

            # gseapy wants [gene, score], sorted descending.
            rnk = (
                rank_df[
                    [
                        "gene_symbol",
                        "score",
                    ]
                ]
                .sort_values(
                    "score",
                    ascending=False,
                    kind="mergesort",
                )
                .copy()
            )

            method_dir = (
                library_dir
                / method
            )

            method_dir.mkdir(
                parents=True,
                exist_ok=True,
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

            res = standardize_gseapy_results(
                pre.res2d
            )

            res["method"] = method
            res["library"] = library_name

            if "FDR_qvalue" not in res.columns:
                raise ValueError(
                    "gseapy result lacks FDR q-value column. "
                    f"Columns: {res.columns.tolist()}"
                )

            res[
                "fdr_significant"
            ] = (
                res["FDR_qvalue"]
                < args.fdr
            )

            all_results[
                library_name
            ][
                method
            ] = res

            res.to_csv(
                method_dir
                / "gsea_all_results.csv",
                index=False,
            )

            sig = res.loc[
                res[
                    "fdr_significant"
                ]
            ].copy()

            sig.to_csv(
                method_dir
                / "gsea_FDR0.05.csv",
                index=False,
            )

            n_pos = int(
                (
                    sig[
                        "NES"
                    ] > 0
                ).sum()
            )

            n_neg = int(
                (
                    sig[
                        "NES"
                    ] < 0
                ).sum()
            )

            summary_rows.append(
                {
                    "library":
                        library_name,
                    "method":
                        method,
                    "ranking_symbols_n":
                        len(rank_df),
                    "gene_sets_tested_n":
                        len(res),
                    "permutations":
                        args.permutations,
                    "fdr_threshold":
                        args.fdr,
                    "fdr_significant_n":
                        len(sig),
                    "fdr_significant_positive_NES_n":
                        n_pos,
                    "fdr_significant_negative_NES_n":
                        n_neg,
                }
            )

            print(
                f"    tested={len(res):,} | "
                f"FDR<0.05={len(sig):,} "
                f"(positive NES={n_pos:,}, "
                f"negative NES={n_neg:,})"
            )

            if not res.empty:
                show_cols = [
                    c
                    for c in [
                        "pathway",
                        "NES",
                        "NOM_pvalue",
                        "FDR_qvalue",
                        "leading_edge_genes",
                    ]
                    if c in res.columns
                ]

                print(
                    res.sort_values(
                        "FDR_qvalue",
                        ascending=True,
                    )[
                        show_cols
                    ]
                    .head(8)
                    .to_string(
                        index=False
                    )
                    .replace(
                        "\n",
                        "\n    ",
                    )
                )

        # --------------------------------------------------------
        # Direct method comparison for this library
        # --------------------------------------------------------
        comparison = None

        for method in [
            "DESeq2",
            "Regular_AREA",
            "Weighted_AREA",
        ]:
            res = all_results[
                library_name
            ][
                method
            ].copy()

            keep_cols = [
                c
                for c in [
                    "pathway",
                    "ES",
                    "NES",
                    "NOM_pvalue",
                    "FDR_qvalue",
                    "fdr_significant",
                    "leading_edge_genes",
                ]
                if c in res.columns
            ]

            keep = res[
                keep_cols
            ].copy()

            rename = {
                c: f"{method}_{c}"
                for c in keep.columns
                if c != "pathway"
            }

            keep = keep.rename(
                columns=rename
            )

            if comparison is None:
                comparison = keep

            else:
                comparison = comparison.merge(
                    keep,
                    on="pathway",
                    how="outer",
                    validate="one_to_one",
                )

        comparison[
            "Weighted_specific_vs_DESeq2"
        ] = (
            comparison[
                "Weighted_AREA_fdr_significant"
            ].fillna(False)
            & ~comparison[
                "DESeq2_fdr_significant"
            ].fillna(False)
        )

        comparison[
            "Regular_specific_vs_DESeq2"
        ] = (
            comparison[
                "Regular_AREA_fdr_significant"
            ].fillna(False)
            & ~comparison[
                "DESeq2_fdr_significant"
            ].fillna(False)
        )

        comparison[
            "AREA_both_specific_vs_DESeq2"
        ] = (
            comparison[
                "Weighted_AREA_fdr_significant"
            ].fillna(False)
            & comparison[
                "Regular_AREA_fdr_significant"
            ].fillna(False)
            & ~comparison[
                "DESeq2_fdr_significant"
            ].fillna(False)
        )

        comparison[
            "Weighted_and_DESeq2"
        ] = (
            comparison[
                "Weighted_AREA_fdr_significant"
            ].fillna(False)
            & comparison[
                "DESeq2_fdr_significant"
            ].fillna(False)
        )

        comparison[
            "all_three_methods"
        ] = (
            comparison[
                "Weighted_AREA_fdr_significant"
            ].fillna(False)
            & comparison[
                "Regular_AREA_fdr_significant"
            ].fillna(False)
            & comparison[
                "DESeq2_fdr_significant"
            ].fillna(False)
        )

        # Direction concordance where both methods have NES.
        comparison[
            "Weighted_vs_DESeq2_same_direction"
        ] = (
            np.sign(
                comparison[
                    "Weighted_AREA_NES"
                ]
            )
            == np.sign(
                comparison[
                    "DESeq2_NES"
                ]
            )
        )

        comparison = comparison.sort_values(
            [
                "Weighted_specific_vs_DESeq2",
                "AREA_both_specific_vs_DESeq2",
                "Weighted_AREA_FDR_qvalue",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="mergesort",
        )

        comparison.to_csv(
            outdir
            / f"{library_name}_method_comparison.csv",
            index=False,
        )

        weighted_specific = comparison.loc[
            comparison[
                "Weighted_specific_vs_DESeq2"
            ]
        ].copy()

        weighted_specific.to_csv(
            outdir
            / f"{library_name}_Weighted_specific_vs_DESeq2.csv",
            index=False,
        )

        both_area_specific = comparison.loc[
            comparison[
                "AREA_both_specific_vs_DESeq2"
            ]
        ].copy()

        both_area_specific.to_csv(
            outdir
            / f"{library_name}_both_AREA_specific_vs_DESeq2.csv",
            index=False,
        )

        print(
            f"\n  Weighted-specific vs DESeq2: "
            f"{len(weighted_specific):,} pathways"
        )

        print(
            f"  Both AREA methods significant, "
            f"DESeq2 not: "
            f"{len(both_area_specific):,} pathways"
        )

    # ------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------
    summary = pd.DataFrame(
        summary_rows
    )

    summary_out = (
        outdir
        / "gsea_summary.csv"
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    (outdir / "analysis_design.txt").write_text(
        "Genome-wide preranked GSEA comparison\n\n"
        "Common Ensembl universe: 32,994 genes\n"
        "12 persistent DESeq2 non-converged genes excluded before comparison\n\n"
        "Ranking orientation:\n"
        "  positive = expression higher with greater disease severity / AD\n"
        "  DESeq2 = Wald statistic\n"
        "  Regular AREA = -adjusted_Regular_AREA_Z\n"
        "  Weighted AREA = -adjusted_AREA_Z\n\n"
        "Gene-symbol handling:\n"
        "  duplicate symbols collapsed per method by largest absolute score\n\n"
        "GSEA implementation:\n"
        "  gseapy.prerank\n"
        f"  permutations = {args.permutations}\n"
        f"  weight = 1\n"
        f"  min gene-set size = {args.min_size}\n"
        f"  max gene-set size = {args.max_size}\n"
        f"  seed = {args.seed}\n"
        f"  FDR threshold = {args.fdr}\n\n"
        "Interpretation:\n"
        "  positive NES = pathway genes concentrated among genes whose "
        "expression increases with disease severity / AD\n"
        "  negative NES = pathway genes concentrated among genes whose "
        "expression decreases with disease severity / AD\n"
    )

    print("\n" + "=" * 92)
    print("DONE")
    print("=" * 92)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nOutputs: {outdir}"
    )


if __name__ == "__main__":
    main()

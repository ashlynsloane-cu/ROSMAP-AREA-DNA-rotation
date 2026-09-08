#!/usr/bin/env python3
"""
run_weighted_area_full_phenotype_gsea.py
========================================

Run genome-wide preranked GSEA across the FULL 9-phenotype Weighted AREA suite.

Weighted AREA phenotypes:
    1. Cognitive_stage
    2. cogn_global_impairment
    3. mmse_impairment
    4. CERAD_equal
    5. CERAD_amyloid_calibrated
    6. amyloid_continuous
    7. Braak_equal
    8. Braak_tangle_calibrated
    9. tangle_continuous

Reference rankings:
    - DESeq2 AD4 vs NCI1
    - Adjusted Regular AREA AD4 vs NCI1

Purpose
-------
Ask which biological programs are:
    - shared across many disease/pathology phenotypes,
    - specific to cognition,
    - specific to amyloid,
    - specific to tau/Braak,
    - stronger for continuous pathology than ordinal pathology,
    - significant in Weighted AREA but not in DESeq2.

Ranking orientation
-------------------
Positive score = expression increases with disease severity / pathology burden.
Negative score = expression decreases with disease severity / pathology burden.

    DESeq2:
        Wald statistic (AD vs NCI), positive = AD higher

    Adjusted Regular AREA:
        - adjusted_Regular_AREA_Z

    Weighted AREA:
        - adjusted_AREA_Z

All analyses use the locked 32,994-gene common comparison universe from the
final Venn membership table. GSEA is performed at the gene-symbol level.
Duplicate gene symbols are collapsed within each ranking by keeping the
Ensembl entry with largest absolute score.

Libraries
---------
- Hallmark 2020
- GO Biological Process 2025
- Reactome Pathways 2024

Implementation
--------------
gseapy.prerank()
Default 1,000 permutations, seed fixed for reproducibility.

Main outputs
------------
results/weighted_area_full_phenotype_gsea/

    gsea_summary.csv
    phenotype_metadata.csv

    ranking_tables/
        DESeq2.csv
        Regular_AREA.csv
        Weighted_<phenotype>.csv

    Hallmark_2020/
    GO_Biological_Process_2025/
    Reactome_Pathways_2024/

    <library>_weighted_phenotype_matrix.csv
        one row per pathway; NES/FDR for all 9 Weighted phenotypes

    <library>_weighted_significance_matrix.csv
        pathway x phenotype boolean significance matrix

    <library>_pathway_recurrence.csv
        how many Weighted phenotypes significantly enrich each pathway

    <library>_cognition_vs_amyloid_vs_tau_summary.csv
        compact family-level recurrence summary

    <library>_weighted_specific_vs_DESeq2.csv
        pathways significant in >=1 Weighted phenotype but not DESeq2

    <library>_weighted_shared_with_DESeq2.csv
        pathways significant in >=1 Weighted phenotype and DESeq2

    <library>_continuous_vs_ordinal_pathology.csv
        direct comparison of continuous vs ordinal pathology encodings
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PHENOTYPES = [
    "Cognitive_stage",
    "cogn_global_impairment",
    "mmse_impairment",
    "CERAD_equal",
    "CERAD_amyloid_calibrated",
    "amyloid_continuous",
    "Braak_equal",
    "Braak_tangle_calibrated",
    "tangle_continuous",
]

PHENOTYPE_FAMILY = {
    "Cognitive_stage": "cognition",
    "cogn_global_impairment": "cognition",
    "mmse_impairment": "cognition",
    "CERAD_equal": "amyloid",
    "CERAD_amyloid_calibrated": "amyloid",
    "amyloid_continuous": "amyloid",
    "Braak_equal": "tau",
    "Braak_tangle_calibrated": "tau",
    "tangle_continuous": "tau",
}

PHENOTYPE_ENCODING = {
    "Cognitive_stage": "ordinal",
    "cogn_global_impairment": "continuous",
    "mmse_impairment": "continuous",
    "CERAD_equal": "ordinal_equal",
    "CERAD_amyloid_calibrated": "ordinal_calibrated",
    "amyloid_continuous": "continuous",
    "Braak_equal": "ordinal_equal",
    "Braak_tangle_calibrated": "ordinal_calibrated",
    "tangle_continuous": "continuous",
}

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
        "--weighted-root",
        default="results/weighted_area_full_phenotype_suite",
    )

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
        "--membership",
        default=(
            "results/visualizations/"
            "final_locked_covariate_DESeq2_Regular_Weighted_"
            "venn_gene_membership.csv"
        ),
    )

    p.add_argument(
        "--permutations",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260908,
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
        default="results/weighted_area_full_phenotype_gsea",
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
            "    python3 -m pip install gseapy\n",
            file=sys.stderr,
        )
        raise


def clean_symbol_series(series):
    s = series.astype("string").str.strip()

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
                f"{label}: invalid term/gene set on line {line_no}."
            )

        pathways[term] = genes

    if len(pathways) < 20:
        raise ValueError(
            f"{label}: parsed only {len(pathways)} terms."
        )

    return pathways


def collapse_to_symbols(df, id_col, symbol_col, score_col, method):
    x = df[
        [
            id_col,
            symbol_col,
            score_col,
        ]
    ].copy()

    x = x.rename(
        columns={
            id_col: "ensembl_id_version",
            symbol_col: "gene_symbol",
            score_col: "score",
        }
    )

    x["score"] = pd.to_numeric(
        x["score"],
        errors="coerce",
    )

    x["gene_symbol"] = clean_symbol_series(
        x["gene_symbol"]
    )

    x = x.loc[
        x["gene_symbol"].notna()
        & x["score"].notna()
        & np.isfinite(x["score"])
    ].copy()

    x["abs_score"] = x["score"].abs()

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
    x = res2d.copy()

    if "Term" not in x.columns:
        x = x.reset_index()

    rename = {
        "Term": "pathway",
        "ES": "ES",
        "NES": "NES",
        "NOM p-val": "NOM_pvalue",
        "FDR q-val": "FDR_qvalue",
        "FWER p-val": "FWER_pvalue",
        "Tag %": "Tag_percent",
        "Gene %": "Gene_percent",
        "Lead_genes": "leading_edge_genes",
    }

    x = x.rename(
        columns={
            k: v
            for k, v in rename.items()
            if k in x.columns
        }
    )

    if "pathway" not in x.columns:
        raise ValueError(
            "Could not identify pathway column in gseapy output. "
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


def run_prerank(
    gp,
    rank_df,
    gene_sets,
    args,
):
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

    return standardize_gseapy_results(
        pre.res2d
    )


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
    # Common universe + annotation
    # ------------------------------------------------------------
    membership = pd.read_csv(
        args.membership
    )

    if not {
        "gene_id",
        "gene_symbol",
    }.issubset(
        membership.columns
    ):
        raise ValueError(
            "Membership table must contain gene_id and gene_symbol."
        )

    membership["gene_id"] = (
        membership["gene_id"]
        .astype(str)
        .str.strip()
    )

    if len(membership) != args.expected_common_genes:
        raise ValueError(
            f"Expected {args.expected_common_genes:,} common genes, "
            f"found {len(membership):,}."
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
    # Reference rankings: DESeq2 + adjusted Regular AREA
    # ------------------------------------------------------------
    deseq = pd.read_csv(
        args.deseq2_file
    )

    if not {
        "ensembl_id_version",
        "stat",
    }.issubset(
        deseq.columns
    ):
        raise ValueError(
            "DESeq2 file must contain ensembl_id_version and stat."
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

    deseq["gsea_score"] = pd.to_numeric(
        deseq["stat"],
        errors="coerce",
    )

    regular = pd.read_csv(
        args.regular_file
    )

    if not {
        "gene_id",
        "adjusted_Regular_AREA_Z",
    }.issubset(
        regular.columns
    ):
        raise ValueError(
            "Regular AREA file must contain gene_id and "
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

    regular["gsea_score"] = -pd.to_numeric(
        regular["adjusted_Regular_AREA_Z"],
        errors="coerce",
    )

    rankings = {
        "DESeq2": collapse_to_symbols(
            deseq,
            "ensembl_id_version",
            "gene_symbol",
            "gsea_score",
            "DESeq2",
        ),
        "Regular_AREA": collapse_to_symbols(
            regular,
            "ensembl_id_version",
            "gene_symbol",
            "gsea_score",
            "Regular_AREA",
        ),
    }

    # ------------------------------------------------------------
    # Weighted AREA full phenotype suite
    # ------------------------------------------------------------
    weighted_metadata_rows = []

    for phenotype in PHENOTYPES:
        path = (
            Path(args.weighted_root)
            / phenotype
            / "results.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Weighted AREA result not found for {phenotype}: {path}"
            )

        w = pd.read_csv(
            path
        )

        required = {
            "gene_id",
            "adjusted_AREA_Z",
            "n_samples",
        }

        if not required.issubset(
            w.columns
        ):
            raise ValueError(
                f"{phenotype} missing required columns: "
                f"{sorted(required - set(w.columns))}"
            )

        w["gene_id"] = (
            w["gene_id"]
            .astype(str)
            .str.strip()
        )

        w = w.loc[
            w["gene_id"].isin(
                common_ids
            )
        ].copy()

        if set(w["gene_id"]) != common_ids:
            raise ValueError(
                f"{phenotype} does not match exact common gene universe."
            )

        w = w.rename(
            columns={
                "gene_id": "ensembl_id_version",
            }
        )

        w = w.merge(
            annotation,
            on="ensembl_id_version",
            how="left",
            validate="one_to_one",
        )

        w["gsea_score"] = -pd.to_numeric(
            w["adjusted_AREA_Z"],
            errors="coerce",
        )

        method_name = (
            "Weighted_"
            + phenotype
        )

        rankings[
            method_name
        ] = collapse_to_symbols(
            w,
            "ensembl_id_version",
            "gene_symbol",
            "gsea_score",
            method_name,
        )

        ns = (
            pd.to_numeric(
                w["n_samples"],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        n_samples = (
            int(ns[0])
            if len(ns) == 1
            else np.nan
        )

        weighted_metadata_rows.append(
            {
                "phenotype": phenotype,
                "family": PHENOTYPE_FAMILY[
                    phenotype
                ],
                "encoding": PHENOTYPE_ENCODING[
                    phenotype
                ],
                "n_samples": n_samples,
                "ranking_symbols_n": len(
                    rankings[
                        method_name
                    ]
                ),
                "results_file": str(path),
            }
        )

    phenotype_metadata = pd.DataFrame(
        weighted_metadata_rows
    )

    phenotype_metadata.to_csv(
        outdir / "phenotype_metadata.csv",
        index=False,
    )

    for name, rank_df in rankings.items():
        rank_df.to_csv(
            rank_dir / f"{name}.csv",
            index=False,
        )

    print("=" * 100)
    print("WEIGHTED AREA FULL-PHENOTYPE GENOME-WIDE PRERANKED GSEA")
    print("=" * 100)
    print(
        f"Common Ensembl universe: "
        f"{len(common_ids):,}"
    )
    print(
        "Rank orientation: positive = higher expression with "
        "greater disease severity/pathology"
    )
    print("\nWeighted phenotypes:")
    print(
        phenotype_metadata[
            [
                "phenotype",
                "family",
                "encoding",
                "n_samples",
                "ranking_symbols_n",
            ]
        ].to_string(
            index=False
        )
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
    all_results = {}

    for library_name, gene_sets in libraries.items():
        print(
            f"\n{'-' * 100}\n"
            f"{library_name}: {len(gene_sets):,} raw gene sets"
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

        # Run both references + all 9 Weighted phenotypes.
        for method_name, rank_df in rankings.items():
            print(
                f"\nRunning {method_name}..."
            )

            res = run_prerank(
                gp,
                rank_df,
                gene_sets,
                args,
            )

            res["method"] = method_name
            res["library"] = library_name

            if "FDR_qvalue" not in res.columns:
                raise ValueError(
                    f"{method_name}/{library_name}: "
                    "gseapy output missing FDR q-value."
                )

            res[
                "fdr_significant"
            ] = (
                res[
                    "FDR_qvalue"
                ]
                < args.fdr
            )

            all_results[
                library_name
            ][
                method_name
            ] = res

            method_dir = (
                library_dir
                / method_name
            )

            method_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

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

            summary_rows.append(
                {
                    "library":
                        library_name,
                    "method":
                        method_name,
                    "phenotype":
                        (
                            method_name.replace(
                                "Weighted_",
                                "",
                            )
                            if method_name.startswith(
                                "Weighted_"
                            )
                            else np.nan
                        ),
                    "family":
                        (
                            PHENOTYPE_FAMILY[
                                method_name.replace(
                                    "Weighted_",
                                    "",
                                )
                            ]
                            if method_name.startswith(
                                "Weighted_"
                            )
                            else "reference"
                        ),
                    "ranking_symbols_n":
                        len(rank_df),
                    "gene_sets_tested_n":
                        len(res),
                    "fdr_significant_n":
                        len(sig),
                    "fdr_significant_positive_NES_n":
                        int(
                            (
                                sig[
                                    "NES"
                                ] > 0
                            ).sum()
                        ),
                    "fdr_significant_negative_NES_n":
                        int(
                            (
                                sig[
                                    "NES"
                                ] < 0
                            ).sum()
                        ),
                    "permutations":
                        args.permutations,
                }
            )

            print(
                f"  tested={len(res):,} | "
                f"FDR<0.05={len(sig):,}"
            )

        # --------------------------------------------------------
        # Build Weighted phenotype matrix for this library.
        # --------------------------------------------------------
        weighted_methods = [
            "Weighted_"
            + p
            for p in PHENOTYPES
        ]

        matrix = None

        for method in weighted_methods:
            phenotype = method.replace(
                "Weighted_",
                "",
            )

            res = all_results[
                library_name
            ][
                method
            ][
                [
                    "pathway",
                    "NES",
                    "FDR_qvalue",
                    "fdr_significant",
                ]
            ].copy()

            res = res.rename(
                columns={
                    "NES":
                        f"{phenotype}_NES",
                    "FDR_qvalue":
                        f"{phenotype}_FDR",
                    "fdr_significant":
                        f"{phenotype}_sig",
                }
            )

            if matrix is None:
                matrix = res
            else:
                matrix = matrix.merge(
                    res,
                    on="pathway",
                    how="outer",
                    validate="one_to_one",
                )

        # Reference FDR/NES.
        for method in [
            "DESeq2",
            "Regular_AREA",
        ]:
            ref = all_results[
                library_name
            ][
                method
            ][
                [
                    "pathway",
                    "NES",
                    "FDR_qvalue",
                    "fdr_significant",
                ]
            ].copy()

            ref = ref.rename(
                columns={
                    "NES":
                        f"{method}_NES",
                    "FDR_qvalue":
                        f"{method}_FDR",
                    "fdr_significant":
                        f"{method}_sig",
                }
            )

            matrix = matrix.merge(
                ref,
                on="pathway",
                how="left",
                validate="one_to_one",
            )

        sig_cols = [
            f"{p}_sig"
            for p in PHENOTYPES
        ]

        for col in sig_cols:
            matrix[col] = (
                matrix[col]
                .fillna(False)
                .astype(bool)
            )

        matrix[
            "n_weighted_phenotypes_significant"
        ] = (
            matrix[
                sig_cols
            ]
            .sum(
                axis=1
            )
        )

        family_sig_cols = {}

        for family in [
            "cognition",
            "amyloid",
            "tau",
        ]:
            cols = [
                f"{p}_sig"
                for p in PHENOTYPES
                if PHENOTYPE_FAMILY[
                    p
                ] == family
            ]

            family_sig_cols[
                family
            ] = cols

            matrix[
                f"n_{family}_phenotypes_significant"
            ] = (
                matrix[
                    cols
                ]
                .sum(
                    axis=1
                )
            )

            matrix[
                f"any_{family}_significant"
            ] = (
                matrix[
                    f"n_{family}_phenotypes_significant"
                ]
                > 0
            )

            matrix[
                f"all_{family}_significant"
            ] = (
                matrix[
                    f"n_{family}_phenotypes_significant"
                ]
                == len(cols)
            )

        matrix[
            "weighted_any_significant"
        ] = (
            matrix[
                "n_weighted_phenotypes_significant"
            ]
            > 0
        )

        matrix[
            "weighted_all9_significant"
        ] = (
            matrix[
                "n_weighted_phenotypes_significant"
            ]
            == len(
                PHENOTYPES
            )
        )

        matrix[
            "weighted_specific_vs_DESeq2"
        ] = (
            matrix[
                "weighted_any_significant"
            ]
            & ~matrix[
                "DESeq2_sig"
            ].fillna(False)
        )

        matrix[
            "weighted_shared_with_DESeq2"
        ] = (
            matrix[
                "weighted_any_significant"
            ]
            & matrix[
                "DESeq2_sig"
            ].fillna(False)
        )

        matrix[
            "both_AREA_any_but_DESeq2_not"
        ] = (
            matrix[
                "weighted_any_significant"
            ]
            & matrix[
                "Regular_AREA_sig"
            ].fillna(False)
            & ~matrix[
                "DESeq2_sig"
            ].fillna(False)
        )

        matrix = matrix.sort_values(
            [
                "n_weighted_phenotypes_significant",
                "n_tau_phenotypes_significant",
                "n_amyloid_phenotypes_significant",
                "n_cognition_phenotypes_significant",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            kind="mergesort",
        )

        matrix.to_csv(
            outdir
            / f"{library_name}_weighted_phenotype_matrix.csv",
            index=False,
        )

        # Boolean significance-only matrix.
        significance_cols = (
            ["pathway"]
            + sig_cols
            + [
                "DESeq2_sig",
                "Regular_AREA_sig",
                "n_weighted_phenotypes_significant",
                "n_cognition_phenotypes_significant",
                "n_amyloid_phenotypes_significant",
                "n_tau_phenotypes_significant",
            ]
        )

        matrix[
            significance_cols
        ].to_csv(
            outdir
            / f"{library_name}_weighted_significance_matrix.csv",
            index=False,
        )

        # Recurrence table.
        recurrence = matrix[
            [
                "pathway",
                "n_weighted_phenotypes_significant",
                "n_cognition_phenotypes_significant",
                "n_amyloid_phenotypes_significant",
                "n_tau_phenotypes_significant",
                "weighted_all9_significant",
                "DESeq2_sig",
                "Regular_AREA_sig",
                "weighted_specific_vs_DESeq2",
                "both_AREA_any_but_DESeq2_not",
            ]
        ].copy()

        recurrence.to_csv(
            outdir
            / f"{library_name}_pathway_recurrence.csv",
            index=False,
        )

        # Weighted-specific vs DESeq2.
        matrix.loc[
            matrix[
                "weighted_specific_vs_DESeq2"
            ]
        ].to_csv(
            outdir
            / f"{library_name}_weighted_specific_vs_DESeq2.csv",
            index=False,
        )

        matrix.loc[
            matrix[
                "weighted_shared_with_DESeq2"
            ]
        ].to_csv(
            outdir
            / f"{library_name}_weighted_shared_with_DESeq2.csv",
            index=False,
        )

        # --------------------------------------------------------
        # Continuous vs ordinal pathology comparison.
        # --------------------------------------------------------
        compare_rows = []

        pathology_pairs = [
            (
                "amyloid",
                "CERAD_equal",
                "CERAD_amyloid_calibrated",
                "amyloid_continuous",
            ),
            (
                "tau",
                "Braak_equal",
                "Braak_tangle_calibrated",
                "tangle_continuous",
            ),
        ]

        for (
            family,
            equal_p,
            calibrated_p,
            continuous_p,
        ) in pathology_pairs:
            cols = [
                "pathway",
                f"{equal_p}_NES",
                f"{equal_p}_FDR",
                f"{equal_p}_sig",
                f"{calibrated_p}_NES",
                f"{calibrated_p}_FDR",
                f"{calibrated_p}_sig",
                f"{continuous_p}_NES",
                f"{continuous_p}_FDR",
                f"{continuous_p}_sig",
            ]

            tmp = matrix[
                cols
            ].copy()

            tmp[
                "family"
            ] = family

            tmp[
                "continuous_specific"
            ] = (
                tmp[
                    f"{continuous_p}_sig"
                ]
                & ~tmp[
                    f"{equal_p}_sig"
                ]
                & ~tmp[
                    f"{calibrated_p}_sig"
                ]
            )

            tmp[
                "ordinal_any_specific"
            ] = (
                (
                    tmp[
                        f"{equal_p}_sig"
                    ]
                    | tmp[
                        f"{calibrated_p}_sig"
                    ]
                )
                & ~tmp[
                    f"{continuous_p}_sig"
                ]
            )

            tmp[
                "significant_all_three_encodings"
            ] = (
                tmp[
                    f"{equal_p}_sig"
                ]
                & tmp[
                    f"{calibrated_p}_sig"
                ]
                & tmp[
                    f"{continuous_p}_sig"
                ]
            )

            compare_rows.append(
                tmp
            )

        continuous_compare = pd.concat(
            compare_rows,
            ignore_index=True,
        )

        continuous_compare.to_csv(
            outdir
            / f"{library_name}_continuous_vs_ordinal_pathology.csv",
            index=False,
        )

        # --------------------------------------------------------
        # Cognition vs amyloid vs tau compact summary.
        # --------------------------------------------------------
        family_summary = matrix[
            [
                "pathway",
                "n_cognition_phenotypes_significant",
                "n_amyloid_phenotypes_significant",
                "n_tau_phenotypes_significant",
                "n_weighted_phenotypes_significant",
                "DESeq2_sig",
                "Regular_AREA_sig",
            ]
        ].copy()

        family_summary[
            "cognition_only"
        ] = (
            (
                family_summary[
                    "n_cognition_phenotypes_significant"
                ]
                > 0
            )
            & (
                family_summary[
                    "n_amyloid_phenotypes_significant"
                ]
                == 0
            )
            & (
                family_summary[
                    "n_tau_phenotypes_significant"
                ]
                == 0
            )
        )

        family_summary[
            "amyloid_only"
        ] = (
            (
                family_summary[
                    "n_amyloid_phenotypes_significant"
                ]
                > 0
            )
            & (
                family_summary[
                    "n_cognition_phenotypes_significant"
                ]
                == 0
            )
            & (
                family_summary[
                    "n_tau_phenotypes_significant"
                ]
                == 0
            )
        )

        family_summary[
            "tau_only"
        ] = (
            (
                family_summary[
                    "n_tau_phenotypes_significant"
                ]
                > 0
            )
            & (
                family_summary[
                    "n_cognition_phenotypes_significant"
                ]
                == 0
            )
            & (
                family_summary[
                    "n_amyloid_phenotypes_significant"
                ]
                == 0
            )
        )

        family_summary[
            "shared_all_families"
        ] = (
            (
                family_summary[
                    "n_cognition_phenotypes_significant"
                ]
                > 0
            )
            & (
                family_summary[
                    "n_amyloid_phenotypes_significant"
                ]
                > 0
            )
            & (
                family_summary[
                    "n_tau_phenotypes_significant"
                ]
                > 0
            )
        )

        family_summary.to_csv(
            outdir
            / f"{library_name}_cognition_vs_amyloid_vs_tau_summary.csv",
            index=False,
        )

        print(
            f"\n{library_name} recurrence:"
        )
        print(
            "  Significant in >=1 Weighted phenotype: "
            f"{int(matrix['weighted_any_significant'].sum()):,}"
        )
        print(
            "  Significant in all 9 Weighted phenotypes: "
            f"{int(matrix['weighted_all9_significant'].sum()):,}"
        )
        print(
            "  Weighted-specific vs DESeq2: "
            f"{int(matrix['weighted_specific_vs_DESeq2'].sum()):,}"
        )
        print(
            "  Both AREA + >=1 Weighted, DESeq2 not: "
            f"{int(matrix['both_AREA_any_but_DESeq2_not'].sum()):,}"
        )

    # ------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------
    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        outdir / "gsea_summary.csv",
        index=False,
    )

    (outdir / "analysis_design.txt").write_text(
        "Weighted AREA full 9-phenotype preranked GSEA\n\n"
        "Weighted phenotype families:\n"
        "  cognition: Cognitive_stage, cogn_global_impairment, "
        "mmse_impairment\n"
        "  amyloid: CERAD_equal, CERAD_amyloid_calibrated, "
        "amyloid_continuous\n"
        "  tau: Braak_equal, Braak_tangle_calibrated, "
        "tangle_continuous\n\n"
        "Reference rankings:\n"
        "  DESeq2 AD4 vs NCI1\n"
        "  Adjusted Regular AREA AD4 vs NCI1\n\n"
        "Ranking orientation:\n"
        "  positive = higher expression with greater disease "
        "severity/pathology\n"
        "  DESeq2 = Wald statistic\n"
        "  Regular AREA = -adjusted_Regular_AREA_Z\n"
        "  Weighted AREA = -adjusted_AREA_Z\n\n"
        f"Permutations: {args.permutations}\n"
        f"FDR threshold: {args.fdr}\n"
        f"Gene-set size: {args.min_size}-{args.max_size}\n"
        f"Seed: {args.seed}\n"
    )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        summary[
            [
                "library",
                "method",
                "family",
                "fdr_significant_n",
                "fdr_significant_positive_NES_n",
                "fdr_significant_negative_NES_n",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nOutputs: {outdir}"
    )


if __name__ == "__main__":
    main()

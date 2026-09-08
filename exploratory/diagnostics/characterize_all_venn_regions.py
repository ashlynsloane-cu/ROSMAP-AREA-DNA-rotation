#!/usr/bin/env python3
"""
characterize_all_venn_regions.py
================================

Compact, reusable characterization of every region in the final locked
DESeq2 / Regular AREA / Weighted AREA Venn diagram.

Regions:
    DESeq2 only
    Regular AREA only
    Weighted AREA only
    DESeq2 + Regular AREA only
    DESeq2 + Weighted AREA only
    Regular AREA + Weighted AREA only
    All three

For every gene, the script computes the same descriptive geometry metrics:
    - DESeq2 BH FDR and log2FC
    - endpoint Cohen's d (AD4 vs NCI1)
    - endpoint Mann-Whitney AUC / rank-biserial effect
    - endpoint AD/NCI variance ratio
    - graded NCI/MCI/AD means and medians
    - Spearman severity-expression correlation
    - monotonic median trajectory across NCI -> MCI -> AD

Then it summarizes these metrics by Venn region and makes comparison plots.

Important:
This is descriptive characterization, not a new inferential screen.
DESeq2 and Regular AREA use the 418 endpoint participants.
Weighted AREA uses the broader 619-person graded cohort.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


LOCKED_COVARIATES = [
    "age_death",
    "sex",
    "rin_numeric",
    "pmi_numeric",
    "sequencing_batch",
]

REGION_ORDER = [
    "DESeq2 only",
    "Adjusted Regular AREA only",
    "Weighted AREA only",
    "DESeq2 + Adjusted Regular AREA only",
    "DESeq2 + Weighted AREA only",
    "Adjusted Regular AREA + Weighted AREA only",
    "All three",
]

SHORT_LABELS = {
    "DESeq2 only": "DESeq2 only",
    "Adjusted Regular AREA only": "Regular only",
    "Weighted AREA only": "Weighted only",
    "DESeq2 + Adjusted Regular AREA only": "DESeq2 + Regular",
    "DESeq2 + Weighted AREA only": "DESeq2 + Weighted",
    "Adjusted Regular AREA + Weighted AREA only": "Regular + Weighted",
    "All three": "All three",
}


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
        "--expression",
        default=(
            "results/preprocessing/"
            "ROSMAP_AREA_normalized_counts_all_samples.csv"
        ),
    )

    p.add_argument(
        "--metadata",
        default=(
            "results/preprocessing/"
            "ROSMAP_RNAseq_master_metadata_locked_covariates.csv"
        ),
    )

    p.add_argument("--expected-endpoint-n", type=int, default=418)
    p.add_argument("--expected-graded-n", type=int, default=619)
    p.add_argument("--representatives-per-region", type=int, default=5)

    p.add_argument(
        "--outdir",
        default="results/venn_region_characterization",
    )

    return p.parse_args()


def to_bool(series):
    if series.dtype == bool:
        return series

    x = series.astype(str).str.strip().str.lower()

    out = x.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )

    if out.isna().any():
        raise ValueError(
            "Could not parse significance boolean column."
        )

    return out.astype(bool)


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not identify {label}. "
        f"Tried {candidates}."
    )


def region_from_flags(de, reg, wgt):
    if de and reg and wgt:
        return "All three"

    if de and reg:
        return "DESeq2 + Adjusted Regular AREA only"

    if de and wgt:
        return "DESeq2 + Weighted AREA only"

    if reg and wgt:
        return "Adjusted Regular AREA + Weighted AREA only"

    if de:
        return "DESeq2 only"

    if reg:
        return "Adjusted Regular AREA only"

    if wgt:
        return "Weighted AREA only"

    return "Not significant"


def cohen_d(ad, nci):
    ad = np.asarray(ad, float)
    nci = np.asarray(nci, float)

    ad = ad[np.isfinite(ad)]
    nci = nci[np.isfinite(nci)]

    if len(ad) < 2 or len(nci) < 2:
        return np.nan

    va = np.var(ad, ddof=1)
    vn = np.var(nci, ddof=1)

    pooled = (
        ((len(ad) - 1) * va + (len(nci) - 1) * vn)
        / (len(ad) + len(nci) - 2)
    )

    if pooled <= 0:
        return 0.0

    return (
        np.mean(ad) - np.mean(nci)
    ) / np.sqrt(pooled)


def rank_biserial(ad, nci):
    ad = np.asarray(ad, float)
    nci = np.asarray(nci, float)

    ad = ad[np.isfinite(ad)]
    nci = nci[np.isfinite(nci)]

    result = mannwhitneyu(
        ad,
        nci,
        alternative="two-sided",
        method="asymptotic",
    )

    auc = float(result.statistic) / (
        len(ad) * len(nci)
    )

    return auc, 2.0 * auc - 1.0


def monotonic_label(nci, mci, ad):
    if (
        nci <= mci <= ad
        and not (nci == mci == ad)
    ):
        return "monotonic_increasing"

    if (
        nci >= mci >= ad
        and not (nci == mci == ad)
    ):
        return "monotonic_decreasing"

    if nci == mci == ad:
        return "flat"

    return "non_monotonic"


def safe_name(text):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    ).strip("_")


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    membership = pd.read_csv(
        args.membership
    )

    required = [
        "gene_id",
        "gene_symbol",
        "DESeq2_pvalue",
        "DESeq2_comparison_padj_BH",
        "log2FoldChange",
        "Regular_comparison_padj_BH",
        "adjusted_Regular_AREA_Z",
        "Weighted_comparison_padj_BH",
        "adjusted_AREA_Z",
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
    ]

    missing = [
        c
        for c in required
        if c not in membership.columns
    ]

    if missing:
        raise ValueError(
            f"Membership table missing columns: {missing}"
        )

    for c in [
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
    ]:
        membership[c] = to_bool(
            membership[c]
        )

    membership["region"] = [
        region_from_flags(de, reg, wgt)
        for de, reg, wgt in zip(
            membership["DESeq2_significant"],
            membership["Regular_AREA_significant"],
            membership["Weighted_AREA_significant"],
        )
    ]

    membership = membership.loc[
        membership["region"] != "Not significant"
    ].copy()

    print("=" * 90)
    print("FINAL VENN REGION CHARACTERIZATION")
    print("=" * 90)

    print("\nREGION COUNTS")

    observed_counts = (
        membership["region"]
        .value_counts()
        .reindex(
            REGION_ORDER,
            fill_value=0,
        )
    )

    for region in REGION_ORDER:
        print(
            f"  {region:<45} "
            f"{observed_counts[region]:,}"
        )

    # ------------------------------------------------------------
    # Cohorts
    # ------------------------------------------------------------
    meta = pd.read_csv(
        args.metadata
    )

    sample_col = pick_col(
        meta,
        [
            "sample_id",
            "SampleID",
            "sample",
            "rna_sample_id",
        ],
        "sample ID column",
    )

    dx_col = pick_col(
        meta,
        [
            "diagnosis",
            "Diagnosis",
            "diagnosis_code",
        ],
        "diagnosis column",
    )

    missing_covariates = [
        c
        for c in LOCKED_COVARIATES
        if c not in meta.columns
    ]

    if missing_covariates:
        raise ValueError(
            f"Metadata missing locked covariates: "
            f"{missing_covariates}"
        )

    meta[sample_col] = (
        meta[sample_col]
        .astype(str)
        .str.strip()
    )

    meta[dx_col] = pd.to_numeric(
        meta[dx_col],
        errors="coerce",
    )

    complete = (
        meta[LOCKED_COVARIATES]
        .notna()
        .all(axis=1)
    )

    endpoint = meta.loc[
        complete
        & meta[dx_col].isin(
            [1, 4]
        )
    ].copy()

    endpoint["group"] = np.where(
        endpoint[dx_col] == 1,
        "NCI",
        "AD",
    )

    graded = meta.loc[
        complete
        & meta[dx_col].isin(
            [1, 2, 3, 4, 5]
        )
    ].copy()

    graded["group"] = np.select(
        [
            graded[dx_col] == 1,
            graded[dx_col].isin(
                [2, 3]
            ),
            graded[dx_col].isin(
                [4, 5]
            ),
        ],
        [
            "NCI",
            "MCI",
            "AD",
        ],
        default="exclude",
    )

    graded["severity"] = np.select(
        [
            graded[dx_col] == 1,
            graded[dx_col].isin(
                [2, 3]
            ),
            graded[dx_col].isin(
                [4, 5]
            ),
        ],
        [
            0.0,
            0.5,
            1.0,
        ],
        default=np.nan,
    )

    if endpoint[sample_col].nunique() != args.expected_endpoint_n:
        raise ValueError(
            f"Endpoint n={endpoint[sample_col].nunique()}, "
            f"expected {args.expected_endpoint_n}."
        )

    if graded[sample_col].nunique() != args.expected_graded_n:
        raise ValueError(
            f"Graded n={graded[sample_col].nunique()}, "
            f"expected {args.expected_graded_n}."
        )

    print("\nCOHORTS")
    print(
        f"  Endpoint cohort: "
        f"{endpoint[sample_col].nunique():,}"
    )
    print(
        f"  Graded cohort:   "
        f"{graded[sample_col].nunique():,}"
    )

    # ------------------------------------------------------------
    # Expression matrix: sample-major in this project.
    # ------------------------------------------------------------
    expr_raw = pd.read_csv(
        args.expression
    )

    if "sample_id" not in expr_raw.columns:
        raise ValueError(
            "Expected sample-major normalized expression matrix "
            "with sample_id column."
        )

    expr_raw["sample_id"] = (
        expr_raw["sample_id"]
        .astype(str)
        .str.strip()
    )

    target_ids = set(
        membership["gene_id"]
        .astype(str)
    )

    missing_genes = (
        target_ids
        - set(
            expr_raw.columns
            .astype(str)
        )
    )

    if missing_genes:
        raise ValueError(
            f"{len(missing_genes)} Venn genes missing "
            "from expression matrix."
        )

    graded_samples = (
        graded[sample_col]
        .tolist()
    )

    endpoint_samples = (
        endpoint[sample_col]
        .tolist()
    )

    missing_samples = (
        set(graded_samples)
        - set(expr_raw["sample_id"])
    )

    if missing_samples:
        raise ValueError(
            f"{len(missing_samples)} graded samples missing "
            "from expression matrix."
        )

    expr = (
        expr_raw
        .set_index("sample_id")
        .loc[
            graded_samples,
            sorted(target_ids),
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .T
    )

    log_expr = np.log2(
        expr + 1.0
    )

    endpoint_idx = (
        endpoint
        .set_index(sample_col)
    )

    graded_idx = (
        graded
        .set_index(sample_col)
    )

    lookup = (
        membership
        .set_index("gene_id")
    )

    # ------------------------------------------------------------
    # Per-gene geometry metrics
    # ------------------------------------------------------------
    rows = []

    for gene_id in membership[
        "gene_id"
    ].astype(str):

        ep = log_expr.loc[
            gene_id,
            endpoint_samples,
        ]

        ad = ep.loc[
            endpoint_idx.index[
                endpoint_idx[
                    "group"
                ] == "AD"
            ]
        ].to_numpy(
            float
        )

        nci = ep.loc[
            endpoint_idx.index[
                endpoint_idx[
                    "group"
                ] == "NCI"
            ]
        ].to_numpy(
            float
        )

        auc, rbc = rank_biserial(
            ad,
            nci,
        )

        d = cohen_d(
            ad,
            nci,
        )

        va = np.var(
            ad,
            ddof=1,
        )

        vn = np.var(
            nci,
            ddof=1,
        )

        var_ratio = (
            va / vn
            if vn > 0
            else np.nan
        )

        gr = log_expr.loc[
            gene_id,
            graded_samples,
        ]

        severity = (
            graded_idx[
                "severity"
            ]
            .to_numpy(
                float
            )
        )

        rho = float(
            spearmanr(
                severity,
                gr.to_numpy(
                    float
                ),
                nan_policy="omit",
            ).statistic
        )

        means = {}
        medians = {}

        for group in [
            "NCI",
            "MCI",
            "AD",
        ]:
            vals = gr.loc[
                graded_idx.index[
                    graded_idx[
                        "group"
                    ] == group
                ]
            ].to_numpy(
                float
            )

            means[group] = float(
                np.nanmean(
                    vals
                )
            )

            medians[group] = float(
                np.nanmedian(
                    vals
                )
            )

        info = lookup.loc[
            gene_id
        ]

        rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": info[
                    "gene_symbol"
                ],
                "region": info[
                    "region"
                ],
                "DESeq2_BH": info[
                    "DESeq2_comparison_padj_BH"
                ],
                "DESeq2_log2FoldChange": info[
                    "log2FoldChange"
                ],
                "Regular_BH": info[
                    "Regular_comparison_padj_BH"
                ],
                "Regular_Z": info[
                    "adjusted_Regular_AREA_Z"
                ],
                "Weighted_BH": info[
                    "Weighted_comparison_padj_BH"
                ],
                "Weighted_Z": info[
                    "adjusted_AREA_Z"
                ],
                "cohen_d_AD_minus_NCI": d,
                "mann_whitney_AUC_AD_gt_NCI": auc,
                "rank_biserial_AD_minus_NCI": rbc,
                "AD_to_NCI_variance_ratio": var_ratio,
                "graded_NCI_mean_log2norm": means[
                    "NCI"
                ],
                "graded_MCI_mean_log2norm": means[
                    "MCI"
                ],
                "graded_AD_mean_log2norm": means[
                    "AD"
                ],
                "graded_NCI_median_log2norm": medians[
                    "NCI"
                ],
                "graded_MCI_median_log2norm": medians[
                    "MCI"
                ],
                "graded_AD_median_log2norm": medians[
                    "AD"
                ],
                "spearman_severity_vs_expression": rho,
                "median_trajectory": monotonic_label(
                    medians["NCI"],
                    medians["MCI"],
                    medians["AD"],
                ),
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    numeric_cols = [
        "DESeq2_BH",
        "DESeq2_log2FoldChange",
        "Regular_BH",
        "Regular_Z",
        "Weighted_BH",
        "Weighted_Z",
        "cohen_d_AD_minus_NCI",
        "rank_biserial_AD_minus_NCI",
        "AD_to_NCI_variance_ratio",
        "spearman_severity_vs_expression",
    ]

    for col in numeric_cols:
        metrics[col] = pd.to_numeric(
            metrics[col],
            errors="coerce",
        )

    metrics[
        "abs_DESeq2_log2FC"
    ] = metrics[
        "DESeq2_log2FoldChange"
    ].abs()

    metrics[
        "abs_cohen_d"
    ] = metrics[
        "cohen_d_AD_minus_NCI"
    ].abs()

    metrics[
        "abs_rank_biserial"
    ] = metrics[
        "rank_biserial_AD_minus_NCI"
    ].abs()

    metrics[
        "abs_spearman_rho"
    ] = metrics[
        "spearman_severity_vs_expression"
    ].abs()

    metrics[
        "is_monotonic"
    ] = metrics[
        "median_trajectory"
    ].isin(
        [
            "monotonic_increasing",
            "monotonic_decreasing",
        ]
    )

    metrics_out = (
        outdir
        / "all_regions_gene_geometry.csv"
    )

    metrics.to_csv(
        metrics_out,
        index=False,
    )

    # ------------------------------------------------------------
    # Region summary
    # ------------------------------------------------------------
    summary_rows = []

    for region in REGION_ORDER:
        sub = metrics.loc[
            metrics[
                "region"
            ] == region
        ].copy()

        if sub.empty:
            continue

        summary_rows.append(
            {
                "region": region,
                "short_label":
                    SHORT_LABELS[
                        region
                    ],
                "n_genes":
                    len(sub),
                "median_DESeq2_BH":
                    sub[
                        "DESeq2_BH"
                    ].median(),
                "fraction_DESeq2_BH_lt_0.10":
                    (
                        sub[
                            "DESeq2_BH"
                        ] < 0.10
                    ).mean(),
                "median_abs_DESeq2_log2FC":
                    sub[
                        "abs_DESeq2_log2FC"
                    ].median(),
                "median_abs_Cohens_d":
                    sub[
                        "abs_cohen_d"
                    ].median(),
                "median_abs_rank_biserial":
                    sub[
                        "abs_rank_biserial"
                    ].median(),
                "median_abs_spearman_rho":
                    sub[
                        "abs_spearman_rho"
                    ].median(),
                "fraction_monotonic":
                    sub[
                        "is_monotonic"
                    ].mean(),
                "median_AD_NCI_variance_ratio":
                    sub[
                        "AD_to_NCI_variance_ratio"
                    ].median(),
                "spearman_abs_d_vs_abs_rank":
                    spearmanr(
                        sub[
                            "abs_cohen_d"
                        ],
                        sub[
                            "abs_rank_biserial"
                        ],
                        nan_policy="omit",
                    ).statistic,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_out = (
        outdir
        / "venn_region_geometry_summary.csv"
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    # ------------------------------------------------------------
    # Representative genes per region
    # ------------------------------------------------------------
    reps = []

    for region in REGION_ORDER:
        sub = metrics.loc[
            metrics[
                "region"
            ] == region
        ].copy()

        if sub.empty:
            continue

        sub = sub.loc[
            sub[
                "gene_symbol"
            ].notna()
        ].copy()

        # Rank representatives by the methods defining the region.
        if region == "DESeq2 only":
            sub[
                "priority"
            ] = (
                -np.log10(
                    sub[
                        "DESeq2_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
            )

        elif region == "Adjusted Regular AREA only":
            sub[
                "priority"
            ] = (
                -np.log10(
                    sub[
                        "Regular_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
            )

        elif region == "Weighted AREA only":
            sub[
                "priority"
            ] = (
                -np.log10(
                    sub[
                        "Weighted_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
            )

        elif region == "DESeq2 + Adjusted Regular AREA only":
            sub[
                "priority"
            ] = (
                -np.log10(
                    sub[
                        "DESeq2_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
                - np.log10(
                    sub[
                        "Regular_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
            )

        elif region == "DESeq2 + Weighted AREA only":
            sub[
                "priority"
            ] = (
                -np.log10(
                    sub[
                        "DESeq2_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
                - np.log10(
                    sub[
                        "Weighted_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
            )

        elif region == "Adjusted Regular AREA + Weighted AREA only":
            sub[
                "priority"
            ] = (
                -np.log10(
                    sub[
                        "Regular_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
                - np.log10(
                    sub[
                        "Weighted_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
            )

        else:
            sub[
                "priority"
            ] = (
                -np.log10(
                    sub[
                        "DESeq2_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
                - np.log10(
                    sub[
                        "Regular_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
                - np.log10(
                    sub[
                        "Weighted_BH"
                    ].clip(
                        lower=1e-300
                    )
                )
            )

        top = (
            sub
            .sort_values(
                [
                    "priority",
                    "abs_spearman_rho",
                ],
                ascending=[
                    False,
                    False,
                ],
                kind="mergesort",
            )
            .head(
                args.representatives_per_region
            )
            .copy()
        )

        reps.append(
            top
        )

    representatives = pd.concat(
        reps,
        ignore_index=True,
    )

    reps_out = (
        outdir
        / "representative_genes_by_region.csv"
    )

    representatives.to_csv(
        reps_out,
        index=False,
    )

    # ------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------
    plotting_order = [
        r
        for r in REGION_ORDER
        if r in set(
            summary["region"]
        )
    ]

    short_order = [
        SHORT_LABELS[
            r
        ]
        for r in plotting_order
    ]

    def boxplot_metric(
        metric,
        ylabel,
        title,
        filename,
    ):
        data = [
            metrics.loc[
                metrics[
                    "region"
                ] == region,
                metric,
            ]
            .dropna()
            .to_numpy()
            for region in plotting_order
        ]

        fig, ax = plt.subplots(
            figsize=(12, 6.5)
        )

        ax.boxplot(
            data,
            tick_labels=short_order,
            showfliers=False,
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title
        )

        ax.tick_params(
            axis="x",
            rotation=30,
        )

        fig.tight_layout()

        fig.savefig(
            outdir
            / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    boxplot_metric(
        "abs_DESeq2_log2FC",
        "|DESeq2 log2 fold-change|",
        "Mean-shift magnitude across Venn regions",
        "compare_abs_deseq2_log2FC.png",
    )

    boxplot_metric(
        "abs_cohen_d",
        "|Cohen's d|",
        "Endpoint mean-shift effect across Venn regions",
        "compare_abs_cohens_d.png",
    )

    boxplot_metric(
        "abs_rank_biserial",
        "|Rank-biserial effect|",
        "Endpoint rank-order effect across Venn regions",
        "compare_abs_rank_biserial.png",
    )

    boxplot_metric(
        "abs_spearman_rho",
        "|Spearman rho|",
        "Graded severity-expression association across Venn regions",
        "compare_abs_spearman_rho.png",
    )

    # Monotonicity fraction.
    mono = summary.set_index(
        "region"
    ).loc[
        plotting_order,
        "fraction_monotonic",
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        short_order,
        mono.to_numpy(),
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Fraction monotonic NCI → MCI → AD"
    )

    ax.set_title(
        "Monotonic graded expression trajectories across Venn regions"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig.tight_layout()

    fig.savefig(
        outdir
        / "compare_monotonic_fraction.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # DESeq2 FDR distributions for non-DESeq2 regions only.
    non_de_regions = [
        "Adjusted Regular AREA only",
        "Weighted AREA only",
        "Adjusted Regular AREA + Weighted AREA only",
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for region in non_de_regions:
        values = metrics.loc[
            metrics[
                "region"
            ] == region,
            "DESeq2_BH",
        ].dropna()

        ax.hist(
            values,
            bins=30,
            alpha=0.35,
            label=SHORT_LABELS[
                region
            ],
            density=True,
        )

    ax.axvline(
        0.05,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel(
        "DESeq2 BH-adjusted p-value"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "How close are AREA-only regions to DESeq2 significance?"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir
        / "compare_deseq2_fdr_area_only_regions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------
    print("\nREGION SUMMARY")
    print(
        summary[
            [
                "short_label",
                "n_genes",
                "median_DESeq2_BH",
                "fraction_DESeq2_BH_lt_0.10",
                "median_abs_DESeq2_log2FC",
                "median_abs_Cohens_d",
                "median_abs_rank_biserial",
                "median_abs_spearman_rho",
                "fraction_monotonic",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nREPRESENTATIVE GENES BY REGION")

    print(
        representatives[
            [
                "region",
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
        ].to_string(
            index=False
        )
    )

    print("\nWROTE")
    print(
        f"  {metrics_out}"
    )
    print(
        f"  {summary_out}"
    )
    print(
        f"  {reps_out}"
    )
    print(
        f"  {outdir / 'compare_abs_deseq2_log2FC.png'}"
    )
    print(
        f"  {outdir / 'compare_abs_cohens_d.png'}"
    )
    print(
        f"  {outdir / 'compare_abs_rank_biserial.png'}"
    )
    print(
        f"  {outdir / 'compare_abs_spearman_rho.png'}"
    )
    print(
        f"  {outdir / 'compare_monotonic_fraction.png'}"
    )
    print(
        f"  {outdir / 'compare_deseq2_fdr_area_only_regions.png'}"
    )


if __name__ == "__main__":
    main()

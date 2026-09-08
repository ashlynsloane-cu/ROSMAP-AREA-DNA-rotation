#!/usr/bin/env python3
"""
Deep characterization of the 468 genes significant in adjusted Regular AREA
and Weighted AREA but not DESeq2.

Outputs descriptive expression-geometry metrics and figures to:
results/characterization_468/expression_geometry/
"""

from pathlib import Path
import argparse
import re

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
    p.add_argument("--expected-target-n", type=int, default=468)
    p.add_argument("--expected-endpoint-n", type=int, default=418)
    p.add_argument("--expected-graded-n", type=int, default=619)
    p.add_argument("--n-representatives", type=int, default=6)
    p.add_argument(
        "--outdir",
        default=(
            "results/characterization_468/"
            "expression_geometry"
        ),
    )
    return p.parse_args()


def to_bool(series):
    if series.dtype == bool:
        return series
    x = series.astype(str).str.strip().str.lower()
    out = x.map({"true": True, "false": False, "1": True, "0": False})
    if out.isna().any():
        raise ValueError("Could not parse significance booleans.")
    return out.astype(bool)


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find {label}. Tried {candidates}. "
        f"Available: {df.columns.tolist()[:30]}"
    )


def infer_gene_col(df):
    # First try common explicit names.
    for c in [
        "gene_id",
        "ensembl_id_version",
        "ensembl_id",
        "Geneid",
        "gene",
        "feature_id",
        "rowname",
        "rownames",
        "Unnamed: 0",
    ]:
        if c in df.columns:
            vals = df[c].astype(str).str.strip()
            if vals.str.startswith("ENSG").mean() > 0.5:
                return c

    # Then scan every column for Ensembl-style IDs. This handles CSVs where
    # R/pandas wrote row names under an unexpected header.
    candidates = []

    for c in df.columns:
        vals = df[c].astype(str).str.strip()
        frac_ensg = vals.str.startswith("ENSG").mean()

        if frac_ensg > 0.5:
            candidates.append((c, frac_ensg))

    if len(candidates) == 1:
        return candidates[0][0]

    if len(candidates) > 1:
        candidates = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True,
        )
        return candidates[0][0]

    raise ValueError(
        "Could not identify expression gene-ID column. "
        f"First 15 columns: {df.columns.tolist()[:15]}"
    )


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
        ((len(ad)-1)*va + (len(nci)-1)*vn)
        / (len(ad)+len(nci)-2)
    )
    if pooled <= 0:
        return 0.0
    return (np.mean(ad)-np.mean(nci)) / np.sqrt(pooled)


def rank_biserial(ad, nci):
    ad = np.asarray(ad, float)
    nci = np.asarray(nci, float)
    ad = ad[np.isfinite(ad)]
    nci = nci[np.isfinite(nci)]
    u = mannwhitneyu(
        ad,
        nci,
        alternative="two-sided",
        method="asymptotic",
    ).statistic
    auc = u / (len(ad) * len(nci))
    return auc, 2*auc - 1


def monotonic_label(a, b, c):
    if a <= b <= c and not (a == b == c):
        return "monotonic_increasing"
    if a >= b >= c and not (a == b == c):
        return "monotonic_decreasing"
    if a == b == c:
        return "flat"
    return "non_monotonic"


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    membership = pd.read_csv(args.membership)
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
    missing = [c for c in required if c not in membership.columns]
    if missing:
        raise ValueError(f"Membership table missing columns: {missing}")

    for c in [
        "DESeq2_significant",
        "Regular_AREA_significant",
        "Weighted_AREA_significant",
    ]:
        membership[c] = to_bool(membership[c])

    target = membership.loc[
        (~membership["DESeq2_significant"])
        & membership["Regular_AREA_significant"]
        & membership["Weighted_AREA_significant"]
    ].copy()

    if len(target) != args.expected_target_n:
        raise ValueError(
            f"Expected {args.expected_target_n} target genes, found {len(target)}."
        )

    meta = pd.read_csv(args.metadata)
    sample_col = pick_col(
        meta,
        ["sample_id", "SampleID", "sample", "rna_sample_id"],
        "sample ID column",
    )
    dx_col = pick_col(
        meta,
        ["diagnosis", "Diagnosis", "diagnosis_code"],
        "diagnosis column",
    )

    missing_cov = [c for c in LOCKED_COVARIATES if c not in meta.columns]
    if missing_cov:
        raise ValueError(f"Metadata missing locked covariates: {missing_cov}")

    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    meta[dx_col] = pd.to_numeric(meta[dx_col], errors="coerce")
    complete = meta[LOCKED_COVARIATES].notna().all(axis=1)

    endpoint = meta.loc[
        complete & meta[dx_col].isin([1, 4])
    ].copy()
    endpoint["group"] = np.where(
        endpoint[dx_col] == 1, "NCI", "AD"
    )

    graded = meta.loc[
        complete & meta[dx_col].isin([1, 2, 3, 4, 5])
    ].copy()
    graded["group"] = np.select(
        [
            graded[dx_col] == 1,
            graded[dx_col].isin([2, 3]),
            graded[dx_col].isin([4, 5]),
        ],
        ["NCI", "MCI", "AD"],
        default="exclude",
    )

    graded["severity"] = np.select(
        [
            graded[dx_col] == 1,
            graded[dx_col].isin([2, 3]),
            graded[dx_col].isin([4, 5]),
        ],
        [0.0, 0.5, 1.0],
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

    expr_raw = pd.read_csv(args.expression)

    target_ids = set(target["gene_id"].astype(str))
    endpoint_samples = endpoint[sample_col].tolist()
    graded_samples = graded[sample_col].tolist()

    # The normalized-count file used in this project is sample-major:
    # rows = samples, columns = genes, with sample_id as the first column.
    #
    # Still support gene-major input as a fallback so this script remains reusable.
    if (
        "sample_id" in expr_raw.columns
        and sum(
            str(c).startswith("ENSG")
            for c in expr_raw.columns
        ) > 100
    ):
        print("\nEXPRESSION MATRIX ORIENTATION")
        print("  Detected sample-major matrix: rows=samples, columns=genes")

        expr_raw["sample_id"] = (
            expr_raw["sample_id"]
            .astype(str)
            .str.strip()
        )

        if expr_raw["sample_id"].duplicated().any():
            raise ValueError(
                "Expression matrix contains duplicate sample_id values."
            )

        missing_target_genes = (
            target_ids
            - set(expr_raw.columns.astype(str))
        )

        if missing_target_genes:
            raise ValueError(
                f"{len(missing_target_genes)} target genes are missing "
                "from the expression matrix."
            )

        missing_endpoint = (
            set(endpoint_samples)
            - set(expr_raw["sample_id"])
        )

        missing_graded = (
            set(graded_samples)
            - set(expr_raw["sample_id"])
        )

        if missing_endpoint:
            raise ValueError(
                f"{len(missing_endpoint)} endpoint samples missing from expression."
            )

        if missing_graded:
            raise ValueError(
                f"{len(missing_graded)} graded samples missing from expression."
            )

        # Keep only the 468 genes and the graded cohort samples, then transpose
        # into the gene x sample layout used by the rest of this script.
        expr_subset = (
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
        )

        expr = expr_subset.T
        expr.index.name = "gene_id"

    else:
        print("\nEXPRESSION MATRIX ORIENTATION")
        print("  Detected gene-major matrix: rows=genes, columns=samples")

        gene_col = infer_gene_col(expr_raw)

        expr_raw[gene_col] = (
            expr_raw[gene_col]
            .astype(str)
            .str.strip()
        )

        missing_target_genes = (
            target_ids
            - set(expr_raw[gene_col])
        )

        if missing_target_genes:
            raise ValueError(
                f"{len(missing_target_genes)} target genes are missing "
                "from the expression matrix."
            )

        expr = (
            expr_raw
            .loc[
                expr_raw[gene_col].isin(target_ids)
            ]
            .set_index(gene_col)
        )

        missing_endpoint = (
            set(endpoint_samples)
            - set(expr.columns)
        )

        missing_graded = (
            set(graded_samples)
            - set(expr.columns)
        )

        if missing_endpoint:
            raise ValueError(
                f"{len(missing_endpoint)} endpoint samples missing from expression."
            )

        if missing_graded:
            raise ValueError(
                f"{len(missing_graded)} graded samples missing from expression."
            )

        expr = (
            expr.loc[
                sorted(target_ids),
                graded_samples,
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

    if expr.shape != (
        len(target_ids),
        len(graded_samples),
    ):
        raise ValueError(
            "Unexpected expression shape after orientation handling: "
            f"{expr.shape}; expected "
            f"({len(target_ids)}, {len(graded_samples)})."
        )

    print(
        f"  Working expression matrix: "
        f"{expr.shape[0]:,} genes x {expr.shape[1]:,} samples"
    )

    log_expr = np.log2(
        expr + 1.0
    )

    endpoint_idx = endpoint.set_index(sample_col)
    graded_idx = graded.set_index(sample_col)
    lookup = target.set_index("gene_id")

    rows = []

    for gene_id in target["gene_id"].astype(str):
        ep = log_expr.loc[gene_id, endpoint_samples]
        ad = ep.loc[
            endpoint_idx.index[
                endpoint_idx["group"] == "AD"
            ]
        ].to_numpy(float)
        nci = ep.loc[
            endpoint_idx.index[
                endpoint_idx["group"] == "NCI"
            ]
        ].to_numpy(float)

        auc, rbc = rank_biserial(ad, nci)
        d = cohen_d(ad, nci)

        va = np.var(ad, ddof=1)
        vn = np.var(nci, ddof=1)
        var_ratio = va / vn if vn > 0 else np.nan

        gr = log_expr.loc[gene_id, graded_samples]
        sev = graded_idx["severity"].to_numpy(float)
        rho = float(
            spearmanr(
                sev,
                gr.to_numpy(float),
                nan_policy="omit",
            ).statistic
        )

        medians = {}
        means = {}
        for group in ["NCI", "MCI", "AD"]:
            vals = gr.loc[
                graded_idx.index[
                    graded_idx["group"] == group
                ]
            ].to_numpy(float)
            medians[group] = float(np.nanmedian(vals))
            means[group] = float(np.nanmean(vals))

        info = lookup.loc[gene_id]

        rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": info["gene_symbol"],
                "DESeq2_pvalue": info["DESeq2_pvalue"],
                "DESeq2_BH": info["DESeq2_comparison_padj_BH"],
                "DESeq2_log2FoldChange": info["log2FoldChange"],
                "Regular_BH": info["Regular_comparison_padj_BH"],
                "Regular_Z": info["adjusted_Regular_AREA_Z"],
                "Weighted_BH": info["Weighted_comparison_padj_BH"],
                "Weighted_Z": info["adjusted_AREA_Z"],
                "endpoint_NCI_mean_log2norm": float(np.nanmean(nci)),
                "endpoint_AD_mean_log2norm": float(np.nanmean(ad)),
                "endpoint_NCI_median_log2norm": float(np.nanmedian(nci)),
                "endpoint_AD_median_log2norm": float(np.nanmedian(ad)),
                "cohen_d_AD_minus_NCI": d,
                "mann_whitney_AUC_AD_gt_NCI": auc,
                "rank_biserial_AD_minus_NCI": rbc,
                "AD_to_NCI_variance_ratio": var_ratio,
                "graded_NCI_mean_log2norm": means["NCI"],
                "graded_MCI_mean_log2norm": means["MCI"],
                "graded_AD_mean_log2norm": means["AD"],
                "graded_NCI_median_log2norm": medians["NCI"],
                "graded_MCI_median_log2norm": medians["MCI"],
                "graded_AD_median_log2norm": medians["AD"],
                "spearman_severity_vs_expression": rho,
                "median_trajectory": monotonic_label(
                    medians["NCI"],
                    medians["MCI"],
                    medians["AD"],
                ),
            }
        )

    metrics = pd.DataFrame(rows)

    for c in [
        "DESeq2_pvalue",
        "DESeq2_BH",
        "DESeq2_log2FoldChange",
        "Regular_BH",
        "Regular_Z",
        "Weighted_BH",
        "Weighted_Z",
    ]:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    metrics["abs_DESeq2_log2FoldChange"] = metrics[
        "DESeq2_log2FoldChange"
    ].abs()
    metrics["abs_cohen_d"] = metrics[
        "cohen_d_AD_minus_NCI"
    ].abs()
    metrics["abs_rank_biserial"] = metrics[
        "rank_biserial_AD_minus_NCI"
    ].abs()
    metrics["abs_spearman_rho"] = metrics[
        "spearman_severity_vs_expression"
    ].abs()
    metrics["joint_AREA_strength"] = (
        -np.log10(metrics["Regular_BH"].clip(lower=1e-300))
        -np.log10(metrics["Weighted_BH"].clip(lower=1e-300))
    )

    metrics_out = outdir / "geometry_metrics_468.csv"
    metrics.to_csv(metrics_out, index=False)

    n_mono = int(
        metrics["median_trajectory"].isin(
            ["monotonic_increasing", "monotonic_decreasing"]
        ).sum()
    )

    summary = pd.DataFrame(
        [
            ("n_genes", len(metrics)),
            (
                "median_abs_DESeq2_log2FC",
                metrics["abs_DESeq2_log2FoldChange"].median(),
            ),
            ("median_DESeq2_BH", metrics["DESeq2_BH"].median()),
            (
                "fraction_DESeq2_BH_lt_0.10",
                (metrics["DESeq2_BH"] < 0.10).mean(),
            ),
            (
                "fraction_DESeq2_BH_lt_0.25",
                (metrics["DESeq2_BH"] < 0.25).mean(),
            ),
            ("median_abs_Cohens_d", metrics["abs_cohen_d"].median()),
            (
                "median_abs_rank_biserial",
                metrics["abs_rank_biserial"].median(),
            ),
            (
                "median_abs_spearman_rho",
                metrics["abs_spearman_rho"].median(),
            ),
            (
                "n_monotonic_median_trajectory",
                n_mono,
            ),
            (
                "fraction_monotonic_median_trajectory",
                n_mono / len(metrics),
            ),
            (
                "spearman_abs_rank_biserial_vs_abs_cohen_d",
                spearmanr(
                    metrics["abs_rank_biserial"],
                    metrics["abs_cohen_d"],
                    nan_policy="omit",
                ).statistic,
            ),
            (
                "spearman_Regular_Z_vs_Weighted_Z",
                spearmanr(
                    metrics["Regular_Z"],
                    metrics["Weighted_Z"],
                    nan_policy="omit",
                ).statistic,
            ),
        ],
        columns=["metric", "value"],
    )

    summary_out = outdir / "geometry_summary.csv"
    summary.to_csv(summary_out, index=False)

    reps = (
        metrics.loc[metrics["gene_symbol"].notna()]
        .assign(
            is_monotonic=lambda d: d["median_trajectory"].isin(
                ["monotonic_increasing", "monotonic_decreasing"]
            )
        )
        .sort_values(
            [
                "is_monotonic",
                "joint_AREA_strength",
                "abs_DESeq2_log2FoldChange",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(args.n_representatives)
        .copy()
    )

    reps_out = outdir / "representative_genes.csv"
    reps.to_csv(reps_out, index=False)

    # Figure 1: DESeq2 FDR distribution.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(metrics["DESeq2_BH"].dropna(), bins=30)
    ax.axvline(0.05, linestyle="--", linewidth=1.5)
    ax.axvline(0.10, linestyle=":", linewidth=1.5)
    ax.set_xlabel("DESeq2 BH-adjusted p-value")
    ax.set_ylabel("Number of genes")
    ax.set_title("DESeq2 FDR distribution for the 468 rank-based-only genes")
    fig.tight_layout()
    fig.savefig(
        outdir / "deseq2_fdr_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: Regular vs Weighted Z.
    rho_rw = spearmanr(
        metrics["Regular_Z"],
        metrics["Weighted_Z"],
        nan_policy="omit",
    ).statistic
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(metrics["Regular_Z"], metrics["Weighted_Z"], alpha=0.65, s=28)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Adjusted Regular AREA Z")
    ax.set_ylabel("Weighted AREA Z")
    ax.set_title(f"Regular vs Weighted AREA Z (rho={rho_rw:.3f})")
    fig.tight_layout()
    fig.savefig(
        outdir / "regular_vs_weighted_z.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: mean-shift vs rank-order effect.
    rho_eff = spearmanr(
        metrics["abs_cohen_d"],
        metrics["abs_rank_biserial"],
        nan_policy="omit",
    ).statistic
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        metrics["abs_cohen_d"],
        metrics["abs_rank_biserial"],
        alpha=0.65,
        s=28,
    )
    ax.set_xlabel("|Cohen's d|, AD4 vs NCI1")
    ax.set_ylabel("|Rank-biserial effect|, AD4 vs NCI1")
    ax.set_title(f"Mean-shift vs rank-order effect (rho={rho_eff:.3f})")
    fig.tight_layout()
    fig.savefig(
        outdir / "rank_vs_mean_effect.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 4: graded median trajectories.
    order = [
        "monotonic_increasing",
        "monotonic_decreasing",
        "non_monotonic",
        "flat",
    ]
    counts = metrics["median_trajectory"].value_counts().reindex(
        order, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Number of genes")
    ax.set_title("Median expression trajectory across NCI → MCI → AD")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(
        outdir / "graded_monotonicity_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # One representative-gene plot per gene.
    for _, rep in reps.iterrows():
        gene_id = rep["gene_id"]
        symbol = str(rep["gene_symbol"])
        values = log_expr.loc[gene_id, graded_samples].rename("expr").to_frame()
        values["group"] = graded_idx["group"]

        grouped = [
            values.loc[values["group"] == g, "expr"].dropna().to_numpy()
            for g in ["NCI", "MCI", "AD"]
        ]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.boxplot(grouped, positions=[0, 1, 2], showfliers=False)
        rng = np.random.default_rng(12345)

        for pos, arr in enumerate(grouped):
            jitter = rng.normal(pos, 0.055, size=len(arr))
            ax.scatter(jitter, arr, alpha=0.22, s=12)

        ax.set_xticks([0, 1, 2], ["NCI", "MCI", "AD"])
        ax.set_ylabel("log2(normalized count + 1)")
        ax.set_title(
            f"{symbol}\n"
            f"DESeq2 BH={rep['DESeq2_BH']:.3g} | "
            f"Regular BH={rep['Regular_BH']:.3g} | "
            f"Weighted BH={rep['Weighted_BH']:.3g}"
        )
        fig.tight_layout()
        fig.savefig(
            outdir / f"representative_gene_{safe_name(symbol)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print("=" * 88)
    print("468-GENE EXPRESSION-GEOMETRY CHARACTERIZATION")
    print("=" * 88)
    print(f"Target genes: {len(metrics):,}")
    print(f"Endpoint cohort: {endpoint[sample_col].nunique():,}")
    print(f"Graded cohort: {graded[sample_col].nunique():,}")
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nREPRESENTATIVE GENES")
    print(
        reps[
            [
                "gene_symbol",
                "gene_id",
                "DESeq2_BH",
                "DESeq2_log2FoldChange",
                "Regular_BH",
                "Regular_Z",
                "Weighted_BH",
                "Weighted_Z",
                "cohen_d_AD_minus_NCI",
                "rank_biserial_AD_minus_NCI",
                "spearman_severity_vs_expression",
                "median_trajectory",
            ]
        ].to_string(index=False)
    )
    print("\nWROTE")
    print(f"  {metrics_out}")
    print(f"  {summary_out}")
    print(f"  {reps_out}")
    print(f"  {outdir / 'deseq2_fdr_distribution.png'}")
    print(f"  {outdir / 'regular_vs_weighted_z.png'}")
    print(f"  {outdir / 'rank_vs_mean_effect.png'}")
    print(f"  {outdir / 'graded_monotonicity_summary.png'}")
    print(f"  {len(reps)} representative-gene plots")


if __name__ == "__main__":
    main()

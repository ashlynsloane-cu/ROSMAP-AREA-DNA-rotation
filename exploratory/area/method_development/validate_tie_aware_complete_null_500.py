#!/usr/bin/env python3
"""
validate_tie_aware_complete_null_500.py
=======================================

Genome-wide complete-null validation for the corrected tie-aware Weighted AREA.

This script validates SIX phenotype encodings on matched RNA-seq sample sets:

AMYLOID AXIS
------------
1. CERAD_equal
2. CERAD_amyloid_calibrated
3. amyloid_continuous

TAU AXIS
--------
1. Braak_equal
2. Braak_tangle_calibrated
3. tangle_continuous

Design
------
For each pathology axis:

1. Restrict to the exact three-way complete-case RNA-seq sample set.
2. For every gene, compute ASCENDING average expression ranks.
   Exact expression ties therefore receive the same rank.
3. Center those ranks within gene.
4. For each phenotype representation, center the phenotype scores.
5. For each of 500 OUTER null replicates:
      - apply ONE shared random phenotype permutation to all three
        representations within the axis;
      - compute the centered linear-rank statistic genome-wide;
      - standardize using the exact permutation variance:

          Var(C_g) =
              sum_i (w_i - mean(w))^2
              * sum_i (r_gi - mean(r_g))^2
              / (n - 1)

      - convert Z to two-sided Gaussian p-values;
      - BH-adjust across all tested genes.

Under the complete null, every discovery is false, so:

    complete-null FDR = P(R > 0)

where R is the number of BH discoveries.

Outputs
-------
For each axis:
    replicate_results.csv
    event_hit_counts.csv
    marginal_p_calibration.csv
    burst_summary.csv
    method_summary.csv
    significant_genes/outer_XXXX__METHOD.csv  [only when R > 0]

At root:
    MASTER_complete_null_summary.csv
    validation_manifest.json

Important
---------
- This is a validation of the NEW tie-aware Weighted AREA inference.
- It does NOT modify the original Dowell-Lab Regular AREA implementation.
- All three phenotype representations within an axis use IDENTICAL samples
  and IDENTICAL outer permutations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc


AMYLOID_METHODS = {
    "CERAD_equal": "CERAD_equal_weight",
    "CERAD_amyloid_calibrated": "CERAD_amyloid_calibrated_weight",
    "amyloid_continuous": "amyloid_continuous_AREA_weight",
}

TAU_METHODS = {
    "Braak_equal": "Braak_equal_weight",
    "Braak_tangle_calibrated": "Braak_tangle_calibrated_weight",
    "tangle_continuous": "tangle_continuous_AREA_weight",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--metadata",
        default=(
            "results/pathology_geometry_abeta_tau/metadata/"
            "ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv"
        ),
    )
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--expression-id-col", default=None)

    p.add_argument("--outer-permutations", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--fdr-threshold", type=float, default=0.05)

    p.add_argument(
        "--rank-chunk-size",
        type=int,
        default=2000,
        help="Genes per chunk while computing average ranks.",
    )

    p.add_argument(
        "--outdir",
        default="results/ams_area_validation/tie_aware_complete_null_500",
    )

    return p.parse_args()


def stable_seed(base_seed, label):
    h = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(h[:8], 16)) % (2**32 - 1)


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)

    order = np.argsort(p, kind="mergesort")
    ranked = p[order]

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = q
    return out


def wilson_interval(successes, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan

    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z * z / n

    center = (
        phat
        + z * z / (2 * n)
    ) / denom

    half = (
        z
        * math.sqrt(
            phat * (1 - phat) / n
            + z * z / (4 * n * n)
        )
        / denom
    )

    return max(0.0, center - half), min(1.0, center + half)


def load_expression(path, explicit_id_col):
    d = pd.read_csv(path)

    id_col = explicit_id_col or d.columns[0]

    if id_col not in d.columns:
        raise ValueError(f"Expression ID column '{id_col}' not found.")

    sample_ids = d[id_col].astype(str).str.strip()

    if sample_ids.duplicated().any():
        raise ValueError("Expression sample IDs are duplicated.")

    genes = [c for c in d.columns if c != id_col]

    xdf = d[genes].apply(pd.to_numeric, errors="coerce")

    if xdf.isna().any().any():
        bad = xdf.columns[xdf.isna().any()].tolist()[:10]
        raise ValueError(
            "Expression contains missing/non-numeric values. "
            f"Example genes: {bad}"
        )

    return (
        pd.Index(sample_ids),
        np.asarray(genes, dtype=str),
        xdf.to_numpy(dtype=np.float64),
        id_col,
    )


def align_axis(
    expression_ids,
    x_all,
    metadata,
    sample_col,
    complete_flag,
    method_map,
):
    needed = [sample_col, complete_flag] + list(method_map.values())

    missing = [c for c in needed if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    d = metadata.loc[
        metadata[complete_flag].astype(bool),
        [sample_col] + list(method_map.values()),
    ].copy()

    d[sample_col] = d[sample_col].astype(str).str.strip()

    if d[sample_col].duplicated().any():
        raise ValueError(
            f"Duplicate sample IDs in matched set '{complete_flag}'."
        )

    pos = pd.Series(
        np.arange(len(expression_ids), dtype=int),
        index=expression_ids.astype(str),
    )

    missing_ids = d.loc[
        ~d[sample_col].isin(pos.index),
        sample_col,
    ]

    if len(missing_ids):
        raise ValueError(
            f"{len(missing_ids)} matched metadata samples are absent "
            "from expression."
        )

    rows = pos.loc[d[sample_col]].to_numpy(dtype=int)
    x = x_all[rows, :]

    wdf = d[list(method_map.values())].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if wdf.isna().any().any():
        raise ValueError(
            f"Matched set '{complete_flag}' contains NA/non-numeric weights."
        )

    W = wdf.to_numpy(dtype=np.float64)

    if np.any(~np.isfinite(W)):
        raise ValueError("Phenotype weights contain non-finite values.")

    return d[sample_col].to_numpy(dtype=str), x, W


def compute_centered_average_ranks(x, chunk_size):
    """
    x: samples x genes

    Returns
    -------
    R : genes x samples, float32
        Average expression ranks centered at (n+1)/2.
    rank_ss : genes, float64
        Sum of squared centered average ranks for each gene.
    tie_fraction : genes, float64
        Fraction of samples belonging to an exact-expression tie group.
    max_tie_group : genes, int
    """
    n_samples, n_genes = x.shape
    rank_mean = (n_samples + 1.0) / 2.0

    R = np.empty(
        (n_genes, n_samples),
        dtype=np.float32,
    )

    rank_ss = np.empty(n_genes, dtype=np.float64)
    tie_fraction = np.empty(n_genes, dtype=np.float64)
    max_tie_group = np.empty(n_genes, dtype=np.int32)

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)

        for gene_idx in range(start, end):
            values = x[:, gene_idx]

            ranks = stats.rankdata(
                values,
                method="average",
            ).astype(np.float64)

            centered = ranks - rank_mean

            R[gene_idx, :] = centered.astype(np.float32)
            rank_ss[gene_idx] = float(np.dot(centered, centered))

            _, counts = np.unique(
                values,
                return_counts=True,
            )

            tied = counts[counts > 1]

            if len(tied):
                tie_fraction[gene_idx] = float(
                    np.sum(tied) / n_samples
                )
                max_tie_group[gene_idx] = int(np.max(tied))
            else:
                tie_fraction[gene_idx] = 0.0
                max_tie_group[gene_idx] = 1

        print(
            f"    ranked genes: {end:,}/{n_genes:,}"
        )

    return R, rank_ss, tie_fraction, max_tie_group


def validate_rank_matrix(R, rank_ss):
    row_sums = np.asarray(
        R.sum(axis=1),
        dtype=np.float64,
    )

    max_abs_sum = float(np.max(np.abs(row_sums)))

    if max_abs_sum > 1e-4:
        raise RuntimeError(
            "Centered rank rows do not sum approximately to zero. "
            f"max abs sum={max_abs_sum}"
        )

    if np.any(rank_ss <= 0):
        bad = int(np.sum(rank_ss <= 0))
        raise RuntimeError(
            f"{bad} genes have non-positive rank variance."
        )

    return max_abs_sum


def run_axis(
    axis_name,
    complete_flag,
    method_map,
    expression_ids,
    gene_ids,
    x_all,
    metadata,
    args,
    root,
):
    axis_dir = root / axis_name
    axis_dir.mkdir(parents=True, exist_ok=True)

    sig_dir = axis_dir / "significant_genes"
    sig_dir.mkdir(exist_ok=True)

    method_names = list(method_map)

    sample_ids, x, W = align_axis(
        expression_ids,
        x_all,
        metadata,
        args.sample_col,
        complete_flag,
        method_map,
    )

    n_samples = len(sample_ids)
    n_genes = len(gene_ids)
    n_methods = len(method_names)

    print("\n" + "=" * 80)
    print(axis_name.upper())
    print("=" * 80)
    print(f"Matched samples: {n_samples:,}")
    print(f"Genes tested:    {n_genes:,}")
    print(f"Methods:         {', '.join(method_names)}")

    pd.DataFrame({
        "position": np.arange(n_samples),
        "sample_id": sample_ids,
    }).to_csv(
        axis_dir / "matched_sample_manifest.csv",
        index=False,
    )

    print("\nComputing tie-aware average expression ranks...")
    R, rank_ss, tie_fraction, max_tie_group = (
        compute_centered_average_ranks(
            x,
            args.rank_chunk_size,
        )
    )

    max_rank_center_error = validate_rank_matrix(
        R,
        rank_ss,
    )

    tie_df = pd.DataFrame({
        "gene_id": gene_ids,
        "rank_sum_squares": rank_ss,
        "fraction_samples_in_ties": tie_fraction,
        "max_tie_group_size": max_tie_group,
    })

    tie_df.to_csv(
        axis_dir / "gene_tie_diagnostics.csv",
        index=False,
    )

    # Center phenotype scores. The phenotype sum-of-squares is invariant
    # to outer permutation.
    Wc = W - np.mean(W, axis=0, keepdims=True)

    phenotype_ss = np.sum(
        Wc * Wc,
        axis=0,
    )

    if np.any(phenotype_ss <= 0):
        raise ValueError(
            "One or more phenotype representations are constant."
        )

    # Gene x method exact permutation SDs.
    sd = np.sqrt(
        rank_ss[:, None]
        * phenotype_ss[None, :]
        / (n_samples - 1)
    )

    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise RuntimeError(
            "Invalid analytic tie-aware null SD."
        )

    rng = np.random.default_rng(
        stable_seed(
            args.seed,
            axis_name + "_outer_null",
        )
    )

    replicate_rows = []

    # Running marginal-p calibration accumulators.
    nominal_thresholds = [0.05, 0.01, 0.001]
    nominal_counts = {
        method: {thr: 0 for thr in nominal_thresholds}
        for method in method_names
    }

    for outer in range(1, args.outer_permutations + 1):
        perm = rng.permutation(n_samples)

        # Same outer phenotype permutation for all methods in this axis.
        Wperm = Wc[perm, :]

        # C = sum centered_rank * permuted_centered_weight.
        # R is genes x samples, Wperm is samples x methods.
        C = R @ Wperm

        # Orientation follows prior tie-aware validation:
        # positive Z means phenotype enrichment toward lower expression
        # under ascending expression ranking.
        Z = -C / sd

        P = erfc(
            np.abs(Z)
            / math.sqrt(2.0)
        )

        for j, method in enumerate(method_names):
            p = P[:, j]
            q = bh_adjust(p)

            sig = q < args.fdr_threshold
            n_sig = int(np.sum(sig))

            row = {
                "axis": axis_name,
                "outer_replicate": outer,
                "method": method,
                "n_samples": n_samples,
                "n_genes": n_genes,
                "n_fdr_sig": n_sig,
                "fraction_p_lt_0.05": float(np.mean(p < 0.05)),
                "fraction_p_lt_0.01": float(np.mean(p < 0.01)),
                "fraction_p_lt_0.001": float(np.mean(p < 0.001)),
                "min_p": float(np.min(p)),
                "min_fdr": float(np.min(q)),
                "max_abs_z": float(np.max(np.abs(Z[:, j]))),
            }

            replicate_rows.append(row)

            for thr in nominal_thresholds:
                nominal_counts[method][thr] += int(np.sum(p < thr))

            if n_sig > 0:
                event = pd.DataFrame({
                    "gene_id": gene_ids[sig],
                    "z": Z[sig, j],
                    "pvalue": p[sig],
                    "padj": q[sig],
                    "fraction_samples_in_ties": tie_fraction[sig],
                    "max_tie_group_size": max_tie_group[sig],
                }).sort_values(
                    ["padj", "pvalue"],
                    kind="mergesort",
                )

                event.to_csv(
                    sig_dir
                    / f"outer_{outer:04d}__{method}.csv",
                    index=False,
                )

        if (
            outer % 25 == 0
            or outer == args.outer_permutations
        ):
            print(
                f"    outer nulls: "
                f"{outer:,}/{args.outer_permutations:,}"
            )

    replicate_df = pd.DataFrame(
        replicate_rows
    )

    replicate_df.to_csv(
        axis_dir / "replicate_results.csv",
        index=False,
    )

    # Event hit table: one row per outer replicate, one column per method.
    event_hits = (
        replicate_df.pivot(
            index="outer_replicate",
            columns="method",
            values="n_fdr_sig",
        )
        .reset_index()
    )

    event_hits.columns.name = None

    event_hits.to_csv(
        axis_dir / "event_hit_counts.csv",
        index=False,
    )

    method_summary_rows = []
    burst_rows = []
    marginal_rows = []

    for method in method_names:
        d = replicate_df.loc[
            replicate_df["method"] == method
        ].copy()

        hits = d["n_fdr_sig"].to_numpy(dtype=int)

        n_any = int(np.sum(hits > 0))
        fdr_complete_null = n_any / args.outer_permutations
        ci_low, ci_high = wilson_interval(
            n_any,
            args.outer_permutations,
        )

        method_summary_rows.append({
            "axis": axis_name,
            "method": method,
            "outer_permutations": args.outer_permutations,
            "complete_null_fdr_P_R_gt_0": fdr_complete_null,
            "n_outer_with_any_BH_hit": n_any,
            "wilson95_low": ci_low,
            "wilson95_high": ci_high,
            "mean_n_fdr_sig": float(np.mean(hits)),
            "median_n_fdr_sig": float(np.median(hits)),
            "p95_n_fdr_sig": float(np.quantile(hits, 0.95)),
            "p99_n_fdr_sig": float(np.quantile(hits, 0.99)),
            "max_n_fdr_sig": int(np.max(hits)),
            "mean_fraction_p_lt_0.05": float(
                d["fraction_p_lt_0.05"].mean()
            ),
            "mean_fraction_p_lt_0.01": float(
                d["fraction_p_lt_0.01"].mean()
            ),
            "mean_fraction_p_lt_0.001": float(
                d["fraction_p_lt_0.001"].mean()
            ),
        })

        for threshold in [1, 10, 100, 250, 500, 1000]:
            count = int(np.sum(hits >= threshold))

            burst_rows.append({
                "axis": axis_name,
                "method": method,
                "threshold_n_fdr_sig": threshold,
                "n_outer_replicates": count,
                "fraction_outer_replicates": (
                    count / args.outer_permutations
                ),
            })

        total_gene_tests = (
            args.outer_permutations
            * n_genes
        )

        for thr in nominal_thresholds:
            observed = (
                nominal_counts[method][thr]
                / total_gene_tests
            )

            marginal_rows.append({
                "axis": axis_name,
                "method": method,
                "p_threshold": thr,
                "observed_fraction": observed,
                "expected_fraction": thr,
                "observed_minus_expected": observed - thr,
                "ratio_observed_to_expected": observed / thr,
            })

    method_summary = pd.DataFrame(
        method_summary_rows
    )

    method_summary.to_csv(
        axis_dir / "method_summary.csv",
        index=False,
    )

    burst_df = pd.DataFrame(
        burst_rows
    )

    burst_df.to_csv(
        axis_dir / "burst_summary.csv",
        index=False,
    )

    marginal_df = pd.DataFrame(
        marginal_rows
    )

    marginal_df.to_csv(
        axis_dir / "marginal_p_calibration.csv",
        index=False,
    )

    print("\nComplete-null summary:")
    print(
        method_summary[
            [
                "method",
                "complete_null_fdr_P_R_gt_0",
                "n_outer_with_any_BH_hit",
                "wilson95_low",
                "wilson95_high",
                "mean_n_fdr_sig",
                "max_n_fdr_sig",
                "mean_fraction_p_lt_0.05",
                "mean_fraction_p_lt_0.01",
                "mean_fraction_p_lt_0.001",
            ]
        ].to_string(index=False)
    )

    return {
        "axis": axis_name,
        "n_samples": n_samples,
        "n_genes": n_genes,
        "max_abs_centered_rank_row_sum": max_rank_center_error,
        "fraction_genes_with_any_ties": float(
            np.mean(tie_fraction > 0)
        ),
        "max_fraction_samples_in_ties": float(
            np.max(tie_fraction)
        ),
        "max_tie_group_size": int(
            np.max(max_tie_group)
        ),
        "method_summary": method_summary,
    }


def main():
    a = parse_args()

    root = Path(a.outdir)
    root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        a.metadata
    )

    (
        expression_ids,
        gene_ids,
        x_all,
        expression_id_col,
    ) = load_expression(
        a.expression,
        a.expression_id_col,
    )

    print("=" * 80)
    print("TIE-AWARE WEIGHTED AREA: 500 COMPLETE-NULL VALIDATION")
    print("=" * 80)
    print(
        f"Expression: {len(expression_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )
    print(
        f"Outer null permutations: {a.outer_permutations:,}"
    )
    print(
        f"BH threshold: {a.fdr_threshold}"
    )

    axis_results = []

    axis_results.append(
        run_axis(
            axis_name="amyloid_axis",
            complete_flag="CERAD_threeway_complete",
            method_map=AMYLOID_METHODS,
            expression_ids=expression_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            metadata=metadata,
            args=a,
            root=root,
        )
    )

    axis_results.append(
        run_axis(
            axis_name="tau_axis",
            complete_flag="Braak_threeway_complete",
            method_map=TAU_METHODS,
            expression_ids=expression_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            metadata=metadata,
            args=a,
            root=root,
        )
    )

    master = pd.concat(
        [
            x["method_summary"]
            for x in axis_results
        ],
        ignore_index=True,
    )

    master.to_csv(
        root / "MASTER_complete_null_summary.csv",
        index=False,
    )

    with open(
        root / "validation_manifest.json",
        "w",
    ) as f:
        json.dump(
            {
                "expression": str(
                    Path(a.expression).resolve()
                ),
                "metadata": str(
                    Path(a.metadata).resolve()
                ),
                "expression_id_col": expression_id_col,
                "sample_col": a.sample_col,
                "outer_permutations": a.outer_permutations,
                "seed": a.seed,
                "fdr_threshold": a.fdr_threshold,
                "pvalue_method": (
                    "Two-sided Gaussian p-values from the exact random-"
                    "permutation variance of the centered average-rank "
                    "linear statistic."
                ),
                "tie_handling": (
                    "Exact expression ties receive average ranks. "
                    "No arbitrary within-tie sample ordering is used."
                ),
                "outer_null": (
                    "Phenotype scores are jointly permuted across participants. "
                    "The same permutation is used for all three phenotype "
                    "representations within each biological axis."
                ),
                "axes": [
                    {
                        "axis": x["axis"],
                        "n_samples": x["n_samples"],
                        "n_genes": x["n_genes"],
                        "fraction_genes_with_any_ties": (
                            x["fraction_genes_with_any_ties"]
                        ),
                        "max_fraction_samples_in_ties": (
                            x["max_fraction_samples_in_ties"]
                        ),
                        "max_tie_group_size": (
                            x["max_tie_group_size"]
                        ),
                    }
                    for x in axis_results
                ],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("MASTER COMPLETE-NULL SUMMARY")
    print("=" * 80)

    print(
        master[
            [
                "axis",
                "method",
                "complete_null_fdr_P_R_gt_0",
                "n_outer_with_any_BH_hit",
                "wilson95_low",
                "wilson95_high",
                "mean_n_fdr_sig",
                "max_n_fdr_sig",
                "mean_fraction_p_lt_0.05",
                "mean_fraction_p_lt_0.01",
                "mean_fraction_p_lt_0.001",
            ]
        ].to_string(index=False)
    )

    print("\nWrote:")
    print(
        root
        / "MASTER_complete_null_summary.csv"
    )

    print(
        "\nInterpretation: under the complete null, FDR equals P(R>0). "
        "At q=0.05, we want the fraction of outer replicates with any BH "
        "discoveries to be approximately <= 0.05, with marginal p-value "
        "fractions close to their nominal thresholds."
    )


if __name__ == "__main__":
    main()

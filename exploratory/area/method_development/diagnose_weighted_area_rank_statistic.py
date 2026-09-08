#!/usr/bin/env python3
"""
diagnose_weighted_area_rank_statistic.py
=========================================

Method-development audit for Weighted AREA.

This script answers four questions BEFORE inspecting biological hit counts:

1. Does the exact production Weighted AREA score satisfy the derived
   linear-rank identity?

2. Are the inferential results invariant to positive affine recodings of the
   phenotype (e.g. x, 7x, x+10, 3x+5), even though the raw ES may change?

3. How closely is Weighted AREA related to the Cuzick / Wilcoxon-type linear
   rank statistic?

4. Does stable sorting of tied expression values introduce sample-order
   dependence relative to conventional average-rank handling?

Exact identity used
-------------------
Let samples be ordered by ASCENDING expression for one gene, with phenotype
weights w_(1), ..., w_(n), total S = sum(w), and ordinary rank positions
r = 1, ..., n.

Define the individual-level Cuzick-like centered linear-rank statistic

    T = (1/n) * sum_i [(w_i - mean(w)) * r_i]

where r_i is the stable expression rank assigned to sample i.

For the CURRENT production trapezoidal Weighted AREA implementation,

    Weighted_AREA_ES
      = 1/n - 2*T/S - w_(1)/(n*S)

up to floating-point precision.

Thus the current score is a scaled/reoriented centered linear-rank statistic
plus a one-sample endpoint term caused by the trapezoidal cumulative-curve
convention.

Affine recoding
---------------
For w' = a*w + c with a > 0 and valid nonnegative transformed weights, every
observed/permuted AREA score is a positive affine transform of the corresponding
score under w. Consequently, when the null is rebuilt for the transformed
weights:

    - Gaussian permutation Z is invariant
    - Gaussian two-sided p is invariant
    - centered empirical permutation p is invariant

Raw ES need not be invariant to additive shifts.

Cuzick comparison
-----------------
Classic Cuzick uses ranked responses and centered numerical group scores. With
individual-level phenotype scores, the same core linear-rank statistic is:

    T_Cuzick = mean[(w - mean(w)) * rank(expression)]

This script reports both:
    - stable ranks (matching production tie-breaking)
    - average ranks (conventional handling of expression ties)

No pathways and no "best" phenotype representation are selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc


DEFAULT_WEIGHTS = [
    "CERAD_equal_weight",
    "CERAD_amyloid_calibrated_weight",
    "amyloid_continuous_AREA_weight",
    "Braak_equal_weight",
    "Braak_tangle_calibrated_weight",
    "tangle_continuous_AREA_weight",
]


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
    p.add_argument(
        "--area-script",
        default="exploratory/area/run_ams_area.py",
    )
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--expression-id-col", default=None)

    p.add_argument(
        "--weight-cols",
        default=",".join(DEFAULT_WEIGHTS),
        help="Comma-separated phenotype-weight columns to audit.",
    )

    p.add_argument(
        "--n-genes",
        type=int,
        default=1000,
        help="Number of genes sampled for rank/tie diagnostics.",
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Paired phenotype permutations for affine-invariance audit.",
    )
    p.add_argument("--seed", type=int, default=20260907)

    p.add_argument(
        "--outdir",
        default="results/ams_area_validation/rank_statistic_diagnostic",
    )

    return p.parse_args()


def import_production_score(path):
    path = Path(path).resolve()

    spec = importlib.util.spec_from_file_location(
        "run_ams_area_current",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "compute_weighted_enrichment_score"):
        raise AttributeError(
            f"{path} lacks compute_weighted_enrichment_score()."
        )

    return mod.compute_weighted_enrichment_score


def stable_seed(base, label):
    h = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (int(base) + int(h[:8], 16)) % (2**32 - 1)


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


def empirical_two_sided_p(observed, null):
    mu = float(np.mean(null))
    null_abs = np.sort(np.abs(null - mu))
    obs_abs = np.abs(np.asarray(observed, dtype=float) - mu)

    left = np.searchsorted(null_abs, obs_abs, side="left")
    n_ge = len(null_abs) - left

    return (n_ge + 1.0) / (len(null_abs) + 1.0)


def load_expression(path, explicit_id_col):
    d = pd.read_csv(path)

    id_col = explicit_id_col or d.columns[0]

    if id_col not in d.columns:
        raise ValueError(
            f"Expression ID column '{id_col}' not found."
        )

    ids = d[id_col].astype(str).str.strip()

    if ids.duplicated().any():
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
        pd.Index(ids),
        np.asarray(genes, dtype=str),
        xdf.to_numpy(dtype=float),
        id_col,
    )


def align_weight(
    expression_ids,
    x_all,
    metadata,
    sample_col,
    weight_col,
):
    if sample_col not in metadata.columns:
        raise ValueError(
            f"Metadata lacks sample column '{sample_col}'."
        )

    if weight_col not in metadata.columns:
        raise ValueError(
            f"Metadata lacks requested weight '{weight_col}'."
        )

    d = metadata[
        [sample_col, weight_col]
    ].copy()

    d[weight_col] = pd.to_numeric(
        d[weight_col],
        errors="coerce",
    )

    d = d.dropna(
        subset=[sample_col, weight_col]
    )

    d[sample_col] = (
        d[sample_col]
        .astype(str)
        .str.strip()
    )

    if d[sample_col].duplicated().any():
        raise ValueError(
            f"Duplicate sample IDs for {weight_col}."
        )

    position = pd.Series(
        np.arange(len(expression_ids), dtype=int),
        index=expression_ids.astype(str),
    )

    missing = d.loc[
        ~d[sample_col].isin(position.index),
        sample_col,
    ]

    if len(missing):
        raise ValueError(
            f"{len(missing)} metadata samples for {weight_col} "
            "are absent from expression."
        )

    rows = position.loc[
        d[sample_col]
    ].to_numpy(dtype=int)

    x = x_all[rows, :]
    w = d[weight_col].to_numpy(dtype=float)

    if np.any(~np.isfinite(w)):
        raise ValueError(
            f"{weight_col} contains NA/Inf after filtering."
        )

    if np.min(w) < -1e-12:
        raise ValueError(
            f"{weight_col} contains negative values."
        )

    if np.sum(w) <= 0:
        raise ValueError(
            f"{weight_col} has zero total weight."
        )

    return d[sample_col].to_numpy(dtype=str), x, w


def stable_rank_and_order(values):
    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    ranks[order] = np.arange(
        1,
        len(values) + 1,
        dtype=float,
    )

    return ranks, order


def reverse_tie_order(values):
    """
    Same ascending expression ordering, but the opposite deterministic
    sample-index order within exact ties.

    Production stable mergesort resolves ties according to incoming sample
    order. This alternative quantifies how much that choice can matter.
    """
    idx = np.arange(
        len(values),
        dtype=int,
    )

    # np.lexsort: last key is primary.
    return np.lexsort(
        (
            -idx,
            values,
        )
    )


def tie_metrics(values):
    _, counts = np.unique(
        values,
        return_counts=True,
    )

    tied = counts[
        counts > 1
    ]

    return {
        "n_unique_expression": int(len(counts)),
        "n_tie_groups": int(len(tied)),
        "fraction_samples_in_ties": (
            float(np.sum(tied) / len(values))
            if len(tied)
            else 0.0
        ),
        "max_tie_group_size": (
            int(np.max(tied))
            if len(tied)
            else 1
        ),
        "has_any_tie": bool(
            len(tied) > 0
        ),
    }


def exact_rank_identity(
    weights,
    stable_ranks,
    first_weight,
):
    """
    Exact algebraic representation of current production Weighted AREA.
    """
    n = len(weights)
    total = float(np.sum(weights))
    centered = weights - np.mean(weights)

    T = float(
        np.mean(
            centered
            * stable_ranks
        )
    )

    es = (
        1.0 / n
        - 2.0 * T / total
        - first_weight / (n * total)
    )

    return T, es


def average_rank_cuzick(
    expression,
    weights,
):
    ranks = stats.rankdata(
        expression,
        method="average",
    )

    return float(
        np.mean(
            (
                weights
                - np.mean(weights)
            )
            * ranks
        )
    )


def generate_permutation_indices(
    n,
    permutations,
    seed,
):
    rng = np.random.default_rng(seed)

    out = np.empty(
        (permutations, n),
        dtype=np.int32,
    )

    for i in range(permutations):
        out[i] = rng.permutation(n)

    return out


def transform_specs():
    return [
        ("base", 1.0, 0.0),
        ("scale_7x", 7.0, 0.0),
        ("shift_plus_10", 1.0, 10.0),
        ("affine_3x_plus_5", 3.0, 5.0),
    ]


def score_null(
    production_score,
    weights,
    permutation_indices,
):
    return np.asarray(
        [
            production_score(
                weights[p]
            )
            for p in permutation_indices
        ],
        dtype=float,
    )


def gaussian_from_null(
    observed,
    null,
):
    mu = float(np.mean(null))
    sd = float(np.std(null))  # production ddof=0

    if sd <= 0:
        raise ValueError("Permutation null has zero SD.")

    z = (
        np.asarray(observed, dtype=float)
        - mu
    ) / sd

    p = erfc(
        np.abs(z)
        / math.sqrt(2.0)
    )

    return mu, sd, z, p


def audit_one_weight(
    production_score,
    weight_col,
    genes,
    x,
    weights,
    gene_indices,
    permutations,
    seed,
    outdir,
):
    n = len(weights)
    total = float(np.sum(weights))

    if np.min(weights) < 0:
        raise ValueError(
            f"{weight_col}: negative base weights."
        )

    print("\n" + "=" * 80)
    print(weight_col)
    print("=" * 80)
    print(f"Samples: {n:,}")
    print(f"Weight min/max: {np.min(weights):.6g} / {np.max(weights):.6g}")
    print(f"Weight mean:    {np.mean(weights):.6g}")
    print(f"Weight total:   {total:.6g}")

    permutation_indices = generate_permutation_indices(
        n=n,
        permutations=permutations,
        seed=seed,
    )

    # -------------------------------------------------------------
    # Base null.
    # -------------------------------------------------------------
    base_null = score_null(
        production_score,
        weights,
        permutation_indices,
    )

    base_mu = float(np.mean(base_null))
    base_sd = float(np.std(base_null))

    # -------------------------------------------------------------
    # Gene-level exact identity, Cuzick relation, ties.
    # -------------------------------------------------------------
    gene_rows = []

    observed_base = []

    for gene_idx in gene_indices:
        gene = genes[gene_idx]
        expression = x[:, gene_idx]

        stable_ranks, order = stable_rank_and_order(
            expression
        )

        first_weight = float(
            weights[
                order[0]
            ]
        )

        production_es = float(
            production_score(
                weights[
                    order
                ]
            )
        )

        T_stable, identity_es = exact_rank_identity(
            weights,
            stable_ranks,
            first_weight,
        )

        T_average = average_rank_cuzick(
            expression,
            weights,
        )

        alt_order = reverse_tie_order(
            expression
        )

        alternate_es = float(
            production_score(
                weights[
                    alt_order
                ]
            )
        )

        tie = tie_metrics(
            expression
        )

        observed_base.append(
            production_es
        )

        gene_rows.append({
            "gene_id": gene,
            "production_ES": production_es,
            "identity_ES": identity_es,
            "identity_abs_diff": abs(
                production_es
                - identity_es
            ),
            "cuzick_T_stable_rank": T_stable,
            "cuzick_T_average_rank": T_average,
            "endpoint_weight_lowest_expression": first_weight,
            "alternate_reverse_tie_ES": alternate_es,
            "tie_break_delta_ES": (
                alternate_es
                - production_es
            ),
            "tie_break_delta_Z_using_base_null": (
                (
                    alternate_es
                    - production_es
                )
                / base_sd
            ),
            **tie,
        })

    gene_df = pd.DataFrame(
        gene_rows
    )

    gene_df.to_csv(
        outdir
        / f"{weight_col}__gene_level_rank_diagnostic.csv",
        index=False,
    )

    observed_base = np.asarray(
        observed_base,
        dtype=float,
    )

    # -------------------------------------------------------------
    # Rank relation summary.
    # -------------------------------------------------------------
    stable_T = gene_df[
        "cuzick_T_stable_rank"
    ].to_numpy(dtype=float)

    average_T = gene_df[
        "cuzick_T_average_rank"
    ].to_numpy(dtype=float)

    no_tie = ~gene_df[
        "has_any_tie"
    ].to_numpy(dtype=bool)

    rank_summary = {
        "weight_col": weight_col,
        "n_samples": n,
        "n_genes_tested": len(gene_df),
        "exact_identity_max_abs_diff": float(
            gene_df[
                "identity_abs_diff"
            ].max()
        ),
        # AREA sign is opposite the usual positive Cuzick direction.
        "pearson_minus_stable_cuzick_vs_AREA": float(
            stats.pearsonr(
                -stable_T,
                observed_base,
            ).statistic
        ),
        "spearman_minus_stable_cuzick_vs_AREA": float(
            stats.spearmanr(
                -stable_T,
                observed_base,
            ).statistic
        ),
        "pearson_minus_average_cuzick_vs_AREA": float(
            stats.pearsonr(
                -average_T,
                observed_base,
            ).statistic
        ),
        "spearman_minus_average_cuzick_vs_AREA": float(
            stats.spearmanr(
                -average_T,
                observed_base,
            ).statistic
        ),
        "fraction_genes_with_any_expression_tie": float(
            gene_df[
                "has_any_tie"
            ].mean()
        ),
        "median_fraction_samples_in_ties": float(
            gene_df[
                "fraction_samples_in_ties"
            ].median()
        ),
        "max_fraction_samples_in_ties": float(
            gene_df[
                "fraction_samples_in_ties"
            ].max()
        ),
        "max_tie_group_size_over_genes": int(
            gene_df[
                "max_tie_group_size"
            ].max()
        ),
        "max_abs_tie_break_delta_Z": float(
            np.max(
                np.abs(
                    gene_df[
                        "tie_break_delta_Z_using_base_null"
                    ].to_numpy(dtype=float)
                )
            )
        ),
        "p95_abs_tie_break_delta_Z": float(
            np.quantile(
                np.abs(
                    gene_df[
                        "tie_break_delta_Z_using_base_null"
                    ].to_numpy(dtype=float)
                ),
                0.95,
            )
        ),
        "fraction_abs_tie_break_delta_Z_gt_0.01": float(
            np.mean(
                np.abs(
                    gene_df[
                        "tie_break_delta_Z_using_base_null"
                    ].to_numpy(dtype=float)
                )
                > 0.01
            )
        ),
        "fraction_abs_tie_break_delta_Z_gt_0.1": float(
            np.mean(
                np.abs(
                    gene_df[
                        "tie_break_delta_Z_using_base_null"
                    ].to_numpy(dtype=float)
                )
                > 0.1
            )
        ),
    }

    if np.sum(no_tie) >= 3:
        rank_summary[
            "pearson_minus_average_cuzick_vs_AREA_no_ties"
        ] = float(
            stats.pearsonr(
                -average_T[no_tie],
                observed_base[no_tie],
            ).statistic
        )
    else:
        rank_summary[
            "pearson_minus_average_cuzick_vs_AREA_no_ties"
        ] = np.nan

    # -------------------------------------------------------------
    # Affine recoding audit.
    # -------------------------------------------------------------
    affine_rows = []

    base_mu, base_sd, base_z, base_p = gaussian_from_null(
        observed_base,
        base_null,
    )

    base_emp = empirical_two_sided_p(
        observed_base,
        base_null,
    )

    for name, a, c in transform_specs():
        transformed = (
            a * weights
            + c
        )

        if np.min(transformed) < -1e-12:
            raise ValueError(
                f"Transform {name} creates negative weights."
            )

        transformed_null = score_null(
            production_score,
            transformed,
            permutation_indices,
        )

        transformed_obs = []

        for gene_idx in gene_indices:
            order = np.argsort(
                x[:, gene_idx],
                kind="mergesort",
            )

            transformed_obs.append(
                float(
                    production_score(
                        transformed[
                            order
                        ]
                    )
                )
            )

        transformed_obs = np.asarray(
            transformed_obs,
            dtype=float,
        )

        (
            transformed_mu,
            transformed_sd,
            transformed_z,
            transformed_p,
        ) = gaussian_from_null(
            transformed_obs,
            transformed_null,
        )

        transformed_emp = empirical_two_sided_p(
            transformed_obs,
            transformed_null,
        )

        # All raw ES values should be an affine transform of base ES.
        slope, intercept = np.polyfit(
            observed_base,
            transformed_obs,
            1,
        )

        predicted_slope = (
            a * total
            / (
                a * total
                + n * c
            )
        )

        affine_rows.append({
            "weight_col": weight_col,
            "transform": name,
            "a": a,
            "c": c,
            "base_null_mean": base_mu,
            "base_null_sd": base_sd,
            "transformed_null_mean": transformed_mu,
            "transformed_null_sd": transformed_sd,
            "raw_ES_affine_fit_slope": float(slope),
            "raw_ES_affine_fit_intercept": float(intercept),
            "theoretical_raw_ES_slope": float(predicted_slope),
            "raw_ES_slope_abs_error": float(
                abs(
                    slope
                    - predicted_slope
                )
            ),
            "max_abs_delta_gaussian_Z": float(
                np.max(
                    np.abs(
                        transformed_z
                        - base_z
                    )
                )
            ),
            "max_abs_delta_gaussian_P": float(
                np.max(
                    np.abs(
                        transformed_p
                        - base_p
                    )
                )
            ),
            "max_abs_delta_empirical_P": float(
                np.max(
                    np.abs(
                        transformed_emp
                        - base_emp
                    )
                )
            ),
            "pearson_raw_ES_base_vs_transformed": float(
                stats.pearsonr(
                    observed_base,
                    transformed_obs,
                ).statistic
            ),
        })

    affine_df = pd.DataFrame(
        affine_rows
    )

    affine_df.to_csv(
        outdir
        / f"{weight_col}__affine_invariance.csv",
        index=False,
    )

    print("\nExact rank identity:")
    print(
        f"  max |production ES - identity ES| = "
        f"{rank_summary['exact_identity_max_abs_diff']:.3e}"
    )

    print("\nRelation to Cuzick-like rank statistic:")
    print(
        "  Pearson(-stable-rank Cuzick T, AREA ES) = "
        f"{rank_summary['pearson_minus_stable_cuzick_vs_AREA']:.8f}"
    )
    print(
        "  Pearson(-average-rank Cuzick T, AREA ES) = "
        f"{rank_summary['pearson_minus_average_cuzick_vs_AREA']:.8f}"
    )

    print("\nTie sensitivity:")
    print(
        "  genes with any exact expression tie = "
        f"{100 * rank_summary['fraction_genes_with_any_expression_tie']:.2f}%"
    )
    print(
        "  p95 |tie-breaking delta Z| = "
        f"{rank_summary['p95_abs_tie_break_delta_Z']:.6g}"
    )
    print(
        "  max |tie-breaking delta Z| = "
        f"{rank_summary['max_abs_tie_break_delta_Z']:.6g}"
    )

    print("\nAffine recoding:")
    print(
        affine_df[
            [
                "transform",
                "raw_ES_affine_fit_slope",
                "theoretical_raw_ES_slope",
                "max_abs_delta_gaussian_Z",
                "max_abs_delta_gaussian_P",
                "max_abs_delta_empirical_P",
            ]
        ].to_string(index=False)
    )

    return rank_summary, affine_df


def main():
    a = parse_args()

    outdir = Path(
        a.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weight_cols = [
        x.strip()
        for x in a.weight_cols.split(",")
        if x.strip()
    ]

    metadata = pd.read_csv(
        a.metadata
    )

    missing_weights = [
        col
        for col in weight_cols
        if col not in metadata.columns
    ]

    if missing_weights:
        print("Missing requested weight columns:")
        for col in missing_weights:
            print("  -", col)

        print("\nAvailable weight-like columns:")
        for col in metadata.columns:
            if (
                "weight" in col.lower()
                or "braak" in col.lower()
                or "cerad" in col.lower()
                or "amy" in col.lower()
                or "tang" in col.lower()
            ):
                print("  ", col)

        raise SystemExit(2)

    (
        expression_ids,
        genes,
        x_all,
        expression_id_col,
    ) = load_expression(
        a.expression,
        a.expression_id_col,
    )

    production_score = import_production_score(
        a.area_script
    )

    rng = np.random.default_rng(
        a.seed
    )

    n_genes = min(
        a.n_genes,
        len(genes),
    )

    gene_indices = np.sort(
        rng.choice(
            len(genes),
            size=n_genes,
            replace=False,
        )
    )

    pd.DataFrame({
        "gene_index": gene_indices,
        "gene_id": genes[
            gene_indices
        ],
    }).to_csv(
        outdir
        / "sampled_genes.csv",
        index=False,
    )

    print("=" * 80)
    print("WEIGHTED AREA: RANK-STATISTIC / AFFINE-INVARIANCE AUDIT")
    print("=" * 80)
    print(
        f"Expression: {len(expression_ids):,} samples x "
        f"{len(genes):,} genes"
    )
    print(
        f"Sampled genes: {n_genes:,}"
    )
    print(
        f"Affine null permutations per phenotype: {a.permutations:,}"
    )

    rank_rows = []
    affine_all = []

    sample_count_rows = []

    for weight_col in weight_cols:
        sample_ids, x, w = align_weight(
            expression_ids,
            x_all,
            metadata,
            a.sample_col,
            weight_col,
        )

        sample_count_rows.append({
            "weight_col": weight_col,
            "n_samples": int(
                len(sample_ids)
            ),
            "weight_min": float(
                np.min(w)
            ),
            "weight_max": float(
                np.max(w)
            ),
            "weight_mean": float(
                np.mean(w)
            ),
            "weight_sum": float(
                np.sum(w)
            ),
        })

        rank_summary, affine_df = audit_one_weight(
            production_score=production_score,
            weight_col=weight_col,
            genes=genes,
            x=x,
            weights=w,
            gene_indices=gene_indices,
            permutations=a.permutations,
            seed=stable_seed(
                a.seed,
                weight_col,
            ),
            outdir=outdir,
        )

        rank_rows.append(
            rank_summary
        )

        affine_all.append(
            affine_df
        )

    rank_df = pd.DataFrame(
        rank_rows
    )

    rank_df.to_csv(
        outdir
        / "MASTER_rank_statistic_relation_summary.csv",
        index=False,
    )

    affine_master = pd.concat(
        affine_all,
        ignore_index=True,
    )

    affine_master.to_csv(
        outdir
        / "MASTER_affine_invariance_summary.csv",
        index=False,
    )

    sample_df = pd.DataFrame(
        sample_count_rows
    )

    sample_df.to_csv(
        outdir
        / "phenotype_sample_weight_summary.csv",
        index=False,
    )

    with open(
        outdir
        / "diagnostic_manifest.json",
        "w",
    ) as f:
        json.dump(
            {
                "expression": str(
                    Path(
                        a.expression
                    ).resolve()
                ),
                "metadata": str(
                    Path(
                        a.metadata
                    ).resolve()
                ),
                "area_script": str(
                    Path(
                        a.area_script
                    ).resolve()
                ),
                "expression_id_col": expression_id_col,
                "sample_col": a.sample_col,
                "weight_cols": weight_cols,
                "n_genes": n_genes,
                "permutations": a.permutations,
                "seed": a.seed,
                "questions": [
                    "Exact production ES vs derived centered linear-rank identity",
                    "Positive affine phenotype-recoding invariance of Z/p",
                    "Relation to stable-rank and average-rank Cuzick statistics",
                    "Sensitivity to expression-tie ordering",
                ],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("MASTER SUMMARY")
    print("=" * 80)

    print("\nRank-statistic relation:")
    print(
        rank_df[
            [
                "weight_col",
                "n_samples",
                "exact_identity_max_abs_diff",
                "pearson_minus_stable_cuzick_vs_AREA",
                "pearson_minus_average_cuzick_vs_AREA",
                "fraction_genes_with_any_expression_tie",
                "p95_abs_tie_break_delta_Z",
                "max_abs_tie_break_delta_Z",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nAffine invariance:")
    print(
        affine_master[
            [
                "weight_col",
                "transform",
                "raw_ES_affine_fit_slope",
                "theoretical_raw_ES_slope",
                "max_abs_delta_gaussian_Z",
                "max_abs_delta_gaussian_P",
                "max_abs_delta_empirical_P",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nWrote:")
    print(
        outdir
        / "MASTER_rank_statistic_relation_summary.csv"
    )
    print(
        outdir
        / "MASTER_affine_invariance_summary.csv"
    )

    print(
        "\nInterpretation rule: raw ES may change after an additive shift, "
        "but inferential Z/p should not if the transformed null is rebuilt. "
        "Nonlinear recodings are a different question and can change inference."
    )


if __name__ == "__main__":
    main()

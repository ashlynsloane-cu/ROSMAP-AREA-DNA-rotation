#!/usr/bin/env python3
"""
simulate_weighted_area_mixed_signal.py
======================================

Mixed-signal / spike-in validation for Weighted AREA.

Purpose
-------
Complete-null calibration has already tested whether Weighted AREA manufactures
discoveries when no genes are truly associated. This script addresses the next
question:

    When true and null genes coexist, does Weighted AREA recover the true genes
    while controlling the false discovery proportion?

It also tests sensitivity to different underlying biological response shapes.

Default traits
--------------
    Cognitive_stage
    Braak_stage
    CERAD_burden

True signal shapes
------------------
    threshold
        Trait-specific Regular-AREA-like transition:
            Cognitive: current weight > 0
            Braak:     current weight >= 0.5
            CERAD:     current weight >= 2/3

    linear
        response = x

    early_saturation
        response = sqrt(x)

    late_acceleration
        response = x^2

    extreme_stage
        response = 1 only at x == 1, otherwise 0

Analysis encodings
------------------
    current
    sqrt
    square
    exp3
    cube

Simulation design
-----------------
For each trait and replicate:

1. Randomly permute the phenotype across the real ROSMAP samples.
   This creates a pseudo-phenotype that is independent of the original
   expression data while preserving the exact phenotype-level counts.

2. Randomly choose a set of "true" genes from variable genes.

3. Add a controlled synthetic signal to those genes only:
       z(log1p(expression)) + direction * beta * z(true_response)
   Half the effects are positive/negative in expectation.

4. Re-rank samples within each spiked gene.

5. Test ALL 33,006 genes with each Weighted AREA encoding using the exact
   production scoring convention and existing 10k method nulls.

6. Apply BH separately for each encoding and record:
       TP, FP, FN, discoveries
       TPR / power
       FDP
       precision
       null-gene nominal p-value rates

Important guardrails
--------------------
* The SAME pseudo-phenotype, true-gene set, directions, signal shape, and effect
  size are used across all five analysis encodings within a replicate.
* Gene sets / encodings are never selected using biological pathway results.
* This script does NOT do pathway enrichment or choose a "winner."
* Existing real-data expression covariance, sample structure, gene-specific
  distributions, and null genes are preserved.
* Existing 10k paired method nulls are reused because the AREA null depends on
  the phenotype-weight distribution and sample count, not gene identity.

Effect-size interpretation
--------------------------
beta is measured in within-gene log1p-expression SD units per 1 SD of the
simulated response. Because AREA is rank-based, beta should be viewed as a
controlled simulation strength, not as a fold change.

Recommended first run
---------------------
50 replicates, 500 true genes, beta = 0.25,0.50,0.75.

If needed, increase to 100 replicates with the SAME base seed. The first 50
replicate definitions remain identical.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfc


DEFAULT_TRAITS = (
    "Cognitive_stage",
    "Braak_stage",
    "CERAD_burden",
)

ENCODINGS = (
    "current",
    "sqrt",
    "square",
    "exp3",
    "cube",
)

ENCODING_ROLE = {
    "current": "primary",
    "sqrt": "sensitivity",
    "square": "sensitivity",
    "exp3": "sensitivity",
    "cube": "stress_test",
}

DEFAULT_SHAPES = (
    "threshold",
    "linear",
    "early_saturation",
    "late_acceleration",
    "extreme_stage",
)

THRESHOLD_RULE = {
    "Cognitive_stage": ("gt", 0.0),
    "Braak_stage": ("ge", 0.5),
    "CERAD_burden": ("ge", 2.0 / 3.0),
}


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Mixed-signal spike-in validation for Weighted AREA."
    )

    p.add_argument(
        "--traits",
        default=",".join(DEFAULT_TRAITS),
    )
    p.add_argument(
        "--shapes",
        default=",".join(DEFAULT_SHAPES),
    )
    p.add_argument(
        "--effect-sizes",
        default="0.25,0.50,0.75",
        help="Comma-separated beta values in within-gene log1p-expression SD units.",
    )
    p.add_argument(
        "--replicates",
        type=int,
        default=50,
    )
    p.add_argument(
        "--true-genes",
        type=int,
        default=500,
    )
    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--area-script",
        default="exploratory/area/run_ams_area.py",
    )
    p.add_argument(
        "--pilot-screen-root",
        default="results/ams_area_validation/weighted_A_to_D/paired_screen",
    )
    p.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260906,
    )
    p.add_argument(
        "--outdir",
        default=(
            "results/ams_area_validation/weighted_A_to_D/"
            "mixed_signal_simulation"
        ),
    )

    return p.parse_args()


# =============================================================================
# Production AREA import + phenotype transforms
# =============================================================================

def import_weighted_score(path):
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


def transform_analysis_weights(v, encoding):
    """
    Pre-specified cohort-independent Weighted AREA encodings.
    """
    v = np.asarray(v, dtype=float)

    if encoding == "current":
        return v.copy()

    lo = float(np.min(v))
    hi = float(np.max(v))

    if np.isclose(lo, hi):
        raise ValueError("Phenotype vector has no variation.")

    x = (v - lo) / (hi - lo)

    if encoding == "sqrt":
        return np.sqrt(x)

    if encoding == "square":
        return x ** 2

    if encoding == "exp3":
        return np.expm1(3.0 * x) / np.expm1(3.0)

    if encoding == "cube":
        return x ** 3

    raise ValueError(f"Unknown encoding: {encoding}")


def true_response(current_weights, trait, shape):
    """
    Biological response used to CREATE the spike-in signal.

    This is distinct from the analysis encoding used to TEST the signal.
    """
    x = np.asarray(current_weights, dtype=float)

    lo = float(np.min(x))
    hi = float(np.max(x))
    if np.isclose(lo, hi):
        raise ValueError("Phenotype vector has no variation.")

    x01 = (x - lo) / (hi - lo)

    if shape == "linear":
        y = x01

    elif shape == "early_saturation":
        y = np.sqrt(x01)

    elif shape == "late_acceleration":
        y = x01 ** 2

    elif shape == "extreme_stage":
        y = np.isclose(x01, 1.0).astype(float)

    elif shape == "threshold":
        if trait not in THRESHOLD_RULE:
            raise ValueError(
                f"No threshold rule defined for trait '{trait}'."
            )

        op, cutoff = THRESHOLD_RULE[trait]

        if op == "gt":
            y = (x01 > cutoff).astype(float)
        elif op == "ge":
            y = (x01 >= cutoff).astype(float)
        else:
            raise ValueError(op)

    else:
        raise ValueError(f"Unknown true response shape: {shape}")

    sd = float(np.std(y))
    if sd <= 0:
        raise ValueError(
            f"{trait}/{shape} true-response vector has zero variance."
        )

    return (y - np.mean(y)) / sd


# =============================================================================
# Weighted AREA vectorization
# =============================================================================

def trapz_unit_spacing(y):
    y = np.asarray(y, dtype=float)

    if y.size < 2:
        return 0.0

    return float(
        0.5 * (
            y[0]
            + y[-1]
            + 2.0 * np.sum(y[1:-1])
        )
    )


def weighted_position_coefficients(n):
    """
    Exact linear positional representation of compute_weighted_enrichment_score.
    """
    if n < 2:
        raise ValueError("Need at least two samples.")

    trap_coeff = np.ones(n, dtype=float)
    trap_coeff[0] = 0.5
    trap_coeff[-1] = 0.5

    # Contribution of a weight placed at rank position k to the trapezoidal
    # area of the cumulative curve.
    position_coeff = np.cumsum(
        trap_coeff[::-1]
    )[::-1]

    bin_width = 1.0 / n

    trend = (
        np.append(
            np.arange(
                0,
                1,
                1.0 / (n - 1),
            ),
            1.0,
        )
        * bin_width
    )

    trend_area = trapz_unit_spacing(trend)

    return position_coeff, trend_area


def build_rank_coefficient_matrix(x):
    """
    genes x samples positional coefficient matrix using the exact production
    ranking convention: ascending expression, stable mergesort.
    """
    n_samples, n_genes = x.shape

    coeff, trend_area = weighted_position_coefficients(
        n_samples
    )

    print(
        f"Building baseline rank coefficients: "
        f"{n_genes:,} genes x {n_samples:,} samples"
    )

    orders = np.argsort(
        x,
        axis=0,
        kind="mergesort",
    ).T

    b = np.empty(
        (n_genes, n_samples),
        dtype=np.float64,
    )

    chunk = 2000

    for start in range(0, n_genes, chunk):
        end = min(
            start + chunk,
            n_genes,
        )

        local_orders = orders[start:end]
        rows = np.arange(
            end - start
        )[:, None]

        b[start:end][
            rows,
            local_orders,
        ] = coeff[None, :]

    del orders

    return b, coeff, trend_area


def coefficient_rows_from_expression(
    x_gene,
    position_coeff,
):
    """
    x_gene: samples x selected_genes

    Returns selected_genes x samples positional coefficient rows.
    """
    orders = np.argsort(
        x_gene,
        axis=0,
        kind="mergesort",
    ).T

    n_genes, n_samples = orders.shape

    b = np.empty(
        (n_genes, n_samples),
        dtype=np.float64,
    )

    rows = np.arange(n_genes)[:, None]

    b[
        rows,
        orders,
    ] = position_coeff[None, :]

    return b


def score_matrix(
    b,
    trend_area,
    weight_matrix,
):
    """
    b: genes x samples
    weight_matrix: samples x K

    returns genes x K Weighted AREA ES.
    """
    n = b.shape[1]

    totals = np.sum(
        weight_matrix,
        axis=0,
    )

    if np.any(totals <= 0):
        raise ValueError(
            "At least one weight vector sums to zero."
        )

    area = (
        b @ weight_matrix
    ) / (
        n * totals[None, :]
    )

    return 2.0 * (
        area - trend_area
    )


def verify_vectorized_scoring(
    score_fn,
    x,
    b,
    trend_area,
    current_weights,
    seed,
):
    rng = np.random.default_rng(seed)

    test_genes = rng.choice(
        x.shape[1],
        size=min(25, x.shape[1]),
        replace=False,
    )

    rows = []

    for enc in ENCODINGS:
        w = transform_analysis_weights(
            current_weights,
            enc,
        )

        vectorized = score_matrix(
            b[test_genes],
            trend_area,
            w[:, None],
        )[:, 0]

        direct = np.empty(
            len(test_genes),
            dtype=float,
        )

        for j, gene_idx in enumerate(test_genes):
            order = np.argsort(
                x[:, gene_idx],
                kind="mergesort",
            )

            direct[j] = float(
                score_fn(
                    w[order]
                )
            )

        diff = np.abs(
            vectorized - direct
        )

        rows.append(
            {
                "encoding": enc,
                "n_test_genes": len(test_genes),
                "max_abs_difference": float(
                    np.max(diff)
                ),
                "allclose": bool(
                    np.allclose(
                        vectorized,
                        direct,
                        rtol=1e-11,
                        atol=1e-13,
                    )
                ),
            }
        )

    out = pd.DataFrame(rows)

    if not out["allclose"].all():
        raise RuntimeError(
            "Vectorized scoring failed validation."
        )

    return out


# =============================================================================
# Data / null loading
# =============================================================================

def load_expression(path):
    d = pd.read_csv(path)

    id_col = d.columns[0]

    sample_ids = (
        d[id_col]
        .astype(str)
        .str.strip()
    )

    if sample_ids.duplicated().any():
        raise ValueError(
            "Expression sample IDs are not unique."
        )

    xdf = (
        d.iloc[:, 1:]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if xdf.isna().any().any():
        raise ValueError(
            "Expression contains NA/non-numeric values."
        )

    return (
        pd.Index(sample_ids),
        np.asarray(
            d.columns[1:],
            dtype=str,
        ),
        xdf.to_numpy(
            dtype=float
        ),
        id_col,
    )


def load_trait_template(
    pilot_trait_dir,
):
    f = (
        Path(pilot_trait_dir)
        / "ALL_outer_exact_permutation_map.csv"
    )

    d = pd.read_csv(f)

    required = {
        "outer_replicate",
        "target_position",
        "target_sample_id",
        "original_current_weight",
    }

    missing = required - set(
        d.columns
    )

    if missing:
        raise ValueError(
            f"{f} missing columns: {sorted(missing)}"
        )

    first_rep = int(
        d["outer_replicate"].min()
    )

    g = (
        d[
            d["outer_replicate"]
            == first_rep
        ]
        .sort_values(
            "target_position"
        )
        .copy()
    )

    sample_ids = (
        g["target_sample_id"]
        .astype(str)
        .to_numpy()
    )

    current_weights = (
        g["original_current_weight"]
        .to_numpy(dtype=float)
    )

    return (
        sample_ids,
        current_weights,
        f,
    )


def align_expression(
    expr_ids,
    x_all,
    target_ids,
):
    pos = pd.Series(
        np.arange(
            len(expr_ids),
            dtype=int,
        ),
        index=expr_ids.astype(str),
    )

    target = pd.Series(
        target_ids.astype(str)
    )

    missing = target[
        ~target.isin(
            pos.index
        )
    ]

    if len(missing):
        raise ValueError(
            f"{len(missing)} target samples "
            "missing from expression."
        )

    rows = pos.loc[
        target
    ].to_numpy(dtype=int)

    return x_all[
        rows,
        :
    ]


def load_inner_null_parameters(
    pilot_trait_dir,
):
    rows = []

    for enc in ENCODINGS:
        f = (
            Path(pilot_trait_dir)
            / f"{enc}_paired_method_null.csv"
        )

        d = pd.read_csv(f)

        null = d[
            "null_es"
        ].to_numpy(dtype=float)

        rows.append(
            {
                "encoding": enc,
                "null_n": len(null),
                "null_mean": float(
                    np.mean(null)
                ),
                # Exact production convention.
                "null_sd": float(
                    np.std(null)
                ),
                "null_file": str(f),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Inference
# =============================================================================

def bh_hits_from_p(
    p,
    q,
):
    p = np.asarray(
        p,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(
        p,
        kind="mergesort",
    )

    ranked = p[order]

    critical = (
        q
        * np.arange(
            1,
            m + 1,
        )
        / m
    )

    ok = ranked <= critical

    if not np.any(ok):
        return np.empty(
            0,
            dtype=int,
        )

    k = int(
        np.where(ok)[0][-1]
        + 1
    )

    return order[:k]


def summarize_one_test(
    p,
    hit_idx,
    true_mask,
):
    n_true = int(
        np.sum(true_mask)
    )

    hit_mask = np.zeros(
        len(p),
        dtype=bool,
    )
    hit_mask[
        hit_idx
    ] = True

    tp = int(
        np.sum(
            hit_mask
            & true_mask
        )
    )

    fp = int(
        np.sum(
            hit_mask
            & ~true_mask
        )
    )

    fn = n_true - tp

    r = tp + fp

    fdp = (
        fp / r
        if r > 0
        else 0.0
    )

    tpr = (
        tp / n_true
        if n_true > 0
        else np.nan
    )

    precision = (
        tp / r
        if r > 0
        else np.nan
    )

    null_p = p[
        ~true_mask
    ]

    true_p = p[
        true_mask
    ]

    return {
        "discoveries": r,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "power_tpr": tpr,
        "fdp": fdp,
        "precision": precision,
        "null_frac_p_lt_0.05": float(
            np.mean(
                null_p < 0.05
            )
        ),
        "null_frac_p_lt_0.01": float(
            np.mean(
                null_p < 0.01
            )
        ),
        "null_frac_p_lt_0.001": float(
            np.mean(
                null_p < 0.001
            )
        ),
        "median_true_p": float(
            np.median(
                true_p
            )
        ),
        "median_null_p": float(
            np.median(
                null_p
            )
        ),
    }


# =============================================================================
# Simulation
# =============================================================================

def choose_true_genes(
    eligible_idx,
    n_true,
    rng,
):
    if n_true > len(
        eligible_idx
    ):
        raise ValueError(
            f"Requested {n_true} true genes "
            f"but only {len(eligible_idx)} eligible."
        )

    return rng.choice(
        eligible_idx,
        size=n_true,
        replace=False,
    )


def spike_true_genes(
    baseline_z_true,
    response_z,
    directions,
    beta,
):
    """
    baseline_z_true: samples x n_true
    response_z: samples
    directions: n_true with +/-1
    """
    return (
        baseline_z_true
        + beta
        * response_z[:, None]
        * directions[None, :]
    )


def aggregate_scenarios(
    result_df,
):
    group_cols = [
        "trait",
        "true_shape",
        "effect_size_beta",
        "encoding",
        "encoding_role",
    ]

    rows = []

    for keys, g in result_df.groupby(
        group_cols,
        sort=False,
    ):
        (
            trait,
            shape,
            beta,
            encoding,
            role,
        ) = keys

        n = len(g)

        rows.append(
            {
                "trait": trait,
                "true_shape": shape,
                "effect_size_beta": beta,
                "encoding": encoding,
                "encoding_role": role,
                "replicates": n,
                "mean_power": float(
                    g["power_tpr"].mean()
                ),
                "sd_power": float(
                    g["power_tpr"].std(
                        ddof=1
                    )
                ),
                "mcse_power": float(
                    g["power_tpr"].std(
                        ddof=1
                    )
                    / math.sqrt(n)
                ),
                "mean_fdp": float(
                    g["fdp"].mean()
                ),
                "median_fdp": float(
                    g["fdp"].median()
                ),
                "p95_fdp": float(
                    g["fdp"].quantile(
                        0.95
                    )
                ),
                "mcse_fdp": float(
                    g["fdp"].std(
                        ddof=1
                    )
                    / math.sqrt(n)
                ),
                "frac_replicates_fdp_gt_0.05": float(
                    np.mean(
                        g["fdp"]
                        > 0.05
                    )
                ),
                "mean_discoveries": float(
                    g["discoveries"].mean()
                ),
                "mean_true_positives": float(
                    g["true_positives"].mean()
                ),
                "mean_false_positives": float(
                    g["false_positives"].mean()
                ),
                "median_false_positives": float(
                    g["false_positives"].median()
                ),
                "max_false_positives": int(
                    g["false_positives"].max()
                ),
                "mean_precision": float(
                    g["precision"].mean(
                        skipna=True
                    )
                ),
                "mean_null_frac_p_lt_0.05": float(
                    g[
                        "null_frac_p_lt_0.05"
                    ].mean()
                ),
                "mean_null_frac_p_lt_0.01": float(
                    g[
                        "null_frac_p_lt_0.01"
                    ].mean()
                ),
                "mean_null_frac_p_lt_0.001": float(
                    g[
                        "null_frac_p_lt_0.001"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_trait(
    trait,
    trait_index,
    args,
    score_fn,
    expr_ids,
    gene_ids,
    x_all,
    shapes,
    betas,
    root_outdir,
):
    trait_seed = (
        args.seed
        + 100000 * trait_index
    )

    trait_rng = np.random.default_rng(
        trait_seed
    )

    pilot_trait_dir = (
        Path(
            args.pilot_screen_root
        )
        / trait
    )

    trait_outdir = (
        root_outdir
        / trait
    )
    trait_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print(
        f"MIXED-SIGNAL SIMULATION: {trait}"
    )
    print("=" * 80)

    (
        sample_ids,
        current_weights,
        template_file,
    ) = load_trait_template(
        pilot_trait_dir
    )

    x = align_expression(
        expr_ids,
        x_all,
        sample_ids,
    )

    n_samples, n_genes = x.shape

    print(
        f"Samples: {n_samples:,}"
    )
    print(
        f"Genes:   {n_genes:,}"
    )

    # log1p expression is used only for controlled spike-in construction.
    # Baseline sample ranking is unchanged because log1p is monotonic.
    logx = np.log1p(x)

    gene_mean = np.mean(
        logx,
        axis=0,
    )
    gene_sd = np.std(
        logx,
        axis=0,
    )

    eligible_idx = np.where(
        np.isfinite(gene_sd)
        & (gene_sd > 1e-12)
    )[0]

    print(
        f"Variable genes eligible for spike-in: "
        f"{len(eligible_idx):,}"
    )

    # Baseline genome-wide rank coefficients.
    (
        baseline_b,
        position_coeff,
        trend_area,
    ) = build_rank_coefficient_matrix(
        x
    )

    # Production scorer bridge.
    vector_check = verify_vectorized_scoring(
        score_fn=score_fn,
        x=x,
        b=baseline_b,
        trend_area=trend_area,
        current_weights=current_weights,
        seed=trait_seed + 999,
    )

    vector_check.insert(
        0,
        "trait",
        trait,
    )

    vector_check.to_csv(
        trait_outdir
        / "vectorized_scoring_validation.csv",
        index=False,
    )

    if not vector_check[
        "allclose"
    ].all():
        raise RuntimeError(
            f"{trait}: vectorized "
            "scoring failed."
        )

    # Existing 10k method null parameters.
    null_params = (
        load_inner_null_parameters(
            pilot_trait_dir
        )
    )

    null_params.insert(
        0,
        "trait",
        trait,
    )

    null_params.to_csv(
        trait_outdir
        / "inner_null_parameters.csv",
        index=False,
    )

    mu_by_enc = {
        r.encoding: float(
            r.null_mean
        )
        for r in null_params.itertuples(
            index=False
        )
    }

    sd_by_enc = {
        r.encoding: float(
            r.null_sd
        )
        for r in null_params.itertuples(
            index=False
        )
    }

    result_rows = []
    replicate_definition_rows = []

    for rep in range(
        1,
        args.replicates + 1,
    ):
        # -------------------------------------------------------------
        # Paired pseudo-phenotype for this replicate.
        # -------------------------------------------------------------
        perm_idx = trait_rng.permutation(
            n_samples
        )

        pseudo_current = (
            current_weights[
                perm_idx
            ]
        )

        # Same true genes and effect directions across every shape,
        # effect size, and analysis encoding within this replicate.
        true_idx = choose_true_genes(
            eligible_idx,
            args.true_genes,
            trait_rng,
        )

        directions = trait_rng.choice(
            np.array(
                [-1.0, 1.0]
            ),
            size=args.true_genes,
            replace=True,
        )

        true_mask = np.zeros(
            n_genes,
            dtype=bool,
        )
        true_mask[
            true_idx
        ] = True

        # Standardized baseline expression for only the selected true genes.
        baseline_z_true = (
            logx[
                :,
                true_idx
            ]
            - gene_mean[
                true_idx
            ][None, :]
        ) / gene_sd[
            true_idx
        ][None, :]

        # -------------------------------------------------------------
        # Analysis weight matrix for all encodings.
        # -------------------------------------------------------------
        analysis_weight_matrix = np.column_stack(
            [
                transform_analysis_weights(
                    pseudo_current,
                    enc,
                )
                for enc in ENCODINGS
            ]
        )

        # Baseline ES for all genes under the pseudo-phenotype.
        # These are the correct ES values for every null gene.
        base_es = score_matrix(
            baseline_b,
            trend_area,
            analysis_weight_matrix,
        )

        # Save compact replicate definition.
        replicate_definition_rows.append(
            {
                "trait": trait,
                "replicate": rep,
                "trait_seed": trait_seed,
                "n_true_genes": args.true_genes,
                "positive_direction_genes": int(
                    np.sum(
                        directions > 0
                    )
                ),
                "negative_direction_genes": int(
                    np.sum(
                        directions < 0
                    )
                ),
                "permutation_checksum": int(
                    np.dot(
                        perm_idx.astype(
                            np.int64
                        ),
                        np.arange(
                            1,
                            n_samples + 1,
                            dtype=np.int64,
                        ),
                    )
                ),
                "true_gene_checksum": int(
                    np.sum(
                        true_idx.astype(
                            np.int64
                        )
                    )
                ),
            }
        )

        for shape in shapes:
            response_z = true_response(
                pseudo_current,
                trait,
                shape,
            )

            for beta in betas:
                # -----------------------------------------------------
                # Spike only true genes.
                # -----------------------------------------------------
                spiked_true = spike_true_genes(
                    baseline_z_true,
                    response_z,
                    directions,
                    beta,
                )

                true_b = coefficient_rows_from_expression(
                    spiked_true,
                    position_coeff,
                )

                true_es = score_matrix(
                    true_b,
                    trend_area,
                    analysis_weight_matrix,
                )

                # -----------------------------------------------------
                # Inference for each analysis encoding.
                # -----------------------------------------------------
                for enc_i, enc in enumerate(
                    ENCODINGS
                ):
                    # Begin from baseline ES of all genes, then replace
                    # the true-gene rows with their post-spike ES.
                    es = base_es[
                        :,
                        enc_i,
                    ].copy()

                    es[
                        true_idx
                    ] = true_es[
                        :,
                        enc_i,
                    ]

                    mu = mu_by_enc[
                        enc
                    ]
                    sd = sd_by_enc[
                        enc
                    ]

                    z = (
                        es - mu
                    ) / sd

                    p = erfc(
                        np.abs(z)
                        / math.sqrt(2.0)
                    )

                    hit_idx = bh_hits_from_p(
                        p,
                        args.fdr_threshold,
                    )

                    metrics = summarize_one_test(
                        p=p,
                        hit_idx=hit_idx,
                        true_mask=true_mask,
                    )

                    result_rows.append(
                        {
                            "trait": trait,
                            "replicate": rep,
                            "true_shape": shape,
                            "effect_size_beta": beta,
                            "n_true_genes": args.true_genes,
                            "encoding": enc,
                            "encoding_role": ENCODING_ROLE[
                                enc
                            ],
                            **metrics,
                        }
                    )

        print(
            f"  completed replicate "
            f"{rep:>3}/{args.replicates}"
        )

    result_df = pd.DataFrame(
        result_rows
    )

    replicate_defs = pd.DataFrame(
        replicate_definition_rows
    )

    summary_df = aggregate_scenarios(
        result_df
    )

    result_df.to_csv(
        trait_outdir
        / "mixed_signal_replicate_results.csv",
        index=False,
    )

    summary_df.to_csv(
        trait_outdir
        / "mixed_signal_scenario_summary.csv",
        index=False,
    )

    replicate_defs.to_csv(
        trait_outdir
        / "replicate_definitions.csv",
        index=False,
    )

    with open(
        trait_outdir
        / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "trait": trait,
                "replicates": args.replicates,
                "true_genes_per_replicate": args.true_genes,
                "true_shapes": shapes,
                "effect_sizes_beta": betas,
                "analysis_encodings": list(
                    ENCODINGS
                ),
                "encoding_roles": ENCODING_ROLE,
                "base_seed": args.seed,
                "trait_seed": trait_seed,
                "fdr_threshold": args.fdr_threshold,
                "expression": str(
                    Path(
                        args.expression
                    ).resolve()
                ),
                "pilot_trait_dir": str(
                    pilot_trait_dir.resolve()
                ),
                "pilot_template_file": str(
                    template_file.resolve()
                ),
                "signal_construction": (
                    "z(log1p(expression)) + direction * beta * "
                    "z(true_response)"
                ),
                "threshold_rule": (
                    THRESHOLD_RULE.get(
                        trait
                    )
                ),
                "guardrail": (
                    "Same pseudo-phenotype, true genes, directions, "
                    "shape, and effect size compared across all analysis "
                    "encodings. No pathway enrichment or method selection."
                ),
            },
            handle,
            indent=2,
        )

    print("\nTrait complete.")
    print(
        summary_df[
            [
                "true_shape",
                "effect_size_beta",
                "encoding",
                "mean_power",
                "mean_fdp",
                "mean_false_positives",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    del baseline_b, logx, x

    return (
        result_df,
        summary_df,
        replicate_defs,
        vector_check,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    traits = [
        x.strip()
        for x in args.traits.split(",")
        if x.strip()
    ]

    shapes = [
        x.strip()
        for x in args.shapes.split(",")
        if x.strip()
    ]

    betas = [
        float(x)
        for x in args.effect_sizes.split(",")
        if x.strip()
    ]

    unknown_shapes = set(
        shapes
    ) - set(
        DEFAULT_SHAPES
    )

    if unknown_shapes:
        raise ValueError(
            f"Unknown shapes: "
            f"{sorted(unknown_shapes)}"
        )

    if args.replicates < 1:
        raise ValueError(
            "--replicates must be >=1"
        )

    if args.true_genes < 1:
        raise ValueError(
            "--true-genes must be >=1"
        )

    if any(
        beta <= 0
        for beta in betas
    ):
        raise ValueError(
            "All effect sizes must be >0."
        )

    root_outdir = Path(
        args.outdir
    )

    root_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "WEIGHTED AREA MIXED-SIGNAL / SPIKE-IN VALIDATION"
    )
    print("=" * 80)
    print(
        f"Traits:       {', '.join(traits)}"
    )
    print(
        f"Shapes:       {', '.join(shapes)}"
    )
    print(
        f"Effect sizes: {betas}"
    )
    print(
        f"Replicates:   {args.replicates} per trait"
    )
    print(
        f"True genes:   {args.true_genes} per replicate"
    )
    print(
        f"BH q:         {args.fdr_threshold}"
    )
    print(
        "\nThis tests mixed true/null settings. "
        "No pathways or method selection."
    )

    score_fn = import_weighted_score(
        args.area_script
    )

    (
        expr_ids,
        gene_ids,
        x_all,
        expression_id_col,
    ) = load_expression(
        args.expression
    )

    print(
        f"\nExpression loaded: "
        f"{len(expr_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )
    print(
        f"Expression ID column: "
        f"{expression_id_col}"
    )

    all_results = []
    all_summaries = []
    all_defs = []
    all_vector_checks = []

    for trait_index, trait in enumerate(
        traits
    ):
        (
            result_df,
            summary_df,
            defs_df,
            vector_df,
        ) = run_trait(
            trait=trait,
            trait_index=trait_index,
            args=args,
            score_fn=score_fn,
            expr_ids=expr_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            shapes=shapes,
            betas=betas,
            root_outdir=root_outdir,
        )

        all_results.append(
            result_df
        )
        all_summaries.append(
            summary_df
        )
        all_defs.append(
            defs_df
        )
        all_vector_checks.append(
            vector_df
        )

    master_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    master_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    master_defs = pd.concat(
        all_defs,
        ignore_index=True,
    )

    master_vector = pd.concat(
        all_vector_checks,
        ignore_index=True,
    )

    master_results.to_csv(
        root_outdir
        / "MASTER_mixed_signal_replicate_results.csv",
        index=False,
    )

    master_summary.to_csv(
        root_outdir
        / "MASTER_mixed_signal_scenario_summary.csv",
        index=False,
    )

    master_defs.to_csv(
        root_outdir
        / "MASTER_replicate_definitions.csv",
        index=False,
    )

    master_vector.to_csv(
        root_outdir
        / "MASTER_vectorized_scoring_validation.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print(
        "MIXED-SIGNAL SIMULATION COMPLETE"
    )
    print("=" * 80)

    # Compact primary-current summary for immediate inspection.
    primary = master_summary[
        master_summary[
            "encoding"
        ] == "current"
    ][
        [
            "trait",
            "true_shape",
            "effect_size_beta",
            "replicates",
            "mean_power",
            "mean_fdp",
            "p95_fdp",
            "frac_replicates_fdp_gt_0.05",
            "mean_false_positives",
            "max_false_positives",
        ]
    ]

    print(
        "\nPRIMARY CURRENT ENCODING:"
    )

    print(
        primary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nMain outputs:"
    )

    for name in [
        "MASTER_mixed_signal_scenario_summary.csv",
        "MASTER_mixed_signal_replicate_results.csv",
        "MASTER_replicate_definitions.csv",
        "MASTER_vectorized_scoring_validation.csv",
    ]:
        print(
            f"  {root_outdir / name}"
        )

    print(
        "\nSTOP HERE. Inspect mixed-signal FDR/power before "
        "real-data robustness, method lock, pathways, or plotting."
    )


if __name__ == "__main__":
    main()

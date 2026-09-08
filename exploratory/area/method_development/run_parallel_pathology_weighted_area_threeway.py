#!/usr/bin/env python3
"""
run_parallel_pathology_weighted_area_threeway.py

Run exactly three matched-sample Weighted AREA analyses per pathology axis,
using metadata from characterize_pathology_geometry_abeta_tau.py.

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

All three methods within an axis use the IDENTICAL complete-case RNA-seq sample
set, so differences reflect phenotype representation rather than sample
composition.

The implementation reproduces production Weighted AREA:
- ascending expression ranks
- stable mergesort
- exact production score bridge
- paired method-null permutations
- Gaussian two-sided inference with np.std(...), ddof=0
- BH across genes

Empirical permutation-tail p-values are retained as diagnostics.
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
    p.add_argument(
        "--area-script",
        default="exploratory/area/run_ams_area.py",
    )

    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--expression-id-col", default=None)

    p.add_argument("--inner-permutations", type=int, default=10000)
    p.add_argument("--null-batch-size", type=int, default=500)

    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--fdr-threshold", type=float, default=0.05)

    p.add_argument(
        "--outdir",
        default="results/pathology_parallel_area_threeway",
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

    if not hasattr(
        mod,
        "compute_weighted_enrichment_score",
    ):
        raise AttributeError(
            f"{path} lacks compute_weighted_enrichment_score()."
        )

    return mod.compute_weighted_enrichment_score


def trapz_unit_spacing(y):
    y = np.asarray(y, dtype=float)

    if y.size < 2:
        return 0.0

    return float(
        0.5
        * (
            y[0]
            + y[-1]
            + 2.0 * np.sum(y[1:-1])
        )
    )


def position_coefficients(n):
    if n < 2:
        raise ValueError("Need at least two samples.")

    trap = np.ones(n, dtype=float)
    trap[0] = 0.5
    trap[-1] = 0.5

    coeff = np.cumsum(
        trap[::-1]
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

    return coeff, trapz_unit_spacing(trend)


def build_rank_coefficient_matrix(x):
    """
    x: samples x genes
    returns B: genes x samples
    """
    n_samples, n_genes = x.shape

    coeff, trend_area = position_coefficients(
        n_samples
    )

    print(
        f"Building stable ascending ranks: "
        f"{n_genes:,} genes x {n_samples:,} samples"
    )

    orders = np.argsort(
        x,
        axis=0,
        kind="mergesort",
    ).T

    B = np.empty(
        (n_genes, n_samples),
        dtype=np.float64,
    )

    chunk = 2000

    for start in range(
        0,
        n_genes,
        chunk,
    ):
        end = min(
            start + chunk,
            n_genes,
        )

        local = orders[start:end]

        rr = np.arange(
            end - start
        )[:, None]

        B[start:end][
            rr,
            local,
        ] = coeff[None, :]

    del orders

    return B, trend_area


def score_matrix(
    B,
    trend_area,
    W,
):
    n = B.shape[1]

    totals = np.sum(
        W,
        axis=0,
    )

    if (
        np.any(~np.isfinite(totals))
        or np.any(totals <= 0)
    ):
        raise ValueError(
            "Every weight vector needs a finite positive sum."
        )

    area = (
        B @ W
    ) / (
        n * totals[None, :]
    )

    return 2.0 * (
        area - trend_area
    )


def load_expression(
    path,
    explicit_id_col,
):
    d = pd.read_csv(path)

    id_col = (
        explicit_id_col
        or d.columns[0]
    )

    if id_col not in d.columns:
        raise ValueError(
            f"Expression ID column '{id_col}' not found."
        )

    sample_ids = (
        d[id_col]
        .astype(str)
        .str.strip()
    )

    if sample_ids.duplicated().any():
        raise ValueError(
            "Expression sample IDs are not unique."
        )

    gene_cols = [
        c
        for c in d.columns
        if c != id_col
    ]

    xdf = d[
        gene_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if xdf.isna().any().any():
        bad = (
            xdf.columns[
                xdf.isna().any()
            ]
            .tolist()[:10]
        )

        raise ValueError(
            "Expression contains missing/non-numeric values. "
            f"Example genes: {bad}"
        )

    return (
        pd.Index(sample_ids),
        np.asarray(
            gene_cols,
            dtype=str,
        ),
        xdf.to_numpy(
            dtype=float
        ),
        id_col,
    )


def align_axis(
    expr_ids,
    x_all,
    metadata,
    sample_col,
    complete_flag,
    method_map,
):
    needed = (
        [sample_col, complete_flag]
        + list(method_map.values())
    )

    missing = [
        col
        for col in needed
        if col not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Metadata missing columns: {missing}"
        )

    d = metadata.loc[
        metadata[
            complete_flag
        ].astype(bool),
        [sample_col]
        + list(method_map.values()),
    ].copy()

    d[sample_col] = (
        d[sample_col]
        .astype(str)
        .str.strip()
    )

    if d[sample_col].duplicated().any():
        raise ValueError(
            f"Duplicate samples in {complete_flag} matched set."
        )

    pos = pd.Series(
        np.arange(
            len(expr_ids),
            dtype=int,
        ),
        index=expr_ids.astype(str),
    )

    missing_ids = d.loc[
        ~d[sample_col].isin(
            pos.index
        ),
        sample_col,
    ]

    if len(missing_ids):
        raise ValueError(
            f"{len(missing_ids)} matched metadata samples "
            "are absent from expression."
        )

    rows = pos.loc[
        d[sample_col]
    ].to_numpy(dtype=int)

    x = x_all[
        rows,
        :
    ]

    W = (
        d[
            list(
                method_map.values()
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if np.any(~np.isfinite(W)):
        raise ValueError(
            f"Missing/non-numeric weights remain in {complete_flag}."
        )

    if np.min(W) < -1e-12:
        raise ValueError(
            f"Negative pathology weights found in {complete_flag}."
        )

    return (
        d[sample_col].to_numpy(
            dtype=str
        ),
        x,
        W,
    )


def stable_seed(
    base_seed,
    label,
):
    h = hashlib.sha256(
        label.encode(
            "utf-8"
        )
    ).hexdigest()

    offset = int(
        h[:8],
        16,
    )

    return (
        int(base_seed)
        + offset
    ) % (
        2**32 - 1
    )


def generate_paired_nulls(
    W,
    n_permutations,
    seed,
    batch_size,
):
    """
    Same permutation indices are applied to all three phenotype
    representations within an axis.
    """
    n_samples, n_methods = W.shape

    coeff, trend_area = position_coefficients(
        n_samples
    )

    totals = np.sum(
        W,
        axis=0,
    )

    rng = np.random.default_rng(
        seed
    )

    nulls = np.empty(
        (
            n_permutations,
            n_methods,
        ),
        dtype=float,
    )

    for start in range(
        0,
        n_permutations,
        batch_size,
    ):
        end = min(
            start + batch_size,
            n_permutations,
        )

        batch = end - start

        perms = np.empty(
            (
                batch,
                n_samples,
            ),
            dtype=np.int32,
        )

        for i in range(batch):
            perms[i] = rng.permutation(
                n_samples
            )

        for j in range(n_methods):
            shuffled = W[
                :,
                j,
            ][perms]

            area = (
                shuffled @ coeff
            ) / (
                n_samples
                * totals[j]
            )

            nulls[
                start:end,
                j,
            ] = 2.0 * (
                area
                - trend_area
            )

        if (
            end % 2000 == 0
            or end == n_permutations
        ):
            print(
                f"    null permutations: "
                f"{end:,}/{n_permutations:,}"
            )

    return nulls


def bh_adjust(p):
    p = np.asarray(
        p,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(
        p,
        kind="mergesort",
    )

    ranked = p[
        order
    ]

    q = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
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
        m,
        dtype=float,
    )

    out[
        order
    ] = q

    return out


def empirical_two_sided_p(
    observed,
    null,
):
    mu = float(
        np.mean(
            null
        )
    )

    null_abs = np.sort(
        np.abs(
            null - mu
        )
    )

    obs_abs = np.abs(
        np.asarray(
            observed,
            dtype=float,
        )
        - mu
    )

    left = np.searchsorted(
        null_abs,
        obs_abs,
        side="left",
    )

    n_ge = (
        len(null_abs)
        - left
    )

    return (
        n_ge + 1.0
    ) / (
        len(null_abs)
        + 1.0
    )


def null_diagnostics(
    method,
    null,
):
    mu = float(
        np.mean(
            null
        )
    )

    sd = float(
        np.std(
            null
        )
    )

    if sd <= 0:
        raise ValueError(
            f"Null SD is zero for {method}."
        )

    z = (
        null - mu
    ) / sd

    return {
        "method": method,
        "inner_null_n": int(
            len(null)
        ),
        "null_mean": mu,
        "null_sd_ddof0": sd,
        "null_skew": float(
            stats.skew(
                null,
                bias=False,
            )
        ),
        "null_excess_kurtosis": float(
            stats.kurtosis(
                null,
                fisher=True,
                bias=False,
            )
        ),
        "null_frac_abs_z_ge_1.96": float(
            np.mean(
                np.abs(z)
                >= 1.95996398454
            )
        ),
        "null_frac_abs_z_ge_2.576": float(
            np.mean(
                np.abs(z)
                >= 2.57582930355
            )
        ),
        "null_frac_abs_z_ge_3.291": float(
            np.mean(
                np.abs(z)
                >= 3.29052673149
            )
        ),
        "null_min": float(
            np.min(
                null
            )
        ),
        "null_max": float(
            np.max(
                null
            )
        ),
    }


def verify_production_scoring(
    production_score,
    x,
    B,
    trend_area,
    W,
    method_names,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    test_genes = rng.choice(
        x.shape[1],
        size=min(
            25,
            x.shape[1],
        ),
        replace=False,
    )

    rows = []

    for j, method in enumerate(
        method_names
    ):
        w = W[:, j]

        vectorized = score_matrix(
            B[test_genes],
            trend_area,
            w[:, None],
        )[:, 0]

        direct = []

        for gene_idx in test_genes:
            order = np.argsort(
                x[:, gene_idx],
                kind="mergesort",
            )

            direct.append(
                float(
                    production_score(
                        w[order]
                    )
                )
            )

        direct = np.asarray(
            direct,
            dtype=float,
        )

        diff = np.abs(
            vectorized
            - direct
        )

        rows.append({
            "method": method,
            "n_test_genes": int(
                len(
                    test_genes
                )
            ),
            "max_abs_difference": float(
                np.max(
                    diff
                )
            ),
            "allclose": bool(
                np.allclose(
                    vectorized,
                    direct,
                    rtol=1e-11,
                    atol=1e-13,
                )
            ),
        })

    out = pd.DataFrame(
        rows
    )

    if not out[
        "allclose"
    ].all():
        raise RuntimeError(
            "Vectorized score failed production bridge."
        )

    return out


def pairwise_method_comparison(
    results,
    qcut,
):
    names = list(
        results
    )

    rows = []

    for i, method_a in enumerate(
        names
    ):
        A = results[
            method_a
        ]

        for method_b in names[
            i + 1:
        ]:
            B = results[
                method_b
            ]

            za = A[
                "z"
            ].to_numpy(
                dtype=float
            )

            zb = B[
                "z"
            ].to_numpy(
                dtype=float
            )

            sa = (
                A[
                    "padj_gaussian"
                ].to_numpy(
                    dtype=float
                )
                < qcut
            )

            sb = (
                B[
                    "padj_gaussian"
                ].to_numpy(
                    dtype=float
                )
                < qcut
            )

            both = (
                sa & sb
            )

            union = (
                sa | sb
            )

            direction_all = (
                np.sign(
                    za
                )
                == np.sign(
                    zb
                )
            )

            direction_union = (
                direction_all[
                    union
                ]
                if np.any(
                    union
                )
                else np.array([])
            )

            rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "pearson_z": float(
                    stats.pearsonr(
                        za,
                        zb,
                    ).statistic
                ),
                "spearman_z": float(
                    stats.spearmanr(
                        za,
                        zb,
                    ).statistic
                ),
                "direction_agreement_all": float(
                    np.mean(
                        direction_all
                    )
                ),
                "direction_agreement_sig_union": (
                    float(
                        np.mean(
                            direction_union
                        )
                    )
                    if len(
                        direction_union
                    )
                    else np.nan
                ),
                "sig_a": int(
                    np.sum(
                        sa
                    )
                ),
                "sig_b": int(
                    np.sum(
                        sb
                    )
                ),
                "sig_both": int(
                    np.sum(
                        both
                    )
                ),
                "sig_union": int(
                    np.sum(
                        union
                    )
                ),
                "jaccard_sig": (
                    float(
                        np.sum(
                            both
                        )
                        / np.sum(
                            union
                        )
                    )
                    if np.sum(
                        union
                    )
                    else np.nan
                ),
            })

    return pd.DataFrame(
        rows
    )


def run_axis(
    axis_name,
    complete_flag,
    method_map,
    expr_ids,
    gene_ids,
    x_all,
    metadata,
    production_score,
    a,
    root_outdir,
):
    axis_dir = (
        root_outdir
        / axis_name
    )

    axis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    method_names = list(
        method_map
    )

    (
        sample_ids,
        x,
        W,
    ) = align_axis(
        expr_ids,
        x_all,
        metadata,
        a.sample_col,
        complete_flag,
        method_map,
    )

    print("\n" + "=" * 80)
    print(
        axis_name.upper()
    )
    print("=" * 80)
    print(
        f"Matched samples: "
        f"{len(sample_ids):,}"
    )
    print(
        f"Genes: "
        f"{len(gene_ids):,}"
    )
    print(
        f"Methods: "
        f"{', '.join(method_names)}"
    )

    pd.DataFrame({
        "position": np.arange(
            len(sample_ids)
        ),
        "sample_id": sample_ids,
    }).to_csv(
        axis_dir
        / "matched_sample_manifest.csv",
        index=False,
    )

    B, trend_area = (
        build_rank_coefficient_matrix(
            x
        )
    )

    validation = (
        verify_production_scoring(
            production_score,
            x,
            B,
            trend_area,
            W,
            method_names,
            stable_seed(
                a.seed,
                axis_name
                + "_validation",
            ),
        )
    )

    validation.to_csv(
        axis_dir
        / "vectorized_scoring_validation.csv",
        index=False,
    )

    print(
        "\nProduction scoring bridge:"
    )
    print(
        validation.to_string(
            index=False
        )
    )

    observed_es = score_matrix(
        B,
        trend_area,
        W,
    )

    print(
        "\nGenerating paired method nulls..."
    )

    nulls = generate_paired_nulls(
        W,
        a.inner_permutations,
        stable_seed(
            a.seed,
            axis_name
            + "_paired_null",
        ),
        a.null_batch_size,
    )

    null_df = pd.DataFrame(
        nulls,
        columns=method_names,
    )

    null_df.insert(
        0,
        "permutation",
        np.arange(
            1,
            a.inner_permutations
            + 1,
        ),
    )

    null_df.to_csv(
        axis_dir
        / "paired_method_nulls.csv",
        index=False,
    )

    results = {}
    null_rows = []

    for j, method in enumerate(
        method_names
    ):
        null = nulls[
            :,
            j,
        ]

        observed = observed_es[
            :,
            j,
        ]

        diag = null_diagnostics(
            method,
            null,
        )

        null_rows.append(
            diag
        )

        mu = diag[
            "null_mean"
        ]

        sd = diag[
            "null_sd_ddof0"
        ]

        z = (
            observed
            - mu
        ) / sd

        p_gaussian = erfc(
            np.abs(
                z
            )
            / math.sqrt(
                2.0
            )
        )

        q_gaussian = bh_adjust(
            p_gaussian
        )

        p_empirical = empirical_two_sided_p(
            observed,
            null,
        )

        q_empirical = bh_adjust(
            p_empirical
        )

        result = pd.DataFrame({
            "gene_id": gene_ids,
            "observed_es": observed,
            "null_mean": mu,
            "null_sd_ddof0": sd,
            "z": z,
            "pvalue_gaussian": p_gaussian,
            "padj_gaussian": q_gaussian,
            "pvalue_empirical": p_empirical,
            "padj_empirical": q_empirical,
        })

        result.to_csv(
            axis_dir
            / f"{method}_Weighted_AREA_results.csv",
            index=False,
        )

        results[
            method
        ] = result

    null_diag_df = pd.DataFrame(
        null_rows
    )

    null_diag_df.to_csv(
        axis_dir
        / "null_diagnostics.csv",
        index=False,
    )

    pairwise = pairwise_method_comparison(
        results,
        a.fdr_threshold,
    )

    pairwise.to_csv(
        axis_dir
        / "pairwise_method_comparisons.csv",
        index=False,
    )

    wide = pd.DataFrame({
        "gene_id": gene_ids
    })

    for method, result in results.items():
        wide[
            f"{method}__ES"
        ] = result[
            "observed_es"
        ]

        wide[
            f"{method}__Z"
        ] = result[
            "z"
        ]

        wide[
            f"{method}__P"
        ] = result[
            "pvalue_gaussian"
        ]

        wide[
            f"{method}__FDR"
        ] = result[
            "padj_gaussian"
        ]

        wide[
            f"{method}__SIG"
        ] = (
            result[
                "padj_gaussian"
            ]
            < a.fdr_threshold
        ).astype(int)

    sig_cols = [
        f"{method}__SIG"
        for method
        in method_names
    ]

    wide[
        "n_methods_significant"
    ] = wide[
        sig_cols
    ].sum(
        axis=1
    )

    wide.to_csv(
        axis_dir
        / "gene_level_threeway_robustness.csv",
        index=False,
    )

    print(
        "\nNull diagnostics:"
    )
    print(
        null_diag_df.to_string(
            index=False
        )
    )

    print(
        "\nPairwise concordance:"
    )
    print(
        pairwise.to_string(
            index=False
        )
    )

    print(
        "\nFDR-significant genes:"
    )

    for method in method_names:
        n_sig = int(
            (
                results[
                    method
                ][
                    "padj_gaussian"
                ]
                < a.fdr_threshold
            ).sum()
        )

        print(
            f"  {method}: "
            f"{n_sig:,}"
        )

    del B, x

    return {
        "axis": axis_name,
        "n_samples": int(
            len(
                sample_ids
            )
        ),
        "methods": method_names,
    }


def main():
    a = parse_args()

    root = Path(
        a.outdir
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        a.metadata
    )

    (
        expr_ids,
        gene_ids,
        x_all,
        expression_id_col,
    ) = load_expression(
        a.expression,
        a.expression_id_col,
    )

    print("=" * 80)
    print(
        "ROSMAP THREE-WAY PATHOLOGY WEIGHTED AREA"
    )
    print("=" * 80)
    print(
        f"Expression: "
        f"{len(expr_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )
    print(
        f"Expression ID column: "
        f"{expression_id_col}"
    )
    print(
        f"Inner null permutations: "
        f"{a.inner_permutations:,}"
    )

    production_score = import_production_score(
        a.area_script
    )

    axes = []

    axes.append(
        run_axis(
            axis_name="amyloid_axis",
            complete_flag="CERAD_threeway_complete",
            method_map=AMYLOID_METHODS,
            expr_ids=expr_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            metadata=metadata,
            production_score=production_score,
            a=a,
            root_outdir=root,
        )
    )

    axes.append(
        run_axis(
            axis_name="tau_axis",
            complete_flag="Braak_threeway_complete",
            method_map=TAU_METHODS,
            expr_ids=expr_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            metadata=metadata,
            production_score=production_score,
            a=a,
            root_outdir=root,
        )
    )

    with open(
        root
        / "threeway_pathology_area_manifest.json",
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
                "seed": a.seed,
                "inner_permutations": a.inner_permutations,
                "fdr_threshold": a.fdr_threshold,
                "axes": axes,
                "design": (
                    "Exactly three representations per pathology axis, "
                    "with identical complete-case RNA-seq samples within axis."
                ),
                "important": (
                    "Do not select a representation by hit count. "
                    "Interpret stage geometry first, then assess gene-level "
                    "concordance. New quantitative/calibrated traits require "
                    "complete-null validation before final method lock."
                ),
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(
        f"Results root: "
        f"{root}"
    )
    print(
        "\nNext: inspect observed concordance, then complete-null validate "
        "the calibrated and continuous pathology versions."
    )


if __name__ == "__main__":
    main()

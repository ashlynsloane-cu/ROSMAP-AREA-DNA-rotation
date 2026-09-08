#!/usr/bin/env python3
"""
Validate tie-aware Weighted AREA before modifying production code.

Checks:
A. geometric curve ES == centered average-rank ES
B. row-order invariance
C. explicit permutation null vs analytic tie-corrected variance
D. current stable-rank Z vs tie-aware Z across sampled genes

No pathways and no biological winner selection.
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

from tie_aware_weighted_area import (
    average_expression_ranks,
    centered_linear_rank_statistic,
    curve_area_es,
    permutation_variance_linear_rank,
    tie_aware_area_test,
    tie_summary,
)

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
    p.add_argument("--area-script", default="exploratory/area/run_ams_area.py")
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--expression-id-col", default=None)
    p.add_argument("--weight-cols", default=",".join(DEFAULT_WEIGHTS))
    p.add_argument("--n-genes", type=int, default=2000)
    p.add_argument("--row-order-repeats", type=int, default=10)
    p.add_argument("--permutation-validation-genes", type=int, default=12)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument(
        "--outdir",
        default="results/ams_area_validation/tie_aware_weighted_area",
    )
    return p.parse_args()


def stable_seed(base, label):
    h = hashlib.sha256(label.encode()).hexdigest()
    return (int(base) + int(h[:8], 16)) % (2**32 - 1)


def import_production_score(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("run_ams_area_current", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "compute_weighted_enrichment_score"):
        raise AttributeError(
            f"{path} lacks compute_weighted_enrichment_score()."
        )
    return mod.compute_weighted_enrichment_score


def load_expression(path, id_col):
    d = pd.read_csv(path)
    id_col = id_col or d.columns[0]
    ids = d[id_col].astype(str).str.strip()

    if ids.duplicated().any():
        raise ValueError("Expression sample IDs are duplicated.")

    genes = [c for c in d.columns if c != id_col]
    xdf = d[genes].apply(pd.to_numeric, errors="coerce")

    if xdf.isna().any().any():
        bad = xdf.columns[xdf.isna().any()].tolist()[:10]
        raise ValueError(f"Expression has NA/non-numeric genes: {bad}")

    return pd.Index(ids), np.asarray(genes), xdf.to_numpy(float), id_col


def align_weight(expr_ids, x_all, meta, sample_col, weight_col):
    d = meta[[sample_col, weight_col]].copy()
    d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce")
    d = d.dropna(subset=[sample_col, weight_col])
    d[sample_col] = d[sample_col].astype(str).str.strip()

    pos = pd.Series(np.arange(len(expr_ids)), index=expr_ids.astype(str))

    if not d[sample_col].isin(pos.index).all():
        raise ValueError(f"Some {weight_col} samples absent from expression.")

    rows = pos.loc[d[sample_col]].to_numpy(int)

    return (
        d[sample_col].to_numpy(str),
        x_all[rows, :],
        d[weight_col].to_numpy(float),
    )


def current_es(production_score, expression, weights):
    order = np.argsort(expression, kind="mergesort")
    return float(production_score(weights[order]))


def current_stable_rank_z(expression, weights):
    order = np.argsort(expression, kind="mergesort")

    ranks = np.empty(len(expression), float)
    ranks[order] = np.arange(1, len(expression) + 1, dtype=float)

    wc = weights - np.mean(weights)
    rc = ranks - np.mean(ranks)

    c = float(np.dot(wc, rc))
    var_c = float(np.dot(wc, wc) * np.dot(rc, rc) / (len(weights) - 1))
    return float(-c / math.sqrt(var_c))


def choose_across_tie_quantiles(tie_fractions, n_choose):
    tie_fractions = np.asarray(tie_fractions, float)

    if n_choose >= len(tie_fractions):
        return np.arange(len(tie_fractions))

    targets = np.quantile(
        tie_fractions,
        np.linspace(0, 1, n_choose),
    )

    available = set(range(len(tie_fractions)))
    chosen = []

    for target in targets:
        idx = min(
            available,
            key=lambda i: abs(tie_fractions[i] - target),
        )
        chosen.append(idx)
        available.remove(idx)

    return np.asarray(chosen, int)


def explicit_permutation_check(
    expression,
    weights,
    observed_c,
    analytic_var,
    permutations,
    seed,
):
    ranks = average_expression_ranks(expression)
    rc = ranks - np.mean(ranks)
    wc = weights - np.mean(weights)

    rng = np.random.default_rng(seed)
    null_c = np.empty(permutations, float)

    for i in range(permutations):
        null_c[i] = np.dot(wc[rng.permutation(len(wc))], rc)

    empirical_mean = float(np.mean(null_c))
    empirical_sd = float(np.std(null_c, ddof=0))
    analytic_sd = float(math.sqrt(analytic_var))

    analytic_z = float(-observed_c / analytic_sd)
    permutation_z = float(
        -(observed_c - empirical_mean) / empirical_sd
    )

    empirical_p = float(
        (
            np.sum(
                np.abs(null_c - empirical_mean)
                >= abs(observed_c - empirical_mean)
            )
            + 1
        )
        / (permutations + 1)
    )

    gaussian_p = float(
        erfc(abs(analytic_z) / math.sqrt(2.0))
    )

    return {
        "permutation_null_mean": empirical_mean,
        "permutation_null_sd": empirical_sd,
        "analytic_null_sd": analytic_sd,
        "sd_ratio_empirical_over_analytic": empirical_sd / analytic_sd,
        "analytic_z": analytic_z,
        "permutation_standardized_z": permutation_z,
        "abs_delta_z": abs(analytic_z - permutation_z),
        "gaussian_p": gaussian_p,
        "empirical_p": empirical_p,
    }


def main():
    a = parse_args()

    root = Path(a.outdir)
    root.mkdir(parents=True, exist_ok=True)

    weight_cols = [x.strip() for x in a.weight_cols.split(",") if x.strip()]

    meta = pd.read_csv(a.metadata)
    expr_ids, genes, x_all, expr_id_col = load_expression(
        a.expression, a.expression_id_col
    )

    production_score = import_production_score(a.area_script)

    rng = np.random.default_rng(a.seed)
    n_genes = min(a.n_genes, len(genes))
    gene_indices = np.sort(
        rng.choice(len(genes), size=n_genes, replace=False)
    )

    print("=" * 80)
    print("TIE-AWARE WEIGHTED AREA VALIDATION")
    print("=" * 80)
    print(
        f"Expression: {len(expr_ids):,} samples x {len(genes):,} genes"
    )
    print(f"Sampled genes: {n_genes:,}")

    master_rows = []
    permutation_rows = []
    row_order_rows = []

    for weight_col in weight_cols:
        print("\n" + "-" * 80)
        print(weight_col)
        print("-" * 80)

        sample_ids, x, weights = align_weight(
            expr_ids,
            x_all,
            meta,
            a.sample_col,
            weight_col,
        )

        gene_rows = []

        for gene_idx in gene_indices:
            expression = x[:, gene_idx]
            tie = tie_summary(expression)

            prod_es = current_es(
                production_score,
                expression,
                weights,
            )

            stable_z = current_stable_rank_z(
                expression,
                weights,
            )

            tie_result = tie_aware_area_test(
                expression,
                weights,
            )

            geom_es = curve_area_es(
                expression,
                weights,
            )

            gene_rows.append({
                "gene_id": genes[gene_idx],
                "gene_index": int(gene_idx),
                "current_production_ES": prod_es,
                "current_stable_rank_Z": stable_z,
                "tie_aware_ES": tie_result.es,
                "tie_aware_Z": tie_result.z,
                "tie_aware_gaussian_P": tie_result.pvalue_gaussian,
                "geometric_curve_ES": geom_es,
                "geometric_identity_abs_diff": abs(
                    geom_es - tie_result.es
                ),
                "delta_ES_tieaware_minus_current": (
                    tie_result.es - prod_es
                ),
                "delta_Z_tieaware_minus_stable": (
                    tie_result.z - stable_z
                ),
                **tie,
            })

        gene_df = pd.DataFrame(gene_rows)

        gene_df.to_csv(
            root / f"{weight_col}__gene_level_tie_comparison.csv",
            index=False,
        )

        # Row-order invariance: deliberately shuffle full sample rows.
        order_test = (
            gene_df
            .sort_values(
                ["fraction_samples_in_ties", "max_tie_group_size"],
                ascending=False,
            )
            .head(min(20, len(gene_df)))
        )

        for _, row in order_test.iterrows():
            gene_idx = int(row["gene_index"])
            expression = x[:, gene_idx]

            baseline = tie_aware_area_test(
                expression,
                weights,
            )

            rrng = np.random.default_rng(
                stable_seed(a.seed, f"{weight_col}_{gene_idx}")
            )

            max_es_delta = 0.0
            max_z_delta = 0.0

            for _ in range(a.row_order_repeats):
                p = rrng.permutation(len(weights))
                rerun = tie_aware_area_test(
                    expression[p],
                    weights[p],
                )
                max_es_delta = max(
                    max_es_delta,
                    abs(rerun.es - baseline.es),
                )
                max_z_delta = max(
                    max_z_delta,
                    abs(rerun.z - baseline.z),
                )

            row_order_rows.append({
                "weight_col": weight_col,
                "gene_id": genes[gene_idx],
                "fraction_samples_in_ties": float(
                    row["fraction_samples_in_ties"]
                ),
                "max_tie_group_size": int(row["max_tie_group_size"]),
                "max_abs_delta_ES_after_row_shuffle": max_es_delta,
                "max_abs_delta_Z_after_row_shuffle": max_z_delta,
            })

        # Explicit permutation-null checks across tie burden.
        chosen = choose_across_tie_quantiles(
            gene_df["fraction_samples_in_ties"].to_numpy(float),
            min(a.permutation_validation_genes, len(gene_df)),
        )

        for local_idx in chosen:
            row = gene_df.iloc[local_idx]
            gene_idx = int(row["gene_index"])
            expression = x[:, gene_idx]

            c, ranks = centered_linear_rank_statistic(
                expression,
                weights,
            )

            var_c = permutation_variance_linear_rank(
                ranks,
                weights,
            )

            check = explicit_permutation_check(
                expression,
                weights,
                c,
                var_c,
                a.permutations,
                stable_seed(
                    a.seed,
                    f"{weight_col}__{gene_idx}",
                ),
            )

            permutation_rows.append({
                "weight_col": weight_col,
                "gene_id": genes[gene_idx],
                "gene_index": gene_idx,
                "fraction_samples_in_ties": float(
                    row["fraction_samples_in_ties"]
                ),
                "max_tie_group_size": int(
                    row["max_tie_group_size"]
                ),
                **check,
            })

        master_rows.append({
            "weight_col": weight_col,
            "n_samples": len(weights),
            "n_genes_compared": len(gene_df),
            "fraction_genes_with_any_ties": float(
                np.mean(
                    gene_df["fraction_samples_in_ties"] > 0
                )
            ),
            "median_fraction_samples_in_ties": float(
                gene_df["fraction_samples_in_ties"].median()
            ),
            "max_fraction_samples_in_ties": float(
                gene_df["fraction_samples_in_ties"].max()
            ),
            "max_tie_group_size": int(
                gene_df["max_tie_group_size"].max()
            ),
            "max_geometric_identity_abs_diff": float(
                gene_df["geometric_identity_abs_diff"].max()
            ),
            "pearson_current_stableZ_vs_tieawareZ": float(
                stats.pearsonr(
                    gene_df["current_stable_rank_Z"],
                    gene_df["tie_aware_Z"],
                ).statistic
            ),
            "spearman_current_stableZ_vs_tieawareZ": float(
                stats.spearmanr(
                    gene_df["current_stable_rank_Z"],
                    gene_df["tie_aware_Z"],
                ).statistic
            ),
            "median_abs_delta_Z": float(
                np.median(
                    np.abs(
                        gene_df["delta_Z_tieaware_minus_stable"]
                    )
                )
            ),
            "p95_abs_delta_Z": float(
                np.quantile(
                    np.abs(
                        gene_df["delta_Z_tieaware_minus_stable"]
                    ),
                    0.95,
                )
            ),
            "max_abs_delta_Z": float(
                np.max(
                    np.abs(
                        gene_df["delta_Z_tieaware_minus_stable"]
                    )
                )
            ),
        })

        print(
            "  genes with any ties: "
            f"{100 * master_rows[-1]['fraction_genes_with_any_ties']:.2f}%"
        )
        print(
            "  geometric identity max error: "
            f"{master_rows[-1]['max_geometric_identity_abs_diff']:.3e}"
        )
        print(
            "  current stable-Z vs tie-aware Z Pearson: "
            f"{master_rows[-1]['pearson_current_stableZ_vs_tieawareZ']:.8f}"
        )
        print(
            "  p95 |delta Z| from tie correction: "
            f"{master_rows[-1]['p95_abs_delta_Z']:.6g}"
        )

    master = pd.DataFrame(master_rows)
    perm_df = pd.DataFrame(permutation_rows)
    row_df = pd.DataFrame(row_order_rows)

    master.to_csv(root / "MASTER_tie_aware_summary.csv", index=False)
    perm_df.to_csv(root / "permutation_null_validation.csv", index=False)
    row_df.to_csv(root / "row_order_invariance_validation.csv", index=False)

    with open(root / "tie_aware_validation_manifest.json", "w") as f:
        json.dump({
            "expression": str(Path(a.expression).resolve()),
            "metadata": str(Path(a.metadata).resolve()),
            "area_script": str(Path(a.area_script).resolve()),
            "expression_id_col": expr_id_col,
            "sample_col": a.sample_col,
            "weight_cols": weight_cols,
            "n_genes": n_genes,
            "row_order_repeats": a.row_order_repeats,
            "permutation_validation_genes_per_weight": (
                a.permutation_validation_genes
            ),
            "permutations": a.permutations,
            "seed": a.seed,
            "purpose": (
                "Validate average-rank, sample-order-invariant tie handling "
                "before modifying production Weighted AREA."
            ),
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("MASTER SUMMARY")
    print("=" * 80)
    print(master.to_string(index=False))

    print("\nPermutation-null validation:")
    print(
        perm_df[
            [
                "weight_col",
                "gene_id",
                "fraction_samples_in_ties",
                "sd_ratio_empirical_over_analytic",
                "abs_delta_z",
                "gaussian_p",
                "empirical_p",
            ]
        ].to_string(index=False)
    )

    print("\nWrote:")
    print(root / "MASTER_tie_aware_summary.csv")
    print(root / "permutation_null_validation.csv")
    print(root / "row_order_invariance_validation.csv")

    print(
        "\nDo not replace run_ams_area.py yet. "
        "Inspect these validation outputs first."
    )


if __name__ == "__main__":
    main()

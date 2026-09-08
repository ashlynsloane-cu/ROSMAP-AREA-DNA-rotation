#!/usr/bin/env python3
"""
diagnose_area_pvalue_calibration.py
==============================================================================
Compare AREA p-value calculations under repeated global-null phenotype shuffles.
==============================================================================

Why this script exists
----------------------
The current official AREA p-value function:
  1. separates positive and negative permutation-null scores,
  2. estimates mean and SD within the selected signed half,
  3. treats that signed-half distribution as an ordinary Gaussian.

A signed half of a roughly Gaussian null is a TRUNCATED distribution, not an
ordinary Gaussian. Fitting an unconstrained Gaussian to that half can therefore
produce anti-conservative tail probabilities.

This script compares four p-value approaches on exactly the same AREA ES values:

  official_signed_gaussian
      Reproduces the current AREA behavior.

  full_null_gaussian_2s
      Fits one Gaussian to the FULL permutation-null ES distribution and uses
      a two-sided tail probability.

  analytic_rank_gaussian_2s
      Uses the exact random-permutation mean and variance of the AREA linear
      rank statistic, followed by a two-sided Gaussian approximation.
      No fitted signed-tail distribution is used.

  empirical_2s
      Direct Monte Carlo two-sided p-value based on distance from the full-null
      mean. This is calibration-friendly but has finite-permutation resolution,
      so with 1,000 inner permutations its minimum p is ~0.001 and it is not
      suitable by itself for genome-wide BH discovery. It serves as a benchmark.

The goal is NOT to pick a replacement in advance. The goal is to find which
method actually returns approximately calibrated p-values under 100+ global-null
phenotype permutations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Intended location:
# exploratory/area/validation/diagnose_area_pvalue_calibration.py
VALIDATION_DIR = Path(__file__).resolve().parent
AREA_SCRIPT_DIR = VALIDATION_DIR.parent
if str(AREA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AREA_SCRIPT_DIR))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from run_ams_area import (
    benjamini_hochberg,
    configure_area_import,
    encode_trait_vector,
    load_expression,
    load_metadata,
    load_sample_manifest,
    permute_weighted_enrichment_scores,
)
from validate_ams_area_nulls import (
    precompute_orders,
    vectorized_area_scores,
    vectorized_nes_pvalues,
)


def get_trait_vector(
    trait: dict,
    method: str,
    metadata: pd.DataFrame,
    sample_order: List[str],
) -> Tuple[np.ndarray, List[str]]:
    source = metadata.loc[sample_order, trait["source_column"]]
    regular = encode_trait_vector(source, trait["regular"], trait["name"], "regular")

    if method == "regular":
        series = regular
    else:
        if trait["type"].lower() == "binary_only":
            raise ValueError(
                f"Trait '{trait['name']}' is binary-only; Weighted AREA is not distinct."
            )
        weighted = encode_trait_vector(source, trait["weighted"], trait["name"], "weighted")
        if not regular.notna().equals(weighted.notna()):
            raise ValueError("Regular and Weighted encodings select different samples.")
        series = weighted

    valid = series.notna()
    ids = series.index[valid].tolist()
    vec = series.loc[ids].to_numpy(dtype=float)
    return vec, ids


def score_genome_es(
    phenotype: np.ndarray,
    orders_chunks,
    method: str,
    n_genes: int,
) -> np.ndarray:
    es_all = np.empty(n_genes, dtype=float)
    regular = method == "regular"

    for start, end, orders in orders_chunks:
        sorted_values = phenotype[orders]
        es_all[start:end] = vectorized_area_scores(sorted_values, regular=regular)

    return es_all


def p_official(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    _, p = vectorized_nes_pvalues(scores, null_scores)
    return p


def p_full_null_gaussian_two_sided(
    scores: np.ndarray,
    null_scores: np.ndarray,
) -> np.ndarray:
    """
    Fit one Gaussian to the full permutation null and use a two-sided p-value.
    """
    null = np.asarray(null_scores, dtype=float)
    mu = float(np.mean(null))
    sigma = float(np.std(null))
    if sigma <= 0:
        raise RuntimeError("Full permutation null has zero variance.")

    z = (np.asarray(scores, dtype=float) - mu) / sigma
    p = 2.0 * stats.norm.sf(np.abs(z))
    return np.clip(p, 0.0, 1.0)


def area_linear_null_moments(weights: np.ndarray) -> Tuple[float, float]:
    """
    Exact random-permutation mean and variance of the AREA enrichment score.

    AREA ES is a linear rank statistic when the phenotype values are fixed and
    randomly permuted across expression ranks.

    If:
        T = sum_i a_i * w_perm(i)

    then for a random permutation:
        Var(T) =
          [sum_i (a_i-a_bar)^2 * sum_i (w_i-w_bar)^2] / (n-1)

    The AREA scale and fixed trend term are then applied exactly.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) < 2:
        raise ValueError("Phenotype vector must be one-dimensional with n>=2.")

    if np.any(~np.isfinite(w)):
        raise ValueError("Phenotype contains non-finite values.")

    # Official Regular AREA thresholds values >0 to 1 before scoring.
    n = len(w)
    total_weight = float(np.sum(w))
    if total_weight <= 0:
        raise ValueError("Phenotype total weight must be >0.")

    # Trapezoid coefficients for cumulative curve.
    trap_weights = np.ones(n, dtype=float)
    trap_weights[0] = 0.5
    trap_weights[-1] = 0.5

    # Coefficient multiplying q_i = w_i / total_weight in AUC(cumulative / n).
    # a_i = (1/n) * sum_{j=i}^{n-1} trap_weight_j
    reverse_cumsum = np.cumsum(trap_weights[::-1])[::-1]
    a = reverse_cumsum / n

    # AREA's fixed trend exactly as implemented in the package.
    trend = np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0) / n
    trend_auc = 0.5 * (
        trend[0] + trend[-1] + 2.0 * np.sum(trend[1:-1])
    )

    mean_score = 2.0 * (float(np.mean(a)) - trend_auc)

    centered_a_ss = float(np.sum((a - np.mean(a)) ** 2))
    centered_w_ss = float(np.sum((w - np.mean(w)) ** 2))

    var_linear_sum = centered_a_ss * centered_w_ss / (n - 1)
    var_score = (2.0 / total_weight) ** 2 * var_linear_sum

    if var_score <= 0:
        raise RuntimeError("Analytic AREA null variance is zero.")

    return mean_score, var_score


def p_analytic_rank_gaussian_two_sided(
    scores: np.ndarray,
    phenotype: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, float, float]:
    if method == "regular":
        weights = (np.asarray(phenotype) > 0).astype(float)
    else:
        weights = np.asarray(phenotype, dtype=float)

    mu, var = area_linear_null_moments(weights)
    sigma = float(np.sqrt(var))
    z = (np.asarray(scores, dtype=float) - mu) / sigma
    p = 2.0 * stats.norm.sf(np.abs(z))
    return np.clip(p, 0.0, 1.0), mu, sigma


def p_empirical_two_sided(
    scores: np.ndarray,
    null_scores: np.ndarray,
) -> np.ndarray:
    """
    Add-one Monte Carlo p-value using distance from the full-null mean.

    Vectorized by sorting null absolute deviations.
    """
    null = np.asarray(null_scores, dtype=float)
    mu = float(np.mean(null))
    null_dev = np.sort(np.abs(null - mu))
    obs_dev = np.abs(np.asarray(scores, dtype=float) - mu)

    # count null_dev >= obs_dev
    left = np.searchsorted(null_dev, obs_dev, side="left")
    extreme = len(null_dev) - left

    return (extreme + 1.0) / (len(null_dev) + 1.0)


def metrics(p: np.ndarray, alpha: float = 0.05) -> dict:
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    fdr = benjamini_hochberg(p)

    return {
        "n_genes": int(len(pv)),
        "n_fdr_lt_0.05": int(np.sum(np.isfinite(fdr) & (fdr < 0.05))),
        "frac_p_lt_0.05": float(np.mean(pv < 0.05)),
        "frac_p_lt_0.01": float(np.mean(pv < 0.01)),
        "frac_p_lt_0.001": float(np.mean(pv < 0.001)),
        "median_p": float(np.median(pv)),
        "min_p": float(np.min(pv)),
    }


def calibration_error(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected = {
        "frac_p_lt_0.05": 0.05,
        "frac_p_lt_0.01": 0.01,
        "frac_p_lt_0.001": 0.001,
    }

    for method, sub in summary.groupby("pvalue_method"):
        row = {
            "pvalue_method": method,
            "null_mean_fdr_sig": float(sub["n_fdr_lt_0.05"].mean()),
            "null_median_fdr_sig": float(sub["n_fdr_lt_0.05"].median()),
            "null_max_fdr_sig": int(sub["n_fdr_lt_0.05"].max()),
            "null_frac_runs_any_fdr_sig": float(np.mean(sub["n_fdr_lt_0.05"] > 0)),
            "null_mean_median_p": float(sub["median_p"].mean()),
        }
        for metric, exp in expected.items():
            observed = float(sub[metric].mean())
            row["null_mean_" + metric] = observed
            row["inflation_ratio_" + metric] = observed / exp
        rows.append(row)

    return pd.DataFrame(rows)


def plot_calibration(compact: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = compact["pvalue_method"].tolist()
    x = np.arange(len(methods))

    for metric, expected, ylabel, filename in [
        ("null_mean_frac_p_lt_0.05", 0.05, "Mean null fraction p < 0.05", "calibration_p005.png"),
        ("null_mean_frac_p_lt_0.01", 0.01, "Mean null fraction p < 0.01", "calibration_p001.png"),
        ("null_mean_frac_p_lt_0.001", 0.001, "Mean null fraction p < 0.001", "calibration_p0001.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(x, compact[metric].values)
        ax.axhline(expected, linestyle="--", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("AREA null p-value calibration")
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x, compact["null_mean_fdr_sig"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Mean BH FDR<0.05 genes under global null")
    ax.set_title("False discoveries under global-null phenotype shuffles")
    fig.tight_layout()
    fig.savefig(outdir / "calibration_false_discoveries.png", dpi=300)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare current and candidate AREA p-value calculations under null."
    )
    p.add_argument("-e", "--expression", required=True)
    p.add_argument("-m", "--metadata", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--method", choices=["regular", "weighted"], default="regular")
    p.add_argument("-o", "--outdir", default="results/ams_area_validation/pvalue_diagnostic")
    p.add_argument("--area-root", default=None)
    p.add_argument("--sample-manifest", default=None)
    p.add_argument("--outer-permutations", type=int, default=100)
    p.add_argument("--inner-permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    configure_area_import(args.area_root)

    from src.area.enrichment import permute_enrichment_scores

    expression = load_expression(Path(args.expression).expanduser())
    metadata = load_metadata(Path(args.metadata).expanduser())

    common = sorted(set(expression.index).intersection(metadata.index))
    if not common:
        raise ValueError("No overlapping expression/metadata samples.")

    manifest = load_sample_manifest(
        Path(args.sample_manifest).expanduser() if args.sample_manifest else None
    )
    if manifest is not None:
        missing = manifest - set(common)
        if missing:
            raise ValueError(
                f"{len(missing)} manifest IDs absent from expression/metadata; "
                f"examples={sorted(missing)[:10]}"
            )
        common = [s for s in common if s in manifest]

    rng_order = random.Random(args.seed)
    rng_order.shuffle(common)

    expression = expression.loc[common]
    metadata = metadata.loc[common]

    with open(args.config) as handle:
        config = json.load(handle)

    matches = [
        t for t in config["traits"]
        if t["name"] == args.trait and t.get("enabled", True)
    ]
    if len(matches) != 1:
        raise ValueError(f"Trait '{args.trait}' not found uniquely.")
    trait = matches[0]

    phenotype, valid_samples = get_trait_vector(
        trait, args.method, metadata, common
    )
    expression = expression.loc[valid_samples]

    if args.method == "regular":
        null_scores = np.asarray(
            permute_enrichment_scores(
                phenotype,
                n_permutations=args.inner_permutations,
                seed=args.seed,
                xp=np,
                verbose=False,
            ),
            dtype=float,
        )
        moment_weights = (phenotype > 0).astype(float)
    else:
        null_scores = np.asarray(
            permute_weighted_enrichment_scores(
                phenotype,
                n_permutations=args.inner_permutations,
                seed=args.seed,
            ),
            dtype=float,
        )
        moment_weights = phenotype.astype(float)

    analytic_mu, analytic_var = area_linear_null_moments(moment_weights)
    analytic_sigma = np.sqrt(analytic_var)

    print("\n" + "=" * 80)
    print("AREA P-VALUE CALIBRATION DIAGNOSTIC")
    print("=" * 80)
    print(f"Trait:                    {args.trait}")
    print(f"AREA method:              {args.method}")
    print(f"Samples:                  {len(valid_samples)}")
    print(f"Genes:                    {expression.shape[1]}")
    print(f"Inner null permutations:  {args.inner_permutations}")
    print(f"Outer null permutations:  {args.outer_permutations}")
    print("")
    print("Permutation-null moments:")
    print(f"  empirical mean: {np.mean(null_scores):.8g}")
    print(f"  empirical SD:   {np.std(null_scores):.8g}")
    print("Analytic linear-rank moments:")
    print(f"  analytic mean:  {analytic_mu:.8g}")
    print(f"  analytic SD:    {analytic_sigma:.8g}")
    print(f"  mean difference: {np.mean(null_scores)-analytic_mu:.3g}")
    print(f"  SD ratio empirical/analytic: {np.std(null_scores)/analytic_sigma:.6f}")

    orders = precompute_orders(expression, chunk_size=args.chunk_size)
    n_genes = expression.shape[1]

    # Score real phenotype once for context.
    real_es = score_genome_es(phenotype, orders, args.method, n_genes)

    p_methods = {
        "official_signed_gaussian": lambda es: p_official(es, null_scores),
        "full_null_gaussian_2s": lambda es: p_full_null_gaussian_two_sided(es, null_scores),
        "analytic_rank_gaussian_2s": lambda es: p_analytic_rank_gaussian_two_sided(
            es, phenotype, args.method
        )[0],
        "empirical_2s": lambda es: p_empirical_two_sided(es, null_scores),
    }

    outdir = Path(args.outdir).expanduser() / args.trait / args.method
    outdir.mkdir(parents=True, exist_ok=True)

    real_rows = []
    for name, fn in p_methods.items():
        p = fn(real_es)
        m = metrics(p)
        m["pvalue_method"] = name
        m["dataset"] = "real"
        real_rows.append(m)
    pd.DataFrame(real_rows).to_csv(outdir / "real_data_pvalue_method_summary.csv", index=False)

    # Global-null outer permutations.
    rng = np.random.default_rng(args.seed + 424242)
    null_rows = []

    for b in range(args.outer_permutations):
        perm = rng.permutation(phenotype)
        es = score_genome_es(perm, orders, args.method, n_genes)

        for name, fn in p_methods.items():
            p = fn(es)
            m = metrics(p)
            m["outer_permutation"] = b + 1
            m["pvalue_method"] = name
            null_rows.append(m)

        if (b + 1) % max(1, args.outer_permutations // 10) == 0 or (b + 1) == args.outer_permutations:
            latest = pd.DataFrame(null_rows)
            this_b = latest[latest["outer_permutation"] == b + 1]
            print(f"\nOuter null {b+1}/{args.outer_permutations}")
            for _, r in this_b.iterrows():
                print(
                    f"  {r['pvalue_method']:<30s} "
                    f"FDR genes={int(r['n_fdr_lt_0.05']):>5,}  "
                    f"p<.05={r['frac_p_lt_0.05']:.4f}  "
                    f"p<.01={r['frac_p_lt_0.01']:.4f}"
                )

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(outdir / "outer_null_pvalue_method_replicates.csv", index=False)

    compact = calibration_error(null_df)
    compact.to_csv(outdir / "P_VALUE_CALIBRATION_COMPARISON.csv", index=False)

    moments = pd.DataFrame([{
        "trait": args.trait,
        "method": args.method,
        "samples": len(valid_samples),
        "empirical_null_mean": float(np.mean(null_scores)),
        "empirical_null_sd": float(np.std(null_scores)),
        "analytic_null_mean": float(analytic_mu),
        "analytic_null_sd": float(analytic_sigma),
        "sd_ratio_empirical_over_analytic": float(np.std(null_scores) / analytic_sigma),
    }])
    moments.to_csv(outdir / "AREA_null_moment_check.csv", index=False)

    plot_calibration(compact, outdir)

    print("\n" + "=" * 80)
    print("CALIBRATION COMPARISON")
    print("=" * 80)
    print(compact.to_string(index=False))
    print(f"\nResults written to: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

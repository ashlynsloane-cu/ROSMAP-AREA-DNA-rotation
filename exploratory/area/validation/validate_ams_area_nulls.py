#!/usr/bin/env python3
"""
validate_ams_area_nulls.py
==============================================================================
Null-calibration suite for Regular AREA and Weighted AREA (AMS-AREA)
==============================================================================

Purpose
-------
Test whether the AREA p-value machinery is calibrated under a global null.

For each selected trait/method:
  1. Build the same phenotype vector used by run_ams_area.py.
  2. Build the method-specific permutation null ONCE.
     (The multiset of phenotype values is unchanged by outer label permutation.)
  3. Compute the real-data genome-wide p-values/FDRs.
  4. Randomly permute phenotype labels many times ("outer permutations").
  5. For every outer-null replicate, recompute genome-wide p-values and BH FDR.
  6. Record how many genes are falsely significant.

If the global-null calibration is good, BH FDR<0.05 should usually yield zero
or very few discoveries in an outer-null replicate. Systematic hundreds or
thousands of discoveries would be strong evidence of anti-conservative p-values.

This script imports phenotype encoding and Weighted AREA geometry directly from
run_ams_area.py so validation uses the same definitions as the primary analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Make the canonical parent AREA runner importable when this script lives in
# exploratory/area/validation/ and is executed directly.
VALIDATION_DIR = Path(__file__).resolve().parent
AREA_SCRIPT_DIR = VALIDATION_DIR.parent
if str(AREA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AREA_SCRIPT_DIR))

import numpy as np
import pandas as pd
from scipy import stats

from run_ams_area import (
    benjamini_hochberg,
    compute_weighted_enrichment_score,
    configure_area_import,
    encode_trait_vector,
    load_expression,
    load_metadata,
    load_sample_manifest,
    permute_weighted_enrichment_scores,
)


def vectorized_area_scores(sorted_values: np.ndarray, regular: bool) -> np.ndarray:
    """
    Vectorized AREA geometry for many genes at once.

    sorted_values shape: samples x genes, phenotype values already sorted by each
    gene's expression rank.

    For regular=True values are thresholded >0 exactly like official AREA.
    For regular=False continuous non-negative weights are retained.
    """
    values = np.asarray(sorted_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("sorted_values must be a 2D samples x genes array.")

    if regular:
        values = (values > 0).astype(float)

    n = values.shape[0]
    if n < 2:
        raise ValueError("At least two samples are required.")

    totals = values.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("At least one phenotype-sorted gene has zero total phenotype weight.")

    bin_width = 1.0 / n
    normalized = (values / totals) * bin_width
    cumulative = np.cumsum(normalized, axis=0)

    trend = np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0) * bin_width

    auc_cum = 0.5 * (
        cumulative[0, :] + cumulative[-1, :] + 2.0 * cumulative[1:-1, :].sum(axis=0)
    )
    auc_trend = 0.5 * (trend[0] + trend[-1] + 2.0 * trend[1:-1].sum())

    return (auc_cum - auc_trend) * 2.0


def null_tail_parameters(null_scores: Iterable[float]) -> Dict[str, float]:
    null = np.asarray(list(null_scores), dtype=float)
    pos = null[null > 0]
    neg = null[null < 0]

    if len(pos) < 2 or len(neg) < 2:
        raise RuntimeError("Permutation null does not contain both positive and negative tails.")

    params = {
        "pos_mu": float(pos.mean()),
        "pos_sigma": float(pos.std()),
        "neg_mu": float(neg.mean()),
        "neg_sigma": float(neg.std()),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }
    if params["pos_sigma"] == 0 or params["neg_sigma"] == 0:
        raise RuntimeError("Permutation null has zero variance in one signed tail.")
    return params


def vectorized_nes_pvalues(
    scores: np.ndarray,
    null_scores: Iterable[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized equivalent of official AREA compute_nes_pvalue().
    """
    scores = np.asarray(scores, dtype=float)
    par = null_tail_parameters(null_scores)

    nes = np.empty_like(scores)
    p = np.empty_like(scores)

    pos = scores > 0
    neg = ~pos

    if np.any(pos):
        nes[pos] = -(scores[pos] / par["pos_mu"])
        p[pos] = 1.0 - stats.norm.cdf(
            scores[pos], loc=par["pos_mu"], scale=par["pos_sigma"]
        )
    if np.any(neg):
        nes[neg] = scores[neg] / par["neg_mu"]
        p[neg] = stats.norm.cdf(
            scores[neg], loc=par["neg_mu"], scale=par["neg_sigma"]
        )

    # Official AREA can return exact 0 from floating-point tail underflow.
    # Preserve that behavior for calibration; BH handles zero values.
    return nes, p


def get_trait_vectors(
    trait: dict,
    metadata: pd.DataFrame,
    sample_order: List[str],
) -> Tuple[pd.Series, Optional[pd.Series], List[str]]:
    source_col = trait["source_column"]
    if source_col not in metadata.columns:
        raise ValueError(f"Missing source column '{source_col}' for trait '{trait['name']}'.")

    source = metadata.loc[sample_order, source_col]
    regular = encode_trait_vector(source, trait["regular"], trait["name"], "regular")

    weighted = None
    if trait["type"].lower() != "binary_only":
        weighted = encode_trait_vector(source, trait["weighted"], trait["name"], "weighted")
        if not regular.notna().equals(weighted.notna()):
            raise ValueError(
                f"Trait '{trait['name']}' uses different samples for Regular and Weighted AREA."
            )

    valid = regular.notna()
    valid_samples = regular.index[valid].tolist()
    return regular.loc[valid_samples], (
        None if weighted is None else weighted.loc[valid_samples]
    ), valid_samples


def precompute_orders(expression: pd.DataFrame, chunk_size: int) -> List[Tuple[int, int, np.ndarray]]:
    """
    Precompute sample-order indices per gene in manageable chunks.
    """
    matrix = expression.to_numpy(dtype=float)
    chunks = []
    for start in range(0, matrix.shape[1], chunk_size):
        end = min(start + chunk_size, matrix.shape[1])
        orders = np.argsort(matrix[:, start:end], axis=0, kind="stable")
        chunks.append((start, end, orders))
    return chunks


def score_genome(
    phenotype: np.ndarray,
    orders_chunks: List[Tuple[int, int, np.ndarray]],
    null_scores: Iterable[float],
    method: str,
    n_genes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    es_all = np.empty(n_genes, dtype=float)
    nes_all = np.empty(n_genes, dtype=float)
    p_all = np.empty(n_genes, dtype=float)

    regular = method == "regular"

    for start, end, orders in orders_chunks:
        sorted_values = phenotype[orders]
        es = vectorized_area_scores(sorted_values, regular=regular)
        nes, p = vectorized_nes_pvalues(es, null_scores)

        es_all[start:end] = es
        nes_all[start:end] = nes
        p_all[start:end] = p

    return es_all, nes_all, p_all


def calibration_metrics(p: np.ndarray, fdr: np.ndarray) -> dict:
    finite = np.isfinite(p)
    pv = p[finite]
    if len(pv) == 0:
        raise RuntimeError("No finite p-values.")

    return {
        "n_genes": int(len(pv)),
        "n_fdr_sig": int(np.sum(np.isfinite(fdr) & (fdr < 0.05))),
        "frac_p_lt_0.05": float(np.mean(pv < 0.05)),
        "frac_p_lt_0.01": float(np.mean(pv < 0.01)),
        "frac_p_lt_0.001": float(np.mean(pv < 0.001)),
        "frac_p_eq_0": float(np.mean(pv == 0)),
        "min_p": float(np.min(pv)),
        "median_p": float(np.median(pv)),
        "p01": float(np.quantile(pv, 0.01)),
        "p05": float(np.quantile(pv, 0.05)),
        "p10": float(np.quantile(pv, 0.10)),
        "ks_uniform_stat": float(stats.kstest(pv, "uniform").statistic),
        "ks_uniform_p": float(stats.kstest(pv, "uniform").pvalue),
    }


def write_plots(
    replicate_df: pd.DataFrame,
    hist_counts: np.ndarray,
    hist_edges: np.ndarray,
    observed_n_sig: int,
    out_prefix: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Null FDR-discovery count distribution
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(replicate_df["n_fdr_sig"], bins="auto")
    ax.axvline(observed_n_sig, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Genes with BH FDR < 0.05")
    ax.set_ylabel("Outer-null replicates")
    ax.set_title("Genome-wide false discoveries under permuted phenotype")
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_null_fdr_discovery_counts.png", dpi=300)
    plt.close(fig)

    # Aggregate raw-p histogram
    centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
    density = hist_counts / max(hist_counts.sum(), 1)
    expected = np.diff(hist_edges)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(centers, density, width=np.diff(hist_edges), align="center", alpha=0.7)
    ax.plot(centers, expected, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Raw p-value")
    ax.set_ylabel("Proportion of null gene tests")
    ax.set_title("Aggregate null p-value calibration")
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_null_pvalue_histogram.png", dpi=300)
    plt.close(fig)


def run_trait_method(
    trait: dict,
    method: str,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_order: List[str],
    outdir: Path,
    outer_permutations: int,
    inner_permutations: int,
    seed: int,
    chunk_size: int,
    official_compute_enrichment_score,
    official_permute_enrichment_scores,
) -> dict:
    regular, weighted, valid_samples = get_trait_vectors(trait, metadata, sample_order)
    phenotype_series = regular if method == "regular" else weighted

    if phenotype_series is None:
        raise ValueError(f"Weighted AREA is not defined for binary trait '{trait['name']}'.")

    phenotype = phenotype_series.to_numpy(dtype=float)
    expr = expression.loc[valid_samples]
    genes = expr.columns.to_numpy()

    method_dir = outdir / trait["name"] / method
    method_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"NULL CALIBRATION: {trait['name']} / {method.upper()}")
    print("=" * 80)
    print(f"Samples:             {len(valid_samples)}")
    print(f"Genes:               {len(genes)}")
    print(f"Inner permutations:  {inner_permutations}")
    print(f"Outer permutations:  {outer_permutations}")

    if method == "regular":
        null_scores = official_permute_enrichment_scores(
            phenotype,
            n_permutations=inner_permutations,
            seed=seed,
            xp=np,
            verbose=False,
        )
    else:
        null_scores = permute_weighted_enrichment_scores(
            phenotype,
            n_permutations=inner_permutations,
            seed=seed,
        )

    null_scores = np.asarray(null_scores, dtype=float)
    pd.DataFrame({"null_score": null_scores}).to_csv(
        method_dir / "method_permutation_null_scores.csv", index=False
    )

    orders_chunks = precompute_orders(expr, chunk_size=chunk_size)

    # Exact parity check for vectorized scoring on the first gene.
    first_gene = genes[0]
    first_ranks = expr.iloc[:, 0].to_numpy(dtype=float)
    first_order = np.argsort(first_ranks, kind="stable")
    sorted_pheno = phenotype[first_order]

    vector_es = vectorized_area_scores(sorted_pheno[:, None], regular=(method == "regular"))[0]
    if method == "regular":
        official_es, *_ = official_compute_enrichment_score(sorted_pheno, xp=np, verbose=False)
        if not np.isclose(vector_es, float(official_es), atol=1e-14, rtol=1e-12):
            raise AssertionError(
                f"Vectorized Regular AREA parity failed for {first_gene}: "
                f"{vector_es} vs {official_es}"
            )
    else:
        direct_es = compute_weighted_enrichment_score(sorted_pheno)
        if not np.isclose(vector_es, direct_es, atol=1e-14, rtol=1e-12):
            raise AssertionError(
                f"Vectorized Weighted AREA parity failed for {first_gene}: "
                f"{vector_es} vs {direct_es}"
            )

    # Real phenotype.
    real_es, real_nes, real_p = score_genome(
        phenotype, orders_chunks, null_scores, method, len(genes)
    )
    real_fdr = benjamini_hochberg(real_p)
    real_df = pd.DataFrame(
        {
            "gene_id": genes,
            "ES": real_es,
            "NES": real_nes,
            "P": real_p,
            "FDR": real_fdr,
        }
    )
    real_df.to_csv(method_dir / "real_data_results.csv", index=False)
    observed_n_sig = int(np.sum(np.isfinite(real_fdr) & (real_fdr < 0.05)))

    rng = np.random.default_rng(seed + 100003)
    replicate_rows = []
    hist_edges = np.linspace(0.0, 1.0, 101)
    hist_counts = np.zeros(len(hist_edges) - 1, dtype=np.int64)

    first_null_df = None

    for b in range(outer_permutations):
        permuted = rng.permutation(phenotype)

        es, nes, p = score_genome(
            permuted, orders_chunks, null_scores, method, len(genes)
        )
        fdr = benjamini_hochberg(p)

        metrics = calibration_metrics(p, fdr)
        metrics["outer_permutation"] = b + 1
        replicate_rows.append(metrics)

        hist_counts += np.histogram(p[np.isfinite(p)], bins=hist_edges)[0]

        if b == 0:
            first_null_df = pd.DataFrame(
                {
                    "gene_id": genes,
                    "ES": es,
                    "NES": nes,
                    "P": p,
                    "FDR": fdr,
                }
            )

        if (b + 1) % max(1, outer_permutations // 10) == 0 or (b + 1) == outer_permutations:
            print(
                f"  Outer null {b + 1:>4}/{outer_permutations}: "
                f"FDR<0.05 genes={metrics['n_fdr_sig']:,}, "
                f"raw p<0.05={metrics['frac_p_lt_0.05']:.3f}"
            )

    replicate_df = pd.DataFrame(replicate_rows)
    replicate_df.to_csv(method_dir / "outer_null_replicate_summary.csv", index=False)

    if first_null_df is not None:
        first_null_df.to_csv(method_dir / "first_outer_null_gene_results.csv", index=False)

    hist_df = pd.DataFrame(
        {
            "p_bin_left": hist_edges[:-1],
            "p_bin_right": hist_edges[1:],
            "count": hist_counts,
        }
    )
    hist_df.to_csv(method_dir / "aggregate_null_pvalue_histogram.csv", index=False)

    null_n_sig = replicate_df["n_fdr_sig"].to_numpy()
    empirical_ge_observed = (1 + np.sum(null_n_sig >= observed_n_sig)) / (outer_permutations + 1)

    summary = {
        "trait": trait["name"],
        "method": method,
        "samples": len(valid_samples),
        "genes": len(genes),
        "inner_permutations": inner_permutations,
        "outer_permutations": outer_permutations,
        "observed_fdr_sig_genes": observed_n_sig,
        "null_mean_fdr_sig_genes": float(np.mean(null_n_sig)),
        "null_median_fdr_sig_genes": float(np.median(null_n_sig)),
        "null_max_fdr_sig_genes": int(np.max(null_n_sig)),
        "null_frac_runs_any_fdr_sig": float(np.mean(null_n_sig > 0)),
        "null_frac_runs_ge_observed": float(empirical_ge_observed),
        "null_mean_frac_p_lt_0.05": float(replicate_df["frac_p_lt_0.05"].mean()),
        "null_mean_frac_p_lt_0.01": float(replicate_df["frac_p_lt_0.01"].mean()),
        "null_mean_frac_p_lt_0.001": float(replicate_df["frac_p_lt_0.001"].mean()),
        "null_mean_frac_p_eq_0": float(replicate_df["frac_p_eq_0"].mean()),
        "null_mean_median_p": float(replicate_df["median_p"].mean()),
        "vectorized_score_parity": "PASSED",
    }

    pd.DataFrame([summary]).to_csv(method_dir / "calibration_summary.csv", index=False)

    write_plots(
        replicate_df=replicate_df,
        hist_counts=hist_counts,
        hist_edges=hist_edges,
        observed_n_sig=observed_n_sig,
        out_prefix=method_dir / f"{trait['name']}_{method}",
    )

    print("\nCalibration summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Outer-null calibration for Regular/Weighted AREA.")
    p.add_argument("-e", "--expression", required=True)
    p.add_argument("-m", "--metadata", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-o", "--outdir", default="results/ams_area_validation/null_calibration")
    p.add_argument("--area-root", default=None)
    p.add_argument("--traits", default=None, help="Comma-separated trait names. Default: all enabled.")
    p.add_argument(
        "--methods",
        choices=["regular", "weighted", "both"],
        default="both",
        help="Which method(s) to validate (default: both).",
    )
    p.add_argument("--sample-manifest", default=None)
    p.add_argument("--outer-permutations", type=int, default=100)
    p.add_argument("--inner-permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=2000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.outer_permutations < 10:
        raise ValueError("--outer-permutations should be >=10; 100 is recommended.")
    if args.inner_permutations < 100:
        raise ValueError("--inner-permutations should be >=100; 1000 matches the current AREA analysis.")

    configure_area_import(args.area_root)
    from src.area.enrichment import compute_enrichment_score, permute_enrichment_scores

    expression = load_expression(Path(args.expression).expanduser())
    metadata = load_metadata(Path(args.metadata).expanduser())

    common = sorted(set(expression.index).intersection(metadata.index))
    if not common:
        raise ValueError("No overlapping sample IDs.")

    manifest = load_sample_manifest(
        Path(args.sample_manifest).expanduser() if args.sample_manifest else None
    )
    if manifest is not None:
        missing = manifest - set(common)
        if missing:
            raise ValueError(
                f"{len(missing)} sample-manifest IDs are absent from expression/metadata; "
                f"examples={sorted(missing)[:10]}"
            )
        common = [s for s in common if s in manifest]

    rng = random.Random(args.seed)
    rng.shuffle(common)

    expression = expression.loc[common]
    metadata = metadata.loc[common]

    with open(args.config) as handle:
        config = json.load(handle)

    traits = [t for t in config["traits"] if t.get("enabled", True)]
    if args.traits:
        requested = {x.strip() for x in args.traits.split(",") if x.strip()}
        available = {t["name"] for t in traits}
        missing = requested - available
        if missing:
            raise ValueError(f"Unknown/unenabled trait(s): {sorted(missing)}")
        traits = [t for t in traits if t["name"] in requested]

    methods_requested = (
        ["regular", "weighted"] if args.methods == "both" else [args.methods]
    )

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for trait in traits:
        for method in methods_requested:
            if method == "weighted" and trait["type"].lower() == "binary_only":
                print(f"Skipping Weighted AREA null calibration for binary-only trait {trait['name']}.")
                continue

            summaries.append(
                run_trait_method(
                    trait=trait,
                    method=method,
                    expression=expression,
                    metadata=metadata,
                    sample_order=common,
                    outdir=outdir,
                    outer_permutations=args.outer_permutations,
                    inner_permutations=args.inner_permutations,
                    seed=args.seed,
                    chunk_size=args.chunk_size,
                    official_compute_enrichment_score=compute_enrichment_score,
                    official_permute_enrichment_scores=permute_enrichment_scores,
                )
            )

    pd.DataFrame(summaries).to_csv(outdir / "NULL_CALIBRATION_MASTER_SUMMARY.csv", index=False)

    print("\n" + "=" * 80)
    print("NULL CALIBRATION COMPLETE")
    print("=" * 80)
    print(pd.DataFrame(summaries).to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

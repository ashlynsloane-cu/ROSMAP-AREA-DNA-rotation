#!/usr/bin/env python3
"""
run_parallel_pathology_weighted_area.py

Run matched-sample parallel Weighted AREA analyses using phenotype weights
created by characterize_pathology_geometry.py.

CERAD / plaque axis:
  1. CERAD_equal
  2. CERAD_plaqn_calibrated
  3. plaq_n_continuous

Braak / tau axis:
  1. Braak_equal
  2. Braak_nft_calibrated
  3. Braak_tangle_calibrated
  4. nft_continuous
  5. tangle_continuous

Within each axis, all methods use the identical complete-case RNA-seq samples.

The score reproduces production Weighted AREA:
  - ascending expression ranking
  - stable mergesort
  - exact compute_weighted_enrichment_score() bridge check
  - phenotype-position permutation null
  - Gaussian two-sided p-values with ddof=0
  - BH across genes

Empirical tail p-values are also written as diagnostics.

This script does not decide which representation is "best." It writes
concordance statistics so equal-stage, calibrated-stage, and continuous
pathology representations can be compared without choosing by hit count.
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


CERAD_METHODS = {
    "CERAD_equal": "CERAD_equal_weight",
    "CERAD_plaqn_calibrated": "CERAD_plaqn_calibrated_weight",
    "plaq_n_continuous": "plaq_n_AREA_weight",
}

TAU_METHODS = {
    "Braak_equal": "Braak_equal_weight",
    "Braak_nft_calibrated": "Braak_nft_calibrated_weight",
    "Braak_tangle_calibrated": "Braak_tangle_calibrated_weight",
    "nft_continuous": "nft_AREA_weight",
    "tangle_continuous": "tangle_AREA_weight",
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--metadata",
        default=(
            "results/pathology_geometry/metadata/"
            "ROSMAP_RNAseq_metadata_with_pathology_area_weights.csv"
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
    p.add_argument("--outdir", default="results/pathology_parallel_area")
    return p.parse_args()


# ---------------------------------------------------------------------
# Production score import + exact vectorization
# ---------------------------------------------------------------------

def import_score(path):
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


def trapz_unit(y):
    y = np.asarray(y, float)
    if len(y) < 2:
        return 0.0
    return float(
        0.5 * (y[0] + y[-1] + 2.0 * np.sum(y[1:-1]))
    )


def position_coefficients(n):
    trap = np.ones(n, float)
    trap[0] = 0.5
    trap[-1] = 0.5
    coeff = np.cumsum(trap[::-1])[::-1]

    bin_width = 1.0 / n
    trend = (
        np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0)
        * bin_width
    )
    return coeff, trapz_unit(trend)


def build_B(x):
    """genes x samples positional coefficient matrix."""
    n_samples, n_genes = x.shape
    coeff, trend_area = position_coefficients(n_samples)

    print(
        f"Building stable ascending ranks: "
        f"{n_genes:,} genes x {n_samples:,} samples"
    )

    orders = np.argsort(x, axis=0, kind="mergesort").T
    B = np.empty((n_genes, n_samples), dtype=np.float64)

    chunk = 2000
    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        local = orders[start:end]
        rr = np.arange(end - start)[:, None]
        B[start:end][rr, local] = coeff[None, :]

    del orders
    return B, trend_area


def score_matrix(B, trend_area, W):
    """B: genes x samples; W: samples x methods."""
    n = B.shape[1]
    totals = W.sum(axis=0)

    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("Every phenotype-weight vector needs a finite positive sum.")

    area = (B @ W) / (n * totals[None, :])
    return 2.0 * (area - trend_area)


# ---------------------------------------------------------------------
# Data loading / matched sample sets
# ---------------------------------------------------------------------

def load_expression(path, id_col):
    d = pd.read_csv(path)
    id_col = id_col or d.columns[0]

    if id_col not in d.columns:
        raise ValueError(f"Expression ID column {id_col} not found.")

    sample_ids = d[id_col].astype(str).str.strip()
    genes = [c for c in d.columns if c != id_col]

    Xdf = d[genes].apply(pd.to_numeric, errors="coerce")
    if Xdf.isna().any().any():
        bad = Xdf.columns[Xdf.isna().any()].tolist()[:10]
        raise ValueError(f"Expression contains missing/non-numeric values: {bad}")

    if sample_ids.duplicated().any():
        raise ValueError("Expression sample IDs are not unique.")

    return (
        pd.Index(sample_ids),
        np.asarray(genes, dtype=str),
        Xdf.to_numpy(float),
        id_col,
    )


def align_axis(expr_ids, Xall, meta, sample_col, flag_col, method_map):
    required = [sample_col, flag_col] + list(method_map.values())
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    d = meta.loc[
        meta[flag_col].astype(bool),
        [sample_col] + list(method_map.values()),
    ].copy()

    d[sample_col] = d[sample_col].astype(str).str.strip()

    if d[sample_col].duplicated().any():
        raise ValueError(f"Duplicate samples in {flag_col} matched set.")

    pos = pd.Series(np.arange(len(expr_ids)), index=expr_ids.astype(str))
    absent = d.loc[~d[sample_col].isin(pos.index), sample_col]
    if len(absent):
        raise ValueError(
            f"{len(absent)} metadata samples are absent from expression."
        )

    rows = pos.loc[d[sample_col]].to_numpy(int)
    X = Xall[rows, :]

    Wdf = d[list(method_map.values())].apply(pd.to_numeric, errors="coerce")
    if Wdf.isna().any().any():
        raise ValueError(f"{flag_col} matched set still contains missing weights.")

    W = Wdf.to_numpy(float)
    if W.min() < -1e-12:
        raise ValueError("Negative phenotype weights found. Inspect pathology scale.")

    return d[sample_col].to_numpy(str), X, W


# ---------------------------------------------------------------------
# Paired nulls + inference
# ---------------------------------------------------------------------

def stable_seed(base, label):
    h = hashlib.sha256(label.encode()).hexdigest()
    return (int(base) + int(h[:8], 16)) % (2**32 - 1)


def paired_nulls(W, n_perm, seed, batch_size):
    """
    Use the same permutation index for every phenotype representation within
    an axis, preserving paired comparability of the nulls.
    """
    n, k = W.shape
    coeff, trend_area = position_coefficients(n)
    totals = W.sum(axis=0)
    rng = np.random.default_rng(seed)

    out = np.empty((n_perm, k), float)

    for start in range(0, n_perm, batch_size):
        end = min(start + batch_size, n_perm)
        batch = end - start

        perms = np.empty((batch, n), dtype=np.int32)
        for i in range(batch):
            perms[i] = rng.permutation(n)

        for j in range(k):
            shuffled = W[:, j][perms]
            area = (shuffled @ coeff) / (n * totals[j])
            out[start:end, j] = 2.0 * (area - trend_area)

        if end % 2000 == 0 or end == n_perm:
            print(f"    null permutations {end:,}/{n_perm:,}")

    return out


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(m)
    out[order] = q
    return out


def empirical_p(obs, null):
    mu = np.mean(null)
    null_abs = np.sort(np.abs(null - mu))
    obs_abs = np.abs(np.asarray(obs) - mu)

    left = np.searchsorted(null_abs, obs_abs, side="left")
    n_ge = len(null_abs) - left
    return (n_ge + 1.0) / (len(null_abs) + 1.0)


def null_summary(method, null):
    mu = float(np.mean(null))
    sd = float(np.std(null))  # exact production convention
    z = (null - mu) / sd

    return {
        "method": method,
        "inner_null_n": len(null),
        "null_mean": mu,
        "null_sd_ddof0": sd,
        "null_skew": float(stats.skew(null, bias=False)),
        "null_excess_kurtosis": float(
            stats.kurtosis(null, fisher=True, bias=False)
        ),
        "null_frac_abs_z_ge_1.96": float(np.mean(np.abs(z) >= 1.95996398454)),
        "null_frac_abs_z_ge_2.576": float(np.mean(np.abs(z) >= 2.57582930355)),
        "null_frac_abs_z_ge_3.291": float(np.mean(np.abs(z) >= 3.29052673149)),
        "null_min": float(np.min(null)),
        "null_max": float(np.max(null)),
    }


# ---------------------------------------------------------------------
# Exact production bridge
# ---------------------------------------------------------------------

def verify_score(prod_score, X, B, trend_area, W, names, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[1], min(25, X.shape[1]), replace=False)
    rows = []

    for j, name in enumerate(names):
        w = W[:, j]
        vec = score_matrix(B[idx], trend_area, w[:, None])[:, 0]

        direct = []
        for g in idx:
            order = np.argsort(X[:, g], kind="mergesort")
            direct.append(float(prod_score(w[order])))
        direct = np.asarray(direct)

        diff = np.abs(vec - direct)
        rows.append({
            "method": name,
            "n_test_genes": len(idx),
            "max_abs_difference": float(np.max(diff)),
            "allclose": bool(
                np.allclose(vec, direct, rtol=1e-11, atol=1e-13)
            ),
        })

    out = pd.DataFrame(rows)
    if not out["allclose"].all():
        raise RuntimeError("Vectorized score failed production bridge.")
    return out


# ---------------------------------------------------------------------
# Cross-method robustness
# ---------------------------------------------------------------------

def compare_methods(results, qcut):
    names = list(results)
    rows = []

    for i, a in enumerate(names):
        A = results[a]

        for b in names[i + 1:]:
            B = results[b]
            za = A["z"].to_numpy(float)
            zb = B["z"].to_numpy(float)

            sa = A["padj_gaussian"].to_numpy(float) < qcut
            sb = B["padj_gaussian"].to_numpy(float) < qcut
            union = sa | sb
            both = sa & sb

            sign_all = np.sign(za) == np.sign(zb)
            sign_union = sign_all[union] if union.any() else np.array([])

            rows.append({
                "method_a": a,
                "method_b": b,
                "pearson_z": float(stats.pearsonr(za, zb).statistic),
                "spearman_z": float(stats.spearmanr(za, zb).statistic),
                "direction_agreement_all": float(np.mean(sign_all)),
                "direction_agreement_sig_union": (
                    float(np.mean(sign_union)) if len(sign_union) else np.nan
                ),
                "sig_a": int(sa.sum()),
                "sig_b": int(sb.sum()),
                "sig_both": int(both.sum()),
                "sig_union": int(union.sum()),
                "jaccard_sig": (
                    float(both.sum() / union.sum()) if union.sum() else np.nan
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Axis runner
# ---------------------------------------------------------------------

def run_axis(
    axis,
    flag,
    method_map,
    expr_ids,
    genes,
    Xall,
    meta,
    prod_score,
    a,
    root,
):
    outdir = root / axis
    outdir.mkdir(parents=True, exist_ok=True)

    names = list(method_map)
    sample_ids, X, W = align_axis(
        expr_ids, Xall, meta, a.sample_col, flag, method_map
    )

    print("\n" + "=" * 80)
    print(axis)
    print("=" * 80)
    print(f"Matched samples: {len(sample_ids):,}")
    print(f"Genes: {len(genes):,}")
    print("Methods:", ", ".join(names))

    manifest = pd.DataFrame({
        "position": np.arange(len(sample_ids)),
        "sample_id": sample_ids,
    })
    manifest.to_csv(outdir / "matched_sample_manifest.csv", index=False)

    B, trend_area = build_B(X)

    validation = verify_score(
        prod_score, X, B, trend_area, W, names,
        stable_seed(a.seed, axis + "_validation"),
    )
    validation.to_csv(
        outdir / "vectorized_scoring_validation.csv",
        index=False,
    )
    print("\nProduction score bridge:")
    print(validation.to_string(index=False))

    observed = score_matrix(B, trend_area, W)

    print("\nGenerating paired method nulls...")
    null = paired_nulls(
        W,
        a.inner_permutations,
        stable_seed(a.seed, axis + "_null"),
        a.null_batch_size,
    )

    null_df = pd.DataFrame(null, columns=names)
    null_df.insert(
        0, "permutation",
        np.arange(1, a.inner_permutations + 1)
    )
    null_df.to_csv(outdir / "paired_method_nulls.csv", index=False)

    results = {}
    null_rows = []

    for j, name in enumerate(names):
        nvec = null[:, j]
        obs = observed[:, j]

        ns = null_summary(name, nvec)
        null_rows.append(ns)

        mu = ns["null_mean"]
        sd = ns["null_sd_ddof0"]

        z = (obs - mu) / sd
        p_gauss = erfc(np.abs(z) / math.sqrt(2.0))
        q_gauss = bh(p_gauss)

        p_emp = empirical_p(obs, nvec)
        q_emp = bh(p_emp)

        r = pd.DataFrame({
            "gene_id": genes,
            "observed_es": obs,
            "null_mean": mu,
            "null_sd_ddof0": sd,
            "z": z,
            "pvalue_gaussian": p_gauss,
            "padj_gaussian": q_gauss,
            "pvalue_empirical": p_emp,
            "padj_empirical": q_emp,
        })
        r.to_csv(
            outdir / f"{name}_Weighted_AREA_results.csv",
            index=False,
        )
        results[name] = r

    null_summary_df = pd.DataFrame(null_rows)
    null_summary_df.to_csv(
        outdir / "null_diagnostics.csv",
        index=False,
    )

    pairwise = compare_methods(results, a.fdr_threshold)
    pairwise.to_csv(
        outdir / "pairwise_method_comparisons.csv",
        index=False,
    )

    wide = pd.DataFrame({"gene_id": genes})
    for name, r in results.items():
        wide[f"{name}__ES"] = r["observed_es"]
        wide[f"{name}__Z"] = r["z"]
        wide[f"{name}__P"] = r["pvalue_gaussian"]
        wide[f"{name}__FDR"] = r["padj_gaussian"]
        wide[f"{name}__SIG"] = (
            r["padj_gaussian"] < a.fdr_threshold
        ).astype(int)

    sigcols = [f"{n}__SIG" for n in names]
    wide["n_methods_significant"] = wide[sigcols].sum(axis=1)
    wide.to_csv(
        outdir / "gene_level_method_robustness_wide.csv",
        index=False,
    )

    print("\nNull diagnostics:")
    print(null_summary_df.to_string(index=False))

    print("\nPairwise concordance:")
    print(pairwise.to_string(index=False))

    print("\nFDR-significant genes:")
    for name in names:
        count = int(
            (results[name]["padj_gaussian"] < a.fdr_threshold).sum()
        )
        print(f"  {name}: {count:,}")

    del B, X

    return {
        "axis": axis,
        "n_samples": len(sample_ids),
        "methods": names,
    }


def main():
    a = args()
    root = Path(a.outdir)
    root.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(a.metadata)
    expr_ids, genes, Xall, expr_id_col = load_expression(
        a.expression, a.expression_id_col
    )

    print("=" * 80)
    print("PARALLEL PATHOLOGY WEIGHTED AREA")
    print("=" * 80)
    print(
        f"Expression: {len(expr_ids):,} samples x {len(genes):,} genes"
    )
    print("Expression sample column:", expr_id_col)
    print("Inner null permutations:", f"{a.inner_permutations:,}")

    prod_score = import_score(a.area_script)

    axes = []
    axes.append(
        run_axis(
            "CERAD_plaque",
            "CERAD_parallel_complete",
            CERAD_METHODS,
            expr_ids, genes, Xall, meta, prod_score, a, root,
        )
    )
    axes.append(
        run_axis(
            "Braak_tau",
            "Braak_parallel_complete",
            TAU_METHODS,
            expr_ids, genes, Xall, meta, prod_score, a, root,
        )
    )

    with open(root / "parallel_pathology_area_manifest.json", "w") as f:
        json.dump({
            "expression": str(Path(a.expression).resolve()),
            "metadata": str(Path(a.metadata).resolve()),
            "area_script": str(Path(a.area_script).resolve()),
            "seed": a.seed,
            "inner_permutations": a.inner_permutations,
            "fdr_threshold": a.fdr_threshold,
            "axes": axes,
            "matched_sample_design": (
                "All phenotype representations within each pathology axis "
                "use exactly the same complete-case RNA-seq participants."
            ),
            "inference": (
                "Exact Weighted AREA score; paired phenotype-position null; "
                "Gaussian two-sided p-values with ddof=0; BH across genes. "
                "Empirical tail p-values retained as diagnostics."
            ),
            "important_next_step": (
                "Calibrated and continuous phenotype versions need their own "
                "complete-null validation before final method lock."
            ),
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Results:", root)
    print(
        "Do not select a pathology representation by number of significant genes. "
        "Inspect phenotype geometry and cross-method concordance first."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
validate_weighted_area_A_to_D.py
================================

Weighted AREA validation workflow implementing A-D:

A. Exact reproduction check
   Re-run the corrected core AMS-AREA command with the original settings:
       traits: Cognitive_stage,Braak_stage,CERAD_burden
       permutations: 1000
       seed: 42
       FDR: 0.05
   Compare the new results to results/ams_area_corrected/core.
   The script STOPS if reproduction fails unless explicitly overridden.

B. Paired permutation benchmark
   Compare pre-specified cohort-independent encodings using the SAME permutation
   indices for every encoding within a trait:
       current  = primary equal-increment weighting
       sqrt     = early-stage emphasis sensitivity analysis
       square   = late-stage emphasis sensitivity analysis
       exp3     = smooth late-stage acceleration sensitivity analysis
       cube     = extreme late-stage emphasis stress test

   linear01 is excluded because it is identical to current for these traits.
   ECDF/population-based weighting is excluded because it is cohort-dependent.

C. Repeat the small complete-null screen with paired shuffles
   Defaults:
       10,000 paired inner permutations
       20 paired outer phenotype shuffles
   These are SCREENING settings, not final validation.

D. Preserve every problematic outer-null event
   For every paired outer shuffle producing >=1 FDR<0.05 hit in ANY encoding:
       - retain outer replicate ID
       - retain exact permutation indices
       - retain sample-to-sample phenotype reassignment
       - retain hit counts for all encodings on that SAME shuffle
       - retain identities of significant genes for each encoding
       - retain cross-encoding Jaccard overlap of false-hit sets

This script performs NO pathway enrichment and makes NO method-selection decision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TRAITS = ("Cognitive_stage", "Braak_stage", "CERAD_burden")
ENCODINGS = ("current", "sqrt", "square", "exp3", "cube")

ENCODING_ROLE = {
    "current": "primary",
    "sqrt": "sensitivity",
    "square": "sensitivity",
    "exp3": "sensitivity",
    "cube": "stress_test",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Weighted AREA validation steps A-D."
    )

    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--metadata",
        default="results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv",
    )
    p.add_argument(
        "--config",
        default="exploratory/area/config/rosmap_ams_area_traits.json",
    )
    p.add_argument(
        "--area-script",
        default="exploratory/area/run_ams_area.py",
    )
    p.add_argument(
        "--area-root",
        default=str(Path.home() / "Developer/area-workspace/AREA"),
    )
    p.add_argument(
        "--reference-root",
        default="results/ams_area_corrected/core",
        help="Existing corrected core results used as the reproduction reference.",
    )
    p.add_argument(
        "--manifest-root",
        default="results/ams_area_corrected/core",
        help="Corrected core sample manifests used for paired benchmark.",
    )
    p.add_argument(
        "--outdir",
        default="results/ams_area_validation/weighted_A_to_D",
    )

    # A: exact corrected-run settings
    p.add_argument("--reproduction-permutations", type=int, default=1000)
    p.add_argument("--reproduction-seed", type=int, default=42)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument(
        "--reuse-reproduction",
        action="store_true",
        help="Reuse A_reproduction_core if present instead of rerunning run_ams_area.py.",
    )
    p.add_argument(
        "--allow-reproduction-mismatch",
        action="store_true",
        help="Continue to B-D even if A fails. Not recommended.",
    )

    # B-C: paired screening settings
    p.add_argument("--inner-permutations", type=int, default=10000)
    p.add_argument("--outer-permutations", type=int, default=20)
    p.add_argument("--paired-seed", type=int, default=424242)
    p.add_argument(
        "--save-observed-gene-results",
        action="store_true",
        help="Save observed per-gene paired-benchmark results for every encoding.",
    )

    return p.parse_args()


def import_area_score(area_script):
    path = Path(area_script).resolve()
    spec = importlib.util.spec_from_file_location("run_ams_area_current", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "compute_weighted_enrichment_score"):
        raise AttributeError(
            f"{path} lacks compute_weighted_enrichment_score()."
        )

    return mod.compute_weighted_enrichment_score


def minmax01(v):
    v = np.asarray(v, dtype=float)
    lo = np.min(v)
    hi = np.max(v)
    if not np.isfinite(v).all():
        raise ValueError("Weights contain NA/Inf.")
    if np.isclose(lo, hi):
        raise ValueError("Weights have no variation.")
    return (v - lo) / (hi - lo)


def transform_weights(v, encoding):
    """
    Pre-specified cohort-independent monotonic transformations.
    Participant counts NEVER alter phenotype weights.
    """
    v = np.asarray(v, dtype=float)

    if encoding == "current":
        out = v.copy()
    else:
        x = minmax01(v)

        if encoding == "sqrt":
            out = np.sqrt(x)
        elif encoding == "square":
            out = x ** 2
        elif encoding == "exp3":
            out = np.expm1(3.0 * x) / np.expm1(3.0)
        elif encoding == "cube":
            out = x ** 3
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

    if not np.isfinite(out).all():
        raise ValueError(f"{encoding}: transformed weights contain NA/Inf.")
    if np.any(out < 0):
        raise ValueError(f"{encoding}: transformed weights contain negatives.")
    if np.sum(out) <= 0:
        raise ValueError(f"{encoding}: transformed weights sum to zero.")

    return out


def load_expression(path):
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Expression CSV has fewer than 2 columns.")

    id_col = df.columns[0]
    sample_ids = df[id_col].astype(str).str.strip()

    if sample_ids.duplicated().any():
        raise ValueError(
            f"Expression first column '{id_col}' is not a unique sample ID."
        )

    xdf = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if xdf.isna().any().any():
        bad = xdf.columns[xdf.isna().any()].tolist()[:10]
        raise ValueError(
            "Expression contains NA/non-numeric values. "
            f"Example columns: {bad}"
        )

    return (
        pd.Index(sample_ids),
        np.asarray(df.columns[1:], dtype=str),
        xdf.to_numpy(dtype=float),
        id_col,
    )


def detect_manifest_id_column(manifest, expression_ids):
    expr_set = set(expression_ids.astype(str))
    best = None

    for col in manifest.columns:
        vals = manifest[col].dropna().astype(str).str.strip()
        if len(vals) == 0:
            continue
        overlap = int(vals.isin(expr_set).sum())
        frac = overlap / len(vals)
        candidate = (frac, overlap, col)
        if best is None or candidate > best:
            best = candidate

    if best is None or best[0] < 0.90:
        raise ValueError(
            f"Could not identify manifest sample-ID column; best={best}"
        )

    return best[2]


def load_manifest(trait, manifest_root, expression_ids):
    path = (
        Path(manifest_root)
        / trait
        / f"{trait}_sample_manifest.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    m = pd.read_csv(path)

    if "weighted_weight" not in m.columns:
        raise ValueError(
            f"{path} lacks weighted_weight. "
            f"Columns: {m.columns.tolist()}"
        )

    id_col = detect_manifest_id_column(m, expression_ids)
    m[id_col] = m[id_col].astype(str).str.strip()
    m["weighted_weight"] = pd.to_numeric(
        m["weighted_weight"], errors="coerce"
    )
    m = m.dropna(subset=[id_col, "weighted_weight"]).copy()

    if m[id_col].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")

    return m, id_col, path


def align_trait(expression_ids, x_all, manifest, id_col):
    positions = pd.Series(
        np.arange(len(expression_ids), dtype=int),
        index=expression_ids.astype(str),
    )

    ids = manifest[id_col].astype(str)
    missing = ids[~ids.isin(positions.index)]

    if len(missing):
        raise ValueError(
            f"{len(missing)} manifest samples missing from expression."
        )

    row_idx = positions.loc[ids].to_numpy(dtype=int)

    return (
        x_all[row_idx, :],
        manifest["weighted_weight"].to_numpy(dtype=float),
        ids.to_numpy(dtype=str),
    )


def rank_orders(x):
    """
    genes x samples ranking matrix, ASCENDING expression within each gene.

    This exactly matches run_ams_area.py:
        order = np.argsort(ranks, kind="mergesort")

    Stable mergesort also preserves the same deterministic tie handling.
    """
    return np.argsort(x, axis=0, kind="mergesort").T


def score_all_genes(score_fn, orders, weights):
    out = np.empty(orders.shape[0], dtype=float)
    for j in range(orders.shape[0]):
        out[j] = float(score_fn(weights[orders[j]]))
    return out


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    finite = np.isfinite(p)
    pv = p[finite]
    if len(pv) == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    q_rank = ranked * m / np.arange(1, m + 1)
    q_rank = np.minimum.accumulate(q_rank[::-1])[::-1]
    q_rank = np.clip(q_rank, 0.0, 1.0)

    restored = np.empty_like(q_rank)
    restored[order] = q_rank
    out[finite] = restored

    return out


def normal_two_sided_p(z):
    z = np.asarray(z, dtype=float)
    return np.fromiter(
        (math.erfc(abs(float(v)) / math.sqrt(2.0)) for v in z),
        dtype=float,
        count=len(z),
    )


def gaussian_from_null(es, null_scores):
    mu = float(np.mean(null_scores))
    sd = float(np.std(null_scores))

    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        raise ValueError("Method-null SD is zero/non-finite.")

    z = (es - mu) / sd
    p = normal_two_sided_p(z)
    return z, p, mu, sd


def empirical_two_sided_p(es, null_scores):
    """
    Diagnostic only. Resolution is 1/(N+1), so 10k permutations cannot provide
    genome-wide-resolution empirical p-values.
    """
    mu = float(np.mean(null_scores))
    null_dev = np.sort(np.abs(null_scores - mu))
    obs_dev = np.abs(es - mu)
    first_ge = np.searchsorted(null_dev, obs_dev, side="left")
    n_ge = len(null_dev) - first_ge
    return (n_ge + 1.0) / (len(null_dev) + 1.0)


def summarize_pvalues(p, q, cutoff):
    return {
        "frac_p_lt_0.05": float(np.mean(p < 0.05)),
        "frac_p_lt_0.01": float(np.mean(p < 0.01)),
        "frac_p_lt_0.001": float(np.mean(p < 0.001)),
        "median_p": float(np.median(p)),
        "min_p": float(np.min(p)),
        "n_fdr_sig": int(np.sum(q < cutoff)),
        "min_fdr": float(np.min(q)),
    }


def null_shape(null_scores):
    x = np.asarray(null_scores, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    z = (x - mu) / sd

    return {
        "null_mean": mu,
        "null_sd": sd,
        "null_skew": float(np.mean(z ** 3)),
        "null_excess_kurtosis": float(np.mean(z ** 4) - 3.0),
        "null_min": float(np.min(x)),
        "null_max": float(np.max(x)),
    }


# ---------------------------------------------------------------------------
# A. Exact reproduction
# ---------------------------------------------------------------------------

def run_exact_reproduction(a, reproduction_root):
    reproduction_root.mkdir(parents=True, exist_ok=True)

    summary_file = reproduction_root / "AMS_AREA_run_summary.csv"

    if a.reuse_reproduction and summary_file.exists():
        print(
            "A: Reusing existing reproduction output:",
            reproduction_root,
        )
        return

    cmd = [
        sys.executable,
        a.area_script,
        "--expression",
        a.expression,
        "--metadata",
        a.metadata,
        "--config",
        a.config,
        "--traits",
        ",".join(TRAITS),
        "--outdir",
        str(reproduction_root),
        "--area-root",
        a.area_root,
        "--permutations",
        str(a.reproduction_permutations),
        "--seed",
        str(a.reproduction_seed),
        "--fdr-threshold",
        str(a.fdr_threshold),
    ]

    print("\nA: Running exact corrected-core reproduction command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def compare_reproduction_trait(
    trait,
    reference_root,
    reproduction_root,
    cutoff,
):
    ref_file = (
        Path(reference_root)
        / trait
        / f"{trait}_AMS_AREA_results.csv"
    )
    rep_file = (
        Path(reproduction_root)
        / trait
        / f"{trait}_AMS_AREA_results.csv"
    )

    ref = pd.read_csv(ref_file)
    rep = pd.read_csv(rep_file)

    required = [
        "gene_id",
        "Weighted_ES",
        "Weighted_Z_fullnull",
        "Weighted_P",
        "Weighted_FDR",
    ]

    for label, df in [("reference", ref), ("reproduction", rep)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{trait} {label} missing columns: {missing}"
            )

    merged = ref[required].merge(
        rep[required],
        on="gene_id",
        suffixes=("_ref", "_rep"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    aligned = merged["_merge"].eq("both").all()
    n_aligned = int((merged["_merge"] == "both").sum())

    metrics = {
        "trait": trait,
        "reference_rows": len(ref),
        "reproduction_rows": len(rep),
        "aligned_rows": n_aligned,
        "gene_sets_exact": bool(aligned and len(ref) == len(rep)),
    }

    numeric_pass = True

    for col in [
        "Weighted_ES",
        "Weighted_Z_fullnull",
        "Weighted_P",
        "Weighted_FDR",
    ]:
        x = merged[f"{col}_ref"].to_numpy(dtype=float)
        y = merged[f"{col}_rep"].to_numpy(dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        same_nan = np.array_equal(np.isnan(x), np.isnan(y))

        if finite.any():
            max_abs = float(np.max(np.abs(x[finite] - y[finite])))
            allclose = bool(
                np.allclose(
                    x[finite],
                    y[finite],
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
        else:
            max_abs = np.nan
            allclose = True

        metrics[f"{col}_max_abs_diff"] = max_abs
        metrics[f"{col}_allclose"] = bool(allclose and same_nan)

        numeric_pass = (
            numeric_pass
            and metrics[f"{col}_allclose"]
        )

    ref_sig = set(
        ref.loc[
            ref["Weighted_FDR"].notna()
            & (ref["Weighted_FDR"] < cutoff),
            "gene_id",
        ].astype(str)
    )
    rep_sig = set(
        rep.loc[
            rep["Weighted_FDR"].notna()
            & (rep["Weighted_FDR"] < cutoff),
            "gene_id",
        ].astype(str)
    )

    metrics["reference_fdr_hits"] = len(ref_sig)
    metrics["reproduction_fdr_hits"] = len(rep_sig)
    metrics["fdr_hit_set_exact"] = bool(ref_sig == rep_sig)

    # Also verify that reference P is mathematically consistent with its Z.
    ref_z = ref["Weighted_Z_fullnull"].to_numpy(dtype=float)
    ref_p = ref["Weighted_P"].to_numpy(dtype=float)
    finite = np.isfinite(ref_z) & np.isfinite(ref_p)

    if finite.any():
        p_from_z = normal_two_sided_p(ref_z[finite])
        metrics["reference_p_from_z_max_abs_diff"] = float(
            np.max(np.abs(p_from_z - ref_p[finite]))
        )
        metrics["reference_p_from_z_allclose"] = bool(
            np.allclose(
                p_from_z,
                ref_p[finite],
                rtol=1e-10,
                atol=1e-12,
            )
        )
    else:
        metrics["reference_p_from_z_max_abs_diff"] = np.nan
        metrics["reference_p_from_z_allclose"] = True

    metrics["reproduction_pass"] = bool(
        metrics["gene_sets_exact"]
        and numeric_pass
        and metrics["fdr_hit_set_exact"]
        and metrics["reference_p_from_z_allclose"]
    )

    return metrics


def run_reproduction_check(a, outdir):
    reproduction_root = outdir / "A_reproduction_core"
    run_exact_reproduction(a, reproduction_root)

    rows = []
    for trait in TRAITS:
        row = compare_reproduction_trait(
            trait,
            a.reference_root,
            reproduction_root,
            a.fdr_threshold,
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(
        outdir / "A_reproduction_check.csv",
        index=False,
    )

    print("\nA. REPRODUCTION CHECK")
    print(
        result[
            [
                "trait",
                "reference_fdr_hits",
                "reproduction_fdr_hits",
                "Weighted_ES_max_abs_diff",
                "Weighted_P_max_abs_diff",
                "Weighted_FDR_max_abs_diff",
                "fdr_hit_set_exact",
                "reproduction_pass",
            ]
        ].to_string(index=False)
    )

    overall = bool(result["reproduction_pass"].all())

    with open(
        outdir / "A_reproduction_status.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "overall_pass": overall,
                "reference_root": str(Path(a.reference_root).resolve()),
                "reproduction_root": str(reproduction_root.resolve()),
                "permutations": a.reproduction_permutations,
                "seed": a.reproduction_seed,
                "fdr_threshold": a.fdr_threshold,
            },
            handle,
            indent=2,
        )

    return overall, reproduction_root


def verify_benchmark_scoring_against_reproduction(
    trait,
    gene_ids,
    observed_es_current,
    reproduction_root,
):
    rep_file = (
        reproduction_root
        / trait
        / f"{trait}_AMS_AREA_results.csv"
    )
    rep = pd.read_csv(rep_file)

    check = pd.DataFrame(
        {
            "gene_id": gene_ids.astype(str),
            "benchmark_current_ES": observed_es_current,
        }
    ).merge(
        rep[["gene_id", "Weighted_ES"]],
        on="gene_id",
        how="inner",
        validate="one_to_one",
    )

    diff = np.abs(
        check["benchmark_current_ES"].to_numpy()
        - check["Weighted_ES"].to_numpy()
    )

    return {
        "trait": trait,
        "n_compared": len(check),
        "max_abs_es_diff": float(np.max(diff)),
        "allclose": bool(
            np.allclose(
                check["benchmark_current_ES"],
                check["Weighted_ES"],
                rtol=1e-10,
                atol=1e-12,
            )
        ),
    }


# ---------------------------------------------------------------------------
# B-C-D. Paired benchmark / paired complete-null screen / event retention
# ---------------------------------------------------------------------------

def paired_inner_nulls(
    score_fn,
    weights_by_encoding,
    n_permutations,
    rng,
):
    """
    One permutation index is generated per inner replicate and applied to every
    encoding. This makes the method comparison strictly paired.
    """
    n_samples = len(next(iter(weights_by_encoding.values())))
    nulls = {
        enc: np.empty(n_permutations, dtype=float)
        for enc in ENCODINGS
    }

    for i in range(n_permutations):
        perm_idx = rng.permutation(n_samples)

        for enc in ENCODINGS:
            nulls[enc][i] = float(
                score_fn(
                    weights_by_encoding[enc][perm_idx]
                )
            )

    return nulls


def jaccard(a, b):
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def save_problematic_event(
    trait_dir,
    trait,
    outer_rep,
    perm_idx,
    sample_ids,
    weights_by_encoding,
    sig_sets,
):
    event_dir = (
        trait_dir
        / "problematic_outer_events"
        / f"outer_{outer_rep:04d}"
    )
    event_dir.mkdir(parents=True, exist_ok=True)

    # Exact sample reassignment / phenotype permutation.
    perm_df = pd.DataFrame(
        {
            "target_position": np.arange(len(sample_ids)),
            "target_sample_id": sample_ids,
            "source_position": perm_idx,
            "source_sample_id": sample_ids[perm_idx],
        }
    )

    for enc in ENCODINGS:
        original = weights_by_encoding[enc]
        perm_df[f"{enc}_original_weight"] = original
        perm_df[f"{enc}_shuffled_weight"] = original[perm_idx]

    perm_df.to_csv(
        event_dir / "exact_phenotype_permutation.csv",
        index=False,
    )

    # Significant genes per encoding on the SAME null shuffle.
    hit_rows = []

    for enc in ENCODINGS:
        genes = sorted(sig_sets[enc])

        pd.DataFrame(
            {
                "gene_id": genes,
                "trait": trait,
                "encoding": enc,
                "outer_replicate": outer_rep,
            }
        ).to_csv(
            event_dir / f"{enc}_fdr_significant_genes.csv",
            index=False,
        )

        hit_rows.append(
            {
                "trait": trait,
                "outer_replicate": outer_rep,
                "encoding": enc,
                "encoding_role": ENCODING_ROLE[enc],
                "n_fdr_sig": len(genes),
            }
        )

    pd.DataFrame(hit_rows).to_csv(
        event_dir / "event_hit_counts.csv",
        index=False,
    )

    # Pairwise overlap of false-hit sets across encodings.
    overlap_rows = []

    for i, a in enumerate(ENCODINGS):
        for b in ENCODINGS[i + 1:]:
            sa = sig_sets[a]
            sb = sig_sets[b]
            overlap_rows.append(
                {
                    "trait": trait,
                    "outer_replicate": outer_rep,
                    "encoding_a": a,
                    "encoding_b": b,
                    "n_a": len(sa),
                    "n_b": len(sb),
                    "intersection": len(sa & sb),
                    "union": len(sa | sb),
                    "jaccard": jaccard(sa, sb),
                }
            )

    pd.DataFrame(overlap_rows).to_csv(
        event_dir / "cross_encoding_false_hit_overlap.csv",
        index=False,
    )

    return event_dir


def aggregate_outer(outer):
    rows = []

    for (trait, enc), g in outer.groupby(
        ["trait", "encoding"],
        sort=False,
    ):
        hits = g["n_fdr_sig"].to_numpy(dtype=float)

        rows.append(
            {
                "trait": trait,
                "encoding": enc,
                "encoding_role": ENCODING_ROLE[enc],
                "outer_runs": len(g),
                "null_mean_fdr_hits": float(np.mean(hits)),
                "null_median_fdr_hits": float(np.median(hits)),
                "null_p95_fdr_hits": float(np.quantile(hits, 0.95)),
                "null_p99_fdr_hits": float(np.quantile(hits, 0.99)),
                "null_max_fdr_hits": int(np.max(hits)),
                "null_frac_runs_any_fdr_hit": float(np.mean(hits > 0)),
                "null_mean_frac_p_lt_0.05": float(
                    g["frac_p_lt_0.05"].mean()
                ),
                "null_mean_frac_p_lt_0.01": float(
                    g["frac_p_lt_0.01"].mean()
                ),
                "null_mean_frac_p_lt_0.001": float(
                    g["frac_p_lt_0.001"].mean()
                ),
                "null_mean_median_p": float(
                    g["median_p"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_paired_trait(
    a,
    trait,
    score_fn,
    expression_ids,
    gene_ids,
    x_all,
    reproduction_root,
    outdir,
    trait_index,
):
    manifest, manifest_id_col, manifest_path = load_manifest(
        trait,
        a.manifest_root,
        expression_ids,
    )

    x, current_weights, sample_ids = align_trait(
        expression_ids,
        x_all,
        manifest,
        manifest_id_col,
    )

    trait_dir = outdir / "paired_screen" / trait
    trait_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"B-C-D: {trait}")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Samples: {len(sample_ids):,}")
    print(f"Genes: {len(gene_ids):,}")

    # Preserve participant counts as descriptive metadata only.
    counts = (
        pd.Series(current_weights)
        .value_counts()
        .sort_index()
        .rename_axis("current_weight")
        .reset_index(name="n_samples")
    )
    counts.to_csv(
        trait_dir / "phenotype_level_counts_descriptive_only.csv",
        index=False,
    )

    weights_by_encoding = {
        enc: transform_weights(current_weights, enc)
        for enc in ENCODINGS
    }

    mapping_rows = []
    for enc in ENCODINGS:
        w = weights_by_encoding[enc]
        for level in np.unique(current_weights):
            mask = current_weights == level
            transformed_unique = np.unique(w[mask])
            if len(transformed_unique) != 1:
                raise AssertionError(
                    f"{trait}/{enc}: equal phenotype levels mapped differently."
                )

            mapping_rows.append(
                {
                    "trait": trait,
                    "encoding": enc,
                    "encoding_role": ENCODING_ROLE[enc],
                    "current_weight": float(level),
                    "transformed_weight": float(transformed_unique[0]),
                    "n_samples_descriptive_only": int(np.sum(mask)),
                }
            )

    pd.DataFrame(mapping_rows).to_csv(
        trait_dir / "locked_weight_mappings.csv",
        index=False,
    )

    print("Precomputing gene expression rank orders...")
    orders = rank_orders(x)

    # Observed ES for each encoding.
    observed_es = {}
    for enc in ENCODINGS:
        print(f"  scoring observed genes: {enc}")
        observed_es[enc] = score_all_genes(
            score_fn,
            orders,
            weights_by_encoding[enc],
        )

    # Bridge benchmark scoring to exact A reproduction.
    current_es_check = verify_benchmark_scoring_against_reproduction(
        trait,
        gene_ids,
        observed_es["current"],
        reproduction_root,
    )

    pd.DataFrame([current_es_check]).to_csv(
        trait_dir / "benchmark_current_ES_vs_reproduction.csv",
        index=False,
    )

    if not current_es_check["allclose"]:
        raise RuntimeError(
            f"{trait}: benchmark current ES does not reproduce "
            "run_ams_area Weighted_ES. B-D aborted."
        )

    # B. PAIRED inner permutations: same perm_idx for all encodings.
    inner_rng = np.random.default_rng(
        a.paired_seed + 100000 * trait_index
    )

    print(
        f"B: Building paired inner nulls: "
        f"{a.inner_permutations:,} permutations"
    )
    nulls = paired_inner_nulls(
        score_fn,
        weights_by_encoding,
        a.inner_permutations,
        inner_rng,
    )

    observed_rows = []
    p_models = {}
    q_models = {}

    for enc in ENCODINGS:
        null = nulls[enc]

        pd.DataFrame({"null_es": null}).to_csv(
            trait_dir / f"{enc}_paired_method_null.csv",
            index=False,
        )

        z, p, mu, sd = gaussian_from_null(
            observed_es[enc],
            null,
        )
        q = bh_adjust(p)
        pemp = empirical_two_sided_p(
            observed_es[enc],
            null,
        )

        p_models[enc] = (mu, sd)
        q_models[enc] = q

        row = {
            "trait": trait,
            "encoding": enc,
            "encoding_role": ENCODING_ROLE[enc],
            "n_samples": len(sample_ids),
            "n_genes": len(gene_ids),
            "inner_permutations": a.inner_permutations,
            **null_shape(null),
            **summarize_pvalues(
                p,
                q,
                a.fdr_threshold,
            ),
            "empirical_p_floor": 1.0 / (a.inner_permutations + 1.0),
            "median_abs_gaussian_minus_empirical_p": float(
                np.median(np.abs(p - pemp))
            ),
        }
        observed_rows.append(row)

        print(
            f"  {enc:8s}: observed FDR hits="
            f"{row['n_fdr_sig']:,}; "
            f"p<.05={row['frac_p_lt_0.05']:.4f}"
        )

        if a.save_observed_gene_results:
            pd.DataFrame(
                {
                    "gene_id": gene_ids,
                    "ES": observed_es[enc],
                    "Z_fullnull": z,
                    "P_gaussian": p,
                    "FDR_gaussian": q,
                    "P_empirical_innernull": pemp,
                }
            ).to_csv(
                trait_dir / f"{enc}_observed_gene_results.csv",
                index=False,
            )

    observed_df = pd.DataFrame(observed_rows)
    observed_df.to_csv(
        trait_dir / "paired_observed_summary.csv",
        index=False,
    )

    # C-D. Same outer permutation is applied to ALL encodings.
    outer_rng = np.random.default_rng(
        a.paired_seed + 100000 * trait_index + 50000
    )

    outer_rows = []
    permutation_map_rows = []
    problem_rows = []

    print(
        f"C: Running {a.outer_permutations} PAIRED complete-null shuffles..."
    )

    for outer_rep in range(1, a.outer_permutations + 1):
        perm_idx = outer_rng.permutation(len(sample_ids))

        # Preserve every outer permutation, not just problematic ones.
        for target_pos, source_pos in enumerate(perm_idx):
            permutation_map_rows.append(
                {
                    "trait": trait,
                    "outer_replicate": outer_rep,
                    "target_position": target_pos,
                    "target_sample_id": sample_ids[target_pos],
                    "source_position": int(source_pos),
                    "source_sample_id": sample_ids[source_pos],
                    "original_current_weight": float(
                        current_weights[target_pos]
                    ),
                    "shuffled_current_weight": float(
                        current_weights[source_pos]
                    ),
                }
            )

        sig_sets = {}
        hit_counts = {}

        for enc in ENCODINGS:
            shuffled_weights = (
                weights_by_encoding[enc][perm_idx]
            )

            es0 = score_all_genes(
                score_fn,
                orders,
                shuffled_weights,
            )

            mu, sd = p_models[enc]
            z0 = (es0 - mu) / sd
            p0 = normal_two_sided_p(z0)
            q0 = bh_adjust(p0)

            summary = summarize_pvalues(
                p0,
                q0,
                a.fdr_threshold,
            )

            sig_mask = q0 < a.fdr_threshold
            sig_set = set(
                gene_ids[sig_mask].astype(str)
            )
            sig_sets[enc] = sig_set
            hit_counts[enc] = len(sig_set)

            outer_rows.append(
                {
                    "trait": trait,
                    "outer_replicate": outer_rep,
                    "encoding": enc,
                    "encoding_role": ENCODING_ROLE[enc],
                    **summary,
                }
            )

        any_problem = any(
            n > 0 for n in hit_counts.values()
        )

        print(
            f"  outer {outer_rep:>3}/{a.outer_permutations}: "
            + ", ".join(
                f"{enc}={hit_counts[enc]:,}"
                for enc in ENCODINGS
            )
        )

        if any_problem:
            event_dir = save_problematic_event(
                trait_dir,
                trait,
                outer_rep,
                perm_idx,
                sample_ids,
                weights_by_encoding,
                sig_sets,
            )

            problem_row = {
                "trait": trait,
                "outer_replicate": outer_rep,
                "event_directory": str(event_dir),
                "max_hits_any_encoding": max(hit_counts.values()),
                "encodings_with_hits": ",".join(
                    enc
                    for enc in ENCODINGS
                    if hit_counts[enc] > 0
                ),
            }
            problem_row.update(
                {
                    f"{enc}_hits": hit_counts[enc]
                    for enc in ENCODINGS
                }
            )
            problem_rows.append(problem_row)

    outer_df = pd.DataFrame(outer_rows)
    outer_df.to_csv(
        trait_dir / "paired_outer_null_replicate_summary.csv",
        index=False,
    )

    pd.DataFrame(permutation_map_rows).to_csv(
        trait_dir / "ALL_outer_exact_permutation_map.csv",
        index=False,
    )

    problem_df = pd.DataFrame(problem_rows)
    if problem_df.empty:
        problem_df = pd.DataFrame(
            columns=[
                "trait",
                "outer_replicate",
                "event_directory",
                "max_hits_any_encoding",
                "encodings_with_hits",
            ]
            + [f"{enc}_hits" for enc in ENCODINGS]
        )

    problem_df.to_csv(
        trait_dir / "problematic_outer_runs.csv",
        index=False,
    )

    return (
        observed_df,
        outer_df,
        problem_df,
        pd.DataFrame([current_es_check]),
    )


def main():
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "workflow": "Weighted AREA A-D validation",
        "A_exact_reproduction": {
            "traits": list(TRAITS),
            "permutations": a.reproduction_permutations,
            "seed": a.reproduction_seed,
            "fdr_threshold": a.fdr_threshold,
            "reference_root": str(Path(a.reference_root).resolve()),
        },
        "B_C_D_paired_screen": {
            "encodings": list(ENCODINGS),
            "encoding_roles": ENCODING_ROLE,
            "inner_permutations": a.inner_permutations,
            "outer_permutations": a.outer_permutations,
            "paired_seed": a.paired_seed,
            "paired_design": (
                "same permutation index applied to all encodings within trait"
            ),
        },
        "excluded_encodings": {
            "linear01": "mathematically redundant with current for these traits",
            "ecdf": "cohort-dependent; excluded for cross-cohort replicability",
        },
        "guardrail": (
            "Participant counts are descriptive only and never change weights. "
            "No pathway enrichment or method selection occurs in this script."
        ),
    }

    with open(
        outdir / "A_to_D_run_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("=" * 80)
    print("WEIGHTED AREA VALIDATION: A-D")
    print("=" * 80)
    print("A. exact corrected-run reproduction")
    print("B. paired inner-null method comparison")
    print("C. paired complete-null screening")
    print("D. preserve every problematic null event")
    print("\nNo pathways. No method selection.")

    # A
    reproduction_pass, reproduction_root = (
        run_reproduction_check(a, outdir)
    )

    if not reproduction_pass:
        message = (
            "\nA FAILED: corrected core results were not reproduced exactly. "
            "B-D will not run because method comparison is not trustworthy "
            "until the discrepancy is resolved."
        )
        print(message)

        if not a.allow_reproduction_mismatch:
            sys.exit(2)

        print(
            "WARNING: --allow-reproduction-mismatch supplied; "
            "continuing despite failed A."
        )

    print("\nA PASSED. Proceeding to paired B-D screen.")

    score_fn = import_area_score(a.area_script)
    expression_ids, gene_ids, x_all, expression_id_col = (
        load_expression(a.expression)
    )

    print(
        f"\nExpression loaded: {len(expression_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )
    print(f"Expression ID column: {expression_id_col}")

    observed_frames = []
    outer_frames = []
    problem_frames = []
    es_check_frames = []

    for trait_index, trait in enumerate(TRAITS):
        observed, outer, problems, escheck = run_paired_trait(
            a=a,
            trait=trait,
            score_fn=score_fn,
            expression_ids=expression_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            reproduction_root=reproduction_root,
            outdir=outdir,
            trait_index=trait_index,
        )

        observed_frames.append(observed)
        outer_frames.append(outer)
        problem_frames.append(problems)
        es_check_frames.append(escheck)

    observed_master = pd.concat(
        observed_frames,
        ignore_index=True,
    )
    outer_master = pd.concat(
        outer_frames,
        ignore_index=True,
    )
    problems_master = pd.concat(
        problem_frames,
        ignore_index=True,
    )
    escheck_master = pd.concat(
        es_check_frames,
        ignore_index=True,
    )

    outer_calibration = aggregate_outer(
        outer_master
    )

    master = observed_master.merge(
        outer_calibration,
        on=["trait", "encoding", "encoding_role"],
        how="left",
        validate="one_to_one",
    )

    observed_master.to_csv(
        outdir / "B_paired_observed_summary.csv",
        index=False,
    )
    outer_master.to_csv(
        outdir / "C_paired_outer_null_replicate_summary.csv",
        index=False,
    )
    outer_calibration.to_csv(
        outdir / "C_paired_outer_null_calibration_summary.csv",
        index=False,
    )
    problems_master.to_csv(
        outdir / "D_problematic_outer_runs_MASTER.csv",
        index=False,
    )
    escheck_master.to_csv(
        outdir / "B_current_ES_bridge_check.csv",
        index=False,
    )
    master.to_csv(
        outdir / "A_TO_D_MASTER_COMPARISON.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("A-D SCREEN COMPLETE")
    print("=" * 80)

    print("\nPaired null calibration:")
    print(
        outer_calibration[
            [
                "trait",
                "encoding",
                "null_frac_runs_any_fdr_hit",
                "null_mean_frac_p_lt_0.05",
                "null_mean_frac_p_lt_0.01",
                "null_max_fdr_hits",
            ]
        ].to_string(index=False)
    )

    print("\nProblematic paired outer runs:")
    if len(problems_master):
        print(problems_master.to_string(index=False))
    else:
        print("None.")

    print("\nMain outputs:")
    for filename in [
        "A_reproduction_check.csv",
        "B_current_ES_bridge_check.csv",
        "B_paired_observed_summary.csv",
        "C_paired_outer_null_replicate_summary.csv",
        "C_paired_outer_null_calibration_summary.csv",
        "D_problematic_outer_runs_MASTER.csv",
        "A_TO_D_MASTER_COMPARISON.csv",
    ]:
        print(f"  {outdir / filename}")

    print(
        "\nSTOP HERE after this screen. Do not proceed to pathways or select an "
        "encoding until A-D outputs have been inspected."
    )


if __name__ == "__main__":
    main()

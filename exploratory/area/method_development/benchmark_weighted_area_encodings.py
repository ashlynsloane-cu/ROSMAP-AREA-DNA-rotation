#!/usr/bin/env python3
"""
benchmark_weighted_area_encodings.py

Phase-1 validation framework for Weighted AREA encodings.

This script intentionally DOES NOT do pathway enrichment or choose a "best"
encoding. It first audits phenotype mappings, then (only when --stage benchmark
is requested) compares observed hit counts and complete-null calibration.

Crucially, it imports compute_weighted_enrichment_score() from the project's
current exploratory/area/run_ams_area.py, so the benchmark uses the exact
Weighted AREA ES implementation already in the repository.

Encodings:
  current  : exact cohort-independent weighting already used by corrected AREA
  sqrt     : sqrt(x); early-stage emphasis
  square   : x^2; late-stage emphasis
  exp3     : (exp(3*x)-1)/(exp(3)-1); smooth late-stage acceleration
  cube     : x^3; extreme late-stage emphasis (stress test only)

Deliberately excluded:
  linear01 : mathematically identical to current for the present traits
  ecdf     : cohort-dependent and therefore unsuitable for the primary benchmark

Recommended workflow:
  1) --stage audit
  2) inspect mappings; do not interpret biology yet
  3) --stage benchmark with 10-20 outer permutations for screening
  4) only after screening, lock candidate methods and run >=100 outer nulls
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VALID_ENCODINGS = (
    "current",   # primary: equal ordinal increments
    "sqrt",      # sensitivity: early-stage emphasis
    "square",    # sensitivity: late-stage emphasis
    "exp3",      # sensitivity: smooth late-stage acceleration
    "cube",      # stress test: extreme late-stage emphasis
)


def args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--manifest-root",
        default="results/ams_area_corrected/core",
    )
    p.add_argument(
        "--area-script",
        default="exploratory/area/run_ams_area.py",
    )
    p.add_argument(
        "--traits",
        default="Cognitive_stage,Braak_stage,CERAD_burden",
    )
    p.add_argument(
        "--encodings",
        default=",".join(VALID_ENCODINGS),
    )
    p.add_argument(
        "--stage",
        choices=("audit", "benchmark"),
        default="audit",
    )
    p.add_argument("--inner-permutations", type=int, default=10000)
    p.add_argument(
        "--outer-permutations",
        type=int,
        default=20,
        help="20 = screening only; use >=100 after candidate methods are locked.",
    )
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--outdir",
        default="results/ams_area_validation/weighted_encoding_benchmark",
    )
    p.add_argument(
        "--save-gene-results",
        action="store_true",
        help="Save observed per-gene ES/P/FDR for each trait x encoding.",
    )
    return p.parse_args()


def import_area_score(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("run_ams_area_current", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "compute_weighted_enrichment_score"):
        raise AttributeError(
            f"{path} has no compute_weighted_enrichment_score()"
        )
    return mod.compute_weighted_enrichment_score


def minmax01(v):
    v = np.asarray(v, dtype=float)
    lo, hi = np.min(v), np.max(v)
    if not np.isfinite(v).all() or np.isclose(lo, hi):
        raise ValueError("Weights must be finite and have variation.")
    return (v - lo) / (hi - lo)


def transform(v, name):
    """
    Apply one of the pre-specified cohort-independent monotonic transforms.

    All non-current transforms operate on a 0-1 normalization of the current
    weights. For the present corrected ROSMAP traits, current already spans
    0-1, so this preserves the intended ordinal endpoints exactly.
    """
    v = np.asarray(v, dtype=float)

    if name == "current":
        out = v.copy()
    else:
        x = minmax01(v)

        if name == "sqrt":
            out = np.sqrt(x)
        elif name == "square":
            out = x ** 2
        elif name == "exp3":
            out = np.expm1(3 * x) / np.expm1(3)
        elif name == "cube":
            out = x ** 3
        else:
            raise ValueError(f"Unknown encoding: {name}")

    if not np.isfinite(out).all() or np.any(out < 0) or np.sum(out) <= 0:
        raise ValueError(f"Invalid transformed weights for {name}")

    return out


def load_expression(path):
    df = pd.read_csv(path)
    id_col = df.columns[0]
    ids = df[id_col].astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError(
            f"First expression column '{id_col}' is not a unique sample ID."
        )

    xdf = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if xdf.isna().any().any():
        raise ValueError(
            "Expression matrix contains NA/non-numeric values after first column."
        )
    return pd.Index(ids), np.asarray(df.columns[1:], dtype=str), xdf.to_numpy(float), id_col


def manifest_for_trait(root, trait, expression_ids):
    path = Path(root) / trait / f"{trait}_sample_manifest.csv"
    m = pd.read_csv(path)
    if "weighted_weight" not in m.columns:
        raise ValueError(f"{path} lacks weighted_weight")

    expr = set(expression_ids.astype(str))
    best = None
    for col in m.columns:
        if col == "weighted_weight":
            continue
        vals = m[col].dropna().astype(str).str.strip()
        if len(vals) == 0:
            continue
        overlap = vals.isin(expr).sum()
        candidate = (overlap / len(vals), overlap, col)
        if best is None or candidate > best:
            best = candidate

    if best is None or best[0] < 0.90:
        raise ValueError(
            f"Could not identify sample-ID column in {path}; best={best}"
        )

    id_col = best[2]
    m[id_col] = m[id_col].astype(str).str.strip()
    m["weighted_weight"] = pd.to_numeric(
        m["weighted_weight"], errors="coerce"
    )
    m = m.dropna(subset=[id_col, "weighted_weight"]).copy()

    if m[id_col].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")
    return m, id_col, path


def align(expr_ids, x, manifest, id_col):
    positions = pd.Series(
        np.arange(len(expr_ids), dtype=int),
        index=expr_ids.astype(str),
    )
    ids = manifest[id_col].astype(str)
    missing = ids[~ids.isin(positions.index)]
    if len(missing):
        raise ValueError(
            f"{len(missing)} manifest samples missing from expression."
        )
    rows = positions.loc[ids].to_numpy(int)
    return x[rows, :], manifest["weighted_weight"].to_numpy(float)


def mapping_table(current, converted, encoding):
    d = pd.DataFrame(
        {
            "current_weight": current,
            "transformed_weight": converted,
        }
    )
    d = (
        d.groupby("current_weight", as_index=False)
        .agg(
            transformed_weight=("transformed_weight", "first"),
            n_samples=("transformed_weight", "size"),
        )
        .sort_values("current_weight")
    )
    d.insert(0, "encoding", encoding)
    return d


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def p_gaussian(es, null):
    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("Null SD is invalid.")
    z = (es - mu) / sd
    p = np.fromiter(
        (math.erfc(abs(float(v)) / math.sqrt(2)) for v in z),
        dtype=float,
        count=len(z),
    )
    return z, p, mu, sd


def p_empirical(es, null):
    mu = float(np.mean(null))
    null_dev = np.sort(np.abs(null - mu))
    obs_dev = np.abs(es - mu)
    first_ge = np.searchsorted(null_dev, obs_dev, side="left")
    nge = len(null_dev) - first_ge
    return (nge + 1) / (len(null_dev) + 1)


def method_null(score_fn, weights, n, rng):
    out = np.empty(n, float)
    for i in range(n):
        out[i] = score_fn(rng.permutation(weights))
    return out


def rank_orders(x):
    # genes x samples; descending expression within each gene
    return np.argsort(-x, axis=0, kind="mergesort").T


def score_all_genes(score_fn, orders, weights):
    out = np.empty(orders.shape[0], float)
    for j in range(orders.shape[0]):
        out[j] = score_fn(weights[orders[j]])
    return out


def null_shape(null):
    mu = np.mean(null)
    sd = np.std(null, ddof=1)
    z = (null - mu) / sd
    return {
        "null_mean": float(mu),
        "null_sd": float(sd),
        "null_skew": float(np.mean(z ** 3)),
        "null_excess_kurtosis": float(np.mean(z ** 4) - 3),
        "null_min": float(np.min(null)),
        "null_max": float(np.max(null)),
    }


def result_summary(p, q, cutoff):
    return {
        "frac_p_lt_0.05": float(np.mean(p < 0.05)),
        "frac_p_lt_0.01": float(np.mean(p < 0.01)),
        "frac_p_lt_0.001": float(np.mean(p < 0.001)),
        "median_p": float(np.median(p)),
        "min_p": float(np.min(p)),
        "n_fdr_sig": int(np.sum(q < cutoff)),
        "min_fdr": float(np.min(q)),
    }


def aggregate_outer(df):
    rows = []
    for (trait, encoding), g in df.groupby(["trait", "encoding"]):
        hits = g["n_fdr_sig"].to_numpy()
        rows.append(
            {
                "trait": trait,
                "encoding": encoding,
                "encoding_role": (
                    "primary"
                    if encoding == "current"
                    else "stress_test"
                    if encoding == "cube"
                    else "sensitivity"
                ),
                "outer_runs": len(g),
                "null_mean_fdr_hits": float(np.mean(hits)),
                "null_median_fdr_hits": float(np.median(hits)),
                "null_p95_fdr_hits": float(np.quantile(hits, 0.95)),
                "null_p99_fdr_hits": float(np.quantile(hits, 0.99)),
                "null_max_fdr_hits": int(np.max(hits)),
                "null_frac_runs_any_fdr_hit": float(np.mean(hits > 0)),
                "null_mean_frac_p_lt_0.05": float(g["frac_p_lt_0.05"].mean()),
                "null_mean_frac_p_lt_0.01": float(g["frac_p_lt_0.01"].mean()),
                "null_mean_frac_p_lt_0.001": float(g["frac_p_lt_0.001"].mean()),
                "null_mean_median_p": float(g["median_p"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    a = args()
    traits = [x.strip() for x in a.traits.split(",") if x.strip()]
    encodings = [x.strip() for x in a.encodings.split(",") if x.strip()]
    bad = [x for x in encodings if x not in VALID_ENCODINGS]
    if bad:
        raise ValueError(f"Unknown encodings {bad}; valid={VALID_ENCODINGS}")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    score_fn = import_area_score(a.area_script)
    expr_ids, gene_ids, x_all, expr_id_col = load_expression(a.expression)

    print("=" * 80)
    print("WEIGHTED AREA ENCODING BENCHMARK")
    print("=" * 80)
    print(f"Stage: {a.stage}")
    print(f"Expression: {len(expr_ids):,} samples x {len(gene_ids):,} genes")
    print(f"Expression ID column: {expr_id_col}")
    print(f"Traits: {traits}")
    print(f"Encodings: {encodings}")

    metadata = {
        "stage": a.stage,
        "expression": str(Path(a.expression).resolve()),
        "manifest_root": str(Path(a.manifest_root).resolve()),
        "area_script": str(Path(a.area_script).resolve()),
        "traits": traits,
        "encodings": encodings,
        "inner_permutations": a.inner_permutations,
        "outer_permutations": a.outer_permutations,
        "fdr_threshold": a.fdr_threshold,
        "seed": a.seed,
        "guardrail": (
            "Candidate encodings were pre-specified before benchmark results. "
            "linear01 was excluded as redundant and ecdf was excluded as "
            "cohort-dependent. This benchmark does not perform pathway "
            "enrichment or choose an encoding from biological results."
        ),
    }
    (outdir / "benchmark_run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    audit_rows = []
    observed_rows = []
    outer_rows = []

    for trait_i, trait in enumerate(traits):
        print("\n" + "=" * 80)
        print(trait)
        print("=" * 80)

        m, mid, mpath = manifest_for_trait(
            a.manifest_root, trait, expr_ids
        )
        x, current = align(expr_ids, x_all, m, mid)
        tdir = outdir / trait
        tdir.mkdir(parents=True, exist_ok=True)

        print(f"Manifest: {mpath}")
        print(f"Samples: {len(current):,}")
        print(f"Current levels: {np.unique(current).tolist()}")

        for enc in encodings:
            w = transform(current, enc)
            mt = mapping_table(current, w, enc)
            mt.to_csv(
                tdir / f"{trait}_{enc}_weight_mapping.csv",
                index=False,
            )
            audit_rows.append(
                {
                    "trait": trait,
                    "encoding": enc,
                    "encoding_role": (
                        "primary"
                        if enc == "current"
                        else "stress_test"
                        if enc == "cube"
                        else "sensitivity"
                    ),
                    "n_samples": len(w),
                    "n_current_levels": len(np.unique(current)),
                    "n_transformed_levels": len(np.unique(w)),
                    "current_min": float(np.min(current)),
                    "current_max": float(np.max(current)),
                    "transformed_min": float(np.min(w)),
                    "transformed_max": float(np.max(w)),
                    "transformed_mean": float(np.mean(w)),
                    "transformed_sd": float(np.std(w, ddof=1)),
                    "identical_to_current": bool(
                        np.allclose(w, current, atol=1e-14, rtol=1e-12)
                    ),
                }
            )

        this_audit = pd.DataFrame(
            [r for r in audit_rows if r["trait"] == trait]
        )
        print(
            this_audit[
                [
                    "encoding",
                    "n_transformed_levels",
                    "transformed_min",
                    "transformed_max",
                    "transformed_mean",
                    "identical_to_current",
                ]
            ].to_string(index=False)
        )

        if a.stage == "audit":
            continue

        print("\nPrecomputing gene rank orders...")
        orders = rank_orders(x)

        for enc_i, enc in enumerate(encodings):
            print("\n" + "-" * 72)
            print(f"{trait} / {enc}")
            print("-" * 72)

            w = transform(current, enc)
            rng = np.random.default_rng(
                a.seed + 100000 * trait_i + 1000 * enc_i
            )

            print(f"Inner null: {a.inner_permutations:,} permutations")
            null = method_null(
                score_fn, w, a.inner_permutations, rng
            )
            pd.DataFrame({"null_es": null}).to_csv(
                tdir / f"{trait}_{enc}_method_null.csv",
                index=False,
            )

            print("Observed genome-wide scoring...")
            es = score_all_genes(score_fn, orders, w)
            z, p, mu, sd = p_gaussian(es, null)
            q = bh(p)
            pemp = p_empirical(es, null)

            row = {
                "trait": trait,
                "encoding": enc,
                "encoding_role": (
                    "primary"
                    if enc == "current"
                    else "stress_test"
                    if enc == "cube"
                    else "sensitivity"
                ),
                "n_samples": len(w),
                "n_genes": len(gene_ids),
                "inner_permutations": a.inner_permutations,
                "outer_permutations": a.outer_permutations,
                **null_shape(null),
                **result_summary(p, q, a.fdr_threshold),
                "empirical_p_floor": 1 / (a.inner_permutations + 1),
                "median_abs_gaussian_minus_empirical_p": float(
                    np.median(np.abs(p - pemp))
                ),
            }
            observed_rows.append(row)

            print(
                f"Observed FDR<{a.fdr_threshold}: "
                f"{row['n_fdr_sig']:,}; "
                f"p<.05={row['frac_p_lt_0.05']:.4f}; "
                f"p<.01={row['frac_p_lt_0.01']:.4f}"
            )

            if a.save_gene_results:
                pd.DataFrame(
                    {
                        "gene_id": gene_ids,
                        "Weighted_ES": es,
                        "Weighted_Z_fullnull": z,
                        "Weighted_P_gaussian": p,
                        "Weighted_FDR_gaussian": q,
                        "Weighted_P_empirical_innernull": pemp,
                    }
                ).to_csv(
                    tdir / f"{trait}_{enc}_observed_gene_results.csv",
                    index=False,
                )

            print(
                f"Outer complete-null calibration: "
                f"{a.outer_permutations} shuffles"
            )
            for outer in range(a.outer_permutations):
                shuffled = rng.permutation(w)
                es0 = score_all_genes(
                    score_fn, orders, shuffled
                )
                _, p0, _, _ = p_gaussian(es0, null)
                q0 = bh(p0)
                s0 = result_summary(
                    p0, q0, a.fdr_threshold
                )
                outer_rows.append(
                    {
                        "trait": trait,
                        "encoding": enc,
                        "encoding_role": (
                            "primary"
                            if enc == "current"
                            else "stress_test"
                            if enc == "cube"
                            else "sensitivity"
                        ),
                        "outer_replicate": outer + 1,
                        **s0,
                    }
                )

                if (outer + 1) % 5 == 0 or outer + 1 == a.outer_permutations:
                    print(
                        f"  {outer + 1:>3}/{a.outer_permutations}: "
                        f"FDR hits={s0['n_fdr_sig']:,}; "
                        f"p<.05={s0['frac_p_lt_0.05']:.4f}"
                    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(outdir / "encoding_audit_master.csv", index=False)
    print(f"\nWrote: {outdir / 'encoding_audit_master.csv'}")

    if a.stage == "audit":
        print("\nAUDIT COMPLETE.")
        print("STOP HERE. Inspect the phenotype mappings before benchmarking.")
        return

    obs = pd.DataFrame(observed_rows)
    outer = pd.DataFrame(outer_rows)
    outer_agg = aggregate_outer(outer)

    obs.to_csv(outdir / "observed_method_summary.csv", index=False)
    outer.to_csv(
        outdir / "outer_null_replicate_summary.csv",
        index=False,
    )
    outer_agg.to_csv(
        outdir / "outer_null_calibration_summary.csv",
        index=False,
    )

    master = obs.merge(
        outer_agg,
        on=["trait", "encoding"],
        how="left",
        validate="one_to_one",
    )
    master.to_csv(
        outdir / "WEIGHTED_ENCODING_MASTER_COMPARISON.csv",
        index=False,
    )

    print("\nBENCHMARK COMPLETE.")
    print(
        "Do not choose a weighting scheme from observed hit count or pathway "
        "biology. Inspect complete-null calibration first."
    )


if __name__ == "__main__":
    main()

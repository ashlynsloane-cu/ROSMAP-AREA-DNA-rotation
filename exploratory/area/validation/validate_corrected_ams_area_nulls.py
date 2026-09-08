#!/usr/bin/env python3
"""
validate_corrected_ams_area_nulls.py
==============================================================================
Global-null validation for the corrected AMS-AREA v2 inference.
==============================================================================

Runs repeated outer phenotype-label shuffles and verifies that corrected
full-null Gaussian p-values remain calibrated for Regular and Weighted AREA.

This script should be used after changing the primary runner, and its summary
can be retained as a validation artifact for the methods/repository.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent
AREA_SCRIPT_DIR = VALIDATION_DIR.parent
if str(AREA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AREA_SCRIPT_DIR))

from run_ams_area import (
    benjamini_hochberg,
    configure_area_import,
    encode_trait_vector,
    full_null_gaussian_pvalue,
    load_expression,
    load_metadata,
    load_sample_manifest,
    permute_weighted_enrichment_scores,
)

def vectorized_area_scores(sorted_values: np.ndarray, regular: bool) -> np.ndarray:
    values = np.asarray(sorted_values, dtype=float)
    if regular:
        values = (values > 0).astype(float)

    n = values.shape[0]
    totals = values.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("Zero total phenotype weight.")

    bw = 1.0 / n
    cumulative = np.cumsum((values / totals) * bw, axis=0)
    trend = np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0) * bw

    auc = 0.5 * (
        cumulative[0] + cumulative[-1] + 2.0 * cumulative[1:-1].sum(axis=0)
    )
    trend_auc = 0.5 * (trend[0] + trend[-1] + 2.0 * trend[1:-1].sum())
    return (auc - trend_auc) * 2.0

def p_vector(scores, null):
    null = np.asarray(null, dtype=float)
    mu = null.mean()
    sd = null.std()
    z = (np.asarray(scores) - mu) / sd
    return 2.0 * __import__("scipy").stats.norm.sf(np.abs(z))

def get_trait(trait, method, metadata, sample_order):
    source = metadata.loc[sample_order, trait["source_column"]]
    reg = encode_trait_vector(source, trait["regular"], trait["name"], "regular")
    if method == "regular":
        series = reg
    else:
        if trait["type"].lower() == "binary_only":
            raise ValueError("Weighted not defined for binary-only trait.")
        wgt = encode_trait_vector(source, trait["weighted"], trait["name"], "weighted")
        if not reg.notna().equals(wgt.notna()):
            raise ValueError("Regular/Weighted sample masks differ.")
        series = wgt
    valid = series.notna()
    ids = series.index[valid].tolist()
    return series.loc[ids].to_numpy(float), ids

def metrics(p):
    p = np.asarray(p)
    fdr = benjamini_hochberg(p)
    return {
        "n_fdr_sig": int(np.sum(fdr < 0.05)),
        "frac_p_lt_0.05": float(np.mean(p < 0.05)),
        "frac_p_lt_0.01": float(np.mean(p < 0.01)),
        "frac_p_lt_0.001": float(np.mean(p < 0.001)),
        "median_p": float(np.median(p)),
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--expression", required=True)
    p.add_argument("-m", "--metadata", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-o", "--outdir", default="results/ams_area_validation/corrected_nulls")
    p.add_argument("--area-root", default=None)
    p.add_argument("--traits", default=None)
    p.add_argument("--methods", choices=["regular", "weighted", "both"], default="both")
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

    expr = load_expression(Path(args.expression).expanduser())
    meta = load_metadata(Path(args.metadata).expanduser())
    common = sorted(set(expr.index).intersection(meta.index))

    manifest = load_sample_manifest(
        Path(args.sample_manifest).expanduser() if args.sample_manifest else None
    )
    if manifest is not None:
        missing = manifest - set(common)
        if missing:
            raise ValueError(f"Manifest IDs missing: {sorted(missing)[:10]}")
        common = [x for x in common if x in manifest]

    rng_order = random.Random(args.seed)
    rng_order.shuffle(common)
    expr = expr.loc[common]
    meta = meta.loc[common]

    with open(args.config) as handle:
        config = json.load(handle)
    traits = [t for t in config["traits"] if t.get("enabled", True)]

    if args.traits:
        wanted = {x.strip() for x in args.traits.split(",") if x.strip()}
        traits = [t for t in traits if t["name"] in wanted]

    methods = ["regular", "weighted"] if args.methods == "both" else [args.methods]

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for trait in traits:
        for method in methods:
            if method == "weighted" and trait["type"].lower() == "binary_only":
                continue

            phenotype, ids = get_trait(trait, method, meta, common)
            e = expr.loc[ids]
            matrix = e.to_numpy(float)

            if method == "regular":
                null = permute_enrichment_scores(
                    phenotype, n_permutations=args.inner_permutations,
                    seed=args.seed, xp=np, verbose=False
                )
            else:
                null = permute_weighted_enrichment_scores(
                    phenotype, n_permutations=args.inner_permutations, seed=args.seed
                )
            null = np.asarray(null, float)

            # Precompute stable expression order by chunks.
            orders = []
            for start in range(0, matrix.shape[1], args.chunk_size):
                end = min(start + args.chunk_size, matrix.shape[1])
                orders.append((start, end, np.argsort(matrix[:, start:end], axis=0, kind="stable")))

            rng = np.random.default_rng(args.seed + 123456)
            rep_rows = []

            for b in range(args.outer_permutations):
                perm = rng.permutation(phenotype)
                p_all = np.empty(matrix.shape[1])

                for start, end, order in orders:
                    sorted_values = perm[order]
                    es = vectorized_area_scores(sorted_values, regular=(method == "regular"))
                    p_all[start:end] = p_vector(es, null)

                m = metrics(p_all)
                m.update({
                    "trait": trait["name"],
                    "method": method,
                    "outer_permutation": b + 1,
                })
                rep_rows.append(m)

            rep = pd.DataFrame(rep_rows)
            rep.to_csv(outdir / f"{trait['name']}_{method}_outer_nulls.csv", index=False)

            rows.append({
                "trait": trait["name"],
                "method": method,
                "samples": len(ids),
                "genes": matrix.shape[1],
                "null_mean_fdr_sig": float(rep["n_fdr_sig"].mean()),
                "null_median_fdr_sig": float(rep["n_fdr_sig"].median()),
                "null_max_fdr_sig": int(rep["n_fdr_sig"].max()),
                "null_frac_runs_any_fdr_sig": float(np.mean(rep["n_fdr_sig"] > 0)),
                "null_mean_frac_p_lt_0.05": float(rep["frac_p_lt_0.05"].mean()),
                "null_mean_frac_p_lt_0.01": float(rep["frac_p_lt_0.01"].mean()),
                "null_mean_frac_p_lt_0.001": float(rep["frac_p_lt_0.001"].mean()),
                "null_mean_median_p": float(rep["median_p"].mean()),
            })

            print(
                f"{trait['name']} / {method}: "
                f"mean FDR hits={rows[-1]['null_mean_fdr_sig']:.2f}, "
                f"p<.05={rows[-1]['null_mean_frac_p_lt_0.05']:.4f}, "
                f"p<.01={rows[-1]['null_mean_frac_p_lt_0.01']:.4f}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "CORRECTED_NULL_CALIBRATION_SUMMARY.csv", index=False)
    print("\n" + summary.to_string(index=False))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""
bootstrap_msi_stability.py
==============================================================================
Bootstrap stability analysis for AMS-AREA MSI
==============================================================================

Use this AFTER null calibration and simulation calibration.

For selected real genes:
  * stratified bootstrap resampling is performed within the Regular AREA
    control/case strata;
  * the bootstrap-specific Regular and Weighted permutation nulls are rebuilt;
  * MSI is recomputed for each bootstrap replicate;
  * per-gene confidence intervals and classification stability are reported.

This is intentionally targeted to a selected gene set rather than all ~33k
genes. Genome-wide 200x bootstrap with nested permutation nulls is unnecessary
for validating the stability of candidate archetypes and is expensive.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

# Make the canonical parent AREA runner importable when this script lives in
# exploratory/area/validation/ and is executed directly.
VALIDATION_DIR = Path(__file__).resolve().parent
AREA_SCRIPT_DIR = VALIDATION_DIR.parent
if str(AREA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AREA_SCRIPT_DIR))

import numpy as np
import pandas as pd

from run_ams_area import (
    calculate_msi,
    compute_weighted_enrichment_score,
    configure_area_import,
    encode_trait_vector,
    load_expression,
    load_metadata,
    permute_weighted_enrichment_scores,
)


def safe_nes_p(observed_es, null_scores, compute_nes_pvalue):
    null = np.asarray(null_scores, dtype=float)
    tail = null[null > 0] if observed_es > 0 else null[null < 0]
    if len(tail) < 2 or np.std(tail) == 0:
        return np.nan, np.nan
    nes, p = compute_nes_pvalue(observed_es, null_scores, use_gpu=False, xp=np)
    return float(nes), float(p)


def get_trait_vectors(trait, metadata, sample_order):
    source = metadata.loc[sample_order, trait["source_column"]]
    reg = encode_trait_vector(source, trait["regular"], trait["name"], "regular")

    if trait["type"].lower() == "binary_only":
        raise ValueError("MSI stability requires an AMS trait with a Weighted representation.")

    wgt = encode_trait_vector(source, trait["weighted"], trait["name"], "weighted")
    if not reg.notna().equals(wgt.notna()):
        raise ValueError("Regular and Weighted encodings select different samples.")

    valid = reg.notna()
    ids = reg.index[valid].tolist()
    return ids, reg.loc[ids].to_numpy(float), wgt.loc[ids].to_numpy(float)


def choose_genes(results: pd.DataFrame, genes_arg: str, gene_file: str, top_n: int) -> List[str]:
    if genes_arg:
        genes = [x.strip() for x in genes_arg.split(",") if x.strip()]
        return genes

    if gene_file:
        path = Path(gene_file).expanduser()
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = "gene_id" if "gene_id" in df.columns else df.columns[0]
            return df[col].dropna().astype(str).tolist()
        with open(path) as handle:
            return [line.strip() for line in handle if line.strip()]

    required = {"gene_id", "MSI", "Regular_FDR", "Weighted_FDR"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results file missing columns needed for automatic gene selection: {missing}")

    sig = results[
        (results["Regular_FDR"] < 0.05) | (results["Weighted_FDR"] < 0.05)
    ].copy()
    sig["abs_MSI"] = sig["MSI"].abs()
    sig = sig.sort_values("abs_MSI", ascending=False)
    return sig["gene_id"].head(top_n).astype(str).tolist()


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap stability for selected AMS-AREA genes.")
    p.add_argument("-e", "--expression", required=True)
    p.add_argument("-m", "--metadata", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--results", required=True, help="Existing <trait>_AMS_AREA_results.csv")
    p.add_argument("-o", "--outdir", default="results/ams_area_validation/bootstrap")
    p.add_argument("--area-root", default=None)
    p.add_argument("--genes", default=None, help="Comma-separated gene IDs")
    p.add_argument("--gene-file", default=None)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--bootstraps", type=int, default=200)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--msi-threshold", type=float, default=2.0)
    p.add_argument("--msi-p-floor", type=float, default=1e-15)
    return p.parse_args()


def main():
    args = parse_args()
    configure_area_import(args.area_root)

    from src.area.enrichment import (
        compute_enrichment_score,
        compute_nes_pvalue,
        permute_enrichment_scores,
    )

    expression = load_expression(Path(args.expression).expanduser())
    metadata = load_metadata(Path(args.metadata).expanduser())
    results = pd.read_csv(args.results)

    with open(args.config) as handle:
        config = json.load(handle)
    matches = [t for t in config["traits"] if t["name"] == args.trait and t.get("enabled", True)]
    if len(matches) != 1:
        raise ValueError(f"Trait '{args.trait}' not found uniquely.")
    trait = matches[0]

    common = sorted(set(expression.index).intersection(metadata.index))
    rng_order = random.Random(args.seed)
    rng_order.shuffle(common)

    ids, reg, wgt = get_trait_vectors(trait, metadata, common)
    expression = expression.loc[ids]

    genes = choose_genes(results, args.genes, args.gene_file, args.top_n)
    missing_genes = [g for g in genes if g not in expression.columns]
    if missing_genes:
        raise ValueError(f"Selected genes absent from expression, examples={missing_genes[:10]}")

    original_lookup = results.set_index("gene_id")
    missing_results = [g for g in genes if g not in original_lookup.index]
    if missing_results:
        raise ValueError(f"Selected genes absent from AMS results, examples={missing_results[:10]}")

    matrix = expression[genes].to_numpy(dtype=float)

    control_idx = np.where(reg == 0)[0]
    case_idx = np.where(reg == 1)[0]
    if len(control_idx) == 0 or len(case_idx) == 0:
        raise ValueError("Regular phenotype must contain both groups.")

    rng = np.random.default_rng(args.seed + 700001)
    rows = []

    print(f"Trait: {args.trait}")
    print(f"Genes selected: {len(genes)}")
    print(f"Bootstrap replicates: {args.bootstraps}")
    print(f"Samples: {len(ids)} ({len(control_idx)} controls / {len(case_idx)} cases)")

    for b in range(args.bootstraps):
        # Stratified bootstrap preserves Regular AREA group counts exactly.
        sampled_control = rng.choice(control_idx, size=len(control_idx), replace=True)
        sampled_case = rng.choice(case_idx, size=len(case_idx), replace=True)
        boot_idx = np.concatenate([sampled_control, sampled_case])
        rng.shuffle(boot_idx)

        reg_b = reg[boot_idx]
        wgt_b = wgt[boot_idx]
        expr_b = matrix[boot_idx, :]

        reg_null = permute_enrichment_scores(
            reg_b,
            n_permutations=args.permutations,
            seed=args.seed + b,
            xp=np,
            verbose=False,
        )
        wgt_null = permute_weighted_enrichment_scores(
            wgt_b,
            n_permutations=args.permutations,
            seed=args.seed + b,
        )

        for g_idx, gene in enumerate(genes):
            ranks = expr_b[:, g_idx]
            order = np.argsort(ranks, kind="stable")

            reg_sorted = reg_b[order]
            wgt_sorted = wgt_b[order]

            reg_es, *_ = compute_enrichment_score(reg_sorted, xp=np, verbose=False)
            reg_nes, reg_p = safe_nes_p(float(reg_es), reg_null, compute_nes_pvalue)

            wgt_es = compute_weighted_enrichment_score(wgt_sorted)
            wgt_nes, wgt_p = safe_nes_p(float(wgt_es), wgt_null, compute_nes_pvalue)

            if np.isfinite(reg_p) and np.isfinite(wgt_p):
                msi = float(calculate_msi(
                    np.array([reg_p]),
                    np.array([wgt_p]),
                    p_floor=args.msi_p_floor,
                )[0])
            else:
                msi = np.nan

            rows.append({
                "bootstrap": b + 1,
                "gene_id": gene,
                "Regular_P": reg_p,
                "Weighted_P": wgt_p,
                "MSI": msi,
            })

        if (b + 1) % max(1, args.bootstraps // 10) == 0 or b + 1 == args.bootstraps:
            print(f"  Completed bootstrap {b + 1}/{args.bootstraps}")

    boot = pd.DataFrame(rows)
    outdir = Path(args.outdir).expanduser() / args.trait
    outdir.mkdir(parents=True, exist_ok=True)
    boot.to_csv(outdir / "MSI_bootstrap_replicates.csv", index=False)

    summary_rows = []
    for gene in genes:
        x = boot.loc[boot["gene_id"] == gene, "MSI"].dropna().to_numpy()
        original = original_lookup.loc[gene]
        original_msi = float(original["MSI"])

        summary_rows.append({
            "gene_id": gene,
            "original_MSI": original_msi,
            "original_Regular_FDR": float(original["Regular_FDR"]),
            "original_Weighted_FDR": float(original["Weighted_FDR"]),
            "original_Classification": original.get("Classification", ""),
            "bootstrap_n_valid": len(x),
            "bootstrap_median_MSI": float(np.median(x)) if len(x) else np.nan,
            "bootstrap_q025_MSI": float(np.quantile(x, 0.025)) if len(x) else np.nan,
            "bootstrap_q975_MSI": float(np.quantile(x, 0.975)) if len(x) else np.nan,
            "frac_MSI_lt_0": float(np.mean(x < 0)) if len(x) else np.nan,
            "frac_MSI_gt_0": float(np.mean(x > 0)) if len(x) else np.nan,
            "frac_MSI_lt_minus_threshold": float(np.mean(x < -args.msi_threshold)) if len(x) else np.nan,
            "frac_MSI_gt_plus_threshold": float(np.mean(x > args.msi_threshold)) if len(x) else np.nan,
            "ci_excludes_zero": bool((np.quantile(x, 0.975) < 0) or (np.quantile(x, 0.025) > 0)) if len(x) else False,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "MSI_bootstrap_stability_summary.csv", index=False)

    print(f"\nResults written to: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

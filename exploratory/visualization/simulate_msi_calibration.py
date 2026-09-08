#!/usr/bin/env python3
"""
simulate_msi_calibration.py
==============================================================================
Simulation calibration for corrected AMS-AREA MSI.
==============================================================================

Uses the corrected full-null Gaussian two-sided p-values from run_ams_area.py.

MSI remains:
    MSI = log10(P_weighted_corrected / P_regular_corrected)

No final +/- threshold is assumed. The script simulates known:
  * null
  * threshold
  * dosage
  * mixed

relationships using the REAL phenotype distribution and sample size for one
ROSMAP AMS trait, then evaluates a grid of candidate |MSI| cutoffs.
"""

from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent
AREA_SCRIPT_DIR = VALIDATION_DIR.parent
if str(AREA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AREA_SCRIPT_DIR))

from run_ams_area import (
    calculate_msi,
    configure_area_import,
    encode_trait_vector,
    load_metadata,
    permute_weighted_enrichment_scores,
)

def zscore(x):
    x = np.asarray(x, float)
    sd = x.std()
    if sd == 0:
        raise ValueError("Zero-variance predictor.")
    return (x - x.mean()) / sd

def area_scores(sorted_values, regular):
    values = np.asarray(sorted_values, float)
    if regular:
        values = (values > 0).astype(float)
    n = values.shape[0]
    totals = values.sum(axis=0)
    bw = 1.0 / n
    cumulative = np.cumsum((values / totals) * bw, axis=0)
    trend = np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0) * bw
    auc = 0.5*(cumulative[0]+cumulative[-1]+2*cumulative[1:-1].sum(axis=0))
    trend_auc = 0.5*(trend[0]+trend[-1]+2*trend[1:-1].sum())
    return (auc-trend_auc)*2.0

def fullnull_p(scores, null):
    from scipy import stats
    null = np.asarray(null, float)
    z = (np.asarray(scores)-null.mean())/null.std()
    return 2.0*stats.norm.sf(np.abs(z))

def trait_vectors(trait, meta, sample_order):
    source = meta.loc[sample_order, trait["source_column"]]
    reg = encode_trait_vector(source, trait["regular"], trait["name"], "regular")
    if trait["type"].lower() == "binary_only":
        raise ValueError("MSI simulation requires an AMS trait.")
    wgt = encode_trait_vector(source, trait["weighted"], trait["name"], "weighted")
    if not reg.notna().equals(wgt.notna()):
        raise ValueError("Regular/Weighted masks differ.")
    valid = reg.notna()
    ids = reg.index[valid].tolist()
    return ids, reg.loc[ids].to_numpy(float), wgt.loc[ids].to_numpy(float)

def simulate_expression(rng, predictor, n_samples, n_genes, effect, noise_sd):
    noise = rng.normal(0, noise_sd, size=(n_samples, n_genes))
    if predictor is None or effect == 0:
        return noise
    signs = rng.choice([-1.0, 1.0], size=n_genes)
    return noise + predictor[:, None]*(effect*signs[None, :])

def evaluate(results, thresholds, p_gate):
    rows = []
    for effect in sorted(results["effect_size"].unique()):
        sub = results[results["effect_size"] == effect]
        for t in thresholds:
            row = {"effect_size": effect, "msi_threshold": t, "association_p_gate": p_gate}
            for scenario in ["null", "mixed", "dosage", "threshold"]:
                s = sub[sub["scenario"] == scenario]
                g = s[np.minimum(s["Regular_P"], s["Weighted_P"]) < p_gate]
                row[f"{scenario}_n_gated"] = len(g)
                if scenario == "null":
                    row["null_false_preference_rate"] = (
                        np.nan if len(g)==0 else float(np.mean(np.abs(g["MSI"]) > t))
                    )
                elif scenario == "mixed":
                    row["mixed_strong_preference_rate"] = (
                        np.nan if len(g)==0 else float(np.mean(np.abs(g["MSI"]) > t))
                    )
                elif scenario == "dosage":
                    row["dosage_correct_rate"] = (
                        np.nan if len(g)==0 else float(np.mean(g["MSI"] < -t))
                    )
                    row["dosage_wrong_rate"] = (
                        np.nan if len(g)==0 else float(np.mean(g["MSI"] > t))
                    )
                elif scenario == "threshold":
                    row["threshold_correct_rate"] = (
                        np.nan if len(g)==0 else float(np.mean(g["MSI"] > t))
                    )
                    row["threshold_wrong_rate"] = (
                        np.nan if len(g)==0 else float(np.mean(g["MSI"] < -t))
                    )
            rows.append(row)
    return pd.DataFrame(rows)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--metadata", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("-o", "--outdir", default="results/ams_area_validation/msi_simulation_corrected")
    p.add_argument("--area-root", default=None)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--genes-per-scenario", type=int, default=2000)
    p.add_argument("--effect-sizes", default="0,0.25,0.5,1.0,1.5")
    p.add_argument("--noise-sd", type=float, default=1.0)
    p.add_argument("--threshold-grid", default="0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5,4")
    p.add_argument("--association-p-gate", type=float, default=0.05)
    p.add_argument("--msi-p-floor", type=float, default=1e-300)
    return p.parse_args()

def main():
    args = parse_args()
    configure_area_import(args.area_root)
    from src.area.enrichment import permute_enrichment_scores

    meta = load_metadata(Path(args.metadata).expanduser())
    with open(args.config) as handle:
        config = json.load(handle)
    matches = [t for t in config["traits"] if t["name"]==args.trait and t.get("enabled", True)]
    if len(matches)!=1:
        raise ValueError(f"Trait '{args.trait}' not found uniquely.")
    trait = matches[0]

    order = sorted(meta.index)
    random.Random(args.seed).shuffle(order)
    ids, reg, wgt = trait_vectors(trait, meta, order)

    reg_null = permute_enrichment_scores(
        reg, n_permutations=args.permutations, seed=args.seed, xp=np, verbose=False
    )
    wgt_null = permute_weighted_enrichment_scores(
        wgt, n_permutations=args.permutations, seed=args.seed
    )

    zr, zw = zscore(reg), zscore(wgt)
    mixed = zscore(0.5*zr + 0.5*zw)

    effect_sizes = [float(x) for x in args.effect_sizes.split(",") if x.strip()]
    thresholds = [float(x) for x in args.threshold_grid.split(",") if x.strip()]
    scenarios = {"null": None, "threshold": zr, "dosage": zw, "mixed": mixed}
    rng = np.random.default_rng(args.seed + 900001)
    rows = []

    for effect in effect_sizes:
        for scenario, predictor in scenarios.items():
            actual = 0.0 if scenario=="null" else effect
            expr = simulate_expression(
                rng, predictor, len(ids), args.genes_per_scenario, actual, args.noise_sd
            )
            orders = np.argsort(expr, axis=0, kind="stable")
            reg_es = area_scores(reg[orders], regular=True)
            wgt_es = area_scores(wgt[orders], regular=False)
            reg_p = fullnull_p(reg_es, reg_null)
            wgt_p = fullnull_p(wgt_es, wgt_null)
            msi = calculate_msi(reg_p, wgt_p, p_floor=args.msi_p_floor)

            for i in range(args.genes_per_scenario):
                rows.append({
                    "trait": args.trait,
                    "scenario": scenario,
                    "effect_size": effect,
                    "simulation_gene": i+1,
                    "Regular_ES": reg_es[i],
                    "Regular_P": reg_p[i],
                    "Weighted_ES": wgt_es[i],
                    "Weighted_P": wgt_p[i],
                    "MSI": msi[i],
                })
            print(f"Completed {scenario:9s} effect={effect:g}")

    results = pd.DataFrame(rows)
    outdir = Path(args.outdir).expanduser()/args.trait
    outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(outdir/"MSI_simulation_gene_results.csv", index=False)

    null_msi = results.loc[results.scenario=="null","MSI"].replace([np.inf,-np.inf],np.nan).dropna()
    qs = [0.90,0.95,0.975,0.99,0.995]
    pd.DataFrame({
        "quantile": qs,
        "abs_MSI_threshold": [float(np.quantile(np.abs(null_msi),q)) for q in qs]
    }).to_csv(outdir/"MSI_null_abs_quantiles.csv", index=False)

    perf = evaluate(results, thresholds, args.association_p_gate)
    perf.to_csv(outdir/"MSI_threshold_performance.csv", index=False)

    summary = results.groupby(["scenario","effect_size"]).agg(
        n=("MSI","size"),
        median_MSI=("MSI","median"),
        q025_MSI=("MSI",lambda x:x.quantile(.025)),
        q975_MSI=("MSI",lambda x:x.quantile(.975)),
    ).reset_index()
    summary.to_csv(outdir/"MSI_simulation_scenario_summary.csv", index=False)

    print(f"Results written to {outdir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

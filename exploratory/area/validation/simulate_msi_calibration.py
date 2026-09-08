#!/usr/bin/env python3
"""
simulate_msi_calibration.py
==============================================================================
Simulation-based calibration of AMS-AREA Method Selection Index (MSI)
==============================================================================

MSI itself remains continuous:
    MSI = log10(p_weighted / p_regular)

This script does NOT assume that +/-2 is correct.

Instead, for an observed ROSMAP trait it preserves the REAL phenotype vectors
and sample size, then simulates gene expression under known relationship shapes:

  null       : no expression-phenotype association
  threshold  : expression changes with the Regular AREA state
  dosage     : expression changes progressively with Weighted AREA severity
  mixed      : equal contribution from standardized threshold + dosage predictors

For a grid of effect sizes, it calculates Regular/Weighted p-values and MSI.
It then reports:
  * MSI distributions by known generative model
  * null-derived |MSI| quantiles
  * classification performance over candidate MSI thresholds
  * false method-preference rates under the null
  * sensitivity for recovering known dosage and threshold relationships

This gives a reviewer-defensible empirical basis for any eventual categorical
MSI cutoff.
"""

from __future__ import annotations

import argparse
import json
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
    calculate_msi,
    configure_area_import,
    encode_trait_vector,
    load_metadata,
    permute_weighted_enrichment_scores,
)
from validate_ams_area_nulls import (
    vectorized_area_scores,
    vectorized_nes_pvalues,
)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0:
        raise ValueError("Cannot z-score a zero-variance predictor.")
    return (x - x.mean()) / sd


def get_trait_vectors(trait: dict, metadata: pd.DataFrame, sample_order: List[str]):
    source = metadata.loc[sample_order, trait["source_column"]]
    regular = encode_trait_vector(source, trait["regular"], trait["name"], "regular")

    if trait["type"].lower() == "binary_only":
        raise ValueError(
            f"Trait '{trait['name']}' is binary-only; MSI calibration requires an AMS trait."
        )

    weighted = encode_trait_vector(source, trait["weighted"], trait["name"], "weighted")
    if not regular.notna().equals(weighted.notna()):
        raise ValueError("Regular and Weighted encodings do not select identical samples.")

    valid = regular.notna()
    ids = regular.index[valid].tolist()
    reg = regular.loc[ids].to_numpy(dtype=float)
    wgt = weighted.loc[ids].to_numpy(dtype=float)

    if set(np.unique(reg)) != {0.0, 1.0}:
        raise ValueError("Regular encoding must contain both 0 and 1.")
    if np.std(wgt) == 0:
        raise ValueError("Weighted phenotype has zero variance.")

    return ids, reg, wgt


def score_simulated_expression(
    expression: np.ndarray,
    regular_vector: np.ndarray,
    weighted_vector: np.ndarray,
    regular_null,
    weighted_null,
    msi_p_floor: float,
):
    """
    expression: samples x genes
    """
    orders = np.argsort(expression, axis=0, kind="stable")

    reg_sorted = regular_vector[orders]
    wgt_sorted = weighted_vector[orders]

    reg_es = vectorized_area_scores(reg_sorted, regular=True)
    wgt_es = vectorized_area_scores(wgt_sorted, regular=False)

    reg_nes, reg_p = vectorized_nes_pvalues(reg_es, regular_null)
    wgt_nes, wgt_p = vectorized_nes_pvalues(wgt_es, weighted_null)

    msi = calculate_msi(reg_p, wgt_p, p_floor=msi_p_floor)

    return reg_es, reg_nes, reg_p, wgt_es, wgt_nes, wgt_p, msi


def simulate_batch(
    rng: np.random.Generator,
    predictor: Optional[np.ndarray],
    n_samples: int,
    n_genes: int,
    effect_size: float,
    noise_sd: float,
) -> np.ndarray:
    noise = rng.normal(0.0, noise_sd, size=(n_samples, n_genes))

    if predictor is None or effect_size == 0:
        return noise

    signs = rng.choice(np.array([-1.0, 1.0]), size=n_genes)
    signal = predictor[:, None] * (effect_size * signs[None, :])
    return signal + noise


def evaluate_thresholds(
    results: pd.DataFrame,
    thresholds: np.ndarray,
    association_p_gate: float,
) -> pd.DataFrame:
    rows = []

    for effect in sorted(results["effect_size"].unique()):
        sub = results[results["effect_size"] == effect]

        for t in thresholds:
            null = sub[sub["scenario"] == "null"]
            dosage = sub[sub["scenario"] == "dosage"]
            threshold = sub[sub["scenario"] == "threshold"]
            mixed = sub[sub["scenario"] == "mixed"]

            def gate(df):
                return df[np.minimum(df["Regular_P"], df["Weighted_P"]) < association_p_gate]

            null_g = gate(null)
            dosage_g = gate(dosage)
            threshold_g = gate(threshold)
            mixed_g = gate(mixed)

            rows.append({
                "effect_size": effect,
                "msi_threshold": float(t),
                "association_p_gate": association_p_gate,
                "null_n_gated": len(null_g),
                "null_false_preference_rate": (
                    np.nan if len(null_g) == 0 else float(np.mean(np.abs(null_g["MSI"]) > t))
                ),
                "mixed_n_gated": len(mixed_g),
                "mixed_strong_preference_rate": (
                    np.nan if len(mixed_g) == 0 else float(np.mean(np.abs(mixed_g["MSI"]) > t))
                ),
                "dosage_n_gated": len(dosage_g),
                "dosage_correct_rate": (
                    np.nan if len(dosage_g) == 0 else float(np.mean(dosage_g["MSI"] < -t))
                ),
                "dosage_wrong_direction_rate": (
                    np.nan if len(dosage_g) == 0 else float(np.mean(dosage_g["MSI"] > t))
                ),
                "threshold_n_gated": len(threshold_g),
                "threshold_correct_rate": (
                    np.nan if len(threshold_g) == 0 else float(np.mean(threshold_g["MSI"] > t))
                ),
                "threshold_wrong_direction_rate": (
                    np.nan if len(threshold_g) == 0 else float(np.mean(threshold_g["MSI"] < -t))
                ),
            })

    return pd.DataFrame(rows)


def make_plots(results: pd.DataFrame, performance: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # MSI distributions by scenario, one figure per effect size.
    for effect in sorted(results["effect_size"].unique()):
        sub = results[results["effect_size"] == effect]
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        scenarios = ["null", "mixed", "dosage", "threshold"]
        data = [sub.loc[sub["scenario"] == s, "MSI"].replace([np.inf, -np.inf], np.nan).dropna().values
                for s in scenarios]
        ax.boxplot(data, labels=scenarios, showfliers=False)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_ylabel("MSI = log10(p_weighted / p_regular)")
        ax.set_title(f"MSI by known simulated relationship (effect={effect})")
        fig.tight_layout()
        fig.savefig(outdir / f"MSI_simulation_effect_{effect:g}.png", dpi=300)
        plt.close(fig)

    # Correct-recovery rate vs MSI threshold, averaged across nonzero effects.
    perf = performance[performance["effect_size"] > 0].copy()
    agg = perf.groupby("msi_threshold", as_index=False).agg(
        dosage_correct_rate=("dosage_correct_rate", "mean"),
        threshold_correct_rate=("threshold_correct_rate", "mean"),
        null_false_preference_rate=("null_false_preference_rate", "mean"),
        mixed_strong_preference_rate=("mixed_strong_preference_rate", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(agg["msi_threshold"], agg["dosage_correct_rate"], label="Dosage correctly favored")
    ax.plot(agg["msi_threshold"], agg["threshold_correct_rate"], label="Threshold correctly favored")
    ax.plot(agg["msi_threshold"], agg["null_false_preference_rate"], label="Null false preference")
    ax.plot(agg["msi_threshold"], agg["mixed_strong_preference_rate"], label="Mixed strong preference")
    ax.set_xlabel("|MSI| classification threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.set_title("MSI threshold calibration")
    fig.tight_layout()
    fig.savefig(outdir / "MSI_threshold_performance.png", dpi=300)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Simulation-based MSI cutoff calibration.")
    p.add_argument("-m", "--metadata", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--trait", required=True, help="One AMS trait name, e.g. Braak_stage")
    p.add_argument("-o", "--outdir", default="results/ams_area_validation/msi_simulation")
    p.add_argument("--area-root", default=None)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--genes-per-scenario", type=int, default=2000)
    p.add_argument("--effect-sizes", default="0,0.25,0.5,1.0,1.5")
    p.add_argument("--noise-sd", type=float, default=1.0)
    p.add_argument("--threshold-grid", default="0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5,4")
    p.add_argument(
        "--association-p-gate",
        type=float,
        default=0.05,
        help="Only evaluate method classification among genes with min(raw p)<this value.",
    )
    p.add_argument("--msi-p-floor", type=float, default=1e-15)
    return p.parse_args()


def main():
    args = parse_args()
    configure_area_import(args.area_root)

    from src.area.enrichment import permute_enrichment_scores

    metadata = load_metadata(Path(args.metadata).expanduser())

    with open(args.config) as handle:
        config = json.load(handle)

    matches = [t for t in config["traits"] if t["name"] == args.trait and t.get("enabled", True)]
    if len(matches) != 1:
        raise ValueError(f"Trait '{args.trait}' not found uniquely among enabled config traits.")
    trait = matches[0]

    sample_order = sorted(metadata.index.tolist())
    rng_order = random.Random(args.seed)
    rng_order.shuffle(sample_order)

    ids, reg, wgt = get_trait_vectors(trait, metadata, sample_order)
    print(f"Trait: {args.trait}")
    print(f"Samples: {len(ids)}")

    regular_null = permute_enrichment_scores(
        reg,
        n_permutations=args.permutations,
        seed=args.seed,
        xp=np,
        verbose=False,
    )
    weighted_null = permute_weighted_enrichment_scores(
        wgt,
        n_permutations=args.permutations,
        seed=args.seed,
    )

    z_reg = zscore(reg)
    z_wgt = zscore(wgt)
    mixed_predictor = zscore(0.5 * z_reg + 0.5 * z_wgt)

    effect_sizes = [float(x) for x in args.effect_sizes.split(",") if x.strip()]
    thresholds = np.array([float(x) for x in args.threshold_grid.split(",") if x.strip()])

    rng = np.random.default_rng(args.seed + 900001)
    rows = []

    scenarios = {
        "null": None,
        "threshold": z_reg,
        "dosage": z_wgt,
        "mixed": mixed_predictor,
    }

    for effect in effect_sizes:
        for scenario, predictor in scenarios.items():
            actual_effect = 0.0 if scenario == "null" else effect
            expr = simulate_batch(
                rng=rng,
                predictor=predictor,
                n_samples=len(ids),
                n_genes=args.genes_per_scenario,
                effect_size=actual_effect,
                noise_sd=args.noise_sd,
            )

            reg_es, reg_nes, reg_p, wgt_es, wgt_nes, wgt_p, msi = score_simulated_expression(
                expression=expr,
                regular_vector=reg,
                weighted_vector=wgt,
                regular_null=regular_null,
                weighted_null=weighted_null,
                msi_p_floor=args.msi_p_floor,
            )

            for i in range(args.genes_per_scenario):
                rows.append({
                    "trait": args.trait,
                    "scenario": scenario,
                    "effect_size": effect,
                    "simulation_gene": i + 1,
                    "Regular_ES": reg_es[i],
                    "Regular_NES": reg_nes[i],
                    "Regular_P": reg_p[i],
                    "Weighted_ES": wgt_es[i],
                    "Weighted_NES": wgt_nes[i],
                    "Weighted_P": wgt_p[i],
                    "MSI": msi[i],
                })

            print(f"Completed scenario={scenario:9s} effect={effect:g}")

    results = pd.DataFrame(rows)
    outdir = Path(args.outdir).expanduser() / args.trait
    outdir.mkdir(parents=True, exist_ok=True)

    results.to_csv(outdir / "MSI_simulation_gene_results.csv", index=False)

    null_msi = results.loc[results["scenario"] == "null", "MSI"].replace([np.inf, -np.inf], np.nan).dropna()
    null_quantiles = pd.DataFrame({
        "quantile": [0.90, 0.95, 0.975, 0.99, 0.995],
        "abs_MSI_threshold": [
            float(np.quantile(np.abs(null_msi), q))
            for q in [0.90, 0.95, 0.975, 0.99, 0.995]
        ],
    })
    null_quantiles.to_csv(outdir / "MSI_null_abs_quantiles.csv", index=False)

    performance = evaluate_thresholds(
        results=results,
        thresholds=thresholds,
        association_p_gate=args.association_p_gate,
    )
    performance.to_csv(outdir / "MSI_threshold_performance.csv", index=False)

    # Compact scenario summary
    scenario_summary = results.groupby(["scenario", "effect_size"]).agg(
        n=("MSI", "size"),
        median_MSI=("MSI", "median"),
        q025_MSI=("MSI", lambda x: x.quantile(0.025)),
        q975_MSI=("MSI", lambda x: x.quantile(0.975)),
        frac_MSI_lt_minus2=("MSI", lambda x: float(np.mean(x < -2))),
        frac_MSI_gt_plus2=("MSI", lambda x: float(np.mean(x > 2))),
        median_min_p=("Regular_P", "median"),
    ).reset_index()
    scenario_summary.to_csv(outdir / "MSI_simulation_scenario_summary.csv", index=False)

    make_plots(results, performance, outdir)

    print("\nNull-derived |MSI| thresholds:")
    print(null_quantiles.to_string(index=False))
    print(f"\nResults written to: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

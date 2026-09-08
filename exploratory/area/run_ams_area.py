#!/usr/bin/env python3
"""
run_ams_area.py
==============================================================================
AMS-AREA v2: calibrated Regular + Weighted AREA
==============================================================================

Primary inferential change
--------------------------
The original AREA implementation calculates p-values by fitting a Gaussian
separately to the positive and negative halves of the permutation null. Global
null tests showed that this signed-half Gaussian procedure is anti-conservative.

AMS-AREA v2 keeps the AREA enrichment statistic and permutation framework, but
uses the FULL permutation-null distribution to calculate a two-sided Gaussian
p-value:

    z = (ES_observed - mean(null_ES)) / sd(null_ES)
    p = 2 * NormalSF(|z|)

This corrected full-null approach was empirically well calibrated under repeated
global-null phenotype shuffles for both Regular and Weighted AREA.

Important:
  * Regular AREA ES is still calculated with the official AREA implementation.
  * Weighted AREA is the continuous-weight AMS extension.
  * NES is retained as a descriptive directional score using AREA's original
    same-signed normalization, but NES is NOT used to calculate the corrected p.
  * BH correction is applied separately to corrected Regular and Weighted p-values.
  * MSI uses corrected p-values:
        MSI = log10(P_weighted_corrected / P_regular_corrected)
  * Missing phenotype values are excluded, never imputed.
  * For AMS traits, Regular and Weighted use exactly the same samples.
  * Strictly binary traits run Regular AREA only; Weighted/MSI are NA.
  * Archetype classification is DISABLED by default until MSI calibration is
    complete. Supply --msi-threshold only after simulation-based calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ==============================================================================
# AREA import / provenance
# ==============================================================================

def _candidate_area_roots(script_path: Path) -> List[Path]:
    candidates: List[Path] = []
    env_root = os.environ.get("AREA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    try:
        rosmap_repo = script_path.resolve().parents[2]
        candidates.append(rosmap_repo.parent / "AREA")
    except IndexError:
        pass
    return candidates


def configure_area_import(area_root_arg: Optional[str]) -> Optional[Path]:
    try:
        import src.area.enrichment  # noqa: F401
        return None
    except ModuleNotFoundError:
        pass

    candidates: List[Path] = []
    if area_root_arg:
        candidates.append(Path(area_root_arg).expanduser())
    candidates.extend(_candidate_area_roots(Path(__file__)))

    seen = set()
    for root in candidates:
        root = root.resolve()
        if str(root) in seen:
            continue
        seen.add(str(root))

        if (root / "src" / "area" / "enrichment.py").exists():
            sys.path.insert(0, str(root))
            try:
                import src.area.enrichment  # noqa: F401
                return root
            except ModuleNotFoundError:
                sys.path.pop(0)

    raise ModuleNotFoundError(
        "Could not import AREA. Supply --area-root /path/to/AREA, set AREA_ROOT, "
        "install AREA in the active environment, or place AREA next to this repo."
    )


def git_sha(repo_root: Optional[Path]) -> Optional[str]:
    if repo_root is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


# ==============================================================================
# Weighted AREA
# ==============================================================================

def _trapz_unit_spacing(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    return float(0.5 * (y[0] + y[-1] + 2.0 * np.sum(y[1:-1])))


def compute_weighted_enrichment_score(weight_vector: Iterable[float]) -> float:
    """
    Continuous-weight generalization of AREA.

    For a 0/1 vector this collapses to the same ES as Regular AREA.
    """
    weights = np.asarray(weight_vector, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Weighted phenotype vector must be one-dimensional.")
    if len(weights) < 2:
        raise ValueError("At least two samples are required.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weighted phenotype contains NA/Inf.")
    if np.any(weights < 0):
        raise ValueError("Weighted AREA requires non-negative phenotype weights.")

    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("Weighted phenotype has zero total weight.")

    n = len(weights)
    bin_width = 1.0 / n
    normalized = (weights / total_weight) * bin_width
    cumulative = np.cumsum(normalized)

    trend = np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0) * bin_width
    return (_trapz_unit_spacing(cumulative) - _trapz_unit_spacing(trend)) * 2.0


def permute_weighted_enrichment_scores(
    weight_vector: Iterable[float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> List[float]:
    weights = np.asarray(weight_vector, dtype=float)
    rng = np.random.default_rng(seed)
    return [
        compute_weighted_enrichment_score(rng.permutation(weights))
        for _ in range(n_permutations)
    ]


# ==============================================================================
# Corrected inference
# ==============================================================================

def full_null_gaussian_pvalue(
    observed_es: float,
    null_scores: Iterable[float],
) -> Tuple[float, float]:
    """
    Calibrated two-sided p-value from the FULL permutation-null ES distribution.

    Returns:
        pvalue, zscore
    """
    null = np.asarray(list(null_scores), dtype=float)
    if null.size < 10:
        raise ValueError("Null distribution is too small.")
    if not np.all(np.isfinite(null)):
        raise ValueError("Null distribution contains non-finite values.")

    mu = float(np.mean(null))
    sigma = float(np.std(null))
    if sigma <= 0:
        raise RuntimeError("Permutation null has zero variance.")

    z = (float(observed_es) - mu) / sigma
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(np.clip(p, 0.0, 1.0)), float(z)


def descriptive_nes(
    observed_es: float,
    null_scores: Iterable[float],
) -> float:
    """
    Preserve AREA's same-signed NES normalization as a descriptive direction score.

    Unlike the legacy implementation, this function does NOT generate a p-value.
    """
    null = np.asarray(list(null_scores), dtype=float)

    if observed_es > 0:
        subset = null[null > 0]
        if subset.size == 0:
            return np.nan
        mu = float(np.mean(subset))
        if mu == 0:
            return np.nan
        return float(-(observed_es / mu))
    else:
        subset = null[null < 0]
        if subset.size == 0:
            return np.nan
        mu = float(np.mean(subset))
        if mu == 0:
            return np.nan
        return float(observed_es / mu)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out

    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def calculate_msi(
    p_regular: np.ndarray,
    p_weighted: np.ndarray,
    p_floor: float = 1e-300,
) -> np.ndarray:
    """
    Continuous Method Selection Index using corrected p-values.

        MSI = log10(p_weighted / p_regular)

    Negative -> Weighted stronger.
    Positive -> Regular stronger.
    """
    pr = np.clip(np.asarray(p_regular, dtype=float), p_floor, 1.0)
    pw = np.clip(np.asarray(p_weighted, dtype=float), p_floor, 1.0)
    return np.log10(pw / pr)


def classify_gene(
    regular_fdr: float,
    weighted_fdr: Optional[float],
    msi: Optional[float],
    fdr_threshold: float,
    msi_threshold: Optional[float],
    binary_only: bool,
) -> str:
    reg_sig = np.isfinite(regular_fdr) and regular_fdr < fdr_threshold

    if binary_only:
        return "Binary Association (Regular-only)" if reg_sig else "Non-Significant Driver"

    wgt_sig = (
        weighted_fdr is not None
        and np.isfinite(weighted_fdr)
        and weighted_fdr < fdr_threshold
    )

    if not reg_sig and not wgt_sig:
        return "Non-Significant Driver"

    if msi_threshold is None:
        return "Association - MSI classification pending calibration"

    if msi is None or not np.isfinite(msi):
        return "Unclassified"

    if msi < -msi_threshold:
        return "Dosage Accumulator (Weighted)"
    if msi > msi_threshold:
        return "State-Transition Trigger (Regular)"
    return "Co-Progressive Driver"


# ==============================================================================
# Trait encoding
# ==============================================================================

def canonical_token(value) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")

    s = str(value).strip()
    if not s:
        return None

    try:
        f = float(s)
        if np.isfinite(f):
            if f.is_integer():
                return str(int(f))
            return format(f, ".15g")
    except ValueError:
        pass
    return s


def encode_mapping(series: pd.Series, spec: dict, trait_name: str, role: str) -> pd.Series:
    mapping = {canonical_token(k): float(v) for k, v in spec["mapping"].items()}
    excluded = {canonical_token(v) for v in spec.get("exclude_values", [])}
    result = pd.Series(np.nan, index=series.index, dtype=float)
    unexpected = set()

    for idx, value in series.items():
        token = canonical_token(value)
        if token is None:
            continue
        if token in mapping:
            result.loc[idx] = mapping[token]
        elif token in excluded:
            continue
        else:
            unexpected.add(token)

    if unexpected:
        raise ValueError(
            f"Trait '{trait_name}' {role} mapping encountered unexpected values "
            f"{sorted(unexpected)}. Add them explicitly to mapping/exclude_values."
        )
    return result


def encode_threshold(series: pd.Series, spec: dict, trait_name: str, role: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = x.notna()

    if "valid_min" in spec and (valid & (x < float(spec["valid_min"]))).any():
        raise ValueError(f"Trait '{trait_name}' {role} has values below valid_min.")
    if "valid_max" in spec and (valid & (x > float(spec["valid_max"]))).any():
        raise ValueError(f"Trait '{trait_name}' {role} has values above valid_max.")

    case = valid.copy()
    control = valid.copy()

    if "case_min" in spec:
        case &= x >= float(spec["case_min"])
    if "case_max" in spec:
        case &= x <= float(spec["case_max"])
    if "control_min" in spec:
        control &= x >= float(spec["control_min"])
    if "control_max" in spec:
        control &= x <= float(spec["control_max"])

    if (case & control).any():
        raise ValueError(f"Trait '{trait_name}' {role} case/control ranges overlap.")

    result.loc[control] = 0.0
    result.loc[case] = 1.0

    unassigned = valid & result.isna()
    if unassigned.any() and not spec.get("allow_unassigned", False):
        examples = sorted(pd.unique(x.loc[unassigned]))[:10]
        raise ValueError(
            f"Trait '{trait_name}' {role} has values in neither threshold group: "
            f"{examples}. Define ranges or set allow_unassigned=true intentionally."
        )
    return result


def encode_linear(series: pd.Series, spec: dict, trait_name: str, role: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = x.notna()

    valid_min = float(spec.get("valid_min", spec["anchor_min"]))
    valid_max = float(spec.get("valid_max", spec["anchor_max"]))
    bad = valid & ((x < valid_min) | (x > valid_max))
    if bad.any():
        examples = sorted(pd.unique(x.loc[bad]))[:10]
        raise ValueError(
            f"Trait '{trait_name}' {role} contains values outside "
            f"[{valid_min}, {valid_max}], examples={examples}."
        )

    anchor_min = float(spec["anchor_min"])
    anchor_max = float(spec["anchor_max"])
    if anchor_max <= anchor_min:
        raise ValueError("anchor_max must exceed anchor_min.")

    w = (x.loc[valid] - anchor_min) / (anchor_max - anchor_min)
    if spec.get("clip", False):
        w = w.clip(0.0, 1.0)

    direction = spec.get("direction", "higher")
    if direction == "lower":
        w = 1.0 - w
    elif direction != "higher":
        raise ValueError("direction must be 'higher' or 'lower'.")

    if (w < 0).any():
        raise ValueError("Linear encoding generated negative weights.")

    result.loc[valid] = w.astype(float)
    return result


def encode_trait_vector(series: pd.Series, spec: dict, trait_name: str, role: str) -> pd.Series:
    kind = spec["kind"].lower()
    if kind == "mapping":
        return encode_mapping(series, spec, trait_name, role)
    if kind == "threshold":
        return encode_threshold(series, spec, trait_name, role)
    if kind == "linear":
        return encode_linear(series, spec, trait_name, role)
    raise ValueError(f"Unsupported encoding kind '{kind}' for trait '{trait_name}'.")


# ==============================================================================
# Input loading
# ==============================================================================

def load_expression(path: Path) -> pd.DataFrame:
    print(f"Loading normalized expression: {path}")
    df = pd.read_csv(path)

    if "sample_id" not in df.columns:
        first = df.columns[0]
        if first.lower() in {"unnamed: 0", "id", "sample", "sampleid"}:
            df = df.rename(columns={first: "sample_id"})
        else:
            raise ValueError(
                "Expression must be samples x genes with a 'sample_id' column."
            )

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in expression.")
    df = df.set_index("sample_id")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate gene columns in expression.")

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Expression contains non-numeric or missing values.")
    if (numeric < 0).any().any():
        raise ValueError("Normalized count matrix contains negative values.")

    print(f"  Expression dimensions: {numeric.shape[0]} samples x {numeric.shape[1]} genes")
    return numeric


def load_metadata(path: Path) -> pd.DataFrame:
    print(f"Loading metadata: {path}")
    meta = pd.read_csv(path)
    if "sample_id" not in meta.columns:
        raise ValueError("Metadata must contain a 'sample_id' column.")

    meta["sample_id"] = meta["sample_id"].astype(str).str.strip()
    if meta["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in metadata.")
    meta = meta.set_index("sample_id")
    print(f"  Metadata rows: {len(meta)}")
    return meta


def load_sample_manifest(path: Optional[Path]) -> Optional[set]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Sample manifest not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        ids = df["sample_id"] if "sample_id" in df.columns else df.iloc[:, 0]
    else:
        with open(path) as handle:
            ids = pd.Series([line.strip() for line in handle if line.strip()])

    sample_set = set(ids.astype(str).str.strip())
    if not sample_set:
        raise ValueError("Sample manifest contains no sample IDs.")
    return sample_set


# ==============================================================================
# Analysis
# ==============================================================================

def analyze_trait(
    trait: dict,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_order: List[str],
    outdir: Path,
    n_permutations: int,
    seed: int,
    fdr_threshold: float,
    msi_threshold: Optional[float],
    msi_p_floor: float,
    compute_enrichment_score,
    permute_enrichment_scores,
) -> dict:

    name = trait["name"]
    source_column = trait["source_column"]
    binary_only = trait["type"].lower() == "binary_only"

    if source_column not in metadata.columns:
        raise ValueError(f"Trait '{name}' requires missing metadata column '{source_column}'.")

    source = metadata.loc[sample_order, source_column]
    regular = encode_trait_vector(source, trait["regular"], name, "regular")

    weighted = None
    if not binary_only:
        weighted = encode_trait_vector(source, trait["weighted"], name, "weighted")
        if not regular.notna().equals(weighted.notna()):
            raise ValueError(
                f"Trait '{name}' would use different Regular vs Weighted samples."
            )

    valid = regular.notna()
    valid_samples = regular.index[valid].tolist()
    if len(valid_samples) < 4:
        raise ValueError(f"Trait '{name}' has fewer than 4 analyzable samples.")

    reg_vec = regular.loc[valid_samples].to_numpy(dtype=float)
    if set(np.unique(reg_vec).tolist()) != {0.0, 1.0}:
        raise ValueError(f"Trait '{name}' Regular encoding must contain both 0 and 1.")

    wgt_vec = None
    if not binary_only:
        wgt_vec = weighted.loc[valid_samples].to_numpy(dtype=float)
        if np.any(~np.isfinite(wgt_vec)) or np.any(wgt_vec < 0) or np.std(wgt_vec) == 0:
            raise ValueError(f"Trait '{name}' has invalid Weighted phenotype values.")

    trait_dir = outdir / name
    trait_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.DataFrame({
        "sample_id": valid_samples,
        "source_value": source.loc[valid_samples].values,
        "regular_state": reg_vec,
    })
    if wgt_vec is not None:
        manifest["weighted_weight"] = wgt_vec
    manifest.to_csv(trait_dir / f"{name}_sample_manifest.csv", index=False)

    print("\n" + "=" * 78)
    print(f"AMS-AREA v2 TRAIT: {name}")
    print("=" * 78)
    print(f"Samples:             {len(valid_samples)}")
    print(f"Regular controls:    {int(np.sum(reg_vec == 0))}")
    print(f"Regular cases:       {int(np.sum(reg_vec == 1))}")
    print(f"Genes:               {expression.shape[1]}")
    print(f"Inference:           full-null Gaussian, two-sided")
    print(f"MSI classification:  {'disabled' if msi_threshold is None else f'|MSI|>{msi_threshold}'}")

    print(f"Building Regular AREA null ({n_permutations} permutations; seed={seed})...")
    reg_null = permute_enrichment_scores(
        reg_vec,
        n_permutations=n_permutations,
        seed=seed,
        xp=np,
        verbose=False,
    )

    wgt_null = None
    if not binary_only:
        print(f"Building Weighted AREA null ({n_permutations} permutations; seed={seed})...")
        wgt_null = permute_weighted_enrichment_scores(
            wgt_vec,
            n_permutations=n_permutations,
            seed=seed,
        )

    expr = expression.loc[valid_samples]
    genes = expr.columns.to_list()
    matrix = expr.to_numpy(dtype=float)

    reg_es = np.empty(len(genes))
    reg_nes = np.empty(len(genes))
    reg_z = np.empty(len(genes))
    reg_p = np.empty(len(genes))

    if not binary_only:
        wgt_es = np.empty(len(genes))
        wgt_nes = np.empty(len(genes))
        wgt_z = np.empty(len(genes))
        wgt_p = np.empty(len(genes))

    update = max(1, len(genes) // 10)

    for i, gene in enumerate(genes):
        ranks = matrix[:, i]
        order = np.argsort(ranks, kind="mergesort")

        reg_sorted = reg_vec[order]
        obs_reg, *_ = compute_enrichment_score(reg_sorted, xp=np, verbose=False)
        p_reg, z_reg = full_null_gaussian_pvalue(float(obs_reg), reg_null)

        reg_es[i] = float(obs_reg)
        reg_nes[i] = descriptive_nes(float(obs_reg), reg_null)
        reg_z[i] = z_reg
        reg_p[i] = p_reg

        if not binary_only:
            wgt_sorted = wgt_vec[order]
            obs_wgt = compute_weighted_enrichment_score(wgt_sorted)
            p_wgt, z_wgt = full_null_gaussian_pvalue(float(obs_wgt), wgt_null)

            wgt_es[i] = obs_wgt
            wgt_nes[i] = descriptive_nes(obs_wgt, wgt_null)
            wgt_z[i] = z_wgt
            wgt_p[i] = p_wgt

        if (i + 1) % update == 0 or (i + 1) == len(genes):
            print(f"  Processed {i+1:,}/{len(genes):,} genes")

    reg_fdr = benjamini_hochberg(reg_p)

    result = pd.DataFrame({
        "gene_id": genes,
        "Regular_ES": reg_es,
        "Regular_NES": reg_nes,
        "Regular_Z_fullnull": reg_z,
        "Regular_P": reg_p,
        "Regular_FDR": reg_fdr,
    })

    if binary_only:
        result["Weighted_ES"] = np.nan
        result["Weighted_NES"] = np.nan
        result["Weighted_Z_fullnull"] = np.nan
        result["Weighted_P"] = np.nan
        result["Weighted_FDR"] = np.nan
        result["MSI"] = np.nan
        result["Preferred_Method"] = "Regular-only"
    else:
        wgt_fdr = benjamini_hochberg(wgt_p)
        msi = calculate_msi(reg_p, wgt_p, p_floor=msi_p_floor)

        result["Weighted_ES"] = wgt_es
        result["Weighted_NES"] = wgt_nes
        result["Weighted_Z_fullnull"] = wgt_z
        result["Weighted_P"] = wgt_p
        result["Weighted_FDR"] = wgt_fdr
        result["MSI"] = msi
        result["Preferred_Method"] = np.where(
            msi < 0, "Weighted",
            np.where(msi > 0, "Regular", "Tie")
        )

    result["Classification"] = [
        classify_gene(
            regular_fdr=float(row.Regular_FDR),
            weighted_fdr=None if binary_only else float(row.Weighted_FDR),
            msi=None if binary_only else float(row.MSI),
            fdr_threshold=fdr_threshold,
            msi_threshold=msi_threshold,
            binary_only=binary_only,
        )
        for row in result.itertuples(index=False)
    ]

    result_path = trait_dir / f"{name}_AMS_AREA_results.csv"
    result.to_csv(result_path, index=False)

    counts = result["Classification"].value_counts().to_dict()

    summary = {
        "trait": name,
        "source_column": source_column,
        "type": trait["type"],
        "samples_analyzed": len(valid_samples),
        "regular_controls": int(np.sum(reg_vec == 0)),
        "regular_cases": int(np.sum(reg_vec == 1)),
        "genes_tested": len(genes),
        "pvalue_method": "full_null_gaussian_two_sided",
        "regular_fdr_significant": int(np.sum(result["Regular_FDR"] < fdr_threshold)),
        "weighted_fdr_significant": (
            None if binary_only else int(np.sum(result["Weighted_FDR"] < fdr_threshold))
        ),
        "msi_threshold": msi_threshold,
        "dosage_accumulator_genes": int(counts.get("Dosage Accumulator (Weighted)", 0)),
        "state_transition_genes": int(counts.get("State-Transition Trigger (Regular)", 0)),
        "co_progressive_genes": int(counts.get("Co-Progressive Driver", 0)),
        "msi_classification_pending": int(
            counts.get("Association - MSI classification pending calibration", 0)
        ),
        "binary_regular_associations": int(
            counts.get("Binary Association (Regular-only)", 0)
        ),
        "non_significant_genes": int(counts.get("Non-Significant Driver", 0)),
        "results_file": str(result_path),
    }

    print("Trait complete.")
    print(f"  Corrected Regular FDR<{fdr_threshold}:  {summary['regular_fdr_significant']:,}")
    if not binary_only:
        print(f"  Corrected Weighted FDR<{fdr_threshold}: {summary['weighted_fdr_significant']:,}")
        print("  MSI archetype labels remain provisional until simulation calibration.")

    return summary


# ==============================================================================
# Self test
# ==============================================================================

def run_self_test(compute_enrichment_score) -> None:
    vectors = [
        np.array([0, 0, 1, 1, 0, 1], dtype=float),
        np.array([1, 0, 0, 1, 1, 0, 0, 1], dtype=float),
    ]
    for v in vectors:
        official, *_ = compute_enrichment_score(v, xp=np, verbose=False)
        weighted = compute_weighted_enrichment_score(v)
        if not np.isclose(float(official), weighted, atol=1e-14, rtol=1e-12):
            raise AssertionError("Weighted ES does not collapse to Regular AREA ES for binary data.")

    bh = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    if not np.allclose(bh, np.array([0.02, 0.04, 0.04, 0.008])):
        raise AssertionError("BH self-test failed.")

    # Symmetric toy full-null check.
    null = np.linspace(-1, 1, 1001)
    p0, z0 = full_null_gaussian_pvalue(0.0, null)
    if not (0.99 <= p0 <= 1.0):
        raise AssertionError("Full-null p-value should be ~1 at the null mean.")

    print("AMS-AREA v2 self-test: PASSED")
    print("  Weighted ES == Regular AREA ES for binary vectors.")
    print("  BH implementation validated.")
    print("  Full-null two-sided p-value sanity check validated.")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="AMS-AREA v2 with calibrated full-null p-values.")
    p.add_argument("-e", "--expression", required=False)
    p.add_argument("-m", "--metadata", required=False)
    p.add_argument("-c", "--config", required=False)
    p.add_argument("-o", "--outdir", default="results/ams_area_corrected")
    p.add_argument("--area-root", default=None)
    p.add_argument("--traits", default=None)
    p.add_argument("--sample-manifest", default=None)
    p.add_argument("-p", "--permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument(
        "--msi-threshold",
        type=float,
        default=None,
        help=(
            "Optional |MSI| threshold for archetype labels. Leave unset until "
            "simulation-based MSI calibration is complete."
        ),
    )
    p.add_argument("--msi-p-floor", type=float, default=1e-300)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.permutations < 100:
        raise ValueError("--permutations should be >=100; 1000 is recommended.")
    if not (0 < args.fdr_threshold < 1):
        raise ValueError("--fdr-threshold must be between 0 and 1.")
    if args.msi_threshold is not None and args.msi_threshold <= 0:
        raise ValueError("--msi-threshold must be >0 when supplied.")

    configured_root = configure_area_import(args.area_root)
    from src.area.enrichment import compute_enrichment_score, permute_enrichment_scores

    if args.self_test:
        run_self_test(compute_enrichment_score)
        return 0

    required = {
        "--expression": args.expression,
        "--metadata": args.metadata,
        "--config": args.config,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(missing))

    expr_path = Path(args.expression).expanduser()
    meta_path = Path(args.metadata).expanduser()
    config_path = Path(args.config).expanduser()
    outdir = Path(args.outdir).expanduser()
    manifest_path = Path(args.sample_manifest).expanduser() if args.sample_manifest else None

    for path in [expr_path, meta_path, config_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    outdir.mkdir(parents=True, exist_ok=True)

    expression = load_expression(expr_path)
    metadata = load_metadata(meta_path)

    common = sorted(set(expression.index).intersection(metadata.index))
    if not common:
        raise ValueError("No overlapping expression/metadata sample IDs.")

    manifest = load_sample_manifest(manifest_path)
    if manifest is not None:
        missing_ids = manifest - set(common)
        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} manifest IDs absent from expression/metadata; "
                f"examples={sorted(missing_ids)[:10]}"
            )
        common = [s for s in common if s in manifest]

    rng = random.Random(args.seed)
    rng.shuffle(common)
    expression = expression.loc[common]
    metadata = metadata.loc[common]

    with open(config_path) as handle:
        config = json.load(handle)

    traits = [t for t in config["traits"] if t.get("enabled", True)]
    if args.traits:
        requested = {x.strip() for x in args.traits.split(",") if x.strip()}
        available = {t["name"] for t in traits}
        missing = requested - available
        if missing:
            raise ValueError(f"Requested trait(s) not found/enabled: {sorted(missing)}")
        traits = [t for t in traits if t["name"] in requested]

    summaries = []
    for trait in traits:
        summaries.append(
            analyze_trait(
                trait=trait,
                expression=expression,
                metadata=metadata,
                sample_order=common,
                outdir=outdir,
                n_permutations=args.permutations,
                seed=args.seed,
                fdr_threshold=args.fdr_threshold,
                msi_threshold=args.msi_threshold,
                msi_p_floor=args.msi_p_floor,
                compute_enrichment_score=compute_enrichment_score,
                permute_enrichment_scores=permute_enrichment_scores,
            )
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(outdir / "AMS_AREA_run_summary.csv", index=False)

    script_path = Path(__file__).resolve()
    try:
        rosmap_repo = script_path.parents[2]
    except IndexError:
        rosmap_repo = None

    provenance = {
        "analysis_version": "AMS-AREA-v2-full-null-gaussian",
        "pvalue_method": "full_null_gaussian_two_sided",
        "nes_method": "same_signed_mean_descriptive_only",
        "expression": str(expr_path.resolve()),
        "metadata": str(meta_path.resolve()),
        "config": str(config_path.resolve()),
        "sample_manifest": str(manifest_path.resolve()) if manifest_path else None,
        "samples_master": len(common),
        "genes": int(expression.shape[1]),
        "permutations": args.permutations,
        "seed": args.seed,
        "fdr_threshold": args.fdr_threshold,
        "msi_threshold": args.msi_threshold,
        "msi_p_floor": args.msi_p_floor,
        "traits": [t["name"] for t in traits],
        "area_root": str(configured_root) if configured_root else "imported_from_environment",
        "area_git_sha": git_sha(configured_root),
        "rosmap_repo_git_sha": git_sha(rosmap_repo),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    with open(outdir / "AMS_AREA_run_metadata.json", "w") as handle:
        json.dump(provenance, handle, indent=2)

    print("\n" + "=" * 78)
    print("AMS-AREA v2 COMPLETE")
    print("=" * 78)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

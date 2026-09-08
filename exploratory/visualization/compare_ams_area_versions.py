#!/usr/bin/env python3
"""
compare_ams_area_versions.py
==============================================================================
Compare legacy vs corrected AMS-AREA result files.
==============================================================================

Reports:
  * old vs corrected FDR-significant counts
  * retained / lost / newly gained significant genes
  * p-value and FDR correlations
  * corrected MSI summary for AMS traits

This is descriptive. It does not decide a final MSI threshold.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy", required=True, help="Legacy *_AMS_AREA_results.csv")
    p.add_argument("--corrected", required=True, help="Corrected *_AMS_AREA_results.csv")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--fdr", type=float, default=0.05)
    return p.parse_args()

def corr_safe(a, b, method="spearman"):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    return float(x[mask].corr(y[mask], method=method))

def main():
    args = parse_args()
    old = pd.read_csv(args.legacy)
    new = pd.read_csv(args.corrected)

    for df, label in [(old, "legacy"), (new, "corrected")]:
        if "gene_id" not in df.columns:
            raise ValueError(f"{label} file lacks gene_id")
        if df["gene_id"].duplicated().any():
            raise ValueError(f"{label} file has duplicate gene_id")

    m = old.merge(new, on="gene_id", suffixes=("_legacy", "_corrected"), how="inner")
    rows = []

    for prefix in ["Regular", "Weighted"]:
        old_fdr = f"{prefix}_FDR_legacy"
        new_fdr = f"{prefix}_FDR_corrected"
        if old_fdr not in m.columns or new_fdr not in m.columns:
            continue
        if m[old_fdr].notna().sum() == 0 or m[new_fdr].notna().sum() == 0:
            continue

        old_sig = m[old_fdr] < args.fdr
        new_sig = m[new_fdr] < args.fdr

        rows.append({
            "method": prefix,
            "genes_compared": len(m),
            "legacy_fdr_sig": int(old_sig.sum()),
            "corrected_fdr_sig": int(new_sig.sum()),
            "retained_sig": int((old_sig & new_sig).sum()),
            "lost_after_correction": int((old_sig & ~new_sig).sum()),
            "new_after_correction": int((~old_sig & new_sig).sum()),
            "legacy_only_fraction_of_legacy_sig": (
                float((old_sig & ~new_sig).sum() / old_sig.sum()) if old_sig.sum() else np.nan
            ),
            "spearman_raw_p": corr_safe(
                m.get(f"{prefix}_P_legacy"), m.get(f"{prefix}_P_corrected")
            ),
            "spearman_fdr": corr_safe(m[old_fdr], m[new_fdr]),
        })

    summary = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    # Gene-level transition table.
    transition = m[["gene_id"]].copy()
    for prefix in ["Regular", "Weighted"]:
        old_fdr = f"{prefix}_FDR_legacy"
        new_fdr = f"{prefix}_FDR_corrected"
        if old_fdr in m and new_fdr in m:
            transition[f"{prefix}_legacy_sig"] = m[old_fdr] < args.fdr
            transition[f"{prefix}_corrected_sig"] = m[new_fdr] < args.fdr
    trans_path = out.with_name(out.stem + "_gene_transitions.csv")
    transition.to_csv(trans_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSummary: {out}")
    print(f"Transitions: {trans_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

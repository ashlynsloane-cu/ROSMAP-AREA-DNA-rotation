#!/usr/bin/env python3
"""
plot_msi_diagnostics.py
==============================================================================
Diagnostic plots for one AMS-AREA result file.
==============================================================================

Produces:
  * MSI histogram
  * Regular vs Weighted -log10(p) scatter
  * MSI vs overall association strength
  * Regular vs Weighted NES scatter
  * threshold sensitivity table for counts at several |MSI| cutoffs

These are descriptive diagnostics. They do not define the final MSI cutoff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot AMS-AREA MSI diagnostics.")
    p.add_argument("-i", "--input", required=True, help="<trait>_AMS_AREA_results.csv")
    p.add_argument("-o", "--outdir", required=True)
    p.add_argument("--thresholds", default="0.5,1,1.5,2,2.5,3")
    return p.parse_args()


def safe_neglog10(x):
    x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return -np.log10(np.clip(x, 1e-300, 1.0))


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    required = {
        "MSI", "Regular_P", "Weighted_P", "Regular_FDR", "Weighted_FDR",
        "Regular_NES", "Weighted_NES"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    sig_any = (df["Regular_FDR"] < 0.05) | (df["Weighted_FDR"] < 0.05)

    rows = []
    for t in thresholds:
        rows.append({
            "abs_MSI_threshold": t,
            "all_genes_weighted_favored": int(np.sum(df["MSI"] < -t)),
            "all_genes_regular_favored": int(np.sum(df["MSI"] > t)),
            "significant_genes_weighted_favored": int(np.sum(sig_any & (df["MSI"] < -t))),
            "significant_genes_regular_favored": int(np.sum(sig_any & (df["MSI"] > t))),
            "significant_genes_no_strong_preference": int(np.sum(sig_any & (df["MSI"].abs() <= t))),
        })
    pd.DataFrame(rows).to_csv(outdir / "MSI_threshold_sensitivity_counts.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite_msi = df["MSI"].replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(finite_msi, bins=80)
    ax.axvline(0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("MSI = log10(p_weighted / p_regular)")
    ax.set_ylabel("Genes")
    ax.set_title("MSI distribution")
    fig.tight_layout()
    fig.savefig(outdir / "MSI_histogram.png", dpi=300)
    plt.close(fig)

    reg_nlp = safe_neglog10(df["Regular_P"])
    wgt_nlp = safe_neglog10(df["Weighted_P"])

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(reg_nlp, wgt_nlp, s=5, alpha=0.3)
    lim = max(np.nanpercentile(reg_nlp, 99.5), np.nanpercentile(wgt_nlp, 99.5), 1)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Regular AREA -log10(p)")
    ax.set_ylabel("Weighted AREA -log10(p)")
    ax.set_title("Regular vs Weighted evidence")
    fig.tight_layout()
    fig.savefig(outdir / "Regular_vs_Weighted_neglog10P.png", dpi=300)
    plt.close(fig)

    overall_strength = np.maximum(reg_nlp, wgt_nlp)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(overall_strength, df["MSI"], s=5, alpha=0.3)
    ax.axhline(0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("max[-log10(Regular p), -log10(Weighted p)]")
    ax.set_ylabel("MSI")
    ax.set_title("MSI versus overall association strength")
    fig.tight_layout()
    fig.savefig(outdir / "MSI_vs_association_strength.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(df["Regular_NES"], df["Weighted_NES"], s=5, alpha=0.3)
    ax.set_xlabel("Regular AREA NES")
    ax.set_ylabel("Weighted AREA NES")
    ax.set_title("Regular vs Weighted NES")
    fig.tight_layout()
    fig.savefig(outdir / "Regular_vs_Weighted_NES.png", dpi=300)
    plt.close(fig)

    print(f"Diagnostics written to: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""
plot_area_pvalue_method_comparison_v2.py

PI-facing comparison of:
  1) Original AREA p-values: signed-half Gaussian + one-sided tail
  2) Calibrated p-values: full permutation-null Gaussian + two-sided tail

Input: two diagnostic directories created by diagnose_area_pvalue_calibration.py.
Each directory must contain:
  P_VALUE_CALIBRATION_COMPARISON.csv
  real_data_pvalue_method_summary.csv

The figure uses real global-null calibration results and real ROSMAP discovery
counts. Panel A is a conceptual schematic; panels B-D are quantitative.

Outputs PNG + PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ORIGINAL = "official_signed_gaussian"
CORRECTED = "full_null_gaussian_2s"

METHOD_LABELS = {
    ORIGINAL: "Original AREA",
    CORRECTED: "Full-null Gaussian",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ad-dir", required=True,
                   help="Diagnostic directory for AD4_vs_NCI1 / regular.")
    p.add_argument("--braak-dir", required=True,
                   help="Diagnostic directory for Braak_stage / weighted.")
    p.add_argument(
        "-o", "--output",
        default="results/ams_area_validation/figures/AREA_pvalue_method_comparison_v2.png"
    )
    p.add_argument(
        "--title",
        default="Full-null calibration improves AREA p-value calibration"
    )
    return p.parse_args()


def load_diagnostic(directory: Path):
    cal_path = directory / "P_VALUE_CALIBRATION_COMPARISON.csv"
    real_path = directory / "real_data_pvalue_method_summary.csv"

    if not cal_path.exists():
        raise FileNotFoundError(cal_path)
    if not real_path.exists():
        raise FileNotFoundError(real_path)

    cal = pd.read_csv(cal_path)
    real = pd.read_csv(real_path)

    needed_cal = {
        "pvalue_method",
        "null_frac_runs_any_fdr_sig",
        "null_mean_frac_p_lt_0.05",
        "null_mean_frac_p_lt_0.01",
        "null_mean_frac_p_lt_0.001",
    }
    needed_real = {"pvalue_method", "n_fdr_lt_0.05"}

    miss = needed_cal - set(cal.columns)
    if miss:
        raise ValueError(f"{cal_path} missing columns: {sorted(miss)}")
    miss = needed_real - set(real.columns)
    if miss:
        raise ValueError(f"{real_path} missing columns: {sorted(miss)}")

    for method in (ORIGINAL, CORRECTED):
        if method not in set(cal["pvalue_method"]):
            raise ValueError(f"{method} missing from {cal_path}")
        if method not in set(real["pvalue_method"]):
            raise ValueError(f"{method} missing from {real_path}")

    return cal, real


def row(df, method):
    x = df.loc[df["pvalue_method"] == method]
    if len(x) != 1:
        raise ValueError(f"Expected one row for {method}, found {len(x)}")
    return x.iloc[0]


def panel_label(ax, letter):
    ax.text(
        -0.055, 1.045, letter,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold",
        ha="left", va="top",
        clip_on=False,
    )


def box(ax, x, y, text, width=0.35, height=0.105, fontsize=9.3, weight=None):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
        bbox=dict(boxstyle="round,pad=0.45", fill=False, linewidth=1.1),
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )


def plot_schematic(ax):
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("P-value calculation: what changed?", loc="left",
                 fontsize=12.5, fontweight="bold", pad=16)

    box(
        ax, 0.50, 0.885,
        "Same AREA enrichment score + same phenotype permutations",
        fontsize=10.0, weight="bold"
    )

    # Split.
    arrow(ax, 0.50, 0.835, 0.24, 0.775)
    arrow(ax, 0.50, 0.835, 0.76, 0.775)

    ax.text(0.25, 0.73, "Original AREA",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10.6, fontweight="bold")
    ax.text(0.75, 0.73, "AMS-AREA v2",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10.6, fontweight="bold")

    # Original branch.
    box(ax, 0.25, 0.61, "Keep same-sign half\nof the null", fontsize=9.2)
    arrow(ax, 0.25, 0.555, 0.25, 0.49)
    box(ax, 0.25, 0.42, "Fit Gaussian to\nthat truncated half", fontsize=9.2)
    arrow(ax, 0.25, 0.365, 0.25, 0.30)
    box(ax, 0.25, 0.23, "One-sided tail", fontsize=9.2)
    ax.text(
        0.25, 0.08,
        "→ p-values too small\nunder the global null",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=9.2, fontweight="bold"
    )

    # Corrected branch.
    box(ax, 0.75, 0.61, "Keep the entire\npermutation null", fontsize=9.2)
    arrow(ax, 0.75, 0.555, 0.75, 0.49)
    box(ax, 0.75, 0.42, "Fit one Gaussian to\nall null ES values", fontsize=9.2)
    arrow(ax, 0.75, 0.365, 0.75, 0.30)
    box(ax, 0.75, 0.23, "Two-sided tail around\nfull-null mean", fontsize=9.2)
    ax.text(
        0.75, 0.08,
        "→ null p-values track\nexpected frequencies",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=9.2, fontweight="bold"
    )


def calibration_values(cal):
    expected = np.array([0.001, 0.01, 0.05], dtype=float)
    cols = [
        "null_mean_frac_p_lt_0.001",
        "null_mean_frac_p_lt_0.01",
        "null_mean_frac_p_lt_0.05",
    ]
    original = np.array([float(row(cal, ORIGINAL)[c]) for c in cols])
    corrected = np.array([float(row(cal, CORRECTED)[c]) for c in cols])
    return expected, original, corrected


def plot_calibration(ax, cal, title):
    panel_letter = title.split()[0]
    clean_title = title[2:] if len(title) > 2 and title[1] == " " else title
    panel_label(ax, panel_letter)
    ax.set_title(clean_title, loc="left", fontsize=12.5, fontweight="bold", pad=16)

    expected, original, corrected = calibration_values(cal)

    # Perfect calibration.
    ref = np.logspace(-3.2, -0.85, 100)
    ax.plot(ref, ref, linestyle=":", linewidth=1.4, label="Expected")

    ax.plot(
        expected, original,
        marker="o", markersize=6.5, linewidth=1.8,
        label=METHOD_LABELS[ORIGINAL]
    )
    ax.plot(
        expected, corrected,
        marker="s", markersize=6.5, linewidth=1.8, linestyle="--",
        label=METHOD_LABELS[CORRECTED]
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6e-4, 7e-2)
    ax.set_ylim(5e-4, 1.2e-1)
    ax.set_xticks([0.001, 0.01, 0.05])
    ax.set_xticklabels(["0.001", "0.01", "0.05"])
    ax.set_yticks([0.001, 0.01, 0.05, 0.1])
    ax.set_yticklabels(["0.001", "0.01", "0.05", "0.10"])
    ax.set_xlabel("Expected null tail rate")
    ax.set_ylabel("Observed null tail rate")
    ax.grid(True, which="major", linewidth=0.5, alpha=0.25)

    # BH global-null behavior.
    old_any = 100 * float(row(cal, ORIGINAL)["null_frac_runs_any_fdr_sig"])
    new_any = 100 * float(row(cal, CORRECTED)["null_frac_runs_any_fdr_sig"])
    ax.text(
        0.04, 0.96,
        f"Null runs with any FDR hit:  {old_any:.0f}% → {new_any:.0f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.30", fill=False, linewidth=0.8),
    )

    # Label only the most stringent tail; this carries the clearest message.
    for y, label in [
        (original[0], f"{original[0] / expected[0]:.1f}× expected"),
        (corrected[0], f"{corrected[0] / expected[0]:.2f}× expected"),
    ]:
        ax.annotate(
            label,
            xy=(expected[0], y),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=8.5,
        )


def plot_discoveries(ax, ad_real, braak_real):
    panel_label(ax, "D")
    ax.set_title(
        "Real-data discoveries decrease after recalibration",
        loc="left", fontsize=12.5, fontweight="bold", pad=16
    )

    labels = ["AD4 vs NCI1\nRegular AREA", "Braak stage\nWeighted AREA"]
    old = np.array([
        int(row(ad_real, ORIGINAL)["n_fdr_lt_0.05"]),
        int(row(braak_real, ORIGINAL)["n_fdr_lt_0.05"]),
    ])
    new = np.array([
        int(row(ad_real, CORRECTED)["n_fdr_lt_0.05"]),
        int(row(braak_real, CORRECTED)["n_fdr_lt_0.05"]),
    ])

    y = np.arange(len(labels))[::-1]

    # Directional arrows make the reduction visually immediate and avoid
    # redundant grouped bars.
    for yi, old_i, new_i in zip(y, old, new):
        ax.annotate(
            "",
            xy=(new_i, yi), xytext=(old_i, yi),
            arrowprops=dict(arrowstyle="-|>", linewidth=2.0),
        )

        reduction = 100 * (old_i - new_i) / old_i
        midpoint = (old_i + new_i) / 2

        ax.text(
            old_i, yi + 0.11,
            f"Original  {old_i:,}",
            ha="right", va="bottom", fontsize=9.0
        )
        ax.text(
            new_i, yi - 0.11,
            f"Full-null  {new_i:,}",
            ha="left", va="top", fontsize=9.0
        )
        ax.text(
            midpoint, yi + 0.16,
            f"−{reduction:.0f}%",
            ha="center", va="bottom",
            fontsize=10.0, fontweight="bold"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Genes with FDR < 0.05")
    xmin = min(new) - 0.08 * (max(old) - min(new))
    xmax = max(old) + 0.05 * (max(old) - min(new))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.60, 1.60)
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.25)
    ax.text(
        0.03, 0.03,
        "Same observed ES values; only the p-value calculation changes.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5
    )

def main():
    args = parse_args()
    ad_dir = Path(args.ad_dir).expanduser()
    braak_dir = Path(args.braak_dir).expanduser()

    ad_cal, ad_real = load_diagnostic(ad_dir)
    braak_cal, braak_real = load_diagnostic(braak_dir)

    out_png = Path(args.output).expanduser()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")

    fig = plt.figure(figsize=(13.2, 8.35))
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.02, 1.12],
        height_ratios=[1.0, 1.0],
        hspace=0.47, wspace=0.31
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_schematic(ax_a)
    plot_calibration(ax_b, ad_cal, "B  AD4 vs NCI1 · Regular AREA")
    plot_calibration(ax_c, braak_cal, "C  Braak stage · Weighted AREA")
    plot_discoveries(ax_d, ad_real, braak_real)

    # One small legend for calibration panels.
    handles, labels = ax_b.get_legend_handles_labels()
    # Remove "Expected" from method legend and explain diagonal directly.
    if labels and labels[0] == "Expected":
        handles = handles[1:]
        labels = labels[1:]
    ax_b.legend(
        handles, labels,
        frameon=False, loc="lower right",
        fontsize=8.7, borderaxespad=0.8
    )
    ax_b.text(
        0.58, 0.69, "ideal calibration",
        transform=ax_b.transAxes, fontsize=8.2, rotation=34
    )

    fig.suptitle(args.title, fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.020,
        "ROSMAP · 33,006 genes · calibration from repeated global-null phenotype shuffles",
        ha="center", va="bottom", fontsize=9.3
    )

    fig.subplots_adjust(top=0.875, bottom=0.125, left=0.07, right=0.985)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

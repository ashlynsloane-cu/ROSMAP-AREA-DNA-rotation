#!/usr/bin/env python3
"""Generate publication-ready ROSMAP AREA Figure 2."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from figure_style import AREA_COLOR, CONV_COLOR, INK, MID, configure_matplotlib, style_axes

CONTRASTS = [
    ("cogdx_NCI_vs_AD", "NCI → AD"),
    ("cogdx_MCI_vs_AD", "MCI → AD"),
    ("Braak_3-4_vs_5-6", "Late Braak"),
    ("CERAD_3-4_vs_1-2", "CERAD"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--output-dir", "--outdir", type=Path, default=None)
    return p.parse_args()


def locate(repo, src):
    if src:
        return src.expanduser().resolve()
    candidates = [
        repo / "results/stage12_brain_convergence/stage12K_convergence_architecture/CONVERGENCE_ARCHITECTURE_SUMMARY.tsv",
        repo / "data/frozen_inputs/results/stage12_brain_convergence/stage12K_convergence_architecture/CONVERGENCE_ARCHITECTURE_SUMMARY.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate CONVERGENCE_ARCHITECTURE_SUMMARY.tsv; use --source.")


def load(repo, src):
    path = locate(repo, src)
    data = pd.read_csv(path, sep="\t")
    required = {"contrast_id", "method", "formal_conjunction_fdr05_n", "gsea_concordant_both_fdr05_n"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Stage 12K source missing columns: {sorted(missing)}")
    data = data[
        data["contrast_id"].isin([x[0] for x in CONTRASTS])
        & data["method"].isin(["AREA", "CONVENTIONAL"])
    ].copy()
    if len(data) != 8:
        raise ValueError(f"Expected 8 Stage 12K rows; found {len(data)}")
    return data, path


def bars(ax, data, column, xlabel, title, xmax):
    y = [3, 2, 1, 0]
    h = 0.30
    for yi, (contrast_id, _) in zip(y, CONTRASTS):
        area = float(data[(data.contrast_id == contrast_id) & (data.method == "AREA")][column].iloc[0])
        conventional = float(data[(data.contrast_id == contrast_id) & (data.method == "CONVENTIONAL")][column].iloc[0])
        ax.barh(yi - h / 2, area, h, color=AREA_COLOR)
        ax.barh(yi + h / 2, conventional, h, color=CONV_COLOR,
                edgecolor=INK, linewidth=0.45)
        offset = xmax * 0.012
        ax.text(area + offset, yi - h / 2, f"{int(area)}", va="center", fontsize=8)
        ax.text(conventional + offset, yi + h / 2, f"{int(conventional)}", va="center", fontsize=8)

    ax.set_yticks(y, [x[1] for x in CONTRASTS])
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, labelpad=3, fontsize=9)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=7)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.45)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    style_axes(ax, grid_axis="x")


def make(data):
    fig = plt.figure(figsize=(15.5, 7.2))
    grid = fig.add_gridspec(1, 2, wspace=0.28)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)

    bars(ax_a, data, "formal_conjunction_fdr05_n", "Number of convergent genes",
         "A  Same genes: significant in RNA + protein", 60)
    bars(ax_b, data, "gsea_concordant_both_fdr05_n", "Number of convergent pathways",
         "B  Same pathways: enriched in RNA + protein", 90)

    fig.suptitle(
        "Figure 2. Cross-omic convergence at the gene and pathway levels",
        x=0.055, y=0.965, ha="left", fontsize=17, fontweight="bold",
    )
    fig.text(
        0.055, 0.925,
        "Convergence requires concordant direction in both modalities",
        fontsize=10, color=MID, ha="left",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.16)
    return fig


def main():
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    out = (args.output_dir or repo / "results/figures/figure_2_cross_modal").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    data, source = load(repo, args.source)
    fig = make(data)
    fig.savefig(out / "Figure2_cross_modal_convergence_rebuilt.png", dpi=600, facecolor="white")
    fig.savefig(out / "Figure2_cross_modal_convergence_rebuilt.pdf", facecolor="white")
    fig.savefig(out / "Figure2_cross_modal_convergence_rebuilt.svg", facecolor="white")
    plt.close(fig)
    data.to_csv(out / "Figure2_convergence_values_audit.csv", index=False)
    (out / "Figure2_source.txt").write_text(str(source) + "\n")
    print(f"Wrote revised Figure 2 to: {out}")


if __name__ == "__main__":
    main()

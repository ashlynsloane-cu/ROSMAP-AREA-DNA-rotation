#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)
BASENAME = OUTDIR / "weighted_area_encoding_tree_refined_v5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

TEXT = "#171717"
SUBTEXT = "#555555"
EDGE = "#555555"
LINE = "#777777"
DOT = "#7264A8"
ROOT_FILL = "#F1F2F4"
PANEL_FILL = "#F8F5FB"
BOTTOM_FILL = "#F1F2F4"


def add_box(ax, x, y, w, h, fill, lw=0.85, radius=0.012):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fill,
        edgecolor=EDGE,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)


def vline(ax, x, y1, y2, lw=0.85):
    ax.plot([x, x], [y1, y2], color=LINE, lw=lw, solid_capstyle="round", zorder=1)


def hline(ax, x1, x2, y, lw=0.85):
    ax.plot([x1, x2], [y, y], color=LINE, lw=lw, solid_capstyle="round", zorder=1)


def split(ax, parent_x, parent_y, branch_y, child_centers, child_y):
    vline(ax, parent_x, parent_y, branch_y)
    hline(ax, min(child_centers), max(child_centers), branch_y)
    for cx in child_centers:
        vline(ax, cx, branch_y, child_y)


def merge(ax, child_centers, child_y, branch_y, target_x, target_y):
    for cx in child_centers:
        vline(ax, cx, child_y, branch_y)
    hline(ax, min(child_centers), max(child_centers), branch_y)
    vline(ax, target_x, branch_y, target_y)


def draw_scale(ax, x0, x1, y, fractions, labels):
    ax.plot([x0, x1], [y, y], color=LINE, lw=0.9, zorder=3)
    xs = [x0 + (x1 - x0) * f for f in fractions]
    for px, lab in zip(xs, labels):
        ax.scatter([px], [y], s=18, color=DOT, zorder=4)
        ax.text(px, y - 0.0155, lab, ha="center", va="top", fontsize=6.0, color=TEXT, zorder=4)


def encoding_box(ax, box, *, title, subtitle, representation, examples, mode):
    x, y, w, h = box
    add_box(ax, x, y, w, h, PANEL_FILL)
    cx = x + w / 2

    ax.text(cx, y + h * 0.84, title, ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=TEXT, zorder=3)
    ax.text(cx, y + h * 0.735, subtitle, ha="center", va="center",
            fontsize=6.45, fontstyle="italic", color=SUBTEXT, zorder=3)

    ax.text(x + w * 0.08, y + h * 0.60, "Representation", ha="left", va="top",
            fontsize=6.3, fontweight="bold", color=TEXT, zorder=3)
    ax.text(x + w * 0.08, y + h * 0.505, representation, ha="left", va="top",
            fontsize=6.35, linespacing=1.05, color=TEXT, zorder=3)

    ax.text(x + w * 0.08, y + h * 0.345, "Examples", ha="left", va="top",
            fontsize=6.3, fontweight="bold", color=TEXT, zorder=3)
    ax.text(x + w * 0.08, y + h * 0.265, examples, ha="left", va="top",
            fontsize=5.95, linespacing=1.02, color=TEXT, zorder=3)

    sx0 = x + w * 0.22
    sx1 = x + w * 0.78
    sy = y + h * 0.105
    if mode == "equal":
        draw_scale(ax, sx0, sx1, sy, [0, 0.5, 1], ["0", "0.5", "1"])
    elif mode == "calibrated":
        draw_scale(ax, sx0, sx1, sy, [0, 0.2, 1], ["0", "0.2", "1"])
    elif mode == "continuous":
        draw_scale(ax, sx0, sx1, sy, [0, 0.62, 1], ["0", "0.62", "1"])


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.965, "Weighted AREA phenotype encoding",
             ha="center", va="top", fontsize=11.0, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.928,
             "Three phenotype representations feed into the same rank-based test.",
             ha="center", va="top", fontsize=7.25, color=SUBTEXT)

    root = (0.30, 0.775, 0.40, 0.088)
    add_box(ax, *root, ROOT_FILL)
    ax.text(0.5, root[1] + root[3] * 0.63, "Weighted AREA",
            ha="center", va="center", fontsize=9.9, fontweight="bold", color=TEXT, zorder=3)
    ax.text(0.5, root[1] + root[3] * 0.29, "How is phenotype severity represented?",
            ha="center", va="center", fontsize=6.75, fontstyle="italic", color=SUBTEXT, zorder=3)

    panel_y, panel_h, panel_w = 0.385, 0.280, 0.300
    gap = 0.020
    total_w = panel_w * 3 + gap * 2
    left0 = (1 - total_w) / 2
    equal = (left0, panel_y, panel_w, panel_h)
    calibrated = (left0 + panel_w + gap, panel_y, panel_w, panel_h)
    continuous = (left0 + 2 * (panel_w + gap), panel_y, panel_w, panel_h)

    encoding_box(ax, equal,
                 title="Equal",
                 subtitle="ordinal; equal spacing",
                 representation="Ordered categories with\nequal spacing.",
                 examples="Cognitive stage\nCERAD; Braak",
                 mode="equal")
    encoding_box(ax, calibrated,
                 title="Calibrated",
                 subtitle="ordinal; unequal spacing",
                 representation="Ordered categories with\nempirically defined spacing.",
                 examples="Calibrated CERAD\nCalibrated Braak",
                 mode="calibrated")
    encoding_box(ax, continuous,
                 title="Continuous",
                 subtitle="fully quantitative",
                 representation="Measured values used directly;\nno discretization.",
                 examples="Amyloid/tangle burden\nGlobal cognition or MMSE",
                 mode="continuous")

    centers = [b[0] + b[2] / 2 for b in (equal, calibrated, continuous)]
    root_cx = root[0] + root[2] / 2
    split(ax, root_cx, root[1], 0.723, centers, panel_y + panel_h)

    bottom = (0.18, 0.108, 0.64, 0.098)
    bottom_cx = bottom[0] + bottom[2] / 2
    merge(ax, centers, panel_y, 0.292, bottom_cx, bottom[1] + bottom[3])

    add_box(ax, *bottom, BOTTOM_FILL)
    ax.text(bottom_cx, bottom[1] + bottom[3] * 0.66, "Same Weighted AREA statistic",
            ha="center", va="center", fontsize=8.85, fontweight="bold", color=TEXT, zorder=3)
    ax.text(bottom_cx, bottom[1] + bottom[3] * 0.33,
            "expression ranks + phenotype weights  →  association Z  →  $p$-value  →  FDR",
            ha="center", va="center", fontsize=6.5, color=TEXT, zorder=3)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.045)
    for ext, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig(BASENAME.with_suffix(ext), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()

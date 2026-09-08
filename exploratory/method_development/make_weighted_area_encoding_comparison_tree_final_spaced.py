#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)
BASENAME = OUTDIR / "weighted_area_encoding_tree_final_spaced"

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
LINE = "#7A7A7A"
DOT = "#7567AE"
ROOT_FILL = "#F2F3F5"
PANEL_FILL = "#F8F5FB"
BOTTOM_FILL = "#F2F3F5"


def add_box(ax, x, y, w, h, fill, lw=0.9, radius=0.012):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fill,
        edgecolor=EDGE,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)


def vline(ax, x, y1, y2, lw=0.9):
    ax.plot([x, x], [y1, y2], color=LINE, lw=lw, solid_capstyle="round", zorder=1)


def hline(ax, x1, x2, y, lw=0.9):
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
    ax.plot([x0, x1], [y, y], color=LINE, lw=1.0, zorder=3)
    xs = [x0 + (x1 - x0) * f for f in fractions]
    for px, lab in zip(xs, labels):
        ax.scatter([px], [y], s=22, color=DOT, zorder=4)
        ax.text(px, y - 0.012, lab, ha="center", va="top", fontsize=6.2, color=TEXT, zorder=4)


def encoding_box(ax, box, *, title, subtitle, representation, examples, mode):
    x, y, w, h = box
    add_box(ax, x, y, w, h, PANEL_FILL)
    cx = x + w / 2

    # Header block
    ax.text(cx, y + h * 0.845, title,
            ha="center", va="center", fontsize=9.2,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(cx, y + h * 0.730, subtitle,
            ha="center", va="center", fontsize=6.55,
            fontstyle="italic", color=SUBTEXT, zorder=3)

    left = x + w * 0.08

    # Body block with explicit spacing to avoid crowding.
    ax.text(left, y + h * 0.610, "Representation",
            ha="left", va="top", fontsize=6.4,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(left, y + h * 0.515, representation,
            ha="left", va="top", fontsize=6.35,
            linespacing=1.07, color=TEXT, zorder=3)

    ax.text(left, y + h * 0.375, "Examples",
            ha="left", va="top", fontsize=6.4,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(left, y + h * 0.295, examples,
            ha="left", va="top", fontsize=6.15,
            linespacing=1.04, color=TEXT, zorder=3)

    # Encoding schematic lifted off the bottom border.
    sx0 = x + w * 0.24
    sx1 = x + w * 0.76
    sy = y + h * 0.085
    if mode == "equal":
        draw_scale(ax, sx0, sx1, sy, [0, 0.5, 1], ["0", "0.5", "1"])
    elif mode == "calibrated":
        draw_scale(ax, sx0, sx1, sy, [0, 0.2, 1], ["0", "0.2", "1"])
    elif mode == "continuous":
        draw_scale(ax, sx0, sx1, sy, [0, 0.62, 1], ["0", "0.62", "1"])


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5, 0.968, "Weighted AREA phenotype encoding",
        ha="center", va="top", fontsize=11.1,
        fontweight="bold", color=TEXT,
    )
    fig.text(
        0.5, 0.930,
        "Three phenotype representations feed into the same rank-based test.",
        ha="center", va="top", fontsize=7.15, color=SUBTEXT,
    )

    # Root node
    root = (0.305, 0.780, 0.390, 0.085)
    add_box(ax, *root, ROOT_FILL)
    root_cx = root[0] + root[2] / 2
    ax.text(root_cx, root[1] + root[3] * 0.64, "Weighted AREA",
            ha="center", va="center", fontsize=10.0,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(root_cx, root[1] + root[3] * 0.30,
            "How is phenotype severity represented?",
            ha="center", va="center", fontsize=6.75,
            fontstyle="italic", color=SUBTEXT, zorder=3)

    # Middle panels: made wider and slightly taller, with tighter group spacing.
    panel_y = 0.350
    panel_h = 0.335
    panel_w = 0.304
    gap = 0.010
    left0 = (1 - (3 * panel_w + 2 * gap)) / 2
    equal = (left0, panel_y, panel_w, panel_h)
    calibrated = (left0 + panel_w + gap, panel_y, panel_w, panel_h)
    continuous = (left0 + 2 * (panel_w + gap), panel_y, panel_w, panel_h)

    encoding_box(
        ax, equal,
        title="Equal",
        subtitle="ordinal; equal spacing",
        representation="Ordered categories with\nequal spacing.",
        examples="Cognitive stage\nCERAD; Braak",
        mode="equal",
    )
    encoding_box(
        ax, calibrated,
        title="Calibrated",
        subtitle="ordinal; unequal spacing",
        representation="Ordered categories with\nempirically defined spacing.",
        examples="Calibrated CERAD\nCalibrated Braak",
        mode="calibrated",
    )
    encoding_box(
        ax, continuous,
        title="Continuous",
        subtitle="fully quantitative",
        representation="Measured values used directly;\nno discretization.",
        examples="Amyloid/tangle burden\nGlobal cognition or MMSE",
        mode="continuous",
    )

    centers = [b[0] + b[2] / 2 for b in (equal, calibrated, continuous)]
    split(ax, root_cx, root[1], 0.723, centers, panel_y + panel_h)

    # Bottom node
    bottom = (0.185, 0.107, 0.630, 0.094)
    bottom_cx = bottom[0] + bottom[2] / 2
    merge(ax, centers, panel_y, 0.265, bottom_cx, bottom[1] + bottom[3])

    add_box(ax, *bottom, BOTTOM_FILL)
    ax.text(bottom_cx, bottom[1] + bottom[3] * 0.66,
            "Same Weighted AREA statistic",
            ha="center", va="center", fontsize=8.9,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(bottom_cx, bottom[1] + bottom[3] * 0.33,
            "expression ranks + phenotype weights  →  association Z  →  $p$-value  →  FDR",
            ha="center", va="center", fontsize=6.45,
            color=TEXT, zorder=3)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.045)

    for ext, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig(BASENAME.with_suffix(ext), bbox_inches="tight", facecolor="white", **kwargs)

    plt.close(fig)


if __name__ == "__main__":
    main()

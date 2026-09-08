#!/usr/bin/env python3
"""
make_weighted_area_encoding_comparison_tree.py
==============================================

Publication-ready tree diagram for the three phenotype encodings used by
Weighted AREA.

Design goals
------------
- classic hierarchy: Weighted AREA -> three encoding choices -> shared test
- full two-column journal width (~7.2 in), compact height
- readable typography at final size
- vector PDF/SVG plus 600-dpi PNG
- minimal text density and no slide-like whitespace

Outputs
-------
results/method_development/
    weighted_area_encoding_tree_publication.png
    weighted_area_encoding_tree_publication.pdf
    weighted_area_encoding_tree_publication.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)
BASENAME = OUTDIR / "weighted_area_encoding_tree_publication"

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
    return patch


def vline(ax, x, y1, y2, lw=0.85):
    ax.plot([x, x], [y1, y2], color=LINE, lw=lw,
            solid_capstyle="round", zorder=1)


def hline(ax, x1, x2, y, lw=0.85):
    ax.plot([x1, x2], [y, y], color=LINE, lw=lw,
            solid_capstyle="round", zorder=1)


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


def draw_scale(ax, x0, x1, y, fractions, labels, *, annotation=None, value_label=None):
    ax.plot([x0, x1], [y, y], color=LINE, lw=0.9, zorder=3)
    xs = [x0 + (x1 - x0) * f for f in fractions]
    for px, lab in zip(xs, labels):
        ax.scatter([px], [y], s=18, color=DOT, zorder=4)
        ax.text(px, y - 0.020, lab, ha="center", va="top",
                fontsize=6.5, color=TEXT, zorder=4)
    if value_label is not None:
        px = xs[1]
        ax.text(px, y + 0.022, value_label, ha="center", va="bottom",
                fontsize=6.2, fontstyle="italic", color=SUBTEXT, zorder=4)
    if annotation:
        ax.text((x0 + x1) / 2, y - 0.050, annotation,
                ha="center", va="top", fontsize=6.0,
                fontstyle="italic", color=SUBTEXT, zorder=4)


def encoding_box(ax, box, *, title, subtitle, assumption, examples, mode):
    x, y, w, h = box
    add_box(ax, x, y, w, h, PANEL_FILL)
    cx = x + w / 2

    ax.text(cx, y + h * 0.86, title,
            ha="center", va="center", fontsize=9.6,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(cx, y + h * 0.74, subtitle,
            ha="center", va="center", fontsize=6.9,
            fontstyle="italic", color=SUBTEXT, zorder=3)

    ax.text(x + w * 0.08, y + h * 0.60, "Assumption",
            ha="left", va="top", fontsize=6.6,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(x + w * 0.08, y + h * 0.51, assumption,
            ha="left", va="top", fontsize=6.7,
            linespacing=1.13, color=TEXT, zorder=3)

    ax.text(x + w * 0.08, y + h * 0.34, "Examples",
            ha="left", va="top", fontsize=6.6,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(x + w * 0.08, y + h * 0.265, examples,
            ha="left", va="top", fontsize=6.45,
            linespacing=1.10, color=TEXT, zorder=3)

    sx0 = x + w * 0.18
    sx1 = x + w * 0.82
    sy = y + h * 0.080

    if mode == "equal":
        draw_scale(ax, sx0, sx1, sy, [0, 0.5, 1], ["0", "0.5", "1"])
    elif mode == "calibrated":
        draw_scale(
            ax, sx0, sx1, sy,
            [0, 0.2, 1], ["0", "0.2", "1"],
        )
    elif mode == "continuous":
        draw_scale(
            ax, sx0, sx1, sy,
            [0, 0.62, 1], ["0", "0.62", "1"],
        )
    else:
        raise ValueError(mode)


def main():
    # Full-width journal figure, deliberately not slide-shaped.
    fig, ax = plt.subplots(figsize=(7.2, 5.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Figure title is intentionally modest; the manuscript caption carries detail.
    fig.text(
        0.5, 0.967, "Weighted AREA phenotype encoding",
        ha="center", va="top", fontsize=11.8,
        fontweight="bold", color=TEXT,
    )
    fig.text(
        0.5, 0.930,
        "Choose the phenotype representation before applying the same rank-based statistic.",
        ha="center", va="top", fontsize=7.8, color=SUBTEXT,
    )

    # Root node.
    root = (0.31, 0.785, 0.38, 0.095)
    add_box(ax, *root, ROOT_FILL)
    ax.text(0.5, root[1] + root[3] * 0.63, "Weighted AREA",
            ha="center", va="center", fontsize=10.4,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(0.5, root[1] + root[3] * 0.30,
            "How is phenotype severity represented?",
            ha="center", va="center", fontsize=7.1,
            fontstyle="italic", color=SUBTEXT, zorder=3)

    # Encoding choice row.
    panel_y, panel_h, panel_w = 0.385, 0.300, 0.275
    equal = (0.035, panel_y, panel_w, panel_h)
    calibrated = (0.3625, panel_y, panel_w, panel_h)
    continuous = (0.690, panel_y, panel_w, panel_h)

    encoding_box(
        ax, equal,
        title="Equal",
        subtitle="ordinal; equal spacing",
        assumption="Category steps are\nequally spaced.",
        examples="Cognitive_stage\nCERAD_equal; Braak_equal",
        mode="equal",
    )
    encoding_box(
        ax, calibrated,
        title="Calibrated",
        subtitle="ordinal; unequal spacing",
        assumption="Ordering is known; gaps\nneed not be biologically equal.",
        examples="CERAD_amyloid_calibrated\nBraak_tangle_calibrated",
        mode="calibrated",
    )
    encoding_box(
        ax, continuous,
        title="Continuous",
        subtitle="fully quantitative",
        assumption="Use the measured value\ndirectly; no discretization.",
        examples="Amyloid/tangle burden\nCognition; MMSE impairment",
        mode="continuous",
    )

    centers = [b[0] + b[2] / 2 for b in (equal, calibrated, continuous)]
    root_cx = root[0] + root[2] / 2
    split(
        ax,
        parent_x=root_cx,
        parent_y=root[1],
        branch_y=0.735,
        child_centers=centers,
        child_y=panel_y + panel_h,
    )

    # Shared downstream statistic: convergence is explicit instead of implied.
    bottom = (0.18, 0.090, 0.64, 0.115)
    bottom_cx = bottom[0] + bottom[2] / 2
    merge(
        ax,
        child_centers=centers,
        child_y=panel_y,
        branch_y=0.285,
        target_x=bottom_cx,
        target_y=bottom[1] + bottom[3],
    )

    add_box(ax, *bottom, BOTTOM_FILL)
    ax.text(
        bottom_cx, bottom[1] + bottom[3] * 0.68,
        "Same Weighted AREA statistic",
        ha="center", va="center", fontsize=9.4,
        fontweight="bold", color=TEXT, zorder=3,
    )
    ax.text(
        bottom_cx, bottom[1] + bottom[3] * 0.35,
        "expression ranks + phenotype weights  →  adjusted association Z  →  p-value  →  FDR",
        ha="center", va="center", fontsize=6.9,
        color=TEXT, zorder=3,
    )

    fig.text(
        0.5, 0.025,
        "Only phenotype encoding changes; the underlying test is identical across branches.",
        ha="center", va="bottom", fontsize=6.8, color=SUBTEXT,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.045)

    for ext, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(
            BASENAME.with_suffix(ext),
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )

    plt.close(fig)
    for ext in (".png", ".pdf", ".svg"):
        print(f"Wrote: {BASENAME.with_suffix(ext)}")


if __name__ == "__main__":
    main()

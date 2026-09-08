#!/usr/bin/env python3
"""
make_weighted_area_encoding_comparison_v2.py
============================================

Clean publication-style figure comparing the three Weighted AREA
phenotype encoding choices: Equal, Calibrated, and Continuous.

This version is designed to match the cleaned-up visual layout:
- generous spacing
- no text overlap
- three equal-sized comparison panels
- simple encoding schematics inside each panel
- one shared downstream Weighted AREA box

Outputs
-------
results/method_development/
    weighted_area_encoding_choices_v2.png
    weighted_area_encoding_choices_v2.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "weighted_area_encoding_choices_v2.png"
PDF = OUTDIR / "weighted_area_encoding_choices_v2.pdf"


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
EDGE = "#4f4f4f"
LINE = "#666666"
TEXT = "#171717"
SUBTEXT = "#4a4a4a"
FOOTER = "#666666"

TOP_FILL = "#f1f1f1"
PANEL_FILL = "#f7f4fb"
BOTTOM_FILL = "#f2f2f2"
DOT = "#7b6bb2"


def add_box(ax, x, y, w, h, fill, lw=1.25):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fill,
        edgecolor=EDGE,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color=LINE,
        shrinkA=0,
        shrinkB=0,
        zorder=0,
    )
    ax.add_patch(arrow)


def add_top_text(ax, x, y, w, h):
    cx = x + w / 2

    ax.text(
        cx,
        y + h * 0.66,
        "Weighted AREA",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )

    ax.text(
        cx,
        y + h * 0.34,
        "same rank-based framework; different phenotype encodings",
        ha="center",
        va="center",
        fontsize=10,
        fontstyle="italic",
        color=SUBTEXT,
    )


def add_panel_text(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    assumption_lines,
    example_lines,
):
    cx = x + w / 2

    ax.text(
        cx,
        y + h * 0.88,
        title,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
    )

    ax.text(
        cx,
        y + h * 0.80,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.5,
        fontstyle="italic",
        color=SUBTEXT,
    )

    # Assumption / use section
    ax.text(
        x + w * 0.09,
        y + h * 0.68,
        assumption_lines[0],
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT,
    )

    ax.text(
        x + w * 0.53,
        y + h * 0.68,
        assumption_lines[1],
        ha="center",
        va="center",
        fontsize=10.5,
        color=TEXT,
    )

    if len(assumption_lines) > 2:
        ax.text(
            x + w * 0.53,
            y + h * 0.61,
            assumption_lines[2],
            ha="center",
            va="center",
            fontsize=10.5,
            color=TEXT,
        )

    # Examples section
    ax.text(
        x + w * 0.09,
        y + h * 0.49,
        "Examples:",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT,
    )

    ex_y = y + h * 0.49
    for i, line in enumerate(example_lines):
        ax.text(
            x + w * 0.57,
            ex_y - i * h * 0.065,
            line,
            ha="center",
            va="center",
            fontsize=10.2,
            color=TEXT,
        )


def draw_equal(ax, x, y, w, h):
    x0 = x + w * 0.15
    x1 = x + w * 0.85
    yy = y + h * 0.18

    pts = [
        x0,
        (x0 + x1) / 2,
        x1,
    ]

    ax.plot(
        [x0, x1],
        [yy, yy],
        lw=1.8,
        color=LINE,
    )

    labels = ["0", "0.5", "1"]

    for px, lab in zip(pts, labels):
        ax.scatter(
            [px],
            [yy],
            s=55,
            color=DOT,
            zorder=3,
        )

        ax.text(
            px,
            yy - h * 0.075,
            lab,
            ha="center",
            va="center",
            fontsize=10,
        )


def draw_calibrated(ax, x, y, w, h):
    x0 = x + w * 0.15
    x1 = x + w * 0.85
    yy = y + h * 0.18

    pts = [
        x0,
        x0 + (x1 - x0) * 0.20,
        x1,
    ]

    ax.plot(
        [x0, x1],
        [yy, yy],
        lw=1.8,
        color=LINE,
    )

    labels = ["0", "0.2", "1"]

    for px, lab in zip(pts, labels):
        ax.scatter(
            [px],
            [yy],
            s=55,
            color=DOT,
            zorder=3,
        )

        ax.text(
            px,
            yy - h * 0.075,
            lab,
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.text(
        x + w / 2,
        y + h * 0.055,
        "category → quantitative anchor → rescaled score",
        ha="center",
        va="center",
        fontsize=8.7,
        fontstyle="italic",
        color=SUBTEXT,
    )


def draw_continuous(ax, x, y, w, h):
    x0 = x + w * 0.15
    x1 = x + w * 0.85
    yy = y + h * 0.18

    ax.plot(
        [x0, x1],
        [yy, yy],
        lw=1.8,
        color=LINE,
    )

    val = x0 + (x1 - x0) * 0.62

    for px in [x0, val, x1]:
        ax.scatter(
            [px],
            [yy],
            s=55,
            color=DOT,
            zorder=3,
        )

    ax.text(
        x0,
        yy - h * 0.075,
        "0",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.text(
        x1,
        yy - h * 0.075,
        "1",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.text(
        val,
        yy + h * 0.07,
        "0.62",
        ha="center",
        va="center",
        fontsize=10,
    )


def add_bottom_text(ax, x, y, w, h):
    cx = x + w / 2

    ax.text(
        cx,
        y + h * 0.72,
        "All three feed into the same Weighted AREA statistic",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
    )

    ax.text(
        cx,
        y + h * 0.45,
        "gene-expression ranks   ↓   phenotype severity weights",
        ha="center",
        va="center",
        fontsize=12.5,
        color=TEXT,
    )

    ax.text(
        cx,
        y + h * 0.23,
        "adjusted association Z   →   p-value   →   FDR",
        ha="center",
        va="center",
        fontsize=12.5,
        color=TEXT,
    )


def main():
    fig, ax = plt.subplots(
        figsize=(14.5, 10.5),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -----------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------
    fig.suptitle(
        "Weighted AREA encoding choices",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )

    fig.text(
        0.5,
        0.928,
        (
            "How should phenotype severity be encoded before applying "
            "the same Weighted AREA statistic?"
        ),
        ha="center",
        fontsize=12.5,
        color="#333333",
    )

    # -----------------------------------------------------------------
    # Top box
    # -----------------------------------------------------------------
    top = (
        0.30,
        0.79,
        0.40,
        0.09,
    )

    add_box(
        ax,
        *top,
        TOP_FILL,
    )

    add_top_text(
        ax,
        *top,
    )

    top_cx = top[0] + top[2] / 2

    # -----------------------------------------------------------------
    # Middle panels
    # -----------------------------------------------------------------
    panel_y = 0.42
    panel_h = 0.27
    panel_w = 0.27

    equal = (
        0.03,
        panel_y,
        panel_w,
        panel_h,
    )

    calibrated = (
        0.365,
        panel_y,
        panel_w,
        panel_h,
    )

    continuous = (
        0.70,
        panel_y,
        panel_w,
        panel_h,
    )

    for node in [
        equal,
        calibrated,
        continuous,
    ]:
        add_box(
            ax,
            *node,
            PANEL_FILL,
        )

    add_panel_text(
        ax,
        *equal,
        title="Equal",
        subtitle="ordinal; equal spacing",
        assumption_lines=[
            "Assumption:",
            "category steps are",
            "equally spaced",
        ],
        example_lines=[
            "Cognitive_stage,",
            "CERAD_equal, Braak_equal",
        ],
    )

    add_panel_text(
        ax,
        *calibrated,
        title="Calibrated",
        subtitle="ordinal; unequal spacing",
        assumption_lines=[
            "Assumption:",
            "ordering is known, but category gaps",
            "are not biologically equal",
        ],
        example_lines=[
            "CERAD_amyloid_calibrated,",
            "Braak_tangle_calibrated",
        ],
    )

    add_panel_text(
        ax,
        *continuous,
        title="Continuous",
        subtitle="fully quantitative",
        assumption_lines=[
            "",
            "Use the measured value directly;",
            "no discretization",
        ],
        example_lines=[
            "amyloid_continuous, tangle_continuous,",
            "cogn_global_impairment, mmse_impairment",
        ],
    )

    draw_equal(
        ax,
        *equal,
    )

    draw_calibrated(
        ax,
        *calibrated,
    )

    draw_continuous(
        ax,
        *continuous,
    )

    # -----------------------------------------------------------------
    # Top arrows
    # -----------------------------------------------------------------
    for node in [
        equal,
        calibrated,
        continuous,
    ]:
        node_cx = node[0] + node[2] / 2

        add_arrow(
            ax,
            top_cx,
            top[1],
            node_cx,
            node[1] + node[3],
        )

    # -----------------------------------------------------------------
    # Bottom common box
    # -----------------------------------------------------------------
    bottom = (
        0.16,
        0.12,
        0.68,
        0.17,
    )

    add_box(
        ax,
        *bottom,
        BOTTOM_FILL,
    )

    add_bottom_text(
        ax,
        *bottom,
    )

    bottom_cx = bottom[0] + bottom[2] / 2

    for node in [
        equal,
        calibrated,
        continuous,
    ]:
        node_cx = node[0] + node[2] / 2

        add_arrow(
            ax,
            node_cx,
            node[1],
            bottom_cx,
            bottom[1] + bottom[3],
        )

    # -----------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------
    fig.text(
        0.5,
        0.045,
        (
            "Only the phenotype encoding changes. "
            "The underlying Weighted AREA test remains the same."
        ),
        ha="center",
        fontsize=10.2,
        color=FOOTER,
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )

    fig.savefig(
        PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote: {PNG}")
    print(f"Wrote: {PDF}")


if __name__ == "__main__":
    main()

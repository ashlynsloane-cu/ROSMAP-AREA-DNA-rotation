#!/usr/bin/env python3
"""
make_rosmap_analysis_tree_publication.py
=======================================

Publication-sized ROSMAP analysis strategy schematic.

Design targets
--------------
- full two-column journal width (~7.2 in), not slide dimensions
- readable 7.5--12 pt typography at final size
- restrained line weights and color fills
- compact two-column hierarchy instead of a 16:9 flowchart
- vector PDF/SVG plus 600-dpi PNG

Outputs
-------
results/method_development/
    rosmap_analysis_strategy_publication.png
    rosmap_analysis_strategy_publication.pdf
    rosmap_analysis_strategy_publication.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# -----------------------------------------------------------------------------
# Publication defaults
# -----------------------------------------------------------------------------
OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)
BASENAME = OUTDIR / "rosmap_analysis_strategy_publication"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,          # editable TrueType text in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",     # keep text as text
    "axes.linewidth": 0.8,
})

# Neutral, color-blind-safe-ish pastel palette; contrast remains acceptable in gray.
TEXT = "#171717"
SUBTEXT = "#515151"
EDGE = "#555555"
LINE = "#777777"
ROOT_FILL = "#F2F3F5"
BINARY_FILL = "#EEF4FA"
DESEQ_FILL = "#E6F0F8"
REGULAR_FILL = "#EDF5E9"
GRADED_FILL = "#F4EFF8"
WEIGHTED_FILL = "#EEE7F6"
CHILD_FILL = "#F9F6FB"


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


def centered_block(
    ax, x, y, w, h, *, title, subtitle=None, body=None,
    title_size=9.5, subtitle_size=7.4, body_size=7.6,
    title_y=0.72, subtitle_y=0.48, body_y=0.20,
):
    cx = x + w / 2
    ax.text(
        cx, y + h * title_y, title,
        ha="center", va="center", fontsize=title_size,
        fontweight="bold", color=TEXT, zorder=3,
    )
    if subtitle:
        ax.text(
            cx, y + h * subtitle_y, subtitle,
            ha="center", va="center", fontsize=subtitle_size,
            fontstyle="italic", color=SUBTEXT, zorder=3,
        )
    if body:
        ax.text(
            cx, y + h * body_y, "\n".join(body),
            ha="center", va="center", fontsize=body_size,
            linespacing=1.16, color=TEXT, zorder=3,
        )


def row_block(ax, x, y, w, h, *, title, descriptor, examples):
    """Compact child row used for Weighted AREA phenotype encodings."""
    ax.text(
        x + w * 0.055, y + h * 0.64, title,
        ha="left", va="center", fontsize=8.7,
        fontweight="bold", color=TEXT, zorder=3,
    )
    ax.text(
        x + w * 0.055, y + h * 0.32, descriptor,
        ha="left", va="center", fontsize=6.45,
        fontstyle="italic", color=SUBTEXT, zorder=3,
    )
    ax.text(
        x + w * 0.57, y + h * 0.48, examples,
        ha="left", va="center", fontsize=6.55,
        color=TEXT, linespacing=1.06, zorder=3,
    )


def vline(ax, x, y1, y2, lw=0.85):
    ax.plot([x, x], [y1, y2], color=LINE, lw=lw, solid_capstyle="round", zorder=1)


def hline(ax, x1, x2, y, lw=0.85):
    ax.plot([x1, x2], [y, y], color=LINE, lw=lw, solid_capstyle="round", zorder=1)


def split(ax, parent_x, parent_y, branch_y, children_x, child_y):
    vline(ax, parent_x, parent_y, branch_y)
    hline(ax, min(children_x), max(children_x), branch_y)
    for cx in children_x:
        vline(ax, cx, branch_y, child_y)


def main():
    # 7.2 in is a common full-width target for two-column journals.
    fig, ax = plt.subplots(figsize=(7.2, 6.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Heading: deliberately modest; journal figure captions carry detail.
    # ------------------------------------------------------------------
    fig.text(
        0.5, 0.965,
        "ROSMAP analysis strategy by phenotype structure",
        ha="center", va="top", fontsize=12.2, fontweight="bold", color=TEXT,
    )
    fig.text(
        0.5, 0.928,
        "Analysis branch is determined by how the phenotype is represented.",
        ha="center", va="top", fontsize=8.0, color=SUBTEXT,
    )

    # ------------------------------------------------------------------
    # Root + top split
    # ------------------------------------------------------------------
    root = (0.29, 0.805, 0.42, 0.085)
    add_box(ax, *root, ROOT_FILL)
    centered_block(
        ax, *root,
        title="Phenotype of interest",
        subtitle="How is it represented?",
        title_size=10.4, subtitle_size=7.5,
        title_y=0.66, subtitle_y=0.31,
    )

    binary = (0.055, 0.655, 0.405, 0.090)
    graded = (0.540, 0.655, 0.405, 0.090)
    add_box(ax, *binary, BINARY_FILL)
    add_box(ax, *graded, GRADED_FILL)

    centered_block(
        ax, *binary,
        title="Binary endpoint", subtitle="two groups",
        body=["Example: AD vs NCI"],
        title_size=9.8, subtitle_size=7.1, body_size=7.4,
        title_y=0.73, subtitle_y=0.47, body_y=0.18,
    )
    centered_block(
        ax, *graded,
        title="Graded / quantitative phenotype", subtitle="ordered or continuous",
        body=["Examples: cognition, amyloid, tau"],
        title_size=9.4, subtitle_size=7.1, body_size=7.4,
        title_y=0.73, subtitle_y=0.47, body_y=0.18,
    )

    root_cx = root[0] + root[2] / 2
    binary_cx = binary[0] + binary[2] / 2
    graded_cx = graded[0] + graded[2] / 2
    split(ax, root_cx, root[1], 0.775, [binary_cx, graded_cx], binary[1] + binary[3])

    # ------------------------------------------------------------------
    # Binary branch: two complementary analyses, stacked for legibility.
    # ------------------------------------------------------------------
    deseq = (0.075, 0.485, 0.365, 0.105)
    regular = (0.075, 0.335, 0.365, 0.105)
    for node, fill in ((deseq, DESEQ_FILL), (regular, REGULAR_FILL)):
        add_box(ax, *node, fill)

    centered_block(
        ax, *deseq,
        title="DESeq2", subtitle="count-based differential expression",
        body=["Tests mean-expression differences", "Negative-binomial GLM"],
        title_size=9.7, subtitle_size=7.0, body_size=7.4,
        title_y=0.72, subtitle_y=0.48, body_y=0.20,
    )
    centered_block(
        ax, *regular,
        title="Regular AREA", subtitle="rank-based association",
        body=["Tests rank separation between groups", "Permutation-derived null"],
        title_size=9.7, subtitle_size=7.0, body_size=7.4,
        title_y=0.72, subtitle_y=0.48, body_y=0.20,
    )

    # One vertical trunk with short horizontal elbows makes the parallel
    # relationship clear without forcing two narrow side-by-side cards.
    trunk_x = 0.050
    vline(ax, binary_cx, binary[1], 0.620)
    hline(ax, trunk_x, binary_cx, 0.620)
    vline(ax, trunk_x, 0.620, regular[1] + regular[3] / 2)
    for node in (deseq, regular):
        cy = node[1] + node[3] / 2
        hline(ax, trunk_x, node[0], cy)

    # ------------------------------------------------------------------
    # Graded branch: Weighted AREA, then encoding choices as compact rows.
    # ------------------------------------------------------------------
    weighted = (0.575, 0.505, 0.335, 0.110)
    add_box(ax, *weighted, WEIGHTED_FILL)
    centered_block(
        ax, *weighted,
        title="Weighted AREA", subtitle="rank-based + severity weights",
        body=["Tests whether expression rank", "tracks phenotype severity"],
        title_size=9.7, subtitle_size=7.0, body_size=7.4,
        title_y=0.72, subtitle_y=0.48, body_y=0.20,
    )
    weighted_cx = weighted[0] + weighted[2] / 2
    vline(ax, graded_cx, graded[1], weighted[1] + weighted[3])

    equal = (0.555, 0.370, 0.375, 0.075)
    calibrated = (0.555, 0.255, 0.375, 0.075)
    continuous = (0.555, 0.140, 0.375, 0.075)
    for node in (equal, calibrated, continuous):
        add_box(ax, *node, CHILD_FILL, lw=0.75, radius=0.010)

    row_block(
        ax, *equal,
        title="Equal",
        descriptor="ordinal; equal spacing",
        examples="Ordered categories\nCognitive stage\nCERAD, Braak",
    )
    row_block(
        ax, *calibrated,
        title="Calibrated",
        descriptor="ordinal; unequal spacing",
        examples="Calibrated spacing\nCERAD, Braak",
    )
    row_block(
        ax, *continuous,
        title="Continuous",
        descriptor="fully quantitative",
        examples="Measured values\nAmyloid, tangle\nCognition",
    )

    enc_trunk_x = 0.535
    vline(ax, weighted_cx, weighted[1], 0.468)
    hline(ax, enc_trunk_x, weighted_cx, 0.468)
    vline(ax, enc_trunk_x, 0.468, continuous[1] + continuous[3] / 2)
    for node in (equal, calibrated, continuous):
        cy = node[1] + node[3] / 2
        hline(ax, enc_trunk_x, node[0], cy)

    # ------------------------------------------------------------------
    # Bottom method distinction -- small enough to function like a figure note.
    # ------------------------------------------------------------------
    fig.text(
        0.5, 0.035,
        "DESeq2 tests mean shifts; AREA methods test rank-based expression–phenotype association.",
        ha="center", va="bottom", fontsize=7.1, color=SUBTEXT,
    )

    fig.subplots_adjust(left=0.025, right=0.975, top=0.91, bottom=0.07)

    fig.savefig(BASENAME.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(BASENAME.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(BASENAME.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for ext in (".png", ".pdf", ".svg"):
        print(f"Wrote: {BASENAME.with_suffix(ext)}")


if __name__ == "__main__":
    main()

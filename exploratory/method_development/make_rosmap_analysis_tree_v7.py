#!/usr/bin/env python3
"""
make_rosmap_analysis_tree_v7.py
===============================

Final typography/spacing pass for the ROSMAP analysis strategy tree.

Primary fixes from v6
---------------------
- italic descriptor lines are moved farther away from bold headers
- descriptor lines are slightly smaller and lighter
- body text is moved lower so it no longer crowds the italic descriptor
- top title/subtitle spacing is increased
- root, branch, method, and encoding nodes each use tailored vertical spacing
- connectors and overall hierarchy are unchanged

Outputs
-------
results/method_development/rosmap_analysis_tree_v7.png
results/method_development/rosmap_analysis_tree_v7.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "rosmap_analysis_tree_v7.png"
PDF = OUTDIR / "rosmap_analysis_tree_v7.pdf"

# ---------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------
EDGE = "#4a4a4a"
LINE = "#7b7b7b"
TEXT = "#171717"
ITALIC = "#4c4c4c"
FOOTER = "#6a6a6a"

ROOT_FILL = "#f1f2f4"
BINARY_FILL = "#eef4fb"
DESEQ_FILL = "#e5eff9"
REGULAR_FILL = "#eaf4e7"
GRADED_FILL = "#f3eef9"
WEIGHTED_FILL = "#eee5f7"
CHILD_FILL = "#f8f5fb"


def add_box(ax, x, y, w, h, fill, lw=1.2):
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


def add_text_block(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle=None,
    body_lines=None,
    title_size=15,
    subtitle_size=9.3,
    body_size=9.0,
    title_y=0.78,
    subtitle_y=0.52,
    body_y=0.24,
):
    """
    Add centered text using explicit relative vertical positions.

    Keeping title/subtitle/body positions explicit avoids the compressed
    appearance caused by stacking text with fixed cursor offsets.
    """
    cx = x + w / 2

    ax.text(
        cx,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT,
    )

    if subtitle:
        ax.text(
            cx,
            y + h * subtitle_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            fontstyle="italic",
            color=ITALIC,
        )

    if body_lines:
        ax.text(
            cx,
            y + h * body_y,
            "\n".join(body_lines),
            ha="center",
            va="center",
            fontsize=body_size,
            linespacing=1.28,
            color=TEXT,
        )


def connect_split(ax, parent_x, parent_y, branch_y, child_centers, child_y):
    ax.plot(
        [parent_x, parent_x],
        [parent_y, branch_y],
        color=LINE,
        lw=1.35,
        solid_capstyle="round",
        zorder=0,
    )

    ax.plot(
        [min(child_centers), max(child_centers)],
        [branch_y, branch_y],
        color=LINE,
        lw=1.35,
        solid_capstyle="round",
        zorder=0,
    )

    for cx in child_centers:
        ax.plot(
            [cx, cx],
            [branch_y, child_y],
            color=LINE,
            lw=1.35,
            solid_capstyle="round",
            zorder=0,
        )


def connect_vertical(ax, x, y1, y2):
    ax.plot(
        [x, x],
        [y1, y2],
        color=LINE,
        lw=1.35,
        solid_capstyle="round",
        zorder=0,
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -----------------------------------------------------------------
    # Title and subtitle
    # -----------------------------------------------------------------
    fig.suptitle(
        "ROSMAP analysis strategy by phenotype structure",
        fontsize=21,
        fontweight="bold",
        y=0.978,
    )

    fig.text(
        0.5,
        0.918,
        "Choose the analysis branch from the structure of the phenotype",
        ha="center",
        fontsize=11.3,
        color="#333333",
    )

    # -----------------------------------------------------------------
    # Root
    # -----------------------------------------------------------------
    root = (0.37, 0.785, 0.26, 0.095)
    add_box(ax, *root, ROOT_FILL)

    add_text_block(
        ax,
        *root,
        title="Phenotype of interest",
        subtitle="How is it represented?",
        title_size=16,
        subtitle_size=9.5,
        title_y=0.70,
        subtitle_y=0.36,
    )

    # -----------------------------------------------------------------
    # Phenotype structure row
    # -----------------------------------------------------------------
    binary = (0.105, 0.585, 0.285, 0.105)
    graded = (0.61, 0.585, 0.285, 0.105)

    add_box(ax, *binary, BINARY_FILL)
    add_box(ax, *graded, GRADED_FILL)

    add_text_block(
        ax,
        *binary,
        title="Binary endpoint",
        subtitle="two groups",
        body_lines=["Example: AD vs NCI"],
        title_size=15,
        subtitle_size=9.4,
        body_size=9.1,
        title_y=0.73,
        subtitle_y=0.48,
        body_y=0.19,
    )

    add_text_block(
        ax,
        *graded,
        title="Graded / quantitative phenotype",
        subtitle="ordered or continuous",
        body_lines=["Examples: cognition, amyloid, tau"],
        title_size=14.5,
        subtitle_size=9.4,
        body_size=9.1,
        title_y=0.73,
        subtitle_y=0.48,
        body_y=0.19,
    )

    root_cx = root[0] + root[2] / 2
    binary_cx = binary[0] + binary[2] / 2
    graded_cx = graded[0] + graded[2] / 2

    connect_split(
        ax,
        parent_x=root_cx,
        parent_y=root[1],
        branch_y=0.735,
        child_centers=[binary_cx, graded_cx],
        child_y=binary[1] + binary[3],
    )

    # -----------------------------------------------------------------
    # Binary methods
    # -----------------------------------------------------------------
    deseq = (0.045, 0.345, 0.235, 0.155)
    regular = (0.300, 0.345, 0.235, 0.155)

    add_box(ax, *deseq, DESEQ_FILL)
    add_box(ax, *regular, REGULAR_FILL)

    add_text_block(
        ax,
        *deseq,
        title="DESeq2",
        subtitle="count-based differential expression",
        body_lines=[
            "Tests mean-expression differences",
            "Negative-binomial GLM",
            "AD vs NCI",
        ],
        title_size=15,
        subtitle_size=8.9,
        body_size=9.0,
        title_y=0.76,
        subtitle_y=0.55,
        body_y=0.24,
    )

    add_text_block(
        ax,
        *regular,
        title="Regular AREA",
        subtitle="rank-based association",
        body_lines=[
            "Tests rank separation between groups",
            "Permutation-derived null",
            "AD vs NCI",
        ],
        title_size=15,
        subtitle_size=8.9,
        body_size=9.0,
        title_y=0.76,
        subtitle_y=0.55,
        body_y=0.24,
    )

    deseq_cx = deseq[0] + deseq[2] / 2
    regular_cx = regular[0] + regular[2] / 2

    connect_split(
        ax,
        parent_x=binary_cx,
        parent_y=binary[1],
        branch_y=0.545,
        child_centers=[deseq_cx, regular_cx],
        child_y=deseq[1] + deseq[3],
    )

    # -----------------------------------------------------------------
    # Weighted AREA
    # -----------------------------------------------------------------
    weighted = (0.655, 0.385, 0.255, 0.125)
    add_box(ax, *weighted, WEIGHTED_FILL)

    add_text_block(
        ax,
        *weighted,
        title="Weighted AREA",
        subtitle="rank-based + severity weights",
        body_lines=[
            "Tests whether expression rank",
            "tracks phenotype severity",
        ],
        title_size=15,
        subtitle_size=8.9,
        body_size=9.1,
        title_y=0.77,
        subtitle_y=0.54,
        body_y=0.23,
    )

    weighted_cx = weighted[0] + weighted[2] / 2

    connect_vertical(
        ax,
        graded_cx,
        graded[1],
        weighted[1] + weighted[3],
    )

    # -----------------------------------------------------------------
    # Weighted AREA encodings
    # -----------------------------------------------------------------
    child_y = 0.105
    child_h = 0.165

    equal = (0.515, child_y, 0.145, child_h)
    calibrated = (0.685, child_y, 0.145, child_h)
    continuous = (0.855, child_y, 0.125, child_h)

    for node in (equal, calibrated, continuous):
        add_box(ax, *node, CHILD_FILL)

    add_text_block(
        ax,
        *equal,
        title="Equal",
        subtitle="ordinal; equal spacing",
        body_lines=[
            "Ordered categories",
            "Examples: Cognitive stage,",
            "CERAD, Braak",
        ],
        title_size=13,
        subtitle_size=8.4,
        body_size=8.2,
        title_y=0.75,
        subtitle_y=0.53,
        body_y=0.24,
    )

    add_text_block(
        ax,
        *calibrated,
        title="Calibrated",
        subtitle="ordinal; unequal spacing",
        body_lines=[
            "Calibrated category spacing",
            "Examples: calibrated",
            "CERAD, Braak",
        ],
        title_size=13,
        subtitle_size=8.4,
        body_size=8.2,
        title_y=0.75,
        subtitle_y=0.53,
        body_y=0.24,
    )

    add_text_block(
        ax,
        *continuous,
        title="Continuous",
        subtitle="fully quantitative",
        body_lines=[
            "No discretization",
            "Examples: amyloid burden,",
            "tangle burden, cognition",
        ],
        title_size=13,
        subtitle_size=8.3,
        body_size=8.0,
        title_y=0.75,
        subtitle_y=0.53,
        body_y=0.24,
    )

    child_centers = [
        equal[0] + equal[2] / 2,
        calibrated[0] + calibrated[2] / 2,
        continuous[0] + continuous[2] / 2,
    ]

    connect_split(
        ax,
        parent_x=weighted_cx,
        parent_y=weighted[1],
        branch_y=0.315,
        child_centers=child_centers,
        child_y=child_y + child_h,
    )

    # -----------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------
    fig.text(
        0.5,
        0.043,
        (
            "DESeq2 tests mean shifts; AREA methods test "
            "rank-based expression–phenotype association."
        ),
        ha="center",
        fontsize=8.9,
        color=FOOTER,
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.90,
        bottom=0.08,
    )

    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")

    plt.close(fig)

    print(f"Wrote: {PNG}")
    print(f"Wrote: {PDF}")


if __name__ == "__main__":
    main()

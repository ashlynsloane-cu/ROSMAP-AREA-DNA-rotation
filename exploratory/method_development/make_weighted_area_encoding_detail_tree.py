#!/usr/bin/env python3
"""
make_weighted_area_encoding_detail_tree.py

Create a zoomed-in tree diagram focused on the three Weighted AREA
encoding choices: Equal, Calibrated, and Continuous.

Outputs
-------
results/method_development/weighted_area_encoding_detail_tree.png
results/method_development/weighted_area_encoding_detail_tree.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "weighted_area_encoding_detail_tree.png"
PDF = OUTDIR / "weighted_area_encoding_detail_tree.pdf"


EDGE = "#4a4a4a"
LINE = "#7b7b7b"
TEXT = "#171717"
SUBTEXT = "#4c4c4c"
FOOT = "#666666"

ROOT_FILL = "#f1f2f4"
QUESTION_FILL = "#f7f7f7"
WEIGHTED_FILL = "#eee5f7"
CHILD_FILL = "#f8f5fb"
DETAIL_FILL = "#fcfbfe"


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
    subtitle_size=9.5,
    body_size=9.0,
    title_y=0.80,
    subtitle_y=0.58,
    body_y=0.25,
):
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
            color=SUBTEXT,
        )

    if body_lines:
        ax.text(
            cx,
            y + h * body_y,
            "\n".join(body_lines),
            ha="center",
            va="center",
            fontsize=body_size,
            color=TEXT,
            linespacing=1.25,
        )


def connect_vertical(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color=LINE, lw=1.4, solid_capstyle="round", zorder=0)


def connect_split(ax, parent_x, parent_y, branch_y, child_centers, child_top_y):
    ax.plot([parent_x, parent_x], [parent_y, branch_y], color=LINE, lw=1.4, zorder=0)
    ax.plot([min(child_centers), max(child_centers)], [branch_y, branch_y], color=LINE, lw=1.4, zorder=0)
    for cx in child_centers:
        ax.plot([cx, cx], [branch_y, child_top_y], color=LINE, lw=1.4, zorder=0)


def main():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Weighted AREA encoding choices",
        fontsize=22,
        fontweight="bold",
        y=0.975,
    )

    fig.text(
        0.5,
        0.932,
        "A zoomed-in view of how equal, calibrated, and continuous phenotype encodings differ",
        ha="center",
        fontsize=11.5,
        color="#333333",
    )

    # Top node
    root = (0.37, 0.84, 0.26, 0.075)
    add_box(ax, *root, ROOT_FILL)
    add_text_block(
        ax,
        *root,
        title="Weighted AREA",
        subtitle="same rank-based framework; different phenotype encodings",
        title_size=17,
        subtitle_size=9.8,
        title_y=0.69,
        subtitle_y=0.34,
    )
    root_cx = root[0] + root[2] / 2

    # Question node
    q = (0.315, 0.705, 0.37, 0.075)
    add_box(ax, *q, QUESTION_FILL)
    add_text_block(
        ax,
        *q,
        title="How should phenotype severity be represented?",
        subtitle="Choose the encoding that matches what you know about the trait",
        title_size=14.5,
        subtitle_size=9.3,
        title_y=0.67,
        subtitle_y=0.32,
    )
    q_cx = q[0] + q[2] / 2

    connect_vertical(ax, root_cx, root[1], q[1] + q[3])

    # Three main branches
    equal = (0.06, 0.43, 0.25, 0.20)
    calib = (0.375, 0.43, 0.25, 0.20)
    cont = (0.69, 0.43, 0.25, 0.20)

    for box in (equal, calib, cont):
        add_box(ax, *box, CHILD_FILL)

    add_text_block(
        ax,
        *equal,
        title="Equal",
        subtitle="ordinal; equally spaced",
        body_lines=[
            "Use when:",
            "you trust the ordering, but not any",
            "special unequal spacing between levels",
            "",
            "Encoding idea:",
            "adjacent category steps are treated",
            "as equally far apart",
            "",
            "ROSMAP examples:",
            "Cognitive_stage, CERAD_equal, Braak_equal",
        ],
        title_size=15,
        subtitle_size=9.2,
        body_size=8.8,
        title_y=0.86,
        subtitle_y=0.72,
        body_y=0.30,
    )

    add_text_block(
        ax,
        *calib,
        title="Calibrated",
        subtitle="ordinal; unequally spaced",
        body_lines=[
            "Use when:",
            "you trust the ordering and also have",
            "reason to think the category gaps",
            "are not biologically equal",
            "",
            "Encoding idea:",
            "ordered categories are mapped to",
            "unequal numeric weights",
            "",
            "ROSMAP examples:",
            "CERAD_amyloid_calibrated,",
            "Braak_tangle_calibrated",
        ],
        title_size=15,
        subtitle_size=9.2,
        body_size=8.8,
        title_y=0.86,
        subtitle_y=0.72,
        body_y=0.29,
    )

    add_text_block(
        ax,
        *cont,
        title="Continuous",
        subtitle="fully quantitative",
        body_lines=[
            "Use when:",
            "a measured numeric phenotype",
            "is already available directly",
            "",
            "Encoding idea:",
            "use the observed value itself;",
            "no binning or discretization",
            "",
            "ROSMAP examples:",
            "amyloid_continuous,",
            "tangle_continuous,",
            "cogn_global_impairment, mmse_impairment",
        ],
        title_size=15,
        subtitle_size=9.2,
        body_size=8.7,
        title_y=0.86,
        subtitle_y=0.72,
        body_y=0.28,
    )

    child_centers = [
        equal[0] + equal[2] / 2,
        calib[0] + calib[2] / 2,
        cont[0] + cont[2] / 2,
    ]
    connect_split(
        ax,
        parent_x=q_cx,
        parent_y=q[1],
        branch_y=0.665,
        child_centers=child_centers,
        child_top_y=equal[1] + equal[3],
    )

    # Detail box under calibrated
    detail = (0.28, 0.12, 0.44, 0.20)
    add_box(ax, *detail, DETAIL_FILL)

    add_text_block(
        ax,
        *detail,
        title="What does calibrated mean?",
        subtitle="You keep the category order, but estimate unequal distances between categories",
        body_lines=[
            "1. Start with ordered categories (for example, low → medium → high).",
            "2. Use an external quantitative anchor or empirical summary to estimate",
            "   how severe each category really is.",
            "3. Rescale those category summaries to a common numeric range",
            "   while preserving the order.",
            "4. Use those unequal scores as the phenotype weights in Weighted AREA.",
            "",
            "Simple intuition:",
            "Equal assumes 0 → 0.5 → 1.0;",
            "Calibrated might instead look like 0 → 0.2 → 1.0 if the middle state is",
            "biologically much closer to the low state than to the high state.",
        ],
        title_size=14.5,
        subtitle_size=9.0,
        body_size=8.7,
        title_y=0.84,
        subtitle_y=0.70,
        body_y=0.30,
    )

    calib_cx = calib[0] + calib[2] / 2
    connect_vertical(ax, calib_cx, calib[1], detail[1] + detail[3])

    # Small note box on framework invariance
    note = (0.72, 0.14, 0.20, 0.11)
    add_box(ax, *note, WEIGHTED_FILL)
    add_text_block(
        ax,
        *note,
        title="Key point",
        subtitle=None,
        body_lines=[
            "The Weighted AREA statistic stays the same.",
            "What changes is only how the phenotype",
            "values are encoded before ranking/association.",
        ],
        title_size=12.5,
        body_size=8.4,
        title_y=0.77,
        body_y=0.35,
    )

    fig.text(
        0.5,
        0.04,
        "Equal and calibrated are ordinal encodings; continuous uses the measured value directly.",
        ha="center",
        fontsize=9,
        color=FOOT,
    )

    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.07)
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {PNG}")
    print(f"Wrote: {PDF}")


if __name__ == "__main__":
    main()

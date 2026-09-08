#!/usr/bin/env python3
"""
make_simple_analysis_tree_v2.py
===============================

Cleaner publication-style tree diagram for the ROSMAP analysis strategy.

Structure
---------
Start
├── DESeq2
├── Regular AREA
└── Weighted AREA
    ├── Equal
    ├── Calibrated
    └── Continuous

The figure is intentionally simple and low-text. It is meant to explain:
- when each method is used
- the core statistical model
- representative ROSMAP variables

Outputs
-------
rosmap_analysis_tree_v2.png
rosmap_analysis_tree_v2.pdf
"""

from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_PNG = "rosmap_analysis_tree_v2.png"
OUTPUT_PDF = "rosmap_analysis_tree_v2.pdf"


def wrap(text, width):
    return "\n".join(
        textwrap.fill(line, width=width)
        for line in text.split("\n")
    )


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    body,
    facecolor,
    edgecolor="#333333",
    title_size=15,
    subtitle_size=10,
    body_size=9.5,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h - 0.028,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
    )

    ax.text(
        x + w / 2,
        y + h - 0.066,
        subtitle,
        ha="center",
        va="top",
        fontsize=subtitle_size,
        fontstyle="italic",
    )

    ax.text(
        x + 0.016,
        y + h - 0.112,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        linespacing=1.32,
    )


def connect_tree(ax, x_parent, y_parent, child_xs, y_child_top, y_branch):
    ax.plot(
        [x_parent, x_parent],
        [y_parent, y_branch],
        linewidth=1.5,
    )

    ax.plot(
        [min(child_xs), max(child_xs)],
        [y_branch, y_branch],
        linewidth=1.5,
    )

    for x_child in child_xs:
        ax.plot(
            [x_child, x_child],
            [y_branch, y_child_top],
            linewidth=1.5,
        )


def main():
    fig, ax = plt.subplots(
        figsize=(15, 9)
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.axis(
        "off"
    )

    fig.suptitle(
        "ROSMAP analysis strategy by phenotype structure",
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )

    fig.text(
        0.5,
        0.93,
        (
            "Count-based differential expression versus rank-based "
            "phenotype association"
        ),
        ha="center",
        fontsize=11,
    )

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------
    root_x = 0.36
    root_y = 0.82
    root_w = 0.28
    root_h = 0.095

    add_box(
        ax,
        root_x,
        root_y,
        root_w,
        root_h,
        "Phenotype of interest",
        "What structure does the trait have?",
        "",
        facecolor="#f0f3f7",
        title_size=16,
        subtitle_size=10,
        body_size=9,
    )

    # ------------------------------------------------------------------
    # Main branches
    # ------------------------------------------------------------------
    main_y = 0.50
    main_w = 0.25
    main_h = 0.23

    deseq_x = 0.04
    regular_x = 0.375
    weighted_x = 0.71

    deseq_body = wrap(
        "Use when\n"
        "Binary endpoint comparison\n\n"
        "Model\n"
        "Negative-binomial GLM on counts\n\n"
        "ROSMAP example\n"
        "AD vs NCI",
        28,
    )

    regular_body = wrap(
        "Use when\n"
        "Rank-based separation matters\n"
        "without explicit severity spacing\n\n"
        "Model\n"
        "AREA enrichment across gene-ranked samples\n\n"
        "ROSMAP examples\n"
        "AD vs NCI; Cognitive_stage",
        28,
    )

    weighted_body = wrap(
        "Use when\n"
        "Phenotype levels carry severity weights\n\n"
        "Model\n"
        "Weighted rank-based association\n"
        "with adjusted Z and FDR\n\n"
        "Next choice\n"
        "Equal, calibrated, or continuous",
        28,
    )

    add_box(
        ax,
        deseq_x,
        main_y,
        main_w,
        main_h,
        "DESeq2",
        "binary endpoint",
        deseq_body,
        facecolor="#eef4fb",
    )

    add_box(
        ax,
        regular_x,
        main_y,
        main_w,
        main_h,
        "Regular AREA",
        "rank-based",
        regular_body,
        facecolor="#f1f7ef",
    )

    add_box(
        ax,
        weighted_x,
        main_y,
        main_w,
        main_h,
        "Weighted AREA",
        "rank-based + severity weights",
        weighted_body,
        facecolor="#f4effa",
    )

    root_center = (
        root_x
        + root_w / 2
    )

    main_centers = [
        deseq_x
        + main_w / 2,
        regular_x
        + main_w / 2,
        weighted_x
        + main_w / 2,
    ]

    connect_tree(
        ax,
        root_center,
        root_y,
        main_centers,
        main_y
        + main_h,
        0.775,
    )

    # ------------------------------------------------------------------
    # Weighted AREA sub-branches
    # ------------------------------------------------------------------
    sub_y = 0.13
    sub_w = 0.16
    sub_h = 0.22

    equal_x = 0.48
    calibrated_x = 0.665
    continuous_x = 0.85

    equal_body = wrap(
        "Use when\n"
        "Ordered categories with no trusted unequal spacing\n\n"
        "Examples\n"
        "Cognitive_stage\n"
        "CERAD_equal\n"
        "Braak_equal",
        24,
    )

    calibrated_body = wrap(
        "Use when\n"
        "Ordered categories with calibrated spacing\n\n"
        "Examples\n"
        "CERAD_amyloid_calibrated\n"
        "Braak_tangle_calibrated",
        24,
    )

    continuous_body = wrap(
        "Use when\n"
        "A quantitative phenotype is available\n\n"
        "Examples\n"
        "amyloid_continuous\n"
        "tangle_continuous\n"
        "cogn_global_impairment\n"
        "mmse_impairment",
        24,
    )

    add_box(
        ax,
        equal_x,
        sub_y,
        sub_w,
        sub_h,
        "Equal",
        "ordinal, equally spaced",
        equal_body,
        facecolor="#faf8fd",
        title_size=13,
        subtitle_size=9,
        body_size=8.8,
    )

    add_box(
        ax,
        calibrated_x,
        sub_y,
        sub_w,
        sub_h,
        "Calibrated",
        "ordinal, unequal spacing",
        calibrated_body,
        facecolor="#faf8fd",
        title_size=13,
        subtitle_size=9,
        body_size=8.8,
    )

    add_box(
        ax,
        continuous_x,
        sub_y,
        sub_w,
        sub_h,
        "Continuous",
        "fully quantitative",
        continuous_body,
        facecolor="#faf8fd",
        title_size=13,
        subtitle_size=9,
        body_size=8.8,
    )

    weighted_center = (
        weighted_x
        + main_w / 2
    )

    sub_centers = [
        equal_x
        + sub_w / 2,
        calibrated_x
        + sub_w / 2,
        continuous_x
        + sub_w / 2,
    ]

    connect_tree(
        ax,
        weighted_center,
        main_y,
        sub_centers,
        sub_y
        + sub_h,
        0.43,
    )

    # ------------------------------------------------------------------
    # Small interpretation line
    # ------------------------------------------------------------------
    ax.text(
        0.5,
        0.055,
        (
            "DESeq2 tests endpoint mean shifts; AREA methods test "
            "rank-based expression–phenotype association."
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    plt.tight_layout(
        rect=[
            0.02,
            0.06,
            0.98,
            0.92,
        ]
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    print(
        f"Wrote {OUTPUT_PNG}"
    )

    print(
        f"Wrote {OUTPUT_PDF}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_weighted_area_encoding_comparison_publication.py
=====================================================

Publication-sized comparison of Equal, Calibrated, and Continuous phenotype
encodings used with the same Weighted AREA statistic.

Design targets
--------------
- full two-column journal width (~7.2 in), not slide dimensions
- vertical row layout to avoid very wide / cramped cards
- minimum ~7 pt body text at final size
- vector PDF/SVG plus 600-dpi PNG
- no overlapping labels or decorative arrow clutter

Outputs
-------
results/method_development/
    weighted_area_encoding_choices_publication.png
    weighted_area_encoding_choices_publication.pdf
    weighted_area_encoding_choices_publication.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTDIR = Path("results/method_development")
OUTDIR.mkdir(parents=True, exist_ok=True)
BASENAME = OUTDIR / "weighted_area_encoding_choices_publication"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

TEXT = "#171717"
SUBTEXT = "#515151"
EDGE = "#555555"
LINE = "#707070"
DOT = "#7264A8"
PANEL_FILL = "#F8F5FB"
BAND_FILL = "#F2F3F5"
BOTTOM_FILL = "#F2F3F5"


def add_box(ax, x, y, w, h, fill, lw=0.85, radius=0.012):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fill, edgecolor=EDGE, linewidth=lw, zorder=2,
    )
    ax.add_patch(patch)
    return patch


def label_value(ax, x, y, label, value, *, label_w=0.115, value_size=7.15):
    ax.text(
        x, y, label, ha="left", va="top",
        fontsize=7.3, fontweight="bold", color=TEXT, zorder=3,
    )
    ax.text(
        x + label_w, y, value, ha="left", va="top",
        fontsize=value_size, color=TEXT, linespacing=1.16, zorder=3,
    )


def draw_scale(ax, x0, x1, y, values, labels, *, annotation=None):
    ax.plot([x0, x1], [y, y], color=LINE, lw=1.0, zorder=3)
    for px, lab in zip(values, labels):
        ax.scatter([px], [y], s=24, color=DOT, zorder=4)
        ax.text(
            px, y - 0.020, lab, ha="center", va="top",
            fontsize=6.9, color=TEXT, zorder=4,
        )
    if annotation:
        ax.text(
            (x0 + x1) / 2, y - 0.052, annotation,
            ha="center", va="top", fontsize=6.5,
            fontstyle="italic", color=SUBTEXT, zorder=4,
        )


def add_encoding_row(
    ax, x, y, w, h, *, title, subtitle, assumption, examples,
    scale_mode,
):
    add_box(ax, x, y, w, h, PANEL_FILL)

    # Left: encoding class
    ax.text(
        x + 0.030, y + h * 0.68, title,
        ha="left", va="center", fontsize=9.6,
        fontweight="bold", color=TEXT, zorder=3,
    )
    ax.text(
        x + 0.030, y + h * 0.43, subtitle,
        ha="left", va="center", fontsize=7.1,
        fontstyle="italic", color=SUBTEXT, zorder=3,
    )

    # Vertical separator between label column and explanatory text.
    sep_x = x + w * 0.235
    ax.plot(
        [sep_x, sep_x], [y + h * 0.17, y + h * 0.83],
        color="#B8B8B8", lw=0.65, zorder=3,
    )

    # Middle: assumption and examples.
    text_x = x + w * 0.275
    label_value(
        ax, text_x, y + h * 0.78, "Assumption", assumption,
        label_w=w * 0.115, value_size=7.05,
    )
    label_value(
        ax, text_x, y + h * 0.47, "Examples", examples,
        label_w=w * 0.115, value_size=6.85,
    )

    # Right: simple quantitative schematic.
    sx0 = x + w * 0.755
    sx1 = x + w * 0.945
    sy = y + h * 0.50

    if scale_mode == "equal":
        vals = [sx0, (sx0 + sx1) / 2, sx1]
        draw_scale(ax, sx0, sx1, sy, vals, ["0", "0.5", "1"])
    elif scale_mode == "calibrated":
        vals = [sx0, sx0 + (sx1 - sx0) * 0.20, sx1]
        draw_scale(
            ax, sx0, sx1, sy, vals, ["0", "0.2", "1"],
            annotation="category → anchor → rescaled score",
        )
    elif scale_mode == "continuous":
        val = sx0 + (sx1 - sx0) * 0.62
        draw_scale(ax, sx0, sx1, sy, [sx0, val, sx1], ["0", "0.62", "1"])
        ax.text(
            val, sy + 0.026, "measured value",
            ha="center", va="bottom", fontsize=6.3,
            fontstyle="italic", color=SUBTEXT, zorder=4,
        )
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5, 0.965, "Weighted AREA encoding choices",
        ha="center", va="top", fontsize=12.2,
        fontweight="bold", color=TEXT,
    )
    fig.text(
        0.5, 0.928,
        "Phenotype severity can be encoded three ways before applying the same rank-based statistic.",
        ha="center", va="top", fontsize=8.0, color=SUBTEXT,
    )

    # Small invariant-method band replaces the oversized top box.
    band = (0.16, 0.820, 0.68, 0.065)
    add_box(ax, *band, BAND_FILL, lw=0.75)
    ax.text(
        0.50, band[1] + band[3] * 0.60,
        "Weighted AREA statistic is unchanged",
        ha="center", va="center", fontsize=8.8,
        fontweight="bold", color=TEXT, zorder=3,
    )
    ax.text(
        0.50, band[1] + band[3] * 0.27,
        "only the phenotype encoding changes",
        ha="center", va="center", fontsize=7.0,
        fontstyle="italic", color=SUBTEXT, zorder=3,
    )

    # Three full-width rows. This is much more robust than three narrow columns
    # after journal down-scaling.
    row_x, row_w, row_h = 0.055, 0.89, 0.165
    y_equal, y_cal, y_cont = 0.610, 0.395, 0.180

    add_encoding_row(
        ax, row_x, y_equal, row_w, row_h,
        title="Equal",
        subtitle="ordinal; equal spacing",
        assumption="Category steps are\nequally spaced.",
        examples="Cognitive_stage,\nCERAD_equal, Braak_equal",
        scale_mode="equal",
    )
    add_encoding_row(
        ax, row_x, y_cal, row_w, row_h,
        title="Calibrated",
        subtitle="ordinal; unequal spacing",
        assumption="Ordering is known; category gaps\nneed not be biologically equal.",
        examples="CERAD_amyloid_calibrated,\nBraak_tangle_calibrated",
        scale_mode="calibrated",
    )
    add_encoding_row(
        ax, row_x, y_cont, row_w, row_h,
        title="Continuous",
        subtitle="fully quantitative",
        assumption="Use the measured value directly;\nno discretization.",
        examples="amyloid_continuous, tangle_continuous,\ncogn_global_impairment,\nmmse_impairment",
        scale_mode="continuous",
    )

    # Shared downstream statistic. Kept compact and aligned with the rows.
    bottom = (0.12, 0.050, 0.76, 0.075)
    add_box(ax, *bottom, BOTTOM_FILL, lw=0.80)
    ax.text(
        0.50, bottom[1] + bottom[3] * 0.63,
        "All three encodings feed into the same Weighted AREA test",
        ha="center", va="center", fontsize=8.7,
        fontweight="bold", color=TEXT, zorder=3,
    )
    ax.text(
        0.50, bottom[1] + bottom[3] * 0.28,
        "expression ranks + phenotype weights  →  adjusted association Z  →  p-value  →  FDR",
        ha="center", va="center", fontsize=7.0,
        color=TEXT, zorder=3,
    )

    fig.subplots_adjust(left=0.025, right=0.975, top=0.91, bottom=0.025)

    fig.savefig(BASENAME.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(BASENAME.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(BASENAME.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for ext in (".png", ".pdf", ".svg"):
        print(f"Wrote: {BASENAME.with_suffix(ext)}")


if __name__ == "__main__":
    main()

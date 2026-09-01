#!/usr/bin/env python3
"""
plot_archetype_signature.py
======================================================================
ROSMAP Genome-Wide Attribute Archetype Visualization Pipeline
----------------------------------------------------------------------
Generates a publication-grade, dual-panel, 100% stacked bar chart
displaying the relative transcriptomic architectures of all 13 ROSMAP
attributes side-by-side. Visually isolates continuous gradients from 
strictly binary demographic and comorbidity traits.

This script features dynamic filesystem detection to run seamlessly 
on both virtualized container backends and local MacBook environments.
======================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless rendering safe for local and server environments
import matplotlib.pyplot as plt

def main():
    print("======================================================================")
    print("Generating Publication-Ready Transcriptomic Archetype Chart (v3)")
    print("======================================================================")

    # Set standard styles
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['svg.fonttype'] = 'none'

    # 1. Path Resolution: Detect environments to prevent read-only filesystem errors on local Macs
    save_dir = "."
    if os.path.exists("/workspace/scratch") and os.access("/workspace/scratch", os.W_OK):
        save_dir = "/workspace/scratch"
    elif os.path.exists("results"):
        save_dir = "results"
    elif os.path.exists("exploratory/visualization"):
        save_dir = "exploratory/visualization"
        
    print(f"Output files will be saved directly to: {os.path.abspath(save_dir)}")

    # 2. Canvas Dimensions: Final physical layout (170 mm wide x 160 mm high)
    # 170 mm = 6.69 inches, 160 mm = 6.30 inches
    fig_width_inch = 6.69
    fig_height_inch = 6.30

    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))

    # GridSpec setup: 
    # Row 0: Top panel (Continuous comparisons, 8 attributes, 1.7 height ratio)
    # Row 1: Bottom panel (Binary counts, 5 attributes, 1.0 height ratio)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, height_ratios=[1.7, 1.0], hspace=0.35)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ======================================================================
    # TOP PANEL: Continuous / Ordinal attributes (parallel comparison)
    # ======================================================================
    variables_top = [
        "Body Mass Index (BMI)",
        "Age at Death",
        "Age at AD Onset",
        "Braak Tangles",
        "CERAD Plaques",
        "Consensus Cognition",
        "Global Cognition",
        "MMSE Score"
    ]

    # Raw counts unchanged (reproduced from genome-wide screen)
    counts_weighted = np.array([101,  391, 3721, 3500,  765, 2063, 1930, 3478]) # Dosage-like
    counts_regular  = np.array([1423,  107,    0,   10,   53,   25,  451, 1661]) # Threshold-like
    counts_shared   = np.array([1989, 2485, 1110, 3501, 7772, 6503, 9139, 7562]) # Shared/robust

    totals_top = counts_weighted + counts_regular + counts_shared

    # Compute percentages
    pct_weighted = (counts_weighted / totals_top) * 100
    pct_regular  = (counts_regular / totals_top) * 100
    pct_shared   = (counts_shared / totals_top) * 100

    # Colors matching requested colorblind palette
    color_dosage = "#1f77b4"     # Blue (Dosage-like)
    color_threshold = "#ff7f0e"  # Orange (Threshold-like)
    color_shared = "#2ca02c"     # Green (Shared)

    y_indices = np.arange(len(variables_top))
    bar_height = 0.65

    # Plot stacked bars
    ax_top.barh(y_indices, pct_weighted, color=color_dosage, height=bar_height, edgecolor='white', linewidth=0.3, label="Dosage-like response")
    ax_top.barh(y_indices, pct_regular, left=pct_weighted, color=color_threshold, height=bar_height, edgecolor='white', linewidth=0.3, label="Threshold-like response")
    ax_top.barh(y_indices, pct_shared, left=pct_weighted + pct_regular, color=color_shared, height=bar_height, edgecolor='white', linewidth=0.3, label="Shared response")

    # Add internal percentages (only for segments >= 5%)
    for i in range(len(variables_top)):
        left_accum = 0.0
        
        # 1. Dosage-like segment
        val_w = pct_weighted[i]
        if val_w >= 5.0:
            ax_top.text(left_accum + val_w/2, i, f"{int(round(val_w))}%", va='center', ha='center', color='white', fontsize=7.5, fontweight='bold')
        left_accum += val_w
        
        # 2. Threshold-like segment
        val_r = pct_regular[i]
        if val_r >= 5.0:
            ax_top.text(left_accum + val_r/2, i, f"{int(round(val_r))}%", va='center', ha='center', color='white', fontsize=7.5, fontweight='bold')
        left_accum += val_r
        
        # 3. Shared segment
        val_s = pct_shared[i]
        if val_s >= 5.0:
            ax_top.text(left_accum + val_s/2, i, f"{int(round(val_s))}%", va='center', ha='center', color='white', fontsize=7.5, fontweight='bold')

    # Annotate absolute N values on the right
    for i, (total, name) in enumerate(zip(totals_top, variables_top)):
        ax_top.text(101.5, i, f"N={total:,}", va='center', ha='left', fontsize=8, color="#2c3e50", fontweight='bold')

    # Formatting top panel
    ax_top.set_xlim(0, 100)
    ax_top.set_yticks(y_indices)
    ax_top.set_yticklabels(variables_top, fontsize=8.5, color="#2c3e50")
    ax_top.set_xticklabels([])  # Hide x ticks on top panel
    ax_top.xaxis.grid(True, linestyle="--", alpha=0.3, color="#bdc3c7")
    ax_top.set_axisbelow(True)

    # Subtle horizontal separators to visually group variables biologically
    for split_idx in [0.5, 2.5, 4.5]:
        ax_top.axhline(split_idx, color="#bdc3c7", linestyle="-", alpha=0.3, linewidth=0.7)

    # Remove spines for top panel
    for spine in ["top", "right", "bottom"]:
        ax_top.spines[spine].set_visible(False)
    ax_top.spines["left"].set_color("#7f8c8d")
    ax_top.spines["left"].set_linewidth(0.5)

    # ======================================================================
    # BOTTOM PANEL: Binary attributes analyzed with standard single binary model
    # ======================================================================
    variables_bot = [
        "APOE \u03b54 Carrier",
        "Diabetes History",
        "Hypertension History",
        "Biological Sex",
        "Stroke History"
    ]
    counts_bot = [26, 43, 570, 2294, 2907] # Raw counts exactly unchanged

    y_indices_bot = np.arange(len(variables_bot))
    color_binary = "#7f8c8d"

    # Plot simple bars representing raw counts
    ax_bot.barh(y_indices_bot, counts_bot, color=color_binary, height=bar_height, edgecolor='white', linewidth=0.3)

    # Annotate absolute counts on the right
    for i, count in enumerate(counts_bot):
        ax_bot.text(count + 35, i, f"N={count:,}", va='center', ha='left', fontsize=8, color="#2c3e50", fontweight='bold')

    # Formatting bottom panel
    ax_bot.set_yticks(y_indices_bot)
    ax_bot.set_yticklabels(variables_bot, fontsize=8.5, color="#2c3e50")
    ax_bot.set_xlabel("Composition of significant transcriptomic signature (%)", fontsize=8.5, fontweight='bold', labelpad=4)
    ax_bot.set_xlim(0, 3200)
    ax_bot.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax_bot.set_xticklabels(["0", "500", "1,000", "1,500", "2,000", "2,500", "3,000"], fontsize=8, color="#2c3e50")
    ax_bot.xaxis.grid(True, linestyle="--", alpha=0.3, color="#bdc3c7")
    ax_bot.set_axisbelow(True)

    # Remove spines for bottom panel
    for spine in ["top", "right"]:
        ax_bot.spines[spine].set_visible(False)
    ax_bot.spines["left"].set_color("#7f8c8d")
    ax_bot.spines["left"].set_linewidth(0.5)
    ax_bot.spines["bottom"].set_color("#7f8c8d")
    ax_bot.spines["bottom"].set_linewidth(0.5)

    # ======================================================================
    # Master Titles, Subtitles, and Legends (No Spacing Overlaps!)
    # ======================================================================
    # 1. Title (Wrapped to prevent clipping at the exact 170 mm border)
    fig.text(0.04, 0.95, "Clinical attributes show distinct dosage-like, threshold-like,\nand shared transcriptomic response architectures", 
             fontsize=10.0, fontweight='bold', color="#2c3e50", ha='left', va='top')

    # 2. Take-Home Subtitle (RE-WRAPPED WITH AN EXPLICIT NEWLINE TO PREVENT TRUNCATION)
    fig.text(0.04, 0.89, "AD onset and Braak contain large dosage-like components, BMI is threshold-enriched,\nwhile CERAD and cognition are predominantly shared.",
             fontsize=7.8, fontweight='medium', color="#34495e", ha='left', va='top')

    # 3. Methodological Category Definition
    fig.text(0.04, 0.83, "Among BH-FDR < 0.05 genes, bars show whether associations are detected only when phenotype severity is retained continuously\n(dosage-like), only after discretization (threshold-like), or under both representations (shared response).",
             fontsize=7.0, color="#7f8c8d", style='italic', ha='left', va='top')

    # 4. Binary Section Header
    fig.text(0.04, 0.35, "Binary attributes analyzed with a single binary model",
             fontsize=8.5, fontweight='bold', color="#2c3e50", ha='left', va='center')

    # Horizontal Legend: Sits cleanly between definitions and the topmost horizontal bar
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(reversed(handles), labels[::-1], loc="upper right", bbox_to_anchor=(1.0, 1.30), 
                  frameon=False, ncol=3, fontsize=7.8, handletextpad=0.5, columnspacing=1.0)

    # Spacing adjustment: Left-shift bars to fit long labels, pull top down for titles
    plt.subplots_adjust(left=0.28, right=0.88, top=0.70, bottom=0.10)

    # ======================================================================
    # Export Outputs
    # ======================================================================
    # Save primary high-res raster & vector PDF
    fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2.png"), dpi=300)
    fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2.pdf"))

    # Generate web-optimized versions at exact pixel widths (1200px and 600px)
    fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2_1200.png"), dpi=179.37)
    fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2_600.png"), dpi=89.68)

    plt.close()
    print("Successfully generated all updated, overlap-free figures.")
    print("======================================================================")

if __name__ == "__main__":
    main()

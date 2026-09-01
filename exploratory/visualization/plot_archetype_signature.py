#!/usr/bin/env python3
"""
plot_archetype_signature.py
======================================================================
Generates a publication-grade 100% stacked bar chart showing the 
relative molecular architectures of all 13 ROSMAP clinical attributes.
======================================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe headless execution on local terminals and remote servers
import matplotlib.pyplot as plt
import seaborn as sns

# Set style - Clean, minimalist journal theme
sns.set_theme(style="white", font="Arial")

def main():
    print("======================================================================")
    print("Generating Publication-Ready Transcriptomic Archetype Chart")
    print("======================================================================")

    # 1. Input Raw Significant Counts from the Unbiased Genome-Wide Screen
    data = {
        "Clinical Attribute": [
            "Global Cognition", "MMSE Score", "Consensus Cognition", 
            "CERAD Plaques", "Braak Tangles", "Age at AD Onset", 
            "Stroke History", "Biological Sex", "Body Mass Index", 
            "Age at Death", "Hypertension", "Diabetes", "APOE e4 Carrier"
        ],
        "Weighted (Dosage)": [1930, 3478, 2063,  765, 3500, 3721,    0,    0,  101,  391,    0,    0,    0],
        "Regular (Trigger)": [ 451, 1661,   25,   53,   10,    0,    0,    0, 1423,  107,    0,    0,    0],
        "Co-Progressive":    [9139, 7562, 6503, 7772, 3501, 1110,    0,    0, 1989, 2485,    0,    0,    0],
        "Strictly Binary":   [   0,    0,    0,    0,    0,    0, 2907, 2294,    0,    0,  570,   43,   26]
    }

    df = pd.DataFrame(data)

    # 2. Calculate Total Significant Genes & Percentages for Stacked Bar Visual
    df["Total"] = df.iloc[:, 1:].sum(axis=1)
    df = df.sort_values(by="Total", ascending=True) # Sort so smallest is on bottom, largest is on top

    for col in ["Weighted (Dosage)", "Regular (Trigger)", "Co-Progressive", "Strictly Binary"]:
        df[f"{col}_pct"] = (df[col] / df["Total"]) * 100

    # 3. Plotting Setup - Dual dimensions designed to look stunning in slide decks or paper panels
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Color Palette: Accessible, distinct, and elegant
    colors = ["#2b7bba", "#d95f02", "#2ca02c", "#7570b3"] # Clean Weighted, Regular, Co-Progressive, Strictly Binary colors
    categories = [
        "Weighted (Dosage)_pct", "Regular (Trigger)_pct", 
        "Co-Progressive_pct", "Strictly Binary_pct"
    ]
    labels = [
        "Continuous Dosage (Weighted-Only rescued)", 
        "Discrete Threshold (Regular-Only rescued)", 
        "Co-Progressive Drivers (Captured by both)", 
        "Strictly Binary Baseline State"
    ]

    left = np.zeros(len(df))

    # Stack the bar segments
    for idx, cat in enumerate(categories):
        ax.barh(
            df["Clinical Attribute"], df[cat], left=left, 
            color=colors[idx], edgecolor="white", height=0.65, label=labels[idx]
        )
        left += df[cat].values

    # Annotate with the absolute number of significant genes on the right axis
    for i, (total, name) in enumerate(zip(df["Total"], df["Clinical Attribute"])):
        ax.text(
            101.5, i, f"N = {int(total):,}", 
            va="center", ha="left", fontsize=9.5, fontweight="bold", color="#2c3e50"
        )

    # Formatting Spines, Axes, and Gridlines
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proportion of Significant Transcriptomic Signature (%)", fontsize=11, fontweight="bold", labelpad=12)
    ax.set_ylabel("Aligned Clinical & Pathological Variables", fontsize=11, fontweight="bold", labelpad=12)

    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#bdc3c7")
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True)

    # Add descriptive title and subtitle matching high-level biological takeaways
    plt.suptitle(
        "Continuous Gradients Account for Up to 17% of Pathology & Onset Signatures,\nWhile Comorbidities Behave as Strictly Binary States",
        x=0.08, y=0.97, ha="left", fontsize=13, fontweight="bold", color="#2c3e50"
    )
    plt.title(
        f"Genome-wide screen (21,433 genes) across {len(df)} attributes under Benjamini-Hochberg FDR < 0.05",
        loc="left", fontsize=9.5, pad=20, color="#7f8c8d", style="italic"
    )

    # Legend placement at the bottom
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, 
        frameon=False, fontsize=9.5
    )

    # Output file
    output_filename = "unbiased_transcriptomic_archetypes.png"
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"[Success] Publication-ready figure compiled and saved as: {output_filename}")
    print("======================================================================")

if __name__ == "__main__":
    main()

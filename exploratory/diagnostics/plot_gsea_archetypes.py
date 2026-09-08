#!/usr/bin/env python3
"""
plot_gsea_archetypes.py
======================================================================
Archetype-Specific GSEA Signature Mapping Pipeline
======================================================================
This script parses your consolidated `gsea_enrichment_summary.csv` file, 
disentangles unweighted (Regular) and expression-weighted (Weighted) GSEA runs, 
classifies the MSigDB Hallmark pathways into three distinct Molecular Response 
Archetypes, and renders three clean, publication-grade bubble plots:

1. Shared GSEA Responses (Greens)
2. Threshold-like GSEA Responses (Oranges)
3. Dosage-like GSEA Responses (Blues)

Eradicates the overlapping density issues of the monolithic 650-dot plot!
======================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Standard publication aesthetics
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42

def classify_pathway_archetypes(df, p_cutoff=0.05):
    """
    Groups GSEA records, aligns Regular and Weighted runs, and classifies
    each pathway into Shared, Threshold-like, or Dosage-like responses.
    """
    print("Classifying GSEA pathways into Molecular Archetypes on the fly...")
    
    # 1. Pivot the dataset to align Regular and Weighted runs per Pathway
    # We will compute the minimum P-Value or FDR across corresponding runs
    # In this dataset, the Attributes are: 'Regular', 'Weighted', 'Mci', 'Braak'
    
    # Let's map Attributes into clear Unweighted (Regular) vs. Weighted groups
    def map_method(attr):
        attr_lower = attr.lower()
        if 'weighted' in attr_lower:
            return 'Weighted'
        elif 'regular' in attr_lower:
            return 'Regular'
        # Default fallback based on column assumptions
        if attr == 'Weighted':
            return 'Weighted'
        return 'Regular'
        
    df['Method'] = df['Attribute'].apply(map_method)
    
    # Group by Pathway and Method to get the best enrichment scores
    pivoted = df.pivot_table(
        index='Pathway',
        columns='Method',
        values=['NES', 'P_Value', 'FDR_BH'],
        aggfunc='min' # Get the most significant result across contrasts
    ).fillna(1.0)
    
    # Flatten columns
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    archetypes = []
    for _, row in pivoted.iterrows():
        pathway = row['Pathway']
        p_reg = row['P_Value_Regular']
        p_wgt = row['P_Value_Weighted']
        
        is_reg_sig = p_reg < p_cutoff
        is_wgt_sig = p_wgt < p_cutoff
        
        if is_reg_sig and is_wgt_sig:
            arch = 'Shared Response'
        elif is_reg_sig and not is_wgt_sig:
            arch = 'Threshold-like Response'
        elif is_wgt_sig and not is_reg_sig:
            arch = 'Dosage-like Response'
        else:
            arch = 'Non-Significant'
            
        archetypes.append({
            'Pathway': pathway,
            'Archetype': arch,
            'P_Value_Regular': p_reg,
            'P_Value_Weighted': p_wgt,
            'NES_Regular': row['NES_Regular'],
            'NES_Weighted': row['NES_Weighted']
        })
        
    return pd.DataFrame(archetypes)


def plot_bubble_archetype(arch_name, df, full_gsea_df, cmap, out_path):
    """
    Renders a clean, publication-grade GSEA bubble plot (dot plot) for a single archetype.
    """
    arch_pathways = df[df['Archetype'] == arch_name]['Pathway'].unique()
    
    if len(arch_pathways) == 0:
        print(f" -> Info: No pathways classified under '{arch_name}'. Skipping plot.")
        return
        
    # Filter the master GSEA data to only contain these pathways
    plot_data = full_gsea_df[full_gsea_df['Pathway'].isin(arch_pathways)].copy()
    
    # Keep only the major clinical attributes to avoid clutter
    # We will exclude raw 'Regular'/'Weighted' labels if they are unannotated
    clean_attributes = [a for a in plot_data['Attribute'].unique() if a not in ['Regular', 'Weighted']]
    if clean_attributes:
        plot_data = plot_data[plot_data['Attribute'].isin(clean_attributes)]
    
    if plot_data.empty:
        # Fallback to whatever attributes are available for display
        plot_data = full_gsea_df[full_gsea_df['Pathway'].isin(arch_pathways)].copy()
        
    # Compute dot size from P-value (-log10)
    plot_data['Bubble_Size'] = -np.log10(plot_data['P_Value'].clip(lower=1e-5)) * 25 + 20
    
    # Set plot dimensions dynamically based on pathway count
    fig_height = max(5, len(arch_pathways) * 0.35 + 2)
    fig_width = max(8, plot_data['Attribute'].nunique() * 1.2 + 3)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Scatter plot as bubble plot
    scatter = ax.scatter(
        x=plot_data['Attribute'],
        y=plot_data['Pathway'],
        s=plot_data['Bubble_Size'],
        c=plot_data['NES'],
        cmap=cmap,
        alpha=0.85,
        edgecolors='#444444',
        linewidths=1.0
    )
    
    # Clean grid lines
    ax.grid(True, linestyle='--', alpha=0.5, color='lightgrey', zorder=0)
    ax.set_axisbelow(True)
    
    # Colorbar & Legend
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Normalized Enrichment Score (NES)', fontsize=10, fontweight='bold', labelpad=10)
    
    ax.set_title(f"GSEA Pathway Signature Map: {arch_name}", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Clinical Contrast / Attribute", fontsize=10, fontweight='bold', labelpad=10)
    ax.set_ylabel("MSigDB Hallmark Pathway", fontsize=10, fontweight='bold', labelpad=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {arch_name} bubble plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Archetype-Specific GSEA Dot Plot Pipeline")
    parser.add_argument("-i", "--input", default="/workspace/knowledge/gsea_enrichment_summary.csv",
                        help="Path to gsea_enrichment_summary.csv.")
    parser.add_argument("-o", "--outdir", default="results/gsea_archetypes",
                        help="Output directory.")
    parser.add_argument("-p", "--p-cutoff", type=float, default=0.05,
                        help="P-Value significance threshold (default: 0.05).")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(args.input)
    os.makedirs(args.outdir, exist_ok=True)
    
    # Classify pathways
    classified_df = classify_pathway_archetypes(df, args.p_cutoff)
    classified_df.to_csv(os.path.join(args.outdir, "gsea_pathway_archetype_classifications.csv"), index=False)
    print(f"Saved pathway archetypes mapping to: {os.path.join(args.outdir, 'gsea_pathway_archetype_classifications.csv')}")
    
    # Count distributions
    print("\nPathway Archetype Distribution:")
    print(classified_df['Archetype'].value_counts())
    print()
    
    # Render three clean panels
    # Greens for Shared Response
    plot_bubble_archetype(
        arch_name="Shared Response",
        df=classified_df,
        full_gsea_df=df,
        cmap="Greens",
        out_path=os.path.join(args.outdir, "gsea_bubble_shared_response.png")
    )
    
    # Oranges for Threshold-like Response
    plot_bubble_archetype(
        arch_name="Threshold-like Response",
        df=classified_df,
        full_gsea_df=df,
        cmap="Oranges",
        out_path=os.path.join(args.outdir, "gsea_bubble_threshold_like_response.png")
    )
    
    # Blues for Dosage-like Response
    plot_bubble_archetype(
        arch_name="Dosage-like Response",
        df=classified_df,
        full_gsea_df=df,
        cmap="Blues",
        out_path=os.path.join(args.outdir, "gsea_bubble_dosage_like_response.png")
    )
    
    print("\nGSEA Archetype Signature Mapping Pipeline Completed successfully!\n")


if __name__ == "__main__":
    main()

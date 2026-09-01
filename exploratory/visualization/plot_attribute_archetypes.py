#!/usr/bin/env python3
"""
plot_attribute_archetypes.py
======================================================================
Generates a publication-grade side-by-side visualization comparing 
the absolute volume of significant genes and their molecular archetype 
compositions across all 13 clinical, demographic, and comorbidity traits.
======================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Set the style to whitegrid with DejaVu Sans for clean typography
    sns.set_theme(style='whitegrid', context='talk', font='DejaVu Sans')
    
    # 1. ENCODE Master Attribute Archetype Matrix Data
    # Values extracted neutrally from the 1,000-permutation genome-wide screen
    data = [
        { "Attribute": "MMSE Score", "Weighted": 3478, "Regular": 1661, "Co_Progressive": 7562, "Binary": 0, "Total_Sig": 12701 },
        { "Attribute": "Global Cognition", "Weighted": 1930, "Regular": 451, "Co_Progressive": 9139, "Binary": 0, "Total_Sig": 11520 },
        { "Attribute": "Consensus Cognition", "Weighted": 2063, "Regular": 25, "Co_Progressive": 6503, "Binary": 0, "Total_Sig": 8591 },
        { "Attribute": "CERAD Plaques", "Weighted": 765, "Regular": 53, "Co_Progressive": 7772, "Binary": 0, "Total_Sig": 8590 },
        { "Attribute": "Braak Tangles", "Weighted": 3500, "Regular": 10, "Co_Progressive": 3501, "Binary": 0, "Total_Sig": 7011 },
        { "Attribute": "Age at AD Onset", "Weighted": 3721, "Regular": 0, "Co_Progressive": 1110, "Binary": 0, "Total_Sig": 4831 },
        { "Attribute": "Body Mass Index", "Weighted": 101, "Regular": 1423, "Co_Progressive": 1989, "Binary": 0, "Total_Sig": 3513 },
        { "Attribute": "Age at Death", "Weighted": 391, "Regular": 107, "Co_Progressive": 2485, "Binary": 0, "Total_Sig": 2983 },
        { "Attribute": "Stroke History", "Weighted": 0, "Regular": 0, "Co_Progressive": 0, "Binary": 2907, "Total_Sig": 2907 },
        { "Attribute": "Biological Sex", "Weighted": 0, "Regular": 0, "Co_Progressive": 0, "Binary": 2294, "Total_Sig": 2294 },
        { "Attribute": "Hypertension History", "Weighted": 0, "Regular": 0, "Co_Progressive": 0, "Binary": 570, "Total_Sig": 570 },
        { "Attribute": "Diabetes History", "Weighted": 0, "Regular": 0, "Co_Progressive": 0, "Binary": 43, "Total_Sig": 43 },
        { "Attribute": "APOE ε4 Carrier", "Weighted": 0, "Regular": 0, "Co_Progressive": 0, "Binary": 26, "Total_Sig": 26 }
    ]
    
    # Sort data by Total Significant Genes descending for optimal visualization flow
    df = pd.DataFrame(data).sort_values(by="Total_Sig", ascending=True)
    
    # Calculate composition percentages among significant genes
    df['pct_weighted'] = df['Weighted'] / df['Total_Sig'] * 100
    df['pct_regular'] = df['Regular'] / df['Total_Sig'] * 100
    df['pct_coprog'] = df['Co_Progressive'] / df['Total_Sig'] * 100
    df['pct_binary'] = df['Binary'] / df['Total_Sig'] * 100
    
    # 2. INITIALIZE FIGURES
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    
    # Takeaway Title: Must contain a verb and numbers to tell the insight
    fig.suptitle("Genome-Wide Screening of 13 Attributes Identifies Distinct Molecular Architectures Across 21,433 Genes", 
                 fontsize=18, fontweight='bold', y=0.97)
    
    y_pos = np.arange(len(df))
    
    # --- PANEL A: ABSOLUTE SIGNIFICANT GENE COUNT ---
    # Colors absolute bars using a clean, neutral dark slate to emphasize the volume neutrally
    bars = ax1.barh(y_pos, df['Total_Sig'], color='#4a5568', edgecolor='none', height=0.7)
    
    # Add value labels directly on the bars for exact numbers
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 150, bar.get_y() + bar.get_height()/2.0, f"{int(width):,}", 
                 va='center', ha='left', fontsize=11, color='#2d3748', fontweight='bold')
        
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['Attribute'], fontsize=13, fontweight='bold')
    ax1.set_title("A. Absolute Volume of Significant Genes (BH-FDR < 0.05)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Number of Genome-Wide Significant Genes", fontsize=12, labelpad=10)
    ax1.set_xlim(0, 14500)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    # --- PANEL B: ARCHETYPE COMPOSITION (100% STACKED BAR) ---
    # Stack segment colors: Weighted (Blue), Regular (Orange), Co-Progressive (Green), Binary (Slate Gray)
    c_weighted = '#3182bd' # Deep vibrant blue
    c_regular  = '#e6550d' # Vibrant warm orange
    c_coprog   = '#31a354' # Deep leaf green
    c_binary   = '#737373' # Mid gray
    
    ax2.barh(y_pos, df['pct_weighted'], color=c_weighted, edgecolor='none', height=0.7, 
             label='Dosage Accumulator (Weighted-only rescued)')
    ax2.barh(y_pos, df['pct_regular'], left=df['pct_weighted'], color=c_regular, edgecolor='none', height=0.7, 
             label='State-Transition Trigger (Regular-only)')
    ax2.barh(y_pos, df['pct_coprog'], left=df['pct_weighted'] + df['pct_regular'], color=c_coprog, edgecolor='none', height=0.7, 
             label='Co-Progressive Driver (Both Models)')
    ax2.barh(y_pos, df['pct_binary'], left=df['pct_weighted'] + df['pct_regular'] + df['pct_coprog'], color=c_binary, edgecolor='none', height=0.7, 
             label='Strictly Binary Demographics / History')
    
    # Label stacked segments with percentages where they are meaningful (> 4%)
    for i in range(len(df)):
        row = df.iloc[i]
        offset = 0.0
        
        # Weighted label
        if row['pct_weighted'] > 4:
            ax2.text(offset + row['pct_weighted']/2, i, f"{row['pct_weighted']:.1f}%", 
                     va='center', ha='center', fontsize=9, color='white', fontweight='bold')
        offset += row['pct_weighted']
        
        # Regular label
        if row['pct_regular'] > 4:
            ax2.text(offset + row['pct_regular']/2, i, f"{row['pct_regular']:.1f}%", 
                     va='center', ha='center', fontsize=9, color='white', fontweight='bold')
        offset += row['pct_regular']
        
        # Co-Progressive label
        if row['pct_coprog'] > 4:
            ax2.text(offset + row['pct_coprog']/2, i, f"{row['pct_coprog']:.1f}%", 
                     va='center', ha='center', fontsize=9, color='white', fontweight='bold')
        offset += row['pct_coprog']
        
        # Binary label
        if row['pct_binary'] > 4:
            ax2.text(offset + row['pct_binary']/2, i, f"{row['pct_binary']:.1f}%", 
                     va='center', ha='center', fontsize=9, color='white', fontweight='bold')
            
    ax2.set_title("B. Composition of Significant Transcriptomic Footprint", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Percentage of Significant Genes (%)", fontsize=12, labelpad=10)
    ax2.set_xlim(0, 100)
    ax2.set_xticks(np.arange(0, 101, 20))
    ax2.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    # Format and place legend at bottom
    ax2.legend(loc='lower center', bbox_to_anchor=(-0.1, -0.18), ncol=2, frameon=True, fontsize=12)
    
    # Source note at bottom-left in small gray text
    fig.text(0.02, 0.01, "Source: Unbiased parallel genome-wide comparative screen across 21,433 genes in 3,196 ROSMAP patients.", 
             fontsize=10, color='#718096', style='italic')
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    
    # Save to scratch folder
    output_path = "/workspace/scratch/unbiased_transcriptomic_archetypes.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated clean, publication-grade plot: {output_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compare_degs_area.py
======================================================================
Pristine Standalone 3-Set Venn Diagram (170 mm Width for Publication)
======================================================================
This script loads genomic results from DESeq2, Regular AREA, and Weighted AREA,
calculates their exact overlaps, and generates a standalone 3-set Venn diagram
perfectly formatted for direct publication (170 mm width, clean centering,
tightly spaced titles).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects

# Standard high-contrast publication color palette
PALETTE = {
    'DESeq2': '#1f77b4',        # Muted Blue (Group Mean Shifts)
    'Regular': '#ff7f0e',       # Orange (State-Transition Switches)
    'Weighted': '#2ca02c',      # Green (Continuous Dosage Model)
    'Shared_AREA': '#d62728',   # Red (AREA-Consistent Shared Drivers)
}

SHADOW_EFFECT = [path_effects.withStroke(linewidth=2.5, foreground="white")]

def load_genes_from_col(df, gene_col, score_col, cutoff):
    if score_col not in df.columns:
        return set()
    sub_df = df[[gene_col, score_col]].dropna()
    sub_df[score_col] = pd.to_numeric(sub_df[score_col], errors='coerce')
    sig_df = sub_df[sub_df[score_col] < cutoff]
    genes = sig_df[gene_col].dropna().astype(str).str.upper().str.strip().unique()
    return set(genes)

def detect_column(columns, candidates):
    for candidate in candidates:
        for col in columns:
            col_clean = col.lower().strip().replace('_', '').replace('-', '')
            cand_clean = candidate.lower().strip().replace('_', '').replace('-', '')
            if cand_clean in col_clean or col_clean == cand_clean:
                return col
    return None

def draw_3set_standalone_venn(sets, labels, out_path):
    """
    Draws a standalone 3-set Venn diagram matching exact journal constraints:
    - Width: 170 mm (6.69 in)
    - Height: 150 mm (5.9 in)
    - Clean alignment, centered labels, tight titles (no weird gaps)
    """
    A, B, C = sets
    lbl_A, lbl_B, lbl_C = labels
    
    # Calculate exact non-overlapping intersection counts
    v_100 = len(A - (B | C))        # DESeq2 Only
    v_010 = len(B - (A | C))        # Regular Only
    v_001 = len(C - (A | B))        # Weighted Only
    v_110 = len((A & B) - C)        # DESeq2 & Regular
    v_101 = len((A & C) - B)        # DESeq2 & Weighted
    v_011 = len((B & C) - A)        # Regular & Weighted
    v_111 = len(A & B & C)          # Core Concordant (All 3)
    
    total_unique = len(A | B | C)
    
    # Convert 170 mm to inches (6.69 in) for width, 150 mm (5.9 in) for height
    fig, ax = plt.subplots(figsize=(6.69, 5.9))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.axis('off')

    # Shift entire Venn Diagram downward to create more space below subtitle
    y_shift = -0.45
    
    # Centers and radii for 3 intersecting circles
    c1_center = (-1.1, -0.6 + y_shift)  # DESeq2 (Bottom-Left)
    c2_center = (1.1, -0.6 + y_shift)   # Regular AREA (Bottom-Right)
    c3_center = (0.0, 1.1 + y_shift)    # Weighted AREA (Top-Center)
    r = 2.45
    
    circle_A = Circle(c1_center, r, facecolor=PALETTE['DESeq2'], alpha=0.18, edgecolor=PALETTE['DESeq2'], linewidth=2)
    circle_B = Circle(c2_center, r, facecolor=PALETTE['Regular'], alpha=0.18, edgecolor=PALETTE['Regular'], linewidth=2)
    circle_C = Circle(c3_center, r, facecolor=PALETTE['Weighted'], alpha=0.18, edgecolor=PALETTE['Weighted'], linewidth=2)
    
    ax.add_patch(circle_A)
    ax.add_patch(circle_B)
    ax.add_patch(circle_C)
    
    # Text helper
    def plot_count(x, y, val, color='#111111', size=11):
        ax.text(x, y, f"{val:,}", fontsize=size, fontweight='bold', color=color,
                ha='center', va='center', path_effects=SHADOW_EFFECT)
        
    # Placements inside intersections
    plot_count(-2.2, -1.2 + y_shift, v_100, PALETTE['DESeq2'], size=11.5)   # DESeq2 Only
    plot_count(2.2, -1.2 + y_shift, v_010, PALETTE['Regular'], size=11.5)    # Regular Only
    plot_count(0.0, 2.3 + y_shift, v_001, PALETTE['Weighted'], size=11.5)    # Weighted Only
    plot_count(0.0, -1.6 + y_shift, v_110, '#333333', size=10)               # DESeq2 & Regular Overlap
    plot_count(-1.1, 0.5 + y_shift, v_101, '#333333', size=10)               # DESeq2 & Weighted Overlap
    plot_count(1.1, 0.5 + y_shift, v_011, PALETTE['Shared_AREA'], size=10)   # Regular & Weighted Overlap
    plot_count(0.0, -0.3 + y_shift, v_111, '#111111', size=13)               # Core Concordant
    
    # Outer labels positioned neatly
    ax.text(-3.4, -2.1 + y_shift, f"{lbl_A}\n(Group-Mean Shifts)", fontsize=9.5, fontweight='bold', color=PALETTE['DESeq2'], ha='center')
    ax.text(3.4, -2.1 + y_shift, f"{lbl_B}\n(State-Transition Switches)", fontsize=9.5, fontweight='bold', color=PALETTE['Regular'], ha='center')
    ax.text(0.0, 3.8 + y_shift, f"{lbl_C}\n(Continuous Dosage Model)", fontsize=9.5, fontweight='bold', color=PALETTE['Weighted'], ha='center')
    
    # Tightly integrated titles (no double Figure 1, no giant gaps)
    # Using a single call block to prevent vertical gaps
    ax.text(0.0, 4.7, "Shared and Method-Specific Gene Signatures in Cognitive AD", 
            fontsize=11.5, fontweight='bold', color='#111111', ha='center', va='center')
    ax.text(0.0, 4.3, f"Total Unique Significant Genes: {total_unique:,}  |  Significance Cutoff: FDR < 0.05", 
            fontsize=8.5, color='#555555', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Standalone publication figure successfully saved to: {out_path}")

def run_self_test():
    print("\n[SELF-TEST] Running self-test with representative mock ROSMAP AD markers...")
    deseq2 = {"APP", "MAPT", "TREM2", "APOE", "BIN1", "CLU", "ABCA7", "PICALM", "CR1", "CD33", "MS4A6A", "SOD1", "GFAP", "STAT3", "IL1B", "ACKR4P1", "ACOT4P1", "AACSP1"}
    regular = {"KDM6A", "JPX", "EFR3B", "SFN", "CLXN", "PNOC", "POLE", "APOE", "BIN1", "TREM2", "MAPT", "APP", "GFAP", "STAT3", "ABCB11", "ACAA1", "ACP6", "ACTN1", "ADAM1B", "AAK1", "AASDHPPT", "AATBC"}
    weighted = {"MFGE8", "MLF1", "HMCN2", "DMKN", "WAC", "MSR1", "NUPR1", "AFF4", "APOE", "BIN1", "CLU", "ABCA7", "TREM2", "APP", "AADAT", "AARS2", "ABCA1", "ABCC11", "ABCC6", "AAK1", "AASDHPPT", "AATBC"}
    
    draw_3set_standalone_venn([deseq2, regular, weighted], ['DESeq2', 'Regular AREA', 'Weighted AREA'], "compare_degs_3set_Venn_test.png")

def main():
    parser = argparse.ArgumentParser(description="Genomic DEG/RAE Overlap Standalone Venn Diagram (170 mm Width)")
    parser.add_argument("--input", default=None, help="Path to consolidated master comparison CSV")
    parser.add_argument("--deseq2-file", default=None, help="Path to separate DESeq2 results CSV")
    parser.add_argument("-o", "--output", default="compare_degs_3set_Venn.png", help="Output PNG path")
    parser.add_argument("--cutoff", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--self-test", action="store_true", help="Runs on-the-fly self-test with mock data")
    args = parser.parse_args()
    
    if args.self_test or (not args.input and not args.deseq2_file):
        run_self_test()
        sys.exit(0)
        
    print("Loading data files...")
    df_area = pd.read_csv(args.input)
    df_area.columns = [c.strip() for c in df_area.columns]
    
    df_de = pd.read_csv(args.deseq2_file)
    df_de.columns = [c.strip() for c in df_de.columns]
    
    # 1. Fuzzy map columns
    gene_col_area = None
    for candidate in ['gene', 'gene_symbol', 'symbol', 'feature', 'id']:
        for col in df_area.columns:
            if col.lower() == candidate:
                gene_col_area = col
                break
        if gene_col_area: break
    if not gene_col_area: gene_col_area = df_area.columns[0]
        
    reg_col = detect_column(df_area.columns, ['regularfdr', 'regular_fdr', 'regular_padj', 'regular_p'])
    wgt_col = detect_column(df_area.columns, ['weightedfdr', 'weighted_fdr', 'weighted_padj', 'weighted_p'])
    
    gene_col_de = None
    for candidate in ['gene_symbol', 'gene', 'symbol', 'feature']:
        for col in df_de.columns:
            if col.lower() == candidate:
                gene_col_de = col
                break
        if gene_col_de: break
    if not gene_col_de: gene_col_de = df_de.columns[0]
        
    de_col = detect_column(df_de.columns, ['padj', 'p.adjust', 'adjusted_p_value', 'pvalue'])
    
    # 2. Extract Significant Sets
    set_regular = load_genes_from_col(df_area, gene_col_area, reg_col, args.cutoff)
    set_weighted = load_genes_from_col(df_area, gene_col_area, wgt_col, args.cutoff)
    set_deseq2 = load_genes_from_col(df_de, gene_col_de, de_col, args.cutoff)
    
    print(f" -> Active Mappings:")
    print(f"    * Regular AREA Genes  : {len(set_regular):,}")
    print(f"    * Weighted AREA Genes : {len(set_weighted):,}")
    print(f"    * DESeq2 Genes        : {len(set_deseq2):,}")
    
    draw_3set_standalone_venn([set_deseq2, set_regular, set_weighted], 
                              ['DESeq2', 'Regular AREA', 'Weighted AREA'], 
                              args.output)

if __name__ == "__main__":
    main()

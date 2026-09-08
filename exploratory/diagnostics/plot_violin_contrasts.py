#!/usr/bin/env python3
"""
plot_violin_contrasts.py
======================================================================
Publication-Grade Split Violin Plots for Multi-Method Validation
======================================================================
This script takes continuous expression data and clinical boolean metadata,
isolates specific genes that were selectively captured by different pipeline
methods (e.g. Regular AREA only, Weighted AREA only, or DESeq2 only), and 
plots high-resolution violin plots comparing the cohorts.

Overlays individual patient jitter points and central box plots. Computes
and annotates Wilcoxon rank-sum p-values and fold changes dynamically!

Includes a fully functional mock self-testing mode!
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
from scipy.stats import ranksums

# Standard publication aesthetics
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42

# Scientific palette
COLORS = {
    'Control': '#b0c4de',       # Slate/Grey-Blue
    'Disease': '#ff7f0e'        # Vibrant Orange (matching the paper style)
}

def clean_patient_ids(series):
    """
    Standardizes patient IDs to strings, stripping decimals and 'R' prefixes
    to guarantee flawless matching across sheets.
    """
    return series.astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.upper().str.replace(r'^R', '', regex=True)


def generate_violin(gene, expression_series, labels_series, contrast_label, out_path):
    """
    Plots a single publication-grade violin plot for a gene, overlaying 
    jitter points and a central box plot, with Wilcoxon p-value annotation.
    """
    # Combine into a plotting DataFrame
    df = pd.DataFrame({
        'Expression': expression_series,
        'Group': labels_series.map({0: 'Control', 1: 'Disease'})
    }).dropna()
    
    if df['Group'].nunique() < 2:
        print(f" -> Warning: Cannot plot {gene} as one group is empty or missing.")
        return
        
    fig, ax = plt.subplots(figsize=(6, 7))
    
    # Calculate statistics
    control_vals = df[df['Group'] == 'Control']['Expression'].values
    disease_vals = df[df['Group'] == 'Disease']['Expression'].values
    
    stat, p_val = ranksums(control_vals, disease_vals)
    mean_ctrl = np.mean(control_vals)
    mean_dis = np.mean(disease_vals)
    fold_change = mean_dis / mean_ctrl if mean_ctrl > 0 else np.nan
    
    # Render violin
    sns.violinplot(
        x='Group',
        y='Expression',
        data=df,
        palette=COLORS,
        inner=None,          # Draw custom boxplot later
        alpha=0.6,
        linewidth=1.5,
        ax=ax
    )
    
    # Render box plot inside
    sns.boxplot(
        x='Group',
        y='Expression',
        data=df,
        width=0.15,
        color='white',
        linewidth=2,
        showfliers=False,
        ax=ax
    )
    
    # Overlay individual patient jitter points
    sns.stripplot(
        x='Group',
        y='Expression',
        data=df,
        color='#333333',
        size=4,
        alpha=0.45,
        jitter=0.2,
        ax=ax
    )
    
    # Titles & labels
    ax.set_title(f"Genomic Profile: {gene}", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(f"Clinical Status: {contrast_label}", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Continuous Expression (VST / Normalized Counts)", fontsize=11, fontweight='bold', labelpad=10)
    
    # Scientific annotation text
    stat_text = (
        f"Wilcoxon p = {p_val:.2e}\n"
        f"Control Mean = {mean_ctrl:.2f}\n"
        f"Disease Mean = {mean_dis:.2f}\n"
        f"Fold Change = {fold_change:.2f}x"
    )
    
    # Coordinate box for annotation text
    ax.text(
        0.05, 0.95, stat_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.8, edgecolor='lightgrey')
    )
    
    # Despine for publication standard
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" -> Saved publication-grade violin plot for {gene} to: {out_path}")


def run_self_test():
    """
    Generates mock expression and clinical metadata to verify plot aesthetics.
    """
    print("\n[SELF-TEST] Simulating patient expression cohort...")
    np.random.seed(42)
    
    # Generate 120 dummy patient IDs
    patient_ids = [f"MAP_{1000+i}" for i in range(120)]
    
    # Generate mock metadata (50% control, 50% disease)
    metadata_df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'AD_vs_NCI': np.random.choice([0, 1], size=120, p=[0.5, 0.5])
    })
    
    # Generate mock expression matrix:
    # 1. PTPN23: High outlier noise, means-based (DESeq2) fails, but rank-based AREA succeeds!
    # 2. MFGE8: Shifted mean, both find it.
    expression_data = {
        'Patient_ID': patient_ids
    }
    
    expression_data['PTPN23'] = np.where(
        metadata_df['AD_vs_NCI'] == 1,
        np.random.normal(loc=12.0, scale=1.5, size=120),  # Disease stable
        np.random.normal(loc=8.0, scale=1.5, size=120)    # Control has outliers
    )
    # Inject heavy outliers into controls to confuse DESeq2!
    control_indices = metadata_df[metadata_df['AD_vs_NCI'] == 0].index
    outlier_idx = np.random.choice(control_indices, size=8, replace=False)
    expression_data['PTPN23'][outlier_idx] += np.random.uniform(25.0, 40.0, size=8)
    
    # MFGE8: Clear shift, clear dosage
    expression_data['MFGE8'] = np.where(
        metadata_df['AD_vs_NCI'] == 1,
        np.random.normal(loc=4.5, scale=1.0, size=120),
        np.random.normal(loc=9.0, scale=1.0, size=120)
    )
    
    expression_df = pd.DataFrame(expression_data)
    
    outdir = "results_violins_test"
    os.makedirs(outdir, exist_ok=True)
    
    # Align and Plot
    # Match on Patient_ID
    metadata_df['Patient_ID_Clean'] = clean_patient_ids(metadata_df['Patient_ID'])
    expression_df['Patient_ID_Clean'] = clean_patient_ids(expression_df['Patient_ID'])
    
    merged = pd.merge(expression_df, metadata_df, on='Patient_ID_Clean')
    
    for gene in ['PTPN23', 'MFGE8']:
        generate_violin(
            gene=gene,
            expression_series=merged[gene],
            labels_series=merged['AD_vs_NCI'],
            contrast_label="Alzheimer's Disease (AD)",
            out_path=os.path.join(outdir, f"violin_{gene}_test.png")
        )
    print(f"Self-test completed! Mock violin plots saved inside folder: {outdir}\n")


def main():
    parser = argparse.ArgumentParser(description="Continuous Genomic Feature Violin Plot Generation")
    parser.add_argument("--expression", default=None, help="Path to continuous expression matrix CSV/TSV.")
    parser.add_argument("--clinical", default=None, help="Path to clinical boolean metadata CSV.")
    parser.add_argument("--genes", default=None, help="Comma-separated list of genes to plot.")
    parser.add_argument("--contrast", default=None, help="Clinical contrast column name (e.g. AD_vs_NCI).")
    parser.add_argument("--id-col-expr", default="Patient_ID", help="Patient ID column name in expression sheet.")
    parser.add_argument("--id-col-clin", default="Patient_ID", help="Patient ID column name in clinical metadata sheet.")
    parser.add_argument("-o", "--outdir", default="results/violins", help="Output directory for plots.")
    parser.add_argument("--self-test", action="store_true", help="Runs functional self-test with dummy data.")
    args = parser.parse_args()
    
    if args.self_test or (not args.expression and not args.clinical):
        run_self_test()
        sys.exit(0)
        
    if not args.genes or not args.contrast:
        print("Error: Please specify genes list (--genes) and target clinical contrast (--contrast).")
        sys.exit(1)
        
    # Read files
    sep_expr = '\t' if args.expression.endswith('.tsv') or args.expression.endswith('.txt') else ','
    sep_clin = '\t' if args.clinical.endswith('.tsv') or args.clinical.endswith('.txt') else ','
    
    try:
        expr_df = pd.read_csv(args.expression, sep=sep_expr)
        clin_df = pd.read_csv(args.clinical, sep=sep_clin)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
        
    # Align and Clean Patient IDs
    if args.id_col_expr not in expr_df.columns:
        print(f"Error: Patient column '{args.id_col_expr}' not found in expression sheet. Available columns: {list(expr_df.columns[:5])}")
        sys.exit(1)
    if args.id_col_clin not in clin_df.columns:
        print(f"Error: Patient column '{args.id_col_clin}' not found in clinical sheet. Available columns: {list(clin_df.columns[:5])}")
        sys.exit(1)
    if args.contrast not in clin_df.columns:
        print(f"Error: Contrast column '{args.contrast}' not found in clinical sheet. Available columns: {list(clin_df.columns)}")
        sys.exit(1)
        
    expr_df['ID_Clean'] = clean_patient_ids(expr_df[args.id_col_expr])
    clin_df['ID_Clean'] = clean_patient_ids(clin_df[args.id_col_clin])
    
    merged = pd.merge(expr_df, clin_df[['ID_Clean', args.contrast]], on='ID_Clean')
    print(f"Successfully aligned and matched {len(merged):,} patient profiles across both datasets.")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    genes_list = [g.strip().upper() for g in args.genes.split(',')]
    for gene in genes_list:
        matching_cols = [col for col in merged.columns if col.upper() == gene]
        if not matching_cols:
            print(f" -> Warning: Gene '{gene}' not found in expression dataset. Skipping.")
            continue
        
        target_col = matching_cols[0]
        out_file = os.path.join(args.outdir, f"violin_{gene}_{args.contrast}.png")
        generate_violin(
            gene=gene,
            expression_series=merged[target_col],
            labels_series=merged[args.contrast],
            contrast_label=args.contrast.replace('_', ' ').title(),
            out_path=out_file
        )


if __name__ == "__main__":
    main()

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_enrichment_plot(ranks_path, bools_path, results_path, gene, attribute, output_dir):
    """
    Generates a publication-quality GSEA-style running enrichment curve and barcode track
    as well as a split violin plot of raw expression for a given gene-attribute pair.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LOAD DATA
    print("Loading datasets...")
    try:
        ranks_df = pd.read_csv(ranks_path, index_col=0)
        bools_df = pd.read_csv(bools_path, index_col=0)
        results_df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
        
    # Check that gene and attribute exist
    if gene not in ranks_df.columns:
        print(f"Error: Gene '{gene}' not found in expression ranks columns.")
        sys.exit(1)
    if attribute not in bools_df.columns:
        print(f"Error: Attribute '{attribute}' not found in boolean matrix columns.")
        sys.exit(1)
        
    # Align samples
    common_samples = ranks_df.index.intersection(bools_df.index)
    if len(common_samples) == 0:
        print("Error: No common samples found between ranks and boolean files.")
        sys.exit(1)
        
    # Subset and drop NaNs for this specific attribute
    expr_series = ranks_df.loc[common_samples, gene]
    bool_series = bools_df.loc[common_samples, attribute]
    
    # Combine into a temporary DataFrame and drop NaNs
    pair_df = pd.DataFrame({'expr': expr_series, 'label': bool_series}).dropna()
    
    if len(pair_df) == 0:
        print("Error: No valid samples left after dropping NaNs.")
        sys.exit(1)
        
    # Sort descending by expression (highest expression rank = 0)
    pair_df = pair_df.sort_values(by='expr', ascending=False)
    
    N = len(pair_df)
    M = int(pair_df['label'].sum())
    
    if M == 0 or M == N:
        print(f"Error: Attribute '{attribute}' has no variation (either all 0s or all 1s).")
        sys.exit(1)
        
    print(f"Analyzing {gene} x {attribute}: {N} total samples, {M} cases (1), {N-M} controls (0)")
    
    # Extract NES and adjusted p-value from results file
    nes = 0.0
    pval_bh = 1.0
    
    # Standardize column names of results file to locate statistics
    results_df.columns = [c.lower().replace('_', '').replace('-', '') for c in results_df.columns]
    
    # Locate row matching gene and attribute
    # Find matching columns for attribute and gene
    attr_col = None
    gene_col = None
    for c in ['boolcolumn', 'booleanattribute', 'booleanfilecol']:
        if c in results_df.columns:
            attr_col = c
            break
    for c in ['rankcolumn', 'rankfilecol', 'feature']:
        if c in results_df.columns:
            gene_col = c
            break
            
    if attr_col and gene_col:
        match = results_df[
            (results_df[attr_col].astype(str).str.lower().str.replace('_', '') == attribute.lower().replace('_', '')) &
            (results_df[gene_col].astype(str).str.lower() == gene.lower())
        ]
        if len(match) > 0:
            # Get NES and BH p-value
            nes_cols = [c for c in results_df.columns if 'nes' in c]
            bh_cols = [c for c in results_df.columns if 'bh' in c or 'benjamini' in c or 'padj' in c]
            if nes_cols:
                nes = float(match[nes_cols[0]].values[0])
            if bh_cols:
                pval_bh = float(match[bh_cols[0]].values[0])
                
    print(f"Retrieved Stats from Results: NES = {nes:.4f}, BH-adjusted p = {pval_bh:.2e}")
    
    # ==========================================================
    # PLOT 1: RUNNING ENRICHMENT CURVE & BARCODE TRACK
    # ==========================================================
    # Calculate GSEA-style running enrichment sum
    labels_sorted = pair_df['label'].values
    
    # Cumulative fraction of cases (P_run)
    p_run = np.cumsum(labels_sorted) / M
    # Line of expectation (P_exp)
    p_exp = np.arange(1, N + 1) / N
    
    # Subtract diagonal expectation to center curve if desired,
    # or plot them against each other as in the paper.
    # Figure 2A: y-axis is Cumulative score, diagonal is line of expectation.
    # Enrichment score is twice the area between them.
    
    # Let's set up a beautiful two-panel GSEA-style figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(10, 7), 
        sharex=True, 
        gridspec_kw={'height_ratios': [4, 1]}
    )
    plt.subplots_adjust(hspace=0.05)
    
    # Top Panel: Running Enrichment Curve
    ax1.plot(range(1, N + 1), p_run, label="Running Sum ($P_{run}$)", color="#1abc9c", linewidth=2.5)
    ax1.plot(range(1, N + 1), p_exp, label="Line of Expectation ($P_{exp}$)", color="#7f8c8d", linestyle="--", linewidth=1.5)
    
    # Fill the area between the curve and the expectation
    # If positive NES, color is blue-ish, if negative, red-ish
    fill_color = "#3498db" if nes >= 0 else "#e74c3c"
    ax1.fill_between(range(1, N + 1), p_run, p_exp, color=fill_color, alpha=0.2, label=f"Enrichment Area (NES = {nes:.2f})")
    
    ax1.set_ylabel("Cumulative Score", fontsize=12)
    ax1.set_title(f"Running Enrichment Curve: {gene} vs {attribute}\nNES: {nes:.3f} | BH-adjusted p: {pval_bh:.2e}", fontsize=14, fontweight='bold')
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Bottom Panel: Barcode Track
    for idx, label in enumerate(labels_sorted):
        if label == 1:
            ax2.axvline(x=idx + 1, color="#2c3e50", alpha=0.6, linewidth=1.2)
            
    ax2.set_yticks([])
    ax2.set_ylabel("Cases", rotation=0, labelpad=20, va="center", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Samples Ranked by Expression (Highest to Lowest)", fontsize=12)
    ax2.set_xlim(1, N)
    
    # Highlight extreme zones
    ax2.axvspan(1, N*0.1, color="#2ecc71", alpha=0.1) # Top 10%
    ax2.axvspan(N*0.9, N, color="#e74c3c", alpha=0.1) # Bottom 10%
    
    # Save the figure
    curve_out_path = os.path.join(output_dir, f"{gene}_{attribute}_enrichment_curve.png")
    plt.savefig(curve_out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved running enrichment curve to: {curve_out_path}")
    
    # ==========================================================
    # PLOT 2: SPLIT VIOLIN / JITTER PLOT
    # ==========================================================
    plt.figure(figsize=(7, 6))
    
    # Map binary categories to strings for plotting
    pair_df['Cohort'] = pair_df['label'].map({0.0: f"Controls\n(N={N-M})", 1.0: f"Cases\n(N={M})"})
    
    # Violin plot
    sns.violinplot(
        x='Cohort', 
        y='expr', 
        data=pair_df, 
        palette=['#95a5a6', '#3498db' if nes >= 0 else '#e74c3c'],
        inner=None, 
        alpha=0.4,
        hue='Cohort',
        legend=False
    )
    
    # Jitter points
    sns.stripplot(
        x='Cohort', 
        y='expr', 
        data=pair_df, 
        color="#2c3e50", 
        size=4, 
        jitter=0.25, 
        alpha=0.6
    )
    
    # Add mean lines
    means = pair_df.groupby('Cohort')['expr'].mean()
    for idx, cohort in enumerate(pair_df['Cohort'].unique()):
        plt.hlines(means[cohort], idx-0.2, idx+0.2, colors='black', linestyles='solid', linewidths=2.5)
        
    # Calculate fold change between means (since values are VST/log-like, we can show difference)
    mean_control = pair_df[pair_df['label'] == 0]['expr'].mean()
    mean_case = pair_df[pair_df['label'] == 1]['expr'].mean()
    diff = mean_case - mean_control
    
    plt.title(f"{gene} Raw Expression by {attribute}\nMean Diff (Case - Control): {diff:+.3f}", fontsize=13, fontweight='bold')
    plt.ylabel(f"{gene} Expression (VST Normalized Counts)", fontsize=11)
    plt.xlabel("")
    plt.grid(True, linestyle=":", alpha=0.5)
    
    # Save violin plot
    violin_out_path = os.path.join(output_dir, f"{gene}_{attribute}_expression_violin.png")
    plt.savefig(violin_out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved expression comparison violin plot to: {violin_out_path}")
    print("--------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create official publication GSEA enrichment curves and split violin plots.")
    parser.add_argument("-r", "--ranks", required=True, help="Path to preprocessed continuous ranks CSV")
    parser.add_argument("-b", "--bools", required=True, help="Path to preprocessed boolean attributes CSV")
    parser.add_argument("-s", "--results", required=True, help="Path to AREA adjusted pvalues CSV")
    parser.add_argument("-g", "--gene", required=True, help="Gene symbol to plot (e.g. RCAN1)")
    parser.add_argument("-a", "--attribute", required=True, help="Boolean attribute to plot (e.g. High_Braak)")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory for generated plots")
    
    args = parser.parse_args()
    generate_enrichment_plot(args.ranks, args.bools, args.results, args.gene, args.attribute, args.outdir)

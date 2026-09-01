import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(area_results_path, output_dir, p_adj_threshold=0.05, top_n=15):
    """
    Analyzes and visualizes the results of an AREA run on ROSMAP data.
    """
    print("--------------------------------------------------")
    print("ROSMAP AREA Results Analyzer and Visualizer (v2)")
    print("--------------------------------------------------")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LOAD RESULTS
    print(f"Loading AREA results from: {area_results_path}")
    try:
        df = pd.read_csv(area_results_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
        
    print(f"Original results contain {len(df)} gene-attribute pairs.")
    
    # Identify column names (lower or upper case mapping)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower().replace('_', '').replace('-', '')
        col_mapping[col_lower] = col
        
    # Standard column checks (robust matching for standard AREA column patterns)
    bool_col = col_mapping.get('boolcolumn', col_mapping.get('booleanattribute', col_mapping.get('booleanfilecol', df.columns[0])))
    rank_col = col_mapping.get('rankcolumn', col_mapping.get('rankfilecol', df.columns[1]))
    nes_col = col_mapping.get('nes', df.columns[2] if len(df.columns) > 2 else 'nes')
    pval_col = col_mapping.get('pvalue', col_mapping.get('rawpvalue', df.columns[3] if len(df.columns) > 3 else 'pvalue'))
    bh_col = col_mapping.get('pvaluebh', col_mapping.get('benjaminihochberg', col_mapping.get('padj', df.columns[6] if len(df.columns) > 6 else 'pvalue_bh')))
    
    print(f"Mapping columns:")
    print(f"  - Attribute Column: {bool_col}")
    print(f"  - Gene/Rank Column: {rank_col}")
    print(f"  - NES Column:       {nes_col}")
    print(f"  - BH-Adjusted P:    {bh_col}")
    
    # 2. FILTER FOR SIGNIFICANT RESULTS
    print(f"Filtering results using BH-adjusted p-value cutoff < {p_adj_threshold}...")
    sig_df = df[df[bh_col] < p_adj_threshold].copy()
    print(f"Found {len(sig_df)} statistically significant associations.")
    
    if len(sig_df) == 0:
        print("Warning: No significant associations found at this threshold. Saving top 100 closest to significance.")
        sig_df = df.sort_values(by=bh_col).head(100) # Save top 100 closest to significance
        
    # Save significant results
    sig_out_path = os.path.join(output_dir, "sig_area_associations.csv")
    sig_df.to_csv(sig_out_path, index=False)
    print(f"Significant associations saved to: {sig_out_path}")
    
    # 3. SUMMARIZE PER COGNITIVE/PATHOLOGICAL TRAIT
    print("\nSummary of Significant Associations by Attribute:")
    summary_data = []
    for trait, group in sig_df.groupby(bool_col):
        total = len(group)
        up_regulated = (group[nes_col] > 0).sum()
        down_regulated = (group[nes_col] < 0).sum()
        print(f"  * {trait}: {total} total genes ({up_regulated} positive NES [high-expression risk], {down_regulated} negative NES [low-expression risk])")
        summary_data.append({
            'Attribute': trait,
            'Total_Significant_Genes': total,
            'Positive_NES_Risk': up_regulated,
            'Negative_NES_Risk': down_regulated
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "attribute_summary.csv"), index=False)
    
    # 4. PLOT TOP ASSOCIATIONS FOR EACH ATTRIBUTE
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    for trait, group in sig_df.groupby(bool_col):
        # Sort by statistical significance (lowest p-adj) and select top N
        top_group = group.sort_values(by=bh_col).head(top_n).copy()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Color bar based on sign of NES
        colors = ['#e74c3c' if x < 0 else '#3498db' for x in top_group[nes_col]]
        
        # Horizontal bar plot
        sns.barplot(
            x=nes_col,
            y=rank_col,
            data=top_group,
            palette=colors,
            hue=rank_col,
            legend=False
        )
        
        plt.title(f"Top {top_n} Genes Associated with {trait}\n(BH-adjusted p-value < {p_adj_threshold})", fontsize=14, fontweight='bold')
        plt.xlabel("Normalized Enrichment Score (NES)\n[Negative: Low Expression Risk | Positive: High Expression Risk]", fontsize=11)
        plt.ylabel("Gene Symbol", fontsize=11)
        
        # Tight layout and save
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"top_genes_{trait.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved bar plot of top genes for {trait} to: {plot_path}")
        
    print("--------------------------------------------------")
    print("Visualization processing completed successfully!")
    print("--------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize and summarize significant gene-disease associations from AREA.")
    parser.add_argument("-i", "--input", required=True, help="Path to AREA output CSV file")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory for charts and tables")
    parser.add_argument("-p", "--p-cutoff", type=float, default=0.05, help="Benjamini-Hochberg adjusted p-value threshold (default: 0.05)")
    parser.add_argument("-n", "--top-n", type=int, default=15, help="Number of top genes to plot per attribute (default: 15)")
    
    args = parser.parse_args()
    visualize_results(args.input, args.outdir, args.p_cutoff, args.top_n)

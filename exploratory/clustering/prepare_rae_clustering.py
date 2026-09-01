import os
import sys
import argparse
import pandas as pd
import numpy as np

def calculate_leading_edge(expression_series, bool_series, nes):
    """
    Calculates the GSEA-style leading edge inflection point for a gene-attribute pair.
    
    Parameters:
    - expression_series: pd.Series of continuous gene expression values (aligned with bool_series)
    - bool_series: pd.Series of binary attribute labels (0.0 or 1.0, aligned with expression_series)
    - nes: float, Normalized Enrichment Score from AREA
    
    Returns:
    - threshold_val: float, the expression value at the leading edge boundary
    - rae_indices: set, sample IDs that fall into the Risk-Associated Expression (RAE) region
    """
    # Align and drop NaNs
    df = pd.DataFrame({'expr': expression_series, 'label': bool_series}).dropna()
    if len(df) == 0:
        return np.nan, set()
        
    # Sort samples by expression descending (highest to lowest)
    df_sorted = df.sort_values(by='expr', ascending=False)
    
    expr_vals = df_sorted['expr'].values
    labels = df_sorted['label'].values
    samples = df_sorted.index.values
    
    N = len(labels)
    M = int(np.sum(labels))
    
    # If no cases or all are cases, leading edge is not meaningful
    if M == 0 or M == N:
        return np.nan, set()
        
    # Compute running sum and expectation
    # P_run[i] = cumulative sum of cases up to index i / total cases
    # P_exp[i] = (i + 1) / N
    p_run = np.cumsum(labels) / M
    p_exp = np.arange(1, N + 1) / N
    
    if nes > 0:
        # Positive NES: high expression drives risk. 
        # We want to find the index that maximizes (P_run - P_exp)
        diff = p_run - p_exp
        idx = np.argmax(diff)
        threshold_val = expr_vals[idx]
        # RAE contains all samples with expression >= threshold_val
        rae_samples = set(df_sorted.iloc[:idx+1].index)
    else:
        # Negative NES: low expression drives risk. 
        # We want to find the index that maximizes (P_exp - P_run)
        diff = p_exp - p_run
        idx = np.argmax(diff)
        threshold_val = expr_vals[idx]
        # RAE contains all samples with expression <= threshold_val
        rae_samples = set(df_sorted.iloc[idx:].index)
        
    return threshold_val, rae_samples

def prepare_rae_matrices(ranks_path, bools_path, results_path, output_dir, p_adj_threshold=0.05):
    """
    Identifies RAE thresholds for all significant gene-attribute pairs and 
    generates a binary patient-by-gene risk matrix for downstream clustering.
    """
    print("--------------------------------------------------")
    print("AREA Leading Edge & RAE Matrix Builder")
    print("--------------------------------------------------")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading continuous ranks from: {ranks_path}")
    ranks_df = pd.read_csv(ranks_path, index_col=0)
    print(f"  -> Shape: {ranks_df.shape} (samples x genes)")
    
    print(f"Loading boolean attributes from: {bools_path}")
    bools_df = pd.read_csv(bools_path, index_col=0)
    print(f"  -> Shape: {bools_df.shape} (samples x attributes)")
    
    print(f"Loading AREA significance results from: {results_path}")
    results_df = pd.read_csv(results_path)
    print(f"  -> Total gene-attribute pairs: {len(results_df)}")
    
    # Identify key columns in results
    col_mapping = {}
    for col in results_df.columns:
        col_lower = col.lower().replace('_', '').replace('-', '')
        col_mapping[col_lower] = col
        
    bool_col = col_mapping.get('booleanattribute', col_mapping.get('boolcolumn', results_df.columns[0]))
    rank_col = col_mapping.get('rankcolumn', col_mapping.get('rankfilecol', results_df.columns[1]))
    nes_col = col_mapping.get('nes', 'nes')
    bh_col = col_mapping.get('pvaluebh', col_mapping.get('benjaminihochberg', 'pvalue_bh'))
    
    print(f"Mapping columns:")
    print(f"  - Attribute Column: {bool_col}")
    print(f"  - Gene Column:      {rank_col}")
    print(f"  - NES Column:       {nes_col}")
    print(f"  - BH-Adjusted P:    {bh_col}")
    
    # 2. Filter for significant associations
    sig_df = results_df[results_df[bh_col] < p_adj_threshold].copy()
    print(f"\nFiltering for significant associations (BH-adjusted p < {p_adj_threshold})...")
    print(f"  -> Found {len(sig_df)} significant gene-attribute associations.")
    
    if len(sig_df) == 0:
        print("Error: No statistically significant associations found at this threshold!")
        print("Please check your adjusted p-values or try a less stringent threshold.")
        sys.exit(1)
        
    # Align samples
    common_samples = sorted(list(set(ranks_df.index).intersection(set(bools_df.index))))
    print(f"  -> Aligned {len(common_samples)} common samples.")
    
    ranks_aligned = ranks_df.loc[common_samples]
    bools_aligned = bools_df.loc[common_samples]
    
    # 3. Group by Clinical Attribute to build individual matrices
    summary_records = []
    
    for attribute, group in sig_df.groupby(bool_col):
        if attribute not in bools_aligned.columns:
            print(f"Warning: Attribute '{attribute}' from results not found in boolean attributes file columns. Skipping.")
            continue
            
        print(f"\nProcessing Attribute: '{attribute}'")
        sig_genes = group[rank_col].unique()
        # Filter genes that are actually in ranks_df
        sig_genes_aligned = [g for g in sig_genes if g in ranks_aligned.columns]
        print(f"  -> Significant genes for this attribute: {len(sig_genes)} ({len(sig_genes_aligned)} present in ranks file)")
        
        if len(sig_genes_aligned) == 0:
            continue
            
        # Initialize binary RAE dataframe (samples x genes) for this attribute
        # Default value is 0
        rae_matrix = pd.DataFrame(0, index=common_samples, columns=sig_genes_aligned)
        
        # We also want to record the sample list where this attribute is not NaN
        attr_non_nan_samples = bools_aligned[attribute].dropna().index
        
        for idx, row in group.iterrows():
            gene = row[rank_col]
            nes = row[nes_col]
            p_val_bh = row[bh_col]
            
            if gene not in sig_genes_aligned:
                continue
                
            # Run leading edge
            thresh, rae_samples = calculate_leading_edge(
                ranks_aligned[gene], 
                bools_aligned[attribute], 
                nes
            )
            
            if pd.isna(thresh):
                continue
                
            # Set 1 for samples in RAE
            rae_matrix.loc[list(rae_samples), gene] = 1
            
            # Mask samples where the attribute was NaN (exclude from RAE analysis)
            all_samples_set = set(common_samples)
            nan_samples = all_samples_set - set(attr_non_nan_samples)
            rae_matrix.loc[list(nan_samples), gene] = np.nan
            
            # Compute prevalence inside vs outside RAE
            aligned_df = pd.DataFrame({
                'rae': rae_matrix[gene], 
                'label': bools_aligned[attribute]
            }).dropna()
            
            prev_inside = aligned_df[aligned_df['rae'] == 1]['label'].mean()
            prev_outside = aligned_df[aligned_df['rae'] == 0]['label'].mean()
            
            summary_records.append({
                'Attribute': attribute,
                'Gene': gene,
                'NES': nes,
                'p_value_BH': p_val_bh,
                'RAE_Direction': 'High' if nes > 0 else 'Low',
                'RAE_Threshold_VST': thresh,
                'RAE_Sample_Count': len(rae_samples),
                'Prevalence_Inside_RAE': prev_inside,
                'Prevalence_Outside_RAE': prev_outside,
                'Risk_Ratio': prev_inside / prev_outside if prev_outside > 0 else np.inf
            })
            
        # Drop rows that are fully NaN (samples where attribute was missing)
        rae_matrix = rae_matrix.dropna(how='all')
        
        # Save RAE matrix
        matrix_filename = f"rae_matrix_{attribute.lower().replace(' ', '_')}.csv"
        matrix_path = os.path.join(output_dir, matrix_filename)
        rae_matrix.to_csv(matrix_path)
        print(f"  -> Saved binary RAE risk matrix to: {matrix_path} (Shape: {rae_matrix.shape})")
        
    # Save leading edge thresholds summary
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, "rae_thresholds_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved comprehensive RAE thresholds summary to: {summary_path}")
    print("--------------------------------------------------")
    print("RAE Preprocessing Completed Successfully!")
    print("--------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculates GSEA-style leading edge thresholds and builds binary RAE matrices.")
    parser.add_argument("-r", "--ranks", required=True, help="Path to preprocessed continuous ranks CSV (samples x genes)")
    parser.add_argument("-b", "--bools", required=True, help="Path to preprocessed boolean attributes CSV (samples x attributes)")
    parser.add_argument("-s", "--sig-results", required=True, help="Path to AREA significance CSV table")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory to save matrices and summary")
    parser.add_argument("-p", "--p-cutoff", type=float, default=0.05, help="BH-adjusted p-value significance cutoff (default: 0.05)")
    
    args = parser.parse_args()
    prepare_rae_matrices(args.ranks, args.bools, args.sig_results, args.outdir, args.p_cutoff)

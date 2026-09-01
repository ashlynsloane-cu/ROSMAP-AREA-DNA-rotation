import os
import sys
import argparse
import pandas as pd
import numpy as np

def preprocess_rosmap(expression_path, metadata_path, output_dir, vst_threshold=None):
    """
    Preprocesses ROSMAP gene expression and clinical metadata files
    for use with the Attribute Rank Enrichment Algorithm (AREA).
    """
    print("--------------------------------------------------")
    print("Preparing ROSMAP Data for AREA Preprocessing")
    print("--------------------------------------------------")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LOAD GENE EXPRESSION DATA
    print(f"Loading gene expression matrix from: {expression_path}")
    try:
        # Check if the file is comma or tab separated
        if expression_path.endswith('.tsv') or expression_path.endswith('.txt'):
            expr_df = pd.read_csv(expression_path, sep='\t')
        else:
            expr_df = pd.read_csv(expression_path)
    except Exception as e:
        print(f"Error loading expression matrix: {e}")
        sys.exit(1)
        
    print(f"Original expression matrix shape: {expr_df.shape}")
    
    # Ensure the first column is the gene symbol
    first_col = expr_df.columns[0]
    print(f"Detected gene identifier column: '{first_col}'")
    expr_df = expr_df.rename(columns={first_col: 'gene_symbol'})
    
    # Set gene_symbol as index and transpose to get (samples x genes)
    expr_df = expr_df.set_index('gene_symbol')
    
    # Strip whitespace from column names (sample IDs) and index (genes)
    expr_df.columns = expr_df.columns.str.strip()
    expr_df.index = expr_df.index.str.strip()
    
    # Transpose matrix to (samples x genes)
    print("Transposing expression matrix to samples x genes (rank file format)...")
    expr_transposed = expr_df.T
    expr_transposed.index.name = 'sample_id'
    
    print(f"Transposed expression matrix shape: {expr_transposed.shape} (samples x genes)")
    
    # 2. FILTERING LOW EXPRESSION GENES
    if vst_threshold is not None:
        print(f"Filtering genes with mean VST expression level < {vst_threshold}...")
        mean_expr = expr_transposed.mean(axis=0)
        genes_to_keep = mean_expr[mean_expr >= vst_threshold].index
        expr_transposed = expr_transposed[genes_to_keep]
        print(f"Expression matrix shape after filtering: {expr_transposed.shape} (samples x genes)")
    else:
        print("No VST expression threshold applied. Keeping all genes (variance filtering will be handled by AREA).")

    # 3. LOAD METADATA
    print(f"Loading clinical metadata from: {metadata_path}")
    try:
        if metadata_path.endswith('.tsv') or metadata_path.endswith('.txt'):
            meta_df = pd.read_csv(metadata_path, sep='\t')
        else:
            meta_df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        sys.exit(1)
        
    print(f"Original metadata shape: {meta_df.shape}")
    
    # Find sample_id column (case insensitive search)
    sample_col = None
    for col in meta_df.columns:
        if col.lower() == 'sample_id':
            sample_col = col
            break
            
    if not sample_col:
        raise ValueError("Could not find 'sample_id' column in the metadata file. Available columns: " + ", ".join(meta_df.columns))
        
    print(f"Using '{sample_col}' as the sample identifier in clinical metadata.")
    meta_df = meta_df.rename(columns={sample_col: 'sample_id'})
    meta_df['sample_id'] = meta_df['sample_id'].astype(str).str.strip()
    
    # Set sample_id as the index
    meta_df = meta_df.set_index('sample_id')
    
    # 4. BINARIZE CLINICAL COGNITIVE & PATHOLOGICAL TRAITS
    print("Binarizing clinical and pathological attributes...")
    bool_df = pd.DataFrame(index=meta_df.index)
    
    # A. Binarize Cognitive Diagnosis (ROSMAP diagnosis: 1=NCI, 2/3=MCI, 4/5=AD, 6=Other dementia)
    if 'diagnosis' in meta_df.columns:
        # 1. Progressive Clinical Levels (Three-way split, no samples dropped!)
        bool_df['NCI_vs_Rest'] = meta_df['diagnosis'].map({1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0})
        bool_df['MCI_vs_Rest'] = meta_df['diagnosis'].map({1: 0.0, 2: 1.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0})
        bool_df['AD_vs_Rest'] = meta_df['diagnosis'].map({1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0, 6: 0.0})
        
        # 2. Classic Cohort Targets
        bool_df['AD_vs_NCI'] = meta_df['diagnosis'].map({1: 0.0, 4: 1.0, 5: 1.0})
        bool_df['Cognitive_Impairment_vs_NCI'] = meta_df['diagnosis'].map({
            1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0
        })
        print("  - Created inclusive 'NCI_vs_Rest', 'MCI_vs_Rest', 'AD_vs_Rest', 'AD_vs_NCI', and 'Cognitive_Impairment_vs_NCI' variables.")
    else:
        print("  - Warning: 'diagnosis' column not found in metadata.")

    # B. Binarize Braak Stage (Tangle pathology: 0-6 scale)
    if 'braak' in meta_df.columns:
        # Group Braak III-VI (Moderate-to-Severe) vs Braak 0-II (None-to-Mild) to keep ALL 578 patients!
        def binarize_braak_inclusive(val):
            if pd.isna(val): return np.nan
            val = float(val)
            if val >= 3: return 1.0
            else: return 0.0
            
        bool_df['Braak_III_VI_vs_0_II'] = meta_df['braak'].apply(binarize_braak_inclusive)
        
        # Keep old binarizer for legacy comparison
        def binarize_braak_extreme(val):
            if pd.isna(val): return np.nan
            val = float(val)
            if val >= 5: return 1.0
            elif val <= 2: return 0.0
            return np.nan
        bool_df['High_Braak'] = meta_df['braak'].apply(binarize_braak_extreme)
        
        print("  - Created 'Braak_III_VI_vs_0_II' (inclusive) and 'High_Braak' (extreme) pathological indicators.")
    else:
        print("  - Warning: 'braak' column not found in metadata.")

    # C. Binarize CERAD Score (Amyloid plaque pathology)
    if 'cerad' in meta_df.columns:
        def binarize_cerad(val):
            if pd.isna(val): return np.nan
            val = float(val)
            if val in [1, 2]: return 1.0
            elif val in [3, 4]: return 0.0
            return np.nan
            
        bool_df['High_CERAD'] = meta_df['cerad'].apply(binarize_cerad)
        print("  - Created 'High_CERAD' pathology indicator.")
    else:
        print("  - Warning: 'cerad' column not found in metadata.")

    # D. Binarize APOE ε4 carrier status
    if 'apoe' in meta_df.columns:
        def binarize_apoe(val):
            if pd.isna(val): return np.nan
            val_str = str(int(val)) if not pd.isna(val) else ""
            if '4' in val_str: return 1.0
            elif val_str in ['22', '23', '33']: return 0.0
            return np.nan
            
        bool_df['APOE_e4_carrier'] = meta_df['apoe'].apply(binarize_apoe)
        print("  - Created 'APOE_e4_carrier' status.")
    else:
        print("  - Warning: 'apoe' column not found in metadata.")

    # E. Binarize Sex - BOTH Sex_Male AND Sex_Female to avoid sex bias!
    if 'sex' in meta_df.columns:
        bool_df['Sex_Male'] = meta_df['sex'].map({1: 1.0, 0: 0.0})
        bool_df['Sex_Female'] = meta_df['sex'].map({1: 0.0, 0: 1.0})
        print("  - Created 'Sex_Male' and 'Sex_Female' binary covariates.")
    else:
        print("  - Warning: 'sex' column not found in metadata.")

    # 5. FIND COMMON SAMPLES (MERGE/ALIGN)
    expr_samples = set(expr_transposed.index)
    bool_samples = set(bool_df.index)
    common_samples = sorted(list(expr_samples.intersection(bool_samples)))
    
    print(f"Alignment Summary:")
    print(f"  - Samples in expression matrix: {len(expr_samples)}")
    print(f"  - Samples in metadata file:    {len(bool_samples)}")
    print(f"  - Common samples found:        {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("Error: No overlapping samples found between expression matrix and metadata!")
        sys.exit(1)
        
    # Subset both to common samples
    expr_final = expr_transposed.loc[common_samples]
    bool_final = bool_df.loc[common_samples]
    
    # 6. SAVE PREPROCESSED FILES
    expr_out_path = os.path.join(output_dir, "rosmap_area_ranks.csv")
    bool_out_path = os.path.join(output_dir, "rosmap_area_bools.csv")
    
    print(f"Saving preprocessed continuous ranks to: {expr_out_path}")
    expr_final.to_csv(expr_out_path)
    
    print(f"Saving preprocessed boolean attributes to: {bool_out_path}")
    bool_final.to_csv(bool_out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocesses ROSMAP gene expression and clinical metadata files.")
    parser.add_argument("-e", "--expression", required=True, help="Path to continuous VST expression CSV/TSV")
    parser.add_argument("-m", "--metadata", required=True, help="Path to clinical metadata CSV/TSV")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory to save preprocessed matrices")
    parser.add_argument("-t", "--vst-threshold", type=float, default=None, help="Optional mean VST expression threshold (e.g. 1.0)")
    
    args = parser.parse_args()
    preprocess_rosmap(args.expression, args.metadata, args.outdir, args.vst_threshold)

#!/usr/bin/env python3
"""
prepare_rae_clustering.py
======================================================================
ROSMAP Analysis Funnel - Tier 3: Leading Edge Thresholding & RAE Discretization
======================================================================
This script calculates GSEA-style leading edge inflection points for 
statistically significant gene-attribute associations from Tier 1. 

It converts continuous patient expression data (VST-normalized counts)
into a binary Risk-Associated Expression (RAE) state (1 for Risk, 0 for Control),
compressing the massive transcriptomic space into a Patient-by-Gene Binary 
Risk Matrix for downstream subtyping and additive synergy modeling.

Improvements in v3:
1. Universal Multi-File Auto-Detector: Scans and processes all 13 individual
   trait files (e.g. area_genome_wide_hypertension_history_comparison.csv)
   directly from the results directory, automatically matching them to 
   clinical traits and avoiding bash-side pattern mismatches.
2. Dynamic Fuzzy Trait Aligner: Maps filenames (like "hypertension_history")
   directly to active columns in "rosmap_area_bools.csv" (like "hypertension_cum").
3. Color-Coded Bio-Archetypes (Tier 1 matched):
   - Co-Progressive Driver -> Shared Response (Green)
   - State-Transition Trigger -> Threshold-like Response (Orange)
   - Dosage Accumulator -> Dosage-like Response (Blue)
4. Model-Specific Splitting: Gracefully handles traits analyzed by both Regular
   and Weighted models, saving separate RAE matrices for each model if significant.
======================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


def clean_id_to_string(val):
    """
    Standardizes participant IDs to string, stripping any 'R' prefix or decimals
    to guarantee a seamless match between ranks, bools, and results.
    """
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.endswith('.0'):
        val_str = val_str[:-2]
    # Strip common prefixes (like 'R' or 'r')
    if val_str.upper().startswith('R'):
        val_str = val_str[1:]
    return val_str.strip()


def clean_attribute_name(col_or_filename):
    """
    Cleans up file or column names into standardized, clean clinical labels.
    """
    name = col_or_filename.lower()
    
    # Strip known junk patterns
    junk = [
        "area_resultsarea_scores_",
        "area_genome_wide_",
        "_comparison",
        "method_comparison",
        "results_",
        "_regular_fdr",
        "_weighted_fdr",
        "_regular_nes",
        "_weighted_nes",
        "_nes",
        ".csv"
    ]
    for pattern in junk:
        name = name.replace(pattern, "")
        
    name = name.strip("_").strip()
    
    # Direct dictionary translation
    translation = {
        "high_braak": "High Braak",
        "high_cerad": "High CERAD",
        "mci_vs_rest": "MCI vs Rest",
        "ad_vs_rest": "AD vs Rest",
        "nci_vs_rest": "NCI vs Rest",
        "ad_vs_nci": "AD vs NCI",
        "cognitive_impairment_vs_nci": "Cognitive Impairment vs NCI",
        "cognitive": "Cognitive Impairment vs NCI",
        "consensus_cognition": "Cognitive Impairment vs NCI",
        "global_cognition": "Global Cognition",
        "apoe_e4_carrier": "APOE ε4 Carrier",
        "biological_sex": "Biological Sex",
        "sex_male": "Male Sex",
        "sex_female": "Female Sex",
        "braak_iii_vi_vs_0_ii": "Braak III-VI vs 0-II",
        "braak_continuous": "Braak Continuous",
        "braak_tangles": "Braak Tangles",
        "cerad_plaques": "Neuritic Plaques",
        "mmse_score": "MMSE Score",
        "body_mass_index": "Body Mass Index",
        "bmi": "Body Mass Index",
        "diabetes_history": "Diabetes History",
        "diabetes_sr_rx": "Diabetes History",
        "hypertension_history": "Hypertension History",
        "hypertension_cum": "Hypertension History",
        "stroke_history": "Stroke History",
        "stroke_cum": "Stroke History"
    }
    
    return translation.get(name, name.replace("_", " ").title())


def map_filename_to_bool_column(filename, bool_columns):
    """
    Fuzzily aligns the input results filename with the active column names
    in the preprocessed boolean attributes CSV matrix.
    """
    clean_fn = filename.lower().replace("area_genome_wide_", "").replace("_comparison", "").replace(".csv", "").strip("_")
    
    # Direct dictionary mapping for Dataset 1731 schemas
    direct_map = {
        "hypertension_history": "hypertension_cum",
        "diabetes_history": "diabetes_sr_rx",
        "stroke_history": "stroke_cum",
        "apoe_e4_carrier": "apoe_e4_carrier",
        "biological_sex": "sex_male",  # standard baseline is male
        "age_at_ad_onset": "age_first_ad_dx",
        "age_at_death": "age_death",
        "body_mass_index": "bmi",
        "mmse_score": "cts_estmmse30",
        "global_cognition": "cogn_global",
        "consensus_cognition": "cogn_global",
        "cognitive_comparison": "cognitive_impairment_vs_nci",
        "cognitive": "cognitive_impairment_vs_nci",
        "cerad_plaques": "ceradsc",
        "braak_tangles": "braaksc",
        "mci": "mci_vs_rest",
        "braak": "braak_iii_vi_vs_0_ii"
    }
    
    # 1. Check direct map
    for key, val in direct_map.items():
        if key in clean_fn:
            for col in bool_columns:
                if col.lower().strip() == val.lower().strip():
                    return col
                    
    # 2. Fuzzy substring fallback matching
    for col in bool_columns:
        col_lower = col.lower().strip()
        if col_lower in clean_fn or clean_fn in col_lower:
            return col
        if "hyper" in clean_fn and "hyper" in col_lower:
            return col
        if "diab" in clean_fn and "diab" in col_lower:
            return col
        if "stroke" in clean_fn and "stroke" in col_lower:
            return col
        if "apoe" in clean_fn and "apoe" in col_lower:
            return col
        if "sex" in clean_fn and "sex" in col_lower:
            return col
        if "cogn" in clean_fn and ("cogn" in col_lower or "mci" in col_lower or "ad_vs" in col_lower):
            return col
        if "cerad" in clean_fn and "cerad" in col_lower:
            return col
        if "braak" in clean_fn and "braak" in col_lower:
            return col
            
    return None


def calculate_leading_edge(expression_series, bool_series, nes):
    """
    Calculates the GSEA-style leading edge inflection point for a gene-attribute pair.
    """
    # Align and drop NaNs
    df = pd.DataFrame({'expr': expression_series, 'label': bool_series}).dropna()
    if len(df) == 0:
        return np.nan, set()
        
    # Sort samples by expression descending (highest expression rank = index 0)
    df_sorted = df.sort_values(by='expr', ascending=False)
    expr_vals = df_sorted['expr'].values
    labels = df_sorted['label'].values
    samples = df_sorted.index.values
    N = len(labels)
    M = int(np.sum(labels))
    
    # If no cases or all are cases, leading edge is not mathematically meaningful
    if M == 0 or M == N:
        return np.nan, set()
        
    # Compute running sum and expectation
    p_run = np.cumsum(labels) / M
    p_exp = np.arange(1, N + 1) / N
    
    if nes >= 0:
        # Positive NES: high expression drives risk.
        diff = p_run - p_exp
        idx = np.argmax(diff)
        threshold_val = expr_vals[idx]
        rae_samples = set(df_sorted.iloc[:idx+1].index)
    else:
        # Negative NES: low expression drives risk.
        diff = p_exp - p_run
        idx = np.argmax(diff)
        threshold_val = expr_vals[idx]
        rae_samples = set(df_sorted.iloc[idx:].index)
        
    return threshold_val, rae_samples


def main():
    parser = argparse.ArgumentParser(description="ROSMAP Funnel Tier 3: Leading Edge & RAE Discretization")
    parser.add_argument("-r", "--ranks", required=True, 
                        help="Path to preprocessed continuous ranks/expression CSV (samples x genes)")
    parser.add_argument("-b", "--bools", required=True, 
                        help="Path to preprocessed boolean attributes CSV (samples x attributes)")
    parser.add_argument("-s", "--sig-results", required=True, 
                        help="Path to significance results CSV file or a directory containing them")
    parser.add_argument("-o", "--outdir", default="results/rae_analysis_strict", 
                        help="Output directory to save binary matrices and summaries")
    parser.add_argument("-p", "--p-cutoff", type=float, default=0.01, 
                        help="BH-adjusted p-value significance cutoff for genes (default: 0.01)")
    args = parser.parse_args()

    print("======================================================================")
    print("ROSMAP Funnel Tier 3: GSEA-Style Leading Edge & RAE Builder (v3)")
    print("======================================================================")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. LOAD DATASETS
    print("\n[STEP 1] Loading and Normalizing Cohort Datasets...")
    try:
        print(f" -> Loading continuous expression: {args.ranks}")
        ranks_df = pd.read_csv(args.ranks, index_col=0)
        ranks_df.index = ranks_df.index.map(clean_id_to_string)
        print(f"    * Loaded shape: {ranks_df.shape} (patients x genes)")
        
        print(f" -> Loading boolean attributes: {args.bools}")
        bools_df = pd.read_csv(args.bools, index_col=0)
        bools_df.index = bools_df.index.map(clean_id_to_string)
        print(f"    * Loaded shape: {bools_df.shape} (patients x attributes)")
        
    except Exception as e:
        print(f"Error loading rank/bool files: {e}")
        sys.exit(1)
        
    # Align common samples across files
    common_samples = sorted(list(set(ranks_df.index).intersection(set(bools_df.index))))
    print(f" -> Aligned {len(common_samples)} common patients with matching genomic and clinical metadata.")
    if len(common_samples) == 0:
        print("Error: Sample IDs do not overlap! Please verify ID formats.")
        sys.exit(1)
        
    ranks_aligned = ranks_df.loc[common_samples]
    bools_aligned = bools_df.loc[common_samples]
    
    # 2. DISCOVER AND LOAD RESULTS
    print("\n[STEP 2] Discovering and Loading AREA Significance Results...")
    input_files = []
    if os.path.isdir(args.sig_results):
        print(f" -> Scanning directory: {args.sig_results}")
        for f in os.listdir(args.sig_results):
            if f.endswith('.csv') and not any(p in f.lower() for p in ["rae", "pathway", "gsea"]):
                input_files.append(os.path.join(args.sig_results, f))
    else:
        if os.path.exists(args.sig_results):
            input_files.append(args.sig_results)
            
    if not input_files:
        print(f"Error: No valid results files found at: {args.sig_results}")
        sys.exit(1)
        
    print(f" -> Found {len(input_files)} results CSV file(s) to process.")
    
    # 3. LEADING EDGE PROCESSING LOOP
    print(f"\n[STEP 3] Running GSEA-style Leading Edge inflection calculations (FDR < {args.p_cutoff})...")
    summary_records = []
    
    # Standard translation mapping for Response Archetypes (clean color codes matched)
    archetype_labels = {
        "co-progressive driver": "Shared Response",
        "dosage accumulator (weighted)": "Dosage-like Response",
        "categorical trigger (regular)": "Threshold-like Response",
        "state-transition trigger (regular)": "Threshold-like Response"
    }
    
    for file_path in input_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing file: {filename}")
        try:
            res_df = pd.read_csv(file_path)
        except Exception as e:
            print(f" -> Error reading {filename}: {e}. Skipping.")
            continue
            
        # Clean and find clinical attribute and match to boolean column
        clean_attr = clean_attribute_name(filename)
        bool_col_target = map_filename_to_bool_column(filename, bools_aligned.columns)
        
        if not bool_col_target:
            print(f"    * Warning: Could not map filename '{filename}' to any column in bools matrix. Skipping.")
            continue
            
        # Standardize column names
        col_lower_map = {col.lower().replace('_', '').replace('-', ''): col for col in res_df.columns}
        gene_col = col_lower_map.get('gene', col_lower_map.get('feature', None))
        if not gene_col:
            for col in res_df.columns:
                if col.lower() in ['gene_symbol', 'symbol', 'genes', 'id', 'feature', 'gene']:
                    gene_col = col
                    break
            if not gene_col:
                print(f"    * Warning: Could not locate gene column. Skipping.")
                continue
                
        fdr_cols = [c for c in res_df.columns if "fdr" in c.lower() or "padj" in c.lower()]
        nes_cols = [c for c in res_df.columns if "nes" in c.lower()]
        
        if not fdr_cols:
            p_cols = [c for c in res_df.columns if "pvalue" in c.lower() or "pval" in c.lower()]
            if p_cols:
                fdr_cols = [p_cols[0]]
            else:
                print(f"    * Warning: No significance (FDR/PValue) columns found. Skipping.")
                continue

        # Split and process Regular vs Weighted runs inside this attribute file
        for fdr_col in fdr_cols:
            col_lbl_lower = fdr_col.lower()
            
            # Match FDR to corresponding NES column
            nes_col = None
            model_suffix = ""
            if "regular" in col_lbl_lower:
                model_suffix = "Regular"
                matching_nes = [c for c in nes_cols if "regular" in c.lower()]
                if matching_nes:
                    nes_col = matching_nes[0]
            elif "weighted" in col_lbl_lower:
                model_suffix = "Weighted"
                matching_nes = [c for c in nes_cols if "weighted" in c.lower()]
                if matching_nes:
                    nes_col = matching_nes[0]
            
            if not nes_col:
                if nes_cols:
                    nes_col = nes_cols[0]
                else:
                    print(f"    * Warning: No matching NES column found for significance column '{fdr_col}'. Skipping.")
                    continue
            
            # Filter significant genes
            sig_group = res_df[res_df[fdr_col] < args.p_cutoff].copy()
            if sig_group.empty:
                continue
                
            model_clean_attr = f"{clean_attr} {model_suffix}".strip()
            print(f"   * Model: '{model_clean_attr}' (Binarized via: '{bool_col_target}') -> Found {len(sig_group)} significant genes.")
            
            sig_genes_aligned = [g for g in sig_group[gene_col].unique() if g in ranks_aligned.columns]
            if not sig_genes_aligned:
                print("     -> No significant genes present in ranks file. Skipping matrix creation.")
                continue
                
            rae_matrix = pd.DataFrame(0, index=common_samples, columns=sig_genes_aligned)
            non_nan_mask = bools_aligned[bool_col_target].dropna().index
            
            for idx, row in sig_group.iterrows():
                gene = row[gene_col]
                if gene not in sig_genes_aligned:
                    continue
                nes = row[nes_col]
                p_val_bh = row[fdr_col]
                
                # Dynamic archetype mapping (Tier 1 color coded matched)
                raw_arch = str(row.get("Classification", "Shared Response")).lower().strip()
                clean_arch = archetype_labels.get(raw_arch, "Shared Response")
                
                # Calculate leading-edge threshold
                thresh, rae_samples = calculate_leading_edge(
                    ranks_aligned[gene],
                    bools_aligned[bool_col_target],
                    nes
                )
                
                if pd.isna(thresh):
                    continue
                    
                # Discretize expression
                rae_matrix.loc[list(rae_samples), gene] = 1
                nan_samples = set(common_samples) - set(non_nan_mask)
                rae_matrix.loc[list(nan_samples), gene] = np.nan
                
                # Compute prevalence and risk ratios
                aligned_df = pd.DataFrame({'rae': rae_matrix[gene], 'label': bools_aligned[bool_col_target]}).dropna()
                prev_inside = aligned_df[aligned_df['rae'] == 1]['label'].mean()
                prev_outside = aligned_df[aligned_df['rae'] == 0]['label'].mean()
                risk_ratio = prev_inside / prev_outside if prev_outside > 0 else np.inf
                
                summary_records.append({
                    "Attribute": model_clean_attr,
                    "Gene": gene,
                    "NES": nes,
                    "p_value_BH": p_val_bh,
                    "Response_Archetype": clean_arch,
                    "RAE_Direction": "High-Expression Risk" if nes >= 0 else "Low-Expression Risk",
                    "RAE_Threshold_VST": thresh,
                    "RAE_Sample_Count": len(rae_samples),
                    "Prevalence_Inside_RAE": prev_inside,
                    "Prevalence_Outside_RAE": prev_outside,
                    "Risk_Ratio": risk_ratio
                })
                
            # Export matrix
            rae_matrix = rae_matrix.dropna(how='all')
            matrix_filename = f"rae_matrix_{model_clean_attr.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
            matrix_path = os.path.join(args.outdir, matrix_filename)
            rae_matrix.to_csv(matrix_path)
            print(f"      -> Saved binary RAE matrix to: {matrix_path} (Shape: {rae_matrix.shape})")

    # 4. EXPORT SUMMARY TABLE
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_cols = ["Attribute", "Gene", "NES", "p_value_BH", "Response_Archetype", "RAE_Direction", "RAE_Threshold_VST", "RAE_Sample_Count", "Prevalence_Inside_RAE", "Prevalence_Outside_RAE", "Risk_Ratio"]
        summary_df = summary_df[summary_cols]
        summary_path = os.path.join(args.outdir, "rae_thresholds_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[STEP 4] Saved comprehensive RAE thresholds summary to: {summary_path}")
    else:
        print("\n[STEP 4] Error: No significant associations met significance threshold across attributes. Summary not saved.")

    print("\n======================================================================")
    print("Tier 3 RAE Pipeline Completed Successfully!")
    print("======================================================================")


if __name__ == '__main__':
    main()

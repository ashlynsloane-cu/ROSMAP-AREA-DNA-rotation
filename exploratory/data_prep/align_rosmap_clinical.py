#!/usr/bin/env python3
"""
align_rosmap_clinical.py
======================================================================
ROSMAP Clinical Integration & Double-Hop ID Alignment Pipeline
----------------------------------------------------------------------
This script bridges your de-identified local clinical Excel files:
  1. data/dataset_1731_cross-sectional_07-30-2026.xlsx
  2. data/dataset_1731_long_07-30-2026.xlsx

With your active genomic metadata:
  - data/ROSMAP_metadata_merged_AREA.csv

By leveraging ROSMAP_clinical.csv as the ID translation bridge!
======================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

def clean_id_to_string(val):
    """
    Standardizes participant IDs to string, stripping any 'R' prefix or decimals
    to guarantee a seamless match between different files.
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

def discover_column(df, candidates, name_for_log):
    """
    Looks for candidate column names in a DataFrame (case-insensitive).
    """
    for col in df.columns:
        if col.lower() in [c.lower() for candidates_list in candidates for c in (candidates_list if isinstance(candidates_list, list) else [candidates_list])]:
            print(f"  -> Discovered {name_for_log} column: '{col}'")
            return col
    return None

def main():
    print("======================================================================")
    print("Starting ROSMAP Clinical Alignment Pipeline (Double-Hop ID Bridge)")
    print("======================================================================")

    # 1. SETUP PATHS
    cross_path = "data/dataset_1731_cross-sectional_07-30-2026.xlsx"
    long_path = "data/dataset_1731_long_07-30-2026.xlsx"
    bridge_path = "ROSMAP_clinical.csv"
    meta_path = "data/ROSMAP_metadata_merged_AREA.csv"

    # Fallback search for bridge_path in data/ if not in root
    if not os.path.exists(bridge_path) and os.path.exists("data/" + bridge_path):
        bridge_path = "data/" + bridge_path

    # Check for files
    missing_files = []
    for path in [cross_path, long_path, bridge_path, meta_path]:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        print("\nError: The following required files were not found in your directory:")
        for mf in missing_files:
            print(f"  * '{mf}'")
        print("\nPlease ensure you have placed your files in the correct folders before running.")
        sys.exit(1)

    # 2. LOAD DATA
    print(f"Loading cross-sectional data from: {cross_path}...")
    cross_df = pd.read_excel(cross_path)
    print(f"Loading longitudinal data from: {long_path}...")
    long_df = pd.read_excel(long_path)
    print(f"Loading ID Bridge file from: {bridge_path}...")
    bridge_df = pd.read_csv(bridge_path)
    print(f"Loading active genomic metadata from: {meta_path}...")
    meta_df = pd.read_csv(meta_path)

    # 3. DISCOVER ID COLUMNS
    cross_id_col = discover_column(cross_df, [['projid', 'individual_id', 'subject_id']], "Cross-sectional Participant ID")
    long_id_col = discover_column(long_df, [['projid', 'individual_id', 'subject_id']], "Longitudinal Participant ID")
    bridge_projid_col = discover_column(bridge_df, [['projid']], "Bridge Participant projid")
    bridge_indid_col = discover_column(bridge_df, [['individualID', 'individual_id']], "Bridge Genomic individualID")
    meta_indid_col = discover_column(meta_df, [['individual_id', 'individualID']], "Genomic Metadata individual_id")

    if not all([cross_id_col, long_id_col, bridge_projid_col, bridge_indid_col, meta_indid_col]):
        print("\nError: Could not identify matching participant ID columns in one or more files.")
        sys.exit(1)

    # Clean IDs to normalized integer strings for perfect matching
    cross_df['clean_projid'] = cross_df[cross_id_col].apply(clean_id_to_string)
    long_df['clean_projid'] = long_df[long_id_col].apply(clean_id_to_string)
    bridge_df['clean_projid'] = bridge_df[bridge_projid_col].apply(clean_id_to_string)
    
    # Keep bridge translation
    id_map = bridge_df[['clean_projid', bridge_indid_col]].dropna()
    id_map = id_map.rename(columns={bridge_indid_col: 'individual_id_mapped'})
    id_map['clean_ind_id'] = id_map['individual_id_mapped'].apply(clean_id_to_string)
    # Deduplicate to prevent fan-out
    id_map = id_map.drop_duplicates(subset=['clean_projid'])

    # 4. PROCESS LONGITUDINAL TRAITS
    print("\nProcessing longitudinal variables...")
    visit_col = discover_column(long_df, [['fu_year', 'visit', 'cycle', 'age_at_visit']], "Longitudinal Visit/Time")
    if visit_col:
        long_df = long_df.sort_values(by=['clean_projid', visit_col])
    else:
        print("  -> Warning: Time/visit column not found. Processing based on sheet row order.")

    cog_col = discover_column(long_df, [['cogn_global', 'cogng_global', 'global_cognition']], "Global Cognition")
    mmse_col = discover_column(long_df, [['cts_estmmse30', 'mmse', 'mmse_lv']], "MMSE Score")
    bmi_col = discover_column(long_df, [['bmi', 'bmi_lv']], "Body Mass Index")

    stroke_col = discover_column(long_df, [['r_stroke', 'stroke_cum', 'stroke']], "Stroke History")
    diab_col = discover_column(long_df, [['diabetes_sr_rx', 'diab', 'diabetes']], "Diabetes History")
    hyper_col = discover_column(long_df, [['hypertension_cum', 'hyperten', 'hypertension']], "Hypertension History")

    # Longitudinal aggregation dictionary
    agg_dict = {}
    for col in [cog_col, mmse_col, bmi_col]:
        if col: agg_dict[col] = 'last'
    for col in [stroke_col, diab_col, hyper_col]:
        if col: agg_dict[col] = 'max'

    # Execute aggregation
    long_grouped = long_df.groupby('clean_projid').agg(agg_dict).reset_index()
    print(f"  -> Successfully aggregated longitudinal data for {len(long_grouped)} participants.")

    # 5. MERGE CLINICAL VARIABLES
    print("\nMerging clinical variables...")
    cross_df = cross_df.set_index('clean_projid')
    long_grouped = long_grouped.set_index('clean_projid')
    clinical_merged = cross_df.copy().join(long_grouped, how='left')

    # Now bridge clinical data onto standard genomics R-prefixed individual_ids
    print("\nBridging de-identified clinical IDs to genomic individual_ids using ROSMAP_clinical.csv...")
    clinical_merged = clinical_merged.join(id_map.set_index('clean_projid'), how='inner')
    print(f"  * Successfully bridged {len(clinical_merged)} clinical records to genomic IDs.")

    # 6. INTEGRATE INTO GENOMIC METADATA
    print("\nAligning with active genomic metadata...")
    meta_df['clean_ind_id'] = meta_df[meta_indid_col].apply(clean_id_to_string)
    print(f"  * Baseline genomic samples: {len(meta_df)}")

    # Columns we want to pull
    variables_to_pull = []
    for col in ['msex', 'apoe_genotype', 'braaksc', 'ceradsc', 'amyloid', 'age_death', 'age_first_ad_dx', 'cogdx']:
        discovered = discover_column(clinical_merged, [[col]], f"Cross-sectional target '{col}'")
        if discovered: 
            variables_to_pull.append(discovered)
            
    for col in [cog_col, mmse_col, bmi_col, stroke_col, diab_col, hyper_col]:
        if col and col in clinical_merged.columns: 
            variables_to_pull.append(col)

    # Keep mapped ID for joining
    clinical_subset = clinical_merged[variables_to_pull + ['clean_ind_id']]
    # Deduplicate in case of duplicate keys
    clinical_subset = clinical_subset.drop_duplicates(subset=['clean_ind_id']).set_index('clean_ind_id')

    # Drop existing duplicate columns in genomic metadata to prevent suffix collisions
    cols_to_drop = [c for c in clinical_subset.columns if c in meta_df.columns and c != 'clean_ind_id']
    if cols_to_drop:
        print(f"  * Updating existing columns in metadata: {', '.join(cols_to_drop)}")
        meta_df = meta_df.drop(columns=cols_to_drop)

    # Merge bridged clinical data onto metadata using normalized IDs
    final_metadata = meta_df.merge(clinical_subset, left_on='clean_ind_id', right_index=True, how='left')

    # Drop temporary clean ID column
    final_metadata = final_metadata.drop(columns=['clean_ind_id'])

    # Save output
    print(f"\nSaving final merged dataset back to: {meta_path}...")
    final_metadata.to_csv(meta_path, index=False)

    print("======================================================================")
    print("ROSMAP Double-Hop Clinical Integration Completed Successfully!")
    print(f"Final Metadata Matrix Shape: {final_metadata.shape} (samples x variables)")
    # Print non-NaN counts for validation
    print("----------------------------------------------------------------------")
    print("Validation: Number of populated (non-blank) patients per variable:")
    for col in clinical_subset.columns:
        populated = final_metadata[col].notna().sum()
        print(f"  * {col:<20}: {populated:>4} / {len(final_metadata)} patients populated")
    print("======================================================================")

if __name__ == "__main__":
    main()


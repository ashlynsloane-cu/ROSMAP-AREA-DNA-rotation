#!/usr/bin/env python3
"""
align_rosmap_clinical_v3.py
======================================================================
ROSMAP Clinical Excel Alignment & Longitudinal Aggregation Script (v3)
----------------------------------------------------------------------
This script robustly integrates your offline clinical Excel files:
  1. data/dataset_1731_cross-sectional_07-30-2026.xlsx
  2. data/dataset_1731_long_07-30-2026.xlsx

With your active genomic metadata:
  - data/ROSMAP_metadata_merged_AREA.csv

UPDATED FOR DATASET 1731 SCHEMAS:
This version directly supports the exact column headers of your 1731 dataset:
  - Global Cognition: 'cogn_global'
  - MMSE Score: 'cts_estmmse30'
  - Stroke: 'r_stroke' / 'stroke_cum'
  - Diabetes: 'diabetes_sr_rx'
  - Hypertension: 'hypertension_cum'
  - Age at First AD Diagnosis: 'age_first_ad_dx' (pulled from cross-sectional)
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
    print("Starting ROSMAP Clinical Alignment Pipeline (v3 - Dataset 1731)")
    print("======================================================================")

    # 1. SETUP PATHS (Adjusted for your 'data/' folder)
    cross_path = "data/dataset_1731_cross-sectional_07-30-2026.xlsx"
    long_path = "data/dataset_1731_long_07-30-2026.xlsx"
    meta_path = "data/ROSMAP_metadata_merged_AREA.csv"

    # Check for files
    missing_files = []
    for path in [cross_path, long_path, meta_path]:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        print("\nError: The following required files were not found in your directory:")
        for mf in missing_files:
            print(f"  * '{mf}'")
        print("\nPlease ensure you have copied the Excel files directly into your")
        print("ROSMAP-AREA-DNA-rotation/data/ directory before running this script.")
        sys.exit(1)

    # 2. LOAD DATA
    print(f"Loading cross-sectional data from: {cross_path}...")
    cross_df = pd.read_excel(cross_path)
    print(f"Loading longitudinal data from: {long_path}...")
    long_df = pd.read_excel(long_path)
    print(f"Loading active genomic metadata from: {meta_path}...")
    meta_df = pd.read_csv(meta_path)

    # 3. DISCOVER ID COLUMNS
    cross_id_col = discover_column(cross_df, [['projid', 'individual_id', 'subject_id']], "Cross-sectional Participant ID")
    long_id_col = discover_column(long_df, [['projid', 'individual_id', 'subject_id']], "Longitudinal Participant ID")
    meta_id_col = discover_column(meta_df, [['individual_id', 'projid', 'subject_id']], "Genomic Metadata Participant ID")

    if not cross_id_col or not long_id_col or not meta_id_col:
        print("\nError: Could not identify matching participant ID columns.")
        sys.exit(1)

    # Normalize ID columns to strings for perfect matching
    cross_df['clean_id'] = cross_df[cross_id_col].apply(clean_id_to_string)
    long_df['clean_id'] = long_df[long_id_col].apply(clean_id_to_string)
    meta_df['clean_id'] = meta_df[meta_id_col].apply(clean_id_to_string)

    # 4. PROCESS LONGITUDINAL TRAITS
    print("\nProcessing longitudinal variables...")
    
    # Sort longitudinal data by participant and visit to ensure proper timeline
    visit_col = discover_column(long_df, [['fu_year', 'visit', 'cycle', 'age_at_visit']], "Longitudinal Visit/Time")
    if visit_col:
        long_df = long_df.sort_values(by=['clean_id', visit_col])
    else:
        print("  -> Warning: Time/visit column not found. Processing based on sheet row order.")

    # A. Last Valid (_lv) continuous scores (average cognitive z-score, mmse, bmi)
    # Mapping exact columns from your dataset 1731 list!
    cog_col = discover_column(long_df, [['cogn_global', 'cogng_global', 'global_cognition']], "Global Cognition")
    mmse_col = discover_column(long_df, [['cts_estmmse30', 'mmse', 'mmse_lv']], "MMSE Score")
    bmi_col = discover_column(long_df, [['bmi', 'bmi_lv']], "Body Mass Index")

    # B. Ever-reported binary history columns (diabetes, stroke, hypertension)
    stroke_col = discover_column(long_df, [['r_stroke', 'stroke_cum', 'stroke']], "Stroke History")
    diab_col = discover_column(long_df, [['diabetes_sr_rx', 'diab', 'diabetes']], "Diabetes History")
    hyper_col = discover_column(long_df, [['hypertension_cum', 'hyperten', 'hypertension']], "Hypertension History")

    # Longitudinal aggregation dictionary
    agg_dict = {}
    
    # Last Valid rule (takes the last non-null entry for that patient)
    for col in [cog_col, mmse_col, bmi_col]:
        if col:
            agg_dict[col] = 'last'

    # Ever-reported rule (any yes/1 in history means yes/1)
    for col in [stroke_col, diab_col, hyper_col]:
        if col:
            agg_dict[col] = 'max'

    # Execute aggregation
    long_grouped = long_df.groupby('clean_id').agg(agg_dict).reset_index()
    print(f"  -> Successfully aggregated longitudinal data for {len(long_grouped)} participants.")

    # 5. MERGE CLINICAL VARIABLES
    print("\nMerging cross-sectional and longitudinal clinical variables...")
    # Set clean_id as index for clinical sheets
    cross_df = cross_df.set_index('clean_id')
    long_grouped = long_grouped.set_index('clean_id')

    # Start with cross-sectional as base
    clinical_merged = cross_df.copy()

    # Join longitudinal features
    clinical_merged = clinical_merged.join(long_grouped, how='left')

    # 6. INTEGRATE INTO METADATA
    print("\nAligning with active genomic metadata...")
    print(f"  * Baseline genomic samples: {len(meta_df)}")

    # Columns we want to extract from clinical merge and add to active metadata
    variables_to_pull = []
    
    # Add cross-sectional columns if found (added age_first_ad_dx and cogdx)
    for col in ['msex', 'apoe_genotype', 'braaksc', 'ceradsc', 'amyloid', 'age_death', 'age_first_ad_dx', 'cogdx']:
        discovered = discover_column(clinical_merged, [[col]], f"Cross-sectional target '{col}'")
        if discovered:
            variables_to_pull.append(discovered)

    # Add longitudinal columns if found
    for col in [cog_col, mmse_col, bmi_col, stroke_col, diab_col, hyper_col]:
        if col and col in clinical_merged.columns:
            variables_to_pull.append(col)

    # Subset clinical merged to only target variables
    clinical_subset = clinical_merged[variables_to_pull]

    # Drop existing duplicates of target columns in metadata to prevent suffix naming like '_x' or '_y'
    cols_to_drop = [c for c in clinical_subset.columns if c in meta_df.columns and c != 'clean_id']
    if cols_to_drop:
        print(f"  * Updating existing columns in metadata: {', '.join(cols_to_drop)}")
        meta_df = meta_df.drop(columns=cols_to_drop)

    # Merge clinical data onto metadata using normalized IDs
    final_metadata = meta_df.merge(clinical_subset, left_on='clean_id', right_index=True, how='left')

    # Drop the temporary clean ID column
    final_metadata = final_metadata.drop(columns=['clean_id'])

    # Save output
    print(f"\nSaving final merged dataset back to: {meta_path}...")
    final_metadata.to_csv(meta_path, index=False)

    print("======================================================================")
    print("ROSMAP Clinical Integration Completed Successfully!")
    print(f"Final Metadata Matrix Shape: {final_metadata.shape} (samples x variables)")
    print("======================================================================")

if __name__ == "__main__":
    main()

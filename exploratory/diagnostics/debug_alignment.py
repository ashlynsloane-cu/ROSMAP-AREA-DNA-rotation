#!/usr/bin/env python3
"""
debug_alignment.py
======================================================================
ROSMAP ID Alignment & Merger Diagnostic Tool
----------------------------------------------------------------------
This script checks the exact format of participant IDs across your:
  1. data/dataset_1731_cross-sectional_07-30-2026.xlsx
  2. data/dataset_1731_long_07-30-2026.xlsx
  3. data/ROSMAP_metadata_merged_AREA.csv

It prints the raw and cleaned heads of each ID column and calculates
the exact overlap (intersection) to pinpoint why merged columns are blank.
======================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

def clean_id_to_string(val):
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.endswith('.0'):
        val_str = val_str[:-2]
    # Strip 'R' or 'r' prefix if present
    if val_str.upper().startswith('R'):
        val_str = val_str[1:]
    return val_str.strip()

def main():
    print("==========================================================")
    print("ROSMAP ID Alignment Diagnostic Tool")
    print("==========================================================")

    cross_path = "data/dataset_1731_cross-sectional_07-30-2026.xlsx"
    long_path = "data/dataset_1731_long_07-30-2026.xlsx"
    meta_path = "data/ROSMAP_metadata_merged_AREA.csv"

    # Verify files exist
    for p in [cross_path, long_path, meta_path]:
        if not os.path.exists(p):
            print(f"Error: File '{p}' not found. Please run this from your root ROSMAP-AREA-DNA-rotation directory.")
            sys.exit(1)

    print("Loading datasets...")
    cross_df = pd.read_excel(cross_path, nrows=10)
    long_df = pd.read_excel(long_path, nrows=10)
    meta_df = pd.read_csv(meta_path, nrows=10)

    # Let's also load full files just for unique set intersection count (nrows=None for unique ID sets)
    print("Checking unique ID intersection across full files...")
    cross_full = pd.read_excel(cross_path, usecols=[0]) # read only first column for speed
    long_full = pd.read_excel(long_path, usecols=[0])
    meta_full = pd.read_csv(meta_path, usecols=['individual_id'])

    # Get column names
    cross_id_col = cross_full.columns[0]
    long_id_col = long_full.columns[0]
    meta_id_col = 'individual_id'

    # Clean IDs
    meta_full['clean_id'] = meta_full[meta_id_col].apply(clean_id_to_string)
    cross_full['clean_id'] = cross_full[cross_id_col].apply(clean_id_to_string)
    long_full['clean_id'] = long_full[long_id_col].apply(clean_id_to_string)

    print("\n--- ID FORMAT EXAMPLES ---")
    print(f"Genomic Metadata Column ('{meta_id_col}'):")
    print("  * Raw examples    :", list(meta_df[meta_id_col].head(5)))
    print("  * Cleaned examples:", list(meta_full['clean_id'].head(5)))

    print(f"\nCross-Sectional Column ('{cross_id_col}'):")
    print("  * Raw examples    :", list(cross_df[cross_id_col].head(5)))
    print("  * Cleaned examples:", list(cross_full['clean_id'].head(5)))

    # Intersections
    meta_set = set(meta_full['clean_id'].dropna().astype(str))
    cross_set = set(cross_full['clean_id'].dropna().astype(str))
    long_set = set(long_full['clean_id'].dropna().astype(str))

    print("\n--- OVERLAP ANALYSIS ---")
    print(f"Total unique IDs in Genomic Metadata (CSV)  : {len(meta_set)}")
    print(f"Total unique IDs in Cross-sectional (Excel): {len(cross_set)}")
    print(f"Total unique IDs in Longitudinal (Excel)   : {len(long_set)}")
    
    overlap = meta_set.intersection(cross_set)
    print(f"Genomic Metadata ∩ Cross-sectional OVERLAP  : {len(overlap)} matches")
    
    if len(overlap) == 0:
        print("\n❌ CRITICAL WARNING: There are ZERO overlapping participant IDs between your genomics sheet and the clinical Excel sheets!")
        print("This means the 'individual_id' in your genomics CSV does not match the 'projid' in your clinical sheets.")
        print("Please check if one of the sheets uses a completely different ID schema (e.g. 'individual_id' vs. 'projid').")
    else:
        print(f"\n✅ SUCCESS: Found {len(overlap)} matching IDs!")
        print("If the columns are still blank, the merger logic may have been applied to an older cached copy of your metadata, or the columns were not aligned correctly.")

    print("==========================================================")

if __name__ == "__main__":
    main()

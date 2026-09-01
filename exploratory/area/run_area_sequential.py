#!/usr/bin/env python3
"""
Ultra-Optimized Sequential Runner for AREA (macOS Apple Silicon Deadlock Workaround).
Bypasses expensive pandas DataFrame merges and multi-threaded locks entirely.
Performs blistering-fast 1D NumPy argsort operations directly on the main thread.
Completes genome-wide runs of 20,000+ genes in seconds instead of hours.
"""

import os
import sys

# Force macOS linear algebra libraries to remain single-threaded
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["ACCELERATE_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

# Add the AREA source directory to Python's search path
sys.path.append("/Users/ashlynsloane/Developer/area-workspace/AREA")

try:
    import src.area.runner as runner
    import src.area.cli as cli
    from src.area.enrichment import permute_enrichment_scores, compute_enrichment_score, compute_nes_pvalue
except ModuleNotFoundError:
    print("Error: Could not locate the 'src' directory inside your AREA workspace.")
    print("Please ensure your AREA path is correct in the script.")
    sys.exit(1)

# --------------------------------------------------------------------------
# Ultra-Optimized Sequential Compute Function (Main Thread Only)
# --------------------------------------------------------------------------
def sequential_compute_pvalues_optimized(out_dir, plan_file, join_column, rank_df, bool_df, 
                                         keep_samples, use_gpu=False, n_threads=1, 
                                         verbose=False):
    """
    Blistering-fast sequential calculation bypassing pandas column merges entirely.
    """
    plan_path = os.path.join(out_dir, plan_file)
    plan_df = pd.read_csv(plan_path)
    plan_df = plan_df[plan_df["plan"] == "run_area"]
 
    bool_columns = plan_df["bool_column"].unique()
    print(f"\n==============================================================")
    print(f"Computing p-values for {len(bool_columns)} attributes (ULTRA-OPTIMIZED MAIN-THREAD)")
    print(f"==============================================================\n")
 
    # 1. Clean and index inputs for rapid alignment
    bool_df_clean = bool_df.dropna(subset=[join_column])
    rank_df_clean = rank_df.dropna(subset=[join_column])
    
    bool_df_indexed = bool_df_clean.set_index(join_column)
    rank_df_indexed = rank_df_clean.set_index(join_column)
    
    common_samples = bool_df_indexed.index.intersection(rank_df_indexed.index)
    if len(keep_samples) > 0:
        common_samples = common_samples.intersection(keep_samples)
        
    common_samples = sorted(list(common_samples))
    
    # 2. Shuffle sample order once to randomize ties globally
    import random
    random.seed(42)
    random.shuffle(common_samples)
    
    # 3. Subset aligned dataframes once
    bool_df_aligned = bool_df_indexed.loc[common_samples]
    rank_df_aligned = rank_df_indexed.loc[common_samples]
    
    results = []
    
    for i, bool_col in enumerate(bool_columns, 1):
        print(f"[{i}/{len(bool_columns)}] Processing attribute: '{bool_col}'...")
        
        # Get target rank columns (genes) for this attribute
        rank_cols_for_attr = plan_df.loc[
            plan_df["bool_column"] == bool_col, "rank_column"
        ].tolist()
        
        # Keep only genes present in rank_df
        rank_cols_for_attr = [col for col in rank_cols_for_attr if col in rank_df_aligned.columns]
        
        if not rank_cols_for_attr:
            print(f"  -> No valid genes found in ranks file for '{bool_col}'. Skipping.")
            continue
            
        # Get binary vector for this attribute, dropping NaNs to be mathematically clean
        attr_series = bool_df_aligned[bool_col]
        valid_samples = attr_series.dropna().index
        
        if len(valid_samples) == 0:
            print(f"  -> No valid non-NaN samples for '{bool_col}'. Skipping.")
            continue
            
        binary_vector = attr_series.loc[valid_samples].values
        binary_vector = np.array([1 if v > 0 else 0 for v in binary_vector])
        
        # Build empirical null distribution once per attribute
        print(f"  -> Building null distribution (1,000 permutations) on {len(valid_samples)} samples...")
        null_scores = permute_enrichment_scores(
            binary_vector, n_permutations=1000, xp=np, verbose=False
        )
        
        # Extract the ranks matrix as a numpy array for blistering speed
        print(f"  -> Extracting matrix and scoring {len(rank_cols_for_attr)} genes...")
        ranks_matrix = rank_df_aligned.loc[valid_samples, rank_cols_for_attr].values
        
        attr_results = []
        
        # Loop over each gene index and run argsort sequentially
        for g_idx, rank_col in enumerate(rank_cols_for_attr):
            ranks = ranks_matrix[:, g_idx]
            
            # Stable mergesort preserves randomized tie-breaking order from the shuffle
            sorted_indices = np.argsort(ranks, kind='mergesort')
            binary_sorted = binary_vector[sorted_indices]
            
            # Compute observed enrichment score and NES/p-value
            observed_es, *_ = compute_enrichment_score(binary_sorted, xp=np)
            nes, pvalue = compute_nes_pvalue(observed_es, null_scores, use_gpu=False, xp=np)
            
            attr_results.append({
                "bool_column": bool_col,
                "rank_column": rank_col,
                "nes": nes,
                "pvalue": pvalue,
            })
            
        results.append(pd.DataFrame(attr_results))
        print(f"[{i}/{len(bool_columns)}] Finished processing '{bool_col}'.\n")
 
    combined = pd.concat(results, ignore_index=True)
    raw_path = os.path.join(out_dir, plan_file + ".raw_pvalues.csv")
    combined.to_csv(raw_path, index=False)
    print("Done computing p-values sequentially (Ultra-Optimized!).")

# --------------------------------------------------------------------------
# Monkey-patch BOTH namespaces to ensure the thread pool is bypassed completely
# --------------------------------------------------------------------------
runner.compute_pvalues = sequential_compute_pvalues_optimized
cli.compute_pvalues = sequential_compute_pvalues_optimized

# --------------------------------------------------------------------------
# Execute CLI main()
# --------------------------------------------------------------------------
from src.area.cli import main

if __name__ == "__main__":
    sys.exit(main())

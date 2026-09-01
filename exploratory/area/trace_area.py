import sys
import os
import pandas as pd
import numpy as np
import time

# Add the AREA source directory to Python's search path
sys.path.append("/Users/ashlynsloane/Developer/area-workspace/AREA")

try:
    import src.area.runner as runner
    import src.area.enrichment as enrichment
    from src.area.runner import _run_single_bool_column
except ModuleNotFoundError:
    print("Error: Could not locate the 'src' directory inside your AREA workspace.")
    sys.exit(1)

def run_diagnostic_trace():
    print("==========================================================")
    print("Starting Detailed Line-by-Line Trace Diagnostic")
    print("==========================================================")
    
    # Load raw inputs to see their sizes
    print("Loading test data frames...")
    bool_df = pd.read_csv("results/rosmap_area_bools.csv")
    rank_df = pd.read_csv("results/rosmap_area_ranks.csv", index_col=0).reset_index()
    
    print(f"Loaded boolean attributes. Shape: {bool_df.shape}")
    print(f"Loaded ranks. Shape: {rank_df.shape}")
    
    # We will test NCI_vs_Rest, but on a TINY subset of genes (e.g., top 5 genes)
    # and a TINY number of permutations (e.g., 5 permutations)
    bool_col = "NCI_vs_Rest"
    rank_columns = rank_df.columns[1:6].tolist() # Just 5 genes
    join_column = "sample_id"
    keep_samples = []
    use_gpu = False
    n_permutations = 5 # Just 5 permutations
    
    print(f"\nRunning test for attribute '{bool_col}' against 5 genes with {n_permutations} permutations...")
    
    t0 = time.time()
    
    # Step-by-step emulation of _run_single_bool_column
    print("\n[Step 1] Loading backend...")
    from src.area.backend import load_array_backend
    xp = load_array_backend(use_gpu)
    print(f"  -> Backend loaded: {xp.__name__}")
    
    print("\n[Step 2] Merging DataFrames...")
    cols_bool = [bool_col, join_column]
    cols_rank = rank_columns + [join_column]
    merged = bool_df[cols_bool].merge(rank_df[cols_rank], on=join_column, how="inner")
    print(f"  -> Merged DataFrame shape: {merged.shape}")
    
    print("\n[Step 3] Shuffling row order...")
    merged = merged.sample(frac=1).reset_index(drop=True)
    print("  -> Shuffled.")
    
    print("\n[Step 4] Extracting binary vector...")
    binary_vector = merged[bool_col].tolist()
    print(f"  -> Binary vector length: {len(binary_vector)}, sum of 1s: {sum(binary_vector)}")
    
    print("\n[Step 5] Entering permute_enrichment_scores loop...")
    xp.random.seed(42)
    binary_array = xp.array(binary_vector)
    
    null_scores = []
    for p in range(n_permutations):
        print(f"  -> Permutation {p+1}/{n_permutations}...")
        shuffled = xp.random.permutation(binary_array)
        
        print("     -> Computing enrichment score...")
        t_es = time.time()
        es, *_ = enrichment.compute_enrichment_score(shuffled, xp=xp, verbose=True)
        print(f"     -> Enrichment score computed: {es:.4f} (took {time.time() - t_es:.6f}s)")
        
        null_scores.append(es)
        
    print(f"  -> Finished permutations loop. Null scores: {null_scores}")
    
    print("\n[Step 6] Scoring individual genes...")
    results = []
    for rank_col in rank_columns:
        print(f"  -> Scoring gene '{rank_col}'...")
        sorted_df = merged.sort_values(rank_col)
        binary_sorted = sorted_df[bool_col].tolist()
        
        observed_es, *_ = enrichment.compute_enrichment_score(binary_sorted, xp=xp)
        print(f"     -> Observed ES: {observed_es:.4f}")
        
        nes, pvalue = enrichment.compute_nes_pvalue(observed_es, null_scores, use_gpu=use_gpu, xp=xp)
        print(f"     -> NES: {nes:.4f}, p-value: {pvalue}")
        
        results.append({
            "bool_column": bool_col,
            "rank_column": rank_col,
            "nes": nes,
            "pvalue": pvalue,
        })
        
    print("\n==========================================================")
    print(f"SUCCESS! Diagnostic run completed in {time.time() - t0:.2f} seconds.")
    print("==========================================================")

if __name__ == "__main__":
    run_diagnostic_trace()

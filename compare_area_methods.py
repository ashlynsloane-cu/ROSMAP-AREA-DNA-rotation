#!/usr/bin/env python3
"""
compare_area_methods_v3.py
======================================================================
ROSMAP Clinical-Genomic Methodological Validation & Benchmarking Tool (v3)
----------------------------------------------------------------------
This script performs a parallel, side-by-side comparison of:
  - Step 1: Regular AREA (using your 3-way multi-state categorical binarizations)
  - Step 2: Weighted AREA (using raw, continuous/ordinal clinical scores)
  - Step 3: Model Selection Index (MSI) to automatically classify genes

This v3 version corrects a mathematical label-swap in the classification logic:
  - Negative MSI (Weighted P < Regular P) -> Dosage Accumulator (Weighted)
  - Positive MSI (Regular P < Weighted P) -> Categorical Trigger (Regular)
======================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm

# Standard Trapezoidal integration helper to maintain parity with AREA backend
def trapz(y, x=None, dx=1.0):
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        d = np.diff(x)
    if x is None or d.ndim == 0:
        return 0.5 * (y[0] + y[-1] + 2 * np.sum(y[1:-1])) * d
    else:
        return 0.5 * np.sum((y[:-1] + y[1:]) * d)

def compute_area_score_regular(binary_vector):
    """
    Step 1: Standard AREA math.
    Step size is binary and normalized (1 / N_cases).
    """
    binary = np.array([1 if v > 0 else 0 for v in binary_vector])
    n = len(binary)
    total_hits = float(np.sum(binary))
    
    if total_hits == 0:
        return 0.0
        
    bin_width = 1.0 / n
    normalized_scores = np.multiply(np.divide(binary, total_hits), bin_width)
    cumulative_scores = np.cumsum(normalized_scores)
    
    trend = np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0)
    trend = np.multiply(trend, bin_width)
    
    enrichment_score = (trapz(cumulative_scores) - trapz(trend)) * 2
    return enrichment_score

def compute_area_score_weighted(weight_vector):
    """
    Step 2: Weighted AREA math.
    Step size is weighted by raw clinical scores and normalized by total cumulative weight.
    Guarantees the running sum finishes exactly at 1.0.
    """
    weights = np.array([float(w) if pd.notna(w) else 0.0 for w in weight_vector])
    n = len(weights)
    total_weight = float(np.sum(weights))
    
    if total_weight == 0:
        return 0.0
        
    bin_width = 1.0 / n
    normalized_scores = np.multiply(np.divide(weights, total_weight), bin_width)
    cumulative_scores = np.cumsum(normalized_scores)
    
    trend = np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0)
    trend = np.multiply(trend, bin_width)
    
    enrichment_score = (trapz(cumulative_scores) - trapz(trend)) * 2
    return enrichment_score

def compute_nes_and_pvalue(observed_es, null_scores):
    """
    Split-signed normalization to absorb skewness in the empirical null.
    """
    if observed_es > 0:
        subset = np.array([x for x in null_scores if x > 0])
        if len(subset) == 0:
            return 0.0, 1.0
        mu = np.mean(subset)
        sigma = np.std(subset)
        nes = -(observed_es / mu)
        pvalue = 1.0 - norm.cdf(observed_es, mu, sigma)
    else:
        subset = np.array([x for x in null_scores if x < 0])
        if len(subset) == 0:
            return 0.0, 1.0
        mu = np.mean(subset)
        sigma = np.std(subset)
        nes = observed_es / mu
        pvalue = norm.cdf(observed_es, mu, sigma)
        
    pvalue = np.clip(pvalue, 1e-15, 1.0)
    return nes, pvalue

def run_area_permutations(vector, is_weighted, n_permutations=1000):
    """
    Generates the empirical null distribution by shuffling labels/weights.
    """
    null_scores = []
    shuffled_vector = np.array(vector, copy=True)
    for _ in range(n_permutations):
        np.random.shuffle(shuffled_vector)
        if is_weighted:
            es = compute_area_score_weighted(shuffled_vector)
        else:
            es = compute_area_score_regular(shuffled_vector)
        null_scores.append(es)
    return null_scores

def calculate_msi(p_regular, p_weighted):
    """
    Step 3: Model Selection Index.
    MSI = log10(p_weighted / p_regular)
    """
    return np.log10(p_weighted / p_regular)

def classify_gene(msi):
    # CORRECTED CLASSIFICATION BOUNDARIES:
    # If msi < -2.0: p_weighted is at least 100x smaller (more significant) than p_regular.
    # Therefore, the gene tracks continuous severity better -> Dosage Accumulator (Weighted).
    # If msi > 2.0: p_regular is at least 100x smaller (more significant) than p_weighted.
    # Therefore, the gene tracks discrete states better -> Categorical Trigger (Regular).
    if msi < -2.0:
        return "Dosage Accumulator (Weighted)"
    elif msi > 2.0:
        return "Categorical Trigger (Regular)"
    else:
        return "Co-Progressive Driver"

def main():
    parser = argparse.ArgumentParser(description="ROSMAP Parallel AREA & Methodological Benchmark")
    parser.add_argument("-r", "--ranks", default="results/rosmap_area_ranks.csv", help="Path to preprocessed rank CSV")
    parser.add_argument("-c", "--clinical", default="data/ROSMAP_metadata_merged_AREA.csv", help="Path to ROSMAP merged metadata CSV")
    parser.add_argument("-p", "--permutations", type=int, default=1000, help="Number of permutations (default 1000)")
    parser.add_argument("-o", "--output", default="results/area_method_comparison.csv", help="Output comparison CSV file")
    
    args = parser.parse_args()
    
    print("======================================================================")
    print("Starting ROSMAP Parallel Method Comparison Pipeline (v3 - Corrected)")
    print("======================================================================")
    
    # 1. LOAD DATA
    if not os.path.exists(args.ranks):
        print(f"Error: Rank file '{args.ranks}' not found.")
        sys.exit(1)
    if not os.path.exists(args.clinical):
        print(f"Error: Clinical metadata '{args.clinical}' not found.")
        sys.exit(1)
        
    print(f"Loading continuous expression ranks from: {args.ranks}")
    ranks_df = pd.read_csv(args.ranks, index_col=0)
    print(f"Loading clinical metadata from: {args.clinical}")
    clinical_df = pd.read_csv(args.clinical, index_col=0)
    
    # 2. ALIGN SAMPLES
    common_samples = ranks_df.index.intersection(clinical_df.index)
    print(f"Aligned samples: {len(common_samples)} patients")
    if len(common_samples) == 0:
        print("Error: Sample IDs do not overlap!")
        sys.exit(1)
        
    ranks_aligned = ranks_df.loc[common_samples]
    clinical_aligned = clinical_df.loc[common_samples]
    
    # Dynamic column discovery for robust metadata schema mapping
    cognitive_col = None
    for col in ['diagnosis', 'cogdx', 'dcfdx']:
        if col in clinical_aligned.columns:
            cognitive_col = col
            break
            
    braak_col = None
    for col in ['braak', 'braaksc', 'braak_stage']:
        if col in clinical_aligned.columns:
            braak_col = col
            break
            
    if not cognitive_col or not braak_col:
        print(f"Error: Could not map cognitive column or Braak column. Available columns: {list(clinical_aligned.columns)}")
        sys.exit(1)
        
    print(f"  -> Discovered Cognitive Column: '{cognitive_col}'")
    print(f"  -> Discovered Braak Pathology Column: '{braak_col}'")
    
    # 3-Way categorical labels (Step 1)
    clinical_aligned['NCI_vs_Rest'] = clinical_aligned[cognitive_col].apply(lambda x: 1 if x == 1 else 0)
    clinical_aligned['MCI_vs_Rest'] = clinical_aligned[cognitive_col].apply(lambda x: 1 if x in [2, 3] else 0)
    clinical_aligned['AD_vs_Rest'] = clinical_aligned[cognitive_col].apply(lambda x: 1 if x in [4, 5] else 0)
    
    # Pathological variables (Braak stage 0 to 6)
    clinical_aligned['Braak_Continuous'] = pd.to_numeric(clinical_aligned[braak_col], errors='coerce').fillna(0.0)
    clinical_aligned['Braak_III_VI_vs_0_II'] = clinical_aligned['Braak_Continuous'].apply(lambda x: 1 if x >= 3 else 0)
    
    # Focus Genes
    candidate_genes = ["RCAN1", "SOD1", "MAPT", "APP", "APOE", "ADAMTS1", "SAMSN1"]
    active_genes = [g for g in candidate_genes if g in ranks_aligned.columns]
    
    if len(active_genes) == 0:
        print("Warning: None of the candidate genes found in ranks. Testing top 5 high-variance genes instead.")
        gene_vars = ranks_aligned.var().sort_values(ascending=False)
        active_genes = gene_vars.head(5).index.tolist()
        
    print(f"Benchmarking genes: {', '.join(active_genes)}")
    
    results = []
    
    # 3. RUN PARALLEL ANALYSIS
    print(f"\nRunning evaluations with n_permutations = {args.permutations}...")
    print("----------------------------------------------------------------------")
    
    for gene in active_genes:
        sorted_samples = ranks_aligned[gene].sort_values().index
        sorted_clinical = clinical_aligned.loc[sorted_samples]
        
        # --- A. MCI vs Rest (Categorical Transition baseline) ---
        mci_vector = sorted_clinical['MCI_vs_Rest'].values
        obs_es_mci = compute_area_score_regular(mci_vector)
        null_mci = run_area_permutations(mci_vector, is_weighted=False, n_permutations=args.permutations)
        nes_mci, p_mci = compute_nes_and_pvalue(obs_es_mci, null_mci)
        
        # --- B. Braak categorical baseline (Braak III-VI vs 0-II) ---
        braak_cat_vector = sorted_clinical['Braak_III_VI_vs_0_II'].values
        obs_es_bcat = compute_area_score_regular(braak_cat_vector)
        null_bcat = run_area_permutations(braak_cat_vector, is_weighted=False, n_permutations=args.permutations)
        nes_bcat, p_bcat = compute_nes_and_pvalue(obs_es_bcat, null_bcat)
        
        # --- C. Braak continuous (Weighted AREA) ---
        braak_cont_vector = sorted_clinical['Braak_Continuous'].values
        obs_es_bcont = compute_area_score_weighted(braak_cont_vector)
        null_bcont = run_area_permutations(braak_cont_vector, is_weighted=True, n_permutations=args.permutations)
        nes_bcont, p_bcont = compute_nes_and_pvalue(obs_es_bcont, null_bcont)
        
        # --- D. MSI Analysis for Pathology (Braak Cat vs Braak Continuous) ---
        msi = calculate_msi(p_bcat, p_bcont)
        classification = classify_gene(msi)
        
        results.append({
            "Gene": gene,
            "MCI_Regular_NES": nes_mci,
            "MCI_Regular_P": p_mci,
            "Braak_Regular_NES": nes_bcat,
            "Braak_Regular_P": p_bcat,
            "Braak_Weighted_NES": nes_bcont,
            "Braak_Weighted_P": p_bcont,
            "MSI": msi,
            "Classification": classification
        })
        
        print(f"Gene: {gene:<9} | MCI Reg P: {p_mci:.2e} | Braak Reg P: {p_bcat:.2e} | Braak Wgt P: {p_bcont:.2e} | MSI: {msi:>6.2f} | {classification}")

    # Compile and Save Output
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    
    print("----------------------------------------------------------------------")
    print(f"Pipeline Completed! Detailed comparative results saved to: {args.output}")
    print("======================================================================")

if __name__ == "__main__":
    main()

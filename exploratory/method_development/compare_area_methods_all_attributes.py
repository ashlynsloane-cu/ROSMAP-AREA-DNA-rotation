#!/usr/bin/env python3
"""
compare_area_methods_all_attributes.py
======================================================================
ROSMAP Genome-Wide Parallel Method Comparison for All Attributes
----------------------------------------------------------------------
Performs a high-performance, parallelized comparison of Regular vs. 
Weighted AREA across all three core clinical-pathological domains:
  1. Braak (Tangle Pathology Progression)
  2. CERAD (Amyloid Plaque Severity)
  3. Cognitive State (Episodic Cognitive Decline)

Pre-computes empirical null distributions once per clinical vector 
to complete the 21,433-gene genome-wide screens in ~30 seconds.
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
    Standard AREA math.
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
    Weighted AREA math.
    Normalizes by cumulative weight to preserve standard geometry.
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
        
    # Boundary control for pvalues
    pvalue = np.clip(pvalue, 1e-15, 1.0)
    return nes, pvalue

def run_area_permutations(vector, is_weighted, n_permutations=1000):
    """
    Generates the empirical null distribution by shuffling labels/weights once.
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
    Model Selection Index.
    MSI = log10(p_weighted / p_regular)
    """
    p_weighted = max(p_weighted, 1e-15)
    p_regular = max(p_regular, 1e-15)
    return np.log10(p_weighted / p_regular)

def classify_gene(msi, p_reg, p_wgt, cutoff=0.05):
    """
    Classifies genes into functional clinical archetypes.
    """
    if p_reg > cutoff and p_wgt > cutoff:
        return "Non-Significant Driver"
        
    if msi < -2.0:
        return "Dosage Accumulator (Weighted)"
    elif msi > 2.0:
        return "State-Transition Trigger (Regular)"
    else:
        return "Co-Progressive Driver"

def benjamini_hochberg_correction(p_values):
    """
    Performs BH multiple-testing correction.
    """
    p_vals = np.array(p_values)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p_vals = p_vals[sorted_indices]
    
    adj_p_vals = np.zeros(n)
    min_adj_p = 1.0
    for i in range(n - 1, -1, -1):
        p = sorted_p_vals[i]
        adj_p = p * n / (i + 1)
        min_adj_p = min(min_adj_p, adj_p)
        adj_p_vals[i] = min_adj_p
        
    original_adj_p_vals = np.zeros(n)
    original_adj_p_vals[sorted_indices] = adj_p_vals
    return original_adj_p_vals

def run_domain_comparison(domain_name, ranks_df, clinical_df, reg_vector, cont_vector, args, all_genes):
    print(f"\n======================================================================")
    print(f"PROCESSING DOMAIN: {domain_name.upper()}")
    print(f"======================================================================")
    
    # Pre-compute Nulls
    print(f"--> Pre-computing {args.permutations} permutations for {domain_name}...")
    null_reg = run_area_permutations(reg_vector, is_weighted=False, n_permutations=args.permutations)
    null_cont = run_area_permutations(cont_vector, is_weighted=True, n_permutations=args.permutations)
    print("  -> Null distributions pre-computed successfully.")
    
    # Vectorized Processing Loop
    print(f"--> Screening {len(all_genes)} genes in optimized vectorized mode...")
    results = []
    update_interval = max(1, len(all_genes) // 10)
    
    for idx, gene in enumerate(all_genes):
        sorted_indices = np.argsort(ranks_df[gene].values)
        
        vector_reg_sorted = reg_vector[sorted_indices]
        vector_cont_sorted = cont_vector[sorted_indices]
        
        obs_es_reg = compute_area_score_regular(vector_reg_sorted)
        obs_es_cont = compute_area_score_weighted(vector_cont_sorted)
        
        nes_reg, p_reg = compute_nes_and_pvalue(obs_es_reg, null_reg)
        nes_cont, p_cont = compute_nes_and_pvalue(obs_es_cont, null_cont)
        
        msi = calculate_msi(p_reg, p_cont)
        
        results.append({
            "Gene": gene,
            "Regular_NES": nes_reg,
            "Regular_P": p_reg,
            "Weighted_NES": nes_cont,
            "Weighted_P": p_cont,
            "MSI": msi
        })
        
        if (idx + 1) % update_interval == 0 or (idx + 1) == len(all_genes):
            print(f"  Processed {idx + 1}/{len(all_genes)} genes ({(idx + 1)/len(all_genes)*100:.1f}%)...")
            
    # Apply FDR correction
    print("\n--> Applying Benjamini-Hochberg False Discovery Rate corrections...")
    res_df = pd.DataFrame(results)
    res_df['Regular_FDR'] = benjamini_hochberg_correction(res_df['Regular_P'].values)
    res_df['Weighted_FDR'] = benjamini_hochberg_correction(res_df['Weighted_P'].values)
    
    # Classify genes using corrected FDR
    print("--> Classifying genes into molecular archetypes...")
    res_df['Classification'] = res_df.apply(
        lambda r: classify_gene(r['MSI'], r['Regular_FDR'], r['Weighted_FDR'], cutoff=0.05), axis=1
    )
    
    # Save output
    out_path = f"results/area_genome_wide_{domain_name.lower()}_comparison.csv"
    os.makedirs("results", exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"  -> Detailed {domain_name} results saved to: {out_path}")
    
    # Archetype breakdown
    counts = res_df['Classification'].value_counts()
    summary_dict = {}
    for arch in ["Non-Significant Driver", "Dosage Accumulator (Weighted)", "State-Transition Trigger (Regular)", "Co-Progressive Driver"]:
        summary_dict[arch] = counts.get(arch, 0)
        
    return res_df, summary_dict

def main():
    parser = argparse.ArgumentParser(description="ROSMAP Parallel Method Comparison for All Attributes")
    parser.add_argument("-r", "--ranks", default="results/rosmap_area_ranks.csv", help="Path to preprocessed rank CSV")
    parser.add_argument("-c", "--clinical", default="data/ROSMAP_metadata_merged_AREA.csv", help="Path to clinical metadata CSV")
    parser.add_argument("-p", "--permutations", type=int, default=1000, help="Number of permutations (default 1000)")
    
    args = parser.parse_args()
    
    print("======================================================================")
    print("Starting ROSMAP Genome-Wide Parallel Method Comparison (All Attributes)")
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
    
    # 3. SCHEMA COLUMN DISCOVERY
    cog_col = None
    for col in ['diagnosis', 'cogdx', 'dcfdx']:
        if col in clinical_aligned.columns:
            cog_col = col
            break
            
    braak_col = None
    for col in ['braak', 'braaksc', 'braak_stage']:
        if col in clinical_aligned.columns:
            braak_col = col
            break
            
    cerad_col = None
    for col in ['cerad', 'ceradsc', 'cerad_score']:
        if col in clinical_aligned.columns:
            cerad_col = col
            break
            
    if not cog_col or not braak_col or not cerad_col:
        print(f"Error: Missing clinical columns in metadata. Discovered: Cog={cog_col}, Braak={braak_col}, CERAD={cerad_col}")
        sys.exit(1)
        
    print(f"  -> Discovered Cog Diagnosis Column : '{cog_col}'")
    print(f"  -> Discovered Braak Stage Column     : '{braak_col}'")
    print(f"  -> Discovered CERAD Plaque Column    : '{cerad_col}'")
    
    # 4. PREPARE ALL DOMAIN VECTORS
    all_genes = ranks_aligned.columns.tolist()
    print(f"Number of genes to process: {len(all_genes)}")
    
    # A. Braak Vectors
    braak_cont = pd.to_numeric(clinical_aligned[braak_col], errors='coerce').fillna(0.0).values
    braak_reg = np.array([1 if v >= 3 else 0 for v in braak_cont])
    
    # B. CERAD Vectors
    # CERAD is 1 (Definite AD) to 4 (No AD). 
    # Continuous Plaque Severity = 4 - CERAD (so 0 is least, 3 is highest plaque burden)
    cerad_raw = pd.to_numeric(clinical_aligned[cerad_col], errors='coerce').values
    cerad_cont = np.array([float(4 - c) if (pd.notna(c) and c in [1, 2, 3, 4]) else 0.0 for c in cerad_raw])
    cerad_reg = np.array([1 if c in [1, 2] else 0 for c in cerad_raw]) # High CERAD: Definite/Probable
    
    # C. Cognitive Vectors
    # diagnosis: 1=NCI, 2/3=MCI, 4/5=AD.
    # Continuous Cognitive State: NCI=0, MCI=1, AD=2 (ignoring 6 to avoid confounding non-AD primary dementias)
    cog_raw = pd.to_numeric(clinical_aligned[cog_col], errors='coerce').values
    cog_cont = np.array([1.0 if c in [2, 3] else (2.0 if c in [4, 5] else 0.0) for c in cog_raw])
    # Cognitive Impairment vs NCI (diagnosis > 1)
    cog_reg = np.array([1 if c in [2, 3, 4, 5, 6] else 0 for c in cog_raw])
    
    # 5. RUN COMPILATION
    domain_summaries = {}
    
    # Run Braak
    _, sum_braak = run_domain_comparison("Braak", ranks_aligned, clinical_aligned, braak_reg, braak_cont, args, all_genes)
    domain_summaries["Braak (Tangle Pathology)"] = sum_braak
    
    # Run CERAD
    _, sum_cerad = run_domain_comparison("CERAD", ranks_aligned, clinical_aligned, cerad_reg, cerad_cont, args, all_genes)
    domain_summaries["CERAD (Plaque Burden)"] = sum_cerad
    
    # Run Cognitive
    _, sum_cog = run_domain_comparison("Cognitive", ranks_aligned, clinical_aligned, cog_reg, cog_cont, args, all_genes)
    domain_summaries["Cognitive (Clinical Decline)"] = sum_cog
    
    # 6. PRINT JOINT ARCHETYPE MATRIX REPORT
    print("\n" + "="*80)
    print("MASTER METRIC-GENE SELECTION MATRIX SUMMARY")
    print("="*80)
    print(f"{'Molecular Archetype':<40} | {'Braak':<12} | {'CERAD':<12} | {'Cognitive':<12}")
    print("-"*80)
    
    archetypes = [
        "Non-Significant Driver",
        "Dosage Accumulator (Weighted)",
        "State-Transition Trigger (Regular)",
        "Co-Progressive Driver"
    ]
    
    for arch in archetypes:
        val_braak = domain_summaries["Braak (Tangle Pathology)"][arch]
        pct_braak = (val_braak / len(all_genes)) * 100
        
        val_cerad = domain_summaries["CERAD (Plaque Burden)"][arch]
        pct_cerad = (val_cerad / len(all_genes)) * 100
        
        val_cog = domain_summaries["Cognitive (Clinical Decline)"][arch]
        pct_cog = (val_cog / len(all_genes)) * 100
        
        print(f"{arch:<40} | {val_braak:>4} ({pct_braak:>4.1f}%) | {val_cerad:>4} ({pct_cerad:>4.1f}%) | {val_cog:>4} ({pct_cog:>4.1f}%)")
        
    print("="*80)
    print("Genome-wide multi-attribute parallel comparative analysis successfully complete!")
    print("="*80)

if __name__ == "__main__":
    main()





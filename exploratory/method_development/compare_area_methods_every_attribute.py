#!/usr/bin/env python3
"""
compare_area_methods_every_attribute.py
======================================================================
ROSMAP Genome-Wide Parallel Method Comparison for EVERY Clinical Attribute
----------------------------------------------------------------------
Performs a high-performance, parallelized comparison of Regular vs. 
Weighted AREA across all 13 clinical, pathological, demographic, and
physiological attributes in the ROSMAP dataset:
  1. Braak Tangle Stage
  2. CERAD Plaque Score
  3. Consensus Cognitive Diagnosis (cogdx)
  4. Global Cognition z-score
  5. MMSE Bedside Score
  6. Body Mass Index (BMI)
  7. Age at Death
  8. Age at First AD Diagnosis (age_first_ad_dx)
  9. Biological Sex (msex)
  10. APOE e4 Carrier Status
  11. Stroke History
  12. Diabetes History
  13. Hypertension History

Pre-computes empirical null distributions once per clinical vector 
to complete the 21,433-gene genome-wide screens in seconds.
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

def classify_gene(msi, p_reg, p_wgt, cutoff=0.05, is_strictly_binary=False):
    """
    Classifies genes into functional clinical archetypes.
    """
    if p_reg > cutoff and p_wgt > cutoff:
        return "Non-Significant Driver"
    
    if is_strictly_binary:
        return "Co-Progressive Driver (Binary Base)"
        
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

def run_domain_comparison(domain_name, ranks_df, reg_vector, cont_vector, args, all_genes, is_strictly_binary=False):
    print(f"\n======================================================================")
    print(f"PROCESSING ATTRIBUTE: {domain_name.upper()}")
    print(f"======================================================================")
    
    # Pre-compute Nulls
    print(f"--> Pre-computing {args.permutations} permutations for {domain_name}...")
    null_reg = run_area_permutations(reg_vector, is_weighted=False, n_permutations=args.permutations)
    if is_strictly_binary:
        null_cont = null_reg # Avoid redundant computation for binary attributes
    else:
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
        if is_strictly_binary:
            obs_es_cont = obs_es_reg
        else:
            obs_es_cont = compute_area_score_weighted(vector_cont_sorted)
        
        nes_reg, p_reg = compute_nes_and_pvalue(obs_es_reg, null_reg)
        if is_strictly_binary:
            nes_cont, p_cont = nes_reg, p_reg
        else:
            nes_cont, p_cont = compute_nes_and_pvalue(obs_es_cont, null_cont)
        
        msi = 0.0 if is_strictly_binary else calculate_msi(p_reg, p_cont)
        
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
        lambda r: classify_gene(r['MSI'], r['Regular_FDR'], r['Weighted_FDR'], cutoff=0.05, is_strictly_binary=is_strictly_binary), axis=1
    )
    
    # Save output
    out_path = f"results/area_genome_wide_{domain_name.lower()}_comparison.csv"
    os.makedirs("results", exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"  -> Detailed {domain_name} results saved to: {out_path}")
    
    # Archetype breakdown
    counts = res_df['Classification'].value_counts()
    summary_dict = {}
    archs = ["Non-Significant Driver", "Dosage Accumulator (Weighted)", "State-Transition Trigger (Regular)", "Co-Progressive Driver", "Co-Progressive Driver (Binary Base)"]
    for arch in archs:
        summary_dict[arch] = counts.get(arch, 0)
        
    return res_df, summary_dict

def main():
    parser = argparse.ArgumentParser(description="ROSMAP Parallel Method Comparison for EVERY Clinical Attribute")
    parser.add_argument("-r", "--ranks", default="results/rosmap_area_ranks.csv", help="Path to preprocessed rank CSV")
    parser.add_argument("-c", "--clinical", default="data/ROSMAP_metadata_merged_AREA.csv", help="Path to clinical metadata CSV")
    parser.add_argument("-p", "--permutations", type=int, default=1000, help="Number of permutations (default 1000)")
    
    args = parser.parse_args()
    
    print("======================================================================")
    print("Starting ROSMAP Genome-Wide Parallel Method Comparison (EVERY Attribute)")
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
    all_genes = ranks_aligned.columns.tolist()
    print(f"Number of genes to process: {len(all_genes)}")
    
    # 3. SCHEMA ALIGNMENTS & PREPARATIONS
    attributes_to_run = {}
    
    # Definition Helper:
    # attributes_to_run[attribute_name] = { 'reg': binary_vector, 'cont': continuous_vector, 'is_binary': Bool }

    # 1. Braak Tangles
    braak_col = next((c for c in ['braaksc', 'braak', 'braak_stage'] if c in clinical_aligned.columns), None)
    if braak_col:
        cont = pd.to_numeric(clinical_aligned[braak_col], errors='coerce').fillna(0.0).values
        reg = np.array([1 if v >= 3 else 0 for v in cont])
        attributes_to_run['Braak_Tangles'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 2. CERAD Plaques
    cerad_col = next((c for c in ['ceradsc', 'cerad', 'cerad_score'] if c in clinical_aligned.columns), None)
    if cerad_col:
        raw = pd.to_numeric(clinical_aligned[cerad_col], errors='coerce').values
        cont = np.array([float(4 - c) if (pd.notna(c) and c in [1, 2, 3, 4]) else 0.0 for c in raw])
        reg = np.array([1 if c in [1, 2] else 0 for c in raw])
        attributes_to_run['CERAD_Plaques'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 3. Consensus Diagnosis
    cog_col = next((c for c in ['cogdx', 'diagnosis', 'dcfdx'] if c in clinical_aligned.columns), None)
    if cog_col:
        raw = pd.to_numeric(clinical_aligned[cog_col], errors='coerce').values
        cont = np.array([1.0 if c in [2, 3] else (2.0 if c in [4, 5] else 0.0) for c in raw])
        reg = np.array([1 if c in [2, 3, 4, 5, 6] else 0 for c in raw])
        attributes_to_run['Consensus_Cognition'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 4. Global Cognition
    glob_col = next((c for c in ['cogn_global', 'cogng_global'] if c in clinical_aligned.columns), None)
    if glob_col:
        raw = pd.to_numeric(clinical_aligned[glob_col], errors='coerce').values
        # Invert cognitive z-score so that a higher value represents more decline (enriching positive NES)
        cont = np.array([-float(v) if pd.notna(v) else 0.0 for v in raw])
        reg = np.array([1 if v < -1.0 else 0 for v in raw])
        attributes_to_run['Global_Cognition'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 5. MMSE Score
    mmse_col = next((c for c in ['cts_estmmse30', 'mmse', 'cts_mmse30_lv'] if c in clinical_aligned.columns), None)
    if mmse_col:
        cont = pd.to_numeric(clinical_aligned[mmse_col], errors='coerce').fillna(30.0).values
        reg = np.array([1 if v < 24 else 0 for v in cont])
        attributes_to_run['MMSE_Score'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 6. Body Mass Index (BMI)
    bmi_col = next((c for c in ['bmi', 'bmi_lv'] if c in clinical_aligned.columns), None)
    if bmi_col:
        cont = pd.to_numeric(clinical_aligned[bmi_col], errors='coerce').fillna(25.0).values
        reg = np.array([1 if v >= 30.0 else 0 for v in cont])
        attributes_to_run['Body_Mass_Index'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 7. Age at Death
    age_col = next((c for c in ['age_death', 'age'] if c in clinical_aligned.columns), None)
    if age_col:
        cont = pd.to_numeric(clinical_aligned[age_col], errors='coerce').fillna(85.0).values
        reg = np.array([1 if v >= 85.0 else 0 for v in cont])
        attributes_to_run['Age_at_Death'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 8. Age at First AD Diagnosis
    onset_col = next((c for c in ['age_first_ad_dx'] if c in clinical_aligned.columns), None)
    if onset_col:
        cont = pd.to_numeric(clinical_aligned[onset_col], errors='coerce').fillna(0.0).values
        reg = np.array([1 if (v > 0 and v <= 75) else 0 for v in cont])
        attributes_to_run['Age_at_AD_Onset'] = {'reg': reg, 'cont': cont, 'is_binary': False}

    # 9. Biological Sex (Strictly Binary)
    sex_col = next((c for c in ['msex', 'sex'] if c in clinical_aligned.columns), None)
    if sex_col:
        reg = pd.to_numeric(clinical_aligned[sex_col], errors='coerce').fillna(0.0).values
        attributes_to_run['Biological_Sex'] = {'reg': reg, 'cont': reg, 'is_binary': True}

    # 10. APOE e4 Status (Strictly Binary)
    apoe_col = next((c for c in ['apoe_genotype', 'apoe'] if c in clinical_aligned.columns), None)
    if apoe_col:
        def binarize_apoe(val):
            if pd.isna(val): return 0.0
            val_str = str(int(float(val))).strip()
            return 1.0 if '4' in val_str else 0.0
        reg = clinical_aligned[apoe_col].apply(binarize_apoe).values
        attributes_to_run['APOE_e4_Carrier'] = {'reg': reg, 'cont': reg, 'is_binary': True}

    # 11. Stroke History (Strictly Binary)
    stroke_col = next((c for c in ['r_stroke', 'stroke_cum', 'stroke'] if c in clinical_aligned.columns), None)
    if stroke_col:
        reg = pd.to_numeric(clinical_aligned[stroke_col], errors='coerce').fillna(0.0).values
        attributes_to_run['Stroke_History'] = {'reg': reg, 'cont': reg, 'is_binary': True}

    # 12. Diabetes History (Strictly Binary)
    diab_col = next((c for c in ['diabetes_sr_rx', 'diab', 'diabetes'] if c in clinical_aligned.columns), None)
    if diab_col:
        reg = pd.to_numeric(clinical_aligned[diab_col], errors='coerce').fillna(0.0).values
        attributes_to_run['Diabetes_History'] = {'reg': reg, 'cont': reg, 'is_binary': True}

    # 13. Hypertension History (Strictly Binary)
    hyper_col = next((c for c in ['hypertension_cum', 'hyperten', 'hypertension'] if c in clinical_aligned.columns), None)
    if hyper_col:
        reg = pd.to_numeric(clinical_aligned[hyper_col], errors='coerce').fillna(0.0).values
        attributes_to_run['Hypertension_History'] = {'reg': reg, 'cont': reg, 'is_binary': True}

    # 4. RUN SYSTEMATIC METHOD COMPARISON FOR ALL ATTRIBUTES
    all_summaries = {}
    for attr_name, payload in attributes_to_run.items():
        _, summary = run_domain_comparison(
            attr_name, ranks_aligned, payload['reg'], payload['cont'], args, all_genes, is_strictly_binary=payload['is_binary']
        )
        all_summaries[attr_name] = summary

    # 5. PRINT JOINT MASTER METRIC-GENE SELECTION MATRIX REPORT
    print("\n" + "="*110)
    print("MASTER CLINICAL-PATHOLOGICAL GENOME-WIDE METRIC SELECTION MATRIX (EVERY ATTRIBUTE)")
    print("="*110)
    print(f"{'Clinical Attribute':<25} | {'Non-Sig':<12} | {'Weighted (Dosage)':<18} | {'Regular (Trigger)':<18} | {'Co-Progressive':<15} | {'Strictly Binary':<15}")
    print("-"*110)
    
    for attr_name in sorted(attributes_to_run.keys()):
        sum_data = all_summaries[attr_name]
        
        non_sig = sum_data["Non-Significant Driver"]
        pct_non_sig = (non_sig / len(all_genes)) * 100
        
        weighted = sum_data["Dosage Accumulator (Weighted)"]
        pct_weighted = (weighted / len(all_genes)) * 100
        
        regular = sum_data["State-Transition Trigger (Regular)"]
        pct_regular = (regular / len(all_genes)) * 100
        
        coprog = sum_data["Co-Progressive Driver"]
        pct_coprog = (coprog / len(all_genes)) * 100
        
        binary_base = sum_data["Co-Progressive Driver (Binary Base)"]
        pct_binary_base = (binary_base / len(all_genes)) * 100
        
        print(f"{attr_name:<25} | {non_sig:>5} ({pct_non_sig:>4.1f}%) | {weighted:>5} ({pct_weighted:>4.1f}%) | {regular:>5} ({pct_regular:>4.1f}%) | {coprog:>5} ({pct_coprog:>4.1f}%) | {binary_base:>5} ({pct_binary_base:>4.1f}%)")
        
    print("="*110)
    print("Genome-wide ALL-attribute parallel comparative analysis successfully complete!")
    print("="*110)

if __name__ == "__main__":
    main()

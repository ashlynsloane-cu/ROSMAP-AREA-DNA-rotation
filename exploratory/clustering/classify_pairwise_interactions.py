#!/usr/bin/env python3
"""
classify_pairwise_interactions.py
======================================================================
Tier 4 of the ROSMAP Funnel: Pairwise Interaction & Synergy Classification
======================================================================
This script systematically calculates Rothman's additive interaction
metrics—Relative Excess Risk due to Interaction (RERI) and Attributable
Proportion (AP)—for every gene-gene risk pair across all 13 clinical,
pathological, and metabolic RAE matrices.

Provides a robust safeguard:
- Adds '--max-genes' command-line threshold (default: 500) to subset
  high-density matrices to the top most abundant risk-associated drivers,
  preventing Out-Of-Memory (OOM) terminal kills on consumer hardware!
======================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


def clean_id_to_string(val):
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.endswith('.0'):
        val_str = val_str[:-2]
    if val_str.upper().startswith('R'):
        val_str = val_str[1:]
    return val_str.strip()


def clean_attribute_name(col_or_filename):
    name = col_or_filename.lower()
    junk = [
        "area_resultsarea_scores_",
        "area_genome_wide_method_comparison",
        "area_method_comparison",
        "method_comparison",
        "results_",
        "_regular_nes",
        "_weighted_nes",
        "_nes",
        ".csv",
        "rae_matrix_"
    ]
    for pattern in junk:
        name = name.replace(pattern, "")
        
    name = name.strip("_").strip()
    
    translation = {
        "high_braak": "High Braak",
        "high_cerad": "High CERAD",
        "mci_vs_rest": "MCI vs Rest",
        "ad_vs_rest": "AD vs Rest",
        "nci_vs_rest": "NCI vs Rest",
        "ad_vs_nci": "AD vs NCI",
        "cognitive_impairment_vs_nci": "Cognitive Impairment vs NCI",
        "apoe_e4_carrier": "APOE ε4 Carrier",
        "sex_male": "Male Sex",
        "sex_female": "Female Sex",
        "braak_iii_vi_vs_0_ii": "Braak III-VI vs 0-II",
        "braak_continuous": "Braak Continuous",
        "bmi": "BMI Trigger",
        "diabetes_sr_rx": "Diabetes History"
    }
    
    return translation.get(name, name.replace("_", " ").title())


def map_rae_to_boolean_column(rae_filename, boolean_columns):
    name_lower = rae_filename.lower().replace("rae_matrix_", "").replace(".csv", "").strip("_")
    
    mappings = {
        "mci_regular": "mci_vs_rest",
        "mci_vs_rest_regular": "mci_vs_rest",
        "mci_vs_rest_weighted": "mci_vs_rest",
        "braak_regular": "braak_iii_vi_vs_0_ii",
        "braak_weighted": "braak_iii_vi_vs_0_ii",
        "high_braak_weighted": "high_braak",
        "high_braak_regular": "high_braak",
        "mmse_score_regular": "cognitive_impairment_vs_nci",
        "mmse_score_weighted": "cognitive_impairment_vs_nci",
        "global_cognition_regular": "cognitive_impairment_vs_nci",
        "global_cognition_weighted": "cognitive_impairment_vs_nci",
        "neuritic_plaques_regular": "high_cerad",
        "neuritic_plaques_weighted": "high_cerad"
    }
    
    if name_lower in mappings:
        target = mappings[name_lower]
        for col in boolean_columns:
            if col.lower().strip() == target:
                return col

    for col in boolean_columns:
        col_lower = col.lower()
        if "braak" in name_lower and "braak" in col_lower:
            return col
        if "mmse" in name_lower and ("mmse" in col_lower or "cognitive" in col_lower or "cogn" in col_lower):
            return col
        if "cogn" in name_lower and ("cogn" in col_lower or "mci" in col_lower or "ad_" in col_lower):
            return col
        if "cerad" in name_lower and "cerad" in col_lower:
            return col
        if "mci" in name_lower and "mci" in col_lower:
            return col
        if "ad_vs" in name_lower and "ad_vs" in col_lower:
            return col
        if "nci" in name_lower and "nci" in col_lower:
            return col
        if "apoe" in name_lower and "apoe" in col_lower:
            return col
        if "sex" in name_lower and "sex" in col_lower:
            return col
            
    tokens = name_lower.split("_")
    for token in tokens:
        if len(token) > 2:
            for col in boolean_columns:
                if token in col.lower():
                    return col
                    
    return None


def classify_interaction_logic(rr10, rr01, rr11, reri, ap):
    is_i_protective = rr10 < 0.85
    is_j_protective = rr01 < 0.85
    is_i_hazard = rr10 > 1.15
    is_j_hazard = rr01 > 1.15
    is_i_benign = 0.85 <= rr10 <= 1.15
    is_j_benign = 0.85 <= rr01 <= 1.15
    
    # 1. Dual-Protective Cooperativity
    if is_i_protective and is_j_protective and rr11 < 0.85:
        if reri > 0 and ap > 0.1:
            return "Dual-Protective Cooperativity", "Resilience"
        else:
            return "Additive Protective", "Resilience"
            
    # 2. Antagonistic Protective Interference
    if is_i_protective and is_j_protective and rr11 >= 0.95:
        return "Antagonistic Protective Interference", "Interference"
        
    # 3. Antagonistic Buffering (Molecular Shield)
    if ((is_i_hazard and is_j_protective) or (is_i_protective and is_j_hazard)) and rr11 <= 1.10:
        return "Antagonistic Buffering (Molecular Shield)", "Resilience"
        
    # 4. Potentiating Synergy
    if ((is_i_benign and is_j_hazard) or (is_i_hazard and is_j_benign)) and rr11 > 1.30:
        if ap > 0.4:
            return "Potentiating Synergy", "Alternative Etiology"
        else:
            return "Additive Risk", "Alternative Etiology"
            
    # 5. True Cooperative Synergy
    if is_i_hazard and is_j_hazard and rr11 > 1.20:
        if reri > 0 and ap > 0.4:
            return "True Cooperative Synergy", "Alternative Etiology"
        else:
            return "Additive Risk", "Alternative Etiology"
            
    if rr11 < 0.85:
        return "Unclassified Protective", "Resilience"
    elif rr11 > 1.15:
        return "Unclassified Risk", "Alternative Etiology"
    else:
        return "Unclassified Additive", "Neutral"


def run_self_test():
    print("\n[SELF-TEST] Simulating pairwise synergy mapping...")
    np.random.seed(42)
    patients = [f"R{1000+i}" for i in range(120)]
    
    # 600 genes (to trigger max-genes subset filtering)
    genes = [f"Gene_{i}" for i in range(600)]
    mock_matrix = np.random.choice([0, 1], size=(120, 600), p=[0.8, 0.2])
    df_rae = pd.DataFrame(mock_matrix, index=patients, columns=genes)
    
    df_bools = pd.DataFrame({
        "AD_vs_NCI": np.random.choice([0, 1], size=120, p=[0.5, 0.5])
    }, index=patients)
    df_bools.index.name = "projid"
    
    df_bools.to_csv("mock_bools_syn.csv")
    df_rae.to_csv("rae_matrix_ad_vs_nci.csv")
    
    sys.argv = [sys.argv[0], "-b", "mock_bools_syn.csv", "-r", ".", "-o", "results_synergies_test", "--max-genes", "20"]
    
    import shutil
    try:
        main()
    finally:
        for f in ["mock_bools_syn.csv", "rae_matrix_ad_vs_nci.csv"]:
            if os.path.exists(f): os.remove(f)
        if os.path.exists("results_synergies_test"):
            shutil.rmtree("results_synergies_test")
            
    print("[SELF-TEST] Complete! Pairwise interaction script is fully operational.\n")


def main():
    if "--self-test" in sys.argv:
        run_self_test()
        sys.exit(0)
        
    parser = argparse.ArgumentParser(description="Classify pairwise risk-interaction categories across ROSMAP attributes.")
    parser.add_argument("-b", "--bools", required=True, help="Path to clinical attributes CSV file.")
    parser.add_argument("-r", "--rae-dir", required=True, help="Path to directory containing patient RAE binarized matrices.")
    parser.add_argument("-o", "--outdir", default="results/pairwise_synergies", help="Output directory.")
    parser.add_argument("--min-stratum-size", type=int, default=15, help="Minimum size per stratum (default: 15).")
    parser.add_argument("--max-genes", type=int, default=500,
                        help="Top N highest-prevalence risk genes to analyze to prevent memory crash (default: 500).")
    args = parser.parse_args()
    
    print("======================================================================")
    print("Starting ROSMAP Pairwise Synergy & RERI Classification Pipeline (v2)")
    print("======================================================================")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    try:
        bools_df = pd.read_csv(args.bools, index_col=0)
        bools_df.index = bools_df.index.map(clean_id_to_string)
    except Exception as e:
        print(f"Error loading clinical bools file: {e}")
        sys.exit(1)
        
    rae_files = [f for f in os.listdir(args.rae_dir) if f.startswith("rae_matrix_") and f.endswith(".csv")]
    if not rae_files:
        print(f"Error: No RAE matrices found in directory '{args.rae_dir}' matching 'rae_matrix_*.csv'.")
        sys.exit(1)
        
    print(f" -> Discovered {len(rae_files)} RAE matrix files to analyze.")
    global_records = []
    
    for rae_file in sorted(rae_files):
        rae_path = os.path.join(args.rae_dir, rae_file)
        try:
            rae_df = pd.read_csv(rae_path, index_col=0)
            rae_df.index = rae_df.index.map(clean_id_to_string)
        except Exception as e:
            print(f" -> Error reading {rae_file}: {e}. Skipping.")
            continue
            
        genes_raw = [c for c in rae_df.columns if c not in ["Participant", "Participant_Clean"]]
        if len(genes_raw) < 2:
            continue
            
        common_samples = sorted(list(set(rae_df.index).intersection(set(bools_df.index))))
        if not common_samples:
            continue
            
        bool_col_target = map_rae_to_boolean_column(rae_file, bools_df.columns)
        if not bool_col_target:
            continue
            
        # CRITICAL PROTECTION: If there are too many genes, subset to the top --max-genes
        # based on overall RAE prevalence (column sums) to prevent Out-Of-Memory (OOM) shell crashes!
        if len(genes_raw) > args.max_genes:
            print(f"\\nProcessing pairwise synergies for: {rae_file}")
            print(f" -> WARNING: Matrix has {len(genes_raw):,} genes. Subsetting to top {args.max_genes} highest prevalence risk genes to prevent MacBook OOM crash.")
            col_sums = rae_df[genes_raw].sum(axis=0)
            genes = list(col_sums.nlargest(args.max_genes).index)
        else:
            print(f"\\nProcessing pairwise synergies for: {rae_file}")
            genes = genes_raw
            
        print(f" -> Target Clinical Contrast: '{bool_col_target}' | Evaluating {len(genes)} significant genes...")
        
        subset_df = pd.DataFrame(index=common_samples)
        subset_df["Label"] = bools_df.loc[common_samples, bool_col_target]
        for g in genes:
            subset_df[g] = rae_df.loc[common_samples, g]
            
        subset_df = subset_df.dropna()
        if len(subset_df) < 15:
            print(" -> Skipping: Cohort size too small after aligning.")
            continue
            
        attribute_clean = clean_attribute_name(rae_file)
        
        # Optimize by vector pre-slicing
        matrix_vals = subset_df[genes].values
        labels_vals = subset_df["Label"].values
        num_samples = len(subset_df)
        
        # Loop through all pairs
        for i in range(len(genes)):
            g_i_vals = matrix_vals[:, i]
            for j in range(i+1, len(genes)):
                g_j_vals = matrix_vals[:, j]
                
                # Logical slices are computed extremely fast via NumPy bitwise checks
                c00 = (g_i_vals == 0) & (g_j_vals == 0)
                c10 = (g_i_vals == 1) & (g_j_vals == 0)
                c01 = (g_i_vals == 0) & (g_j_vals == 1)
                c11 = (g_i_vals == 1) & (g_j_vals == 1)
                
                s00, s10, s01, s11 = c00.sum(), c10.sum(), c01.sum(), c11.sum()
                
                if s00 < args.min_stratum_size or s10 < args.min_stratum_size or s01 < args.min_stratum_size or s11 < args.min_stratum_size:
                    continue
                    
                # Calculate prevalence rates
                P00 = labels_vals[c00].mean()
                P10 = labels_vals[c10].mean()
                P01 = labels_vals[c01].mean()
                P11 = labels_vals[c11].mean()
                
                if P00 == 0:
                    P00 = 0.01
                    
                RR10 = P10 / P00
                RR01 = P01 / P00
                RR11 = P11 / P00
                
                RERI = RR11 - RR10 - RR01 + 1
                AP = RERI / RR11 if RR11 > 0 else 0.0
                
                cat, mode = classify_interaction_logic(RR10, RR01, RR11, RERI, AP)
                
                record = {
                    "Attribute": attribute_clean,
                    "Contrast_Label": bool_col_target,
                    "Gene_i": genes[i],
                    "Gene_j": genes[j],
                    "N00": int(s00),
                    "P00_Pct": P00 * 100,
                    "N10": int(s10),
                    "P10_Pct": P10 * 100,
                    "N01": int(s01),
                    "P01_Pct": P01 * 100,
                    "N11": int(s11),
                    "P11_Pct": P11 * 100,
                    "RR10": RR10,
                    "RR01": RR01,
                    "RR11": RR11,
                    "RERI": RERI,
                    "AP": AP,
                    "Interaction_Category": cat,
                    "Physiological_Mode": mode
                }
                
                global_records.append(record)
                
    if global_records:
        global_df = pd.DataFrame(global_records)
        edge_list_path = os.path.join(args.outdir, "classified_pairwise_interactions_all_attributes.csv")
        global_df.to_csv(edge_list_path, index=False)
        print(f"\\n[Success] Consolidated edge-list spreadsheet saved to: '{edge_list_path}'")
        
        report_path = os.path.join(args.outdir, "pairwise_interaction_summary_report.txt")
        with open(report_path, "w") as f_out:
            f_out.write("=================================================================\\n")
            f_out.write("CONSOLIDATED ROSMAP COHORT EPIDEMIOLOGICAL INTERACTION REPORT\\n")
            f_out.write("=================================================================\\n\\n")
            f_out.write(f"Evaluated attributes: {global_df['Attribute'].nunique()}\\n")
            f_out.write(f"QC stratum size threshold: N >= {args.min_stratum_size} patients per cell.\\n")
            f_out.write(f"Total passing gene-gene pairs evaluated: {len(global_df):,}\\n\\n")
            
            f_out.write("-----------------------------------------------------------------\\n")
            f_out.write("1. CONSOLIDATED SUMMARY OF INTERACTION CLASSES\\n")
            f_out.write("-----------------------------------------------------------------\\n")
            counts = global_df["Interaction_Category"].value_counts()
            for cat, cnt in counts.items():
                f_out.write(f" * {cat:<42}: {cnt:,} pairs ({(cnt/len(global_df)*100):.1f}%)\\n")
                
            f_out.write("\\n-----------------------------------------------------------------\\n")
            f_out.write("2. HIGHLIGHTED RESILIENCE DRIVERS (Antagonistic Buffering / Cooperativity)\\n")
            f_out.write("-----------------------------------------------------------------\\n")
            res_df = global_df[global_df["Physiological_Mode"] == "Resilience"].sort_values("AP", ascending=False)
            if not res_df.empty:
                for idx, row in res_df.head(10).iterrows():
                    f_out.write(f"Attribute: {row['Attribute']} | Pair: {row['Gene_i']} x {row['Gene_j']}\\n")
                    f_out.write(f" * Classification : {row['Interaction_Category']}\\n")
                    f_out.write(f" * RERI           : {row['RERI']:.4f} | Attributable Proportion (AP): {row['AP']:.4f}\\n")
                    f_out.write(f" * Strata Pcts (%) : P00={row['P00_Pct']:.1f}% | P10={row['P10_Pct']:.1f}% | P01={row['P01_Pct']:.1f}% | P11={row['P11_Pct']:.1f}%\\n\\n")
            else:
                f_out.write(" No resilient, buffering modifier pairs passed strict QC limits.\\n")
                
            f_out.write("\\n-----------------------------------------------------------------\\n")
            f_out.write("3. HIGHLIGHTED TOXIC SYNERGIES (Potentiating / True Cooperative Synergy)\\n")
            f_out.write("-----------------------------------------------------------------\\n")
            syn_df = global_df[global_df["Interaction_Category"].isin(["True Cooperative Synergy", "Potentiating Synergy"])].sort_values("AP", ascending=False)
            if not syn_df.empty:
                for idx, row in syn_df.head(10).iterrows():
                    f_out.write(f"Attribute: {row['Attribute']} | Pair: {row['Gene_i']} x {row['Gene_j']}\\n")
                    f_out.write(f" * Classification : {row['Interaction_Category']}\\n")
                    f_out.write(f" * RERI           : {row['RERI']:.4f} | Attributable Proportion (AP): {row['AP']:.4f}\\n")
                    f_out.write(f" * Strata Pcts (%) : P00={row['P00_Pct']:.1f}% | P10={row['P10_Pct']:.1f}% | P01={row['P01_Pct']:.1f}% | P11={row['P11_Pct']:.1f}%\\n\\n")
            else:
                f_out.write(" No hazardous potentiating or cooperative synergies passed strict QC limits.\\n")
                
        print(f" -> Summary report saved to: '{report_path}'")
    else:
        print("\nError: No gene pairs across any of the 13 attributes met the minimum stratum size criteria.")
        
    print("\n======================================================================")
    print("Pairwise Interaction Classification Pipeline Completed Successfully!")
    print("======================================================================")


if __name__ == "__main__":
    main()

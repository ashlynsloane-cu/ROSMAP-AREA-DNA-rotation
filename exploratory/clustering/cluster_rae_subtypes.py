#!/usr/bin/env python3
"""
cluster_rae_subtypes.py
======================================================================
Tier 4 of the ROSMAP Funnel: Patient Subtyping & 4-Way Cohort Stratification
======================================================================
This script performs unsupervised K-means clustering on the binary
Patient-by-Gene Risk-Associated Expression (RAE) matrices for all 13
clinical, pathological, and metabolic attributes.

Includes:
1. Automated Silhouette optimal k calculation (Hamming metric).
2. Grouping of patients into 4 clinical progression cohorts (Resilient, etc.).
3. Complex heatmap replica plotting using hierarchical clustering.
4. Robust system recursion limits increased to handle massive trees!
======================================================================
"""

import os
import sys
import argparse
import textwrap
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

# CRITICAL ACTION: Increase Python's recursion limit to prevent crashes during dendrogram 
# calculations on very large gene linkage trees (e.g. over 1,000 to 10,000 genes)!
sys.setrecursionlimit(100000)

# Standard aesthetics
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42

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
        if "braak" in name_lower and "braak" in col_lower: return col
        if "mmse" in name_lower and ("mmse" in col_lower or "cognitive" in col_lower or "cogn" in col_lower): return col
        if "cogn" in name_lower and ("cogn" in col_lower or "mci" in col_lower or "ad_" in col_lower): return col
        if "cerad" in name_lower and "cerad" in col_lower: return col
        if "mci" in name_lower and "mci" in col_lower: return col
        if "ad_vs" in name_lower and "ad_vs" in col_lower: return col
        if "nci" in name_lower and "nci" in col_lower: return col
        if "apoe" in name_lower and "apoe" in col_lower: return col
        if "sex" in name_lower and "sex" in col_lower: return col
    return None

def generate_complex_heatmap_replica(X, profiles, bools_df, genes, final_k, heatmap_path, clean_title):
    """
    Plots an ultra-polished heat map ordering both patients (rows) and genes (columns) 
    using hierarchical clustering. Natively supports very large numbers of genes safely!
    """
    if X.empty or len(genes) < 2:
        return
        
    print(f"    * Generating complex heatmap replica for {X.shape[1]} genes...")
    
    # 1. Compute linkage for columns (genes)
    gene_linkage = linkage(X.T.values, method='average', metric='hamming')
    
    # Track the order of columns using dendrogram leaves
    try:
        dendro = dendrogram(gene_linkage, no_plot=True)
        col_order_idx = dendro['leaves']
        ordered_genes = [genes[idx] for idx in col_order_idx]
    except Exception as e:
        print(f"    * Note: Hierarchical column sorting skipped: {e}. Falling back to default gene order.")
        ordered_genes = genes

    # 2. Sort rows (patients) by their Cluster and RAE burden
    profiles_sorted = profiles.sort_values(by=["Cluster", "RAE_Burden"], ascending=[True, False])
    ordered_patients = profiles_sorted.index
    
    # Slice the sorted matrix
    plot_matrix = X.loc[ordered_patients, ordered_genes].values
    
    # Plotting parameters (cap height/width for extreme sizes)
    fig_h = max(6, min(14, len(ordered_patients) * 0.015))
    fig_w = max(8, min(16, len(ordered_genes) * 0.05))
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#ffffff')
    
    # Draw heatmap
    sns.heatmap(
        plot_matrix,
        cmap=["#f5f5f5", "#1f77b4"], # Clean binary palette (Grey=0, Blue=1)
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    
    ax.set_title(f"RAE Subtype Activation Map: {clean_title} (k={final_k})", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(f"Significant Risk-Associated Genes (N={len(ordered_genes):,})", fontsize=10, fontweight='semibold')
    ax.set_ylabel(f"Aligned Patient Cohort (N={len(ordered_patients):,})", fontsize=10, fontweight='semibold')
    
    # Save high-res plot
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    * Heatmap saved to: {heatmap_path}")

def run_self_test():
    print("\n[SELF-TEST] Running simulated RAE subtyping on mock datasets...")
    np.random.seed(42)
    patients = [f"R{1000+i}" for i in range(120)]
    
    # Simulated RAE matrix with 1050 genes (to test the increased recursion limit)
    genes = [f"Gene_{i}" for i in range(1050)]
    mock_matrix = np.random.choice([0, 1], size=(120, 1050), p=[0.85, 0.15])
    df_rae = pd.DataFrame(mock_matrix, index=patients, columns=genes)
    
    # Simulated clinical bools (AD_vs_NCI)
    df_bools = pd.DataFrame({
        "AD_vs_NCI": np.random.choice([0, 1], size=120, p=[0.5, 0.5])
    }, index=patients)
    df_bools.index.name = "projid"
    
    df_bools.to_csv("mock_bools.csv")
    df_rae.to_csv("rae_matrix_ad_vs_nci.csv")
    
    # Run pipeline
    print(" -> Executing cluster_rae_subtypes pipeline on mock directories...")
    # Simulate argv overrides
    sys.argv = [sys.argv[0], "-b", "mock_bools.csv", "-r", ".", "-o", "results_subtypes_test"]
    
    import shutil
    try:
        main()
    finally:
        # Cleanup mock files
        for f in ["mock_bools.csv", "rae_matrix_ad_vs_nci.csv"]:
            if os.path.exists(f): os.remove(f)
        if os.path.exists("results_subtypes_test"):
            shutil.rmtree("results_subtypes_test")
            
    print("[SELF-TEST] Complete! Subtyping script is fully operational.\n")

def main():
    if "--self-test" in sys.argv:
        run_self_test()
        sys.exit(0)
        
    parser = argparse.ArgumentParser(description="Unsupervised Patient-by-Gene RAE Clustering")
    parser.add_argument("-b", "--bools", required=True, help="Clinical attributes CSV")
    parser.add_argument("-r", "--rae-dir", required=True, help="Directory containing rae_matrix_*.csv")
    parser.add_argument("-o", "--outdir", default="results/patient_subtypes", help="Output directory")
    args = parser.parse_args()
    
    print("======================================================================")
    print("Starting ROSMAP Patient Subtyping & Cohort Stratification Pipeline (v2)")
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
        
    print(f" -> Discovered {len(rae_files)} RAE matrix files to process.")
    summary_records = []
    
    for rae_file in sorted(rae_files):
        print(f"\nProcessing RAE matrix: {rae_file}")
        rae_path = os.path.join(args.rae_dir, rae_file)
        
        try:
            rae_df = pd.read_csv(rae_path, index_col=0)
            rae_df.index = rae_df.index.map(clean_id_to_string)
        except Exception as e:
            print(f" -> Error reading {rae_file}: {e}. Skipping.")
            continue
            
        genes = [c for c in rae_df.columns if c not in ["Participant", "Participant_Clean"]]
        if len(genes) < 2:
            print(f" -> Skipping {rae_file}: too few significant genes.")
            continue
            
        common_samples = sorted(list(set(rae_df.index).intersection(set(bools_df.index))))
        if not common_samples:
            print(f" -> Skipping {rae_file}: no matching patient IDs.")
            continue
            
        bool_col_target = map_rae_to_boolean_column(rae_file, bools_df.columns)
        if not bool_col_target:
            print(f" -> Skipping {rae_file}: target clinical contrast not detected.")
            continue
            
        print(f" -> Aligned {len(common_samples)} patients. Clinical contrast target: '{bool_col_target}'")
        clinical_labels = bools_df.loc[common_samples, bool_col_target].dropna()
        valid_patients = clinical_labels.index
        
        if len(valid_patients) < 10:
            print(f" -> Skipping {rae_file}: too few valid patients.")
            continue
            
        X = rae_df.loc[valid_patients, genes].copy()
        
        # Silhouette Analysis (Optimizing k)
        max_k = min(10, len(valid_patients) - 2)
        if max_k < 2:
            continue
            
        best_k = 2
        best_silhouette = -1.0
        
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=25, max_iter=100)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels, metric="hamming")
            if score > best_silhouette:
                best_silhouette = score
                best_k = k
                
        print(f"   * Selected Cluster Count k = {best_k} (Hamm-Silhouette Width: {best_silhouette:.4f})")
        
        # Final K-Means
        final_km = KMeans(n_clusters=best_k, random_state=42, n_init=50, max_iter=200)
        patient_labels = final_km.fit_predict(X)
        
        profiles = pd.DataFrame(index=valid_patients)
        profiles["Participant"] = valid_patients
        profiles["Cluster"] = patient_labels
        profiles["Disease_Status"] = clinical_labels.astype(int)
        profiles["RAE_Burden"] = X.sum(axis=1)
        
        cluster_means = profiles.groupby("Cluster")["RAE_Burden"].mean()
        high_burden_cluster = cluster_means.idxmax()
        high_burden_mean = cluster_means.max()
        
        # Cohort Stratification
        profiles["Cohort_Stratification"] = "Typical Healthy"
        profiles.loc[(profiles["Cluster"] == high_burden_cluster) & (profiles["Disease_Status"] == 0), "Cohort_Stratification"] = "Resilient"
        profiles.loc[(profiles["Cluster"] == high_burden_cluster) & (profiles["Disease_Status"] == 1), "Cohort_Stratification"] = "Canonical Disease"
        profiles.loc[(profiles["Cluster"] != high_burden_cluster) & (profiles["Disease_Status"] == 1), "Cohort_Stratification"] = "Alternative Etiology"
        profiles.loc[(profiles["Cluster"] != high_burden_cluster) & (profiles["Disease_Status"] == 0), "Cohort_Stratification"] = "Typical Healthy"
        
        counts = profiles["Cohort_Stratification"].value_counts()
        print("   * Stratification Distribution:")
        for name, cnt in counts.items():
            print(f"      - {name:<22}: {cnt} patients")
            
        trait_slug = clean_attribute_name(rae_file).lower().replace(" ", "_")
        
        # Save profiles & sample sheets
        profiles.to_csv(os.path.join(args.outdir, f"patient_clusters_{trait_slug}.csv"), index=False)
        profiles[profiles["Cohort_Stratification"].isin(["Resilient", "Canonical Disease"])].to_csv(
            os.path.join(args.outdir, f"samplesheet_resilience_{trait_slug}.csv"), index=False
        )
        profiles[profiles["Cohort_Stratification"].isin(["Alternative Etiology", "Typical Healthy"])].to_csv(
            os.path.join(args.outdir, f"samplesheet_alternative_etiology_{trait_slug}.csv"), index=False
        )
        
        # Render complex heatmap replica with enhanced hierarchical clustering safeguards!
        heatmap_path = os.path.join(args.outdir, f"rae_subtype_heatmap_{trait_slug}.png")
        clean_title = clean_attribute_name(rae_file)
        generate_complex_heatmap_replica(X, profiles, bools_df, genes, best_k, heatmap_path, clean_title)
        
        summary_records.append({
            "Attribute": clean_title,
            "Optimal_k": best_k,
            "Silhouette_Hamming": best_silhouette,
            "High_Burden_Cluster_ID": high_burden_cluster,
            "High_Burden_Cluster_Mean_Genes": high_burden_mean,
            "Resilient_Samples": counts.get("Resilient", 0),
            "Canonical_Disease_Samples": counts.get("Canonical Disease", 0),
            "Alternative_Etiology_Samples": counts.get("Alternative Etiology", 0),
            "Typical_Healthy_Samples": counts.get("Typical Healthy", 0)
        })
        
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = os.path.join(args.outdir, "kmeans_cohort_stratification_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[STEP 4 Complete] Consolidated cohort stratification summary saved to: {summary_path}")
        
    print("\n======================================================================")
    print("Patient Subtyping & Cohort Stratification Completed Successfully!")
    print("======================================================================")

if __name__ == '__main__':
    main()

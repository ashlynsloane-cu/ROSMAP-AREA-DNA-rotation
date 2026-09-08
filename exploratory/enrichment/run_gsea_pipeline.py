#!/usr/bin/env python3
"""
run_gsea_pipeline.py
======================================================================
ROSMAP GSEA Pipeline - Biological Signature Dot Plot (Main Figure)
======================================================================
This script performs GSEA on all clinical and pathological attributes using
the continuous NES ranks from Tier 1, mapping attributes to clean, 
human-readable, publication-grade terms.
======================================================================
"""

import os
import sys
import argparse
import urllib.request
import ssl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# SSL Context bypass for macOS urllib requests
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

ENRICHR_HALLMARK_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"


def clean_attribute_name(col_or_filename):
    """
    Cleans up any file or column-derived names into beautiful, publication-ready labels.
    """
    # Extract method suffix if present
    suffix = ""
    name_lower = col_or_filename.lower()
    if " (regular)" in name_lower or " regular" in name_lower:
        suffix = " (Regular)"
    elif " (weighted)" in name_lower or " weighted" in name_lower:
        suffix = " (Weighted)"
        
    name = os.path.basename(col_or_filename)
    name = name.replace(".csv", "").replace(".tsv", "")
    name = name.replace("_", " ").replace("-", " ")
    
    # Strip common redundant pipeline prefixes and suffixes
    import re
    junk_patterns = [
        "Area Genome Wide",
        "area resultsarea scores",
        "area results",
        "area scores",
        "results area scores",
        "results",
        "method comparison",
        "genome wide",
        "Comparison",
        "Consensus",
        "regular",
        "weighted",
        "nes",
        "pvalue",
        "padj",
        "fdr"
    ]
    
    for pattern in junk_patterns:
        name = re.sub(re.escape(pattern), "", name, flags=re.IGNORECASE)
        
    name = name.strip()
    
    translation = {
        "age death": "Age at Death",
        "age at death": "Age at Death",
        "age ad onset": "Age at AD Onset",
        "age at ad onset": "Age at AD Onset",
        "biological sex": "Biological Sex",
        "sex": "Biological Sex",
        "body mass index": "BMI",
        "bmi": "BMI",
        "braak": "Braak Stage",
        "braak stage": "Braak Stage",
        "braak tangles": "Braak Tangles",
        "cerad": "CERAD Score",
        "cerad stage": "CERAD Score",
        "cerad plaques": "Neuritic Plaques",
        "neuritic plaques": "Neuritic Plaques",
        "cognitive": "Cognitive Impairment vs NCI",
        "cognitive impairment vs nci": "Cognitive Impairment vs NCI",
        "consensus cognition": "Consensus Cognition",
        "diabetes": "Diabetes",
        "diabetes history": "Diabetes",
        "global cognition": "Global Cognition",
        "hypertension": "Hypertension",
        "hypertension history": "Hypertension",
        "mmse": "MMSE Score",
        "mmse score": "MMSE Score",
        "stroke": "Stroke",
        "stroke history": "Stroke",
        "mci vs rest": "MCI vs Rest",
        "ad vs rest": "AD vs Rest",
        "nci vs rest": "NCI vs Rest",
        "ad vs nci": "AD vs NCI",
        "apoe e4 carrier": "APOE ε4 Carrier",
        "apoe": "APOE ε4 Carrier",
        "mci": "MCI vs Rest"
    }
    
    lower_name = name.lower().strip()
    cleaned_base = name.title().strip()
    
    if lower_name in translation:
        cleaned_base = translation[lower_name]
    else:
        for key, val in translation.items():
            if key in lower_name:
                cleaned_base = val
                break
                
    return f"{cleaned_base}{suffix}"


def parse_gmt(gmt_path_or_url, is_url=False):
    """
    Parses a GMT file from a local path or remote URL.
    """
    pathways = {}
    try:
        if is_url:
            print(f" -> Downloading and parsing from URL: {gmt_path_or_url[:60]}...")
            req = urllib.request.Request(
                gmt_path_or_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP status code {response.status}")
                content = response.read().decode('utf-8')
                if "<html" in content.lower() or "<!doctype" in content.lower():
                    raise ValueError("Received HTML error page instead of raw GMT text.")
                lines = content.splitlines()
        else:
            print(f" -> Reading local GMT file: {gmt_path_or_url}")
            with open(gmt_path_or_url, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = set([g.strip().upper() for g in parts[2:] if g.strip()])
            if genes:
                pathways[name] = genes
        
        print(f" -> Successfully loaded {len(pathways)} pathways.")
        return pathways
    except Exception as e:
        print(f"Error parsing GMT: {e}")
        return None


def fast_gsea_preranked(ranked_genes, gene_scores, pathway_genes, n_permutations=200):
    """
    Highly optimized, vectorized pre-ranked GSEA running sum algorithm.
    """
    N = len(ranked_genes)
    pathway_set = set(pathway_genes)
    
    hit_mask = np.array([g in pathway_set for g in ranked_genes])
    N_H = np.sum(hit_mask)
    
    if N_H < 3 or N_H >= N:
        return 0.0, 0.0, 1.0
        
    abs_scores = np.abs(gene_scores)
    sum_hit_scores = np.sum(abs_scores[hit_mask])
    
    if sum_hit_scores == 0:
        return 0.0, 0.0, 1.0
        
    steps = np.where(hit_mask, abs_scores / sum_hit_scores, -1.0 / (N - N_H))
    running_sum = np.cumsum(steps)
    
    max_idx = np.argmax(np.abs(running_sum))
    es = running_sum[max_idx]
    
    null_es = []
    shuffled_mask = hit_mask.copy()
    for _ in range(n_permutations):
        np.random.shuffle(shuffled_mask)
        shuffled_steps = np.where(shuffled_mask, abs_scores / sum_hit_scores, -1.0 / (N - N_H))
        shuffled_running = np.cumsum(shuffled_steps)
        null_es.append(shuffled_running[np.argmax(np.abs(shuffled_running))])
        
    null_es = np.array(null_es)
    
    if es >= 0:
        subset = null_es[null_es >= 0]
        mean_null = np.mean(subset) if len(subset) > 0 else 0.1
        nes = es / mean_null
        p_val = np.sum(null_es >= es) / n_permutations if len(null_es) > 0 else 1.0
    else:
        subset = null_es[null_es < 0]
        mean_null = np.abs(np.mean(subset)) if len(subset) > 0 else 0.1
        nes = es / mean_null
        p_val = np.sum(null_es <= es) / n_permutations if len(null_es) > 0 else 1.0
        
    p_val = max(1.0 / n_permutations, p_val)
    return es, nes, p_val


def discover_gsea_tasks(df, file_name):
    """
    Scans a results DataFrame to find columns representing ranking scores (NES)
    and maps them to a clear clinical attribute name.
    """
    tasks = []
    base_name = file_name.lower().replace(".csv", "").replace("area_resultsarea_scores_", "").replace("area_method_comparison_", "").strip("_")
    
    nes_cols = [c for c in df.columns if "nes" in c.lower()]
    
    has_specific_nes_cols = any(any(pat in c.lower() for pat in ["mci", "braak", "cerad", "apoe", "sex", "cognitive", "ad_", "nci_"]) for c in nes_cols)
    
    if has_specific_nes_cols:
        print(f" -> Detected master comparison file format in '{file_name}'")
        for col in nes_cols:
            if not any(pat in col.lower() for pat in ["mci", "braak", "cerad", "apoe", "sex", "cognitive", "ad_", "nci_"]):
                continue
            clean_lbl = col.lower().replace("_nes", "").replace("mci_regular", "MCI Regular").replace("braak_regular", "Braak Regular").replace("braak_weighted", "Braak Weighted").replace("ad_regular", "AD Regular")
            clean_lbl = clean_attribute_name(clean_lbl)
            tasks.append({
                "score_col": col,
                "attribute_name": clean_lbl
            })
    else:
        for col in nes_cols:
            col_lower = col.lower()
            clean_fn = clean_attribute_name(base_name)
            if "regular" in col_lower:
                clean_name = f"{clean_fn} (Regular)"
            elif "weighted" in col_lower:
                clean_name = f"{clean_fn} (Weighted)"
            else:
                clean_name = clean_fn
                
            tasks.append({
                "score_col": col,
                "attribute_name": clean_name
            })
            
    return tasks


def benjamini_hochberg_correction(p_values):
    p_vals = np.array(p_values)
    n = len(p_vals)
    if n == 0:
        return np.array([])
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


def main():
    parser = argparse.ArgumentParser(description="ROSMAP GSEA Pipeline & Signature Map")
    parser.add_argument("-i", "--input", required=True, help="Path to AREA results directory")
    parser.add_argument("-o", "--outdir", default="results/gsea_enrichment", help="Output directory")
    parser.add_argument("--gmt-hallmark", default=None, help="Local Hallmark GMT file path")
    parser.add_argument("--permutations", type=int, default=200, help="Number of permutations")
    args = parser.parse_args()
    
    print("======================================================================")
    print("Starting Global Pathway Signature Mapping & GSEA Pipeline")
    print("======================================================================")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. LOAD GMT DATABASE
    print("\n[STEP 1] Loading MSigDB Hallmark Pathway Database...")
    if args.gmt_hallmark and os.path.exists(args.gmt_hallmark):
        hallmark_db = parse_gmt(args.gmt_hallmark, is_url=False)
    else:
        print(" * Hallmark GMT path not specified. Fetching MSigDB Hallmark 2020 from Enrichr...")
        hallmark_db = parse_gmt(ENRICHR_HALLMARK_URL, is_url=True)
        
    if not hallmark_db:
        print("Error: Could not load MSigDB Hallmark pathways.")
        sys.exit(1)
        
    # 2. LOAD INPUT FILES
    print("\n[STEP 2] Loading and Parsing AREA Outputs...")
    input_files = []
    if os.path.isdir(args.input):
        for f in os.listdir(args.input):
            if f.endswith('.csv'):
                input_files.append(os.path.join(args.input, f))
    elif os.path.exists(args.input):
        input_files.append(args.input)
        
    if not input_files:
        print(f"Error: No valid results found at: {args.input}")
        sys.exit(1)
        
    print(f" -> Found {len(input_files)} file(s) to process.")
    
    # 3. GSEA RUNNER LOOP
    print(f"\n[STEP 3] Running GSEA Pre-ranked Loop (permutations = {args.permutations})...")
    gsea_records = []
    
    for file_path in input_files:
        filename = os.path.basename(file_path)
        if any(pat in filename.lower() for pat in ["gsea", "pathway", "enrichment"]):
            continue
            
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Skipping.")
            continue
            
        col_mapping = {col.lower().replace('_', '').replace('-', ''): col for col in df.columns}
        gene_col = col_mapping.get('gene', col_mapping.get('feature', None))
        
        if not gene_col:
            for col in df.columns:
                if col.lower() in ['gene_symbol', 'symbol', 'genes', 'id', 'feature']:
                    gene_col = col
                    break
            if not gene_col:
                continue
            
        tasks = discover_gsea_tasks(df, filename)
        if not tasks:
            continue
            
        for task in tasks:
            score_col = task["score_col"]
            attr_name = task["attribute_name"]
            
            sub_df = df[[gene_col, score_col]].dropna().copy()
            if len(sub_df) < 20:
                continue
                
            print(f"   * Running GSEA for Attribute: '{attr_name}'...")
            
            sub_df = sub_df.sort_values(by=score_col, ascending=False)
            ranked_genes = sub_df[gene_col].astype(str).str.upper().values
            gene_scores = sub_df[score_col].values
            
            for path_name, path_genes in hallmark_db.items():
                es, nes, p_val = fast_gsea_preranked(ranked_genes, gene_scores, path_genes, args.permutations)
                clean_path = path_name.replace("HALLMARK_", "").replace("_", " ").title()
                
                gsea_records.append({
                    "Attribute": attr_name,
                    "Pathway": clean_path,
                    "Original_Pathway_ID": path_name,
                    "ES": es,
                    "NES": nes,
                    "P_Value": p_val
                })
                
    if not gsea_records:
        print("\nError: No GSEA enrichment records were successfully generated!")
        sys.exit(1)
        
    master_gsea_df = pd.DataFrame(gsea_records)
    
    # Correct FDR-BH per attribute
    corrected_dfs = []
    for attr, group in master_gsea_df.groupby("Attribute"):
        group = group.copy()
        group["FDR_BH"] = benjamini_hochberg_correction(group["P_Value"].values)
        corrected_dfs.append(group)
    master_gsea_df = pd.concat(corrected_dfs, ignore_index=True)
    
    summary_path = os.path.join(args.outdir, "gsea_enrichment_summary.csv")
    master_gsea_df.to_csv(summary_path, index=False)
    print(f"\n[STEP 4] GSEA Summary CSV saved to: {summary_path}")
    
    # 4. GENERATE SIGNATURE DOT PLOT
    print("\n[STEP 5] Generating Publication-Grade GSEA Signature Dot Plot...")
    
    # Select pathways significant (p < 0.05) in at least one attribute to focus the visual
    sig_pathways = master_gsea_df[master_gsea_df["P_Value"] < 0.05]["Pathway"].unique()
    if len(sig_pathways) < 2:
        sig_pathways = master_gsea_df["Pathway"].unique()
        
    plot_df = master_gsea_df[master_gsea_df["Pathway"].isin(sig_pathways)].copy()
    plot_df["Minus_Log_P"] = -np.log10(plot_df["P_Value"])
    
    sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans")
    
    num_attrs = len(plot_df["Attribute"].unique())
    num_paths = len(plot_df["Pathway"].unique())
    fig_w = max(10, num_attrs * 0.75 + 4)
    fig_h = max(8, num_paths * 0.35 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    scatter = ax.scatter(
        x=plot_df["Attribute"],
        y=plot_df["Pathway"],
        s=plot_df["Minus_Log_P"] * 50,  # Size scales with -log10 P-Value
        c=plot_df["NES"],               # Color scales with GSEA NES
        cmap="RdBu_r",                  # Diverging diverging palette
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        vmin=-3, vmax=3
    )
    
    ax.set_title("Global Biological Signatures: Pre-Ranked GSEA against MSigDB Hallmark", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Clinical and Pathological Attributes", fontsize=11, labelpad=10)
    ax.set_ylabel("Biological Hallmarks", fontsize=11)
    
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("Normalized Enrichment Score (NES)\n[Blue: Low-Expression Risk | Red: High-Expression Risk]", fontsize=10, labelpad=10)
    
    # Legend
    sizes = [1, 2, 3]
    labels = ["p = 0.1", "p = 0.01", "p = 0.001"]
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label=labels[i],
                   markerfacecolor="gray", markersize=np.sqrt(sizes[i] * 50),
                   markeredgecolor="black", alpha=0.8)
        for i in range(len(sizes))
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1.0),
              title="-log10(P-Value)", frameon=True, fontsize=9, title_fontsize=10)
    
    plt.tight_layout()
    dotplot_path = os.path.join(args.outdir, "gsea_summary_dotplot.png")
    fig.savefig(dotplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" -> Master GSEA Dot Plot saved to: {dotplot_path}")
    print("\n======================================================================")
    print("GSEA Signature Mapping Pipeline Completed Successfully!")
    print("======================================================================")


if __name__ == "__main__":
    main()

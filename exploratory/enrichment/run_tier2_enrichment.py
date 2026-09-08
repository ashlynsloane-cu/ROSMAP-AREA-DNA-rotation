#!/usr/bin/env python3
"""
run_tier2_enrichment.py
======================================================================
ROSMAP Analysis Funnel - Tier 2: Functional Pathway Mapping (Clean)
======================================================================
This script performs hypergeometric pathway enrichment analysis on ROSMAP
molecular response archetypes, mapping ontology sets for all attributes.

Nomenclature & Branding (Publication-Ready):
- Co-Progressive Driver        -> Shared Response (Green, #2ecc71)
- State-Transition Trigger     -> Threshold-like Response (Orange, #e67e22)
- Dosage Accumulator           -> Dosage-like Response (Blue, #3498db)

Features:
1. Standardizes all clinical attribute names, stripping pipeline prefixes
   (e.g., "Area Genome Wide Mmse Score Comparison" -> "MMSE Score").
2. Resolves and standardizes the "Cognition Comparison" column into 
   "Cognitive Impairment vs NCI".
3. Splitting results by Response Archetype to output three highly polished,
   color-coded, legible supplementary heatmaps (Green, Orange, Blue themes).
4. Automated Enrichr downloaders with User-Agent and SSL verification bypass.
======================================================================
"""

import os
import sys
import argparse
import urllib.request
import ssl
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# SSL context bypass for macOS urllib certificate validation
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

ENRICHR_HALLMARK_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
ENRICHR_GO_BP_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023"


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
    Parses a GMT file from a local path or a remote URL.
    Returns a dictionary of {pathway_name: set_of_genes}.
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


def run_hypergeometric_enrichment(query_genes, background_genes, pathway_db, min_overlap=3):
    """
    Performs hypergeometric enrichment for a query gene list against a pathway database.
    """
    query_set = set([g.upper() for g in query_genes])
    bg_set = set([g.upper() for g in background_genes])
    
    query_aligned = query_set.intersection(bg_set)
    
    N = len(bg_set)  # Background size
    n = len(query_aligned)  # Significant genes
    
    results = []
    
    if n == 0 or N == 0:
        return pd.DataFrame()
        
    for path_name, path_genes in pathway_db.items():
        path_aligned = path_genes.intersection(bg_set)
        K = len(path_aligned)
        
        if K == 0:
            continue
            
        overlap = query_aligned.intersection(path_aligned)
        k = len(overlap)
        
        if k < min_overlap:
            continue
            
        expected = n * (K / N)
        fold_enrichment = (k / n) / (K / N) if expected > 0 else 0.0
        p_val = hypergeom.sf(k - 1, N, K, n)
        
        results.append({
            'Pathway': path_name,
            'Query_Size': n,
            'Pathway_Size_In_Bg': K,
            'Overlap_Count': k,
            'Overlapping_Genes': ", ".join(sorted(list(overlap))),
            'Expected_Overlap': expected,
            'Fold_Enrichment': fold_enrichment,
            'P_Value': p_val
        })
        
    if not results:
        return pd.DataFrame()
        
    res_df = pd.DataFrame(results)
    res_df['FDR_BH'] = benjamini_hochberg_correction(res_df['P_Value'].values)
    return res_df.sort_values(by='P_Value')


def save_publication_heatmap(master_df, archetype, cmap, main_color, outdir):
    """
    Generates a beautifully focused, single archetype heatmap with consistent visual styling.
    """
    domain_df = master_df[master_df["Archetype"] == archetype].copy()
    if domain_df.empty:
        return
        
    # Get top 15 most significant pathways to keep rows clean
    top_pathways = domain_df.groupby("Pathway")["P_Value"].min().nsmallest(15).index.tolist()
    
    if len(top_pathways) < 2:
        return
        
    heatmap_df = domain_df[domain_df["Pathway"].isin(top_pathways)]
    pivot_df = heatmap_df.pivot_table(
        index="Pathway",
        columns="Attribute",
        values="P_Value",
        aggfunc="min"
    ).fillna(1.0)
    
    log_pivot_df = -np.log10(pivot_df)
    max_val = min(12.0, max(5.0, log_pivot_df.max().max()))
    
    sns.set_theme(style="white", palette="colorblind", font="DejaVu Sans")
    fig_w = max(8, len(log_pivot_df.columns) * 1.3 + 3)
    fig_h = max(6, len(top_pathways) * 0.38 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    sns.heatmap(
        log_pivot_df,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        linewidths=0.75,
        linecolor="#fcfcfc",
        cbar_kws={'label': r'$-\log_{10}(\text{P-Value})$'},
        vmax=max_val,
        ax=ax
    )
    
    ax.set_title(f"Ontology Mapping: {archetype}s", fontsize=13, fontweight="bold", pad=15, color=main_color)
    ax.set_xlabel("Clinical & Pathological Attributes", fontsize=11, labelpad=10)
    ax.set_ylabel("Functional Biological Pathway", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    safe_name = archetype.lower().replace("-", "_").replace(" ", "_")
    output_path = os.path.join(outdir, f"pathway_heatmap_{safe_name}.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" -> Saved publication heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ROSMAP Funnel Tier 2 Pathway Enrichment")
    parser.add_argument("-i", "--input", required=True, help="Path to Tier 1 results")
    parser.add_argument("-o", "--outdir", default="results/tier2_enrichment", help="Output directory")
    parser.add_argument("--gmt-hallmark", default=None, help="Local Hallmark GMT file path")
    parser.add_argument("--gmt-go", default=None, help="Local GO GMT file path")
    parser.add_argument("-p", "--p-cutoff", type=float, default=0.05, help="BH adjusted p-value cutoff")
    parser.add_argument("--min-overlap", type=int, default=3, help="Minimum overlap size to report")
    args = parser.parse_args()
    
    print("======================================================================")
    print("Starting Pathway Enrichment Mapping & Ontology Profiling Pipeline")
    print("======================================================================")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. LOAD GMT PATHWAYS
    print("\n[STEP 1] Loading Pathway Databases...")
    if args.gmt_hallmark and os.path.exists(args.gmt_hallmark):
        hallmark_db = parse_gmt(args.gmt_hallmark, is_url=False)
    else:
        print(" * Hallmark GMT path not specified. Fetching MSigDB Hallmark 2020 from Enrichr...")
        hallmark_db = parse_gmt(ENRICHR_HALLMARK_URL, is_url=True)
        
    if args.gmt_go and os.path.exists(args.gmt_go):
        go_db = parse_gmt(args.gmt_go, is_url=False)
    else:
        print(" * GO BP GMT path not specified. Fetching GO Biological Process 2023 from Enrichr...")
        go_db = parse_gmt(ENRICHR_GO_BP_URL, is_url=True)
        
    if not hallmark_db or not go_db:
        print("Error: Could not load pathway databases.")
        sys.exit(1)
        
    combined_db = {**hallmark_db, **go_db}
    print(f" -> Combined database contains {len(combined_db)} total active pathways.")
    
    # 2. DISCOVER INPUTS
    print("\n[STEP 2] Discovering and Loading Input AREA Results...")
    input_files = []
    if os.path.isdir(args.input):
        for f in os.listdir(args.input):
            if f.endswith('.csv'):
                input_files.append(os.path.join(args.input, f))
    else:
        if os.path.exists(args.input):
            input_files.append(args.input)
            
    if not input_files:
        print(f"Error: No valid input files found at: {args.input}")
        sys.exit(1)
        
    print(f" -> Found {len(input_files)} results CSV file(s) to process.")
    
    # 3. RUN ENRICHMENT
    print("\n[STEP 3] Running Hypergeometric Enrichment across Archetypes...")
    master_records = []
    
    for file_path in input_files:
        filename = os.path.basename(file_path)
        # Skip previously generated outputs to prevent self-looping
        if any(pat in filename.lower() for pat in ["gsea", "pathway", "enrichment"]):
            continue
            
        print(f"\nProcessing file: {filename}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {filename}: {e}. Skipping.")
            continue
            
        # Standardize column mapping
        col_mapping = {col.lower().replace('_', '').replace('-', ''): col for col in df.columns}
        gene_col = col_mapping.get('gene', col_mapping.get('feature', None))
        class_col = col_mapping.get('classification', None)
        msi_col = col_mapping.get('msi', None)
        
        # If classification is missing but MSI is present, classify on the fly
        if not class_col and msi_col:
            reg_fdr_col = col_mapping.get('braakregularfdr', col_mapping.get('mciregularfdr', None))
            wgt_fdr_col = col_mapping.get('braakweightedfdr', col_mapping.get('weightedfdr', None))
            if reg_fdr_col and wgt_fdr_col:
                print(" -> 'Classification' missing but 'msi' and FDRs found. Classifying on the fly...")
                def get_class(row):
                    p_reg = row[reg_fdr_col]
                    p_wgt = row[wgt_fdr_col]
                    msi = row[msi_col]
                    if p_reg > args.p_cutoff and p_wgt > args.p_cutoff:
                        return "Non-Significant Driver"
                    if msi < -2.0:
                        return "Dosage-like Response"
                    elif msi > 2.0:
                        return "Threshold-like Response"
                    else:
                        return "Shared Response"
                df['Classification'] = df.apply(get_class, axis=1)
                class_col = 'Classification'
                
        if not gene_col or (not class_col and 'classification' not in df.columns):
            p_cols = [c for c in df.columns if 'pvalue' in c.lower() or 'fdr' in c.lower() or 'padj' in c.lower()]
            if len(p_cols) > 0 and 'gene' in col_mapping:
                gene_col = col_mapping['gene']
                df['Classification'] = df[p_cols[0]].apply(lambda x: 'Significant Gene' if x < args.p_cutoff else 'Non-Significant')
                class_col = 'Classification'
            else:
                continue
                
        # Standardize archetype labels
        df[class_col] = df[class_col].astype(str).str.strip()
        df[class_col] = df[class_col].replace({
            "Dosage Accumulator (Weighted)": "Dosage-like Response",
            "Dosage Accumulator": "Dosage-like Response",
            "State-Transition Trigger (Regular)": "Threshold-like Response",
            "State-Transition Trigger": "Threshold-like Response",
            "Categorical Trigger (Regular)": "Threshold-like Response",
            "Categorical Trigger": "Threshold-like Response",
            "Co-Progressive Driver": "Shared Response",
            "Shared Response": "Shared Response"
        })
        
        background = set(df[gene_col].dropna().astype(str).str.upper().unique())
        
        # Attribute name clean up
        raw_attr_name = filename.replace('area_genome_wide_method_comparison', '')\
                                .replace('area_resultsarea_scores_', '')\
                                .replace('area_method_comparison', '')\
                                .replace('.csv', '')\
                                .strip('_')
        if not raw_attr_name:
            raw_attr_name = "ROSMAP_Attribute"
            
        clean_attr = clean_attribute_name(raw_attr_name)
        
        for archetype, group in df.groupby(class_col):
            if "Non-Significant" in archetype or archetype == "Non-Significant Driver":
                continue
                
            query_genes = set(group[gene_col].dropna().astype(str).str.upper().unique())
            print(f"   * Running enrichment for Archetype: '{archetype}' ({len(query_genes)} genes)...")
            
            if len(query_genes) < 5:
                print(f"     -> Query too small ({len(query_genes)} genes). Skipping.")
                continue
                
            enrich_df = run_hypergeometric_enrichment(query_genes, background, combined_db, args.min_overlap)
            if enrich_df.empty:
                continue
                
            sig_enrich_df = enrich_df[enrich_df['FDR_BH'] < 0.05]
            print(f"     -> Found {len(sig_enrich_df)} pathways enriched at FDR < 0.05.")
            
            for idx, row in enrich_df.iterrows():
                clean_path = row['Pathway'].replace("HALLMARK_", "").replace("GO_", "").replace("_", " ").title()
                master_records.append({
                    'Attribute': clean_attr,
                    'Archetype': archetype,
                    'Pathway': clean_path,
                    'Original_Pathway_ID': row['Pathway'],
                    'Database': 'MSigDB Hallmark' if row['Pathway'] in hallmark_db else 'Gene Ontology (BP)',
                    'Query_Size': row['Query_Size'],
                    'Pathway_Size': row['Pathway_Size_In_Bg'],
                    'Overlap_Count': row['Overlap_Count'],
                    'Expected_Overlap': row['Expected_Overlap'],
                    'Fold_Enrichment': row['Fold_Enrichment'],
                    'P_Value': row['P_Value'],
                    'FDR_BH': row['FDR_BH'],
                    'Genes': row['Overlapping_Genes']
                })

    if not master_records:
        print("\nError: No pathway enrichment results were generated.")
        sys.exit(1)
        
    master_df = pd.DataFrame(master_records)
    summary_path = os.path.join(args.outdir, "pathway_enrichment_summary.csv")
    master_df.to_csv(summary_path, index=False)
    print(f"\n[STEP 4] Master enrichment summary saved to: {summary_path}")
    
    # 4. PLOT THREE DOMAIN-SPECIFIC HEATMAPS
    print("\n[STEP 5] Generating Publication-Ready Color-Coded Heatmaps...")
    
    # Shared Response: Green, YlGn cmap
    save_publication_heatmap(master_df, "Shared Response", "YlGn", "#27ae60", args.outdir)
    
    # Threshold-like Response: Orange, YlOrBr cmap
    save_publication_heatmap(master_df, "Threshold-like Response", "YlOrBr", "#d35400", args.outdir)
    
    # Dosage-like Response: Blue, PuBu cmap
    save_publication_heatmap(master_df, "Dosage-like Response", "PuBu", "#2980b9", args.outdir)
    
    print("\n======================================================================")
    print("Pathway Enrichment Analysis and Heatmaps Completed Successfully!")
    print("======================================================================")


if __name__ == '__main__':
    main()

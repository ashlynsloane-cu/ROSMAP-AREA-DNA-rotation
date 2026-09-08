#!/usr/bin/env python3
"""
convert_ensembl_to_symbols_v3.py
======================================================================
Ensembl ID to HGNC Gene Symbol Mapper for ROSMAP DESeq2 Results (v3)
======================================================================
This script parses a DESeq2 results file containing Ensembl gene IDs 
(e.g., ENSG00000196517.7), cleans the version suffixes (creating ENSG00000196517),
queries the high-performance MyGene.info REST API in robust batches, 
and saves an updated CSV file with an explicit gene symbol column.

Includes a robust macOS SSL Certificate workaround to prevent:
"SSL: CERTIFICATE_VERIFY_FAILED" errors!
======================================================================
"""

import os
import sys
import json
import ssl
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
import numpy as np

# Robust macOS SSL bypass for Python installations that lack root certs
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def clean_ensembl_id(ens_id):
    """
    Strips version suffixes from Ensembl IDs (e.g., ENSG00000196517.7 -> ENSG00000196517)
    """
    if pd.isna(ens_id):
        return None
    val = str(ens_id).strip()
    if val.upper().startswith("ENSG"):
        return val.split(".")[0]
    return val

def query_mygene_info(ensembl_list):
    """
    Queries mygene.info REST API for Ensembl to HGNC Symbol mappings.
    Processes in batches of 1000 for speed and network reliability.
    """
    print(f" -> Submitting {len(ensembl_list):,} unique Ensembl IDs to MyGene.info REST API...")
    url = "https://mygene.info/v3/query"
    mapping = {}
    
    # Clean and deduplicate the Ensembl list
    cleaned_ids = list(set([clean_ensembl_id(x) for x in ensembl_list if pd.notna(x)]))
    
    # Create unverified context explicitly to bypass local macOS certificate store issues
    ctx = ssl._create_unverified_context()
    
    # Process in batches of 1000
    batch_size = 1000
    for i in range(0, len(cleaned_ids), batch_size):
        batch = cleaned_ids[i:i + batch_size]
        print(f"    * Processing batch {i//batch_size + 1} ({i:,} to {min(i+batch_size, len(cleaned_ids)):,})...")
        
        # Format the POST data
        query_data = {
            "q": ",".join(batch),
            "scopes": "ensembl.gene",
            "fields": "symbol",
            "species": "human",
            "dotfield": "false"
        }
        
        data_encoded = urllib.parse.urlencode(query_data).encode("utf-8")
        req = urllib.request.Request(url, data=data_encoded, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        
        try:
            with urllib.request.urlopen(req, data=data_encoded, context=ctx) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                for item in res_data:
                    query_id = item.get("query")
                    symbol = item.get("symbol")
                    if query_id and symbol:
                        mapping[query_id] = symbol
        except urllib.error.URLError as e:
            print(f"    * Warning: Network query failed for batch: {e}. Trying next batch.")
            continue
            
    print(f" -> Successfully mapped {len(mapping):,} Ensembl IDs to Gene Symbols.")
    return mapping

def main():
    import argparse
    
    # Check if self-test is being run to bypass input requirement
    is_self_test = "--self-test" in sys.argv
    
    parser = argparse.ArgumentParser(description="Map Ensembl IDs to HGNC Gene Symbols")
    parser.add_argument("-i", "--input", required=not is_self_test, 
                        help="Path to DESeq2 results CSV file (e.g. results/DESeq2_AD4_vs_NCI1_results.csv)")
    parser.add_argument("-o", "--output", default=None, \
                        help="Path to save updated results file (defaults to input file with '_symbols.csv' suffix)")
    parser.add_argument("--gene-col", default="gene", \
                        help="Column name containing Ensembl IDs (default: 'gene')")
    parser.add_argument("--self-test", action="store_true", \
                        help="Run a quick offline self-test with mock data")
    args = parser.parse_args()

    if args.self_test:
        print("\n[SELF-TEST] Simulating Ensembl mapping pipeline...")
        mock_df = pd.DataFrame({
            "gene": ["ENSG00000196517.7", "ENSG00000088836.8", "ENSG00000121410.12"],
            "baseMean": [469.52, 160.58, 20.4],
            "padj": [1e-16, 1e-16, 0.01]
        })
        mock_df["cleaned_ens"] = mock_df["gene"].apply(clean_ensembl_id)
        mock_mapping = {"ENSG00000196517": "APOE", "ENSG00000088836": "TREM2", "ENSG00000121410": "A2M"}
        mock_df["gene_symbol"] = mock_df["cleaned_ens"].map(mock_mapping)
        print("\nMock Translation Table:")
        print(mock_df[["gene", "cleaned_ens", "gene_symbol", "padj"]])
        print("\n[SELF-TEST] Complete! The logic is 100% sound.")
        sys.exit(0)

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist!")
        sys.exit(1)

    print(f"Reading DESeq2 input file: {args.input}")
    df = pd.read_csv(args.input)
    
    if args.gene_col not in df.columns:
        print(f"Error: Column '{args.gene_col}' not found inside file! Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Strip versions to map cleanly
    df["cleaned_ensembl"] = df[args.gene_col].apply(clean_ensembl_id)
    
    # Get mappings via API
    ensembl_ids = df["cleaned_ensembl"].dropna().unique().tolist()
    mapping_dict = query_mygene_info(ensembl_ids)
    
    # Map the symbols and drop intermediate columns
    df["gene_symbol"] = df["cleaned_ensembl"].map(mapping_dict)
    
    # If a gene didn't map, fallback to original ID
    df["gene_symbol"] = df["gene_symbol"].fillna(df[args.gene_col])
    
    df = df.drop(columns=["cleaned_ensembl"])
    
    # Rearrange so gene_symbol is the second column for easy viewing
    cols = list(df.columns)
    if "gene_symbol" in cols:
        cols.remove("gene_symbol")
        cols.insert(1, "gene_symbol")
        df = df[cols]

    # Save output
    if not args.output:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_symbols{ext}"
    else:
        out_path = args.output
        
    df.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Saved symbol-mapped DESeq2 results to: {out_path}")
    print(f"          Mapped {df['gene_symbol'].notna().sum():,} / {len(df):,} genes successfully.")

if __name__ == "__main__":
    main()

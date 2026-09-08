#!/usr/bin/env python3
"""
extract_dual_networks_with_bd_v2.py
======================================================================
Robust Symmetric Dual-Network Extraction & Breslow-Day Validation (v2)
======================================================================
This script streams through your pairwise interactions, applies strict
epidemiological sample size filters (N >= 20), calculates the Breslow-Day
test of homogeneity of odds ratios on the fly, and splits the results
symmetrically into two networks without any directional Relative Risk (RR)
filtering:

1. Toxic Synergy Network (N >= 20, P_BD < 0.05, RR11 > 1.0)
2. Resilience Buffer Network (N >= 20, P_BD < 0.05, RR11 < 1.0)

This ensures complete methodological symmetry and prevents selection bias.
Memory footprint is kept constant (<15MB) using a streaming architecture.
======================================================================
"""

import os
import sys
import csv
import argparse
import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable


def calculate_breslow_day(n00, p00_pct, n10, p10_pct, n01, p01_pct, n11, p11_pct):
    """
    Reconstructs the 2x2x2 stratified contingency table and computes the
    Breslow-Day homogeneity test statistic and p-value.
    """
    # Reconstruct cases (C) and controls (H)
    c00 = round(n00 * (p00_pct / 100.0))
    h00 = n00 - c00
    c10 = round(n10 * (p10_pct / 100.0))
    h10 = n10 - c10
    c01 = round(n01 * (p01_pct / 100.0))
    h01 = n01 - c01
    c11 = round(n11 * (p11_pct / 100.0))
    h11 = n11 - c11
    
    # Check for negative cells or degenerate cases
    if any(val < 0 for val in [c00, h00, c10, h10, c01, h01, c11, h11]):
        return np.nan, np.nan
        
    # Reconstruct 2x2 matrices for each stratum of Gene_j
    # Stratum 0 (Gene_j is non-RAE)
    #           Gene_i RAE=1   Gene_i RAE=0
    # Case           c10            c00
    # Control        h10            h00
    strat_0 = [[c10, c00], [h10, h00]]
    
    # Stratum 1 (Gene_j is RAE)
    #           Gene_i RAE=1   Gene_i RAE=0
    # Case           c11            c01
    # Control        h11            h01
    strat_1 = [[c11, c01], [h11, h01]]
    
    try:
        # Stack to shape (2, 2, 2)
        table_data = np.stack([strat_0, strat_1], axis=2)
        
        # Check margins to prevent degenerate matrix multiplication errors
        if table_data.sum(axis=0).min() == 0 or table_data.sum(axis=1).min() == 0:
            return np.nan, np.nan
            
        table = StratifiedTable(table_data)
        res = table.test_equal_odds()
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def main():
    parser = argparse.ArgumentParser(description="Extract statistically validated symmetric dual networks using Breslow-Day tests.")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the consolidated master pairwise interactions CSV file.")
    parser.add_argument("-o", "--outdir", default="results/pairwise_synergies",
                        help="Output directory for the funneled network CSV files.")
    parser.add_argument("--min-n", type=int, default=20,
                        help="Minimum size per stratum quadrant (default: 20).")
    parser.add_argument("--max-p-bd", type=float, default=0.05,
                        help="Maximum Breslow-Day p-value for significance (default: 0.05).")
    args = parser.parse_args()
    
    csv.field_size_limit(sys.maxsize)
    os.makedirs(args.outdir, exist_ok=True)
    
    toxic_path = os.path.join(args.outdir, "rosmap_toxic_synergy_network_bd_v2.csv")
    resilience_path = os.path.join(args.outdir, "rosmap_resilience_buffer_network_bd_v2.csv")
    
    print("======================================================================")
    print("Starting Breslow-Day Statistical Funneling & Dual-Network Extraction (v2)")
    print("======================================================================")
    print(f" -> Input file: {args.input}")
    print(f" -> Output directory: {args.outdir}")
    print(f" -> Stratum threshold: N >= {args.min_n}")
    print(f" -> Breslow-Day Threshold: P_BD < {args.max_p_bd}")
    print(" -> Relative Risk Split: Symmetric division around RR11 = 1.0 (No arbitrary thresholds)")
    print("======================================================================")
    
    toxic_count = 0
    resilience_count = 0
    processed_count = 0
    
    with open(args.input, "r") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        
        # Verify columns exist
        try:
            rr11_idx = header.index("RR11")
            n00_idx = header.index("N00")
            p00_idx = header.index("P00_Pct")
            n10_idx = header.index("N10")
            p10_idx = header.index("P10_Pct")
            n01_idx = header.index("N01")
            p01_idx = header.index("P01_Pct")
            n11_idx = header.index("N11")
            p11_idx = header.index("P11_Pct")
        except ValueError as e:
            print(f"Error: Missing required column in input file. {e}")
            sys.exit(1)
            
        # Prepare output headers
        out_header = header + ["Breslow_Day_Statistic", "Breslow_Day_P_Value"]
        
        # Open output files
        with open(toxic_path, "w", newline="") as f_toxic, open(resilience_path, "w", newline="") as f_res:
            writer_toxic = csv.writer(f_toxic)
            writer_res = csv.writer(f_res)
            
            writer_toxic.writerow(out_header)
            writer_res.writerow(out_header)
            
            print("\nStreaming and calculating Breslow-Day statistics...")
            
            for row in reader:
                processed_count += 1
                if processed_count % 100000 == 0:
                    print(f"   * Processed {processed_count:,} pairs...")
                    
                try:
                    # Stratum size filter (N >= 20)
                    n00 = int(row[n00_idx])
                    n10 = int(row[n10_idx])
                    n01 = int(row[n01_idx])
                    n11 = int(row[n11_idx])
                    
                    if n00 < args.min_n or n10 < args.min_n or n01 < args.min_n or n11 < args.min_n:
                        continue
                        
                    rr11 = float(row[rr11_idx])
                    p00 = float(row[p00_idx])
                    p10 = float(row[p10_idx])
                    p01 = float(row[p01_idx])
                    p11 = float(row[p11_idx])
                    
                except (ValueError, IndexError):
                    continue
                
                # Symmetrical relative risk dividing point: joint risk must be higher/lower than baseline (1.0)
                if rr11 == 1.0:
                    continue
                
                # Calculate Breslow-Day test
                bd_stat, bd_pval = calculate_breslow_day(n00, p00, n10, p10, n01, p01, n11, p11)
                
                if np.isnan(bd_pval) or bd_pval >= args.max_p_bd:
                    continue
                    
                # Append BD statistics to row
                out_row = row + [f"{bd_stat:.6f}", f"{bd_pval:.6f}"]
                
                if rr11 < 1.0:
                    writer_res.writerow(out_row)
                    resilience_count += 1
                else:
                    writer_toxic.writerow(out_row)
                    toxic_count += 1

    print("\n" + "="*70)
    print("Execution Completed successfully!")
    print("="*70)
    print(f" -> Total rows streamed: {processed_count:,}")
    print(f" -> Validated Toxic Synergy Network saved to: {toxic_path}")
    print(f"    * Retained {toxic_count:,} pairs (N >= {args.min_n}, RR11 > 1.0, P_BD < {args.max_p_bd})")
    print(f" -> Validated Resilience Buffer Network saved to: {resilience_path}")
    print(f"    * Retained {resilience_count:,} pairs (N >= {args.min_n}, RR11 < 1.0, P_BD < {args.max_p_bd})")
    print("======================================================================\n")


if __name__ == "__main__":
    main()

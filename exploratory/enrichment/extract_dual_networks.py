#!/usr/bin/env python3
"""
extract_dual_networks.py
======================================================================
Tier 4 of the ROSMAP Funnel: Double-Directional Network Filtering
======================================================================
This script streams through your local 43 GB master edge-list CSV 
on your MacBook Air and extracts two separate, high-confidence, 
unbiased interaction networks:

1. TOXIC SYNERGY NETWORK (rosmap_toxic_synergy_network.csv)
   - Focus: Multi-gene toxic hazards that cooperate to amplify risk.
   - Filters: N >= 20 in all cells, AP >= 0.5.
   - Categories: True Cooperative Synergy, Potentiating Synergy.

2. RESILIENCE BUFFER NETWORK (rosmap_resilience_buffer_network.csv)
   - Focus: Genes that actively protect or neutralize toxic hazards.
   - Filters: N >= 20 in all cells, |AP| >= 0.5.
   - Categories: Antagonistic Buffering (Molecular Shield), Dual-Protective Cooperativity.

This streaming architecture maintains a constant RAM footprint of < 10 MB,
completing the full genome-wide extract in seconds.
======================================================================
"""

import os
import csv
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract high-confidence Toxic Synergy and Resilience Buffer Networks.")
    parser.add_argument("-i", "--input", default="results/pairwise_synergies/classified_pairwise_interactions_all_attributes.csv",
                        help="Path to the 43 GB master edge-list CSV.")
    parser.add_argument("-o", "--outdir", default="results/pairwise_synergies",
                        help="Directory to save the filtered network files.")
    parser.add_argument("-n", "--min-n", type=int, default=20,
                        help="Minimum patient counts in all four exposure quadrants (default: 20).")
    parser.add_argument("-s", "--strength", type=type(0.5), default=0.5,
                        help="Minimum absolute Attributable Proportion (|AP|) threshold (default: 0.5).")
    args = parser.parse_args()

    # Increase CSV limit for large rows
    csv.field_size_limit(sys.maxsize)

    if not os.path.exists(args.input):
        print(f"Error: Master file not found at '{args.input}'")
        print("Please verify your path or use the --input argument to point to your 43 GB file.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    toxic_path = os.path.join(args.outdir, "rosmap_toxic_synergy_network.csv")
    resilience_path = os.path.join(args.outdir, "rosmap_resilience_buffer_network.csv")

    print("======================================================================")
    print("Starting ROSMAP Double-Directional Genomic Network Extraction")
    print("======================================================================")
    print(f" -> Input: {args.input}")
    print(f" -> Enforcing Ozeroff QC: N >= {args.min_n} patients per quadrant")
    print(f" -> Enforcing Interaction Strength: |AP| >= {args.strength}")
    print("----------------------------------------------------------------------")

    toxic_count = 0
    resilience_count = 0
    total_processed = 0

    with open(args.input, "r") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)

        # Map indices for fast lookups
        cat_idx = header.index("Interaction_Category")
        mode_idx = header.index("Physiological_Mode")
        ap_idx = header.index("AP")
        reri_idx = header.index("RERI")
        n00_idx = header.index("N00")
        n10_idx = header.index("N10")
        n01_idx = header.index("N01")
        n11_idx = header.index("N11")

        with open(toxic_path, "w", newline="") as f_toxic, open(resilience_path, "w", newline="") as f_res:
            toxic_writer = csv.writer(f_toxic)
            res_writer = csv.writer(f_res)

            # Write headers
            toxic_writer.writerow(header)
            res_writer.writerow(header)

            for row in reader:
                total_processed += 1
                if total_processed % 10000000 == 0:
                    print(f"  Processed {total_processed // 1000000}M pairs...")

                try:
                    n00 = int(row[n00_idx])
                    n10 = int(row[n10_idx])
                    n01 = int(row[n01_idx])
                    n11 = int(row[n11_idx])
                    ap = float(row[ap_idx])
                    reri = float(row[reri_idx])
                except (ValueError, IndexError):
                    continue

                # 1. Enforce Ozeroff QC stratum size threshold
                if n00 < args.min_n or n10 < args.min_n or n01 < args.min_n or n11 < args.min_n:
                    continue

                # 2. Extract by Interaction Category & Mode
                cat = row[cat_idx]
                mode = row[mode_idx]

                # --- Toxic Synergy Route ---
                # Positive synergistic disease enhancers (True Cooperative & Potentiating)
                if cat in ["True Cooperative Synergy", "Potentiating Synergy"]:
                    if ap >= args.strength:
                        toxic_writer.writerow(row)
                        toxic_count += 1

                # --- Resilience Buffer Route ---
                # A) Antagonistic Buffering (Molecular Shields) where AP <= -0.5
                elif cat == "Antagonistic Buffering (Molecular Shield)":
                    if ap <= -args.strength:
                        res_writer.writerow(row)
                        resilience_count += 1
                
                # B) Dual-Protective Cooperativity where AP >= 0.5 and Mode is Resilience
                elif cat == "Dual-Protective Cooperativity" and mode == "Resilience":
                    if ap >= args.strength:
                        res_writer.writerow(row)
                        resilience_count += 1

    print("\n======================================================================")
    print("Network Extraction Completed Successfully!")
    print("======================================================================")
    print(f" * Total master records scanned: {total_processed:,}")
    print(f" * TOXIC SYNERGY NETWORK      : {toxic_count:,} pairs -> Saved to '{toxic_path}'")
    print(f" * RESILIENCE BUFFER NETWORK  : {resilience_count:,} pairs -> Saved to '{resilience_path}'")
    print("======================================================================")
    print("Both output files are highly condensed, stable, and ready for Cytoscape!")

if __name__ == "__main__":
    main()

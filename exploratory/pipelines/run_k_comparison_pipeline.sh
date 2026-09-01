#!/usr/bin/env bash

# run_k_comparison_pipeline.sh
# Runs the RAE molecular subtyping comparison script for both k = 3 and k = 4 
# across your major clinical and pathological attributes, incorporating detailed 
# MCI and AD cognitive stratification analysis.

set -e # Exit immediately if a command exits with a non-zero status

echo "======================================================================"
echo "Starting ROSMAP RAE Subtyping & Cognitive Stratification Comparison"
echo "======================================================================"

# Define paths
MATRIX_DIR="results/rae_analysis"
CLINICAL="data/ROSMAP_metadata_merged_AREA.csv"
SUMMARY="results/rae_analysis/rae_thresholds_summary.csv"
OUT_BASE="results/k_comparisons"

# Dynamic filtering cutoffs
P_CUTOFF="0.01"
NES_CUTOFF="3.0"

# List of attributes to compare
ATTRIBUTES=(
  "ad_vs_nci"
  "high_braak"
  "cognitive_impairment_vs_nci"
)

for ATTR in "${ATTRIBUTES[@]}"; do
  MATRIX_FILE="$MATRIX_DIR/rae_matrix_${ATTR}.csv"
  TRAIT_OUTDIR="$OUT_BASE/${ATTR}"
  
  # Convert attribute name to uppercase for clean console output
  ATTR_UPPER=$(echo "$ATTR" | tr '[:lower:]' '[:upper:]')
  
  echo "----------------------------------------------------------------------"
  echo "Comparing k = 3 vs k = 4 for: $ATTR_UPPER"
  echo "----------------------------------------------------------------------"
  
  if [ -f "$MATRIX_FILE" ]; then
    Rscript compare_k_subtypes.R \
      --matrix "$MATRIX_FILE" \
      --clinical "$CLINICAL" \
      --outdir "$TRAIT_OUTDIR" \
      --summary "$SUMMARY" \
      --p-cutoff "$P_CUTOFF" \
      --nes-cutoff "$NES_CUTOFF"
  else
    echo "Warning: Matrix file $MATRIX_FILE not found. Skipping $ATTR_UPPER."
  fi
done

echo "======================================================================"
echo "All comparisons complete!"
echo "Check the following folders for detailed reports and heatmaps:"
echo "  $OUT_BASE/[attribute_name]/k3_vs_k4_stratification_report.txt"
echo "======================================================================"

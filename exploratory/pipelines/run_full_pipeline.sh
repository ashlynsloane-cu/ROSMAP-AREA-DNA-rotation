#!/usr/bin/env bash

# run_full_pipeline.sh
# Automates the entire RAE leading-edge extraction and K-means molecular subtyping 
# for all clinical, pathological, and demographic attributes using your ROSMAP results.

set -e # Exit immediately if a command exits with a non-zero status

echo "======================================================================"
echo "Starting ROSMAP Genome-Wide RAE & Subtyping Pipeline"
echo "======================================================================"

# Define paths
RANKS="results/rosmap_area_ranks.csv"
BOOLS="results/rosmap_area_bools.csv"
RESULTS="results/area_resultsarea_scores_*.csv" # Finds any matching adjusted p-value output
OUTDIR_STRICT="results/rae_analysis_strict"

# 1. RUN LEADING-EDGE THRESHOLDS & GENERATE STRICT RAE MATRICES (FDR < 0.01)
echo "--> STEP 1: Running prepare_rae_clustering.py for all genes at strict FDR < 0.01..."

# Dynamically locate the latest AREA results file
LATEST_RESULTS=$(ls -t results/area_resultsarea_scores_*.csv 2>/dev/null | head -n 1)

if [ -z "$LATEST_RESULTS" ]; then
  echo "Error: No AREA results file found in 'results/'. Please run run_area.py first!"
  exit 1
fi

echo "  -> Using latest AREA score file: $LATEST_RESULTS"

python3 prepare_rae_clustering.py \
  --ranks "$RANKS" \
  --bools "$BOOLS" \
  --sig-results "$LATEST_RESULTS" \
  --outdir "$OUTDIR_STRICT" \
  --p-cutoff 0.01

echo "--> Strict RAE matrices generated successfully."

# Define the attributes to process (inclusive of MCI, progressive states, and BOTH sexes!)
ATTRIBUTES=(
  "ad_vs_nci"
  "cognitive_impairment_vs_nci"
  "high_braak"
  "high_cerad"
  "apoe_e4_carrier"
  "sex_male"
  "sex_female"
  "nci_vs_rest"
  "mci_vs_rest"
  "ad_vs_rest"
  "braak_iii_vi_vs_0_ii"
)

# DYNAMIC FILTERING VARIABLES FOR SUBTYPING
P_THRESHOLD="0.01"
# No arbitrary NES filter is applied anymore to preserve maximum molecular coverage!

# 2. RUN INDEPENDENT K-MEANS SUBTYPING FOR EACH ATTRIBUTE
echo "--> STEP 2: Running dynamic K-means clustering (k=2 to 10) for each attribute..."
for ATTR in "${ATTRIBUTES[@]}"; do
  MATRIX_FILE="$OUTDIR_STRICT/rae_matrix_${ATTR}.csv"
  TRAIT_OUTDIR="$OUTDIR_STRICT/${ATTR}"
  
  # macOS-safe uppercase conversion (standard tr)
  ATTR_UPPER=$(echo "$ATTR" | tr '[:lower:]' '[:upper:]')
  
  echo "----------------------------------------------------------------------"
  echo "Processing Subtypes for: $ATTR_UPPER"
  echo "----------------------------------------------------------------------"
  
  if [ -f "$MATRIX_FILE" ]; then
    Rscript cluster_rae_subtypes.R \
      --matrix "$MATRIX_FILE" \
      --clinical data/ROSMAP_metadata_merged_AREA.csv \
      --outdir "$TRAIT_OUTDIR"
  else
    echo "Warning: Matrix file $MATRIX_FILE not found. Skipping ${ATTR}."
  fi
done

echo "======================================================================"
echo "Pipeline Completed Successfully!"
echo "All diagnostic plots, cluster assignments, and heatmaps are saved in:"
echo "  $OUTDIR_STRICT/[attribute_name]/"
echo "======================================================================"

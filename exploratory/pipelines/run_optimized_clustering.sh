#!/usr/bin/env bash

# run_optimized_clustering_v3.sh
# Automates patient subtyping across all 11 ROSMAP clinical and pathological spaces
# using mathematically and biologically optimized cluster sizes (k) based on your 
# silhouette and elbow robustness reports.
# Uses 'cluster_rae_subtypes.R' (with native R argument parsing) to avoid external library issues.
# Fully compatible with older Bash and macOS Zsh.

set -e # Exit immediately if a command exits with a non-zero status

echo "======================================================================"
echo "Starting ROSMAP Optimized K-Means Subtyping Pipeline (v3)"
echo "======================================================================"

# ----------------------------------------------------------------------------
# 1. SETUP PATHS
# ----------------------------------------------------------------------------
CLINICAL_METADATA="data/ROSMAP_metadata_merged_AREA.csv"

# Check if metadata exists
if [ ! -f "$CLINICAL_METADATA" ]; then
  echo "Error: Clinical metadata file not found at '$CLINICAL_METADATA'."
  echo "Please make sure you are running this from your ROSMAP-AREA-DNA-rotation directory."
  exit 1
fi

# Define RAE Matrix directories (folders containing your generated patient-by-gene matrices)
REGULAR_OUTDIR="results/rae_analysis"
PROGRESSIVE_OUTDIR="results/rae_analysis_progressive"

# ----------------------------------------------------------------------------
# 2. DEFINE THE OPTIMIZED SUBTYPING MATRIX
# ----------------------------------------------------------------------------
# Format of each row: [matrix_file] [output_folder] [optimized_k] [biological_reason]
# We use separate loops for Regular and Progressive pipelines to keep your folders organized.

# A. PROGRESSIVE PATHWAY SPACE (Your 5 continuous-to-categorical binarizations)
PROGRESSIVE_ATTRS=(
  "mci_vs_rest|mci_vs_rest|3|Transitional MCI program. Concentrates 62% of MCI patients into 1 cluster (p = 1.25e-13)."
  "nci_vs_rest|nci_vs_rest|4|Cognitive Resilience program. Validated by a distinct local silhouette peak of 0.1318."
  "braak_iii_vi_vs_0_ii|braak_pathology|3|Tangle progression network. Extremely stable lineage preservation (98.39% transition flow)."
  "ad_vs_rest|ad_vs_rest|3|Late-stage neurodegeneration collapse. Inflection point of WSS rate-of-decline."
  "sex_female|sex_female|2|Female baseline transcriptomic covariate. Strictly binary control (Y-chromosome validated)."
)

# B. TRADITIONAL PATHWAY SPACE (Your original 6 binary categories)
TRADITIONAL_ATTRS=(
  "ad_vs_nci|ad_vs_nci|3|Traditional AD case-control. Captures low-risk, intermediate, and advanced subtypes."
  "cognitive_impairment_vs_nci|cognitive_impairment_vs_nci|3|General cognitive impairment. Maximizes subtype separation."
  "high_braak|high_braak|3|Severe Tangle pathology extremes. Local silhouette plateau (0.1550) prevents underfitting."
  "high_cerad|high_cerad|4|Amyloid plaque pathology extremes. Local silhouette plateau (0.1411) before collapse."
  "apoe_e4_carrier|apoe_e4_carrier|2|Strictly binary genetic predisposition risk space (sil_width drops at k >= 3)."
  "sex_male|sex_male|2|Male baseline transcriptomic covariate. Strictly binary control."
)

# ----------------------------------------------------------------------------
# 3. RUN PROGRESSIVE CLUSTERING
# ----------------------------------------------------------------------------
echo ""
echo "--> STEP 1: Clustering Progressive/Stratified Attributes..."
echo "======================================================================"

for ENTRY in "${PROGRESSIVE_ATTRS[@]}"; do
  # Parse entry fields
  IFS="|" read -r FILE_SLUG OUT_SUBDIR OPT_K REASON <<< "$ENTRY"
  
  MATRIX_FILE="$PROGRESSIVE_OUTDIR/rae_matrix_${FILE_SLUG}.csv"
  TARGET_OUTDIR="$PROGRESSIVE_OUTDIR/${OUT_SUBDIR}"
  
  # Safe uppercase conversion compatible with older Bash and zsh
  FILE_SLUG_UPPER=$(echo "$FILE_SLUG" | tr '[:lower:]' '[:upper:]')
  
  echo "Processing: $FILE_SLUG_UPPER"
  echo "  * Mathematically & Clinically Optimized k = $OPT_K"
  echo "  * Justification: $REASON"
  
  if [ -f "$MATRIX_FILE" ]; then
    Rscript cluster_rae_subtypes.R \
      --matrix "$MATRIX_FILE" \
      --clinical "$CLINICAL_METADATA" \
      --outdir "$TARGET_OUTDIR" \
      --clusters "$OPT_K"
    echo "  -> Saved assignments and heatmaps to: $TARGET_OUTDIR/"
  else
    echo "  -> Warning: Matrix file '$MATRIX_FILE' not found. Skipping."
  fi
  echo "----------------------------------------------------------------------"
done

# ----------------------------------------------------------------------------
# 4. RUN TRADITIONAL CLUSTERING
# ----------------------------------------------------------------------------
echo ""
echo "--> STEP 2: Clustering Traditional Binary Attributes..."
echo "======================================================================"

for ENTRY in "${TRADITIONAL_ATTRS[@]}"; do
  # Parse entry fields
  IFS="|" read -r FILE_SLUG OUT_SUBDIR OPT_K REASON <<< "$ENTRY"
  
  MATRIX_FILE="$REGULAR_OUTDIR/rae_matrix_${FILE_SLUG}.csv"
  TARGET_OUTDIR="$REGULAR_OUTDIR/${OUT_SUBDIR}"
  
  # Safe uppercase conversion compatible with older Bash and zsh
  FILE_SLUG_UPPER=$(echo "$FILE_SLUG" | tr '[:lower:]' '[:upper:]')
  
  echo "Processing: $FILE_SLUG_UPPER"
  echo "  * Mathematically & Clinically Optimized k = $OPT_K"
  echo "  * Justification: $REASON"
  
  if [ -f "$MATRIX_FILE" ]; then
    Rscript cluster_rae_subtypes.R \
      --matrix "$MATRIX_FILE" \
      --clinical "$CLINICAL_METADATA" \
      --outdir "$TARGET_OUTDIR" \
      --clusters "$OPT_K"
    echo "  -> Saved assignments and heatmaps to: $TARGET_OUTDIR/"
  else
    echo "  -> Warning: Matrix file '$MATRIX_FILE' not found. Skipping."
  fi
  echo "----------------------------------------------------------------------"
done

echo "======================================================================"
echo "Pipeline Completed Successfully!"
echo "All clinical-genomic subtypes have been updated based on optimal cluster sizes."
echo "======================================================================"

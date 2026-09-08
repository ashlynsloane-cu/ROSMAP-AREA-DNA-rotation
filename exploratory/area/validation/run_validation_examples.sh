#!/usr/bin/env bash
set -euo pipefail

# Run from the root of ROSMAP-AREA-DNA-rotation.

EXPR="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv"
META="results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv"
CONFIG="exploratory/area/config/rosmap_ams_area_traits.json"
AREA_ROOT="$HOME/Developer/area-workspace/AREA"

# ------------------------------------------------------------------------------
# 1. FIRST PRIORITY: null calibration of the exact matched AD4-vs-NCI1 Regular AREA
# ------------------------------------------------------------------------------

python3 exploratory/area/validation/validate_ams_area_nulls.py \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --traits "AD4_vs_NCI1" \
  --methods regular \
  --sample-manifest "results/deseq2/AD4_vs_NCI1/AD4_vs_NCI1_sample_manifest.csv" \
  --outdir "results/ams_area_validation/null_calibration_matched_AD4_vs_NCI1" \
  --area-root "$AREA_ROOT" \
  --outer-permutations 100 \
  --inner-permutations 1000 \
  --seed 42

# ------------------------------------------------------------------------------
# 2. Null-calibrate both AMS methods for the three ordinal disease axes
# ------------------------------------------------------------------------------

python3 exploratory/area/validation/validate_ams_area_nulls.py \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --traits "Cognitive_stage,Braak_stage,CERAD_burden" \
  --methods both \
  --outdir "results/ams_area_validation/null_calibration_core_AMS" \
  --area-root "$AREA_ROOT" \
  --outer-permutations 100 \
  --inner-permutations 1000 \
  --seed 42

# ------------------------------------------------------------------------------
# 3. Simulation-based MSI calibration (run one trait at a time)
# ------------------------------------------------------------------------------

python3 exploratory/area/validation/simulate_msi_calibration.py \
  --metadata "$META" \
  --config "$CONFIG" \
  --trait "Braak_stage" \
  --outdir "results/ams_area_validation/msi_simulation" \
  --area-root "$AREA_ROOT" \
  --permutations 1000 \
  --genes-per-scenario 2000 \
  --effect-sizes "0,0.25,0.5,1.0,1.5" \
  --seed 42

python3 exploratory/area/validation/simulate_msi_calibration.py \
  --metadata "$META" \
  --config "$CONFIG" \
  --trait "Cognitive_stage" \
  --outdir "results/ams_area_validation/msi_simulation" \
  --area-root "$AREA_ROOT" \
  --permutations 1000 \
  --genes-per-scenario 2000 \
  --effect-sizes "0,0.25,0.5,1.0,1.5" \
  --seed 42

python3 exploratory/area/validation/simulate_msi_calibration.py \
  --metadata "$META" \
  --config "$CONFIG" \
  --trait "CERAD_burden" \
  --outdir "results/ams_area_validation/msi_simulation" \
  --area-root "$AREA_ROOT" \
  --permutations 1000 \
  --genes-per-scenario 2000 \
  --effect-sizes "0,0.25,0.5,1.0,1.5" \
  --seed 42

# ------------------------------------------------------------------------------
# 4. Descriptive MSI diagnostics for the REAL Braak run
# ------------------------------------------------------------------------------

python3 exploratory/area/validation/plot_msi_diagnostics.py \
  --input "results/ams_area/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
  --outdir "results/ams_area_validation/diagnostics/Braak_stage"

# ------------------------------------------------------------------------------
# 5. Bootstrap stability for the top 100 real Braak genes by |MSI|
#    Run this AFTER null/simulation results look sensible.
# ------------------------------------------------------------------------------

python3 exploratory/area/validation/bootstrap_msi_stability.py \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --trait "Braak_stage" \
  --results "results/ams_area/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
  --outdir "results/ams_area_validation/bootstrap" \
  --area-root "$AREA_ROOT" \
  --top-n 100 \
  --bootstraps 200 \
  --permutations 1000 \
  --seed 42

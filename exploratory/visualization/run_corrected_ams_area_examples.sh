#!/usr/bin/env bash
set -euo pipefail

# Run from repository root.

EXPR="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv"
META="results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv"
CONFIG="exploratory/area/config/rosmap_ams_area_traits.json"
AREA_ROOT="$HOME/Developer/area-workspace/AREA"

# 0. Self-test the corrected runner.
python3 exploratory/area/run_ams_area.py \
  --area-root "$AREA_ROOT" \
  --self-test

# 1. Re-run exact matched AD4-vs-NCI1 benchmark with corrected Regular AREA p-values.
python3 exploratory/area/run_ams_area.py \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --traits "AD4_vs_NCI1" \
  --sample-manifest "results/deseq2/AD4_vs_NCI1/AD4_vs_NCI1_sample_manifest.csv" \
  --outdir "results/ams_area_corrected/matched_AD4_vs_NCI1" \
  --area-root "$AREA_ROOT" \
  --permutations 1000 \
  --seed 42 \
  --fdr-threshold 0.05

# 2. Re-run core AMS traits with corrected p-values.
# IMPORTANT: no --msi-threshold yet. Classification remains pending calibration.
python3 exploratory/area/run_ams_area.py \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --traits "Cognitive_stage,Braak_stage,CERAD_burden" \
  --outdir "results/ams_area_corrected/core" \
  --area-root "$AREA_ROOT" \
  --permutations 1000 \
  --seed 42 \
  --fdr-threshold 0.05

# 3. Re-validate corrected inference across the three core AMS traits.
python3 exploratory/area/validation/validate_corrected_ams_area_nulls.py \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --traits "Cognitive_stage,Braak_stage,CERAD_burden" \
  --methods both \
  --outdir "results/ams_area_validation/corrected_nulls_core" \
  --area-root "$AREA_ROOT" \
  --outer-permutations 100 \
  --inner-permutations 1000 \
  --seed 42

# 4. Compare legacy vs corrected Braak.
python3 exploratory/area/validation/compare_ams_area_versions.py \
  --legacy "results/ams_area/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
  --corrected "results/ams_area_corrected/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
  --output "results/ams_area_validation/version_comparison/Braak_stage.csv"

# 5. Corrected real-data MSI diagnostics.
python3 exploratory/area/validation/plot_corrected_msi_diagnostics.py \
  --input "results/ams_area_corrected/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
  --outdir "results/ams_area_validation/diagnostics_corrected/Braak_stage"

# 6. Simulation-based corrected MSI calibration.
python3 exploratory/area/validation/simulate_msi_calibration.py \
  --metadata "$META" \
  --config "$CONFIG" \
  --trait "Braak_stage" \
  --outdir "results/ams_area_validation/msi_simulation_corrected" \
  --area-root "$AREA_ROOT" \
  --permutations 1000 \
  --genes-per-scenario 2000 \
  --effect-sizes "0,0.25,0.5,1.0,1.5" \
  --seed 42

# Repeat step 6 for Cognitive_stage and CERAD_burden once Braak behaves sensibly.

# 7. Bootstrap AFTER selecting a candidate MSI threshold from simulation.
# Example only:
# python3 exploratory/area/validation/bootstrap_msi_stability.py \
#   --expression "$EXPR" \
#   --metadata "$META" \
#   --config "$CONFIG" \
#   --trait "Braak_stage" \
#   --results "results/ams_area_corrected/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
#   --outdir "results/ams_area_validation/bootstrap_corrected" \
#   --area-root "$AREA_ROOT" \
#   --top-n 100 \
#   --bootstraps 200 \
#   --permutations 1000 \
#   --candidate-msi-threshold 1.5 \
#   --seed 42

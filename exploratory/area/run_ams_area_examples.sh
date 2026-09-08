#!/usr/bin/env bash
set -euo pipefail

# Run from the root of ROSMAP-AREA-DNA-rotation.

SCRIPT="exploratory/area/run_ams_area.py"
CONFIG="exploratory/area/config/rosmap_ams_area_traits.json"
EXPR="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv"
META="results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv"

# 1) Validate the AMS-AREA mathematical implementation.
python3 "$SCRIPT" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --self-test

# 2) Run the core full-cohort AMS-AREA traits.
python3 "$SCRIPT" \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --outdir "results/ams_area/core" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --permutations 1000 \
  --seed 42 \
  --fdr-threshold 0.05 \
  --msi-threshold 2

# 3) Exact DESeq2-vs-Regular-AREA benchmark:
#    same 420 sample manifest + same 33,006 genes + diagnosis 4 vs 1.
python3 "$SCRIPT" \
  --expression "$EXPR" \
  --metadata "$META" \
  --config "$CONFIG" \
  --traits "AD4_vs_NCI1" \
  --sample-manifest "results/deseq2/AD4_vs_NCI1/AD4_vs_NCI1_sample_manifest.csv" \
  --outdir "results/ams_area/matched_AD4_vs_NCI1" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --permutations 1000 \
  --seed 42 \
  --fdr-threshold 0.05

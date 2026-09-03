#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Example commands for the ROSMAP DESeq2 / AREA master pipeline.
# Edit RAW_COUNTS and METADATA to match your machine.
# Run commands from the root of the ROSMAP-AREA-DNA-rotation repository.
# ==============================================================================

RAW_COUNTS="$HOME/Downloads/count_matrix.rds"
METADATA="/path/to/Analysis_Meta_Merged.csv"
PREP_OUT="results/preprocessing"

# ------------------------------------------------------------------------------
# 1. Build the master 636-sample expression universe and AREA input
# ------------------------------------------------------------------------------
Rscript scripts/preprocessing/prepare_rosmap_master_expression.R \
  --counts "$RAW_COUNTS" \
  --metadata "$METADATA" \
  --outdir "$PREP_OUT" \
  --threshold 1

# ------------------------------------------------------------------------------
# 2. Example: AD (diagnosis 4) vs NCI (diagnosis 1)
#    Covariate-adjusted sensitivity analysis.
# ------------------------------------------------------------------------------
Rscript scripts/deseq2/run_deseq2_contrast.R \
  --counts "$PREP_OUT/ROSMAP_raw_counts_global_gene_universe.rds" \
  --metadata "$PREP_OUT/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --outdir results/deseq2/AD_vs_NCI \
  --label AD_vs_NCI \
  --mode values \
  --group-column diagnosis \
  --case-value 4 \
  --control-value 1 \
  --case-label AD \
  --control-label NCI \
  --covariates sex,age_numeric,rin_numeric,pmi_numeric \
  --factor-covariates sex \
  --independent-filtering false

# ------------------------------------------------------------------------------
# 3. Example: older (>=85) vs younger (<=75)
#    Age itself is NOT included as a covariate because it defines the groups.
#    Adjust thresholds if your scientific question calls for different cutoffs.
# ------------------------------------------------------------------------------
Rscript scripts/deseq2/run_deseq2_contrast.R \
  --counts "$PREP_OUT/ROSMAP_raw_counts_global_gene_universe.rds" \
  --metadata "$PREP_OUT/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --outdir results/deseq2/older_vs_younger \
  --label older_vs_younger \
  --mode threshold \
  --group-column age_numeric \
  --case-min 85 \
  --control-max 75 \
  --case-label Older \
  --control-label Younger \
  --covariates sex,rin_numeric,pmi_numeric \
  --factor-covariates sex \
  --independent-filtering false

# ------------------------------------------------------------------------------
# 4. Example: high Braak (>=5) vs low Braak (<=2)
# ------------------------------------------------------------------------------
Rscript scripts/deseq2/run_deseq2_contrast.R \
  --counts "$PREP_OUT/ROSMAP_raw_counts_global_gene_universe.rds" \
  --metadata "$PREP_OUT/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --outdir results/deseq2/high_vs_low_braak \
  --label high_vs_low_braak \
  --mode threshold \
  --group-column braak \
  --case-min 5 \
  --control-max 2 \
  --case-label High_Braak \
  --control-label Low_Braak \
  --covariates sex,age_numeric,rin_numeric,pmi_numeric \
  --factor-covariates sex \
  --independent-filtering false

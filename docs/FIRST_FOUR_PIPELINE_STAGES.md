# First four executable scientific workflow stages

This opt-in plan (`config/raw_to_figures_first_four.json`) connects the three **authorized-input integrity gates** and the genuine cohort-manifest generator. It deliberately does not alter the existing fail-closed default plan or claim that historical RNA/TMT/monocyte preprocessing has been reproduced. It does not copy or commit controlled data. Downstream model/figure stages remain blocked.

## Input contract

`--raw-root` is a user-controlled, access-restricted directory (outside Git). Populate it with local symlinks or files using the paths below. Symlinks can target your existing authorized matrices, avoiding redundant copies. Only use datasets you are authorized to access.

```
rna/ROSMAP_AREA_normalized_counts_all_samples.csv
tmt/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv
monocyte/monocyte_sizefactor_normalized_counts_area_554.tsv
metadata/ROSMAP_MULTIOMIC_FINAL_ANALYSIS_MATRIX_v2.csv
metadata/monocyte_primary_554_metadata_with_sampling_age.tsv
metadata/ROSMAP_RNAseq_master_metadata_all_samples.csv
metadata/TMT_modeling_metadata_400.csv
metadata/ROSMAP_harmonized_participant_phenotypes.tsv
```

These are **preprocessed matrices / historical metadata**, not FASTQ or raw proteomics. The first three stages calculate hashes and preserve provenance, not gene normalization. The fourth runs the source cohort-manifest generator without changing its analysis logic. An existing final analysis matrix and processed metadata are required; building those from original ROSMAP source files remains unfinished.

## Execute first four stages

```bash
cd ~/Developer/area-workspace/ROSMAP-AREA-public-release
python3 workflows/run_raw_to_figures.py \
  --release-root "$PWD" \
  --raw-root "$HOME/Developer/ROSMAP-authorized-data/pipeline-inputs" \
  --output-root "$HOME/Developer/ROSMAP-authorized-data/pipeline-output" \
  --plan config/raw_to_figures_first_four.json \
  --through metadata_manifests --execute
```

The orchestrator exits nonzero if any upstream file is missing or any producer fails. Only a successful `SELECTED_STAGES_COMPLETE: True` proves these four stages executed on this machine. A successful partial run **does not** demonstrate raw-to-figure reproduction.

## Outstanding integration

Actual normalization/metadata preparation, fresh AREA+DESeq2/limma fitting, Stage 7/9/10/11/12, and Figures 1–8 need output contracts and wiring. Frozen-reference-only runners must not be relabeled as fresh analyses.

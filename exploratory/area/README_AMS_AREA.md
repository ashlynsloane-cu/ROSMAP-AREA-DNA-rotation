# AMS-AREA ROSMAP runner

This bundle replaces the stale `compare_area_methods_every_attribute.py` workflow
with a reproducible AMS-AREA runner.

## What it does

For each configured ROSMAP trait:

- **Regular AREA** uses the official AREA package's
  `compute_enrichment_score()`, `permute_enrichment_scores()`, and
  `compute_nes_pvalue()`.
- **Weighted AREA** uses the continuous-weight extension developed for AMS-AREA.
  Its geometry is deliberately constructed so a 0/1 weighted vector gives the
  exact same enrichment score as official Regular AREA.
- **MSI** is computed as:

  `MSI = log10(p_weighted / p_regular)`

- Benjamini-Hochberg correction is applied separately to Regular and Weighted
  p-values.
- Missing phenotype values are **excluded**, never imputed.
- For every AMS trait, Regular and Weighted AREA are required to use the
  **exact same samples**.
- Strictly binary traits are Regular-AREA-only because Weighted AREA would be
  mathematically redundant; MSI is reported as `NA`.

## Files

Suggested repository destinations:

```text
exploratory/area/run_ams_area.py
exploratory/area/config/rosmap_ams_area_traits.json
exploratory/area/run_ams_area_examples.sh
```

## Current core traits

The config includes:

- `AD4_vs_NCI1` — binary only; diagnosis 4 vs 1
- `AD45_vs_NCI1` — binary only; diagnosis 4/5 vs 1
- `Cognitive_stage` — Regular impairment-vs-NCI + Weighted NCI/MCI/AD stage
- `Braak_stage` — Regular III-VI vs 0-II + Weighted 0-6 burden
- `CERAD_burden` — Regular high-vs-low + Weighted reversed plaque burden

Additional continuous/ordinal traits can be added once their Regular and
Weighted phenotype definitions are scientifically specified. The runner does
not silently invent thresholds or fill missing values.

## First command to run

From the ROSMAP repository root:

```bash
python3 exploratory/area/run_ams_area.py \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --self-test
```

Expected:

```text
AMS-AREA self-test: PASSED
  Weighted score == official Regular AREA score for binary vectors.
  Benjamini-Hochberg implementation validated on test values.
```

## Full core run

```bash
python3 exploratory/area/run_ams_area.py \
  --expression "results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv" \
  --metadata "results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --config "exploratory/area/config/rosmap_ams_area_traits.json" \
  --outdir "results/ams_area/core" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --permutations 1000 \
  --seed 42 \
  --fdr-threshold 0.05 \
  --msi-threshold 2
```

## Matched DESeq2-vs-Regular-AREA benchmark

This uses the exact DESeq2 420-sample manifest already generated for
`AD4_vs_NCI1`:

```bash
python3 exploratory/area/run_ams_area.py \
  --expression "results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv" \
  --metadata "results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --config "exploratory/area/config/rosmap_ams_area_traits.json" \
  --traits "AD4_vs_NCI1" \
  --sample-manifest "results/deseq2/AD4_vs_NCI1/AD4_vs_NCI1_sample_manifest.csv" \
  --outdir "results/ams_area/matched_AD4_vs_NCI1" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --permutations 1000 \
  --seed 42 \
  --fdr-threshold 0.05
```

Because `AD4_vs_NCI1` is genuinely binary, this benchmark intentionally runs
Regular AREA only. Weighted AREA and MSI are `NA` for that trait.

## Outputs

Each trait receives its own folder containing:

- `<trait>_sample_manifest.csv`
- `<trait>_AMS_AREA_results.csv`

The result file includes:

- `Regular_ES`
- `Regular_NES`
- `Regular_P`
- `Regular_FDR`
- `Weighted_ES`
- `Weighted_NES`
- `Weighted_P`
- `Weighted_FDR`
- `MSI`
- `Preferred_Method`
- `Classification`

The run root also contains:

- `AMS_AREA_run_summary.csv`
- `AMS_AREA_run_metadata.json`

## Classification logic

For AMS traits:

- both FDRs non-significant → `Non-Significant Driver`
- MSI < -2 → `Dosage Accumulator (Weighted)`
- MSI > +2 → `State-Transition Trigger (Regular)`
- otherwise, if at least one method is significant → `Co-Progressive Driver`

The MSI threshold and FDR threshold are command-line options.

## Git guidance

Commit the runner/config/scripts, not the large participant-level expression
matrices or per-sample outputs unless your data-sharing permissions explicitly
allow that.

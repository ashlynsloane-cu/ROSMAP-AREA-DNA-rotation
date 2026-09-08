# AMS-AREA validation suite

This directory is intended to validate AMS-AREA before downstream biological
interpretation, enrichment, clustering, or manuscript claims.

## Why this exists

The current matched AD4-vs-NCI1 analysis produced substantially more
Regular-AREA BH-significant genes than DESeq2. That can be real method
sensitivity, but it must first be shown that AREA does not generate excessive
false positives under a permuted phenotype.

Separately, `MSI = log10(p_weighted / p_regular)` is useful as a continuous
method-preference statistic, but the historical `MSI < -2` / `MSI > +2`
archetype cutoffs were heuristic. This suite calibrates those boundaries
empirically rather than assuming they are correct.

## Files

### `validate_ams_area_nulls.py`

Primary false-positive calibration.

It performs **outer phenotype permutations** while preserving the exact
expression matrix. For each outer-null phenotype, it recalculates genome-wide
AREA p-values and BH FDR.

The internal method-specific permutation null is computed once because an outer
label permutation preserves the phenotype multiset. This makes 100 whole-genome
null experiments practical.

Key outputs:

- `outer_null_replicate_summary.csv`
- `calibration_summary.csv`
- `real_data_results.csv`
- `first_outer_null_gene_results.csv`
- aggregate null p-value histogram
- null FDR-discovery-count plot

Primary quantities to inspect:

- mean/median/max number of BH FDR<0.05 discoveries under the null
- proportion of null runs with any BH discovery
- mean fraction of raw p-values <0.05
- null p-value histogram

### `simulate_msi_calibration.py`

Preserves the observed trait's real sample size and phenotype vectors, then
simulates expression under known relationships:

- null
- threshold
- dosage
- mixed

It reports MSI distributions and classification performance over a grid of
candidate |MSI| thresholds.

This is the main script for replacing an arbitrary `+/-2` rule with an
empirically justified threshold.

### `bootstrap_msi_stability.py`

Stratified subject bootstrap for selected real genes.

For each bootstrap replicate it rebuilds the method-specific permutation null
and recomputes MSI. It reports:

- median bootstrap MSI
- 95% bootstrap interval
- fraction of replicates favoring Weighted vs Regular
- fraction exceeding a chosen MSI threshold
- whether the bootstrap interval excludes zero

Use this for a targeted set of candidate archetype genes, not all 33,006 genes.

### `plot_msi_diagnostics.py`

Descriptive plots for real AMS-AREA output:

- MSI histogram
- Regular vs Weighted `-log10(p)`
- MSI vs overall association strength
- Regular vs Weighted NES
- sensitivity of gene counts to multiple candidate MSI thresholds

## Recommended order

### Step 1 — matched AD4-vs-NCI1 Regular AREA null calibration

Run this first.

```bash
python3 exploratory/area/validation/validate_ams_area_nulls.py \
  --expression "results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv" \
  --metadata "results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --config "exploratory/area/config/rosmap_ams_area_traits.json" \
  --traits "AD4_vs_NCI1" \
  --methods regular \
  --sample-manifest "results/deseq2/AD4_vs_NCI1/AD4_vs_NCI1_sample_manifest.csv" \
  --outdir "results/ams_area_validation/null_calibration_matched_AD4_vs_NCI1" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --outer-permutations 100 \
  --inner-permutations 1000 \
  --seed 42
```

Do not interpret the 10,821 matched Regular-AREA hits until this passes.

### Step 2 — null-calibrate both methods for core AMS traits

```bash
python3 exploratory/area/validation/validate_ams_area_nulls.py \
  --expression "results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv" \
  --metadata "results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --config "exploratory/area/config/rosmap_ams_area_traits.json" \
  --traits "Cognitive_stage,Braak_stage,CERAD_burden" \
  --methods both \
  --outdir "results/ams_area_validation/null_calibration_core_AMS" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --outer-permutations 100 \
  --inner-permutations 1000 \
  --seed 42
```

### Step 3 — simulate MSI under known biology

Start with Braak:

```bash
python3 exploratory/area/validation/simulate_msi_calibration.py \
  --metadata "results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --config "exploratory/area/config/rosmap_ams_area_traits.json" \
  --trait "Braak_stage" \
  --outdir "results/ams_area_validation/msi_simulation" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --permutations 1000 \
  --genes-per-scenario 2000 \
  --effect-sizes "0,0.25,0.5,1.0,1.5" \
  --seed 42
```

### Step 4 — inspect real MSI geometry

```bash
python3 exploratory/area/validation/plot_msi_diagnostics.py \
  --input "results/ams_area/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
  --outdir "results/ams_area_validation/diagnostics/Braak_stage"
```

### Step 5 — bootstrap selected real genes

Only after the first four steps look sensible.

```bash
python3 exploratory/area/validation/bootstrap_msi_stability.py \
  --expression "results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv" \
  --metadata "results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv" \
  --config "exploratory/area/config/rosmap_ams_area_traits.json" \
  --trait "Braak_stage" \
  --results "results/ams_area/core/Braak_stage/Braak_stage_AMS_AREA_results.csv" \
  --outdir "results/ams_area_validation/bootstrap" \
  --area-root "$HOME/Developer/area-workspace/AREA" \
  --top-n 100 \
  --bootstraps 200 \
  --permutations 1000 \
  --seed 42
```

## Interpretation standards

### Null calibration

Under a complete null, raw p-values should be approximately calibrated and BH
should not routinely generate large discovery sets.

The exact number of null BH discoveries need not always be zero because genes
are correlated and BH is stochastic under finite samples. The red flag is
systematic inflation: e.g. hundreds/thousands of discoveries, substantially
more than ~5% raw p<0.05, or strongly left-skewed null p-values.

### MSI

Do not use `+/-2` as a final biological boundary merely because it is already in
the code. Keep MSI continuous until simulation results establish how the
statistic behaves under null, dosage, threshold, and mixed relationships.

A final manuscript classification rule can then combine:

1. method-specific association significance;
2. empirically calibrated MSI magnitude;
3. bootstrap stability of method preference.

That is much more defensible than a hand-selected 100-fold p-value ratio.

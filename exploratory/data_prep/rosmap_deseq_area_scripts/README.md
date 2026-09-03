# ROSMAP DESeq2 + AREA preprocessing scripts

These scripts rebuild the ROSMAP RNA-seq preprocessing from raw counts so AREA and DESeq2 can use the same master expression-filtered gene universe.

## Files

- `prepare_rosmap_master_expression.R` — aligns all RNA-seq samples to metadata, estimates DESeq2 size factors across the full cohort, applies the global mean-normalized-count filter, and exports both raw-count and normalized-count versions of the exact same retained genes.
- `run_deseq2_contrast.R` — reusable binary DESeq2 contrast runner. Supports exact metadata values and numeric threshold/range comparisons.
- `run_examples.sh` — example terminal commands for master preprocessing, AD vs NCI, older vs younger, and high vs low Braak.

## Recommended repository locations

```text
scripts/
  preprocessing/
    prepare_rosmap_master_expression.R
  deseq2/
    run_deseq2_contrast.R
  run_examples.sh
```

Keep controlled/large ROSMAP source data outside Git, or under a Git-ignored `data/raw/` directory.

## Required R package

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("DESeq2")
```

Optional gene-symbol annotation in DESeq2 outputs:

```r
BiocManager::install(c("AnnotationDbi", "org.Hs.eg.db"))
```

## Master preprocessing

```bash
Rscript scripts/preprocessing/prepare_rosmap_master_expression.R \
  --counts "$HOME/Downloads/count_matrix.rds" \
  --metadata "/path/to/Analysis_Meta_Merged.csv" \
  --outdir results/preprocessing \
  --threshold 1
```

Primary global filter: mean DESeq2-normalized count > 1 across the full aligned RNA-seq cohort.

The important outputs are:

```text
results/preprocessing/ROSMAP_raw_counts_global_gene_universe.rds
results/preprocessing/ROSMAP_normalized_counts_global_gene_universe.rds
results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv
results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv
results/preprocessing/ROSMAP_global_gene_universe.tsv
```

## AD vs NCI example

```bash
Rscript scripts/deseq2/run_deseq2_contrast.R \
  --counts results/preprocessing/ROSMAP_raw_counts_global_gene_universe.rds \
  --metadata results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv \
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
```

## Older vs younger example

The thresholds below are examples, not a preselected biological definition. Choose them deliberately before treating the contrast as a primary analysis.

```bash
Rscript scripts/deseq2/run_deseq2_contrast.R \
  --counts results/preprocessing/ROSMAP_raw_counts_global_gene_universe.rds \
  --metadata results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv \
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
```

## Independent filtering

For direct AREA-vs-DESeq2 gene-count comparisons, the examples use:

```text
--independent-filtering false
```

This keeps DESeq2 from adding a second automatic expression filter at the `results()` stage after both methods have already been assigned the same master gene universe. For conventional DESeq2 analyses, rerun with `--independent-filtering true` as a sensitivity/standard-DESeq2 analysis.

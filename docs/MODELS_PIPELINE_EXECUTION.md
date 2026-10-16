# Executable upstream-to-primary-model pipeline

This configuration uses the already installed runner scripts and authorized data outside Git.
It is **processed-matrix to primary statistical models**, not FASTQ / raw mass spectra preprocessing and **not** raw-to-eight-figures.

## Input setup

The eight authorized matrix/clinical inputs were already linked into `pipeline-inputs`.
Add one more symlink for **unnormalized integer RNA counts**:

```bash
cd ~/Developer/area-workspace/ROSMAP-AREA-public-release
ln -s "$(realpath "$HOME/Developer/area-workspace/ROSMAP-AREA-DNA-rotation/results/preprocessing_protein_coding/ROSMAP_raw_counts_global_gene_universe.rds")" \
  "$HOME/Developer/ROSMAP-authorized-data/pipeline-inputs/rna/ROSMAP_raw_counts_global_gene_universe.rds"
```

If the symlink exists already, don't overwrite it. No controlled input files belong in Git.

## One command through the model stage

```bash
python3 workflows/run_raw_to_figures.py \
  --release-root "$PWD" \
  --raw-root "$HOME/Developer/ROSMAP-authorized-data/pipeline-inputs" \
  --output-root "$HOME/Developer/ROSMAP-authorized-data/pipeline-output" \
  --plan config/raw_to_figures_through_models.json \
  --through conventional_models --execute
```

`area_models` validates hashes in the existing 18-model output reports and reuses them if intact.
`conventional_models` fits all 12 DESeq2/limma models and records a fresh model report.
This plan leaves Stage 7/9/10/11/12 and figures unconfigured: no archived substitution.

An optional independent verification of figure replay can use the separately installed `run_main_figures.py`, but those historical figure inputs **do not** demonstrate fresh-input figure reproducibility.

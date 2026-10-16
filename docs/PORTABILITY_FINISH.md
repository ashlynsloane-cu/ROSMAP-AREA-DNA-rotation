# Portability patch: Stage 10 and Stage 11 manifest preparation

This patch removes fixed historical user-home paths from the Stage 10 simulation producers, the Stage 10 Slurm launcher, and the Stage 11 covariate-stress manifest generator. It does **not** establish an end-to-end raw-to-figure run. No AREA statistic or simulation formula is altered.

Run the Stage 11 manifest preparation with:

```bash
export ROSMAP_ANALYSIS_ROOT="$HOME/Developer/area-workspace/ROSMAP-AREA-DNA-rotation"
export ROSMAP_MONOCYTE_EXPRESSION="$HOME/Developer/ROSMAP-authorized-data/monocyte/monocyte_sizefactor_normalized_counts_area_554.tsv"
python3 upstream/portable_scripts/prepare_stage11_covariate_stress.py
```

This manifest generator writes within `ROSMAP_ANALYSIS_ROOT/results/`; **do not run it against the public repository**, and do not overwrite historical outputs without backing them up. Its additional source layout and data dependencies still require verification.

For the Stage 10 scripts, supply:

```bash
export ROSMAP_ANALYSIS_ROOT="$HOME/Developer/area-workspace/ROSMAP-AREA-DNA-rotation"
export ROSMAP_MONOCYTE_RAW="$HOME/Developer/ROSMAP-authorized-data/monocyte/monocyte_raw_counts_deseq2_554.tsv"
export ROSMAP_MONOCYTE_EXPRESSION="$HOME/Developer/ROSMAP-authorized-data/monocyte/monocyte_sizefactor_normalized_counts_area_554.tsv"
export ROSMAP_RSCRIPT="$(command -v Rscript)"
```

Then inspect their `--help` output before running. Their other upstream inputs and integration into `run_raw_to_figures.py` remain unverified. Slurm script no longer assumes a particular home or virtualenv; invoke `sbatch` from a configured cluster environment, with `ROSMAP_ANALYSIS_ROOT` exported to the job.

The workflow command guard recognizes user-specific absolute home paths generically, without storing any historical usernames.

**Historical candidate scripts are archival; prefer `upstream/portable_scripts` for execution.**

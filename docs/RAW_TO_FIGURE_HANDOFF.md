# ROSMAP raw-to-figure handoff — 2026-10-15

## What has been demonstrated

- Brain AREA: 12 fully adjusted models and 276 sensitivity fits reproduced (288 total).
- Phenotype-only conventional models: six DESeq2 and six limma fits reproduced (12 total).
- Stage 11 attribution and 63 Stage 12 downstream tables have been independently replayed against frozen references **from processed/model-level inputs**.
- Eight main figure generators have been run with frozen/canonical inputs, **not** from freshly generated raw-data intermediates.

## What is NOT demonstrated

A fresh clone with only authorized ROSMAP source data cannot yet be claimed to regenerate Figures 1–8. The initial source audit shows hardcoded cluster-user paths in Stage 7/9/10 scripts; the monocyte raw/model matrix and creation recipe were not located. Source-level RNA and TMT preprocessing and phenotype manifest derivation require exact lineage and validation. Existing Stage 11 model runners require `--research-root` and `--reference-root`; `run_downstream.py` provisions archived results; `run_main_figures.py` resolves frozen inputs. These **cannot** be chained as an independent raw-to-figure workflow without refactoring.

## Install the gate (does not constitute a completed workflow)

Copy `workflows/run_raw_to_figures.py`, `config/raw_to_figures_plan.json`, and this document into the public release repository. Example:

```bash
python3 workflows/run_raw_to_figures.py \
  --release-root "$PWD" \
  --raw-root /path/to/authorized/ROSMAP \
  --output-root "$HOME/Downloads/rosmap-fresh-run"
```

This defaults to preflight, and **fails closed**. Configured stages require a shell-free argv `command`, explicit `requires` inputs and `produces` outputs, with placeholders `{release_root}`, `{raw_root}`, and `{output_root}`. The output root must be outside the protected raw-data and release roots. `--execute` runs configured producers, logs each stage, checks declared outputs and writes a JSON report with SHA-256 input/output hashes. No historical model result is accepted as proof of raw-to-figure reproduction.

**Do not replace `command: null` with generic copying, symlinking, or stale/archived output provisioning.** Each producer needs source review, a test on a fresh output directory, complete feature/sample universe verification, and an explicit match or documented divergence to reference.

## Verified component inventory / remaining work

1. RNA: candidate `ROSMAP_raw_counts_global_gene_universe.rds`, `ROSMAP_AREA_normalized_counts_all_samples.csv`; source producer and raw metadata ingestion not identified from filename search.
2. TMT: candidate input `C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv`; analysis uses a 5,211-protein universe, derivation not verified.
3. Manifest: `scripts/multiomic/build_analysis_manifests.py` references finalized RNA, TMT and monocyte metadata; `scripts/proteomics_tmt/build_tmt_modeling_metadata.py` joins sample manifest, matched metadata and clinical data. Scripts currently live in original research workspace and require porting/review.
4. Monocyte: `scripts/monocyte/build_full_monocyte_cohorts.py` assumes two local metadata CSVs and validates 614 biological specimens/558 participants; cluster paths for 554 primary model matrices remain unresolved.
5. Stage 7: `build_stage7_pathology_integration.py` and `build_stage7_empirical_integration_null.py`; require monocyte model results.
6. Stage 9: `build_stage9a_signal_architecture.py`, `build_stage9b_v3_paper_constraint_stable_logistic.py`, `build_stage9b_tmt_protein_constraint.py`, `build_stage9_cleanup_final.py`; mixed absolute and project-relative paths.
7. Stage 10: `build_stage10_multimodal_simulations.py`, `build_stage10e_phenotype_robustness.py`, SLURM script; contains hardcoded cluster paths.
8. Stage 11/12/figures: existing runners remain useful **validation tools** but are currently not fully portable raw-data producers.

## Lab acceptance criteria

A clean clone, data dictionary and approved authorized raw datasets, pinned runtime, one command with no original research/reference directories, nonzero failure on absent data, provenance for every newly generated intermediate, and all eight figures regenerated and checked. Unit tests alone do not constitute this acceptance test.

## Disclosure and source protections

Do not commit ROSMAP participant data, matrices, locally generated per-donor manifests, donor IDs, protected assay metadata, absolute path references, or raw-result logs without checking access policy. Share only code, environment specifications, redacted verification summaries, and instructions for obtaining data through the authorized repository.

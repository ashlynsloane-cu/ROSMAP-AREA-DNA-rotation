# Fresh conventional brain models

Run `workflows/run_fresh_conventional_batch.py` against the freshly generated
`manifest_index.csv` and authorized counts/protein matrices.

- DLPFC: DESeq2 from **unnormalized integer RNA counts**. Do not substitute the
  normalized AREA matrix. Pass `--rna-column-map` if raw count sample headers
  need mapping to manifest `sample_id`.
- TMT: limma from the authorized corrected log2 abundance matrix, globally
  complete 5,211-protein universe.
- All 12 jobs use **phenotype-only** adjustment, matching the previously
  validated conventional Stage 11 comparisons. They are **not** the separate
  covariate-adjusted conventional comparisons used elsewhere in the project.
- CERAD uses `3-4` as group A and `1-2` as group B (historical convention).
- First run without `--execute` to inspect 12 cohorts; then add `--execute`.
- Outputs and execution logs remain outside Git, under `--outdir`.
- Historical results are not inputs and no comparison with them is performed.

The driver refuses to run if any of the required source files is absent. An
execution report is written after each completed job. A failed job halts the
batch, with details in its `execution.log`.

This does **not** complete the raw-to-eight-figures pipeline: full Stage 11
sensitivity, Stage 7–12 fresh intermediates and figure producers remain separate.

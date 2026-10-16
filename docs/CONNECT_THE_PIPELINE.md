# Connect the ROSMAP pipeline — honest incremental integration

This patch adds the ability to register actual producer commands and execute the canonical DAG only through an explicitly selected stage. It does **not** substitute archived files for independent preprocessing, nor does it claim the raw-to-figures pipeline is complete. The default plan's `command: null` stages remain blocked.

The supported roots are `{release_root}` (repository), `{raw_root}` (authorized input directory, outside Git), and `{output_root}` (generated artifacts, outside Git). A stage is considered successful only when its command exits zero and every declared output exists, is nonempty, and hashes successfully. The pipeline additionally checks upstream dependencies and captures input hashes and command logs. Use `--through STAGE` to test a contiguous prefix; this prints `SELECTED_STAGES_COMPLETE` separately from `RAW_TO_FIGURE_COMPLETE`.

For each genuine producer script, register an argv command, actual input paths, and at least one output file using `workflows/connect_pipeline_stage.py`. Example, **only after providing a verified RNA preprocessing script with these arguments and contract**:

```bash
python3 workflows/connect_pipeline_stage.py \
  --plan config/raw_to_figures_plan.json \
  --output-plan "$HOME/Downloads/rosmap-connected-plan.json" \
  --stage rna_preprocess \
  --command-json '["python3","{release_root}/scripts/real_rna_preprocessor.py","--input","{raw_root}/rna/raw_counts.rds","--out","{output_root}/rna/normalized.csv"]' \
  --requires-json '["{raw_root}/rna/raw_counts.rds"]' \
  --produces-json '["{output_root}/rna/normalized.csv"]'
```

The example names are *illustrative*, not assertions that those paths or CLI parameters exist. Reuse the output plan as `--plan` when registering later stages. `--execute` really runs commands and writes only outside the repo.

```bash
python3 workflows/run_raw_to_figures.py \
  --release-root "$PWD" \
  --raw-root "$HOME/Developer/ROSMAP-authorized-data" \
  --output-root "$HOME/Downloads/rosmap-fresh-output" \
  --plan "$HOME/Downloads/rosmap-connected-plan.json" \
  --through rna_preprocess --execute
```

No legitimate producer commands have been invented. Remaining task: port and test count/abundance preprocessing, manifest construction, regenerated AREA+conventional models, Stage 7/9/10/11/12 producers, and update figure producers to take output-root paths rather than frozen hashes. Do not claim complete reproduction before the final, full-run provenance report passes.

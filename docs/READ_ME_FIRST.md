# Final-day integration inventory — not a completed pipeline

The included `workflows/audit_raw_to_figures.py` inventories key local artifacts, candidate preprocessing/figure scripts, existing runners, and publication figure manifest. It reads filenames only, does not copy protected ROSMAP data, does not mutate research/reference repositories, and produces a local-only JSON/Markdown report.

Run from the ROSMAP public-release worktree:

```bash
python3 workflows/audit_raw_to_figures.py \
  --release-root "$PWD" \
  --research-root "$HOME/Developer/area-workspace/ROSMAP-AREA-DNA-rotation" \
  --reference-root "$HOME/Developer/stage11_offline_2026-09-28" \
  --output-root "$HOME/Downloads/rosmap-integration-gate-v1"
```

If you have a separate directory containing authorized raw input files, add `--raw-root /path/to/raw_inputs`.

**Important:** The tool *does not* create a raw-to-figure pipeline. The next developer step is to inspect candidate producer scripts and establish explicit source→output contracts. Only then can a supported DAG with raw input checks and figure reruns be built. An archived count matrix is not the original raw data, and frozen Stage11/12 tables do not prove raw lineage.

Do not commit the generated local report before reviewing local absolute paths and data-use permissions. Protect raw ROSMAP data and participant-level metadata.

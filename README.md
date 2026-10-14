# ROSMAP AREA Analysis

Research code for exploratory development of an AREA-based analysis of ROSMAP molecular and clinical data.

## Project status

**Active exploratory analysis.**

The ROSMAP analysis cohort, phenotype definitions, and downstream analysis workflow are currently being refined. A final canonical analysis pipeline has **not yet been frozen**.

Code generated during method development is retained in `exploratory/` for provenance and reference. These scripts should not be interpreted as a single sequential pipeline, and some represent superseded approaches, diagnostics, sensitivity analyses, or intermediate methodological experiments.

Once the final ROSMAP dataset and analysis decisions are established, the reproducible primary workflow will be separated from this exploratory code and documented here.

## Repository structure

```text
ROSMAP-AREA-DNA-rotation/
├── README.md
├── .gitignore
│
├── exploratory/
│   ├── area/             # AREA method development and comparisons
│   ├── clustering/       # RAE construction, clustering, and k evaluation
│   ├── config/           # exploratory configuration/input files
│   ├── data_prep/        # ROSMAP data integration and preprocessing
│   ├── diagnostics/      # debugging and diagnostic analyses
│   ├── enrichment/       # exploratory enrichment analyses
│   ├── pipelines/        # historical/development shell workflows
│   └── visualization/    # exploratory plotting code
│
├── data/                 # local/restricted source and derived data
└── results/              # locally generated analysis outputs



o


qqq


exit()

## Canonical figure reproduction

This repository provides eight canonical figure generators comparing
rank-based AREA with conventional differential-expression and
differential-abundance approaches in ROSMAP transcriptomic and
proteomic analyses.

The canonical scripts and their SHA-256 checksums are registered in
`finalization_audit/repository_freeze/canonical_figure_manifest.csv`.

### Reproduction

From the repository root, first verify the inputs and scripts:

    python3 workflows/run_main_figures.py --dry-run

To generate the eight figures:

    python3 workflows/run_main_figures.py --all

Use `--output-root` with a new directory for subsequent runs to
avoid overwriting previous results.

The runner checks frozen input hashes, figure script hashes, and
additional registered input hashes before running the figures.

### Data availability

Some frozen analysis inputs are stored under `data/frozen_inputs/`.
This directory is excluded from Git and its contents are not
authorized for public redistribution.

A fresh clone cannot reproduce the complete figure set without
provisioning these inputs separately.

This workflow reproduces figures from frozen analysis results,
not the entire upstream analysis from raw ROSMAP data.

Figure 3 Panel A currently uses nine frozen numeric values embedded
in its script because the available Stage 9 interpretation text
does not contain those values.

See `docs/METHODS.md` and `docs/FIGURE_PROVENANCE.md`.

### Supplemental investigations

The `supplemental/` directory documents longitudinal monocyte
cohort feasibility, Weighted AREA, Stage 12L/M rank trajectories,
and phosphoproteomics. Their validation status differs.

### Data governance

Participant-level ROSMAP data, sample identifiers, and potentially
restricted derived results must not be publicly redistributed
without appropriate authorization.

### Checking required input files

The canonical figures require frozen intermediate analysis results.
These files are not all distributed with the public source code.

After obtaining any required data through an authorized mechanism,
place them at the paths recorded in the input manifests and run:

    python3 tools/check_figure_inputs.py

Use `--show-present` to display successfully verified inputs.

The validator checks file presence and SHA-256 integrity. It does
not download data, grant data access, or reconstruct upstream results.

Additional source files used by individual figures, including the
Figure 8 protein-evidence input, must also be provisioned where
specified by the figure runner.

Do not commit restricted data or locally provisioned frozen inputs.

# ROSMAP AREA Analysis

Computational analysis of ROSMAP transcriptomic, proteomic, and
clinical data using AREA (Activity-by-Rank Expression Analysis)
and conventional differential-analysis methods.

This repository contains the finalized eight-figure analysis,
reproducible figure-generation scripts, methodological documentation,
and exploratory analyses retained for provenance.

## Project status

**Eight canonical figures finalized and publicly released.**

The eight canonical figures and their generating scripts have been
validated against frozen intermediate analysis inputs.

The figure-generation workflow is documented and versioned.
This does not mean that every upstream analysis, exploratory
experiment, or raw-data processing step has been frozen into
one end-to-end pipeline.

Historical and exploratory code remains available under
`exploratory/`. Some scripts represent superseded methods,
diagnostics, or sensitivity analyses rather than components of
the canonical workflow.

## Quick links

- [Eight canonical figures](figures/README.md)
- [Figure-generating scripts](scripts/multiomic/)
- [Methods](docs/METHODS.md)
- [Figure provenance](docs/FIGURE_PROVENANCE.md)
- [Canonical figure manifest](finalization_audit/repository_freeze/canonical_figure_manifest.csv)
- [Validated Python requirements](requirements-figures.txt)

## Repository organization

| Directory or file | Purpose |
|---|---|
| `figures/` | Eight final figures in PDF/PNG and available SVG formats |
| `scripts/multiomic/` | Canonical figure-generation scripts and helpers |
| `workflows/run_main_figures.py` | Figure reproduction runner |
| `config/` | Frozen-input manifest |
| `docs/` | Methods and figure provenance |
| `supplemental/` | Documented supplementary investigations |
| `exploratory/` | Historical, diagnostic, and exploratory analyses |
| `tools/` | Input-provisioning validation |
| `results/` | Analysis outputs; not all are distributed |
| `data/` | Local or restricted datasets; not all are distributed |

## Software environment

The validated figure-generation environment used Python 3.14.3.
The recorded package versions are in `requirements-figures.txt`.

Install dependencies using:

    python3 -m pip install -r requirements-figures.txt

Input data are provisioned separately, subject to the relevant
data-use permissions.

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

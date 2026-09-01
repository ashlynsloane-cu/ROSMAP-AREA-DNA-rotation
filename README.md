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

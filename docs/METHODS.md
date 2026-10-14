# Methods and reproducibility scope

## Project overview

This repository documents exploratory and comparative applications of AREA (a rank-based association framework) to ROSMAP molecular measurements and Alzheimer's disease-related phenotypes. The canonical release provides eight figure generators, a frozen-input manifest, and a unified figure runner. It does **not** constitute a fully reproducible pipeline from raw ROSMAP files to every underlying analysis table.

## Molecular data and comparisons

The work includes analyses of monocyte RNA sequencing, dorsolateral prefrontal cortex (DLPFC) RNA sequencing, and DLPFC TMT proteomics. Participant inclusion, feature filtering, contrast definitions, and covariates depend on the underlying analysis and should be taken from the corresponding input manifests and scripts rather than inferred from figure-level counts.

AREA results are compared, where appropriate, to conventional differential-analysis methods, including DESeq2 for RNA-seq and limma for proteomics. Results across methods are not necessarily estimates of the same statistical quantity. The distinction between nominal significance, Benjamini–Hochberg FDR, an individual-modality finding, and a formal joint/conjunction result should be preserved.

## Statistical methods

The frozen AREA implementation used for validated analyses accounts for tied ranks, covariates via partial-rank/residualized association, residual degrees of freedom, a two-sided Gaussian analytical p-value, and BH multiple-testing correction. The exact implementation and covariate model are defined by each generating analysis and its archived metadata; the figure-generation runner does not rerun all upstream inference.

Cross-modal analyses summarize concordance, formal joint significance, enrichment, rank overlap, and gene-set leading-edge results. Pathway associations should not be interpreted as evidence of causal regulation. Gene-level functional descriptions and literature-based protein evidence are not independent validation experiments.

## Reproducing the canonical figures

From the repository root, with the required Python/R dependencies and authorized frozen inputs available:

```bash
python3 workflows/run_main_figures.py --dry-run
python3 workflows/run_main_figures.py --all
```

Inspect `workflows/run_main_figures.py --help` for supported output-directory arguments. The runner verifies registered input and script hashes before generating figures. It must not be assumed that a fresh public clone contains all necessary frozen inputs: some outputs are derived from controlled-access ROSMAP data and are intentionally excluded from version control pending redistribution approval.

The canonical figure script inventory is held in `finalization_audit/repository_freeze/canonical_figure_manifest.csv` in the local finalization workspace. The input bindings are held in `config/canonical_figure_inputs.csv`. See `docs/FIGURE_PROVENANCE.md` for the per-figure scope.

## Known limitations

1. Raw-to-derived upstream analyses are not all incorporated into the eight-figure runner.
2. Some frozen Stage 11/12-derived aggregate inputs are housed in a local, Git-ignored `data/frozen_inputs/` tree; distribution rights require separate confirmation.
3. Figure 3 Panel A currently includes nine frozen summary medians embedded in its script, pending independent verification against the corresponding upstream numeric table and externalization.
4. Local execution success does not by itself demonstrate reproducibility on another computer; environment locking and a clean-clone test remain outstanding.
5. The supplemental investigations have differing levels of completion and validation. Consult each `supplemental/*/README.md` rather than assuming they represent finalized results.

## Data governance

ROSMAP data access and redistribution must comply with their source agreements. Never commit participant IDs, specimen/sample metadata, controlled molecular matrices, credentials, or unreviewed exports. `.gitignore` does not remove data already committed to Git history; tracked content and history must be reviewed separately.

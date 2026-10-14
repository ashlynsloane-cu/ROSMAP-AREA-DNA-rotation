# ROSMAP phosphosite abundance: AREA versus limma (NCI vs AD)

## Scientific question

Can covariate-adjusted Regular AREA identify Alzheimer disease–associated differences in measured phosphosite abundance, and how does its discovery pattern compare with limma in the same samples?

## Data and cohort

The frozen AREA run manifest reports **134 participants (70 NCI; 64 AD)** and **8,016 tested phosphosites**. The analysis considers measured phosphosite abundance; these estimates should not be described as changes in phosphorylation occupancy unless normalized for corresponding total-protein abundance and otherwise justified.

## Methods

The AREA run manifest records adjustment for continuous age and postmortem interval plus categorical sex and batch; model design rank **15** and residual degrees of freedom **119**. The specified AREA inference is FWL partial ranks, residual-df-corrected analytic permutation variance, Gaussian two-sided p values, and BH FDR control at **0.05**. The conventional comparison uses limma; model comparability, filtering and annotation should be checked against its full script and manifests before stronger methodological claims.

## Frozen results

The method comparison summary reports:

| Classification at FDR < 0.05 | Phosphosites |
| --- | ---: |
| Significant in AREA and limma | 437 |
| AREA only | 30 |
| limma only | 364 |
| Neither | 7,185 |
| **Total tested** | **8,016** |

Thus, AREA detected **467** sites and limma detected **801** sites at the stated threshold. On this particular contrast, limma reports more discoveries; neither count alone is evidence of superior calibration, specificity, or biological validity.

Primary frozen artifacts:

- AREA run metadata: `results/phosphoproteomics/area/NCI_vs_AD/run_manifest.json`
- AREA results: `results/phosphoproteomics/area/NCI_vs_AD/phosphosite_NCI_vs_AD_Regular_AREA_all_results.csv`
- limma results: `results/phosphoproteomics/differential/NCI_vs_AD/phosphosite_NCI_vs_AD_limma_all_results.csv`
- Method overlap and signal metrics: `results/phosphoproteomics/comparison/phosphosite_method_signal_architecture_summary.tsv`
- limma implementation: `scripts/phosphoproteomics_run_limma_NCI_vs_AD.R`

## Completion status and limitations

**Frozen differential comparison available; independent end-to-end reproduction pending.** The counts above were read from the existing comparison table and cross-checked arithmetically with the AREA run manifest; the underlying test pipeline has not yet been rerun for this supplemental audit. Potential protein-level abundance confounding, site assignment ambiguity, missingness and batch effects merit qualification. The sample metadata, participant-level matrix and phosphopeptide QC matrices should not automatically be published because of possible controlled-access data restrictions.

## Reproduction and audit

SHA-256 hashes for the five designated source files are recorded in `finalization_audit/repository_freeze/supplemental_source_manifest.csv`. The limma and AREA input matrices are currently maintained separately from this README; a public reproducibility package should distribute only materials permitted by the relevant data-use agreements. A one-command validated supplementary workflow has not yet been established.

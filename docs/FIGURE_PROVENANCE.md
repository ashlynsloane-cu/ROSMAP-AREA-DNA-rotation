# Canonical figure provenance

The eight scripts below are the frozen figure-generation entry points. Successful local execution has been reported for the original unified run and the runner adapted to local frozen inputs. These checks confirm figure reproduction from precomputed inputs, **not** a complete rerun of all upstream ROSMAP analyses.

| Figure | Canonical script | Principal scope |
| --- | --- | --- |
| 1 | `scripts/multiomic/make_figure1_cross_phenotype_recurrence_canonical.py` | Cross-phenotype recurrence and method comparison |
| 2 | `scripts/multiomic/make_figure2_cross_modal.py` | RNA/protein cross-modal relationships and convergence architecture |
| 3 | `scripts/multiomic/make_figure3_signal_architecture_manuscript_v14.py` | Signal architecture, ranking and discovery characteristics |
| 4 | `scripts/multiomic/make_figure4_signal_geometry_with_outliers.py` | Rank/signal geometry, including selected outliers |
| 5 | `scripts/multiomic/make_figure5_covariate_sensitivity_v13.py` | Sensitivity of AREA and conventional methods to covariate adjustment |
| 6 | `scripts/multiomic/make_figure6_crossomic_convergence_canonical.py` | Formal cross-modal findings and leading-edge/pathway summaries |
| 7 | `scripts/multiomic/make_figure7_biological_programs_and_genes-3.py` | Biological enrichment programs and gene examples |
| 8 | `scripts/multiomic/AREA_measured_AD_protein_recovery/make_ten_protein_recovery.py` | Ten measured AD-relevant proteins and literature-context evidence |

## Frozen inputs and integrity

`config/canonical_figure_inputs.csv` records 15 original figure-to-input bindings. Eight bindings reference six distinct files from the separate Stage 11/12 offline workspace. The local publication preparation copied those six files into `data/frozen_inputs/` with SHA-256 identity preserved. An additional Stage 12K formal-comparison input for Figure 6 was copied and hashed, and Figure 8's `all_tested_protein_evidence.csv` input was registered in a separate local audit record.

The auxiliary input record is `finalization_audit/repository_freeze/additional_figure_inputs.json`. Local verification records, generated outputs, and controlled or unreviewed inputs are **not presumed publication-approved**. A reader cloning this repository will need access to any non-distributed inputs through an authorized source.

## Figure-specific caveats

- **Figure 2:** Uses Stage 12K `CONVERGENCE_ARCHITECTURE_SUMMARY.tsv` for cross-modal architecture metrics.
- **Figure 3:** Nine Panel A median values are embedded in the generator; upstream numeric verification and externalization remain outstanding.
- **Figure 5:** Uses Stage 11 covariate sensitivity and Shapley-attribution summaries.
- **Figure 6:** Uses cross-modal GSEA, Stage 12K architecture, leading-edge memberships, and the additional formal-comparison table. Formal significance and descriptive agreement are distinct quantities. The Figure 6 canonical script is the `crossomic_convergence_canonical` version, not the older `crossomic_biology` variant.
- **Figure 7:** Uses Stage 12F enrichment and Stage 12K leading-edge data. Leading-edge membership is an enrichment-algorithm result, not independent functional validation.
- **Figure 8:** Compares AREA and limma evidence for ten selected AD-related proteins and summarizes published biological context; it is not an independent validation cohort.

## Validation boundaries

The frozen manifests, SHA-256 checks, and local unified-run tests support computational provenance of the plotted summaries. They do not establish redistribution permissions, the validity of all upstream cohort models, or fresh-clone portability. See `docs/METHODS.md` and the supplemental README files for related scope and limitations.

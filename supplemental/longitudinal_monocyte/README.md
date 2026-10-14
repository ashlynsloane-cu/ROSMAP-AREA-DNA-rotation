# Longitudinal monocyte cohort (exploratory)

## Scientific question

Can repeated monocyte RNA-seq specimens from the same ROSMAP participants support within-person analyses of transcriptional change over time, potentially in relation to Alzheimer disease phenotypes?

## Data and cohort

The frozen cohort files identify **56 participants with two monocyte specimens each (112 draws)**. The paired manifest records earlier and later specimen IDs, sampling ages, observed interval, and technical concordance indicators for RNA, library, and sequencing batches. Some sampling ages are represented using a capped value for ages 90+; analyses of time intervals should inspect the corresponding raw and capped fields before modeling.

## Methods and provenance

- Paired cohort: `results/monocyte_full_cohort/monocyte_longitudinal_pairs_56.tsv`
- Draw-level metadata: `results/monocyte_full_cohort/monocyte_longitudinal_draws_112.tsv`
- Prior design notes: `results/multiomic_concordance/AREA_longitudinal_extension_notes.txt`
- Related cohort construction: `scripts/monocyte/build_full_monocyte_cohorts.py`

The paired manifest includes fields identifying whether specimens share RNA, library and sequencing batches; these factors may influence within-person comparisons. The present audit has not established a canonical repeated-measures differential-expression model or its inferential outputs.

## Verified results

- 56 two-specimen participant pairs.
- 112 draw-level specimen records.

These are **cohort feasibility findings**, not evidence of longitudinal differential expression or disease progression.

## Completion status and limitations

**Exploratory; cohort verified, inference not verified.** Before making longitudinal biological claims, identify the model, sample exclusions, participant effects, age/time handling, batch-adjustment strategy, multiple-testing correction, and frozen statistical results. Subject/specimen-level manifests may contain controlled-access identifiers and should not automatically be committed to a public repository.

## Reproduction and audit

The existence and SHA-256 hashes of the three frozen source files were recorded in `finalization_audit/repository_freeze/supplemental_source_manifest.csv`. This verifies file provenance, not analytic reproduction. No canonical longitudinal-analysis command is designated yet.

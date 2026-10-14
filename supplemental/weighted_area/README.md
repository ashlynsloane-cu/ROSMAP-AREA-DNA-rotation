# Weighted AREA phenotype encodings (methods development)

## Scientific question

How sensitive are Weighted AREA results to the encoding of a clinical or pathological phenotype, and how do encoding choices affect discoveries, calibration and biological interpretation relative to Regular AREA and conventional differential-expression methods?

## Data and analyses

The repository contains substantial methodological experiments, including phenotype-encoding comparisons, matched-sample sensitivity, covariate-adjusted runs, age-related preliminary analyses and enrichment. Candidate phenotype mappings tested in prior work include `current`, `linear01`, `sqrt`, `square`, `cube`, `exp3` and `ecdf`. These are methodological alternatives, not interchangeable representations of a fixed estimand.

## Methods and provenance

Candidate analysis entry points:

- `exploratory/area/run_ams_area.py` — Regular/Weighted AREA development and calibration.
- `exploratory/method_development/run_weighted_area_matched_sample_sensitivity_v3.py` — matched-sample sensitivity investigation.

Relevant result directories include:

- `results/weighted_area_encoding_comparison/`
- `results/weighted_area_encoding_validation/`
- `results/weighted_area_matched_sample_sensitivity/`
- `results/weighted_area_observed_covariate_adjusted/`
- `results/weighted_area_full_phenotype_suite/`
- `results/weighted_area_biology_final_v2/`
- `results/ams_area_validation/`

These locations are an inventory of candidate outputs, not a declaration that every historical version is canonical.

## Findings available for audit

Encoding choices have produced materially different significance counts in prior exploratory comparisons. For example, one recorded CERAD-burden validation replicate reported **495 FDR-significant features using the current encoding** and **986 with the square-root encoding**. This illustrates sensitivity to encoding; it **does not demonstrate** that the larger discovery set is more accurate or has better false-discovery control.

## Completion status and limitations

**Methods development; canonical validation selection pending.** A preferred encoding should not be chosen solely for its number of significant features. Before citing a particular method comparison, verify the matched samples, effect definition, null calibration, BH/FDR procedure, held-out or simulation validation, and exact source files. The many historical outputs are not all suitable for publication or GitHub inclusion.

## Reproduction and audit

Hashes of the two identified candidate entry-point scripts appear in `finalization_audit/repository_freeze/supplemental_source_manifest.csv`. No single canonical Weighted AREA run or complete frozen-output comparison has yet been independently reproduced. Keep the main-figure AREA workflow distinct from these exploratory weighting experiments.

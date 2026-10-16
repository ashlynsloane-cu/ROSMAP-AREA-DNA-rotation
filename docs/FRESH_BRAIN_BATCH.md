# Fresh brain AREA model batch

Uses 111 freshly generated ROSMAP cohort manifest entries, authorized DLPFC normalized expression and TMT corrected abundance matrices, and the existing AREA executable. It does not read historical model outputs or Stage 11 task tables.

The historical Stage 11 covariate specification is preserved: DLPFC `age_numeric,rin_numeric,pmi_numeric` and `sex,sequencing_batch` (sample-by-feature expression); TMT `age_numeric,pmi_numeric` and `msex,Batch` (feature-by-sample expression with globally complete proteins). CERAD is explicitly oriented `3-4` vs `1-2` to match the historical Stage 11 model.

Run preflight then add `--execute` for actual fits. `FRESH_BRAIN_AREA_COMPLETE: 12` confirms fresh execution, not strict equivalence against historical results. The RNA input is already normalized, and this batch does not replace the original normalization step or connect Stage 7–12 and figures.

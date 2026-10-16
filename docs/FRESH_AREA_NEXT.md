# Fresh AREA analyses — next runnable models

`run_fresh_monocyte_all_area.py` runs all six primary monocyte contrasts using
freshly generated manifests and the authorized normalized monocyte expression
matrix. It does not use historical frozen results and does not mark the entire
multimodal AREA stage as reproduced. The original pandas reader is retained.

`run_fresh_brain_area.py` fits one DLPFC/TMT discovery contrast, using the exact
covariates explicitly supplied. The correct biological covariates, matrix
orientation, feature identifiers, and completeness rules must match the
historical scientific model design; this script does not guess them.

No change to the existing 12-stage plan is made: `area_models` still refers to
all required modalities and contrasts, not merely the six monocyte models.

Source files and unit tests may be committed; output matrices and results
remain under the authorized external output root and must not be committed.

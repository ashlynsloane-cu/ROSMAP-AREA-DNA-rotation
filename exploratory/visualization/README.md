# AMS-AREA v2: corrected inference + MSI calibration

This bundle implements the changes motivated by the global-null calibration
experiments.

## Statistical change

The AREA enrichment statistic is retained.

The p-value calculation changes from the legacy signed-half Gaussian approach to
a two-sided Gaussian calculated from the **full permutation-null ES
distribution**:

```text
mu    = mean(all permuted ES)
sigma = sd(all permuted ES)
z     = (observed ES - mu) / sigma
p     = 2 * P[Normal >= |z|]
```

Global-null testing showed that the old signed-half method was anti-conservative,
while the full-null method was close to the expected 5%, 1%, and 0.1% tail
frequencies for both Regular and Weighted AREA.

## What remains unchanged

- Regular AREA ES uses the official AREA implementation.
- Weighted AREA remains the AMS continuous-weight extension.
- 1,000 phenotype-label permutations remain the default null.
- BH correction remains separate for Regular and Weighted.
- Missing phenotype values are excluded rather than imputed.
- Binary-only traits use Regular AREA only.

## NES

The original same-signed NES is retained as a descriptive score because its sign
and magnitude are useful for direction. It no longer supplies the p-value.

## MSI

MSI now uses corrected p-values:

```text
MSI = log10(P_weighted_corrected / P_regular_corrected)
```

The runner deliberately does NOT assign dosage/trigger archetypes by default.
The historical +/-2 boundary is considered provisional until calibrated by
simulation.

## Suggested repository layout

```text
exploratory/area/
    run_ams_area.py                          # replace current runner

exploratory/area/validation/
    validate_corrected_ams_area_nulls.py
    compare_ams_area_versions.py
    simulate_msi_calibration.py              # replace prior validation version
    bootstrap_msi_stability.py               # replace prior validation version
    plot_corrected_msi_diagnostics.py
    run_corrected_ams_area_examples.sh
```

## Recommended execution order

1. `run_ams_area.py --self-test`
2. corrected matched AD4-vs-NCI1 Regular AREA
3. corrected Cognitive/Braak/CERAD AMS-AREA
4. corrected global-null validation across core traits
5. old-vs-corrected result comparison
6. corrected MSI simulations
7. choose a candidate MSI threshold based on false-preference and recovery rates
8. bootstrap selected real genes
9. only then freeze archetype labels

Do not run downstream clustering, GSEA, RAE, or interaction analyses from the
legacy significant-gene sets after adopting the corrected inference.

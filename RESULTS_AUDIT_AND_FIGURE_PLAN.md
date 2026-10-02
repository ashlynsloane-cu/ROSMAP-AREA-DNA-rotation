# Results Audit & Figure Plan

**Project:** ROSMAP AREA  
**Date:** 2026-10-02  
**Purpose:** Freeze the scientific question → evidence → figure mapping before generating final figures.

## Figure-design principle

We should make **one main figure per scientific question**, with multiple panels when necessary. Figures should answer the question directly, use the frozen analysis outputs rather than analysis-process chronology, and distinguish biological findings from methodological characterization and null/boundary results.

We should **not** make one figure per analysis stage. Stages are analysis provenance; figures are the scientific argument.

## Results audit table

| Result family | Scientific question | What the frozen analysis shows | Interpretation / role |
|---|---|---|---|
| 1. Within-modality recurrence | Do disease-associated features recur across related clinical/pathology contrasts within each modality? | DLPFC and TMT show substantial recurrence under both AREA and conventional analyses. Monocyte AREA has no recurrent/significant features under the frozen definition, while conventional monocyte signal recurs. | Supporting main result; establishes within-modality recurrence and modality dependence. |
| 2. Braak progression architecture | Does AREA identify early, late, progressive, extreme-only, or non-monotonic patterns across tau stages? | DLPFC AREA: 16,208/17,646 have no classified Braak signal; 635 extreme-only; 477 late; 326 other partial. Conventional analysis assigns more early/progressive patterns. | Supporting/compact; shows that methods can produce different progression architectures. |
| 3. Longitudinal monocyte | Do repeated blood samples show systematic within-person molecular changes and relationships to later pathology? | 608 QC-pass specimens from 554 people; 54 repeated participants. Paired AREA analysis: 0 FDR discoveries. Pathology analysis: 0 FDR discoveries; null calibration close to nominal. | Secondary/supplement. Important boundary result, not a biological centerpiece. |
| 4. Monocyte ↔ DLPFC RNA | Does blood RNA reproduce brain RNA disease-associated signal? | Across 14,211 shared genes, signed-statistic correlations were essentially zero across contrasts; 0/100 top-k overlap tests survived FDR. | Main result: no broad individual-gene blood–brain transcriptomic concordance. |
| 5. DLPFC RNA ↔ TMT protein | Does disease-associated brain RNA signal recur at the protein level? | AREA DLPFC–TMT overlap: top-500 = 20 observed vs 15.1 expected (1.32×, P=.115); top-1000 = 100 vs 76.5 expected (1.31×, FDR=.0047). Conventional was significant at all three thresholds. | Main result: broad/modest RNA–protein relationship rather than a highly concentrated top-ranked overlap. |
| 6. Strict three-way integration | Is there a universal gene-level signal shared by monocyte RNA, DLPFC RNA, and TMT protein? | No significant three-modality enrichment; at top-500, zero genes were integrated across all three modalities under either framework. | Main negative/boundary result. |
| 7. Phenotype integration + permutation null | Do the same genes recur across diagnosis, Braak, and CERAD more than expected from list size alone? | Strong enrichment over empirical nulls. At top-500 DLPFC AREA: 429 genes supported ≥2 axes vs 125 expected; 93 supported all 3 vs 1.9 expected. TMT: 546 ≥2 axes and 215 all 3. Monocyte: 252 ≥2 axes and 14 all 3. | Main result: phenotype-associated signal is structured rather than explained by list size alone. |
| 8. AREA ↔ conventional phenotype-integrated overlap | Do AREA and DESeq2/limma recover overlapping phenotype-integrated biology? | At top-500: DLPFC 266 shared vs ~11.8 expected; TMT 417 vs ~58 expected; monocyte 38 vs ~7.4 expected. | Main result/bridge: methods differ in rankings but recover substantial integrated biology. |
| 9. Signal architecture | What distinguishes AREA-predominant from conventional-predominant features? | AREA-predominant features consistently have lower NCI/AD participant-distribution overlap. NCI→AD ORs at top250/500/1000: monocyte 0.529/0.496/0.487; DLPFC 0.276/0.211/0.150; TMT 0.028/0.030/0.074. Across 45 phenotype-generalization models, 40/45 had OR<1 and 33/45 had OR<1 + FDR<.05. | Major main result: the clearest empirical characterization of what AREA selects. This is signal geometry, not method superiority. |
| 10. ROSMAP-calibrated simulations | Under known ground truth, what signal geometry produces different AREA vs conventional behavior? | AREA is well calibrated; it is not a generic outlier/heterogeneity detector. Sparse subset effects/outliers often favor conventional methods; symmetric dispersion does not favor AREA; asymmetric mean-preserving RNA rank redistribution can favor AREA. | Major main result: mechanistic/statistical explanation for the empirical Stage 9 architecture. |
| 11. Covariate-adjustment stress test | How much does covariate adjustment change AREA, and which covariates contribute? | 18/18 A-vs-B validations; 336/336 covariate states; 72/72 Shapley rows. DLPFC batch dominates; monocyte batch dominates 5/6 contrasts; TMT age generally dominates. Primary-vs-Shapley Spearman rho=.934 overall. | Methodological validation; important for interpretation but not a biological headline. |
| 12. Formal cross-omic gene-level convergence | Which individual genes have sufficiently strong RNA + protein evidence in the same direction? | Among 4,903 mapped genes: NCI→AD has 52 AREA formal conjunction genes, MCI→AD 0, late Braak 0, CERAD 10. AREA discoveries include shared AREA+conventional, AREA-only, and conventional-only sets. | Main biological result; demonstrates that cross-omic convergence varies by phenotype axis. |
| 13. Pathway-level cross-omic convergence | Can RNA/protein converge at the pathway level when individual-gene conjunction does not? | MCI→AD and late Braak have zero formal conjunction genes but strong concordant pathway enrichment, including oxidative phosphorylation, respiratory electron transport, Complex I, mitochondrial ATP synthesis, and translation/ribosome biology. | Major main result: distributed cross-omic biology can converge at pathway level without individual genes reaching conjunction FDR. |
| 14. RRHO / whole-profile RNA–protein concordance | Is RNA–protein agreement distributed across complete ranked profiles rather than only significant genes? | Approximate signed-rank rho values ~0.27–0.32 across major contrasts, with particularly strong concordant RRHO structure for MCI→AD and late Braak. | Main result, likely paired with pathway-level convergence. |
| 15. Targeted monocyte follow-up | Do strongest brain RNA–protein candidates reproduce in blood monocytes? | NCI→AD: 0/67 brain-derived candidates significant by monocyte AREA; 3/67 by DESeq2; 0 by both. CERAD: 0/23 by either. Some genes are not testable in monocyte. | Secondary/supplement; reinforces brain-specificity but is targeted rather than genome-wide discovery. |
| 16. Stage 12L/M trajectory architecture | Do AREA-significant genes fall into recurring participant-rank trajectory shapes? | 2,326 NCI→AD AREA-significant genes clustered by trajectory shape. K=4 has four stable structural groups; 2,192 genes have perfect assignment across resamples and 2,243 have ≥95% stability. Cluster sizes: 721/481/492/632. | Exploratory/hold until biological interpretation is fully settled; do not equate with patient subtypes. |

## Proposed main-figure question map

### Figure 1 — Do disease-associated signals recur across phenotypic axes and molecular layers?

**Question:** Is the observed disease-associated signal reproducible across clinical diagnosis, tau pathology, and amyloid pathology, and does that recurrence differ by molecular layer?

Potential panels:
- A: recurrence/convergence across diagnosis, Braak, CERAD
- B: empirical observed-vs-null recurrence
- C: modality comparison (monocyte RNA, DLPFC RNA, TMT)
- D: AREA vs conventional overlap in phenotype-integrated signal

Primary evidence: Results 7–8, with Results 1–2 as supporting context.

### Figure 2 — Do the same molecular signals appear across molecular layers?

**Question:** How much concordance exists between blood RNA, brain RNA, and brain protein?

Potential panels:
- A: monocyte ↔ DLPFC gene-level concordance
- B: DLPFC RNA ↔ TMT protein overlap
- C: strict three-way convergence
- D: possibly whole-profile/RRHO summary

Primary evidence: Results 4–6 and 14.

### Figure 3 — What does AREA add beyond conventional differential-expression methods?

**Question:** What distinguishes AREA-predominant features from conventional-predominant features?

Potential panels:
- A: method-specific top-k overlap/classification
- B: distribution-overlap effect across modalities and thresholds
- C: phenotype-generalization ORs
- D: potentially phenotype/modality summary rather than dozens of individual comparisons

Primary evidence: Result 9.

**Guardrail:** This figure should characterize the difference between methods, not claim that AREA is universally superior.

### Figure 4 — What kind of signal produces the AREA-specific behavior?

**Question:** Under controlled ground truth, which signal geometries generate different AREA vs conventional behavior?

Potential panels:
- A: broad/homogeneous mean-shift regime
- B: penetrance / heterogeneous-effect regime
- C: outlier / tail regimes
- D: asymmetric mean-preserving rank-redistribution regime
- E: calibration/type-I-error check if needed

Primary evidence: Result 10.

### Figure 5 — How much does covariate adjustment change AREA?

**Question:** Is covariate adjustment merely cosmetic, or does it materially alter AREA rankings?

Potential panels:
- A: adjusted vs unadjusted AREA concordance
- B: rank/feature-set changes across modalities
- C: covariate-attribution/Shapley summary
- D: modality-specific dominant covariates

Primary evidence: Result 11.

### Figure 6 — What is the biological content of the cross-omic signal?

**Question:** What biological programs emerge when DLPFC RNA and protein are integrated?

Potential panels:
- A: formal gene-level RNA–protein conjunction by phenotype axis
- B: RRHO/whole-profile concordance
- C: pathway-level convergence
- D: representative mitochondrial/respiratory programs or leading-edge genes

Primary evidence: Results 12–14.

## Supporting/supplementary figure candidates

- Longitudinal monocyte repeated-sample analysis
- Braak progression architecture
- Targeted monocyte follow-up of brain-derived candidates
- Strict three-way negative result in greater detail
- Stage 12L/M trajectory clustering and K=4 diagnostics
- Detailed sensitivity analyses and alternative overlap metrics

## Figure-design rules

1. **One scientific question per figure.**
2. Multiple panels are encouraged when they answer different parts of the same question.
3. Use **data-driven plots**, not decorative conceptual diagrams, for Results figures.
4. Keep AREA and conventional methods visually consistent across figures.
5. Use the same modality labels and ordering throughout.
6. Preserve the distinction between diagnosis, Braak/tau, CERAD/amyloid, blood monocyte RNA, DLPFC RNA, and DLPFC TMT protein.
7. Do not turn negative results into visual omissions. Important nulls/boundaries should be shown when they constrain interpretation.
8. Do not interpret tiny p-values as large biological effects.
9. Do not call Stage 9 evidence proof of AREA superiority.
10. Do not describe Stage 12M trajectory clusters as patient subtypes without additional evidence.
11. Main figures should show the **effect/structure first**, with sample sizes, thresholds, and statistical details available but not visually dominant.
12. Every main panel should be traceable to a frozen source table/output.

## Current proposed main-figure sequence

**Fig 1:** Phenotype recurrence / integrated signal  
→ **Fig 2:** Cross-modal concordance  
→ **Fig 3:** AREA signal architecture  
→ **Fig 4:** Ground-truth simulations  
→ **Fig 5:** Covariate-adjustment sensitivity  
→ **Fig 6:** Cross-omic biological interpretation

This sequence moves from **whether the signal is reproducible → where it exists → what is distinctive about AREA → why that distinction occurs → how robust it is to covariate adjustment → what biology it contains.**

**Status:** Figure architecture is a proposal, not yet frozen. Do not generate final figures until each figure's panel-level source tables, estimand, denominator, statistical test, and intended takeaway have been audited.

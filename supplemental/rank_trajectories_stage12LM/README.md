# Participant-rank trajectories and Stage 12L/M architecture (exploratory)

## Scientific questions

How do feature-level participant ranks and their configurations differ across disease contrasts, and can recurring gene-level rank patterns be summarized through trajectory clustering? The Stage 12L leading-edge sensitivity investigation and Stage 12M K=4 architecture visualization form one exploratory thread. Earlier rank-insertion and stable-versus-lost examples are related but distinct descriptive analyses.

## Methods and provenance

Candidate entry points:

- `scripts/run_stage12L_NCI_vs_AD.sh` — Stage 12L NCI-versus-AD leading-edge sensitivity.
- `scripts/multiomic/stage12M_make_architecture_figure.py` — candidate K=4 architecture visualization.
- `scripts/multiomic/AREA_diagnosis_drivers/plot_participant_ranks.py` — descriptive participant-rank displays.
- `scripts/multiomic/AREA_diagnosis_drivers/plot_rank_insertion.py` — before/after group-insertion displays.
- `scripts/multiomic/AREA_diagnosis_drivers/plot_stable_vs_lost_examples.py` — descriptive stable/lost examples.

Frozen or pre-existing supporting artifacts:

- `results/figures/stage12M_architecture/Figure_candidate_AREA_gene_architecture_K4.png`
- `results/stage12_brain_convergence/stage12M_clustering/NCI_vs_AD/`
- `results/figures/area_diagnosis_participant_ranks/`
- `results/figures/area_rank_insertion/`

Related methodology notes exist in the separate offline workspace under `results/stage12_brain_convergence/stage12_method_notes/`.

## Verified results

A candidate K=4 architecture figure and supporting analysis directories exist. Their presence does **not** establish cluster stability, biological enrichment, generalization, or independent validation.

## Completion status and limitations

**Exploratory; visualization and run entry points located.** Before elevating any K=4 cluster as a biological result, confirm the exact participant/feature selection, rank normalization, distance metric, clustering parameters, cluster memberships, and stability checks. Descriptive rank plots must not be conflated with formal covariate-adjusted AREA significance.

## Reproduction and audit

The Stage 12L runner, Stage 12M figure generator, and a saved figure have SHA-256 entries in `finalization_audit/repository_freeze/supplemental_source_manifest.csv`. A canonical, fully independently verified Stage 12L/M workflow is not yet designated. Avoid publishing sample-level rank data or controlled-access participant identifiers.

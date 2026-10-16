#!/usr/bin/env python3
"""
Stage 12E — brain biological evidence matrix.

Goal
----
Create the exact evidence tables needed for the biological deep dive without
introducing new discovery thresholds or pooling biologically distinct sets.

Primary formal groups
---------------------
For NCI-vs-AD and CERAD:
    formal_shared
    formal_AREA_only
    formal_conventional_only

Secondary near-convergence groups
---------------------------------
For MCI-vs-AD and late Braak:
    near_shared
    near_AREA_only
    near_conventional_only

"Near-convergence" is inherited from Stage 12D:
    same RNA/protein direction
    AND RNA padj < 0.05
    AND protein padj < 0.05
    AND conjunction BH FDR >= 0.05

Key improvement over Stage 12D
------------------------------
For every candidate gene, this script returns FULL statistics from BOTH
analysis frameworks, even when only one framework crossed the formal
conjunction threshold.

Thus an AREA-only formal gene can be distinguished as:
    - conventional result nearly significant
    - conventional result moderately supported
    - conventional result weak / discordant

without inventing a new threshold.

No pathway enrichment is performed here. This script creates the frozen,
auditable gene sets and evidence matrix that should be used for the subsequent
literature/pathway analysis.

Outputs
-------
results/stage12_brain_convergence/stage12E_biological_evidence_matrix/

    MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv
    FORMAL_SHARED_FULL_EVIDENCE.tsv
    FORMAL_AREA_ONLY_FULL_EVIDENCE.tsv
    FORMAL_CONVENTIONAL_ONLY_FULL_EVIDENCE.tsv
    NEAR_CONVERGENCE_FULL_EVIDENCE.tsv
    CROSS_CONTRAST_GENE_RECURRENCE.tsv
    EVIDENCE_CLASS_SUMMARY.tsv
    FORMAL_SHARED_GENES.txt
    FORMAL_AREA_ONLY_GENES.txt
    FORMAL_CONVENTIONAL_ONLY_GENES.txt
    NEAR_SHARED_GENES.txt
    README_STAGE12E.txt
"""

from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(os.environ["ROSMAP_OFFLINE_ROOT"]).expanduser().resolve()
AREA_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
    / "ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv"
)

CONV_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12B_deseq2_limma_conjunction"
    / "ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv"
)

STAGE12D_FORMAL = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12D_biological_candidate_sets"
    / "FORMAL_BRAIN_DISCOVERY_CATEGORIES.tsv"
)

STAGE12D_NEAR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12D_biological_candidate_sets"
    / "MCI_AD_LATE_BRAAK_NEAR_CONVERGENCE.tsv"
)

MONOCYTE_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12C_monocyte_followup"
    / "BRAIN_CANDIDATE_MONOCYTE_EVIDENCE.tsv"
)

OUTDIR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12E_biological_evidence_matrix"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


FORMAL_CONTRASTS = {
    "cogdx_NCI_vs_AD",
    "CERAD_3-4_vs_1-2",
}

NEAR_CONTRASTS = {
    "cogdx_MCI_vs_AD",
    "Braak_3-4_vs_5-6",
}


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def require_columns(df, columns, label):
    missing = set(columns) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {sorted(missing)}"
        )


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def load_inputs():
    for path in [
        AREA_PATH,
        CONV_PATH,
        STAGE12D_FORMAL,
        STAGE12D_NEAR,
        MONOCYTE_PATH,
    ]:
        require_file(path)

    area = pd.read_csv(
        AREA_PATH,
        sep="\t",
        low_memory=False,
    )

    conv = pd.read_csv(
        CONV_PATH,
        sep="\t",
        low_memory=False,
    )

    formal = pd.read_csv(
        STAGE12D_FORMAL,
        sep="\t",
        low_memory=False,
    )

    near = pd.read_csv(
        STAGE12D_NEAR,
        sep="\t",
        low_memory=False,
    )

    mono = pd.read_csv(
        MONOCYTE_PATH,
        sep="\t",
        low_memory=False,
    )

    return area, conv, formal, near, mono


def prepare_area(area):
    require_columns(
        area,
        [
            "contrast_id",
            "gene_symbol",
            "direction_concordant",
            "conjunction_pvalue",
            "conjunction_fdr_bh",
            "crossomic_fdr05_bh",
            "both_individual_fdr05",
            "rna_pvalue",
            "rna_padj",
            "rna_direction",
            "protein_pvalue",
            "protein_padj",
            "protein_direction",
        ],
        "AREA",
    )

    x = area.copy()

    x = x.rename(
        columns={
            "direction_concordant":
                "AREA_direction_concordant",
            "conjunction_pvalue":
                "AREA_conjunction_pvalue",
            "conjunction_fdr_bh":
                "AREA_conjunction_fdr_bh",
            "crossomic_fdr05_bh":
                "AREA_formal_BH",
            "both_individual_fdr05":
                "AREA_both_individual_FDR05",
            "rna_pvalue":
                "AREA_RNA_pvalue",
            "rna_padj":
                "AREA_RNA_padj",
            "rna_direction":
                "AREA_RNA_direction",
            "protein_pvalue":
                "AREA_protein_pvalue",
            "protein_padj":
                "AREA_protein_padj",
            "protein_direction":
                "AREA_protein_direction",
        }
    )

    keep = [
        "contrast_id",
        "gene_symbol",
        "AREA_direction_concordant",
        "AREA_conjunction_pvalue",
        "AREA_conjunction_fdr_bh",
        "AREA_formal_BH",
        "AREA_both_individual_FDR05",
        "AREA_RNA_pvalue",
        "AREA_RNA_padj",
        "AREA_RNA_direction",
        "AREA_protein_pvalue",
        "AREA_protein_padj",
        "AREA_protein_direction",
    ]

    x = x[keep].copy()

    for c in [
        "AREA_direction_concordant",
        "AREA_formal_BH",
        "AREA_both_individual_FDR05",
    ]:
        x[c] = bool_series(x[c])

    if x.duplicated(
        ["contrast_id", "gene_symbol"]
    ).any():
        raise RuntimeError(
            "Duplicate AREA contrast/gene rows."
        )

    return x


def prepare_conv(conv):
    require_columns(
        conv,
        [
            "contrast_id",
            "gene_symbol",
            "direction_concordant",
            "conjunction_pvalue",
            "conjunction_fdr_bh",
            "crossomic_fdr05_bh",
            "both_individual_fdr05",
            "rna_pvalue",
            "rna_padj",
            "rna_effect",
            "protein_pvalue",
            "protein_padj",
            "protein_effect",
        ],
        "conventional",
    )

    x = conv.copy()

    x = x.rename(
        columns={
            "direction_concordant":
                "conventional_direction_concordant",
            "conjunction_pvalue":
                "conventional_conjunction_pvalue",
            "conjunction_fdr_bh":
                "conventional_conjunction_fdr_bh",
            "crossomic_fdr05_bh":
                "conventional_formal_BH",
            "both_individual_fdr05":
                "conventional_both_individual_FDR05",
            "rna_pvalue":
                "DESeq2_pvalue",
            "rna_padj":
                "DESeq2_padj",
            "rna_effect":
                "DESeq2_log2FC",
            "protein_pvalue":
                "limma_pvalue",
            "protein_padj":
                "limma_padj",
            "protein_effect":
                "limma_logFC",
        }
    )

    keep = [
        "contrast_id",
        "gene_symbol",
        "conventional_direction_concordant",
        "conventional_conjunction_pvalue",
        "conventional_conjunction_fdr_bh",
        "conventional_formal_BH",
        "conventional_both_individual_FDR05",
        "DESeq2_pvalue",
        "DESeq2_padj",
        "DESeq2_log2FC",
        "limma_pvalue",
        "limma_padj",
        "limma_logFC",
    ]

    x = x[keep].copy()

    for c in [
        "conventional_direction_concordant",
        "conventional_formal_BH",
        "conventional_both_individual_FDR05",
    ]:
        x[c] = bool_series(x[c])

    if x.duplicated(
        ["contrast_id", "gene_symbol"]
    ).any():
        raise RuntimeError(
            "Duplicate conventional contrast/gene rows."
        )

    return x


def candidate_classes(formal, near):
    require_columns(
        formal,
        [
            "contrast_id",
            "gene_symbol",
            "brain_category",
        ],
        "Stage12D formal",
    )

    formal_x = formal[
        [
            "contrast_id",
            "gene_symbol",
            "brain_category",
        ]
    ].copy()

    formal_x["evidence_track"] = "formal"
    formal_x["evidence_class"] = (
        "formal_"
        + formal_x["brain_category"].astype(str)
    )

    if len(near) > 0:
        require_columns(
            near,
            [
                "contrast_id",
                "gene_symbol",
                "near_category",
            ],
            "Stage12D near",
        )

        near_x = near[
            [
                "contrast_id",
                "gene_symbol",
                "near_category",
            ]
        ].copy()

        near_x["evidence_track"] = "near_convergence"
        near_x["evidence_class"] = (
            "near_"
            + near_x["near_category"]
            .astype(str)
            .str.replace("_near", "", regex=False)
        )
    else:
        near_x = pd.DataFrame(
            columns=[
                "contrast_id",
                "gene_symbol",
                "near_category",
                "evidence_track",
                "evidence_class",
            ]
        )

    formal_x["near_category"] = np.nan
    near_x["brain_category"] = np.nan

    cols = [
        "contrast_id",
        "gene_symbol",
        "evidence_track",
        "evidence_class",
        "brain_category",
        "near_category",
    ]

    candidates = pd.concat(
        [
            formal_x[cols],
            near_x[cols],
        ],
        ignore_index=True,
    )

    if candidates.duplicated(
        ["contrast_id", "gene_symbol"]
    ).any():
        dup = candidates.loc[
            candidates.duplicated(
                ["contrast_id", "gene_symbol"],
                keep=False,
            )
        ]
        raise RuntimeError(
            "A contrast/gene appears in multiple Stage12D "
            f"candidate classes:\n{dup.to_string(index=False)}"
        )

    return candidates


def prepare_monocyte(mono):
    if len(mono) == 0:
        return pd.DataFrame(
            columns=["contrast_id", "gene_symbol"]
        )

    base = [
        "contrast_id",
        "gene_symbol",
    ]

    optional = [
        "area_pvalue",
        "area_padj",
        "area_direction",
        "deseq2_log2fc_stage11_orientation",
        "deseq2_pvalue",
        "deseq2_padj",
        "monocyte_area_fdr05",
        "monocyte_deseq2_fdr05",
        "monocyte_methods_direction_agree",
        "monocyte_area_matches_brain_area_direction",
        "monocyte_deseq2_matches_brain_conventional_direction",
    ]

    keep = base + [
        c
        for c in optional
        if c in mono.columns
    ]

    x = mono[keep].copy()

    rename = {
        "area_pvalue": "monocyte_AREA_pvalue",
        "area_padj": "monocyte_AREA_padj",
        "area_direction": "monocyte_AREA_direction",
        "deseq2_log2fc_stage11_orientation":
            "monocyte_DESeq2_log2FC",
        "deseq2_pvalue": "monocyte_DESeq2_pvalue",
        "deseq2_padj": "monocyte_DESeq2_padj",
        "monocyte_area_fdr05":
            "monocyte_AREA_FDR05",
        "monocyte_deseq2_fdr05":
            "monocyte_DESeq2_FDR05",
    }

    x = x.rename(columns=rename)

    if x.duplicated(
        ["contrast_id", "gene_symbol"]
    ).any():
        raise RuntimeError(
            "Duplicate monocyte contrast/gene rows."
        )

    return x


def build_master(area, conv, candidates, mono):
    master = candidates.merge(
        area,
        on=["contrast_id", "gene_symbol"],
        how="left",
        validate="one_to_one",
    )

    master = master.merge(
        conv,
        on=["contrast_id", "gene_symbol"],
        how="left",
        validate="one_to_one",
    )

    master = master.merge(
        mono,
        on=["contrast_id", "gene_symbol"],
        how="left",
        validate="one_to_one",
    )

    # Explicit support descriptors based only on already-established
    # inferential states; no new cutoffs are introduced.
    master["AREA_status"] = np.select(
        [
            master["AREA_formal_BH"].fillna(False),
            master[
                "AREA_both_individual_FDR05"
            ].fillna(False),
        ],
        [
            "formal_conjunction_BH",
            "both_modalities_individual_FDR05_only",
        ],
        default="no_dual_FDR05_support",
    )

    master["conventional_status"] = np.select(
        [
            master[
                "conventional_formal_BH"
            ].fillna(False),
            master[
                "conventional_both_individual_FDR05"
            ].fillna(False),
        ],
        [
            "formal_conjunction_BH",
            "both_modalities_individual_FDR05_only",
        ],
        default="no_dual_FDR05_support",
    )

    # Is the "other" method direction-concordant even if it did not cross
    # formal BH? This is useful for method-specific biological interpretation.
    master[
        "both_frameworks_direction_concordant"
    ] = (
        master["AREA_direction_concordant"].fillna(False)
        & master[
            "conventional_direction_concordant"
        ].fillna(False)
    )

    return master


def build_recurrence(master):
    rows = []

    for gene, g in master.groupby(
        "gene_symbol",
        sort=False,
    ):
        classes = sorted(
            set(g["evidence_class"].astype(str))
        )

        contrasts = sorted(
            set(g["contrast_id"].astype(str))
        )

        formal = g.loc[
            g["evidence_track"].eq("formal")
        ]

        near = g.loc[
            g["evidence_track"].eq("near_convergence")
        ]

        rows.append(
            {
                "gene_symbol": gene,
                "n_candidate_contrasts": len(contrasts),
                "n_formal_contrasts": len(formal),
                "n_near_convergence_contrasts": len(near),
                "candidate_contrasts": ",".join(contrasts),
                "evidence_classes": ",".join(classes),
                "formal_shared_n":
                    int(
                        formal["evidence_class"]
                        .eq("formal_shared")
                        .sum()
                    ),
                "formal_AREA_only_n":
                    int(
                        formal["evidence_class"]
                        .eq("formal_AREA_only")
                        .sum()
                    ),
                "formal_conventional_only_n":
                    int(
                        formal["evidence_class"]
                        .eq(
                            "formal_conventional_only"
                        )
                        .sum()
                    ),
                "near_shared_n":
                    int(
                        near["evidence_class"]
                        .eq("near_shared")
                        .sum()
                    ),
                "near_AREA_only_n":
                    int(
                        near["evidence_class"]
                        .eq("near_AREA_only")
                        .sum()
                    ),
                "near_conventional_only_n":
                    int(
                        near["evidence_class"]
                        .eq(
                            "near_conventional_only"
                        )
                        .sum()
                    ),
            }
        )

    out = pd.DataFrame(rows)

    return (
        out
        .sort_values(
            [
                "n_formal_contrasts",
                "n_near_convergence_contrasts",
                "n_candidate_contrasts",
                "gene_symbol",
            ],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def write_gene_list(path, df):
    genes = sorted(
        set(df["gene_symbol"].dropna().astype(str))
    )

    path.write_text(
        "\n".join(genes)
        + ("\n" if genes else "")
    )


def main():
    area_raw, conv_raw, formal, near, mono_raw = (
        load_inputs()
    )

    area = prepare_area(area_raw)
    conv = prepare_conv(conv_raw)
    candidates = candidate_classes(
        formal,
        near,
    )
    mono = prepare_monocyte(mono_raw)

    master = build_master(
        area,
        conv,
        candidates,
        mono,
    )

    recurrence = build_recurrence(master)

    summary = (
        master
        .groupby(
            [
                "evidence_track",
                "contrast_id",
                "evidence_class",
            ],
            dropna=False,
            sort=False,
        )
        .size()
        .rename("n_genes")
        .reset_index()
    )

    master_path = (
        OUTDIR / "MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv"
    )
    shared_path = (
        OUTDIR / "FORMAL_SHARED_FULL_EVIDENCE.tsv"
    )
    area_only_path = (
        OUTDIR / "FORMAL_AREA_ONLY_FULL_EVIDENCE.tsv"
    )
    conv_only_path = (
        OUTDIR
        / "FORMAL_CONVENTIONAL_ONLY_FULL_EVIDENCE.tsv"
    )
    near_path = (
        OUTDIR / "NEAR_CONVERGENCE_FULL_EVIDENCE.tsv"
    )
    recurrence_path = (
        OUTDIR / "CROSS_CONTRAST_GENE_RECURRENCE.tsv"
    )
    summary_path = (
        OUTDIR / "EVIDENCE_CLASS_SUMMARY.tsv"
    )

    master.to_csv(
        master_path,
        sep="\t",
        index=False,
    )

    master.loc[
        master["evidence_class"].eq("formal_shared")
    ].to_csv(
        shared_path,
        sep="\t",
        index=False,
    )

    master.loc[
        master["evidence_class"].eq("formal_AREA_only")
    ].to_csv(
        area_only_path,
        sep="\t",
        index=False,
    )

    master.loc[
        master["evidence_class"].eq(
            "formal_conventional_only"
        )
    ].to_csv(
        conv_only_path,
        sep="\t",
        index=False,
    )

    master.loc[
        master["evidence_track"].eq("near_convergence")
    ].to_csv(
        near_path,
        sep="\t",
        index=False,
    )

    recurrence.to_csv(
        recurrence_path,
        sep="\t",
        index=False,
    )

    summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
    )

    write_gene_list(
        OUTDIR / "FORMAL_SHARED_GENES.txt",
        master.loc[
            master["evidence_class"].eq(
                "formal_shared"
            )
        ],
    )

    write_gene_list(
        OUTDIR / "FORMAL_AREA_ONLY_GENES.txt",
        master.loc[
            master["evidence_class"].eq(
                "formal_AREA_only"
            )
        ],
    )

    write_gene_list(
        OUTDIR / "FORMAL_CONVENTIONAL_ONLY_GENES.txt",
        master.loc[
            master["evidence_class"].eq(
                "formal_conventional_only"
            )
        ],
    )

    write_gene_list(
        OUTDIR / "NEAR_SHARED_GENES.txt",
        master.loc[
            master["evidence_class"].eq(
                "near_shared"
            )
        ],
    )

    readme = """STAGE 12E — BIOLOGICAL EVIDENCE MATRIX
======================================

Purpose
-------
Freeze the gene sets and full statistical evidence used for the biological
deep dive.

No new discovery thresholds are introduced.

PRIMARY BIOLOGICAL COMPARISON
-----------------------------
formal_shared
    Cross-omic conjunction BH FDR < 0.05 in BOTH:
        AREA
        DESeq2 RNA + limma protein

formal_AREA_only
    Formal cross-omic conjunction discovery in AREA but not in the
    conventional framework.

formal_conventional_only
    Formal cross-omic conjunction discovery in DESeq2/limma but not AREA.

IMPORTANT
---------
"AREA-only" does NOT mean the conventional method has no evidence.

MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv therefore contains the FULL
conventional statistics for every AREA-only gene, and vice versa. This is
the correct table for asking whether a method-specific discovery reflects:

    different rank/statistical strength,
    failure of one modality in the other framework,
    direction discordance,
    or a near-threshold difference in conjunction FDR.

NEAR-CONVERGENCE
----------------
For MCI->AD and late Braak, near-convergence remains:

    same RNA/protein direction
    AND RNA individual FDR < 0.05
    AND protein individual FDR < 0.05
    AND conjunction BH FDR >= 0.05

These are NOT formal cross-omic discoveries.

They are retained because both AREA and conventional methods showed abundant
within-modality signal in these contrasts despite zero formal conjunction
discoveries.

MONOCYTE
--------
Where available, Stage 12C monocyte evidence is attached to the formal
brain-derived genes.

Monocyte is a peripheral follow-up layer, not a filter on brain discovery.

NEXT STEP
---------
Use these frozen sets for biological interpretation:

    1. formal_shared
    2. formal_AREA_only
    3. formal_conventional_only as comparator
    4. MCI->AD / late-Braak near-convergence

Pathway/literature interpretation should keep these evidence classes separate.
"""

    readme_path = (
        OUTDIR / "README_STAGE12E.txt"
    )
    readme_path.write_text(readme)

    print("=" * 120)
    print("STAGE 12E — BRAIN BIOLOGICAL EVIDENCE MATRIX")
    print("=" * 120)

    print("\nEVIDENCE CLASS SUMMARY")
    print("-" * 120)
    print(
        summary.to_string(index=False)
    )

    print("\nCROSS-CONTRAST RECURRENCE — TOP")
    print("-" * 120)
    print(
        recurrence.head(60).to_string(index=False)
    )

    print("\nFORMAL AREA-ONLY — FULL OTHER-METHOD EVIDENCE")
    print("-" * 120)

    area_only = master.loc[
        master["evidence_class"].eq(
            "formal_AREA_only"
        )
    ].copy()

    show = [
        "contrast_id",
        "gene_symbol",
        "AREA_conjunction_fdr_bh",
        "conventional_conjunction_fdr_bh",
        "conventional_direction_concordant",
        "DESeq2_padj",
        "limma_padj",
        "DESeq2_log2FC",
        "limma_logFC",
    ]

    print(
        area_only[show]
        .round(6)
        .to_string(index=False)
    )

    print("\nFORMAL SHARED — FULL EVIDENCE")
    print("-" * 120)

    shared = master.loc[
        master["evidence_class"].eq(
            "formal_shared"
        )
    ].copy()

    show_shared = [
        "contrast_id",
        "gene_symbol",
        "AREA_conjunction_fdr_bh",
        "conventional_conjunction_fdr_bh",
        "AREA_RNA_padj",
        "AREA_protein_padj",
        "DESeq2_padj",
        "limma_padj",
    ]

    print(
        shared[show_shared]
        .round(6)
        .to_string(index=False)
    )

    print("\nWROTE:")
    for path in [
        master_path,
        shared_path,
        area_only_path,
        conv_only_path,
        near_path,
        recurrence_path,
        summary_path,
        OUTDIR / "FORMAL_SHARED_GENES.txt",
        OUTDIR / "FORMAL_AREA_ONLY_GENES.txt",
        OUTDIR / "FORMAL_CONVENTIONAL_ONLY_GENES.txt",
        OUTDIR / "NEAR_SHARED_GENES.txt",
        readme_path,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()

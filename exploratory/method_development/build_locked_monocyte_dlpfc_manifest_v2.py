#!/usr/bin/env python3
"""
build_locked_monocyte_dlpfc_manifest_v2.py
=======================================

Build the locked same-participant ROSMAP monocyte RNA-seq ↔ DLPFC RNA-seq
analysis manifest for the cross-tissue AREA / Weighted AREA study.

This script starts from the successfully rebuilt matched participant/specimen
tables with continuous pathology and produces a publication-ready locked cohort
manifest.

Expected inputs
---------------
results/monocyte_dlpfc_feasibility/pathology_diagnostic/
    monocyte_dlpfc_matched_specimen_table_with_continuous_pathology.csv
    monocyte_dlpfc_matched_participant_table_with_continuous_pathology.csv

Optional technical metadata
---------------------------
The script also searches for:
    ROSMAP_assay_rnaSeq_metadata.csv
    ROSMAP_biospecimen_metadata.csv

and, when available, merges monocyte and DLPFC technical covariates such as:
    RNA batch
    sequencing batch
    library prep / assay metadata
    RIN
    PMI

Primary outputs
---------------
results/monocyte_dlpfc_cross_tissue/

    locked_cross_tissue_manifest.csv
    locked_cross_tissue_manifest_minimal.csv
    locked_cross_tissue_phenotype_counts.csv
    locked_cross_tissue_covariate_audit.csv
    locked_cross_tissue_exclusions.csv
    locked_cross_tissue_summary.txt

Design
------
- one row per matched participant
- exactly one monocyte specimen per participant
- one DLPFC specimen identifier field per participant
- preserve all 168 matched participants unless a required identifier is missing
- phenotype-specific missingness is recorded rather than globally excluding
  participants
- exact age_death is used; age_numeric is not used
- 90+ monocyte sampling age is preserved as censored and is not converted to an
  exact blood-to-death interval

Recommended downstream use
--------------------------
Use this locked manifest as the participant backbone for BOTH tissues so that
blood and brain analyses use the same people and same phenotype definitions.

Run from repo root:
    python3 exploratory/method_development/build_locked_monocyte_dlpfc_manifest_v2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_INPUT_DIR = Path(
    "results/monocyte_dlpfc_feasibility/pathology_diagnostic"
)

DEFAULT_OUTDIR = Path(
    "results/monocyte_dlpfc_cross_tissue"
)


# =============================================================================
# HELPERS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--specimen-table",
        default=str(
            DEFAULT_INPUT_DIR
            / "monocyte_dlpfc_matched_specimen_table_with_continuous_pathology.csv"
        ),
    )

    p.add_argument(
        "--participant-table",
        default=str(
            DEFAULT_INPUT_DIR
            / "monocyte_dlpfc_matched_participant_table_with_continuous_pathology.csv"
        ),
    )

    p.add_argument(
        "--biospecimen",
        default=None,
    )

    p.add_argument(
        "--assay-metadata",
        default=None,
    )

    p.add_argument(
        "--outdir",
        default="results/monocyte_dlpfc_cross_tissue_v2",
    )

    return p.parse_args()


def first_existing(paths: Iterable[str]) -> Optional[Path]:
    for x in paths:
        p = Path(x).expanduser()
        if p.exists():
            return p
    return None


def resolve(df: pd.DataFrame, candidates, required=False):
    lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]

    if required:
        raise ValueError(
            f"Could not resolve required column from {candidates}.\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def first_nonnull(series):
    x = series.dropna()
    return x.iloc[0] if len(x) else np.nan


def semicolon_unique(series):
    values = [
        str(x).strip()
        for x in series
        if pd.notna(x)
        and str(x).strip()
        and str(x).strip().lower() != "nan"
    ]
    return ";".join(sorted(set(values))) if values else np.nan


def split_first_specimen(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    return text.split(";")[0]


# =============================================================================
# OPTIONAL METADATA DISCOVERY
# =============================================================================

def discover_optional(args):
    biospec = (
        Path(args.biospecimen).expanduser()
        if args.biospecimen
        else first_existing(
            [
                "data/ROSMAP_biospecimen_metadata.csv",
                "~/Downloads/ROSMAP_biospecimen_metadata.csv",
                "ROSMAP_biospecimen_metadata.csv",
            ]
        )
    )

    assay = (
        Path(args.assay_metadata).expanduser()
        if args.assay_metadata
        else first_existing(
            [
                "data/ROSMAP_assay_rnaSeq_metadata.csv",
                "~/Downloads/ROSMAP_assay_rnaSeq_metadata.csv",
                "ROSMAP_assay_rnaSeq_metadata.csv",
            ]
        )
    )

    return biospec, assay


# =============================================================================
# BUILD LOCKED MANIFEST
# =============================================================================

def build_base_manifest(specimen, participant):
    # Participant table should already be one row/person.
    if participant["individualID"].duplicated().any():
        raise ValueError(
            "Participant table contains duplicate individualID values."
        )

    manifest = participant.copy()

    # Pull specimen-level identifiers/sampling age back into participant table.
    specimen_summary = (
        specimen.groupby("individualID", as_index=False)
        .agg(
            monocyte_specimen_id_from_specimen=(
                "monocyte_specimen_id",
                semicolon_unique,
            ),
            monocyte_sampling_age_raw_from_specimen=(
                "monocyte_sampling_age_raw",
                first_nonnull,
            ),
            monocyte_sampling_age_exact_from_specimen=(
                "monocyte_sampling_age_exact",
                first_nonnull,
            ),
            monocyte_sampling_age_lower_bound_from_specimen=(
                "monocyte_sampling_age_lower_bound",
                first_nonnull,
            ),
            dlpfc_specimen_ids_from_specimen=(
                "dlpfc_specimen_ids",
                first_nonnull,
            ),
        )
    )

    manifest = manifest.merge(
        specimen_summary,
        on="individualID",
        how="left",
        validate="one_to_one",
    )

    # Standardize identifiers.
    if "monocyte_specimen_id" not in manifest.columns:
        manifest["monocyte_specimen_id"] = (
            manifest["monocyte_specimen_id_from_specimen"]
        )
    else:
        manifest["monocyte_specimen_id"] = (
            manifest["monocyte_specimen_id"]
            .where(
                manifest["monocyte_specimen_id"].notna(),
                manifest["monocyte_specimen_id_from_specimen"],
            )
        )

    if "dlpfc_specimen_ids" not in manifest.columns:
        manifest["dlpfc_specimen_ids"] = (
            manifest["dlpfc_specimen_ids_from_specimen"]
        )
    else:
        manifest["dlpfc_specimen_ids"] = (
            manifest["dlpfc_specimen_ids"]
            .where(
                manifest["dlpfc_specimen_ids"].notna(),
                manifest["dlpfc_specimen_ids_from_specimen"],
            )
        )

    manifest["dlpfc_specimen_id_primary"] = (
        manifest["dlpfc_specimen_ids"].map(
            split_first_specimen
        )
    )

    # Ensure monocyte sampling ages are present.
    for target, source in [
        (
            "monocyte_sampling_age_raw",
            "monocyte_sampling_age_raw_from_specimen",
        ),
        (
            "monocyte_sampling_age_exact",
            "monocyte_sampling_age_exact_from_specimen",
        ),
        (
            "monocyte_sampling_age_lower_bound",
            "monocyte_sampling_age_lower_bound_from_specimen",
        ),
    ]:
        if target not in manifest.columns:
            manifest[target] = manifest[source]
        else:
            manifest[target] = (
                manifest[target]
                .where(
                    manifest[target].notna(),
                    manifest[source],
                )
            )

    return manifest


# =============================================================================
# OPTIONAL TECHNICAL COVARIATES
# =============================================================================

def merge_biospecimen_technical(manifest, biospec_path):
    audit = []

    expected_rows = len(manifest)
    expected_unique = manifest["individualID"].nunique()

    if biospec_path is None or not biospec_path.exists():
        audit.append(
            {
                "source": "biospecimen",
                "status": "not_found",
                "details": "",
                "rows_before": expected_rows,
                "rows_after": expected_rows,
                "unique_participants_before": expected_unique,
                "unique_participants_after": expected_unique,
            }
        )
        return manifest, audit

    bio = pd.read_csv(
        biospec_path,
        low_memory=False,
    )

    specimen_col = resolve(
        bio,
        ["specimenID", "specimen_id"],
        required=True,
    )

    candidates = [
        "rnaBatch",
        "batch",
        "sequencingBatch",
        "sequencing_batch",
        "libraryPrep",
        "library_prep",
        "rin",
        "RIN",
        "pmi",
        "PMI",
        "samplingAge",
        "isPostMortem",
        "organ",
        "tissue",
        "cellType",
        "assay",
    ]

    available = [
        c for c in candidates
        if c in bio.columns
    ]

    # Collapse to exactly one row per specimen before any merge.
    collapsed = _collapse_one_row_per_specimen(
        bio,
        specimen_col,
        available,
    )

    mono = collapsed.copy().rename(
        columns={
            specimen_col: "monocyte_specimen_id",
            **{
                c: f"monocyte_{c}"
                for c in available
            },
        }
    )

    manifest = manifest.merge(
        mono,
        on="monocyte_specimen_id",
        how="left",
        validate="many_to_one",
    )

    _assert_manifest_integrity(
        manifest,
        expected_rows,
        expected_unique,
        "after monocyte biospecimen metadata merge",
    )

    brain = collapsed.copy().rename(
        columns={
            specimen_col: "dlpfc_specimen_id_primary",
            **{
                c: f"dlpfc_{c}"
                for c in available
            },
        }
    )

    manifest = manifest.merge(
        brain,
        on="dlpfc_specimen_id_primary",
        how="left",
        validate="many_to_one",
    )

    _assert_manifest_integrity(
        manifest,
        expected_rows,
        expected_unique,
        "after DLPFC biospecimen metadata merge",
    )

    audit.append(
        {
            "source": "biospecimen",
            "status": "merged_collapsed_one_row_per_specimen",
            "details": (
                f"{biospec_path}; raw_rows={len(bio)}; "
                f"collapsed_rows={len(collapsed)}; "
                f"fields=" + ",".join(available)
            ),
            "rows_before": expected_rows,
            "rows_after": len(manifest),
            "unique_participants_before": expected_unique,
            "unique_participants_after": manifest["individualID"].nunique(),
        }
    )

    return manifest, audit


def _collapse_one_row_per_specimen(df, specimen_col, fields):
    """
    Collapse metadata to exactly one row per specimen.

    For each field:
      - if there is one unique non-null value, keep it
      - if there are multiple unique non-null values, join them with ';'
        so the ambiguity is preserved rather than duplicating rows
    """
    def collapse_value(series):
        vals = (
            series.dropna()
            .astype(str)
            .map(str.strip)
        )
        vals = [
            v for v in vals
            if v and v.lower() not in {"nan", "none"}
        ]

        if not vals:
            return np.nan

        unique = sorted(set(vals))

        if len(unique) == 1:
            return unique[0]

        return ";".join(unique)

    agg = {
        field: collapse_value
        for field in fields
    }

    collapsed = (
        df[[specimen_col] + fields]
        .groupby(
            specimen_col,
            as_index=False,
            dropna=False,
        )
        .agg(agg)
    )

    return collapsed


def _assert_manifest_integrity(manifest, expected_rows, expected_unique, stage):
    """
    Hard fail if a metadata merge expands or duplicates the locked cohort.
    """
    observed_rows = len(manifest)
    observed_unique = manifest["individualID"].nunique()

    if observed_rows != expected_rows:
        raise RuntimeError(
            f"{stage}: row count changed from {expected_rows} to "
            f"{observed_rows}. A metadata merge expanded the cohort."
        )

    if observed_unique != expected_unique:
        raise RuntimeError(
            f"{stage}: unique participant count changed from "
            f"{expected_unique} to {observed_unique}."
        )

    if manifest["individualID"].duplicated().any():
        dupes = (
            manifest.loc[
                manifest["individualID"].duplicated(keep=False),
                "individualID",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"{stage}: duplicate individualID rows remain after merge. "
            f"Examples: {dupes[:10]}"
        )


def merge_assay_metadata(manifest, assay_path):
    audit = []

    expected_rows = len(manifest)
    expected_unique = manifest["individualID"].nunique()

    if assay_path is None or not assay_path.exists():
        audit.append(
            {
                "source": "rna_assay_metadata",
                "status": "not_found",
                "details": "",
                "rows_before": expected_rows,
                "rows_after": expected_rows,
                "unique_participants_before": expected_unique,
                "unique_participants_after": expected_unique,
            }
        )
        return manifest, audit

    assay = pd.read_csv(
        assay_path,
        low_memory=False,
    )

    specimen_col = resolve(
        assay,
        [
            "specimenID",
            "specimen_id",
            "sample_id",
        ],
    )

    if specimen_col is None:
        audit.append(
            {
                "source": "rna_assay_metadata",
                "status": "unusable",
                "details": (
                    f"{assay_path}; no specimen/sample ID"
                ),
                "rows_before": expected_rows,
                "rows_after": expected_rows,
                "unique_participants_before": expected_unique,
                "unique_participants_after": expected_unique,
            }
        )
        return manifest, audit

    useful = [
        c for c in assay.columns
        if c != specimen_col
        and any(
            key in c.lower()
            for key in [
                "batch",
                "platform",
                "library",
                "rin",
                "read",
                "sequenc",
                "assay",
            ]
        )
    ]

    if not useful:
        audit.append(
            {
                "source": "rna_assay_metadata",
                "status": "no_useful_fields",
                "details": str(assay_path),
                "rows_before": expected_rows,
                "rows_after": expected_rows,
                "unique_participants_before": expected_unique,
                "unique_participants_after": expected_unique,
            }
        )
        return manifest, audit

    # -----------------------------------------------------------------
    # CRITICAL FIX:
    # collapse assay metadata to ONE row per specimen BEFORE merging.
    # -----------------------------------------------------------------
    assay_collapsed = _collapse_one_row_per_specimen(
        assay,
        specimen_col,
        useful,
    )

    n_raw = len(assay)
    n_collapsed = len(assay_collapsed)
    n_unique_specimens = assay[specimen_col].nunique(dropna=True)

    # Monocyte side.
    mono = assay_collapsed.copy()

    mono = mono.rename(
        columns={
            specimen_col: "monocyte_specimen_id",
            **{
                c: f"monocyte_assay_{c}"
                for c in useful
            },
        }
    )

    manifest = manifest.merge(
        mono,
        on="monocyte_specimen_id",
        how="left",
        validate="many_to_one",
    )

    _assert_manifest_integrity(
        manifest,
        expected_rows,
        expected_unique,
        "after monocyte assay metadata merge",
    )

    # DLPFC side.
    brain = assay_collapsed.copy()

    brain = brain.rename(
        columns={
            specimen_col: "dlpfc_specimen_id_primary",
            **{
                c: f"dlpfc_assay_{c}"
                for c in useful
            },
        }
    )

    manifest = manifest.merge(
        brain,
        on="dlpfc_specimen_id_primary",
        how="left",
        validate="many_to_one",
    )

    _assert_manifest_integrity(
        manifest,
        expected_rows,
        expected_unique,
        "after DLPFC assay metadata merge",
    )

    audit.append(
        {
            "source": "rna_assay_metadata",
            "status": "merged_collapsed_one_row_per_specimen",
            "details": (
                f"{assay_path}; raw_rows={n_raw}; "
                f"collapsed_rows={n_collapsed}; "
                f"unique_specimens={n_unique_specimens}; "
                f"fields=" + ",".join(useful)
            ),
            "rows_before": expected_rows,
            "rows_after": len(manifest),
            "unique_participants_before": expected_unique,
            "unique_participants_after": manifest["individualID"].nunique(),
        }
    )

    return manifest, audit


# =============================================================================
# PHENOTYPE AVAILABILITY / EXCLUSIONS
# =============================================================================

def add_analysis_flags(manifest):
    phenotype_map = {
        "cognition_global": "global_cognition",
        "mmse": "mmse",
        "cerad": "cerad",
        "braak": "braak",
        "amyloid_continuous": "amyloid_continuous",
        "tangles_continuous": "tangles_continuous",
    }

    for label, col in phenotype_map.items():
        if col in manifest.columns:
            manifest[
                f"eligible_{label}"
            ] = manifest[col].notna()
        else:
            manifest[
                f"eligible_{label}"
            ] = False

    # Core matched-tissue identifiers.
    manifest[
        "has_monocyte_rnaseq"
    ] = manifest[
        "monocyte_specimen_id"
    ].notna()

    manifest[
        "has_dlpfc_rnaseq"
    ] = manifest[
        "dlpfc_specimen_id_primary"
    ].notna()

    manifest[
        "locked_cross_tissue_core"
    ] = (
        manifest["has_monocyte_rnaseq"]
        & manifest["has_dlpfc_rnaseq"]
    )

    return manifest


def exclusion_table(manifest):
    rows = []

    for _, row in manifest.iterrows():
        reasons = []

        if not bool(
            row["has_monocyte_rnaseq"]
        ):
            reasons.append(
                "missing_monocyte_specimen"
            )

        if not bool(
            row["has_dlpfc_rnaseq"]
        ):
            reasons.append(
                "missing_dlpfc_specimen"
            )

        if reasons:
            rows.append(
                {
                    "individualID": row[
                        "individualID"
                    ],
                    "projid": row.get(
                        "projid",
                        np.nan,
                    ),
                    "reason": ";".join(
                        reasons
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "individualID",
            "projid",
            "reason",
        ],
    )


def phenotype_counts(manifest):
    rows = []

    phenotype_cols = [
        (
            "global_cognition",
            "eligible_cognition_global",
        ),
        (
            "MMSE",
            "eligible_mmse",
        ),
        (
            "CERAD",
            "eligible_cerad",
        ),
        (
            "Braak",
            "eligible_braak",
        ),
        (
            "continuous_amyloid",
            "eligible_amyloid_continuous",
        ),
        (
            "continuous_tangles",
            "eligible_tangles_continuous",
        ),
    ]

    for label, flag in phenotype_cols:
        rows.append(
            {
                "phenotype": label,
                "n_core_matched": int(
                    manifest[
                        "locked_cross_tissue_core"
                    ].sum()
                ),
                "n_eligible": int(
                    (
                        manifest[
                            "locked_cross_tissue_core"
                        ]
                        & manifest[flag]
                    ).sum()
                ),
                "n_missing_phenotype": int(
                    (
                        manifest[
                            "locked_cross_tissue_core"
                        ]
                        & ~manifest[flag]
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def minimal_manifest(manifest):
    preferred = [
        "individualID",
        "projid",
        "monocyte_specimen_id",
        "dlpfc_specimen_id_primary",
        "monocyte_sampling_age_raw",
        "monocyte_sampling_age_exact",
        "age_death_exact",
        "years_monocyte_to_death_exact",
        "cognitive_diagnosis_last_valid",
        "cognitive_diagnosis_at_death",
        "global_cognition",
        "mmse",
        "cerad",
        "braak",
        "amyloid_continuous",
        "tangles_continuous",
        "sex",
        "pmi",
        "eligible_cognition_global",
        "eligible_mmse",
        "eligible_cerad",
        "eligible_braak",
        "eligible_amyloid_continuous",
        "eligible_tangles_continuous",
        "locked_cross_tissue_core",
    ]

    cols = [
        c for c in preferred
        if c in manifest.columns
    ]

    return manifest[
        cols
    ].copy()


def write_summary(
    path,
    manifest,
    counts,
    audit,
):
    core_n = int(
        manifest[
            "locked_cross_tissue_core"
        ].sum()
    )

    exact_interval = (
        int(
            manifest[
                "years_monocyte_to_death_exact"
            ]
            .notna()
            .sum()
        )
        if "years_monocyte_to_death_exact"
        in manifest.columns
        else 0
    )

    lines = [
        "ROSMAP LOCKED MONOCYTE ↔ DLPFC CROSS-TISSUE COHORT",
        "=" * 72,
        "",
        f"Rows in manifest: {len(manifest)}",
        f"Core matched monocyte+DLPFC participants: {core_n}",
        f"Exact blood-to-death interval available: {exact_interval}",
        "",
        "Phenotype availability:",
        counts.to_string(index=False),
        "",
        "Technical metadata audit:",
        pd.DataFrame(audit).to_string(index=False),
        "",
        "Design note:",
        (
            "This manifest is intended to serve as the shared participant backbone "
            "for monocyte and DLPFC AREA / Weighted AREA analyses."
        ),
    ]

    path.write_text(
        "\n".join(lines)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print("=" * 100)
    print("BUILD LOCKED ROSMAP MONOCYTE ↔ DLPFC CROSS-TISSUE MANIFEST V2")
    print("=" * 100)
    print()

    specimen_path = Path(
        args.specimen_table
    )

    participant_path = Path(
        args.participant_table
    )

    if not specimen_path.exists():
        raise FileNotFoundError(
            f"Missing specimen table: {specimen_path}"
        )

    if not participant_path.exists():
        raise FileNotFoundError(
            f"Missing participant table: {participant_path}"
        )

    biospec_path, assay_path = discover_optional(
        args
    )

    print(
        f"Specimen table:   {specimen_path}"
    )
    print(
        f"Participant table:{participant_path}"
    )
    print(
        f"Biospecimen meta: {biospec_path if biospec_path else 'NOT FOUND'}"
    )
    print(
        f"RNA assay meta:   {assay_path if assay_path else 'NOT FOUND'}"
    )
    print()

    specimen = pd.read_csv(
        specimen_path,
        low_memory=False,
    )

    participant = pd.read_csv(
        participant_path,
        low_memory=False,
    )

    manifest = build_base_manifest(
        specimen,
        participant,
    )

    # The pathology-diagnostic participant table is the source of truth for
    # the locked matched cohort. No downstream technical metadata merge is
    # allowed to change this cohort size.
    expected_locked_rows = len(participant)
    expected_locked_unique = participant["individualID"].nunique()

    _assert_manifest_integrity(
        manifest,
        expected_locked_rows,
        expected_locked_unique,
        "base locked manifest",
    )

    print(
        f"Expected locked cohort: {expected_locked_rows} rows / "
        f"{expected_locked_unique} unique participants"
    )
    print()

    audit = []

    manifest, a = merge_biospecimen_technical(
        manifest,
        biospec_path,
    )
    audit.extend(a)

    manifest, a = merge_assay_metadata(
        manifest,
        assay_path,
    )
    audit.extend(a)

    manifest = add_analysis_flags(
        manifest
    )

    _assert_manifest_integrity(
        manifest,
        expected_locked_rows,
        expected_locked_unique,
        "final locked manifest",
    )

    # Sort deterministically.
    manifest = manifest.sort_values(
        [
            "individualID",
        ]
    ).reset_index(
        drop=True
    )

    exclusions = exclusion_table(
        manifest
    )

    counts = phenotype_counts(
        manifest
    )

    minimal = minimal_manifest(
        manifest
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        outdir
        / "locked_cross_tissue_manifest.csv"
    )

    minimal_path = (
        outdir
        / "locked_cross_tissue_manifest_minimal.csv"
    )

    counts_path = (
        outdir
        / "locked_cross_tissue_phenotype_counts.csv"
    )

    audit_path = (
        outdir
        / "locked_cross_tissue_covariate_audit.csv"
    )

    exclusion_path = (
        outdir
        / "locked_cross_tissue_exclusions.csv"
    )

    summary_path = (
        outdir
        / "locked_cross_tissue_summary.txt"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    minimal.to_csv(
        minimal_path,
        index=False,
    )

    counts.to_csv(
        counts_path,
        index=False,
    )

    pd.DataFrame(
        audit
    ).to_csv(
        audit_path,
        index=False,
    )

    exclusions.to_csv(
        exclusion_path,
        index=False,
    )

    write_summary(
        summary_path,
        manifest,
        counts,
        audit,
    )

    print(
        f"Locked rows: {len(manifest)}"
    )
    print(
        f"Core matched participants: "
        f"{int(manifest['locked_cross_tissue_core'].sum())}"
    )
    print()
    print(
        counts.to_string(
            index=False
        )
    )
    print()
    print("Outputs:")
    print(f"  {manifest_path}")
    print(f"  {minimal_path}")
    print(f"  {counts_path}")
    print(f"  {audit_path}")
    print(f"  {exclusion_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()

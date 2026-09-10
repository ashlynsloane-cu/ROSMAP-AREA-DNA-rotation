#!/usr/bin/env python3
"""
diagnose_monocyte_dlpfc_pathology_merge.py
==========================================

Diagnose why continuous amyloid and tangle phenotypes did not merge into the
matched ROSMAP monocyte ↔ DLPFC feasibility table, identify the exact pathology
columns available in the current project metadata, and rebuild the matched
table with the best-supported continuous pathology variables.

Primary goals
-------------
1. Inspect the pathology metadata and report candidate amyloid/tangle columns.
2. Quantify how many of the 168 matched participants can be linked to each
   candidate variable.
3. Prefer exact continuous pathology variables over ordinal proxies.
4. Rebuild the matched specimen- and participant-level tables with the selected
   continuous pathology columns.
5. Preserve the original CERAD and Braak columns.
6. Write a clear audit CSV showing which variables were considered and selected.

Expected inputs
---------------
results/monocyte_dlpfc_feasibility/
    monocyte_dlpfc_matched_specimen_table.csv
    monocyte_dlpfc_matched_participant_table.csv

Current project pathology metadata, typically:
    results/pathology_geometry_abeta_tau/metadata/
        ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv

Optional additional metadata searched automatically:
    data/ROSMAP_metadata_merged_AREA.csv
    results/preprocessing/ROSMAP_RNAseq_master_metadata_locked_covariates.csv
    data/ROSMAP_clinical.csv

Outputs
-------
results/monocyte_dlpfc_feasibility/pathology_diagnostic/

    pathology_candidate_columns.csv
    pathology_candidate_overlap_summary.csv
    pathology_selected_variables.csv
    pathology_merge_audit.csv

    monocyte_dlpfc_matched_specimen_table_with_continuous_pathology.csv
    monocyte_dlpfc_matched_participant_table_with_continuous_pathology.csv
    monocyte_dlpfc_continuous_pathology_missingness.csv

Notes
-----
- The script does NOT guess that any column is continuous just because its name
  contains "amyloid" or "tangle". It scores candidates using:
      * column name
      * numeric type / coercibility
      * number of unique numeric values
      * matched-participant coverage
  and prints the selected variable.
- Ordinal variables such as CERAD and Braak remain in the table but are not used
  as substitutes for the continuous fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Iterable
import re

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_FEASIBILITY_DIR = Path(
    "results/monocyte_dlpfc_feasibility"
)

DEFAULT_PATHOLOGY_CANDIDATES = [
    Path(
        "results/pathology_geometry_abeta_tau/metadata/"
        "ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv"
    ),
    Path("data/ROSMAP_metadata_merged_AREA.csv"),
]

DEFAULT_LOCKED_BRAIN = Path(
    "results/preprocessing/"
    "ROSMAP_RNAseq_master_metadata_locked_covariates.csv"
)

DEFAULT_CLINICAL = Path(
    "data/ROSMAP_clinical.csv"
)

# Terms that suggest continuous amyloid/tau pathology.
AMYLOID_TERMS = [
    "amyloid",
    "abeta",
    "a_beta",
    "plaque",
]

TAU_TERMS = [
    "tangle",
    "tau",
    "nft",
    "neurofibrillary",
]

# Terms that make a candidate less likely to be the desired quantitative field.
ORDINAL_PENALTY_TERMS = [
    "cerad",
    "braak",
    "stage",
    "category",
    "class",
    "binary",
    "dx",
    "diagnosis",
    "equal",
    "calibrated",
    "weight",
]


# =============================================================================
# ARGUMENTS / DISCOVERY
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--specimen-table",
        default=str(
            DEFAULT_FEASIBILITY_DIR
            / "monocyte_dlpfc_matched_specimen_table.csv"
        ),
    )

    p.add_argument(
        "--participant-table",
        default=str(
            DEFAULT_FEASIBILITY_DIR
            / "monocyte_dlpfc_matched_participant_table.csv"
        ),
    )

    p.add_argument(
        "--pathology-metadata",
        default=None,
        help=(
            "Optional explicit pathology metadata file. "
            "If omitted, common project locations are searched."
        ),
    )

    p.add_argument(
        "--locked-brain-metadata",
        default=str(DEFAULT_LOCKED_BRAIN),
    )

    p.add_argument(
        "--clinical",
        default=str(DEFAULT_CLINICAL),
    )

    p.add_argument(
        "--outdir",
        default=str(
            DEFAULT_FEASIBILITY_DIR
            / "pathology_diagnostic"
        ),
    )

    return p.parse_args()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        p = Path(p).expanduser()
        if p.exists():
            return p
    return None


def find_pathology(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"Explicit pathology metadata not found: {p}"
            )
        return p

    found = first_existing(
        DEFAULT_PATHOLOGY_CANDIDATES
    )

    if found is not None:
        return found

    # Last resort recursive search.
    root = Path("results")
    if root.exists():
        hits = sorted(
            root.rglob("*abeta*tau*.csv")
        )
        if hits:
            return hits[0]

    raise FileNotFoundError(
        "Could not locate pathology metadata.\n"
        "Pass --pathology-metadata <path>."
    )


# =============================================================================
# HELPERS
# =============================================================================

def resolve(
    df: pd.DataFrame,
    candidates,
    required=False,
):
    lower = {
        c.lower(): c
        for c in df.columns
    }

    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]

    if required:
        raise ValueError(
            f"Could not resolve required column from {candidates}.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def numeric_profile(series: pd.Series):
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    n_nonmissing = int(
        numeric.notna().sum()
    )

    n_unique = int(
        numeric.dropna().nunique()
    )

    return numeric, n_nonmissing, n_unique


def looks_like_candidate(
    name: str,
    terms,
):
    low = name.lower()
    return any(
        term in low
        for term in terms
    )


def candidate_score(
    column: str,
    numeric_unique: int,
    matched_coverage: int,
    matched_total: int,
):
    low = column.lower()

    score = 0.0

    # Prefer broad quantitative structure.
    if numeric_unique >= 50:
        score += 5.0
    elif numeric_unique >= 20:
        score += 4.0
    elif numeric_unique >= 10:
        score += 3.0
    elif numeric_unique >= 5:
        score += 1.5
    elif numeric_unique >= 3:
        score += 0.5

    # Prefer good matched-participant coverage.
    if matched_total > 0:
        score += 4.0 * (
            matched_coverage
            / matched_total
        )

    # Name cues for quantitative measures.
    positive_terms = [
        "continuous",
        "burden",
        "density",
        "mean",
        "average",
        "quant",
        "measure",
        "load",
        "percent",
        "pct",
        "level",
    ]

    for term in positive_terms:
        if term in low:
            score += 0.5

    # Penalize obvious ordinal / encoding columns.
    for term in ORDINAL_PENALTY_TERMS:
        if term in low:
            score -= 2.0

    return score


# =============================================================================
# LINK PATHOLOGY METADATA TO MATCHED PARTICIPANTS
# =============================================================================

def build_individual_bridge(
    pathology: pd.DataFrame,
    matched_participants: pd.DataFrame,
    locked_path: Optional[Path],
):
    """
    Return pathology metadata with a standardized `_individualID` column.

    Match priority:
      1. pathology individualID
      2. pathology projid -> matched projid
      3. pathology sample_id/specimenID -> locked brain metadata -> individualID
    """
    out = pathology.copy()

    individual_col = resolve(
        out,
        ["individualID", "individual_id"],
    )

    if individual_col is not None:
        out["_individualID"] = (
            out[individual_col]
            .astype("string")
        )
        return out, f"direct:{individual_col}"

    proj_col = resolve(
        out,
        ["projid"],
    )

    if (
        proj_col is not None
        and "projid" in matched_participants.columns
    ):
        bridge = (
            matched_participants[
                ["individualID", "projid"]
            ]
            .dropna()
            .drop_duplicates()
        )

        out = out.merge(
            bridge,
            left_on=proj_col,
            right_on="projid",
            how="left",
        )

        out["_individualID"] = (
            out["individualID"]
            .astype("string")
        )

        return out, f"projid:{proj_col}"

    sample_col = resolve(
        out,
        [
            "sample_id",
            "specimenID",
            "specimen_id",
        ],
    )

    if (
        sample_col is not None
        and locked_path is not None
        and locked_path.exists()
    ):
        locked = pd.read_csv(
            locked_path,
            low_memory=False,
        )

        locked_sample = resolve(
            locked,
            [
                "sample_id",
                "specimenID",
                "specimen_id",
            ],
            required=True,
        )

        locked_individual = resolve(
            locked,
            [
                "individualID",
                "individual_id",
            ],
        )

        if locked_individual is not None:
            bridge = (
                locked[
                    [
                        locked_sample,
                        locked_individual,
                    ]
                ]
                .dropna()
                .drop_duplicates()
                .rename(
                    columns={
                        locked_sample: "_sample_bridge",
                        locked_individual: "_individualID",
                    }
                )
            )

            out = out.merge(
                bridge,
                left_on=sample_col,
                right_on="_sample_bridge",
                how="left",
            )

            return out, (
                f"sample:{sample_col}->"
                f"{locked_sample}->"
                f"{locked_individual}"
            )

        # If locked metadata only has projid, bridge via matched participant table.
        locked_proj = resolve(
            locked,
            ["projid"],
        )

        if (
            locked_proj is not None
            and "projid" in matched_participants.columns
        ):
            pbridge = (
                matched_participants[
                    ["individualID", "projid"]
                ]
                .dropna()
                .drop_duplicates()
            )

            bridge = (
                locked[
                    [
                        locked_sample,
                        locked_proj,
                    ]
                ]
                .drop_duplicates()
                .merge(
                    pbridge,
                    left_on=locked_proj,
                    right_on="projid",
                    how="left",
                )
                [[locked_sample, "individualID"]]
                .rename(
                    columns={
                        locked_sample: "_sample_bridge",
                        "individualID": "_individualID",
                    }
                )
            )

            out = out.merge(
                bridge,
                left_on=sample_col,
                right_on="_sample_bridge",
                how="left",
            )

            return out, (
                f"sample:{sample_col}->"
                f"{locked_sample}->projid"
            )

    raise ValueError(
        "Could not link pathology metadata to matched participants. "
        "Need individualID, projid, or sample_id/specimenID plus a usable bridge."
    )


# =============================================================================
# CANDIDATE AUDIT
# =============================================================================

def audit_candidates(
    pathology_linked: pd.DataFrame,
    matched_ids: set,
):
    rows = []

    matched_total = len(
        matched_ids
    )

    for col in pathology_linked.columns:
        if col.startswith("_"):
            continue

        is_amyloid = looks_like_candidate(
            col,
            AMYLOID_TERMS,
        )

        is_tau = looks_like_candidate(
            col,
            TAU_TERMS,
        )

        if not (
            is_amyloid
            or is_tau
        ):
            continue

        numeric, n_numeric, n_unique = numeric_profile(
            pathology_linked[col]
        )

        tmp = pd.DataFrame(
            {
                "_individualID": pathology_linked[
                    "_individualID"
                ],
                "_value": numeric,
            }
        )

        tmp = tmp[
            tmp["_individualID"].isin(
                matched_ids
            )
        ]

        matched_nonmissing = int(
            tmp.loc[
                tmp["_value"].notna(),
                "_individualID",
            ]
            .nunique()
        )

        axis = (
            "amyloid"
            if is_amyloid and not is_tau
            else "tau"
            if is_tau and not is_amyloid
            else "amyloid_and_tau_name"
        )

        rows.append(
            {
                "column": col,
                "axis_name_match": axis,
                "dtype": str(
                    pathology_linked[col].dtype
                ),
                "n_numeric_all_rows": n_numeric,
                "n_unique_numeric_all_rows": n_unique,
                "matched_participants_nonmissing": matched_nonmissing,
                "matched_participants_total": matched_total,
                "matched_percent_nonmissing": (
                    100.0
                    * matched_nonmissing
                    / matched_total
                    if matched_total
                    else np.nan
                ),
                "candidate_score": candidate_score(
                    col,
                    n_unique,
                    matched_nonmissing,
                    matched_total,
                ),
                "ordinal_name_penalty": any(
                    term in col.lower()
                    for term in ORDINAL_PENALTY_TERMS
                ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    if len(audit):
        audit = audit.sort_values(
            [
                "axis_name_match",
                "candidate_score",
                "matched_participants_nonmissing",
                "n_unique_numeric_all_rows",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )

    return audit


def choose_best_candidate(
    audit: pd.DataFrame,
    axis: str,
):
    subset = audit[
        audit["axis_name_match"].isin(
            [
                axis,
                "amyloid_and_tau_name",
            ]
        )
    ].copy()

    # Exclude obviously ordinal candidates where possible.
    nonordinal = subset[
        ~subset["ordinal_name_penalty"]
    ]

    if len(nonordinal):
        subset = nonordinal

    # Require at least some quantitative structure.
    quantitative = subset[
        subset[
            "n_unique_numeric_all_rows"
        ] >= 5
    ]

    if len(quantitative):
        subset = quantitative

    if len(subset) == 0:
        return None

    return subset.sort_values(
        [
            "candidate_score",
            "matched_participants_nonmissing",
            "n_unique_numeric_all_rows",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).iloc[0]


# =============================================================================
# COLLAPSE / MERGE
# =============================================================================

def collapse_by_person(
    linked: pd.DataFrame,
    column: str,
):
    numeric = pd.to_numeric(
        linked[column],
        errors="coerce",
    )

    tmp = pd.DataFrame(
        {
            "individualID": linked[
                "_individualID"
            ],
            column: numeric,
        }
    )

    tmp = tmp.dropna(
        subset=[
            "individualID",
            column,
        ]
    )

    # Usually one brain RNA-seq row/person. If duplicates exist, use the mean
    # and flag this in the audit.
    out = (
        tmp.groupby(
            "individualID",
            as_index=False,
        )
        .agg(
            **{
                column: (
                    column,
                    "mean",
                ),
                f"{column}__n_source_rows": (
                    column,
                    "size",
                ),
            }
        )
    )

    return out


def merge_selected(
    specimen: pd.DataFrame,
    participant: pd.DataFrame,
    linked: pd.DataFrame,
    selected,
):
    merge_audit = []

    for axis, selected_row in selected.items():
        if selected_row is None:
            merge_audit.append(
                {
                    "axis": axis,
                    "selected_column": None,
                    "status": "no_candidate_selected",
                    "n_matched_after_merge": 0,
                }
            )
            continue

        col = selected_row["column"]

        collapsed = collapse_by_person(
            linked,
            col,
        )

        standard_name = (
            "amyloid_continuous"
            if axis == "amyloid"
            else "tangles_continuous"
        )

        collapsed = collapsed.rename(
            columns={
                col: standard_name,
                f"{col}__n_source_rows": (
                    f"{standard_name}__n_source_rows"
                ),
            }
        )

        # Remove existing empty/broken standardized column before re-merge.
        for df in [specimen, participant]:
            if standard_name in df.columns:
                df.drop(
                    columns=[
                        standard_name
                    ],
                    inplace=True,
                )

        specimen = specimen.merge(
            collapsed,
            on="individualID",
            how="left",
        )

        participant = participant.merge(
            collapsed,
            on="individualID",
            how="left",
        )

        merge_audit.append(
            {
                "axis": axis,
                "selected_column": col,
                "status": "merged",
                "n_matched_after_merge": int(
                    participant[
                        standard_name
                    ]
                    .notna()
                    .sum()
                ),
            }
        )

    return (
        specimen,
        participant,
        pd.DataFrame(
            merge_audit
        ),
    )


# =============================================================================
# MISSINGNESS
# =============================================================================

def make_missingness(
    participant: pd.DataFrame,
):
    variables = [
        "global_cognition",
        "mmse",
        "cerad",
        "braak",
        "amyloid_continuous",
        "tangles_continuous",
        "age_death_exact",
        "years_monocyte_to_death_exact",
    ]

    rows = []

    n = len(
        participant
    )

    for var in variables:
        present = (
            var in participant.columns
        )

        nn = (
            int(
                participant[
                    var
                ]
                .notna()
                .sum()
            )
            if present
            else 0
        )

        rows.append(
            {
                "variable": var,
                "column_present": present,
                "n_nonmissing": nn,
                "n_missing": n - nn,
                "percent_nonmissing": (
                    100.0 * nn / n
                    if n
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print("=" * 100)
    print("ROSMAP MONOCYTE ↔ DLPFC CONTINUOUS PATHOLOGY DIAGNOSTIC")
    print("=" * 100)
    print()

    specimen_path = Path(
        args.specimen_table
    )

    participant_path = Path(
        args.participant_table
    )

    pathology_path = find_pathology(
        args.pathology_metadata
    )

    locked_path = Path(
        args.locked_brain_metadata
    )

    if not locked_path.exists():
        locked_path = None

    outdir = Path(
        args.outdir
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not specimen_path.exists():
        raise FileNotFoundError(
            f"Missing specimen table: {specimen_path}"
        )

    if not participant_path.exists():
        raise FileNotFoundError(
            f"Missing participant table: {participant_path}"
        )

    print(
        f"Matched specimen table:   {specimen_path}"
    )
    print(
        f"Matched participant table:{participant_path}"
    )
    print(
        f"Pathology metadata:       {pathology_path}"
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

    pathology = pd.read_csv(
        pathology_path,
        low_memory=False,
    )

    linked, bridge_method = build_individual_bridge(
        pathology,
        participant,
        locked_path,
    )

    print(
        f"Pathology → participant bridge: {bridge_method}"
    )
    print()

    matched_ids = set(
        participant[
            "individualID"
        ]
        .dropna()
        .astype(str)
    )

    linked[
        "_individualID"
    ] = linked[
        "_individualID"
    ].astype(str)

    audit = audit_candidates(
        linked,
        matched_ids,
    )

    audit_path = (
        outdir
        / "pathology_candidate_columns.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    if len(audit) == 0:
        raise ValueError(
            "No amyloid/tangle-like columns found "
            "in pathology metadata."
        )

    amyloid = choose_best_candidate(
        audit,
        "amyloid",
    )

    tau = choose_best_candidate(
        audit,
        "tau",
    )

    selected_rows = []

    for axis, row in [
        ("amyloid", amyloid),
        ("tau", tau),
    ]:
        if row is None:
            selected_rows.append(
                {
                    "axis": axis,
                    "selected_column": None,
                    "candidate_score": np.nan,
                    "matched_participants_nonmissing": 0,
                    "n_unique_numeric_all_rows": 0,
                }
            )
        else:
            selected_rows.append(
                {
                    "axis": axis,
                    "selected_column": row[
                        "column"
                    ],
                    "candidate_score": row[
                        "candidate_score"
                    ],
                    "matched_participants_nonmissing": row[
                        "matched_participants_nonmissing"
                    ],
                    "n_unique_numeric_all_rows": row[
                        "n_unique_numeric_all_rows"
                    ],
                }
            )

    selected_df = pd.DataFrame(
        selected_rows
    )

    selected_df.to_csv(
        outdir
        / "pathology_selected_variables.csv",
        index=False,
    )

    print("Selected variables:")
    print(
        selected_df.to_string(
            index=False
        )
    )
    print()

    specimen2, participant2, merge_audit = merge_selected(
        specimen.copy(),
        participant.copy(),
        linked,
        {
            "amyloid": amyloid,
            "tau": tau,
        },
    )

    merge_audit.to_csv(
        outdir
        / "pathology_merge_audit.csv",
        index=False,
    )

    specimen2.to_csv(
        outdir
        / (
            "monocyte_dlpfc_matched_specimen_table_"
            "with_continuous_pathology.csv"
        ),
        index=False,
    )

    participant2.to_csv(
        outdir
        / (
            "monocyte_dlpfc_matched_participant_table_"
            "with_continuous_pathology.csv"
        ),
        index=False,
    )

    missing = make_missingness(
        participant2
    )

    missing.to_csv(
        outdir
        / "monocyte_dlpfc_continuous_pathology_missingness.csv",
        index=False,
    )

    # Additional overlap summary for the top candidates.
    top_overlap = audit[
        [
            "column",
            "axis_name_match",
            "n_unique_numeric_all_rows",
            "matched_participants_nonmissing",
            "matched_percent_nonmissing",
            "candidate_score",
            "ordinal_name_penalty",
        ]
    ].copy()

    top_overlap.to_csv(
        outdir
        / "pathology_candidate_overlap_summary.csv",
        index=False,
    )

    print("=" * 100)
    print("MERGE RESULT")
    print("=" * 100)
    print()
    print(
        merge_audit.to_string(
            index=False
        )
    )
    print()
    print(
        missing.to_string(
            index=False
        )
    )
    print()
    print("Outputs:")
    print(
        f"  {audit_path}"
    )
    print(
        f"  {outdir / 'pathology_selected_variables.csv'}"
    )
    print(
        f"  {outdir / 'pathology_merge_audit.csv'}"
    )
    print(
        f"  {outdir / 'monocyte_dlpfc_matched_participant_table_with_continuous_pathology.csv'}"
    )
    print(
        f"  {outdir / 'monocyte_dlpfc_matched_specimen_table_with_continuous_pathology.csv'}"
    )


if __name__ == "__main__":
    main()

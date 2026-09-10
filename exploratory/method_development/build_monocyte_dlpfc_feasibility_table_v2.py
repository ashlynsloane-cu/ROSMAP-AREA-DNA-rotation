#!/usr/bin/env python3
"""
build_monocyte_dlpfc_feasibility_table_v2.py
=========================================

Build a same-participant ROSMAP ante-mortem monocyte RNA-seq ↔ postmortem
DLPFC bulk RNA-seq feasibility table.

Primary outputs
---------------
results/monocyte_dlpfc_feasibility/
    monocyte_dlpfc_matched_specimen_table.csv
    monocyte_dlpfc_matched_participant_table.csv
    monocyte_dlpfc_feasibility_summary.csv
    monocyte_dlpfc_missingness_summary.csv

What the specimen-level table contains
--------------------------------------
- individualID / projid
- monocyte specimen ID
- monocyte sampling age (raw + numeric where exact)
- DLPFC specimen ID(s)
- age at death
- blood-to-death interval where calculable
- cognitive diagnosis
- global cognition, if available
- MMSE
- CERAD
- Braak
- continuous amyloid, if available
- continuous tangles, if available

Important:
- Exact project `age_death` is used for blood-to-death interval calculations.
- `age_numeric` is deliberately not used.
- If only a censored age such as `90+` is available, it is preserved as
  censored and not treated as an exact age.
- By default, the script uses the LOCKED DLPFC analysis cohort if it can find
  results/preprocessing/ROSMAP_RNAseq_master_metadata_locked_covariates.csv.
  If that file is absent, it falls back to all DLPFC bulk RNA-seq specimens
  in the biospecimen metadata and prints a warning.
- The script automatically searches common repo locations for required files.

Run from the ROSMAP-AREA-DNA-rotation repo root:
    python3 exploratory/method_development/build_monocyte_dlpfc_feasibility_table_v2.py

Optional explicit paths:
    python3 exploratory/method_development/build_monocyte_dlpfc_feasibility_table_v2.py \
      --biospecimen data/ROSMAP_biospecimen_metadata.csv \
      --clinical data/ROSMAP_clinical.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Iterable
import re

import numpy as np
import pandas as pd


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def first_existing(candidates: Iterable[str]) -> Optional[Path]:
    for x in candidates:
        p = Path(x).expanduser()
        if p.exists():
            return p
    return None


def find_by_name(names: Iterable[str], roots=("data", "results", ".")) -> Optional[Path]:
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for name in names:
            hits = list(rp.rglob(name))
            if hits:
                return sorted(hits)[0]
    return None


def discover_inputs(args):
    biospec = Path(args.biospecimen).expanduser() if args.biospecimen else first_existing([
        "data/ROSMAP_biospecimen_metadata.csv",
        "ROSMAP_biospecimen_metadata.csv",
        "~/Downloads/ROSMAP_biospecimen_metadata.csv",
    ])
    if biospec is None:
        biospec = find_by_name(["ROSMAP_biospecimen_metadata.csv"])

    clinical = Path(args.clinical).expanduser() if args.clinical else first_existing([
        "data/ROSMAP_clinical.csv",
        "data/ROSMAP_clinical(1).csv",
        "ROSMAP_clinical.csv",
        "ROSMAP_clinical(1).csv",
        "~/Downloads/ROSMAP_clinical.csv",
        "~/Downloads/ROSMAP_clinical(1).csv",
    ])
    if clinical is None:
        clinical = find_by_name(["ROSMAP_clinical.csv", "ROSMAP_clinical(1).csv"])

    locked = Path(args.locked_brain_metadata) if args.locked_brain_metadata else first_existing([
        "results/preprocessing/ROSMAP_RNAseq_master_metadata_locked_covariates.csv",
        "results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv",
    ])

    pathology = Path(args.pathology_metadata) if args.pathology_metadata else first_existing([
        "results/pathology_geometry_abeta_tau/metadata/ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv",
        "data/ROSMAP_metadata_merged_AREA.csv",
    ])

    assay = Path(args.assay_metadata) if args.assay_metadata else first_existing([
        "data/ROSMAP_assay_rnaSeq_metadata.csv",
        "ROSMAP_assay_rnaSeq_metadata.csv",
    ])
    if assay is None:
        assay = find_by_name(["ROSMAP_assay_rnaSeq_metadata.csv"])

    if biospec is None or not biospec.exists():
        raise FileNotFoundError(
            "Could not find ROSMAP_biospecimen_metadata.csv.\n"
            "Searched the repo and ~/Downloads.\n"
            "Either move it to data/ or pass --biospecimen <path>."
        )

    if clinical is None or not clinical.exists():
        raise FileNotFoundError(
            "Could not find ROSMAP_clinical.csv.\n"
            "Place it in data/ or pass --clinical <path>."
        )

    return biospec, clinical, locked, pathology, assay


# =============================================================================
# COLUMN HELPERS
# =============================================================================

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


def exact_numeric_age(series: pd.Series) -> pd.Series:
    """
    Convert genuinely numeric age values to float.
    Censored values such as '90+' remain NaN.
    """
    s = series.astype("string").str.strip()
    exact = s.str.fullmatch(r"\d+(\.\d+)?", na=False)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[exact] = pd.to_numeric(s.loc[exact], errors="coerce")
    return out


def lower_bound_age(series: pd.Series) -> pd.Series:
    """
    Numeric lower bound for descriptive purposes.
    E.g. 90+ -> 90. This is NEVER used as an exact death interval.
    """
    s = series.astype("string").str.strip()
    extracted = s.str.extract(r"^\s*(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def boolish_false(series):
    return (
        series.isna()
        | series.astype(str).str.strip().str.lower().isin(
            ["false", "0", "no", "nan", "none", ""]
        )
    )


def semicolon_unique(values):
    vals = [str(x) for x in values if pd.notna(x) and str(x).strip() not in ("", "nan")]
    return ";".join(sorted(set(vals)))


# =============================================================================
# SOURCE STANDARDIZATION
# =============================================================================

def get_monocytes(bio: pd.DataFrame) -> pd.DataFrame:
    organ = bio["organ"].astype(str).str.lower()
    cell = bio["cellType"].astype(str).str.lower()
    assay = bio["assay"].astype(str).str.lower()

    keep = (
        organ.eq("blood")
        & cell.str.contains("monocyte", na=False)
        & assay.eq("rnaseq")
    )

    mono = bio.loc[keep].copy()

    if "exclude" in mono.columns:
        # Keep an audit column, but exclude only rows explicitly marked True.
        excluded = mono["exclude"].astype(str).str.strip().str.lower().eq("true")
        mono = mono.loc[~excluded].copy()

    mono["monocyte_sampling_age_raw"] = mono["samplingAge"]
    mono["monocyte_sampling_age_exact"] = exact_numeric_age(mono["samplingAge"])
    mono["monocyte_sampling_age_lower_bound"] = lower_bound_age(mono["samplingAge"])

    keep_cols = [
        "individualID",
        "specimenID",
        "monocyte_sampling_age_raw",
        "monocyte_sampling_age_exact",
        "monocyte_sampling_age_lower_bound",
    ]

    for c in ["visitNumber", "samplingDate"]:
        if c in mono.columns:
            keep_cols.append(c)

    mono = mono[keep_cols].rename(
        columns={
            "specimenID": "monocyte_specimen_id",
            "visitNumber": "monocyte_visit_number",
            "samplingDate": "monocyte_sampling_date",
        }
    )

    # Repeated monocyte indicator.
    counts = mono["individualID"].value_counts()
    mono["n_monocyte_rnaseq_samples_for_person"] = mono["individualID"].map(counts)
    mono["repeated_monocyte_rnaseq"] = (
        mono["n_monocyte_rnaseq_samples_for_person"] > 1
    )

    return mono


def get_brain_specimens(
    bio: pd.DataFrame,
    locked_path: Optional[Path],
):
    organ = bio["organ"].astype(str).str.lower()
    tissue = bio["tissue"].astype(str).str.lower()
    assay = bio["assay"].astype(str).str.lower()

    dlpfc = bio.loc[
        organ.eq("brain")
        & tissue.str.contains("dorsolateral prefrontal", na=False)
        & assay.eq("rnaseq")
    ].copy()

    if "exclude" in dlpfc.columns:
        excluded = dlpfc["exclude"].astype(str).str.strip().str.lower().eq("true")
        dlpfc = dlpfc.loc[~excluded].copy()

    source = "all_DLPFC_RNAseq_in_biospecimen_metadata"

    if locked_path is not None and locked_path.exists():
        locked = pd.read_csv(locked_path, low_memory=False)
        sid = resolve(
            locked,
            ["sample_id", "specimenID", "specimen_id"],
            required=True,
        )
        locked_ids = set(locked[sid].astype(str))
        dlpfc = dlpfc.loc[
            dlpfc["specimenID"].astype(str).isin(locked_ids)
        ].copy()
        source = f"locked_DLPFC_cohort:{locked_path}"

    out = (
        dlpfc.groupby("individualID", as_index=False)
        .agg(
            dlpfc_specimen_ids=("specimenID", semicolon_unique),
            n_dlpfc_rnaseq_specimens=("specimenID", "nunique"),
        )
    )

    return out, source


def clinical_standardized(clin: pd.DataFrame) -> pd.DataFrame:
    individual = resolve(clin, ["individualID"], required=True)

    mapping = {
        "projid": ["projid"],
        "age_death_raw": ["age_death"],
        "mmse_last_valid": [
            "cts_mmse30_lv",
            "mmse_lv",
            "mmse",
            "mmse_impairment",
        ],
        "braak_stage": ["braaksc", "braak_stage", "Braak_stage"],
        "cerad_score": ["ceradsc", "cerad", "CERAD_burden"],
        "cognitive_diagnosis_at_death": ["cogdx"],
        "cognitive_diagnosis_last_valid": ["dcfdx_lv", "dcfdx"],
        "sex": ["msex", "sex"],
        "pmi": ["pmi", "pmi_numeric"],
    }

    out = pd.DataFrame({"individualID": clin[individual]})

    for new, candidates in mapping.items():
        col = resolve(clin, candidates)
        if col is not None:
            out[new] = clin[col]

    if "age_death_raw" in out.columns:
        out["age_death_exact_from_clinical"] = exact_numeric_age(out["age_death_raw"])
        out["age_death_lower_bound"] = lower_bound_age(out["age_death_raw"])

    return out.drop_duplicates("individualID")


def add_optional_phenotypes(
    table: pd.DataFrame,
    source_path: Optional[Path],
    bio: pd.DataFrame,
    source_label: str,
):
    """
    Merge optional phenotype values from project metadata.
    Supports matching by individualID, projid, or brain specimen/sample_id.
    """
    if source_path is None or not source_path.exists():
        print(f"[optional] {source_label}: not found; skipping.")
        return table

    src = pd.read_csv(source_path, low_memory=False)
    print(f"[optional] {source_label}: {source_path}")

    # Build join key.
    if "individualID" in src.columns:
        src["_join_individualID"] = src["individualID"].astype(str)
    else:
        sample_col = resolve(src, ["sample_id", "specimenID", "specimen_id"])
        if sample_col is not None:
            bridge = bio[["specimenID", "individualID"]].drop_duplicates()
            src = src.merge(
                bridge,
                left_on=sample_col,
                right_on="specimenID",
                how="left",
            )
            src["_join_individualID"] = src["individualID"].astype("string")
        else:
            proj_col = resolve(src, ["projid"])
            if proj_col is not None and "projid" in table.columns:
                # Merge directly by projid for this source.
                key = "projid"
                phenos = extract_optional_columns(src)
                if not phenos:
                    return table
                use = src[[proj_col] + list(phenos.values())].copy()
                use = use.rename(columns={proj_col: key})
                use = collapse_source(use, key)
                ren = {v: k for k, v in phenos.items()}
                use = use.rename(columns=ren)
                return merge_fill(table, use, key)
            else:
                print(
                    f"[optional] {source_label}: could not resolve individualID, "
                    "sample_id/specimenID, or projid; skipping."
                )
                return table

    phenos = extract_optional_columns(src)
    if not phenos:
        return table

    cols = ["_join_individualID"] + list(dict.fromkeys(phenos.values()))
    use = src[cols].copy()
    use = collapse_source(use, "_join_individualID")
    use = use.rename(columns={"_join_individualID": "individualID"})
    use = use.rename(columns={v: k for k, v in phenos.items()})

    return merge_fill(table, use, "individualID")


def extract_optional_columns(src: pd.DataFrame):
    specs = {
        "global_cognition": [
            "cogn_global",
            "cogn_global_lv",
            "global_cognition",
            "cogn_global_impairment",
        ],
        "mmse": [
            "cts_mmse30_lv",
            "mmse",
            "mmse_impairment",
        ],
        "cerad": [
            "CERAD_burden",
            "ceradsc",
            "cerad",
            "CERAD_equal",
            "CERAD_amyloid_calibrated",
        ],
        "braak": [
            "Braak_stage",
            "braaksc",
            "braak_stage",
            "Braak_equal",
            "Braak_tangle_calibrated",
        ],
        "amyloid_continuous": [
            "amyloid_continuous",
            "amyloid",
            "amyloid_burden",
            "amyloid_scaled",
        ],
        "tangles_continuous": [
            "tangle_continuous",
            "tangles_continuous",
            "tangles",
            "tangle_burden",
            "tangles_scaled",
        ],
        "age_death_project": [
            "age_death",
        ],
    }

    found = {}
    for standardized, candidates in specs.items():
        col = resolve(src, candidates)
        if col is not None:
            found[standardized] = col
    return found


def collapse_source(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    One row per key. For each variable use the first non-null value.
    """
    def first_nonnull(s):
        x = s.dropna()
        return x.iloc[0] if len(x) else np.nan

    agg = {
        c: first_nonnull
        for c in df.columns
        if c != key
    }
    return df.groupby(key, as_index=False, dropna=False).agg(agg)


def merge_fill(left, right, key):
    """
    Merge without destroying values already present.
    If a duplicate standardized column exists, fill left NA from right.
    """
    overlapping = [
        c for c in right.columns
        if c in left.columns and c != key
    ]

    merged = left.merge(
        right,
        on=key,
        how="left",
        suffixes=("", "__new"),
    )

    for c in overlapping:
        nc = c + "__new"
        merged[c] = merged[c].where(
            merged[c].notna(),
            merged[nc],
        )
        merged = merged.drop(columns=[nc])

    return merged


# =============================================================================
# INTERVAL CALCULATION
# =============================================================================

def choose_best_age_death(table: pd.DataFrame) -> pd.DataFrame:
    """
    Choose the exact age at death used for blood-to-death interval calculations.

    Priority:
      1. project metadata `age_death`
      2. exact numeric clinical `age_death`

    IMPORTANT:
    - `age_numeric` is deliberately NOT used.
    - censored strings such as `90+` are never converted to an exact death age.
    """
    project_raw = table.get(
        "age_death_project",
        pd.Series(np.nan, index=table.index),
    )

    # Project `age_death` is expected to be the exact numeric death age in the
    # ROSMAP RNA-seq metadata used in this project.
    project_exact = pd.to_numeric(
        project_raw,
        errors="coerce",
    )

    clinical_exact = pd.to_numeric(
        table.get(
            "age_death_exact_from_clinical",
            pd.Series(np.nan, index=table.index),
        ),
        errors="coerce",
    )

    table["age_death_exact"] = project_exact.where(
        project_exact.notna(),
        clinical_exact,
    )

    table["age_death_source"] = np.where(
        project_exact.notna(),
        "project_age_death",
        np.where(
            clinical_exact.notna(),
            "clinical_age_death_exact",
            "censored_or_missing",
        ),
    )

    table["years_monocyte_to_death_exact"] = (
        table["age_death_exact"]
        - table["monocyte_sampling_age_exact"]
    )

    bad = (
        table["years_monocyte_to_death_exact"].notna()
        & (table["years_monocyte_to_death_exact"] < 0)
    )

    table["interval_metadata_flag"] = np.where(
        bad,
        "negative_interval_check_metadata",
        "",
    )

    table.loc[
        bad,
        "years_monocyte_to_death_exact",
    ] = np.nan

    return table


# =============================================================================
# OUTPUT SUMMARIES
# =============================================================================

def participant_table(specimen_table: pd.DataFrame) -> pd.DataFrame:
    # Keep one row/person while summarizing repeated monocyte samples.
    agg = {
        "projid": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "monocyte_specimen_id": semicolon_unique,
        "n_monocyte_rnaseq_samples_for_person": "max",
        "repeated_monocyte_rnaseq": "max",
        "dlpfc_specimen_ids": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "n_dlpfc_rnaseq_specimens": "max",
        "age_death_raw": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "age_death_exact": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "cognitive_diagnosis_at_death": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "cognitive_diagnosis_last_valid": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "global_cognition": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "mmse": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "cerad": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "braak": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "amyloid_continuous": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "tangles_continuous": lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        "years_monocyte_to_death_exact": lambda x: (
            ";".join(f"{v:.3f}" for v in sorted(x.dropna().unique()))
            if x.notna().any()
            else np.nan
        ),
    }

    # Only aggregate columns that actually exist.
    agg = {k: v for k, v in agg.items() if k in specimen_table.columns}

    out = (
        specimen_table.groupby("individualID", as_index=False)
        .agg(agg)
    )

    return out


def make_summary(
    mono_all: pd.DataFrame,
    brain: pd.DataFrame,
    matched: pd.DataFrame,
    brain_source: str,
):
    rows = []

    def add(metric, value):
        rows.append({"metric": metric, "value": value})

    add("monocyte_rnaseq_specimens_total", len(mono_all))
    add("monocyte_rnaseq_participants_total", mono_all["individualID"].nunique())
    add(
        "monocyte_participants_with_repeated_rnaseq",
        int(
            mono_all.loc[
                mono_all["repeated_monocyte_rnaseq"],
                "individualID",
            ].nunique()
        ),
    )
    add("DLPFC_participants_in_selected_brain_cohort", brain["individualID"].nunique())
    add("matched_monocyte_DLPFC_participants", matched["individualID"].nunique())
    add("matched_monocyte_specimens", len(matched))
    add(
        "matched_participants_with_repeated_monocyte_rnaseq",
        int(
            matched.loc[
                matched["repeated_monocyte_rnaseq"],
                "individualID",
            ].nunique()
        ),
    )
    add("brain_cohort_source", brain_source)

    if "years_monocyte_to_death_exact" in matched.columns:
        vals = matched["years_monocyte_to_death_exact"].dropna()
        add("matched_monocyte_samples_with_exact_death_interval", len(vals))
        if len(vals):
            add("years_to_death_median", float(vals.median()))
            add("years_to_death_mean", float(vals.mean()))
            add("years_to_death_min", float(vals.min()))
            add("years_to_death_max", float(vals.max()))

    return pd.DataFrame(rows)


def missingness_summary(df: pd.DataFrame):
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
    n = len(df)

    for c in variables:
        if c not in df.columns:
            rows.append({
                "variable": c,
                "column_present": False,
                "n_nonmissing": 0,
                "n_missing": n,
                "percent_nonmissing": 0.0,
            })
        else:
            nn = int(df[c].notna().sum())
            rows.append({
                "variable": c,
                "column_present": True,
                "n_nonmissing": nn,
                "n_missing": n - nn,
                "percent_nonmissing": 100.0 * nn / n if n else np.nan,
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--biospecimen", default=None)
    p.add_argument("--clinical", default=None)
    p.add_argument("--locked-brain-metadata", default=None)
    p.add_argument("--pathology-metadata", default=None)
    p.add_argument("--assay-metadata", default=None)

    p.add_argument(
        "--outdir",
        default="results/monocyte_dlpfc_feasibility",
    )

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 100)
    print("ROSMAP MONOCYTE ↔ DLPFC MATCHED FEASIBILITY TABLE")
    print("=" * 100)
    print()

    biospec_path, clinical_path, locked_path, pathology_path, assay_path = (
        discover_inputs(args)
    )

    print(f"Biospecimen metadata: {biospec_path}")
    print(f"Clinical metadata:    {clinical_path}")
    print(f"Locked brain metadata:{locked_path if locked_path else ' NOT FOUND'}")
    print(f"Pathology metadata:   {pathology_path if pathology_path else ' NOT FOUND'}")
    print(f"RNA assay metadata:   {assay_path if assay_path else ' NOT FOUND'}")
    print()

    bio = pd.read_csv(biospec_path, low_memory=False)
    clin = pd.read_csv(clinical_path, low_memory=False)

    mono = get_monocytes(bio)
    brain, brain_source = get_brain_specimens(
        bio,
        locked_path,
    )

    print(
        f"Monocyte RNA-seq: {len(mono):,} specimens / "
        f"{mono['individualID'].nunique():,} participants"
    )
    print(
        f"Selected DLPFC cohort: {brain['individualID'].nunique():,} participants"
    )

    matched = mono.merge(
        brain,
        on="individualID",
        how="inner",
        validate="many_to_one",
    )

    print(
        f"Matched overlap: {matched['individualID'].nunique():,} participants / "
        f"{len(matched):,} monocyte specimens"
    )
    print()

    # Core clinical merge.
    cstd = clinical_standardized(clin)
    matched = matched.merge(
        cstd,
        on="individualID",
        how="left",
        validate="many_to_one",
    )

    # Normalize a few standard names from clinical immediately.
    if "mmse_last_valid" in matched.columns:
        matched["mmse"] = matched["mmse_last_valid"]

    if "cerad_score" in matched.columns:
        matched["cerad"] = matched["cerad_score"]

    if "braak_stage" in matched.columns:
        matched["braak"] = matched["braak_stage"]

    # Optional project metadata enrichments.
    matched = add_optional_phenotypes(
        matched,
        locked_path,
        bio,
        "locked brain metadata",
    )

    matched = add_optional_phenotypes(
        matched,
        pathology_path,
        bio,
        "pathology / merged AREA metadata",
    )

    matched = choose_best_age_death(matched)

    # Order useful columns first.
    preferred = [
        "individualID",
        "projid",
        "monocyte_specimen_id",
        "n_monocyte_rnaseq_samples_for_person",
        "repeated_monocyte_rnaseq",
        "monocyte_sampling_age_raw",
        "monocyte_sampling_age_exact",
        "monocyte_sampling_age_lower_bound",
        "age_death_raw",
        "age_death_exact",
        "age_death_source",
        "years_monocyte_to_death_exact",
        "interval_metadata_flag",
        "dlpfc_specimen_ids",
        "n_dlpfc_rnaseq_specimens",
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
    ]

    for c in preferred:
        if c not in matched.columns:
            matched[c] = np.nan

    remaining = [
        c for c in matched.columns
        if c not in preferred
    ]

    matched = matched[
        preferred + remaining
    ].sort_values(
        [
            "individualID",
            "monocyte_sampling_age_exact",
            "monocyte_specimen_id",
        ],
        na_position="last",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    specimen_out = (
        outdir
        / "monocyte_dlpfc_matched_specimen_table.csv"
    )
    participant_out = (
        outdir
        / "monocyte_dlpfc_matched_participant_table.csv"
    )
    summary_out = (
        outdir
        / "monocyte_dlpfc_feasibility_summary.csv"
    )
    missing_out = (
        outdir
        / "monocyte_dlpfc_missingness_summary.csv"
    )

    matched.to_csv(
        specimen_out,
        index=False,
    )

    participants = participant_table(
        matched
    )

    participants.to_csv(
        participant_out,
        index=False,
    )

    summary = make_summary(
        mono,
        brain,
        matched,
        brain_source,
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    missingness_summary(
        participants
    ).to_csv(
        missing_out,
        index=False,
    )

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print()
    print(f"Specimen table:    {specimen_out}")
    print(f"Participant table: {participant_out}")
    print(f"Summary:           {summary_out}")
    print(f"Missingness:       {missing_out}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

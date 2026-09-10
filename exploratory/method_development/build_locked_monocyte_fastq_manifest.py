#!/usr/bin/env python3
"""
build_locked_monocyte_fastq_manifest.py
=======================================

Build the exact Synapse FASTQ manifest for the LOCKED ROSMAP monocyte ↔ DLPFC
cross-tissue cohort.

The script does NOT download FASTQs. It queries Synapse metadata, recursively
indexes FASTQ files under the monocyte RNA-seq folder, and matches them to the
locked monocyte specimen IDs.

Designed specifically for:
    results/monocyte_dlpfc_cross_tissue_v2/
        locked_cross_tissue_manifest.csv

Default Synapse monocyte parent:
    syn22024496

Key safeguards
--------------
- Only specimens present in the locked cross-tissue manifest are retained.
- BC15Control is excluded if encountered.
- Batch-3 lane-split FASTQs are preserved as separate files.
- If both apparent merged and lane-split files coexist for the same specimen,
  the script DOES NOT silently choose one. It flags the specimen as ambiguous.
- Read-pair balance is audited.
- Every locked specimen gets a status:
      PASS
      MISSING_FASTQ
      UNPAIRED_READS
      POSSIBLE_MERGED_AND_LANE_SPLIT_CONFLICT
      MULTIPLE_NAMING_PATTERNS
- No FASTQ is downloaded until the manifest passes QC.

Outputs
-------
results/monocyte_dlpfc_fastq_manifest/

    locked_monocyte_fastq_manifest_all_matches.csv
    locked_monocyte_fastq_manifest_download_ready.csv
    locked_monocyte_fastq_specimen_qc.csv
    locked_monocyte_fastq_unmatched_synapse_files.csv
    locked_monocyte_fastq_summary.txt

Requirements
------------
    python3 -m pip install synapseclient pandas

Authentication
--------------
Recommended:
    synapse login

or configure a Synapse personal access token in the usual synapseclient way.

Run from repo root
------------------
    python3 exploratory/method_development/build_locked_monocyte_fastq_manifest.py

If the actual FASTQ folder is a child of syn22024496, recursive traversal will
find it automatically.

Optional:
    --synapse-root syn22024496
    --locked-manifest results/monocyte_dlpfc_cross_tissue_v2/locked_cross_tissue_manifest.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_LOCKED_MANIFEST = (
    "results/monocyte_dlpfc_cross_tissue_v2/"
    "locked_cross_tissue_manifest.csv"
)

DEFAULT_OUTDIR = (
    "results/monocyte_dlpfc_fastq_manifest"
)

DEFAULT_SYNAPSE_ROOT = "syn22024496"

CONTROL_TOKENS = [
    "bc15control",
    "control",
]

FASTQ_SUFFIXES = (
    ".fastq.gz",
    ".fq.gz",
    ".fastq",
    ".fq",
)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--locked-manifest",
        default=DEFAULT_LOCKED_MANIFEST,
    )

    p.add_argument(
        "--synapse-root",
        default=DEFAULT_SYNAPSE_ROOT,
        help=(
            "Synapse folder/project containing the monocyte FASTQs. "
            "Traversal is recursive."
        ),
    )

    p.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
    )

    p.add_argument(
        "--include-ambiguous",
        action="store_true",
        help=(
            "Include specimens with ambiguous merged-vs-lane-split files "
            "in the download-ready manifest. Default: exclude them."
        ),
    )

    return p.parse_args()


# =============================================================================
# LOCKED COHORT
# =============================================================================

def load_locked_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Locked cross-tissue manifest not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = [
        "individualID",
        "monocyte_specimen_id",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Locked manifest is missing required columns: {missing}"
        )

    # Keep only the actual locked cross-tissue cohort if the flag exists.
    if "locked_cross_tissue_core" in df.columns:
        core = (
            df["locked_cross_tissue_core"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df.loc[core].copy()

    if df["individualID"].duplicated().any():
        raise ValueError(
            "Locked manifest contains duplicate individualID rows."
        )

    if df["monocyte_specimen_id"].duplicated().any():
        dupes = (
            df.loc[
                df["monocyte_specimen_id"].duplicated(keep=False),
                "monocyte_specimen_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Locked manifest contains duplicate monocyte specimen IDs. "
            f"Examples: {dupes[:10]}"
        )

    df["monocyte_specimen_id"] = (
        df["monocyte_specimen_id"]
        .astype(str)
        .str.strip()
    )

    return df


# =============================================================================
# SYNAPSE TRAVERSAL
# =============================================================================

def import_synapseclient():
    try:
        import synapseclient
    except ImportError as exc:
        raise RuntimeError(
            "synapseclient is not installed.\n"
            "Install it with:\n"
            "  python3 -m pip install synapseclient\n"
        ) from exc

    return synapseclient


def entity_annotations_to_dict(entity):
    out = {}

    # Newer synapseclient entity objects often expose annotations directly.
    ann = getattr(entity, "annotations", None)

    if ann is not None:
        try:
            items = dict(ann)
            for k, v in items.items():
                out[k] = v
        except Exception:
            pass

    return out


def recurse_synapse_files(syn, root_id: str):
    """
    Recursively yield current child file entities under root_id.
    """
    queue = deque(
        [(root_id, "")]
    )

    seen_containers = set()

    while queue:
        parent_id, parent_path = queue.popleft()

        if parent_id in seen_containers:
            continue

        seen_containers.add(
            parent_id
        )

        children = list(
            syn.getChildren(
                parent_id,
                includeTypes=[
                    "file",
                    "folder",
                    "project",
                ],
            )
        )

        for child in children:
            child_id = child.get("id")
            child_name = child.get("name", "")
            child_type = str(
                child.get("type", "")
            ).lower()

            rel_path = (
                f"{parent_path}/{child_name}"
                if parent_path
                else child_name
            )

            if "folder" in child_type or "project" in child_type:
                queue.append(
                    (child_id, rel_path)
                )
                continue

            if "file" not in child_type:
                continue

            # Metadata only: downloadFile=False.
            try:
                entity = syn.get(
                    child_id,
                    downloadFile=False,
                )
            except Exception:
                entity = None

            if entity is None:
                yield {
                    "synapse_id": child_id,
                    "file_name": child_name,
                    "relative_path": rel_path,
                    "parent_id": parent_id,
                    "file_version": np.nan,
                    "content_size": np.nan,
                    "md5": np.nan,
                    "is_restricted": np.nan,
                    "annotations": {},
                }
                continue

            yield {
                "synapse_id": child_id,
                "file_name": getattr(entity, "name", child_name),
                "relative_path": rel_path,
                "parent_id": parent_id,
                "file_version": getattr(entity, "versionNumber", np.nan),
                "content_size": getattr(entity, "contentSize", np.nan),
                "md5": getattr(entity, "md5", np.nan),
                "is_restricted": getattr(entity, "isRestricted", np.nan),
                "annotations": entity_annotations_to_dict(entity),
            }


# =============================================================================
# FASTQ NAME PARSING
# =============================================================================

def normalize_filename(name: str) -> str:
    return str(name).strip()


def is_fastq(name: str) -> bool:
    low = normalize_filename(name).lower()
    return low.endswith(
        FASTQ_SUFFIXES
    )


def is_control_file(name: str) -> bool:
    low = normalize_filename(name).lower()
    return any(
        token in low
        for token in CONTROL_TOKENS
    )


def extract_read_end(name: str) -> Optional[str]:
    """
    Recognize common paired-end conventions:
      END1 / END2
      R1 / R2
      _1 / _2 before fastq suffix
    """
    stem = normalize_filename(name)

    patterns = [
        (r"(?:^|[_\-.])END1(?:[_\-.]|$)", "R1"),
        (r"(?:^|[_\-.])END2(?:[_\-.]|$)", "R2"),
        (r"(?:^|[_\-.])R1(?:[_\-.]|$)", "R1"),
        (r"(?:^|[_\-.])R2(?:[_\-.]|$)", "R2"),
        (r"(?:^|[_\-.])READ1(?:[_\-.]|$)", "R1"),
        (r"(?:^|[_\-.])READ2(?:[_\-.]|$)", "R2"),
    ]

    for pat, label in patterns:
        if re.search(
            pat,
            stem,
            flags=re.IGNORECASE,
        ):
            return label

    # Conservative fallback for names ending _1.fastq.gz / _2.fastq.gz
    if re.search(
        r"_1\.(?:fastq|fq)(?:\.gz)?$",
        stem,
        flags=re.IGNORECASE,
    ):
        return "R1"

    if re.search(
        r"_2\.(?:fastq|fq)(?:\.gz)?$",
        stem,
        flags=re.IGNORECASE,
    ):
        return "R2"

    return None


def extract_lane(name: str) -> Optional[str]:
    """
    Recognize common lane conventions such as L001, lane1, lane_1.
    """
    patterns = [
        r"(?:^|[_\-.])(L00\d)(?:[_\-.]|$)",
        r"(?:^|[_\-.])lane[_-]?(\d+)(?:[_\-.]|$)",
    ]

    for pat in patterns:
        m = re.search(
            pat,
            name,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1)

    return None


def detect_naming_pattern(name: str) -> str:
    low = name.lower()

    if "end1" in low or "end2" in low:
        return "END1_END2"

    if re.search(
        r"(?:^|[_\-.])r[12](?:[_\-.]|$)",
        low,
    ):
        return "R1_R2"

    if re.search(
        r"_[12]\.(?:fastq|fq)(?:\.gz)?$",
        low,
    ):
        return "suffix_1_2"

    return "unrecognized"


def specimen_matches_filename(
    specimen_id: str,
    file_name: str,
) -> bool:
    """
    Match locked specimen IDs such as Sample_001 against FASTQ names while
    avoiding Sample_001 matching Sample_0010.
    """
    specimen = re.escape(
        specimen_id.strip()
    )

    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){specimen}(?![A-Za-z0-9])",
            file_name,
            flags=re.IGNORECASE,
        )
        or file_name.lower().startswith(
            specimen_id.lower()
        )
    )


def infer_file_structure(group: pd.DataFrame):
    lanes = [
        x for x in group["lane"].dropna().astype(str).unique()
    ]

    has_lane_split = len(lanes) > 0

    # "Merged-looking" means a paired FASTQ with no lane annotation.
    has_unlaned = group["lane"].isna().any()

    if has_lane_split and has_unlaned:
        return "mixed_lane_split_and_unlaned"

    if has_lane_split:
        return "lane_split"

    return "unlaned"


# =============================================================================
# BUILD MATCHES
# =============================================================================

def build_fastq_index(synapse_records):
    rows = []

    for record in synapse_records:
        name = record["file_name"]

        if not is_fastq(name):
            continue

        annotations = record.pop(
            "annotations",
            {},
        )

        row = dict(record)
        row["read_end"] = extract_read_end(name)
        row["lane"] = extract_lane(name)
        row["naming_pattern"] = detect_naming_pattern(name)
        row["control_like_name"] = is_control_file(name)

        # Preserve a few useful annotations if present.
        for key in [
            "individualID",
            "specimenID",
            "batch",
            "rnaBatch",
            "deprecated",
            "isDeprecated",
        ]:
            value = annotations.get(
                key,
                np.nan,
            )
            if isinstance(
                value,
                (list, tuple),
            ):
                value = ";".join(
                    map(str, value)
                )
            row[f"annotation_{key}"] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def match_locked_specimens(
    locked: pd.DataFrame,
    fastqs: pd.DataFrame,
):
    rows = []

    for _, person in locked.iterrows():
        specimen = person[
            "monocyte_specimen_id"
        ]

        matches = fastqs[
            fastqs["file_name"].map(
                lambda x: specimen_matches_filename(
                    specimen,
                    x,
                )
            )
        ].copy()

        for _, fq in matches.iterrows():
            row = {
                "individualID": person[
                    "individualID"
                ],
                "projid": person.get(
                    "projid",
                    np.nan,
                ),
                "monocyte_specimen_id": specimen,
                "synapse_id": fq[
                    "synapse_id"
                ],
                "file_name": fq[
                    "file_name"
                ],
                "relative_path": fq[
                    "relative_path"
                ],
                "read_end": fq[
                    "read_end"
                ],
                "lane": fq[
                    "lane"
                ],
                "naming_pattern": fq[
                    "naming_pattern"
                ],
                "file_structure_component": (
                    "lane_split"
                    if pd.notna(fq["lane"])
                    else "unlaned"
                ),
                "file_version": fq[
                    "file_version"
                ],
                "content_size": fq[
                    "content_size"
                ],
                "md5": fq[
                    "md5"
                ],
                "control_like_name": fq[
                    "control_like_name"
                ],
                "annotation_deprecated": fq.get(
                    "annotation_deprecated",
                    np.nan,
                ),
                "annotation_isDeprecated": fq.get(
                    "annotation_isDeprecated",
                    np.nan,
                ),
            }

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# QC
# =============================================================================

def truthy_deprecated(value) -> bool:
    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "deprecated",
    }


def specimen_qc(
    locked: pd.DataFrame,
    matches: pd.DataFrame,
):
    rows = []

    for _, person in locked.iterrows():
        specimen = person[
            "monocyte_specimen_id"
        ]

        g = matches[
            matches[
                "monocyte_specimen_id"
            ].eq(specimen)
        ].copy()

        if len(g) == 0:
            rows.append(
                {
                    "individualID": person[
                        "individualID"
                    ],
                    "projid": person.get(
                        "projid",
                        np.nan,
                    ),
                    "monocyte_specimen_id": specimen,
                    "n_fastq_files": 0,
                    "n_R1": 0,
                    "n_R2": 0,
                    "n_unrecognized_read_end": 0,
                    "n_lane_split_files": 0,
                    "n_unlaned_files": 0,
                    "n_deprecated_flagged": 0,
                    "file_structure": "missing",
                    "naming_patterns": "",
                    "status": "MISSING_FASTQ",
                }
            )
            continue

        n_r1 = int(
            g["read_end"].eq("R1").sum()
        )
        n_r2 = int(
            g["read_end"].eq("R2").sum()
        )
        n_unknown = int(
            g["read_end"].isna().sum()
        )
        n_lane = int(
            g["lane"].notna().sum()
        )
        n_unlaned = int(
            g["lane"].isna().sum()
        )

        deprecated = (
            g["annotation_deprecated"].map(
                truthy_deprecated
            )
            | g["annotation_isDeprecated"].map(
                truthy_deprecated
            )
        )

        n_deprecated = int(
            deprecated.sum()
        )

        structure = infer_file_structure(
            g
        )

        patterns = sorted(
            set(
                g["naming_pattern"]
                .dropna()
                .astype(str)
            )
        )

        statuses = []

        if n_unknown > 0:
            statuses.append(
                "UNRECOGNIZED_READ_END"
            )

        if n_r1 != n_r2:
            statuses.append(
                "UNPAIRED_READS"
            )

        if structure == "mixed_lane_split_and_unlaned":
            statuses.append(
                "POSSIBLE_MERGED_AND_LANE_SPLIT_CONFLICT"
            )

        if len(patterns) > 1:
            statuses.append(
                "MULTIPLE_NAMING_PATTERNS"
            )

        if n_deprecated > 0:
            statuses.append(
                "DEPRECATED_FILE_PRESENT"
            )

        if not statuses:
            statuses = [
                "PASS"
            ]

        rows.append(
            {
                "individualID": person[
                    "individualID"
                ],
                "projid": person.get(
                    "projid",
                    np.nan,
                ),
                "monocyte_specimen_id": specimen,
                "n_fastq_files": len(g),
                "n_R1": n_r1,
                "n_R2": n_r2,
                "n_unrecognized_read_end": n_unknown,
                "n_lane_split_files": n_lane,
                "n_unlaned_files": n_unlaned,
                "n_deprecated_flagged": n_deprecated,
                "file_structure": structure,
                "naming_patterns": ";".join(
                    patterns
                ),
                "status": ";".join(
                    statuses
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_download_ready(
    matches: pd.DataFrame,
    qc: pd.DataFrame,
    include_ambiguous: bool,
):
    allowed = {
        "PASS",
    }

    if include_ambiguous:
        # Still do not allow missing/unpaired/unrecognized/deprecated files.
        acceptable = qc[
            ~qc["status"].str.contains(
                "MISSING_FASTQ|UNPAIRED_READS|UNRECOGNIZED_READ_END|DEPRECATED_FILE_PRESENT",
                regex=True,
                na=False,
            )
        ]
    else:
        acceptable = qc[
            qc["status"].isin(
                allowed
            )
        ]

    specimens = set(
        acceptable[
            "monocyte_specimen_id"
        ]
    )

    ready = matches[
        matches[
            "monocyte_specimen_id"
        ].isin(specimens)
    ].copy()

    # Exclude any explicitly deprecated files, even if the specimen is otherwise OK.
    dep = (
        ready["annotation_deprecated"].map(
            truthy_deprecated
        )
        | ready["annotation_isDeprecated"].map(
            truthy_deprecated
        )
    )

    ready = ready.loc[
        ~dep
    ].copy()

    ready = ready.sort_values(
        [
            "monocyte_specimen_id",
            "lane",
            "read_end",
            "file_name",
        ],
        na_position="last",
    )

    return ready


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary(
    path: Path,
    locked: pd.DataFrame,
    fastqs: pd.DataFrame,
    matches: pd.DataFrame,
    qc: pd.DataFrame,
    ready: pd.DataFrame,
    root_id: str,
):
    status_counts = (
        qc["status"]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "status"
        )
        .reset_index(
            name="n_specimens"
        )
    )

    lines = [
        "ROSMAP LOCKED MONOCYTE FASTQ MANIFEST",
        "=" * 72,
        "",
        f"Synapse root: {root_id}",
        f"Locked participants/specimens: {len(locked)}",
        f"FASTQ files indexed under Synapse root: {len(fastqs)}",
        f"FASTQ files matched to locked specimens: {len(matches)}",
        f"Locked specimens with ≥1 FASTQ match: {matches['monocyte_specimen_id'].nunique() if len(matches) else 0}",
        f"Download-ready specimens: {ready['monocyte_specimen_id'].nunique() if len(ready) else 0}",
        f"Download-ready FASTQ files: {len(ready)}",
        "",
        "Specimen QC status counts:",
        status_counts.to_string(index=False),
        "",
        "Important batch-3 rule:",
        (
            "Lane-split FASTQs are preserved. Specimens containing both lane-split "
            "and unlaned FASTQs are flagged rather than silently resolved."
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
    print("BUILD LOCKED ROSMAP MONOCYTE FASTQ MANIFEST")
    print("=" * 100)
    print()

    locked_path = Path(
        args.locked_manifest
    )

    outdir = Path(
        args.outdir
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    locked = load_locked_manifest(
        locked_path
    )

    print(
        f"Locked manifest: {locked_path}"
    )
    print(
        f"Locked specimens: {len(locked)}"
    )
    print(
        f"Synapse root: {args.synapse_root}"
    )
    print()

    synapseclient = import_synapseclient()

    syn = synapseclient.Synapse(
        silent=True
    )

    try:
        syn.login(
            silent=True
        )
    except TypeError:
        syn.login()

    print(
        "Recursively indexing Synapse files..."
    )

    records = list(
        recurse_synapse_files(
            syn,
            args.synapse_root,
        )
    )

    fastqs = build_fastq_index(
        records
    )

    print(
        f"FASTQ files found: {len(fastqs)}"
    )

    if len(fastqs) == 0:
        raise RuntimeError(
            "No FASTQ files were found under the supplied Synapse root."
        )

    # Controls are never eligible for locked download.
    fastqs_noncontrol = fastqs.loc[
        ~fastqs[
            "control_like_name"
        ]
    ].copy()

    matches = match_locked_specimens(
        locked,
        fastqs_noncontrol,
    )

    qc = specimen_qc(
        locked,
        matches,
    )

    ready = build_download_ready(
        matches,
        qc,
        args.include_ambiguous,
    )

    # Synapse FASTQs that did not map to locked specimens.
    matched_ids = set(
        matches[
            "synapse_id"
        ]
    ) if len(matches) else set()

    unmatched = fastqs.loc[
        ~fastqs[
            "synapse_id"
        ].isin(
            matched_ids
        )
    ].copy()

    all_path = (
        outdir
        / "locked_monocyte_fastq_manifest_all_matches.csv"
    )

    ready_path = (
        outdir
        / "locked_monocyte_fastq_manifest_download_ready.csv"
    )

    qc_path = (
        outdir
        / "locked_monocyte_fastq_specimen_qc.csv"
    )

    unmatched_path = (
        outdir
        / "locked_monocyte_fastq_unmatched_synapse_files.csv"
    )

    summary_path = (
        outdir
        / "locked_monocyte_fastq_summary.txt"
    )

    matches.to_csv(
        all_path,
        index=False,
    )

    ready.to_csv(
        ready_path,
        index=False,
    )

    qc.to_csv(
        qc_path,
        index=False,
    )

    unmatched.to_csv(
        unmatched_path,
        index=False,
    )

    write_summary(
        summary_path,
        locked,
        fastqs,
        matches,
        qc,
        ready,
        args.synapse_root,
    )

    print()
    print("QC status:")
    print(
        qc["status"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )
    print()

    print(
        f"Download-ready specimens: "
        f"{ready['monocyte_specimen_id'].nunique() if len(ready) else 0}"
    )
    print(
        f"Download-ready FASTQ files: {len(ready)}"
    )
    print()

    print("Outputs:")
    print(f"  {all_path}")
    print(f"  {ready_path}")
    print(f"  {qc_path}")
    print(f"  {unmatched_path}")
    print(f"  {summary_path}")
    print()
    print(
        "Do not start the FASTQ download until the specimen QC table has "
        "been inspected, especially any batch-3 lane-split conflicts."
    )


if __name__ == "__main__":
    main()

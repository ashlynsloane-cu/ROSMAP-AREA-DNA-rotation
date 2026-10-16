#!/usr/bin/env python3
"""
Stage 12C — targeted monocyte follow-up of brain RNA-protein discoveries.

Purpose
-------
Ask whether genes prioritized by formal DLPFC RNA-protein conjunction analyses
show corresponding signal in blood monocyte RNA.

This is deliberately NOT a new genome-wide monocyte discovery analysis.
The candidate set is fixed upstream from the brain analyses and preserves
brain-method provenance:

    shared              = significant by AREA and DESeq2/limma brain conjunction
    AREA_only           = significant only by AREA brain conjunction
    conventional_only   = significant only by DESeq2/limma brain conjunction

Monocyte evidence
-----------------
For the matched contrast, report fully adjusted:

    monocyte AREA:
        pvalue, padj, direction

    monocyte DESeq2:
        pvalue, padj, log2FoldChange

No arbitrary top-X% threshold and no new candidate-set FDR are introduced.
Existing genome-wide monocyte FDR values are retained as the inferential
quantities. Nominal p<0.05 is reported descriptively.

Direction
---------
AREA direction strings are normalized:
    *_associated_higher_rank -> +1
    *_associated_lower_rank  -> -1

DESeq2 log2FoldChange is interpreted as group B - group A using the frozen
Stage 11 contrast orientation. If an adjusted DESeq2 run_manifest.json is
available, the script audits its group_a/group_b labels and flips the sign
only when the manifest is explicitly reversed relative to Stage 11.

Outputs
-------
results/stage12_brain_convergence/stage12C_monocyte_followup/
    BRAIN_CANDIDATE_MONOCYTE_EVIDENCE.tsv
    MONOCYTE_SUPPORT_SUMMARY_BY_BRAIN_CATEGORY.tsv
    MONOCYTE_INPUT_AUDIT.tsv
"""

from __future__ import annotations
import os

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
OFFLINE = Path(os.environ["ROSMAP_OFFLINE_ROOT"]).expanduser().resolve()
REPO = Path(os.environ["ROSMAP_RESEARCH_ROOT"]).expanduser().resolve()
TASK_FILE = (
    OFFLINE
    / "results/stage11_covariate_stress/stage11_tasks.tsv"
)

GENE_UNIVERSE = (
    REPO
    / "results/preprocessing_protein_coding"
    / "ROSMAP_global_gene_universe.tsv"
)

BRAIN_COMPARISON = (
    OFFLINE
    / "results/stage12_brain_convergence"
    / "stage12B_deseq2_limma_conjunction"
    / "AREA_VS_CONVENTIONAL_CONJUNCTION_COMPARISON.tsv"
)

AREA_BRAIN = (
    OFFLINE
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
    / "ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv"
)

CONVENTIONAL_BRAIN = (
    OFFLINE
    / "results/stage12_brain_convergence"
    / "stage12B_deseq2_limma_conjunction"
    / "ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv"
)

OUTDIR = (
    OFFLINE
    / "results/stage12_brain_convergence"
    / "stage12C_monocyte_followup"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def localize_area_result(path_string: str) -> Path:
    marker = "/results/multiomic_models/"
    s = str(path_string)

    if marker not in s:
        raise ValueError(f"Cannot localize AREA result: {s}")

    return REPO / "results/multiomic_models" / s.split(marker, 1)[1]


def find_adjusted_deseq2(task) -> tuple[Path, Path | None]:
    area_local = localize_area_result(task.A_result)
    mean_dir = area_local.parent.parent / "mean_based"

    hits = sorted(mean_dir.glob("*_DESeq2_all_results.csv"))

    if len(hits) != 1:
        raise RuntimeError(
            f"Task {task.task_id}: expected exactly one adjusted DESeq2 "
            f"result in {mean_dir}; found {hits}"
        )

    manifest = mean_dir / "run_manifest.json"

    return hits[0], manifest if manifest.exists() else None


def load_gene_map() -> pd.DataFrame:
    g = pd.read_csv(
        GENE_UNIVERSE,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    needed = {"ensembl_id_version", "ensembl_id", "gene_name"}
    missing = needed - set(g.columns)

    if missing:
        raise RuntimeError(
            f"Gene universe missing columns: {sorted(missing)}"
        )

    g = g[list(needed)].copy()

    g["ensembl_id_version"] = (
        g["ensembl_id_version"].astype(str).str.strip()
    )
    g["ensembl_id"] = g["ensembl_id"].astype(str).str.strip()
    g["gene_name"] = g["gene_name"].astype(str).str.strip()

    return g


def parse_area_sign(direction: object) -> float:
    s = str(direction)

    if "_associated_higher_rank" in s:
        return 1.0

    if "_associated_lower_rank" in s:
        return -1.0

    return np.nan


def normalize_label(x: object) -> str:
    s = str(x).strip().lower()

    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    s = s.replace("cerad", "")

    return s


def recursive_find(obj, target_key: str):
    """Return first value under target_key in an arbitrary JSON structure."""
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]

        for value in obj.values():
            found = recursive_find(value, target_key)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find(value, target_key)
            if found is not None:
                return found

    return None


def orientation_multiplier(
    run_manifest: Path | None,
    stage11_group_a: str,
    stage11_group_b: str,
) -> tuple[float, str, str | None, str | None]:
    """
    Return multiplier needed to express conventional effect as Stage11 group B-A.

    If no usable manifest exists, retain +1 but mark orientation as unverified.
    """
    if run_manifest is None:
        return 1.0, "unverified_no_run_manifest", None, None

    try:
        obj = json.loads(run_manifest.read_text())
    except Exception:
        return 1.0, "unverified_manifest_unreadable", None, None

    run_a = recursive_find(obj, "group_a")
    run_b = recursive_find(obj, "group_b")

    if run_a is None or run_b is None:
        return 1.0, "unverified_group_labels_not_found", None, None

    a = normalize_label(run_a)
    b = normalize_label(run_b)
    sa = normalize_label(stage11_group_a)
    sb = normalize_label(stage11_group_b)

    if a == sa and b == sb:
        return 1.0, "verified_same_orientation", str(run_a), str(run_b)

    if a == sb and b == sa:
        return -1.0, "verified_reversed_and_flipped", str(run_a), str(run_b)

    raise RuntimeError(
        "Adjusted DESeq2 run-manifest group labels do not match the "
        "Stage 11 contrast in either orientation:\n"
        f"  Stage11: {stage11_group_a} -> {stage11_group_b}\n"
        f"  run:     {run_a} -> {run_b}\n"
        f"  manifest: {run_manifest}"
    )


def map_ids_to_symbols(
    ids: pd.Series,
    gene_map: pd.DataFrame,
) -> pd.Series:
    """
    Map common feature-ID forms to gene symbols.

    Supported:
      versioned Ensembl gene ID
      unversioned Ensembl gene ID
      exact gene symbol
      SYMBOL|accession
    """
    ids = ids.astype(str).str.strip()

    by_version = dict(
        zip(
            gene_map["ensembl_id_version"],
            gene_map["gene_name"],
        )
    )

    by_ens = dict(
        zip(
            gene_map["ensembl_id"],
            gene_map["gene_name"],
        )
    )

    known_symbols = set(gene_map["gene_name"])

    out = []

    for x in ids:
        if x in by_version:
            out.append(by_version[x])
            continue

        stripped = re.sub(r"\.\d+$", "", x)

        if stripped in by_ens:
            out.append(by_ens[stripped])
            continue

        if x in known_symbols:
            out.append(x)
            continue

        if "|" in x:
            symbol = x.split("|", 1)[0].strip()

            if symbol in known_symbols:
                out.append(symbol)
                continue

        out.append(np.nan)

    return pd.Series(out, index=ids.index, dtype=object)


def first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_monocyte_area(
    path: Path,
    expected_n: int,
    gene_map: pd.DataFrame,
    candidate_symbols: set[str],
) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(path, low_memory=False)

    needed = {"feature_id", "pvalue", "padj", "direction"}
    missing = needed - set(d.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing AREA columns: {sorted(missing)}"
        )

    if len(d) != expected_n:
        raise RuntimeError(
            f"{path}: expected {expected_n} rows; found {len(d)}."
        )

    d = d.copy()

    d["gene_symbol"] = map_ids_to_symbols(
        d["feature_id"],
        gene_map,
    )

    d["area_pvalue"] = pd.to_numeric(
        d["pvalue"],
        errors="coerce",
    )
    d["area_padj"] = pd.to_numeric(
        d["padj"],
        errors="coerce",
    )
    d["area_direction"] = d["direction"].astype(str)
    d["area_direction_sign"] = d["area_direction"].map(
        parse_area_sign
    )

    mapped = int(d["gene_symbol"].notna().sum())

    mapped_all = d.loc[d["gene_symbol"].notna()].copy()
    global_dupes = (
        mapped_all.loc[
            mapped_all["gene_symbol"].duplicated(keep=False),
            "gene_symbol",
        ]
        .value_counts()
    )

    audit = {
        "area_path": str(path),
        "area_n_features": len(d),
        "area_n_symbol_mapped": mapped,
        "area_mapping_fraction": mapped / len(d),
        "area_global_duplicate_symbols_n": int(len(global_dupes)),
        "area_global_duplicate_symbol_examples": ",".join(
            map(str, global_dupes.head(20).index.tolist())
        ),
    }

    keep = [
        "gene_symbol",
        "feature_id",
        "area_pvalue",
        "area_padj",
        "area_direction",
        "area_direction_sign",
    ]

    d = d[keep].rename(
        columns={"feature_id": "monocyte_area_feature_id"}
    )

    d = d.loc[
        d["gene_symbol"].notna()
        & d["gene_symbol"].isin(candidate_symbols)
    ].copy()

    if d["gene_symbol"].duplicated().any():
        # Only candidate-level ambiguity is analysis-blocking.
        dupes = (
            d.loc[
                d["gene_symbol"].duplicated(keep=False),
                "gene_symbol",
            ]
            .value_counts()
            .to_dict()
        )
        raise RuntimeError(
            "Monocyte AREA has ambiguous mappings among the fixed "
            f"brain-candidate genes: {dupes}. "
            "Do not collapse these arbitrarily."
        )

    return d, audit


def load_monocyte_deseq2(
    path: Path,
    expected_n: int,
    gene_map: pd.DataFrame,
    multiplier: float,
    candidate_symbols: set[str],
) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(path, low_memory=False)

    id_col = first_present(
        d.columns,
        ["feature_id", "gene_id", "gene", d.columns[0]],
    )
    p_col = first_present(
        d.columns,
        ["pvalue", "p_value", "P.Value"],
    )
    q_col = first_present(
        d.columns,
        ["padj", "fdr", "adj.P.Val"],
    )
    effect_col = first_present(
        d.columns,
        ["log2FoldChange", "effect", "estimate"],
    )

    if any(x is None for x in [id_col, p_col, q_col, effect_col]):
        raise RuntimeError(
            f"Could not identify DESeq2 columns in {path}. "
            f"Columns={list(d.columns)}"
        )

    if len(d) != expected_n:
        raise RuntimeError(
            f"{path}: expected {expected_n} rows; found {len(d)}."
        )

    out = pd.DataFrame(
        {
            "feature_id": d[id_col].astype(str).str.strip(),
            "deseq2_pvalue": pd.to_numeric(
                d[p_col],
                errors="coerce",
            ),
            "deseq2_padj": pd.to_numeric(
                d[q_col],
                errors="coerce",
            ),
            "deseq2_log2fc_raw": pd.to_numeric(
                d[effect_col],
                errors="coerce",
            ),
        }
    )

    out["deseq2_log2fc_stage11_orientation"] = (
        out["deseq2_log2fc_raw"] * multiplier
    )

    out["deseq2_direction_sign"] = np.sign(
        out["deseq2_log2fc_stage11_orientation"]
    )

    out["gene_symbol"] = map_ids_to_symbols(
        out["feature_id"],
        gene_map,
    )

    mapped = int(out["gene_symbol"].notna().sum())

    mapped_all = out.loc[out["gene_symbol"].notna()].copy()
    global_dupes = (
        mapped_all.loc[
            mapped_all["gene_symbol"].duplicated(keep=False),
            "gene_symbol",
        ]
        .value_counts()
    )

    audit = {
        "deseq2_path": str(path),
        "deseq2_n_features": len(out),
        "deseq2_n_symbol_mapped": mapped,
        "deseq2_mapping_fraction": mapped / len(out),
        "deseq2_global_duplicate_symbols_n": int(len(global_dupes)),
        "deseq2_global_duplicate_symbol_examples": ",".join(
            map(str, global_dupes.head(20).index.tolist())
        ),
    }

    out = out.rename(
        columns={"feature_id": "monocyte_deseq2_feature_id"}
    )

    out = out.loc[
        out["gene_symbol"].notna()
        & out["gene_symbol"].isin(candidate_symbols)
    ].copy()

    if out["gene_symbol"].duplicated().any():
        dupes = (
            out.loc[
                out["gene_symbol"].duplicated(keep=False),
                "gene_symbol",
            ]
            .value_counts()
            .to_dict()
        )
        raise RuntimeError(
            "Monocyte DESeq2 has ambiguous mappings among the fixed "
            f"brain-candidate genes: {dupes}. "
            "Do not collapse these arbitrarily."
        )

    return out, audit


def split_csv_genes(value) -> list[str]:
    if pd.isna(value):
        return []

    s = str(value).strip()

    if not s:
        return []

    return [x.strip() for x in s.split(",") if x.strip()]


def build_brain_candidates() -> pd.DataFrame:
    comparison = pd.read_csv(
        BRAIN_COMPARISON,
        sep="\t",
        low_memory=False,
    )

    rows = []

    for r in comparison.itertuples(index=False):
        contrast_id = str(r.contrast_id)

        for gene in split_csv_genes(r.shared_genes):
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "gene_symbol": gene,
                    "brain_category": "shared",
                    "brain_AREA_BH": True,
                    "brain_conventional_BH": True,
                }
            )

        for gene in split_csv_genes(r.AREA_only_genes):
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "gene_symbol": gene,
                    "brain_category": "AREA_only",
                    "brain_AREA_BH": True,
                    "brain_conventional_BH": False,
                }
            )

        for gene in split_csv_genes(r.conventional_only_genes):
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "gene_symbol": gene,
                    "brain_category": "conventional_only",
                    "brain_AREA_BH": False,
                    "brain_conventional_BH": True,
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("No brain candidates found.")

    if out.duplicated(
        ["contrast_id", "gene_symbol"]
    ).any():
        raise RuntimeError(
            "Duplicate brain candidate membership within a contrast."
        )

    return out


def attach_brain_direction(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    area = pd.read_csv(
        AREA_BRAIN,
        sep="\t",
        low_memory=False,
    )

    conv = pd.read_csv(
        CONVENTIONAL_BRAIN,
        sep="\t",
        low_memory=False,
    )

    area_small = area[
        [
            "contrast_id",
            "gene_symbol",
            "rna_direction",
            "conjunction_fdr_bh",
        ]
    ].copy()

    area_small["brain_area_direction_sign"] = (
        area_small["rna_direction"].map(parse_area_sign)
    )

    area_small = area_small.rename(
        columns={
            "rna_direction": "brain_area_rna_direction",
            "conjunction_fdr_bh":
                "brain_area_conjunction_fdr_bh",
        }
    )

    conv_small = conv[
        [
            "contrast_id",
            "gene_symbol",
            "rna_effect",
            "protein_effect",
            "conjunction_fdr_bh",
        ]
    ].copy()

    conv_small["brain_conventional_direction_sign"] = np.sign(
        pd.to_numeric(
            conv_small["rna_effect"],
            errors="coerce",
        )
    )

    conv_small = conv_small.rename(
        columns={
            "rna_effect": "brain_deseq2_log2fc",
            "protein_effect": "brain_limma_logfc",
            "conjunction_fdr_bh":
                "brain_conventional_conjunction_fdr_bh",
        }
    )

    out = candidates.merge(
        area_small,
        on=["contrast_id", "gene_symbol"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        conv_small,
        on=["contrast_id", "gene_symbol"],
        how="left",
        validate="one_to_one",
    )

    return out


def main():
    for path in [
        TASK_FILE,
        GENE_UNIVERSE,
        BRAIN_COMPARISON,
        AREA_BRAIN,
        CONVENTIONAL_BRAIN,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    gene_map = load_gene_map()

    candidates = attach_brain_direction(
        build_brain_candidates()
    )

    # Only contrasts with formal brain conjunction discoveries contribute
    # candidates. Keep this dynamic rather than hard-coding NCI->AD/CERAD.
    candidate_contrasts = list(
        dict.fromkeys(candidates["contrast_id"].astype(str))
    )

    tasks = pd.read_csv(
        TASK_FILE,
        sep="\t",
    )

    monocyte_tasks = tasks.loc[
        tasks["modality"].astype(str).eq("monocyte")
        & tasks["contrast_id"].astype(str).isin(candidate_contrasts)
    ].copy()

    expected = set(candidate_contrasts)
    observed = set(monocyte_tasks["contrast_id"].astype(str))

    if observed != expected:
        raise RuntimeError(
            f"Missing monocyte Stage11 tasks. Expected {expected}; "
            f"found {observed}."
        )

    all_evidence = []
    audit_rows = []

    for task in monocyte_tasks.itertuples(index=False):
        contrast_id = str(task.contrast_id)

        area_path = localize_area_result(task.A_result)
        deseq_path, run_manifest = find_adjusted_deseq2(task)

        multiplier, orientation_status, run_a, run_b = (
            orientation_multiplier(
                run_manifest,
                str(task.group_a),
                str(task.group_b),
            )
        )

        candidate_subset = candidates.loc[
            candidates["contrast_id"].astype(str).eq(contrast_id)
        ].copy()

        candidate_symbols = set(
            candidate_subset["gene_symbol"].astype(str)
        )

        area, area_audit = load_monocyte_area(
            area_path,
            int(task.A_n_features),
            gene_map,
            candidate_symbols,
        )

        deseq, deseq_audit = load_monocyte_deseq2(
            deseq_path,
            int(task.A_n_features),
            gene_map,
            multiplier,
            candidate_symbols,
        )

        evidence = candidate_subset.merge(
            area,
            on="gene_symbol",
            how="left",
            validate="one_to_one",
        )

        evidence = evidence.merge(
            deseq,
            on="gene_symbol",
            how="left",
            validate="one_to_one",
        )

        evidence["stage11_group_a"] = str(task.group_a)
        evidence["stage11_group_b"] = str(task.group_b)
        evidence["deseq2_orientation_status"] = orientation_status

        evidence["monocyte_area_present"] = (
            evidence["area_pvalue"].notna()
            | evidence["area_padj"].notna()
        )

        evidence["monocyte_deseq2_present"] = (
            evidence["deseq2_pvalue"].notna()
            | evidence["deseq2_padj"].notna()
        )

        evidence["monocyte_area_nominal_p05"] = (
            evidence["area_pvalue"] < 0.05
        )

        evidence["monocyte_area_fdr05"] = (
            evidence["area_padj"] < 0.05
        )

        evidence["monocyte_deseq2_nominal_p05"] = (
            evidence["deseq2_pvalue"] < 0.05
        )

        evidence["monocyte_deseq2_fdr05"] = (
            evidence["deseq2_padj"] < 0.05
        )

        evidence["monocyte_methods_direction_agree"] = (
            evidence["area_direction_sign"].notna()
            & evidence["deseq2_direction_sign"].notna()
            & evidence["area_direction_sign"].eq(
                evidence["deseq2_direction_sign"]
            )
        )

        evidence["monocyte_both_nominal_p05"] = (
            evidence["monocyte_area_nominal_p05"]
            & evidence["monocyte_deseq2_nominal_p05"]
        )

        evidence["monocyte_both_fdr05"] = (
            evidence["monocyte_area_fdr05"]
            & evidence["monocyte_deseq2_fdr05"]
        )

        evidence["monocyte_area_matches_brain_area_direction"] = (
            evidence["area_direction_sign"].notna()
            & evidence["brain_area_direction_sign"].notna()
            & evidence["area_direction_sign"].eq(
                evidence["brain_area_direction_sign"]
            )
        )

        evidence[
            "monocyte_deseq2_matches_brain_conventional_direction"
        ] = (
            evidence["deseq2_direction_sign"].notna()
            & evidence["brain_conventional_direction_sign"].notna()
            & evidence["deseq2_direction_sign"].eq(
                evidence["brain_conventional_direction_sign"]
            )
        )

        all_evidence.append(evidence)

        audit_rows.append(
            {
                "contrast_id": contrast_id,
                "task_id": int(task.task_id),
                "stage11_group_a": str(task.group_a),
                "stage11_group_b": str(task.group_b),
                "n_candidates": len(candidate_subset),
                "adjusted_deseq2_run_manifest":
                    str(run_manifest) if run_manifest else "",
                "adjusted_deseq2_run_group_a":
                    run_a if run_a is not None else "",
                "adjusted_deseq2_run_group_b":
                    run_b if run_b is not None else "",
                "deseq2_orientation_status": orientation_status,
                **area_audit,
                **deseq_audit,
            }
        )

    evidence = pd.concat(
        all_evidence,
        ignore_index=True,
    )

    audit = pd.DataFrame(audit_rows)

    summary_rows = []

    for (contrast_id, category), g in evidence.groupby(
        ["contrast_id", "brain_category"],
        sort=False,
    ):
        summary_rows.append(
            {
                "contrast_id": contrast_id,
                "brain_category": category,
                "n_brain_candidates": len(g),
                "monocyte_AREA_nominal_p05_n":
                    int(g["monocyte_area_nominal_p05"].sum()),
                "monocyte_AREA_fdr05_n":
                    int(g["monocyte_area_fdr05"].sum()),
                "monocyte_DESeq2_nominal_p05_n":
                    int(g["monocyte_deseq2_nominal_p05"].sum()),
                "monocyte_DESeq2_fdr05_n":
                    int(g["monocyte_deseq2_fdr05"].sum()),
                "monocyte_both_methods_nominal_p05_n":
                    int(g["monocyte_both_nominal_p05"].sum()),
                "monocyte_both_methods_fdr05_n":
                    int(g["monocyte_both_fdr05"].sum()),
                "monocyte_methods_direction_agree_n":
                    int(g["monocyte_methods_direction_agree"].sum()),
                "AREA_monocyte_matches_AREA_brain_direction_n":
                    int(
                        g[
                            "monocyte_area_matches_brain_area_direction"
                        ].sum()
                    ),
                "DESeq2_monocyte_matches_conventional_brain_direction_n":
                    int(
                        g[
                            "monocyte_deseq2_matches_brain_conventional_direction"
                        ].sum()
                    ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Useful display ordering: shared first, then method-specific.
    category_order = {
        "shared": 0,
        "AREA_only": 1,
        "conventional_only": 2,
    }

    summary["_category_order"] = summary[
        "brain_category"
    ].map(category_order)

    summary = (
        summary
        .sort_values(
            ["contrast_id", "_category_order"],
            kind="mergesort",
        )
        .drop(columns="_category_order")
        .reset_index(drop=True)
    )

    evidence_path = (
        OUTDIR / "BRAIN_CANDIDATE_MONOCYTE_EVIDENCE.tsv"
    )
    summary_path = (
        OUTDIR / "MONOCYTE_SUPPORT_SUMMARY_BY_BRAIN_CATEGORY.tsv"
    )
    audit_path = (
        OUTDIR / "MONOCYTE_INPUT_AUDIT.tsv"
    )

    evidence.to_csv(
        evidence_path,
        sep="\t",
        index=False,
    )

    summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
    )

    audit.to_csv(
        audit_path,
        sep="\t",
        index=False,
    )

    print("=" * 120)
    print("STAGE 12C — TARGETED MONOCYTE FOLLOW-UP")
    print("=" * 120)
    print(
        "Brain candidate categories are fixed upstream; "
        "no new monocyte selection threshold is introduced."
    )

    print("\nINPUT / ORIENTATION AUDIT")
    print("-" * 120)
    print(
        audit[
            [
                "contrast_id",
                "task_id",
                "stage11_group_a",
                "stage11_group_b",
                "n_candidates",
                "area_n_features",
                "area_n_symbol_mapped",
                "area_global_duplicate_symbols_n",
                "area_global_duplicate_symbol_examples",
                "deseq2_n_features",
                "deseq2_n_symbol_mapped",
                "deseq2_global_duplicate_symbols_n",
                "deseq2_global_duplicate_symbol_examples",
                "deseq2_orientation_status",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nMONOCYTE SUPPORT BY BRAIN DISCOVERY CATEGORY")
    print("-" * 120)
    print(
        summary
        .to_string(index=False)
    )

    print("\nCANDIDATES WITH MONOCYTE FDR < 0.05 IN EITHER METHOD")
    print("-" * 120)

    hit = evidence.loc[
        evidence["monocyte_area_fdr05"]
        | evidence["monocyte_deseq2_fdr05"]
    ].copy()

    if hit.empty:
        print("None.")
    else:
        cols = [
            "contrast_id",
            "gene_symbol",
            "brain_category",
            "area_direction",
            "area_pvalue",
            "area_padj",
            "deseq2_log2fc_stage11_orientation",
            "deseq2_pvalue",
            "deseq2_padj",
            "monocyte_methods_direction_agree",
            "monocyte_area_matches_brain_area_direction",
            "monocyte_deseq2_matches_brain_conventional_direction",
        ]

        print(
            hit[cols]
            .sort_values(
                [
                    "contrast_id",
                    "brain_category",
                    "gene_symbol",
                ],
                kind="mergesort",
            )
            .round(7)
            .to_string(index=False)
        )

    print("\nSHARED BRAIN DISCOVERIES — ALL MONOCYTE EVIDENCE")
    print("-" * 120)

    shared = evidence.loc[
        evidence["brain_category"].eq("shared")
    ].copy()

    cols = [
        "contrast_id",
        "gene_symbol",
        "brain_category",
        "area_direction",
        "area_pvalue",
        "area_padj",
        "deseq2_log2fc_stage11_orientation",
        "deseq2_pvalue",
        "deseq2_padj",
        "monocyte_methods_direction_agree",
        "monocyte_area_matches_brain_area_direction",
        "monocyte_deseq2_matches_brain_conventional_direction",
    ]

    print(
        shared[cols]
        .sort_values(
            ["contrast_id", "gene_symbol"],
            kind="mergesort",
        )
        .round(7)
        .to_string(index=False)
    )

    print("\nWROTE:")
    print(" ", evidence_path)
    print(" ", summary_path)
    print(" ", audit_path)

    print(
        "\nINTERPRETATION RULE:\n"
        "  Monocyte evidence is a peripheral follow-up to brain-derived candidates.\n"
        "  Existing monocyte genome-wide padj values remain primary. Nominal p<0.05\n"
        "  is descriptive only. Lack of monocyte support does not invalidate a\n"
        "  brain RNA-protein discovery."
    )


if __name__ == "__main__":
    main()

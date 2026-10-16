#!/usr/bin/env python3
"""
Compare fully adjusted vs phenotype-only conventional rankings for DLPFC
DESeq2 and TMT limma, and place them beside Stage 11 AREA A-vs-B metrics.

Uses the frozen Stage 11 deterministic ranking convention:
    p ascending
    feature_id ascending for exact p ties
    stable mergesort
    unique integer ranks 1..N
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OFFLINE = Path.home() / "Developer/stage11_offline_2026-09-28"
REPO = Path.home() / "Developer/area-workspace/ROSMAP-AREA-DNA-rotation"

TASK_FILE = (
    OFFLINE
    / "results/stage11_covariate_stress/stage11_tasks.tsv"
)

PHENO_ROOT = (
    OFFLINE
    / "results/stage11_conventional_covariate_sensitivity"
    / "phenotype_only"
)

OUTDIR = (
    OFFLINE
    / "results/stage11_conventional_covariate_sensitivity"
    / "comparison"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

TOPKS = [250, 500, 1000]


def bare_id(x):
    return str(x).split(".")[0]


def localize_multiomic_result(path_string: str) -> Path:
    marker = "/results/multiomic_models/"
    s = str(path_string)

    if marker not in s:
        raise ValueError(f"Cannot localize result path: {s}")

    suffix = s.split(marker, 1)[1]
    return REPO / "results/multiomic_models" / suffix


def deterministic_rank(df, p_col):
    work = pd.DataFrame(
        {
            "feature_id": df["feature_id"].astype(str),
            "p": pd.to_numeric(
                df[p_col],
                errors="coerce",
            ).fillna(np.inf),
        },
        index=df.index,
    )

    ordered = work.sort_values(
        ["p", "feature_id"],
        ascending=[True, True],
        kind="mergesort",
    ).index

    ranks = pd.Series(
        np.arange(1, len(df) + 1),
        index=ordered,
        dtype=int,
    )

    return ranks.reindex(df.index).to_numpy()


def identify_columns(d, method):
    id_candidates = [
        "feature_id",
        "gene_id",
        "protein_id_raw",
        "protein_id",
        "Geneid",
    ]

    id_col = next(
        (c for c in id_candidates if c in d.columns),
        d.columns[0],
    )

    if method == "AREA":
        p_candidates = ["p_value", "pvalue", "p"]
        q_candidates = ["fdr", "padj", "FDR"]
    elif method == "DESeq2":
        p_candidates = ["pvalue", "p_value", "p"]
        q_candidates = ["padj", "fdr", "FDR"]
    elif method == "limma":
        p_candidates = ["P.Value", "p_value", "pvalue"]
        q_candidates = ["adj.P.Val", "fdr", "FDR"]
    else:
        raise ValueError(method)

    p_col = next(
        (c for c in p_candidates if c in d.columns),
        None,
    )
    q_col = next(
        (c for c in q_candidates if c in d.columns),
        None,
    )

    if p_col is None:
        raise ValueError(
            f"Cannot find p-value column for {method}. "
            f"Columns={list(d.columns)}"
        )

    return id_col, p_col, q_col


def read_result(path, method):
    d = pd.read_csv(path, low_memory=False)

    id_col, p_col, q_col = identify_columns(d, method)

    out = pd.DataFrame(
        {
            "feature_id": d[id_col].map(bare_id),
            "p": pd.to_numeric(
                d[p_col],
                errors="coerce",
            ),
        }
    )

    if q_col is not None:
        out["q"] = pd.to_numeric(
            d[q_col],
            errors="coerce",
        )
    else:
        out["q"] = np.nan

    if out["feature_id"].duplicated().any():
        dup = out.loc[
            out["feature_id"].duplicated(keep=False),
            "feature_id",
        ].head(10).tolist()
        raise ValueError(
            f"{path}: duplicate feature IDs after normalization: {dup}"
        )

    return out


def compare_two(left, right, left_label, right_label):
    if set(left["feature_id"]) != set(right["feature_id"]):
        only_left = (
            set(left["feature_id"])
            - set(right["feature_id"])
        )
        only_right = (
            set(right["feature_id"])
            - set(left["feature_id"])
        )
        raise RuntimeError(
            "Feature universes differ.\n"
            f"Only {left_label}: {len(only_left)}\n"
            f"Only {right_label}: {len(only_right)}"
        )

    d = left.merge(
        right,
        on="feature_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )

    n = len(d)

    d["rank_left"] = deterministic_rank(
        d.rename(columns={"p_left": "p"}),
        "p",
    )

    d["rank_right"] = deterministic_rank(
        d.rename(columns={"p_right": "p"}),
        "p",
    )

    rho = float(
        spearmanr(
            d["rank_left"],
            d["rank_right"],
        ).statistic
    )

    shift = (
        (
            d["rank_left"]
            - d["rank_right"]
        )
        .abs()
        / n
    )

    row = {
        "n_features": n,
        "spearman_rho": rho,
        "median_normalized_rank_shift":
            float(shift.median()),
        "p90_normalized_rank_shift":
            float(shift.quantile(0.90)),
        "median_raw_rank_shift":
            float(
                (
                    d["rank_left"]
                    - d["rank_right"]
                )
                .abs()
                .median()
            ),
        "p90_raw_rank_shift":
            float(
                (
                    d["rank_left"]
                    - d["rank_right"]
                )
                .abs()
                .quantile(0.90)
            ),
    }

    for k in TOPKS:
        if k > n:
            continue

        a = set(
            d.loc[
                d["rank_left"] <= k,
                "feature_id",
            ]
        )

        b = set(
            d.loc[
                d["rank_right"] <= k,
                "feature_id",
            ]
        )

        shared = len(a & b)

        row[f"TOP{k}_shared"] = shared
        row[f"TOP{k}_overlap_fraction"] = shared / k

    sig_left = set(
        d.loc[
            pd.to_numeric(
                d["q_left"],
                errors="coerce",
            ) < 0.05,
            "feature_id",
        ]
    )

    sig_right = set(
        d.loc[
            pd.to_numeric(
                d["q_right"],
                errors="coerce",
            ) < 0.05,
            "feature_id",
        ]
    )

    row.update(
        {
            "fdr05_left": len(sig_left),
            "fdr05_right": len(sig_right),
            "fdr05_shared": len(sig_left & sig_right),
            "fdr05_left_only": len(sig_left - sig_right),
            "fdr05_right_only": len(sig_right - sig_left),
            "fdr05_union": len(sig_left | sig_right),
            "fdr05_jaccard": (
                len(sig_left & sig_right)
                / len(sig_left | sig_right)
                if len(sig_left | sig_right)
                else 1.0
            ),
        }
    )

    return row


def find_adjusted_conventional(task, method):
    area_local = localize_multiomic_result(
        task.A_result
    )

    mean_dir = (
        area_local.parent.parent
        / "mean_based"
    )

    pattern = (
        "*_DESeq2_all_results.csv"
        if method == "DESeq2"
        else "*_limma_all_results.csv"
    )

    hits = sorted(mean_dir.glob(pattern))

    if len(hits) != 1:
        raise RuntimeError(
            f"Task {task.task_id}: expected one adjusted "
            f"{method} result in {mean_dir}; found {hits}"
        )

    return hits[0]


def find_pheno_result(task, method):
    contrast_id = str(task.contrast_id)
    modality = str(task.modality)

    outdir = PHENO_ROOT / contrast_id / modality

    contrast_name = (
        f"Stage11_PHENOONLY__{contrast_id}__{modality}"
    )

    return (
        outdir
        / f"{contrast_name}_{method}_all_results.csv"
    )


def find_area_a(task):
    task_id = int(task.task_id)
    contrast_id = str(task.contrast_id)
    modality = str(task.modality)

    p = (
        OFFLINE
        / "results/A_fully_adjusted"
        / f"task_{task_id:02d}"
        / contrast_id
        / modality
        / "A_all_results.csv"
    )

    if p.exists():
        return p

    # Fallback: use localized frozen model endpoint from original project.
    p = localize_multiomic_result(task.A_result)
    if not p.exists():
        raise FileNotFoundError(p)

    return p


def find_area_b(task):
    contrast_id = str(task.contrast_id)
    modality = str(task.modality)

    base = (
        OFFLINE
        / "results/stage11_covariate_stress"
        / "B_intercept_only"
        / contrast_id
        / modality
    )

    hits = sorted(
        base.glob("*_Regular_AREA_all_results.csv")
    )

    if len(hits) != 1:
        raise RuntimeError(
            f"Task {task.task_id}: expected one AREA B result "
            f"under {base}; found {hits}"
        )

    return hits[0]


def main():
    tasks = pd.read_csv(TASK_FILE, sep="\t")
    tasks = tasks.loc[
        tasks["modality"].isin(["dlpfc", "tmt"])
    ].copy()

    conventional_rows = []
    area_rows = []

    for task in tasks.itertuples(index=False):
        modality = str(task.modality)
        method = (
            "DESeq2"
            if modality == "dlpfc"
            else "limma"
        )

        adjusted = find_adjusted_conventional(
            task,
            method,
        )

        pheno = find_pheno_result(
            task,
            method,
        )

        if not pheno.exists():
            raise FileNotFoundError(
                f"Phenotype-only result missing. Run fits first:\n{pheno}"
            )

        adj_df = read_result(adjusted, method)
        pheno_df = read_result(pheno, method)

        conv = compare_two(
            adj_df,
            pheno_df,
            "adjusted",
            "phenotype_only",
        )

        conv.update(
            {
                "task_id": int(task.task_id),
                "contrast_id": task.contrast_id,
                "modality": modality,
                "method": method,
                "left_model": "fully_adjusted",
                "right_model": "phenotype_only",
                "adjusted_result": str(adjusted),
                "phenotype_only_result": str(pheno),
            }
        )

        conventional_rows.append(conv)

        area_a = read_result(
            find_area_a(task),
            "AREA",
        )
        area_b = read_result(
            find_area_b(task),
            "AREA",
        )

        area = compare_two(
            area_a,
            area_b,
            "A_adjusted",
            "B_intercept_only",
        )

        area.update(
            {
                "task_id": int(task.task_id),
                "contrast_id": task.contrast_id,
                "modality": modality,
                "method": "AREA",
                "left_model": "fully_adjusted",
                "right_model": "intercept_only",
            }
        )

        area_rows.append(area)

    conv = pd.DataFrame(conventional_rows)
    area = pd.DataFrame(area_rows)

    order_cols = [
        "task_id",
        "contrast_id",
        "modality",
        "method",
        "n_features",
        "spearman_rho",
        "median_normalized_rank_shift",
        "p90_normalized_rank_shift",
        "median_raw_rank_shift",
        "p90_raw_rank_shift",
        "TOP250_overlap_fraction",
        "TOP500_overlap_fraction",
        "TOP1000_overlap_fraction",
        "fdr05_left",
        "fdr05_right",
        "fdr05_shared",
        "fdr05_left_only",
        "fdr05_right_only",
        "fdr05_jaccard",
    ]

    conv.to_csv(
        OUTDIR / "CONVENTIONAL_ADJUSTED_VS_PHENOTYPE_ONLY.tsv",
        sep="\t",
        index=False,
    )

    area.to_csv(
        OUTDIR / "AREA_ADJUSTED_VS_INTERCEPT_ONLY_RECOMPUTED.tsv",
        sep="\t",
        index=False,
    )

    # Side-by-side summary.
    metric_cols = [
        "spearman_rho",
        "median_normalized_rank_shift",
        "p90_normalized_rank_shift",
        "TOP250_overlap_fraction",
        "TOP500_overlap_fraction",
        "TOP1000_overlap_fraction",
        "fdr05_left",
        "fdr05_right",
        "fdr05_shared",
        "fdr05_jaccard",
    ]

    a = area[
        ["task_id", "contrast_id", "modality", *metric_cols]
    ].copy()

    c = conv[
        ["task_id", "contrast_id", "modality", "method", *metric_cols]
    ].copy()

    a = a.rename(
        columns={
            x: f"AREA_{x}"
            for x in metric_cols
        }
    )

    c = c.rename(
        columns={
            x: f"CONVENTIONAL_{x}"
            for x in metric_cols
        }
    )

    side = a.merge(
        c,
        on=["task_id", "contrast_id", "modality"],
        how="inner",
        validate="one_to_one",
    )

    side[
        "CONVENTIONAL_minus_AREA_median_normalized_shift"
    ] = (
        side["CONVENTIONAL_median_normalized_rank_shift"]
        - side["AREA_median_normalized_rank_shift"]
    )

    side[
        "CONVENTIONAL_to_AREA_median_shift_ratio"
    ] = (
        side["CONVENTIONAL_median_normalized_rank_shift"]
        / side["AREA_median_normalized_rank_shift"]
    )

    side.to_csv(
        OUTDIR / "AREA_VS_CONVENTIONAL_ADJUSTMENT_SENSITIVITY.tsv",
        sep="\t",
        index=False,
    )

    # Modality summaries.
    rows = []
    for modality, d in side.groupby("modality"):
        rows.append(
            {
                "modality": modality,
                "n_contrasts": len(d),
                "AREA_median_spearman":
                    d["AREA_spearman_rho"].median(),
                "CONVENTIONAL_median_spearman":
                    d["CONVENTIONAL_spearman_rho"].median(),
                "AREA_median_normalized_rank_shift":
                    d[
                        "AREA_median_normalized_rank_shift"
                    ].median(),
                "CONVENTIONAL_median_normalized_rank_shift":
                    d[
                        "CONVENTIONAL_median_normalized_rank_shift"
                    ].median(),
                "AREA_median_TOP500_overlap":
                    d[
                        "AREA_TOP500_overlap_fraction"
                    ].median(),
                "CONVENTIONAL_median_TOP500_overlap":
                    d[
                        "CONVENTIONAL_TOP500_overlap_fraction"
                    ].median(),
            }
        )

    modality_summary = pd.DataFrame(rows)
    modality_summary.to_csv(
        OUTDIR / "MODALITY_ADJUSTMENT_SENSITIVITY_SUMMARY.tsv",
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 130)
    print("CONVENTIONAL ADJUSTED vs PHENOTYPE-ONLY")
    print("=" * 130)

    print(
        conv[
            order_cols
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 130)
    print("AREA vs CONVENTIONAL — SIDE BY SIDE")
    print("=" * 130)

    show = side[
        [
            "modality",
            "contrast_id",
            "method",
            "AREA_spearman_rho",
            "CONVENTIONAL_spearman_rho",
            "AREA_median_normalized_rank_shift",
            "CONVENTIONAL_median_normalized_rank_shift",
            "AREA_TOP500_overlap_fraction",
            "CONVENTIONAL_TOP500_overlap_fraction",
            "CONVENTIONAL_to_AREA_median_shift_ratio",
        ]
    ].copy()

    print(
        show
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 130)
    print("MODALITY SUMMARY")
    print("=" * 130)

    print(
        modality_summary
        .round(4)
        .to_string(index=False)
    )

    print("\nWROTE:")
    for f in [
        "CONVENTIONAL_ADJUSTED_VS_PHENOTYPE_ONLY.tsv",
        "AREA_ADJUSTED_VS_INTERCEPT_ONLY_RECOMPUTED.tsv",
        "AREA_VS_CONVENTIONAL_ADJUSTMENT_SENSITIVITY.tsv",
        "MODALITY_ADJUSTMENT_SENSITIVITY_SUMMARY.tsv",
    ]:
        print(" ", OUTDIR / f)


if __name__ == "__main__":
    main()

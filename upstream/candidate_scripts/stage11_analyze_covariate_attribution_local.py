#!/usr/bin/env python3
"""
Local Stage 11 covariate-attribution analysis.

This script is designed for the self-contained offline bundle:
    ~/Developer/stage11_offline_2026-09-28

It:
1. Reconstructs all 2^k covariate states for each of the 18 Stage 11 analyses.
2. Validates locally recomputed A-vs-B metrics against the frozen primary summary.
3. Computes state-wise distance-to-A, Spearman closure, TOP-k overlap, and FDR behavior.
4. Computes exact Shapley attribution for:
      - PRIMARY: median normalized rank-displacement closure
      - SENSITIVITY: Spearman rank-correlation closure
5. Writes single-covariate and leave-one-out diagnostics.
6. Writes modality/group summaries.

No original expression matrices are required.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TOPKS = (250, 500, 1000)
FDR_THRESHOLD = 0.05

ID_CANDIDATES = (
    "feature_id",
    "gene_id",
    "protein_id_raw",
    "protein_id",
)

P_CANDIDATES = (
    "pvalue",
    "p_value",
    "p",
)

Q_CANDIDATES = (
    "padj",
    "fdr",
    "FDR",
)

STAT_CANDIDATES = (
    "adjusted_Regular_AREA_Z",
    "z",
    "Z",
    "stat",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=".",
        help="Offline bundle root (default: current directory).",
    )
    p.add_argument(
        "--outdir",
        default="results/stage11_covariate_attribution_analysis",
    )
    return p.parse_args()


def detect_col(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not detect {label}. "
        f"Tried {list(candidates)}; columns={list(df.columns)}"
    )


def read_area_result(path):
    path = Path(path)
    d = pd.read_csv(path, low_memory=False)

    id_col = detect_col(d, ID_CANDIDATES, "feature ID")
    p_col = detect_col(d, P_CANDIDATES, "p-value")
    q_col = detect_col(d, Q_CANDIDATES, "FDR")
    stat_col = detect_col(d, STAT_CANDIDATES, "AREA statistic")

    out = pd.DataFrame(
        {
            "feature_id": d[id_col].astype(str).str.strip(),
            "p": pd.to_numeric(d[p_col], errors="coerce"),
            "q": pd.to_numeric(d[q_col], errors="coerce"),
            "stat": pd.to_numeric(d[stat_col], errors="coerce"),
        }
    )

    if out["feature_id"].duplicated().any():
        dup = (
            out.loc[
                out["feature_id"].duplicated(keep=False),
                "feature_id",
            ]
            .head(10)
            .tolist()
        )
        raise ValueError(f"{path}: duplicate feature IDs: {dup}")

    return out.set_index("feature_id", drop=True)


def rank_pvalues(x):
    """
    Match frozen Stage 11 deterministic_rank() exactly.

    Sort by:
      1. p-value ascending
      2. feature_id ascending

    Then assign unique integer ranks 1..N.
    """
    work = pd.DataFrame(
        {
            "feature_id": x.index.astype(str).to_numpy(),
            "p": pd.to_numeric(
                x,
                errors="coerce",
            ).fillna(np.inf).to_numpy(),
        }
    )

    ordered = work.sort_values(
        ["p", "feature_id"],
        ascending=[True, True],
        kind="mergesort",
    ).index

    ranks = pd.Series(
        np.arange(
            1,
            len(x) + 1,
        ),
        index=ordered,
        dtype=int,
    )

    ranks_original_order = ranks.reindex(
        work.index
    ).to_numpy()

    return pd.Series(
        ranks_original_order,
        index=x.index,
    )


def align_to_reference(reference, other, label):
    ref_ids = reference.index
    other_ids = other.index

    if len(reference) != len(other):
        raise ValueError(
            f"{label}: feature counts differ "
            f"({len(reference)} vs {len(other)})"
        )

    if set(ref_ids) != set(other_ids):
        only_ref = list(set(ref_ids) - set(other_ids))[:10]
        only_other = list(set(other_ids) - set(ref_ids))[:10]
        raise ValueError(
            f"{label}: feature universes differ. "
            f"Only reference={only_ref}; only other={only_other}"
        )

    return other.reindex(ref_ids)


def topk_mask(ranks, k):
    # With continuous analytic p-values, ties at TOP-k boundaries should be rare.
    # This definition is consistent with the frozen Stage 11 ranking framework.
    return ranks <= k


def pair_metrics(reference, other):
    """
    Metrics of OTHER relative to REFERENCE (A).
    """
    other = align_to_reference(reference, other, "pair_metrics")

    n = len(reference)
    r_ref = rank_pvalues(reference["p"])
    r_other = rank_pvalues(other["p"])

    delta = (r_other - r_ref).abs()

    rho = spearmanr(
        r_ref.to_numpy(),
        r_other.to_numpy(),
        nan_policy="omit",
    ).statistic

    out = {
        "n_features": n,
        "rho_rank_to_A": float(rho),
        # Frozen Stage 11 normalization is by n_features.
        "median_normalized_rank_shift_to_A": float(
            np.nanmedian(delta.to_numpy()) / n
        ),
        "p90_normalized_rank_shift_to_A": float(
            np.nanpercentile(delta.to_numpy(), 90) / n
        ),
        "n_fdr05": int(
            np.sum(
                pd.to_numeric(other["q"], errors="coerce")
                < FDR_THRESHOLD
            )
        ),
    }

    sig_ref = (
        pd.to_numeric(reference["q"], errors="coerce")
        < FDR_THRESHOLD
    )
    sig_other = (
        pd.to_numeric(other["q"], errors="coerce")
        < FDR_THRESHOLD
    )

    out.update(
        {
            "n_fdr05_shared_with_A": int(np.sum(sig_ref & sig_other)),
            "n_fdr05_state_only": int(np.sum(~sig_ref & sig_other)),
            "n_fdr05_A_only": int(np.sum(sig_ref & ~sig_other)),
        }
    )

    for k in TOPKS:
        a = topk_mask(r_ref, k)
        b = topk_mask(r_other, k)

        overlap = int(np.sum(a & b))
        union = int(np.sum(a | b))

        out[f"TOP{k}_overlap_n_to_A"] = overlap
        out[f"TOP{k}_overlap_fraction_to_A"] = overlap / k
        out[f"TOP{k}_jaccard_to_A"] = (
            overlap / union if union else np.nan
        )

    return out


def state_key(groups):
    groups = tuple(groups)
    if not groups:
        return "EMPTY"
    return "|".join(groups)


def local_a_path(root, base_row):
    return (
        root
        / "results"
        / "A_fully_adjusted"
        / f"task_{int(base_row.task_id):02d}"
        / str(base_row.contrast_id)
        / str(base_row.modality)
        / "A_all_results.csv"
    )


def local_b_path(root, base_row):
    d = (
        root
        / "results"
        / "stage11_covariate_stress"
        / "B_intercept_only"
        / str(base_row.contrast_id)
        / str(base_row.modality)
    )
    hits = sorted(d.glob("*_Regular_AREA_all_results.csv"))
    if len(hits) != 1:
        raise ValueError(
            f"Expected one B result in {d}; found {len(hits)}: {hits}"
        )
    return hits[0]


def local_subset_path(root, subset_row):
    folder = "__".join(
        [
            x.strip()
            for x in str(subset_row.subset_groups).split(",")
            if x.strip()
        ]
    )
    d = (
        root
        / "results"
        / "stage11_covariate_stress"
        / "covariate_attribution"
        / "subset_runs"
        / str(subset_row.contrast_id)
        / str(subset_row.modality)
        / folder
    )
    hits = sorted(d.glob("*_Regular_AREA_all_results.csv"))
    if len(hits) != 1:
        raise ValueError(
            f"Expected one subset result in {d}; "
            f"found {len(hits)}: {hits}"
        )
    return hits[0]


def canonical_group_order(spec, modality):
    rows = spec["covariate_groups"][modality]
    return [row[0] for row in rows]


def reconstruct_states(
    root,
    base_row,
    subset_tasks,
    spec,
):
    modality = str(base_row.modality)
    group_order = canonical_group_order(spec, modality)
    full = tuple(group_order)

    states = {}

    states[tuple()] = local_b_path(root, base_row)
    states[full] = local_a_path(root, base_row)

    sub = subset_tasks.loc[
        subset_tasks["base_task_id"].astype(int)
        == int(base_row.task_id)
    ].copy()

    for row in sub.itertuples(index=False):
        selected_raw = [
            x.strip()
            for x in str(row.subset_groups).split(",")
            if x.strip()
        ]
        selected = tuple(
            g for g in group_order if g in set(selected_raw)
        )

        if not selected or selected == full:
            raise ValueError(
                f"Intermediate task unexpectedly encoded endpoint: {selected}"
            )

        if selected in states:
            raise ValueError(
                f"Duplicate covariate state for task "
                f"{base_row.task_id}: {selected}"
            )

        states[selected] = local_subset_path(root, row)

    expected = 2 ** len(group_order)
    if len(states) != expected:
        missing = []
        for r in range(len(group_order) + 1):
            for comb in itertools.combinations(group_order, r):
                if comb not in states:
                    missing.append(comb)

        raise ValueError(
            f"Task {base_row.task_id} {modality}: "
            f"expected {expected} states, found {len(states)}. "
            f"Missing={missing}"
        )

    return group_order, states


def compare_close(actual, expected, label, tol=5e-10):
    if pd.isna(expected) and pd.isna(actual):
        return 0.0

    if not np.isfinite(actual) or not np.isfinite(expected):
        raise ValueError(
            f"{label}: non-finite comparison "
            f"actual={actual}, expected={expected}"
        )

    diff = abs(float(actual) - float(expected))

    if diff > tol:
        raise ValueError(
            f"{label}: validation failed: "
            f"actual={actual:.15g}, expected={expected:.15g}, "
            f"|diff|={diff:.3g}"
        )

    return diff


def validate_ab(
    base_row,
    primary_summary,
    a_df,
    b_df,
):
    metrics = pair_metrics(a_df, b_df)

    hits = primary_summary.loc[
        (primary_summary["task_id"].astype(int)
         == int(base_row.task_id))
    ]

    if len(hits) != 1:
        raise ValueError(
            f"Could not uniquely resolve frozen primary summary "
            f"for task {base_row.task_id}"
        )

    expected = hits.iloc[0]

    diffs = {}

    diffs["rho"] = compare_close(
        metrics["rho_rank_to_A"],
        expected["rho_rank_A_B"],
        f"task {base_row.task_id} rho_rank_A_B",
        tol=5e-10,
    )

    diffs["median_shift"] = compare_close(
        metrics["median_normalized_rank_shift_to_A"],
        expected["median_normalized_rank_shift_A_B"],
        f"task {base_row.task_id} median rank shift",
        tol=5e-12,
    )

    for k in TOPKS:
        diffs[f"top{k}"] = compare_close(
            metrics[f"TOP{k}_overlap_fraction_to_A"],
            expected[f"A_B_TOP{k}_overlap_fraction"],
            f"task {base_row.task_id} TOP{k} overlap",
            tol=5e-12,
        )

    if "A_B_sig_1" in expected.index:
        actual_a_sig = int(
            np.sum(
                pd.to_numeric(a_df["q"], errors="coerce")
                < FDR_THRESHOLD
            )
        )
        actual_b_sig = metrics["n_fdr05"]

        if actual_a_sig != int(expected["A_B_sig_1"]):
            raise ValueError(
                f"Task {base_row.task_id}: A FDR count "
                f"{actual_a_sig} != frozen {int(expected['A_B_sig_1'])}"
            )

        if actual_b_sig != int(expected["A_B_sig_2"]):
            raise ValueError(
                f"Task {base_row.task_id}: B FDR count "
                f"{actual_b_sig} != frozen {int(expected['A_B_sig_2'])}"
            )

    return metrics, diffs


def exact_shapley(group_order, values):
    """
    Exact Shapley value for a complete power set.

    values maps tuple(canonical selected groups) -> scalar value.
    """
    m = len(group_order)
    phi = {g: 0.0 for g in group_order}
    all_groups = set(group_order)

    for g in group_order:
        others = [x for x in group_order if x != g]

        for r in range(len(others) + 1):
            for comb in itertools.combinations(others, r):
                s_set = set(comb)
                s = tuple(
                    x for x in group_order if x in s_set
                )
                sg_set = s_set | {g}
                sg = tuple(
                    x for x in group_order if x in sg_set
                )

                weight = (
                    math.factorial(len(s))
                    * math.factorial(m - len(s) - 1)
                    / math.factorial(m)
                )

                phi[g] += weight * (
                    float(values[sg]) - float(values[s])
                )

    return phi


def standard_group(group):
    if group in {"sequencing_batch", "batch"}:
        return "batch"
    return group


def summarize_grouped(df, value_col, group_cols):
    rows = []

    for keys, d in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        vals = pd.to_numeric(d[value_col], errors="coerce")
        finite = vals[np.isfinite(vals)]

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": int(len(finite)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "n_negative": int(np.sum(finite < 0)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    s11 = (
        root
        / "results"
        / "stage11_covariate_stress"
    )

    base_tasks = pd.read_csv(
        s11 / "stage11_tasks.tsv",
        sep="\t",
    )

    subset_tasks = pd.read_csv(
        s11
        / "covariate_attribution"
        / "covariate_subset_tasks.tsv",
        sep="\t",
    )

    with open(
        s11
        / "covariate_attribution"
        / "COVARIATE_ATTRIBUTION_SPEC.json"
    ) as fh:
        spec = json.load(fh)

    primary_summary = pd.read_csv(
        s11
        / "primary_comparison"
        / "STAGE11_PRIMARY_COMPARISON_SUMMARY.tsv",
        sep="\t",
    )

    if len(base_tasks) != 18:
        raise ValueError(
            f"Expected 18 base tasks; found {len(base_tasks)}"
        )

    if len(subset_tasks) != 300:
        raise ValueError(
            f"Expected 300 intermediate subset tasks; "
            f"found {len(subset_tasks)}"
        )

    state_rows = []
    validation_rows = []
    shapley_rows = []

    print("=" * 88)
    print("STAGE 11 LOCAL COVARIATE ATTRIBUTION")
    print("=" * 88)
    print(f"Root: {root}")
    print(f"Output: {outdir}")
    print()

    for base_row in base_tasks.itertuples(index=False):
        task_id = int(base_row.task_id)
        contrast_id = str(base_row.contrast_id)
        modality = str(base_row.modality)

        group_order, states = reconstruct_states(
            root,
            base_row,
            subset_tasks,
            spec,
        )

        a_key = tuple(group_order)
        b_key = tuple()

        a_df = read_area_result(states[a_key])
        b_df = read_area_result(states[b_key])

        frozen_b_metrics, diffs = validate_ab(
            base_row,
            primary_summary,
            a_df,
            b_df,
        )

        baseline_distance = (
            frozen_b_metrics[
                "median_normalized_rank_shift_to_A"
            ]
        )
        baseline_rho = frozen_b_metrics["rho_rank_to_A"]

        if baseline_distance <= 0:
            raise ValueError(
                f"Task {task_id}: B-to-A rank distance is "
                f"{baseline_distance}; cannot normalize closure"
            )

        rho_denom = 1.0 - baseline_rho

        if abs(rho_denom) < 1e-12:
            raise ValueError(
                f"Task {task_id}: B-to-A rho too close to 1 "
                f"for Spearman closure"
            )

        validation_rows.append(
            {
                "task_id": task_id,
                "contrast_id": contrast_id,
                "family": base_row.family,
                "modality": modality,
                "n_features": len(a_df),
                "computed_rho_A_B": baseline_rho,
                "frozen_rho_A_B": float(
                    primary_summary.loc[
                        primary_summary["task_id"].astype(int)
                        == task_id,
                        "rho_rank_A_B",
                    ].iloc[0]
                ),
                "computed_median_rank_shift_A_B":
                    baseline_distance,
                "frozen_median_rank_shift_A_B": float(
                    primary_summary.loc[
                        primary_summary["task_id"].astype(int)
                        == task_id,
                        "median_normalized_rank_shift_A_B",
                    ].iloc[0]
                ),
                "max_validation_abs_diff": max(diffs.values()),
                "validation": "PASS",
            }
        )

        value_distance = {}
        value_rho = {}
        state_lookup = {}

        for selected, path in states.items():
            if selected == a_key:
                d = a_df
            elif selected == b_key:
                d = b_df
            else:
                d = read_area_result(path)

            metrics = pair_metrics(a_df, d)

            closure_distance = (
                1.0
                - (
                    metrics[
                        "median_normalized_rank_shift_to_A"
                    ]
                    / baseline_distance
                )
            )

            closure_rho = (
                metrics["rho_rank_to_A"] - baseline_rho
            ) / rho_denom

            # Anchor mathematical endpoints exactly.
            if selected == b_key:
                closure_distance = 0.0
                closure_rho = 0.0
            elif selected == a_key:
                closure_distance = 1.0
                closure_rho = 1.0

            value_distance[selected] = float(closure_distance)
            value_rho[selected] = float(closure_rho)

            row = {
                "task_id": task_id,
                "contrast_id": contrast_id,
                "family": base_row.family,
                "modality": modality,
                "cohort": base_row.cohort,
                "state_key": state_key(selected),
                "n_groups": len(selected),
                "selected_groups": ",".join(selected),
                "is_B_empty": selected == b_key,
                "is_A_full": selected == a_key,
                "result_path_local": str(path),
                "value_rank_distance_closure": float(
                    closure_distance
                ),
                "value_spearman_closure": float(
                    closure_rho
                ),
            }
            row.update(metrics)

            state_rows.append(row)
            state_lookup[selected] = row

        phi_distance = exact_shapley(
            group_order,
            value_distance,
        )
        phi_rho = exact_shapley(
            group_order,
            value_rho,
        )

        sum_distance = sum(phi_distance.values())
        sum_rho = sum(phi_rho.values())

        if abs(sum_distance - 1.0) > 1e-10:
            raise ValueError(
                f"Task {task_id}: rank-distance Shapley "
                f"does not sum to 1: {sum_distance}"
            )

        if abs(sum_rho - 1.0) > 1e-10:
            raise ValueError(
                f"Task {task_id}: Spearman Shapley "
                f"does not sum to 1: {sum_rho}"
            )

        full_set = set(group_order)

        for g in group_order:
            singleton = (g,)
            loo_set = full_set - {g}
            loo = tuple(
                x for x in group_order if x in loo_set
            )

            shapley_rows.append(
                {
                    "task_id": task_id,
                    "contrast_id": contrast_id,
                    "family": base_row.family,
                    "modality": modality,
                    "cohort": base_row.cohort,
                    "group": g,
                    "group_standardized": standard_group(g),
                    "shapley_rank_distance":
                        phi_distance[g],
                    "shapley_spearman":
                        phi_rho[g],
                    "single_covariate_value_rank_distance":
                        value_distance[singleton],
                    "single_covariate_value_spearman":
                        value_rho[singleton],
                    "leave_one_out_value_rank_distance":
                        value_distance[loo],
                    "leave_one_out_value_spearman":
                        value_rho[loo],
                    "conditional_full_marginal_rank_distance":
                        1.0 - value_distance[loo],
                    "conditional_full_marginal_spearman":
                        1.0 - value_rho[loo],
                    "single_distance_to_A":
                        state_lookup[singleton][
                            "median_normalized_rank_shift_to_A"
                        ],
                    "leave_one_out_distance_to_A":
                        state_lookup[loo][
                            "median_normalized_rank_shift_to_A"
                        ],
                    "single_rho_to_A":
                        state_lookup[singleton][
                            "rho_rank_to_A"
                        ],
                    "leave_one_out_rho_to_A":
                        state_lookup[loo][
                            "rho_rank_to_A"
                        ],
                    "single_n_fdr05":
                        state_lookup[singleton]["n_fdr05"],
                    "leave_one_out_n_fdr05":
                        state_lookup[loo]["n_fdr05"],
                    "A_n_fdr05":
                        state_lookup[a_key]["n_fdr05"],
                    "B_n_fdr05":
                        state_lookup[b_key]["n_fdr05"],
                }
            )

        print(
            f"PASS task {task_id:02d} | "
            f"{contrast_id:22s} | "
            f"{modality:8s} | "
            f"{len(states):2d} states | "
            f"A-B rho={baseline_rho:.4f} | "
            f"distance={baseline_distance:.4f}"
        )

    validation = pd.DataFrame(validation_rows)
    states_df = pd.DataFrame(state_rows)
    shapley = pd.DataFrame(shapley_rows)

    expected_states = (
        6 * (2 ** 5)
        + 6 * (2 ** 3)
        + 6 * (2 ** 4)
    )

    if len(states_df) != expected_states:
        raise ValueError(
            f"Expected {expected_states} reconstructed states; "
            f"found {len(states_df)}"
        )

    if len(shapley) != 72:
        raise ValueError(
            f"Expected 72 task-by-covariate Shapley rows; "
            f"found {len(shapley)}"
        )

    validation.to_csv(
        outdir / "AB_FROZEN_VALIDATION.tsv",
        sep="\t",
        index=False,
    )

    states_df.to_csv(
        outdir / "ALL_336_COVARIATE_STATES.tsv",
        sep="\t",
        index=False,
    )

    shapley.to_csv(
        outdir / "EXACT_SHAPLEY_BY_DATASET.tsv",
        sep="\t",
        index=False,
    )

    single = shapley[
        [
            "task_id",
            "contrast_id",
            "family",
            "modality",
            "group",
            "group_standardized",
            "single_covariate_value_rank_distance",
            "single_covariate_value_spearman",
            "single_distance_to_A",
            "single_rho_to_A",
            "single_n_fdr05",
            "A_n_fdr05",
            "B_n_fdr05",
        ]
    ].copy()

    single.to_csv(
        outdir / "SINGLE_COVARIATE_EFFECTS.tsv",
        sep="\t",
        index=False,
    )

    loo = shapley[
        [
            "task_id",
            "contrast_id",
            "family",
            "modality",
            "group",
            "group_standardized",
            "leave_one_out_value_rank_distance",
            "leave_one_out_value_spearman",
            "conditional_full_marginal_rank_distance",
            "conditional_full_marginal_spearman",
            "leave_one_out_distance_to_A",
            "leave_one_out_rho_to_A",
            "leave_one_out_n_fdr05",
            "A_n_fdr05",
            "B_n_fdr05",
        ]
    ].copy()

    loo.to_csv(
        outdir / "LEAVE_ONE_OUT_EFFECTS.tsv",
        sep="\t",
        index=False,
    )

    modality_summary = summarize_grouped(
        shapley,
        "shapley_rank_distance",
        ["modality", "group"],
    ).sort_values(
        ["modality", "median"],
        ascending=[True, False],
    )

    modality_summary.to_csv(
        outdir / "SHAPLEY_MODALITY_SUMMARY.tsv",
        sep="\t",
        index=False,
    )

    standardized_summary = summarize_grouped(
        shapley,
        "shapley_rank_distance",
        ["group_standardized"],
    ).sort_values(
        "median",
        ascending=False,
    )

    standardized_summary.to_csv(
        outdir / "SHAPLEY_STANDARDIZED_GROUP_SUMMARY.tsv",
        sep="\t",
        index=False,
    )

    sensitivity_summary = summarize_grouped(
        shapley,
        "shapley_spearman",
        ["modality", "group"],
    ).sort_values(
        ["modality", "median"],
        ascending=[True, False],
    )

    sensitivity_summary.to_csv(
        outdir / "SHAPLEY_SPEARMAN_SENSITIVITY_SUMMARY.tsv",
        sep="\t",
        index=False,
    )

    check = (
        shapley.groupby(
            ["task_id", "contrast_id", "modality"]
        )[
            [
                "shapley_rank_distance",
                "shapley_spearman",
            ]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "shapley_rank_distance":
                    "sum_shapley_rank_distance",
                "shapley_spearman":
                    "sum_shapley_spearman",
            }
        )
    )

    check.to_csv(
        outdir / "SHAPLEY_SUM_CHECK.tsv",
        sep="\t",
        index=False,
    )

    readme = f"""STAGE 11 COVARIATE ATTRIBUTION ANALYSIS
=======================================

Offline bundle root:
{root}

Validation:
- 18/18 frozen A-vs-B analyses reproduced.
- 336/336 covariate states reconstructed.
- 72 task-by-covariate exact Shapley rows generated.
- Exact Shapley values sum to 1 within every dataset.

PRIMARY VALUE FUNCTION
----------------------
d(S,A) = median absolute p-value-rank displacement between
subset state S and fully adjusted model A, divided by n_features.

v_rank(S) = 1 - d(S,A) / d(B,A)

Therefore:
- B / intercept-only has v_rank = 0.
- A / fully adjusted has v_rank = 1.
- Exact Shapley values partition the observed B-to-A rank transformation.
- Negative Shapley values are allowed and indicate non-monotonic/interacting
  covariate effects; they should not be truncated or treated as errors.

SENSITIVITY VALUE FUNCTION
--------------------------
rho(S,A) = Spearman correlation between p-value rankings.

v_rho(S) = [rho(S,A) - rho(B,A)] / [1 - rho(B,A)]

Again B=0 and A=1.

INTERPRETATION LIMIT
--------------------
These Shapley values are contributions to the observed statistical-model
transformation from intercept-only B to fully adjusted A. They are NOT causal
percentages of biological confounding.

FILES
-----
AB_FROZEN_VALIDATION.tsv
ALL_336_COVARIATE_STATES.tsv
EXACT_SHAPLEY_BY_DATASET.tsv
SINGLE_COVARIATE_EFFECTS.tsv
LEAVE_ONE_OUT_EFFECTS.tsv
SHAPLEY_MODALITY_SUMMARY.tsv
SHAPLEY_STANDARDIZED_GROUP_SUMMARY.tsv
SHAPLEY_SPEARMAN_SENSITIVITY_SUMMARY.tsv
SHAPLEY_SUM_CHECK.tsv
"""

    (outdir / "README_ANALYSIS.txt").write_text(readme)

    print()
    print("=" * 88)
    print("ALL VALIDATION PASSED")
    print("=" * 88)
    print(f"Frozen A-B validations: {len(validation)}/18")
    print(f"Reconstructed states:   {len(states_df)}/336")
    print(f"Shapley rows:           {len(shapley)}/72")
    print(f"Outputs: {outdir}")
    print()
    print("Primary modality summary:")
    print(modality_summary.to_string(index=False))


if __name__ == "__main__":
    main()

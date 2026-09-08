#!/usr/bin/env python3
"""
run_covariate_adjusted_regular_area.py
======================================

Production covariate-adjusted binary AREA for the ROSMAP AD4-vs-NCI1
endpoint comparison, using the SAME locked inference framework as the final
covariate-adjusted Weighted AREA analyses.

Binary phenotype:
    NCI = 0
    AD4 = 1

Locked covariates:
    age_death + sex + rin_numeric + pmi_numeric + sequencing_batch

Inference:
    1. Average ranks for exact expression ties.
    2. Residualize expression ranks against the locked covariate design.
    3. Residualize binary AD/NCI state against the same covariates.
    4. Compute the FWL partial linear-rank statistic.
    5. Use the validated within-sequencing-batch permutation variance.
    6. Two-sided Gaussian p-values.
    7. BH FDR across all tested genes.

Raw effect representation:
    Tie-aware binary AREA ES on the SAME final analysis samples.

Important sample handling:
    The supplied DESeq2 AD4-vs-NCI1 manifest is used as the authoritative
    endpoint cohort. By default, comma-containing ambiguous sequencing-batch
    labels are excluded, matching the locked Weighted AREA pipeline.
    In the current ROSMAP data this is expected to remove sample 492_120515,
    changing the endpoint cohort from 420 (220 AD / 200 NCI) to
    419 (219 AD / 200 NCI).

Sign convention with ascending expression ranks:
    positive adjusted_Regular_AREA_Z:
        AD is associated with LOWER expression
    negative adjusted_Regular_AREA_Z:
        AD is associated with HIGHER expression
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc


CONTINUOUS_COVARIATES = [
    "age_death",
    "rin_numeric",
    "pmi_numeric",
]

BINARY_COVARIATES = ["sex"]
BATCH_COL = "sequencing_batch"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--expression",
        default=(
            "results/preprocessing/"
            "ROSMAP_AREA_normalized_counts_all_samples.csv"
        ),
    )
    p.add_argument(
        "--binary-manifest",
        default=(
            "results/deseq2/AD4_vs_NCI1/"
            "AD4_vs_NCI1_sample_manifest.csv"
        ),
        help=(
            "Authoritative covariate-complete AD4-vs-NCI1 endpoint manifest."
        ),
    )
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--group-col", default="analysis_group")
    p.add_argument("--expression-id-col", default=None)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--rank-chunk-size", type=int, default=2000)

    p.add_argument(
        "--keep-multibatch",
        action="store_true",
        help=(
            "Keep comma-containing sequencing_batch values as literal "
            "categories. Default is to exclude them, matching the locked "
            "Weighted AREA pipeline."
        ),
    )

    p.add_argument(
        "--expected-starting-n",
        type=int,
        default=420,
    )
    p.add_argument(
        "--expected-final-n",
        type=int,
        default=419,
    )
    p.add_argument(
        "--expected-final-ad-n",
        type=int,
        default=219,
    )
    p.add_argument(
        "--expected-final-nci-n",
        type=int,
        default=200,
    )

    p.add_argument(
        "--outdir",
        default=(
            "results/regular_area_covariate_adjusted/"
            "AD4_vs_NCI1"
        ),
    )

    return p.parse_args()


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)

    order = np.argsort(p, kind="mergesort")
    ranked = p[order]

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = q

    return out


def load_expression(path, explicit_id_col):
    d = pd.read_csv(path)

    id_col = explicit_id_col or d.columns[0]

    if id_col not in d.columns:
        raise ValueError(
            f"Expression ID column '{id_col}' not found."
        )

    ids = d[id_col].astype(str).str.strip()

    if ids.duplicated().any():
        raise ValueError("Expression sample IDs are duplicated.")

    genes = [c for c in d.columns if c != id_col]

    xdf = d[genes].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if xdf.isna().any().any():
        bad = xdf.columns[
            xdf.isna().any()
        ].tolist()[:10]

        raise ValueError(
            "Expression contains missing/non-numeric values; "
            f"example genes: {bad}"
        )

    return (
        pd.Index(ids),
        np.asarray(genes, dtype=str),
        xdf.to_numpy(np.float64),
        id_col,
    )


def prepare_binary_manifest(
    manifest,
    sample_col,
    group_col,
    keep_multibatch,
):
    required = [
        sample_col,
        group_col,
        *CONTINUOUS_COVARIATES,
        *BINARY_COVARIATES,
        BATCH_COL,
    ]

    missing = [
        c for c in required
        if c not in manifest.columns
    ]

    if missing:
        raise ValueError(
            f"Binary manifest missing required columns: {missing}"
        )

    d = manifest.copy()
    d[sample_col] = (
        d[sample_col]
        .astype(str)
        .str.strip()
    )

    if d[sample_col].duplicated().any():
        raise ValueError(
            "Binary manifest contains duplicate sample IDs."
        )

    exclusions = []

    valid_group = d[group_col].isin(["AD", "NCI"])

    for _, row in d.loc[~valid_group].iterrows():
        exclusions.append({
            "sample_id": row[sample_col],
            "reason": "invalid_analysis_group",
            "value": row[group_col],
        })

    d = d.loc[valid_group].copy()

    # Numeric coercion for locked covariates.
    for c in CONTINUOUS_COVARIATES + BINARY_COVARIATES:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce",
        )

    if not keep_multibatch:
        ambiguous = (
            d[BATCH_COL]
            .astype(str)
            .str.contains(",", regex=False)
        )

        for _, row in d.loc[ambiguous].iterrows():
            exclusions.append({
                "sample_id": row[sample_col],
                "reason": "ambiguous_multibatch",
                "value": row[BATCH_COL],
            })

        d = d.loc[~ambiguous].copy()

    complete_cols = (
        CONTINUOUS_COVARIATES
        + BINARY_COVARIATES
        + [BATCH_COL]
    )

    missing_mask = d[complete_cols].isna().any(axis=1)

    if missing_mask.any():
        for _, row in d.loc[missing_mask].iterrows():
            missing_here = [
                c for c in complete_cols
                if pd.isna(row[c])
            ]

            exclusions.append({
                "sample_id": row[sample_col],
                "reason": "missing_primary_covariate",
                "value": ";".join(missing_here),
            })

        d = d.loc[~missing_mask].copy()

    d["binary_state"] = np.where(
        d[group_col].eq("AD"),
        1.0,
        0.0,
    )

    exclusion_df = pd.DataFrame(
        exclusions,
        columns=[
            "sample_id",
            "reason",
            "value",
        ],
    )

    return d, exclusion_df


def align_expression(
    expression_ids,
    x_all,
    d,
    sample_col,
):
    pos = pd.Series(
        np.arange(
            len(expression_ids),
            dtype=int,
        ),
        index=expression_ids.astype(str),
    )

    absent = d.loc[
        ~d[sample_col].isin(pos.index),
        sample_col,
    ]

    if len(absent):
        raise ValueError(
            f"{len(absent)} endpoint samples are absent "
            "from expression."
        )

    rows = pos.loc[
        d[sample_col]
    ].to_numpy(dtype=int)

    return x_all[rows, :]


def build_covariate_design(d):
    pieces = [
        np.ones(
            len(d),
            dtype=float,
        )
    ]

    names = ["intercept"]

    for c in CONTINUOUS_COVARIATES:
        x = d[c].to_numpy(dtype=float)

        sd = float(
            np.std(
                x,
                ddof=0,
            )
        )

        if (
            not np.isfinite(sd)
            or sd <= 0
        ):
            raise ValueError(
                f"Covariate '{c}' has invalid SD."
            )

        z = (
            x - np.mean(x)
        ) / sd

        pieces.append(z)
        names.append(c + "_z")

    sex = d["sex"].to_numpy(dtype=float)

    pieces.append(
        sex - np.mean(sex)
    )

    names.append("sex_centered")

    batch_dummies = pd.get_dummies(
        d[BATCH_COL].astype(str),
        prefix="batch",
        drop_first=True,
        dtype=float,
    )

    for c in batch_dummies.columns:
        pieces.append(
            batch_dummies[c].to_numpy(
                dtype=float,
            )
        )
        names.append(c)

    C = np.column_stack(
        pieces
    ).astype(np.float64)

    rank = np.linalg.matrix_rank(C)

    if rank != C.shape[1]:
        raise ValueError(
            "Covariate design is rank deficient: "
            f"rank={rank}, columns={C.shape[1]}"
        )

    condition_number = float(
        np.linalg.cond(C)
    )

    Q, _ = np.linalg.qr(
        C,
        mode="reduced",
    )

    return (
        C,
        Q,
        names,
        condition_number,
    )


def residualize_matrix(Q, Y):
    return (
        Y
        - Q @ (
            Q.T @ Y
        )
    )


def compute_average_rank_matrix(
    x,
    chunk_size,
):
    n_samples, n_genes = x.shape

    mean_rank = (
        n_samples + 1.0
    ) / 2.0

    R = np.empty(
        (
            n_genes,
            n_samples,
        ),
        dtype=np.float32,
    )

    tie_fraction = np.empty(
        n_genes,
        dtype=np.float64,
    )

    max_tie_group = np.empty(
        n_genes,
        dtype=np.int32,
    )

    for start in range(
        0,
        n_genes,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_genes,
        )

        for g in range(
            start,
            end,
        ):
            values = x[:, g]

            ranks = stats.rankdata(
                values,
                method="average",
            ).astype(np.float64)

            R[g, :] = (
                ranks - mean_rank
            ).astype(np.float32)

            _, counts = np.unique(
                values,
                return_counts=True,
            )

            tied = counts[
                counts > 1
            ]

            if len(tied):
                tie_fraction[g] = float(
                    np.sum(tied)
                    / n_samples
                )

                max_tie_group[g] = int(
                    np.max(tied)
                )
            else:
                tie_fraction[g] = 0.0
                max_tie_group[g] = 1

        print(
            f"    ranked genes: "
            f"{end:,}/{n_genes:,}"
        )

    return (
        R,
        tie_fraction,
        max_tie_group,
    )


def residualize_gene_ranks(
    R,
    Q,
    chunk_size,
):
    n_genes = R.shape[0]

    E = np.empty_like(
        R,
        dtype=np.float32,
    )

    for start in range(
        0,
        n_genes,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_genes,
        )

        Y = (
            R[start:end, :]
            .T
            .astype(np.float64)
        )

        residual = residualize_matrix(
            Q,
            Y,
        )

        E[start:end, :] = (
            residual
            .T
            .astype(np.float32)
        )

        print(
            f"    residualized genes: "
            f"{end:,}/{n_genes:,}"
        )

    return E


def build_exchangeability_blocks(d):
    labels = (
        d[BATCH_COL]
        .astype(str)
        .to_numpy()
    )

    blocks = []

    for label in sorted(
        pd.unique(labels).tolist()
    ):
        idx = np.flatnonzero(
            labels == label
        )

        if len(idx) < 2:
            raise ValueError(
                f"Sequencing batch '{label}' "
                "has fewer than 2 samples."
            )

        blocks.append(
            (label, idx)
        )

    return blocks


def blockwise_variance(
    E,
    u,
    blocks,
):
    var = np.zeros(
        E.shape[0],
        dtype=np.float64,
    )

    for _, idx in blocks:
        e_block = E[
            :,
            idx,
        ].astype(
            np.float64,
            copy=False,
        )

        u_block = u[
            idx
        ].astype(
            np.float64,
            copy=False,
        )

        ss_e = np.sum(
            e_block * e_block,
            axis=1,
        )

        ss_u = float(
            np.sum(
                u_block * u_block
            )
        )

        var += (
            ss_e
            * ss_u
            / (
                len(idx) - 1
            )
        )

    if (
        np.any(var <= 0)
        or np.any(
            ~np.isfinite(var)
        )
    ):
        raise RuntimeError(
            "Invalid blockwise variance."
        )

    return var


def raw_tie_aware_binary_area_es(
    R_centered,
    w,
):
    n = R_centered.shape[1]

    wc = (
        w - np.mean(w)
    )

    C = (
        R_centered
        .astype(
            np.float64,
            copy=False,
        )
        @ wc
    )

    total = float(
        np.sum(w)
    )

    if total <= 0:
        raise ValueError(
            "Raw AREA geometry requires "
            "at least one AD sample."
        )

    return (
        -2.0
        * C
        / (
            n * total
        )
    )


def main():
    args = parse_args()

    root = Path(
        args.outdir
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    expression_ids, gene_ids, x_all, expression_id_col = (
        load_expression(
            args.expression,
            args.expression_id_col,
        )
    )

    manifest = pd.read_csv(
        args.binary_manifest
    )

    starting_n = len(manifest)

    if (
        args.expected_starting_n is not None
        and starting_n
        != args.expected_starting_n
    ):
        raise ValueError(
            "Starting endpoint manifest count mismatch: "
            f"found n={starting_n}, "
            f"expected n={args.expected_starting_n}."
        )

    d, exclusions = prepare_binary_manifest(
        manifest=manifest,
        sample_col=args.sample_col,
        group_col=args.group_col,
        keep_multibatch=args.keep_multibatch,
    )

    n = len(d)

    group_counts = (
        d[args.group_col]
        .value_counts()
        .to_dict()
    )

    ad_n = int(
        group_counts.get(
            "AD",
            0,
        )
    )

    nci_n = int(
        group_counts.get(
            "NCI",
            0,
        )
    )

    if (
        args.expected_final_n is not None
        and n
        != args.expected_final_n
    ):
        raise ValueError(
            "Final locked-QC sample count mismatch: "
            f"found n={n}, "
            f"expected n={args.expected_final_n}."
        )

    if (
        args.expected_final_ad_n is not None
        and ad_n
        != args.expected_final_ad_n
    ):
        raise ValueError(
            "Final AD count mismatch: "
            f"found n={ad_n}, "
            f"expected n={args.expected_final_ad_n}."
        )

    if (
        args.expected_final_nci_n is not None
        and nci_n
        != args.expected_final_nci_n
    ):
        raise ValueError(
            "Final NCI count mismatch: "
            f"found n={nci_n}, "
            f"expected n={args.expected_final_nci_n}."
        )

    print(
        "=" * 80
    )
    print(
        "COVARIATE-ADJUSTED REGULAR AREA: "
        "AD4 vs NCI1"
    )
    print(
        "=" * 80
    )

    print(
        f"Expression: {len(expression_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )

    print(
        f"Starting endpoint manifest: "
        f"n={starting_n:,}"
    )

    print(
        f"Final locked-QC cohort: "
        f"n={n:,} "
        f"(AD={ad_n:,}, NCI={nci_n:,})"
    )

    print(
        "Primary covariates: "
        "age_death + sex + rin_numeric + "
        "pmi_numeric + sequencing_batch"
    )

    print(
        "Exchangeability blocks: sequencing_batch"
    )

    print(
        "Tie handling: exact expression ties -> average ranks"
    )

    print(
        "Inference: FWL partial rank association -> "
        "validated blockwise permutation variance -> "
        "two-sided Gaussian p -> BH FDR"
    )

    if len(exclusions):
        print(
            "\nExcluded after endpoint manifest:"
        )

        print(
            exclusions.to_string(
                index=False,
            )
        )

    exclusions.to_csv(
        root
        / "excluded_samples.csv",
        index=False,
    )

    d.to_csv(
        root
        / "matched_sample_manifest_full.csv",
        index=False,
    )

    pd.DataFrame({
        "position": np.arange(
            n,
            dtype=int,
        ),
        "sample_id": d[
            args.sample_col
        ].to_numpy(),
        "analysis_group": d[
            args.group_col
        ].to_numpy(),
        "binary_state": d[
            "binary_state"
        ].to_numpy(
            dtype=float,
        ),
        BATCH_COL: d[
            BATCH_COL
        ].astype(str).to_numpy(),
    }).to_csv(
        root
        / "matched_sample_manifest.csv",
        index=False,
    )

    x = align_expression(
        expression_ids,
        x_all,
        d,
        args.sample_col,
    )

    C, Q, covariate_names, condition_number = (
        build_covariate_design(d)
    )

    w = d[
        "binary_state"
    ].to_numpy(
        dtype=np.float64,
    )

    u = residualize_matrix(
        Q,
        w[:, None],
    )[:, 0]

    phenotype_summary = pd.DataFrame(
        [
            {
                "phenotype": "AD4_vs_NCI1",
                "n_samples": n,
                "n_AD": ad_n,
                "n_NCI": nci_n,
                "raw_mean_binary_state": float(
                    np.mean(w)
                ),
                "raw_sd_binary_state": float(
                    np.std(
                        w,
                        ddof=0,
                    )
                ),
                "residual_mean": float(
                    np.mean(u)
                ),
                "residual_sd": float(
                    np.std(
                        u,
                        ddof=0,
                    )
                ),
                "fraction_raw_variance_remaining_after_covariates": float(
                    np.var(
                        u,
                        ddof=0,
                    )
                    / np.var(
                        w,
                        ddof=0,
                    )
                ),
                "raw_vs_residual_correlation": float(
                    np.corrcoef(
                        w,
                        u,
                    )[0, 1]
                ),
                "covariate_design_condition_number": condition_number,
            }
        ]
    )

    phenotype_summary.to_csv(
        root
        / "phenotype_residualization_summary.csv",
        index=False,
    )

    print(
        "\nComputing tie-aware average expression ranks..."
    )

    R, tie_fraction, max_tie_group = (
        compute_average_rank_matrix(
            x,
            args.rank_chunk_size,
        )
    )

    raw_es = (
        raw_tie_aware_binary_area_es(
            R,
            w,
        )
    )

    print(
        "\nResidualizing expression ranks "
        "against covariates..."
    )

    E = residualize_gene_ranks(
        R,
        Q,
        args.rank_chunk_size,
    )

    blocks = build_exchangeability_blocks(
        d
    )

    var = blockwise_variance(
        E,
        u,
        blocks,
    )

    sd = np.sqrt(var)

    T = (
        E.astype(
            np.float64,
            copy=False,
        )
        @ u
    )

    z = -T / sd

    p = erfc(
        np.abs(z)
        / math.sqrt(2.0)
    )

    q = bh_adjust(p)

    ss_e = np.sum(
        E.astype(
            np.float64,
            copy=False,
        ) ** 2,
        axis=1,
    )

    ss_u = float(
        np.sum(
            u * u
        )
    )

    partial_r = (
        T
        / np.sqrt(
            ss_e * ss_u
        )
    )

    sig = (
        q < args.fdr_threshold
    )

    result = pd.DataFrame({
        "gene_id": gene_ids,
        "phenotype": "AD4_vs_NCI1",
        "n_samples": n,
        "n_AD": ad_n,
        "n_NCI": nci_n,
        "raw_tieaware_Regular_AREA_ES": raw_es,
        "adjusted_partial_rank_statistic": T,
        "adjusted_Regular_AREA_Z": z,
        "adjusted_pvalue": p,
        "adjusted_padj_BH": q,
        "adjusted_partial_rank_correlation": partial_r,
        "fraction_samples_in_ties": tie_fraction,
        "max_tie_group_size": max_tie_group,
        "fdr_significant": sig,
    })

    result[
        "direction"
    ] = np.where(
        z > 0,
        "AD_lower_expression",
        np.where(
            z < 0,
            "AD_higher_expression",
            "zero",
        ),
    )

    result = result.sort_values(
        [
            "adjusted_padj_BH",
            "adjusted_pvalue",
        ],
        kind="mergesort",
    )

    result.to_csv(
        root
        / "results.csv",
        index=False,
    )

    pd.DataFrame({
        "gene_id": gene_ids,
        "fraction_samples_in_ties": tie_fraction,
        "max_tie_group_size": max_tie_group,
    }).to_csv(
        root
        / "gene_tie_diagnostics.csv",
        index=False,
    )

    n_sig = int(
        np.sum(sig)
    )

    n_lower = int(
        np.sum(
            sig
            & (
                z > 0
            )
        )
    )

    n_higher = int(
        np.sum(
            sig
            & (
                z < 0
            )
        )
    )

    summary = pd.DataFrame(
        [
            {
                "phenotype": "AD4_vs_NCI1",
                "starting_n": starting_n,
                "final_n": n,
                "n_AD": ad_n,
                "n_NCI": nci_n,
                "n_genes_tested": len(gene_ids),
                "fdr_threshold": args.fdr_threshold,
                "n_fdr_significant": n_sig,
                "n_sig_AD_lower_expression": n_lower,
                "n_sig_AD_higher_expression": n_higher,
            }
        ]
    )

    summary.to_csv(
        root
        / "run_summary.csv",
        index=False,
    )

    analysis_manifest = {
        "analysis": (
            "covariate_adjusted_regular_AREA_binary_AD4_vs_NCI1"
        ),
        "expression": str(
            Path(
                args.expression
            ).resolve()
        ),
        "binary_manifest": str(
            Path(
                args.binary_manifest
            ).resolve()
        ),
        "expression_id_col": expression_id_col,
        "sample_col": args.sample_col,
        "group_col": args.group_col,
        "starting_n": starting_n,
        "analysis_n": n,
        "n_AD": ad_n,
        "n_NCI": nci_n,
        "primary_covariates": (
            CONTINUOUS_COVARIATES
            + BINARY_COVARIATES
            + [BATCH_COL]
        ),
        "continuous_covariate_handling": (
            "Centered and scaled to unit population SD."
        ),
        "batch_handling": (
            "Categorical dummy variables in covariate model; "
            "inference variance uses sequencing-batch "
            "exchangeability blocks."
        ),
        "ambiguous_multibatch_handling": (
            "Excluded if sequencing_batch contains a comma."
            if not args.keep_multibatch
            else (
                "Kept as a literal categorical level "
                "by user request."
            )
        ),
        "tie_handling": (
            "Exact expression ties receive average ranks "
            "before covariate residualization."
        ),
        "adjusted_statistic": (
            "Frisch-Waugh-Lovell partial association between "
            "covariate-residualized average expression ranks "
            "and covariate-residualized binary AD/NCI state."
        ),
        "gaussian_variance": (
            "Sum across sequencing-batch exchangeability blocks "
            "of SS_expression_residual(block) * "
            "SS_binary_state_residual(block) / (n_block - 1)."
        ),
        "raw_effect_representation": (
            "Tie-aware binary AREA ES on the same final "
            "locked-QC sample set."
        ),
        "sign_convention": (
            "Positive adjusted Regular AREA Z / raw ES = "
            "AD enriched toward lower expression; "
            "negative = AD enriched toward higher expression."
        ),
        "fdr_threshold": args.fdr_threshold,
        "covariate_design_terms": covariate_names,
        "design_condition_number": condition_number,
    }

    with open(
        root
        / "analysis_manifest.json",
        "w",
    ) as f:
        json.dump(
            analysis_manifest,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "OBSERVED RESULT"
    )

    print(
        "=" * 80
    )

    print(
        f"FDR < {args.fdr_threshold:g}: "
        f"{n_sig:,} genes"
    )

    print(
        f"  AD -> lower expression: "
        f"{n_lower:,}"
    )

    print(
        f"  AD -> higher expression: "
        f"{n_higher:,}"
    )

    print(
        "\nWrote:"
    )

    for name in [
        "results.csv",
        "run_summary.csv",
        "matched_sample_manifest.csv",
        "matched_sample_manifest_full.csv",
        "excluded_samples.csv",
        "phenotype_residualization_summary.csv",
        "gene_tie_diagnostics.csv",
        "analysis_manifest.json",
    ]:
        print(
            "  "
            + str(
                root / name
            )
        )


if __name__ == "__main__":
    main()

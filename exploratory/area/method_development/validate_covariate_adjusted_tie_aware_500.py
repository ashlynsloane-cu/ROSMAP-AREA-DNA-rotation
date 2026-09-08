#!/usr/bin/env python3
"""
validate_covariate_adjusted_tie_aware_500.py
============================================

Genome-wide complete-null validation of COVARIATE-ADJUSTED, TIE-AWARE
Weighted AREA inference for ROSMAP.

Primary covariates
------------------
- age_death          continuous
- sex                binary
- rin_numeric        continuous
- pmi_numeric        continuous
- sequencing_batch   categorical

The one ambiguous multi-batch value such as "0, 6, 7" is excluded by default
rather than treated as a legitimate singleton batch category.

Statistical formulation
-----------------------
For each gene g:

1. Rank expression across the analysis sample using average ranks for exact ties.
2. Residualize those ranks against the covariate design C.
3. Residualize the phenotype score w against the same C.
4. Test the partial rank association using

       T_g = e_g' u

   where e_g = M_C r_g and u = M_C w.

This is the Frisch-Waugh-Lovell form of the phenotype coefficient after
adjusting the ranked expression response for C.

Permutation scheme
------------------
We use a Freedman-Lane-style residual permutation score test:

- The reduced-model expression-rank residuals e_g are permuted.
- The SAME participant permutation is used across all genes, preserving
  transcriptome-wide cross-gene dependence.
- Permutations are restricted WITHIN sequencing batch, which is the
  exchangeability block.

Because batch indicators are in C, both e_g and u have zero block means.
For independent uniform permutation within each batch b,

    Var(T_g) = sum_b [
        SS_e(g,b) * SS_u(method,b) / (n_b - 1)
    ]

This gives a gene- and phenotype-specific Gaussian Z approximation.
The script explicitly validates that variance formula against 5,000 within-
batch residual permutations for representative genes.

Complete-null validation
------------------------
For each of 500 outer null replicates:

- independently permute residuals within each sequencing batch;
- use one shared blockwise permutation across the entire transcriptome;
- compute Z, two-sided Gaussian p values, and BH FDR genome-wide.

Under the complete null, every discovery is false, so FDR = P(R > 0).

IMPORTANT
---------
This is a covariate-adjusted inference layer for Weighted AREA. The adjusted
statistic is a partial rank association; it is not itself the original raw
cumulative AREA geometry. For eventual observed analyses, retain the raw
tie-aware AREA ES/curve for geometric effect visualization and use this
adjusted partial-rank Z/P for covariate-aware inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc


AMYLOID_METHODS = {
    "CERAD_equal": "CERAD_equal_weight",
    "CERAD_amyloid_calibrated": "CERAD_amyloid_calibrated_weight",
    "amyloid_continuous": "amyloid_continuous_AREA_weight",
}

TAU_METHODS = {
    "Braak_equal": "Braak_equal_weight",
    "Braak_tangle_calibrated": "Braak_tangle_calibrated_weight",
    "tangle_continuous": "tangle_continuous_AREA_weight",
}

CONTINUOUS_COVARIATES = [
    "age_death",
    "rin_numeric",
    "pmi_numeric",
]

BINARY_COVARIATES = [
    "sex",
]

BATCH_COL = "sequencing_batch"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--metadata",
        default=(
            "results/pathology_geometry_abeta_tau/metadata/"
            "ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv"
        ),
    )
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--expression-id-col", default=None)

    p.add_argument("--outer-permutations", type=int, default=500)
    p.add_argument(
        "--explicit-null-permutations",
        type=int,
        default=5000,
    )
    p.add_argument(
        "--explicit-null-genes",
        type=int,
        default=12,
    )

    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--rank-chunk-size", type=int, default=2000)

    p.add_argument(
        "--keep-multibatch",
        action="store_true",
        help=(
            "Keep comma-containing sequencing_batch values as literal "
            "categories. Default is to exclude them as ambiguous."
        ),
    )

    p.add_argument(
        "--outdir",
        default=(
            "results/ams_area_validation/"
            "covariate_adjusted_tie_aware_complete_null_500"
        ),
    )

    return p.parse_args()


def stable_seed(base_seed, label):
    h = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(h[:8], 16)) % (2**32 - 1)


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


def wilson_interval(successes, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z * z / n

    center = (phat + z * z / (2 * n)) / denom
    half = (
        z
        * math.sqrt(
            phat * (1 - phat) / n
            + z * z / (4 * n * n)
        )
        / denom
    )

    return max(0.0, center - half), min(1.0, center + half)


def load_expression(path, explicit_id_col):
    d = pd.read_csv(path)

    id_col = explicit_id_col or d.columns[0]

    if id_col not in d.columns:
        raise ValueError(f"Expression ID column '{id_col}' not found.")

    sample_ids = d[id_col].astype(str).str.strip()

    if sample_ids.duplicated().any():
        raise ValueError("Expression sample IDs are duplicated.")

    genes = [c for c in d.columns if c != id_col]

    xdf = d[genes].apply(pd.to_numeric, errors="coerce")

    if xdf.isna().any().any():
        bad = xdf.columns[xdf.isna().any()].tolist()[:10]
        raise ValueError(
            "Expression contains missing/non-numeric values. "
            f"Example genes: {bad}"
        )

    return (
        pd.Index(sample_ids),
        np.asarray(genes, dtype=str),
        xdf.to_numpy(dtype=np.float64),
        id_col,
    )


def clean_axis_metadata(
    metadata,
    sample_col,
    complete_flag,
    method_map,
    keep_multibatch,
):
    required = (
        [sample_col, complete_flag]
        + list(method_map.values())
        + CONTINUOUS_COVARIATES
        + BINARY_COVARIATES
        + [BATCH_COL]
    )

    missing_cols = [c for c in required if c not in metadata.columns]

    if missing_cols:
        raise ValueError(
            f"Metadata missing required columns: {missing_cols}"
        )

    d = metadata.loc[
        metadata[complete_flag].astype(bool),
        required,
    ].copy()

    starting_n = len(d)

    # Canonical string sample IDs.
    d[sample_col] = d[sample_col].astype(str).str.strip()

    # Numeric conversion.
    for c in CONTINUOUS_COVARIATES + BINARY_COVARIATES + list(method_map.values()):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d[BATCH_COL] = d[BATCH_COL].astype("string").str.strip()

    exclusion_rows = []

    # Ambiguous multi-batch annotation: exclude rather than invent a category.
    multibatch_mask = d[BATCH_COL].str.contains(",", na=False)

    if not keep_multibatch and multibatch_mask.any():
        for _, row in d.loc[multibatch_mask].iterrows():
            exclusion_rows.append({
                "sample_id": row[sample_col],
                "reason": "ambiguous_multi_batch_annotation",
                "value": str(row[BATCH_COL]),
            })

        d = d.loc[~multibatch_mask].copy()

    # Complete-case filtering for phenotype + primary covariates.
    required_complete = (
        list(method_map.values())
        + CONTINUOUS_COVARIATES
        + BINARY_COVARIATES
        + [BATCH_COL]
    )

    missing_mask = d[required_complete].isna().any(axis=1)

    if missing_mask.any():
        for _, row in d.loc[missing_mask].iterrows():
            missing_here = [
                c
                for c in required_complete
                if pd.isna(row[c])
            ]

            exclusion_rows.append({
                "sample_id": row[sample_col],
                "reason": "missing_primary_covariate_or_phenotype",
                "value": ";".join(missing_here),
            })

        d = d.loc[~missing_mask].copy()

    if d[sample_col].duplicated().any():
        raise ValueError("Duplicate sample IDs after axis filtering.")

    # Ensure binary sex coding is really binary.
    sex_values = sorted(d["sex"].unique().tolist())

    if not set(sex_values).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Unexpected sex coding after filtering: {sex_values}"
        )

    exclusion_df = pd.DataFrame(
        exclusion_rows,
        columns=["sample_id", "reason", "value"],
    )

    return d, exclusion_df, starting_n


def align_expression(expression_ids, x_all, d, sample_col):
    pos = pd.Series(
        np.arange(len(expression_ids), dtype=int),
        index=expression_ids.astype(str),
    )

    absent = d.loc[
        ~d[sample_col].isin(pos.index),
        sample_col,
    ]

    if len(absent):
        raise ValueError(
            f"{len(absent)} metadata samples are absent from expression."
        )

    rows = pos.loc[d[sample_col]].to_numpy(dtype=int)

    return x_all[rows, :]


def build_covariate_design(d):
    """
    Intercept + standardized continuous covariates + sex + batch dummies.
    Uses reduced QR to obtain an orthonormal basis for the covariate column space.
    """
    pieces = []
    names = []

    n = len(d)

    pieces.append(np.ones(n, dtype=float))
    names.append("intercept")

    for c in CONTINUOUS_COVARIATES:
        x = d[c].to_numpy(dtype=float)

        sd = float(np.std(x, ddof=0))

        if sd <= 0 or not np.isfinite(sd):
            raise ValueError(f"Covariate '{c}' has zero/invalid SD.")

        z = (x - np.mean(x)) / sd

        pieces.append(z)
        names.append(c + "_z")

    # Center binary sex for numerical conditioning.
    sex = d["sex"].to_numpy(dtype=float)
    sex_centered = sex - np.mean(sex)

    pieces.append(sex_centered)
    names.append("sex_centered")

    batch_dummies = pd.get_dummies(
        d[BATCH_COL].astype(str),
        prefix="batch",
        drop_first=True,
        dtype=float,
    )

    for c in batch_dummies.columns:
        pieces.append(batch_dummies[c].to_numpy(dtype=float))
        names.append(c)

    C = np.column_stack(pieces).astype(np.float64)

    rank = np.linalg.matrix_rank(C)

    if rank != C.shape[1]:
        raise ValueError(
            "Covariate design is rank deficient: "
            f"rank={rank}, columns={C.shape[1]}"
        )

    condition_number = float(np.linalg.cond(C))

    Q, _ = np.linalg.qr(C, mode="reduced")

    return C, Q, names, condition_number


def residualize_matrix(Q, Y):
    """
    Residualize columns of Y against the covariate column space.
    Y shape: n_samples x n_columns
    """
    return Y - Q @ (Q.T @ Y)


def compute_average_rank_matrix(x, chunk_size):
    """
    Returns genes x samples centered average ranks and tie diagnostics.
    """
    n_samples, n_genes = x.shape
    mean_rank = (n_samples + 1.0) / 2.0

    R = np.empty(
        (n_genes, n_samples),
        dtype=np.float32,
    )

    tie_fraction = np.empty(n_genes, dtype=np.float64)
    max_tie_group = np.empty(n_genes, dtype=np.int32)

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)

        for g in range(start, end):
            values = x[:, g]

            ranks = stats.rankdata(
                values,
                method="average",
            ).astype(np.float64)

            R[g, :] = (ranks - mean_rank).astype(np.float32)

            _, counts = np.unique(
                values,
                return_counts=True,
            )

            tied = counts[counts > 1]

            if len(tied):
                tie_fraction[g] = float(
                    np.sum(tied) / n_samples
                )
                max_tie_group[g] = int(np.max(tied))
            else:
                tie_fraction[g] = 0.0
                max_tie_group[g] = 1

        print(f"    ranked genes: {end:,}/{n_genes:,}")

    return R, tie_fraction, max_tie_group


def residualize_gene_ranks(R, Q, chunk_size):
    """
    R shape genes x samples. Returns E shape genes x samples.

    We residualize R.T against Q in gene chunks to control temporary memory.
    """
    n_genes, n_samples = R.shape

    E = np.empty_like(
        R,
        dtype=np.float32,
    )

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)

        Y = R[start:end, :].T.astype(np.float64)

        residual = residualize_matrix(
            Q,
            Y,
        )

        E[start:end, :] = residual.T.astype(np.float32)

        print(f"    residualized genes: {end:,}/{n_genes:,}")

    return E


def build_exchangeability_blocks(d):
    labels = d[BATCH_COL].astype(str).to_numpy()

    unique = sorted(pd.unique(labels).tolist())

    blocks = []

    for label in unique:
        idx = np.flatnonzero(labels == label)

        if len(idx) < 2:
            raise ValueError(
                f"Sequencing batch '{label}' has fewer than 2 samples "
                "after filtering; cannot permute within block."
            )

        blocks.append((label, idx))

    return blocks


def check_block_means(E, U, blocks):
    max_abs_gene_block_mean = 0.0
    max_abs_pheno_block_mean = 0.0

    for _, idx in blocks:
        gene_means = np.mean(
            E[:, idx],
            axis=1,
            dtype=np.float64,
        )

        pheno_means = np.mean(
            U[idx, :],
            axis=0,
            dtype=np.float64,
        )

        max_abs_gene_block_mean = max(
            max_abs_gene_block_mean,
            float(np.max(np.abs(gene_means))),
        )

        max_abs_pheno_block_mean = max(
            max_abs_pheno_block_mean,
            float(np.max(np.abs(pheno_means))),
        )

    return max_abs_gene_block_mean, max_abs_pheno_block_mean


def blockwise_variance(E, U, blocks):
    """
    Var(T_gj) = sum_b SS_e(g,b) * SS_u(j,b) / (n_b - 1)
    """
    n_genes = E.shape[0]
    n_methods = U.shape[1]

    var = np.zeros(
        (n_genes, n_methods),
        dtype=np.float64,
    )

    block_rows = []

    for label, idx in blocks:
        e_block = E[:, idx].astype(np.float64, copy=False)
        u_block = U[idx, :].astype(np.float64, copy=False)

        ss_e = np.sum(
            e_block * e_block,
            axis=1,
        )

        ss_u = np.sum(
            u_block * u_block,
            axis=0,
        )

        var += (
            ss_e[:, None]
            * ss_u[None, :]
            / (len(idx) - 1)
        )

        block_rows.append({
            "batch": label,
            "n": len(idx),
            **{
                f"phenotype_SS_{j}": float(ss_u[j])
                for j in range(n_methods)
            },
        })

    if np.any(var <= 0) or np.any(~np.isfinite(var)):
        raise RuntimeError(
            "Invalid blockwise permutation variance."
        )

    return var, pd.DataFrame(block_rows)


def make_blockwise_permutation(n, blocks, rng):
    """
    Permute residual positions independently within each sequencing batch.

    Returned p satisfies E_perm = E[:, p].
    """
    p = np.arange(n, dtype=int)

    for _, idx in blocks:
        p[idx] = rng.permutation(idx)

    return p


def choose_across_tie_quantiles(tie_fraction, n_choose):
    tie_fraction = np.asarray(tie_fraction, dtype=float)

    if n_choose >= len(tie_fraction):
        return np.arange(len(tie_fraction), dtype=int)

    targets = np.quantile(
        tie_fraction,
        np.linspace(0, 1, n_choose),
    )

    available = set(range(len(tie_fraction)))
    chosen = []

    for target in targets:
        idx = min(
            available,
            key=lambda i: abs(tie_fraction[i] - target),
        )
        chosen.append(idx)
        available.remove(idx)

    return np.asarray(chosen, dtype=int)


def explicit_null_check(
    e,
    u,
    analytic_sd,
    blocks,
    n_permutations,
    seed,
):
    rng = np.random.default_rng(seed)

    observed = float(np.dot(e, u))

    null = np.empty(
        n_permutations,
        dtype=float,
    )

    for i in range(n_permutations):
        p = make_blockwise_permutation(
            len(e),
            blocks,
            rng,
        )

        null[i] = np.dot(
            e[p],
            u,
        )

    empirical_mean = float(np.mean(null))
    empirical_sd = float(np.std(null, ddof=0))

    analytic_z = float(
        -observed / analytic_sd
    )

    empirical_z = float(
        -(observed - empirical_mean) / empirical_sd
    )

    empirical_p = float(
        (
            np.sum(
                np.abs(null - empirical_mean)
                >= abs(observed - empirical_mean)
            )
            + 1
        )
        / (n_permutations + 1)
    )

    gaussian_p = float(
        erfc(
            abs(analytic_z)
            / math.sqrt(2.0)
        )
    )

    return {
        "observed_statistic": observed,
        "permutation_null_mean": empirical_mean,
        "permutation_null_sd": empirical_sd,
        "analytic_null_sd": float(analytic_sd),
        "sd_ratio_empirical_over_analytic": (
            empirical_sd / analytic_sd
        ),
        "analytic_z": analytic_z,
        "permutation_standardized_z": empirical_z,
        "abs_delta_z": abs(analytic_z - empirical_z),
        "gaussian_p": gaussian_p,
        "empirical_p": empirical_p,
    }


def run_axis(
    axis_name,
    complete_flag,
    method_map,
    expression_ids,
    gene_ids,
    x_all,
    metadata,
    args,
    root,
):
    axis_dir = root / axis_name
    axis_dir.mkdir(parents=True, exist_ok=True)

    sig_dir = axis_dir / "significant_genes"
    sig_dir.mkdir(exist_ok=True)

    method_names = list(method_map)

    d, exclusions, starting_n = clean_axis_metadata(
        metadata,
        args.sample_col,
        complete_flag,
        method_map,
        args.keep_multibatch,
    )

    exclusions.to_csv(
        axis_dir / "excluded_samples.csv",
        index=False,
    )

    x = align_expression(
        expression_ids,
        x_all,
        d,
        args.sample_col,
    )

    n = len(d)
    n_genes = len(gene_ids)
    n_methods = len(method_names)

    print("\n" + "=" * 80)
    print(axis_name.upper())
    print("=" * 80)
    print(f"Three-way starting samples: {starting_n:,}")
    print(f"Adjusted complete cases:    {n:,}")
    print(f"Excluded:                   {starting_n - n:,}")
    print(f"Genes tested:               {n_genes:,}")

    batch_counts = (
        d[BATCH_COL]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    print("\nSequencing-batch counts:")
    print(batch_counts.to_string())

    pd.DataFrame({
        "position": np.arange(n),
        "sample_id": d[args.sample_col].to_numpy(),
        BATCH_COL: d[BATCH_COL].astype(str).to_numpy(),
    }).to_csv(
        axis_dir / "matched_sample_manifest.csv",
        index=False,
    )

    C, Q, covariate_names, condition_number = (
        build_covariate_design(d)
    )

    pd.DataFrame(
        C,
        columns=covariate_names,
    ).assign(
        sample_id=d[args.sample_col].to_numpy()
    ).to_csv(
        axis_dir / "covariate_design_matrix.csv",
        index=False,
    )

    print("\nCovariate design:")
    print(f"  columns:          {C.shape[1]}")
    print(f"  matrix rank:      {np.linalg.matrix_rank(C)}")
    print(f"  condition number: {condition_number:.6g}")
    print("  terms:")
    for name in covariate_names:
        print(f"    {name}")

    W = (
        d[list(method_map.values())]
        .to_numpy(dtype=np.float64)
    )

    U = residualize_matrix(
        Q,
        W,
    )

    phenotype_rows = []

    for j, method in enumerate(method_names):
        raw = W[:, j]
        adj = U[:, j]

        phenotype_rows.append({
            "method": method,
            "raw_mean": float(np.mean(raw)),
            "raw_sd": float(np.std(raw, ddof=0)),
            "residual_mean": float(np.mean(adj)),
            "residual_sd": float(np.std(adj, ddof=0)),
            "raw_vs_residual_correlation": float(
                np.corrcoef(raw, adj)[0, 1]
            ),
            "fraction_raw_variance_remaining_after_covariates": float(
                np.var(adj, ddof=0)
                / np.var(raw, ddof=0)
            ),
        })

    pd.DataFrame(
        phenotype_rows
    ).to_csv(
        axis_dir / "phenotype_residualization_summary.csv",
        index=False,
    )

    print("\nComputing tie-aware expression ranks...")
    R, tie_fraction, max_tie_group = compute_average_rank_matrix(
        x,
        args.rank_chunk_size,
    )

    print("\nResidualizing expression ranks against covariates...")
    E = residualize_gene_ranks(
        R,
        Q,
        args.rank_chunk_size,
    )

    # Free the raw rank matrix before permutation-intensive work.
    del R

    blocks = build_exchangeability_blocks(
        d
    )

    max_gene_block_mean, max_pheno_block_mean = (
        check_block_means(
            E,
            U,
            blocks,
        )
    )

    print("\nExchangeability blocks:")
    for label, idx in blocks:
        print(f"  batch {label}: n={len(idx)}")

    print(
        "  max |gene residual block mean|: "
        f"{max_gene_block_mean:.3e}"
    )
    print(
        "  max |phenotype residual block mean|: "
        f"{max_pheno_block_mean:.3e}"
    )

    var, block_variance_df = blockwise_variance(
        E,
        U,
        blocks,
    )

    sd = np.sqrt(var)

    block_variance_df.to_csv(
        axis_dir / "block_variance_components.csv",
        index=False,
    )

    pd.DataFrame({
        "gene_id": gene_ids,
        "fraction_samples_in_ties": tie_fraction,
        "max_tie_group_size": max_tie_group,
    }).to_csv(
        axis_dir / "gene_tie_diagnostics.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Explicit permutation validation of the blockwise SD formula.
    # ------------------------------------------------------------
    chosen = choose_across_tie_quantiles(
        tie_fraction,
        min(
            args.explicit_null_genes,
            n_genes,
        ),
    )

    explicit_rows = []

    for g in chosen:
        for j, method in enumerate(method_names):
            result = explicit_null_check(
                e=E[g, :].astype(np.float64),
                u=U[:, j].astype(np.float64),
                analytic_sd=sd[g, j],
                blocks=blocks,
                n_permutations=args.explicit_null_permutations,
                seed=stable_seed(
                    args.seed,
                    f"{axis_name}__{method}__{g}",
                ),
            )

            explicit_rows.append({
                "axis": axis_name,
                "method": method,
                "gene_id": gene_ids[g],
                "gene_index": int(g),
                "fraction_samples_in_ties": float(
                    tie_fraction[g]
                ),
                "max_tie_group_size": int(
                    max_tie_group[g]
                ),
                **result,
            })

    explicit_df = pd.DataFrame(
        explicit_rows
    )

    explicit_df.to_csv(
        axis_dir / "explicit_blockwise_null_validation.csv",
        index=False,
    )

    print("\nExplicit blockwise-null validation:")
    print(
        explicit_df.groupby("method").agg(
            mean_sd_ratio=(
                "sd_ratio_empirical_over_analytic",
                "mean",
            ),
            min_sd_ratio=(
                "sd_ratio_empirical_over_analytic",
                "min",
            ),
            max_sd_ratio=(
                "sd_ratio_empirical_over_analytic",
                "max",
            ),
            max_abs_delta_z=(
                "abs_delta_z",
                "max",
            ),
        ).to_string()
    )

    # ------------------------------------------------------------
    # 500 complete-null residual permutations.
    # ------------------------------------------------------------
    rng = np.random.default_rng(
        stable_seed(
            args.seed,
            axis_name + "__outer_blockwise_null",
        )
    )

    replicate_rows = []

    nominal_thresholds = [0.05, 0.01, 0.001]

    nominal_counts = {
        method: {
            thr: 0
            for thr in nominal_thresholds
        }
        for method in method_names
    }

    for outer in range(
        1,
        args.outer_permutations + 1,
    ):
        pidx = make_blockwise_permutation(
            n,
            blocks,
            rng,
        )

        # Freedman-Lane-style residual permutation:
        # same blockwise residual permutation across every gene.
        T = (
            E[:, pidx].astype(np.float64, copy=False)
            @ U
        )

        Z = -T / sd

        P = erfc(
            np.abs(Z)
            / math.sqrt(2.0)
        )

        for j, method in enumerate(method_names):
            pvals = P[:, j]
            qvals = bh_adjust(pvals)

            sig = qvals < args.fdr_threshold
            n_sig = int(np.sum(sig))

            replicate_rows.append({
                "axis": axis_name,
                "outer_replicate": outer,
                "method": method,
                "n_samples": n,
                "n_genes": n_genes,
                "n_fdr_sig": n_sig,
                "fraction_p_lt_0.05": float(
                    np.mean(pvals < 0.05)
                ),
                "fraction_p_lt_0.01": float(
                    np.mean(pvals < 0.01)
                ),
                "fraction_p_lt_0.001": float(
                    np.mean(pvals < 0.001)
                ),
                "min_p": float(np.min(pvals)),
                "min_fdr": float(np.min(qvals)),
                "max_abs_z": float(
                    np.max(np.abs(Z[:, j]))
                ),
            })

            for thr in nominal_thresholds:
                nominal_counts[method][thr] += int(
                    np.sum(pvals < thr)
                )

            if n_sig > 0:
                event = pd.DataFrame({
                    "gene_id": gene_ids[sig],
                    "z": Z[sig, j],
                    "pvalue": pvals[sig],
                    "padj": qvals[sig],
                    "fraction_samples_in_ties": (
                        tie_fraction[sig]
                    ),
                    "max_tie_group_size": (
                        max_tie_group[sig]
                    ),
                }).sort_values(
                    ["padj", "pvalue"],
                    kind="mergesort",
                )

                event.to_csv(
                    sig_dir
                    / f"outer_{outer:04d}__{method}.csv",
                    index=False,
                )

        if (
            outer % 25 == 0
            or outer == args.outer_permutations
        ):
            print(
                f"    outer nulls: "
                f"{outer:,}/{args.outer_permutations:,}"
            )

    replicate_df = pd.DataFrame(
        replicate_rows
    )

    replicate_df.to_csv(
        axis_dir / "replicate_results.csv",
        index=False,
    )

    event_hits = (
        replicate_df.pivot(
            index="outer_replicate",
            columns="method",
            values="n_fdr_sig",
        )
        .reset_index()
    )

    event_hits.columns.name = None

    event_hits.to_csv(
        axis_dir / "event_hit_counts.csv",
        index=False,
    )

    method_rows = []
    burst_rows = []
    marginal_rows = []

    for method in method_names:
        dm = replicate_df.loc[
            replicate_df["method"] == method
        ]

        hits = dm["n_fdr_sig"].to_numpy(dtype=int)

        n_any = int(np.sum(hits > 0))
        prob_any = n_any / args.outer_permutations

        ci_low, ci_high = wilson_interval(
            n_any,
            args.outer_permutations,
        )

        method_rows.append({
            "axis": axis_name,
            "method": method,
            "n_samples": n,
            "outer_permutations": args.outer_permutations,
            "complete_null_fdr_P_R_gt_0": prob_any,
            "n_outer_with_any_BH_hit": n_any,
            "wilson95_low": ci_low,
            "wilson95_high": ci_high,
            "mean_n_fdr_sig": float(np.mean(hits)),
            "median_n_fdr_sig": float(np.median(hits)),
            "p95_n_fdr_sig": float(
                np.quantile(hits, 0.95)
            ),
            "p99_n_fdr_sig": float(
                np.quantile(hits, 0.99)
            ),
            "max_n_fdr_sig": int(np.max(hits)),
            "mean_fraction_p_lt_0.05": float(
                dm["fraction_p_lt_0.05"].mean()
            ),
            "mean_fraction_p_lt_0.01": float(
                dm["fraction_p_lt_0.01"].mean()
            ),
            "mean_fraction_p_lt_0.001": float(
                dm["fraction_p_lt_0.001"].mean()
            ),
        })

        for threshold in [1, 10, 100, 250, 500, 1000]:
            count = int(np.sum(hits >= threshold))

            burst_rows.append({
                "axis": axis_name,
                "method": method,
                "threshold_n_fdr_sig": threshold,
                "n_outer_replicates": count,
                "fraction_outer_replicates": (
                    count / args.outer_permutations
                ),
            })

        total_tests = (
            args.outer_permutations
            * n_genes
        )

        for thr in nominal_thresholds:
            observed = (
                nominal_counts[method][thr]
                / total_tests
            )

            marginal_rows.append({
                "axis": axis_name,
                "method": method,
                "p_threshold": thr,
                "observed_fraction": observed,
                "expected_fraction": thr,
                "observed_minus_expected": (
                    observed - thr
                ),
                "ratio_observed_to_expected": (
                    observed / thr
                ),
            })

    method_summary = pd.DataFrame(
        method_rows
    )

    method_summary.to_csv(
        axis_dir / "method_summary.csv",
        index=False,
    )

    pd.DataFrame(
        burst_rows
    ).to_csv(
        axis_dir / "burst_summary.csv",
        index=False,
    )

    pd.DataFrame(
        marginal_rows
    ).to_csv(
        axis_dir / "marginal_p_calibration.csv",
        index=False,
    )

    print("\nComplete-null summary:")
    print(
        method_summary[
            [
                "method",
                "n_samples",
                "complete_null_fdr_P_R_gt_0",
                "n_outer_with_any_BH_hit",
                "wilson95_low",
                "wilson95_high",
                "mean_n_fdr_sig",
                "max_n_fdr_sig",
                "mean_fraction_p_lt_0.05",
                "mean_fraction_p_lt_0.01",
                "mean_fraction_p_lt_0.001",
            ]
        ].to_string(index=False)
    )

    return {
        "axis": axis_name,
        "starting_n": starting_n,
        "analysis_n": n,
        "n_excluded": starting_n - n,
        "covariate_names": covariate_names,
        "condition_number": condition_number,
        "max_abs_gene_residual_block_mean": (
            max_gene_block_mean
        ),
        "max_abs_phenotype_residual_block_mean": (
            max_pheno_block_mean
        ),
        "method_summary": method_summary,
    }


def main():
    args = parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        args.metadata
    )

    (
        expression_ids,
        gene_ids,
        x_all,
        expression_id_col,
    ) = load_expression(
        args.expression,
        args.expression_id_col,
    )

    print("=" * 80)
    print("COVARIATE-ADJUSTED TIE-AWARE WEIGHTED AREA")
    print("500 COMPLETE-NULL VALIDATION")
    print("=" * 80)
    print(
        f"Expression: {len(expression_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )
    print(
        "Primary covariates: "
        "age_death + sex + rin_numeric + pmi_numeric + sequencing_batch"
    )
    print(
        "Exchangeability blocks: sequencing_batch"
    )
    print(
        f"Outer null permutations: {args.outer_permutations:,}"
    )
    print(
        f"Explicit null permutations/gene: "
        f"{args.explicit_null_permutations:,}"
    )

    results = []

    results.append(
        run_axis(
            axis_name="amyloid_axis",
            complete_flag="CERAD_threeway_complete",
            method_map=AMYLOID_METHODS,
            expression_ids=expression_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            metadata=metadata,
            args=args,
            root=root,
        )
    )

    results.append(
        run_axis(
            axis_name="tau_axis",
            complete_flag="Braak_threeway_complete",
            method_map=TAU_METHODS,
            expression_ids=expression_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            metadata=metadata,
            args=args,
            root=root,
        )
    )

    master = pd.concat(
        [
            r["method_summary"]
            for r in results
        ],
        ignore_index=True,
    )

    master.to_csv(
        root / "MASTER_covariate_adjusted_complete_null_summary.csv",
        index=False,
    )

    manifest = {
        "expression": str(
            Path(args.expression).resolve()
        ),
        "metadata": str(
            Path(args.metadata).resolve()
        ),
        "expression_id_col": expression_id_col,
        "sample_col": args.sample_col,
        "primary_covariates": (
            CONTINUOUS_COVARIATES
            + BINARY_COVARIATES
            + [BATCH_COL]
        ),
        "continuous_covariate_handling": (
            "Centered and scaled to unit population SD."
        ),
        "batch_handling": (
            "Categorical dummy variables in the covariate model; "
            "residual permutations restricted within sequencing batch."
        ),
        "ambiguous_multibatch_handling": (
            "Excluded by default if sequencing_batch contains a comma."
            if not args.keep_multibatch
            else "Kept as literal categorical level by user request."
        ),
        "tie_handling": (
            "Exact expression ties receive average ranks before "
            "covariate residualization."
        ),
        "adjusted_statistic": (
            "Frisch-Waugh-Lovell partial association between "
            "covariate-residualized average expression ranks and "
            "covariate-residualized phenotype score."
        ),
        "permutation_scheme": (
            "Freedman-Lane-style reduced-model expression-rank residual "
            "permutation within sequencing-batch exchangeability blocks. "
            "One shared blockwise permutation is used across the entire "
            "transcriptome."
        ),
        "gaussian_variance": (
            "Sum across exchangeability blocks of "
            "SS_expression_residual(block) * "
            "SS_phenotype_residual(block) / (n_block - 1)."
        ),
        "outer_permutations": args.outer_permutations,
        "explicit_null_permutations": args.explicit_null_permutations,
        "explicit_null_genes": args.explicit_null_genes,
        "seed": args.seed,
        "fdr_threshold": args.fdr_threshold,
        "axes": [
            {
                "axis": r["axis"],
                "starting_n": r["starting_n"],
                "analysis_n": r["analysis_n"],
                "n_excluded": r["n_excluded"],
                "covariate_design_terms": r["covariate_names"],
                "design_condition_number": r["condition_number"],
                "max_abs_gene_residual_block_mean": (
                    r["max_abs_gene_residual_block_mean"]
                ),
                "max_abs_phenotype_residual_block_mean": (
                    r["max_abs_phenotype_residual_block_mean"]
                ),
            }
            for r in results
        ],
    }

    with open(
        root / "validation_manifest.json",
        "w",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("MASTER COVARIATE-ADJUSTED COMPLETE-NULL SUMMARY")
    print("=" * 80)

    print(
        master[
            [
                "axis",
                "method",
                "n_samples",
                "complete_null_fdr_P_R_gt_0",
                "n_outer_with_any_BH_hit",
                "wilson95_low",
                "wilson95_high",
                "mean_n_fdr_sig",
                "max_n_fdr_sig",
                "mean_fraction_p_lt_0.05",
                "mean_fraction_p_lt_0.01",
                "mean_fraction_p_lt_0.001",
            ]
        ].to_string(index=False)
    )

    print("\nWrote:")
    print(
        root
        / "MASTER_covariate_adjusted_complete_null_summary.csv"
    )

    print(
        "\nDo not run observed covariate-adjusted associations until "
        "the explicit blockwise null validation and 500 complete-null "
        "summary have been inspected."
    )


if __name__ == "__main__":
    main()

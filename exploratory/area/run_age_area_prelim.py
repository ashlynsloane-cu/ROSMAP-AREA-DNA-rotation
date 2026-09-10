#!/usr/bin/env python3
"""
Covariate-adjusted Regular AREA + Weighted AREA for preliminary ROSMAP age analysis.

Matched phenotype design
------------------------
Regular AREA: younger <= 75 years vs older >= 85 years.
Weighted AREA: exact age_death retained continuously within the SAME extreme-age cohort.
Weighted age is shifted by the youngest analyzed age so the youngest participant has
weight 0. This shift is recorded in the run summary; multiplying weights by a positive
constant would not change the raw Weighted AREA geometry.

Adjustment
----------
sex + rin_numeric + pmi_numeric + sequencing_batch
(age_death is the phenotype and is intentionally NOT a covariate.)

Inference matches the project's locked covariate-adjusted AREA framework:
average expression ranks -> FWL residualization -> within-batch permutation variance
-> two-sided Gaussian p-values -> BH FDR.
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

CONTINUOUS_COVARIATES = ["rin_numeric", "pmi_numeric"]
BINARY_COVARIATES = ["sex"]
BATCH_COL = "sequencing_batch"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--metadata",
        default="results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv",
    )
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--age-col", default="age_death")
    p.add_argument("--expression-id-col", default=None)
    p.add_argument("--younger-max", type=float, default=75.0)
    p.add_argument("--older-min", type=float, default=85.0)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--rank-chunk-size", type=int, default=2000)
    p.add_argument("--keep-multibatch", action="store_true")
    p.add_argument("--outdir", default="results/age_prelim/area")
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
        raise ValueError(f"Expression ID column '{id_col}' not found.")
    ids = d[id_col].astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError("Expression sample IDs are duplicated.")
    genes = [c for c in d.columns if c != id_col]
    xdf = d[genes].apply(pd.to_numeric, errors="coerce")
    if xdf.isna().any().any():
        bad = xdf.columns[xdf.isna().any()].tolist()[:10]
        raise ValueError(f"Expression contains missing/non-numeric values; examples: {bad}")
    return pd.Index(ids), np.asarray(genes, dtype=str), xdf.to_numpy(np.float64), id_col


def prepare_metadata(meta, args):
    required = [
        args.sample_col,
        args.age_col,
        *CONTINUOUS_COVARIATES,
        *BINARY_COVARIATES,
        BATCH_COL,
    ]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    d = meta.copy()
    d[args.sample_col] = d[args.sample_col].astype(str).str.strip()
    if d[args.sample_col].duplicated().any():
        raise ValueError("Metadata sample IDs are duplicated.")

    for c in [args.age_col, *CONTINUOUS_COVARIATES, *BINARY_COVARIATES]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d[BATCH_COL] = d[BATCH_COL].astype("string").str.strip()

    exclusions = []

    age_missing = d[args.age_col].isna()
    for _, row in d.loc[age_missing].iterrows():
        exclusions.append({"sample_id": row[args.sample_col], "reason": "missing_age", "value": ""})
    d = d.loc[~age_missing].copy()

    extreme = (d[args.age_col] <= args.younger_max) | (d[args.age_col] >= args.older_min)
    for _, row in d.loc[~extreme].iterrows():
        exclusions.append({
            "sample_id": row[args.sample_col],
            "reason": "middle_age_excluded",
            "value": row[args.age_col],
        })
    d = d.loc[extreme].copy()

    if not args.keep_multibatch:
        multibatch = d[BATCH_COL].str.contains(",", na=False)
        for _, row in d.loc[multibatch].iterrows():
            exclusions.append({
                "sample_id": row[args.sample_col],
                "reason": "ambiguous_multibatch",
                "value": row[BATCH_COL],
            })
        d = d.loc[~multibatch].copy()

    complete_cols = CONTINUOUS_COVARIATES + BINARY_COVARIATES + [BATCH_COL]
    miss = d[complete_cols].isna().any(axis=1)
    for _, row in d.loc[miss].iterrows():
        missing_here = [c for c in complete_cols if pd.isna(row[c])]
        exclusions.append({
            "sample_id": row[args.sample_col],
            "reason": "missing_adjustment_covariate",
            "value": ";".join(missing_here),
        })
    d = d.loc[~miss].copy()

    d["analysis_group"] = np.where(d[args.age_col] >= args.older_min, "Older", "Younger")
    d["regular_state"] = np.where(d["analysis_group"].eq("Older"), 1.0, 0.0)

    min_age = float(d[args.age_col].min())
    d["weighted_age"] = d[args.age_col] - min_age
    if np.any(d["weighted_age"] < 0) or float(d["weighted_age"].sum()) <= 0:
        raise ValueError("Invalid continuous age weights.")

    if set(d["analysis_group"].unique()) != {"Older", "Younger"}:
        raise ValueError("Both Older and Younger groups must be represented.")

    exclusion_df = pd.DataFrame(exclusions, columns=["sample_id", "reason", "value"])
    return d, exclusion_df, min_age


def align_expression(expression_ids, x_all, d, sample_col):
    pos = pd.Series(np.arange(len(expression_ids), dtype=int), index=expression_ids.astype(str))
    absent = d.loc[~d[sample_col].isin(pos.index), sample_col]
    if len(absent):
        raise ValueError(f"{len(absent)} analysis samples are absent from expression.")
    rows = pos.loc[d[sample_col]].to_numpy(dtype=int)
    return x_all[rows, :]


def build_covariate_design(d):
    pieces = [np.ones(len(d), dtype=float)]
    names = ["intercept"]

    for c in CONTINUOUS_COVARIATES:
        x = d[c].to_numpy(dtype=float)
        sd = float(np.std(x, ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Covariate '{c}' has invalid SD.")
        pieces.append((x - np.mean(x)) / sd)
        names.append(c + "_z")

    sex = d["sex"].to_numpy(dtype=float)
    pieces.append(sex - np.mean(sex))
    names.append("sex_centered")

    batch_dummies = pd.get_dummies(
        d[BATCH_COL].astype(str), prefix="batch", drop_first=True, dtype=float
    )
    for c in batch_dummies.columns:
        pieces.append(batch_dummies[c].to_numpy(dtype=float))
        names.append(c)

    C = np.column_stack(pieces).astype(np.float64)
    rank = np.linalg.matrix_rank(C)
    if rank != C.shape[1]:
        raise ValueError(f"Covariate design rank deficient: rank={rank}, columns={C.shape[1]}")
    condition_number = float(np.linalg.cond(C))
    Q, _ = np.linalg.qr(C, mode="reduced")
    return C, Q, names, condition_number


def residualize_matrix(Q, Y):
    return Y - Q @ (Q.T @ Y)


def compute_average_rank_matrix(x, chunk_size):
    n_samples, n_genes = x.shape
    mean_rank = (n_samples + 1.0) / 2.0
    R = np.empty((n_genes, n_samples), dtype=np.float32)
    tie_fraction = np.empty(n_genes, dtype=np.float64)
    max_tie_group = np.empty(n_genes, dtype=np.int32)

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        for g in range(start, end):
            values = x[:, g]
            ranks = stats.rankdata(values, method="average").astype(np.float64)
            R[g, :] = (ranks - mean_rank).astype(np.float32)
            _, counts = np.unique(values, return_counts=True)
            tied = counts[counts > 1]
            if len(tied):
                tie_fraction[g] = float(np.sum(tied) / n_samples)
                max_tie_group[g] = int(np.max(tied))
            else:
                tie_fraction[g] = 0.0
                max_tie_group[g] = 1
        print(f"  ranked genes: {end:,}/{n_genes:,}")
    return R, tie_fraction, max_tie_group


def residualize_gene_ranks(R, Q, chunk_size):
    n_genes = R.shape[0]
    E = np.empty_like(R, dtype=np.float32)
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        Y = R[start:end, :].T.astype(np.float64)
        residual = residualize_matrix(Q, Y)
        E[start:end, :] = residual.T.astype(np.float32)
        print(f"  residualized genes: {end:,}/{n_genes:,}")
    return E


def build_exchangeability_blocks(d):
    labels = d[BATCH_COL].astype(str).to_numpy()
    blocks = []
    for label in sorted(pd.unique(labels).tolist()):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            raise ValueError(f"Sequencing batch '{label}' has fewer than 2 samples.")
        blocks.append((label, idx))
    return blocks


def blockwise_variance(E, u, blocks):
    var = np.zeros(E.shape[0], dtype=np.float64)
    for _, idx in blocks:
        e_block = E[:, idx].astype(np.float64, copy=False)
        u_block = u[idx].astype(np.float64, copy=False)
        ss_e = np.sum(e_block * e_block, axis=1)
        ss_u = float(np.sum(u_block * u_block))
        var += ss_e * ss_u / (len(idx) - 1)
    if np.any(var <= 0) or np.any(~np.isfinite(var)):
        raise RuntimeError("Invalid blockwise variance.")
    return var


def raw_tie_aware_area_es(R_centered, w):
    n = R_centered.shape[1]
    wc = w - np.mean(w)
    C = R_centered.astype(np.float64, copy=False) @ wc
    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("Raw AREA geometry requires positive phenotype weight sum.")
    return -2.0 * C / (n * total)


def score_phenotype(name, R, E, Q, blocks, w, gene_ids, tie_fraction, max_tie_group, fdr):
    w = np.asarray(w, dtype=np.float64)
    u = residualize_matrix(Q, w[:, None])[:, 0]
    if float(np.std(u, ddof=0)) <= 0:
        raise ValueError(f"Residualized phenotype '{name}' has zero variance.")
    raw_es = raw_tie_aware_area_es(R, w)
    var = blockwise_variance(E, u, blocks)
    sd = np.sqrt(var)
    T = E.astype(np.float64, copy=False) @ u
    z = -T / sd
    p = erfc(np.abs(z) / math.sqrt(2.0))
    q = bh_adjust(p)
    ss_e = np.sum(E.astype(np.float64, copy=False) ** 2, axis=1)
    ss_u = float(np.sum(u * u))
    partial_r = T / np.sqrt(ss_e * ss_u)
    sig = q < fdr

    out = pd.DataFrame({
        "gene_id": gene_ids,
        "phenotype": name,
        "n_samples": len(w),
        "raw_tieaware_AREA_ES": raw_es,
        "adjusted_partial_rank_statistic": T,
        "adjusted_AREA_Z": z,
        "adjusted_pvalue": p,
        "adjusted_padj_BH": q,
        "adjusted_partial_rank_correlation": partial_r,
        "fraction_samples_in_ties": tie_fraction,
        "max_tie_group_size": max_tie_group,
        "fdr_significant": sig,
    })
    out["direction"] = np.where(
        z > 0,
        "higher_age_lower_expression",
        np.where(z < 0, "higher_age_higher_expression", "zero"),
    )
    return out.sort_values(["adjusted_padj_BH", "adjusted_pvalue"], kind="mergesort")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading expression...")
    expression_ids, gene_ids, x_all, expression_id_col = load_expression(
        args.expression, args.expression_id_col
    )
    print(f"Expression: {len(expression_ids):,} samples x {len(gene_ids):,} genes")

    print("Loading metadata...")
    meta = pd.read_csv(args.metadata)
    d, exclusions, age_zero = prepare_metadata(meta, args)
    x = align_expression(expression_ids, x_all, d, args.sample_col)

    counts = d["analysis_group"].value_counts().to_dict()
    print("\nAGE PRELIMINARY COHORT")
    print(f"  Younger <= {args.younger_max:g}: {counts.get('Younger', 0):,}")
    print(f"  Older   >= {args.older_min:g}: {counts.get('Older', 0):,}")
    print(f"  Total matched: {len(d):,}")
    print(f"  Age range: {d[args.age_col].min():.3f} to {d[args.age_col].max():.3f}")
    print(f"  Weighted-age zero: {age_zero:.3f} years (youngest analyzed participant)")
    print("  Covariates: sex + rin_numeric + pmi_numeric + sequencing_batch")
    print("  age_death is NOT included as a covariate.")

    # Save the exact common cohort before scoring. DESeq2 should use this file.
    d.to_csv(outdir / "age_extremes_metadata_for_deseq2.csv", index=False)
    exclusions.to_csv(outdir / "excluded_samples.csv", index=False)
    d[[
        args.sample_col, args.age_col, "analysis_group", "regular_state", "weighted_age",
        "sex", "rin_numeric", "pmi_numeric", BATCH_COL,
    ]].to_csv(outdir / "matched_sample_manifest.csv", index=False)

    C, Q, covariate_names, condition_number = build_covariate_design(d)
    print(f"  Covariate design columns: {len(covariate_names)}; condition number={condition_number:.3g}")

    print("\nComputing tie-aware average expression ranks...")
    R, tie_fraction, max_tie_group = compute_average_rank_matrix(x, args.rank_chunk_size)
    print("\nResidualizing expression ranks against covariates...")
    E = residualize_gene_ranks(R, Q, args.rank_chunk_size)
    blocks = build_exchangeability_blocks(d)

    regular = score_phenotype(
        "Age_extremes_regular",
        R, E, Q, blocks,
        d["regular_state"].to_numpy(dtype=float),
        gene_ids, tie_fraction, max_tie_group, args.fdr_threshold,
    )
    weighted = score_phenotype(
        "Age_continuous_weighted",
        R, E, Q, blocks,
        d["weighted_age"].to_numpy(dtype=float),
        gene_ids, tie_fraction, max_tie_group, args.fdr_threshold,
    )

    regular.to_csv(outdir / "regular_area_age_results.csv", index=False)
    weighted.to_csv(outdir / "weighted_area_age_results.csv", index=False)

    n_reg = int((regular["adjusted_padj_BH"] < args.fdr_threshold).sum())
    n_wgt = int((weighted["adjusted_padj_BH"] < args.fdr_threshold).sum())
    reg_ids = set(regular.loc[regular["adjusted_padj_BH"] < args.fdr_threshold, "gene_id"])
    wgt_ids = set(weighted.loc[weighted["adjusted_padj_BH"] < args.fdr_threshold, "gene_id"])

    summary = {
        "analysis": "ROSMAP age preliminary Regular vs Weighted AREA",
        "n_samples": int(len(d)),
        "n_younger": int(counts.get("Younger", 0)),
        "n_older": int(counts.get("Older", 0)),
        "younger_max": float(args.younger_max),
        "older_min": float(args.older_min),
        "age_min": float(d[args.age_col].min()),
        "age_max": float(d[args.age_col].max()),
        "weighted_age_zero": float(age_zero),
        "genes_tested": int(len(gene_ids)),
        "covariates": ["sex", "rin_numeric", "pmi_numeric", "sequencing_batch"],
        "regular_fdr_significant": n_reg,
        "weighted_fdr_significant": n_wgt,
        "regular_weighted_intersection": int(len(reg_ids & wgt_ids)),
        "regular_weighted_union": int(len(reg_ids | wgt_ids)),
        "regular_weighted_jaccard": float(len(reg_ids & wgt_ids) / len(reg_ids | wgt_ids)) if (reg_ids | wgt_ids) else None,
        "fdr_threshold": float(args.fdr_threshold),
        "condition_number": condition_number,
        "expression_id_column": expression_id_col,
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nRESULTS")
    print(f"  Regular AREA FDR<{args.fdr_threshold:g}:  {n_reg:,}")
    print(f"  Weighted AREA FDR<{args.fdr_threshold:g}: {n_wgt:,}")
    print(f"  Intersection: {len(reg_ids & wgt_ids):,}")
    print(f"  Outputs: {outdir}")


if __name__ == "__main__":
    main()

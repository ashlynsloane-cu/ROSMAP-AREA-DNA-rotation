#!/usr/bin/env python3
"""Run the locked observed covariate-adjusted, tie-aware Weighted AREA analysis.

Primary inference:
  average expression ranks for exact ties -> residualize ranks and phenotype
  against age_death + sex + rin_numeric + pmi_numeric + sequencing_batch ->
  partial linear-rank statistic -> validated within-batch permutation variance ->
  two-sided Gaussian p -> BH FDR.

Raw effect representation:
  unadjusted tie-aware AREA ES on the same final complete-case samples.

Sign convention (ascending expression ranks):
  positive AREA ES / adjusted AREA Z = higher severity toward LOWER expression.
  negative = higher severity toward HIGHER expression.

No pathway analysis is performed here.
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
CONTINUOUS_COVARIATES = ["age_death", "rin_numeric", "pmi_numeric"]
BINARY_COVARIATES = ["sex"]
BATCH_COL = "sequencing_batch"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--expression", default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv")
    p.add_argument("--metadata", default="results/pathology_geometry_abeta_tau/metadata/ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv")
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--expression-id-col", default=None)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--rank-chunk-size", type=int, default=2000)
    p.add_argument("--keep-multibatch", action="store_true")
    p.add_argument("--outdir", default="results/weighted_area_observed_covariate_adjusted")
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
    ids = d[id_col].astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError("Expression sample IDs are duplicated.")
    genes = [c for c in d.columns if c != id_col]
    xdf = d[genes].apply(pd.to_numeric, errors="coerce")
    if xdf.isna().any().any():
        bad = xdf.columns[xdf.isna().any()].tolist()[:10]
        raise ValueError(f"Expression has missing/non-numeric values; example genes: {bad}")
    return pd.Index(ids), np.asarray(genes, dtype=str), xdf.to_numpy(np.float64), id_col


def clean_axis_metadata(metadata, sample_col, complete_flag, method_map, keep_multibatch):
    required = [sample_col, complete_flag] + list(method_map.values()) + CONTINUOUS_COVARIATES + BINARY_COVARIATES + [BATCH_COL]
    missing = [c for c in required if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    d = metadata.loc[metadata[complete_flag].astype(bool), required].copy()
    starting_n = len(d)
    d[sample_col] = d[sample_col].astype(str).str.strip()

    for c in list(method_map.values()) + CONTINUOUS_COVARIATES + BINARY_COVARIATES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d[BATCH_COL] = d[BATCH_COL].astype("string").str.strip()

    excluded = []
    multi = d[BATCH_COL].str.contains(",", na=False)
    if not keep_multibatch and multi.any():
        for _, row in d.loc[multi].iterrows():
            excluded.append({"sample_id": row[sample_col], "reason": "ambiguous_multi_batch_annotation", "value": str(row[BATCH_COL])})
        d = d.loc[~multi].copy()

    complete_cols = list(method_map.values()) + CONTINUOUS_COVARIATES + BINARY_COVARIATES + [BATCH_COL]
    miss = d[complete_cols].isna().any(axis=1)
    if miss.any():
        for _, row in d.loc[miss].iterrows():
            which = [c for c in complete_cols if pd.isna(row[c])]
            excluded.append({"sample_id": row[sample_col], "reason": "missing_primary_covariate_or_phenotype", "value": ";".join(which)})
        d = d.loc[~miss].copy()

    if d[sample_col].duplicated().any():
        raise ValueError("Duplicate sample IDs after filtering.")

    return d, pd.DataFrame(excluded, columns=["sample_id", "reason", "value"]), starting_n


def align_expression(expression_ids, x_all, d, sample_col):
    pos = pd.Series(np.arange(len(expression_ids), dtype=int), index=expression_ids.astype(str))
    absent = d.loc[~d[sample_col].isin(pos.index), sample_col]
    if len(absent):
        raise ValueError(f"{len(absent)} metadata samples absent from expression.")
    rows = pos.loc[d[sample_col]].to_numpy(int)
    return x_all[rows, :]


def build_covariate_design(d):
    pieces = [np.ones(len(d), dtype=float)]
    names = ["intercept"]
    for c in CONTINUOUS_COVARIATES:
        x = d[c].to_numpy(float)
        sd = float(np.std(x, ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Covariate {c} has invalid SD.")
        pieces.append((x - np.mean(x)) / sd)
        names.append(c + "_z")

    sex = d["sex"].to_numpy(float)
    pieces.append(sex - np.mean(sex))
    names.append("sex_centered")

    batch_dummies = pd.get_dummies(d[BATCH_COL].astype(str), prefix="batch", drop_first=True, dtype=float)
    for c in batch_dummies.columns:
        pieces.append(batch_dummies[c].to_numpy(float))
        names.append(c)

    C = np.column_stack(pieces).astype(np.float64)
    rank = np.linalg.matrix_rank(C)
    if rank != C.shape[1]:
        raise ValueError(f"Covariate design rank deficient: rank={rank}, columns={C.shape[1]}")
    condition_number = float(np.linalg.cond(C))
    Q, _ = np.linalg.qr(C, mode="reduced")
    return C, Q, names, condition_number


def residualize(Q, Y):
    return Y - Q @ (Q.T @ Y)


def compute_average_rank_matrix(x, chunk_size):
    n, g = x.shape
    mean_rank = (n + 1.0) / 2.0
    R = np.empty((g, n), dtype=np.float32)
    tie_fraction = np.empty(g, dtype=np.float64)
    max_tie = np.empty(g, dtype=np.int32)

    for start in range(0, g, chunk_size):
        end = min(start + chunk_size, g)
        for j in range(start, end):
            vals = x[:, j]
            ranks = stats.rankdata(vals, method="average").astype(np.float64)
            R[j, :] = (ranks - mean_rank).astype(np.float32)
            _, counts = np.unique(vals, return_counts=True)
            tied = counts[counts > 1]
            tie_fraction[j] = float(np.sum(tied) / n) if len(tied) else 0.0
            max_tie[j] = int(np.max(tied)) if len(tied) else 1
        print(f"    ranked genes: {end:,}/{g:,}")
    return R, tie_fraction, max_tie


def residualize_gene_ranks(R, Q, chunk_size):
    g, _ = R.shape
    E = np.empty_like(R, dtype=np.float32)
    for start in range(0, g, chunk_size):
        end = min(start + chunk_size, g)
        Y = R[start:end, :].T.astype(np.float64)
        E[start:end, :] = residualize(Q, Y).T.astype(np.float32)
        print(f"    residualized genes: {end:,}/{g:,}")
    return E


def build_blocks(d):
    labels = d[BATCH_COL].astype(str).to_numpy()
    blocks = []
    for label in sorted(pd.unique(labels).tolist()):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            raise ValueError(f"Batch {label} has fewer than 2 samples.")
        blocks.append((label, idx))
    return blocks


def blockwise_variance(E, U, blocks):
    g = E.shape[0]
    m = U.shape[1]
    var = np.zeros((g, m), dtype=np.float64)
    for _, idx in blocks:
        eb = E[:, idx].astype(np.float64, copy=False)
        ub = U[idx, :].astype(np.float64, copy=False)
        ss_e = np.sum(eb * eb, axis=1)
        ss_u = np.sum(ub * ub, axis=0)
        var += ss_e[:, None] * ss_u[None, :] / (len(idx) - 1)
    if np.any(var <= 0) or np.any(~np.isfinite(var)):
        raise RuntimeError("Invalid blockwise variance.")
    return var


def raw_tieaware_es(R, W):
    n = R.shape[1]
    Wc = W - np.mean(W, axis=0, keepdims=True)
    C = R.astype(np.float64, copy=False) @ Wc
    totals = np.sum(W, axis=0)
    if np.any(totals <= 0):
        raise ValueError("Raw AREA geometry requires positive phenotype score totals.")
    return -2.0 * C / (n * totals[None, :])


def partial_rank_corr(E, U):
    Ef = E.astype(np.float64, copy=False)
    num = Ef @ U
    ss_e = np.sum(Ef * Ef, axis=1)
    ss_u = np.sum(U * U, axis=0)
    return num / np.sqrt(ss_e[:, None] * ss_u[None, :])


def build_overlap(gene_ids, method_names, masks):
    out = pd.DataFrame({"gene_id": gene_ids})
    for method in method_names:
        out[f"sig_{method}"] = masks[method]

    labels = []
    for i in range(len(gene_ids)):
        active = [m for m in method_names if masks[m][i]]
        labels.append(" + ".join(active) if active else "none")
    out["membership"] = labels
    return out


def run_axis(axis_name, complete_flag, method_map, expression_ids, gene_ids, x_all, metadata, args, root):
    axis_dir = root / axis_name
    axis_dir.mkdir(parents=True, exist_ok=True)
    method_names = list(method_map)

    d, exclusions, starting_n = clean_axis_metadata(metadata, args.sample_col, complete_flag, method_map, args.keep_multibatch)
    exclusions.to_csv(axis_dir / "excluded_samples.csv", index=False)
    x = align_expression(expression_ids, x_all, d, args.sample_col)
    n = len(d)

    print("\n" + "=" * 80)
    print(axis_name.upper())
    print("=" * 80)
    print(f"Starting three-way samples: {starting_n:,}")
    print(f"Final adjusted samples:     {n:,}")
    print(f"Excluded:                   {starting_n - n:,}")
    print(f"Genes tested:               {len(gene_ids):,}")

    pd.DataFrame({
        "position": np.arange(n),
        "sample_id": d[args.sample_col].to_numpy(),
        BATCH_COL: d[BATCH_COL].astype(str).to_numpy(),
    }).to_csv(axis_dir / "matched_sample_manifest.csv", index=False)

    C, Q, covariate_names, condition_number = build_covariate_design(d)
    W = d[list(method_map.values())].to_numpy(np.float64)
    U = residualize(Q, W)

    pd.DataFrame([
        {
            "method": method,
            "raw_mean": float(np.mean(W[:, j])),
            "raw_sd": float(np.std(W[:, j], ddof=0)),
            "residual_mean": float(np.mean(U[:, j])),
            "residual_sd": float(np.std(U[:, j], ddof=0)),
            "fraction_raw_variance_remaining_after_covariates": float(np.var(U[:, j], ddof=0) / np.var(W[:, j], ddof=0)),
            "raw_vs_residual_correlation": float(np.corrcoef(W[:, j], U[:, j])[0, 1]),
        }
        for j, method in enumerate(method_names)
    ]).to_csv(axis_dir / "phenotype_residualization_summary.csv", index=False)

    print("\nComputing tie-aware ranks...")
    R, tie_fraction, max_tie = compute_average_rank_matrix(x, args.rank_chunk_size)
    raw_es = raw_tieaware_es(R, W)

    print("\nResidualizing expression ranks...")
    E = residualize_gene_ranks(R, Q, args.rank_chunk_size)
    blocks = build_blocks(d)
    sd = np.sqrt(blockwise_variance(E, U, blocks))
    T = E.astype(np.float64, copy=False) @ U
    Z = -T / sd
    P = erfc(np.abs(Z) / math.sqrt(2.0))
    PR = partial_rank_corr(E, U)

    pd.DataFrame({
        "gene_id": gene_ids,
        "fraction_samples_in_ties": tie_fraction,
        "max_tie_group_size": max_tie,
    }).to_csv(axis_dir / "gene_tie_diagnostics.csv", index=False)

    long_parts = []
    masks = {}
    count_rows = []

    for j, method in enumerate(method_names):
        q = bh_adjust(P[:, j])
        sig = q < args.fdr_threshold
        masks[method] = sig

        result = pd.DataFrame({
            "gene_id": gene_ids,
            "axis": axis_name,
            "method": method,
            "n_samples": n,
            "raw_tieaware_AREA_ES": raw_es[:, j],
            "adjusted_partial_rank_statistic": T[:, j],
            "adjusted_AREA_Z": Z[:, j],
            "adjusted_pvalue": P[:, j],
            "adjusted_padj_BH": q,
            "adjusted_partial_rank_correlation": PR[:, j],
            "fraction_samples_in_ties": tie_fraction,
            "max_tie_group_size": max_tie,
            "fdr_significant": sig,
        })
        result["direction"] = np.where(
            result["adjusted_AREA_Z"] > 0,
            "higher_severity_lower_expression",
            np.where(result["adjusted_AREA_Z"] < 0, "higher_severity_higher_expression", "zero"),
        )
        result = result.sort_values(["adjusted_padj_BH", "adjusted_pvalue"], kind="mergesort")
        result.to_csv(axis_dir / f"{method}__results.csv", index=False)
        long_parts.append(result)

        n_sig = int(np.sum(sig))
        n_pos = int(np.sum(sig & (Z[:, j] > 0)))
        n_neg = int(np.sum(sig & (Z[:, j] < 0)))
        count_rows.append({
            "axis": axis_name,
            "method": method,
            "n_samples": n,
            "n_genes_tested": len(gene_ids),
            "fdr_threshold": args.fdr_threshold,
            "n_fdr_significant": n_sig,
            "n_sig_higher_severity_lower_expression": n_pos,
            "n_sig_higher_severity_higher_expression": n_neg,
        })
        print(f"{method}: FDR<{args.fdr_threshold:g} = {n_sig:,} (lower-expression={n_pos:,}; higher-expression={n_neg:,})")

    membership = build_overlap(gene_ids, method_names, masks)
    membership.to_csv(axis_dir / "encoding_overlap_membership.csv", index=False)
    overlap_counts = membership["membership"].value_counts().rename_axis("membership").reset_index(name="n_genes")
    overlap_counts["axis"] = axis_name
    overlap_counts["fdr_threshold"] = args.fdr_threshold
    overlap_counts.to_csv(axis_dir / "encoding_overlap_counts.csv", index=False)

    return {
        "axis": axis_name,
        "starting_n": starting_n,
        "analysis_n": n,
        "n_excluded": starting_n - n,
        "covariate_names": covariate_names,
        "condition_number": condition_number,
        "results_long": pd.concat(long_parts, ignore_index=True),
        "counts": pd.DataFrame(count_rows),
        "overlap": overlap_counts,
    }


def main():
    args = parse_args()
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    expression_ids, gene_ids, x_all, expression_id_col = load_expression(args.expression, args.expression_id_col)

    print("=" * 80)
    print("OBSERVED COVARIATE-ADJUSTED TIE-AWARE WEIGHTED AREA")
    print("=" * 80)
    print(f"Expression: {len(expression_ids):,} samples x {len(gene_ids):,} genes")
    print("Primary covariates: age_death + sex + rin_numeric + pmi_numeric + sequencing_batch")
    print("Primary inference: validated adjusted partial-rank Z/P + BH")
    print("Raw effect: tie-aware AREA ES")

    results = [
        run_axis("amyloid_axis", "CERAD_threeway_complete", AMYLOID_METHODS, expression_ids, gene_ids, x_all, metadata, args, root),
        run_axis("tau_axis", "Braak_threeway_complete", TAU_METHODS, expression_ids, gene_ids, x_all, metadata, args, root),
    ]

    master_long = pd.concat([r["results_long"] for r in results], ignore_index=True)
    master_counts = pd.concat([r["counts"] for r in results], ignore_index=True)
    master_overlap = pd.concat([r["overlap"] for r in results], ignore_index=True)

    master_long.to_csv(root / "MASTER_observed_weighted_area_results_long.csv", index=False)
    master_counts.to_csv(root / "MASTER_significant_gene_counts.csv", index=False)
    master_overlap.to_csv(root / "MASTER_encoding_overlap_counts.csv", index=False)

    manifest = {
        "expression": str(Path(args.expression).resolve()),
        "metadata": str(Path(args.metadata).resolve()),
        "expression_id_col": expression_id_col,
        "sample_col": args.sample_col,
        "fdr_threshold": args.fdr_threshold,
        "primary_covariates": CONTINUOUS_COVARIATES + BINARY_COVARIATES + [BATCH_COL],
        "tie_handling": "Exact expression ties receive average ranks.",
        "primary_inference": "Covariate-adjusted partial linear-rank statistic with validated within-sequencing-batch permutation variance; two-sided Gaussian p-values; BH FDR.",
        "raw_effect_representation": "Unadjusted tie-aware AREA ES on the same final complete-case sample set.",
        "sign_convention": "Positive raw ES / adjusted AREA Z = higher severity enriched toward lower expression; negative = toward higher expression.",
        "ambiguous_multibatch_handling": "Excluded by default if sequencing_batch contains a comma." if not args.keep_multibatch else "Kept as literal category by user request.",
        "axes": [
            {
                "axis": r["axis"],
                "starting_n": r["starting_n"],
                "analysis_n": r["analysis_n"],
                "n_excluded": r["n_excluded"],
                "covariate_design_terms": r["covariate_names"],
                "design_condition_number": r["condition_number"],
            }
            for r in results
        ],
    }
    with open(root / "analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("MASTER SIGNIFICANT GENE COUNTS")
    print("=" * 80)
    print(master_counts.to_string(index=False))
    print("\nEncoding overlap counts:")
    print(master_overlap.to_string(index=False))
    print("\nWrote:")
    print(root / "MASTER_observed_weighted_area_results_long.csv")
    print(root / "MASTER_significant_gene_counts.csv")
    print(root / "MASTER_encoding_overlap_counts.csv")
    print("\nNo pathway analysis was run. Inspect gene-level results first.")


if __name__ == "__main__":
    main()

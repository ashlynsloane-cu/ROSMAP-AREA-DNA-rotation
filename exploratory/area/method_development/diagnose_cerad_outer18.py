#!/usr/bin/env python3
"""
diagnose_cerad_outer18.py
=========================

Focused diagnostic for the CERAD_burden paired complete-null event in outer
replicate 18.

Questions addressed:
1. Does the event survive when Gaussian inference exactly matches run_ams_area.py?
   (population SD: np.std(null), ddof=0)
2. At the BH boundary, is the Gaussian tail materially more liberal than the
   empirical 10,000-permutation null?
3. Does outer replicate 18 align unusually strongly with a major transcriptomic
   principal component relative to the other 19 paired null shuffles?

This script does NOT perform pathway enrichment and does NOT select a weighting
scheme.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


ENCODINGS = ("current", "sqrt", "square", "exp3", "cube")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--area-script",
        default="exploratory/area/run_ams_area.py",
    )
    p.add_argument(
        "--screen-dir",
        default=(
            "results/ams_area_validation/weighted_A_to_D/"
            "paired_screen/CERAD_burden"
        ),
    )
    p.add_argument("--outer-replicate", type=int, default=18)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--top-variable-genes", type=int, default=5000)
    p.add_argument("--n-pcs", type=int, default=20)
    p.add_argument(
        "--outdir",
        default=(
            "results/ams_area_validation/weighted_A_to_D/"
            "CERAD_outer18_diagnostic"
        ),
    )
    return p.parse_args()


def import_score(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("run_ams_area_current", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_weighted_enrichment_score


def transform(v, encoding):
    v = np.asarray(v, dtype=float)
    if encoding == "current":
        return v.copy()

    lo, hi = np.min(v), np.max(v)
    x = (v - lo) / (hi - lo)

    if encoding == "sqrt":
        return np.sqrt(x)
    if encoding == "square":
        return x ** 2
    if encoding == "exp3":
        return np.expm1(3.0 * x) / np.expm1(3.0)
    if encoding == "cube":
        return x ** 3
    raise ValueError(encoding)


def load_expression(path):
    d = pd.read_csv(path)
    id_col = d.columns[0]
    ids = d[id_col].astype(str).str.strip()
    xdf = d.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if xdf.isna().any().any():
        raise ValueError("Expression matrix contains NA/non-numeric values.")
    return pd.Index(ids), np.asarray(d.columns[1:], dtype=str), xdf.to_numpy(float)


def rank_orders(x):
    # Exact run_ams_area.py convention: ascending expression, stable ties.
    return np.argsort(x, axis=0, kind="mergesort").T


def score_all(score_fn, orders, weights):
    out = np.empty(orders.shape[0], dtype=float)
    for j in range(orders.shape[0]):
        out[j] = float(score_fn(weights[orders[j]]))
    return out


def normal_two_sided_p(z):
    z = np.asarray(z, dtype=float)
    return np.fromiter(
        (math.erfc(abs(float(v)) / math.sqrt(2.0)) for v in z),
        dtype=float,
        count=len(z),
    )


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def empirical_two_sided_p(es, null):
    """
    Diagnostic only. With 10k null draws, minimum possible p ~= 1e-4.
    """
    mu = float(np.mean(null))
    null_dev = np.sort(np.abs(null - mu))
    obs_dev = np.abs(es - mu)
    first = np.searchsorted(null_dev, obs_dev, side="left")
    nge = len(null_dev) - first
    return (nge + 1.0) / (len(null_dev) + 1.0)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_pcs(x, n_top, n_pcs):
    """
    Diagnostic PCA on log1p normalized counts.
    Select top-variable genes, center genes, then SVD.
    """
    logx = np.log1p(x)
    var = np.var(logx, axis=0)
    n_top = min(n_top, logx.shape[1])
    keep = np.argsort(var)[-n_top:]
    m = logx[:, keep]
    m = m - np.mean(m, axis=0, keepdims=True)

    u, s, vt = np.linalg.svd(m, full_matrices=False)
    n_pcs = min(n_pcs, u.shape[1])
    scores = u[:, :n_pcs] * s[:n_pcs]

    total_ss = np.sum(s ** 2)
    var_exp = (s[:n_pcs] ** 2) / total_ss

    return scores, var_exp, keep


def main():
    a = parse_args()
    screen_dir = Path(a.screen_dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    score_fn = import_score(a.area_script)
    expr_ids, gene_ids, x_all = load_expression(a.expression)

    perm_file = screen_dir / "ALL_outer_exact_permutation_map.csv"
    perm = pd.read_csv(perm_file)

    required = {
        "outer_replicate",
        "target_sample_id",
        "source_sample_id",
        "original_current_weight",
        "shuffled_current_weight",
    }
    missing = required - set(perm.columns)
    if missing:
        raise ValueError(f"Permutation map missing columns: {sorted(missing)}")

    # Align expression once to the target sample order used in the permutation map.
    first_rep = perm["outer_replicate"].min()
    template = perm[perm["outer_replicate"] == first_rep].copy()
    target_ids = template["target_sample_id"].astype(str).to_numpy()

    pos = pd.Series(
        np.arange(len(expr_ids), dtype=int),
        index=expr_ids.astype(str),
    )
    if not pd.Series(target_ids).isin(pos.index).all():
        raise ValueError("Some CERAD target samples are missing from expression.")

    x = x_all[pos.loc[target_ids].to_numpy(int), :]
    orders = rank_orders(x)

    # ------------------------------------------------------------------
    # 1. Reconstruct outer event and compare Gaussian vs empirical tails.
    # ------------------------------------------------------------------
    event = perm[perm["outer_replicate"] == a.outer_replicate].copy()
    event_ids = event["target_sample_id"].astype(str).to_numpy()
    if not np.array_equal(event_ids, target_ids):
        raise ValueError("Target sample order differs across outer replicates.")

    current_shuffled = event["shuffled_current_weight"].to_numpy(float)

    boundary_rows = []
    gene_frames = []

    print("=" * 80)
    print(f"CERAD OUTER {a.outer_replicate}: EXACT-RUNNER GAUSSIAN + EMPIRICAL DIAGNOSTIC")
    print("=" * 80)

    for enc in ENCODINGS:
        null_file = screen_dir / f"{enc}_paired_method_null.csv"
        null = pd.read_csv(null_file)["null_es"].to_numpy(float)

        weights = transform(current_shuffled, enc)
        es = score_all(score_fn, orders, weights)

        # Exact run_ams_area.py inference: population SD, ddof=0.
        mu = float(np.mean(null))
        sd = float(np.std(null))
        z = (es - mu) / sd
        p_gauss = normal_two_sided_p(z)
        q_gauss = bh(p_gauss)

        p_emp = empirical_two_sided_p(es, null)

        sig = q_gauss < a.fdr_threshold
        n_sig = int(np.sum(sig))

        if n_sig > 0:
            sig_idx = np.where(sig)[0]
            # Largest Gaussian p among BH discoveries = practical BH boundary.
            boundary_i = sig_idx[np.argmax(p_gauss[sig_idx])]
            boundary_p = float(p_gauss[boundary_i])
            boundary_z = float(abs(z[boundary_i]))

            null_z_abs = np.abs((null - mu) / sd)
            empirical_tail_at_boundary = float(
                (np.sum(null_z_abs >= boundary_z) + 1.0) / (len(null) + 1.0)
            )

            emp_sig = p_emp[sig]
            gauss_sig = p_gauss[sig]
            ratio = emp_sig / np.maximum(gauss_sig, np.finfo(float).tiny)

            # Among Gaussian hits, compare empirical p to the same gene's
            # rank-specific BH critical value. This is diagnostic only.
            ordered_sig = sig_idx[np.argsort(p_gauss[sig_idx])]
            ranks = np.arange(1, len(ordered_sig) + 1)
            bh_critical = a.fdr_threshold * ranks / len(gene_ids)
            emp_below_rank_critical = int(
                np.sum(p_emp[ordered_sig] <= bh_critical)
            )

            med_emp = float(np.median(emp_sig))
            med_gauss = float(np.median(gauss_sig))
            med_ratio = float(np.median(ratio))
            frac_emp_gt_gauss = float(np.mean(emp_sig > gauss_sig))
        else:
            boundary_p = np.nan
            boundary_z = np.nan
            empirical_tail_at_boundary = np.nan
            emp_below_rank_critical = 0
            med_emp = np.nan
            med_gauss = np.nan
            med_ratio = np.nan
            frac_emp_gt_gauss = np.nan

        boundary_rows.append(
            {
                "encoding": enc,
                "n_gaussian_fdr_hits": n_sig,
                "gaussian_bh_boundary_p": boundary_p,
                "gaussian_bh_boundary_abs_z": boundary_z,
                "empirical_null_tail_at_gaussian_boundary": empirical_tail_at_boundary,
                "empirical_to_gaussian_boundary_ratio": (
                    empirical_tail_at_boundary / boundary_p
                    if n_sig > 0 and boundary_p > 0
                    else np.nan
                ),
                "median_gaussian_p_among_gaussian_hits": med_gauss,
                "median_empirical_p_among_gaussian_hits": med_emp,
                "median_empirical_to_gaussian_p_ratio_among_hits": med_ratio,
                "fraction_gaussian_hits_with_empirical_p_gt_gaussian_p": frac_emp_gt_gauss,
                "gaussian_hits_empirical_p_below_rank_specific_BH_critical": (
                    emp_below_rank_critical
                ),
                "empirical_p_floor": 1.0 / (len(null) + 1.0),
            }
        )

        gene_frames.append(
            pd.DataFrame(
                {
                    "gene_id": gene_ids,
                    "encoding": enc,
                    "outer_replicate": a.outer_replicate,
                    "ES": es,
                    "Z_gaussian_exact_runner": z,
                    "P_gaussian_exact_runner": p_gauss,
                    "FDR_gaussian_exact_runner": q_gauss,
                    "P_empirical_10k_diagnostic": p_emp,
                    "gaussian_fdr_significant": sig,
                }
            )
        )

        print(
            f"{enc:8s}: Gaussian FDR hits={n_sig:,}; "
            f"boundary p={boundary_p if np.isfinite(boundary_p) else float('nan'):.6g}; "
            f"empirical tail at boundary="
            f"{empirical_tail_at_boundary if np.isfinite(empirical_tail_at_boundary) else float('nan'):.6g}"
        )

    boundary = pd.DataFrame(boundary_rows)
    boundary.to_csv(outdir / "outer18_gaussian_vs_empirical_tail_summary.csv", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(
        outdir / "outer18_gene_level_gaussian_empirical.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. PCA alignment: is outer18 unusually aligned with expression PCs?
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PCA ALIGNMENT DIAGNOSTIC")
    print("=" * 80)
    print(
        f"Building PCA from log1p normalized counts using top "
        f"{a.top_variable_genes:,} variable genes..."
    )

    pc_scores, var_exp, keep = build_pcs(
        x,
        a.top_variable_genes,
        a.n_pcs,
    )

    pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(pc_scores.shape[1])],
            "variance_explained": var_exp,
        }
    ).to_csv(outdir / "pca_variance_explained.csv", index=False)

    pc_rows = []

    for rep, g in perm.groupby("outer_replicate", sort=True):
        g = g.copy()
        ids = g["target_sample_id"].astype(str).to_numpy()
        if not np.array_equal(ids, target_ids):
            raise ValueError(f"Target sample order mismatch in outer {rep}")

        shuffled_current = g["shuffled_current_weight"].to_numpy(float)

        for enc in ENCODINGS:
            w = transform(shuffled_current, enc)
            rs = np.array(
                [pearson(w, pc_scores[:, j]) for j in range(pc_scores.shape[1])],
                dtype=float,
            )
            best_j = int(np.nanargmax(np.abs(rs)))

            pc_rows.append(
                {
                    "outer_replicate": int(rep),
                    "encoding": enc,
                    "max_abs_pc_correlation": float(abs(rs[best_j])),
                    "best_pc": best_j + 1,
                    "signed_best_pc_correlation": float(rs[best_j]),
                    **{
                        f"PC{j+1}_r": float(rs[j])
                        for j in range(len(rs))
                    },
                }
            )

    pc_df = pd.DataFrame(pc_rows)

    # Rank each replicate within encoding by strongest PC alignment.
    pc_df["max_abs_pc_rank_within_encoding"] = (
        pc_df.groupby("encoding")["max_abs_pc_correlation"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    pc_df.to_csv(outdir / "all_outer_pc_alignment.csv", index=False)

    event_pc = pc_df[
        pc_df["outer_replicate"] == a.outer_replicate
    ].copy()
    event_pc.to_csv(outdir / "outer18_pc_alignment.csv", index=False)

    print("\nOuter-event PC alignment:")
    print(
        event_pc[
            [
                "encoding",
                "max_abs_pc_correlation",
                "best_pc",
                "signed_best_pc_correlation",
                "max_abs_pc_rank_within_encoding",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(f"Wrote: {outdir / 'outer18_gaussian_vs_empirical_tail_summary.csv'}")
    print(f"Wrote: {outdir / 'outer18_pc_alignment.csv'}")
    print(f"Wrote: {outdir / 'all_outer_pc_alignment.csv'}")
    print(f"Wrote: {outdir / 'outer18_gene_level_gaussian_empirical.csv'}")
    print("\nSTOP HERE. No pathway analysis or method selection.")


if __name__ == "__main__":
    main()

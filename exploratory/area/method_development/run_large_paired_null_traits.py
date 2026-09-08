#!/usr/bin/env python3
"""
run_large_paired_null_traits.py
===============================

Fresh large paired complete-null validation for Weighted AREA traits.

Default traits
--------------
Cognitive_stage
Braak_stage

Primary question
----------------
For each trait, how often do random phenotype permutations produce any BH
discovery, and how often do they produce large burst events?

Design
------
* Default: 500 fresh outer phenotype permutations PER TRAIT.
* SAME permutation indices are applied to all five pre-specified encodings
  within each trait:
      current, sqrt, square, exp3, cube
* Reuses each trait's existing 10,000-permutation paired method nulls from:
      results/ams_area_validation/weighted_A_to_D/paired_screen/<TRAIT>/
* Gaussian inference exactly matches run_ams_area.py:
      mu = mean(null)
      sigma = np.std(null)       # ddof=0
      two-sided Normal p-value
* BH is applied genome-wide at q=0.05 separately for each encoding.
* Every outer permutation is retained.
* Every outer replicate with >=1 BH discovery in any encoding gets a dedicated
  event directory containing the exact sample reassignment and significant genes.

No pathway enrichment. No method selection.

Implementation note
-------------------
Genome-wide Weighted AREA scoring is vectorized, but the script verifies the
vectorized formula against compute_weighted_enrichment_score() imported from
the current run_ams_area.py before running each trait.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfc


DEFAULT_TRAITS = ("Cognitive_stage", "Braak_stage")
ENCODINGS = ("current", "sqrt", "square", "exp3", "cube")
ROLES = {
    "current": "primary",
    "sqrt": "sensitivity",
    "square": "sensitivity",
    "exp3": "sensitivity",
    "cube": "stress_test",
}

BURST_THRESHOLDS = (1, 10, 100, 250, 500, 1000)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fresh large paired complete-null validation for Weighted AREA traits."
    )
    p.add_argument(
        "--traits",
        default=",".join(DEFAULT_TRAITS),
        help="Comma-separated traits. Default: Cognitive_stage,Braak_stage",
    )
    p.add_argument(
        "--expression",
        default="results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv",
    )
    p.add_argument(
        "--area-script",
        default="exploratory/area/run_ams_area.py",
    )
    p.add_argument(
        "--pilot-screen-root",
        default="results/ams_area_validation/weighted_A_to_D/paired_screen",
        help="Root containing <TRAIT>/ALL_outer_exact_permutation_map.csv and paired method nulls.",
    )
    p.add_argument(
        "--outer-permutations",
        type=int,
        default=500,
        help="Fresh outer phenotype permutations per trait. Default: 500.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260905,
        help=(
            "Base seed. Trait-specific deterministic offsets are added. "
            "Keep this seed fixed if extending later."
        ),
    )
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of outer replicates scored per matrix batch.",
    )
    p.add_argument(
        "--outdir",
        default=(
            "results/ams_area_validation/weighted_A_to_D/"
            "large_paired_null_500_Cognitive_Braak"
        ),
    )
    return p.parse_args()


def import_weighted_score(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("run_ams_area_current", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "compute_weighted_enrichment_score"):
        raise AttributeError(
            f"{path} lacks compute_weighted_enrichment_score()."
        )

    return mod.compute_weighted_enrichment_score


def transform(v, encoding):
    """
    Cohort-independent monotonic phenotype transformations.
    Participant counts never alter phenotype weights.
    """
    v = np.asarray(v, dtype=float)

    if encoding == "current":
        return v.copy()

    lo = float(np.min(v))
    hi = float(np.max(v))
    if np.isclose(lo, hi):
        raise ValueError("Phenotype vector has no variation.")

    x = (v - lo) / (hi - lo)

    if encoding == "sqrt":
        return np.sqrt(x)
    if encoding == "square":
        return x ** 2
    if encoding == "exp3":
        return np.expm1(3.0 * x) / np.expm1(3.0)
    if encoding == "cube":
        return x ** 3

    raise ValueError(f"Unknown encoding: {encoding}")


def trapz_unit_spacing(y):
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    return float(
        0.5 * (y[0] + y[-1] + 2.0 * np.sum(y[1:-1]))
    )


def weighted_position_coefficients(n):
    """
    Exact linear positional-coefficient representation of the production
    Weighted AREA statistic.
    """
    if n < 2:
        raise ValueError("Need at least two samples.")

    t = np.ones(n, dtype=float)
    t[0] = 0.5
    t[-1] = 0.5

    a = np.cumsum(t[::-1])[::-1]

    bin_width = 1.0 / n
    trend = (
        np.append(np.arange(0, 1, 1.0 / (n - 1)), 1.0)
        * bin_width
    )
    trend_area = trapz_unit_spacing(trend)

    return a, trend_area


def load_expression(path):
    d = pd.read_csv(path)
    if d.shape[1] < 2:
        raise ValueError("Expression file has fewer than two columns.")

    id_col = d.columns[0]
    sample_ids = d[id_col].astype(str).str.strip()

    if sample_ids.duplicated().any():
        raise ValueError(
            f"Expression sample IDs in '{id_col}' are not unique."
        )

    xdf = d.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if xdf.isna().any().any():
        bad = xdf.columns[xdf.isna().any()].tolist()[:10]
        raise ValueError(
            "Expression contains NA/non-numeric values. "
            f"Example columns: {bad}"
        )

    return (
        pd.Index(sample_ids),
        np.asarray(d.columns[1:], dtype=str),
        xdf.to_numpy(dtype=float),
        id_col,
    )


def load_trait_template(trait_dir):
    """
    Recover exact sample order and original current weights from the pilot map.
    The pilot shuffled values are NOT reused.
    """
    f = Path(trait_dir) / "ALL_outer_exact_permutation_map.csv"
    d = pd.read_csv(f)

    required = {
        "outer_replicate",
        "target_position",
        "target_sample_id",
        "original_current_weight",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"{f} is missing columns: {sorted(missing)}"
        )

    first_rep = int(d["outer_replicate"].min())
    g = (
        d[d["outer_replicate"] == first_rep]
        .sort_values("target_position")
        .copy()
    )

    sample_ids = g["target_sample_id"].astype(str).to_numpy()
    current = g["original_current_weight"].to_numpy(dtype=float)

    if len(sample_ids) != len(np.unique(sample_ids)):
        raise ValueError("Pilot target sample IDs are not unique.")

    return sample_ids, current, f


def align_expression(expr_ids, x_all, target_ids):
    positions = pd.Series(
        np.arange(len(expr_ids), dtype=int),
        index=expr_ids.astype(str),
    )

    target = pd.Series(target_ids.astype(str))
    missing = target[~target.isin(positions.index)]

    if len(missing):
        raise ValueError(
            f"{len(missing)} trait samples are missing from expression."
        )

    rows = positions.loc[target].to_numpy(dtype=int)
    return x_all[rows, :]


def build_rank_coefficient_matrix(x):
    """
    B[g, s] = positional coefficient assigned to sample s for gene g when
    samples are sorted by ASCENDING expression using stable mergesort.

    Exact match to:
        order = np.argsort(ranks, kind="mergesort")
    """
    n_samples, n_genes = x.shape
    a, trend_area = weighted_position_coefficients(n_samples)

    print(
        f"Building stable ascending rank orders for "
        f"{n_genes:,} genes x {n_samples:,} samples..."
    )

    orders = np.argsort(
        x,
        axis=0,
        kind="mergesort",
    ).T

    b = np.empty(
        (n_genes, n_samples),
        dtype=np.float64,
    )

    chunk = 2000
    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        local = orders[start:end]
        r = np.arange(end - start)[:, None]
        b[start:end][r, local] = a[None, :]

    del orders
    return b, trend_area


def score_matrix(b, trend_area, weight_matrix):
    """
    weight_matrix: samples x K
    returns: genes x K Weighted ES matrix.
    """
    n = b.shape[1]
    totals = np.sum(weight_matrix, axis=0)

    if np.any(totals <= 0):
        raise ValueError("At least one transformed weight vector sums to zero.")

    weighted_area = (b @ weight_matrix) / (
        n * totals[None, :]
    )

    return 2.0 * (weighted_area - trend_area)


def verify_vectorized_scoring(
    score_fn,
    x,
    b,
    trend_area,
    current_weights,
    seed,
):
    """
    Guardrail: vectorized ES must reproduce production score exactly.
    """
    rng = np.random.default_rng(seed)
    n_genes = x.shape[1]
    test_genes = rng.choice(
        n_genes,
        size=min(25, n_genes),
        replace=False,
    )

    rows = []

    for enc in ENCODINGS:
        w = transform(current_weights, enc)
        vectorized = score_matrix(
            b[test_genes],
            trend_area,
            w[:, None],
        )[:, 0]

        direct = np.empty(len(test_genes), dtype=float)

        for j, gene_idx in enumerate(test_genes):
            order = np.argsort(
                x[:, gene_idx],
                kind="mergesort",
            )
            direct[j] = float(
                score_fn(w[order])
            )

        diff = np.abs(vectorized - direct)

        rows.append(
            {
                "encoding": enc,
                "n_test_genes": len(test_genes),
                "max_abs_difference": float(np.max(diff)),
                "allclose": bool(
                    np.allclose(
                        vectorized,
                        direct,
                        rtol=1e-11,
                        atol=1e-13,
                    )
                ),
            }
        )

    check = pd.DataFrame(rows)

    if not check["allclose"].all():
        raise RuntimeError(
            "Vectorized scoring does not reproduce "
            "compute_weighted_enrichment_score()."
        )

    return check


def load_inner_null_parameters(trait_dir):
    rows = []

    for enc in ENCODINGS:
        f = Path(trait_dir) / f"{enc}_paired_method_null.csv"
        d = pd.read_csv(f)

        if "null_es" not in d.columns:
            raise ValueError(f"{f} lacks null_es")

        null = d["null_es"].to_numpy(dtype=float)

        if len(null) < 100:
            raise ValueError(
                f"{f} contains only {len(null)} null draws."
            )

        mu = float(np.mean(null))
        sd = float(np.std(null))  # EXACT runner convention: ddof=0.

        rows.append(
            {
                "encoding": enc,
                "null_file": str(f),
                "inner_null_n": len(null),
                "null_mean": mu,
                "null_sd_ddof0": sd,
                "null_min": float(np.min(null)),
                "null_max": float(np.max(null)),
            }
        )

    return pd.DataFrame(rows)


def bh_hits_from_p(p, q):
    p = np.asarray(p, dtype=float)
    m = len(p)

    order = np.argsort(
        p,
        kind="mergesort",
    )
    ranked = p[order]
    critical = q * np.arange(1, m + 1) / m
    ok = ranked <= critical

    if not np.any(ok):
        return np.empty(0, dtype=int)

    k = int(np.where(ok)[0][-1] + 1)
    return order[:k]


def wilson_interval(successes, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan

    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )

    return max(0.0, center - half), min(1.0, center + half)


def save_event(
    event_root,
    trait,
    rep,
    perm_idx,
    sample_ids,
    current_weights,
    hit_indices_by_encoding,
    gene_ids,
):
    event_dir = event_root / f"outer_{rep:04d}"
    event_dir.mkdir(parents=True, exist_ok=True)

    exact = pd.DataFrame(
        {
            "trait": trait,
            "target_position": np.arange(len(sample_ids)),
            "target_sample_id": sample_ids,
            "source_position": perm_idx,
            "source_sample_id": sample_ids[perm_idx],
            "original_current_weight": current_weights,
            "shuffled_current_weight": current_weights[perm_idx],
        }
    )

    for enc in ENCODINGS:
        original_enc = transform(
            current_weights,
            enc,
        )
        exact[f"{enc}_original_weight"] = original_enc
        exact[f"{enc}_shuffled_weight"] = original_enc[perm_idx]

    exact.to_csv(
        event_dir / "exact_phenotype_permutation.csv",
        index=False,
    )

    hit_rows = []

    for enc in ENCODINGS:
        idx = hit_indices_by_encoding[enc]
        genes = gene_ids[idx]

        pd.DataFrame(
            {
                "gene_id": genes,
                "trait": trait,
                "encoding": enc,
                "outer_replicate": rep,
            }
        ).to_csv(
            event_dir / f"{enc}_fdr_significant_genes.csv",
            index=False,
        )

        hit_rows.append(
            {
                "trait": trait,
                "outer_replicate": rep,
                "encoding": enc,
                "encoding_role": ROLES[enc],
                "n_fdr_sig": len(idx),
            }
        )

    pd.DataFrame(hit_rows).to_csv(
        event_dir / "event_hit_counts.csv",
        index=False,
    )


def aggregate_results(trait, replicate_df, q):
    rows = []

    for enc in ENCODINGS:
        g = replicate_df[
            replicate_df["encoding"] == enc
        ].copy()

        hits = g["n_fdr_sig"].to_numpy(dtype=int)
        n = len(hits)

        row = {
            "trait": trait,
            "encoding": enc,
            "encoding_role": ROLES[enc],
            "outer_permutations": n,
            "nominal_bh_q": q,
            "mean_fdr_hits": float(np.mean(hits)),
            "median_fdr_hits": float(np.median(hits)),
            "p95_fdr_hits": float(np.quantile(hits, 0.95)),
            "p99_fdr_hits": float(np.quantile(hits, 0.99)),
            "max_fdr_hits": int(np.max(hits)),
            "mean_frac_p_lt_0.05": float(
                g["frac_p_lt_0.05"].mean()
            ),
            "mean_frac_p_lt_0.01": float(
                g["frac_p_lt_0.01"].mean()
            ),
            "mean_frac_p_lt_0.001": float(
                g["frac_p_lt_0.001"].mean()
            ),
        }

        for threshold in BURST_THRESHOLDS:
            count = int(np.sum(hits >= threshold))
            rate = count / n
            lo, hi = wilson_interval(count, n)

            label = (
                "any"
                if threshold == 1
                else f"ge_{threshold}"
            )

            row[f"{label}_events"] = count
            row[f"{label}_rate"] = rate
            row[f"{label}_wilson95_low"] = lo
            row[f"{label}_wilson95_high"] = hi

        # Under the complete/global null, FDR = P(R > 0).
        row["global_null_fdr_estimate"] = row["any_rate"]
        row["global_null_fdr_wilson95_low"] = row[
            "any_wilson95_low"
        ]
        row["global_null_fdr_wilson95_high"] = row[
            "any_wilson95_high"
        ]

        rows.append(row)

    return pd.DataFrame(rows)


def paired_event_concordance(trait, replicate_df):
    wide = replicate_df.pivot(
        index="outer_replicate",
        columns="encoding",
        values="n_fdr_sig",
    )

    rows = []

    for i, a in enumerate(ENCODINGS):
        for b in ENCODINGS[i + 1:]:
            aa = wide[a] > 0
            bb = wide[b] > 0

            both = int(np.sum(aa & bb))
            a_only = int(np.sum(aa & ~bb))
            b_only = int(np.sum(~aa & bb))
            neither = int(np.sum(~aa & ~bb))

            event_union = both + a_only + b_only

            rows.append(
                {
                    "trait": trait,
                    "encoding_a": a,
                    "encoding_b": b,
                    "both_any_hit": both,
                    "a_only_any_hit": a_only,
                    "b_only_any_hit": b_only,
                    "neither_any_hit": neither,
                    "event_jaccard": (
                        both / event_union
                        if event_union > 0
                        else np.nan
                    ),
                    "hit_count_correlation": float(
                        wide[a].corr(wide[b])
                    ),
                }
            )

    return pd.DataFrame(rows)


def run_trait(
    trait,
    trait_index,
    a,
    score_fn,
    expr_ids,
    gene_ids,
    x_all,
    expr_id_col,
    root_outdir,
):
    trait_seed = a.seed + 100000 * trait_index
    trait_pilot_dir = Path(a.pilot_screen_root) / trait
    trait_outdir = root_outdir / trait
    trait_outdir.mkdir(parents=True, exist_ok=True)

    event_root = trait_outdir / "problematic_outer_events"
    event_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"{trait}: LARGE PAIRED COMPLETE-NULL VALIDATION")
    print("=" * 80)
    print(f"Fresh outer permutations: {a.outer_permutations:,}")
    print(f"Trait seed:               {trait_seed}")
    print(f"BH q:                     {a.fdr_threshold}")

    sample_ids, current_weights, template_file = (
        load_trait_template(trait_pilot_dir)
    )

    x = align_expression(
        expr_ids,
        x_all,
        sample_ids,
    )

    print(
        f"Expression aligned: {x.shape[0]:,} samples x "
        f"{x.shape[1]:,} genes"
    )
    print(f"Current phenotype levels: {np.unique(current_weights).tolist()}")

    pd.DataFrame(
        {
            "position": np.arange(len(sample_ids)),
            "sample_id": sample_ids,
            "current_weight": current_weights,
        }
    ).to_csv(
        trait_outdir / "sample_order_and_current_weights.csv",
        index=False,
    )

    null_params = load_inner_null_parameters(
        trait_pilot_dir
    )
    null_params.insert(0, "trait", trait)
    null_params.to_csv(
        trait_outdir / "inner_null_parameters_exact_runner.csv",
        index=False,
    )

    mu_by_enc = {
        row.encoding: float(row.null_mean)
        for row in null_params.itertuples(index=False)
    }
    sd_by_enc = {
        row.encoding: float(row.null_sd_ddof0)
        for row in null_params.itertuples(index=False)
    }

    b, trend_area = build_rank_coefficient_matrix(x)

    vector_check = verify_vectorized_scoring(
        score_fn,
        x,
        b,
        trend_area,
        current_weights,
        seed=trait_seed + 999,
    )
    vector_check.insert(0, "trait", trait)
    vector_check.to_csv(
        trait_outdir / "vectorized_scoring_validation.csv",
        index=False,
    )

    print("\nVectorized scoring validation:")
    print(vector_check.to_string(index=False))
    print("PASS: vectorized ES matches production Weighted AREA.")

    # Generate all fresh paired permutations upfront.
    rng = np.random.default_rng(trait_seed)
    n_samples = len(sample_ids)

    perm_indices = np.empty(
        (a.outer_permutations, n_samples),
        dtype=np.int16
        if n_samples < np.iinfo(np.int16).max
        else np.int32,
    )

    for i in range(a.outer_permutations):
        perm_indices[i] = rng.permutation(n_samples)

    perm_cols = {
        "outer_replicate": np.arange(
            1,
            a.outer_permutations + 1,
        )
    }
    for pos in range(n_samples):
        perm_cols[f"source_pos_{pos:03d}"] = perm_indices[:, pos]

    pd.DataFrame(perm_cols).to_csv(
        trait_outdir / "ALL_fresh_outer_permutation_indices.csv",
        index=False,
    )

    with open(
        trait_outdir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "trait": trait,
                "analysis": "large paired complete-null validation",
                "fresh_outer_permutations": a.outer_permutations,
                "base_seed": a.seed,
                "trait_seed": trait_seed,
                "fdr_threshold": a.fdr_threshold,
                "encodings": list(ENCODINGS),
                "encoding_roles": ROLES,
                "burst_thresholds": list(BURST_THRESHOLDS),
                "expression": str(Path(a.expression).resolve()),
                "area_script": str(Path(a.area_script).resolve()),
                "pilot_trait_dir": str(trait_pilot_dir.resolve()),
                "pilot_template_file": str(template_file.resolve()),
                "expression_id_column": expr_id_col,
                "inference": (
                    "full-null Gaussian two-sided using existing paired 10k nulls; "
                    "np.std(null), ddof=0; BH separately per encoding"
                ),
                "paired_design": (
                    "same exact fresh phenotype permutation applied to all encodings "
                    "within trait"
                ),
                "guardrail": (
                    "Pilot permutations are retained and not replaced. "
                    "No pathways or method selection."
                ),
            },
            handle,
            indent=2,
        )

    replicate_rows = []
    problematic_rows = []

    print("\nRunning fresh paired outer-null permutations...")

    for batch_start in range(
        0,
        a.outer_permutations,
        a.batch_size,
    ):
        batch_end = min(
            batch_start + a.batch_size,
            a.outer_permutations,
        )
        reps = list(range(batch_start, batch_end))

        weight_columns = []
        column_meta = []

        for rep0 in reps:
            perm_idx = perm_indices[rep0]
            shuffled_current = current_weights[perm_idx]

            for enc in ENCODINGS:
                weight_columns.append(
                    transform(shuffled_current, enc)
                )
                column_meta.append(
                    (rep0 + 1, enc)
                )

        wmat = np.column_stack(weight_columns)
        es = score_matrix(
            b,
            trend_area,
            wmat,
        )

        mus = np.asarray(
            [mu_by_enc[enc] for _, enc in column_meta],
            dtype=float,
        )
        sds = np.asarray(
            [sd_by_enc[enc] for _, enc in column_meta],
            dtype=float,
        )

        z = (es - mus[None, :]) / sds[None, :]
        pmat = erfc(np.abs(z) / math.sqrt(2.0))

        batch_event_hits = {}

        for col, (rep, enc) in enumerate(column_meta):
            p = pmat[:, col]
            hit_idx = bh_hits_from_p(
                p,
                a.fdr_threshold,
            )

            replicate_rows.append(
                {
                    "trait": trait,
                    "outer_replicate": rep,
                    "encoding": enc,
                    "encoding_role": ROLES[enc],
                    "n_fdr_sig": int(len(hit_idx)),
                    "frac_p_lt_0.05": float(
                        np.mean(p < 0.05)
                    ),
                    "frac_p_lt_0.01": float(
                        np.mean(p < 0.01)
                    ),
                    "frac_p_lt_0.001": float(
                        np.mean(p < 0.001)
                    ),
                    "median_p": float(np.median(p)),
                    "min_p": float(np.min(p)),
                }
            )

            batch_event_hits.setdefault(
                rep,
                {},
            )[enc] = hit_idx

        for rep, hits_by_enc in batch_event_hits.items():
            counts = {
                enc: int(len(hits_by_enc[enc]))
                for enc in ENCODINGS
            }

            if any(v > 0 for v in counts.values()):
                save_event(
                    event_root=event_root,
                    trait=trait,
                    rep=rep,
                    perm_idx=perm_indices[rep - 1],
                    sample_ids=sample_ids,
                    current_weights=current_weights,
                    hit_indices_by_encoding=hits_by_enc,
                    gene_ids=gene_ids,
                )

                row = {
                    "trait": trait,
                    "outer_replicate": rep,
                    "encodings_with_hits": ",".join(
                        enc
                        for enc in ENCODINGS
                        if counts[enc] > 0
                    ),
                    "max_hits_any_encoding": max(
                        counts.values()
                    ),
                }
                row.update(
                    {
                        f"{enc}_hits": counts[enc]
                        for enc in ENCODINGS
                    }
                )
                problematic_rows.append(row)

        print(
            f"  completed {batch_end:>4}/{a.outer_permutations} "
            f"fresh outer permutations"
        )

    replicate_df = pd.DataFrame(replicate_rows)
    replicate_df.to_csv(
        trait_outdir / "fresh_outer_replicate_summary.csv",
        index=False,
    )

    if problematic_rows:
        problem_df = (
            pd.DataFrame(problematic_rows)
            .sort_values("outer_replicate")
            .reset_index(drop=True)
        )
    else:
        problem_df = pd.DataFrame(
            columns=[
                "trait",
                "outer_replicate",
                "encodings_with_hits",
                "max_hits_any_encoding",
            ]
            + [f"{enc}_hits" for enc in ENCODINGS]
        )

    problem_df.to_csv(
        trait_outdir / "problematic_outer_runs.csv",
        index=False,
    )

    calibration = aggregate_results(
        trait,
        replicate_df,
        a.fdr_threshold,
    )
    calibration.to_csv(
        trait_outdir / "FRESH_500_NULL_CALIBRATION_SUMMARY.csv",
        index=False,
    )

    concordance = paired_event_concordance(
        trait,
        replicate_df,
    )
    concordance.to_csv(
        trait_outdir / "paired_encoding_event_concordance.csv",
        index=False,
    )

    print("\nSummary:")
    cols = [
        "trait",
        "encoding",
        "outer_permutations",
        "any_events",
        "any_rate",
        "any_wilson95_low",
        "any_wilson95_high",
        "ge_100_events",
        "ge_500_events",
        "ge_1000_events",
        "max_fdr_hits",
    ]
    print(
        calibration[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    del b, x

    return calibration, replicate_df, problem_df, concordance, vector_check


def main():
    a = parse_args()

    traits = [
        t.strip()
        for t in a.traits.split(",")
        if t.strip()
    ]

    if not traits:
        raise ValueError("No traits supplied.")
    if a.outer_permutations < 1:
        raise ValueError("--outer-permutations must be >=1.")
    if a.batch_size < 1:
        raise ValueError("--batch-size must be >=1.")

    root_outdir = Path(a.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LARGE PAIRED COMPLETE-NULL VALIDATION")
    print("=" * 80)
    print(f"Traits:                   {', '.join(traits)}")
    print(f"Fresh outer permutations: {a.outer_permutations:,} PER TRAIT")
    print(f"Base seed:                {a.seed}")
    print(f"BH q:                     {a.fdr_threshold}")
    print(f"Encodings:                {', '.join(ENCODINGS)}")
    print(
        "\nExisting pilot null events are retained. "
        "These are fresh null runs, not replacement permutations."
    )

    score_fn = import_weighted_score(a.area_script)

    expr_ids, gene_ids, x_all, expr_id_col = (
        load_expression(a.expression)
    )

    print(
        f"\nExpression loaded: {len(expr_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )

    all_cal = []
    all_rep = []
    all_problem = []
    all_concord = []
    all_vector = []

    for trait_index, trait in enumerate(traits):
        cal, rep, problem, concord, vector = run_trait(
            trait=trait,
            trait_index=trait_index,
            a=a,
            score_fn=score_fn,
            expr_ids=expr_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            expr_id_col=expr_id_col,
            root_outdir=root_outdir,
        )

        all_cal.append(cal)
        all_rep.append(rep)
        all_problem.append(problem)
        all_concord.append(concord)
        all_vector.append(vector)

    master_cal = pd.concat(all_cal, ignore_index=True)
    master_rep = pd.concat(all_rep, ignore_index=True)
    master_problem = pd.concat(all_problem, ignore_index=True)
    master_concord = pd.concat(all_concord, ignore_index=True)
    master_vector = pd.concat(all_vector, ignore_index=True)

    master_cal.to_csv(
        root_outdir / "MASTER_FRESH_500_NULL_CALIBRATION_SUMMARY.csv",
        index=False,
    )
    master_rep.to_csv(
        root_outdir / "MASTER_fresh_outer_replicate_summary.csv",
        index=False,
    )
    master_problem.to_csv(
        root_outdir / "MASTER_problematic_outer_runs.csv",
        index=False,
    )
    master_concord.to_csv(
        root_outdir / "MASTER_paired_encoding_event_concordance.csv",
        index=False,
    )
    master_vector.to_csv(
        root_outdir / "MASTER_vectorized_scoring_validation.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("ALL TRAITS COMPLETE")
    print("=" * 80)

    cols = [
        "trait",
        "encoding",
        "outer_permutations",
        "any_events",
        "any_rate",
        "any_wilson95_low",
        "any_wilson95_high",
        "ge_100_events",
        "ge_500_events",
        "ge_1000_events",
        "max_fdr_hits",
    ]

    print(
        master_cal[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nMain master outputs:")
    for name in [
        "MASTER_FRESH_500_NULL_CALIBRATION_SUMMARY.csv",
        "MASTER_fresh_outer_replicate_summary.csv",
        "MASTER_problematic_outer_runs.csv",
        "MASTER_paired_encoding_event_concordance.csv",
        "MASTER_vectorized_scoring_validation.csv",
    ]:
        print(f"  {root_outdir / name}")

    print(
        "\nSTOP HERE. Inspect calibration before simulations, pathways, "
        "or selecting an encoding."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_weighted_area_full_phenotype_suite.py
========================================

Production ROSMAP Weighted AREA suite using the LOCKED method:

- average ranks for exact expression ties
- covariate adjustment:
      age_death + sex + rin_numeric + pmi_numeric + sequencing_batch
- partial linear-rank association
- validated within-sequencing-batch permutation variance
- two-sided Gaussian p-values
- BH FDR across all tested genes
- raw tie-aware AREA ES retained as the geometric effect representation

PHENOTYPES INCLUDED
-------------------
Cognitive:
    1. Cognitive_stage
       diagnosis codes:
           1 -> NCI -> 0
           2,3 -> MCI -> 0.5
           4,5 -> AD/dementia -> 1
           6 -> excluded
    2. cogn_global_impairment
       continuous global cognition, reversed so higher = worse:
           max(cogn_global) - cogn_global
    3. mmse_impairment
       continuous MMSE impairment:
           30 - cts_estmmse30

Amyloid:
    4. CERAD_equal
    5. CERAD_amyloid_calibrated
    6. amyloid_continuous

Tau:
    7. Braak_equal
    8. Braak_tangle_calibrated
    9. tangle_continuous

NOT INCLUDED AS PRIMARY WEIGHTED PHENOTYPES
-------------------------------------------
- binary variables (sex, stroke, diabetes, hypertension): Regular AREA is
  more natural for these.
- technical variables (RIN, PMI, batch): covariates, not targets.
- age/age_death: covariates, not disease phenotypes in this analysis.
- APOE: genotype/risk factor, not an ordered disease-severity phenotype.
- age_first_ad_dx: onset timing, not a cross-cohort severity axis.
- raw CERAD/Braak duplicate codings: redundant with equal-weight encodings.
- cogdx: not included separately because diagnosis/Cognitive_stage already
  represents the ordered clinical-state axis used in the project.

SIGN CONVENTION
---------------
Expression ranks are ascending.

Positive adjusted_AREA_Z / raw_tieaware_AREA_ES:
    higher phenotype severity -> lower expression

Negative adjusted_AREA_Z / raw_tieaware_AREA_ES:
    higher phenotype severity -> higher expression

OUTPUTS
-------
Root:
    MASTER_weighted_area_results_long.csv
    MASTER_significant_gene_counts.csv
    MASTER_pairwise_overlap.csv
    MASTER_pairwise_z_correlations.csv
    MASTER_phenotype_summary.csv
    analysis_manifest.json

Per phenotype:
    <phenotype>/results.csv
    <phenotype>/matched_sample_manifest.csv
    <phenotype>/excluded_samples.csv
    <phenotype>/phenotype_residualization_summary.csv
    <phenotype>/gene_tie_diagnostics.csv

No pathway analysis is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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


@dataclass
class PhenotypeSpec:
    name: str
    family: str
    source: str
    transform: str
    description: str


PHENOTYPES = [
    PhenotypeSpec(
        name="Cognitive_stage",
        family="cognitive",
        source="diagnosis",
        transform="cognitive_stage",
        description="NCI=0; MCI=0.5; AD/dementia=1; diagnosis 6 excluded.",
    ),
    PhenotypeSpec(
        name="cogn_global_impairment",
        family="cognitive",
        source="cogn_global",
        transform="reverse_from_max",
        description="Continuous global cognition reversed so higher values indicate worse cognition.",
    ),
    PhenotypeSpec(
        name="mmse_impairment",
        family="cognitive",
        source="cts_estmmse30",
        transform="mmse_impairment",
        description="30 - estimated MMSE; higher values indicate greater cognitive impairment.",
    ),
    PhenotypeSpec(
        name="CERAD_equal",
        family="amyloid",
        source="CERAD_equal_weight",
        transform="identity",
        description="Equal-spaced CERAD burden score.",
    ),
    PhenotypeSpec(
        name="CERAD_amyloid_calibrated",
        family="amyloid",
        source="CERAD_amyloid_calibrated_weight",
        transform="identity",
        description="CERAD score calibrated to quantitative cortical amyloid burden.",
    ),
    PhenotypeSpec(
        name="amyloid_continuous",
        family="amyloid",
        source="amyloid_continuous_AREA_weight",
        transform="identity",
        description="Continuous quantitative amyloid burden.",
    ),
    PhenotypeSpec(
        name="Braak_equal",
        family="tau",
        source="Braak_equal_weight",
        transform="identity",
        description="Equal-spaced Braak stage score.",
    ),
    PhenotypeSpec(
        name="Braak_tangle_calibrated",
        family="tau",
        source="Braak_tangle_calibrated_weight",
        transform="identity",
        description="Braak stage score calibrated to quantitative tangle burden.",
    ),
    PhenotypeSpec(
        name="tangle_continuous",
        family="tau",
        source="tangle_continuous_AREA_weight",
        transform="identity",
        description="Continuous quantitative tangle burden.",
    ),
]


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
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--rank-chunk-size", type=int, default=2000)
    p.add_argument(
        "--keep-multibatch",
        action="store_true",
        help="Keep comma-containing sequencing_batch labels as literal categories.",
    )
    p.add_argument(
        "--outdir",
        default="results/weighted_area_full_phenotype_suite",
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


def derive_phenotype(metadata, spec):
    if spec.source not in metadata.columns:
        raise ValueError(
            f"Phenotype '{spec.name}' requires missing column '{spec.source}'."
        )

    raw = pd.to_numeric(metadata[spec.source], errors="coerce")

    if spec.transform == "identity":
        w = raw.copy()

    elif spec.transform == "cognitive_stage":
        # Project coding:
        # 1=NCI; 2/3=MCI; 4/5=AD/dementia; 6=other dementia/excluded.
        mapping = {
            1.0: 0.0,
            2.0: 0.5,
            3.0: 0.5,
            4.0: 1.0,
            5.0: 1.0,
        }
        w = raw.map(mapping)

    elif spec.transform == "reverse_from_max":
        valid = raw.dropna()
        if valid.empty:
            w = raw * np.nan
        else:
            w = valid.max() - raw

    elif spec.transform == "mmse_impairment":
        # Estimated MMSE is expected on the conventional 0-30 scale.
        bad = raw.dropna()[(raw.dropna() < 0) | (raw.dropna() > 30)]
        if len(bad):
            raise ValueError(
                f"{len(bad)} MMSE values fall outside 0-30; inspect before analysis."
            )
        w = 30.0 - raw

    else:
        raise ValueError(f"Unknown transform: {spec.transform}")

    w = pd.to_numeric(w, errors="coerce")

    # Weighted AREA raw ES representation requires nonnegative weights.
    if (w.dropna() < 0).any():
        raise ValueError(
            f"Phenotype '{spec.name}' produced negative weights."
        )

    return w


def prepare_phenotype_metadata(
    metadata,
    spec,
    sample_col,
    keep_multibatch,
):
    required = (
        [sample_col, spec.source]
        + CONTINUOUS_COVARIATES
        + BINARY_COVARIATES
        + [BATCH_COL]
    )

    missing = [c for c in required if c not in metadata.columns]
    if missing:
        raise ValueError(
            f"Phenotype '{spec.name}' missing required metadata columns: {missing}"
        )

    d = metadata[required].copy()
    d[sample_col] = d[sample_col].astype(str).str.strip()

    d["phenotype_weight"] = derive_phenotype(metadata, spec)

    for c in CONTINUOUS_COVARIATES + BINARY_COVARIATES:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d[BATCH_COL] = d[BATCH_COL].astype("string").str.strip()

    exclusions = []

    multibatch = d[BATCH_COL].str.contains(",", na=False)
    if not keep_multibatch and multibatch.any():
        for _, row in d.loc[multibatch].iterrows():
            exclusions.append({
                "sample_id": row[sample_col],
                "reason": "ambiguous_multi_batch_annotation",
                "value": str(row[BATCH_COL]),
            })
        d = d.loc[~multibatch].copy()

    complete_cols = (
        ["phenotype_weight"]
        + CONTINUOUS_COVARIATES
        + BINARY_COVARIATES
        + [BATCH_COL]
    )

    missing_mask = d[complete_cols].isna().any(axis=1)

    if missing_mask.any():
        for _, row in d.loc[missing_mask].iterrows():
            missing_here = [c for c in complete_cols if pd.isna(row[c])]
            exclusions.append({
                "sample_id": row[sample_col],
                "reason": "missing_phenotype_or_primary_covariate",
                "value": ";".join(missing_here),
            })
        d = d.loc[~missing_mask].copy()

    if d[sample_col].duplicated().any():
        raise ValueError(f"Duplicate sample IDs for phenotype '{spec.name}'.")

    if d["phenotype_weight"].nunique() < 2:
        raise ValueError(f"Phenotype '{spec.name}' is constant after filtering.")

    exclusion_df = pd.DataFrame(
        exclusions,
        columns=["sample_id", "reason", "value"],
    )

    return d, exclusion_df


def align_expression(expression_ids, x_all, d, sample_col):
    pos = pd.Series(
        np.arange(len(expression_ids), dtype=int),
        index=expression_ids.astype(str),
    )

    absent = d.loc[~d[sample_col].isin(pos.index), sample_col]
    if len(absent):
        raise ValueError(
            f"{len(absent)} metadata samples are absent from expression."
        )

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
        z = (x - np.mean(x)) / sd
        pieces.append(z)
        names.append(c + "_z")

    sex = d["sex"].to_numpy(dtype=float)
    pieces.append(sex - np.mean(sex))
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
            f"Covariate design rank deficient: rank={rank}, columns={C.shape[1]}"
        )

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

        print(f"    ranked genes: {end:,}/{n_genes:,}")

    return R, tie_fraction, max_tie_group


def residualize_gene_ranks(R, Q, chunk_size):
    n_genes = R.shape[0]
    E = np.empty_like(R, dtype=np.float32)

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)

        Y = R[start:end, :].T.astype(np.float64)
        residual = residualize_matrix(Q, Y)
        E[start:end, :] = residual.T.astype(np.float32)

        print(f"    residualized genes: {end:,}/{n_genes:,}")

    return E


def build_exchangeability_blocks(d):
    labels = d[BATCH_COL].astype(str).to_numpy()
    blocks = []

    for label in sorted(pd.unique(labels).tolist()):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            raise ValueError(
                f"Sequencing batch '{label}' has fewer than 2 samples."
            )
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


def run_phenotype(
    spec,
    expression_ids,
    gene_ids,
    x_all,
    metadata,
    args,
    root,
):
    outdir = root / spec.name
    outdir.mkdir(parents=True, exist_ok=True)

    d, exclusions = prepare_phenotype_metadata(
        metadata=metadata,
        spec=spec,
        sample_col=args.sample_col,
        keep_multibatch=args.keep_multibatch,
    )

    exclusions.to_csv(outdir / "excluded_samples.csv", index=False)

    x = align_expression(
        expression_ids,
        x_all,
        d,
        args.sample_col,
    )

    n = len(d)
    n_genes = len(gene_ids)

    print("\n" + "=" * 80)
    print(spec.name)
    print("=" * 80)
    print(f"Family:       {spec.family}")
    print(f"Samples:      {n:,}")
    print(f"Genes tested: {n_genes:,}")
    print(f"Definition:   {spec.description}")

    pd.DataFrame({
        "position": np.arange(n, dtype=int),
        "sample_id": d[args.sample_col].to_numpy(),
        "phenotype_weight": d["phenotype_weight"].to_numpy(dtype=float),
        BATCH_COL: d[BATCH_COL].astype(str).to_numpy(),
    }).to_csv(outdir / "matched_sample_manifest.csv", index=False)

    C, Q, covariate_names, condition_number = build_covariate_design(d)

    w = d["phenotype_weight"].to_numpy(dtype=np.float64)
    u = residualize_matrix(Q, w[:, None])[:, 0]

    phenotype_summary = pd.DataFrame([{
        "phenotype": spec.name,
        "family": spec.family,
        "n_samples": n,
        "raw_min": float(np.min(w)),
        "raw_max": float(np.max(w)),
        "raw_mean": float(np.mean(w)),
        "raw_sd": float(np.std(w, ddof=0)),
        "n_unique_values": int(pd.Series(w).nunique()),
        "residual_mean": float(np.mean(u)),
        "residual_sd": float(np.std(u, ddof=0)),
        "fraction_raw_variance_remaining_after_covariates": float(
            np.var(u, ddof=0) / np.var(w, ddof=0)
        ),
        "raw_vs_residual_correlation": float(np.corrcoef(w, u)[0, 1]),
        "covariate_design_condition_number": condition_number,
    }])

    phenotype_summary.to_csv(
        outdir / "phenotype_residualization_summary.csv",
        index=False,
    )

    print("\nComputing tie-aware average expression ranks...")
    R, tie_fraction, max_tie_group = compute_average_rank_matrix(
        x,
        args.rank_chunk_size,
    )

    raw_es = raw_tie_aware_area_es(R, w)

    print("\nResidualizing expression ranks against covariates...")
    E = residualize_gene_ranks(
        R,
        Q,
        args.rank_chunk_size,
    )

    blocks = build_exchangeability_blocks(d)
    var = blockwise_variance(E, u, blocks)
    sd = np.sqrt(var)

    T = E.astype(np.float64, copy=False) @ u
    z = -T / sd
    p = erfc(np.abs(z) / math.sqrt(2.0))
    q = bh_adjust(p)

    ss_e = np.sum(E.astype(np.float64, copy=False) ** 2, axis=1)
    ss_u = float(np.sum(u * u))
    partial_r = T / np.sqrt(ss_e * ss_u)

    sig = q < args.fdr_threshold

    result = pd.DataFrame({
        "gene_id": gene_ids,
        "family": spec.family,
        "phenotype": spec.name,
        "n_samples": n,
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

    result["direction"] = np.where(
        z > 0,
        "higher_severity_lower_expression",
        np.where(
            z < 0,
            "higher_severity_higher_expression",
            "zero",
        ),
    )

    result = result.sort_values(
        ["adjusted_padj_BH", "adjusted_pvalue"],
        kind="mergesort",
    )

    result.to_csv(outdir / "results.csv", index=False)

    pd.DataFrame({
        "gene_id": gene_ids,
        "fraction_samples_in_ties": tie_fraction,
        "max_tie_group_size": max_tie_group,
    }).to_csv(
        outdir / "gene_tie_diagnostics.csv",
        index=False,
    )

    n_sig = int(np.sum(sig))
    n_pos = int(np.sum(sig & (z > 0)))
    n_neg = int(np.sum(sig & (z < 0)))

    print(
        f"\nFDR < {args.fdr_threshold:g}: {n_sig:,} genes "
        f"(higher severity -> lower expression: {n_pos:,}; "
        f"higher severity -> higher expression: {n_neg:,})"
    )

    count_row = {
        "family": spec.family,
        "phenotype": spec.name,
        "n_samples": n,
        "n_genes_tested": n_genes,
        "fdr_threshold": args.fdr_threshold,
        "n_fdr_significant": n_sig,
        "n_sig_higher_severity_lower_expression": n_pos,
        "n_sig_higher_severity_higher_expression": n_neg,
    }

    return {
        "spec": spec,
        "results": result,
        "count_row": count_row,
        "phenotype_summary": phenotype_summary,
        "covariate_names": covariate_names,
        "condition_number": condition_number,
    }


def pairwise_summaries(run_results, gene_ids):
    names = [r["spec"].name for r in run_results]

    sig_map = {}
    z_map = {}

    for r in run_results:
        df = r["results"].set_index("gene_id").reindex(gene_ids)
        sig_map[r["spec"].name] = df["fdr_significant"].to_numpy(dtype=bool)
        z_map[r["spec"].name] = df["adjusted_AREA_Z"].to_numpy(dtype=float)

    overlap_rows = []
    corr_rows = []

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = sig_map[a]
            sb = sig_map[b]

            inter = int(np.sum(sa & sb))
            union = int(np.sum(sa | sb))
            only_a = int(np.sum(sa & ~sb))
            only_b = int(np.sum(~sa & sb))

            overlap_rows.append({
                "phenotype_A": a,
                "phenotype_B": b,
                "n_sig_A": int(np.sum(sa)),
                "n_sig_B": int(np.sum(sb)),
                "intersection": inter,
                "A_only": only_a,
                "B_only": only_b,
                "union": union,
                "jaccard": (inter / union) if union else np.nan,
            })

            za = z_map[a]
            zb = z_map[b]
            mask = np.isfinite(za) & np.isfinite(zb)

            corr_rows.append({
                "phenotype_A": a,
                "phenotype_B": b,
                "n_genes": int(np.sum(mask)),
                "pearson_adjusted_Z": float(
                    np.corrcoef(za[mask], zb[mask])[0, 1]
                ),
                "spearman_adjusted_Z": float(
                    pd.Series(za[mask]).corr(
                        pd.Series(zb[mask]),
                        method="spearman",
                    )
                ),
            })

    return pd.DataFrame(overlap_rows), pd.DataFrame(corr_rows)


def main():
    args = parse_args()
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)

    expression_ids, gene_ids, x_all, expression_id_col = load_expression(
        args.expression,
        args.expression_id_col,
    )

    print("=" * 80)
    print("ROSMAP WEIGHTED AREA: FULL PHENOTYPE SUITE")
    print("=" * 80)
    print(
        f"Expression: {len(expression_ids):,} samples x "
        f"{len(gene_ids):,} genes"
    )
    print(
        "Primary covariates: "
        "age_death + sex + rin_numeric + pmi_numeric + sequencing_batch"
    )
    print(f"Phenotypes scheduled: {len(PHENOTYPES)}")

    run_results = []

    for spec in PHENOTYPES:
        run_results.append(
            run_phenotype(
                spec=spec,
                expression_ids=expression_ids,
                gene_ids=gene_ids,
                x_all=x_all,
                metadata=metadata,
                args=args,
                root=root,
            )
        )

    master_results = pd.concat(
        [r["results"] for r in run_results],
        ignore_index=True,
    )
    master_results.to_csv(
        root / "MASTER_weighted_area_results_long.csv",
        index=False,
    )

    master_counts = pd.DataFrame(
        [r["count_row"] for r in run_results]
    )
    master_counts.to_csv(
        root / "MASTER_significant_gene_counts.csv",
        index=False,
    )

    master_pheno = pd.concat(
        [r["phenotype_summary"] for r in run_results],
        ignore_index=True,
    )
    master_pheno.to_csv(
        root / "MASTER_phenotype_summary.csv",
        index=False,
    )

    pairwise_overlap, pairwise_corr = pairwise_summaries(
        run_results,
        gene_ids,
    )

    pairwise_overlap.to_csv(
        root / "MASTER_pairwise_overlap.csv",
        index=False,
    )

    pairwise_corr.to_csv(
        root / "MASTER_pairwise_z_correlations.csv",
        index=False,
    )

    manifest = {
        "expression": str(Path(args.expression).resolve()),
        "metadata": str(Path(args.metadata).resolve()),
        "expression_id_col": expression_id_col,
        "sample_col": args.sample_col,
        "fdr_threshold": args.fdr_threshold,
        "primary_covariates": (
            CONTINUOUS_COVARIATES
            + BINARY_COVARIATES
            + [BATCH_COL]
        ),
        "tie_handling": "Exact expression ties receive average ranks.",
        "primary_inference": (
            "Covariate-adjusted partial linear-rank statistic with "
            "validated within-sequencing-batch permutation variance, "
            "two-sided Gaussian p-values, and BH FDR."
        ),
        "raw_effect_representation": (
            "Unadjusted tie-aware AREA ES on each phenotype's final "
            "complete-case sample set."
        ),
        "sign_convention": (
            "Positive adjusted AREA Z/raw ES = higher phenotype severity "
            "associated with lower expression; negative = higher severity "
            "associated with higher expression."
        ),
        "phenotypes": [
            {
                "name": r["spec"].name,
                "family": r["spec"].family,
                "source": r["spec"].source,
                "transform": r["spec"].transform,
                "description": r["spec"].description,
                "n_samples": int(r["count_row"]["n_samples"]),
                "covariate_design_terms": r["covariate_names"],
                "design_condition_number": r["condition_number"],
            }
            for r in run_results
        ],
        "excluded_primary_targets": {
            "binary_traits": (
                "sex, r_stroke, diabetes_sr_rx, hypertension_cum; "
                "Regular AREA is more natural."
            ),
            "technical_covariates": "rin, pmi, library/sequencing batch",
            "demographic_covariate": "age/age_death",
            "risk_factor": "APOE",
            "timing_trait": "age_first_ad_dx",
            "duplicate_stage_codings": "raw braaksc/ceradsc and stage ordinals",
        },
    }

    with open(root / "analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("MASTER SIGNIFICANT GENE COUNTS")
    print("=" * 80)
    print(master_counts.to_string(index=False))

    print("\nWrote:")
    for name in [
        "MASTER_weighted_area_results_long.csv",
        "MASTER_significant_gene_counts.csv",
        "MASTER_pairwise_overlap.csv",
        "MASTER_pairwise_z_correlations.csv",
        "MASTER_phenotype_summary.csv",
        "analysis_manifest.json",
    ]:
        print(f"  {root / name}")

    print(
        "\nNo pathway analysis was run. "
        "This is the production Weighted AREA result suite."
    )


if __name__ == "__main__":
    main()

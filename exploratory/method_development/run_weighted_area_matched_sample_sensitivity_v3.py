#!/usr/bin/env python3
"""
run_weighted_area_matched_sample_sensitivity.py
================================================

Matched-sample sensitivity analysis for the locked ROSMAP Weighted AREA method.

PURPOSE
-------
Ask how much the Weighted AREA results change when the analysis is restricted
to the same locked AD-vs-NCI participant cohort used for the controlled
DESeq2 / Regular AREA comparison.

This script deliberately REUSES the validated production implementation in:

    exploratory/area/method_development/
        run_weighted_area_full_phenotype_suite.py

It does not reimplement the Weighted AREA statistic.

WHAT IT DOES
------------
1. Finds or loads the authoritative matched AD-vs-NCI sample manifest.
2. Restricts the metadata to those participant IDs.
3. Re-runs all 9 Weighted AREA phenotypes with the exact locked method.
4. Compares FULL-COHORT vs MATCHED-COHORT results:
       - sample counts
       - genome-wide adjusted-Z Spearman correlation
       - direction concordance
       - FDR-significant gene counts
       - significant-gene Jaccard overlap
5. By default, runs the existing full-phenotype GSEA pipeline on the matched
   Weighted AREA outputs and compares FULL vs MATCHED pathway results:
       - pathway NES Spearman correlation
       - FDR-significant pathway counts
       - significant-pathway Jaccard overlap
6. Writes manuscript-friendly summary tables and a compact validation figure.

IMPORTANT INTERPRETATION
------------------------
The matched cohort is the binary endpoint AD-vs-NCI sample set.

Therefore:
- Cognitive_stage loses its MCI middle category in the matched analysis.
  Its FULL-vs-MATCHED comparison reflects BOTH cohort restriction and a
  change in phenotype support, not sample size alone.
- Continuous cognition / amyloid / tau phenotypes retain within-cohort
  quantitative variation and are cleaner cohort-restriction sensitivity tests.
- Pathology phenotypes may have fewer than 418 complete cases because the
  phenotype itself can be missing.

DEFAULT INPUTS
--------------
Expression:
    results/preprocessing/ROSMAP_AREA_normalized_counts_all_samples.csv

Metadata:
    results/pathology_geometry_abeta_tau/metadata/
        ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv

Full Weighted AREA:
    results/weighted_area_full_phenotype_suite

Full GSEA:
    results/weighted_area_full_phenotype_gsea

Production Weighted AREA runner:
    exploratory/area/method_development/
        run_weighted_area_full_phenotype_suite.py

GSEA runner:
    exploratory/enrichment/run_weighted_area_full_phenotype_gsea.py

OUTPUTS
-------
results/weighted_area_matched_sample_sensitivity/

    matched_weighted_area/
        <phenotype>/results.csv
        ... normal production per-phenotype outputs ...

    matched_gsea/
        ... normal full-phenotype GSEA outputs ...

    matched_cohort_manifest_used.csv
    matched_weighted_area_gene_sensitivity.csv
    matched_weighted_area_pathway_sensitivity_v2.csv
    matched_weighted_area_sensitivity_summary_v3.png
    matched_weighted_area_sensitivity_summary_v3.pdf
    analysis_manifest.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

PHENOTYPES = [
    "Cognitive_stage",
    "cogn_global_impairment",
    "mmse_impairment",
    "CERAD_equal",
    "CERAD_amyloid_calibrated",
    "amyloid_continuous",
    "Braak_equal",
    "Braak_tangle_calibrated",
    "tangle_continuous",
]

PHENOTYPE_LABELS = {
    "Cognitive_stage": "Cognitive\nstage",
    "cogn_global_impairment": "Global\ncognition",
    "mmse_impairment": "MMSE",
    "CERAD_equal": "CERAD\nequal",
    "CERAD_amyloid_calibrated": "CERAD\ncalibrated",
    "amyloid_continuous": "Amyloid\ncontinuous",
    "Braak_equal": "Braak\nequal",
    "Braak_tangle_calibrated": "Braak\ncalibrated",
    "tangle_continuous": "Tangling\ncontinuous",
}

LIBRARIES = {
    "Hallmark": "Hallmark_2020",
    "GO BP": "GO_Biological_Process_2025",
    "Reactome": "Reactome_Pathways_2024",
}

EXPECTED_MATCHED_N = 418


# =============================================================================
# CLI
# =============================================================================

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

    p.add_argument(
        "--weighted-runner",
        default=(
            "exploratory/area/method_development/"
            "run_weighted_area_full_phenotype_suite.py"
        ),
    )

    p.add_argument(
        "--gsea-runner",
        default=(
            "exploratory/enrichment/"
            "run_weighted_area_full_phenotype_gsea.py"
        ),
    )

    p.add_argument(
        "--full-weighted-root",
        default="results/weighted_area_full_phenotype_suite",
    )

    p.add_argument(
        "--full-gsea-root",
        default="results/weighted_area_full_phenotype_gsea",
    )

    p.add_argument(
        "--matched-manifest",
        default=None,
        help=(
            "Optional explicit matched-cohort manifest. If omitted, the script "
            "tries locked Regular AREA / DESeq2 manifest candidates and requires "
            f"exactly {EXPECTED_MATCHED_N} unique sample IDs."
        ),
    )

    p.add_argument(
        "--sample-col",
        default="sample_id",
    )

    p.add_argument(
        "--expression-id-col",
        default=None,
    )

    p.add_argument(
        "--rank-chunk-size",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--gsea-permutations",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--gsea-threads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--gsea-seed",
        type=int,
        default=20260908,
    )

    p.add_argument(
        "--skip-gsea",
        action="store_true",
        help="Run only gene-level matched sensitivity and skip matched GSEA.",
    )

    p.add_argument(
        "--outdir",
        default="results/weighted_area_matched_sample_sensitivity",
    )

    return p.parse_args()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require(path: Path, label: str = "file") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def safe_spearman(x, y) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return np.nan
    return float(pair["x"].corr(pair["y"], method="spearman"))


def jaccard(a: set[str], b: set[str]):
    inter = a & b
    union = a | b
    if not union:
        return np.nan, 0, 0
    return len(inter) / len(union), len(inter), len(union)


def direction_concordance(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    pair = pair[(pair["x"] != 0) & (pair["y"] != 0)]

    if len(pair) == 0:
        return np.nan, 0

    same = np.sign(pair["x"]) == np.sign(pair["y"])
    return float(same.mean()), int(len(pair))


def import_module_from_path(path: Path, module_name: str):
    require(path, "Python runner")

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# =============================================================================
# MATCHED COHORT MANIFEST
# =============================================================================

def read_manifest_sample_ids(
    path: Path,
    preferred_sample_col: str,
):
    require(path, "matched cohort manifest")

    df = pd.read_csv(path)

    candidates = [
        preferred_sample_col,
        "sample_id",
        "sample",
        "Sample",
        "RNAseq_sample_id",
    ]

    sample_col = next(
        (c for c in candidates if c in df.columns),
        None,
    )

    if sample_col is None:
        raise ValueError(
            f"Could not identify sample ID column in {path}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    ids = (
        df[sample_col]
        .astype(str)
        .str.strip()
    )

    ids = ids[
        ~ids.isin(["", "nan", "None", "<NA>"])
    ]

    if ids.duplicated().any():
        duplicated = ids[ids.duplicated()].tolist()[:10]
        raise ValueError(
            f"Matched manifest has duplicated sample IDs: {duplicated}"
        )

    return df, sample_col, list(ids)


def select_matched_manifest(args):
    if args.matched_manifest is not None:
        path = Path(args.matched_manifest)
        df, sample_col, ids = read_manifest_sample_ids(
            path,
            args.sample_col,
        )

        if len(ids) != EXPECTED_MATCHED_N:
            raise ValueError(
                f"Explicit matched manifest contains {len(ids)} unique samples; "
                f"expected {EXPECTED_MATCHED_N}."
            )

        return path, df, sample_col, ids

    candidates = [
        Path(
            "results/regular_area_covariate_adjusted/"
            "AD4_vs_NCI1/matched_sample_manifest.csv"
        ),
        Path(
            "results/regular_area_covariate_adjusted/"
            "AD4_vs_NCI1/sample_manifest.csv"
        ),
        Path(
            "results/deseq2_locked_covariates/"
            "AD4_vs_NCI1/AD4_vs_NCI1_sample_manifest.csv"
        ),
        Path(
            "results/deseq2_locked_covariates/"
            "AD4_vs_NCI1/sample_manifest.csv"
        ),
        Path(
            "results/deseq2/AD4_vs_NCI1/"
            "AD4_vs_NCI1_sample_manifest.csv"
        ),
    ]

    inspected = []

    for path in candidates:
        if not path.exists():
            inspected.append((str(path), "missing"))
            continue

        try:
            df, sample_col, ids = read_manifest_sample_ids(
                path,
                args.sample_col,
            )
        except Exception as exc:
            inspected.append((str(path), f"error: {exc}"))
            continue

        inspected.append((str(path), f"n={len(ids)}"))

        if len(ids) == EXPECTED_MATCHED_N:
            return path, df, sample_col, ids

    message = [
        "Could not automatically identify an authoritative 418-sample matched manifest.",
        "",
        "Candidates inspected:",
    ]

    for path, status in inspected:
        message.append(f"  {path}: {status}")

    message.extend(
        [
            "",
            "Supply the correct file explicitly with:",
            "",
            "  --matched-manifest PATH",
        ]
    )

    raise RuntimeError("\n".join(message))


# =============================================================================
# RUN MATCHED WEIGHTED AREA
# =============================================================================

def validate_metadata_and_subset(
    metadata_path: Path,
    matched_ids: list[str],
    sample_col: str,
):
    require(metadata_path, "metadata")

    metadata = pd.read_csv(metadata_path)

    if sample_col not in metadata.columns:
        raise ValueError(
            f"Metadata does not contain sample column '{sample_col}'."
        )

    metadata = metadata.copy()
    metadata[sample_col] = (
        metadata[sample_col]
        .astype(str)
        .str.strip()
    )

    if metadata[sample_col].duplicated().any():
        raise ValueError("Metadata sample IDs are duplicated.")

    meta_ids = set(metadata[sample_col])
    missing = [sid for sid in matched_ids if sid not in meta_ids]

    if missing:
        raise ValueError(
            f"{len(missing)} matched cohort IDs are absent from metadata. "
            f"Examples: {missing[:10]}"
        )

    order = {
        sid: i
        for i, sid in enumerate(matched_ids)
    }

    subset = (
        metadata[
            metadata[sample_col].isin(matched_ids)
        ]
        .copy()
    )

    subset["_matched_order"] = subset[sample_col].map(order)
    subset = subset.sort_values("_matched_order").drop(
        columns="_matched_order"
    )

    if len(subset) != EXPECTED_MATCHED_N:
        raise AssertionError(
            f"Metadata subset has n={len(subset)}, expected {EXPECTED_MATCHED_N}."
        )

    return metadata, subset


def run_matched_weighted_area(
    args,
    module,
    metadata_subset: pd.DataFrame,
    outroot: Path,
):
    expression_ids, gene_ids, x_all, expression_id_col = module.load_expression(
        args.expression,
        args.expression_id_col,
    )

    module_args = SimpleNamespace(
        sample_col=args.sample_col,
        expression_id_col=args.expression_id_col,
        fdr_threshold=args.fdr_threshold,
        rank_chunk_size=args.rank_chunk_size,
        keep_multibatch=False,
    )

    run_results = []

    for spec in module.PHENOTYPES:
        result = module.run_phenotype(
            spec=spec,
            expression_ids=expression_ids,
            gene_ids=gene_ids,
            x_all=x_all,
            metadata=metadata_subset,
            args=module_args,
            root=outroot,
        )

        run_results.append(result)

    # Write a few master outputs analogous to the production suite.
    master_counts = pd.DataFrame(
        [r["count_row"] for r in run_results]
    )

    master_counts.to_csv(
        outroot / "MASTER_significant_gene_counts.csv",
        index=False,
    )

    master_results = pd.concat(
        [r["results"] for r in run_results],
        ignore_index=True,
    )

    master_results.to_csv(
        outroot / "MASTER_weighted_area_results_long.csv",
        index=False,
    )

    phenotype_summary = pd.concat(
        [r["phenotype_summary"] for r in run_results],
        ignore_index=True,
    )

    phenotype_summary.to_csv(
        outroot / "MASTER_phenotype_summary.csv",
        index=False,
    )

    return run_results


# =============================================================================
# GENE-LEVEL FULL VS MATCHED COMPARISON
# =============================================================================

def load_weighted_results(root: Path, phenotype: str):
    path = root / phenotype / "results.csv"
    require(path, f"{phenotype} Weighted AREA result")

    df = pd.read_csv(path)

    needed = [
        "gene_id",
        "adjusted_AREA_Z",
        "adjusted_padj_BH",
    ]

    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing}"
        )

    out = df[needed].copy()
    out["gene_id"] = out["gene_id"].astype(str)
    out["adjusted_AREA_Z"] = pd.to_numeric(
        out["adjusted_AREA_Z"],
        errors="coerce",
    )
    out["adjusted_padj_BH"] = pd.to_numeric(
        out["adjusted_padj_BH"],
        errors="coerce",
    )

    return out.drop_duplicates("gene_id").set_index("gene_id")


def compare_gene_results(
    full_root: Path,
    matched_root: Path,
):
    rows = []

    for phenotype in PHENOTYPES:
        full = load_weighted_results(
            full_root,
            phenotype,
        )

        matched = load_weighted_results(
            matched_root,
            phenotype,
        )

        joined = (
            full.rename(
                columns={
                    "adjusted_AREA_Z": "full_Z",
                    "adjusted_padj_BH": "full_q",
                }
            )
            .join(
                matched.rename(
                    columns={
                        "adjusted_AREA_Z": "matched_Z",
                        "adjusted_padj_BH": "matched_q",
                    }
                ),
                how="inner",
            )
            .dropna(
                subset=[
                    "full_Z",
                    "matched_Z",
                ]
            )
        )

        rho = safe_spearman(
            joined["full_Z"],
            joined["matched_Z"],
        )

        direction, n_direction = direction_concordance(
            joined["full_Z"],
            joined["matched_Z"],
        )

        full_sig = set(
            joined.index[
                joined["full_q"] < 0.05
            ]
        )

        matched_sig = set(
            joined.index[
                joined["matched_q"] < 0.05
            ]
        )

        jac, inter, union = jaccard(
            full_sig,
            matched_sig,
        )

        full_n_samples = int(
            pd.read_csv(
                full_root
                / phenotype
                / "results.csv",
                nrows=1,
            )["n_samples"].iloc[0]
        )

        matched_n_samples = int(
            pd.read_csv(
                matched_root
                / phenotype
                / "results.csv",
                nrows=1,
            )["n_samples"].iloc[0]
        )

        rows.append(
            {
                "phenotype": phenotype,
                "full_n_samples": full_n_samples,
                "matched_n_samples": matched_n_samples,
                "n_common_genes": int(len(joined)),
                "spearman_full_vs_matched_Z": rho,
                "direction_concordance": direction,
                "n_direction_compared": n_direction,
                "full_n_fdr_significant": len(full_sig),
                "matched_n_fdr_significant": len(matched_sig),
                "significant_intersection": inter,
                "significant_union": union,
                "significant_jaccard": jac,
                "matched_fraction_of_full_hits": (
                    inter / len(full_sig)
                    if full_sig
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MATCHED GSEA
# =============================================================================

def run_matched_gsea(
    args,
    matched_weighted_root: Path,
    matched_gsea_root: Path,
):
    runner = Path(args.gsea_runner)
    require(runner, "GSEA runner")

    cmd = [
        sys.executable,
        str(runner),
        "--weighted-root",
        str(matched_weighted_root),
        "--outdir",
        str(matched_gsea_root),
        "--permutations",
        str(args.gsea_permutations),
        "--threads",
        str(args.gsea_threads),
        "--seed",
        str(args.gsea_seed),
        "--fdr",
        str(args.fdr_threshold),
    ]

    print("\nRunning matched-cohort genome-wide preranked GSEA:")
    print(" ".join(cmd))
    print()

    subprocess.run(
        cmd,
        check=True,
    )


def resolve_gsea_columns(df: pd.DataFrame):
    term_candidates = [
        "pathway",
        "Pathway",
        "Term",
        "term",
        "Name",
    ]

    nes_candidates = [
        "NES",
        "nes",
    ]

    q_candidates = [
        "FDR_qvalue",
        "FDR q-val",
        "FDR q-value",
        "FDR",
        "fdr",
        "qvalue",
        "q_value",
    ]

    term_col = next(
        (c for c in term_candidates if c in df.columns),
        None,
    )

    nes_col = next(
        (c for c in nes_candidates if c in df.columns),
        None,
    )

    q_col = next(
        (c for c in q_candidates if c in df.columns),
        None,
    )

    if term_col is None or nes_col is None or q_col is None:
        raise ValueError(
            "Could not resolve GSEA term/NES/FDR columns. "
            f"Available: {', '.join(df.columns)}"
        )

    return term_col, nes_col, q_col


def load_gsea_result(
    root: Path,
    library_dir: str,
    phenotype: str,
):
    path = (
        root
        / library_dir
        / f"Weighted_{phenotype}"
        / "gsea_all_results.csv"
    )

    require(path, "GSEA result")

    df = pd.read_csv(path)

    term_col, nes_col, q_col = resolve_gsea_columns(df)

    out = df[[term_col, nes_col, q_col]].copy()
    out.columns = ["term", "NES", "q"]

    out["term"] = out["term"].astype(str).str.strip()
    out["NES"] = pd.to_numeric(out["NES"], errors="coerce")
    out["q"] = pd.to_numeric(out["q"], errors="coerce")

    out = out.dropna(subset=["term", "NES"])

    # A valid per-pathway GSEA table should contain many unique pathway terms.
    # This catches accidental selection of a constant metadata column such as
    # gseapy's 'Name' field instead of the actual 'pathway' field.
    n_unique_terms = out["term"].nunique()
    if n_unique_terms < 5 and len(out) >= 5:
        raise ValueError(
            f"GSEA term column '{term_col}' produced only {n_unique_terms} unique "
            f"terms across {len(out)} rows. Available columns: {', '.join(df.columns)}"
        )

    return (
        out.drop_duplicates("term")
        .set_index("term")
    )


def compare_pathway_results(
    full_gsea_root: Path,
    matched_gsea_root: Path,
):
    rows = []

    for phenotype in PHENOTYPES:
        for library_label, library_dir in LIBRARIES.items():
            full = load_gsea_result(
                full_gsea_root,
                library_dir,
                phenotype,
            )

            matched = load_gsea_result(
                matched_gsea_root,
                library_dir,
                phenotype,
            )

            joined = (
                full.rename(
                    columns={
                        "NES": "full_NES",
                        "q": "full_q",
                    }
                )
                .join(
                    matched.rename(
                        columns={
                            "NES": "matched_NES",
                            "q": "matched_q",
                        }
                    ),
                    how="inner",
                )
                .dropna(
                    subset=[
                        "full_NES",
                        "matched_NES",
                    ]
                )
            )

            full_sig = set(
                joined.index[
                    joined["full_q"] < 0.05
                ]
            )

            matched_sig = set(
                joined.index[
                    joined["matched_q"] < 0.05
                ]
            )

            jac, inter, union = jaccard(
                full_sig,
                matched_sig,
            )

            rows.append(
                {
                    "phenotype": phenotype,
                    "library": library_label,
                    "n_common_pathways": int(len(joined)),
                    "spearman_full_vs_matched_NES": safe_spearman(
                        joined["full_NES"],
                        joined["matched_NES"],
                    ),
                    "full_n_fdr_significant": len(full_sig),
                    "matched_n_fdr_significant": len(matched_sig),
                    "significant_intersection": inter,
                    "significant_union": union,
                    "significant_jaccard": jac,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# FIGURE
# =============================================================================

def make_summary_figure(
    gene_summary: pd.DataFrame,
    pathway_summary: pd.DataFrame | None,
    output_png: Path,
    output_pdf: Path,
):
    """
    Publication-style matched-sample sensitivity summary.

    Design priorities:
    - no overlapping phenotype labels
    - enough horizontal room for nine phenotypes
    - consistent panel sizes
    - compact paper-ready typography
    - separate legend placement
    """
    fig = plt.figure(figsize=(7.2, 7.8))

    gs = fig.add_gridspec(
        3,
        2,
        left=0.10,
        right=0.985,
        top=0.89,
        bottom=0.14,
        wspace=0.33,
        hspace=0.78,
    )

    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        ]
    )

    x = np.arange(len(PHENOTYPES))

    # Short labels are essential at paper width.
    short_labels = [
        "Cog. stage",
        "Global cog.",
        "MMSE",
        "CERAD eq.",
        "CERAD cal.",
        "Amyloid cont.",
        "Braak eq.",
        "Braak cal.",
        "Tangling cont.",
    ]

    def finish_axis(ax, letter, title, ylabel=None, ylim=None):
        ax.set_title(
            title,
            fontsize=7.2,
            fontweight="bold",
            pad=7,
        )

        ax.text(
            -0.12,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            va="top",
        )

        if ylabel is not None:
            ax.set_ylabel(
                ylabel,
                fontsize=5.8,
            )

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xticks(x)
        ax.set_xticklabels(
            short_labels,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=4.9,
        )

        ax.tick_params(
            axis="y",
            labelsize=5.3,
        )

        ax.grid(
            axis="y",
            alpha=0.16,
            linewidth=0.5,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # A. Sample retention
    # ------------------------------------------------------------------
    ax = axes[0, 0]

    full_n = gene_summary["full_n_samples"].to_numpy()
    matched_n = gene_summary["matched_n_samples"].to_numpy()

    ax.bar(
        x - 0.18,
        full_n,
        width=0.36,
        label="Full cohort",
    )

    ax.bar(
        x + 0.18,
        matched_n,
        width=0.36,
        label="Matched AD-vs-NCI cohort",
    )

    finish_axis(
        ax,
        "A",
        "Sample retention",
        ylabel="Participants",
    )

    ax.legend(
        frameon=False,
        fontsize=5.0,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
    )

    # ------------------------------------------------------------------
    # B. Genome-wide Z concordance
    # ------------------------------------------------------------------
    ax = axes[0, 1]

    vals = gene_summary["spearman_full_vs_matched_Z"].to_numpy()
    bars = ax.bar(
        x,
        vals,
        width=0.58,
    )

    for bar, value in zip(bars, vals):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=4.5,
            )

    finish_axis(
        ax,
        "B",
        "Genome-wide Z concordance",
        ylabel="Spearman ρ",
        ylim=(0, 1.05),
    )

    # ------------------------------------------------------------------
    # C. Direction concordance
    # ------------------------------------------------------------------
    ax = axes[1, 0]

    vals = gene_summary["direction_concordance"].to_numpy()
    bars = ax.bar(
        x,
        vals,
        width=0.58,
    )

    for bar, value in zip(bars, vals):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=4.5,
            )

    finish_axis(
        ax,
        "C",
        "Direction concordance",
        ylabel="Fraction same sign",
        ylim=(0, 1.05),
    )

    # ------------------------------------------------------------------
    # D. Gene-level discovery
    # ------------------------------------------------------------------
    ax = axes[1, 1]

    full_hits = gene_summary["full_n_fdr_significant"].to_numpy()
    matched_hits = gene_summary["matched_n_fdr_significant"].to_numpy()

    ax.bar(
        x - 0.18,
        full_hits,
        width=0.36,
        label="Full cohort",
    )

    ax.bar(
        x + 0.18,
        matched_hits,
        width=0.36,
        label="Matched cohort",
    )

    finish_axis(
        ax,
        "D",
        "Gene-level discovery",
        ylabel="FDR-significant genes",
    )

    # ------------------------------------------------------------------
    # E. Significant-gene overlap
    # ------------------------------------------------------------------
    ax = axes[2, 0]

    vals = gene_summary["significant_jaccard"].to_numpy()
    bars = ax.bar(
        x,
        vals,
        width=0.58,
    )

    for bar, value in zip(bars, vals):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=4.5,
            )

    finish_axis(
        ax,
        "E",
        "Significant-gene overlap",
        ylabel="Jaccard similarity",
        ylim=(0, 1.05),
    )

    # ------------------------------------------------------------------
    # F. Pathway NES concordance
    # ------------------------------------------------------------------
    ax = axes[2, 1]

    if pathway_summary is not None and len(pathway_summary) > 0:
        path_mean = (
            pathway_summary
            .groupby("phenotype")["spearman_full_vs_matched_NES"]
            .mean()
            .reindex(PHENOTYPES)
        )

        vals = path_mean.to_numpy()

        bars = ax.bar(
            x,
            vals,
            width=0.58,
        )

        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=4.5,
                )

        finish_axis(
            ax,
            "F",
            "Pathway NES concordance",
            ylabel="Mean Spearman ρ\nacross 3 libraries",
            ylim=(0, 1.05),
        )

    else:
        finish_axis(
            ax,
            "F",
            "Pathway NES concordance",
            ylabel="Mean Spearman ρ\nacross 3 libraries",
            ylim=(0, 1.05),
        )

        ax.text(
            0.5,
            0.50,
            "GSEA unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=6.0,
            color="#666666",
        )

    # ------------------------------------------------------------------
    # Global title / notes
    # ------------------------------------------------------------------
    fig.suptitle(
        "Weighted AREA matched-sample sensitivity",
        fontsize=10.4,
        fontweight="bold",
        y=0.982,
    )

    fig.text(
        0.5,
        0.955,
        (
            "Full phenotype cohorts compared with the locked n=418 "
            "AD-vs-NCI cohort"
        ),
        ha="center",
        fontsize=6.0,
        color="#3F3F3F",
    )

    fig.text(
        0.5,
        0.032,
        (
            "Cognitive_stage is diagnostic only: restricting to AD-vs-NCI removes "
            "the MCI category and changes both cohort and phenotype support."
        ),
        ha="center",
        fontsize=5.0,
        color="#5A5A5A",
    )

    fig.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    matched_weighted_root = outroot / "matched_weighted_area"
    matched_gsea_root = outroot / "matched_gsea"

    matched_weighted_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Resolve matched cohort
    # -------------------------------------------------------------------------
    (
        matched_manifest_path,
        matched_manifest_df,
        matched_manifest_sample_col,
        matched_ids,
    ) = select_matched_manifest(args)

    print("=" * 100)
    print("WEIGHTED AREA MATCHED-SAMPLE SENSITIVITY")
    print("=" * 100)
    print()
    print(f"Matched manifest: {matched_manifest_path}")
    print(f"Matched cohort n: {len(matched_ids)}")
    print()

    matched_manifest_used = pd.DataFrame(
        {
            "position": np.arange(len(matched_ids)),
            "sample_id": matched_ids,
        }
    )

    matched_manifest_used.to_csv(
        outroot / "matched_cohort_manifest_used.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Metadata subset
    # -------------------------------------------------------------------------
    _, metadata_subset = validate_metadata_and_subset(
        Path(args.metadata),
        matched_ids,
        args.sample_col,
    )

    metadata_subset.to_csv(
        outroot / "matched_metadata_subset.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Re-run exact production Weighted AREA method
    # -------------------------------------------------------------------------
    weighted_module = import_module_from_path(
        Path(args.weighted_runner),
        "locked_weighted_area_runner",
    )

    run_matched_weighted_area(
        args=args,
        module=weighted_module,
        metadata_subset=metadata_subset,
        outroot=matched_weighted_root,
    )

    # -------------------------------------------------------------------------
    # Gene-level comparison
    # -------------------------------------------------------------------------
    gene_summary = compare_gene_results(
        Path(args.full_weighted_root),
        matched_weighted_root,
    )

    gene_summary.to_csv(
        outroot / "matched_weighted_area_gene_sensitivity.csv",
        index=False,
    )

    print("\nGENE-LEVEL FULL VS MATCHED SUMMARY")
    print()
    print(gene_summary.to_string(index=False))
    print()

    # -------------------------------------------------------------------------
    # GSEA sensitivity
    # -------------------------------------------------------------------------
    pathway_summary = None

    if not args.skip_gsea:
        run_matched_gsea(
            args,
            matched_weighted_root,
            matched_gsea_root,
        )

        pathway_summary = compare_pathway_results(
            Path(args.full_gsea_root),
            matched_gsea_root,
        )

        pathway_summary.to_csv(
            outroot / "matched_weighted_area_pathway_sensitivity_v2.csv",
            index=False,
        )

        print("\nPATHWAY FULL VS MATCHED SUMMARY")
        print()
        print(pathway_summary.to_string(index=False))
        print()

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    figure_png = (
        outroot
        / "matched_weighted_area_sensitivity_summary_v3.png"
    )

    figure_pdf = (
        outroot
        / "matched_weighted_area_sensitivity_summary_v3.pdf"
    )

    make_summary_figure(
        gene_summary,
        pathway_summary,
        figure_png,
        figure_pdf,
    )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------
    manifest_lines = [
        "Weighted AREA matched-sample sensitivity",
        "",
        f"Matched manifest: {matched_manifest_path}",
        f"Matched cohort n: {len(matched_ids)}",
        f"Expression: {args.expression}",
        f"Metadata: {args.metadata}",
        f"Production Weighted runner: {args.weighted_runner}",
        f"Full Weighted root: {args.full_weighted_root}",
        f"Matched Weighted root: {matched_weighted_root}",
        f"GSEA run: {not args.skip_gsea}",
        f"Full GSEA root: {args.full_gsea_root}",
        f"Matched GSEA root: {matched_gsea_root}",
        "",
        "Interpretation note:",
        (
            "Cognitive_stage loses its MCI category after restriction to the "
            "binary endpoint cohort; its comparison therefore reflects both "
            "cohort restriction and phenotype-support change."
        ),
    ]

    (
        outroot
        / "analysis_manifest.txt"
    ).write_text(
        "\n".join(manifest_lines)
        + "\n"
    )

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print()
    print(f"Wrote: {gene_summary.shape[0]} phenotype gene-sensitivity rows")
    print(f"Wrote: {figure_png}")
    print(f"Wrote: {figure_pdf}")

    if pathway_summary is not None:
        print(
            f"Wrote: {len(pathway_summary)} pathway-sensitivity rows"
        )


if __name__ == "__main__":
    main()

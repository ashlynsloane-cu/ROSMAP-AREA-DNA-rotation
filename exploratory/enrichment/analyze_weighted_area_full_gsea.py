#!/usr/bin/env python3
"""
analyze_weighted_area_full_gsea.py
==================================

Post-GSEA synthesis for the full 9-phenotype Weighted AREA suite.

This script DOES NOT rerun GSEA. It analyzes outputs from:
    results/weighted_area_full_phenotype_gsea/

It performs five major tasks:

1) VALIDATE DIRECTIONALITY
   - checks the sign relationship between adjusted_AREA_Z and
     adjusted_partial_rank_correlation for every phenotype
   - checks whether the sign convention is stable across the full suite
   - summarizes positive/negative gene-level directions

2) SUMMARIZE PATHWAY RECURRENCE
   - identifies pathways significant across many of the 9 Weighted phenotypes
   - summarizes recurrence within cognition, amyloid, and tau families

3) FIND FAMILY-SPECIFIC PATHWAYS
   - cognition-only
   - amyloid-only
   - tau-only
   - amyloid+tau but not cognition
   - cognition+amyloid but not tau
   - cognition+tau but not amyloid
   - shared across all three families

4) COMPARE ORDINAL/CALIBRATED/CONTINUOUS PATHOLOGY
   - amyloid: CERAD_equal vs CERAD_amyloid_calibrated vs amyloid_continuous
   - tau: Braak_equal vs Braak_tangle_calibrated vs tangle_continuous
   - counts continuous-only / ordinal-only / shared
   - flags direction flips in NES

5) COMPARE WEIGHTED AREA WITH DESeq2 / REGULAR AREA
   - pathways significant in >=1 Weighted phenotype but not DESeq2
   - pathways significant in both AREA approaches but not DESeq2
   - prioritizes recurrent AREA-specific pathways

Outputs
-------
results/weighted_area_full_phenotype_gsea_synthesis/

    directionality_validation.csv
    gsea_count_summary_pivot.csv
    family_specificity_summary.csv
    pathology_encoding_summary.csv
    weighted_vs_deseq2_summary.csv

    <library>_family_specificity.csv
    <library>_pathology_encoding_detailed.csv
    <library>_weighted_specific_vs_DESeq2_prioritized.csv
    <library>_both_AREA_not_DESeq2_prioritized.csv
    <library>_top_recurrent_pathways.csv

    figures/
        gsea_significant_pathway_counts.png
        family_specificity_counts_<library>.png
        pathology_encoding_counts_<library>.png
        top_recurrent_NES_heatmap_<library>.png

Notes
-----
- This is a synthesis/characterization step, not an independent inferential test.
- Family-specific labels are based on FDR<0.05 significance patterns.
- NES sign is interpreted using the prior ranking convention:
      positive = higher expression with greater disease severity/pathology
      negative = lower expression with greater disease severity/pathology
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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

FAMILIES = {
    "cognition": [
        "Cognitive_stage",
        "cogn_global_impairment",
        "mmse_impairment",
    ],
    "amyloid": [
        "CERAD_equal",
        "CERAD_amyloid_calibrated",
        "amyloid_continuous",
    ],
    "tau": [
        "Braak_equal",
        "Braak_tangle_calibrated",
        "tangle_continuous",
    ],
}

PATHOLOGY_ENCODINGS = {
    "amyloid": {
        "equal": "CERAD_equal",
        "calibrated": "CERAD_amyloid_calibrated",
        "continuous": "amyloid_continuous",
    },
    "tau": {
        "equal": "Braak_equal",
        "calibrated": "Braak_tangle_calibrated",
        "continuous": "tangle_continuous",
    },
}

LIBRARIES = [
    "Hallmark_2020",
    "GO_Biological_Process_2025",
    "Reactome_Pathways_2024",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--gsea-root",
        default="results/weighted_area_full_phenotype_gsea",
    )

    p.add_argument(
        "--weighted-root",
        default="results/weighted_area_full_phenotype_suite",
    )

    p.add_argument(
        "--fdr",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--top-recurrent",
        type=int,
        default=30,
    )

    p.add_argument(
        "--outdir",
        default="results/weighted_area_full_phenotype_gsea_synthesis",
    )

    return p.parse_args()


def safe_bool(series):
    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def validate_directionality(weighted_root):
    """
    Empirically validate the AREA Z sign convention.

    We expect adjusted_AREA_Z and adjusted_partial_rank_correlation
    to have opposite signs if positive Z means lower expression with
    increasing phenotype severity.

    This does NOT independently validate biological phenotype orientation.
    It validates that the statistic/rank-correlation sign relationship
    is internally stable across phenotypes.
    """
    rows = []

    for phenotype in PHENOTYPES:
        path = (
            Path(weighted_root)
            / phenotype
            / "results.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Weighted AREA results: {path}"
            )

        df = pd.read_csv(path)

        required = [
            "adjusted_AREA_Z",
            "adjusted_partial_rank_correlation",
            "adjusted_pvalue",
            "adjusted_padj_BH",
            "n_samples",
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{phenotype} missing columns: {missing}"
            )

        z = pd.to_numeric(
            df["adjusted_AREA_Z"],
            errors="coerce",
        )

        rho = pd.to_numeric(
            df["adjusted_partial_rank_correlation"],
            errors="coerce",
        )

        valid = (
            z.notna()
            & rho.notna()
            & np.isfinite(z)
            & np.isfinite(rho)
            & (z != 0)
            & (rho != 0)
        )

        zv = z.loc[valid]
        rv = rho.loc[valid]

        opposite = (
            np.sign(zv)
            == -np.sign(rv)
        )

        same = (
            np.sign(zv)
            == np.sign(rv)
        )

        corr = spearmanr(
            zv,
            rv,
            nan_policy="omit",
        ).statistic

        ns = (
            pd.to_numeric(
                df["n_samples"],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        n_samples = (
            int(ns[0])
            if len(ns) == 1
            else np.nan
        )

        padj = pd.to_numeric(
            df["adjusted_padj_BH"],
            errors="coerce",
        )

        sig = padj < 0.05

        rows.append(
            {
                "phenotype": phenotype,
                "n_samples": n_samples,
                "n_genes": len(df),
                "n_valid_sign_pairs": int(valid.sum()),
                "spearman_Z_vs_partial_rank_rho": corr,
                "fraction_opposite_sign_Z_vs_rho": opposite.mean(),
                "fraction_same_sign_Z_vs_rho": same.mean(),
                "n_FDR_sig": int(sig.sum()),
                "n_positive_Z_FDR_sig": int(
                    ((z > 0) & sig).sum()
                ),
                "n_negative_Z_FDR_sig": int(
                    ((z < 0) & sig).sum()
                ),
                "n_positive_partial_rho_FDR_sig": int(
                    ((rho > 0) & sig).sum()
                ),
                "n_negative_partial_rho_FDR_sig": int(
                    ((rho < 0) & sig).sum()
                ),
                "direction_convention_pass": bool(
                    opposite.mean() > 0.99
                    and corr < -0.99
                ),
            }
        )

    return pd.DataFrame(rows)


def load_matrix(gsea_root, library):
    path = (
        Path(gsea_root)
        / f"{library}_weighted_phenotype_matrix.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing GSEA matrix: {path}"
        )

    df = pd.read_csv(path)

    for phenotype in PHENOTYPES:
        sig_col = f"{phenotype}_sig"
        if sig_col in df.columns:
            df[sig_col] = safe_bool(
                df[sig_col]
            )

    for c in [
        "DESeq2_sig",
        "Regular_AREA_sig",
        "weighted_any_significant",
        "weighted_all9_significant",
        "weighted_specific_vs_DESeq2",
        "weighted_shared_with_DESeq2",
        "both_AREA_any_but_DESeq2_not",
    ]:
        if c in df.columns:
            df[c] = safe_bool(
                df[c]
            )

    return df


def add_family_specificity(df):
    out = df.copy()

    for family, phenotypes in FAMILIES.items():
        sig_cols = [
            f"{p}_sig"
            for p in phenotypes
        ]

        out[
            f"n_{family}"
        ] = (
            out[
                sig_cols
            ]
            .sum(axis=1)
        )

        out[
            f"any_{family}"
        ] = (
            out[
                f"n_{family}"
            ] > 0
        )

    c = out["any_cognition"]
    a = out["any_amyloid"]
    t = out["any_tau"]

    conditions = [
        c & ~a & ~t,
        ~c & a & ~t,
        ~c & ~a & t,
        c & a & ~t,
        c & ~a & t,
        ~c & a & t,
        c & a & t,
    ]

    labels = [
        "cognition_only",
        "amyloid_only",
        "tau_only",
        "cognition_plus_amyloid",
        "cognition_plus_tau",
        "amyloid_plus_tau",
        "shared_all_three_families",
    ]

    out[
        "family_specificity"
    ] = np.select(
        conditions,
        labels,
        default="not_significant_in_weighted",
    )

    return out


def pathology_encoding_table(df):
    rows = []

    for family, enc in PATHOLOGY_ENCODINGS.items():
        eq = enc["equal"]
        cal = enc["calibrated"]
        con = enc["continuous"]

        x = pd.DataFrame(
            {
                "pathway": df["pathway"],
                "family": family,
                "equal_sig": df[f"{eq}_sig"],
                "calibrated_sig": df[f"{cal}_sig"],
                "continuous_sig": df[f"{con}_sig"],
                "equal_NES": pd.to_numeric(
                    df[f"{eq}_NES"],
                    errors="coerce",
                ),
                "calibrated_NES": pd.to_numeric(
                    df[f"{cal}_NES"],
                    errors="coerce",
                ),
                "continuous_NES": pd.to_numeric(
                    df[f"{con}_NES"],
                    errors="coerce",
                ),
                "equal_FDR": pd.to_numeric(
                    df[f"{eq}_FDR"],
                    errors="coerce",
                ),
                "calibrated_FDR": pd.to_numeric(
                    df[f"{cal}_FDR"],
                    errors="coerce",
                ),
                "continuous_FDR": pd.to_numeric(
                    df[f"{con}_FDR"],
                    errors="coerce",
                ),
            }
        )

        x["continuous_only"] = (
            x["continuous_sig"]
            & ~x["equal_sig"]
            & ~x["calibrated_sig"]
        )

        x["ordinal_any_only"] = (
            (
                x["equal_sig"]
                | x["calibrated_sig"]
            )
            & ~x["continuous_sig"]
        )

        x["all_three_encodings"] = (
            x["equal_sig"]
            & x["calibrated_sig"]
            & x["continuous_sig"]
        )

        x["equal_and_calibrated_only"] = (
            x["equal_sig"]
            & x["calibrated_sig"]
            & ~x["continuous_sig"]
        )

        x["continuous_plus_equal"] = (
            x["continuous_sig"]
            & x["equal_sig"]
        )

        x["continuous_plus_calibrated"] = (
            x["continuous_sig"]
            & x["calibrated_sig"]
        )

        x["equal_vs_continuous_direction_flip"] = (
            x["equal_NES"].notna()
            & x["continuous_NES"].notna()
            & (
                np.sign(
                    x["equal_NES"]
                )
                != np.sign(
                    x["continuous_NES"]
                )
            )
        )

        x["calibrated_vs_continuous_direction_flip"] = (
            x["calibrated_NES"].notna()
            & x["continuous_NES"].notna()
            & (
                np.sign(
                    x["calibrated_NES"]
                )
                != np.sign(
                    x["continuous_NES"]
                )
            )
        )

        rows.append(x)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def plot_count_summary(summary, outpath):
    methods = summary["method"].tolist()
    values = summary["fdr_significant_n"].to_numpy()

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.bar(
        methods,
        values,
    )

    ax.set_ylabel(
        "FDR-significant pathways"
    )

    ax.set_title(
        "Genome-wide GSEA: significant pathway counts"
    )

    ax.tick_params(
        axis="x",
        rotation=70,
    )

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_family_specificity_counts(df, library, outpath):
    counts = (
        df[
            "family_specificity"
        ]
        .value_counts()
        .drop(
            labels=[
                "not_significant_in_weighted"
            ],
            errors="ignore",
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_ylabel(
        "Pathways"
    )

    ax.set_title(
        f"{library}: phenotype-family specificity"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_pathology_encoding_counts(detail, library, outpath):
    rows = []

    for family in [
        "amyloid",
        "tau",
    ]:
        sub = detail.loc[
            detail["family"] == family
        ]

        rows.append(
            {
                "family": family,
                "continuous_only": int(
                    sub[
                        "continuous_only"
                    ].sum()
                ),
                "ordinal_any_only": int(
                    sub[
                        "ordinal_any_only"
                    ].sum()
                ),
                "all_three": int(
                    sub[
                        "all_three_encodings"
                    ].sum()
                ),
            }
        )

    x = pd.DataFrame(rows)

    labels = x["family"].tolist()
    positions = np.arange(
        len(labels)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        positions - width,
        x["continuous_only"],
        width,
        label="Continuous only",
    )

    ax.bar(
        positions,
        x["ordinal_any_only"],
        width,
        label="Ordinal/calibrated only",
    )

    ax.bar(
        positions + width,
        x["all_three"],
        width,
        label="All 3 encodings",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Pathways"
    )

    ax.set_title(
        f"{library}: continuous vs ordinal pathology"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_top_recurrent_heatmap(df, library, outpath, top_n):
    x = df.loc[
        df[
            "n_weighted_phenotypes_significant"
        ] > 0
    ].copy()

    if x.empty:
        return

    x = x.sort_values(
        [
            "n_weighted_phenotypes_significant",
            "n_tau",
            "n_amyloid",
            "n_cognition",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        kind="mergesort",
    ).head(
        top_n
    )

    nes_cols = [
        f"{p}_NES"
        for p in PHENOTYPES
    ]

    values = (
        x[
            nes_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    fig_height = max(
        7,
        0.32 * len(x) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(13, fig_height)
    )

    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            len(PHENOTYPES)
        )
    )

    ax.set_xticklabels(
        PHENOTYPES,
        rotation=55,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(x)
        )
    )

    ax.set_yticklabels(
        x["pathway"]
    )

    ax.set_title(
        f"{library}: NES of top recurrent Weighted AREA pathways"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Normalized enrichment score (NES)"
    )

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figdir = (
        outdir
        / "figures"
    )

    figdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gsea_root = Path(
        args.gsea_root
    )

    summary_path = (
        gsea_root
        / "gsea_summary.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing GSEA summary: {summary_path}"
        )

    summary = pd.read_csv(
        summary_path
    )

    # ------------------------------------------------------------
    # 1) Directionality validation
    # ------------------------------------------------------------
    direction = validate_directionality(
        args.weighted_root
    )

    direction.to_csv(
        outdir
        / "directionality_validation.csv",
        index=False,
    )

    print("=" * 100)
    print("DIRECTIONALITY VALIDATION")
    print("=" * 100)

    print(
        direction[
            [
                "phenotype",
                "n_samples",
                "spearman_Z_vs_partial_rank_rho",
                "fraction_opposite_sign_Z_vs_rho",
                "direction_convention_pass",
            ]
        ].to_string(
            index=False
        )
    )

    if not direction[
        "direction_convention_pass"
    ].all():
        print(
            "\nWARNING: at least one phenotype failed "
            "the expected Z/rho sign-convention check."
        )

    # ------------------------------------------------------------
    # 2) Overall GSEA summary pivot
    # ------------------------------------------------------------
    weighted_summary = summary.loc[
        summary[
            "method"
        ].astype(str).str.startswith(
            "Weighted_"
        )
    ].copy()

    if weighted_summary.empty:
        raise ValueError(
            "No Weighted phenotype rows found in gsea_summary.csv"
        )

    weighted_summary[
        "phenotype"
    ] = (
        weighted_summary[
            "method"
        ]
        .str.replace(
            "Weighted_",
            "",
            regex=False,
        )
    )

    weighted_summary[
        "family"
    ] = weighted_summary[
        "phenotype"
    ].map(
        {
            p: family
            for family, phenos in FAMILIES.items()
            for p in phenos
        }
    )

    pivot = weighted_summary.pivot_table(
        index=[
            "phenotype",
            "family",
        ],
        columns="library",
        values="fdr_significant_n",
        aggfunc="first",
    ).reset_index()

    pivot.to_csv(
        outdir
        / "gsea_count_summary_pivot.csv",
        index=False,
    )

    # Figure across all methods/libraries.
    plot_input = (
        summary[
            [
                "method",
                "library",
                "fdr_significant_n",
            ]
        ]
        .copy()
    )

    plot_input["plot_label"] = (
        plot_input["method"].astype(str)
        + " | "
        + plot_input["library"].astype(str)
    )

    plot_input = (
        plot_input[
            [
                "plot_label",
                "fdr_significant_n",
            ]
        ]
        .rename(
            columns={
                "plot_label": "method",
            }
        )
    )

    plot_count_summary(
        plot_input,
        figdir
        / "gsea_significant_pathway_counts.png",
    )

    # ------------------------------------------------------------
    # 3-5) Per-library synthesis
    # ------------------------------------------------------------
    family_summary_rows = []
    pathology_summary_rows = []
    method_summary_rows = []

    for library in LIBRARIES:
        matrix = load_matrix(
            gsea_root,
            library,
        )

        matrix = add_family_specificity(
            matrix
        )

        matrix.to_csv(
            outdir
            / f"{library}_family_specificity.csv",
            index=False,
        )

        counts = (
            matrix[
                "family_specificity"
            ]
            .value_counts()
        )

        for category, n in counts.items():
            if category == "not_significant_in_weighted":
                continue

            family_summary_rows.append(
                {
                    "library": library,
                    "family_specificity": category,
                    "n_pathways": int(n),
                }
            )

        plot_family_specificity_counts(
            matrix,
            library,
            figdir
            / f"family_specificity_counts_{library}.png",
        )

        # Top recurrent pathways.
        recurrent = matrix.loc[
            matrix[
                "n_weighted_phenotypes_significant"
            ] > 0
        ].copy()

        recurrent = recurrent.sort_values(
            [
                "n_weighted_phenotypes_significant",
                "n_tau",
                "n_amyloid",
                "n_cognition",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            kind="mergesort",
        )

        recurrent.head(
            max(
                args.top_recurrent,
                100,
            )
        ).to_csv(
            outdir
            / f"{library}_top_recurrent_pathways.csv",
            index=False,
        )

        plot_top_recurrent_heatmap(
            matrix,
            library,
            figdir
            / f"top_recurrent_NES_heatmap_{library}.png",
            args.top_recurrent,
        )

        # Pathology encoding analysis.
        pathology = pathology_encoding_table(
            matrix
        )

        pathology.to_csv(
            outdir
            / f"{library}_pathology_encoding_detailed.csv",
            index=False,
        )

        for family in [
            "amyloid",
            "tau",
        ]:
            sub = pathology.loc[
                pathology["family"] == family
            ]

            pathology_summary_rows.append(
                {
                    "library": library,
                    "family": family,
                    "continuous_only": int(
                        sub[
                            "continuous_only"
                        ].sum()
                    ),
                    "ordinal_any_only": int(
                        sub[
                            "ordinal_any_only"
                        ].sum()
                    ),
                    "all_three_encodings": int(
                        sub[
                            "all_three_encodings"
                        ].sum()
                    ),
                    "equal_vs_continuous_direction_flips_all_pathways": int(
                        sub[
                            "equal_vs_continuous_direction_flip"
                        ].sum()
                    ),
                    "calibrated_vs_continuous_direction_flips_all_pathways": int(
                        sub[
                            "calibrated_vs_continuous_direction_flip"
                        ].sum()
                    ),
                    "equal_vs_continuous_direction_flips_if_either_sig": int(
                        (
                            sub[
                                "equal_vs_continuous_direction_flip"
                            ]
                            & (
                                sub[
                                    "equal_sig"
                                ]
                                | sub[
                                    "continuous_sig"
                                ]
                            )
                        ).sum()
                    ),
                    "calibrated_vs_continuous_direction_flips_if_either_sig": int(
                        (
                            sub[
                                "calibrated_vs_continuous_direction_flip"
                            ]
                            & (
                                sub[
                                    "calibrated_sig"
                                ]
                                | sub[
                                    "continuous_sig"
                                ]
                            )
                        ).sum()
                    ),
                }
            )

        plot_pathology_encoding_counts(
            pathology,
            library,
            figdir
            / f"pathology_encoding_counts_{library}.png",
        )

        # Weighted vs DESeq2.
        if "DESeq2_sig" not in matrix.columns:
            raise ValueError(
                f"{library} matrix lacks DESeq2_sig."
            )

        if "Regular_AREA_sig" not in matrix.columns:
            raise ValueError(
                f"{library} matrix lacks Regular_AREA_sig."
            )

        matrix["DESeq2_sig"] = safe_bool(
            matrix["DESeq2_sig"]
        )

        matrix["Regular_AREA_sig"] = safe_bool(
            matrix["Regular_AREA_sig"]
        )

        weighted_specific = matrix.loc[
            (
                matrix[
                    "n_weighted_phenotypes_significant"
                ]
                > 0
            )
            & ~matrix[
                "DESeq2_sig"
            ]
        ].copy()

        weighted_specific = weighted_specific.sort_values(
            [
                "n_weighted_phenotypes_significant",
                "n_tau",
                "n_amyloid",
                "n_cognition",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            kind="mergesort",
        )

        weighted_specific.to_csv(
            outdir
            / f"{library}_weighted_specific_vs_DESeq2_prioritized.csv",
            index=False,
        )

        both_area = weighted_specific.loc[
            weighted_specific[
                "Regular_AREA_sig"
            ]
        ].copy()

        both_area.to_csv(
            outdir
            / f"{library}_both_AREA_not_DESeq2_prioritized.csv",
            index=False,
        )

        method_summary_rows.append(
            {
                "library": library,
                "n_weighted_any_significant": int(
                    (
                        matrix[
                            "n_weighted_phenotypes_significant"
                        ]
                        > 0
                    ).sum()
                ),
                "n_weighted_specific_vs_DESeq2": len(
                    weighted_specific
                ),
                "n_both_AREA_not_DESeq2": len(
                    both_area
                ),
                "n_weighted_all9": int(
                    (
                        matrix[
                            "n_weighted_phenotypes_significant"
                        ]
                        == 9
                    ).sum()
                ),
                "n_shared_all_three_families": int(
                    (
                        matrix[
                            "family_specificity"
                        ]
                        == "shared_all_three_families"
                    ).sum()
                ),
            }
        )

    # ------------------------------------------------------------
    # Global synthesis tables
    # ------------------------------------------------------------
    family_summary = pd.DataFrame(
        family_summary_rows
    )

    family_summary.to_csv(
        outdir
        / "family_specificity_summary.csv",
        index=False,
    )

    pathology_summary = pd.DataFrame(
        pathology_summary_rows
    )

    pathology_summary.to_csv(
        outdir
        / "pathology_encoding_summary.csv",
        index=False,
    )

    method_summary = pd.DataFrame(
        method_summary_rows
    )

    method_summary.to_csv(
        outdir
        / "weighted_vs_deseq2_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WEIGHTED PHENOTYPE GSEA COUNTS")
    print("=" * 100)

    print(
        pivot.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("FAMILY-SPECIFIC PATHWAY COUNTS")
    print("=" * 100)

    print(
        family_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("PATHOLOGY ENCODING SUMMARY")
    print("=" * 100)

    print(
        pathology_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("WEIGHTED AREA VS DESEQ2")
    print("=" * 100)

    print(
        method_summary.to_string(
            index=False
        )
    )

    print("\nWROTE")
    print(
        f"  {outdir / 'directionality_validation.csv'}"
    )
    print(
        f"  {outdir / 'gsea_count_summary_pivot.csv'}"
    )
    print(
        f"  {outdir / 'family_specificity_summary.csv'}"
    )
    print(
        f"  {outdir / 'pathology_encoding_summary.csv'}"
    )
    print(
        f"  {outdir / 'weighted_vs_deseq2_summary.csv'}"
    )
    print(
        f"  figures -> {figdir}"
    )


if __name__ == "__main__":
    main()

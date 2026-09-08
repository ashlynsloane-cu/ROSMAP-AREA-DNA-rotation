#!/usr/bin/env python3
"""
build_weighted_area_priority_tables.py
======================================

Build interpretable priority tables from the full 9-phenotype Weighted AREA GSEA.

This script uses:
    results/weighted_area_full_phenotype_gsea/
        <library>_weighted_phenotype_matrix.csv

It produces the seven biology-focused tables requested:

1. Pathways significant across all 9 Weighted phenotypes
2. Pathways shared across cognition + amyloid + tau
3. Amyloid-only recurrent pathways
4. Tau-only recurrent pathways
5. Cognition-only recurrent pathways
6. Weighted-specific vs DESeq2, prioritized by phenotype recurrence
7. Both AREA methods significant while DESeq2 is not

It also creates exploratory, NOT-final, pathway figures:
- top recurrent pathway NES heatmap
- top Weighted-specific-vs-DESeq2 NES heatmap
- family-specific pathway-count bar chart

Important:
- No new hypothesis testing is performed.
- "Specific" means FDR<0.05 significance-pattern specific.
- Recurrent pathway tables are ranked by number of significant Weighted
  phenotypes first, then by median |NES| among significant phenotypes.
- GO/Reactome redundancy is NOT semantically collapsed here. That should be
  curated after inspecting the priority tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        "--outdir",
        default="results/weighted_area_priority_tables",
    )

    p.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Rows in exploratory heatmaps.",
    )

    p.add_argument(
        "--min-family-recurrence",
        type=int,
        default=2,
        help=(
            "For family-only tables, require significance in at least this "
            "many of that family's 3 phenotypes. Default=2."
        ),
    )

    return p.parse_args()


def to_bool(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
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


def load_matrix(root, library):
    path = (
        Path(root)
        / f"{library}_weighted_phenotype_matrix.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing phenotype matrix: {path}"
        )

    df = pd.read_csv(path)

    required = ["pathway"]

    for p in PHENOTYPES:
        required.extend(
            [
                f"{p}_NES",
                f"{p}_FDR",
                f"{p}_sig",
            ]
        )

    required.extend(
        [
            "DESeq2_sig",
            "Regular_AREA_sig",
            "DESeq2_NES",
            "Regular_AREA_NES",
        ]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{library} matrix missing columns: {missing}"
        )

    for p in PHENOTYPES:
        df[f"{p}_sig"] = to_bool(
            df[f"{p}_sig"]
        )

        df[f"{p}_NES"] = pd.to_numeric(
            df[f"{p}_NES"],
            errors="coerce",
        )

        df[f"{p}_FDR"] = pd.to_numeric(
            df[f"{p}_FDR"],
            errors="coerce",
        )

    df["DESeq2_sig"] = to_bool(
        df["DESeq2_sig"]
    )

    df["Regular_AREA_sig"] = to_bool(
        df["Regular_AREA_sig"]
    )

    df["DESeq2_NES"] = pd.to_numeric(
        df["DESeq2_NES"],
        errors="coerce",
    )

    df["Regular_AREA_NES"] = pd.to_numeric(
        df["Regular_AREA_NES"],
        errors="coerce",
    )

    return df


def add_metrics(df):
    x = df.copy()

    all_sig_cols = [
        f"{p}_sig"
        for p in PHENOTYPES
    ]

    x["n_weighted_sig"] = (
        x[all_sig_cols]
        .sum(axis=1)
    )

    for family, phenotypes in FAMILIES.items():
        cols = [
            f"{p}_sig"
            for p in phenotypes
        ]

        x[f"n_{family}_sig"] = (
            x[cols]
            .sum(axis=1)
        )

        x[f"any_{family}"] = (
            x[f"n_{family}_sig"] > 0
        )

        x[f"all_{family}"] = (
            x[f"n_{family}_sig"] == 3
        )

    med_abs = []
    med_signed = []
    max_abs = []
    direction_consistency = []

    for _, row in x.iterrows():
        vals = []

        for p in PHENOTYPES:
            if bool(row[f"{p}_sig"]):
                val = row[f"{p}_NES"]

                if pd.notna(val):
                    vals.append(float(val))

        if vals:
            arr = np.asarray(vals, float)

            med_abs.append(
                float(
                    np.median(
                        np.abs(arr)
                    )
                )
            )

            med_signed.append(
                float(
                    np.median(arr)
                )
            )

            max_abs.append(
                float(
                    np.max(
                        np.abs(arr)
                    )
                )
            )

            signs = np.sign(arr)

            direction_consistency.append(
                float(
                    max(
                        (signs > 0).mean(),
                        (signs < 0).mean(),
                    )
                )
            )
        else:
            med_abs.append(np.nan)
            med_signed.append(np.nan)
            max_abs.append(np.nan)
            direction_consistency.append(np.nan)

    x["median_abs_NES_when_sig"] = med_abs
    x["median_signed_NES_when_sig"] = med_signed
    x["max_abs_NES_when_sig"] = max_abs
    x["direction_consistency_when_sig"] = direction_consistency

    x["shared_all_three_families"] = (
        x["any_cognition"]
        & x["any_amyloid"]
        & x["any_tau"]
    )

    x["weighted_specific_vs_DESeq2"] = (
        (x["n_weighted_sig"] > 0)
        & ~x["DESeq2_sig"]
    )

    x["both_AREA_not_DESeq2"] = (
        x["weighted_specific_vs_DESeq2"]
        & x["Regular_AREA_sig"]
    )

    return x


def ranked(df):
    return df.sort_values(
        [
            "n_weighted_sig",
            "median_abs_NES_when_sig",
            "direction_consistency_when_sig",
            "max_abs_NES_when_sig",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        kind="mergesort",
    )


def select_output_columns(df):
    cols = [
        "pathway",
        "n_weighted_sig",
        "n_cognition_sig",
        "n_amyloid_sig",
        "n_tau_sig",
        "median_abs_NES_when_sig",
        "median_signed_NES_when_sig",
        "max_abs_NES_when_sig",
        "direction_consistency_when_sig",
        "DESeq2_sig",
        "DESeq2_NES",
        "Regular_AREA_sig",
        "Regular_AREA_NES",
    ]

    for p in PHENOTYPES:
        cols.extend(
            [
                f"{p}_sig",
                f"{p}_NES",
                f"{p}_FDR",
            ]
        )

    return df[
        [
            c for c in cols
            if c in df.columns
        ]
    ]


def save_heatmap(
    df,
    title,
    outpath,
    top_n,
):
    if df.empty:
        return

    plot_df = ranked(df).head(
        top_n
    ).copy()

    nes_cols = [
        f"{p}_NES"
        for p in PHENOTYPES
    ]

    values = (
        plot_df[nes_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    height = max(
        6,
        0.35 * len(plot_df) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(14, height)
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
            len(plot_df)
        )
    )

    ax.set_yticklabels(
        plot_df["pathway"]
    )

    ax.set_title(
        title
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


def save_family_count_plot(
    counts,
    library,
    outpath,
):
    categories = [
        "all9",
        "shared_all_three_families",
        "cognition_only_recurrent",
        "amyloid_only_recurrent",
        "tau_only_recurrent",
        "weighted_specific_vs_DESeq2",
        "both_AREA_not_DESeq2",
    ]

    labels = [
        "All 9",
        "All 3 families",
        "Cognition-only",
        "Amyloid-only",
        "Tau-only",
        "Weighted not DESeq2",
        "Both AREA not DESeq2",
    ]

    vals = [
        counts.get(
            c,
            0,
        )
        for c in categories
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        labels,
        vals,
    )

    ax.set_ylabel(
        "Pathways"
    )

    ax.set_title(
        f"{library}: prioritized pathway categories"
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
        / "exploratory_figures"
    )

    figdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    print("=" * 100)
    print("WEIGHTED AREA PRIORITY TABLES")
    print("=" * 100)

    for library in LIBRARIES:
        print(
            f"\n{library}"
        )

        df = add_metrics(
            load_matrix(
                args.gsea_root,
                library,
            )
        )

        # 1. Significant across all 9 phenotypes.
        all9 = ranked(
            df.loc[
                df["n_weighted_sig"] == 9
            ].copy()
        )

        # 2. Shared across all phenotype families.
        shared_families = ranked(
            df.loc[
                df[
                    "shared_all_three_families"
                ]
            ].copy()
        )

        # 3-5. Recurrent family-specific pathways.
        cognition_only = ranked(
            df.loc[
                (
                    df[
                        "n_cognition_sig"
                    ]
                    >= args.min_family_recurrence
                )
                & (
                    df[
                        "n_amyloid_sig"
                    ]
                    == 0
                )
                & (
                    df[
                        "n_tau_sig"
                    ]
                    == 0
                )
            ].copy()
        )

        amyloid_only = ranked(
            df.loc[
                (
                    df[
                        "n_amyloid_sig"
                    ]
                    >= args.min_family_recurrence
                )
                & (
                    df[
                        "n_cognition_sig"
                    ]
                    == 0
                )
                & (
                    df[
                        "n_tau_sig"
                    ]
                    == 0
                )
            ].copy()
        )

        tau_only = ranked(
            df.loc[
                (
                    df[
                        "n_tau_sig"
                    ]
                    >= args.min_family_recurrence
                )
                & (
                    df[
                        "n_cognition_sig"
                    ]
                    == 0
                )
                & (
                    df[
                        "n_amyloid_sig"
                    ]
                    == 0
                )
            ].copy()
        )

        # 6. Weighted-specific vs DESeq2.
        weighted_specific = ranked(
            df.loc[
                df[
                    "weighted_specific_vs_DESeq2"
                ]
            ].copy()
        )

        # 7. Both AREA, DESeq2 not.
        both_area = ranked(
            df.loc[
                df[
                    "both_AREA_not_DESeq2"
                ]
            ].copy()
        )

        tables = {
            "01_all9_weighted": all9,
            "02_shared_all_three_families": shared_families,
            "03_cognition_only_recurrent": cognition_only,
            "04_amyloid_only_recurrent": amyloid_only,
            "05_tau_only_recurrent": tau_only,
            "06_weighted_specific_vs_DESeq2": weighted_specific,
            "07_both_AREA_not_DESeq2": both_area,
        }

        libdir = (
            outdir
            / library
        )

        libdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, table in tables.items():
            select_output_columns(
                table
            ).to_csv(
                libdir
                / f"{name}.csv",
                index=False,
            )

        counts = {
            "all9": len(all9),
            "shared_all_three_families": len(
                shared_families
            ),
            "cognition_only_recurrent": len(
                cognition_only
            ),
            "amyloid_only_recurrent": len(
                amyloid_only
            ),
            "tau_only_recurrent": len(
                tau_only
            ),
            "weighted_specific_vs_DESeq2": len(
                weighted_specific
            ),
            "both_AREA_not_DESeq2": len(
                both_area
            ),
        }

        summary_rows.append(
            {
                "library": library,
                **counts,
            }
        )

        print(
            f"  all 9 phenotypes:             {len(all9):4d}"
        )
        print(
            f"  shared all 3 families:        {len(shared_families):4d}"
        )
        print(
            f"  cognition-only recurrent:     {len(cognition_only):4d}"
        )
        print(
            f"  amyloid-only recurrent:       {len(amyloid_only):4d}"
        )
        print(
            f"  tau-only recurrent:           {len(tau_only):4d}"
        )
        print(
            f"  Weighted-specific vs DESeq2:  {len(weighted_specific):4d}"
        )
        print(
            f"  both AREA, DESeq2 not:        {len(both_area):4d}"
        )

        # Exploratory figures.
        save_heatmap(
            shared_families,
            (
                f"{library}: top pathways shared across "
                "cognition, amyloid, and tau"
            ),
            figdir
            / f"{library}_shared_all_three_families_heatmap.png",
            args.top_n,
        )

        save_heatmap(
            weighted_specific,
            (
                f"{library}: top recurrent Weighted AREA pathways "
                "not significant in DESeq2"
            ),
            figdir
            / f"{library}_weighted_specific_vs_DESeq2_heatmap.png",
            args.top_n,
        )

        save_heatmap(
            both_area,
            (
                f"{library}: both AREA methods significant, "
                "DESeq2 not"
            ),
            figdir
            / f"{library}_both_AREA_not_DESeq2_heatmap.png",
            args.top_n,
        )

        save_family_count_plot(
            counts,
            library,
            figdir
            / f"{library}_priority_category_counts.png",
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        outdir
        / "priority_table_counts.csv",
        index=False,
    )

    (outdir / "README.txt").write_text(
        "Weighted AREA priority tables\n\n"
        "01_all9_weighted.csv\n"
        "  Significant at FDR<0.05 in all 9 Weighted phenotypes.\n\n"
        "02_shared_all_three_families.csv\n"
        "  Significant in >=1 cognition, >=1 amyloid, and >=1 tau phenotype.\n\n"
        "03_cognition_only_recurrent.csv\n"
        f"  Significant in >={args.min_family_recurrence}/3 cognition "
        "phenotypes and 0 amyloid/tau phenotypes.\n\n"
        "04_amyloid_only_recurrent.csv\n"
        f"  Significant in >={args.min_family_recurrence}/3 amyloid "
        "phenotypes and 0 cognition/tau phenotypes.\n\n"
        "05_tau_only_recurrent.csv\n"
        f"  Significant in >={args.min_family_recurrence}/3 tau "
        "phenotypes and 0 cognition/amyloid phenotypes.\n\n"
        "06_weighted_specific_vs_DESeq2.csv\n"
        "  Significant in >=1 Weighted phenotype and not significant "
        "in DESeq2 GSEA.\n\n"
        "07_both_AREA_not_DESeq2.csv\n"
        "  Significant in >=1 Weighted phenotype and Regular AREA, "
        "but not DESeq2.\n\n"
        "Ranking within tables:\n"
        "  1) number of significant Weighted phenotypes\n"
        "  2) median absolute NES among significant phenotypes\n"
        "  3) direction consistency among significant phenotypes\n"
        "  4) max absolute NES\n\n"
        "GO/Reactome redundancy is intentionally not collapsed yet.\n"
    )

    print("\nWROTE")
    print(
        f"  {outdir / 'priority_table_counts.csv'}"
    )
    print(
        f"  per-library tables -> {outdir}"
    )
    print(
        f"  exploratory figures -> {figdir}"
    )


if __name__ == "__main__":
    main()

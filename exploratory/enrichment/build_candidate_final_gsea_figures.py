#!/usr/bin/env python3
"""
build_candidate_final_gsea_figures.py
=====================================

Create biologically themed candidate figures from the Weighted AREA priority tables.

INPUT
-----
results/weighted_area_priority_tables/<library>/
    01_all9_weighted.csv
    02_shared_all_three_families.csv
    06_weighted_specific_vs_DESeq2.csv
    07_both_AREA_not_DESeq2.csv

PRIMARY PURPOSE
---------------
Reduce Reactome/GO redundancy and create a first-pass, biologically interpretable
figure set centered on recurrent pathway themes.

This script:
1. assigns pathways to broad biological themes using transparent keyword rules
2. prioritizes pathways supported by BOTH AREA methods but not DESeq2
3. supplements with highly recurrent all-9/shared pathways
4. chooses a small representative panel per theme
5. writes a curation table you can manually edit
6. makes candidate heatmaps and summary plots

IMPORTANT
---------
The keyword-based theme assignment is intentionally transparent and exploratory.
It is NOT ontology clustering and should be manually reviewed before final figures.

Ranking orientation:
    positive NES = higher expression with greater disease severity/pathology
    negative NES = lower expression with greater disease severity/pathology
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

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

DISPLAY_PHENOTYPES = [
    "Cognitive\nstage",
    "Global cognitive\nimpairment",
    "MMSE\nimpairment",
    "CERAD\nequal",
    "CERAD\ncalibrated",
    "Amyloid\ncontinuous",
    "Braak\nequal",
    "Braak\ncalibrated",
    "Tangles\ncontinuous",
]

# Order is intentional: more specific themes first.
THEME_RULES = {
    "Mitochondrial energetics": [
        r"oxidative phosphorylation",
        r"respiratory electron",
        r"electron transport",
        r"respiratory chain",
        r"complex i\b",
        r"complex ii\b",
        r"complex iii\b",
        r"complex iv\b",
        r"complex v\b",
        r"atp synthesis",
        r"tca",
        r"citric acid",
        r"mitochondrial.*respir",
        r"mitochondrial protein import",
        r"cristae",
    ],
    "Autophagy / mitophagy / lysosome": [
        r"mitophagy",
        r"pink1",
        r"prkn",
        r"autophagy",
        r"lysosom",
        r"mitf",
        r"selective autophagy",
    ],
    "Synaptic / neurotransmission": [
        r"synap",
        r"neurotransmitter",
        r"glutamate",
        r"gaba",
        r"acetylcholine",
        r"norepinephrine",
        r"dopamine",
        r"serotonin",
        r"nmda",
        r"ampa",
        r"long[- ]term potentiation",
        r"postsynaptic",
        r"presynaptic",
    ],
    "Protein synthesis / folding": [
        r"translation",
        r"ribosom",
        r"trna",
        r"aminoacyl",
        r"prefoldin",
        r"cct",
        r"tric",
        r"protein folding",
        r"chaperon",
        r"proteostasis",
    ],
    "Vesicle trafficking / endocytosis": [
        r"vesicle",
        r"endocyt",
        r"clathrin",
        r"golgi",
        r"er to golgi",
        r"transport along microtubule",
        r"membrane trafficking",
    ],
    "Immune / inflammatory signaling": [
        r"nf[- ]?kappa",
        r"nfkb",
        r"interferon",
        r"cytokine",
        r"toll[- ]like",
        r"myd88",
        r"immune",
        r"inflamm",
        r"interleukin",
        r"tnf",
    ],
    "Rho GTPase / cytoskeleton": [
        r"rho[abcq]?\b",
        r"cdc42",
        r"rac\b",
        r"gtpase",
        r"actin",
        r"cytoskelet",
        r"microtubule",
    ],
    "Chromatin / transcription": [
        r"chromatin",
        r"histone",
        r"epigen",
        r"transcription",
        r"rora",
        r"e2f",
        r"dream",
        r"hdac",
    ],
    "RNA processing": [
        r"rna",
        r"splic",
        r"mirna",
        r"mrna",
        r"rrna",
        r"rna degradation",
        r"rna metabolism",
    ],
    "ECM / adhesion": [
        r"collagen",
        r"extracellular matrix",
        r"\becm\b",
        r"integrin",
        r"adhesion",
        r"laminin",
        r"matrix organization",
    ],
    "Lipid / cholesterol metabolism": [
        r"cholesterol",
        r"lipid",
        r"hdl",
        r"lipoprotein",
        r"fatty acid",
        r"sterol",
    ],
    "Cell death / stress": [
        r"apopt",
        r"cell death",
        r"p53",
        r"stress response",
        r"unfolded protein",
        r"hypoxia",
    ],
    "Cell cycle": [
        r"cell cycle",
        r"mitotic",
        r"mitosis",
        r"checkpoint",
        r"dna replication",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--priority-root",
        default="results/weighted_area_priority_tables",
    )

    p.add_argument(
        "--outdir",
        default="results/candidate_final_gsea_figures",
    )

    p.add_argument(
        "--libraries",
        nargs="+",
        default=[
            "Reactome_Pathways_2024",
            "GO_Biological_Process_2025",
        ],
    )

    p.add_argument(
        "--themes-per-figure",
        type=int,
        default=6,
    )

    p.add_argument(
        "--pathways-per-theme",
        type=int,
        default=3,
    )

    p.add_argument(
        "--max-pathways",
        type=int,
        default=18,
    )

    return p.parse_args()


def assign_theme(pathway):
    text = str(pathway).lower()

    for theme, patterns in THEME_RULES.items():
        for pattern in patterns:
            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                return theme

    return "Other / unassigned"


def read_table(root, library, filename):
    path = (
        Path(root)
        / library
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing priority table: {path}"
        )

    df = pd.read_csv(path)

    if "pathway" not in df.columns:
        raise ValueError(
            f"{path} lacks pathway column."
        )

    return df


def normalize_bool(series):
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


def add_source_priority(df, source_name):
    x = df.copy()

    x["source_set"] = source_name

    source_rank = {
        "both_AREA_not_DESeq2": 4,
        "weighted_specific_vs_DESeq2": 3,
        "all9_weighted": 2,
        "shared_all_three_families": 1,
    }

    x["source_priority"] = source_rank[
        source_name
    ]

    return x


def combine_priority_sources(root, library):
    files = [
        (
            "07_both_AREA_not_DESeq2.csv",
            "both_AREA_not_DESeq2",
        ),
        (
            "06_weighted_specific_vs_DESeq2.csv",
            "weighted_specific_vs_DESeq2",
        ),
        (
            "01_all9_weighted.csv",
            "all9_weighted",
        ),
        (
            "02_shared_all_three_families.csv",
            "shared_all_three_families",
        ),
    ]

    frames = []

    for filename, source_name in files:
        df = read_table(
            root,
            library,
            filename,
        )

        frames.append(
            add_source_priority(
                df,
                source_name,
            )
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # A pathway may appear in several source tables.
    # Keep its highest-priority source, but also retain all source memberships.
    memberships = (
        combined.groupby(
            "pathway"
        )[
            "source_set"
        ]
        .apply(
            lambda s:
                ";".join(
                    sorted(
                        set(
                            s.astype(str)
                        )
                    )
                )
        )
        .rename(
            "all_source_sets"
        )
    )

    combined = (
        combined.sort_values(
            [
                "source_priority",
                "n_weighted_sig",
                "median_abs_NES_when_sig",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            "pathway",
            keep="first",
        )
        .merge(
            memberships,
            on="pathway",
            how="left",
            validate="one_to_one",
        )
    )

    combined["theme"] = (
        combined[
            "pathway"
        ]
        .map(
            assign_theme
        )
    )

    return combined


def choose_candidate_panel(
    df,
    pathways_per_theme,
    max_pathways,
):
    x = df.copy()

    x["n_weighted_sig"] = pd.to_numeric(
        x["n_weighted_sig"],
        errors="coerce",
    )

    x["median_abs_NES_when_sig"] = pd.to_numeric(
        x["median_abs_NES_when_sig"],
        errors="coerce",
    )

    x["direction_consistency_when_sig"] = pd.to_numeric(
        x["direction_consistency_when_sig"],
        errors="coerce",
    )

    x = x.sort_values(
        [
            "source_priority",
            "n_weighted_sig",
            "median_abs_NES_when_sig",
            "direction_consistency_when_sig",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        kind="mergesort",
    )

    # Exclude unassigned from automatic final candidate panel;
    # they remain in full curation table for manual review.
    themed = x.loc[
        x["theme"]
        != "Other / unassigned"
    ].copy()

    selected = (
        themed.groupby(
            "theme",
            group_keys=False,
        )
        .head(
            pathways_per_theme
        )
        .copy()
    )

    selected = selected.sort_values(
        [
            "source_priority",
            "n_weighted_sig",
            "median_abs_NES_when_sig",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        kind="mergesort",
    ).head(
        max_pathways
    )

    # Re-sort by biological theme for plotting.
    theme_order = list(
        THEME_RULES.keys()
    )

    selected["theme_order"] = (
        pd.Categorical(
            selected["theme"],
            categories=theme_order,
            ordered=True,
        )
    )

    selected = selected.sort_values(
        [
            "theme_order",
            "n_weighted_sig",
            "median_abs_NES_when_sig",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        kind="mergesort",
    ).drop(
        columns=[
            "theme_order"
        ]
    )

    return selected


def make_heatmap(
    df,
    title,
    outpath,
):
    if df.empty:
        return

    nes_cols = [
        f"{p}_NES"
        for p in PHENOTYPES
    ]

    values = (
        df[
            nes_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    labels = [
        f"{row['theme']} | {row['pathway']}"
        for _, row in df.iterrows()
    ]

    fig_height = max(
        7,
        0.48 * len(df) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(14, fig_height)
    )

    vmax = np.nanmax(
        np.abs(values)
    )

    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
        cmap="coolwarm",
    )

    ax.set_xticks(
        np.arange(
            len(PHENOTYPES)
        )
    )

    ax.set_xticklabels(
        DISPLAY_PHENOTYPES,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(df)
        )
    )

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.set_title(
        title
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "NES\n(+ higher expression with severity; − lower)"
    )

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_theme_summary_plot(
    df,
    library,
    outpath,
):
    x = (
        df.loc[
            df["theme"]
            != "Other / unassigned"
        ]
        .groupby(
            "theme"
        )
        .agg(
            n_pathways=(
                "pathway",
                "nunique",
            ),
            median_recurrence=(
                "n_weighted_sig",
                "median",
            ),
        )
        .sort_values(
            [
                "n_pathways",
                "median_recurrence",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    if x.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        x.index[::-1],
        x["n_pathways"].to_numpy()[::-1],
    )

    ax.set_xlabel(
        "Prioritized pathways assigned to theme"
    )

    ax.set_title(
        f"{library}: exploratory biological-theme summary"
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

    combined_all = []

    print("=" * 100)
    print("CANDIDATE FINAL GSEA FIGURE BUILD")
    print("=" * 100)

    for library in args.libraries:
        print(
            f"\n{library}"
        )

        combined = combine_priority_sources(
            args.priority_root,
            library,
        )

        combined["library"] = library

        combined_all.append(
            combined
        )

        combined.to_csv(
            outdir
            / f"{library}_theme_curation_full.csv",
            index=False,
        )

        # Manual curation scaffold.
        scaffold_cols = [
            "pathway",
            "theme",
            "source_set",
            "all_source_sets",
            "n_weighted_sig",
            "n_cognition_sig",
            "n_amyloid_sig",
            "n_tau_sig",
            "median_abs_NES_when_sig",
            "direction_consistency_when_sig",
            "DESeq2_sig",
            "Regular_AREA_sig",
        ]

        scaffold = combined[
            [
                c for c in scaffold_cols
                if c in combined.columns
            ]
        ].copy()

        scaffold["keep_for_final_figure"] = ""
        scaffold["manual_theme_override"] = ""
        scaffold["notes"] = ""

        scaffold.to_csv(
            outdir
            / f"{library}_manual_curation_scaffold.csv",
            index=False,
        )

        candidate = choose_candidate_panel(
            combined,
            args.pathways_per_theme,
            args.max_pathways,
        )

        candidate.to_csv(
            outdir
            / f"{library}_candidate_figure_panel.csv",
            index=False,
        )

        make_heatmap(
            candidate,
            (
                f"{library}: candidate recurrent disease-associated "
                "pathway panel"
            ),
            figdir
            / f"{library}_candidate_recurrent_pathway_heatmap.png",
        )

        both_area = combined.loc[
            combined[
                "all_source_sets"
            ].str.contains(
                "both_AREA_not_DESeq2",
                na=False,
            )
        ].copy()

        both_area_candidate = choose_candidate_panel(
            both_area,
            args.pathways_per_theme,
            args.max_pathways,
        )

        both_area_candidate.to_csv(
            outdir
            / f"{library}_candidate_both_AREA_not_DESeq2_panel.csv",
            index=False,
        )

        make_heatmap(
            both_area_candidate,
            (
                f"{library}: candidate pathways supported by both AREA "
                "methods but not DESeq2"
            ),
            figdir
            / f"{library}_candidate_both_AREA_not_DESeq2_heatmap.png",
        )

        make_theme_summary_plot(
            combined,
            library,
            figdir
            / f"{library}_theme_summary.png",
        )

        print(
            f"  prioritized pathways: {len(combined):,}"
        )
        print(
            f"  auto-themed: "
            f"{int((combined['theme'] != 'Other / unassigned').sum()):,}"
        )
        print(
            f"  candidate figure panel: {len(candidate):,}"
        )
        print(
            f"  candidate both-AREA-not-DESeq2 panel: "
            f"{len(both_area_candidate):,}"
        )

    all_df = pd.concat(
        combined_all,
        ignore_index=True,
        sort=False,
    )

    theme_summary = (
        all_df.groupby(
            [
                "library",
                "theme",
            ]
        )
        .agg(
            n_unique_pathways=(
                "pathway",
                "nunique",
            ),
            median_n_weighted_sig=(
                "n_weighted_sig",
                "median",
            ),
            median_abs_NES=(
                "median_abs_NES_when_sig",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "library",
                "n_unique_pathways",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    theme_summary.to_csv(
        outdir
        / "theme_summary_across_libraries.csv",
        index=False,
    )

    (outdir / "README.txt").write_text(
        "Candidate final GSEA figure outputs\n\n"
        "Key files:\n"
        "  *_theme_curation_full.csv\n"
        "      all prioritized pathways with automatic broad-theme assignment\n\n"
        "  *_manual_curation_scaffold.csv\n"
        "      same pathways with blank columns for manual final decisions\n\n"
        "  *_candidate_figure_panel.csv\n"
        "      automatically selected representative pathways across themes\n\n"
        "  *_candidate_both_AREA_not_DESeq2_panel.csv\n"
        "      representative pathways supported by both AREA methods but not DESeq2\n\n"
        "  figures/*candidate*.png\n"
        "      exploratory heatmaps only; inspect before treating as final\n\n"
        "Theme assignment uses transparent keyword rules and MUST be manually reviewed.\n"
    )

    print("\nWROTE")
    print(
        f"  {outdir}"
    )
    print(
        f"  figures -> {figdir}"
    )


if __name__ == "__main__":
    main()

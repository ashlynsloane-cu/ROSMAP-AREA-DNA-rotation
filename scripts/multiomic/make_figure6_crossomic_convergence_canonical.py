#!/usr/bin/env python3
"""
Cross-omic convergence overview figure.

Scientific purpose
------------------
Show the complete, prespecified RNA/protein convergence landscape before
showing individual genes. The figure has three panels:

A. Formal cross-omic discoveries (same-direction RNA–protein genes):
       shared between AREA and conventional
       AREA-only
       conventional-only
   for every phenotype contrast.

B. Evidence funnel:
       genes significant in both RNA and protein
       -> formal same-direction RNA/protein conjunction
   for AREA and conventional analyses.

C. Recurrent cross-phenotype genes:
   a compact matrix of the 10 recurrent genes identified in the Stage 12E
   biological-evidence audit.

The figure intentionally does NOT include RRHO or leading-edge results.
Those answer different questions and should remain separate.

Input
-----
Stage 12K:
    ~/Developer/stage11_offline_2026-09-28/results/
        stage12_brain_convergence/stage12K_convergence_architecture/
        CONVERGENCE_ARCHITECTURE_SUMMARY.tsv

Formal AREA-vs-conventional comparison:
    The script searches the same Stage 12K directory for
    CROSS_METHOD_ARCHITECTURE_COMPARISON.tsv.

If the formal comparison table is absent, the script stops rather than
silently hard-coding the formal overlap counts.

Output
------
Defaults to:
    <repo-root>/results/figures/

Run:
    python3 scripts/multiomic/make_figure_cross_omic_convergence_overview.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


# ---------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------

WIDTH_MM = 170
HEIGHT_MM = 170
DPI = 600

AREA_COLOR = "#7A8FA8"
CONVENTIONAL_COLOR = "#B8C0CA"
SHARED_COLOR = "#596F86"
AREA_ONLY_COLOR = "#9BAFC1"
CONVENTIONAL_ONLY_COLOR = "#D1D6DB"
NEAR_COLOR = "#8A929A"
AREA_NEAR_COLOR = "#B7C0C8"

INK = "#283038"
MID = "#66727C"
GRID = "#E5E9EC"
WHITE = "#FFFFFF"


# ---------------------------------------------------------------------
# Prespecified phenotype contrasts
# ---------------------------------------------------------------------

CONTRASTS = [
    ("cogdx_NCI_vs_AD", "NCI → AD"),
    ("cogdx_MCI_vs_AD", "MCI → AD"),
    ("Braak_3-4_vs_5-6", "Late Braak"),
    ("CERAD_3-4_vs_1-2", "CERAD"),
]

METHODS = ["AREA", "CONVENTIONAL"]


# ---------------------------------------------------------------------
# Recurrent-gene panel
#
# These are the 10 recurrent genes explicitly identified in the frozen
# Stage 12E biological-evidence audit. They are NOT used to define Panels
# A or B. Their inclusion is downstream of the exhaustive analysis.
# ---------------------------------------------------------------------

RECURRENT_GENES = [
    "PPEF1", "SLC38A2", "VGF", "SST", "CSRP1",
    "KIF13B", "NRIP2", "RAB9A", "DHRS7B", "TAPT1",
]

CONTRAST_LABELS = dict(CONTRASTS)


def load_recurrent_evidence(path: Path) -> dict[str, dict[str, str | None]]:
    """Derive displayed classifications from frozen Stage 12E evidence."""
    df = pd.read_csv(path, sep="\t")
    necessary = {"gene_symbol", "contrast_id", "evidence_track",
                 "brain_category", "near_category"}
    if not necessary.issubset(df.columns):
        raise ValueError(f"Missing Stage 12E columns: {sorted(necessary - set(df.columns))}")
    selected = df.loc[df.gene_symbol.isin(RECURRENT_GENES) &
                      df.contrast_id.isin(CONTRAST_LABELS)].copy()
    if selected.duplicated(["gene_symbol", "contrast_id"]).any():
        raise ValueError("Duplicate gene/contrast pairs in Stage 12E evidence")
    labels = {
        "shared": "shared", "AREA_only": "AREA-only",
        "conventional_only": "conventional-only",
        "shared_near": "shared-near", "AREA_only_near": "AREA-near",
    }
    recurrent = {gene: {label: None for _, label in CONTRASTS}
                 for gene in RECURRENT_GENES}
    for row in selected.itertuples(index=False):
        category = (row.brain_category if row.evidence_track == "formal"
                    else row.near_category)
        if pd.isna(category) or category not in labels:
            raise ValueError(f"Unrecognized classification: {row.gene_symbol}, {row.contrast_id}, {category}")
        recurrent[row.gene_symbol][CONTRAST_LABELS[row.contrast_id]] = labels[category]
    if any(all(v is None for v in mapping.values()) for mapping in recurrent.values()):
        raise ValueError("At least one prespecified gene has no Stage 12E evidence")
    return recurrent


STATUS_COLORS = {
    "shared": SHARED_COLOR,
    "AREA-only": AREA_ONLY_COLOR,
    "conventional-only": CONVENTIONAL_ONLY_COLOR,
    "shared-near": NEAR_COLOR,
    "AREA-near": AREA_NEAR_COLOR,
}

STATUS_LABELS = {
    "shared": "Formal shared",
    "AREA-only": "AREA-only",
    "conventional-only": "Conventional-only",
    "shared-near": "Shared near-convergence",
    "AREA-near": "AREA-only near-convergence",
}


# ---------------------------------------------------------------------
# Arguments and paths
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cross-omic convergence overview figure."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Explicit path to CONVERGENCE_ARCHITECTURE_SUMMARY.tsv.",
    )
    parser.add_argument(
        "--formal-comparison",
        type=Path,
        default=None,
        help="Explicit path to CROSS_METHOD_ARCHITECTURE_COMPARISON.tsv.",
    )
    parser.add_argument(
        "--evidence-source", type=Path, default=None,
        help="Frozen Stage 12E MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def locate_source(repo_root: Path, source: Path | None) -> Path:
    if source is not None:
        path = source.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        repo_root
        / "results/stage12_brain_convergence/"
        / "stage12K_convergence_architecture/"
        / "CONVERGENCE_ARCHITECTURE_SUMMARY.tsv",
        repo_root
        / "data/frozen_inputs/results/"
        / "stage12_brain_convergence/stage12K_convergence_architecture/"
        / "CONVERGENCE_ARCHITECTURE_SUMMARY.tsv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate CONVERGENCE_ARCHITECTURE_SUMMARY.tsv. "
        "Use --source /path/to/CONVERGENCE_ARCHITECTURE_SUMMARY.tsv."
    )


def locate_formal_comparison(
    repo_root: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        repo_root
        / "results/stage12_brain_convergence/"
        / "stage12K_convergence_architecture/"
        / "CROSS_METHOD_ARCHITECTURE_COMPARISON.tsv",
        repo_root
        / "data/frozen_inputs/results/"
        / "stage12_brain_convergence/stage12K_convergence_architecture/"
        / "CROSS_METHOD_ARCHITECTURE_COMPARISON.tsv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate CROSS_METHOD_ARCHITECTURE_COMPARISON.tsv. "
        "This table is required so the formal AREA/conventional overlap "
        "counts are read from frozen results rather than hard-coded."
    )


def locate_output_dir(
    repo_root: Path,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return repo_root / "results" / "figures"


# ---------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------


def load_architecture(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {
        "contrast_id",
        "method",
        "both_individual_fdr05_n",
        "formal_conjunction_fdr05_n",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "Stage 12K table is missing: " + ", ".join(sorted(missing))
        )

    contrast_ids = [x[0] for x in CONTRASTS]
    df = df.loc[
        df["contrast_id"].isin(contrast_ids)
        & df["method"].isin(METHODS)
    ].copy()

    expected = {
        (contrast, method)
        for contrast, _ in CONTRASTS
        for method in METHODS
    }
    observed = {
        (str(row.contrast_id), str(row.method))
        for row in df.itertuples(index=False)
    }

    if observed != expected:
        raise RuntimeError(
            "Stage 12K does not contain exactly the expected "
            "4 contrasts × 2 methods.\n"
            f"Missing: {sorted(expected - observed)}\n"
            f"Extra: {sorted(observed - expected)}"
        )

    if len(df) != 8:
        raise RuntimeError(
            f"Expected 8 Stage 12K rows; found {len(df)}."
        )

    return df


def load_formal_comparison(path: Path) -> pd.DataFrame:
    """Load the Stage 12K AREA-vs-conventional formal comparison.

    Stage 12K stores the formal comparison using these canonical fields:
        area_formal_genes_n
        conventional_formal_genes_n
        formal_gene_overlap_n

    The plotting code uses the equivalent partition:
        shared = overlap
        AREA-only = AREA total - overlap
        conventional-only = conventional total - overlap

    This is preferable to hard-coding the counts and keeps Panel A tied to
    the frozen Stage 12K output.
    """
    df = pd.read_csv(path, sep="\t")

    required = {
        "contrast_id",
        "area_formal_genes_n",
        "conventional_formal_genes_n",
        "formal_gene_overlap_n",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "Stage 12K formal comparison table is missing: "
            + ", ".join(sorted(missing))
            + "\nFound columns: "
            + ", ".join(df.columns.astype(str))
        )

    contrast_ids = [x[0] for x in CONTRASTS]
    df = df.loc[df["contrast_id"].isin(contrast_ids)].copy()

    if len(df) != 4:
        raise RuntimeError(
            "Expected exactly four phenotype contrasts in formal "
            f"comparison table; found {len(df)}."
        )

    df["shared_BH_n"] = df["formal_gene_overlap_n"].astype(int)
    df["AREA_only_BH_n"] = (
        df["area_formal_genes_n"].astype(int)
        - df["formal_gene_overlap_n"].astype(int)
    )
    df["conventional_only_BH_n"] = (
        df["conventional_formal_genes_n"].astype(int)
        - df["formal_gene_overlap_n"].astype(int)
    )

    if (df[[
        "AREA_only_BH_n",
        "conventional_only_BH_n",
    ]] < 0).any().any():
        raise RuntimeError(
            "Invalid formal comparison: overlap exceeds one of the "
            "method-specific formal discovery counts."
        )

    return df


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def mm_to_inches(mm: float) -> float:
    return mm / 25.4


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(
        axis="both",
        colors=MID,
        labelsize=8,
        length=0,
    )
    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.7,
        zorder=0,
    )
    ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(
        -0.075,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


# ---------------------------------------------------------------------
# Panel A — formal AREA vs conventional
# ---------------------------------------------------------------------


def plot_panel_a(ax, formal):
    """
    Show the complete formal-discovery landscape.

    Each stacked bar partitions the union of formal discoveries into:
        shared
        AREA-only
        conventional-only

    Thus no individual gene is selected for display here.
    """

    labels = [label for _, label in CONTRASTS]
    y = list(range(len(labels)))

    for yi, (contrast_id, label) in zip(y, CONTRASTS):
        row = formal.loc[
            formal["contrast_id"].eq(contrast_id)
        ].iloc[0]

        shared = int(row["shared_BH_n"])
        area_only = int(row["AREA_only_BH_n"])
        conventional_only = int(row["conventional_only_BH_n"])

        left = 0

        if shared:
            ax.barh(
                yi,
                shared,
                left=left,
                height=0.52,
                color=SHARED_COLOR,
                zorder=3,
            )
            ax.text(
                left + shared / 2,
                yi,
                str(shared),
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color=WHITE,
                zorder=4,
            )
        left += shared

        if area_only:
            ax.barh(
                yi,
                area_only,
                left=left,
                height=0.52,
                color=AREA_ONLY_COLOR,
                zorder=3,
            )
            ax.text(
                left + area_only / 2,
                yi,
                str(area_only),
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK,
                zorder=4,
            )
        left += area_only

        if conventional_only:
            ax.barh(
                yi,
                conventional_only,
                left=left,
                height=0.52,
                color=CONVENTIONAL_ONLY_COLOR,
                zorder=3,
            )
            ax.text(
                left + conventional_only / 2,
                yi,
                str(conventional_only),
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK,
                zorder=4,
            )

        total = shared + area_only + conventional_only
        ax.text(
            total + 2,
            yi,
            f"{total} unique",
            ha="left",
            va="center",
            fontsize=7.2,
            color=MID,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(3.45, -1.65)
    ax.legend(
        handles=[
            Patch(facecolor=SHARED_COLOR, label="Shared"),
            Patch(facecolor=AREA_ONLY_COLOR, label="AREA-only"),
            Patch(facecolor=CONVENTIONAL_ONLY_COLOR, label="Conventional-only"),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.97),
        fontsize=5.0,
        ncol=3,
        handlelength=0.75,
        handletextpad=0.25,
        columnspacing=1.2,
    )

    max_total = max(
        int(formal.loc[
            formal["contrast_id"].eq(contrast_id),
            ["shared_BH_n", "AREA_only_BH_n", "conventional_only_BH_n"],
        ].sum(axis=1).iloc[0])
        for contrast_id, _ in CONTRASTS
    )

    ax.set_xlim(0, max_total * 1.23)


    ax.set_title(
        "Formal cross-omic discoveries (same-direction RNA–protein genes)",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        pad=5,
    )

    style_axis(ax)


# ---------------------------------------------------------------------
# Panel B — evidence funnel
# ---------------------------------------------------------------------


def plot_panel_b(ax, architecture):
    """
    Show how many genes remain as cross-omic evidence becomes stricter.

    For each method and contrast:
        both_individual_fdr05_n
            -> formal_conjunction_fdr05_n

    This makes the zero-gene MCI -> AD / Late Braak formal result explicit
    without implying absence of RNA-protein structure.
    """

    rows = []

    for contrast_id, contrast_label in CONTRASTS:
        for method in METHODS:
            row = architecture.loc[
                architecture["contrast_id"].eq(contrast_id)
                & architecture["method"].eq(method)
            ].iloc[0]

            rows.append(
                (
                    contrast_id,
                    contrast_label,
                    method,
                    int(row["both_individual_fdr05_n"]),
                    int(row["formal_conjunction_fdr05_n"]),
                )
            )

    y = list(range(len(rows)))[::-1]

    for yi, (_, contrast_label, method, both, formal) in zip(y, rows):
        color = AREA_COLOR if method == "AREA" else CONVENTIONAL_COLOR

        ax.hlines(
            yi,
            0,
            both,
            color=color,
            linewidth=2.5,
            zorder=2,
        )

        ax.scatter(
            both,
            yi,
            s=28,
            color=color,
            edgecolor=WHITE,
            linewidth=0.7,
            zorder=4,
        )

        ax.scatter(
            formal,
            yi,
            s=31,
            color=SHARED_COLOR,
            edgecolor=WHITE,
            linewidth=0.7,
            zorder=5,
        )

        ax.text(
            both + 7,
            yi + 0.02,
            str(both),
            fontsize=7.0,
            color=color,
            va="center",
            ha="left",
        )

        ax.annotate(
            str(formal),
            xy=(formal, yi),
            xytext=(7, -3),
            textcoords="offset points",
            fontsize=6.5,
            fontweight="bold",
            color=SHARED_COLOR,
            va="top",
            ha="left",
        )

    ylabels = [
        f"{contrast_label}  •  {method}"
        for _, contrast_label, method, _, _ in rows
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=6.4, color=INK)
    ax.set_ylim(-0.65, len(rows) + 1.15)
    ax.scatter([], [], s=24, color=AREA_COLOR,
               label="Both individually FDR < 0.05")
    ax.scatter([], [], s=27, color=SHARED_COLOR,
               label="Formal conjunction BH FDR < 0.05")
    ax.legend(
        frameon=False,
        fontsize=5.0,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
        ncol=2,
        handletextpad=0.22,
        columnspacing=1.2,
    )

    max_value = max(
        int(architecture["both_individual_fdr05_n"].max()),
        int(architecture["formal_conjunction_fdr05_n"].max()),
    )
    ax.set_xlim(0, max_value * 1.22)



    ax.set_title(
        "Formal conjunction vs individual significance",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        pad=5,
    )

    style_axis(ax)


# ---------------------------------------------------------------------
# Panel C — recurrent genes
# ---------------------------------------------------------------------


def plot_panel_c(ax, recurrent):
    labels = [label for _, label in CONTRASTS]
    genes = list(recurrent)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(-0.5, len(genes) - 0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.0, color=INK)

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=7.0, color=INK)

    for x in range(len(labels)):
        ax.axvline(
            x,
            color=GRID,
            linewidth=0.7,
            zorder=0,
        )

    for y in range(len(genes)):
        ax.axhline(
            y,
            color=GRID,
            linewidth=0.7,
            zorder=0,
        )

    for yi, gene in enumerate(genes):
        for xi, contrast in enumerate(labels):
            status = recurrent[gene][contrast]

            if status is None:
                continue

            ax.scatter(
                xi,
                yi,
                s=75,
                marker="o",
                color=STATUS_COLORS[status],
                edgecolor=WHITE,
                linewidth=0.8,
                zorder=3,
            )

    ax.legend(
        handles=[
            Patch(
                facecolor=STATUS_COLORS[key],
                label=STATUS_LABELS[key],
            )
            for key in [
                "shared",
                "AREA-only",
                "conventional-only",
                "shared-near",
                "AREA-near",
            ]
        ],
        frameon=False,
        fontsize=5.9,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        handlelength=0.9,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    ax.set_title(
        "Recurrent genes across phenotype definitions",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        pad=5,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        length=0,
        colors=INK,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    source = locate_source(repo_root, args.source)
    formal_source = locate_formal_comparison(
        repo_root,
        args.formal_comparison,
    )
    output_dir = locate_output_dir(
        repo_root,
        args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture = load_architecture(source)
    formal = load_formal_comparison(formal_source)
    evidence_source = (args.evidence_source.expanduser().resolve()
        if args.evidence_source is not None else
        repo_root / "results/stage12_brain_convergence/"
        "stage12E_biological_evidence_matrix/MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv")
    if not evidence_source.is_file():
        raise FileNotFoundError(f"Frozen Stage 12E evidence not found: {evidence_source}")
    recurrent = load_recurrent_evidence(evidence_source)

    fig = plt.figure(
        figsize=(
            mm_to_inches(WIDTH_MM),
            mm_to_inches(HEIGHT_MM),
        )
    )

    # Manuscript-sized canvas. Keep the previously validated compact vertical
    # geometry: enough room inside B/C without creating large inter-panel gaps.
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.00, 1.40, 1.55],
        hspace=0.34,
        left=0.30,
        right=0.985,
        top=0.88,
        bottom=0.16,
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")

    plot_panel_a(ax_a, formal)
    plot_panel_b(ax_b, architecture)
    plot_panel_c(ax_c, recurrent)

    fig.text(
        0.30,
        0.985,
        "RNA–protein convergence across AD phenotype definitions",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )





    fig.text(
        0.30,
        0.045,
        (
            "Formal = joint BH FDR < 0.05; near = both individual FDR < 0.05, same direction."
        ),
        fontsize=5.5,
        color=MID,
        ha="left",
        va="bottom",
    )

    png = output_dir / "Figure6_crossomic_convergence_canonical.png"
    pdf = output_dir / "Figure6_crossomic_convergence_canonical.pdf"
    svg = output_dir / "Figure6_crossomic_convergence_canonical.svg"

    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)

    print(f"Stage 12K source: {source}")
    print(f"Formal comparison source: {formal_source}")
    print(f"Stage 12E evidence source: {evidence_source}")
    import csv
    with (output_dir / "Figure6_formal_partition_audit.csv").open("w", newline="") as h:
        w = csv.writer(h); w.writerow(["contrast_id", "shared", "AREA_only", "conventional_only"])
        for contrast_id, _ in CONTRASTS:
            r = formal.loc[formal.contrast_id.eq(contrast_id)].iloc[0]
            w.writerow([contrast_id, int(r.shared_BH_n), int(r.AREA_only_BH_n), int(r.conventional_only_BH_n)])
    architecture[["contrast_id", "method", "both_individual_fdr05_n", "formal_conjunction_fdr05_n"]].to_csv(
        output_dir / "Figure6_significance_funnel_audit.csv", index=False)
    with (output_dir / "Figure6_gene_status_audit.csv").open("w", newline="") as h:
        w = csv.writer(h); w.writerow(["gene_symbol", "contrast_id", "classification"])
        for gene, statuses in recurrent.items():
            for contrast_id, label in CONTRASTS:
                w.writerow([gene, contrast_id, statuses[label] or "none"])

    print(f"Saved PNG: {png}")
    print(f"Saved PDF: {pdf}")
    print(f"Saved SVG: {svg}")


if __name__ == "__main__":
    main()

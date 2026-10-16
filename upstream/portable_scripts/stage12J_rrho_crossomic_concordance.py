#!/usr/bin/env python3
"""
Stage 12J — directional RRHO cross-omic rank concordance.

Purpose
-------
Compare the COMPLETE RNA and protein ranked profiles without choosing a DEG
threshold.

Stage 12J reads the exact 4,903-gene signed ranks produced by Stage 12I:

    results/stage12_brain_convergence/stage12I_full_rank_gsea/
        ALL_RANKED_PROFILES.tsv

Pairs
-----
For each contrast:
    AREA RNA       vs AREA protein
    DESeq2 RNA     vs limma protein

Directional RRHO quadrants
--------------------------
    up_up:
        strongest positive RNA vs strongest positive protein

    down_down:
        strongest negative RNA vs strongest negative protein

    up_down:
        strongest positive RNA vs strongest negative protein

    down_up:
        strongest negative RNA vs strongest positive protein

No significance cutoff is used to define the tails. Every gene with a positive
or negative signed statistic participates in its corresponding directional
tail.

For each pair of rank thresholds, an upper-tail hypergeometric overlap P value
is calculated using the common N=4,903-gene universe.

Multiple testing
----------------
The original RRHO framework recommends either sample-label permutation or an
analytical Benjamini-Yekutieli correction for the highly dependent grid tests.

We cannot perform the preferred sample-label permutation correction from the
Stage 12 summary tables because that would require refitting AREA, DESeq2, and
limma for every permutation.

Therefore this script applies Benjamini-Yekutieli FDR jointly across ALL grid
cells in the four directional quadrants for each RNA/protein profile pair.

The step size controls map resolution only; it is NOT a biological threshold.
Default = 50 genes, appropriate for ~5,000-gene lists and consistent with the
original RRHO literature's worked scale.

Supporting global metrics
-------------------------
For every RNA/protein pair:
    - Spearman rho across all 4,903 signed scores
    - fraction of genes with the same non-zero direction

Outputs
-------
results/stage12_brain_convergence/stage12J_rrho/
    PROFILE_GLOBAL_CONCORDANCE.tsv
    RRHO_GRID.tsv.gz
    RRHO_QUADRANT_PEAKS.tsv
    RRHO_PEAK_GENES.tsv
    PRIORITY_GENE_RRHO_PEAK_HITS.tsv
    heatmaps/*.png
    README_STAGE12J.txt

Important interpretation
------------------------
A strong up_up or down_down region means continuous concordant cross-omic
signal extends beyond a fixed significance threshold.

A strong up_down or down_up region means systematic cross-modal discordance.

RRHO is especially useful for MCI-vs-AD and late Braak, where Stage 12A/12B
showed substantial within-modality signal but no formal gene-level conjunction
discoveries.
"""

from __future__ import annotations
import os

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

import matplotlib.pyplot as plt
ROOT = Path(os.environ["ROSMAP_OFFLINE_ROOT"]).expanduser().resolve()
RANK_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12I_full_rank_gsea"
    / "ALL_RANKED_PROFILES.tsv"
)

OUTDIR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12J_rrho"
)

HEATMAP_DIR = OUTDIR / "heatmaps"

CONTRASTS = [
    "cogdx_NCI_vs_AD",
    "CERAD_3-4_vs_1-2",
    "cogdx_MCI_vs_AD",
    "Braak_3-4_vs_5-6",
]

METHODS = [
    "AREA",
    "CONVENTIONAL",
]

PRIORITY_GENES = {
    "CSRP1",
    "KIF13B",
    "PPEF1",
    "RAB9A",
    "NDUFAF7",
    "MIGA1",
    "FAM73A",
    "NFS1",
    "DBT",
    "DDAH2",
    "PAFAH1B3",
    "PALM3",
    "CAMKK2",
    "CAMK4",
    "CACNG3",
    "ADAM11",
    "SST",
    "CLPTM1",
    "VGF",
    "NRIP2",
    "SLC38A2",
    "NDUFB5",
    "NRN1",
    "PLXNB1",
}

QUADRANTS = {
    "up_up": {
        "rna_sign": 1,
        "protein_sign": 1,
        "rna_ascending": False,
        "protein_ascending": False,
        "class": "concordant",
    },
    "down_down": {
        "rna_sign": -1,
        "protein_sign": -1,
        "rna_ascending": True,
        "protein_ascending": True,
        "class": "concordant",
    },
    "up_down": {
        "rna_sign": 1,
        "protein_sign": -1,
        "rna_ascending": False,
        "protein_ascending": True,
        "class": "discordant",
    },
    "down_up": {
        "rna_sign": -1,
        "protein_sign": 1,
        "rna_ascending": True,
        "protein_ascending": False,
        "class": "discordant",
    },
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--step",
        type=int,
        default=50,
        help="RRHO rank-grid resolution in genes; not a biological cutoff.",
    )

    p.add_argument(
        "--min-threshold",
        type=int,
        default=50,
        help="Smallest directional tail size scanned.",
    )

    return p.parse_args()


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def bh_log_adjust(
    log_p: np.ndarray,
) -> np.ndarray:
    """
    Benjamini-Hochberg adjustment in natural-log space.

    Input:
        log(p), <= 0

    Output:
        log(q_BH), <= 0
    """
    x = np.asarray(
        log_p,
        dtype=float,
    )

    m = len(x)

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranked = x[order]

    ranks = np.arange(
        1,
        m + 1,
        dtype=float,
    )

    log_q_ranked = (
        ranked
        + math.log(m)
        - np.log(ranks)
    )

    log_q_ranked = np.minimum.accumulate(
        log_q_ranked[::-1]
    )[::-1]

    log_q_ranked = np.minimum(
        log_q_ranked,
        0.0,
    )

    out = np.empty(
        m,
        dtype=float,
    )

    out[order] = log_q_ranked

    return out


def by_log_adjust(
    log_p: np.ndarray,
) -> np.ndarray:
    """
    Benjamini-Yekutieli adjustment in natural-log space.
    """
    x = np.asarray(
        log_p,
        dtype=float,
    )

    m = len(x)

    harmonic = float(
        np.sum(
            1.0
            / np.arange(
                1,
                m + 1,
                dtype=float,
            )
        )
    )

    log_bh = bh_log_adjust(x)

    log_by = (
        log_bh
        + math.log(harmonic)
    )

    return np.minimum(
        log_by,
        0.0,
    )


def thresholds(
    n: int,
    step: int,
    minimum: int,
) -> np.ndarray:
    if n <= 0:
        return np.array(
            [],
            dtype=int,
        )

    start = min(
        minimum,
        n,
    )

    values = list(
        range(
            start,
            n + 1,
            step,
        )
    )

    if not values:
        values = [n]

    if values[-1] != n:
        values.append(n)

    return np.array(
        sorted(set(values)),
        dtype=int,
    )


def get_profile(
    ranks: pd.DataFrame,
    contrast: str,
    method: str,
    modality: str,
) -> pd.DataFrame:
    d = ranks.loc[
        ranks["contrast_id"].astype(str).eq(contrast)
        & ranks["method"].astype(str).eq(method)
        & ranks["modality"].astype(str).eq(modality)
    ].copy()

    if len(d) != 4903:
        raise RuntimeError(
            f"{contrast} {method} {modality}: "
            f"expected 4,903 genes; found {len(d)}."
        )

    if d["gene_symbol"].duplicated().any():
        raise RuntimeError(
            f"{contrast} {method} {modality}: duplicate genes."
        )

    needed = {
        "gene_symbol",
        "signed_normal_score",
        "direction_sign",
    }

    missing = needed - set(d.columns)

    if missing:
        raise RuntimeError(
            f"Rank table missing: {sorted(missing)}"
        )

    d["signed_normal_score"] = pd.to_numeric(
        d["signed_normal_score"],
        errors="coerce",
    )

    d["direction_sign"] = pd.to_numeric(
        d["direction_sign"],
        errors="coerce",
    )

    if d[
        [
            "signed_normal_score",
            "direction_sign",
        ]
    ].isna().any().any():
        raise RuntimeError(
            f"{contrast} {method} {modality}: "
            "NA rank values."
        )

    return d


def rank_tail(
    profile: pd.DataFrame,
    sign: int,
    ascending: bool,
) -> list[str]:
    d = profile.loc[
        profile["direction_sign"].eq(sign)
    ].copy()

    d = d.sort_values(
        [
            "signed_normal_score",
            "gene_symbol",
        ],
        ascending=[
            ascending,
            True,
        ],
        kind="mergesort",
    )

    return d[
        "gene_symbol"
    ].astype(str).tolist()


def rrho_grid(
    list1: list[str],
    list2: list[str],
    universe_n: int,
    step: int,
    minimum: int,
):
    """
    Efficiently calculate hypergeometric overlap at a grid of tail sizes.
    """
    t1 = thresholds(
        len(list1),
        step,
        minimum,
    )

    t2 = thresholds(
        len(list2),
        step,
        minimum,
    )

    if len(t1) == 0 or len(t2) == 0:
        return None

    rank1 = {
        gene: i
        for i, gene in enumerate(
            list1,
            start=1,
        )
    }

    rank2 = {
        gene: i
        for i, gene in enumerate(
            list2,
            start=1,
        )
    }

    common = set(rank1) & set(rank2)

    # Each overlapping gene contributes from the first threshold that contains
    # it onward. A 2D histogram followed by cumulative sums gives every overlap.
    hist = np.zeros(
        (len(t1), len(t2)),
        dtype=int,
    )

    for gene in common:
        i = np.searchsorted(
            t1,
            rank1[gene],
            side="left",
        )
        j = np.searchsorted(
            t2,
            rank2[gene],
            side="left",
        )

        if (
            i < len(t1)
            and j < len(t2)
        ):
            hist[i, j] += 1

    overlap = np.cumsum(
        np.cumsum(
            hist,
            axis=0,
        ),
        axis=1,
    )

    n1 = t1[:, None]
    n2 = t2[None, :]

    expected = (
        n1
        * n2
        / float(universe_n)
    )

    # sf(k - 1) = P[X >= k]
    log_p = hypergeom.logsf(
        overlap - 1,
        universe_n,
        n1,
        n2,
    )

    # For k=0, sf(-1) is exactly 1 => log p = 0.
    log_p = np.where(
        np.isfinite(log_p),
        log_p,
        -np.inf,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        enrichment_ratio = (
            overlap
            / expected
        )

    return {
        "thresholds_1": t1,
        "thresholds_2": t2,
        "overlap": overlap,
        "expected": expected,
        "enrichment_ratio":
            enrichment_ratio,
        "log_p": log_p,
        "rank1": rank1,
        "rank2": rank2,
    }


def matrix_to_rows(
    contrast: str,
    method: str,
    quadrant: str,
    quadrant_class: str,
    grid,
):
    rows = []

    t1 = grid["thresholds_1"]
    t2 = grid["thresholds_2"]

    for i, n1 in enumerate(t1):
        for j, n2 in enumerate(t2):
            lp = float(
                grid["log_p"][i, j]
            )

            rows.append(
                {
                    "contrast_id": contrast,
                    "method": method,
                    "quadrant": quadrant,
                    "quadrant_class":
                        quadrant_class,
                    "rna_tail_n": int(n1),
                    "protein_tail_n":
                        int(n2),
                    "overlap_n":
                        int(
                            grid[
                                "overlap"
                            ][i, j]
                        ),
                    "expected_overlap":
                        float(
                            grid[
                                "expected"
                            ][i, j]
                        ),
                    "enrichment_ratio":
                        float(
                            grid[
                                "enrichment_ratio"
                            ][i, j]
                        ),
                    "log_pvalue": lp,
                    "neglog10_pvalue":
                        (
                            -lp
                            / math.log(10)
                            if np.isfinite(lp)
                            else np.inf
                        ),
                }
            )

    return rows


def make_heatmap(
    d: pd.DataFrame,
    path: Path,
    title: str,
):
    pivot = d.pivot(
        index="rna_tail_n",
        columns="protein_tail_n",
        values="neglog10_by_fdr",
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    matrix = pivot.to_numpy(dtype=float)

    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        cap = float(finite.max()) + 1.0
    else:
        cap = 1.0

    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=cap,
        neginf=0.0,
    )

    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel(
        "Protein directional-tail size"
    )
    ax.set_ylabel(
        "RNA directional-tail size"
    )

    x_positions = np.linspace(
        0,
        len(pivot.columns) - 1,
        min(
            6,
            len(pivot.columns),
        ),
        dtype=int,
    )

    y_positions = np.linspace(
        0,
        len(pivot.index) - 1,
        min(
            6,
            len(pivot.index),
        ),
        dtype=int,
    )

    ax.set_xticks(
        x_positions,
        labels=[
            str(
                int(
                    pivot.columns[i]
                )
            )
            for i in x_positions
        ],
    )

    ax.set_yticks(
        y_positions,
        labels=[
            str(
                int(
                    pivot.index[i]
                )
            )
            for i in y_positions
        ],
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )
    cbar.set_label(
        "-log10(BY FDR)"
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


def main():
    args = parse_args()

    require(RANK_PATH)

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    HEATMAP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranks = pd.read_csv(
        RANK_PATH,
        sep="\t",
        low_memory=False,
    )

    universe = set(
        ranks[
            "gene_symbol"
        ].astype(str)
    )

    if len(universe) != 4903:
        raise RuntimeError(
            f"Expected common 4,903-gene universe; "
            f"found {len(universe)}."
        )

    print("=" * 120)
    print(
        "STAGE 12J — DIRECTIONAL RRHO "
        "RNA/PROTEIN CONCORDANCE"
    )
    print("=" * 120)
    print(
        "Universe: 4,903 genes"
    )
    print(
        f"Grid step: {args.step} genes "
        "(map resolution, not biological threshold)"
    )
    print(
        "Correction: Benjamini-Yekutieli "
        "across all four quadrant grids per profile pair"
    )

    grid_rows = []
    global_rows = []
    pair_tail_data = {}

    for contrast in CONTRASTS:
        for method in METHODS:
            rna = get_profile(
                ranks,
                contrast,
                method,
                "RNA",
            )

            protein = get_profile(
                ranks,
                contrast,
                method,
                "PROTEIN",
            )

            merged = rna[
                [
                    "gene_symbol",
                    "signed_normal_score",
                    "direction_sign",
                ]
            ].merge(
                protein[
                    [
                        "gene_symbol",
                        "signed_normal_score",
                        "direction_sign",
                    ]
                ],
                on="gene_symbol",
                how="inner",
                suffixes=(
                    "_rna",
                    "_protein",
                ),
                validate="one_to_one",
            )

            rho, rho_p = spearmanr(
                merged[
                    "signed_normal_score_rna"
                ],
                merged[
                    "signed_normal_score_protein"
                ],
            )

            nonzero = (
                merged[
                    "direction_sign_rna"
                ].ne(0)
                & merged[
                    "direction_sign_protein"
                ].ne(0)
            )

            same_sign = (
                merged.loc[
                    nonzero,
                    "direction_sign_rna",
                ].to_numpy()
                == merged.loc[
                    nonzero,
                    "direction_sign_protein",
                ].to_numpy()
            )

            global_rows.append(
                {
                    "contrast_id": contrast,
                    "method": method,
                    "n_genes": len(merged),
                    "spearman_rho":
                        float(rho),
                    "spearman_pvalue":
                        float(rho_p),
                    "nonzero_direction_n":
                        int(nonzero.sum()),
                    "same_direction_n":
                        int(same_sign.sum()),
                    "same_direction_fraction":
                        float(
                            same_sign.mean()
                        )
                        if len(same_sign)
                        else np.nan,
                }
            )

            pair_key = (
                contrast,
                method,
            )

            pair_tail_data[
                pair_key
            ] = {}

            for quadrant, spec in QUADRANTS.items():
                list1 = rank_tail(
                    rna,
                    spec["rna_sign"],
                    spec["rna_ascending"],
                )

                list2 = rank_tail(
                    protein,
                    spec["protein_sign"],
                    spec[
                        "protein_ascending"
                    ],
                )

                grid = rrho_grid(
                    list1,
                    list2,
                    4903,
                    args.step,
                    args.min_threshold,
                )

                pair_tail_data[
                    pair_key
                ][quadrant] = {
                    "list1": list1,
                    "list2": list2,
                    "grid": grid,
                }

                if grid is None:
                    continue

                grid_rows.extend(
                    matrix_to_rows(
                        contrast,
                        method,
                        quadrant,
                        spec["class"],
                        grid,
                    )
                )

    grid_df = pd.DataFrame(
        grid_rows
    )

    if grid_df.empty:
        raise RuntimeError(
            "RRHO grid is empty."
        )

    # BY correction jointly across all four directional maps per profile pair.
    corrected_parts = []

    for (
        contrast,
        method,
    ), g in grid_df.groupby(
        [
            "contrast_id",
            "method",
        ],
        sort=False,
    ):
        x = g.copy()

        log_by = by_log_adjust(
            x[
                "log_pvalue"
            ].to_numpy()
        )

        x["log_by_fdr"] = log_by

        x["neglog10_by_fdr"] = np.where(
            np.isfinite(log_by),
            -log_by / math.log(10),
            np.inf,
        )

        # Conventional q-value column. May underflow to zero for extremely
        # strong signals; neglog10_by_fdr preserves the magnitude.
        x["by_fdr"] = np.exp(
            np.maximum(
                log_by,
                math.log(
                    np.finfo(float).tiny
                ),
            )
        )

        corrected_parts.append(x)

    grid_df = pd.concat(
        corrected_parts,
        ignore_index=True,
    )

    peak_rows = []
    peak_gene_rows = []
    priority_rows = []

    for (
        contrast,
        method,
        quadrant,
    ), g in grid_df.groupby(
        [
            "contrast_id",
            "method",
            "quadrant",
        ],
        sort=False,
    ):
        g = g.copy()

        peak_idx = (
            g["log_by_fdr"]
            .astype(float)
            .idxmin()
        )

        peak = grid_df.loc[
            peak_idx
        ]

        tail = pair_tail_data[
            (
                contrast,
                method,
            )
        ][quadrant]

        list1 = tail["list1"]
        list2 = tail["list2"]

        n1 = int(
            peak["rna_tail_n"]
        )
        n2 = int(
            peak["protein_tail_n"]
        )

        genes = sorted(
            set(
                list1[:n1]
            )
            & set(
                list2[:n2]
            )
        )

        priority = sorted(
            set(genes)
            & PRIORITY_GENES
        )

        peak_rows.append(
            {
                "contrast_id": contrast,
                "method": method,
                "quadrant": quadrant,
                "quadrant_class":
                    peak[
                        "quadrant_class"
                    ],
                "peak_rna_tail_n": n1,
                "peak_protein_tail_n":
                    n2,
                "peak_overlap_n":
                    int(
                        peak[
                            "overlap_n"
                        ]
                    ),
                "peak_expected_overlap":
                    float(
                        peak[
                            "expected_overlap"
                        ]
                    ),
                "peak_enrichment_ratio":
                    float(
                        peak[
                            "enrichment_ratio"
                        ]
                    ),
                "peak_neglog10_raw_p":
                    float(
                        peak[
                            "neglog10_pvalue"
                        ]
                    ),
                "peak_by_fdr":
                    float(
                        peak[
                            "by_fdr"
                        ]
                    ),
                "peak_neglog10_by_fdr":
                    float(
                        peak[
                            "neglog10_by_fdr"
                        ]
                    ),
                "peak_genes":
                    ",".join(genes),
                "priority_genes":
                    ",".join(priority),
            }
        )

        for gene in genes:
            peak_gene_rows.append(
                {
                    "contrast_id":
                        contrast,
                    "method": method,
                    "quadrant":
                        quadrant,
                    "gene_symbol":
                        gene,
                    "priority_gene":
                        gene
                        in PRIORITY_GENES,
                }
            )

        for gene in priority:
            priority_rows.append(
                {
                    "contrast_id":
                        contrast,
                    "method": method,
                    "quadrant":
                        quadrant,
                    "quadrant_class":
                        peak[
                            "quadrant_class"
                        ],
                    "peak_by_fdr":
                        float(
                            peak[
                                "by_fdr"
                            ]
                        ),
                    "peak_neglog10_by_fdr":
                        float(
                            peak[
                                "neglog10_by_fdr"
                            ]
                        ),
                    "gene_symbol":
                        gene,
                }
            )

        heatmap_path = (
            HEATMAP_DIR
            / (
                f"{contrast}"
                f"__{method}"
                f"__{quadrant}.png"
            )
        )

        make_heatmap(
            g,
            heatmap_path,
            (
                f"{contrast} | {method} | "
                f"{quadrant}"
            ),
        )

    global_df = pd.DataFrame(
        global_rows
    )

    peaks = pd.DataFrame(
        peak_rows
    )

    peak_genes = pd.DataFrame(
        peak_gene_rows
    )

    priority_hits = pd.DataFrame(
        priority_rows
    )

    # Compact pair-level comparison of concordant versus discordant maxima.
    pair_summary_rows = []

    for (
        contrast,
        method,
    ), g in peaks.groupby(
        [
            "contrast_id",
            "method",
        ],
        sort=False,
    ):
        concordant = g.loc[
            g[
                "quadrant_class"
            ].eq("concordant")
        ]

        discordant = g.loc[
            g[
                "quadrant_class"
            ].eq("discordant")
        ]

        c_idx = (
            concordant[
                "peak_neglog10_by_fdr"
            ].idxmax()
        )

        d_idx = (
            discordant[
                "peak_neglog10_by_fdr"
            ].idxmax()
        )

        c = peaks.loc[c_idx]
        d = peaks.loc[d_idx]

        pair_summary_rows.append(
            {
                "contrast_id": contrast,
                "method": method,
                "strongest_concordant_quadrant":
                    c["quadrant"],
                "strongest_concordant_neglog10_BY":
                    c[
                        "peak_neglog10_by_fdr"
                    ],
                "strongest_concordant_overlap_n":
                    c["peak_overlap_n"],
                "strongest_concordant_priority_genes":
                    c["priority_genes"],
                "strongest_discordant_quadrant":
                    d["quadrant"],
                "strongest_discordant_neglog10_BY":
                    d[
                        "peak_neglog10_by_fdr"
                    ],
                "strongest_discordant_overlap_n":
                    d["peak_overlap_n"],
                "strongest_discordant_priority_genes":
                    d["priority_genes"],
            }
        )

    pair_summary = pd.DataFrame(
        pair_summary_rows
    )

    pair_summary = pair_summary.merge(
        global_df,
        on=[
            "contrast_id",
            "method",
        ],
        how="left",
        validate="one_to_one",
    )

    grid_path = (
        OUTDIR
        / "RRHO_GRID.tsv.gz"
    )

    global_path = (
        OUTDIR
        / "PROFILE_GLOBAL_CONCORDANCE.tsv"
    )

    peaks_path = (
        OUTDIR
        / "RRHO_QUADRANT_PEAKS.tsv"
    )

    peak_genes_path = (
        OUTDIR
        / "RRHO_PEAK_GENES.tsv"
    )

    priority_path = (
        OUTDIR
        / "PRIORITY_GENE_RRHO_PEAK_HITS.tsv"
    )

    pair_summary_path = (
        OUTDIR
        / "RRHO_PAIR_SUMMARY.tsv"
    )

    readme_path = (
        OUTDIR
        / "README_STAGE12J.txt"
    )

    grid_df.to_csv(
        grid_path,
        sep="\t",
        index=False,
        compression="gzip",
    )

    global_df.to_csv(
        global_path,
        sep="\t",
        index=False,
    )

    peaks.to_csv(
        peaks_path,
        sep="\t",
        index=False,
    )

    peak_genes.to_csv(
        peak_genes_path,
        sep="\t",
        index=False,
    )

    priority_hits.to_csv(
        priority_path,
        sep="\t",
        index=False,
    )

    pair_summary.to_csv(
        pair_summary_path,
        sep="\t",
        index=False,
    )

    readme = f"""STAGE 12J — DIRECTIONAL RRHO
=================================

Input:
    {RANK_PATH}

Universe:
    4,903 exact one-to-one cross-omic genes.

Contrasts:
    {", ".join(CONTRASTS)}

Pairs:
    AREA RNA vs AREA protein
    conventional DESeq2 RNA vs limma protein

Directional quadrants:
    up_up       concordant
    down_down   concordant
    up_down     discordant
    down_up     discordant

Grid:
    step = {args.step} genes
    minimum tail = {args.min_threshold} genes

The step is only computational/map resolution. It is not a biological
selection threshold.

Multiple testing:
    Benjamini-Yekutieli FDR is applied jointly across all scanned cells in
    all four quadrants for each RNA/protein pair.

Why BY rather than permutations?
    The preferred RRHO sample-label permutation procedure would require
    refitting AREA, DESeq2, and limma for every permutation. Stage 12J begins
    from frozen differential-association results, so BY provides the analytical
    dependency-robust correction described in the original RRHO framework.

Key outputs:
    RRHO_PAIR_SUMMARY.tsv
        Compact comparison of the strongest concordant and discordant map
        regions plus whole-profile Spearman correlation.

    RRHO_QUADRANT_PEAKS.tsv
        Most enriched BY-corrected point in each directional quadrant.

    RRHO_PEAK_GENES.tsv
        Genes inside each quadrant's peak overlap.

    PRIORITY_GENE_RRHO_PEAK_HITS.tsv
        Priority genes captured by those peak overlap regions.

    RRHO_GRID.tsv.gz
        Complete scanned map with raw hypergeometric P and BY FDR.

    heatmaps/
        One corrected RRHO heatmap per pair and directional quadrant.

Interpretation:
    Concordant up_up/down_down signal indicates cross-omic rank agreement that
    extends beyond a fixed gene-level significance threshold.

    Discordant up_down/down_up signal indicates systematic RNA/protein
    directional disagreement.

    This analysis is intended to complement, not replace, Stage 12A/12B formal
    gene-level conjunction testing.
"""

    readme_path.write_text(
        readme
    )

    print("\n" + "=" * 120)
    print("RRHO PAIR SUMMARY")
    print("=" * 120)
    print(
        pair_summary
        .round(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("QUADRANT PEAKS")
    print("=" * 120)

    show = [
        "contrast_id",
        "method",
        "quadrant",
        "peak_rna_tail_n",
        "peak_protein_tail_n",
        "peak_overlap_n",
        "peak_enrichment_ratio",
        "peak_by_fdr",
        "peak_neglog10_by_fdr",
        "priority_genes",
    ]

    print(
        peaks[
            show
        ]
        .sort_values(
            [
                "contrast_id",
                "method",
                "quadrant",
            ]
        )
        .round(5)
        .to_string(index=False)
    )

    print("\nPRIORITY GENES IN RRHO PEAK OVERLAPS")
    print("-" * 120)

    if priority_hits.empty:
        print("None.")
    else:
        print(
            priority_hits
            .sort_values(
                [
                    "contrast_id",
                    "method",
                    "quadrant",
                    "gene_symbol",
                ]
            )
            .round(5)
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in [
        global_path,
        pair_summary_path,
        grid_path,
        peaks_path,
        peak_genes_path,
        priority_path,
        readme_path,
    ]:
        print(" ", path)

    print(
        " ",
        HEATMAP_DIR,
    )


if __name__ == "__main__":
    main()

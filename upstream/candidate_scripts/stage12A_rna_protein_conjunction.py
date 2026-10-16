#!/usr/bin/env python3
"""
Stage 12A revised — formal DLPFC RNA / TMT protein conjunction analysis.

Primary inferential question
----------------------------
For each gene and phenotype/pathology contrast, is there evidence that BOTH
DLPFC RNA and DLPFC protein are associated in the SAME direction?

No arbitrary top-X-percent threshold is used.

Cross-omic conjunction test
---------------------------
For each one-to-one mapped gene:

    if AREA directions agree:
        p_conjunction = max(p_RNA, p_protein)
    else:
        p_conjunction = 1

Why max(p)?
Under the conjunction null, at least one modality is null. Rejecting only
when BOTH p-values are small is an intersection-union test. max(p1, p2) is
therefore a valid conservative conjunction p-value without requiring
independence between the two modalities.

Multiple testing
----------------
Per contrast, p_conjunction is adjusted across the full one-to-one mapped
gene universe using:
    - Benjamini-Hochberg FDR (primary, matching standard omics practice)
    - Benjamini-Yekutieli FDR (dependency-robust sensitivity analysis)

Also reported:
    - whether both modality-specific AREA padj values are < 0.05
    - continuous rank-product score for descriptive ordering only

The prior top-10% / top-5% heuristic is NOT used in this analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OFFLINE = Path.home() / "Developer/stage11_offline_2026-09-28"
REPO = Path.home() / "Developer/area-workspace/ROSMAP-AREA-DNA-rotation"

TASK_FILE = (
    OFFLINE
    / "results/stage11_covariate_stress/stage11_tasks.tsv"
)

GENE_UNIVERSE = (
    REPO
    / "results/preprocessing_protein_coding"
    / "ROSMAP_global_gene_universe.tsv"
)

OUTDIR = (
    OFFLINE
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


CONTRAST_ORDER = [
    "cogdx_NCI_vs_MCI",
    "cogdx_MCI_vs_AD",
    "cogdx_NCI_vs_AD",
    "Braak_0-2_vs_3-4",
    "Braak_3-4_vs_5-6",
    "CERAD_3-4_vs_1-2",
]


def localize_area_result(path_string: str) -> Path:
    marker = "/results/multiomic_models/"
    s = str(path_string)

    if marker not in s:
        raise ValueError(f"Cannot localize AREA result: {s}")

    return REPO / "results/multiomic_models" / s.split(marker, 1)[1]


def deterministic_rank(df: pd.DataFrame) -> np.ndarray:
    work = pd.DataFrame(
        {
            "feature_id": df["feature_id"].astype(str),
            "p": pd.to_numeric(
                df["pvalue"],
                errors="coerce",
            ).fillna(np.inf),
        },
        index=df.index,
    )

    ordered = work.sort_values(
        ["p", "feature_id"],
        ascending=[True, True],
        kind="mergesort",
    ).index

    ranks = pd.Series(
        np.arange(1, len(df) + 1),
        index=ordered,
        dtype=int,
    )

    return ranks.reindex(df.index).to_numpy()


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    p = np.clip(p, 0.0, 1.0)

    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]

    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = q_ranked
    return q


def by_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values for arbitrary dependence."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    return np.clip(bh_adjust(p) * harmonic, 0.0, 1.0)


def load_gene_map() -> pd.DataFrame:
    g = pd.read_csv(
        GENE_UNIVERSE,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    needed = {"ensembl_id_version", "ensembl_id", "gene_name"}
    missing = needed - set(g.columns)

    if missing:
        raise RuntimeError(
            f"Gene universe missing columns: {sorted(missing)}"
        )

    if len(g) != 17646:
        raise RuntimeError(
            f"Expected 17,646 DLPFC genes; found {len(g)}."
        )

    if g["ensembl_id_version"].duplicated().any():
        raise RuntimeError(
            "Duplicate Ensembl versioned IDs in frozen gene universe."
        )

    g["gene_name"] = g["gene_name"].astype(str).str.strip()

    return g[
        ["ensembl_id_version", "ensembl_id", "gene_name"]
    ].copy()


def load_area(path: Path, expected_n: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    d = pd.read_csv(path, low_memory=False)

    needed = {"feature_id", "pvalue", "padj", "direction"}
    missing = needed - set(d.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing columns: {sorted(missing)}"
        )

    if len(d) != expected_n:
        raise RuntimeError(
            f"{path}: expected {expected_n}, found {len(d)} features."
        )

    if d["feature_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{path}: duplicate feature IDs.")

    d = d.copy()

    d["feature_id"] = d["feature_id"].astype(str).str.strip()
    d["pvalue"] = pd.to_numeric(d["pvalue"], errors="coerce")
    d["padj"] = pd.to_numeric(d["padj"], errors="coerce")
    d["direction"] = d["direction"].astype(str).str.strip()

    d["rank"] = deterministic_rank(d)
    d["rank_fraction"] = d["rank"] / len(d)

    return d


def make_one_to_one_symbols(
    gene_map: pd.DataFrame,
    representative_tmt: pd.DataFrame,
):
    dlpfc_counts = (
        gene_map.groupby("gene_name")
        .size()
        .rename("n_dlpfc_genes")
        .reset_index()
    )

    t = representative_tmt[["feature_id"]].copy()

    parsed = t["feature_id"].str.split(
        "|",
        n=1,
        regex=False,
        expand=True,
    )

    if parsed.shape[1] != 2:
        raise RuntimeError(
            "TMT IDs do not parse as GENE_SYMBOL|UniProt."
        )

    t["gene_symbol"] = parsed[0].str.strip()
    t["uniprot_accession"] = parsed[1].str.strip()

    tmt_counts = (
        t.groupby("gene_symbol")
        .size()
        .rename("n_tmt_features")
        .reset_index()
    )

    audit = dlpfc_counts.merge(
        tmt_counts,
        left_on="gene_name",
        right_on="gene_symbol",
        how="outer",
    )

    audit["one_to_one"] = (
        audit["n_dlpfc_genes"].eq(1)
        & audit["n_tmt_features"].eq(1)
    )

    symbols = set(
        audit.loc[
            audit["one_to_one"],
            "gene_name",
        ].dropna().astype(str)
    )

    return audit, symbols


def integrate_pair(
    contrast_id: str,
    dlpfc: pd.DataFrame,
    tmt: pd.DataFrame,
    gene_map: pd.DataFrame,
    one_to_one_symbols: set[str],
):
    # RNA mapping.
    rna = dlpfc.merge(
        gene_map,
        left_on="feature_id",
        right_on="ensembl_id_version",
        how="left",
        validate="one_to_one",
    )

    if rna["gene_name"].isna().any():
        raise RuntimeError(
            f"{contrast_id}: DLPFC Ensembl mapping incomplete."
        )

    rna = rna.loc[
        rna["gene_name"].isin(one_to_one_symbols)
    ].copy()

    rna = rna.rename(
        columns={
            "feature_id": "rna_feature_id",
            "gene_name": "gene_symbol",
            "pvalue": "rna_pvalue",
            "padj": "rna_padj",
            "direction": "rna_direction",
            "rank": "rna_rank",
            "rank_fraction": "rna_rank_fraction",
        }
    )

    # Protein mapping.
    prot = tmt.copy()

    parsed = prot["feature_id"].str.split(
        "|",
        n=1,
        regex=False,
        expand=True,
    )

    if parsed.shape[1] != 2:
        raise RuntimeError(
            f"{contrast_id}: TMT IDs do not parse SYMBOL|accession."
        )

    prot["gene_symbol"] = parsed[0].str.strip()
    prot["uniprot_accession"] = parsed[1].str.strip()

    prot = prot.loc[
        prot["gene_symbol"].isin(one_to_one_symbols)
    ].copy()

    prot = prot.rename(
        columns={
            "feature_id": "protein_feature_id",
            "pvalue": "protein_pvalue",
            "padj": "protein_padj",
            "direction": "protein_direction",
            "rank": "protein_rank",
            "rank_fraction": "protein_rank_fraction",
        }
    )

    rna_cols = [
        "gene_symbol",
        "ensembl_id_version",
        "ensembl_id",
        "rna_feature_id",
        "rna_pvalue",
        "rna_padj",
        "rna_direction",
        "rna_rank",
        "rna_rank_fraction",
    ]

    prot_cols = [
        "gene_symbol",
        "uniprot_accession",
        "protein_feature_id",
        "protein_pvalue",
        "protein_padj",
        "protein_direction",
        "protein_rank",
        "protein_rank_fraction",
    ]

    d = rna[rna_cols].merge(
        prot[prot_cols],
        on="gene_symbol",
        how="inner",
        validate="one_to_one",
    )

    if len(d) != len(one_to_one_symbols):
        raise RuntimeError(
            f"{contrast_id}: expected {len(one_to_one_symbols)} "
            f"one-to-one genes; merged {len(d)}."
        )

    d["contrast_id"] = contrast_id

    d["direction_concordant"] = (
        d["rna_direction"] == d["protein_direction"]
    )

    rna_p = pd.to_numeric(
        d["rna_pvalue"],
        errors="coerce",
    ).fillna(1.0)

    prot_p = pd.to_numeric(
        d["protein_pvalue"],
        errors="coerce",
    ).fillna(1.0)

    d["conjunction_pvalue"] = np.maximum(
        rna_p,
        prot_p,
    )

    # Discordant direction cannot qualify as same-direction convergence.
    d.loc[
        ~d["direction_concordant"],
        "conjunction_pvalue",
    ] = 1.0

    d["conjunction_fdr_bh"] = bh_adjust(
        d["conjunction_pvalue"].to_numpy()
    )

    d["conjunction_fdr_by"] = by_adjust(
        d["conjunction_pvalue"].to_numpy()
    )

    d["crossomic_fdr05_bh"] = (
        d["conjunction_fdr_bh"] < 0.05
    )

    d["crossomic_fdr05_by"] = (
        d["conjunction_fdr_by"] < 0.05
    )

    d["both_individual_fdr05"] = (
        d["direction_concordant"]
        & d["rna_padj"].lt(0.05)
        & d["protein_padj"].lt(0.05)
    )

    d["both_nominal_p05"] = (
        d["direction_concordant"]
        & d["rna_pvalue"].lt(0.05)
        & d["protein_pvalue"].lt(0.05)
    )

    # Descriptive ordering only; NOT a discovery threshold.
    d["crossomic_rank_product"] = np.sqrt(
        d["rna_rank_fraction"]
        * d["protein_rank_fraction"]
    )

    d = d.sort_values(
        [
            "conjunction_fdr_bh",
            "conjunction_pvalue",
            "crossomic_rank_product",
            "gene_symbol",
        ],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    d["crossomic_display_rank"] = np.arange(
        1,
        len(d) + 1,
    )

    summary = {
        "contrast_id": contrast_id,
        "mapped_genes": len(d),
        "direction_concordant_n":
            int(d["direction_concordant"].sum()),
        "direction_concordant_fraction":
            float(d["direction_concordant"].mean()),
        "both_nominal_p05_n":
            int(d["both_nominal_p05"].sum()),
        "both_individual_fdr05_n":
            int(d["both_individual_fdr05"].sum()),
        "conjunction_fdr05_bh_n":
            int(d["crossomic_fdr05_bh"].sum()),
        "conjunction_fdr05_by_n":
            int(d["crossomic_fdr05_by"].sum()),
        "rna_fdr05_among_mapped":
            int(d["rna_padj"].lt(0.05).sum()),
        "protein_fdr05_among_mapped":
            int(d["protein_padj"].lt(0.05).sum()),
        "min_conjunction_fdr_bh":
            float(d["conjunction_fdr_bh"].min()),
        "min_conjunction_fdr_by":
            float(d["conjunction_fdr_by"].min()),
    }

    return d, summary


def build_recurrence(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for gene, g in all_results.groupby(
        "gene_symbol",
        sort=False,
    ):
        rows.append(
            {
                "gene_symbol": gene,
                "n_contrasts_direction_concordant":
                    int(g["direction_concordant"].sum()),
                "n_contrasts_conjunction_fdr05_bh":
                    int(g["crossomic_fdr05_bh"].sum()),
                "n_contrasts_conjunction_fdr05_by":
                    int(g["crossomic_fdr05_by"].sum()),
                "n_contrasts_both_individual_fdr05":
                    int(g["both_individual_fdr05"].sum()),
                "best_conjunction_pvalue":
                    float(g["conjunction_pvalue"].min()),
                "best_conjunction_fdr_bh":
                    float(g["conjunction_fdr_bh"].min()),
                "bh_significant_contrasts": ",".join(
                    g.loc[
                        g["crossomic_fdr05_bh"],
                        "contrast_id",
                    ].astype(str)
                ),
                "by_significant_contrasts": ",".join(
                    g.loc[
                        g["crossomic_fdr05_by"],
                        "contrast_id",
                    ].astype(str)
                ),
            }
        )

    out = pd.DataFrame(rows)

    return out.sort_values(
        [
            "n_contrasts_conjunction_fdr05_bh",
            "best_conjunction_fdr_bh",
            "best_conjunction_pvalue",
            "gene_symbol",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def main():
    gene_map = load_gene_map()

    tasks = pd.read_csv(
        TASK_FILE,
        sep="\t",
    )

    brain = tasks.loc[
        tasks["modality"].isin(["dlpfc", "tmt"])
    ].copy()

    rep = brain.loc[
        brain["modality"].eq("tmt")
    ].iloc[0]

    representative_tmt = load_area(
        localize_area_result(rep["A_result"]),
        int(rep["A_n_features"]),
    )

    mapping_audit, one_to_one_symbols = (
        make_one_to_one_symbols(
            gene_map,
            representative_tmt,
        )
    )

    print("=" * 110)
    print("STAGE 12A REVISED — FORMAL RNA/PROTEIN CONJUNCTION")
    print("=" * 110)
    print(f"One-to-one mapped genes: {len(one_to_one_symbols):,}")
    print("Primary threshold: conjunction BH FDR < 0.05")
    print("Sensitivity threshold: conjunction BY FDR < 0.05")
    print("No top-X-percent selection is used.")

    mapping_path = (
        OUTDIR / "RNA_PROTEIN_MAPPING_AUDIT.tsv"
    )

    mapping_audit.to_csv(
        mapping_path,
        sep="\t",
        index=False,
    )

    tables = []
    summaries = []

    for contrast_id in CONTRAST_ORDER:
        pair = brain.loc[
            brain["contrast_id"].astype(str).eq(contrast_id)
        ]

        if set(pair["modality"]) != {"dlpfc", "tmt"}:
            raise RuntimeError(
                f"{contrast_id}: missing DLPFC/TMT pair."
            )

        loaded = {}

        for r in pair.itertuples(index=False):
            loaded[str(r.modality)] = load_area(
                localize_area_result(r.A_result),
                int(r.A_n_features),
            )

        merged, summary = integrate_pair(
            contrast_id,
            loaded["dlpfc"],
            loaded["tmt"],
            gene_map,
            one_to_one_symbols,
        )

        tables.append(merged)
        summaries.append(summary)

        merged.to_csv(
            OUTDIR / (
                f"{contrast_id}"
                "__RNA_PROTEIN_CONJUNCTION.tsv"
            ),
            sep="\t",
            index=False,
        )

    all_results = pd.concat(
        tables,
        ignore_index=True,
    )

    summary = pd.DataFrame(summaries)

    summary["contrast_id"] = pd.Categorical(
        summary["contrast_id"],
        categories=CONTRAST_ORDER,
        ordered=True,
    )

    summary = summary.sort_values(
        "contrast_id"
    ).reset_index(drop=True)

    summary["contrast_id"] = summary[
        "contrast_id"
    ].astype(str)

    recurrence = build_recurrence(
        all_results
    )

    all_path = (
        OUTDIR
        / "ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv"
    )
    summary_path = (
        OUTDIR
        / "CONTRAST_CONJUNCTION_SUMMARY.tsv"
    )
    recurrence_path = (
        OUTDIR
        / "CROSS_CONTRAST_CONJUNCTION_RECURRENCE.tsv"
    )

    all_results.to_csv(
        all_path,
        sep="\t",
        index=False,
    )

    summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
    )

    recurrence.to_csv(
        recurrence_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 110)
    print("CONJUNCTION SUMMARY")
    print("=" * 110)
    print(
        summary
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("BH-FDR CROSS-OMIC DISCOVERIES")
    print("=" * 110)

    for contrast_id in CONTRAST_ORDER:
        d = all_results.loc[
            all_results["contrast_id"].eq(contrast_id)
            & all_results["crossomic_fdr05_bh"]
        ].copy()

        print(f"\n--- {contrast_id}: {len(d)} genes ---")

        if d.empty:
            print("No same-direction RNA/protein conjunction discoveries at BH FDR < 0.05.")
            continue

        print(
            d[
                [
                    "gene_symbol",
                    "rna_direction",
                    "rna_pvalue",
                    "protein_pvalue",
                    "conjunction_pvalue",
                    "conjunction_fdr_bh",
                    "conjunction_fdr_by",
                    "rna_padj",
                    "protein_padj",
                ]
            ]
            .head(50)
            .round(7)
            .to_string(index=False)
        )

    print("\n" + "=" * 110)
    print("MOST RECURRENT FORMAL CROSS-OMIC DISCOVERIES")
    print("=" * 110)

    recurrent_sig = recurrence.loc[
        recurrence[
            "n_contrasts_conjunction_fdr05_bh"
        ] > 0
    ]

    if recurrent_sig.empty:
        print("No conjunction-BH discoveries in any contrast.")
    else:
        print(
            recurrent_sig
            .head(50)
            .round(7)
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in [
        mapping_path,
        summary_path,
        all_path,
        recurrence_path,
    ]:
        print(" ", path)

    print(
        "\nINTERPRETATION:\n"
        "  conjunction_fdr_bh < 0.05 is the primary cross-omic discovery rule.\n"
        "  conjunction_fdr_by < 0.05 is a stricter dependence-robust sensitivity.\n"
        "  both_individual_fdr05 is reported for comparison.\n"
        "  crossomic_rank_product is descriptive ordering only and has no\n"
        "  discovery threshold."
    )


if __name__ == "__main__":
    main()

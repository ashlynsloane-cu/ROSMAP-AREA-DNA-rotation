#!/usr/bin/env python3
"""
Stage 12B — adjusted DESeq2 RNA / adjusted limma protein conjunction analysis.

Purpose
-------
Repeat the formal Stage 12A cross-omic conjunction analysis using the
fully adjusted conventional methods:

    DLPFC RNA: DESeq2
    DLPFC protein: limma

Then compare formal BH-FDR cross-omic discoveries directly against the
fully adjusted AREA conjunction discoveries from Stage 12A.

Primary cross-omic test
-----------------------
For each one-to-one mapped gene and each contrast:

    if RNA and protein effect signs agree:
        p_conjunction = max(p_RNA, p_protein)
    else:
        p_conjunction = 1

Then adjust the 4,903 conjunction p-values per contrast using:
    - Benjamini-Hochberg FDR (primary)
    - Benjamini-Yekutieli FDR (sensitivity)

No arbitrary top-X-percent threshold is used.

Direction
---------
DESeq2:
    log2FoldChange > 0 => group B higher than group A

limma:
    logFC > 0 => group B higher than group A

The script validates that each adjusted result contains an appropriate signed
effect column before analysis.

Outputs
-------
results/stage12_brain_convergence/stage12B_deseq2_limma_conjunction/
    RNA_PROTEIN_MAPPING_AUDIT.tsv
    CONVENTIONAL_CONJUNCTION_SUMMARY.tsv
    ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv
    CROSS_CONTRAST_CONVENTIONAL_RECURRENCE.tsv
    AREA_VS_CONVENTIONAL_CONJUNCTION_COMPARISON.tsv
    AREA_VS_CONVENTIONAL_DISCOVERY_MEMBERSHIP.tsv
    <contrast>__DESEQ2_LIMMA_CONJUNCTION.tsv
"""

from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
OFFLINE = Path(os.environ["ROSMAP_OFFLINE_ROOT"]).expanduser().resolve()
REPO = Path(os.environ["ROSMAP_RESEARCH_ROOT"]).expanduser().resolve()
TASK_FILE = (
    OFFLINE
    / "results/stage11_covariate_stress/stage11_tasks.tsv"
)

GENE_UNIVERSE = (
    REPO
    / "results/preprocessing_protein_coding"
    / "ROSMAP_global_gene_universe.tsv"
)

AREA_CONJUNCTION = (
    OFFLINE
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
    / "ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv"
)

OUTDIR = (
    OFFLINE
    / "results/stage12_brain_convergence"
    / "stage12B_deseq2_limma_conjunction"
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

    return (
        REPO
        / "results/multiomic_models"
        / s.split(marker, 1)[1]
    )


def find_adjusted_result(task, method: str) -> Path:
    """
    Derive the frozen conventional mean_based directory from the frozen AREA
    endpoint for the same modality/cohort, then identify the single adjusted
    result file.
    """
    area_local = localize_area_result(task.A_result)
    mean_dir = area_local.parent.parent / "mean_based"

    pattern = (
        "*_DESeq2_all_results.csv"
        if method == "DESeq2"
        else "*_limma_all_results.csv"
    )

    hits = sorted(mean_dir.glob(pattern))

    if len(hits) != 1:
        raise RuntimeError(
            f"Task {task.task_id}: expected exactly one adjusted {method} "
            f"result in {mean_dir}; found {hits}"
        )

    return hits[0]


def first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
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
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    return np.clip(bh_adjust(p) * harmonic, 0.0, 1.0)


def load_gene_map() -> pd.DataFrame:
    if not GENE_UNIVERSE.exists():
        raise FileNotFoundError(GENE_UNIVERSE)

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
            f"Frozen gene universe missing columns: {sorted(missing)}"
        )

    if len(g) != 17646:
        raise RuntimeError(
            f"Expected 17,646 DLPFC genes; found {len(g)}."
        )

    if g["ensembl_id_version"].duplicated().any():
        raise RuntimeError(
            "Duplicate versioned Ensembl IDs in frozen gene universe."
        )

    g["gene_name"] = g["gene_name"].astype(str).str.strip()

    return g[
        ["ensembl_id_version", "ensembl_id", "gene_name"]
    ].copy()


def load_conventional(
    path: Path,
    method: str,
    expected_n: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    d = pd.read_csv(path, low_memory=False)

    id_col = first_present(
        d.columns,
        ["feature_id", "gene_id", "protein_id", d.columns[0]],
    )

    if method == "DESeq2":
        p_col = first_present(
            d.columns,
            ["pvalue", "p_value", "P.Value"],
        )
        q_col = first_present(
            d.columns,
            ["padj", "fdr", "adj.P.Val"],
        )
        effect_col = first_present(
            d.columns,
            ["log2FoldChange", "effect", "estimate"],
        )

    elif method == "limma":
        p_col = first_present(
            d.columns,
            ["P.Value", "pvalue", "p_value"],
        )
        q_col = first_present(
            d.columns,
            ["adj.P.Val", "padj", "fdr"],
        )
        effect_col = first_present(
            d.columns,
            ["logFC", "effect", "estimate"],
        )

    else:
        raise ValueError(method)

    missing = {
        "id_col": id_col,
        "p_col": p_col,
        "q_col": q_col,
        "effect_col": effect_col,
    }

    bad = [k for k, v in missing.items() if v is None]
    if bad:
        raise RuntimeError(
            f"{path}: cannot identify {bad}. Columns={list(d.columns)}"
        )

    if len(d) != expected_n:
        raise RuntimeError(
            f"{path}: expected {expected_n} rows; found {len(d)}."
        )

    out = pd.DataFrame(
        {
            "feature_id": d[id_col].astype(str).str.strip(),
            "pvalue": pd.to_numeric(d[p_col], errors="coerce"),
            "padj": pd.to_numeric(d[q_col], errors="coerce"),
            "effect": pd.to_numeric(d[effect_col], errors="coerce"),
        }
    )

    if out["feature_id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate feature IDs.")

    if out["effect"].isna().all():
        raise RuntimeError(f"{path}: effect column is entirely NA.")

    out["effect_sign"] = np.sign(out["effect"])

    return out


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
            "TMT feature IDs do not parse as GENE_SYMBOL|UniProt."
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
    rna: pd.DataFrame,
    prot: pd.DataFrame,
    gene_map: pd.DataFrame,
    one_to_one_symbols: set[str],
):
    # DLPFC Ensembl -> frozen symbol.
    r = rna.merge(
        gene_map,
        left_on="feature_id",
        right_on="ensembl_id_version",
        how="left",
        validate="one_to_one",
    )

    if r["gene_name"].isna().any():
        raise RuntimeError(
            f"{contrast_id}: incomplete DLPFC Ensembl mapping."
        )

    r = r.loc[
        r["gene_name"].isin(one_to_one_symbols)
    ].copy()

    r = r.rename(
        columns={
            "gene_name": "gene_symbol",
            "feature_id": "rna_feature_id",
            "pvalue": "rna_pvalue",
            "padj": "rna_padj",
            "effect": "rna_effect",
            "effect_sign": "rna_effect_sign",
        }
    )

    # TMT SYMBOL|UniProt -> symbol.
    p = prot.copy()

    parsed = p["feature_id"].str.split(
        "|",
        n=1,
        regex=False,
        expand=True,
    )

    if parsed.shape[1] != 2:
        raise RuntimeError(
            f"{contrast_id}: TMT feature IDs do not parse SYMBOL|accession."
        )

    p["gene_symbol"] = parsed[0].str.strip()
    p["uniprot_accession"] = parsed[1].str.strip()

    p = p.loc[
        p["gene_symbol"].isin(one_to_one_symbols)
    ].copy()

    p = p.rename(
        columns={
            "feature_id": "protein_feature_id",
            "pvalue": "protein_pvalue",
            "padj": "protein_padj",
            "effect": "protein_effect",
            "effect_sign": "protein_effect_sign",
        }
    )

    keep_r = [
        "gene_symbol",
        "ensembl_id_version",
        "ensembl_id",
        "rna_feature_id",
        "rna_pvalue",
        "rna_padj",
        "rna_effect",
        "rna_effect_sign",
    ]

    keep_p = [
        "gene_symbol",
        "uniprot_accession",
        "protein_feature_id",
        "protein_pvalue",
        "protein_padj",
        "protein_effect",
        "protein_effect_sign",
    ]

    d = r[keep_r].merge(
        p[keep_p],
        on="gene_symbol",
        how="inner",
        validate="one_to_one",
    )

    if len(d) != len(one_to_one_symbols):
        raise RuntimeError(
            f"{contrast_id}: expected {len(one_to_one_symbols)} mapped genes; "
            f"found {len(d)}."
        )

    d["contrast_id"] = contrast_id

    # Same non-zero sign in RNA and protein.
    d["direction_concordant"] = (
        d["rna_effect_sign"].ne(0)
        & d["protein_effect_sign"].ne(0)
        & d["rna_effect_sign"].eq(d["protein_effect_sign"])
    )

    rp = pd.to_numeric(
        d["rna_pvalue"],
        errors="coerce",
    ).fillna(1.0)

    pp = pd.to_numeric(
        d["protein_pvalue"],
        errors="coerce",
    ).fillna(1.0)

    d["conjunction_pvalue"] = np.maximum(rp, pp)

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

    d = d.sort_values(
        [
            "conjunction_fdr_bh",
            "conjunction_pvalue",
            "gene_symbol",
        ],
        ascending=[True, True, True],
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
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "n_contrasts_conjunction_fdr05_bh",
                "best_conjunction_fdr_bh",
                "best_conjunction_pvalue",
                "gene_symbol",
            ],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def compare_with_area(
    conventional: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not AREA_CONJUNCTION.exists():
        raise FileNotFoundError(
            f"AREA conjunction result missing: {AREA_CONJUNCTION}"
        )

    area = pd.read_csv(
        AREA_CONJUNCTION,
        sep="\t",
        low_memory=False,
    )

    needed = {
        "contrast_id",
        "gene_symbol",
        "crossomic_fdr05_bh",
    }

    missing = needed - set(area.columns)
    if missing:
        raise RuntimeError(
            f"AREA conjunction table missing columns: {sorted(missing)}"
        )

    comparison_rows = []
    membership_rows = []

    for contrast_id in CONTRAST_ORDER:
        a = area.loc[
            area["contrast_id"].astype(str).eq(contrast_id)
        ].copy()

        c = conventional.loc[
            conventional["contrast_id"].astype(str).eq(contrast_id)
        ].copy()

        a_sig = set(
            a.loc[
                a["crossomic_fdr05_bh"].astype(bool),
                "gene_symbol",
            ].astype(str)
        )

        c_sig = set(
            c.loc[
                c["crossomic_fdr05_bh"].astype(bool),
                "gene_symbol",
            ].astype(str)
        )

        union = a_sig | c_sig
        shared = a_sig & c_sig

        comparison_rows.append(
            {
                "contrast_id": contrast_id,
                "AREA_BH_n": len(a_sig),
                "DESeq2_limma_BH_n": len(c_sig),
                "shared_BH_n": len(shared),
                "AREA_only_BH_n": len(a_sig - c_sig),
                "conventional_only_BH_n": len(c_sig - a_sig),
                "BH_jaccard": (
                    len(shared) / len(union)
                    if union
                    else 1.0
                ),
                "shared_genes": ",".join(sorted(shared)),
                "AREA_only_genes": ",".join(sorted(a_sig - c_sig)),
                "conventional_only_genes": ",".join(
                    sorted(c_sig - a_sig)
                ),
            }
        )

        for gene in sorted(union):
            membership_rows.append(
                {
                    "contrast_id": contrast_id,
                    "gene_symbol": gene,
                    "AREA_BH": gene in a_sig,
                    "DESeq2_limma_BH": gene in c_sig,
                    "shared_BH": gene in shared,
                }
            )

    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(membership_rows),
    )


def main():
    gene_map = load_gene_map()

    tasks = pd.read_csv(
        TASK_FILE,
        sep="\t",
    )

    brain = tasks.loc[
        tasks["modality"].isin(["dlpfc", "tmt"])
    ].copy()

    # Representative adjusted TMT result determines the frozen TMT symbol
    # multiplicity for the same 5,211-feature universe.
    rep_task = next(
        r
        for r in brain.itertuples(index=False)
        if str(r.modality) == "tmt"
    )

    rep_limma_path = find_adjusted_result(
        rep_task,
        "limma",
    )

    representative_tmt = load_conventional(
        rep_limma_path,
        "limma",
        int(rep_task.A_n_features),
    )

    mapping_audit, one_to_one_symbols = (
        make_one_to_one_symbols(
            gene_map,
            representative_tmt,
        )
    )

    print("=" * 115)
    print("STAGE 12B — ADJUSTED DESEQ2 / LIMMA RNA-PROTEIN CONJUNCTION")
    print("=" * 115)
    print(f"One-to-one mapped genes: {len(one_to_one_symbols):,}")
    print("Primary threshold: conjunction BH FDR < 0.05")
    print("Sensitivity: conjunction BY FDR < 0.05")
    print("No top-X-percent threshold.")

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
                f"{contrast_id}: missing DLPFC/TMT task pair."
            )

        dlpfc_task = next(
            r
            for r in pair.itertuples(index=False)
            if str(r.modality) == "dlpfc"
        )

        tmt_task = next(
            r
            for r in pair.itertuples(index=False)
            if str(r.modality) == "tmt"
        )

        deseq_path = find_adjusted_result(
            dlpfc_task,
            "DESeq2",
        )

        limma_path = find_adjusted_result(
            tmt_task,
            "limma",
        )

        rna = load_conventional(
            deseq_path,
            "DESeq2",
            int(dlpfc_task.A_n_features),
        )

        prot = load_conventional(
            limma_path,
            "limma",
            int(tmt_task.A_n_features),
        )

        merged, summary = integrate_pair(
            contrast_id,
            rna,
            prot,
            gene_map,
            one_to_one_symbols,
        )

        summary["deseq2_result"] = str(deseq_path)
        summary["limma_result"] = str(limma_path)

        tables.append(merged)
        summaries.append(summary)

        merged.to_csv(
            OUTDIR
            / f"{contrast_id}__DESEQ2_LIMMA_CONJUNCTION.tsv",
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

    summary = (
        summary
        .sort_values("contrast_id")
        .reset_index(drop=True)
    )

    summary["contrast_id"] = summary[
        "contrast_id"
    ].astype(str)

    recurrence = build_recurrence(
        all_results
    )

    comparison, membership = compare_with_area(
        all_results
    )

    all_path = (
        OUTDIR
        / "ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv"
    )
    summary_path = (
        OUTDIR
        / "CONVENTIONAL_CONJUNCTION_SUMMARY.tsv"
    )
    recurrence_path = (
        OUTDIR
        / "CROSS_CONTRAST_CONVENTIONAL_RECURRENCE.tsv"
    )
    comparison_path = (
        OUTDIR
        / "AREA_VS_CONVENTIONAL_CONJUNCTION_COMPARISON.tsv"
    )
    membership_path = (
        OUTDIR
        / "AREA_VS_CONVENTIONAL_DISCOVERY_MEMBERSHIP.tsv"
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

    comparison.to_csv(
        comparison_path,
        sep="\t",
        index=False,
    )

    membership.to_csv(
        membership_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 115)
    print("CONVENTIONAL CONJUNCTION SUMMARY")
    print("=" * 115)

    show = [
        "contrast_id",
        "mapped_genes",
        "direction_concordant_n",
        "direction_concordant_fraction",
        "both_nominal_p05_n",
        "both_individual_fdr05_n",
        "conjunction_fdr05_bh_n",
        "conjunction_fdr05_by_n",
        "rna_fdr05_among_mapped",
        "protein_fdr05_among_mapped",
        "min_conjunction_fdr_bh",
        "min_conjunction_fdr_by",
    ]

    print(
        summary[show]
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 115)
    print("AREA VS DESEQ2/LIMMA FORMAL CROSS-OMIC DISCOVERIES")
    print("=" * 115)

    print(
        comparison
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 115)
    print("BH-FDR DESEQ2/LIMMA DISCOVERIES")
    print("=" * 115)

    for contrast_id in CONTRAST_ORDER:
        d = all_results.loc[
            all_results["contrast_id"].eq(contrast_id)
            & all_results["crossomic_fdr05_bh"]
        ].copy()

        print(f"\n--- {contrast_id}: {len(d)} genes ---")

        if d.empty:
            print("No same-direction DESeq2/limma conjunction discoveries.")
            continue

        print(
            d[
                [
                    "gene_symbol",
                    "rna_effect",
                    "protein_effect",
                    "rna_pvalue",
                    "protein_pvalue",
                    "conjunction_pvalue",
                    "conjunction_fdr_bh",
                    "conjunction_fdr_by",
                    "rna_padj",
                    "protein_padj",
                ]
            ]
            .head(60)
            .round(7)
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in [
        mapping_path,
        summary_path,
        all_path,
        recurrence_path,
        comparison_path,
        membership_path,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()

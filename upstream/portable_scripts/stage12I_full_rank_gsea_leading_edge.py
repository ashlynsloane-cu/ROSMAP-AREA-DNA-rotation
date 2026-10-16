#!/usr/bin/env python3
"""
Stage 12I — full-rank cross-omic GSEA + leading-edge analysis.

Primary question
----------------
Are the formal / near-convergent genes isolated discoveries, or are they
extreme members of broader biological programs that shift across the full
RNA and protein ranked distributions?

Design
------
Universe:
    exact 4,903 one-to-one DLPFC RNA / TMT protein genes from Stage 12A/12B.

Contrasts:
    cogdx_NCI_vs_AD
    CERAD_3-4_vs_1-2
    cogdx_MCI_vs_AD
    Braak_3-4_vs_5-6

Profiles:
    AREA RNA
    AREA protein
    DESeq2 RNA
    limma protein

Ranking statistic:
    signed two-sided normal score from the raw modality-specific p-value:

        sign * Phi^{-1}(1 - p/2)

    AREA sign:
        *_associated_higher_rank -> +1
        *_associated_lower_rank  -> -1

    Conventional sign:
        sign(log2FoldChange/logFC), using the frozen Stage 12B orientation.

Why this statistic?
-------------------
AREA does not provide a conventional effect-size/test-statistic directly.
The signed normal score therefore provides a monotonic, full-rank statistic
that uses association strength plus direction and can be defined identically
for all four profiles. GSEA is performed separately within each profile.

Zero p-values are replaced deterministically by one tenth of the smallest
positive p-value in that profile. Exact score ties are given a negligible,
deterministic gene-symbol tie-break for GSEA ordering only. The untied signed
normal score is retained in the output.

Gene sets
---------
Downloaded through GSEApy / Enrichr and then frozen locally for this run:
    Reactome
    GO Biological Process
    MSigDB Hallmark

The script automatically chooses the latest available year-stamped library
for each requested family and records the exact library names. You can pin
library names with CLI arguments on future reruns.

Gene-set membership is explicitly restricted to the 4,903-gene universe.
Default post-restriction gene-set sizes are 10-500 genes.

Inference
---------
Preranked GSEA uses gene-set permutations. Primary pathway criterion:
    GSEA FDR q-value < 0.05

No DEG threshold, top-X-percent gene cutoff, or ORA selection is used.

Leading-edge analysis
---------------------
For every FDR-significant pathway, the GSEA leading-edge genes are retained.
RNA/protein leading-edge intersections are computed for matched
contrast/method/pathway combinations.

Outputs
-------
results/stage12_brain_convergence/stage12I_full_rank_gsea/
    ALL_RANKED_PROFILES.tsv
    RANKING_AUDIT.tsv
    METHOD_DIRECTION_AUDIT.tsv
    GENE_SET_LIBRARY_AUDIT.tsv
    ALL_GSEA_RESULTS.tsv
    CROSS_MODAL_GSEA_COMPARISON.tsv
    CONCORDANT_FDR05_PATHWAYS.tsv
    LEADING_EDGE_LONG_FDR05.tsv
    PRIORITY_GENE_LEADING_EDGE_HITS.tsv
    gene_sets/*.gmt
    README_STAGE12I.txt
"""

from __future__ import annotations
import os

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import gseapy as gp
except ImportError as exc:
    raise RuntimeError(
        "Stage 12I requires GSEApy. Install it in the active environment:\n"
        "  python3 -m pip install 'gseapy==1.3.0'\n"
    ) from exc
ROOT = Path(os.environ["ROSMAP_OFFLINE_ROOT"]).expanduser().resolve()
AREA_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
    / "ALL_CONTRASTS_RNA_PROTEIN_CONJUNCTION.tsv"
)

CONVENTIONAL_PATH = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12B_deseq2_limma_conjunction"
    / "ALL_CONTRASTS_CONVENTIONAL_CONJUNCTION.tsv"
)

OUTDIR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12I_full_rank_gsea"
)

GENESET_DIR = OUTDIR / "gene_sets"

CONTRASTS = [
    "cogdx_NCI_vs_AD",
    "CERAD_3-4_vs_1-2",
    "cogdx_MCI_vs_AD",
    "Braak_3-4_vs_5-6",
]

PRIORITY_GENES = {
    # Deep novelty audit.
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
    # AREA-only neuronal signaling module.
    "CAMKK2",
    "CAMK4",
    "CACNG3",
    "ADAM11",
    "SST",
    "CLPTM1",
    # Strong recurrent/shared contextual genes.
    "VGF",
    "NRIP2",
    "SLC38A2",
    "NDUFB5",
    "NRN1",
    "PLXNB1",
}

LIBRARY_FAMILIES = {
    "REAC": "Reactome_",
    "GO_BP": "GO_Biological_Process_",
    "HALLMARK": "MSigDB_Hallmark_",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="Gene-set permutations per preranked GSEA profile.",
    )

    p.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Minimum gene-set size after restriction to the 4,903-gene universe.",
    )

    p.add_argument(
        "--max-size",
        type=int,
        default=500,
        help="Maximum gene-set size after restriction to the 4,903-gene universe.",
    )

    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--reactome-library",
        default="auto",
    )

    p.add_argument(
        "--go-library",
        default="auto",
    )

    p.add_argument(
        "--hallmark-library",
        default="auto",
    )

    return p.parse_args()


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def parse_area_sign(direction: object) -> float:
    s = str(direction)

    if "_associated_higher_rank" in s:
        return 1.0

    if "_associated_lower_rank" in s:
        return -1.0

    return np.nan


def choose_latest_library(
    available: list[str],
    prefix: str,
    explicit: str,
) -> str:
    if explicit != "auto":
        if explicit not in available:
            raise RuntimeError(
                f"Requested library {explicit!r} is not available.\n"
                f"Matching available libraries:\n"
                + "\n".join(
                    x for x in available if x.startswith(prefix)
                )
            )
        return explicit

    candidates = [
        x
        for x in available
        if x.startswith(prefix)
    ]

    if not candidates:
        raise RuntimeError(
            f"No Enrichr library found with prefix {prefix!r}."
        )

    def year_key(name: str):
        hits = re.findall(r"(?:19|20)\d{2}", name)
        year = int(hits[-1]) if hits else -1
        return year, name

    return max(candidates, key=year_key)


def write_gmt(
    gene_sets: dict[str, list[str]],
    path: Path,
):
    with path.open("w") as handle:
        for term in sorted(gene_sets):
            genes = gene_sets[term]
            handle.write(
                "\t".join(
                    [term, "NA", *genes]
                )
                + "\n"
            )


def pvalue_floor(p: pd.Series) -> tuple[pd.Series, float]:
    x = pd.to_numeric(
        p,
        errors="coerce",
    ).copy()

    x = x.where(np.isfinite(x), 1.0)
    x = x.clip(lower=0.0, upper=1.0)

    positive = x.loc[x > 0]

    if positive.empty:
        floor = 1e-300
    else:
        floor = max(
            float(positive.min()) / 10.0,
            np.finfo(float).tiny,
        )

    x = x.mask(x <= 0, floor)

    return x, floor


def deterministic_tie_break(
    genes: pd.Series,
    scores: pd.Series,
) -> pd.Series:
    """
    Break exact GSEA score ties by an extremely small deterministic offset.

    The offset is many orders of magnitude smaller than the observed score
    scale and exists only to make rank ordering reproducible.
    """
    out = pd.to_numeric(
        scores,
        errors="coerce",
    ).astype(float).copy()

    tmp = pd.DataFrame(
        {
            "gene": genes.astype(str),
            "score": out,
        },
        index=out.index,
    )

    scale = max(
        float(np.nanmax(np.abs(out.to_numpy()))),
        1.0,
    )

    eps = scale * 1e-12

    for _, idx in tmp.groupby(
        "score",
        dropna=False,
        sort=False,
    ).groups.items():
        idx = list(idx)

        if len(idx) <= 1:
            continue

        ordered = sorted(
            idx,
            key=lambda i: str(tmp.loc[i, "gene"]),
        )

        offsets = np.linspace(
            eps,
            -eps,
            len(ordered),
        )

        for i, off in zip(ordered, offsets):
            out.loc[i] = out.loc[i] + off

    return out


def build_profile(
    df: pd.DataFrame,
    contrast: str,
    method: str,
    modality: str,
) -> tuple[pd.DataFrame, dict]:
    d = df.loc[
        df["contrast_id"].astype(str).eq(contrast)
    ].copy()

    if len(d) != 4903:
        raise RuntimeError(
            f"{contrast} {method} {modality}: expected 4,903 genes; "
            f"found {len(d)}."
        )

    if d["gene_symbol"].duplicated().any():
        raise RuntimeError(
            f"{contrast} {method} {modality}: duplicate gene symbols."
        )

    if method == "AREA":
        p_col = (
            "rna_pvalue"
            if modality == "RNA"
            else "protein_pvalue"
        )
        direction_col = (
            "rna_direction"
            if modality == "RNA"
            else "protein_direction"
        )

        sign = d[direction_col].map(
            parse_area_sign
        )

        if sign.isna().any():
            bad = (
                d.loc[
                    sign.isna(),
                    direction_col,
                ]
                .astype(str)
                .value_counts()
                .head(20)
                .to_dict()
            )
            raise RuntimeError(
                f"{contrast} AREA {modality}: cannot parse "
                f"direction labels: {bad}"
            )

        source_direction = d[
            direction_col
        ].astype(str)

    elif method == "CONVENTIONAL":
        p_col = (
            "rna_pvalue"
            if modality == "RNA"
            else "protein_pvalue"
        )
        effect_col = (
            "rna_effect"
            if modality == "RNA"
            else "protein_effect"
        )

        effect = pd.to_numeric(
            d[effect_col],
            errors="coerce",
        )

        if effect.isna().any():
            raise RuntimeError(
                f"{contrast} conventional {modality}: "
                f"NA effect values found."
            )

        sign = np.sign(effect)

        if (sign == 0).any():
            # Exact zero effects receive zero signed statistic.
            pass

        source_direction = effect.astype(str)

    else:
        raise ValueError(method)

    p_clean, floor = pvalue_floor(
        d[p_col]
    )

    z_abs = pd.Series(
        norm.isf(
            p_clean.to_numpy() / 2.0
        ),
        index=d.index,
        dtype=float,
    )

    signed_z = (
        pd.Series(
            np.asarray(sign, dtype=float),
            index=d.index,
        )
        * z_abs
    )

    gsea_score = deterministic_tie_break(
        d["gene_symbol"],
        signed_z,
    )

    profile_id = (
        f"{contrast}__{method}__{modality}"
    )

    out = pd.DataFrame(
        {
            "profile_id": profile_id,
            "contrast_id": contrast,
            "method": method,
            "modality": modality,
            "gene_symbol":
                d["gene_symbol"].astype(str).str.strip(),
            "raw_pvalue":
                pd.to_numeric(d[p_col], errors="coerce"),
            "pvalue_used": p_clean,
            "direction_sign":
                np.asarray(sign, dtype=float),
            "source_direction":
                source_direction.to_numpy(),
            "signed_normal_score": signed_z,
            "gsea_rank_score": gsea_score,
        }
    )

    out = (
        out.sort_values(
            [
                "gsea_rank_score",
                "gene_symbol",
            ],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    out["rank_descending"] = np.arange(
        1,
        len(out) + 1,
    )

    audit = {
        "profile_id": profile_id,
        "contrast_id": contrast,
        "method": method,
        "modality": modality,
        "n_genes": len(out),
        "positive_n":
            int((out["direction_sign"] > 0).sum()),
        "negative_n":
            int((out["direction_sign"] < 0).sum()),
        "zero_n":
            int((out["direction_sign"] == 0).sum()),
        "raw_zero_p_n":
            int(
                pd.to_numeric(
                    d[p_col],
                    errors="coerce",
                ).eq(0).sum()
            ),
        "pvalue_floor": floor,
        "max_signed_score":
            float(out["signed_normal_score"].max()),
        "min_signed_score":
            float(out["signed_normal_score"].min()),
    }

    return out, audit


def build_all_profiles(
    area: pd.DataFrame,
    conventional: pd.DataFrame,
):
    profiles = []
    audits = []

    for contrast in CONTRASTS:
        for method, table in [
            ("AREA", area),
            ("CONVENTIONAL", conventional),
        ]:
            for modality in ["RNA", "PROTEIN"]:
                profile, audit = build_profile(
                    table,
                    contrast,
                    method,
                    modality,
                )
                profiles.append(profile)
                audits.append(audit)

    all_profiles = pd.concat(
        profiles,
        ignore_index=True,
    )

    audit = pd.DataFrame(audits)

    return all_profiles, audit


def direction_audit(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for contrast in CONTRASTS:
        for modality in ["RNA", "PROTEIN"]:
            a = profiles.loc[
                profiles["contrast_id"].eq(contrast)
                & profiles["method"].eq("AREA")
                & profiles["modality"].eq(modality),
                [
                    "gene_symbol",
                    "direction_sign",
                ],
            ].rename(
                columns={
                    "direction_sign": "area_sign"
                }
            )

            c = profiles.loc[
                profiles["contrast_id"].eq(contrast)
                & profiles["method"].eq("CONVENTIONAL")
                & profiles["modality"].eq(modality),
                [
                    "gene_symbol",
                    "direction_sign",
                ],
            ].rename(
                columns={
                    "direction_sign": "conventional_sign"
                }
            )

            d = a.merge(
                c,
                on="gene_symbol",
                how="inner",
                validate="one_to_one",
            )

            nonzero = (
                d["area_sign"].ne(0)
                & d["conventional_sign"].ne(0)
            )

            tested = d.loc[nonzero]

            agree = (
                tested["area_sign"]
                == tested["conventional_sign"]
            )

            rows.append(
                {
                    "contrast_id": contrast,
                    "modality": modality,
                    "n_compared": len(tested),
                    "n_sign_agree": int(agree.sum()),
                    "sign_agreement_fraction":
                        float(agree.mean())
                        if len(tested)
                        else np.nan,
                }
            )

    return pd.DataFrame(rows)


def load_gene_sets(
    universe: set[str],
    args,
):
    available = gp.get_library_name(
        organism="Human"
    )

    chosen = {
        "REAC": choose_latest_library(
            available,
            LIBRARY_FAMILIES["REAC"],
            args.reactome_library,
        ),
        "GO_BP": choose_latest_library(
            available,
            LIBRARY_FAMILIES["GO_BP"],
            args.go_library,
        ),
        "HALLMARK": choose_latest_library(
            available,
            LIBRARY_FAMILIES["HALLMARK"],
            args.hallmark_library,
        ),
    }

    combined = {}
    audit_rows = []

    for short, library_name in chosen.items():
        raw = gp.get_library(
            name=library_name,
            organism="Human",
        )

        restricted = {}

        raw_sizes = []
        restricted_sizes = []

        for term, genes in raw.items():
            raw_clean = {
                str(g).strip()
                for g in genes
                if str(g).strip()
            }

            overlap = sorted(
                raw_clean & universe
            )

            raw_sizes.append(
                len(raw_clean)
            )
            restricted_sizes.append(
                len(overlap)
            )

            if not (
                args.min_size
                <= len(overlap)
                <= args.max_size
            ):
                continue

            key = f"{short}||{term}"
            restricted[key] = overlap
            combined[key] = overlap

        gmt_path = (
            GENESET_DIR
            / f"{short}__{library_name}__restricted_4903.gmt"
        )

        write_gmt(
            restricted,
            gmt_path,
        )

        audit_rows.append(
            {
                "source": short,
                "library_name": library_name,
                "n_raw_terms": len(raw),
                "n_terms_after_4903_and_size_filter":
                    len(restricted),
                "min_size": args.min_size,
                "max_size": args.max_size,
                "median_raw_size":
                    float(np.median(raw_sizes))
                    if raw_sizes else np.nan,
                "median_restricted_size":
                    float(np.median(restricted_sizes))
                    if restricted_sizes else np.nan,
                "gmt_path": str(gmt_path),
            }
        )

    return (
        combined,
        pd.DataFrame(audit_rows),
    )


def normalize_gsea_columns(
    res: pd.DataFrame,
) -> pd.DataFrame:
    out = res.copy()

    rename = {
        "ES": "es",
        "NES": "nes",
        "NOM p-val": "nominal_pvalue",
        "FDR q-val": "fdr_qvalue",
        "FWER p-val": "fwer_pvalue",
        "Tag %": "tag_fraction",
        "Gene %": "gene_fraction",
        "Lead_genes": "leading_edge_genes",
        "Matched_genes": "matched_genes",
    }

    out = out.rename(
        columns={
            k: v
            for k, v in rename.items()
            if k in out.columns
        }
    )

    if "Term" not in out.columns:
        if out.index.name == "Term":
            out = out.reset_index()
        else:
            out = out.reset_index().rename(
                columns={"index": "Term"}
            )

    required = {
        "Term",
        "nes",
        "nominal_pvalue",
        "fdr_qvalue",
        "leading_edge_genes",
    }

    missing = required - set(out.columns)

    if missing:
        raise RuntimeError(
            "Unexpected GSEApy result columns. "
            f"Missing={sorted(missing)}; "
            f"columns={list(out.columns)}"
        )

    parts = out["Term"].astype(str).str.split(
        "||",
        n=1,
        regex=False,
        expand=True,
    )

    if parts.shape[1] != 2:
        raise RuntimeError(
            "Cannot split prefixed GSEA term names."
        )

    out["gene_set_source"] = parts[0]
    out["term_name"] = parts[1]

    return out


def parse_gene_string(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        x.strip()
        for x in re.split(r"[;,]", text)
        if x.strip()
    ]


def run_gsea(
    profiles: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    args,
):
    all_res = []

    profile_ids = (
        profiles[
            [
                "profile_id",
                "contrast_id",
                "method",
                "modality",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "contrast_id",
                "method",
                "modality",
            ]
        )
    )

    n_profiles = len(profile_ids)

    for counter, row in enumerate(
        profile_ids.itertuples(index=False),
        start=1,
    ):
        p = profiles.loc[
            profiles["profile_id"].eq(
                row.profile_id
            ),
            [
                "gene_symbol",
                "gsea_rank_score",
            ],
        ].copy()

        print(
            f"[{counter}/{n_profiles}] "
            f"{row.profile_id}"
        )

        pre = gp.prerank(
            rnk=p,
            gene_sets=gene_sets,
            min_size=args.min_size,
            max_size=args.max_size,
            permutation_num=args.permutations,
            weight=1.0,
            ascending=False,
            threads=args.threads,
            seed=args.seed,
            outdir=None,
            no_plot=True,
            verbose=False,
        )

        res = normalize_gsea_columns(
            pre.res2d
        )

        res["profile_id"] = row.profile_id
        res["contrast_id"] = row.contrast_id
        res["method"] = row.method
        res["modality"] = row.modality

        numeric_cols = [
            "es",
            "nes",
            "nominal_pvalue",
            "fdr_qvalue",
            "fwer_pvalue",
        ]

        for c in numeric_cols:
            if c in res.columns:
                res[c] = pd.to_numeric(
                    res[c],
                    errors="coerce",
                )

        res["fdr05"] = (
            res["fdr_qvalue"] < 0.05
        )

        res["direction"] = np.where(
            res["nes"] > 0,
            "positive",
            np.where(
                res["nes"] < 0,
                "negative",
                "zero",
            ),
        )

        all_res.append(res)

    return pd.concat(
        all_res,
        ignore_index=True,
    )


def cross_modal_comparison(
    gsea: pd.DataFrame,
):
    key = [
        "contrast_id",
        "method",
        "gene_set_source",
        "term_name",
    ]

    rna = gsea.loc[
        gsea["modality"].eq("RNA")
    ].copy()

    prot = gsea.loc[
        gsea["modality"].eq("PROTEIN")
    ].copy()

    keep = [
        *key,
        "nes",
        "nominal_pvalue",
        "fdr_qvalue",
        "leading_edge_genes",
    ]

    rna = rna[keep].rename(
        columns={
            "nes": "rna_nes",
            "nominal_pvalue": "rna_nominal_pvalue",
            "fdr_qvalue": "rna_fdr_qvalue",
            "leading_edge_genes":
                "rna_leading_edge_genes",
        }
    )

    prot = prot[keep].rename(
        columns={
            "nes": "protein_nes",
            "nominal_pvalue":
                "protein_nominal_pvalue",
            "fdr_qvalue":
                "protein_fdr_qvalue",
            "leading_edge_genes":
                "protein_leading_edge_genes",
        }
    )

    d = rna.merge(
        prot,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    d["same_nes_direction"] = (
        np.sign(d["rna_nes"])
        == np.sign(d["protein_nes"])
    )

    d["both_fdr05"] = (
        d["rna_fdr_qvalue"].lt(0.05)
        & d["protein_fdr_qvalue"].lt(0.05)
    )

    d["concordant_both_fdr05"] = (
        d["same_nes_direction"]
        & d["both_fdr05"]
    )

    d["max_crossmodal_fdr"] = np.maximum(
        d["rna_fdr_qvalue"],
        d["protein_fdr_qvalue"],
    )

    d["min_abs_nes"] = np.minimum(
        d["rna_nes"].abs(),
        d["protein_nes"].abs(),
    )

    rna_edges = d[
        "rna_leading_edge_genes"
    ].map(
        lambda x: set(parse_gene_string(x))
    )

    prot_edges = d[
        "protein_leading_edge_genes"
    ].map(
        lambda x: set(parse_gene_string(x))
    )

    intersections = [
        sorted(a & b)
        for a, b in zip(
            rna_edges,
            prot_edges,
        )
    ]

    d["rna_leading_edge_n"] = [
        len(x)
        for x in rna_edges
    ]
    d["protein_leading_edge_n"] = [
        len(x)
        for x in prot_edges
    ]
    d["shared_leading_edge_n"] = [
        len(x)
        for x in intersections
    ]
    d["shared_leading_edge_genes"] = [
        ",".join(x)
        for x in intersections
    ]

    d["priority_genes_in_shared_leading_edge"] = [
        ",".join(
            sorted(
                set(x) & PRIORITY_GENES
            )
        )
        for x in intersections
    ]

    return (
        d.sort_values(
            [
                "concordant_both_fdr05",
                "max_crossmodal_fdr",
                "min_abs_nes",
            ],
            ascending=[
                False,
                True,
                False,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def leading_edge_long(
    gsea: pd.DataFrame,
):
    rows = []

    sig = gsea.loc[
        gsea["fdr_qvalue"].lt(0.05)
    ].copy()

    for r in sig.itertuples(index=False):
        genes = parse_gene_string(
            r.leading_edge_genes
        )

        for gene in genes:
            rows.append(
                {
                    "profile_id": r.profile_id,
                    "contrast_id": r.contrast_id,
                    "method": r.method,
                    "modality": r.modality,
                    "gene_set_source":
                        r.gene_set_source,
                    "term_name": r.term_name,
                    "nes": r.nes,
                    "fdr_qvalue": r.fdr_qvalue,
                    "gene_symbol": gene,
                    "priority_gene":
                        gene in PRIORITY_GENES,
                }
            )

    return pd.DataFrame(rows)


def priority_hits(
    cross_modal: pd.DataFrame,
):
    rows = []

    d = cross_modal.loc[
        cross_modal[
            "concordant_both_fdr05"
        ]
    ].copy()

    for r in d.itertuples(index=False):
        genes = parse_gene_string(
            r.shared_leading_edge_genes
        )

        for gene in sorted(
            set(genes) & PRIORITY_GENES
        ):
            rows.append(
                {
                    "contrast_id":
                        r.contrast_id,
                    "method": r.method,
                    "gene_set_source":
                        r.gene_set_source,
                    "term_name": r.term_name,
                    "rna_nes": r.rna_nes,
                    "protein_nes":
                        r.protein_nes,
                    "rna_fdr_qvalue":
                        r.rna_fdr_qvalue,
                    "protein_fdr_qvalue":
                        r.protein_fdr_qvalue,
                    "gene_symbol": gene,
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    require(AREA_PATH)
    require(CONVENTIONAL_PATH)

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    GENESET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    area = pd.read_csv(
        AREA_PATH,
        sep="\t",
        low_memory=False,
    )
    conventional = pd.read_csv(
        CONVENTIONAL_PATH,
        sep="\t",
        low_memory=False,
    )

    print("=" * 120)
    print(
        "STAGE 12I — FULL-RANK CROSS-OMIC GSEA "
        "+ LEADING EDGE"
    )
    print("=" * 120)
    print(
        "Universe: exact Stage 12 one-to-one "
        "RNA/protein genes (expected 4,903)"
    )
    print(
        f"Permutations: {args.permutations:,}; "
        f"seed={args.seed}"
    )
    print(
        f"Gene-set size after 4,903-gene restriction: "
        f"{args.min_size}-{args.max_size}"
    )

    profiles, ranking_audit = (
        build_all_profiles(
            area,
            conventional,
        )
    )

    universe = set(
        profiles["gene_symbol"].astype(str)
    )

    if len(universe) != 4903:
        raise RuntimeError(
            f"Expected 4,903 unique genes across profiles; "
            f"found {len(universe)}."
        )

    # Every profile must have exactly the same gene set.
    for profile_id, g in profiles.groupby(
        "profile_id"
    ):
        genes = set(
            g["gene_symbol"].astype(str)
        )
        if genes != universe:
            raise RuntimeError(
                f"{profile_id}: gene universe differs "
                "from the common 4,903-gene set."
            )

    direction_check = direction_audit(
        profiles
    )

    print("\nMETHOD-DIRECTION AUDIT")
    print("-" * 120)
    print(
        direction_check
        .round(4)
        .to_string(index=False)
    )

    gene_sets, library_audit = (
        load_gene_sets(
            universe,
            args,
        )
    )

    print("\nGENE-SET LIBRARIES")
    print("-" * 120)
    print(
        library_audit.to_string(
            index=False
        )
    )
    print(
        f"\nCombined restricted gene sets: "
        f"{len(gene_sets):,}"
    )

    gsea = run_gsea(
        profiles,
        gene_sets,
        args,
    )

    cross_modal = cross_modal_comparison(
        gsea
    )

    concordant = cross_modal.loc[
        cross_modal[
            "concordant_both_fdr05"
        ]
    ].copy()

    edge_long = leading_edge_long(
        gsea
    )

    priority = priority_hits(
        cross_modal
    )

    paths = {
        "profiles":
            OUTDIR
            / "ALL_RANKED_PROFILES.tsv",
        "ranking_audit":
            OUTDIR
            / "RANKING_AUDIT.tsv",
        "direction_audit":
            OUTDIR
            / "METHOD_DIRECTION_AUDIT.tsv",
        "library_audit":
            OUTDIR
            / "GENE_SET_LIBRARY_AUDIT.tsv",
        "gsea":
            OUTDIR
            / "ALL_GSEA_RESULTS.tsv",
        "cross_modal":
            OUTDIR
            / "CROSS_MODAL_GSEA_COMPARISON.tsv",
        "concordant":
            OUTDIR
            / "CONCORDANT_FDR05_PATHWAYS.tsv",
        "edge_long":
            OUTDIR
            / "LEADING_EDGE_LONG_FDR05.tsv",
        "priority":
            OUTDIR
            / "PRIORITY_GENE_LEADING_EDGE_HITS.tsv",
        "readme":
            OUTDIR
            / "README_STAGE12I.txt",
    }

    profiles.to_csv(
        paths["profiles"],
        sep="\t",
        index=False,
    )
    ranking_audit.to_csv(
        paths["ranking_audit"],
        sep="\t",
        index=False,
    )
    direction_check.to_csv(
        paths["direction_audit"],
        sep="\t",
        index=False,
    )
    library_audit.to_csv(
        paths["library_audit"],
        sep="\t",
        index=False,
    )
    gsea.to_csv(
        paths["gsea"],
        sep="\t",
        index=False,
    )
    cross_modal.to_csv(
        paths["cross_modal"],
        sep="\t",
        index=False,
    )
    concordant.to_csv(
        paths["concordant"],
        sep="\t",
        index=False,
    )
    edge_long.to_csv(
        paths["edge_long"],
        sep="\t",
        index=False,
    )
    priority.to_csv(
        paths["priority"],
        sep="\t",
        index=False,
    )

    readme = f"""STAGE 12I — FULL-RANK CROSS-OMIC GSEA + LEADING EDGE
=======================================================

Universe:
    4,903 exact one-to-one RNA/protein genes.

Contrasts:
    {", ".join(CONTRASTS)}

Profiles:
    AREA RNA
    AREA protein
    conventional DESeq2 RNA
    conventional limma protein

Ranking:
    sign * two-sided normal score from raw p-value.
    AREA sign follows the Stage 12C convention:
        higher_rank = +1
        lower_rank  = -1
    Conventional sign is sign(effect).

GSEA:
    engine: GSEApy {getattr(gp, "__version__", "unknown")}
    permutations: {args.permutations}
    seed: {args.seed}
    min size after universe restriction: {args.min_size}
    max size after universe restriction: {args.max_size}
    primary pathway criterion: FDR q < 0.05

Exact downloaded library names are recorded in:
    GENE_SET_LIBRARY_AUDIT.tsv

Frozen, universe-restricted GMTs are stored in:
    gene_sets/

Key outputs:
    ALL_RANKED_PROFILES.tsv
        Exact ranks used for GSEA. Stage 12J reads this file.

    ALL_GSEA_RESULTS.tsv
        Every tested GSEA result.

    CROSS_MODAL_GSEA_COMPARISON.tsv
        RNA vs protein result for every matched pathway.

    CONCORDANT_FDR05_PATHWAYS.tsv
        Pathways with FDR < .05 in BOTH RNA and protein and the same NES sign.

    LEADING_EDGE_LONG_FDR05.tsv
        Long-form leading-edge membership for every FDR-significant profile/pathway.

    PRIORITY_GENE_LEADING_EDGE_HITS.tsv
        Priority genes that are in the RNA AND protein leading edge of a
        concordant pathway significant in both modalities.

Interpretation:
    GSEA leading edge is distinct from AREA's participant-level RAE /
    AREA leading-edge threshold. Stage 12I asks which genes drive
    pathway-level full-rank enrichment.

No DEG threshold or top-X-percent gene selection is used.
"""

    paths["readme"].write_text(
        readme
    )

    print("\n" + "=" * 120)
    print(
        "CONCORDANT RNA/PROTEIN GSEA "
        "(FDR < .05 in both, same NES direction)"
    )
    print("=" * 120)

    if concordant.empty:
        print(
            "No pathways met the concordant "
            "dual-FDR criterion."
        )
    else:
        show = [
            "contrast_id",
            "method",
            "gene_set_source",
            "term_name",
            "rna_nes",
            "protein_nes",
            "rna_fdr_qvalue",
            "protein_fdr_qvalue",
            "shared_leading_edge_n",
            "priority_genes_in_shared_leading_edge",
        ]

        print(
            concordant[
                show
            ]
            .head(100)
            .round(5)
            .to_string(index=False)
        )

    print("\nPRIORITY-GENE SHARED LEADING-EDGE HITS")
    print("-" * 120)

    if priority.empty:
        print("None.")
    else:
        print(
            priority
            .sort_values(
                [
                    "contrast_id",
                    "method",
                    "gene_symbol",
                ]
            )
            .round(5)
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in paths.values():
        print(" ", path)


if __name__ == "__main__":
    main()

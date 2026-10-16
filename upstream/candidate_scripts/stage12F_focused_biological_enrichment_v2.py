#!/usr/bin/env python3
"""
Stage 12F — focused biological enrichment of frozen Stage 12E gene sets.

This is the first pathway-level biological analysis.

No new gene-discovery threshold is introduced.

Statistical universe
--------------------
The enrichment background is the exact one-to-one DLPFC RNA / TMT protein
mapping universe used for Stage 12 cross-omic analyses (~4,903 genes), NOT the
whole genome.

This is required because only genes represented in both omic layers could have
entered the Stage 12 candidate sets.

Enrichment service
------------------
g:Profiler g:GOSt API
Species: human
Sources:
    REAC   Reactome
    GO:BP  Gene Ontology Biological Process

Multiple testing:
    Benjamini-Hochberg FDR

Domain:
    custom background = Stage 12 cross-omic mapping universe

Primary query sets
------------------
1. NCI-vs-AD formal shared
2. NCI-vs-AD formal AREA-only
3. CERAD formal shared
4. CERAD formal AREA-only
5. MCI-vs-AD shared-near
6. late-Braak shared-near

Conventional-only sets are retained as secondary comparators.

Very small queries are still written and may be submitted, but absence of
significant enrichment should not be interpreted as absence of biology.

Reproducibility
---------------
The exact API payload and raw JSON response are saved for every query.
The script also attempts to save g:Profiler datasource-version metadata.

Requires an internet connection.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path.home() / "Developer/stage11_offline_2026-09-28"

MASTER = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12E_biological_evidence_matrix"
    / "MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv"
)

MAPPING_AUDIT = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12A_rna_protein_conjunction"
    / "RNA_PROTEIN_MAPPING_AUDIT.tsv"
)

OUTDIR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12F_focused_biological_enrichment"
)

RAW_DIR = OUTDIR / "raw_gprofiler"
OUTDIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


GPROFILER_PROFILE_URL = (
    "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
)

GPROFILER_VERSION_URL = (
    "https://biit.cs.ut.ee/gprofiler/api/util/data_versions/"
    "?organism=hsapiens"
)

SOURCES = ["REAC", "GO:BP"]


QUERY_SPECS = [
    {
        "query_id": "NCI_AD__formal_shared",
        "contrast_id": "cogdx_NCI_vs_AD",
        "evidence_class": "formal_shared",
        "priority": "primary",
    },
    {
        "query_id": "NCI_AD__formal_AREA_only",
        "contrast_id": "cogdx_NCI_vs_AD",
        "evidence_class": "formal_AREA_only",
        "priority": "primary",
    },
    {
        "query_id": "NCI_AD__formal_conventional_only",
        "contrast_id": "cogdx_NCI_vs_AD",
        "evidence_class": "formal_conventional_only",
        "priority": "secondary_comparator",
    },
    {
        "query_id": "CERAD__formal_shared",
        "contrast_id": "CERAD_3-4_vs_1-2",
        "evidence_class": "formal_shared",
        "priority": "primary",
    },
    {
        "query_id": "CERAD__formal_AREA_only",
        "contrast_id": "CERAD_3-4_vs_1-2",
        "evidence_class": "formal_AREA_only",
        "priority": "primary_small_n",
    },
    {
        "query_id": "CERAD__formal_conventional_only",
        "contrast_id": "CERAD_3-4_vs_1-2",
        "evidence_class": "formal_conventional_only",
        "priority": "secondary_comparator",
    },
    {
        "query_id": "MCI_AD__near_shared",
        "contrast_id": "cogdx_MCI_vs_AD",
        "evidence_class": "near_shared",
        "priority": "primary_near_convergence",
    },
    {
        "query_id": "late_Braak__near_shared",
        "contrast_id": "Braak_3-4_vs_5-6",
        "evidence_class": "near_shared",
        "priority": "primary_near_convergence",
    },
]


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "yes"])
    )


def load_background() -> list[str]:
    m = pd.read_csv(
        MAPPING_AUDIT,
        sep="\t",
        low_memory=False,
    )

    if "one_to_one" not in m.columns:
        raise RuntimeError(
            "Mapping audit does not contain one_to_one column."
        )

    one = m.loc[
        as_bool(m["one_to_one"])
    ].copy()

    symbol_col = None

    for candidate in ["gene_name", "gene_symbol"]:
        if candidate in one.columns:
            symbol_col = candidate
            break

    if symbol_col is None:
        raise RuntimeError(
            "Could not identify symbol column in mapping audit."
        )

    genes = sorted(
        set(
            one[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
    )

    if len(genes) < 4000:
        raise RuntimeError(
            f"Unexpectedly small cross-omic background: {len(genes)}"
        )

    return genes


def load_queries(master: pd.DataFrame):
    queries = []

    for spec in QUERY_SPECS:
        subset = master.loc[
            master["contrast_id"].astype(str).eq(
                spec["contrast_id"]
            )
            & master["evidence_class"].astype(str).eq(
                spec["evidence_class"]
            )
        ].copy()

        genes = sorted(
            set(
                subset["gene_symbol"]
                .dropna()
                .astype(str)
                .str.strip()
            )
        )

        queries.append(
            {
                **spec,
                "n_genes": len(genes),
                "genes": genes,
            }
        )

    return queries


def http_json(url: str, payload=None, timeout=90):
    headers = {
        "Accept": "application/json",
        "User-Agent": "Stage12F-ROSMAP-crossomic-enrichment",
    }

    if payload is None:
        request = urllib.request.Request(
            url,
            headers=headers,
            method="GET",
        )
    else:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def flatten_results(
    query_id: str,
    query_priority: str,
    response: dict,
) -> pd.DataFrame:
    result = response.get("result", [])

    rows = []

    for r in result:
        rows.append(
            {
                "query_id": query_id,
                "priority": query_priority,
                "source": r.get("source"),
                "term_id": r.get("native"),
                "term_name": r.get("name"),
                "p_value": r.get("p_value"),
                "significant": r.get("significant"),
                "term_size": r.get("term_size"),
                "query_size": r.get("query_size"),
                "intersection_size":
                    r.get("intersection_size"),
                "effective_domain_size":
                    r.get("effective_domain_size"),
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                "intersection": ",".join(
                    map(
                        str,
                        r.get("intersections", [])
                        or r.get("intersection", [])
                        or [],
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    require(MASTER)
    require(MAPPING_AUDIT)

    master = pd.read_csv(
        MASTER,
        sep="\t",
        low_memory=False,
    )

    needed = {
        "contrast_id",
        "gene_symbol",
        "evidence_class",
    }

    missing = needed - set(master.columns)

    if missing:
        raise RuntimeError(
            f"MASTER missing columns: {sorted(missing)}"
        )

    background = load_background()
    background_set = set(background)

    queries = load_queries(master)

    print("=" * 120)
    print("STAGE 12F — FOCUSED BIOLOGICAL ENRICHMENT")
    print("=" * 120)
    print(f"Custom cross-omic background: {len(background):,} genes")
    print("Sources: Reactome + GO Biological Process")
    print("Correction: BH FDR")
    print()

    audit_rows = []

    for q in queries:
        outside = sorted(
            set(q["genes"]) - background_set
        )

        if outside:
            raise RuntimeError(
                f'{q["query_id"]}: query genes outside background: '
                f"{outside}"
            )

        audit_rows.append(
            {
                "query_id": q["query_id"],
                "priority": q["priority"],
                "contrast_id": q["contrast_id"],
                "evidence_class": q["evidence_class"],
                "n_genes": q["n_genes"],
                "background_n": len(background),
                "genes": ",".join(q["genes"]),
            }
        )

    audit = pd.DataFrame(audit_rows)

    audit_path = OUTDIR / "ENRICHMENT_QUERY_AUDIT.tsv"

    audit.to_csv(
        audit_path,
        sep="\t",
        index=False,
    )

    # Save exact background.
    (
        OUTDIR / "CROSSOMIC_BACKGROUND_GENES.txt"
    ).write_text(
        "\n".join(background) + "\n"
    )

    # Capture data versions when available.
    try:
        versions = http_json(
            GPROFILER_VERSION_URL,
            payload=None,
        )

        (
            OUTDIR / "GPROFILER_DATA_VERSIONS.json"
        ).write_text(
            json.dumps(
                versions,
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as exc:
        (
            OUTDIR / "GPROFILER_DATA_VERSIONS_ERROR.txt"
        ).write_text(str(exc) + "\n")

    result_tables = []

    for i, q in enumerate(queries, start=1):
        print(
            f'[{i}/{len(queries)}] {q["query_id"]}: '
            f'{q["n_genes"]} genes'
        )

        if q["n_genes"] == 0:
            print("  skipped: empty query")
            continue

        payload = {
            "organism": "hsapiens",
            "query": q["genes"],
            "sources": SOURCES,
            "user_threshold": 0.05,
            "all_results": False,
            "ordered": False,
            "combined": False,
            "measure_underrepresentation": False,
            "no_iea": False,
            "domain_scope": "custom",
            "background": background,
            "significance_threshold_method": "fdr",
            "no_evidences": False,
        }

        payload_path = (
            RAW_DIR
            / f'{q["query_id"]}__request.json'
        )

        payload_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
        )

        try:
            response = http_json(
                GPROFILER_PROFILE_URL,
                payload=payload,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f'g:Profiler HTTP error for {q["query_id"]}: '
                f"{exc.code}\n{body}"
            ) from exc

        response_path = (
            RAW_DIR
            / f'{q["query_id"]}__response.json'
        )

        response_path.write_text(
            json.dumps(
                response,
                indent=2,
                sort_keys=True,
            )
        )

        flat = flatten_results(
            q["query_id"],
            q["priority"],
            response,
        )

        if flat.empty:
            print("  no FDR-significant REAC/GO:BP terms")
        else:
            print(
                f"  significant terms: {len(flat)}"
            )
            result_tables.append(flat)

        # Be polite to the public endpoint.
        time.sleep(0.5)

    if result_tables:
        all_results = pd.concat(
            result_tables,
            ignore_index=True,
        )

        all_results["p_value"] = pd.to_numeric(
            all_results["p_value"],
            errors="coerce",
        )

        all_results = (
            all_results
            .sort_values(
                [
                    "query_id",
                    "p_value",
                    "source",
                    "term_name",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
    else:
        all_results = pd.DataFrame(
            columns=[
                "query_id",
                "priority",
                "source",
                "term_id",
                "term_name",
                "p_value",
                "significant",
                "term_size",
                "query_size",
                "intersection_size",
                "effective_domain_size",
                "precision",
                "recall",
                "intersection",
            ]
        )

    result_path = (
        OUTDIR / "ALL_SIGNIFICANT_ENRICHMENT_TERMS.tsv"
    )

    all_results.to_csv(
        result_path,
        sep="\t",
        index=False,
    )

    # Compact top-term table. This is display-only; no new significance rule.
    top_rows = []

    for query_id, g in all_results.groupby(
        "query_id",
        sort=False,
    ):
        for source, s in g.groupby(
            "source",
            sort=False,
        ):
            top_rows.append(
                s.nsmallest(
                    10,
                    "p_value",
                )
            )

    if top_rows:
        top = pd.concat(
            top_rows,
            ignore_index=True,
        )
    else:
        top = all_results.copy()

    top_path = OUTDIR / "TOP10_TERMS_PER_SOURCE.tsv"

    top.to_csv(
        top_path,
        sep="\t",
        index=False,
    )

    readme = """STAGE 12F — FOCUSED BIOLOGICAL ENRICHMENT
==========================================

No new gene-discovery threshold was introduced.

Background:
    exact Stage 12 one-to-one RNA/protein mappable universe

Sources:
    Reactome
    GO Biological Process

Multiple-testing correction:
    Benjamini-Hochberg FDR

Primary queries:
    NCI-vs-AD formal shared
    NCI-vs-AD formal AREA-only
    CERAD formal shared
    CERAD formal AREA-only
    MCI-vs-AD shared-near
    late-Braak shared-near

Secondary comparators:
    NCI-vs-AD formal conventional-only
    CERAD formal conventional-only

Important:
    The CERAD AREA-only query contains only two genes. Lack of enrichment in
    such a small set is not evidence of absent biological coherence.

All request payloads and raw API responses are saved under raw_gprofiler/.
The g:Profiler datasource-version response is also saved when retrievable.
"""

    (OUTDIR / "README_STAGE12F.txt").write_text(
        readme
    )

    print("\n" + "=" * 120)
    print("SIGNIFICANT TERM COUNTS")
    print("=" * 120)

    counts = (
        all_results
        .groupby(
            ["query_id", "source"],
            dropna=False,
        )
        .size()
        .rename("n_significant_terms")
        .reset_index()
    )

    if counts.empty:
        print("No FDR-significant terms in any query.")
    else:
        print(
            counts.to_string(index=False)
        )

    print("\n" + "=" * 120)
    print("TOP ENRICHMENT TERMS")
    print("=" * 120)

    if top.empty:
        print("No significant terms.")
    else:
        show = [
            "query_id",
            "source",
            "term_id",
            "term_name",
            "p_value",
            "intersection_size",
            "query_size",
            "term_size",
            "intersection",
        ]

        print(
            top[show]
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in [
        audit_path,
        OUTDIR / "CROSSOMIC_BACKGROUND_GENES.txt",
        result_path,
        top_path,
        OUTDIR / "README_STAGE12F.txt",
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()

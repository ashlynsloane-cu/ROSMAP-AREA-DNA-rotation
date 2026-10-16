#!/usr/bin/env python3
"""
Stage 12F v4 — reparse existing g:Profiler raw JSON and recover true
query-gene intersections.

Why this exists
---------------
The Stage 12F enrichment statistics were valid, but the original display
parser incorrectly printed g:Profiler's `intersections` field directly.

For the JSON API, `intersections` is NOT a gene list. It is a list of
per-query-gene evidence-code lists, aligned to:

    meta -> genes_metadata -> query -> <query_name> -> ensgs

The official gprofiler2 client reconstructs the actual query genes by using:
    - the aligned `ensgs`
    - the query `mapping`
    - non-empty entries in `intersections`

This script reproduces that logic locally from the raw JSON files already
saved by Stage 12F. It does NOT rerun enrichment and does NOT contact the API.

Outputs
-------
results/stage12_brain_convergence/stage12F_focused_biological_enrichment/

    ALL_SIGNIFICANT_ENRICHMENT_TERMS_CORRECTED.tsv
    TOP10_TERMS_PER_SOURCE_CORRECTED.tsv
    MULTIGENE_SUPPORTED_ENRICHMENT_TERMS.tsv
    REPARSE_VALIDATION.tsv
    README_STAGE12F_V4.txt

Interpretation note
-------------------
MULTIGENE_SUPPORTED_ENRICHMENT_TERMS.tsv is NOT a second statistical
significance filter. Every row remains g:Profiler FDR-significant.

It simply separates terms supported by >=2 query genes from singleton-overlap
terms. A singleton term can be statistically enriched, especially for a tiny
query and a very small annotation set, but it cannot by itself demonstrate
coherence among multiple genes in the candidate set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path.home() / "Developer/stage11_offline_2026-09-28"

OUTDIR = (
    ROOT
    / "results/stage12_brain_convergence"
    / "stage12F_focused_biological_enrichment"
)

RAW_DIR = OUTDIR / "raw_gprofiler"

AUDIT_PATH = OUTDIR / "ENRICHMENT_QUERY_AUDIT.tsv"


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def reverse_mapping(mapping):
    """
    Convert g:Profiler query mapping into:
        ENSG -> [original query IDs]

    Expected API representation is a dict:
        original_input -> ENSG
    but lists are handled defensively.
    """
    reverse = {}

    if isinstance(mapping, dict):
        for original, mapped in mapping.items():
            values = mapped if isinstance(mapped, list) else [mapped]

            for ensg in values:
                if ensg is None:
                    continue

                ensg = str(ensg)
                reverse.setdefault(ensg, []).append(str(original))

        return reverse

    # Defensive fallback for an unexpected JSON representation.
    return reverse


def reconstruct_intersection(response: dict, result: dict):
    query_name = str(result.get("query", "query_1"))

    meta = response.get("meta", {})
    genes_meta = meta.get("genes_metadata", {})
    query_meta_all = genes_meta.get("query", {})

    if query_name not in query_meta_all:
        raise RuntimeError(
            f"Query metadata missing {query_name}. "
            f"Available: {list(query_meta_all)}"
        )

    qmeta = query_meta_all[query_name]

    ensgs = qmeta.get("ensgs", [])
    mapping = qmeta.get("mapping", {})
    intersections = result.get("intersections", [])

    if len(ensgs) != len(intersections):
        raise RuntimeError(
            "g:Profiler intersections/ENSG alignment mismatch: "
            f"{len(intersections)} intersections vs {len(ensgs)} ENSGs"
        )

    reverse = reverse_mapping(mapping)

    hit_genes = []
    hit_ensgs = []
    evidence_codes = []

    for ensg, evidence in zip(ensgs, intersections):
        # null means not evaluated; [] means not in the term.
        if evidence is None:
            continue

        if not isinstance(evidence, list):
            raise RuntimeError(
                f"Unexpected intersection evidence type: {type(evidence)}"
            )

        if len(evidence) == 0:
            continue

        ensg = str(ensg)
        hit_ensgs.append(ensg)

        originals = reverse.get(ensg, [])

        if originals:
            # With ordinary HGNC-symbol queries this should normally be one ID.
            hit_genes.extend(originals)
        else:
            # Preserve an auditable identifier rather than guessing.
            hit_genes.append(ensg)

        evidence_codes.append(
            " ".join(map(str, evidence))
        )

    # Deduplicate while preserving order.
    hit_genes = list(dict.fromkeys(hit_genes))
    hit_ensgs = list(dict.fromkeys(hit_ensgs))

    return {
        "intersection_genes": ",".join(hit_genes),
        "intersection_ensgs": ",".join(hit_ensgs),
        "evidence_codes": ",".join(evidence_codes),
        "reconstructed_intersection_n": len(hit_genes),
        "reconstructed_ensg_n": len(hit_ensgs),
        "mapping_fallback_to_ensg": any(
            gene.startswith("ENSG")
            for gene in hit_genes
        ),
    }


def flatten_response(query_id: str, response: dict):
    rows = []
    validation = []

    for r in response.get("result", []):
        rebuilt = reconstruct_intersection(
            response,
            r,
        )

        reported_n = int(r.get("intersection_size", 0))
        reconstructed_n = int(
            rebuilt["reconstructed_intersection_n"]
        )

        valid = reported_n == reconstructed_n

        validation.append(
            {
                "query_id": query_id,
                "source": r.get("source"),
                "term_id": r.get("native"),
                "term_name": r.get("name"),
                "reported_intersection_size": reported_n,
                "reconstructed_intersection_size":
                    reconstructed_n,
                "intersection_count_matches": valid,
                "mapping_fallback_to_ensg":
                    rebuilt["mapping_fallback_to_ensg"],
            }
        )

        rows.append(
            {
                "query_id": query_id,
                "source": r.get("source"),
                "term_id": r.get("native"),
                "term_name": r.get("name"),
                # g:Profiler's p_value is already corrected according to
                # significance_threshold_method = fdr in the saved request.
                "gprofiler_fdr": r.get("p_value"),
                "significant": r.get("significant"),
                "term_size": r.get("term_size"),
                "query_size": r.get("query_size"),
                "intersection_size": reported_n,
                "effective_domain_size":
                    r.get("effective_domain_size"),
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                **rebuilt,
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(validation),
    )


def main():
    require(OUTDIR)
    require(RAW_DIR)
    require(AUDIT_PATH)

    audit = pd.read_csv(
        AUDIT_PATH,
        sep="\t",
        low_memory=False,
    )

    if "query_id" not in audit.columns:
        raise RuntimeError(
            "ENRICHMENT_QUERY_AUDIT.tsv lacks query_id."
        )

    all_tables = []
    all_validation = []

    print("=" * 120)
    print("STAGE 12F v4 — CORRECT g:Profiler GENE INTERSECTIONS")
    print("=" * 120)
    print("Using existing raw JSON only; no API queries will be run.")

    for query_id in audit["query_id"].astype(str):
        response_path = (
            RAW_DIR
            / f"{query_id}__response.json"
        )

        if not response_path.exists():
            # Queries with no result should still have a response JSON,
            # but tolerate absent files and report them.
            print(
                f"{query_id}: raw response missing; skipped"
            )
            continue

        response = json.loads(
            response_path.read_text()
        )

        flat, validation = flatten_response(
            query_id,
            response,
        )

        if not flat.empty:
            all_tables.append(flat)

        if not validation.empty:
            all_validation.append(validation)

        print(
            f"{query_id}: {len(flat)} significant terms"
        )

    if all_tables:
        results = pd.concat(
            all_tables,
            ignore_index=True,
        )
    else:
        results = pd.DataFrame()

    if all_validation:
        validation = pd.concat(
            all_validation,
            ignore_index=True,
        )
    else:
        validation = pd.DataFrame()

    if not validation.empty:
        bad = validation.loc[
            ~validation["intersection_count_matches"]
        ]

        fallback = validation.loc[
            validation["mapping_fallback_to_ensg"]
        ]

        if not bad.empty:
            raise RuntimeError(
                "Reconstructed intersection counts do not match "
                "g:Profiler's reported intersection_size:\n"
                + bad.to_string(index=False)
            )

        if not fallback.empty:
            raise RuntimeError(
                "Some intersections could not be mapped back to the "
                "original query symbols and fell back to ENSG IDs:\n"
                + fallback.to_string(index=False)
            )

    if not results.empty:
        results["gprofiler_fdr"] = pd.to_numeric(
            results["gprofiler_fdr"],
            errors="coerce",
        )

        results = (
            results
            .sort_values(
                [
                    "query_id",
                    "gprofiler_fdr",
                    "source",
                    "term_name",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    corrected_path = (
        OUTDIR
        / "ALL_SIGNIFICANT_ENRICHMENT_TERMS_CORRECTED.tsv"
    )

    validation_path = (
        OUTDIR / "REPARSE_VALIDATION.tsv"
    )

    results.to_csv(
        corrected_path,
        sep="\t",
        index=False,
    )

    validation.to_csv(
        validation_path,
        sep="\t",
        index=False,
    )

    # Reporting-only subset: FDR-significant terms supported by >=2 query genes.
    # This is not a new statistical significance filter.
    if not results.empty:
        multigene = results.loc[
            results["intersection_size"] >= 2
        ].copy()
    else:
        multigene = results.copy()

    multigene_path = (
        OUTDIR
        / "MULTIGENE_SUPPORTED_ENRICHMENT_TERMS.tsv"
    )

    multigene.to_csv(
        multigene_path,
        sep="\t",
        index=False,
    )

    # Top 10 per source for compact inspection.
    top_parts = []

    if not results.empty:
        for (query_id, source), g in results.groupby(
            ["query_id", "source"],
            sort=False,
        ):
            top_parts.append(
                g.nsmallest(
                    10,
                    "gprofiler_fdr",
                )
            )

    if top_parts:
        top = pd.concat(
            top_parts,
            ignore_index=True,
        )
    else:
        top = results.copy()

    top_path = (
        OUTDIR
        / "TOP10_TERMS_PER_SOURCE_CORRECTED.tsv"
    )

    top.to_csv(
        top_path,
        sep="\t",
        index=False,
    )

    readme = """STAGE 12F v4 — CORRECTED INTERSECTION PARSING
================================================

The original Stage 12F enrichment statistics were not rerun.

The raw g:Profiler JSON responses were reparsed according to the documented
API structure:

    result.intersections
        per-query-gene evidence-code lists

aligned to:

    meta.genes_metadata.query.<query_name>.ensgs

and converted back to the original query identifiers using:

    meta.genes_metadata.query.<query_name>.mapping

This mirrors the reconstruction logic used by the official gprofiler2 client.

ALL_SIGNIFICANT_ENRICHMENT_TERMS_CORRECTED.tsv
    All g:Profiler FDR-significant terms with correct contributing genes.

MULTIGENE_SUPPORTED_ENRICHMENT_TERMS.tsv
    Reporting-only subset with intersection_size >= 2.

    This is NOT a second statistical threshold. Singleton terms remain
    statistically significant according to g:Profiler, but a term supported
    by one query gene cannot establish biological coherence among multiple
    members of a candidate gene set.

REPARSE_VALIDATION.tsv
    Confirms that reconstructed gene counts equal g:Profiler's reported
    intersection_size for every term.
"""

    readme_path = (
        OUTDIR / "README_STAGE12F_V4.txt"
    )
    readme_path.write_text(readme)

    print("\n" + "=" * 120)
    print("VALIDATION")
    print("=" * 120)

    if validation.empty:
        print("No significant terms to validate.")
    else:
        print(
            f"Validated terms: {len(validation)}"
        )
        print(
            "Intersection counts matched: "
            f"{int(validation['intersection_count_matches'].sum())}"
            f"/{len(validation)}"
        )

    print("\n" + "=" * 120)
    print("CORRECTED SIGNIFICANT TERMS")
    print("=" * 120)

    if results.empty:
        print("No significant terms.")
    else:
        show = [
            "query_id",
            "source",
            "term_id",
            "term_name",
            "gprofiler_fdr",
            "intersection_size",
            "query_size",
            "term_size",
            "intersection_genes",
        ]

        print(
            results[show]
            .to_string(index=False)
        )

    print("\n" + "=" * 120)
    print("MULTI-GENE SUPPORTED SIGNIFICANT TERMS")
    print("=" * 120)

    if multigene.empty:
        print("None.")
    else:
        show = [
            "query_id",
            "source",
            "term_id",
            "term_name",
            "gprofiler_fdr",
            "intersection_size",
            "intersection_genes",
        ]

        print(
            multigene[show]
            .to_string(index=False)
        )

    print("\nWROTE:")
    for path in [
        corrected_path,
        top_path,
        multigene_path,
        validation_path,
        readme_path,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()

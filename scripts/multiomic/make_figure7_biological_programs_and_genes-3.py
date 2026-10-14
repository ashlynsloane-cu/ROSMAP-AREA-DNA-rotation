#!/usr/bin/env python3
"""
Figure 7 — Biological content of cross-omic discoveries.

Scientific question
-------------------
What biological content sits underneath the cross-omic discoveries, and what
biological signal is specific to the AREA formal discovery set?

Panel A
-------
Shared mitochondrial / metabolic program.
The four audited mitochondrial leading-edge genes are mapped to broad pathway
contexts using the frozen Stage 12K shared RNA/protein leading-edge table.
Dot size is the number of unique contrast × method combinations in which the
gene appears in a leading edge assigned to that context. This is descriptive
and annotation-level; it is not a count of independent pathways.

Panel B
-------
NCI→AD formal AREA-only neuronal signaling module.
The frozen Stage 12F enrichment output is restricted to multi-gene-supported
significant terms. Genes are mapped to five exact biological contexts using
term-name rules derived from the Stage 12G focal-module audit. This panel shows
which genes support the enriched neuronal/calcium-signaling annotations.

Panel C
-------
Functional roles of recurrent individual-gene cross-omic discoveries.
The ten recurrent genes identified in the frozen Stage 12E biological-evidence
audit are shown once with source-derived functional annotations. Panel C does
NOT repeat Figure 6C's contrast × evidence matrix and does not use hand-written
functional labels. Reviewed UniProtKB/Swiss-Prot annotations are primary; NCBI
Gene summaries are used only when a reviewed UniProt record is unavailable.

No new discovery threshold is introduced. All gene/pathway membership and
recurrence values are read from frozen Stage 12E/12F/12G/12K outputs.

Run
---
From the ROSMAP-AREA-DNA-rotation repository root:

python3 scripts/multiomic/make_figure7_biological_programs_and_genes.py

Optional explicit source paths can be supplied with --leading-edge-source,
--enrichment-source, and --evidence-source.
"""

from __future__ import annotations

import argparse
import json
import re
from io import StringIO
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WIDTH_MM = 170
HEIGHT_MM = 160
DPI = 600

AREA_COLOR = "#7A8FA8"
AREA_DARK = "#596F86"
NEAR_COLOR = "#8A929A"
CONV_COLOR = "#B8C0CA"
INK = "#283038"
MID = "#66727C"
GRID = "#E3E7EA"
WHITE = "#FFFFFF"

CONTRASTS = [
    ("cogdx_NCI_vs_AD", "NCI → AD"),
    ("cogdx_MCI_vs_AD", "MCI → AD"),
    ("Braak_3-4_vs_5-6", "Late Braak"),
    ("CERAD_3-4_vs_1-2", "CERAD"),
]

METHODS = ["AREA", "CONVENTIONAL"]

MITO_GENES = ["NDUFAF7", "NDUFB5", "NFS1", "DBT"]
MITO_CONTEXTS = [
    "Respiration / OXPHOS",
    "Complex I / assembly",
    "ATP synthesis",
    "Amino-acid / TCA metabolism",
]

FOCAL_GENES = ["CAMKK2", "CAMK4", "CACNG3", "ADAM11", "SST"]
NEURONAL_CONTEXTS = [
    "CaM / CaMK / CREB",
    "Ca²⁺ / DAG / IP3",
    "Postsynaptic / neuronal",
    "LGI–ADAM",
    "MECP2 / neuronal transcription",
]

RECURRENT_ORDER = [
    "KIF13B",
    "CSRP1",
    "VGF",
    "SST",
    "NRIP2",
    "RAB9A",
    "PPEF1",
    "SLC38A2",
    "DHRS7B",
    "TAPT1",
]

# Panel C annotations are NOT hard-coded. They are pulled from authoritative
# human gene/protein resources at runtime and cached as a frozen TSV.
# Primary source: UniProtKB/Swiss-Prot (reviewed human protein records).
# Fallback: NCBI Gene human gene summaries.

ANNOTATION_USER_AGENT = "ROSMAP-AREA-Figure7/1.0"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


STATUS_COLORS = {
    "formal_shared": AREA_DARK,
    "formal_AREA_only": AREA_COLOR,
    "formal_conventional_only": CONV_COLOR,
    "near_shared": NEAR_COLOR,
    "near_AREA_only": "#B7C0C8",
    "near_conventional_only": "#D1D6DB",
}

STATUS_LABELS = {
    "formal_shared": "Formal shared",
    "formal_AREA_only": "AREA-only formal",
    "formal_conventional_only": "Conventional-only formal",
    "near_shared": "Shared near-convergence",
    "near_AREA_only": "AREA-only near-convergence",
    "near_conventional_only": "Conventional-only near-convergence",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--leading-edge-source", type=Path, default=None)
    p.add_argument("--enrichment-source", type=Path, default=None)
    p.add_argument("--evidence-source", type=Path, default=None)
    p.add_argument("--gene-annotation-source", type=Path, default=None,
                   help="Frozen TSV of gene annotations; if omitted, use/create the Figure 7 cache.")
    p.add_argument("--refresh-gene-annotations", action="store_true",
                   help="Refresh Panel C annotations from UniProtKB/Swiss-Prot and NCBI Gene.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        path = path.expanduser().resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not locate {label}. Tried:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def locate_sources(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.repo_root.expanduser().resolve()
    stage12 = root / "data/frozen_inputs"

    leading = (
        args.leading_edge_source.expanduser().resolve()
        if args.leading_edge_source is not None
        else first_existing(
            [
                root / "results/stage12_brain_convergence/stage12K_convergence_architecture/CONTRAST_METHOD_SHARED_LEADING_EDGE_GENES.tsv",
                stage12 / "results/stage12_brain_convergence/stage12K_convergence_architecture/CONTRAST_METHOD_SHARED_LEADING_EDGE_GENES.tsv",
            ],
            "Stage 12K shared leading-edge table",
        )
    )

    enrichment = (
        args.enrichment_source.expanduser().resolve()
        if args.enrichment_source is not None
        else first_existing(
            [
                root / "results/stage12_brain_convergence/stage12F_focused_biological_enrichment/ALL_SIGNIFICANT_ENRICHMENT_TERMS_CORRECTED.tsv",
                stage12 / "results/stage12_brain_convergence/stage12F_focused_biological_enrichment/ALL_SIGNIFICANT_ENRICHMENT_TERMS_CORRECTED.tsv",
            ],
            "Stage 12F corrected enrichment table",
        )
    )

    evidence = (
        args.evidence_source.expanduser().resolve()
        if args.evidence_source is not None
        else first_existing(
            [
                root / "results/stage12_brain_convergence/stage12E_biological_evidence_matrix/MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv",
                stage12 / "results/stage12_brain_convergence/stage12E_biological_evidence_matrix/MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv",
            ],
            "Stage 12E master biological evidence table",
        )
    )

    return leading, enrichment, evidence


def split_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} is missing columns: {sorted(missing)}")


def classify_mito_context(term: object) -> str | None:
    text = str(term).lower()
    if any(x in text for x in [
        "complex i biogenesis",
        "complex i assembly",
        "complex i",
        "complex ii",
        "complex iii",
        "complex iv",
        "complex v",
    ]):
        return "Complex I / assembly"
    if any(x in text for x in [
        "atp synthesis",
        "proton motive force",
        "mitochondrial atp",
    ]):
        return "ATP synthesis"
    if any(x in text for x in [
        "oxidative phosphorylation",
        "respiratory electron transport",
        "aerobic respiration",
        "cellular respiration",
        "respiratory chain",
        "electron transport",
    ]):
        return "Respiration / OXPHOS"
    if any(x in text for x in [
        "amino acid",
        "branched-chain",
        "tca cycle",
        "maturation of tca",
        "amino-acid",
    ]):
        return "Amino-acid / TCA metabolism"
    return None


def build_mito_matrix(leading: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        leading,
        {"contrast_id", "method", "gene_symbol", "term_name"},
        "Stage 12K leading-edge source",
    )

    frame = leading.loc[
        leading["gene_symbol"].astype(str).isin(MITO_GENES)
    ].copy()
    frame["context"] = frame["term_name"].map(classify_mito_context)
    frame = frame.loc[frame["context"].notna()].copy()

    rows = []
    for gene in MITO_GENES:
        for context in MITO_CONTEXTS:
            sub = frame.loc[
                frame["gene_symbol"].astype(str).eq(gene)
                & frame["context"].eq(context)
            ]
            combos = sub[["contrast_id", "method"]].drop_duplicates()
            rows.append({
                "gene_symbol": gene,
                "context": context,
                "n_contrast_method": len(combos),
                "n_terms": len(sub),
            })
    return pd.DataFrame(rows)


def classify_neuronal_context(term: object) -> str | None:
    text = str(term).lower()
    if any(x in text for x in [
        "creb1 phosphorylation",
        "camk iv-mediated phosphorylation of creb",
    ]):
        return "CaM / CaMK / CREB"
    if any(x in text for x in [
        "ca-dependent events",
        "cam pathway",
        "calmodulin induced events",
        "dag and ip3",
        "plc beta",
    ]):
        return "Ca²⁺ / DAG / IP3"
    if "neurotransmitter receptors and postsynaptic signal transmission" in text:
        return "Postsynaptic / neuronal"
    if "lgi-adam interactions" in text:
        return "LGI–ADAM"
    if any(x in text for x in [
        "mecp2",
        "transcriptional regulation by mecp2",
    ]):
        return "MECP2 / neuronal transcription"
    return None


def build_neuronal_matrix(enrichment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        enrichment,
        {"query_id", "source", "term_name", "gprofiler_fdr", "intersection_size", "intersection_genes"},
        "Stage 12F enrichment source",
    )

    frame = enrichment.loc[
        enrichment["query_id"].astype(str).eq("NCI_AD__formal_AREA_only")
        & (pd.to_numeric(enrichment["gprofiler_fdr"], errors="coerce") < 0.05)
        & (pd.to_numeric(enrichment["intersection_size"], errors="coerce") >= 2)
    ].copy()
    frame["context"] = frame["term_name"].map(classify_neuronal_context)
    frame = frame.loc[frame["context"].notna()].copy()

    rows = []
    for _, r in frame.iterrows():
        for gene in split_genes(r["intersection_genes"]):
            if gene in FOCAL_GENES:
                rows.append({
                    "gene_symbol": gene,
                    "context": r["context"],
                    "term_name": r["term_name"],
                    "gprofiler_fdr": float(r["gprofiler_fdr"]),
                    "intersection_size": int(r["intersection_size"]),
                })
    membership = pd.DataFrame(rows)

    matrix_rows = []
    for gene in FOCAL_GENES:
        for context in NEURONAL_CONTEXTS:
            sub = membership.loc[
                membership["gene_symbol"].eq(gene)
                & membership["context"].eq(context)
            ]
            matrix_rows.append({
                "gene_symbol": gene,
                "context": context,
                "detected": not sub.empty,
                "min_fdr": sub["gprofiler_fdr"].min() if not sub.empty else np.nan,
            })
    return pd.DataFrame(matrix_rows), membership


def http_get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": ANNOTATION_USER_AGENT})
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def http_get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": ANNOTATION_USER_AGENT})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def fetch_uniprot_annotations(genes: list[str]) -> pd.DataFrame:
    """Fetch reviewed human UniProtKB annotations for the requested genes."""
    clauses = " OR ".join(f"gene:{gene}" for gene in genes)
    query = f"({clauses}) AND (organism_id:9606) AND (reviewed:true)"
    fields = "accession,id,gene_names,protein_name,cc_function,go_p,go_f"
    url = f"{UNIPROT_SEARCH_URL}?query={quote_plus(query)}&format=tsv&fields={fields}"
    text = http_get_text(url)
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return pd.DataFrame(columns=[
            "gene_symbol", "source", "accession", "protein_name",
            "function_text", "go_bp", "go_mf", "retrieved_date",
        ])

    frame = pd.read_csv(StringIO("\n".join(lines)), sep="\t")
    rename = {
        "Entry": "accession",
        "Entry Name": "entry_name",
        "Gene Names": "gene_names",
        "Protein names": "protein_name",
        "Function [CC]": "function_text",
        "Gene Ontology (biological process)": "go_bp",
        "Gene Ontology (molecular function)": "go_mf",
    }
    frame = frame.rename(columns=rename)

    rows = []
    for _, row in frame.iterrows():
        gene_names = str(row.get("gene_names", "")).split()
        matched = [g for g in genes if g in gene_names]
        if not matched:
            # UniProt may omit a gene-name field in edge cases; do not guess.
            continue
        gene = matched[0]
        rows.append({
            "gene_symbol": gene,
            "source": "UniProtKB/Swiss-Prot",
            "accession": str(row.get("accession", "")),
            "protein_name": str(row.get("protein_name", "")),
            "function_text": str(row.get("function_text", "")),
            "go_bp": str(row.get("go_bp", "")),
            "go_mf": str(row.get("go_mf", "")),
            "retrieved_date": date.today().isoformat(),
        })
    return pd.DataFrame(rows)


def fetch_ncbi_gene_summaries(genes: list[str]) -> pd.DataFrame:
    """Fallback to NCBI Gene when a reviewed human UniProt record is unavailable."""
    rows = []
    for gene in genes:
        term = f"{gene}[Gene Name] AND Homo sapiens[Organism]"
        url = (f"{NCBI_ESEARCH_URL}?db=gene&term={quote_plus(term)}"
               "&retmode=json&retmax=5")
        result = http_get_json(url)
        ids = result.get("esearchresult", {}).get("idlist", [])
        if not ids:
            continue
        # One gene symbol + human organism should resolve to one primary record.
        gene_id = ids[0]
        summary_url = (f"{NCBI_ESUMMARY_URL}?db=gene&id={quote_plus(gene_id)}"
                       "&retmode=json")
        summary = http_get_json(summary_url)
        record = summary.get("result", {}).get(str(gene_id), {})
        rows.append({
            "gene_symbol": gene,
            "source": "NCBI Gene",
            "accession": str(gene_id),
            "protein_name": str(record.get("description", "")),
            "function_text": str(record.get("summary", "")),
            "go_bp": "",
            "go_mf": "",
            "retrieved_date": date.today().isoformat(),
        })
        time.sleep(0.34)
    return pd.DataFrame(rows)


def load_or_fetch_gene_annotations(
    genes: list[str],
    cache_path: Path,
    explicit_source: Path | None,
    refresh: bool,
) -> pd.DataFrame:
    """Load a frozen annotation table or refresh it from authoritative resources."""
    if explicit_source is not None:
        source = explicit_source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Gene annotation source does not exist: {source}")
        return pd.read_csv(source, sep="\t", dtype=str).fillna("")

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, sep="\t", dtype=str).fillna("")

    try:
        annotations = fetch_uniprot_annotations(genes)
    except Exception as exc:
        print(f"WARNING: UniProt annotation fetch failed: {exc}")
        annotations = pd.DataFrame()

    found = set(annotations.get("gene_symbol", pd.Series(dtype=str)).astype(str))
    missing = [gene for gene in genes if gene not in found]
    if missing:
        try:
            fallback = fetch_ncbi_gene_summaries(missing)
        except Exception as exc:
            raise RuntimeError(
                "Could not retrieve missing gene annotations from NCBI Gene after "
                f"UniProt lookup failed. Missing: {missing}. Error: {exc}"
            ) from exc
        annotations = pd.concat([annotations, fallback], ignore_index=True)

    annotations = annotations.drop_duplicates("gene_symbol", keep="first")
    missing_final = [gene for gene in genes if gene not in set(annotations["gene_symbol"])]
    if missing_final:
        raise RuntimeError(
            "No authoritative function annotation was retrieved for: "
            f"{missing_final}. The script will not invent labels."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(cache_path, sep="\t", index=False)
    return annotations


def clean_function_text(text: str) -> str:
    """Remove database markup/references, retaining biological qualifiers.

    The untouched text is retained in function_text in the audit output.
    In particular, 'May', 'Could', and '(By similarity)' are never removed.
    """
    text = re.sub(r"\{[^}]*\}", "", str(text))
    text = re.sub(r"\s*\(PubMed:[^)]*\)", "", text)
    text = re.sub(r"\bFUNCTION:\s*", "", text)
    # Preserve the name of a processed peptide, but remove its field brackets.
    text = re.sub(r"\[([^\]]+)\]:", r"\1:", text)
    return re.sub(r"\s+", " ", text).strip()


def select_function_excerpt(row: pd.Series, gene: str) -> tuple[str, str]:
    """Select source text or a GO term; never turn a tentative claim into fact."""
    raw_text = str(row.get("function_text", ""))
    cleaned = clean_function_text(raw_text)
    sentences = [s.strip(" ;.") for s in re.split(r"(?<=[.!?])\s+", cleaned)]
    sentences = [s for s in sentences if s]
    sentence = sentences[0] if sentences else ""

    # Prefer a gene-specific sentence over an introductory Rab-family overview.
    if sentence.startswith("The small GTPases Rab"):
        sentence = next(
            (s for s in sentences if re.search(rf"\b{re.escape(gene)}\b", s)),
            sentence,
        )

    if not sentence or re.search(r"\b(?:could|may|might|possibly|potentially)\b", sentence, re.I):
        terms = [t.strip() for t in str(row.get("go_mf", "")).split(";") if t.strip()]
        if terms:
            # Terms retain their GO identifiers: this is an annotation, not a
            # claim of experimental verification in human brain.
            return terms[0], "GO molecular function"

    if sentence:
        # Some entries encode similarity evidence only in an ECO evidence tag.
        # Make that qualifier visible if all the function evidence is similarity.
        codes = set(re.findall(r"ECO:(\d+)", raw_text))
        if codes == {"0000250"} and "(By similarity)" not in sentence:
            sentence += " (By similarity)"
        return sentence + ".", "Function excerpt"
    return "No function annotation available.", "Unavailable"


def build_recurrence_annotations(
    evidence: pd.DataFrame,
    annotation_source: Path | None,
    cache_path: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Validate recurrent genes and attach source-derived function annotations."""
    require_columns(
        evidence,
        {"gene_symbol"},
        "Stage 12E master biological evidence source",
    )

    observed = set(evidence["gene_symbol"].astype(str))
    missing = [gene for gene in RECURRENT_ORDER if gene not in observed]
    if missing:
        raise RuntimeError(
            "Stage 12E master evidence is missing recurrent genes: "
            f"{missing}"
        )

    annotations = load_or_fetch_gene_annotations(
        RECURRENT_ORDER, cache_path, annotation_source, refresh
    )
    annotations["gene_symbol"] = annotations["gene_symbol"].astype(str)

    rows = []
    for gene in RECURRENT_ORDER:
        sub = annotations.loc[annotations["gene_symbol"].eq(gene)]
        if sub.empty:
            raise RuntimeError(f"No annotation retrieved for recurrent gene {gene}.")
        row = sub.iloc[0]
        function_text = str(row.get("function_text", "")).strip()
        go_bp = str(row.get("go_bp", "")).strip()
        go_mf = str(row.get("go_mf", "")).strip()
        description, description_basis = select_function_excerpt(row, gene)
        rows.append({
            "gene_symbol": gene,
            "annotation_source": row.get("source", ""),
            "accession": row.get("accession", ""),
            "function_text": function_text,
            "functional_description": description,
            "description_basis": description_basis,
            "go_bp": go_bp,
            "go_mf": go_mf,
            "retrieved_date": row.get("retrieved_date", ""),
        })
    return pd.DataFrame(rows)

def style_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=MID, labelsize=7)


def add_letter(ax, letter: str) -> None:
    ax.text(
        -0.055, 1.02, letter,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def draw_mito_panel(ax, matrix: pd.DataFrame) -> None:
    x = {c: i for i, c in enumerate(MITO_CONTEXTS)}
    y = {g: i for i, g in enumerate(MITO_GENES)}

    for i in range(len(MITO_GENES) + 1):
        ax.axhline(i - 0.5, color=GRID, lw=0.6, zorder=0)
    for i in range(len(MITO_CONTEXTS) + 1):
        ax.axvline(i - 0.5, color=GRID, lw=0.6, zorder=0)

    for _, r in matrix.iterrows():
        n = int(r["n_contrast_method"])
        if n == 0:
            continue
        size = 45 + 42 * n
        ax.scatter(
            x[r["context"]],
            y[r["gene_symbol"]],
            s=size,
            facecolor=AREA_DARK,
            edgecolor=AREA_DARK,
            zorder=3,
        )
        ax.text(
            x[r["context"]],
            y[r["gene_symbol"]],
            str(n),
            ha="center",
            va="center",
            fontsize=5.4,
            color=WHITE,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xticks(range(len(MITO_CONTEXTS)))
    ax.set_xticklabels(
        ["Respiration /\nOXPHOS", "Complex I /\nassembly", "ATP\nsynthesis", "Amino-acid /\nTCA"],
        fontsize=6.2,
        color=INK,
        fontweight="bold",
    )
    ax.set_yticks(range(len(MITO_GENES)))
    ax.set_yticklabels(MITO_GENES, fontsize=7.2, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(MITO_CONTEXTS) - 0.5)
    ax.set_ylim(len(MITO_GENES) - 0.5, -0.5)

    ax.text(
        0, 1.08,
        "Shared mitochondrial leading-edge genes",
        transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=INK, ha="left", va="bottom",
    )
    ax.text(
        1.0, 1.08,
        "4 audited genes",
        transform=ax.transAxes, fontsize=5.5, color=MID, ha="right", va="bottom",
    )
    style_axis(ax)


def draw_neuronal_panel(ax, matrix: pd.DataFrame, membership: pd.DataFrame) -> None:
    x = {c: i for i, c in enumerate(NEURONAL_CONTEXTS)}
    y = {g: i for i, g in enumerate(FOCAL_GENES)}

    for i in range(len(FOCAL_GENES) + 1):
        ax.axhline(i - 0.5, color=GRID, lw=0.6, zorder=0)
    for i in range(len(NEURONAL_CONTEXTS) + 1):
        ax.axvline(i - 0.5, color=GRID, lw=0.6, zorder=0)

    for _, r in matrix.iterrows():
        if not bool(r["detected"]):
            continue
        ax.scatter(
            x[r["context"]],
            y[r["gene_symbol"]],
            s=54,
            facecolor=AREA_COLOR,
            edgecolor=AREA_DARK,
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(range(len(NEURONAL_CONTEXTS)))
    ax.set_xticklabels(
        ["CaM / CaMK /\nCREB", "Ca²⁺ / DAG /\nIP3", "Postsynaptic /\nneuronal", "LGI–ADAM", "MECP2 / neuronal\ntranscription"],
        fontsize=5.9,
        color=INK,
        fontweight="bold",
    )
    ax.set_yticks(range(len(FOCAL_GENES)))
    ax.set_yticklabels(FOCAL_GENES, fontsize=7.0, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(NEURONAL_CONTEXTS) - 0.5)
    ax.set_ylim(len(FOCAL_GENES) - 0.5, -0.5)

    ax.text(
        0, 1.08,
        "AREA-only NCI → AD neuronal signaling module",
        transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=INK, ha="left", va="bottom",
    )
    ax.text(
        1.0, 1.08,
        "16 AREA-only genes · 14 significant terms",
        transform=ax.transAxes, fontsize=5.5, color=MID, ha="right", va="bottom",
    )
    style_axis(ax)


def wrap_to_width(ax, text: str, width: float, fontsize: float) -> str:
    """Wrap to a measured width in axis coordinates, without truncating text."""
    from matplotlib.font_manager import FontProperties

    renderer = ax.figure.canvas.get_renderer()
    available = ax.get_window_extent(renderer).width * width
    prop = FontProperties(size=fontsize)
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        measured = renderer.get_text_width_height_descent(candidate, prop, False)[0]
        if line and measured > available:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines)


def draw_function_panel(ax, annotations: pd.DataFrame) -> None:
    """Two-column table of readable excerpts with source text kept in the audit."""
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.text(
        0, 1.085, "Functional annotations of recurrent genes",
        transform=ax.transAxes, fontsize=8, fontweight="bold",
        color=INK, ha="left", va="bottom",
    )
    ax.text(
        0.012, 1.012, "Gene", transform=ax.transAxes,
        fontsize=6, fontweight="bold", color=MID, ha="left", va="bottom",
    )
    ax.text(
        0.15, 1.012, "Curated function / molecular activity",
        transform=ax.transAxes, fontsize=6, fontweight="bold",
        color=MID, ha="left", va="bottom",
    )

    descriptions = [
        wrap_to_width(ax, str(row["functional_description"]), 0.835, 6.1)
        for _, row in annotations.iterrows()
    ]
    # Allocate a larger row for a longer excerpt rather than squeezing fonts.
    heights = [max(1, d.count("\n") + 1) + 0.58 for d in descriptions]
    total = sum(heights)
    ax.set_ylim(total, 0)
    top = 0
    for i, ((_, row), description, height) in enumerate(
        zip(annotations.iterrows(), descriptions, heights)
    ):
        if i % 2 == 0:
            ax.axhspan(top, top + height, color="#F3F5F6", zorder=0)
        y = top + height / 2
        ax.text(
            0.012, y, row["gene_symbol"], fontsize=6.5,
            fontweight="bold", color=INK, ha="left", va="center",
        )
        ax.text(
            0.15, y, description, fontsize=6.1, linespacing=1.12,
            color=INK, ha="left", va="center",
        )
        top += height

    ax.text(
        0, -0.045,
        "Function excerpts: UniProtKB/Swiss-Prot (NCBI Gene fallback). GO molecular function is used for tentative summaries.\n"
        "Source qualifiers are retained; full annotations, GO terms, accessions and retrieval dates are saved with the figure.",
        transform=ax.transAxes, fontsize=4.8, linespacing=1.25,
        color=MID, ha="left", va="top",
    )

def make_figure(
    mito: pd.DataFrame,
    neuronal: pd.DataFrame,
    annotations: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        facecolor="white",
    )
    # Keep A/B compact and give the wrapped function table its own height.
    ax_a = fig.add_axes([0.17, 0.735, 0.815, 0.118])
    ax_b = fig.add_axes([0.17, 0.525, 0.815, 0.130])
    ax_c = fig.add_axes([0.17, 0.115, 0.815, 0.310])

    add_letter(ax_a, "A")
    add_letter(ax_b, "B")
    add_letter(ax_c, "C")

    draw_mito_panel(ax_a, mito)
    draw_neuronal_panel(ax_b, neuronal, pd.DataFrame())
    draw_function_panel(ax_c, annotations)

    fig.text(
        0.17, 0.968,
        "Figure 7. Biology of cross-omic discoveries",
        fontsize=9.7, fontweight="bold", color=INK, ha="left", va="top",
    )
    fig.text(
        0.17, 0.935,
        "Mitochondrial leading edges, AREA-only neuronal signaling, and recurrent gene functions",
        fontsize=5.75, color=MID, ha="left", va="top",
    )
    fig.text(
        0.17, 0.035,
        "A: leading-edge membership across concordant RNA–protein pathways. B: focal genes in significant AREA-only enrichment terms.",
        fontsize=4.45, color=MID, ha="left", va="bottom",
    )
    return fig

def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "results/figures/figure_7_biology"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    leading_path, enrichment_path, evidence_path = locate_sources(args)
    leading = pd.read_csv(leading_path, sep="\t", low_memory=False)
    enrichment = pd.read_csv(enrichment_path, sep="\t", low_memory=False)
    evidence = pd.read_csv(evidence_path, sep="\t", low_memory=False)

    mito = build_mito_matrix(leading)
    neuronal, membership = build_neuronal_matrix(enrichment)
    annotation_cache = output_dir / "Figure7_gene_function_annotations.tsv"
    annotations = build_recurrence_annotations(
        evidence,
        args.gene_annotation_source,
        annotation_cache,
        args.refresh_gene_annotations,
    )

    fig = make_figure(mito, neuronal, annotations)
    stem = "Figure7_biological_programs_functional_roles"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    mito.to_csv(output_dir / "Figure7_mitochondrial_program_audit.csv", index=False)
    neuronal.to_csv(output_dir / "Figure7_neuronal_module_audit.csv", index=False)
    membership.to_csv(output_dir / "Figure7_neuronal_enrichment_membership.csv", index=False)
    annotations.to_csv(output_dir / "Figure7_recurrent_gene_function_annotations.csv", index=False)

    (output_dir / "Figure7_source.txt").write_text(
        f"Stage 12K leading edge: {leading_path}\n"
        f"Stage 12F corrected enrichment: {enrichment_path}\n"
        f"Stage 12E master evidence: {evidence_path}\n"
        "No new statistical thresholds introduced.\n"
        "Panel C functional annotations are source-derived from UniProtKB/Swiss-Prot, with NCBI Gene fallback.\n"
        f"Gene annotation cache: {annotation_cache}\n"
    )

    print("FIGURE 7 SOURCES")
    print(f"  Leading edge: {leading_path}")
    print(f"  Enrichment:   {enrichment_path}")
    print(f"  Stage 12E evidence: {evidence_path}")
    print("\nMITOCHONDRIAL PROGRAM AUDIT")
    print(mito.to_string(index=False))
    print("\nNEURONAL MODULE AUDIT")
    print(neuronal.to_string(index=False))
    print("\nNEURONAL ENRICHMENT MEMBERSHIP")
    print(membership.to_string(index=False))
    print("\nRECURRENT GENE FUNCTION ANNOTATIONS")
    print(annotations.to_string(index=False))
    print("\nFIGURE 7 OUTPUTS")
    print(f"  PNG: {png}")
    print(f"  PDF: {pdf}")
    print(f"  SVG: {svg}")


if __name__ == "__main__":
    main()

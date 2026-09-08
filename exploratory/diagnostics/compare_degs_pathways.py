#!/usr/bin/env python3
"""
Comparative Hallmark pathway enrichment for the seven corrected Venn regions.

This revision resumes from completed per-region CSVs and backs off automatically
when Enrichr returns HTTP 429 or other transient server errors.

This version deliberately DOES NOT re-merge the raw DESeq2 and AREA result files.
It reads the gene lists exported from the corrected Venn membership table so that
the enrichment analysis is guaranteed to use the exact same region definitions
as the Venn diagram.

Default enrichment library:
    MSigDB_Hallmark_2020 via Enrichr

Default plotting rule:
    Plot up to --top-n pathways per region with Enrichr adjusted P < --fdr-cutoff.
    Raw P values are retained in the output tables but are not used as the default
    significance criterion.

Outputs:
    <outdir>/enrichr_<region>.csv
    <outdir>/enrichr_all_regions.csv
    <outdir>/enrichment_region_summary.csv
    <output>.png
    <output>.pdf
"""

import argparse
import json
import math
import os
import ssl
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


REGION_INFO = {
    "DESeq2_only": {
        "label": "DESeq2 only",
        "description": "Group-mean only",
        "color": "#1f77b4",
    },
    "Regular_AREA_only": {
        "label": "Regular AREA only",
        "description": "State-transition only",
        "color": "#ff7f0e",
    },
    "Weighted_AREA_only": {
        "label": "Weighted AREA only",
        "description": "Continuous-progression only",
        "color": "#2ca02c",
    },
    "DESeq2_Regular_only": {
        "label": "DESeq2 + Regular",
        "description": "Mean + state-transition",
        "color": "#9467bd",
    },
    "DESeq2_Weighted_only": {
        "label": "DESeq2 + Weighted",
        "description": "Mean + continuous progression",
        "color": "#17becf",
    },
    "Regular_Weighted_only": {
        "label": "Regular + Weighted AREA",
        "description": "AREA-consistent",
        "color": "#d62728",
    },
    "All_three": {
        "label": "All three",
        "description": "Cross-method concordant",
        "color": "#111111",
    },
}

REGION_ORDER = list(REGION_INFO.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hallmark Enrichr analysis on seven corrected Venn gene sets."
    )
    parser.add_argument(
        "--sets-dir",
        required=True,
        help="Directory containing *_genes.txt from export_updated_venn_gene_sets.py.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for enrichment result CSVs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output PNG path for the comparative bar plot.",
    )
    parser.add_argument(
        "--library",
        default="MSigDB_Hallmark_2020",
        help="Enrichr library. Default: MSigDB_Hallmark_2020",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Maximum number of significant pathways to plot per region. Default: 3",
    )
    parser.add_argument(
        "--fdr-cutoff",
        type=float,
        default=0.05,
        help="Adjusted-p-value cutoff for plotted pathways. Default: 0.05",
    )
    parser.add_argument(
        "--regions",
        default="all",
        help=(
            "Comma-separated region names to analyze/plot, or 'all'. Valid names: "
            + ",".join(REGION_ORDER)
        ),
    )
    parser.add_argument(
        "--show-nonsignificant-top",
        action="store_true",
        help=(
            "If a region has no FDR-significant Hallmark pathways, show its top "
            "pathways anyway. These bars are marked with an asterisk."
        ),
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.0,
        help="Seconds to wait between Enrichr gene-set queries. Default: 3.0",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retries after HTTP 429/transient server errors. Default: 6",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="Initial retry wait; doubles after each failed attempt. Default: 5 seconds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached per-region CSVs and query Enrichr again.",
    )
    return parser.parse_args()


def encode_multipart_formdata(fields):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    lines = []
    for name, value in fields.items():
        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="{name}"')
        lines.append("")
        lines.append(str(value))
    lines.append(f"--{boundary}--")
    body = "\r\n".join(lines).encode("utf-8")
    content_type = f"multipart/form-data; boundary={boundary}"
    return content_type, body


def ssl_context():
    return ssl._create_unverified_context()


def read_gene_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing gene list: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        genes = [line.strip().upper() for line in handle if line.strip()]

    return list(dict.fromkeys(genes))



def urlopen_with_retry(req, max_retries, retry_base_seconds):
    """Open a URL with exponential backoff for rate limits/transient errors."""
    transient_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            return urllib.request.urlopen(req, context=ssl_context())

        except urllib.error.HTTPError as exc:
            if exc.code not in transient_codes or attempt >= max_retries:
                raise

            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = retry_base_seconds * (2 ** attempt)
            else:
                wait = retry_base_seconds * (2 ** attempt)

            wait = min(wait, 120.0)
            print(
                f"    HTTP {exc.code} from Enrichr. "
                f"Waiting {wait:.0f}s before retry "
                f"{attempt + 1}/{max_retries}..."
            )
            time.sleep(wait)

        except urllib.error.URLError as exc:
            if attempt >= max_retries:
                raise

            wait = min(retry_base_seconds * (2 ** attempt), 120.0)
            print(
                f"    Temporary network error: {exc}. "
                f"Waiting {wait:.0f}s before retry "
                f"{attempt + 1}/{max_retries}..."
            )
            time.sleep(wait)

    raise RuntimeError("Retry loop exited unexpectedly.")


def query_enrichr(gene_list, list_desc, library, max_retries, retry_base_seconds):
    if not gene_list:
        print(f" -> {list_desc}: empty gene list; skipping.")
        return pd.DataFrame(
            columns=[
                "Pathway",
                "P_Value",
                "Adjusted_P_Value",
                "Z_Score",
                "Combined_Score",
                "Overlap",
            ]
        )

    print(f" -> Submitting {len(gene_list):,} genes for {list_desc}...")

    add_list_url = "https://maayanlab.cloud/Enrichr/addList"
    payload = {
        "list": "\n".join(gene_list),
        "description": list_desc,
    }
    content_type, body = encode_multipart_formdata(payload)

    req = urllib.request.Request(add_list_url, data=body, method="POST")
    req.add_header("Content-Type", content_type)

    with urlopen_with_retry(req, max_retries, retry_base_seconds) as response:
        res_json = json.loads(response.read().decode("utf-8"))

    user_list_id = res_json.get("userListId")
    if user_list_id is None:
        raise RuntimeError(f"Enrichr did not return a userListId for {list_desc}.")

    query = urllib.parse.urlencode(
        {
            "userListId": user_list_id,
            "backgroundType": library,
        }
    )
    enrich_url = f"https://maayanlab.cloud/Enrichr/enrich?{query}"
    req_enrich = urllib.request.Request(enrich_url, method="GET")

    with urlopen_with_retry(req_enrich, max_retries, retry_base_seconds) as response:
        res_json = json.loads(response.read().decode("utf-8"))

    results = res_json.get(library, [])

    rows = []
    for item in results:
        rows.append(
            {
                "Pathway": item[1],
                "P_Value": item[2],
                "Z_Score": item[3],
                "Combined_Score": item[4],
                "Overlap": item[5],
                "Adjusted_P_Value": item[6],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["P_Value", "Adjusted_P_Value", "Z_Score", "Combined_Score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(
        ["Adjusted_P_Value", "P_Value"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def clean_pathway_label(term):
    label = str(term)
    parts = label.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].startswith("M") and parts[1][1:].isdigit():
        label = parts[0]
    return label.replace("_", " ")


def select_plot_terms(df, top_n, fdr_cutoff, show_nonsignificant_top):
    if df.empty:
        return df.copy()

    sig = df[
        df["Adjusted_P_Value"].notna()
        & (df["Adjusted_P_Value"] < fdr_cutoff)
    ].copy()

    if not sig.empty:
        chosen = sig.head(top_n).copy()
        chosen["is_fdr_significant"] = True
        return chosen

    if show_nonsignificant_top:
        chosen = df.head(top_n).copy()
        chosen["is_fdr_significant"] = False
        return chosen

    return df.head(0).copy()


def build_plot(selected_by_region, regions, output, fdr_cutoff):
    rows = []
    y = 0.0
    gap = 0.8
    separators = []
    region_bounds = []

    for region in regions:
        df = selected_by_region[region].copy()
        start = y

        for _, row in df.iterrows():
            adj = float(row["Adjusted_P_Value"])
            plot_value = -math.log10(max(adj, 1e-300))
            significant = bool(row["is_fdr_significant"])

            label = clean_pathway_label(row["Pathway"])
            if not significant:
                label += " *"

            rows.append(
                {
                    "region": region,
                    "y": y,
                    "label": label,
                    "value": plot_value,
                    "significant": significant,
                }
            )
            y += 1.0

        end = y - 1.0
        if not df.empty:
            region_bounds.append((region, start, end))

        y += gap
        separators.append(y - gap / 2.0)

    if separators:
        separators = separators[:-1]

    fig_height = max(5.5, 0.42 * max(len(rows), 1) + 2.5)
    fig, ax = plt.subplots(figsize=(6.69, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if rows:
        y_values = [r["y"] for r in rows]
        x_values = [r["value"] for r in rows]
        bar_colors = [REGION_INFO[r["region"]]["color"] for r in rows]

        ax.barh(
            y_values,
            x_values,
            color=bar_colors,
            edgecolor=bar_colors,
            height=0.62,
        )
        ax.set_yticks(y_values)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=8.3)
        ax.invert_yaxis()

        for sep in separators:
            ax.axhline(sep, color="#d0d0d0", linewidth=0.8)

        xmax = max(x_values) if x_values else 1.0
        for region, start, end in region_bounds:
            mid = (start + end) / 2.0
            ax.text(
                xmax * 1.02,
                mid,
                REGION_INFO[region]["label"],
                va="center",
                ha="left",
                fontsize=8.1,
                fontweight="bold",
                color=REGION_INFO[region]["color"],
            )

        ax.set_xlim(0, xmax * 1.28)
    else:
        ax.text(
            0.5,
            0.5,
            f"No Hallmark pathways met adjusted P < {fdr_cutoff}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_yticks([])

    ax.set_xlabel("-log10 (adjusted p-value)", fontsize=9.5, fontweight="semibold")
    ax.set_title(
        "Pathway Enrichment Across Corrected Venn Regions",
        fontsize=11.5,
        fontweight="bold",
        pad=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=REGION_INFO[r]["color"],
            edgecolor=REGION_INFO[r]["color"],
            label=REGION_INFO[r]["label"],
        )
        for r in regions
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=7.8,
    )

    plt.tight_layout()

    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    root, _ = os.path.splitext(output)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote: {output}")
    print(f"Wrote: {root + '.pdf'}")


def main():
    args = parse_args()

    if args.regions.strip().lower() == "all":
        regions = REGION_ORDER
    else:
        regions = [x.strip() for x in args.regions.split(",") if x.strip()]
        invalid = [x for x in regions if x not in REGION_INFO]
        if invalid:
            raise ValueError(
                f"Invalid region(s): {invalid}\nValid regions: {REGION_ORDER}"
            )

    os.makedirs(args.outdir, exist_ok=True)

    combined = []
    selected_by_region = {}
    summary_rows = []

    print("=" * 80)
    print("CORRECTED VENN HALLMARK ENRICHMENT")
    print("=" * 80)
    print(f"Library: {args.library}")
    print(f"FDR cutoff for plotted terms: {args.fdr_cutoff}")
    print(f"Top pathways per region: {args.top_n}")

    for i, region in enumerate(regions):
        info = REGION_INFO[region]
        gene_file = os.path.join(args.sets_dir, f"{region}_genes.txt")
        genes = read_gene_list(gene_file)

        print(f"\n{info['label']}: {len(genes):,} genes")
        region_out = os.path.join(args.outdir, f"enrichr_{region}.csv")

        # Resume safely: reuse completed region output unless --force is supplied.
        if os.path.exists(region_out) and not args.force:
            print(f"  Cached result found; reusing: {region_out}")
            df = pd.read_csv(region_out)
        else:
            df = query_enrichr(
                genes,
                info["label"],
                args.library,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
            )

            if not df.empty:
                df["Region"] = region
                df["Region_Label"] = info["label"]
                df["N_Input_Genes"] = len(genes)

            df.to_csv(region_out, index=False)

        n_fdr = (
            int((df["Adjusted_P_Value"] < args.fdr_cutoff).sum())
            if not df.empty
            else 0
        )

        selected = select_plot_terms(
            df,
            top_n=args.top_n,
            fdr_cutoff=args.fdr_cutoff,
            show_nonsignificant_top=args.show_nonsignificant_top,
        )
        selected_by_region[region] = selected

        summary_rows.append(
            {
                "region": region,
                "region_label": info["label"],
                "n_input_genes": len(genes),
                "n_enrichr_terms": len(df),
                f"n_terms_adjusted_p_lt_{args.fdr_cutoff}": n_fdr,
                "results_file": region_out,
            }
        )

        print(f"  Hallmark terms returned: {len(df):,}")
        print(f"  Adjusted P < {args.fdr_cutoff}: {n_fdr:,}")
        print(f"  Wrote: {region_out}")

        if not df.empty:
            combined.append(df)

        if i < len(regions) - 1 and args.request_delay > 0:
            time.sleep(args.request_delay)

    combined_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    combined_path = os.path.join(args.outdir, "enrichr_all_regions.csv")
    combined_df.to_csv(combined_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "enrichment_region_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nWrote: {combined_path}")
    print(f"Wrote: {summary_path}")

    build_plot(
        selected_by_region=selected_by_region,
        regions=regions,
        output=args.output,
        fdr_cutoff=args.fdr_cutoff,
    )

    if args.show_nonsignificant_top:
        print(
            "\nNOTE: Pathway labels ending in '*' did not meet the adjusted-p-value "
            "cutoff and are shown only because --show-nonsignificant-top was used."
        )


if __name__ == "__main__":
    main()

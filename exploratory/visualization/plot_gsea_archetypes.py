#!/usr/bin/env python3
"""Classify and plot paired Regular/Weighted AREA GSEA results.

Classification is performed for each pathway-by-contrast pair. This avoids
averaging p-values across unrelated clinical contrasts. By default, significance
means BH-adjusted FDR < 0.05.
"""

import argparse
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "axes.titleweight": "bold",
})

ARCHETYPES = {
    "Shared Response": {
        "title": "Pathways detected by both AREA models",
        "methods": ("Regular", "Weighted"),
        "slug": "shared_response",
    },
    "Threshold-like Response": {
        "title": "Pathways detected only by Regular AREA",
        "methods": ("Regular",),
        "slug": "threshold_like_response",
    },
    "Dosage-like Response": {
        "title": "Pathways detected only by Weighted AREA",
        "methods": ("Weighted",),
        "slug": "dosage_like_response",
    },
}

METHOD_STYLE = {
    "Regular": {"marker": "o", "offset": -0.12, "label": "Regular AREA"},
    "Weighted": {"marker": "D", "offset": 0.12, "label": "Weighted AREA"},
}


def normalize_method(value):
    text = str(value).strip().lower()
    # Test unweighted before weighted: "weighted" is a substring of "unweighted".
    if re.search(r"\b(unweighted|regular)\b", text):
        return "Regular"
    if re.search(r"\bweighted\b", text):
        return "Weighted"
    return "Unknown"


def contrast_from_attribute(value):
    text = str(value).strip()
    text = re.sub(
        r"[\s_\-]*(\(|\[)?\b(unweighted|regular|weighted)(\s+area)?\b(\)|\])?[\s_\-]*$",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"[_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -_()[]")
    return text or "All contrasts"


def standardize_input(df):
    lower = {str(col).lower(): col for col in df.columns}
    required = {"pathway": "Pathway", "nes": "NES", "p_value": "P_Value"}
    rename = {}
    for key, canonical in required.items():
        if key not in lower:
            raise ValueError(f"Required column '{canonical}' was not found. Columns: {list(df.columns)}")
        rename[lower[key]] = canonical
    for key, canonical in (("fdr_bh", "FDR_BH"), ("attribute", "Attribute"),
                           ("method", "Method"), ("contrast", "Contrast")):
        if key in lower:
            rename[lower[key]] = canonical
    out = df.rename(columns=rename).copy()

    if "FDR_BH" not in out:
        print("Warning: FDR_BH is absent; copying nominal P_Value into FDR_BH.")
        out["FDR_BH"] = out["P_Value"]
    if "Method" in out:
        out["Method"] = out["Method"].map(normalize_method)
    elif "Attribute" in out:
        out["Method"] = out["Attribute"].map(normalize_method)
    else:
        raise ValueError("Input needs either a 'Method' column or method names inside 'Attribute'.")

    if "Contrast" not in out:
        if "Attribute" not in out:
            raise ValueError("Input needs either a 'Contrast' column or an 'Attribute' column.")
        out["Contrast"] = out["Attribute"].map(contrast_from_attribute)

    unknown = sorted(out.loc[out["Method"] == "Unknown", "Attribute"].astype(str).unique()) \
        if "Attribute" in out else []
    if (out["Method"] == "Unknown").any():
        raise ValueError(
            "Could not identify Regular versus Weighted AREA for: "
            + ", ".join(unknown[:8])
        )
    for col in ("NES", "P_Value", "FDR_BH"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Pathway", "Contrast", "Method", "NES", "P_Value", "FDR_BH"])
    return out


def collapse_duplicates(df, policy="conservative"):
    keys = ["Pathway", "Contrast", "Method"]
    duplicate_rows = int(df.duplicated(keys, keep=False).sum())
    if duplicate_rows:
        print(
            f"Warning: {duplicate_rows} rows share a pathway/contrast/method key; "
            f"using '{policy}' duplicate aggregation."
        )
    if policy == "conservative":
        agg = {"NES": "mean", "P_Value": "max", "FDR_BH": "max"}
    elif policy == "mean":
        agg = {"NES": "mean", "P_Value": "mean", "FDR_BH": "mean"}
    else:
        agg = {"NES": "mean", "P_Value": "min", "FDR_BH": "min"}
    return df.groupby(keys, as_index=False).agg(agg)


def classify_pairs(df, significance_column="FDR_BH", cutoff=0.05):
    records = []
    for (pathway, contrast), group in df.groupby(["Pathway", "Contrast"], sort=False):
        by_method = group.set_index("Method")
        row = {"Pathway": pathway, "Contrast": contrast}
        for method in ("Regular", "Weighted"):
            if method in by_method.index:
                method_row = by_method.loc[method]
                row[f"NES_{method}"] = float(method_row["NES"])
                row[f"P_Value_{method}"] = float(method_row["P_Value"])
                row[f"FDR_BH_{method}"] = float(method_row["FDR_BH"])
                row[f"Significant_{method}"] = bool(method_row[significance_column] < cutoff)
            else:
                row[f"NES_{method}"] = np.nan
                row[f"P_Value_{method}"] = np.nan
                row[f"FDR_BH_{method}"] = np.nan
                row[f"Significant_{method}"] = False

        reg = row["Significant_Regular"]
        wgt = row["Significant_Weighted"]
        if reg and wgt:
            row["Archetype"] = "Shared Response"
        elif reg:
            row["Archetype"] = "Threshold-like Response"
        elif wgt:
            row["Archetype"] = "Dosage-like Response"
        else:
            row["Archetype"] = "Not Significant"
        row["Direction_Agreement"] = (
            np.sign(row["NES_Regular"]) == np.sign(row["NES_Weighted"])
            if reg and wgt else np.nan
        )
        records.append(row)
    return pd.DataFrame(records)


def clean_pathway(value):
    text = re.sub(r"^(HALLMARK|REACTOME|KEGG)_", "", str(value), flags=re.IGNORECASE)
    text = text.replace("_", " ").lower().capitalize()
    replacements = {
        "Mtorc1": "mTORC1", "Tnfa": "TNFα", "Nfkb": "NF-κB",
        "Il6": "IL-6", "Jak stat3": "JAK–STAT3", "Apoptosis": "Apoptosis",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return textwrap.fill(text, width=25)


def clean_contrast(value):
    return textwrap.fill(str(value).replace("_", " "), width=28)


def bubble_size(values):
    strength = -np.log10(np.clip(np.asarray(values, dtype=float), 1e-10, 1.0))
    return np.clip(30 + 34 * strength, 30, 240)


def plot_archetype(arch_name, pair_df, outdir, significance_column, cutoff):
    config = ARCHETYPES[arch_name]
    selected = pair_df[pair_df["Archetype"] == arch_name].copy()
    if selected.empty:
        print(f" -> No pathway/contrast pairs classified as {arch_name}; no plot created.")
        return

    pathways = list(dict.fromkeys(selected["Pathway"]))
    contrasts = list(dict.fromkeys(selected["Contrast"]))
    pathways.sort(key=lambda x: str(x).lower())
    contrasts.sort(key=lambda x: str(x).lower())
    x_lookup = {value: i for i, value in enumerate(pathways)}
    y_lookup = {value: i for i, value in enumerate(contrasts)}

    width = min(16.0, max(8.5, 3.6 + 2.2 * len(pathways)))
    height = min(13.0, max(4.5, 2.3 + 0.48 * len(contrasts)))
    fig, ax = plt.subplots(figsize=(width, height))

    all_nes = []
    for method in config["methods"]:
        all_nes.extend(selected[f"NES_{method}"].dropna().tolist())
    limit = max(1.0, float(np.nanpercentile(np.abs(all_nes), 95)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    last_scatter = None

    for method in config["methods"]:
        metric_col = f"{significance_column}_{method}"
        subset = selected.dropna(subset=[f"NES_{method}", metric_col])
        method_offset = METHOD_STYLE[method]["offset"] if len(config["methods"]) > 1 else 0
        x = [x_lookup[p] + method_offset for p in subset["Pathway"]]
        y = [y_lookup[c] for c in subset["Contrast"]]
        last_scatter = ax.scatter(
            x, y, s=bubble_size(subset[metric_col]), c=subset[f"NES_{method}"],
            cmap=cmap, norm=norm, marker=METHOD_STYLE[method]["marker"],
            edgecolors="#303030", linewidths=0.7, alpha=0.92, zorder=3,
        )

    ax.set_xticks(range(len(pathways)), [clean_pathway(x) for x in pathways], fontsize=8.8)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(contrasts)), [clean_contrast(x) for x in contrasts], fontsize=9)
    ax.invert_yaxis()
    ax.grid(color="#E4E4E4", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel("Pathway", fontsize=10.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("Clinical contrast", fontsize=10.5, fontweight="bold", labelpad=10)
    fig.suptitle(config["title"], fontsize=13.5, fontweight="bold", y=0.985)
    ax.set_title(
        f"Color = NES  •  bubble size = -log10({significance_column.replace('_BH', '')})  •  cutoff < {cutoff:g}",
        fontsize=9.5, color="#555555", pad=12,
    )

    cbar = fig.colorbar(last_scatter, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Normalized enrichment score (NES)", fontsize=9.5)
    handles = [
        Line2D([0], [0], marker=METHOD_STYLE[m]["marker"], linestyle="none",
               markerfacecolor="#BDBDBD", markeredgecolor="#303030", markersize=7,
               label=METHOD_STYLE[m]["label"])
        for m in config["methods"]
    ]
    size_values = [0.05, 0.01, 0.001]
    handles.extend(
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#555555", markersize=np.sqrt(bubble_size([v])[0]),
               label=f"{significance_column.replace('_BH', '')} = {v:g}")
        for v in size_values
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.47, 0.012),
               frameon=False, fontsize=8.5, ncol=len(handles),
               handletextpad=0.6, columnspacing=1.4)
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))

    png_path = os.path.join(outdir, f"gsea_bubble_{config['slug']}.png")
    pdf_path = os.path.join(outdir, f"gsea_bubble_{config['slug']}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" -> Saved {png_path} and {pdf_path}")


def make_self_test_data():
    rng = np.random.default_rng(7)
    pathways = ["HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_HYPOXIA",
                "HALLMARK_MTORC1_SIGNALING", "HALLMARK_OXIDATIVE_PHOSPHORYLATION"]
    contrasts = ["Cognitive impairment vs NCI", "Braak high vs low", "APOE4 carrier"]
    rows = []
    for p_idx, pathway in enumerate(pathways):
        for c_idx, contrast in enumerate(contrasts):
            for method in ("Regular", "Weighted"):
                fdr = 0.01 if (p_idx + c_idx + (method == "Weighted")) % 3 else 0.12
                rows.append({
                    "Pathway": pathway,
                    "Attribute": f"{contrast} ({method})",
                    "NES": rng.uniform(-2.4, 2.4),
                    "P_Value": fdr / 2,
                    "FDR_BH": fdr,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Paired Regular/Weighted AREA GSEA plots.")
    parser.add_argument("-i", "--input", help="Path to gsea_enrichment_summary.csv.")
    parser.add_argument("-o", "--outdir", default="results/gsea_archetypes")
    parser.add_argument("--significance-column", choices=["FDR_BH", "P_Value"], default="FDR_BH")
    parser.add_argument("--cutoff", "--p-cutoff", dest="cutoff", type=float, default=0.05)
    parser.add_argument("--agg", choices=["conservative", "mean", "min"], default="conservative",
                        help="How to combine true duplicate pathway/contrast/method rows.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raw = make_self_test_data()
    else:
        if not args.input:
            parser.error("--input is required unless --self-test is used")
        if not os.path.exists(args.input):
            print(f"Error: input file '{args.input}' not found.")
            sys.exit(1)
        raw = pd.read_csv(args.input)

    try:
        standardized = standardize_input(raw)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    collapsed = collapse_duplicates(standardized, args.agg)
    classified = classify_pairs(collapsed, args.significance_column, args.cutoff)
    os.makedirs(args.outdir, exist_ok=True)
    classification_path = os.path.join(args.outdir, "gsea_pathway_archetype_classifications.csv")
    classified.to_csv(classification_path, index=False)
    print(f"Saved pair-level classifications: {classification_path}")
    print("\nPathway/contrast-pair distribution:")
    print(classified["Archetype"].value_counts().to_string())

    discordant = classified[
        (classified["Archetype"] == "Shared Response")
        & (classified["Direction_Agreement"] == False)  # noqa: E712
    ]
    if len(discordant):
        print(
            f"Warning: {len(discordant)} shared significant pair(s) have opposite NES "
            "directions; inspect Direction_Agreement in the classification CSV."
        )
    for arch_name in ARCHETYPES:
        plot_archetype(
            arch_name, classified, args.outdir,
            args.significance_column, args.cutoff,
        )
    print("\nCompleted GSEA archetype plotting.")


if __name__ == "__main__":
    main()

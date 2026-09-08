#!/usr/bin/env python3
"""Plot gene expression by a binary clinical contrast.

The expression input may be either gene-by-sample (a ``gene_symbol`` column
followed by sample columns) or sample-by-gene (a patient ID column followed by
gene columns). ROSMAP sample labels such as ``492_120515`` are matched to
clinical ``projid`` values such as ``492``.
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "axes.titleweight": "bold",
})

COLORS = {"Group 0": "#8DA9C4", "Group 1": "#E07A5F"}
GENE_COLUMN_CANDIDATES = ("gene_symbol", "gene", "symbol", "gene_name")


def canonical_id(value):
    """Normalize one patient ID without destroying non-numeric IDs."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    text = re.sub(r"\.0$", "", text)
    text = re.sub(r"^R(?=\d+$)", "", text)
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return text


def match_sample_to_patient(sample_id, known_patient_ids):
    """Match a sample label to a clinical ID, including ROSMAP suffixes."""
    exact = canonical_id(sample_id)
    if exact in known_patient_ids:
        return exact
    first_piece = re.split(r"[_|:/.-]", str(sample_id).strip(), maxsplit=1)[0]
    candidate = canonical_id(first_piece)
    if candidate in known_patient_ids:
        return candidate
    return exact


def find_gene_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Gene column '{requested}' was not found. Available columns begin: "
                f"{list(df.columns[:8])}"
            )
        return requested
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for candidate in GENE_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def expression_to_patient_rows(expr_df, genes, clinical_ids, id_col_expr, gene_col=None):
    """Return a patient-by-gene numeric matrix from either input orientation."""
    detected_gene_col = find_gene_column(expr_df, gene_col)
    if detected_gene_col is not None:
        print(
            f"Detected gene-by-sample expression matrix using gene column "
            f"'{detected_gene_col}'. Transposing samples into rows."
        )
        gene_names = expr_df[detected_gene_col].astype(str).str.strip().str.upper()
        selected = expr_df.loc[gene_names.isin(set(genes))].copy()
        selected[detected_gene_col] = gene_names[gene_names.isin(set(genes))].values
        if selected.empty:
            return pd.DataFrame(columns=["ID_Clean", *genes])

        sample_cols = [col for col in expr_df.columns if col != detected_gene_col]
        numeric = selected.set_index(detected_gene_col)[sample_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.index.duplicated().any():
            duplicate_genes = sorted(numeric.index[numeric.index.duplicated()].unique())
            print("Warning: duplicate gene rows were averaged for: " + ", ".join(duplicate_genes))
            numeric = numeric.groupby(level=0).mean()
        patient_rows = numeric.T
        patient_rows.index.name = "Sample_ID"
        patient_rows = patient_rows.reset_index()
        patient_rows["ID_Clean"] = patient_rows["Sample_ID"].map(
            lambda x: match_sample_to_patient(x, clinical_ids)
        )
        return patient_rows

    if id_col_expr not in expr_df.columns:
        raise ValueError(
            "Could not determine expression orientation. No gene column was found, "
            f"and patient column '{id_col_expr}' is absent. Available columns begin: "
            f"{list(expr_df.columns[:8])}"
        )
    print(f"Detected sample-by-gene expression matrix using ID column '{id_col_expr}'.")
    upper_to_original = {str(col).upper(): col for col in expr_df.columns}
    keep = [upper_to_original[g] for g in genes if g in upper_to_original]
    patient_rows = expr_df[[id_col_expr, *keep]].copy()
    patient_rows = patient_rows.rename(
        columns={id_col_expr: "Sample_ID", **{col: str(col).upper() for col in keep}}
    )
    patient_rows["ID_Clean"] = patient_rows["Sample_ID"].map(
        lambda x: match_sample_to_patient(x, clinical_ids)
    )
    for gene in genes:
        if gene in patient_rows:
            patient_rows[gene] = pd.to_numeric(patient_rows[gene], errors="coerce")
    return patient_rows


def infer_group_labels(contrast, group0=None, group1=None):
    if group0 and group1:
        return group0, group1
    normalized = contrast.lower()
    if "cognitive_impairment" in normalized and "nci" in normalized:
        return group0 or "NCI", group1 or "Cognitive impairment"
    if "ad" in normalized and "nci" in normalized:
        return group0 or "NCI", group1 or "AD"
    return group0 or "Group 0", group1 or "Group 1"


def binary_group(series, label0, label1):
    """Coerce common numeric, boolean, and text encodings to two labels."""
    numeric = pd.to_numeric(series, errors="coerce")
    observed_numeric = set(numeric.dropna().unique())
    if observed_numeric and observed_numeric.issubset({0, 1}):
        return numeric.map({0: label0, 1: label1})
    mapping = {
        "0": label0, "FALSE": label0, "F": label0, "NO": label0,
        "CONTROL": label0, "NCI": label0,
        "1": label1, "TRUE": label1, "T": label1, "YES": label1,
        "DISEASE": label1, "AD": label1, "COGNITIVE IMPAIRMENT": label1,
        "COGNITIVE_IMPAIRMENT": label1,
    }
    text = series.astype("string").str.strip().str.upper()
    result = text.map(mapping)
    unresolved = sorted(text[result.isna() & text.notna()].unique())
    if unresolved:
        raise ValueError(
            "Contrast must be binary (0/1, boolean, or recognized labels). "
            f"Unrecognized values: {unresolved[:8]}"
        )
    return result


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def calculate_stats(values, groups, label0, label1):
    x0 = values[groups == label0].dropna().to_numpy()
    x1 = values[groups == label1].dropna().to_numpy()
    if len(x0) == 0 or len(x1) == 0:
        return None
    test = mannwhitneyu(x1, x0, alternative="two-sided")
    rank_biserial = 2 * test.statistic / (len(x1) * len(x0)) - 1
    return {
        "p": float(test.pvalue), "n0": len(x0), "n1": len(x1),
        "median_difference": float(np.median(x1) - np.median(x0)),
        "rank_biserial": float(rank_biserial),
    }


def generate_violin(gene, values, groups, label0, label1, stats, q_value, out_path):
    plot_df = pd.DataFrame({"Expression": values, "Group": groups}).dropna()
    order = [label0, label1]
    palette = {label0: COLORS["Group 0"], label1: COLORS["Group 1"]}
    fig, ax = plt.subplots(figsize=(5.4, 5.8))
    sns.violinplot(
        data=plot_df, x="Group", y="Expression", order=order, hue="Group",
        palette=palette, legend=False, inner=None, cut=0, density_norm="width",
        linewidth=1.1, saturation=0.9, ax=ax,
    )
    sns.boxplot(
        data=plot_df, x="Group", y="Expression", order=order, width=0.18,
        showfliers=False, color="white", linewidth=1.2, ax=ax,
    )
    sns.stripplot(
        data=plot_df, x="Group", y="Expression", order=order,
        color="#222222", size=2.5, alpha=0.35, jitter=0.18, ax=ax,
    )
    ax.set_title(f"{gene} expression by cognitive status", fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("VST-normalized expression", fontsize=11)
    ax.set_xticks([0, 1], [f"{label0}\n(n = {stats['n0']})", f"{label1}\n(n = {stats['n1']})"])
    annotation = (
        f"Wilcoxon rank-sum p = {stats['p']:.2g}\n"
        f"BH-adjusted q = {q_value:.2g}\n"
        f"Median difference = {stats['median_difference']:+.2f}\n"
        f"Rank-biserial r = {stats['rank_biserial']:+.2f}"
    )
    ax.text(
        0.03, 0.97, annotation, transform=ax.transAxes, va="top", ha="left",
        fontsize=9.2, bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
                              "edgecolor": "#C7C7C7", "alpha": 0.92},
    )
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" -> Saved {gene}: {out_path}")


def make_self_test_data():
    rng = np.random.default_rng(42)
    clinical = pd.DataFrame({
        "projid": np.arange(1, 81),
        "cognitive_impairment_vs_nci": np.r_[np.zeros(40), np.ones(40)],
    })
    expression = pd.DataFrame({
        "gene_symbol": ["ABCA7", "ACTN1"],
        **{f"{i:02d}_120405": [rng.normal(8 + (i > 40) * 0.8, 0.7),
                                  rng.normal(6 - (i > 40) * 0.5, 0.8)]
           for i in range(1, 81)},
    })
    return expression, clinical


def main():
    parser = argparse.ArgumentParser(description="Plot gene expression for a binary clinical contrast.")
    parser.add_argument("--expression", help="Expression matrix CSV/TSV.")
    parser.add_argument("--clinical", help="Clinical metadata CSV/TSV.")
    parser.add_argument("--genes", help="Comma-separated genes.")
    parser.add_argument("--contrast", help="Binary clinical contrast column.")
    parser.add_argument("--id-col-expr", default="Patient_ID",
                        help="Patient ID column for sample-by-gene matrices.")
    parser.add_argument("--id-col-clin", default="Patient_ID",
                        help="Patient ID column in clinical metadata.")
    parser.add_argument("--id-col", default=None,
                        help="Shortcut for --id-col-clin and --id-col-expr.")
    parser.add_argument("--gene-col", default=None,
                        help="Gene column for gene-by-sample matrices (auto-detected by default).")
    parser.add_argument("--group0-label", default=None)
    parser.add_argument("--group1-label", default=None)
    parser.add_argument("-o", "--outdir", default="results/visualizations/violins")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        expr_df, clin_df = make_self_test_data()
        genes = ["ABCA7", "ACTN1"]
        contrast = "cognitive_impairment_vs_nci"
        id_col_clin = "projid"
    else:
        missing = [name for name in ("expression", "clinical", "genes", "contrast")
                   if getattr(args, name) is None]
        if missing:
            parser.error("Missing required arguments: " + ", ".join("--" + x for x in missing))
        sep_expr = "\t" if args.expression.lower().endswith((".tsv", ".txt")) else ","
        sep_clin = "\t" if args.clinical.lower().endswith((".tsv", ".txt")) else ","
        expr_df = pd.read_csv(args.expression, sep=sep_expr)
        clin_df = pd.read_csv(args.clinical, sep=sep_clin)
        genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
        contrast = args.contrast
        id_col_clin = args.id_col or args.id_col_clin

    id_col_expr = args.id_col or args.id_col_expr
    if id_col_clin not in clin_df.columns:
        print(f"Error: clinical patient column '{id_col_clin}' not found. Available columns begin: {list(clin_df.columns[:8])}")
        sys.exit(1)
    if contrast not in clin_df.columns:
        print(f"Error: contrast '{contrast}' not found in clinical data.")
        sys.exit(1)

    clin = clin_df[[id_col_clin, contrast]].copy()
    clin["ID_Clean"] = clin[id_col_clin].map(canonical_id)
    clin = clin.dropna(subset=["ID_Clean"])
    conflicts = clin.groupby("ID_Clean")[contrast].nunique(dropna=True)
    if (conflicts > 1).any():
        print(f"Error: conflicting clinical labels for patient IDs: {list(conflicts[conflicts > 1].index[:5])}")
        sys.exit(1)
    clin = clin.drop_duplicates("ID_Clean", keep="first")

    try:
        patient_expr = expression_to_patient_rows(
            expr_df, genes, set(clin["ID_Clean"]), id_col_expr, args.gene_col
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    present_genes = [gene for gene in genes if gene in patient_expr.columns]
    for gene in [gene for gene in genes if gene not in patient_expr.columns]:
        print(f" -> Warning: gene '{gene}' was not found; skipping.")
    if not present_genes:
        print("Error: none of the requested genes were found in the expression matrix.")
        sys.exit(1)

    duplicate_count = int(patient_expr["ID_Clean"].duplicated().sum())
    if duplicate_count:
        print(f"Warning: {duplicate_count} extra sample column(s) mapped to an existing patient; expression was averaged within patient.")
    patient_expr = patient_expr.groupby("ID_Clean", as_index=False)[present_genes].mean()
    merged = patient_expr.merge(clin[["ID_Clean", contrast]], on="ID_Clean", how="inner")
    print(f"Matched {len(merged):,} patients (expression IDs: {len(patient_expr):,}; clinical IDs: {len(clin):,}).")
    if merged.empty:
        print(f"Error: no patient IDs matched. Inspect expression sample names and clinical '{id_col_clin}' values.")
        sys.exit(1)

    label0, label1 = infer_group_labels(contrast, args.group0_label, args.group1_label)
    try:
        groups = binary_group(merged[contrast], label0, label1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    stats_by_gene = {}
    for gene in present_genes:
        stats = calculate_stats(merged[gene], groups, label0, label1)
        if stats is None:
            print(f" -> Warning: {gene} lacks observations in one group; skipping.")
        else:
            stats_by_gene[gene] = stats
    if not stats_by_gene:
        print("Error: no genes had data in both clinical groups.")
        sys.exit(1)

    q_values = bh_adjust([stats_by_gene[g]["p"] for g in stats_by_gene])
    os.makedirs(args.outdir, exist_ok=True)
    for gene, q_value in zip(stats_by_gene, q_values):
        generate_violin(
            gene, merged[gene], groups, label0, label1, stats_by_gene[gene], q_value,
            os.path.join(args.outdir, f"violin_{gene}_{contrast}.png"),
        )


if __name__ == "__main__":
    main()

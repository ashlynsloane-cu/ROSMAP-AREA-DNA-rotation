#!/usr/bin/env python3
"""
characterize_pathology_geometry_abeta_tau.py

Use the quantitative pathology variables already present in the RADC-style
ROSMAP dataset:

    amylsqrt_est_8reg
    tangsqrt_est_8reg

to characterize and calibrate the semiquantitative pathology stages:

    CERAD -> quantitative amyloid burden
    Braak -> quantitative tangle burden

This script is phenotype-only: it never reads gene expression.

It writes frozen weights into the RNA-seq metadata so downstream Weighted AREA
can run three matched-sample analyses per pathology axis:

AMYLOID:
    1. CERAD_equal
    2. CERAD_amyloid_calibrated
    3. amyloid_continuous

TAU:
    1. Braak_equal
    2. Braak_tangle_calibrated
    3. tangle_continuous

Important interpretation
------------------------
amylsqrt_est_8reg is quantitative Aβ burden, whereas CERAD specifically
summarizes neuritic plaques. Therefore the CERAD mapping should be described as
"amyloid-burden-informed" rather than a literal neuritic-plaque calibration.

Braak is a topographic NFT staging system, whereas tangsqrt_est_8reg measures
quantitative tangle density. Therefore the Braak mapping should be described as
"tangle-burden-informed" rather than a literal true distance between stages.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CERAD_TEXT = {
    "none": 0,
    "absent": 0,
    "sparse": 1,
    "moderate": 2,
    "frequent": 3,
}

BRAAK_TEXT = {
    "0": 0,
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--calibration-metadata", required=True)
    p.add_argument("--rna-metadata", required=True)

    p.add_argument("--person-id-col", default="projid")
    p.add_argument("--rna-person-id-col", default="projid")
    p.add_argument("--rna-sample-col", default="sample_id")

    p.add_argument("--cerad-col", default="ceradsc")
    p.add_argument("--braak-col", default="braaksc")
    p.add_argument("--amyloid-col", default="amylsqrt_est_8reg")
    p.add_argument("--tangle-col", default="tangsqrt_est_8reg")

    p.add_argument(
        "--cerad-convention",
        choices=["auto", "ceradsc_reverse", "ascending"],
        default="ceradsc_reverse",
    )

    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--outdir", default="results/pathology_geometry_abeta_tau")

    return p.parse_args()


def clean(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())


def parse_cerad(series, colname, convention):
    """
    Return burden-oriented ordinal:
        0 = none
        1 = sparse
        2 = moderate
        3 = frequent
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    non = series.dropna()

    txt = non.astype(str).map(clean)
    text_map = txt.map(CERAD_TEXT)

    if len(non) and text_map.notna().mean() >= 0.90:
        out.loc[non.index] = text_map.astype(float)
        return out, {"mode": "text"}

    num = pd.to_numeric(non, errors="coerce")

    if len(non) and num.notna().mean() < 0.95:
        raise ValueError(
            f"Could not parse {colname}; example values: "
            f"{non.astype(str).unique()[:20].tolist()}"
        )

    vals = np.sort(num.dropna().unique())

    if convention == "auto":
        if set(vals).issubset({1.0, 2.0, 3.0, 4.0}):
            mode = "ceradsc_reverse"
        elif set(vals).issubset({0.0, 1.0, 2.0, 3.0}):
            mode = "ascending"
        else:
            raise ValueError(
                f"Ambiguous CERAD values {vals.tolist()}; "
                "specify --cerad-convention."
            )
    else:
        mode = convention

    if mode == "ceradsc_reverse":
        ordinal = 4.0 - num
    else:
        ordinal = num - 1.0 if len(num) and num.min() >= 1 else num

    out.loc[non.index] = ordinal.to_numpy(dtype=float)

    return out, {
        "mode": mode,
        "raw_values": vals.tolist(),
    }


def parse_braak(series, colname):
    out = pd.Series(np.nan, index=series.index, dtype=float)
    non = series.dropna()

    num = pd.to_numeric(non, errors="coerce")

    if len(non) and num.notna().mean() >= 0.95:
        if num.min() >= 0 and num.max() <= 6:
            out.loc[non.index] = num.to_numpy(dtype=float)
            return out, {
                "mode": "numeric_0_to_6",
                "raw_values": np.sort(num.unique()).tolist(),
            }

    bad = []

    for idx, value in non.items():
        t = (
            clean(value)
            .replace("braak", "")
            .replace("stage", "")
            .replace("-", "")
            .strip()
        )

        if t in BRAAK_TEXT:
            out.loc[idx] = float(BRAAK_TEXT[t])
        else:
            bad.append(str(value))

    if bad:
        raise ValueError(
            f"Could not parse {colname}; example values: {bad[:20]}"
        )

    return out, {"mode": "roman_or_text"}


def collapse_person_level(df, person_col, cols):
    """
    Dataset may be longitudinal. Keep one person-level pathology record after
    confirming that nonmissing pathology values are consistent across rows.
    """
    d = df[[person_col] + cols].copy()

    conflicting = []

    for pid, g in d.groupby(person_col, dropna=False):
        for col in cols:
            # pandas >=3 no longer accepts errors="ignore" in to_numeric().
            # For duplicate-record consistency checking, canonicalize values
            # without changing their meaning. Numeric-looking values are
            # compared numerically so "1" and "1.0" are treated as identical;
            # otherwise compare trimmed strings.
            raw_vals = g[col].dropna()

            numeric_vals = pd.to_numeric(raw_vals, errors="coerce")

            if len(raw_vals) > 0 and numeric_vals.notna().all():
                vals = np.unique(numeric_vals.to_numpy(dtype=float))
            else:
                vals = (
                    raw_vals.astype(str)
                    .str.strip()
                    .replace({"": np.nan})
                    .dropna()
                    .unique()
                )

            if len(vals) > 1:
                conflicting.append((pid, col, vals[:5].tolist()))

    if conflicting:
        print("\nConflicting repeated person-level pathology values detected.")
        print("First 10 conflicts:")
        for x in conflicting[:10]:
            print(x)
        raise ValueError(
            "Do not silently collapse longitudinal rows with conflicting "
            "pathology values."
        )

    return d.groupby(person_col, as_index=False).first()


def weighted_pava(y, weights):
    """
    Weighted nondecreasing PAVA.
    Used only on stage-level quantitative pathology medians.
    """
    y = np.asarray(y, float)
    weights = np.asarray(weights, float)

    blocks = []

    for i, (yi, wi) in enumerate(zip(y, weights)):
        blocks.append([i, i, float(wi), float(yi)])

        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            b2 = blocks.pop()
            b1 = blocks.pop()

            new_w = b1[2] + b2[2]
            new_y = (b1[2] * b1[3] + b2[2] * b2[3]) / new_w

            blocks.append([
                b1[0],
                b2[1],
                new_w,
                new_y,
            ])

    fit = np.empty(len(y), float)

    for start, end, _, value in blocks:
        fit[start:end + 1] = value

    return fit


def normalize_endpoints(x):
    x = np.asarray(x, float)

    if x[-1] <= x[0]:
        raise ValueError(
            "Highest stage does not have greater fitted pathology burden "
            "than lowest stage."
        )

    return (x - x[0]) / (x[-1] - x[0])


def stage_summary(df, stage_col, quantitative_col):
    d = df[[stage_col, quantitative_col]].dropna()

    rows = []

    for stage, g in d.groupby(stage_col, sort=True):
        x = g[quantitative_col].to_numpy(float)

        rows.append({
            "stage_ordinal": float(stage),
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("stage_ordinal")
        .reset_index(drop=True)
    )


def derive_mapping(summary):
    stage = summary["stage_ordinal"].to_numpy(float)
    med = summary["median"].to_numpy(float)
    n = summary["n"].to_numpy(float)

    iso = weighted_pava(med, n)
    calibrated = normalize_endpoints(iso)

    equal = (
        (stage - stage.min())
        / (stage.max() - stage.min())
    )

    out = summary[[
        "stage_ordinal",
        "n",
        "mean",
        "median",
        "q25",
        "q75",
    ]].copy()

    out["isotonic_stage_location"] = iso
    out["equal_weight"] = equal
    out["calibrated_weight"] = calibrated

    return out


def bootstrap_mapping(
    df,
    stage_col,
    quantitative_col,
    mapping,
    reps,
    seed,
):
    rng = np.random.default_rng(seed)

    stages = mapping["stage_ordinal"].to_numpy(float)

    by_stage = {
        stage: df.loc[
            df[stage_col] == stage,
            quantitative_col,
        ].dropna().to_numpy(float)
        for stage in stages
    }

    boot = np.full((reps, len(stages)), np.nan)

    for b in range(reps):
        medians = []
        counts = []

        for stage in stages:
            x = by_stage[stage]
            xb = rng.choice(
                x,
                size=len(x),
                replace=True,
            )

            medians.append(np.median(xb))
            counts.append(len(x))

        iso = weighted_pava(medians, counts)

        if iso[-1] > iso[0]:
            boot[b, :] = normalize_endpoints(iso)

    valid = np.all(np.isfinite(boot), axis=1)
    boot = boot[valid]

    if len(boot) == 0:
        raise ValueError("All bootstrap calibrations were degenerate.")

    return pd.DataFrame({
        "stage_ordinal": stages,
        "bootstrap_valid_reps": int(len(boot)),
        "weight_boot_mean": np.mean(boot, axis=0),
        "weight_boot_median": np.median(boot, axis=0),
        "weight_ci95_low": np.quantile(boot, 0.025, axis=0),
        "weight_ci95_high": np.quantile(boot, 0.975, axis=0),
    })


def association_summary(df, stage_col, quantitative_col):
    d = df[[stage_col, quantitative_col]].dropna()

    sp = stats.spearmanr(
        d[stage_col].to_numpy(float),
        d[quantitative_col].to_numpy(float),
    )

    kt = stats.kendalltau(
        d[stage_col].to_numpy(float),
        d[quantitative_col].to_numpy(float),
    )

    return {
        "n": int(len(d)),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
    }


def make_distribution_plot(
    df,
    stage_col,
    quantitative_col,
    title,
    path,
):
    d = df[[stage_col, quantitative_col]].dropna()

    stages = sorted(d[stage_col].unique())

    groups = [
        d.loc[
            d[stage_col] == stage,
            quantitative_col,
        ].to_numpy(float)
        for stage in stages
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        groups,
        tick_labels=[
            str(int(s)) if float(s).is_integer() else str(s)
            for s in stages
        ],
    )

    ax.set_xlabel("Ordinal stage")
    ax.set_ylabel(quantitative_col)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_mapping_plot(mapping, title, path):
    x = mapping["stage_ordinal"].to_numpy(float)

    eq = mapping["equal_weight"].to_numpy(float)
    cal = mapping["calibrated_weight"].to_numpy(float)

    low = mapping["weight_ci95_low"].to_numpy(float)
    high = mapping["weight_ci95_high"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        x,
        eq,
        marker="o",
        label="Equal stage spacing",
    )

    ax.plot(
        x,
        cal,
        marker="o",
        label="Quantitative-pathology-informed",
    )

    ax.errorbar(
        x,
        cal,
        yerr=np.vstack([
            cal - low,
            high - cal,
        ]),
        fmt="none",
        capsize=3,
    )

    ax.set_xlabel("Ordinal stage")
    ax.set_ylabel("Normalized stage weight")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def map_weight(stage_series, mapping, column):
    lookup = dict(
        zip(
            mapping["stage_ordinal"].astype(float),
            mapping[column].astype(float),
        )
    )

    return stage_series.map(lookup)


def continuous_area_weight(series):
    """
    Use the supplied nonnegative quantitative pathology directly.

    No min-max shift is performed. Weighted AREA is invariant to positive
    multiplicative scaling but NOT to additive shifting.
    """
    x = pd.to_numeric(series, errors="coerce")

    if x.dropna().empty:
        return x

    if x.dropna().min() < -1e-12:
        raise ValueError(
            "Quantitative pathology contains negative values. "
            "Inspect the measurement definition before using it as AREA weights."
        )

    return x



def read_table(path):
    """Read CSV/TSV/TXT/XLSX/XLS based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)

    raise ValueError(
        f"Unsupported table extension '{suffix}' for {path}. "
        "Use CSV, TSV/TXT, XLSX, or XLS."
    )


def main():
    a = parse_args()

    root = Path(a.outdir)

    calibration_dir = root / "calibration"
    plot_dir = root / "diagnostic_plots"
    metadata_dir = root / "metadata"

    for d in [
        root,
        calibration_dir,
        plot_dir,
        metadata_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    raw = read_table(a.calibration_metadata)

    required = [
        a.person_id_col,
        a.cerad_col,
        a.braak_col,
        a.amyloid_col,
        a.tangle_col,
    ]

    missing = [
        col
        for col in required
        if col not in raw.columns
    ]

    if missing:
        print("Missing columns:", missing)
        print("\nAvailable columns:")
        print("\n".join(map(str, raw.columns)))
        raise SystemExit(2)

    # The source may be longitudinal. Collapse to unique participant pathology.
    cal = collapse_person_level(
        raw,
        a.person_id_col,
        [
            a.cerad_col,
            a.braak_col,
            a.amyloid_col,
            a.tangle_col,
        ],
    )

    cal["CERAD_stage_ordinal"], cerad_parse = parse_cerad(
        cal[a.cerad_col],
        a.cerad_col,
        a.cerad_convention,
    )

    cal["Braak_stage_ordinal"], braak_parse = parse_braak(
        cal[a.braak_col],
        a.braak_col,
    )

    cal[a.amyloid_col] = pd.to_numeric(
        cal[a.amyloid_col],
        errors="coerce",
    )

    cal[a.tangle_col] = pd.to_numeric(
        cal[a.tangle_col],
        errors="coerce",
    )

    print("=" * 80)
    print("ROSMAP QUANTITATIVE PATHOLOGY GEOMETRY")
    print("=" * 80)
    print(f"Source rows:          {len(raw):,}")
    print(f"Unique participants:  {len(cal):,}")
    print("CERAD parsing:", cerad_parse)
    print("Braak parsing:", braak_parse)

    specs = [
        (
            "CERAD_amyloid",
            "CERAD_stage_ordinal",
            a.amyloid_col,
            "CERAD vs quantitative Aβ burden",
        ),
        (
            "Braak_tangle",
            "Braak_stage_ordinal",
            a.tangle_col,
            "Braak vs quantitative tangle burden",
        ),
    ]

    mappings = {}
    association_rows = []

    for i, (
        name,
        stage_col,
        quantitative_col,
        title,
    ) in enumerate(specs):

        summary = stage_summary(
            cal,
            stage_col,
            quantitative_col,
        )

        mapping = derive_mapping(summary)

        ci = bootstrap_mapping(
            cal,
            stage_col,
            quantitative_col,
            mapping,
            reps=a.bootstrap,
            seed=a.seed + i * 1000,
        )

        mapping = mapping.merge(
            ci,
            on="stage_ordinal",
            validate="one_to_one",
        )

        assoc = association_summary(
            cal,
            stage_col,
            quantitative_col,
        )

        assoc.update({
            "comparison": name,
            "stage_col": stage_col,
            "quantitative_col": quantitative_col,
        })

        summary.to_csv(
            calibration_dir
            / f"{name}_stage_distribution_summary.csv",
            index=False,
        )

        mapping.to_csv(
            calibration_dir
            / f"{name}_calibrated_stage_mapping.csv",
            index=False,
        )

        make_distribution_plot(
            cal,
            stage_col,
            quantitative_col,
            title,
            plot_dir / f"{name}_distribution.png",
        )

        make_mapping_plot(
            mapping,
            f"{title}: equal vs burden-informed spacing",
            plot_dir / f"{name}_mapping.png",
        )

        mappings[name] = mapping
        association_rows.append(assoc)

        print("\n" + "-" * 80)
        print(name)
        print("-" * 80)
        print(mapping.to_string(index=False))
        print("\nAssociation:")
        print(assoc)

    pd.DataFrame(
        association_rows
    ).to_csv(
        calibration_dir
        / "stage_quantitative_association_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Merge pathology into RNA metadata.
    # -----------------------------------------------------------------
    rna = read_table(a.rna_metadata)

    for col in [
        a.rna_person_id_col,
        a.rna_sample_col,
    ]:
        if col not in rna.columns:
            raise ValueError(
                f"RNA metadata lacks required column '{col}'."
            )

    person = cal[[
        a.person_id_col,
        a.cerad_col,
        a.braak_col,
        a.amyloid_col,
        a.tangle_col,
        "CERAD_stage_ordinal",
        "Braak_stage_ordinal",
    ]].copy()

    if a.person_id_col != a.rna_person_id_col:
        person = person.rename(
            columns={
                a.person_id_col:
                a.rna_person_id_col
            }
        )

    # Calibration table is authoritative for the pathology variables used here.
    drop_overlap = [
        c
        for c in person.columns
        if c != a.rna_person_id_col
        and c in rna.columns
    ]

    rna_base = rna.drop(
        columns=drop_overlap,
    )

    aug = rna_base.merge(
        person,
        on=a.rna_person_id_col,
        how="left",
        validate="many_to_one",
    )

    cerad_map = mappings["CERAD_amyloid"]
    braak_map = mappings["Braak_tangle"]

    aug["CERAD_equal_weight"] = map_weight(
        aug["CERAD_stage_ordinal"],
        cerad_map,
        "equal_weight",
    )

    aug["CERAD_amyloid_calibrated_weight"] = map_weight(
        aug["CERAD_stage_ordinal"],
        cerad_map,
        "calibrated_weight",
    )

    aug["amyloid_continuous_AREA_weight"] = continuous_area_weight(
        aug[a.amyloid_col]
    )

    aug["Braak_equal_weight"] = map_weight(
        aug["Braak_stage_ordinal"],
        braak_map,
        "equal_weight",
    )

    aug["Braak_tangle_calibrated_weight"] = map_weight(
        aug["Braak_stage_ordinal"],
        braak_map,
        "calibrated_weight",
    )

    aug["tangle_continuous_AREA_weight"] = continuous_area_weight(
        aug[a.tangle_col]
    )

    # Matched sample sets for exactly fair 3-way comparisons.
    aug["CERAD_threeway_complete"] = (
        aug["CERAD_equal_weight"].notna()
        & aug["CERAD_amyloid_calibrated_weight"].notna()
        & aug["amyloid_continuous_AREA_weight"].notna()
    )

    aug["Braak_threeway_complete"] = (
        aug["Braak_equal_weight"].notna()
        & aug["Braak_tangle_calibrated_weight"].notna()
        & aug["tangle_continuous_AREA_weight"].notna()
    )

    out_metadata = (
        metadata_dir
        / "ROSMAP_RNAseq_metadata_abeta_tau_threeway.csv"
    )

    aug.to_csv(
        out_metadata,
        index=False,
    )

    with open(
        root / "pathology_geometry_manifest.json",
        "w",
    ) as f:
        json.dump(
            {
                "calibration_source": str(
                    Path(
                        a.calibration_metadata
                    ).resolve()
                ),
                "rna_metadata": str(
                    Path(
                        a.rna_metadata
                    ).resolve()
                ),
                "source_rows": int(len(raw)),
                "unique_calibration_participants": int(len(cal)),
                "cerad_parse": cerad_parse,
                "braak_parse": braak_parse,
                "amyloid_variable": a.amyloid_col,
                "tangle_variable": a.tangle_col,
                "bootstrap": a.bootstrap,
                "seed": a.seed,
                "RNA_threeway_complete_counts": {
                    "CERAD_amyloid": int(
                        aug[
                            "CERAD_threeway_complete"
                        ].sum()
                    ),
                    "Braak_tangle": int(
                        aug[
                            "Braak_threeway_complete"
                        ].sum()
                    ),
                },
                "interpretation": {
                    "CERAD_mapping": (
                        "Amyloid-burden-informed CERAD spacing; "
                        "not a literal neuritic-plaque calibration."
                    ),
                    "Braak_mapping": (
                        "Tangle-burden-informed Braak spacing; "
                        "Braak remains a topographic staging construct."
                    ),
                },
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("Augmented RNA metadata:")
    print(out_metadata)
    print(
        "CERAD/amyloid matched RNA n:",
        int(
            aug[
                "CERAD_threeway_complete"
            ].sum()
        ),
    )
    print(
        "Braak/tangle matched RNA n:",
        int(
            aug[
                "Braak_threeway_complete"
            ].sum()
        ),
    )
    print(
        "\nInspect the two stage-mapping CSVs and plots before running AREA."
    )


if __name__ == "__main__":
    main()

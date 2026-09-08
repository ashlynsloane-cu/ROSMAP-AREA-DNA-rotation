#!/usr/bin/env python3
"""
characterize_pathology_geometry.py

Phenotype-only calibration for ROSMAP pathology.

This script does NOT read gene expression. It:
  1) characterizes quantitative pathology within CERAD/Braak stages;
  2) derives pathology-informed stage weights from an external/full pathology cohort;
  3) bootstraps uncertainty in those weights;
  4) writes RNA metadata augmented with frozen weights for downstream AREA.

CERAD is calibrated against plaq_n.
Braak is calibrated independently against nft and tangsqrt_est_8reg.

The monotone PAVA step is applied only to stage-level pathology medians. It is
not a per-gene adaptive model and therefore does not use expression to choose
weights.
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
    "none": 0, "absent": 0,
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


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration-metadata", required=True)
    p.add_argument("--rna-metadata", required=True)

    p.add_argument("--person-id-col", default="projid")
    p.add_argument("--rna-person-id-col", default="projid")
    p.add_argument("--rna-sample-col", default="sample_id")

    p.add_argument("--cerad-col", default="ceradsc")
    p.add_argument("--braak-col", default="braaksc")
    p.add_argument("--plaq-col", default="plaq_n")
    p.add_argument("--nft-col", default="nft")
    p.add_argument("--tangle-col", default="tangsqrt_est_8reg")

    p.add_argument(
        "--cerad-convention",
        choices=["auto", "ceradsc_reverse", "ascending"],
        default="auto",
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--outdir", default="results/pathology_geometry")
    return p.parse_args()


def clean(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())


def parse_cerad(series, colname, convention):
    out = pd.Series(np.nan, index=series.index, dtype=float)
    non = series.dropna()
    if non.empty:
        return out, {"mode": "empty"}

    txt = non.astype(str).map(clean)
    mapped = txt.map(CERAD_TEXT)
    if mapped.notna().mean() >= 0.90:
        out.loc[non.index] = mapped.astype(float)
        return out, {"mode": "text"}

    num = pd.to_numeric(non, errors="coerce")
    if num.notna().mean() < 0.95:
        raise ValueError(
            f"Could not parse CERAD column {colname}. "
            f"Values include: {non.astype(str).unique()[:20].tolist()}"
        )

    vals = np.sort(num.unique())

    normalized = np.array([0.0, 1/3, 2/3, 1.0])
    if len(vals) <= 4 and all(np.min(np.abs(normalized - v)) < 1e-6 for v in vals):
        out.loc[non.index] = np.rint(num.to_numpy() * 3)
        return out, {"mode": "already_normalized_burden", "raw_values": vals.tolist()}

    mode = convention
    if mode == "auto":
        lname = colname.lower()
        if set(vals).issubset({1.0, 2.0, 3.0, 4.0}) and (
            "ceradsc" in lname or lname == "cerad"
        ):
            mode = "ceradsc_reverse"
        elif set(vals).issubset({0.0, 1.0, 2.0, 3.0}):
            mode = "ascending"
        else:
            raise ValueError(
                f"Ambiguous CERAD coding in {colname}: {vals.tolist()}. "
                "Specify --cerad-convention."
            )

    if mode == "ceradsc_reverse":
        # Current project convention: 1=frequent/high burden, 4=none/low burden.
        ordinal = 4.0 - num
    else:
        ordinal = num - 1.0 if num.min() >= 1 and num.max() <= 4 else num

    out.loc[non.index] = ordinal.to_numpy(dtype=float)
    return out, {"mode": mode, "raw_values": vals.tolist()}


def parse_braak(series, colname):
    out = pd.Series(np.nan, index=series.index, dtype=float)
    non = series.dropna()
    if non.empty:
        return out, {"mode": "empty"}

    num = pd.to_numeric(non, errors="coerce")
    if num.notna().mean() >= 0.95 and num.min() >= 0 and num.max() <= 6:
        out.loc[non.index] = num.to_numpy(dtype=float)
        return out, {"mode": "numeric_0_to_6", "raw_values": np.sort(num.unique()).tolist()}

    bad = []
    for idx, val in non.items():
        t = clean(val).replace("braak", "").replace("stage", "").replace("-", "").strip()
        if t in BRAAK_TEXT:
            out.loc[idx] = float(BRAAK_TEXT[t])
        else:
            bad.append(str(val))
    if bad:
        raise ValueError(f"Could not parse Braak column {colname}: {bad[:20]}")
    return out, {"mode": "roman_or_text"}


def pava(y, w):
    """Weighted nondecreasing pool-adjacent-violators algorithm."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    blocks = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        blocks.append([i, i, wi, yi])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            nw = b1[2] + b2[2]
            nv = (b1[2] * b1[3] + b2[2] * b2[3]) / nw
            blocks.append([b1[0], b2[1], nw, nv])

    fit = np.empty(len(y), float)
    for start, end, _, value in blocks:
        fit[start:end + 1] = value
    return fit


def normalize_endpoints(x):
    x = np.asarray(x, float)
    if x[-1] <= x[0]:
        raise ValueError("Calibrated pathology does not increase from lowest to highest stage.")
    return (x - x[0]) / (x[-1] - x[0])


def summarize(df, stage_col, quant_col):
    d = df[[stage_col, quant_col]].dropna()
    rows = []
    for stage, g in d.groupby(stage_col, sort=True):
        x = g[quant_col].to_numpy(float)
        rows.append({
            "stage_ordinal": float(stage),
            "n": len(x),
            "mean": np.mean(x),
            "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
            "median": np.median(x),
            "q25": np.quantile(x, .25),
            "q75": np.quantile(x, .75),
            "min": np.min(x),
            "max": np.max(x),
        })
    return pd.DataFrame(rows).sort_values("stage_ordinal").reset_index(drop=True)


def derive_mapping(summary):
    stage = summary["stage_ordinal"].to_numpy(float)
    med = summary["median"].to_numpy(float)
    n = summary["n"].to_numpy(float)

    iso = pava(med, n)
    calibrated = normalize_endpoints(iso)
    equal = (stage - stage.min()) / (stage.max() - stage.min())

    out = summary[["stage_ordinal", "n"]].copy()
    out["raw_stage_median"] = med
    out["isotonic_stage_location"] = iso
    out["equal_weight"] = equal
    out["calibrated_weight"] = calibrated
    return out


def bootstrap_mapping(df, stage_col, quant_col, mapping, reps, seed):
    rng = np.random.default_rng(seed)
    stages = mapping["stage_ordinal"].to_numpy(float)
    vals = {
        s: df.loc[df[stage_col] == s, quant_col].dropna().to_numpy(float)
        for s in stages
    }

    boot = np.full((reps, len(stages)), np.nan)
    for b in range(reps):
        meds, counts = [], []
        for s in stages:
            x = vals[s]
            xb = rng.choice(x, len(x), replace=True)
            meds.append(np.median(xb))
            counts.append(len(x))
        iso = pava(meds, counts)
        if iso[-1] > iso[0]:
            boot[b] = normalize_endpoints(iso)

    boot = boot[np.all(np.isfinite(boot), axis=1)]
    if len(boot) == 0:
        raise ValueError("All bootstrap mappings were degenerate.")

    return pd.DataFrame({
        "stage_ordinal": stages,
        "bootstrap_valid_reps": len(boot),
        "weight_boot_mean": np.mean(boot, axis=0),
        "weight_boot_median": np.median(boot, axis=0),
        "weight_ci95_low": np.quantile(boot, .025, axis=0),
        "weight_ci95_high": np.quantile(boot, .975, axis=0),
    })


def assoc(df, stage_col, quant_col):
    d = df[[stage_col, quant_col]].dropna()
    sp = stats.spearmanr(d[stage_col], d[quant_col])
    kt = stats.kendalltau(d[stage_col], d[quant_col])
    return {
        "n": len(d),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
    }


def boxplot(df, stage_col, quant_col, title, ylabel, path):
    d = df[[stage_col, quant_col]].dropna()
    stages = sorted(d[stage_col].unique())
    groups = [d.loc[d[stage_col] == s, quant_col].to_numpy(float) for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=[str(int(s)) if float(s).is_integer() else str(s) for s in stages])
    ax.set_xlabel("Ordinal stage")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def mapping_plot(mapping, title, path):
    x = mapping["stage_ordinal"].to_numpy(float)
    eq = mapping["equal_weight"].to_numpy(float)
    cal = mapping["calibrated_weight"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, eq, marker="o", label="Equal spacing")
    ax.plot(x, cal, marker="o", label="Pathology-calibrated")
    ax.errorbar(
        x, cal,
        yerr=np.vstack([
            cal - mapping["weight_ci95_low"].to_numpy(float),
            mapping["weight_ci95_high"].to_numpy(float) - cal,
        ]),
        fmt="none", capsize=3,
    )
    ax.set_xlabel("Ordinal stage")
    ax.set_ylabel("Normalized stage weight")
    ax.set_ylim(-.05, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dedupe_people(df, id_col, cols):
    d = df[[id_col] + cols].copy()
    if not d[id_col].duplicated().any():
        return d

    bad = []
    for pid, g in d.groupby(id_col, dropna=False):
        for c in cols:
            if len(g[c].dropna().astype(str).unique()) > 1:
                bad.append(pid)
                break
    if bad:
        raise ValueError(f"Conflicting duplicate pathology rows. Example projids: {bad[:10]}")
    return d.groupby(id_col, as_index=False).first()


def weight_lookup(stage_series, mapping, column):
    lookup = dict(zip(mapping["stage_ordinal"].astype(float), mapping[column].astype(float)))
    return stage_series.map(lookup)


def continuous_weight(rna_series, calibration_series):
    """
    Positive multiplicative scaling only. This leaves Weighted AREA unchanged
    because AREA divides by the total weight.
    """
    cal = pd.to_numeric(calibration_series, errors="coerce").dropna().astype(float)
    if len(cal) == 0:
        raise ValueError("No quantitative pathology values.")
    if cal.min() < -1e-12:
        raise ValueError("Negative pathology values found; inspect scale before AREA.")
    scale = float(cal.max())
    if scale <= 0:
        raise ValueError("Quantitative pathology maximum must be positive.")
    return pd.to_numeric(rna_series, errors="coerce") / scale, scale


def main():
    a = args()
    root = Path(a.outdir)
    cdir = root / "calibration"
    pdir = root / "diagnostic_plots"
    mdir = root / "metadata"
    for d in [root, cdir, pdir, mdir]:
        d.mkdir(parents=True, exist_ok=True)

    cal = pd.read_csv(a.calibration_metadata)
    rna = pd.read_csv(a.rna_metadata)

    needed_cal = [
        a.person_id_col, a.cerad_col, a.braak_col,
        a.plaq_col, a.nft_col, a.tangle_col,
    ]
    missing = [c for c in needed_cal if c not in cal.columns]
    if missing:
        print("Missing calibration columns:", missing)
        print("Available columns:")
        print("\n".join(map(str, cal.columns)))
        raise SystemExit(2)

    for c in [a.rna_person_id_col, a.rna_sample_col]:
        if c not in rna.columns:
            raise ValueError(f"RNA metadata lacks {c}")

    cal["CERAD_stage_ordinal"], cerad_parse = parse_cerad(
        cal[a.cerad_col], a.cerad_col, a.cerad_convention
    )
    cal["Braak_stage_ordinal"], braak_parse = parse_braak(
        cal[a.braak_col], a.braak_col
    )

    comparisons = [
        ("CERAD_plaq_n", "CERAD_stage_ordinal", a.plaq_col),
        ("Braak_nft", "Braak_stage_ordinal", a.nft_col),
        ("Braak_tangle", "Braak_stage_ordinal", a.tangle_col),
    ]

    mappings = {}
    assoc_rows = []

    print("=" * 80)
    print("PATHOLOGY GEOMETRY")
    print("=" * 80)
    print("CERAD parsing:", cerad_parse)
    print("Braak parsing:", braak_parse)

    for i, (name, stage_col, quant_col) in enumerate(comparisons):
        s = summarize(cal, stage_col, quant_col)
        m = derive_mapping(s)
        ci = bootstrap_mapping(
            cal, stage_col, quant_col, m, a.bootstrap, a.seed + 1000 * i
        )
        m = m.merge(ci, on="stage_ordinal", validate="one_to_one")

        s.to_csv(cdir / f"{name}_stage_distribution_summary.csv", index=False)
        m.to_csv(cdir / f"{name}_calibrated_stage_mapping.csv", index=False)

        ar = assoc(cal, stage_col, quant_col)
        ar.update({"comparison": name, "quantitative_col": quant_col})
        assoc_rows.append(ar)
        mappings[name] = m

        boxplot(
            cal, stage_col, quant_col,
            f"{name}: quantitative pathology by stage",
            quant_col,
            pdir / f"{name}_distribution.png",
        )
        mapping_plot(
            m,
            f"{name}: equal vs pathology-calibrated spacing",
            pdir / f"{name}_weight_mapping.png",
        )

        print("\n", name)
        print(m.to_string(index=False))
        print(ar)

    pd.DataFrame(assoc_rows).to_csv(
        cdir / "stage_quantitative_association_summary.csv", index=False
    )

    # Merge frozen pathology information into RNA metadata.
    person = dedupe_people(
        cal, a.person_id_col,
        [
            a.cerad_col, a.braak_col, a.plaq_col, a.nft_col, a.tangle_col,
            "CERAD_stage_ordinal", "Braak_stage_ordinal",
        ],
    )

    if a.person_id_col != a.rna_person_id_col:
        person = person.rename(columns={a.person_id_col: a.rna_person_id_col})

    # Drop overlapping pathology fields from RNA metadata before merge so the
    # calibration table is the single source of truth for these new analyses.
    pathology_cols = [
        a.cerad_col, a.braak_col, a.plaq_col, a.nft_col, a.tangle_col,
        "CERAD_stage_ordinal", "Braak_stage_ordinal",
    ]
    rna_base = rna.drop(columns=[c for c in pathology_cols if c in rna.columns])

    aug = rna_base.merge(
        person,
        on=a.rna_person_id_col,
        how="left",
        validate="many_to_one",
    )

    cerad_map = mappings["CERAD_plaq_n"]
    bnft_map = mappings["Braak_nft"]
    btangle_map = mappings["Braak_tangle"]

    aug["CERAD_equal_weight"] = weight_lookup(
        aug["CERAD_stage_ordinal"], cerad_map, "equal_weight"
    )
    aug["CERAD_plaqn_calibrated_weight"] = weight_lookup(
        aug["CERAD_stage_ordinal"], cerad_map, "calibrated_weight"
    )

    aug["Braak_equal_weight"] = weight_lookup(
        aug["Braak_stage_ordinal"], bnft_map, "equal_weight"
    )
    aug["Braak_nft_calibrated_weight"] = weight_lookup(
        aug["Braak_stage_ordinal"], bnft_map, "calibrated_weight"
    )
    aug["Braak_tangle_calibrated_weight"] = weight_lookup(
        aug["Braak_stage_ordinal"], btangle_map, "calibrated_weight"
    )

    aug["plaq_n_AREA_weight"], plaq_scale = continuous_weight(
        aug[a.plaq_col], cal[a.plaq_col]
    )
    aug["nft_AREA_weight"], nft_scale = continuous_weight(
        aug[a.nft_col], cal[a.nft_col]
    )
    aug["tangle_AREA_weight"], tangle_scale = continuous_weight(
        aug[a.tangle_col], cal[a.tangle_col]
    )

    # Identical samples within each parallel comparison.
    aug["CERAD_parallel_complete"] = (
        aug["CERAD_equal_weight"].notna()
        & aug["CERAD_plaqn_calibrated_weight"].notna()
        & aug["plaq_n_AREA_weight"].notna()
    )

    aug["Braak_parallel_complete"] = (
        aug["Braak_equal_weight"].notna()
        & aug["Braak_nft_calibrated_weight"].notna()
        & aug["Braak_tangle_calibrated_weight"].notna()
        & aug["nft_AREA_weight"].notna()
        & aug["tangle_AREA_weight"].notna()
    )

    outmeta = mdir / "ROSMAP_RNAseq_metadata_with_pathology_area_weights.csv"
    aug.to_csv(outmeta, index=False)

    manifest = {
        "calibration_metadata": str(Path(a.calibration_metadata).resolve()),
        "rna_metadata": str(Path(a.rna_metadata).resolve()),
        "cerad_parse": cerad_parse,
        "braak_parse": braak_parse,
        "bootstrap": a.bootstrap,
        "seed": a.seed,
        "continuous_scaling_divisors": {
            "plaq_n": plaq_scale,
            "nft": nft_scale,
            "tangle": tangle_scale,
        },
        "complete_case_counts": {
            "CERAD_parallel_complete": int(aug["CERAD_parallel_complete"].sum()),
            "Braak_parallel_complete": int(aug["Braak_parallel_complete"].sum()),
        },
        "note": (
            "Stage mappings were estimated from quantitative pathology only. "
            "Gene expression was not used to estimate weights."
        ),
    }
    with open(root / "pathology_geometry_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Augmented metadata:", outmeta)
    print("CERAD matched RNA n =", int(aug["CERAD_parallel_complete"].sum()))
    print("Braak+nft+tangle matched RNA n =", int(aug["Braak_parallel_complete"].sum()))
    print("Inspect calibration CSVs and diagnostic plots before AREA.")


if __name__ == "__main__":
    main()

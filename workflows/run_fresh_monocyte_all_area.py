#!/usr/bin/env python3
"""Run six fresh, covariate-adjusted monocyte AREA discovery/pathology comparisons.

Inputs come solely from an authorized expression matrix and freshly built
manifest index. No historical task CSV, model result, or reference is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

CONTRASTS = (("cogdx", "NCI_vs_AD", "Monocyte discovery"),
             ("cogdx", "NCI_vs_MCI", "Monocyte discovery"),
             ("cogdx", "MCI_vs_AD", "Monocyte discovery"),
             ("Braak", "0-2_vs_3-4", "Monocyte pathology subset"),
             ("Braak", "3-4_vs_5-6", "Monocyte pathology subset"),
             ("CERAD", "1-2_vs_3-4", "Monocyte pathology subset"))
COVARIATES_CONT = "age_sampling_capped,age_sampling_90plus"
COVARIATES_CAT = "sex,sequencingBatch"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_rows(index):
    needed = {"axis", "contrast", "cohort", "role", "modality", "manifest", "n", "group_A", "group_B"}
    missing = needed.difference(index.columns)
    if missing:
        raise ValueError(f"Missing manifest index columns: {sorted(missing)}")
    keys = set(CONTRASTS)
    # Cognitive discovery comparisons must be primary. The pathology subset
    # is a designated cohort but may carry a non-primary role in the study plan.
    # Require its exact axis/contrast/cohort below and fail on duplicates.
    mask = index["modality"].eq("monocyte") & (
        index["axis"].isin(["Braak", "CERAD"])
        | (index["axis"].eq("cogdx") & index["role"].eq("primary"))
    )
    rows = index.loc[mask].copy()
    rows = rows.loc[rows.apply(
        lambda r: (str(r["axis"]), str(r["contrast"]), str(r["cohort"])) in keys,
        axis=1,
    )]
    actual = [(str(r["axis"]), str(r["contrast"]), str(r["cohort"]))
              for r in rows.to_dict("records")]
    if len(actual) != len(keys) or set(actual) != keys:
        missing_keys = sorted(keys - set(actual))
        duplicates = sorted(k for k in set(actual) if actual.count(k) > 1)
        raise ValueError(f"Expected six unique primary monocyte manifests; "
                         f"missing={missing_keys}; duplicates={duplicates}")
    rows["_run_order"] = rows.apply(
        lambda r: CONTRASTS.index((str(r["axis"]), str(r["contrast"]),
                                   str(r["cohort"]))), axis=1,
    )
    return rows.sort_values("_run_order")



def normalize_group_label(label, axis):
    """Normalize display punctuation/prefixes, never change model data labels."""
    value = str(label).strip()
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        value = value.replace(dash, "-")
    if axis in ("Braak", "CERAD"):
        # Index display labels include "CERAD 1–2"; manifests may use
        # "CERAD_1-2". Strip only the known axis prefix.
        for separator in ("_", " "):
            prefix = axis + separator
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
                break
    return value


def resolve_groups(row, manifest):
    """Validate semantic groups and return the *actual* manifest labels.

    Index display labels may use Unicode en dashes (0–2), ASCII hyphens
    (0-2), or pathology prefixes. Do not rewrite analysis_group values.
    """
    axis = str(row["axis"])
    contrast = str(row["contrast"])
    expected_a, expected_b = contrast.split("_vs_", 1)
    expected = (normalize_group_label(expected_a, axis),
                normalize_group_label(expected_b, axis))

    index_labels = (normalize_group_label(row["group_A"], axis),
                    normalize_group_label(row["group_B"], axis))
    if index_labels != expected:
        raise ValueError(
            f"Manifest index contrast labels conflict with {axis} {contrast}: "
            f"A={row['group_A']!r} B={row['group_B']!r}")

    observed = manifest["analysis_group"].dropna().astype(str).unique().tolist()
    groups = {}
    for label in observed:
        normalized = normalize_group_label(label, axis)
        if normalized in groups and groups[normalized] != label:
            raise ValueError(f"Ambiguous manifest labels: {groups[normalized]!r}, {label!r}")
        groups[normalized] = label
    if set(groups) != set(expected) or len(groups) != 2:
        raise ValueError(
            f"Group mismatch in manifest: {axis} {contrast}: "
            f"expected={list(expected)} observed={observed}")
    return groups[expected[0]], groups[expected[1]]


def run(args):
    index_path = args.manifests_root / "manifest_index.csv"
    index = pd.read_csv(index_path)
    rows = select_rows(index)
    if not args.expression.is_file():
        raise FileNotFoundError(args.expression)
    if not args.area_script.is_file():
        raise FileNotFoundError(args.area_script)
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows.to_dict("records"):
        contrast = str(row["contrast"])
        # Ignore absolute manifest paths encoded in the index, which can be
        # machine-specific; resolve by the original relative suffix instead.
        manifest_record = Path(str(row["manifest"]))
        try:
            relative = manifest_record.relative_to(args.manifests_root)
        except ValueError:
            # The index is generated under manifests_root by the upstream producer.
            # Do not accept an arbitrary external absolute path.
            raise ValueError(f"Manifest path escapes output root: {manifest_record}")
        manifest = (args.manifests_root / relative).resolve()
        if not manifest.is_relative_to(args.manifests_root.resolve()):
            raise ValueError("Manifest symlink resolves outside generated manifest root")
        tab = pd.read_csv(manifest)
        required = ["sample_id", "analysis_group", "age_sampling_capped", "age_sampling_90plus", "sex", "sequencingBatch"]
        if any(x not in tab.columns for x in required):
            raise ValueError(f"Missing manifest model covariates: {manifest}")
        if len(tab) != int(row["n"]):
            raise ValueError(f"Manifest size mismatch: {contrast}")
        if tab[required].isna().any().any():
            raise ValueError(f"Missing model inputs: {contrast}")
        group_a, group_b = resolve_groups(row, tab)
        destination = args.outdir / str(row["axis"]) / contrast
        cmd = [sys.executable, str(args.area_script),
               "--expression", str(args.expression),
               "--manifest", str(manifest),
               "--outdir", str(destination),
               "--contrast-name", f"monocyte_{row['axis']}_{contrast}",
               "--group-a", group_a,
               "--group-b", group_b,
               "--orientation", "features_by_samples",
               "--feature-id-col", "Geneid",
               "--continuous", COVARIATES_CONT,
               "--categorical", COVARIATES_CAT]
        print(f"FRESH AREA {contrast}: n={len(tab)}", flush=True)
        subprocess.run(cmd, check=True)
        result = destination / f"monocyte_{row['axis']}_{contrast}_Regular_AREA_all_results.csv"
        if not result.is_file() or not (destination / "run_manifest.json").is_file():
            raise RuntimeError(f"AREA outputs missing: {contrast}")
        records.append({"axis": str(row["axis"]), "contrast": contrast, "n": len(tab),
                        "manifest_sha256": sha256(manifest),
                        "result_path": str(result), "result_sha256": sha256(result)})
    report = {"status": "fresh_models_executed", "source": "authorized_expression_and_generated_manifests",
              "expression_sha256": sha256(args.expression), "models": records,
              "scope": "six adjusted primary monocyte contrasts; not entire multimodal AREA stage"}
    (args.outdir / "fresh_area_models.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FRESH_MONOCYTE_MODELS_COMPLETE:", len(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-root", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--area-script", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

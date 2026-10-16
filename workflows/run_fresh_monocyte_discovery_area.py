#!/usr/bin/env python3
"""Run fresh, covariate-adjusted monocyte AREA discovery comparisons.

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

CONTRASTS = ("NCI_vs_AD", "NCI_vs_MCI", "MCI_vs_AD")
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
    rows = index.loc[
        index["axis"].eq("cogdx")
        & index["contrast"].isin(CONTRASTS)
        & index["cohort"].eq("Monocyte discovery")
        & index["role"].eq("primary")
        & index["modality"].eq("monocyte")
    ]
    if len(rows) != len(CONTRASTS) or set(rows["contrast"]) != set(CONTRASTS):
        raise ValueError("Expected one primary monocyte discovery manifest per cognitive contrast")
    return rows.sort_values("contrast")


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
        if set(tab["analysis_group"].astype(str)) != {str(row["group_A"]), str(row["group_B"])}:
            raise ValueError(f"Group mismatch in manifest: {contrast}")
        destination = args.outdir / contrast
        cmd = [sys.executable, str(args.area_script),
               "--expression", str(args.expression),
               "--manifest", str(manifest),
               "--outdir", str(destination),
               "--contrast-name", f"monocyte_{contrast}",
               "--group-a", str(row["group_A"]),
               "--group-b", str(row["group_B"]),
               "--orientation", "features_by_samples",
               "--feature-id-col", "Geneid",
               "--continuous", COVARIATES_CONT,
               "--categorical", COVARIATES_CAT]
        print(f"FRESH AREA {contrast}: n={len(tab)}", flush=True)
        subprocess.run(cmd, check=True)
        result = destination / f"monocyte_{contrast}_Regular_AREA_all_results.csv"
        if not result.is_file() or not (destination / "run_manifest.json").is_file():
            raise RuntimeError(f"AREA outputs missing: {contrast}")
        records.append({"contrast": contrast, "n": len(tab),
                        "manifest_sha256": sha256(manifest),
                        "result_path": str(result), "result_sha256": sha256(result)})
    report = {"status": "fresh_models_executed", "source": "authorized_expression_and_generated_manifests",
              "expression_sha256": sha256(args.expression), "models": records,
              "scope": "covariate-adjusted monocyte cognitive discovery AREA only; not entire model stage"}
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

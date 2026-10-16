#!/usr/bin/env python3
"""Fresh six-contrast DLPFC/TMT AREA batch from newly generated manifests.

No archived model outputs or reference task tables are read. Specifications mirror
Stage11's frozen methodology; see prepare_stage11_covariate_stress.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unicodedata

import pandas as pd

CONTRASTS = (("cogdx", "NCI_vs_AD"), ("cogdx", "NCI_vs_MCI"),
             ("cogdx", "MCI_vs_AD"), ("Braak", "0-2_vs_3-4"),
             ("Braak", "3-4_vs_5-6"), ("CERAD", "1-2_vs_3-4"))
SPECS = {
    "dlpfc": {"cohort": "DLPFC discovery", "orientation": "samples_by_features",
              "continuous": "age_numeric,rin_numeric,pmi_numeric",
              "categorical": "sex,sequencing_batch", "global_complete": False},
    "tmt": {"cohort": "TMT discovery", "orientation": "features_by_samples",
            "continuous": "age_numeric,pmi_numeric", "categorical": "msex,Batch",
            "global_complete": True},
}


def canonical(s):
    """Normalize typographical dashes and optional CERAD label prefix."""
    x = unicodedata.normalize("NFKC", str(s)).strip()
    for dash in "\u2010\u2011\u2012\u2013\u2014\u2212":
        x = x.replace(dash, "-")
    x = x.replace("_", " ")
    if x.lower().startswith("cerad "):
        x = x[6:]
    return x.strip()


def groups(axis, contrast, observed):
    a, b = contrast.split("_vs_", 1)
    expected = {canonical(a), canonical(b)}
    mapping = {}
    for value in observed:
        key = canonical(value)
        if key in mapping:
            raise ValueError("Ambiguous equivalent group labels")
        mapping[key] = str(value)
    if set(mapping) != expected:
        raise ValueError(f"Unexpected groups in {axis}/{contrast}: {sorted(observed)}")
    # Reproduce historical Stage 11 CERAD orientation: 3-4 = reference group A.
    if axis == "CERAD":
        a, b = b, a
    return mapping[canonical(a)], mapping[canonical(b)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select(index, modality, axis, contrast):
    """Select the historical primary discovery specification, never by row order.

    Cognitive contrasts contain both PURE primary and BROAD sensitivity entries.
    Only the PURE definition corresponds to the frozen Stage 11 primary analysis.
    """
    cohort = SPECS[modality]["cohort"]
    selection = index.loc[
        (index["modality"] == modality)
        & (index["cohort"] == cohort)
        & (index["axis"] == axis)
        & (index["contrast"] == contrast)
    ]
    if axis == "cogdx":
        needed = {"role", "phenotype_definition"}
        if not needed.issubset(index.columns):
            raise ValueError("Cognitive selection needs role and phenotype_definition")
        selection = selection.loc[
            selection["role"].astype(str).str.strip().eq("primary")
            & selection["phenotype_definition"].astype(str)
                .str.strip().str.startswith("PURE:")
        ]
    if len(selection) != 1:
        detail = selection[["role", "phenotype_definition"]].to_dict("records") if {"role", "phenotype_definition"}.issubset(selection.columns) else []
        raise ValueError(
            f"Expected exactly one primary fresh manifest for "
            f"{modality} {axis} {contrast}; got {len(selection)}; matches={detail}"
        )
    return selection.iloc[0]


def run(args):
    root = args.manifests_root.resolve(strict=True)
    index_file = root / "manifest_index.csv"
    index = pd.read_csv(index_file)
    required = {"axis", "contrast", "cohort", "modality", "manifest", "n"}
    if required - set(index.columns):
        raise ValueError(f"Manifest index missing: {sorted(required - set(index.columns))}")
    script = args.area_script.resolve(strict=True)
    expression = {"dlpfc": args.rna_expression.resolve(strict=True),
                  "tmt": args.tmt_expression.resolve(strict=True)}
    jobs = []
    for modality in ("dlpfc", "tmt"):
        if args.modality != "both" and modality != args.modality:
            continue
        for axis, contrast in CONTRASTS:
            row = select(index, modality, axis, contrast)
            manifest = Path(str(row.manifest)).resolve(strict=True)
            if not manifest.is_relative_to(root):
                raise ValueError(f"Manifest escapes fresh manifest root: {manifest}")
            meta = pd.read_csv(manifest)
            spec = SPECS[modality]
            required_cols = ["sample_id", "analysis_group"] + spec["continuous"].split(",") + spec["categorical"].split(",")
            missing = sorted(set(required_cols) - set(meta.columns))
            if missing:
                raise ValueError(f"{modality} {axis} {contrast} missing covariates: {missing}")
            if len(meta) != int(row.n) or meta.sample_id.duplicated().any() or meta[required_cols].isna().any().any():
                raise ValueError(f"Invalid cohort/covariates for {modality} {axis} {contrast}")
            a, b = groups(axis, contrast, meta.analysis_group.astype(str).unique())
            destination = args.outdir / modality / axis / contrast
            name = f"{modality}_{axis}_{contrast}"
            command = [sys.executable, str(script), "--expression", str(expression[modality]),
                       "--manifest", str(manifest), "--outdir", str(destination),
                       "--contrast-name", name, "--group-a", a, "--group-b", b,
                       "--orientation", spec["orientation"], "--continuous", spec["continuous"],
                       "--categorical", spec["categorical"]]
            if spec["global_complete"]:
                command.append("--global-complete-features")
            jobs.append((modality, axis, contrast, manifest, destination, name, command, len(meta)))
    if not jobs:
        raise ValueError("No selected jobs")
    print(f"Fresh brain AREA plan: {len(jobs)} models", flush=True)
    for modality, axis, contrast, manifest, dest, name, cmd, n in jobs:
        print(f"{modality} {axis} {contrast}: n={n}", flush=True)
    if not args.execute:
        print("PREFLIGHT_ONLY: run with --execute to fit the models")
        return
    records = []
    for modality, axis, contrast, manifest, dest, name, cmd, n in jobs:
        print(f"FITTING {name}: n={n}", flush=True)
        subprocess.run(cmd, check=True)
        result = dest / f"{name}_Regular_AREA_all_results.csv"
        model_manifest = dest / "run_manifest.json"
        if not result.is_file() or not model_manifest.is_file():
            raise RuntimeError(f"Missing fresh result: {name}")
        records.append({"model": name, "n": n, "result": str(result),
                        "result_sha256": sha256(result), "manifest_sha256": sha256(manifest)})
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {"status": "fresh_brain_area_complete", "models": records,
              "area_script_sha256": sha256(script), "expression_sha256":
              {modality: sha256(expression[modality]) for modality in {j[0] for j in jobs}}}
    (args.outdir / "fresh_brain_area_models.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"FRESH_BRAIN_AREA_COMPLETE: {len(records)}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifests-root", "rna-expression", "tmt-expression", "area-script", "outdir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--modality", choices=("both", "dlpfc", "tmt"), default="both")
    p.add_argument("--execute", action="store_true", help="Fit models; default preflight only")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()

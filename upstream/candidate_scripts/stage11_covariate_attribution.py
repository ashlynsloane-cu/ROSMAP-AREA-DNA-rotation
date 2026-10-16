#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path.home() / "rosmap_multiomic_slurm"
S11 = ROOT / "results/stage11_covariate_stress"
BASE = S11 / "stage11_tasks.tsv"
OUT = S11 / "covariate_attribution"
TASKS = OUT / "covariate_subset_tasks.tsv"
AREA = ROOT / "scripts/multiomic/run_regular_area_binary.py"

GROUPS = {
    "monocyte": [
        ("age", ["age_sampling_capped"], ["age_sampling_90plus"]),
        ("sex", [], ["sex"]),
        ("sequencing_batch", [], ["sequencingBatch"]),
    ],
    "dlpfc": [
        ("age", ["age_numeric"], []),
        ("rin", ["rin_numeric"], []),
        ("pmi", ["pmi_numeric"], []),
        ("sex", [], ["sex"]),
        ("sequencing_batch", [], ["sequencing_batch"]),
    ],
    "tmt": [
        ("age", ["age_numeric"], []),
        ("pmi", ["pmi_numeric"], []),
        ("sex", [], ["msex"]),
        ("batch", [], ["Batch"]),
    ],
}


def csv_list(x):
    if pd.isna(x) or not str(x).strip():
        return []
    return [v.strip() for v in str(x).split(",") if v.strip()]


def boolish(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def full_covariates(modality):
    cont, cat = [], []
    for _, c, k in GROUPS[modality]:
        cont += c
        cat += k
    return cont, cat


def subset_covariates(modality, selected):
    cont, cat = [], []
    selected = set(selected)
    for name, c, k in GROUPS[modality]:
        if name in selected:
            cont += c
            cat += k
    return cont, cat


def validate_a(row):
    modality = row.modality
    expected_cont, expected_cat = full_covariates(modality)

    if csv_list(row.A_continuous) != expected_cont:
        raise ValueError(
            f"{row.contrast_id} {modality}: A continuous covariates changed"
        )
    if csv_list(row.A_categorical) != expected_cat:
        raise ValueError(
            f"{row.contrast_id} {modality}: A categorical covariates changed"
        )

    with open(row.A_run_manifest) as fh:
        m = json.load(fh)

    if m.get("block") is not None:
        raise ValueError(
            f"{row.contrast_id} {modality}: frozen A has non-null block"
        )
    if int(m["n"]) != int(row.A_n):
        raise ValueError(f"{row.contrast_id} {modality}: A n changed")
    if int(m["n_features"]) != int(row.A_n_features):
        raise ValueError(
            f"{row.contrast_id} {modality}: A feature universe changed"
        )

    b_manifest = Path(row.B_outdir) / "run_manifest.json"
    with open(b_manifest) as fh:
        b = json.load(fh)

    if b.get("block") is not None:
        raise ValueError(
            f"{row.contrast_id} {modality}: frozen B has non-null block"
        )
    if b.get("continuous_covariates") != []:
        raise ValueError(f"{row.contrast_id} {modality}: B not intercept-only")
    if b.get("categorical_covariates") != []:
        raise ValueError(f"{row.contrast_id} {modality}: B not intercept-only")


def prepare():
    base = pd.read_csv(BASE, sep="\t")
    if len(base) != 18:
        raise ValueError(f"Expected 18 Stage 11 tasks; found {len(base)}")

    rows = []
    new_id = 0

    for row in base.itertuples(index=False):
        validate_a(row)

        names = [x[0] for x in GROUPS[row.modality]]

        # Empty=B and full=A already exist. Generate only intermediate subsets.
        for size in range(1, len(names)):
            for selected in itertools.combinations(names, size):
                new_id += 1
                cont, cat = subset_covariates(row.modality, selected)
                sid = "__".join(selected)
                outdir = (
                    OUT / "subset_runs" / row.contrast_id / row.modality / sid
                )
                cname = (
                    f"Stage11_SUBSET__{row.contrast_id}__"
                    f"{row.modality}__{sid}"
                )
                result = (
                    outdir / f"{cname}_Regular_AREA_all_results.csv"
                )

                rows.append({
                    "subset_task_id": new_id,
                    "base_task_id": row.task_id,
                    "contrast_id": row.contrast_id,
                    "family": row.family,
                    "modality": row.modality,
                    "cohort": row.cohort,
                    "manifest": row.manifest,
                    "expression": row.expression,
                    "orientation": row.orientation,
                    "group_a": row.group_a,
                    "group_b": row.group_b,
                    "global_complete_features": row.global_complete_features,
                    "subset_groups": ",".join(selected),
                    "continuous": ",".join(cont),
                    "categorical": ",".join(cat),
                    "outdir": str(outdir),
                    "contrast_name": cname,
                    "result_path": str(result),
                    "expected_n": row.A_n,
                    "expected_n_features": row.A_n_features,
                })

    tasks = pd.DataFrame(rows)

    if len(tasks) != 300:
        raise ValueError(f"Expected 300 new subset runs; got {len(tasks)}")
    if tasks["result_path"].duplicated().any():
        raise ValueError("Duplicate subset output paths")

    OUT.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(TASKS, sep="\t", index=False)

    spec = {
        "endpoint_B": "existing intercept-only B",
        "endpoint_A": "existing fully adjusted A",
        "block": None,
        "n_base_tasks": 18,
        "n_new_subset_runs": 300,
        "covariate_groups": GROUPS,
    }
    with open(OUT / "COVARIATE_ATTRIBUTION_SPEC.json", "w") as fh:
        json.dump(spec, fh, indent=2)

    print(f"Wrote: {TASKS}")
    print(f"Total new subset tasks: {len(tasks)}")
    print(tasks.groupby("modality").size().to_string())
    print("Expected: dlpfc=180, monocyte=36, tmt=84")


def run_task(task_id):
    tasks = pd.read_csv(TASKS, sep="\t")
    hit = tasks.loc[tasks.subset_task_id == task_id]

    if len(hit) != 1:
        raise ValueError(f"Could not uniquely resolve subset task {task_id}")

    row = hit.iloc[0]
    outdir = Path(row.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(AREA),
        "--expression", row.expression,
        "--manifest", row.manifest,
        "--outdir", str(outdir),
        "--contrast-name", row.contrast_name,
        "--group-a", str(row.group_a),
        "--group-b", str(row.group_b),
        "--orientation", row.orientation,
    ]

    if boolish(row.global_complete_features):
        cmd.append("--global-complete-features")
    if isinstance(row.continuous, str) and row.continuous.strip():
        cmd += ["--continuous", row.continuous]
    if isinstance(row.categorical, str) and row.categorical.strip():
        cmd += ["--categorical", row.categorical]

    # Intentionally NO --block: frozen A and B both have block=null.
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    result = Path(row.result_path)
    run_manifest = outdir / "run_manifest.json"

    if not result.exists():
        raise FileNotFoundError(result)
    if not run_manifest.exists():
        raise FileNotFoundError(run_manifest)

    with open(run_manifest) as fh:
        m = json.load(fh)

    if m.get("block") is not None:
        raise ValueError("Subset unexpectedly used --block")
    if m.get("continuous_covariates") != csv_list(row.continuous):
        raise ValueError("Subset continuous covariates do not match task table")
    if m.get("categorical_covariates") != csv_list(row.categorical):
        raise ValueError("Subset categorical covariates do not match task table")
    if int(m["n"]) != int(row.expected_n):
        raise ValueError("Subset changed participant count")
    if int(m["n_features"]) != int(row.expected_n_features):
        raise ValueError("Subset changed feature universe")

    print(f"Validated subset task {task_id}: {result}")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--run-task", type=int)
    a = p.parse_args()

    if a.prepare:
        prepare()
    else:
        run_task(a.run_task)


if __name__ == "__main__":
    main()

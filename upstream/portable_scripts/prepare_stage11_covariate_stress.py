import os
#!/usr/bin/env python3

from pathlib import Path
import json

import pandas as pd
ROOT = Path(os.environ["ROSMAP_ANALYSIS_ROOT"]).expanduser().resolve()
OUT = ROOT / "results/stage11_covariate_stress"

PURE = "PURE_1_NCI_2_MCI_4_AD_exclude_3_5_6"

MONO_EXPR = Path(os.environ["ROSMAP_MONOCYTE_EXPRESSION"]).expanduser().resolve()

DLPFC_EXPR = (
    ROOT
    / "results/preprocessing_protein_coding/"
      "ROSMAP_AREA_normalized_counts_all_samples.csv"
)

TMT_EXPR = (
    ROOT
    / "data/proteomics_tmt/input/"
      "C2.median_polish_corrected_log2"
      "(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv"
)


CONTRASTS = [
    {
        "contrast_id": "cogdx_NCI_vs_AD",
        "family": "cogdx",
        "contrast_dir": "NCI_vs_AD",
        "reference": "NCI",
        "case": "AD",
    },
    {
        "contrast_id": "cogdx_NCI_vs_MCI",
        "family": "cogdx",
        "contrast_dir": "NCI_vs_MCI",
        "reference": "NCI",
        "case": "MCI",
    },
    {
        "contrast_id": "cogdx_MCI_vs_AD",
        "family": "cogdx",
        "contrast_dir": "MCI_vs_AD",
        "reference": "MCI",
        "case": "AD",
    },
    {
        "contrast_id": "Braak_0-2_vs_3-4",
        "family": "Braak",
        "contrast_dir": "0-2_vs_3-4",
        "reference": "0-2",
        "case": "3-4",
    },
    {
        "contrast_id": "Braak_3-4_vs_5-6",
        "family": "Braak",
        "contrast_dir": "3-4_vs_5-6",
        "reference": "3-4",
        "case": "5-6",
    },
    {
        "contrast_id": "CERAD_3-4_vs_1-2",
        "family": "CERAD",
        "contrast_dir": "1-2_vs_3-4",
        "reference": "CERAD_3-4",
        "case": "CERAD_1-2",
    },
]


MODALITIES = {
    "monocyte": {
        "expression": MONO_EXPR,
        "orientation": "features_by_samples",
        "global_complete": False,
        "continuous": "age_sampling_capped",
        "categorical": "age_sampling_90plus,sex,sequencingBatch",
    },
    "dlpfc": {
        "expression": DLPFC_EXPR,
        "orientation": "samples_by_features",
        "global_complete": False,
        "continuous": "age_numeric,rin_numeric,pmi_numeric",
        "categorical": "sex,sequencing_batch",
    },
    "tmt": {
        "expression": TMT_EXPR,
        "orientation": "features_by_samples",
        "global_complete": True,
        "continuous": "age_numeric,pmi_numeric",
        "categorical": "msex,Batch",
    },
}


def cohort_name(family, modality):
    if modality == "monocyte":
        if family in {"Braak", "CERAD"}:
            return "Monocyte_pathology_subset"
        return "Monocyte_discovery"

    if modality == "dlpfc":
        return "DLPFC_discovery"

    return "TMT_discovery"


def definition_root(base, spec):
    root = (
        base
        / spec["family"]
        / spec["contrast_dir"]
    )

    if spec["family"] == "cogdx":
        root = root / PURE

    return root


def normalize_group(x):
    x = str(x).strip()

    replacements = {
        "0–2": "0-2",
        "3–4": "3-4",
        "5–6": "5-6",
        "1–2": "1-2",
        "1-2": "1-2",
        "CERAD 1–2": "CERAD_1-2",
        "CERAD 3–4": "CERAD_3-4",
        "CERAD_1–2": "CERAD_1-2",
        "CERAD_3–4": "CERAD_3-4",
    }

    return replacements.get(x, x)


def normalized_for_match(x, family):
    y = normalize_group(x)

    if family == "CERAD":
        if y == "1-2":
            y = "CERAD_1-2"
        elif y == "3-4":
            y = "CERAD_3-4"

    return y


def find_manifest(spec, modality, cohort):
    base = definition_root(
        ROOT / "results/multiomic_analysis_manifests",
        spec,
    )

    if spec["family"] == "cogdx":
        hits = [
            base / cohort / f"{modality}_manifest.csv"
        ]
        hits = [x for x in hits if x.exists()]
    else:
        hits = sorted(
            base.glob(
                f"**/{cohort}/{modality}_manifest.csv"
            )
        )

    if len(hits) != 1:
        raise RuntimeError(
            f"{spec['contrast_id']} {modality}: "
            f"manifest hits={hits}"
        )

    return hits[0]


def find_a_result(spec, modality, cohort):
    base = definition_root(
        ROOT / "results/multiomic_models",
        spec,
    )

    hits = sorted(
        base.glob(
            f"**/{cohort}/{modality}/regular_area/"
            "*_Regular_AREA_all_results.csv"
        )
    )

    if len(hits) != 1:
        raise RuntimeError(
            f"{spec['contrast_id']} {modality}: "
            f"A result hits={hits}"
        )

    result = hits[0]
    manifest = result.parent / "run_manifest.json"

    if not manifest.exists():
        raise RuntimeError(
            f"Missing A run manifest: {manifest}"
        )

    info = json.loads(manifest.read_text())

    if info.get("block") is not None:
        raise RuntimeError(
            f"{spec['contrast_id']} {modality}: "
            f"A block={info.get('block')}; expected null"
        )

    inference = str(info.get("inference", ""))

    if "residual-df-corrected" not in inference:
        raise RuntimeError(
            f"{spec['contrast_id']} {modality}: "
            "A is not residual-df-corrected production AREA"
        )

    return result, manifest, info


rows = []
task_id = 0

for spec in CONTRASTS:
    for modality, cfg in MODALITIES.items():
        task_id += 1

        cohort = cohort_name(
            spec["family"],
            modality,
        )

        manifest = find_manifest(
            spec,
            modality,
            cohort,
        )

        m = pd.read_csv(manifest)

        observed = list(
            pd.unique(
                m["analysis_group"]
                .dropna()
                .astype(str)
            )
        )

        raw_reference = [
            x for x in observed
            if normalized_for_match(
                x,
                spec["family"],
            ) == spec["reference"]
        ]

        raw_case = [
            x for x in observed
            if normalized_for_match(
                x,
                spec["family"],
            ) == spec["case"]
        ]

        if (
            len(raw_reference) != 1
            or len(raw_case) != 1
        ):
            raise RuntimeError(
                f"{spec['contrast_id']} {modality}: "
                f"cannot identify groups from {observed}; "
                f"reference hits={raw_reference}; "
                f"case hits={raw_case}"
            )

        a_result, a_manifest, a_info = (
            find_a_result(
                spec,
                modality,
                cohort,
            )
        )

        b_outdir = (
            OUT
            / "B_intercept_only"
            / spec["contrast_id"]
            / modality
        )

        rows.append(
            {
                "task_id": task_id,
                "contrast_id": spec[
                    "contrast_id"
                ],
                "family": spec["family"],
                "modality": modality,
                "cohort": cohort,
                "manifest": str(manifest),
                "expression": str(
                    cfg["expression"]
                ),
                "orientation": cfg[
                    "orientation"
                ],
                "group_a": raw_reference[0],
                "group_b": raw_case[0],
                "global_complete_features": (
                    cfg["global_complete"]
                ),
                "A_continuous": cfg[
                    "continuous"
                ],
                "A_categorical": cfg[
                    "categorical"
                ],
                "A_result": str(a_result),
                "A_run_manifest": str(
                    a_manifest
                ),
                "A_n": a_info.get("n"),
                "A_n_features": a_info.get(
                    "n_features"
                ),
                "A_residual_df": a_info.get(
                    "residual_df"
                ),
                "B_outdir": str(b_outdir),
            }
        )


tasks = pd.DataFrame(rows)

if len(tasks) != 18:
    raise RuntimeError(
        f"Expected 18 tasks; got {len(tasks)}"
    )

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

task_file = OUT / "stage11_tasks.tsv"

tasks.to_csv(
    task_file,
    sep="\t",
    index=False,
)

print(tasks.to_string(index=False))
print()
print("Wrote:", task_file)
print("Tasks:", len(tasks))

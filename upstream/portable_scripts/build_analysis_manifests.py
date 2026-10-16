#!/usr/bin/env python3
"""
Build modality-specific binary-analysis manifests from the finalized ROSMAP
multi-omic analysis matrix.

This script does NOT fit any model. It freezes the exact participants/samples
for each allowed analysis so DESeq2/limma and Regular AREA consume the same
cohort.

Run from ROSMAP-AREA-DNA-rotation repo root.

Outputs:
  results/multiomic_analysis_manifests/<axis>/<contrast>/<cohort_slug>/<modality>_manifest.csv
  results/multiomic_analysis_manifests/manifest_index.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-matrix",
        default="results/multiomic_comparison_planning/ROSMAP_MULTIOMIC_FINAL_ANALYSIS_MATRIX_v2.csv")
    p.add_argument("--monocyte",
        default="results/monocyte_full_cohort/final_primary_554/monocyte_primary_554_metadata_with_sampling_age.tsv")
    p.add_argument("--dlpfc",
        default="results/preprocessing/ROSMAP_RNAseq_master_metadata_all_samples.csv")
    p.add_argument("--tmt",
        default="results/proteomics_tmt/model_inputs/TMT_modeling_metadata_400.csv")
    p.add_argument("--clinical", default="data/ROSMAP_harmonized_participant_phenotypes.tsv")
    p.add_argument("--outdir", default="results/multiomic_analysis_manifests")
    return p.parse_args()

def read_any(path):
    path = Path(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower()==".tsv" else ",", low_memory=False)

def clean(s):
    return s.astype("string").str.strip()

def slug(x):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x)).strip("_")

def first(df, names, label):
    for c in names:
        if c in df.columns:
            return c
    raise SystemExit(f"ERROR: cannot identify {label}. Tried {names}")

def num(s):
    return pd.to_numeric(s, errors="coerce")

def parse_age90(s):
    raw = s.astype("string").str.strip()
    return pd.to_numeric(raw.str.replace("+","",regex=False), errors="coerce").clip(upper=90)

def phenotype_group(clin, axis, contrast, definition):
    out = pd.Series(pd.NA, index=clin.index, dtype="string")

    if axis == "cogdx":
        x = num(clin["cogdx"])
        broad = str(definition).startswith("BROAD")
        if contrast == "NCI_vs_MCI":
            out[x.eq(1)] = "NCI"
            out[x.isin([2,3] if broad else [2])] = "MCI"
        elif contrast == "MCI_vs_AD":
            out[x.isin([2,3] if broad else [2])] = "MCI"
            out[x.isin([4,5] if broad else [4])] = "AD"
        elif contrast == "NCI_vs_AD":
            out[x.eq(1)] = "NCI"
            out[x.isin([4,5] if broad else [4])] = "AD"
        else:
            raise ValueError(f"Unsupported cogdx contrast: {contrast}")

    elif axis == "Braak":
        x = num(clin["braaksc"])
        groups = {
            "0-2": x.isin([0,1,2]),
            "3-4": x.isin([3,4]),
            "5-6": x.isin([5,6]),
        }
        a,b = contrast.split("_vs_")
        out[groups[a]] = a
        out[groups[b]] = b

    elif axis == "CERAD":
        x = num(clin["ceradsc"])
        if contrast != "1-2_vs_3-4":
            raise ValueError(f"Unsupported CERAD contrast: {contrast}")
        out[x.isin([1,2])] = "CERAD_1-2"
        out[x.isin([3,4])] = "CERAD_3-4"
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    return out

def main():
    a = parse_args()
    outroot = Path(a.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    plan = read_any(a.analysis_matrix)
    mono = read_any(a.monocyte)
    dlp = read_any(a.dlpfc)
    tmt = read_any(a.tmt)
    clin = read_any(a.clinical)

    mid = first(mono, ["individualID","individual_id"], "monocyte participant ID")
    did = first(dlp, ["individual_id","individualID"], "DLPFC participant ID")
    tid = first(tmt, ["individualID","IndividualID","individual_id"], "TMT participant ID")
    cid = first(clin, ["individualID","individual_id","IndividualID"], "clinical participant ID")

    msamp = first(mono, ["specimen","sample_id","specimenID","monocyte_specimen_id"], "monocyte specimen ID")
    dsamp = first(dlp, ["sample_id","SampleID"], "DLPFC sample ID")
    tsamp = first(tmt, ["SampleID","sample_id"], "TMT sample ID")

    for df,col in [(mono,mid),(dlp,did),(tmt,tid),(clin,cid)]:
        df[col] = clean(df[col])

    for label,df,col in [("monocyte",mono,mid),("DLPFC",dlp,did),("TMT",tmt,tid),("clinical",clin,cid)]:
        if df[col].duplicated().any():
            raise SystemExit(f"ERROR: {label} table is not one row per participant.")

    # Join authoritative clinical phenotype + sex to all modalities.
    keep = [cid,"cogdx","braaksc","ceradsc"]
    if "msex" in clin.columns:
        keep.append("msex")
    elif "sex" in clin.columns:
        keep.append("sex")
    c = clin[keep].copy().rename(columns={cid:"participant_id"})
    c = c.set_index("participant_id")

    # Standardized modality metadata.
    M = mono.copy()
    M["participant_id"] = M[mid]
    M["sample_id"] = clean(M[msamp])
    if "samplingAge_numeric_capped" in M:
        M["age_sampling_capped"] = num(M["samplingAge_numeric_capped"])
    elif "age_sampling_capped" not in M:
        src = first(M, ["samplingAge_raw","samplingAge"], "monocyte sampling age")
        M["age_sampling_capped"] = parse_age90(M[src])
    if "samplingAge_90plus" in M:
        M["age_sampling_90plus"] = num(M["samplingAge_90plus"])
    elif "age_sampling_90plus" not in M:
        src = first(M, ["samplingAge_raw","samplingAge"], "monocyte sampling age")
        M["age_sampling_90plus"] = M[src].astype("string").str.contains(r"\+",regex=True,na=False).astype(int)
    if "sex" not in M.columns:
        sex_source = "msex" if "msex" in c.columns else "sex"
        M = M.merge(c[[sex_source]].rename(columns={sex_source:"sex"}),
                    left_on="participant_id", right_index=True, how="left", validate="one_to_one")
    if "sequencingBatch" not in M.columns:
        raise SystemExit("ERROR: monocyte table lacks sequencingBatch")

    D = dlp.copy()
    D["participant_id"] = D[did]
    D["sample_id"] = clean(D[dsamp])
    for col in ["age_numeric","sex","rin_numeric","pmi_numeric","sequencing_batch"]:
        if col not in D.columns:
            raise SystemExit(f"ERROR: DLPFC metadata lacks {col}")

    # Frozen DLPFC QC exclusion.
    dlpfc_qc_exclude_samples = {"492_120515"}
    dlpfc_qc_rows = D["sample_id"].isin(dlpfc_qc_exclude_samples)
    found = set(D.loc[dlpfc_qc_rows, "sample_id"].astype(str))
    if found != dlpfc_qc_exclude_samples:
        raise SystemExit(
            f"ERROR: expected DLPFC QC exclusion sample(s) "
            f"{sorted(dlpfc_qc_exclude_samples)}, found {sorted(found)}"
        )
    print("DLPFC frozen QC exclusion: " + ", ".join(sorted(found)))
    D = D.loc[~dlpfc_qc_rows].copy()

    T = tmt.copy()
    T["participant_id"] = T[tid]
    T["sample_id"] = clean(T[tsamp])
    for col in ["age_numeric","msex","pmi_numeric","Batch"]:
        if col not in T.columns:
            raise SystemExit(f"ERROR: TMT metadata lacks {col}")

    modality_tables = {"monocyte":M, "dlpfc":D, "tmt":T}
    modality_ids = {k:set(v["participant_id"]) for k,v in modality_tables.items()}
    modality_covars = {
        "monocyte":["age_sampling_capped","age_sampling_90plus","sex","sequencingBatch"],
        "dlpfc":["age_numeric","sex","rin_numeric","pmi_numeric","sequencing_batch"],
        "tmt":["age_numeric","msex","pmi_numeric","Batch"],
    }

    cohort_sets = {
        "Monocyte discovery": modality_ids["monocyte"],
        "Monocyte pathology subset": modality_ids["monocyte"],
        "DLPFC discovery": modality_ids["dlpfc"],
        "TMT discovery": modality_ids["tmt"],
        "Monocyte ∩ DLPFC": modality_ids["monocyte"] & modality_ids["dlpfc"],
        "Monocyte ∩ TMT": modality_ids["monocyte"] & modality_ids["tmt"],
        "DLPFC ∩ TMT": modality_ids["dlpfc"] & modality_ids["tmt"],
        "Three-way": modality_ids["monocyte"] & modality_ids["dlpfc"] & modality_ids["tmt"],
    }

    index_rows = []
    seen_specs = set()

    for _, r in plan.iterrows():
        axis = str(r["axis"])
        contrast = str(r["contrast"])
        cohort = str(r["cohort"])
        definition = str(r["phenotype_definition"])
        mods_text = str(r["modalities_analyzed"])

        spec_key = (axis,contrast,cohort,definition)
        # Matrix has one row per scientific comparison, not per modality. Process once.
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)

        if cohort not in cohort_sets:
            raise SystemExit(f"ERROR: unknown cohort in matrix: {cohort}")
        cohort_ids = cohort_sets[cohort]

        clin_sub = c.loc[c.index.intersection(cohort_ids)].copy()
        clin_sub["analysis_group"] = phenotype_group(clin_sub, axis, contrast, definition)
        eligible_pheno = set(clin_sub.index[clin_sub["analysis_group"].notna()])

        requested = []
        if "Monocyte RNA" in mods_text: requested.append("monocyte")
        if "DLPFC RNA" in mods_text: requested.append("dlpfc")
        if "TMT proteomics" in mods_text: requested.append("tmt")

        # Joint complete-case participant set across every requested modality.
        joint = cohort_ids & eligible_pheno
        for mod in requested:
            tab = modality_tables[mod].set_index("participant_id")
            cc = set(tab.index[tab[modality_covars[mod]].notna().all(axis=1)])
            joint &= cc

        # One manifest per modality but identical participant set for matched analyses.
        for mod in requested:
            tab = modality_tables[mod].set_index("participant_id").loc[sorted(joint)].copy()
            tab["participant_id"] = tab.index
            tab["analysis_group"] = clin_sub.loc[tab.index, "analysis_group"].astype(str)

            keepcols = ["participant_id","sample_id","analysis_group"] + modality_covars[mod]
            out = tab[keepcols].reset_index(drop=True)

            if out["sample_id"].duplicated().any():
                raise SystemExit(f"ERROR: duplicate sample IDs in {axis}/{contrast}/{cohort}/{mod}")

            expected_n = int(r["joint_covariate_complete_n"])
            if len(out) != expected_n:
                raise SystemExit(
                    f"ERROR: N mismatch for {axis}/{contrast}/{cohort}/{mod}: "
                    f"matrix expects {expected_n}, rebuilt {len(out)}"
                )

            # Include phenotype definition in the output path.
            # This is essential for cogdx because PURE and BROAD sensitivity
            # analyses otherwise share axis/contrast/cohort/modality names and
            # would overwrite one another.
            od = (
                outroot
                / slug(axis)
                / slug(contrast)
                / slug(definition)
                / slug(cohort)
            )
            od.mkdir(parents=True, exist_ok=True)
            path = od / f"{mod}_manifest.csv"
            out.to_csv(path,index=False)

            index_rows.append({
                "axis":axis, "contrast":contrast, "cohort":cohort,
                "role":r["role"], "modality":mod,
                "manifest":str(path), "n":len(out),
                "group_A":r["group_A"], "group_B":r["group_B"],
                "phenotype_definition":definition,
                "dlpfc_qc_exclusion": (
                    "492_120515|R5693901|sequencing_batch=0, 6, 7"
                    if "dlpfc" in requested else ""
                ),
            })

    idx = pd.DataFrame(index_rows)
    idx.to_csv(outroot/"manifest_index.csv",index=False)
    print("ROSMAP BINARY MANIFEST BUILD")
    print("="*72)
    print(f"Manifests written: {len(idx)}")
    print(f"Unique scientific specifications: {len(seen_specs)}")
    print(f"Index: {outroot/'manifest_index.csv'}")
    print("\nCounts by modality:")
    print(idx.groupby("modality").size().to_string())

if __name__ == "__main__":
    main()

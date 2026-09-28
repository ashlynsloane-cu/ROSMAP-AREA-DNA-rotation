#!/usr/bin/env python3
# build_stage10_multimodal_simulations.py
#
# Stage 10 multimodal ROSMAP-calibrated simulations.
#
# Modalities:
#   10A monocyte RNA: AREA vs DESeq2
#   10B DLPFC RNA:    AREA vs DESeq2
#   10C TMT protein:  AREA vs limma
#   10D cross-modality synthesis
#
# The scenario grid is frozen and shared conceptually across modalities:
# BROAD_MEAN_SHIFT, RANDOM_SUBSET_SHIFT, ONE_TAIL_AMPLIFICATION,
# SYMMETRIC_TWO_TAIL, ISOLATED_OUTLIERS, VARIANCE_ONLY.
#
# RNA is simulated with negative-binomial counts calibrated to real NCI data.
# TMT is simulated on the real-data log2 abundance scale using an
# empirical NCI residual bootstrap by default; Gaussian generation is retained
# as a pre-specified sensitivity mode.
#
# IMPORTANT:
# * Uses existing production method runners.
# * AREA is run WITHOUT --block.
# * Pilot results are machinery checks only.
# * The scenario grid must not be altered after results are inspected.
# * v5 makes RNA VARIANCE_ONLY strictly finite-sample mean-preserving by
#   rescaling realized lognormal multipliers to arithmetic mean 1. This is a
#   pre-freeze technical correction motivated by the pilot audit, not power.
# * SYMMETRIC_TWO_TAIL is explicitly mean-balanced on
#   the realized normalized-expression scale before anchored perturbation.

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path.home() / "rosmap_multiomic_slurm"
RSCRIPT = Path("/home/assl5508/envs/rosmap-r/bin/Rscript")

AREA_RUNNER = ROOT / "scripts/multiomic/run_regular_area_binary.py"
DESEQ_RUNNER = ROOT / "scripts/multiomic/run_rna_deseq2_binary.R"
LIMMA_RUNNER = ROOT / "scripts/multiomic/run_tmt_limma_binary.R"

PURE_BASE = (
    ROOT
    / "results/multiomic_analysis_manifests/cogdx/NCI_vs_AD/"
      "PURE_1_NCI_2_MCI_4_AD_exclude_3_5_6"
)

MONO_RAW = Path(
    "/home/assl5508/rosmap_monocyte_cross_tissue/"
    "analysis_inputs/monocyte_primary_554/model_inputs/"
    "monocyte_raw_counts_deseq2_554.tsv"
)
MONO_NORM = Path(
    "/home/assl5508/rosmap_monocyte_cross_tissue/"
    "analysis_inputs/monocyte_primary_554/model_inputs/"
    "monocyte_sizefactor_normalized_counts_area_554.tsv"
)

DLPFC_RAW_RDS = (
    ROOT
    / "results/preprocessing_protein_coding/"
      "ROSMAP_raw_counts_global_gene_universe.rds"
)
DLPFC_COLUMN_MAP = (
    ROOT
    / "results/preprocessing_protein_coding/"
      "ROSMAP_count_column_to_sample_id.csv"
)
DLPFC_NORM = (
    ROOT
    / "results/preprocessing_protein_coding/"
      "ROSMAP_AREA_normalized_counts_all_samples.csv"
)

TMT_MATRIX = (
    ROOT
    / "data/proteomics_tmt/input/"
      "C2.median_polish_corrected_log2"
      "(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv"
)

OUTROOT = ROOT / "results/stage10_multimodal_simulations"

PENETRANCE_GRID = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
EFFECT_GRID = [0.5, 1.0, 2.0, 3.0]
OUTLIER_COUNTS = [1, 3, 7]
OUTLIER_EFFECT_GRID = [2.0, 3.0, 4.0]
VARIANCE_GRID = [0.5, 1.0, 2.0, 3.0]

FDR = 0.05
TOPK_REFERENCE = [250, 500, 1000]

CONFIG = {
    "monocyte": {
        "kind": "rna",
        "manifest": PURE_BASE / "Monocyte_discovery/monocyte_manifest.csv",
        "n_nci": 152,
        "n_ad": 146,
        "universe": 14756,
        "raw": MONO_RAW,
        "norm": MONO_NORM,
        "continuous": ["age_sampling_capped"],
        "categorical": ["age_sampling_90plus", "sex", "sequencingBatch"],
        "conventional": "DESeq2",
    },
    "dlpfc": {
        "kind": "rna",
        "manifest": PURE_BASE / "DLPFC_discovery/dlpfc_manifest.csv",
        "n_nci": 200,
        "n_ad": 219,
        "universe": 17646,
        "raw_rds": DLPFC_RAW_RDS,
        "column_map": DLPFC_COLUMN_MAP,
        "norm": DLPFC_NORM,
        "continuous": ["age_numeric", "rin_numeric", "pmi_numeric"],
        "categorical": ["sex", "sequencing_batch"],
        "conventional": "DESeq2",
    },
    "tmt": {
        "kind": "continuous",
        "manifest": PURE_BASE / "TMT_discovery/tmt_manifest.csv",
        "n_nci": 167,
        "n_ad": 109,
        "universe": 5211,
        "matrix": TMT_MATRIX,
        "continuous": ["age_numeric", "pmi_numeric"],
        "categorical": ["msex", "Batch"],
        "conventional": "limma",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modality",
        choices=["all", "monocyte", "dlpfc", "tmt"],
        default="all",
    )
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--replicate-id", type=int, default=None)
    p.add_argument("--replicates", type=int, default=25)
    p.add_argument("--genes-per-scenario", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tmt-baseline",
        choices=["empirical_bootstrap", "gaussian"],
        default="empirical_bootstrap",
        help=(
            "TMT null/baseline generator. Primary: empirical NCI centered-"
            "residual bootstrap; gaussian is a pre-specified sensitivity."
        ),
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def require(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def run_checked(cmd, log_path=None):
    if log_path is None:
        proc = subprocess.run(cmd, text=True)
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}"
            + (f"\nSee {log_path}" if log_path else "")
        )


def check_environment():
    require(AREA_RUNNER, "AREA runner")
    require(DESEQ_RUNNER, "DESeq2 runner")
    require(LIMMA_RUNNER, "limma runner")
    require(RSCRIPT, "ROSMAP Rscript")

    code = (
        'cat("DESeq2=", requireNamespace("DESeq2", quietly=TRUE), "\\n", sep="");'
        'cat("limma=", requireNamespace("limma", quietly=TRUE), "\\n", sep="");'
        'if(!requireNamespace("DESeq2", quietly=TRUE) || '
        '!requireNamespace("limma", quietly=TRUE)) quit(status=42)'
    )
    proc = subprocess.run(
        [str(RSCRIPT), "-e", code],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Required R packages unavailable in ~/envs/rosmap-r.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    print(proc.stdout.strip())


def validate_manifest(d, cfg):
    required = [
        "sample_id",
        "analysis_group",
        *cfg["continuous"],
        *cfg["categorical"],
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    if d[required].isna().any().any():
        bad = d[required].isna().sum()
        bad = bad[bad > 0]
        raise ValueError(
            "Manifest contains missing required covariates:\n"
            + bad.to_string()
        )

    counts = d["analysis_group"].astype(str).value_counts().to_dict()
    expected = {"NCI": cfg["n_nci"], "AD": cfg["n_ad"]}
    if counts != expected:
        raise ValueError(
            f"Frozen cohort mismatch: observed={counts}; expected={expected}"
        )


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out

    pv = p[valid]
    order = np.argsort(pv)
    q = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    restored = np.empty_like(q)
    restored[order] = q
    out[valid] = restored
    return out


def scenario_grid(n_ad):
    rows = []
    num = 0

    def add(geometry, effect, penetrance=np.nan, outlier_count=np.nan):
        nonlocal num
        num += 1
        frac = (
            float(outlier_count) / n_ad
            if pd.notna(outlier_count)
            else penetrance
        )
        rows.append({
            "scenario_number": num,
            "scenario_id": f"S{num:03d}",
            "geometry": geometry,
            "penetrance": penetrance,
            "outlier_count": outlier_count,
            "actual_carrier_fraction": frac,
            "effect_sd": effect,
        })

    for e in EFFECT_GRID:
        add("BROAD_MEAN_SHIFT", e, penetrance=1.0)

    for p in PENETRANCE_GRID:
        for e in EFFECT_GRID:
            add("RANDOM_SUBSET_SHIFT", e, penetrance=p)

    for p in PENETRANCE_GRID:
        for e in EFFECT_GRID:
            add("ONE_TAIL_AMPLIFICATION", e, penetrance=p)

    for p in PENETRANCE_GRID:
        for e in EFFECT_GRID:
            add("SYMMETRIC_TWO_TAIL", e, penetrance=p)

    for k in OUTLIER_COUNTS:
        for e in OUTLIER_EFFECT_GRID:
            add("ISOLATED_OUTLIERS", e, outlier_count=k)

    for e in VARIANCE_GRID:
        add("VARIANCE_ONLY", e, penetrance=1.0)

    g = pd.DataFrame(rows)
    if len(g) != 101:
        raise AssertionError(f"Expected 101 scenarios, got {len(g)}")
    return g


def read_features_by_samples_tsv(path, sample_ids):
    d = pd.read_csv(path, sep="\t", low_memory=False)
    d.columns = [str(x) for x in d.columns]
    sample_ids = [str(x) for x in sample_ids]

    missing = sorted(set(sample_ids) - set(d.columns))
    if missing:
        raise ValueError(
            f"{path}: missing {len(missing)} samples, e.g. {missing[:8]}"
        )

    non_sample = [c for c in d.columns if c not in sample_ids]
    if not non_sample:
        raise ValueError(f"{path}: cannot identify feature ID column.")
    id_col = non_sample[0]

    ids = (
        d[id_col]
        .astype(str)
        .str.replace(r"\.\d+$", "", regex=True)
        .to_numpy()
    )
    x = (
        d[sample_ids]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite matrix values.")
    return ids, x, id_col


def read_dlpfc_norm(path, sample_ids):
    d = pd.read_csv(path, low_memory=False)
    d.columns = [str(x) for x in d.columns]
    sample_ids = [str(x) for x in sample_ids]

    if set(sample_ids).issubset(d.columns):
        non_sample = [c for c in d.columns if c not in sample_ids]
        id_col = non_sample[0]
        ids = (
            d[id_col]
            .astype(str)
            .str.replace(r"\.\d+$", "", regex=True)
            .to_numpy()
        )
        x = d[sample_ids].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return ids, x, "features_by_samples"

    first = d.columns[0]
    first_vals = d[first].astype(str)
    if set(sample_ids).issubset(set(first_vals)):
        dd = d.set_index(first)
        x = (
            dd.loc[sample_ids]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
            .T
        )
        ids = (
            pd.Index(dd.columns)
            .astype(str)
            .str.replace(r"\.\d+$", "", regex=True)
            .to_numpy()
        )
        return ids, x, "samples_by_features"

    raise ValueError(
        "Could not align DLPFC normalized matrix to manifest sample IDs."
    )


def export_dlpfc_raw_rds(rds, column_map, output_tsv):
    output_tsv = Path(output_tsv)
    if output_tsv.exists():
        return

    require(rds, "DLPFC raw-count RDS")
    require(column_map, "DLPFC count-column map")
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    helper = output_tsv.parent / "export_dlpfc_raw_for_stage10.R"
    helper.write_text(r'''
args <- commandArgs(trailingOnly=TRUE)
rds <- args[[1]]
mapfile <- args[[2]]
outfile <- args[[3]]

x <- readRDS(rds)
x <- as.matrix(x)

m <- read.csv(mapfile, check.names=FALSE, stringsAsFactors=FALSE)

if (nrow(m) != ncol(x)) {
    stop(
        paste0(
            "Column-map rows (", nrow(m),
            ") != count-matrix columns (", ncol(x), ")."
        )
    )
}

sample_candidates <- c(
    "sample_id", "SampleID", "sample", "sampleId",
    "sampleID", "individual_sample_id"
)

sample_col <- sample_candidates[sample_candidates %in% names(m)][1]

if (is.na(sample_col) || length(sample_col) == 0) {
    good <- names(m)[vapply(
        m,
        function(z) length(unique(z)) == nrow(m),
        logical(1)
    )]
    if (length(good) == 0) {
        stop("Could not identify sample-ID column in DLPFC column map.")
    }
    sample_col <- good[[length(good)]]
}

colnames(x) <- as.character(m[[sample_col]])

gene_id <- rownames(x)
if (is.null(gene_id)) {
    gene_id <- seq_len(nrow(x))
}

out <- data.frame(
    gene_id=gene_id,
    x,
    check.names=FALSE
)

write.table(
    out,
    file=outfile,
    sep="\t",
    quote=FALSE,
    row.names=FALSE
)

cat("sample_col=", sample_col, "\n", sep="")
cat("features=", nrow(x), "\n", sep="")
cat("samples=", ncol(x), "\n", sep="")
''')

    run_checked([
        str(RSCRIPT),
        str(helper),
        str(rds),
        str(column_map),
        str(output_tsv),
    ])


def infer_size_factors(raw, norm):
    if raw.shape != norm.shape:
        raise ValueError(
            f"raw/norm shape mismatch: {raw.shape} vs {norm.shape}"
        )

    sf = []
    for j in range(raw.shape[1]):
        valid = (
            np.isfinite(raw[:, j])
            & np.isfinite(norm[:, j])
            & (raw[:, j] > 0)
            & (norm[:, j] > 0)
        )
        if valid.sum() < 100:
            raise ValueError(
                f"Too few positive genes to infer size factor for sample {j}."
            )
        sf.append(np.median(raw[valid, j] / norm[valid, j]))

    sf = np.asarray(sf, dtype=float)
    sf /= np.exp(np.mean(np.log(sf)))
    return sf


def deseq_ratio_size_factors(counts):
    positive = (counts > 0).all(axis=1)
    use = counts[positive]
    if len(use) < 50:
        raise ValueError("Too few all-positive genes for ratio size factors.")

    gm = np.exp(np.mean(np.log(use), axis=1))
    sf = np.median(use / gm[:, None], axis=0)
    sf /= np.exp(np.mean(np.log(sf)))
    return sf


def rna_calibration(ids, raw, norm, groups, sf):
    nci = groups == "NCI"
    raw_nci = raw[:, nci]
    norm_nci = norm[:, nci]
    sf_nci = sf[nci]

    mu = raw_nci.sum(axis=1) / sf_nci.sum()
    expected = mu[:, None] * sf_nci[None, :]

    num = np.sum((raw_nci - expected) ** 2 - expected, axis=1)
    den = np.sum(expected ** 2, axis=1)
    alpha_raw = np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=float),
        where=den > 0,
    )
    alpha = np.clip(alpha_raw, 1e-4, 5.0)
    sd = np.std(np.log2(norm_nci + 0.5), axis=1, ddof=1)
    zero = np.mean(raw_nci == 0, axis=1)

    d = pd.DataFrame({
        "template_feature_id": ids,
        "baseline_mean": mu,
        "dispersion": alpha,
        "dispersion_raw": alpha_raw,
        "baseline_sd": sd,
        "zero_fraction": zero,
    })
    d["eligible"] = (
        np.isfinite(d["baseline_mean"])
        & np.isfinite(d["dispersion"])
        & np.isfinite(d["baseline_sd"])
        & (d["baseline_mean"] >= 1.0)
        & (d["baseline_sd"] >= 0.10)
        & (d["baseline_sd"] <= 2.0)
        & (d["dispersion"] <= 5.0)
    )
    return d


def read_tmt(path, sample_ids):
    d = pd.read_csv(path, low_memory=False)
    d.columns = [str(x) for x in d.columns]
    sample_ids = [str(x) for x in sample_ids]

    if not set(sample_ids).issubset(d.columns):
        missing = sorted(set(sample_ids) - set(d.columns))
        raise ValueError(
            f"TMT matrix missing manifest samples, e.g. {missing[:8]}"
        )

    # Frozen primary TMT universe:
    # completeness is defined across ALL 400 biological sample columns,
    # BEFORE phenotype-specific sample subsetting. Checking only the current
    # NCI/AD manifest can incorrectly retain proteins whose missing value lies
    # in a participant outside the current contrast.
    id_col = d.columns[0]
    all_sample_cols = d.columns[1:].tolist()

    if len(all_sample_cols) != 400:
        raise ValueError(
            f"Expected 400 TMT biological sample columns; "
            f"found {len(all_sample_cols)}."
        )

    all_expr = (
        d.loc[:, all_sample_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    complete_global = np.isfinite(
        all_expr.to_numpy(dtype=float)
    ).all(axis=1)

    ids_all = d[id_col].astype(str).to_numpy()
    ids = ids_all[complete_global]

    if len(ids) != 5211:
        raise ValueError(
            "Frozen TMT universe assertion failed: expected 5,211 proteins "
            f"complete across all 400 biological samples; found {len(ids):,}."
        )

    x = (
        all_expr.loc[complete_global, sample_ids]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "TMT phenotype-subset matrix contains non-finite values after "
            "global-complete filtering."
        )

    return ids, x


def tmt_calibration(ids, x, groups):
    nci = groups == "NCI"
    xn = x[:, nci]

    mean = np.mean(xn, axis=1)
    sd = np.std(xn, axis=1, ddof=1)
    q10 = np.quantile(xn, 0.10, axis=1)
    q90 = np.quantile(xn, 0.90, axis=1)

    d = pd.DataFrame({
        "template_feature_id": ids,
        "baseline_mean": mean,
        "baseline_sd": sd,
        "p90_p10_span": q90 - q10,
    })
    d["eligible"] = (
        np.isfinite(d["baseline_mean"])
        & np.isfinite(d["baseline_sd"])
        & (d["baseline_sd"] > 0)
    )
    return d


def prepare_modality(modality, args, base_root):
    cfg = CONFIG[modality]
    out = base_root / modality
    spec_dir = out / "specification"
    cal_dir = out / "calibration"

    spec_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(require(cfg["manifest"], f"{modality} manifest"))
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    validate_manifest(manifest, cfg)

    sample_ids = manifest["sample_id"].tolist()
    groups = manifest["analysis_group"].astype(str).to_numpy()

    if cfg["kind"] == "rna":
        if modality == "monocyte":
            ids_raw, raw, _ = read_features_by_samples_tsv(
                require(cfg["raw"], "monocyte raw counts"),
                sample_ids,
            )
            ids_norm, norm, _ = read_features_by_samples_tsv(
                require(cfg["norm"], "monocyte normalized counts"),
                sample_ids,
            )
        else:
            exported = cal_dir / "dlpfc_raw_counts_stage10.tsv"
            export_dlpfc_raw_rds(
                cfg["raw_rds"],
                cfg["column_map"],
                exported,
            )
            ids_raw, raw, _ = read_features_by_samples_tsv(
                exported,
                sample_ids,
            )
            ids_norm, norm, orientation = read_dlpfc_norm(
                require(cfg["norm"], "DLPFC normalized counts"),
                sample_ids,
            )
            print(f"DLPFC normalized input detected as {orientation}")

        raw_map = {g: i for i, g in enumerate(ids_raw)}
        norm_map = {g: i for i, g in enumerate(ids_norm)}
        common = [g for g in ids_raw if g in norm_map]

        if len(common) != cfg["universe"]:
            raise ValueError(
                f"{modality}: expected {cfg['universe']:,} aligned features; "
                f"found {len(common):,}."
            )

        ridx = [raw_map[g] for g in common]
        nidx = [norm_map[g] for g in common]
        raw = raw[ridx]
        norm = norm[nidx]
        ids = np.asarray(common, dtype=object)

        sf = infer_size_factors(raw, norm)
        cal = rna_calibration(ids, raw, norm, groups, sf)

        pd.DataFrame({
            "sample_id": sample_ids,
            "production_size_factor_inferred": sf,
        }).to_csv(
            cal_dir / "production_size_factors.tsv",
            sep="\t",
            index=False,
        )

    else:
        ids, x = read_tmt(
            require(cfg["matrix"], "TMT matrix"),
            sample_ids,
        )
        cal = tmt_calibration(ids, x, groups)

        # Primary TMT baseline generator preserves the empirical marginal
        # distribution shape of real NCI proteins. Store centered NCI
        # residuals once during preparation so each simulation replicate can
        # bootstrap them without repeatedly re-reading the source matrix.
        nci = groups == "NCI"
        x_nci = x[:, nci]
        nci_mean = np.mean(x_nci, axis=1, keepdims=True)
        residuals = x_nci - nci_mean

        if residuals.shape != (cfg["universe"], cfg["n_nci"]):
            raise ValueError(
                f"TMT residual matrix shape mismatch: {residuals.shape}; "
                f"expected {(cfg['universe'], cfg['n_nci'])}."
            )

        residual_df = pd.DataFrame(
            residuals,
            columns=[
                f"NCI_residual_{i:03d}"
                for i in range(1, residuals.shape[1] + 1)
            ],
        )
        residual_df.insert(
            0,
            "template_feature_id",
            ids,
        )
        residual_df.to_csv(
            cal_dir / "tmt_nci_centered_residuals.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

    if int(cal["eligible"].sum()) < 1000:
        raise ValueError(
            f"{modality}: too few eligible calibration features: "
            f"{int(cal['eligible'].sum())}"
        )

    grid = scenario_grid(cfg["n_ad"])
    grid.to_csv(
        spec_dir / "scenario_grid.tsv",
        sep="\t",
        index=False,
    )
    cal.to_csv(
        cal_dir / "calibration_parameters.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    manifest.to_csv(
        cal_dir / "frozen_manifest.csv",
        index=False,
    )

    spec = {
        "modality": modality,
        "kind": cfg["kind"],
        "sample_n": len(manifest),
        "n_NCI": cfg["n_nci"],
        "n_AD": cfg["n_ad"],
        "feature_universe": cfg["universe"],
        "continuous_covariates": cfg["continuous"],
        "categorical_covariates": cfg["categorical"],
        "conventional_method": cfg["conventional"],
        "AREA_block": None,
        "n_scenarios": len(grid),
        "genes_per_scenario": args.genes_per_scenario,
        "replicates": args.replicates,
        "seed": args.seed,
        "pilot": args.pilot,
        "tmt_baseline_generator_primary": (
            args.tmt_baseline
            if modality == "tmt"
            else None
        ),
        "tmt_gaussian_sensitivity_available": (
            modality == "tmt"
        ),
        "rna_variance_only_mean_definition": (
            "finite-sample arithmetic expected normalized-expression mean "
            "preserved exactly by normalizing realized lognormal multipliers "
            "to mean 1; log2-scale SD unchanged by this normalization"
            if cfg["kind"] == "rna"
            else None
        ),
        "rna_symmetric_two_tail_mean_definition": (
            "realized carrier normalized-expression mean balanced exactly "
            "before anchored Poisson perturbation"
            if cfg["kind"] == "rna"
            else None
        ),
    }
    with open(spec_dir / "specification.json", "w") as f:
        json.dump(spec, f, indent=2)

    qcols = [
        c
        for c in [
            "baseline_mean",
            "dispersion",
            "baseline_sd",
            "zero_fraction",
            "p90_p10_span",
        ]
        if c in cal.columns
    ]
    q = cal.loc[cal["eligible"], qcols].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    q.to_csv(
        cal_dir / "calibration_quantiles.tsv",
        sep="\t",
    )

    print("\n" + "=" * 100)
    print(f"STAGE 10 PREPARED: {modality.upper()}")
    print("=" * 100)
    print(
        f"n={len(manifest)} "
        f"(NCI={cfg['n_nci']}, AD={cfg['n_ad']}), "
        f"universe={cfg['universe']:,}, "
        f"eligible calibration features={int(cal['eligible'].sum()):,}"
    )
    print(q.to_string())


def load_prepared(modality, base_root):
    cfg = CONFIG[modality]
    out = base_root / modality

    manifest = pd.read_csv(out / "calibration/frozen_manifest.csv")
    manifest["sample_id"] = manifest["sample_id"].astype(str)

    cal = pd.read_csv(
        out / "calibration/calibration_parameters.tsv.gz",
        sep="\t",
        low_memory=False,
    )
    grid = pd.read_csv(
        out / "specification/scenario_grid.tsv",
        sep="\t",
    )

    sf = None
    tmt_residuals = None

    if cfg["kind"] == "rna":
        sf_d = pd.read_csv(
            out / "calibration/production_size_factors.tsv",
            sep="\t",
        )
        sf_map = dict(zip(
            sf_d["sample_id"].astype(str),
            sf_d["production_size_factor_inferred"].astype(float),
        ))
        sf = np.array(
            [sf_map[s] for s in manifest["sample_id"].astype(str)],
            dtype=float,
        )
    else:
        residual_file = (
            out
            / "calibration/tmt_nci_centered_residuals.tsv.gz"
        )
        if residual_file.exists():
            rd = pd.read_csv(
                residual_file,
                sep="\t",
                low_memory=False,
            )
            if "template_feature_id" not in rd.columns:
                raise ValueError(
                    "TMT residual file missing template_feature_id."
                )

            residual_ids = rd["template_feature_id"].astype(str)
            cal_ids = cal["template_feature_id"].astype(str)

            if residual_ids.duplicated().any():
                raise ValueError(
                    "Duplicate template_feature_id values in TMT residual file."
                )

            if not residual_ids.equals(cal_ids):
                raise ValueError(
                    "TMT residual rows do not exactly match calibration rows."
                )

            tmt_residuals = rd.drop(
                columns=["template_feature_id"]
            ).to_numpy(dtype=float)

            expected_shape = (
                cfg["universe"],
                cfg["n_nci"],
            )
            if tmt_residuals.shape != expected_shape:
                raise ValueError(
                    f"TMT residual matrix shape={tmt_residuals.shape}; "
                    f"expected={expected_shape}."
                )
            if not np.isfinite(tmt_residuals).all():
                raise ValueError(
                    "TMT empirical residual matrix contains non-finite values."
                )

    return manifest, cal, grid, sf, tmt_residuals


def truth_table(grid, genes_per_scenario, universe, replicate_id):
    rows = []
    idx = 0

    for _, s in grid.iterrows():
        for within in range(genes_per_scenario):
            idx += 1
            rows.append({
                "sim_feature_id": f"SIM_R{replicate_id:04d}_{idx:05d}",
                "truth_signal": True,
                "scenario_id": s["scenario_id"],
                "geometry": s["geometry"],
                "penetrance": s["penetrance"],
                "outlier_count": s["outlier_count"],
                "actual_carrier_fraction": s["actual_carrier_fraction"],
                "effect_sd": s["effect_sd"],
                "scenario_feature_index": within + 1,
            })

    while idx < universe:
        idx += 1
        rows.append({
            "sim_feature_id": f"SIM_R{replicate_id:04d}_{idx:05d}",
            "truth_signal": False,
            "scenario_id": "NULL",
            "geometry": "NULL",
            "penetrance": np.nan,
            "outlier_count": np.nan,
            "actual_carrier_fraction": 0.0,
            "effect_sd": 0.0,
            "scenario_feature_index": np.nan,
        })

    return pd.DataFrame(rows)


def rna_multiplier_spec(
    rng,
    geometry,
    effect_sd,
    baseline_sd,
    baseline_values,
    ad_idx,
    penetrance,
    outlier_count,
):
    """Return multiplicative RNA signal factors on the linear count scale.

    Effect sizes are calibrated in empirical NCI SD units on
    log2(normalized_count + 0.5), but the actual RNA signal is injected as a
    multiplier on the expected count scale.

    VARIANCE_ONLY uses a lognormal draw followed by finite-sample arithmetic-mean
    normalization so mean(multiplier_AD) == 1 exactly while preserving the
    realized log2-scale SD.
    SYMMETRIC_TWO_TAIL is additionally rebalanced in simulate_rna using the
    realized anchored normalized expression values, so the carrier arithmetic
    mean is exactly preserved before the Poisson perturbation.
    """
    n = len(baseline_values)
    mult = np.ones(n, dtype=float)
    direction = int(rng.choice([-1, 1]))
    delta = float(effect_sd) * float(baseline_sd)
    baseline_ad = baseline_values[ad_idx]

    if geometry == "BROAD_MEAN_SHIFT":
        carriers = np.asarray(ad_idx)
        mult[carriers] = 2.0 ** (direction * delta)

    elif geometry == "RANDOM_SUBSET_SHIFT":
        k = max(1, int(round(float(penetrance) * len(ad_idx))))
        carriers = rng.choice(
            ad_idx,
            size=min(k, len(ad_idx)),
            replace=False,
        )
        mult[carriers] = 2.0 ** (direction * delta)

    elif geometry == "ONE_TAIL_AMPLIFICATION":
        k = max(1, int(round(float(penetrance) * len(ad_idx))))
        order = np.argsort(baseline_ad)
        loc = order[-k:] if direction > 0 else order[:k]
        carriers = np.asarray(ad_idx)[loc]
        mult[carriers] = 2.0 ** (direction * delta)

    elif geometry == "SYMMETRIC_TWO_TAIL":
        k = max(2, int(round(float(penetrance) * len(ad_idx))))
        k = min(k, len(ad_idx))
        lo_n = k // 2
        hi_n = k - lo_n

        order = np.argsort(baseline_ad)
        low = np.asarray(ad_idx)[order[:lo_n]]
        high = np.asarray(ad_idx)[order[-hi_n:]]
        carriers = np.concatenate([low, high])

        mult[low] = 2.0 ** (-delta)
        mult[high] = 2.0 ** delta
        direction = 0

    elif geometry == "ISOLATED_OUTLIERS":
        k = int(outlier_count)
        carriers = rng.choice(
            ad_idx,
            size=k,
            replace=False,
        )
        mult[carriers] = 2.0 ** (direction * delta)

    elif geometry == "VARIANCE_ONLY":
        carriers = np.asarray(ad_idx)
        tau = delta

        # eta is on the log2 scale. If eta ~ N(mu, tau^2), then
        # E[2^eta] = exp(mu*ln(2) + 0.5*tau^2*ln(2)^2).
        # Choosing mu = -0.5*ln(2)*tau^2 makes E[2^eta] = 1, so the
        # arithmetic expected normalized-expression mean is unchanged.
        eta_mean = -0.5 * math.log(2.0) * tau * tau
        eta = rng.normal(
            loc=eta_mean,
            scale=tau,
            size=len(carriers),
        )
        mult[carriers] = 2.0 ** eta

        # Make VARIANCE_ONLY a strict finite-sample mean-preserving
        # perturbation, not merely mean-preserving in expectation. Dividing
        # by the realized arithmetic mean leaves the log2-scale SD unchanged
        # (it is only a constant shift on the log2 scale) while ensuring that
        # mean(multiplier_AD) == 1 for every simulated feature.
        realized_mean = float(
            np.mean(
                mult[carriers]
            )
        )
        if not np.isfinite(realized_mean) or realized_mean <= 0:
            raise ValueError(
                "Invalid RNA variance-only realized mean multiplier."
            )
        mult[carriers] /= realized_mean
        direction = 0

    else:
        raise ValueError(geometry)

    return mult, np.asarray(carriers), direction, delta



def tmt_additive_shift_spec(
    rng,
    geometry,
    effect_sd,
    baseline_sd,
    baseline_values,
    ad_idx,
    penetrance,
    outlier_count,
):
    n = len(baseline_values)
    shift = np.zeros(n, dtype=float)
    direction = int(rng.choice([-1, 1]))
    delta = float(effect_sd) * float(baseline_sd)
    baseline_ad = baseline_values[ad_idx]

    if geometry == "BROAD_MEAN_SHIFT":
        carriers = np.asarray(ad_idx)
        shift[carriers] = direction * delta

    elif geometry == "RANDOM_SUBSET_SHIFT":
        k = max(1, int(round(float(penetrance) * len(ad_idx))))
        carriers = rng.choice(
            ad_idx,
            size=min(k, len(ad_idx)),
            replace=False,
        )
        shift[carriers] = direction * delta

    elif geometry == "ONE_TAIL_AMPLIFICATION":
        k = max(1, int(round(float(penetrance) * len(ad_idx))))
        order = np.argsort(baseline_ad)
        loc = order[-k:] if direction > 0 else order[:k]
        carriers = np.asarray(ad_idx)[loc]
        shift[carriers] = direction * delta

    elif geometry == "SYMMETRIC_TWO_TAIL":
        k = max(2, int(round(float(penetrance) * len(ad_idx))))
        k = min(k, len(ad_idx))
        lo_n = k // 2
        hi_n = k - lo_n
        order = np.argsort(baseline_ad)
        low = np.asarray(ad_idx)[order[:lo_n]]
        high = np.asarray(ad_idx)[order[-hi_n:]]
        carriers = np.concatenate([low, high])

        shift[low] = -delta
        shift[high] = delta
        shift[carriers] -= np.mean(shift[carriers])
        direction = 0

    elif geometry == "ISOLATED_OUTLIERS":
        k = int(outlier_count)
        carriers = rng.choice(ad_idx, size=k, replace=False)
        shift[carriers] = direction * delta

    elif geometry == "VARIANCE_ONLY":
        carriers = np.asarray(ad_idx)
        tau = delta
        shift[carriers] = rng.normal(
            0.0,
            tau,
            size=len(carriers),
        )
        shift[carriers] -= np.mean(shift[carriers])
        direction = 0

    else:
        raise ValueError(geometry)

    return shift, np.asarray(carriers), direction, delta


def sample_nb(rng, means, alpha):
    alpha = max(float(alpha), 1e-8)
    means = np.asarray(means, dtype=float)
    size = 1.0 / alpha
    prob = size / (size + means)
    prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
    return rng.negative_binomial(size, prob).astype(np.int64)


def simulate_rna(modality, cfg, manifest, cal, grid, sf, args, rep_dir, rng):
    truth = truth_table(
        grid,
        args.genes_per_scenario,
        cfg["universe"],
        args.replicate_id,
    )

    eligible = cal.loc[cal["eligible"]].reset_index(drop=True)
    pick = rng.integers(0, len(eligible), size=len(truth))
    templates = eligible.iloc[pick].reset_index(drop=True)

    truth["template_feature_id"] = templates["template_feature_id"].astype(str)
    truth["baseline_mean"] = templates["baseline_mean"].astype(float).to_numpy()
    truth["dispersion"] = templates["dispersion"].astype(float).to_numpy()
    truth["baseline_sd"] = templates["baseline_sd"].astype(float).to_numpy()

    n = len(manifest)
    counts = np.empty((len(truth), n), dtype=np.int64)

    for i, row in truth.iterrows():
        means = sf * float(row["baseline_mean"])
        counts[i] = sample_nb(
            rng,
            means,
            row["dispersion"],
        )

    groups = manifest["analysis_group"].astype(str).to_numpy()
    ad_idx = np.where(groups == "AD")[0]
    realized = []

    for i in np.where(truth["truth_signal"].to_numpy(bool))[0]:
        row = truth.iloc[i]
        baseline_norm = counts[i] / sf

        mult, carriers, direction, delta = rna_multiplier_spec(
            rng=rng,
            geometry=str(row["geometry"]),
            effect_sd=float(row["effect_sd"]),
            baseline_sd=float(row["baseline_sd"]),
            baseline_values=np.log2(baseline_norm + 0.5),
            ad_idx=ad_idx,
            penetrance=(
                float(row["penetrance"])
                if pd.notna(row["penetrance"])
                else np.nan
            ),
            outlier_count=(
                int(row["outlier_count"])
                if pd.notna(row["outlier_count"])
                else np.nan
            ),
        )

        geometry = str(row["geometry"])

        # Pre-signal arithmetic normalized-expression mean for audit only.
        # This is recorded before any scenario-specific perturbation and does
        # not affect the simulation RNG sequence or injected signal.
        pre_signal_norm_mean_ad = float(
            np.mean(
                baseline_norm[ad_idx]
            )
        )

        # For the RNA symmetric-two-tail scenario, equal +/- log2 shifts do
        # not generally preserve the arithmetic mean because the upper-tail
        # carriers have higher baseline abundance than the lower-tail carriers.
        # Rebalance the two carrier multipliers using the same realized
        # normalized abundance anchors that will be perturbed below. This
        # preserves the carrier arithmetic mean exactly while keeping low-tail
        # factors <1 and high-tail factors >1.
        if geometry == "SYMMETRIC_TWO_TAIL":
            anchor_norm = (
                np.maximum(
                    counts[i, carriers].astype(float),
                    0.25,
                )
                / sf[carriers]
            )
            denom = float(
                np.sum(
                    anchor_norm
                    * mult[carriers]
                )
            )
            numer = float(
                np.sum(anchor_norm)
            )
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError(
                    "Invalid symmetric-two-tail RNA mean-balance denominator."
                )
            mult[carriers] *= numer / denom

        changed = np.where(
            np.abs(mult - 1.0) > 1e-15
        )[0]

        # Audit the arithmetic mean implied by the injected perturbation
        # BEFORE the stochastic redraw. For SYMMETRIC_TWO_TAIL, the reference
        # uses the exact anchored values (including the 0.25 floor) used in
        # mean balancing, so expected/reference should be 1 up to floating-
        # point precision. For other RNA geometries this is descriptive.
        mean_balance_reference = baseline_norm.copy()
        mean_balance_expected = baseline_norm.copy()

        if geometry == "SYMMETRIC_TWO_TAIL":
            carrier_anchor_norm = (
                np.maximum(
                    counts[i, carriers].astype(float),
                    0.25,
                )
                / sf[carriers]
            )
            mean_balance_reference[carriers] = carrier_anchor_norm
            mean_balance_expected[carriers] = (
                carrier_anchor_norm
                * mult[carriers]
            )
        else:
            mean_balance_expected[changed] = (
                float(row["baseline_mean"])
                * mult[changed]
            )

        mean_balance_reference_ad = float(
            np.mean(
                mean_balance_reference[ad_idx]
            )
        )
        injected_expected_norm_mean_ad = float(
            np.mean(
                mean_balance_expected[ad_idx]
            )
        )
        injected_expected_mean_ratio_ad = (
            injected_expected_norm_mean_ad
            / mean_balance_reference_ad
            if mean_balance_reference_ad != 0
            else np.nan
        )

        if len(changed):
            if geometry in {
                "ONE_TAIL_AMPLIFICATION",
                "SYMMETRIC_TWO_TAIL",
            }:
                anchored = np.maximum(
                    counts[i, changed].astype(float),
                    0.25,
                )
                means = anchored * mult[changed]
                counts[i, changed] = rng.poisson(
                    means
                ).astype(np.int64)
            else:
                means = (
                    sf[changed]
                    * float(row["baseline_mean"])
                    * mult[changed]
                )
                counts[i, changed] = sample_nb(
                    rng,
                    means,
                    row["dispersion"],
                )

        mean_mult_ad = float(
            np.mean(
                mult[ad_idx]
            )
        )

        realized_norm_mean_ad = float(
            np.mean(
                counts[i, ad_idx]
                / sf[ad_idx]
            )
        )

        realized.append({
            "sim_feature_id": row["sim_feature_id"],
            "n_carriers": len(carriers),
            "direction": direction,
            "delta_log2_or_tau": delta,
            "mean_multiplier_AD": mean_mult_ad,
            "min_multiplier_AD": float(np.min(mult[ad_idx])),
            "max_multiplier_AD": float(np.max(mult[ad_idx])),
            "pre_signal_normalized_mean_AD": pre_signal_norm_mean_ad,
            "mean_balance_reference_AD": mean_balance_reference_ad,
            "injected_expected_normalized_mean_AD": (
                injected_expected_norm_mean_ad
            ),
            "injected_expected_mean_ratio_AD": (
                injected_expected_mean_ratio_ad
            ),
            "realized_normalized_mean_AD": realized_norm_mean_ad,
        })

    truth = truth.merge(
        pd.DataFrame(realized),
        on="sim_feature_id",
        how="left",
    )
    truth.loc[
        ~truth["truth_signal"],
        [
            "n_carriers",
            "direction",
            "delta_log2_or_tau",
            "mean_multiplier_AD",
            "min_multiplier_AD",
            "max_multiplier_AD",
            "pre_signal_normalized_mean_AD",
            "mean_balance_reference_AD",
            "injected_expected_normalized_mean_AD",
            "injected_expected_mean_ratio_AD",
            "realized_normalized_mean_AD",
        ],
    ] = [
        0,
        0,
        0.0,
        1.0,
        1.0,
        1.0,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    analysis_sf = deseq_ratio_size_factors(counts)
    norm = counts / analysis_sf[None, :]

    sample_ids = manifest["sample_id"].astype(str).tolist()

    raw_file = rep_dir / "simulated_raw_counts.tsv"
    norm_file = rep_dir / "simulated_normalized_counts_for_AREA.tsv"
    manifest_file = rep_dir / "simulated_manifest.csv"

    raw_df = pd.DataFrame(counts, columns=sample_ids)
    raw_df.insert(0, "gene_id", truth["sim_feature_id"])
    raw_df.to_csv(
        raw_file,
        sep="\t",
        index=False,
    )

    norm_df = pd.DataFrame(norm, columns=sample_ids)
    norm_df.insert(0, "gene_id", truth["sim_feature_id"])
    norm_df.to_csv(
        norm_file,
        sep="\t",
        index=False,
    )

    manifest.to_csv(
        manifest_file,
        index=False,
    )
    truth.to_csv(
        rep_dir / "simulation_truth.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    sf_corr = float(np.corrcoef(sf, analysis_sf)[0, 1])
    pd.DataFrame({
        "sample_id": sample_ids,
        "data_generating_size_factor": sf,
        "analysis_ratio_size_factor": analysis_sf,
    }).to_csv(
        rep_dir / "size_factor_audit.tsv",
        sep="\t",
        index=False,
    )

    return truth, raw_file, norm_file, manifest_file, sf_corr


def simulate_tmt(
    cfg,
    manifest,
    cal,
    grid,
    args,
    rep_dir,
    rng,
    tmt_residuals,
):
    truth = truth_table(
        grid,
        args.genes_per_scenario,
        cfg["universe"],
        args.replicate_id,
    )

    eligible = cal.loc[cal["eligible"]].reset_index(drop=True)
    pick = rng.integers(0, len(eligible), size=len(truth))
    templates = eligible.iloc[pick].reset_index(drop=True)

    truth["template_feature_id"] = templates["template_feature_id"].astype(str)
    truth["baseline_mean"] = templates["baseline_mean"].astype(float).to_numpy()
    truth["baseline_sd"] = templates["baseline_sd"].astype(float).to_numpy()

    n = len(manifest)
    x = np.empty((len(truth), n), dtype=float)

    if args.tmt_baseline == "empirical_bootstrap":
        if tmt_residuals is None:
            raise ValueError(
                "TMT empirical baseline requested but prepared NCI residuals "
                "are unavailable. Re-run --prepare --modality tmt."
            )

        cal_ids = cal["template_feature_id"].astype(str).tolist()
        residual_row = {
            feature_id: idx
            for idx, feature_id in enumerate(cal_ids)
        }

        for i, row in truth.iterrows():
            template_id = str(row["template_feature_id"])
            if template_id not in residual_row:
                raise ValueError(
                    f"TMT template missing from empirical residual matrix: "
                    f"{template_id}"
                )
            residual_pool = tmt_residuals[
                residual_row[template_id]
            ]
            sampled = rng.choice(
                residual_pool,
                size=n,
                replace=True,
            )
            x[i] = (
                float(row["baseline_mean"])
                + sampled
            )

    elif args.tmt_baseline == "gaussian":
        for i, row in truth.iterrows():
            x[i] = rng.normal(
                float(row["baseline_mean"]),
                float(row["baseline_sd"]),
                size=n,
            )

    else:
        raise ValueError(
            f"Unknown TMT baseline generator: {args.tmt_baseline}"
        )

    truth["tmt_baseline_generator"] = args.tmt_baseline

    groups = manifest["analysis_group"].astype(str).to_numpy()
    ad_idx = np.where(groups == "AD")[0]
    realized = []

    for i in np.where(truth["truth_signal"].to_numpy(bool))[0]:
        row = truth.iloc[i]

        shift, carriers, direction, delta = tmt_additive_shift_spec(
            rng=rng,
            geometry=str(row["geometry"]),
            effect_sd=float(row["effect_sd"]),
            baseline_sd=float(row["baseline_sd"]),
            baseline_values=x[i].copy(),
            ad_idx=ad_idx,
            penetrance=(
                float(row["penetrance"])
                if pd.notna(row["penetrance"])
                else np.nan
            ),
            outlier_count=(
                int(row["outlier_count"])
                if pd.notna(row["outlier_count"])
                else np.nan
            ),
        )

        x[i] += shift

        realized.append({
            "sim_feature_id": row["sim_feature_id"],
            "n_carriers": len(carriers),
            "direction": direction,
            "delta_log2_or_tau": delta,
        })

    truth = truth.merge(
        pd.DataFrame(realized),
        on="sim_feature_id",
        how="left",
    )
    truth.loc[
        ~truth["truth_signal"],
        ["n_carriers", "direction", "delta_log2_or_tau"],
    ] = [0, 0, 0.0]

    sample_ids = manifest["sample_id"].astype(str).tolist()
    matrix_file = rep_dir / "simulated_TMT_log2.csv"
    manifest_file = rep_dir / "simulated_manifest.csv"

    d = pd.DataFrame(x, columns=sample_ids)
    d.insert(0, "protein_id", truth["sim_feature_id"])
    d.to_csv(
        matrix_file,
        index=False,
    )

    manifest.to_csv(
        manifest_file,
        index=False,
    )
    truth.to_csv(
        rep_dir / "simulation_truth.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    return truth, matrix_file, manifest_file


def detect_id_column(d):
    for c in [
        "gene_id",
        "feature_id",
        "protein_id",
        "Gene",
        "gene",
        "id",
    ]:
        if c in d.columns:
            return c
    return d.columns[0]


def detect_col(d, candidates, label):
    for c in candidates:
        if c in d.columns:
            return c
    raise ValueError(
        f"Could not find {label}; columns={list(d.columns)}"
    )


def parse_result(path, method):
    d = pd.read_csv(path, low_memory=False)
    id_col = detect_id_column(d)

    if method == "AREA":
        pcol = detect_col(
            d,
            ["p_value", "pvalue", "p"],
            "AREA p",
        )
        qcol = detect_col(
            d,
            ["fdr", "padj", "FDR"],
            "AREA FDR",
        )
        stats = [
            "adjusted_Regular_AREA_Z",
            "z",
            "Z",
            "stat",
        ]

    elif method == "DESeq2":
        pcol = detect_col(
            d,
            ["pvalue", "p_value", "p"],
            "DESeq2 p",
        )
        qcol = detect_col(
            d,
            ["padj", "fdr", "FDR"],
            "DESeq2 FDR",
        )
        stats = ["stat", "Wald_stat"]

    elif method == "limma":
        pcol = detect_col(
            d,
            ["P.Value", "p_value", "pvalue"],
            "limma p",
        )
        qcol = detect_col(
            d,
            ["adj.P.Val", "fdr", "FDR"],
            "limma FDR",
        )
        stats = ["t", "stat"]

    else:
        raise ValueError(method)

    statcol = next(
        (c for c in stats if c in d.columns),
        None,
    )

    out = pd.DataFrame({
        "sim_feature_id": (
            d[id_col]
            .astype(str)
            .str.replace(r"\.\d+$", "", regex=True)
        ),
        f"{method}_p": pd.to_numeric(
            d[pcol],
            errors="coerce",
        ),
        f"{method}_fdr": pd.to_numeric(
            d[qcol],
            errors="coerce",
        ),
        f"{method}_stat": (
            pd.to_numeric(
                d[statcol],
                errors="coerce",
            )
            if statcol is not None
            else np.nan
        ),
    })
    return out


def find_one(directory, pattern):
    files = list(Path(directory).glob(pattern))
    if len(files) != 1:
        raise ValueError(
            f"Expected 1 file matching {pattern} in {directory}; "
            f"found {len(files)}: {files}"
        )
    return files[0]


def add_ranks(d, method, universe):
    p = pd.to_numeric(
        d[f"{method}_p"],
        errors="coerce",
    )
    rank = p.rank(
        method="min",
        ascending=True,
        na_option="bottom",
    )
    d[f"{method}_rank"] = rank
    d[f"{method}_rank_percentile"] = (
        rank - 1
    ) / max(
        1,
        universe - 1,
    )

    for k in TOPK_REFERENCE:
        d[f"{method}_TOP{k}"] = rank <= k


def run_models(modality, cfg, rep_dir, truth, sim_files):
    manifest_file = sim_files["manifest"]
    contrast = (
        f"Stage10_{modality}_rep"
        f"{sim_files['replicate_id']:04d}"
    )
    area_dir = rep_dir / "area"
    conventional_dir = rep_dir / "conventional"

    if cfg["kind"] == "rna":
        conventional_cmd = [
            str(RSCRIPT),
            str(DESEQ_RUNNER),
            "--counts",
            str(sim_files["raw"]),
            "--manifest",
            str(manifest_file),
            "--outdir",
            str(conventional_dir),
            "--contrast-name",
            contrast,
            "--group-a",
            "NCI",
            "--group-b",
            "AD",
            "--continuous",
            ",".join(cfg["continuous"]),
            "--categorical",
            ",".join(cfg["categorical"]),
        ]
        area_expr = sim_files["norm"]
        conventional_method = "DESeq2"

    else:
        conventional_cmd = [
            str(RSCRIPT),
            str(LIMMA_RUNNER),
            "--matrix",
            str(sim_files["matrix"]),
            "--manifest",
            str(manifest_file),
            "--outdir",
            str(conventional_dir),
            "--contrast-name",
            contrast,
            "--group-a",
            "NCI",
            "--group-b",
            "AD",
        ]
        area_expr = sim_files["matrix"]
        conventional_method = "limma"

    area_cmd = [
        sys.executable,
        str(AREA_RUNNER),
        "--expression",
        str(area_expr),
        "--manifest",
        str(manifest_file),
        "--outdir",
        str(area_dir),
        "--contrast-name",
        contrast,
        "--group-a",
        "NCI",
        "--group-b",
        "AD",
        "--orientation",
        "features_by_samples",
        "--continuous",
        ",".join(cfg["continuous"]),
        "--categorical",
        ",".join(cfg["categorical"]),
    ]

    if modality == "tmt":
        area_cmd.append("--global-complete-features")

    run_checked(
        conventional_cmd,
        rep_dir / "conventional_console.txt",
    )
    run_checked(
        area_cmd,
        rep_dir / "area_console.txt",
    )

    area_manifest_path = area_dir / "run_manifest.json"
    require(area_manifest_path, "AREA run manifest")

    with open(area_manifest_path) as f:
        area_manifest = json.load(f)

    if area_manifest.get("block") is not None:
        raise ValueError("AREA unexpectedly used a block.")

    if "residual-df-corrected" not in str(
        area_manifest.get("inference", "")
    ):
        raise ValueError(
            "AREA run does not report residual-df-corrected inference."
        )

    area_file = find_one(
        area_dir,
        "*_Regular_AREA_all_results.csv",
    )

    if conventional_method == "DESeq2":
        conv_file = find_one(
            conventional_dir,
            "*_DESeq2_all_results.csv",
        )
    else:
        candidates = list(conventional_dir.glob("*.csv"))
        candidates = [
            x
            for x in candidates
            if "FDR05" not in x.name
            and "manifest" not in x.name.lower()
            and "summary" not in x.name.lower()
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Could not uniquely identify limma all-results CSV: "
                f"{candidates}"
            )
        conv_file = candidates[0]

    a = parse_result(
        area_file,
        "AREA",
    )
    c = parse_result(
        conv_file,
        conventional_method,
    )

    out = (
        truth.merge(
            a,
            on="sim_feature_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            c,
            on="sim_feature_id",
            how="left",
            validate="one_to_one",
        )
    )

    out["AREA_detect_FDR05"] = out["AREA_fdr"] < FDR
    out[f"{conventional_method}_detect_FDR05"] = (
        out[f"{conventional_method}_fdr"] < FDR
    )
    out["AREA_detect_p05"] = out["AREA_p"] < 0.05
    out[f"{conventional_method}_detect_p05"] = (
        out[f"{conventional_method}_p"] < 0.05
    )

    add_ranks(
        out,
        "AREA",
        cfg["universe"],
    )
    add_ranks(
        out,
        conventional_method,
        cfg["universe"],
    )

    return out, conventional_method, area_manifest


def run_replicate(modality, args, base_root):
    cfg = CONFIG[modality]
    mod_root = base_root / modality
    rep = args.replicate_id

    rep_dir = (
        mod_root
        / "replicates"
        / f"replicate_{rep:04d}"
    )
    marker = rep_dir / "REPLICATE_COMPLETE.txt"

    if marker.exists() and not args.force:
        print(
            f"{modality} replicate {rep:04d} already complete."
        )
        return

    if rep_dir.exists() and args.force:
        shutil.rmtree(rep_dir)

    rep_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest, cal, grid, sf, tmt_residuals = load_prepared(
        modality,
        base_root,
    )
    validate_manifest(
        manifest,
        cfg,
    )

    rng = np.random.default_rng(
        args.seed
        + rep * 1000003
        + {
            "monocyte": 11,
            "dlpfc": 23,
            "tmt": 37,
        }[modality]
        * 100000000
    )

    if cfg["kind"] == "rna":
        (
            truth,
            raw_file,
            norm_file,
            manifest_file,
            sf_corr,
        ) = simulate_rna(
            modality,
            cfg,
            manifest,
            cal,
            grid,
            sf,
            args,
            rep_dir,
            rng,
        )
        sim_files = {
            "raw": raw_file,
            "norm": norm_file,
            "manifest": manifest_file,
            "replicate_id": rep,
        }

    else:
        (
            truth,
            matrix_file,
            manifest_file,
        ) = simulate_tmt(
            cfg,
            manifest,
            cal,
            grid,
            args,
            rep_dir,
            rng,
            tmt_residuals,
        )
        sf_corr = np.nan
        sim_files = {
            "matrix": matrix_file,
            "manifest": manifest_file,
            "replicate_id": rep,
        }

    combined, conventional_method, area_manifest = run_models(
        modality,
        cfg,
        rep_dir,
        truth,
        sim_files,
    )

    combined["modality"] = modality
    combined["conventional_method"] = conventional_method
    combined["replicate_id"] = rep

    combined.to_csv(
        rep_dir / "combined_method_results.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    qc = {
        "modality": modality,
        "replicate_id": rep,
        "n_features": len(combined),
        "n_signal": int(
            combined["truth_signal"].sum()
        ),
        "n_null": int(
            (
                ~combined["truth_signal"].astype(bool)
            ).sum()
        ),
        "size_factor_correlation": (
            None
            if not np.isfinite(sf_corr)
            else sf_corr
        ),
        "AREA_design_rank": area_manifest.get(
            "design_rank"
        ),
        "AREA_residual_df": area_manifest.get(
            "residual_df"
        ),
        "AREA_variance_df": area_manifest.get(
            "variance_df"
        ),
        "AREA_inference": area_manifest.get(
            "inference"
        ),
        "TMT_baseline_generator": (
            args.tmt_baseline
            if modality == "tmt"
            else None
        ),
    }

    with open(
        rep_dir / "replicate_qc.json",
        "w",
    ) as f:
        json.dump(
            qc,
            f,
            indent=2,
        )

    marker.write_text("complete\n")

    print(
        f"{modality} replicate {rep:04d} complete: "
        f"{len(combined):,} features."
    )


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan

    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    ctr = (
        p + z * z / (2 * n)
    ) / den
    half = (
        z
        * math.sqrt(
            p * (1 - p) / n
            + z * z / (4 * n * n)
        )
        / den
    )
    return max(0, ctr - half), min(1, ctr + half)


def summarize_modality(modality, args, base_root):
    cfg = CONFIG[modality]
    mod_root = base_root / modality

    files = sorted(
        mod_root.glob(
            "replicates/replicate_*/combined_method_results.tsv.gz"
        )
    )

    frames = []
    for f in files:
        if (
            f.parent
            / "REPLICATE_COMPLETE.txt"
        ).exists():
            frames.append(
                pd.read_csv(
                    f,
                    sep="\t",
                    low_memory=False,
                )
            )

    if not frames:
        raise FileNotFoundError(
            f"No completed {modality} replicates."
        )

    d = pd.concat(
        frames,
        ignore_index=True,
    )

    conventional = cfg["conventional"]

    summary_dir = mod_root / "summary"
    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nulls = d.loc[
        ~d["truth_signal"].astype(bool)
    ].copy()

    null_rows = []
    for method in ["AREA", conventional]:
        p = pd.to_numeric(
            nulls[f"{method}_p"],
            errors="coerce",
        )
        q = pd.to_numeric(
            nulls[f"{method}_fdr"],
            errors="coerce",
        )
        null_rows.append({
            "modality": modality,
            "method": method,
            "n_null_feature_instances": int(
                p.notna().sum()
            ),
            "null_p_lt_0.01": float(
                np.nanmean(p < 0.01)
            ),
            "null_p_lt_0.05": float(
                np.nanmean(p < 0.05)
            ),
            "null_p_median": float(
                np.nanmedian(p)
            ),
            "null_FDR05_call_fraction": float(
                np.nanmean(q < 0.05)
            ),
        })

    null_summary = pd.DataFrame(
        null_rows
    )

    signal = d.loc[
        d["truth_signal"].astype(bool)
    ].copy()

    group_cols = [
        "scenario_id",
        "geometry",
        "penetrance",
        "outlier_count",
        "actual_carrier_fraction",
        "effect_sd",
    ]

    method_rows = []
    paired_rows = []

    for keys, ds in signal.groupby(
        group_cols,
        dropna=False,
        sort=True,
    ):
        base = dict(
            zip(
                group_cols,
                keys,
            )
        )

        for method in ["AREA", conventional]:
            det = (
                ds[f"{method}_detect_FDR05"]
                .fillna(False)
                .astype(bool)
            )
            k = int(det.sum())
            n = len(det)
            lo, hi = wilson(k, n)

            row = {
                "modality": modality,
                **base,
                "method": method,
                "n_signal_instances": n,
                "power_FDR05": float(
                    det.mean()
                ),
                "power_FDR05_CI_low": lo,
                "power_FDR05_CI_high": hi,
                "power_p05": float(
                    ds[
                        f"{method}_detect_p05"
                    ]
                    .fillna(False)
                    .mean()
                ),
                "median_rank_percentile": float(
                    ds[
                        f"{method}_rank_percentile"
                    ].median()
                ),
            }

            for topk in TOPK_REFERENCE:
                row[f"TOP{topk}_recovery"] = float(
                    ds[f"{method}_TOP{topk}"]
                    .fillna(False)
                    .mean()
                )

            method_rows.append(row)

        a = (
            ds["AREA_detect_FDR05"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
        c = (
            ds[
                f"{conventional}_detect_FDR05"
            ]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        area_only = int(
            np.sum(
                a & ~c
            )
        )
        conv_only = int(
            np.sum(
                c & ~a
            )
        )

        discordant = area_only + conv_only
        pval = (
            float(
                binomtest(
                    area_only,
                    discordant,
                    0.5,
                ).pvalue
            )
            if discordant
            else 1.0
        )

        paired_rows.append({
            "modality": modality,
            **base,
            "n_signal_instances": len(ds),
            "AREA_power_FDR05": float(
                a.mean()
            ),
            f"{conventional}_power_FDR05": float(
                c.mean()
            ),
            "power_difference_AREA_minus_conventional": float(
                a.mean() - c.mean()
            ),
            "AREA_only_detected": area_only,
            "conventional_only_detected": conv_only,
            "both_detected": int(
                np.sum(
                    a & c
                )
            ),
            "neither_detected": int(
                np.sum(
                    ~a & ~c
                )
            ),
            "paired_exact_p": pval,
        })

    method_summary = pd.DataFrame(
        method_rows
    )
    paired = pd.DataFrame(
        paired_rows
    )
    paired["paired_exact_fdr"] = bh(
        paired["paired_exact_p"]
    )

    fdr_rows = []
    for rep, dr in d.groupby(
        "replicate_id"
    ):
        truth = (
            dr["truth_signal"]
            .astype(bool)
            .to_numpy()
        )

        for method in ["AREA", conventional]:
            called = (
                dr[
                    f"{method}_detect_FDR05"
                ]
                .fillna(False)
                .astype(bool)
                .to_numpy()
            )
            ncall = int(
                called.sum()
            )
            fp = int(
                np.sum(
                    called & ~truth
                )
            )
            tp = int(
                np.sum(
                    called & truth
                )
            )

            fdr_rows.append({
                "modality": modality,
                "replicate_id": rep,
                "method": method,
                "discoveries": ncall,
                "true_discoveries": tp,
                "false_discoveries": fp,
                "realized_FDR": (
                    fp / ncall
                    if ncall
                    else 0.0
                ),
            })

    fdr_rep = pd.DataFrame(
        fdr_rows
    )

    null_summary.to_csv(
        summary_dir / "null_calibration.tsv",
        sep="\t",
        index=False,
    )
    method_summary.to_csv(
        summary_dir / "scenario_method_summary.tsv",
        sep="\t",
        index=False,
    )
    paired.to_csv(
        summary_dir / "phase_diagram.tsv",
        sep="\t",
        index=False,
    )
    fdr_rep.to_csv(
        summary_dir / "realized_fdr_by_replicate.tsv",
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 100)
    print(f"STAGE 10 SUMMARY: {modality.upper()}")
    print("=" * 100)
    print(
        f"Replicates: "
        f"{d['replicate_id'].nunique()}"
    )
    print("\nNULL CALIBRATION")
    print(
        null_summary.to_string(
            index=False
        )
    )

    return {
        "null": null_summary,
        "method": method_summary,
        "paired": paired,
        "fdr": fdr_rep,
    }


def cross_modal_summary(base_root, modalities):
    pieces = []
    nulls = []
    fdrs = []

    for modality in modalities:
        sdir = base_root / modality / "summary"

        phase = sdir / "phase_diagram.tsv"
        null = sdir / "null_calibration.tsv"
        fdr = sdir / "realized_fdr_by_replicate.tsv"

        if phase.exists():
            pieces.append(
                pd.read_csv(
                    phase,
                    sep="\t",
                )
            )
        if null.exists():
            nulls.append(
                pd.read_csv(
                    null,
                    sep="\t",
                )
            )
        if fdr.exists():
            fdrs.append(
                pd.read_csv(
                    fdr,
                    sep="\t",
                )
            )

    if not pieces:
        return

    out = base_root / "cross_modality_summary"
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase = pd.concat(
        pieces,
        ignore_index=True,
    )
    phase.to_csv(
        out / "cross_modality_phase_diagram.tsv",
        sep="\t",
        index=False,
    )

    if nulls:
        pd.concat(
            nulls,
            ignore_index=True,
        ).to_csv(
            out / "cross_modality_null_calibration.tsv",
            sep="\t",
            index=False,
        )

    if fdrs:
        pd.concat(
            fdrs,
            ignore_index=True,
        ).to_csv(
            out / "cross_modality_realized_fdr.tsv",
            sep="\t",
            index=False,
        )

    print("\n" + "=" * 100)
    print("STAGE 10D CROSS-MODALITY SYNTHESIS")
    print("=" * 100)
    print(
        phase[
            [
                "modality",
                "scenario_id",
                "geometry",
                "penetrance",
                "outlier_count",
                "effect_sd",
                "AREA_power_FDR05",
                "power_difference_AREA_minus_conventional",
                "AREA_only_detected",
                "conventional_only_detected",
                "paired_exact_fdr",
            ]
        ]
        .sort_values(
            [
                "geometry",
                "penetrance",
                "outlier_count",
                "effect_sd",
                "modality",
            ],
            na_position="last",
        )
        .to_string(
            index=False
        )
    )


def selected_modalities(arg):
    if arg == "all":
        return [
            "monocyte",
            "dlpfc",
            "tmt",
        ]
    return [arg]


def main():
    args = parse_args()
    check_environment()

    if args.pilot:
        args.replicates = 3
        args.genes_per_scenario = 1
        base_root = OUTROOT / "pilot"
    else:
        base_root = OUTROOT

    mods = selected_modalities(
        args.modality
    )

    for modality in mods:
        spec = (
            base_root
            / modality
            / "specification/specification.json"
        )
        if args.prepare or not spec.exists():
            prepare_modality(
                modality,
                args,
                base_root,
            )

    if args.prepare:
        return

    if args.summarize:
        for modality in mods:
            summarize_modality(
                modality,
                args,
                base_root,
            )
        if len(mods) > 1:
            cross_modal_summary(
                base_root,
                mods,
            )
        return

    if args.replicate_id is not None:
        if (
            args.replicate_id < 1
            or args.replicate_id > args.replicates
        ):
            raise ValueError(
                f"--replicate-id must be between 1 and {args.replicates}"
            )

        for modality in mods:
            run_replicate(
                modality,
                args,
                base_root,
            )
        return

    for rep in range(
        1,
        args.replicates + 1,
    ):
        args.replicate_id = rep

        for modality in mods:
            run_replicate(
                modality,
                args,
                base_root,
            )

    for modality in mods:
        summarize_modality(
            modality,
            args,
            base_root,
        )

    if len(mods) > 1:
        cross_modal_summary(
            base_root,
            mods,
        )


if __name__ == "__main__":
    main()

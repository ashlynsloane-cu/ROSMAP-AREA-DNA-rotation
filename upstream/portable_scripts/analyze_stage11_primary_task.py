import os
#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT = Path(os.environ["ROSMAP_ANALYSIS_ROOT"]).expanduser().resolve()
TASK_FILE = (
    ROOT
    / "results/stage11_covariate_stress/"
      "stage11_tasks.tsv"
)

OUTROOT = (
    ROOT
    / "results/stage11_covariate_stress/"
      "primary_comparison"
)

TOPKS = [250, 500, 1000]


def bare_id(x):
    x = str(x)

    if x.startswith("ENSG"):
        return x.split(".")[0]

    return x


def choose_column(df, candidates, description, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Could not identify {description}. "
            f"Columns={list(df.columns)}"
        )

    return None


def read_analytic(path, label):
    d = pd.read_csv(
        path,
        low_memory=False,
    )

    id_col = choose_column(
        d,
        [
            "feature_id",
            "gene_id",
            "protein_id_raw",
            "protein_id",
        ],
        f"{label} feature ID",
    )

    p_col = choose_column(
        d,
        [
            "p_value",
            "pvalue",
            "p",
            "adjusted_pvalue",
        ],
        f"{label} p-value",
    )

    q_col = choose_column(
        d,
        [
            "fdr",
            "padj",
            "FDR",
            "adjusted_padj_BH",
        ],
        f"{label} FDR",
    )

    stat_col = choose_column(
        d,
        [
            "adjusted_Regular_AREA_Z",
            "adjusted_AREA_Z",
            "Z",
            "z",
            "stat",
        ],
        f"{label} statistic",
        required=False,
    )

    out = pd.DataFrame(
        {
            "feature_id": d[id_col].map(bare_id),
            f"{label}_p": pd.to_numeric(
                d[p_col],
                errors="coerce",
            ),
            f"{label}_fdr": pd.to_numeric(
                d[q_col],
                errors="coerce",
            ),
        }
    )

    if stat_col is not None:
        out[f"{label}_stat"] = pd.to_numeric(
            d[stat_col],
            errors="coerce",
        )
    else:
        out[f"{label}_stat"] = np.nan

    if out["feature_id"].duplicated().any():
        raise ValueError(
            f"{label}: duplicate normalized feature IDs"
        )

    return out, {
        "id_col": id_col,
        "p_col": p_col,
        "q_col": q_col,
        "stat_col": stat_col,
    }


def read_c(path):
    d = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    required = [
        "feature_id",
        "AREA_ES",
        "legacy_NES",
        "legacy_P",
        "legacy_FDR",
        "fullnull_Z",
        "fullnull_P",
        "fullnull_FDR",
    ]

    missing = [
        x for x in required
        if x not in d.columns
    ]

    if missing:
        raise ValueError(
            f"C missing columns: {missing}. "
            f"Observed={list(d.columns)}"
        )

    out = pd.DataFrame(
        {
            "feature_id": d["feature_id"].map(bare_id),
            "C_ES": pd.to_numeric(
                d["AREA_ES"],
                errors="coerce",
            ),
            "C_legacy_NES": pd.to_numeric(
                d["legacy_NES"],
                errors="coerce",
            ),
            "C_legacy_p": pd.to_numeric(
                d["legacy_P"],
                errors="coerce",
            ),
            "C_legacy_fdr": pd.to_numeric(
                d["legacy_FDR"],
                errors="coerce",
            ),
            "C_full_stat": pd.to_numeric(
                d["fullnull_Z"],
                errors="coerce",
            ),
            "C_full_p": pd.to_numeric(
                d["fullnull_P"],
                errors="coerce",
            ),
            "C_full_fdr": pd.to_numeric(
                d["fullnull_FDR"],
                errors="coerce",
            ),
        }
    )

    if out["feature_id"].duplicated().any():
        raise ValueError(
            "C: duplicate normalized feature IDs"
        )

    return out


def deterministic_rank(df, p_col):
    work = pd.DataFrame(
        {
            "feature_id": df["feature_id"],
            "p": pd.to_numeric(
                df[p_col],
                errors="coerce",
            ).fillna(np.inf),
        },
        index=df.index,
    )

    ordered = work.sort_values(
        ["p", "feature_id"],
        ascending=[True, True],
        kind="mergesort",
    ).index

    ranks = pd.Series(
        np.arange(
            1,
            len(df) + 1,
        ),
        index=ordered,
        dtype=int,
    )

    return ranks.reindex(
        df.index
    ).to_numpy()


def rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if good.sum() < 3:
        return np.nan

    return float(
        spearmanr(
            x[good],
            y[good],
        ).statistic
    )


def top_overlap(d, rank1, rank2, k):
    a = set(
        d.loc[
            d[rank1] <= k,
            "feature_id",
        ]
    )

    b = set(
        d.loc[
            d[rank2] <= k,
            "feature_id",
        ]
    )

    overlap = len(
        a & b
    )

    union = len(
        a | b
    )

    return (
        overlap,
        overlap / k,
        overlap / union,
    )


def fdr_pair(d, q1, q2):
    a = (
        pd.to_numeric(
            d[q1],
            errors="coerce",
        )
        < 0.05
    )

    b = (
        pd.to_numeric(
            d[q2],
            errors="coerce",
        )
        < 0.05
    )

    return {
        "sig_1": int(a.sum()),
        "sig_2": int(b.sum()),
        "shared": int(
            (a & b).sum()
        ),
        "only_1": int(
            (a & ~b).sum()
        ),
        "only_2": int(
            (~a & b).sum()
        ),
    }


parser = argparse.ArgumentParser()

parser.add_argument(
    "--task-id",
    type=int,
    required=True,
)

args = parser.parse_args()


tasks = pd.read_csv(
    TASK_FILE,
    sep="\t",
)

hit = tasks.loc[
    tasks["task_id"].eq(
        args.task_id
    )
]

if len(hit) != 1:
    raise RuntimeError(
        f"Expected one task for "
        f"{args.task_id}; got {len(hit)}"
    )

task = hit.iloc[0]


# ------------------------------------------------------------------
# Locate A, B, C
# ------------------------------------------------------------------

a_path = Path(
    task["A_result"]
)

b_root = Path(
    task["B_outdir"]
)

b_hits = sorted(
    b_root.glob(
        "*_Regular_AREA_all_results.csv"
    )
)

if len(b_hits) != 1:
    raise RuntimeError(
        f"Expected one B result in "
        f"{b_root}; found {b_hits}"
    )

b_path = b_hits[0]

c_path = (
    ROOT
    / "results/stage11_covariate_stress/"
      "C_original_area"
    / str(task["contrast_id"])
    / str(task["modality"])
    / "C_original_AREA_results.tsv"
)

if not a_path.exists():
    raise FileNotFoundError(a_path)

if not c_path.exists():
    raise FileNotFoundError(c_path)


# ------------------------------------------------------------------
# Read + merge exact feature universes
# ------------------------------------------------------------------

A, A_schema = read_analytic(
    a_path,
    "A",
)

B, B_schema = read_analytic(
    b_path,
    "B",
)

C = read_c(
    c_path
)

nA = len(A)
nB = len(B)
nC = len(C)

d = (
    A.merge(
        B,
        on="feature_id",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        C,
        on="feature_id",
        how="inner",
        validate="one_to_one",
    )
)

if not (
    len(d) == nA == nB == nC
):
    raise RuntimeError(
        "Feature-universe mismatch: "
        f"A={nA}, B={nB}, C={nC}, "
        f"intersection={len(d)}"
    )


# ------------------------------------------------------------------
# Deterministic significance ranks
# ------------------------------------------------------------------

rank_specs = {
    "A_rank": "A_p",
    "B_rank": "B_p",
    "C_full_rank": "C_full_p",
    "C_legacy_rank": "C_legacy_p",
}

for rank_col, p_col in rank_specs.items():
    d[rank_col] = deterministic_rank(
        d,
        p_col,
    )


# ------------------------------------------------------------------
# Primary rank correlations
# ------------------------------------------------------------------

row = {
    "task_id": int(
        task["task_id"]
    ),
    "contrast_id": str(
        task["contrast_id"]
    ),
    "family": str(
        task["family"]
    ),
    "modality": str(
        task["modality"]
    ),
    "cohort": str(
        task["cohort"]
    ),
    "n_features": int(
        len(d)
    ),

    "rho_rank_A_B": rho(
        d["A_rank"],
        d["B_rank"],
    ),

    "rho_rank_B_Cfull": rho(
        d["B_rank"],
        d["C_full_rank"],
    ),

    "rho_rank_A_Cfull": rho(
        d["A_rank"],
        d["C_full_rank"],
    ),

    "rho_rank_Clegacy_Cfull": rho(
        d["C_legacy_rank"],
        d["C_full_rank"],
    ),

    "rho_rank_B_Clegacy": rho(
        d["B_rank"],
        d["C_legacy_rank"],
    ),
}


# ------------------------------------------------------------------
# Analytic statistic agreement A vs B
# ------------------------------------------------------------------

row[
    "rho_stat_A_B_signed"
] = rho(
    d["A_stat"],
    d["B_stat"],
)

row[
    "rho_stat_A_B_absolute"
] = rho(
    np.abs(
        d["A_stat"]
    ),
    np.abs(
        d["B_stat"]
    ),
)


# ------------------------------------------------------------------
# Rank-shift magnitude
# ------------------------------------------------------------------

for left, right, name in [
    (
        "A_rank",
        "B_rank",
        "A_B",
    ),
    (
        "B_rank",
        "C_full_rank",
        "B_Cfull",
    ),
    (
        "A_rank",
        "C_full_rank",
        "A_Cfull",
    ),
]:
    delta = (
        np.abs(
            d[left]
            - d[right]
        )
        / len(d)
    )

    d[
        f"normalized_rank_shift_{name}"
    ] = delta

    row[
        f"median_normalized_rank_shift_{name}"
    ] = float(
        np.median(delta)
    )

    row[
        f"p90_normalized_rank_shift_{name}"
    ] = float(
        np.quantile(
            delta,
            0.90,
        )
    )


# ------------------------------------------------------------------
# TOP-k turnover
# ------------------------------------------------------------------

for k in TOPKS:
    for left, right, name in [
        (
            "A_rank",
            "B_rank",
            "A_B",
        ),
        (
            "B_rank",
            "C_full_rank",
            "B_Cfull",
        ),
        (
            "A_rank",
            "C_full_rank",
            "A_Cfull",
        ),
        (
            "C_legacy_rank",
            "C_full_rank",
            "Clegacy_Cfull",
        ),
    ]:
        overlap, frac, jaccard = (
            top_overlap(
                d,
                left,
                right,
                k,
            )
        )

        row[
            f"{name}_TOP{k}_overlap_n"
        ] = overlap

        row[
            f"{name}_TOP{k}_overlap_fraction"
        ] = frac

        row[
            f"{name}_TOP{k}_jaccard"
        ] = jaccard


# ------------------------------------------------------------------
# FDR calls
# ------------------------------------------------------------------

pairs = [
    (
        "A_fdr",
        "B_fdr",
        "A_B",
    ),
    (
        "B_fdr",
        "C_full_fdr",
        "B_Cfull",
    ),
    (
        "A_fdr",
        "C_full_fdr",
        "A_Cfull",
    ),
    (
        "C_legacy_fdr",
        "C_full_fdr",
        "Clegacy_Cfull",
    ),
]

for q1, q2, name in pairs:
    counts = fdr_pair(
        d,
        q1,
        q2,
    )

    for key, value in counts.items():
        row[
            f"{name}_{key}"
        ] = value


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

outdir = (
    OUTROOT
    / f"task_{args.task_id:02d}"
)

outdir.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame(
    [row]
).to_csv(
    outdir / "summary.tsv",
    sep="\t",
    index=False,
)

d.to_csv(
    outdir
    / "feature_level_comparison.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)

pd.DataFrame(
    [
        {
            "A_path": str(a_path),
            "B_path": str(b_path),
            "C_path": str(c_path),
            **{
                f"A_{k}": v
                for k, v
                in A_schema.items()
            },
            **{
                f"B_{k}": v
                for k, v
                in B_schema.items()
            },
        }
    ]
).to_csv(
    outdir / "schema_audit.tsv",
    sep="\t",
    index=False,
)

print(
    pd.DataFrame(
        [row]
    ).to_string(
        index=False
    )
)

print(
    f"\nSTAGE11_PRIMARY_ANALYSIS_COMPLETE "
    f"task={args.task_id}"
)

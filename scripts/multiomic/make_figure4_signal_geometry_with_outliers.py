#!/usr/bin/env python3
"""
Figure 4 — AREA sensitivity depends on signal geometry.

Scientific question
-------------------
What kind of disease-associated molecular signal produces the AREA-specific
behavior observed in ROSMAP?

Design
------
Each row shows the same controlled simulation in two linked views:

    LEFT  = the realized signal geometry in one representative DLPFC RNA
            simulated feature from the frozen Stage 10 replicate.

    RIGHT = the corresponding FDR<0.05 power difference
            (AREA - conventional) across all three modalities.

The five deliberately contrasting geometries are:

    A. Broad directional shift
    B. Sparse subset shift
    C. Symmetric two-tail spread
    D. Asymmetric mean-preserving redistribution
    E. Rare outlier shift (selected from frozen Stage 10 scenarios)

The figure is designed to make the mechanistic claim visible without requiring
the reader to know the simulation terminology.

Publication format
------------------
170 x 180 mm exact canvas, following the Hsp60/10 publication-figure
formatting system:

    - DejaVu Sans
    - compact typography
    - restrained slate-blue / cool-gray palette
    - minimal axes
    - very light gridlines
    - bold panel letters
    - no tight bounding-box export
    - PDF/SVG vector output
    - 600-dpi PNG

Data provenance
---------------
The figure reads the frozen Stage 10 production outputs. It does not rerun
simulations or recompute statistical inference.

Default Stage 10 root:
    ~/rosmap_multiomic_slurm/results/stage10_multimodal_simulations

Representative distribution:
    DLPFC RNA, replicate 1, scenario feature index 1

Frozen representative scenarios:
    S003  BROAD_MEAN_SHIFT
          100% penetrance, 2 SD

    S015  RANDOM_SUBSET_SHIFT
          20% penetrance, 2 SD

    S088  SYMMETRIC_TWO_TAIL
          100% penetrance, 3 SD

    S100  VARIANCE_ONLY
          100% penetrance, 2 SD

    E     Rare subset scenario chosen from frozen phase tables

These scenario choices are made from the pre-specified Stage 10 grid rather
than selected by looking for a favorable result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Hsp60/10 publication style
# ---------------------------------------------------------------------

FIG_WIDTH_MM = 170
FIG_HEIGHT_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FIG_HEIGHT_IN = FIG_HEIGHT_MM / 25.4
PNG_DPI = 600

INK = "#283038"
MID = "#66727C"
GRID = "#E5E9EC"
LIGHT = "#D5DADF"
AREA = "#7A8FA8"
CONV = "#B8C0CA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": PNG_DPI,
    }
)


# ---------------------------------------------------------------------
# Frozen figure design
# ---------------------------------------------------------------------

SCENARIOS = [
    {
        "letter": "A",
        "scenario_id": "S003",
        "geometry": "BROAD_MEAN_SHIFT",
        "penetrance": 1.0,
        "effect_sd": 2.0,
        "title": "Broad directional shift",
        "definition": "All AD participants shift in one direction.",
        "parameter_label": "100% penetrance · 2 SD",
    },
    {
        "letter": "B",
        "scenario_id": "S015",
        "geometry": "RANDOM_SUBSET_SHIFT",
        "penetrance": 0.20,
        "effect_sd": 2.0,
        "title": "Sparse subset shift",
        "definition": "20% of AD participants carry the shift.",
        "parameter_label": "20% penetrance · 2 SD",
    },
    {
        "letter": "C",
        "scenario_id": "S088",
        "geometry": "SYMMETRIC_TWO_TAIL",
        "penetrance": 1.0,
        "effect_sd": 3.0,
        "title": "Symmetric two-tail spread",
        "definition": "AD values expand in both tails.",
        "parameter_label": "100% penetrance · 3 SD",
    },
    {
        "letter": "D",
        "scenario_id": "S100",
        "geometry": "VARIANCE_ONLY",
        "penetrance": 1.0,
        "effect_sd": 2.0,
        "title": "Asymmetric rank redistribution",
        "definition": "Mean preserved; ranks redistribute asymmetrically.",
        "parameter_label": "mean multiplier = 1.00 · 2 SD",
    },
]

MODALITIES = [
    ("monocyte", "Monocyte RNA"),
    ("dlpfc", "DLPFC RNA"),
    ("tmt", "DLPFC TMT"),
]

MODALITY_ORDER = ["Monocyte RNA", "DLPFC RNA", "DLPFC TMT"]


# ---------------------------------------------------------------------
# Arguments and path discovery
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready ROSMAP AREA Figure 4."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="ROSMAP-AREA-DNA-rotation repository root.",
    )
    parser.add_argument(
        "--stage10-root",
        type=Path,
        default=(
            Path.home()
            / "rosmap_multiomic_slurm"
            / "results"
            / "stage10_multimodal_simulations"
        ),
        help="Frozen Stage 10 multimodal simulation output root.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Representative DLPFC RNA replicate used for geometry plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "repo/results/figures/figure_4_signal_geometry."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n  {path}"
        )
    return path


def locate_phase_source(stage10_root: Path, modality: str) -> Path:
    candidates = [
        (
            stage10_root
            / modality
            / "summary"
            / "phase_diagram.tsv"
        ),
        (
            stage10_root
            / modality
            / "summary"
            / "stage10_phase_diagram_table.tsv"
        ),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not locate Stage 10 phase-diagram table for {modality}. "
        "Tried:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def locate_replicate_file(
    stage10_root: Path,
    modality: str,
    replicate: int,
    filename: str,
) -> Path:
    return require_file(
        (
            stage10_root
            / modality
            / "replicates"
            / f"replicate_{replicate:04d}"
            / filename
        ),
        f"Stage 10 {modality} replicate file",
    )


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_phase_data(stage10_root: Path) -> pd.DataFrame:
    pieces = []

    for modality, modality_label in MODALITIES:
        source = locate_phase_source(stage10_root, modality)
        data = pd.read_csv(source, sep="\t")

        required = {
            "scenario_id",
            "geometry",
            "penetrance",
            "effect_sd",
            "AREA_power_FDR05",
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"{source} is missing required columns: {sorted(missing)}"
            )

        conventional_candidates = [
            c
            for c in data.columns
            if c.endswith("_power_FDR05")
            and c != "AREA_power_FDR05"
        ]

        if len(conventional_candidates) != 1:
            raise ValueError(
                f"Could not identify exactly one conventional power column "
                f"in {source}. Found: {conventional_candidates}"
            )

        conventional_col = conventional_candidates[0]

        data = data.copy()
        data["modality"] = modality_label
        data["conventional_method"] = conventional_col.replace(
            "_power_FDR05",
            "",
        )
        data["conventional_power_FDR05"] = pd.to_numeric(
            data[conventional_col],
            errors="coerce",
        )
        data["AREA_power_FDR05"] = pd.to_numeric(
            data["AREA_power_FDR05"],
            errors="coerce",
        )
        data["power_difference"] = (
            data["AREA_power_FDR05"]
            - data["conventional_power_FDR05"]
        )

        pieces.append(data)

    return pd.concat(
        pieces,
        ignore_index=True,
    )



def choose_outlier_scenario(phase: pd.DataFrame) -> dict:
    """Choose a real rare-subset scenario from frozen Stage 10 results.

    Favor lowest available penetrance and, within that, largest injected
    effect. No power values or distributions are fabricated.
    """
    candidates = phase.loc[
        (phase["geometry"].astype(str) == "RANDOM_SUBSET_SHIFT")
        & (pd.to_numeric(phase["penetrance"], errors="coerce") < 0.20)
    ].copy()
    if candidates.empty:
        raise ValueError(
            "Frozen Stage 10 has no RANDOM_SUBSET_SHIFT scenario "
            "with penetrance below 20%. Cannot make an honest outlier panel."
        )
    candidates["penetrance"] = pd.to_numeric(candidates["penetrance"])
    candidates["effect_sd"] = pd.to_numeric(candidates["effect_sd"])
    candidates = candidates.sort_values(
        ["penetrance", "effect_sd", "scenario_id"],
        ascending=[True, False, True],
    )
    row = candidates.iloc[0]
    penetration = float(row["penetrance"])
    effect = float(row["effect_sd"])
    return {
        "letter": "E",
        "scenario_id": str(row["scenario_id"]),
        "geometry": "RANDOM_SUBSET_SHIFT",
        "penetrance": penetration,
        "effect_sd": effect,
        "title": "Rare outlier shift",
        "definition": "A small fraction of AD participants carry extreme values.",
        "parameter_label": f"{penetration:.0%} penetrance · {effect:g} SD",
    }


def validate_phase_data(phase: pd.DataFrame) -> None:
    expected_modalities = set(MODALITY_ORDER)

    if set(phase["modality"]) != expected_modalities:
        raise ValueError(
            "Stage 10 phase table does not contain exactly the expected "
            f"modalities: {sorted(expected_modalities)}"
        )

    for scenario in SCENARIOS:
        subset = phase[
            phase["scenario_id"].astype(str)
            == scenario["scenario_id"]
        ].copy()

        if len(subset) != 3:
            raise ValueError(
                f"{scenario['scenario_id']}: expected exactly 3 modality "
                f"rows, found {len(subset)}."
            )

        if set(subset["modality"]) != expected_modalities:
            raise ValueError(
                f"{scenario['scenario_id']}: missing one or more modalities."
            )

        geometry = set(subset["geometry"].astype(str))
        if geometry != {scenario["geometry"]}:
            raise ValueError(
                f"{scenario['scenario_id']}: geometry mismatch. "
                f"Expected {scenario['geometry']}; found {sorted(geometry)}."
            )

        penetrance = subset["penetrance"].dropna().unique()
        if len(penetrance) != 1 or not np.isclose(
            penetrance[0],
            scenario["penetrance"],
        ):
            raise ValueError(
                f"{scenario['scenario_id']}: penetrance mismatch. "
                f"Expected {scenario['penetrance']}; found {penetrance}."
            )

        effects = subset["effect_sd"].dropna().unique()
        if len(effects) != 1 or not np.isclose(
            effects[0],
            scenario["effect_sd"],
        ):
            raise ValueError(
                f"{scenario['scenario_id']}: effect mismatch. "
                f"Expected {scenario['effect_sd']}; found {effects}."
            )

        if not np.isfinite(subset["power_difference"]).all():
            raise ValueError(
                f"{scenario['scenario_id']}: non-finite power difference."
            )


def load_representative_dlpfc(
    stage10_root: Path,
    replicate: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
    Path,
]:
    matrix_path = locate_replicate_file(
        stage10_root,
        "dlpfc",
        replicate,
        "simulated_normalized_counts_for_AREA.tsv",
    )
    truth_path = locate_replicate_file(
        stage10_root,
        "dlpfc",
        replicate,
        "simulation_truth.tsv.gz",
    )
    manifest_path = locate_replicate_file(
        stage10_root,
        "dlpfc",
        replicate,
        "simulated_manifest.csv",
    )

    truth = pd.read_csv(
        truth_path,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )
    manifest = pd.read_csv(manifest_path)

    if "sim_feature_id" not in truth.columns:
        raise ValueError(
            f"{truth_path}: missing sim_feature_id."
        )

    if "analysis_group" not in manifest.columns:
        raise ValueError(
            f"{manifest_path}: missing analysis_group."
        )

    matrix = pd.read_csv(
        matrix_path,
        sep="\t",
        low_memory=False,
    )

    if "gene_id" not in matrix.columns:
        raise ValueError(
            f"{matrix_path}: expected feature ID column 'gene_id'. "
            f"Found first columns: {list(matrix.columns[:5])}"
        )

    matrix["gene_id"] = matrix["gene_id"].astype(str)
    truth["sim_feature_id"] = truth["sim_feature_id"].astype(str)

    return matrix, truth, manifest, matrix_path, truth_path


def extract_signal_geometry_data(
    matrix: pd.DataFrame,
    truth: pd.DataFrame,
    scenario: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    selected = truth[
        (truth["scenario_id"].astype(str) == scenario["scenario_id"])
        & truth["truth_signal"].astype(bool)
        & (
            pd.to_numeric(
                truth["scenario_feature_index"],
                errors="coerce",
            )
            == 1
        )
    ].copy()

    if len(selected) != 1:
        raise ValueError(
            f"{scenario['scenario_id']}: expected exactly one "
            "scenario_feature_index=1 signal feature; found "
            f"{len(selected)}."
        )

    row = selected.iloc[0]

    sim_id = str(row["sim_feature_id"])
    values = matrix.loc[
        matrix["gene_id"] == sim_id
    ].copy()

    if len(values) != 1:
        raise ValueError(
            f"{scenario['scenario_id']}: could not uniquely locate "
            f"{sim_id} in normalized matrix."
        )

    sample_columns = [
        c
        for c in matrix.columns
        if c != "gene_id"
    ]

    sample_values = pd.to_numeric(
        values.iloc[0][sample_columns],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(sample_values).all():
        raise ValueError(
            f"{scenario['scenario_id']}: non-finite normalized values."
        )

    metadata = {
        "sim_feature_id": sim_id,
        "direction": int(row.get("direction", 0)),
        "n_carriers": int(row.get("n_carriers", 0)),
        "mean_multiplier_AD": float(
            row.get("mean_multiplier_AD", np.nan)
        ),
    }

    return sample_values, np.asarray(sample_columns), metadata


def standardize_to_nci(
    values: np.ndarray,
    sample_ids: np.ndarray,
    manifest: pd.DataFrame,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    manifest_ids = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["analysis_group"].astype(str).to_numpy()

    lookup = {
        sample_id: i
        for i, sample_id in enumerate(manifest_ids)
    }

    try:
        order = np.asarray(
            [lookup[str(sample_id)] for sample_id in sample_ids],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(
            f"Normalized matrix sample ID not found in manifest: {exc}"
        ) from exc

    groups_in_matrix_order = groups[order]

    nci = groups_in_matrix_order == "NCI"
    ad = groups_in_matrix_order == "AD"

    if nci.sum() < 20 or ad.sum() < 20:
        raise ValueError(
            "Unexpectedly small NCI/AD groups in representative simulation."
        )

    nci_values = values[nci]
    ad_values = values[ad]

    nci_log = np.log2(nci_values + 0.5)
    ad_log = np.log2(ad_values + 0.5)

    center = float(np.mean(nci_log))
    scale = float(np.std(nci_log, ddof=1))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "NCI reference SD is non-positive or non-finite."
        )

    nci_z = (nci_log - center) / scale
    ad_z = (ad_log - center) / scale

    if direction in (-1, 1):
        nci_z = nci_z * direction
        ad_z = ad_z * direction

    return nci_z, ad_z


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def style_axis(ax, xgrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    ax.tick_params(
        axis="x",
        labelsize=5.8,
        length=2.4,
        width=0.5,
        color=INK,
        labelcolor=INK,
        pad=2,
    )
    ax.tick_params(
        axis="y",
        labelsize=6.1,
        length=0,
        color=INK,
        labelcolor=INK,
        pad=2,
    )

    if xgrid:
        ax.grid(
            axis="x",
            color=GRID,
            linewidth=0.45,
            zorder=0,
        )
        ax.set_axisbelow(True)


def add_panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.15,
        1.285,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def add_row_heading(
    ax,
    title: str,
    definition: str,
    parameter_label: str,
) -> None:
    ax.text(
        0.0,
        1.285,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )

    ax.text(
        0.0,
        1.145,
        definition,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=MID,
        clip_on=False,
    )

    ax.text(
        0.0,
        1.035,
        parameter_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.9,
        color=MID,
        clip_on=False,
    )


def plot_geometry(
    ax,
    nci_z: np.ndarray,
    ad_z: np.ndarray,
    show_xlabel: bool,
) -> None:
    bins = np.linspace(-4.0, 4.0, 25)

    ax.hist(
        nci_z,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=CONV,
        alpha=0.42,
        edgecolor=CONV,
        linewidth=0.8,
        label="NCI",
        zorder=2,
    )
    ax.hist(
        ad_z,
        bins=bins,
        density=True,
        histtype="step",
        color=AREA,
        linewidth=1.1,
        label="AD",
        zorder=3,
    )

    ax.axvline(
        0.0,
        color=LIGHT,
        linewidth=0.7,
        zorder=1,
    )

    ax.set_xlim(-4.0, 4.0)
    ax.set_yticks([])
    ax.set_ylabel("")
    style_axis(ax)
    ax.spines["left"].set_visible(False)

    if show_xlabel:
        ax.set_xlabel(
            "Expression relative to NCI mean (SD)",
            fontsize=6.8,
            fontweight="bold",
            color=INK,
            labelpad=3,
        )
    else:
        ax.tick_params(
            axis="x",
            labelbottom=False,
        )



def plot_power_difference(
    ax,
    subset: pd.DataFrame,
    show_xlabel: bool,
) -> None:
    subset = subset.copy()

    subset["modality"] = pd.Categorical(
        subset["modality"],
        categories=MODALITY_ORDER,
        ordered=True,
    )
    subset = subset.sort_values("modality")

    y = np.arange(len(MODALITY_ORDER))[::-1]
    values = (
        subset.set_index("modality")
        .loc[MODALITY_ORDER, "power_difference"]
        .to_numpy(dtype=float)
        * 100.0
    )

    ax.axvline(
        0.0,
        color=MID,
        linewidth=0.75,
        linestyle="--",
        zorder=1,
    )

    for yi, value in zip(y, values):
        ax.plot(
            [0.0, value],
            [yi, yi],
            color=LIGHT,
            linewidth=0.9,
            zorder=2,
        )
        ax.scatter(
            value,
            yi,
            s=28,
            color=AREA if value > 0 else CONV,
            edgecolor="none",
            zorder=3,
        )

        label = f"{value:+.0f} pp"

        if value >= 0:
            # Keep large positive labels inside the plotting area.
            if value > 68:
                ha = "right"
                x_text = value - 3.0
            else:
                ha = "left"
                x_text = value + 3.0
        else:
            ha = "left"
            x_text = value + 3.0

        ax.text(
            x_text,
            yi,
            label,
            ha=ha,
            va="center",
            fontsize=5.8,
            color=INK,
        )

    ax.set_yticks(
        y,
        MODALITY_ORDER,
    )
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(-40, 80)
    ax.set_xticks([-40, 0, 40, 80])

    if show_xlabel:
        ax.set_xlabel(
            "AREA − conventional power (percentage points)",
            fontsize=6.8,
            fontweight="bold",
            color=INK,
            labelpad=3,
        )
    else:
        ax.tick_params(
            axis="x",
            labelbottom=False,
        )

    style_axis(ax)



def make_figure(
    phase: pd.DataFrame,
    matrix: pd.DataFrame,
    truth: pd.DataFrame,
    manifest: pd.DataFrame,
    replicate: int,
    output_dir: Path,
    matrix_path: Path,
    truth_path: Path,
    stage10_root: Path,
) -> None:
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        facecolor="white",
    )

    gs = fig.add_gridspec(
        nrows=len(SCENARIOS),
        ncols=2,
        width_ratios=[1.0, 1.08],
        left=0.095,
        right=0.975,
        top=0.845,
        bottom=0.105,
        hspace=1.55,
        wspace=0.68,
    )

    fig.text(
        0.06,
        0.970,
        "AREA sensitivity depends on signal geometry",
        ha="left",
        va="top",
        fontsize=11.3,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.925,
        (
            "ROSMAP-calibrated simulations identify directional/asymmetric "
            "rank redistribution as the clearest AREA-favorable regime."
        ),
        ha="left",
        va="top",
        fontsize=7.3,
        color=MID,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=AREA,
            linewidth=1.1,
            label="AD",
        ),
        Line2D(
            [0],
            [0],
            color=CONV,
            linewidth=4.0,
            alpha=0.42,
            label="NCI",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.285, 0.865),
        frameon=False,
        ncol=2,
        fontsize=6.0,
        handlelength=1.3,
        columnspacing=0.8,
    )

    for i, scenario in enumerate(SCENARIOS):
        ax_geometry = fig.add_subplot(gs[i, 0])
        ax_power = fig.add_subplot(gs[i, 1])

        add_panel_letter(
            ax_geometry,
            scenario["letter"],
        )
        add_row_heading(
            ax_geometry,
            scenario["title"],
            scenario["definition"],
            scenario["parameter_label"],
        )

        sample_values, sample_ids, metadata = extract_signal_geometry_data(
            matrix,
            truth,
            scenario,
        )

        nci_z, ad_z = standardize_to_nci(
            sample_values,
            sample_ids,
            manifest,
            metadata["direction"],
        )

        plot_geometry(
            ax_geometry,
            nci_z,
            ad_z,
            show_xlabel=(i == len(SCENARIOS) - 1),
        )

        power = phase[
            phase["scenario_id"].astype(str)
            == scenario["scenario_id"]
        ].copy()

        plot_power_difference(
            ax_power,
            power,
            show_xlabel=(i == len(SCENARIOS) - 1),
        )


    fig.text(
        0.06,
        0.035,
        (
            "Power difference = AREA − conventional power at FDR < 0.05. "
            "Distributions show one frozen DLPFC RNA feature per scenario.\n"
            "Directional signals are oriented right."
        ),
        ha="left",
        va="bottom",
        fontsize=4.9,
        linespacing=1.15,
        color=MID,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = output_dir / "Figure4_signal_geometry_with_outliers"

    # Exact canvas export: intentionally no bbox_inches='tight'.
    fig.savefig(
        stem.with_suffix(".pdf"),
        format="pdf",
    )
    fig.savefig(
        stem.with_suffix(".svg"),
        format="svg",
    )
    fig.savefig(
        stem.with_suffix(".png"),
        format="png",
        dpi=PNG_DPI,
    )

    audit_rows = []
    for scenario in SCENARIOS:
        rows = phase[
            phase["scenario_id"].astype(str)
            == scenario["scenario_id"]
        ].copy()

        for _, row in rows.iterrows():
            audit_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "geometry": scenario["geometry"],
                    "penetrance": scenario["penetrance"],
                    "effect_sd": scenario["effect_sd"],
                    "modality": row["modality"],
                    "AREA_power_FDR05": row["AREA_power_FDR05"],
                    "conventional_method": row["conventional_method"],
                    "conventional_power_FDR05": (
                        row["conventional_power_FDR05"]
                    ),
                    "AREA_minus_conventional_power": (
                        row["power_difference"]
                    ),
                }
            )

    pd.DataFrame(audit_rows).to_csv(
        output_dir / "Figure4_signal_geometry_audit.csv",
        index=False,
    )

    notes = [
        "Figure 4 — AREA signal geometry",
        "",
        f"Stage 10 root: {stage10_root}",
        f"Representative DLPFC replicate: {replicate}",
        f"Distribution matrix: {matrix_path}",
        f"Distribution truth: {truth_path}",
        "",
        "Frozen representative scenarios:",
    ]

    for scenario in SCENARIOS:
        notes.append(
            (
                f"  {scenario['scenario_id']} | "
                f"{scenario['geometry']} | "
                f"penetrance={scenario['penetrance']} | "
                f"effect_sd={scenario['effect_sd']}"
            )
        )

    notes.extend(
        [
            "",
            "The script reads frozen Stage 10 outputs and does not rerun",
            "simulations or recompute inference.",
            "",
            "Directional scenarios are visually oriented so the injected",
            "disease direction points right; this is a display transformation only.",
        ]
    )

    (output_dir / "Figure4_signal_geometry_notes.txt").write_text(
        "\n".join(notes) + "\n"
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    stage10_root = args.stage10_root.expanduser().resolve()

    if args.output_dir is None:
        output_dir = (
            repo_root
            / "results"
            / "figures"
            / "figure_4_signal_geometry"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()

    phase = load_phase_data(stage10_root)
    SCENARIOS.append(choose_outlier_scenario(phase))
    validate_phase_data(phase)

    matrix, truth, manifest, matrix_path, truth_path = (
        load_representative_dlpfc(
            stage10_root,
            args.replicate,
        )
    )

    make_figure(
        phase=phase,
        matrix=matrix,
        truth=truth,
        manifest=manifest,
        replicate=args.replicate,
        output_dir=output_dir,
        matrix_path=matrix_path,
        truth_path=truth_path,
        stage10_root=stage10_root,
    )

    print(f"Saved: {output_dir / 'Figure4_signal_geometry_manuscript.pdf'}")
    print(f"Saved: {output_dir / 'Figure4_signal_geometry_manuscript.svg'}")
    print(f"Saved: {output_dir / 'Figure4_signal_geometry_manuscript.png'}")
    print(f"Saved: {output_dir / 'Figure4_signal_geometry_audit.csv'}")
    print(f"Saved: {output_dir / 'Figure4_signal_geometry_notes.txt'}")


if __name__ == "__main__":
    main()

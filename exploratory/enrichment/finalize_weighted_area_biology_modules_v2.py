#!/usr/bin/env python3
"""
finalize_weighted_area_biology_modules.py
=========================================

Second-stage refinement of Weighted AREA biological modules.

This script addresses the two main limitations of the first module pass:

1. Hundreds of representative pathways were still unassigned.
2. A few broad modules mixed pathways with opposite biological directions.

The goal is NOT to force every pathway into a category. The goal is to produce
a smaller, directionally coherent, publication-interpretable set of modules
while preserving an audit trail for everything left unassigned.

INPUTS
------
From the broad synthesis:
    results/weighted_area_biology/pathway_long_all.csv
    results/weighted_area_biology/representative_pathways.csv

From the first module pass:
    results/weighted_area_biology_refined/module_assignment_audit.csv
    results/weighted_area_biology_refined/unassigned_representative_pathways.csv

OUTPUTS
-------
results/weighted_area_biology_final/

Tables:
    final_module_assignment_audit.csv
    final_unassigned_representative_pathways.csv
    final_module_summary.csv
    final_module_phenotype_profiles.csv
    final_module_axis_profiles.csv
    final_module_driver_genes.csv
    final_module_representative_pathways.csv
    final_module_rules.csv
    final_priority_modules.csv

Figures:
    final_module_phenotype_heatmap.png/.pdf
    final_module_axis_profiles.png/.pdf
    final_module_driver_genes.png/.pdf
    final_module_coverage.png/.pdf

DESIGN PRINCIPLES
-----------------
- Explicit keyword rules only; every assignment is auditable.
- More specific rules are evaluated before broad rules.
- Direction-sensitive rules are used where biology would otherwise be mixed.
- Module summaries use nonredundant representative pathways.
- "Shared core" vs "axis-enhanced" is based on effect magnitude across the
  three phenotype axes, not merely FDR thresholding.
- Truly axis-specific calls are intentionally strict.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PHENOTYPE STRUCTURE
# =============================================================================

PHENOTYPES = [
    "Cognitive_stage",
    "cogn_global_impairment",
    "mmse_impairment",
    "CERAD_equal",
    "CERAD_amyloid_calibrated",
    "amyloid_continuous",
    "Braak_equal",
    "Braak_tangle_calibrated",
    "tangle_continuous",
]

PHENOTYPE_LABELS = {
    "Cognitive_stage": "Cognitive\nstage",
    "cogn_global_impairment": "Global\ncognition",
    "mmse_impairment": "MMSE",
    "CERAD_equal": "CERAD\nequal",
    "CERAD_amyloid_calibrated": "CERAD\ncalibrated",
    "amyloid_continuous": "Amyloid\ncontinuous",
    "Braak_equal": "Braak\nequal",
    "Braak_tangle_calibrated": "Braak\ncalibrated",
    "tangle_continuous": "Tangles\ncontinuous",
}

AXIS_MAP = {
    "Cognitive_stage": "Cognition",
    "cogn_global_impairment": "Cognition",
    "mmse_impairment": "Cognition",
    "CERAD_equal": "Amyloid",
    "CERAD_amyloid_calibrated": "Amyloid",
    "amyloid_continuous": "Amyloid",
    "Braak_equal": "Tau",
    "Braak_tangle_calibrated": "Tau",
    "tangle_continuous": "Tau",
}

AXES = ["Cognition", "Amyloid", "Tau"]

ENHANCEMENT_DELTA = 0.25
MIN_AXIS_SIGNAL_PHENOTYPES = 2
OTHER_AXIS_WEAK_ABS_NES = 1.30

MODULE_SIG_FRACTION_FOR_DOT = 0.50

MIN_DRIVER_RECURRENCE = 2
MIN_DRIVER_SPECIFICITY = 0.35


# =============================================================================
# FINAL MODULE RULES
# =============================================================================
#
# direction:
#     None       -> any direction
#     "Positive" -> pathway representative must have positive broad direction
#     "Negative" -> pathway representative must have negative broad direction
#
# Rules are intentionally ordered from specific to broad.
# =============================================================================

RULES = [
    # -------------------------------------------------------------------------
    # Mitochondrial / energy biology
    # -------------------------------------------------------------------------
    {
        "module": "Mitochondrial energetics",
        "direction": "Negative",
        "include": [
            r"oxidative phosphorylation",
            r"respiratory chain",
            r"respiratory electron transport",
            r"electron transport chain",
            r"aerobic respiration",
            r"cellular respiration",
            r"proton transmembrane transport",
            r"atp biosynthetic",
            r"atp synth",
            r"proton motive force",
            r"chemiosmotic",
            r"complex i assembly",
            r"complex i biogenesis",
            r"complex iii assembly",
            r"complex iv",
            r"cytochrome c oxidase",
            r"citric acid cycle",
            r"tricarboxylic acid cycle",
            r"tca cycle",
            r"energy derivation by oxidation",
        ],
        "exclude": [
            r"mitophagy",
            r"translation",
            r"protein import",
            r"calcium",
        ],
    },
    {
        "module": "Mitochondrial translation",
        "direction": "Negative",
        "include": [
            r"mitochondrial translation",
            r"mitochondrial ribosom",
            r"mitochondrial trna",
        ],
        "exclude": [],
    },
    {
        "module": "Mitochondrial quality control",
        "direction": "Negative",
        "include": [
            r"mitophagy",
            r"mitochondrial protein import",
            r"protein targeting to mitochondrion",
            r"protein targeting to mitochondria",
            r"import into the mitochondrion",
            r"mitochondrial protein degradation",
            r"cristae formation",
            r"mitochondrial organization",
            r"mitochondrial membrane organization",
            r"mitochondrial dynamics",
            r"pink1",
            r"prkn",
            r"parkin",
        ],
        "exclude": [],
    },
    {
        "module": "Mitochondrial ion homeostasis",
        "direction": "Negative",
        "include": [
            r"mitochondrial calcium",
            r"mitochondrial ion",
        ],
        "exclude": [],
    },

    # -------------------------------------------------------------------------
    # Protein synthesis / proteostasis / trafficking
    # -------------------------------------------------------------------------
    {
        "module": "Translation / tRNA biology",
        "direction": "Negative",
        "include": [
            r"\btranslation\b",
            r"translational initiation",
            r"aminoacylation",
            r"\btrna\b",
            r"ribosome",
            r"ribosomal",
            r"peptide biosynthetic",
            r"protein targeting to er",
            r"protein targeting to membrane",
            r"cotranslational",
        ],
        "exclude": [
            r"mitochondrial translation",
        ],
    },
    {
        "module": "Protein folding / proteostasis",
        "direction": "Negative",
        "include": [
            r"prefoldin",
            r"\btric\b",
            r"\bcct\b",
            r"protein folding",
            r"chaperon",
            r"proteasom",
            r"ubiquitin",
            r"protein deneddylation",
            r"protein neddylation",
            r"unfolded protein",
            r"protein quality control",
        ],
        "exclude": [],
    },
    {
        "module": "Vesicle / intracellular transport",
        "direction": "Negative",
        "include": [
            r"vesicle coating",
            r"vesicle transport",
            r"protein localization",
            r"intracellular transport",
            r"endocyt",
            r"clathrin",
            r"golgi",
            r"endosomal",
            r"protein secretion",
            r"insertion of tail-anchored",
            r"translocation of slc2a4",
        ],
        "exclude": [
            r"synaptic vesicle",
        ],
    },

    # -------------------------------------------------------------------------
    # Neuronal biology
    # -------------------------------------------------------------------------
    {
        "module": "Synaptic transmission",
        "direction": "Negative",
        "include": [
            r"synap",
            r"neurotransmitter",
            r"glutamate release",
            r"gaba",
            r"acetylcholine",
            r"vesicle exocytosis",
            r"neurexin",
            r"neuroligin",
            r"nmda",
            r"ampa receptor",
            r"glur2",
            r"clostridium toxin",
            r"lgi-adam",
        ],
        "exclude": [],
    },
    {
        "module": "Neurite / axon guidance",
        "direction": "Negative",
        "include": [
            r"slit",
            r"robo",
            r"axon guidance",
            r"dendrite",
            r"neurotrophin",
            r"l1 and ankyrin",
            r"neurite",
        ],
        "exclude": [],
    },

    # -------------------------------------------------------------------------
    # Signaling / remodeling
    # -------------------------------------------------------------------------
    {
        "module": "RHO / cytoskeletal signaling",
        "direction": "Positive",
        "include": [
            r"rho[a-z]* gtpase",
            r"\brhoa\b",
            r"\brhob\b",
            r"\brhoc\b",
            r"cdc42",
            r"rac1",
            r"small gtpase",
            r"actin filament",
            r"cytoskeleton",
            r"microtubule",
            r"tubulin",
            r"cell motility",
        ],
        "exclude": [
            r"vesicle transport along microtubule",
        ],
    },
    {
        "module": "TGF / ECM remodeling",
        "direction": "Positive",
        "include": [
            r"\btgf",
            r"tgfbr",
            r"extracellular matrix",
            r"\becm\b",
            r"integrin",
            r"collagen",
            r"elastic fibre",
            r"dermatan sulfate",
            r"focal adhesion",
            r"matrix organization",
            r"cadherin",
            r"type ii classical cadherins",
            r"epithelial mesenchymal transition",
        ],
        "exclude": [],
    },
    {
        "module": "Vascular / endothelial remodeling",
        "direction": "Positive",
        "include": [
            r"endothelial",
            r"angiogenesis",
            r"blood-brain barrier",
            r"blood brain barrier",
            r"pecam",
            r"vascular endothelial growth factor",
            r"\bvegf\b",
            r"barrier",
        ],
        "exclude": [],
    },
    {
        "module": "Growth-factor / RTK signaling",
        "direction": "Positive",
        "include": [
            r"growth factor stimulus",
            r"epidermal growth factor",
            r"\begf\b",
            r"signaling by pdgf",
            r"\bpdgf\b",
            r"\bfgfr",
            r"signaling by scf-kit",
            r"\bflt3\b",
            r"erythropoietin",
            r"\bmet promotes",
            r"receptor tyrosine kinase",
        ],
        "exclude": [
            r"vascular endothelial",
        ],
    },
    {
        "module": "WNT / developmental signaling",
        "direction": "Positive",
        "include": [
            r"\bwnt\b",
            r"beta.?catenin",
            r"notch",
            r"hedgehog",
            r"hippo signaling",
            r"embryonic",
            r"morphogenesis",
            r"developmental",
            r"cell differentiation",
            r"yap1",
            r"wwtr1",
            r"taz-stimulated gene expression",
        ],
        "exclude": [
            r"fat cell differentiation",
            r"myoblast differentiation",
        ],
    },

    # -------------------------------------------------------------------------
    # Inflammation / stress
    # -------------------------------------------------------------------------
    {
        "module": "Inflammatory / NF-kB signaling",
        "direction": "Positive",
        "include": [
            r"nf.?kb",
            r"nf-kappa",
            r"canonical nf-kappa",
            r"traf3-dependent irf",
            r"irf activation",
            r"tumor necrosis factor",
            r"\btnf\b",
            r"interferon",
            r"cytokine",
            r"inflammatory",
            r"innate immune",
            r"tlr signaling",
            r"interleukin",
            r"jak.stat",
            r"stat3",
            r"stat5",
        ],
        "exclude": [],
    },
    {
        "module": "Antigen presentation / phagosome",
        "direction": None,
        "include": [
            r"mhc class",
            r"antigen presentation",
            r"phagosome",
            r"phagosomal",
        ],
        "exclude": [],
    },
    {
        "module": "Metal / hypoxia response",
        "direction": "Positive",
        "include": [
            r"response to metal",
            r"metal ion",
            r"zinc ion",
            r"cellular response to zinc",
            r"hypoxia",
        ],
        "exclude": [],
    },
    {
        "module": "Antioxidant / detoxification",
        "direction": "Negative",
        "include": [
            r"reactive oxygen",
            r"oxidative stress",
            r"glutathione",
            r"detox",
            r"peroxide",
            r"superoxide",
        ],
        "exclude": [],
    },

    # -------------------------------------------------------------------------
    # Nuclear / RNA / genome biology
    # -------------------------------------------------------------------------
    {
        "module": "Chromatin / transcriptional regulation",
        "direction": "Positive",
        "include": [
            r"chromatin",
            r"histone",
            r"epigen",
            r"\be2f\b",
            r"\bdream\b",
            r"\bhdac",
            r"acetylation",
            r"rora activates gene expression",
            r"runx1",
            r"regulation of rna biosynthetic process",
            r"negative regulation of rna biosynthetic process",
        ],
        "exclude": [],
    },
    {
        "module": "RNA processing / stability",
        "direction": "Negative",
        "include": [
            r"mrna capping",
            r"pre-mrna",
            r"mrna stability",
            r"mrna decay",
            r"exoribonuclease",
            r"rna polymerase i transcription termination",
            r"destabilizes mrna",
            r"rna processing",
            r"rna splicing",
            r"intronless",
            r"au-rich",
        ],
        "exclude": [],
    },
    {
        "module": "DNA repair / genome maintenance",
        "direction": "Negative",
        "include": [
            r"dna repair",
            r"nucleotide excision",
            r"\bner\b",
            r"incision complex",
            r"dna damage",
            r"genome maintenance",
        ],
        "exclude": [],
    },

    # -------------------------------------------------------------------------
    # Cell-state / metabolism
    # -------------------------------------------------------------------------
    {
        "module": "Cell cycle / senescence",
        "direction": None,
        "include": [
            r"g2.?m",
            r"g2 m",
            r"cell cycle",
            r"mitotic",
            r"senescence",
            r"g0 to g1",
            r"checkpoint",
        ],
        "exclude": [],
    },
    {
        "module": "Cell death / apoptosis",
        "direction": "Positive",
        "include": [
            r"apopt",
            r"cell death",
            r"p53 pathway",
            r"nrage signals death",
            r"caspase",
        ],
        "exclude": [],
    },
    {
        "module": "mTOR / nutrient signaling",
        "direction": "Negative",
        "include": [
            r"mtorc1",
            r"\bmtor\b",
            r"cellular response to starvation",
            r"response to starvation",
            r"nutrient signaling",
        ],
        "exclude": [],
    },
    {
        "module": "Lipid metabolism",
        "direction": None,
        "include": [
            r"fatty acid",
            r"lipid biosynthetic",
            r"sphingolipid",
            r"adipogenesis",
            r"fat cell differentiation",
        ],
        "exclude": [],
    },
    {
        "module": "Nucleotide / biosynthetic metabolism",
        "direction": "Negative",
        "include": [
            r"nucleotide biosynthetic",
            r"nucleoside",
            r"purine",
            r"pyrimidine",
        ],
        "exclude": [],
    },
    {
        "module": "MYC / biosynthetic program",
        "direction": "Negative",
        "include": [
            r"myc targets",
            r"myc target",
        ],
        "exclude": [],
    },
    {
        "module": "Protein glycosylation / processing",
        "direction": "Negative",
        "include": [
            r"n-linked glycosylation",
            r"n linked glycosylation",
            r"protein glycosylation",
            r"glycoprotein",
        ],
        "exclude": [],
    },
    {
        "module": "Amino acid metabolism",
        "direction": "Negative",
        "include": [
            r"metabolism of amino acids",
            r"amino acid catabolism",
            r"branched-chain amino acid",
            r"branched chain amino acid",
            r"asparagine",
        ],
        "exclude": [
            r"aminoacylation",
        ],
    },
    {
        "module": "Organelle biogenesis / maintenance",
        "direction": "Negative",
        "include": [
            r"organelle biogenesis and maintenance",
            r"organelle biogenesis",
        ],
        "exclude": [
            r"mitochondrial",
        ],
    },
    {
        "module": "Ion homeostasis",
        "direction": "Negative",
        "include": [
            r"ion homeostasis",
            r"ion transport",
            r"calcium ion transport",
        ],
        "exclude": [
            r"mitochondrial",
        ],
    },
    {
        "module": "Autophagy / lysosome",
        "direction": "Negative",
        "include": [
            r"autophagy",
            r"autophagosome",
            r"lysosome",
            r"lysosomal",
            r"\bmitf",
        ],
        "exclude": [
            r"mitophagy",
        ],
    },
]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--biology-root",
        default="results/weighted_area_biology",
    )

    p.add_argument(
        "--first-pass-root",
        default="results/weighted_area_biology_refined",
    )

    p.add_argument(
        "--outdir",
        default="results/weighted_area_biology_final_v2",
    )

    p.add_argument(
        "--top-drivers-per-module",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max-modules-in-main-figure",
        type=int,
        default=16,
    )

    return p.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def normalize_text(text) -> str:
    text = str(text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_leading_edge(value) -> set[str]:
    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    parts = re.split(r"[;,/|]+", text)

    if len(parts) == 1:
        parts = text.split()

    return {
        p.strip()
        for p in parts
        if p.strip()
        and p.strip().lower() not in {"nan", "none", "na"}
    }


def broad_direction(row) -> str:
    if "direction" in row and pd.notna(row["direction"]):
        value = str(row["direction"])
        if value in {"Positive", "Negative"}:
            return value

    means = [
        row.get("mean_NES_cognition", np.nan),
        row.get("mean_NES_amyloid", np.nan),
        row.get("mean_NES_tau", np.nan),
    ]

    mean_value = np.nanmean(means)

    return "Positive" if mean_value >= 0 else "Negative"


def matches_rule(pathway: str, direction: str, rule: dict) -> bool:
    text = normalize_text(pathway).lower()

    if (
        rule["direction"] is not None
        and direction != rule["direction"]
    ):
        return False

    if not any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in rule["include"]
    ):
        return False

    if any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in rule["exclude"]
    ):
        return False

    return True


def assign_module(pathway: str, direction: str):
    for i, rule in enumerate(RULES, start=1):
        if matches_rule(pathway, direction, rule):
            return rule["module"], i

    return None, None


# =============================================================================
# LOAD
# =============================================================================

def load_inputs(args):
    biology_root = Path(args.biology_root)

    long_path = biology_root / "pathway_long_all.csv"
    reps_path = biology_root / "representative_pathways.csv"

    require(long_path)
    require(reps_path)

    long_df = pd.read_csv(long_path)
    reps = pd.read_csv(reps_path)

    return long_df, reps


def rule_table():
    rows = []

    for i, rule in enumerate(RULES, start=1):
        rows.append(
            {
                "rule_order": i,
                "module": rule["module"],
                "required_direction": rule["direction"],
                "include_patterns": "; ".join(rule["include"]),
                "exclude_patterns": "; ".join(rule["exclude"]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# ASSIGN
# =============================================================================

def assign_all_representatives(reps: pd.DataFrame):
    rows = []

    for _, row in reps.iterrows():
        direction = broad_direction(row)
        module, rule_order = assign_module(
            row["pathway"],
            direction,
        )

        out = row.to_dict()
        out["broad_direction"] = direction
        out["module"] = module
        out["assigned"] = module is not None
        out["matched_rule_order"] = rule_order

        rows.append(out)

    return pd.DataFrame(rows)


def make_rep_long(long_df, assignments):
    assigned = assignments[
        assignments["assigned"]
    ][
        ["library", "pathway", "module"]
    ].drop_duplicates()

    return long_df.merge(
        assigned,
        on=["library", "pathway"],
        how="inner",
    )


# =============================================================================
# MODULE PROFILES
# =============================================================================

def phenotype_profiles(rep_long):
    rows = []

    for (module, phenotype), sub in rep_long.groupby(
        ["module", "phenotype"]
    ):
        unique_reps = sub[
            ["library", "pathway"]
        ].drop_duplicates()

        sig_reps = sub[
            sub["significant"]
        ][
            ["library", "pathway"]
        ].drop_duplicates()

        rows.append(
            {
                "module": module,
                "phenotype": phenotype,
                "axis": AXIS_MAP[phenotype],
                "n_representative_pathways": len(unique_reps),
                "n_significant_representatives": len(sig_reps),
                "significant_fraction": (
                    len(sig_reps) / len(unique_reps)
                    if len(unique_reps)
                    else np.nan
                ),
                "median_NES": float(sub["NES"].median()),
                "mean_NES": float(sub["NES"].mean()),
            }
        )

    return pd.DataFrame(rows)


def axis_profiles(profiles):
    rows = []

    for (module, axis), sub in profiles.groupby(
        ["module", "axis"]
    ):
        rows.append(
            {
                "module": module,
                "axis": axis,
                "mean_axis_NES": float(
                    sub["median_NES"].mean()
                ),
                "mean_axis_abs_NES": float(
                    sub["median_NES"].abs().mean()
                ),
                "signal_phenotypes": int(
                    (
                        sub["n_significant_representatives"] > 0
                    ).sum()
                ),
                "mean_significant_fraction": float(
                    sub["significant_fraction"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def classify_module(module, ax_prof):
    sub = ax_prof[
        ax_prof["module"] == module
    ].set_index("axis")

    if not all(a in sub.index for a in AXES):
        return {
            "module_class": "Incomplete",
            "enhanced_axis": "",
            "enhancement_delta": np.nan,
        }

    nes = {
        a: float(sub.loc[a, "mean_axis_NES"])
        for a in AXES
    }

    abs_nes = {
        a: abs(nes[a])
        for a in AXES
    }

    sig_n = {
        a: int(sub.loc[a, "signal_phenotypes"])
        for a in AXES
    }

    # Strict specificity.
    for axis in AXES:
        others = [a for a in AXES if a != axis]

        if (
            sig_n[axis] >= MIN_AXIS_SIGNAL_PHENOTYPES
            and all(sig_n[o] == 0 for o in others)
            and all(
                abs_nes[o] < OTHER_AXIS_WEAK_ABS_NES
                for o in others
            )
        ):
            return {
                "module_class": f"{axis}-specific",
                "enhanced_axis": axis,
                "enhancement_delta": (
                    abs_nes[axis]
                    - np.mean([abs_nes[o] for o in others])
                ),
            }

    all_axes_signal = all(
        sig_n[a] >= 1
        for a in AXES
    )

    nonzero_signs = [
        np.sign(nes[a])
        for a in AXES
        if nes[a] != 0
    ]

    same_sign = (
        len(nonzero_signs) == 3
        and len(set(nonzero_signs)) == 1
    )

    if all_axes_signal and same_sign:
        strongest = max(
            AXES,
            key=lambda a: abs_nes[a],
        )

        others = [
            a for a in AXES
            if a != strongest
        ]

        delta = (
            abs_nes[strongest]
            - np.mean([abs_nes[o] for o in others])
        )

        if delta >= ENHANCEMENT_DELTA:
            return {
                "module_class": f"{strongest}-enhanced shared",
                "enhanced_axis": strongest,
                "enhancement_delta": delta,
            }

        return {
            "module_class": "Shared core",
            "enhanced_axis": "",
            "enhancement_delta": delta,
        }

    return {
        "module_class": "Mixed / context-dependent",
        "enhanced_axis": "",
        "enhancement_delta": np.nan,
    }


def module_summary(assignments, profiles, ax_prof):
    rows = []

    for module, sub in assignments[
        assignments["assigned"]
    ].groupby("module"):

        reps = sub[
            ["library", "pathway"]
        ].drop_duplicates()

        class_info = classify_module(
            module,
            ax_prof,
        )

        axis_lookup = ax_prof[
            ax_prof["module"] == module
        ].set_index("axis")

        row = {
            "module": module,
            "n_representative_pathways": len(reps),
            "n_libraries": sub["library"].nunique(),
            "libraries": ";".join(
                sorted(sub["library"].unique())
            ),
            **class_info,
        }

        for axis in AXES:
            if axis in axis_lookup.index:
                row[f"{axis.lower()}_mean_NES"] = float(
                    axis_lookup.loc[
                        axis,
                        "mean_axis_NES",
                    ]
                )
                row[f"{axis.lower()}_signal_phenotypes"] = int(
                    axis_lookup.loc[
                        axis,
                        "signal_phenotypes",
                    ]
                )
            else:
                row[f"{axis.lower()}_mean_NES"] = np.nan
                row[f"{axis.lower()}_signal_phenotypes"] = 0

        pp = profiles[
            profiles["module"] == module
        ]

        row["overall_mean_abs_NES"] = float(
            pp["median_NES"].abs().mean()
        )

        row["coverage_score"] = (
            row["n_representative_pathways"]
            * row["overall_mean_abs_NES"]
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "coverage_score",
            "overall_mean_abs_NES",
        ],
        ascending=False,
    )


# =============================================================================
# DRIVERS
# =============================================================================

def driver_genes(rep_long):
    rows = []

    sig = rep_long[
        rep_long["significant"]
    ]

    for _, row in sig.iterrows():
        for gene in parse_leading_edge(
            row["leading_edge_genes"]
        ):
            rows.append(
                {
                    "module": row["module"],
                    "gene": gene,
                    "pathway": row["pathway"],
                    "phenotype": row["phenotype"],
                    "axis": row["axis"],
                    "NES": row["NES"],
                }
            )

    if not rows:
        return pd.DataFrame()

    exploded = pd.DataFrame(rows)

    module_counts = (
        exploded
        .groupby(["module", "gene"])
        .agg(
            module_instances=("pathway", "size"),
            module_unique_pathways=("pathway", "nunique"),
            module_phenotypes=("phenotype", "nunique"),
            module_axes=("axis", "nunique"),
            mean_context_NES=("NES", "mean"),
        )
        .reset_index()
    )

    global_counts = (
        exploded
        .groupby("gene")
        .agg(
            global_instances=("pathway", "size"),
            global_modules=("module", "nunique"),
        )
        .reset_index()
    )

    out = module_counts.merge(
        global_counts,
        on="gene",
        how="left",
    )

    out["module_specificity"] = (
        out["module_instances"]
        / out["global_instances"]
    )

    out["driver_score"] = (
        np.sqrt(out["module_instances"])
        * out["module_specificity"]
        * (1 + 0.15 * (out["module_axes"] - 1))
    )

    out["passes_driver_filter"] = (
        (out["module_instances"] >= MIN_DRIVER_RECURRENCE)
        & (out["module_specificity"] >= MIN_DRIVER_SPECIFICITY)
    )

    return out.sort_values(
        [
            "module",
            "passes_driver_filter",
            "driver_score",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )


# =============================================================================
# PRIORITY
# =============================================================================

def priority_modules(summary, max_n):
    """
    Rank modules by biological evidence strength rather than module-class order.

    Priority score balances:
      1. effect magnitude: overall mean |NES|
      2. breadth: sqrt(number of nonredundant representative pathways)
      3. coherence: shared-core and axis-enhanced shared modules receive a
         small bonus, but neither is categorically ranked above the other.

    This prevents biologically strong enhanced modules (especially
    mitochondrial energetics) from being pushed out simply because they are
    not labeled "Shared core".
    """
    x = summary.copy()

    class_bonus = {
        "Shared core": 1.10,
        "Amyloid-enhanced shared": 1.10,
        "Tau-enhanced shared": 1.10,
        "Cognition-enhanced shared": 1.10,
        "Amyloid-specific": 1.00,
        "Tau-specific": 1.00,
        "Cognition-specific": 1.00,
        "Mixed / context-dependent": 0.85,
        "Incomplete": 0.75,
    }

    x["coherence_bonus"] = (
        x["module_class"]
        .map(class_bonus)
        .fillna(0.85)
    )

    x["priority_score"] = (
        x["overall_mean_abs_NES"]
        * np.sqrt(x["n_representative_pathways"].clip(lower=1))
        * x["coherence_bonus"]
    )

    # Also calculate a transparent rank for the exported table.
    x = x.sort_values(
        [
            "priority_score",
            "overall_mean_abs_NES",
            "n_representative_pathways",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).copy()

    x["priority_rank"] = np.arange(1, len(x) + 1)

    return x.head(max_n)


# =============================================================================
# FIGURES
# =============================================================================

def make_heatmap(profiles, priority, outdir):
    modules = priority["module"].tolist()

    pivot = profiles.pivot_table(
        index="module",
        columns="phenotype",
        values="median_NES",
        aggfunc="first",
    ).reindex(
        index=modules,
        columns=PHENOTYPES,
    )

    sig = profiles.pivot_table(
        index="module",
        columns="phenotype",
        values="significant_fraction",
        aggfunc="first",
    ).reindex(
        index=modules,
        columns=PHENOTYPES,
    )

    fig_h = max(
        4.8,
        0.34 * len(modules) + 1.8,
    )

    fig, ax = plt.subplots(
        figsize=(7.2, fig_h)
    )

    vmax = max(
        1.0,
        float(np.nanmax(np.abs(pivot.to_numpy()))),
    )

    im = ax.imshow(
        pivot.to_numpy(),
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(PHENOTYPES)))
    ax.set_xticklabels(
        [PHENOTYPE_LABELS[p] for p in PHENOTYPES],
        rotation=42,
        ha="right",
    )

    ax.set_yticks(np.arange(len(modules)))
    ax.set_yticklabels(
        modules,
        fontsize=5.5,
    )

    for i in range(len(modules)):
        for j in range(len(PHENOTYPES)):
            value = sig.iloc[i, j]

            if (
                pd.notna(value)
                and value >= MODULE_SIG_FRACTION_FOR_DOT
            ):
                ax.text(
                    j,
                    i,
                    "•",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="black",
                )

    ax.axvline(2.5, linewidth=0.8)
    ax.axvline(5.5, linewidth=0.8)

    ax.set_title(
        "Prioritized Weighted AREA biological modules",
        fontsize=9.4,
        fontweight="bold",
        pad=11,
    )

    ax.text(
        0.5,
        1.015,
        (
            "Cell = median NES across nonredundant representative pathways; "
            "dot = ≥50% of module representatives significant"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=5.2,
        color="#5A5A5A",
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.028,
        pad=0.025,
    )

    cbar.set_label(
        "Median NES",
        fontsize=5.7,
    )

    fig.subplots_adjust(
        left=0.27,
        right=0.93,
        top=0.89,
        bottom=0.20,
    )

    fig.savefig(
        outdir / "final_module_phenotype_heatmap.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "final_module_phenotype_heatmap.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def make_axis_figure(ax_prof, priority, outdir):
    modules = priority["module"].tolist()

    pivot = ax_prof.pivot_table(
        index="module",
        columns="axis",
        values="mean_axis_NES",
        aggfunc="first",
    ).reindex(
        index=modules,
        columns=AXES,
    )

    fig_h = max(
        4.8,
        0.32 * len(modules) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(6.8, fig_h)
    )

    y = np.arange(len(modules))
    width = 0.22

    for i, axis in enumerate(AXES):
        ax.barh(
            y + (i - 1) * width,
            pivot[axis],
            height=width,
            label=axis,
        )

    ax.axvline(0, linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(
        modules,
        fontsize=5.4,
    )

    ax.invert_yaxis()
    ax.set_xlabel("Mean module NES")

    ax.set_title(
        "Relative strength of prioritized modules across phenotype axes",
        fontsize=8.8,
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15)

    fig.tight_layout()

    fig.savefig(
        outdir / "final_module_axis_profiles.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "final_module_axis_profiles.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def make_driver_figure(drivers, priority, outdir, top_n):
    if len(drivers) == 0:
        return

    rows = []

    for module in priority["module"]:
        sub = drivers[
            (drivers["module"] == module)
            & (drivers["passes_driver_filter"])
        ].head(top_n)

        for _, row in sub.iterrows():
            rows.append(
                {
                    "label": f"{module} | {row['gene']}",
                    "driver_score": row["driver_score"],
                }
            )

    if not rows:
        return

    plot_df = pd.DataFrame(rows).iloc[::-1]

    fig_h = max(
        5.0,
        0.17 * len(plot_df) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, fig_h)
    )

    ax.barh(
        plot_df["label"],
        plot_df["driver_score"],
    )

    ax.set_xlabel(
        "Module-specific driver score"
    )

    ax.set_title(
        "Top module-specific leading-edge genes",
        fontsize=8.8,
        fontweight="bold",
    )

    ax.tick_params(
        axis="y",
        labelsize=4.7,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15)

    fig.tight_layout()

    fig.savefig(
        outdir / "final_module_driver_genes.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "final_module_driver_genes.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def make_coverage_figure(assignments, outdir):
    n_total = len(
        assignments[
            ["library", "pathway"]
        ].drop_duplicates()
    )

    n_assigned = len(
        assignments[
            assignments["assigned"]
        ][
            ["library", "pathway"]
        ].drop_duplicates()
    )

    values = [
        n_assigned,
        n_total - n_assigned,
    ]

    fig, ax = plt.subplots(
        figsize=(4.2, 3.0)
    )

    bars = ax.bar(
        ["Assigned to\nfinal modules", "Still\nunassigned"],
        values,
        width=0.58,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:,}\n({100*value/n_total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=6.0,
        )

    ax.set_ylabel(
        "Nonredundant representative pathways"
    )

    ax.set_title(
        "Coverage of final biological module scheme",
        fontsize=8.2,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()

    fig.savefig(
        outdir / "final_module_coverage.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "final_module_coverage.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("FINAL WEIGHTED AREA BIOLOGICAL MODULE REFINEMENT V2")
    print("=" * 100)
    print()

    long_df, reps = load_inputs(args)

    rules = rule_table()
    rules.to_csv(
        outdir / "final_module_rules.csv",
        index=False,
    )

    assignments = assign_all_representatives(
        reps
    )

    assignments.to_csv(
        outdir / "final_module_assignment_audit.csv",
        index=False,
    )

    unassigned = assignments[
        ~assignments["assigned"]
    ].copy()

    unassigned.to_csv(
        outdir / "final_unassigned_representative_pathways.csv",
        index=False,
    )

    rep_long = make_rep_long(
        long_df,
        assignments,
    )

    profiles = phenotype_profiles(
        rep_long
    )
    profiles.to_csv(
        outdir / "final_module_phenotype_profiles.csv",
        index=False,
    )

    ax_prof = axis_profiles(
        profiles
    )
    ax_prof.to_csv(
        outdir / "final_module_axis_profiles.csv",
        index=False,
    )

    summary = module_summary(
        assignments,
        profiles,
        ax_prof,
    )
    summary.to_csv(
        outdir / "final_module_summary.csv",
        index=False,
    )

    drivers = driver_genes(
        rep_long
    )
    drivers.to_csv(
        outdir / "final_module_driver_genes.csv",
        index=False,
    )

    rep_table = assignments[
        assignments["assigned"]
    ].copy()

    rep_table.to_csv(
        outdir / "final_module_representative_pathways.csv",
        index=False,
    )

    priority = priority_modules(
        summary,
        args.max_modules_in_main_figure,
    )

    priority.to_csv(
        outdir / "final_priority_modules.csv",
        index=False,
    )

    make_heatmap(
        profiles,
        priority,
        outdir,
    )

    make_axis_figure(
        ax_prof,
        priority,
        outdir,
    )

    make_driver_figure(
        drivers,
        priority,
        outdir,
        args.top_drivers_per_module,
    )

    make_coverage_figure(
        assignments,
        outdir,
    )

    # Console summary
    n_total = assignments[
        ["library", "pathway"]
    ].drop_duplicates().shape[0]

    n_assigned = assignments[
        assignments["assigned"]
    ][
        ["library", "pathway"]
    ].drop_duplicates().shape[0]

    print(
        f"Representative pathways: {n_total:,}"
    )
    print(
        f"Assigned: {n_assigned:,} "
        f"({100*n_assigned/n_total:.1f}%)"
    )
    print(
        f"Unassigned: {n_total - n_assigned:,}"
    )
    print()

    print("PRIORITY MODULES")
    print()

    cols = [
        "module",
        "module_class",
        "n_representative_pathways",
        "cognition_mean_NES",
        "amyloid_mean_NES",
        "tau_mean_NES",
        "overall_mean_abs_NES",
    ]

    print(
        priority[cols].to_string(index=False)
    )

    print()
    print("TOP DRIVERS")
    print()

    if len(drivers) > 0:
        for module in priority["module"]:
            top = drivers[
                (drivers["module"] == module)
                & (drivers["passes_driver_filter"])
            ].head(
                args.top_drivers_per_module
            )

            if len(top) == 0:
                continue

            print(
                f"{module}: "
                + ", ".join(top["gene"].tolist())
            )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print()
    print(f"Outputs: {outdir}")


if __name__ == "__main__":
    main()

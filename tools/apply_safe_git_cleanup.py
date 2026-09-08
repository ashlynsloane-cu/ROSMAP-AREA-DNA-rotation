#!/usr/bin/env python3
"""
apply_safe_git_cleanup.py

Safe cleanup for the current ROSMAP-AREA-DNA-rotation exploratory tree.

This script removes only files that the audit identified as clearly obsolete:
  - lower-numbered versions when a higher-numbered version exists
  - unversioned predecessors when numbered successors exist
  - exact duplicate copies in the wrong folder
  - README(4).md duplicate
  - macOS .DS_Store files

It deliberately DOES NOT touch same-named files whose contents differ
substantially (e.g. plot_gsea_archetypes.py in diagnostics vs visualization).

Dry-run is the default.

Usage:
    python3 apply_safe_git_cleanup.py

Then, if the plan looks correct:
    python3 apply_safe_git_cleanup.py --apply
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


# Explicitly reviewed cleanup targets from git_cleanup_report.txt.
REMOVE_FILES = [
    # ------------------------------------------------------------------
    # Enrichment version families: keep highest numbered versions.
    # ------------------------------------------------------------------
    "exploratory/enrichment/make_curated_weighted_area_gsea_figure.py",
    "exploratory/enrichment/make_curated_weighted_area_gsea_figure_v2.py",
    "exploratory/enrichment/make_curated_weighted_area_gsea_figure_v3.py",
    "exploratory/enrichment/make_curated_weighted_area_gsea_figure_v4.py",
    # KEEP: ..._v5.py

    "exploratory/enrichment/make_final_gsea_figures_v6.py",
    "exploratory/enrichment/make_final_gsea_figures_v7.py",
    # KEEP: ..._v8.py

    # Final Venn figure: v2 supersedes the unversioned predecessor.
    "exploratory/diagnostics/plot_final_deseq_regular_weighted_venn.py",
    # KEEP: ..._v2.py

    # ------------------------------------------------------------------
    # Method-development version families.
    # ------------------------------------------------------------------
    "exploratory/method_development/make_rosmap_analysis_tree_v3.py",
    "exploratory/method_development/make_rosmap_analysis_tree_v4.py",
    "exploratory/method_development/make_rosmap_analysis_tree_v5.py",
    "exploratory/method_development/make_rosmap_analysis_tree_v6.py",
    # KEEP: ..._v7.py
    # KEEP: ..._publication.py

    "exploratory/method_development/make_weighted_area_encoding_comparison_tree_refined.py",
    "exploratory/method_development/make_weighted_area_encoding_comparison_tree_refined_v4.py",
    # KEEP: ..._v5.py

    # There is an unversioned + v2 pair; based on the project convention,
    # v2 is the later/final numbered version.
    "exploratory/method_development/make_controlled_deseq2_vs_regular_area_figure.py",
    # KEEP: ..._v2.py
    # KEEP: ..._publication.py

    "exploratory/method_development/make_weighted_area_encoding_comparison.py",
    # KEEP: ..._v2.py
    # KEEP: ..._publication.py
    # NOTE: *_direct.py and *_tree*.py are distinct figure concepts and are kept.

    # ------------------------------------------------------------------
    # Exact duplicates: keep the copy in the semantically appropriate folder.
    # ------------------------------------------------------------------
    "exploratory/visualization/run_ams_area.py",
    # KEEP: exploratory/area/run_ams_area.py

    "exploratory/visualization/validate_corrected_ams_area_nulls.py",
    # KEEP: exploratory/area/validation/validate_corrected_ams_area_nulls.py

    "exploratory/visualization/README(4).md",
    # KEEP: exploratory/visualization/README.md
]


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def find_repo_root(start: Path) -> Path:
    result = git(start, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise SystemExit(
            f"Not inside a Git repository:\n{result.stderr.strip()}"
        )
    return Path(result.stdout.strip()).resolve()


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def collect_ds_store(root: Path) -> list[Path]:
    exploratory = root / "exploratory"
    if not exploratory.exists():
        return []
    return sorted(exploratory.rglob(".DS_Store"))


def existing_explicit_targets(root: Path) -> list[Path]:
    return [
        root / item
        for item in REMOVE_FILES
        if (root / item).exists()
    ]


def print_keep_summary() -> None:
    print("\nIMPORTANT FILES THIS SCRIPT WILL KEEP:")
    print("  exploratory/enrichment/make_curated_weighted_area_gsea_figure_v5.py")
    print("  exploratory/enrichment/make_final_gsea_figures_v8.py")
    print("  exploratory/diagnostics/plot_final_deseq_regular_weighted_venn_v2.py")
    print("  exploratory/method_development/make_rosmap_analysis_tree_v7.py")
    print("  exploratory/method_development/make_rosmap_analysis_tree_publication.py")
    print("  exploratory/method_development/make_weighted_area_encoding_comparison_tree_refined_v5.py")
    print("  exploratory/method_development/make_controlled_deseq2_vs_regular_area_figure_v2.py")
    print("  exploratory/method_development/make_controlled_deseq2_vs_regular_area_figure_publication.py")
    print("  exploratory/method_development/make_weighted_area_encoding_comparison_v2.py")
    print("  exploratory/method_development/make_weighted_area_encoding_comparison_publication.py")
    print("  exploratory/area/run_ams_area.py")
    print("  exploratory/area/validation/validate_corrected_ams_area_nulls.py")
    print()
    print("It also leaves these DIFFERENT same-name files alone:")
    print("  bootstrap_msi_stability.py")
    print("  plot_gsea_archetypes.py")
    print("  plot_violin_contrasts.py")
    print("  simulate_msi_calibration.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the reviewed cleanup targets.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Any path inside the repository (default: current directory).",
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(args.root).expanduser().resolve())

    explicit = existing_explicit_targets(repo)
    ds_store = collect_ds_store(repo)
    targets = sorted(set(explicit + ds_store))

    print("=" * 78)
    print("ROSMAP SAFE GIT CLEANUP")
    print("=" * 78)
    print(f"Repository: {repo}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print()

    print("FILES TO REMOVE:")
    if not targets:
        print("  None found.")
    else:
        for path in targets:
            reason = "macOS metadata" if path.name == ".DS_Store" else "reviewed obsolete/duplicate"
            print(f"  {rel(path, repo)}  [{reason}]")

    print_keep_summary()

    if not args.apply:
        print("=" * 78)
        print("DRY RUN ONLY — NOTHING WAS DELETED.")
        print("If this list looks correct, run:")
        print("  python3 apply_safe_git_cleanup.py --apply")
        print("=" * 78)
        return 0

    answer = input(
        "\nType CLEAN to delete exactly the files listed above: "
    ).strip()

    if answer != "CLEAN":
        print("Cancelled. Nothing deleted.")
        return 0

    deleted = 0
    for path in targets:
        try:
            path.unlink()
            print(f"Deleted: {rel(path, repo)}")
            deleted += 1
        except OSError as exc:
            print(f"ERROR: {rel(path, repo)}: {exc}")

    print(f"\nDeleted {deleted} file(s).")

    # Show concise status immediately after cleanup.
    status = git(repo, "status", "--short")
    print("\nCURRENT GIT STATUS:")
    print(status.stdout.rstrip() or "Working tree clean.")

    print("\nNext recommended commands:")
    print("  git status --short")
    print("  git add -A")
    print("  git diff --cached --stat")
    print()
    print("Do not commit until the staged --stat looks sensible.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

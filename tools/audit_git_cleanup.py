#!/usr/bin/env python3
"""
audit_git_cleanup.py

Audit a Git repository for:
  1. Current Git status
  2. Modified/deleted tracked files
  3. Version families such as foo_v2.py, foo_v3.py, foo_v8.py
  4. Exact duplicate files
  5. Same-named files in different directories with different content

By default this script DOES NOT delete anything.

Optional:
    --delete-old-versions

removes lower-numbered versions in a version family while keeping the
highest-numbered version. Files such as *_publication.py are treated as
separate files and are never deleted automatically.

Recommended usage:
    python3 audit_git_cleanup.py
    python3 audit_git_cleanup.py --root . --write-report git_cleanup_report.txt

After reviewing the report:
    python3 audit_git_cleanup.py --delete-old-versions
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


VERSION_RE = re.compile(r"^(?P<base>.+)_v(?P<version>\d+)(?P<suffix>\.[^.]+)$")


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def ensure_git_repo(root: Path) -> None:
    result = run_git(root, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise SystemExit(f"Not a Git repository: {root}\n{result.stderr.strip()}")


def repo_root(root: Path) -> Path:
    result = run_git(root, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


def git_status_lines(root: Path) -> list[str]:
    result = run_git(root, "status", "--short")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.splitlines()


def git_diff_stat(root: Path) -> str:
    result = run_git(root, "diff", "--stat")
    return result.stdout.rstrip()


def tracked_change_names(root: Path) -> list[str]:
    result = run_git(root, "diff", "--name-status")
    return result.stdout.splitlines()


def should_ignore(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    ignored_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "node_modules",
    }
    return any(part in ignored_dirs for part in rel_parts)


def candidate_files(root: Path, scan_dirs: list[str]) -> list[Path]:
    files: list[Path] = []

    for scan_dir in scan_dirs:
        base = root / scan_dir
        if not base.exists():
            continue

        if base.is_file():
            if not should_ignore(base, root):
                files.append(base)
            continue

        for path in base.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and not should_ignore(path, root)
            ):
                files.append(path)

    return sorted(set(files))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return None


def similarity(a: Path, b: Path) -> float | None:
    a_lines = text_lines(a)
    b_lines = text_lines(b)
    if a_lines is None or b_lines is None:
        return None

    a_text = "\n".join(a_lines)
    b_text = "\n".join(b_lines)

    # Avoid pathological cost on extremely large text files.
    if len(a_text) + len(b_text) > 2_000_000:
        return None

    return difflib.SequenceMatcher(None, a_text, b_text).ratio()


def line_count(path: Path) -> int | None:
    lines = text_lines(path)
    return len(lines) if lines is not None else None


def find_version_families(files: list[Path], root: Path):
    families: dict[tuple[Path, str, str], list[tuple[int, Path]]] = defaultdict(list)

    for path in files:
        match = VERSION_RE.match(path.name)
        if not match:
            continue

        key = (
            path.parent.relative_to(root),
            match.group("base"),
            match.group("suffix"),
        )
        families[key].append((int(match.group("version")), path))

    return {
        key: sorted(items, key=lambda x: x[0])
        for key, items in families.items()
        if len(items) >= 2
    }


def find_exact_duplicates(files: list[Path]) -> list[list[Path]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        try:
            by_size[path.stat().st_size].append(path)
        except OSError:
            pass

    duplicate_groups: list[list[Path]] = []

    for same_size in by_size.values():
        if len(same_size) < 2:
            continue

        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in same_size:
            try:
                by_hash[sha256(path)].append(path)
            except OSError:
                continue

        for group in by_hash.values():
            if len(group) >= 2:
                duplicate_groups.append(sorted(group))

    return sorted(duplicate_groups, key=lambda g: str(g[0]))


def find_same_basename_groups(files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[path.name].append(path)

    return {
        name: sorted(paths)
        for name, paths in groups.items()
        if len(paths) >= 2
    }


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_report(root: Path, files: list[Path]) -> tuple[str, list[Path]]:
    out: list[str] = []
    old_version_candidates: list[Path] = []

    out.append("=" * 78)
    out.append("GIT CLEANUP AUDIT")
    out.append("=" * 78)
    out.append(f"Repository: {root}")
    out.append("")

    # ------------------------------------------------------------------
    # Git status
    # ------------------------------------------------------------------
    out.append("1. GIT STATUS")
    out.append("-" * 78)
    status = git_status_lines(root)
    if status:
        out.extend(status)
    else:
        out.append("Working tree clean.")
    out.append("")

    # ------------------------------------------------------------------
    # Tracked diffs
    # ------------------------------------------------------------------
    out.append("2. TRACKED FILE CHANGES")
    out.append("-" * 78)
    changed = tracked_change_names(root)
    if changed:
        out.extend(changed)
        stat = git_diff_stat(root)
        if stat:
            out.append("")
            out.append("git diff --stat:")
            out.append(stat)
    else:
        out.append("No tracked-file changes.")
    out.append("")

    # ------------------------------------------------------------------
    # Version families
    # ------------------------------------------------------------------
    out.append("3. VERSION FAMILIES")
    out.append("-" * 78)
    families = find_version_families(files, root)

    if not families:
        out.append("No multi-version families found.")
    else:
        for (directory, base, suffix), items in sorted(
            families.items(), key=lambda x: (str(x[0][0]), x[0][1])
        ):
            highest_version, highest_path = items[-1]
            out.append(f"\n{directory}/{base}_vN{suffix}")
            out.append(f"  KEEP (highest version): {relative(highest_path, root)}")

            for version, path in items[:-1]:
                old_version_candidates.append(path)
                out.append(
                    f"  OLD  (v{version}):          {relative(path, root)}"
                )

            # If an unversioned sibling exists, report it but never delete it.
            unversioned = root / directory / f"{base}{suffix}"
            if unversioned.exists():
                out.append(
                    f"  REVIEW unversioned:         {relative(unversioned, root)}"
                )

            publication = root / directory / f"{base}_publication{suffix}"
            if publication.exists():
                out.append(
                    f"  KEEP/REVIEW publication:    {relative(publication, root)}"
                )

    out.append("")

    # ------------------------------------------------------------------
    # Exact duplicates
    # ------------------------------------------------------------------
    out.append("4. EXACT DUPLICATES")
    out.append("-" * 78)
    exact = find_exact_duplicates(files)
    if not exact:
        out.append("No exact duplicate files found.")
    else:
        for idx, group in enumerate(exact, start=1):
            out.append(f"\nDuplicate group {idx}:")
            for path in group:
                out.append(f"  {relative(path, root)}")
    out.append("")

    # ------------------------------------------------------------------
    # Same basenames across folders
    # ------------------------------------------------------------------
    out.append("5. SAME BASENAME IN MULTIPLE DIRECTORIES")
    out.append("-" * 78)

    same_name = find_same_basename_groups(files)
    if not same_name:
        out.append("No repeated basenames found.")
    else:
        for name, paths in sorted(same_name.items()):
            hashes = []
            for path in paths:
                try:
                    hashes.append(sha256(path))
                except OSError:
                    hashes.append("ERROR")

            if len(set(hashes)) == 1:
                # Already covered under exact duplicates.
                continue

            out.append(f"\n{name}")
            for path in paths:
                lc = line_count(path)
                lc_text = f"{lc} lines" if lc is not None else "binary/non-UTF8"
                out.append(
                    f"  {relative(path, root)} "
                    f"({path.stat().st_size} bytes, {lc_text}, sha={sha256(path)[:10]})"
                )

            if len(paths) == 2:
                ratio = similarity(paths[0], paths[1])
                if ratio is not None:
                    out.append(f"  Text similarity: {ratio:.1%}")
                out.append(
                    "  Suggested diff command:"
                )
                out.append(
                    f'    diff -u "{relative(paths[0], root)}" '
                    f'"{relative(paths[1], root)}"'
                )
            else:
                out.append("  More than two copies; review manually.")
    out.append("")

    # ------------------------------------------------------------------
    # Proposed cleanup
    # ------------------------------------------------------------------
    out.append("6. AUTOMATIC CLEANUP CANDIDATES")
    out.append("-" * 78)
    if old_version_candidates:
        out.append(
            "These are lower-numbered versions where a higher-numbered version "
            "exists in the same directory:"
        )
        for path in sorted(old_version_candidates):
            out.append(f"  {relative(path, root)}")
    else:
        out.append("No lower-numbered version candidates found.")

    out.append("")
    out.append("=" * 78)
    out.append("NOTHING HAS BEEN DELETED.")
    out.append(
        "To delete ONLY the lower-numbered version candidates, rerun with:"
    )
    out.append("  python3 audit_git_cleanup.py --delete-old-versions")
    out.append("=" * 78)

    return "\n".join(out), sorted(set(old_version_candidates))


def delete_old_versions(root: Path, paths: list[Path]) -> None:
    if not paths:
        print("No lower-numbered versions to delete.")
        return

    print("\nThe following lower-numbered versions will be deleted:\n")
    for path in paths:
        print(f"  {relative(path, root)}")

    print("\nHighest-numbered versions, *_publication files, and unversioned files")
    print("will NOT be deleted automatically.")

    answer = input("\nType DELETE to continue: ").strip()
    if answer != "DELETE":
        print("Cancelled. Nothing deleted.")
        return

    deleted = 0
    for path in paths:
        try:
            path.unlink()
            deleted += 1
            print(f"Deleted: {relative(path, root)}")
        except OSError as exc:
            print(f"ERROR deleting {relative(path, root)}: {exc}")

    print(f"\nDeleted {deleted} file(s).")
    print("\nNow review:")
    print("  git status --short")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a Git repository for duplicate and versioned files."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Path anywhere inside the Git repository (default: current directory).",
    )
    parser.add_argument(
        "--scan",
        nargs="+",
        default=["exploratory"],
        help="Directories/files to scan relative to repo root (default: exploratory).",
    )
    parser.add_argument(
        "--write-report",
        default="git_cleanup_report.txt",
        help="Report filename relative to repo root (default: git_cleanup_report.txt).",
    )
    parser.add_argument(
        "--delete-old-versions",
        action="store_true",
        help=(
            "Interactively delete lower-numbered _vN files when a higher-numbered "
            "version exists in the same family."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = Path(args.root).expanduser().resolve()

    ensure_git_repo(start)
    root = repo_root(start)

    files = candidate_files(root, args.scan)
    report, old_versions = build_report(root, files)

    print(report)

    report_path = root / args.write_report
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport written to: {report_path.relative_to(root)}")

    if args.delete_old_versions:
        delete_old_versions(root, old_versions)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check availability and integrity of canonical figure inputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_MANIFEST = (
    ROOT / "config/canonical_figure_inputs.csv"
)
ADDITIONAL_MANIFEST = (
    ROOT / "finalization_audit/repository_freeze/"
    "additional_figure_inputs.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def primary_inputs():
    with INPUT_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        relative = Path(row["relative_path"])
        workspace = row["workspace"]

        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative path: {relative}")

        if workspace == "repo":
            path = ROOT / relative
        elif workspace == "offline":
            path = ROOT / "data/frozen_inputs" / relative
        else:
            raise ValueError(f"Unknown workspace: {workspace}")

        yield row["source_id"], path, row["sha256"]


def additional_inputs():
    data = json.loads(ADDITIONAL_MANIFEST.read_text())

    def walk(value, label="additional"):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                relative = Path(value["path"])

                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"Unsafe additional input path: {relative}"
                    )

                yield label, ROOT / relative, value["sha256"]
            else:
                for key, child in value.items():
                    yield from walk(child, f"{label}.{key}")

        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f"{label}[{index}]")

    yield from walk(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-present",
        action="store_true",
        help="Also display verified inputs",
    )
    args = parser.parse_args()

    records = list(primary_inputs()) + list(additional_inputs())

    missing = []
    mismatched = []
    verified = []

    for label, path, expected in records:
        relative = path.relative_to(ROOT)

        if not path.is_file():
            missing.append((label, relative))
        elif sha256(path) != expected:
            mismatched.append((label, relative))
        else:
            verified.append((label, relative))

    print(f"Registered input bindings: {len(records)}")
    print(f"Verified: {len(verified)}")
    print(f"Missing: {len(missing)}")
    print(f"Hash mismatches: {len(mismatched)}")

    if args.show_present:
        for label, path in verified:
            print(f"PASS    {label}: {path}")

    for label, path in missing:
        print(f"MISSING {label}: {path}")

    for label, path in mismatched:
        print(f"MISMATCH {label}: {path}")

    if missing or mismatched:
        print(
            "\nProvision authorized inputs at the paths above. "
            "Do not commit restricted inputs to Git."
        )
        return 1

    print("\nPASS: all registered inputs match their frozen hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

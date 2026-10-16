#!/usr/bin/env python3
"""Fail-closed, provenance-recording raw-to-figures workflow orchestrator.

Only stages with an explicitly configured producer command may run. It never
substitutes historical references for freshly generated artifacts.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

STAGE_IDS = ("rna_preprocess", "tmt_preprocess", "monocyte_preprocess", "metadata_manifests", "area_models", "conventional_models",
             "stage7", "stage9", "stage10", "stage11_sensitivity",
             "stage12", "figures")
DANGEROUS = re.compile(r"(?:/home/assl5508/|/Users/ashlynsloane/|stage11_offline_|data/frozen_inputs)")


def resolve_path(s, roots):
    if not isinstance(s, str):
        raise ValueError("Path must be a string")
    for token, root in roots.items():
        s = s.replace("{" + token + "}", str(root))
    if "{" in s or "}" in s:
        raise ValueError("Unknown variable in path: " + s)
    return Path(s).expanduser().resolve()


def check_config(cfg):
    stages = cfg.get("stages", [])
    if not isinstance(stages, list) or [x.get("id") for x in stages] != list(STAGE_IDS):
        raise ValueError("Plan must specify all canonical stages in dependency order")
    seen = set()
    for stage in stages:
        sid = stage["id"]
        if not isinstance(stage.get("requires"), list) or not isinstance(stage.get("produces"), list):
            raise ValueError(f"{sid}: requires and produces must be lists")
        if any(dep not in seen for dep in stage.get("depends_on", [])):
            raise ValueError(f"{sid}: dependencies must precede dependent stage")
        cmd = stage.get("command")
        if cmd is not None and (not isinstance(cmd, list) or not cmd or
                                any(not isinstance(v, str) for v in cmd)):
            raise ValueError(f"{sid}: command must be an argv string list or null")
        seen.add(sid)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cfg, roots, execute=False, target=None):
    check_config(cfg)
    output = roots["output_root"]
    repo = roots["release_root"]
    restricted = [roots["raw_root"], roots["output_root"]]
    if any(output == x or x in output.parents or output in x.parents for x in (repo, roots["raw_root"])):
        raise ValueError("Output root may not contain or equal release/raw roots")
    if repo == output or repo in output.parents:
        raise ValueError("Output root must be outside release repository")
    stages = cfg["stages"]
    result = {"schema_version": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "mode": "execute" if execute else "preflight", "stages": [],
              "raw_to_figure_complete": False}
    passed = set()
    for stage in stages:
        sid = stage["id"]
        if target and sid not in target:
            continue
        entry = {"id": sid, "status": "BLOCKED", "reason": None, "outputs": []}
        result["stages"].append(entry)
        if any(d not in passed for d in stage.get("depends_on", [])):
            entry["reason"] = "Upstream stage not executed successfully"
            continue
        cmd = stage.get("command")
        if cmd is None:
            entry["reason"] = "No verified producer configured; historical output cannot substitute"
            continue
        if any(DANGEROUS.search(part) for part in cmd):
            entry["reason"] = "Historical machine-specific/frozen-input path in command"
            continue
        command = []
        try:
            for part in cmd:
                value = part
                for key, root in roots.items():
                    value = value.replace("{" + key + "}", str(root))
                if "{" in value or "}" in value:
                    raise ValueError("Unresolved placeholder in argv")
                command.append(value)
            inputs = [resolve_path(x, roots) for x in stage["requires"]]
            outputs = [resolve_path(x, roots) for x in stage["produces"]]
            for path in inputs:
                if not path.is_file():
                    raise ValueError("Missing required input: " + str(path))
            for path in outputs:
                if output not in path.parents:
                    raise ValueError("Produced output must be inside output-root: " + str(path))
            entry["inputs"] = [{"path": str(x), "sha256": sha256(x)} for x in inputs]
            entry["command"] = command
            if not execute:
                entry["status"] = "READY"
                entry["reason"] = "Command configured and required inputs present; not executed"
                continue
            log = output / "logs" / f"{sid}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as f:
                proc = subprocess.run(command, cwd=str(repo), stdout=f, stderr=subprocess.STDOUT,
                                      check=False)
            entry["log"] = str(log)
            entry["exit_code"] = proc.returncode
            if proc.returncode:
                entry["status"] = "FAILED"
                entry["reason"] = "Producer exited nonzero"
                continue
            missing = [str(x) for x in outputs if not x.is_file() or x.stat().st_size == 0]
            if missing:
                entry["status"] = "FAILED"
                entry["reason"] = "Declared nonempty outputs missing: " + ", ".join(missing)
                continue
            entry["outputs"] = [{"path": str(x), "sha256": sha256(x)} for x in outputs]
            entry["status"] = "PASS"
            passed.add(sid)
        except (OSError, ValueError) as exc:
            entry["reason"] = str(exc)
    result["raw_to_figure_complete"] = (execute and len(passed) == len(STAGE_IDS) and not target)
    if execute:
        output.mkdir(parents=True, exist_ok=True)
        (output / "raw_to_figures_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=Path("config/raw_to_figures_plan.json"))
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    repo = a.release_root.expanduser().resolve()
    raw = a.raw_root.expanduser().resolve()
    out = a.output_root.expanduser().resolve()
    if not repo.is_dir() or not raw.is_dir():
        ap.error("Release root and authorized raw-root directories must exist")
    plan = a.plan if a.plan.is_absolute() else repo / a.plan
    data = json.loads(plan.read_text())
    result = run(data, {"release_root": repo, "raw_root": raw, "output_root": out}, a.execute)
    for entry in result["stages"]:
        print(f"{entry['id']:<25} {entry['status']:<8} {entry['reason'] or ''}")
    print("RAW_TO_FIGURE_COMPLETE:", result["raw_to_figure_complete"])
    if a.execute:
        print("Report:", out / "raw_to_figures_report.json")
    if not result["raw_to_figure_complete"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

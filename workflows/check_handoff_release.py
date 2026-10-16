#!/usr/bin/env python3
"""Read-only, conservative release hygiene and handoff inventory check.

Does not stage, delete, or commit files. Does not read donor-level datasets.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

BLOCK_PARTS = {'__pycache__', '.pytest_cache', '.DS_Store', 'data', 'results', 'outputs', 'frozen_inputs'}
BLOCK_SUFFIXES = {'.pyc', '.rds', '.rdata', '.bam', '.fastq', '.fq', '.h5ad', '.loom', '.parquet'}
BLOCK_NAMES = {'run_stage11_batch.py.pre_cerad_fix.bak'}
REQUIRED = [
    'workflows/run_stage11_batch.py',
    'workflows/run_stage11_sensitivity_batch.py',
    'workflows/run_stage11_conventional_batch.py',
    'workflows/run_stage11_monocyte_area.py',
    'workflows/run_downstream.py',
    'workflows/run_main_figures.py',
    'workflows/run_raw_to_figures.py',
    'workflows/validate_monocyte_inputs.py',
    'config/raw_to_figures_plan.json',
]

def unsafe_path(path):
    p = Path(path)
    low = path.lower()
    if set(p.parts) & BLOCK_PARTS or p.suffix.lower() in BLOCK_SUFFIXES or p.name in BLOCK_NAMES:
        return True
    if any(t in low for t in ('monocyte_raw_counts', 'normalized_counts_all_samples', 'biospecimen_metadata', 'sample_manifest_400')):
        return True
    if re.search(r'(^|/)(\.venv[^/]*|venv|node_modules)(/|$)', path):
        return True
    return False

def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], text=True, capture_output=True, check=True).stdout

def audit(repo):
    tracked = [p for p in git(repo, 'ls-files', '-z').split('\0') if p]
    staged = [p for p in git(repo, 'diff', '--cached', '--name-only', '-z').split('\0') if p]
    untracked = [p for p in git(repo, 'ls-files', '--others', '--exclude-standard', '-z').split('\0') if p]
    missing = [p for p in REQUIRED if not (repo / p).is_file()]
    unsafe_tracked = [p for p in tracked if unsafe_path(p)]
    unsafe_staged = [p for p in staged if unsafe_path(p)]
    unsafe_untracked = [p for p in untracked if unsafe_path(p)]
    return {
        'required_source_missing': missing,
        'unsafe_tracked_paths': unsafe_tracked,
        'unsafe_staged_paths': unsafe_staged,
        'unsafe_untracked_paths': unsafe_untracked,
        'tracked_file_count': len(tracked), 'untracked_file_count': len(untracked),
        'release_safe_to_commit_current_index': not (missing or unsafe_tracked or unsafe_staged),
        'raw_to_figure_reproduced': False,
        'raw_to_figure_note': 'Not proven by this checker; requires independent data-to-figure run.',
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, help='Optional JSON report outside the release repository')
    a = p.parse_args()
    repo = a.release_root.resolve()
    out = audit(repo)
    print(json.dumps(out, indent=2))
    if a.output:
        target = a.output.resolve()
        if target.is_relative_to(repo):
            p.error('Report output must be outside the release repository')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=2) + '\n')
    return 0 if out['release_safe_to_commit_current_index'] else 1

if __name__ == '__main__':
    raise SystemExit(main())

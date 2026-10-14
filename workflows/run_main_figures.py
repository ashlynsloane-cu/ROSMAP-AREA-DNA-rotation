#!/usr/bin/env python3
"""Run frozen ROSMAP-AREA main figure generators without overwriting published outputs.

Uses config/canonical_figure_inputs.csv, the canonical figure script manifest,
and additional_figure_inputs.json. Frozen external inputs are resolved locally.
This is local portability, not a redistribution approval or a fresh-clone guarantee.

Examples:
  python3 workflows/run_main_figures.py --dry-run
  python3 workflows/run_main_figures.py --figures 6,8
  python3 workflows/run_main_figures.py --all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows or any(key not in r for r in rows):
        raise ValueError(f'Invalid manifest: {path}')
    out = {r[key]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f'Duplicate keys in {path}')
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frozen-root', type=Path, default=None,
                        help='Local package root (default: REPO/data/frozen_inputs)')
    parser.add_argument('--output-root', type=Path, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true')
    group.add_argument('--figures', help='Comma-separated figure numbers, e.g. 1,6,8')
    parser.add_argument('--dry-run', action='store_true', help='Verify and display commands without executing generators')
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve()
    frozen = (args.frozen_root or repo / 'data/frozen_inputs').expanduser().resolve()
    output_root = (args.output_root or repo / 'finalization_audit/reproduction_tests/unified_run').expanduser().resolve()
    figures = read_manifest(repo / 'finalization_audit/repository_freeze/canonical_figure_manifest.csv', 'figure')
    inputs = read_manifest(repo / 'config/canonical_figure_inputs.csv', 'source_id')
    additional_path = repo / 'finalization_audit/repository_freeze/additional_figure_inputs.json'
    with additional_path.open(encoding='utf-8') as fh:
        additional = json.load(fh)
    if not all(k in additional for k in ('Figure6_formal_comparison', 'Figure8_protein_evidence')):
        raise ValueError(f'Missing required additional inputs in {additional_path}')
    chosen = [str(n) for n in range(1, 9)] if args.all or not args.figures else [s.strip() for s in args.figures.split(',')]
    if len(chosen) != len(set(chosen)) or any(x not in {str(n) for n in range(1, 9)} for x in chosen):
        parser.error('Expected unique figure numbers between 1 and 8')
    for n in chosen:
        if n not in figures:
            raise RuntimeError(f'Figure {n} not in figure manifest')

    def source(key: str) -> Path:
        r = inputs[key]
        base = repo if r['workspace'] == 'repo' else frozen if r['workspace'] == 'offline' else None
        if base is None:
            raise ValueError(f'Invalid workspace for {key}: {r["workspace"]}')
        p = (base / r['relative_path']).resolve()
        if not p.is_file():
            raise FileNotFoundError(f'{key}: {p}')
        if digest(p) != r['sha256']:
            raise RuntimeError(f'SHA-256 mismatch in {key}: {p}')
        return p

    # Check every recorded input, not only the selected figures.
    for key in inputs:
        source(key)
    print(f'PASS: {len(inputs)} frozen input hashes')

    for n in chosen:
        row = figures[n]
        script = (repo / row['script']).resolve()
        if not script.is_file():
            raise FileNotFoundError(script)
        if digest(script) != row['sha256']:
            raise RuntimeError(f'Figure {n} script SHA-256 mismatch: {script}')
    print(f'PASS: {len(chosen)} figure script hashes')

    def additional_source(key: str) -> Path:
        entry = additional[key]
        if not isinstance(entry, dict) or not all(k in entry for k in ('path', 'sha256')):
            raise ValueError(f'Malformed additional input record: {key}')
        p = (repo / entry['path']).resolve()
        if not p.is_relative_to(repo):
            raise ValueError(f'Additional source escapes repository: {key}')
        if not p.is_file():
            raise FileNotFoundError(f'{key}: {p}')
        if digest(p) != entry['sha256']:
            raise RuntimeError(f'SHA-256 mismatch in {key}: {p}')
        return p

    formal_comparison = additional_source('Figure6_formal_comparison')
    protein_input = additional_source('Figure8_protein_evidence')
    print('PASS: 2 additional figure input hashes')

    for n in chosen:
        path = str(output_root / f'figure{n}')
        script = str((repo / figures[n]['script']).resolve())
        common = ['--repo-root', str(repo)]
        if n == '1':
            params = common + ['--output-dir', path]
        elif n == '2':
            params = common + ['--source', str(source('F2_architecture')), '--output-dir', path]
        elif n == '3':
            interpretation = repo / 'results/multiomic_concordance/STAGE9_SIGNAL_ARCHITECTURE_FINAL_INTERPRETATION.txt'
            if not interpretation.is_file():
                raise FileNotFoundError(interpretation)
            params = common + ['--model-source', str(source('F3_adjusted_models')), '--overlap-source', str(interpretation), '--output-dir', path]
        elif n == '4':
            params = common + ['--stage10-root', str(repo / 'results/stage10_multimodal_simulations'), '--replicate', '1', '--output-dir', path]
        elif n == '5':
            params = common + ['--comparison-dir', str(source('F5_comparison').parent), '--attribution-dir', str(source('F5_shapley').parent), '--output-dir', path]
        elif n == '6':
            params = common + ['--source', str(source('F6_architecture')), '--formal-comparison', str(formal_comparison), '--evidence-source', str(source('F7_evidence')), '--output-dir', path]
        elif n == '7':
            params = common + ['--leading-edge-source', str(source('F7_leading_edges')), '--enrichment-source', str(source('F7_enrichment')), '--evidence-source', str(source('F7_evidence')), '--gene-annotation-source', str(source('F7_annotations')), '--output-dir', path]
        else:
            params = ['--input', str(protein_input), '--output-root', path]
        cmd = [sys.executable, script] + params
        print(f'FIGURE {n}:', ' '.join(subprocess.list2cmdline([part]) for part in cmd), flush=True)
        if not args.dry_run:
            dest = Path(path)
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f'Refusing to overwrite non-empty output directory: {dest}')
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, cwd=repo, check=True)
    print('PASS: commands validated' if args.dry_run else 'PASS: figure generators completed')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Audit fresh primary models and optionally reproduce separately labelled historical figures.

Does not claim that historical figure replay is downstream of fresh models.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_outputs(root):
    root = Path(root)
    contrasts = ('NCI_vs_AD', 'NCI_vs_MCI', 'MCI_vs_AD', '0-2_vs_3-4',
                 '3-4_vs_5-6', '1-2_vs_3-4')
    axes = ('cogdx', 'cogdx', 'cogdx', 'Braak', 'Braak', 'CERAD')
    expected = [(axis, contrast) for axis, contrast in zip(axes, contrasts)]
    findings = {}
    for modality, folder in [('monocyte', 'monocyte_all'), ('dlpfc', 'brain_area/dlpfc'),
                             ('tmt', 'brain_area/tmt')]:
        for axis, contrast in expected:
            d = root / 'models' / folder / axis / contrast
            # Require a model result CSV, not just a run directory.
            files = sorted(p for p in d.glob('*.csv') if p.is_file()) if d.is_dir() else []
            findings[f'area/{modality}/{axis}/{contrast}'] = {
                'directory': str(d), 'result_csvs': len(files),
                'present': bool(files),
            }
    for modality in ('dlpfc', 'tmt'):
        for axis, contrast in expected:
            d = root / 'models/conventional' / modality / axis / contrast
            files = sorted(p for p in d.glob('*.csv') if p.is_file()) if d.is_dir() else []
            findings[f'conventional/{modality}/{axis}/{contrast}'] = {
                'directory': str(d), 'result_csvs': len(files),
                'present': bool(files),
            }
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--pipeline-output', required=True, type=Path)
    parser.add_argument('--handoff-root', required=True, type=Path)
    parser.add_argument('--frozen-root', type=Path,
                        help='Historical offline reference root, required for --replay-figures')
    parser.add_argument('--replay-figures', action='store_true',
                        help='Generate eight HISTORICAL-REPLAY figures (not from fresh models)')
    args = parser.parse_args(argv)
    repo = args.repo_root.expanduser().resolve()
    outputs = args.pipeline_output.expanduser().resolve()
    handoff = args.handoff_root.expanduser().resolve()
    if not repo.is_dir() or not outputs.is_dir():
        parser.error('Repository and completed pipeline output directory must exist')
    if handoff == repo or handoff == outputs or repo in handoff.parents or outputs in handoff.parents:
        parser.error('Handoff output must be separate from repository and controlled-data output')
    findings = find_outputs(outputs)
    fresh_present = sum(x['present'] for x in findings.values())
    manifest = outputs / 'manifests/manifest_index.csv'
    receipts = {
        'fresh_area': outputs / 'models/fresh_area_stage_complete.json',
        'raw_to_figures': outputs / 'raw_to_figures_report.json',
    }
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'handoff_audit_and_optional_historical_figure_replay',
        'fresh_model_expected': 30,
        'fresh_model_csv_directories_found': fresh_present,
        'manifest_index_present': manifest.is_file(),
        'model_directories': findings,
        'pipeline_receipts': {k: {'exists': v.is_file(), 'sha256': sha256(v) if v.is_file() else None}
                              for k, v in receipts.items()},
        'fresh_output_to_figures_executed': False,
        'historical_figure_replay_executed': False,
    }
    handoff.mkdir(parents=True, exist_ok=True)
    if args.replay_figures:
        if not args.frozen_root:
            parser.error('--frozen-root is required for --replay-figures')
        figure_script = repo / 'workflows/run_main_figures.py'
        if not figure_script.is_file():
            parser.error(f'Historical figure runner missing: {figure_script}')
        figure_dest = handoff / 'historical_replay_figures_1_to_8'
        if figure_dest.exists() and any(figure_dest.iterdir()):
            parser.error(f'Refusing to overwrite nonempty figure directory: {figure_dest}')
        cmd = [sys.executable, str(figure_script), '--repo-root', str(repo),
               '--frozen-root', str(args.frozen_root.expanduser().resolve()),
               '--output-root', str(figure_dest), '--all']
        subprocess.run(cmd, check=True)
        report['historical_figure_replay_executed'] = True
        report['historical_figure_folder'] = str(figure_dest)
    target = handoff / 'HANDOFF_PROVENANCE.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'FRESH_MODEL_RESULT_DIRS_PRESENT: {fresh_present}/30')
    print(f'HISTORICAL_FIGURE_REPLAY_EXECUTED: {report["historical_figure_replay_executed"]}')
    print('FRESH_OUTPUT_TO_FIGURES_COMPLETE: False')
    print(f'HANDOFF_REPORT: {target}')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f'Figure replay failed with exit status {exc.returncode}', file=sys.stderr)
        sys.exit(exc.returncode or 1)

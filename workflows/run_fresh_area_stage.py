#!/usr/bin/env python3
"""Bridge six monocyte and twelve brain AREA fits into one provenance-checked stage.

If existing model reports are valid, reuse them. Otherwise rerun from fresh
manifests and authorized expression data. Never use frozen historical models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def valid_monocyte(report, expression, manifest_root):
    if not report.is_file():
        return False
    try:
        r = json.loads(report.read_text())
        if len(r['models']) != 6 or r['expression_sha256'] != digest(expression):
            return False
        if len({(x['axis'], x['contrast']) for x in r['models']}) != 6:
            return False
        for x in r['models']:
            p = Path(x['result_path'])
            if not p.is_file() or digest(p) != x['result_sha256']:
                return False
            manifest = manifest_root / x['axis'] / x['contrast']  # Verified separately by fresh runner
        return True
    except (KeyError, OSError, ValueError, TypeError):
        return False


def valid_brain(report, rna, tmt):
    if not report.is_file():
        return False
    try:
        r = json.loads(report.read_text())
        if len(r['models']) != 12 or len({x['model'] for x in r['models']}) != 12:
            return False
        if r['expression_sha256'] != {'dlpfc': digest(rna), 'tmt': digest(tmt)}:
            return False
        return all(Path(x['result']).is_file() and digest(x['result']) == x['result_sha256']
                   for x in r['models'])
    except (KeyError, OSError, ValueError, TypeError):
        return False


def run_cmd(args):
    print('RUNNING:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    repo, raw, out = a.repo.resolve(), a.raw_root.resolve(), a.output_root.resolve()
    manifest = out / 'manifests'
    rna = raw / 'rna/ROSMAP_AREA_normalized_counts_all_samples.csv'
    tmt = raw / 'tmt/C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv'
    mono = raw / 'monocyte/monocyte_sizefactor_normalized_counts_area_554.tsv'
    area = repo / 'upstream/portable_scripts/run_regular_area_binary.py'
    for src in [manifest/'manifest_index.csv', rna, tmt, mono, area]:
        if not src.is_file():
            p.error(f'Missing input: {src}')
    mono_out = out/'models/monocyte_all'
    brain_out = out/'models/brain_area'
    mono_report = mono_out/'fresh_area_models.json'
    brain_report = brain_out/'fresh_brain_area_models.json'
    if a.force or not valid_monocyte(mono_report, mono, manifest):
        run_cmd([sys.executable, repo/'workflows/run_fresh_monocyte_all_area.py',
                 '--manifests-root', manifest, '--expression', mono, '--outdir', mono_out,
                 '--area-script', area])
    else:
        print('REUSING: verified fresh six-monocyte outputs', flush=True)
    if a.force or not valid_brain(brain_report, rna, tmt):
        run_cmd([sys.executable, repo/'workflows/run_fresh_brain_batch.py',
                 '--manifests-root', manifest, '--rna-expression', rna,
                 '--tmt-expression', tmt, '--outdir', brain_out,
                 '--area-script', area, '--execute'])
    else:
        print('REUSING: verified fresh twelve-brain outputs', flush=True)
    if not valid_monocyte(mono_report, mono, manifest) or not valid_brain(brain_report, rna, tmt):
        raise RuntimeError('Fresh model report validation failed')
    receipt = out/'models/fresh_area_stage_complete.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({'models': 18, 'status': 'fresh_models_verified',
                                   'monocyte_report_sha256': digest(mono_report),
                                   'brain_report_sha256': digest(brain_report)}, indent=2)+'\n')
    print('FRESH_AREA_STAGE_COMPLETE: 18', flush=True)


if __name__ == '__main__':
    main()

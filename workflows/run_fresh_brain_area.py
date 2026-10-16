#!/usr/bin/env python3
"""Fit AREA for one fresh DLPFC or TMT discovery cohort.

Fail closed on unsupported columns; the analyst must supply covariates matching
its validated scientific specification. No historical paths or results used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def select_manifest(index, modality, contrast, axis):
    cohort = 'DLPFC discovery' if modality == 'dlpfc' else 'TMT discovery'
    mask = (index['axis'].eq(axis) & index['contrast'].eq(contrast)
            & index['cohort'].eq(cohort) & index['role'].eq('primary')
            & index['modality'].eq(modality))
    matched = index.loc[mask]
    if len(matched) != 1:
        raise ValueError(f'Expected one {modality} {axis} {contrast} discovery cohort, found {len(matched)}')
    return matched.iloc[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('manifests-root', 'expression', 'outdir', 'area-script'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--modality', choices=['dlpfc', 'tmt'], required=True)
    p.add_argument('--axis', choices=['cogdx', 'Braak', 'CERAD'], default='cogdx')
    p.add_argument('--contrast', required=True)
    p.add_argument('--continuous', required=True, help='Comma-delimited, validated continuous covariates')
    p.add_argument('--categorical', required=True, help='Comma-delimited, validated categorical covariates; use empty string if none')
    p.add_argument('--orientation', choices=['features_by_samples', 'samples_by_features'], required=True)
    p.add_argument('--feature-id-col', default=None)
    p.add_argument('--global-complete-features', action='store_true')
    a = p.parse_args(argv)
    root = a.manifests_root.resolve(strict=True)
    index_path = root / 'manifest_index.csv'
    index = pd.read_csv(index_path)
    row = select_manifest(index, a.modality, a.contrast, a.axis)
    manifest = Path(str(row['manifest'])).resolve(strict=True)
    if not manifest.is_relative_to(root):
        raise ValueError('Manifest path points outside freshly generated manifest directory')
    m = pd.read_csv(manifest)
    covariates = [v for group in (a.continuous, a.categorical) for v in group.split(',') if v]
    needed = ['sample_id', 'analysis_group'] + covariates
    missing = [v for v in needed if v not in m.columns]
    if missing:
        raise ValueError(f'Missing required covariates in fresh manifest: {missing}. Available: {list(m.columns)}')
    if m[needed].isna().any().any() or m.sample_id.duplicated().any():
        raise ValueError('Missing values/duplicate samples in model manifest')
    if len(m) != int(row['n']):
        raise ValueError('Fresh manifest cohort size mismatch')
    groups = {str(row['group_A']), str(row['group_B'])}
    if set(m['analysis_group'].astype(str)) != groups:
        raise ValueError('Fresh manifest group labels differ from manifest index')
    expression = a.expression.resolve(strict=True)
    script = a.area_script.resolve(strict=True)
    outdir = a.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    contrast_name = f'{a.modality}_{a.axis}_{a.contrast}'
    cmd = [sys.executable, str(script), '--expression', str(expression),
           '--manifest', str(manifest), '--outdir', str(outdir),
           '--contrast-name', contrast_name, '--group-a', str(row['group_A']),
           '--group-b', str(row['group_B']), '--orientation', a.orientation,
           '--continuous', a.continuous, '--categorical', a.categorical]
    if a.feature_id_col:
        cmd.extend(['--feature-id-col', a.feature_id_col])
    if a.global_complete_features:
        cmd.append('--global-complete-features')
    subprocess.run(cmd, check=True)
    result = outdir / f'{contrast_name}_Regular_AREA_all_results.csv'
    model_manifest = outdir / 'run_manifest.json'
    if not result.is_file() or not model_manifest.is_file():
        raise RuntimeError('Missing generated AREA result or model manifest')
    receipt = {'status': 'fresh_brain_area_executed', 'axis': a.axis,
               'contrast': a.contrast, 'modality': a.modality,
               'n': len(m), 'manifest_sha256': digest(manifest),
               'expression_sha256': digest(expression),
               'model_script_sha256': digest(script),
               'result_sha256': digest(result),
               'result': str(result), 'covariates': covariates,
               'global_complete_features': a.global_complete_features}
    (outdir / 'fresh_model_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('FRESH_BRAIN_AREA_COMPLETE:', contrast_name)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reproduce historical phenotype-only Stage11 DESeq2 and limma models.

Reads frozen reference and original research inputs; writes only to output root.
A preflight must succeed before execution. Comparison is intentionally strict.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTRASTS = ('cogdx_NCI_vs_AD', 'cogdx_NCI_vs_MCI', 'cogdx_MCI_vs_AD',
             'Braak_0-2_vs_3-4', 'Braak_3-4_vs_5-6', 'CERAD_3-4_vs_1-2')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release-root', 'research-root', 'reference-root', 'output-root'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-jobs', type=int, default=0)
    p.add_argument('--modalities', choices=('both', 'dlpfc', 'tmt'), default='both')
    p.add_argument('--stop-on-failure', action='store_true')
    p.add_argument('--rtol', type=float, default=1e-7)
    p.add_argument('--atol', type=float, default=1e-10)
    return p.parse_args(argv)


def resolve_historical(value, roots):
    if pd.isna(value) or not str(value).strip():
        return None
    raw = str(value).strip()
    marker = '/rosmap_multiomic_slurm/'
    candidates = []
    if marker in raw:
        rel = raw.split(marker, 1)[1]
        candidates += [r / rel for r in roots]
    else:
        raw_path = Path(raw).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend(r / raw_path for r in roots)
    matches = [p.resolve() for p in candidates if p.is_file()]
    return matches[0] if matches else None


def compare_table(expected, observed, rtol, atol):
    a = pd.read_csv(expected, low_memory=False)
    b = pd.read_csv(observed, low_memory=False)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {'status': 'fail', 'reason': 'different columns or row count'}
    if 'feature_id' not in a or a.feature_id.duplicated().any() or b.feature_id.duplicated().any():
        return {'status': 'fail', 'reason': 'invalid feature IDs'}
    a = a.set_index('feature_id').sort_index()
    b = b.set_index('feature_id').sort_index()
    if not a.index.equals(b.index):
        return {'status': 'fail', 'reason': 'different feature universes'}
    bad = {}
    max_delta = 0.0
    for col in a:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xx = x.to_numpy(dtype=float)
            yy = y.to_numpy(dtype=float)
            good = np.isclose(xx, yy, rtol=rtol, atol=atol, equal_nan=True)
            if not good.all():
                bad[col] = int((~good).sum())
            finite = np.isfinite(xx) & np.isfinite(yy)
            if finite.any():
                max_delta = max(max_delta, float(np.max(np.abs(xx[finite] - yy[finite]))))
        else:
            xx = x.fillna('<NA>').astype(str)
            yy = y.fillna('<NA>').astype(str)
            nbad = int((xx != yy).sum())
            if nbad:
                bad[col] = nbad
    return {'status': 'pass' if not bad else 'fail',
            'features': len(a), 'mismatches': bad,
            'max_numeric_abs_difference': max_delta}


def compare_summary(expected, observed, rtol, atol):
    a = pd.read_csv(expected).iloc[0]
    b = pd.read_csv(observed).iloc[0]
    if set(a.index) != set(b.index):
        return {'status': 'fail', 'reason': 'summary fields differ'}
    fields = {}
    # File paths may legitimately change; provenance captured separately.
    for key in a.index:
        if key in ('counts_file', 'column_map'):
            continue
        x, y = a[key], b[key]
        if pd.isna(x) and pd.isna(y):
            continue
        if pd.api.types.is_number(x) and pd.api.types.is_number(y):
            if not np.isclose(float(x), float(y), rtol=rtol, atol=atol, equal_nan=True):
                fields[key] = {'expected': str(x), 'actual': str(y)}
        elif str(x) != str(y):
            fields[key] = {'expected': str(x), 'actual': str(y)}
    return {'status': 'pass' if not fields else 'fail', 'mismatches': fields}


def make_jobs(args):
    reference = args.reference_root / 'results/stage11_conventional_covariate_sensitivity/phenotype_only'
    tasks = pd.read_csv(args.reference_root / 'results/stage11_covariate_stress/stage11_tasks.tsv', sep='\t')
    roots = [args.reference_root, args.research_root]
    jobs = []
    selected_modalities = ('dlpfc', 'tmt') if args.modalities == 'both' else (args.modalities,)
    for contrast in CONTRASTS:
        for mode in selected_modalities:
            hit = tasks[(tasks.contrast_id == contrast) & (tasks.modality == mode)]
            if len(hit) != 1:
                raise ValueError(f'Missing/duplicate Stage11 task: {contrast} {mode}')
            task = hit.iloc[0]
            folder = reference / contrast / mode
            summary_path = folder / 'run_summary.csv'
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summary = pd.read_csv(summary_path).iloc[0]
            group_a, group_b = str(summary.group_A), str(summary.group_B)
            if (group_a, group_b) != (str(task.group_a), str(task.group_b)):
                raise ValueError(f'Historical conventional group orientation differs from task manifest: {contrast} {mode}')
            label = str(summary.contrast)
            method = 'DESeq2' if mode == 'dlpfc' else 'limma'
            suffix = '_DESeq2_all_results.csv' if mode == 'dlpfc' else '_limma_all_results.csv'
            result = folder / f'{label}{suffix}'
            if not result.is_file():
                raise FileNotFoundError(result)
            manifest = resolve_historical(task.manifest, roots)
            if manifest is None:
                raise FileNotFoundError(f'Manifest not found for {contrast} {mode}: {task.manifest}')
            if mode == 'dlpfc':
                data = resolve_historical(summary.counts_file, roots)
                column_map = resolve_historical(summary.column_map, roots)
                if data is None:
                    raise FileNotFoundError(f'Raw DESeq2 counts not found for {contrast}: {summary.counts_file}')
                if pd.notna(summary.column_map) and str(summary.column_map).strip() and column_map is None:
                    raise FileNotFoundError(f'DESeq2 column map missing: {summary.column_map}')
                script = args.research_root / 'scripts/multiomic/run_rna_deseq2_binary.R'
            else:
                data = resolve_historical(task.expression, roots)
                column_map = None
                if data is None:
                    raise FileNotFoundError(f'TMT input missing: {task.expression}')
                script = args.research_root / 'scripts/multiomic/run_tmt_limma_binary.R'
            if not script.is_file():
                raise FileNotFoundError(script)
            # Phenotype-only: no adjusted covariates, explicitly suppress limma defaults.
            command = ['Rscript', str(script),
                       '--manifest', str(manifest), '--outdir', '__OUTPUT__',
                       '--contrast-name', label,
                       '--group-a', group_a, '--group-b', group_b,
                       '--continuous', '', '--categorical', '']
            if mode == 'dlpfc':
                command += ['--counts', str(data)]
                if column_map is not None:
                    command += ['--column-map', str(column_map)]
            else:
                command += ['--matrix', str(data), '--expected-features', str(int(summary.n_features))]
            jobs.append({'id': f'{contrast}__{mode}', 'label': label,
                         'method': method, 'contrast': contrast, 'modality': mode,
                         'source_file': str(data), 'manifest_file': str(manifest),
                         'historical_summary': str(summary_path), 'historical_result': str(result),
                         'command': command})
    return jobs[:args.max_jobs] if args.max_jobs else jobs


def verify(job, target, args):
    filename = Path(job['historical_result']).name
    actual = target / filename
    summ = target / 'run_summary.csv'
    if not actual.is_file() or not summ.is_file():
        return {'status': 'fail', 'reason': 'missing result or summary'}
    result = compare_table(job['historical_result'], actual, args.rtol, args.atol)
    summary = compare_summary(job['historical_summary'], summ, args.rtol, args.atol)
    return {'status': 'pass' if result['status'] == summary['status'] == 'pass' else 'fail',
            'result': result, 'summary': summary}


def main(argv=None):
    args = arguments(argv)
    args.release_root = args.release_root.expanduser().resolve()
    args.research_root = args.research_root.expanduser().resolve()
    args.reference_root = args.reference_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for source in (args.release_root, args.research_root, args.reference_root):
        if args.output_root == source or source in args.output_root.parents:
            raise SystemExit(f'Unsafe output root inside source tree: {args.output_root}')
    jobs = make_jobs(args)
    if not jobs:
        raise SystemExit('No jobs selected')
    print(f'Conventional Stage11 phenotype-only models: {len(jobs)} jobs')
    for i, job in enumerate(jobs, 1):
        print(f"{i:02d} {job['id']} [{job['method']}] counts/matrix: {job['source_file']}")
    if not args.execute:
        print('PREFLIGHT ONLY. No files created; use --execute to run.')
        return 0
    if args.output_root.exists() and not args.resume:
        raise SystemExit('Output root exists; use a fresh root or --resume')
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / 'conventional_batch_report.json'
    records = []
    for index, job in enumerate(jobs, 1):
        dest = args.output_root / job['id']
        record = {'id': job['id'], 'method': job['method'], 'reference': job['historical_result'],
                  'source_file': job['source_file']}
        if dest.exists():
            if not args.resume:
                raise SystemExit(f'Output folder already exists: {dest}')
            result = verify(job, dest, args)
            record.update(result)
            record['execution'] = 'resume-verified' if result['status'] == 'pass' else 'existing-failed'
        else:
            cmd = [str(dest) if a == '__OUTPUT__' else a for a in job['command']]
            dest.mkdir(parents=True, exist_ok=False)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            (args.output_root / f"{job['id']}.log").write_text(proc.stdout + '\n' + proc.stderr)
            record['exit_code'] = proc.returncode
            record['execution'] = 'executed'
            if proc.returncode:
                record.update(status='fail', reason='R subprocess failed; inspect log')
            else:
                record.update(verify(job, dest, args))
        records.append(record)
        report_path.write_text(json.dumps({'jobs': records, 'rtol': args.rtol, 'atol': args.atol}, indent=2))
        print(f"[{index}/{len(jobs)}] {job['id']} {record['status']}", flush=True)
        if record['status'] != 'pass' and args.stop_on_failure:
            break
    n = sum(row['status'] == 'pass' for row in records)
    print(f'Passed {n}/{len(jobs)}; report: {report_path}')
    return 0 if n == len(jobs) else 1


if __name__ == '__main__':
    sys.exit(main())

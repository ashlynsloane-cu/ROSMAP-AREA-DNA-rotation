#!/usr/bin/env python3
"""Reproduce Stage 11 brain intercept-only and intermediate AREA models.

Historical A models (fully adjusted) are intentionally not rerun here.
Outputs must be outside release, research and reference trees.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = {'dlpfc', 'tmt'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release-root', 'reference-root', 'research-root', 'output-root'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--execute', action='store_true', help='Otherwise preflight only')
    p.add_argument('--resume', action='store_true', help='Reuse verified results in existing output root')
    p.add_argument('--task-ids', default='', help='Base IDs, e.g. 2 or 2,3 (default all brain)')
    p.add_argument('--kind', choices=['all', 'b', 'subset'], default='all')
    p.add_argument('--max-jobs', type=int, default=0, help='Limit selected jobs, useful for pilot')
    p.add_argument('--stop-on-failure', action='store_true')
    p.add_argument('--rtol', type=float, default=1e-9)
    p.add_argument('--atol', type=float, default=1e-12)
    return p.parse_args(argv)


def locate(value, roots):
    original = str(value)
    marker = '/rosmap_multiomic_slurm/'
    if marker in original:
        relative = original.split(marker, 1)[1]
        for root in roots:
            candidate = root / relative
            if candidate.exists():
                return candidate.resolve()
    candidate = Path(original).expanduser()
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f'Cannot locate historical path: {original}')


def covars(value):
    if pd.isna(value):
        return ''
    return ','.join(part.strip() for part in str(value).split(',') if part.strip())


def yes(value):
    return str(value).strip().lower() in ('true', '1', 'yes', 'y', 't')


def reference_result(folder):
    hits = sorted(folder.glob('*_Regular_AREA_all_results.csv'))
    if len(hits) != 1:
        raise ValueError(f'Expected one reference result in {folder}; found {len(hits)}')
    return hits[0]


def make_jobs(args):
    roots = (args.reference_root, args.research_root)
    s11 = args.reference_root / 'results/stage11_covariate_stress'
    base = pd.read_csv(s11 / 'stage11_tasks.tsv', sep='\t')
    subset = pd.read_csv(s11 / 'covariate_attribution/covariate_subset_tasks.tsv', sep='\t')
    if len(base) != 18 or len(subset) != 300:
        raise ValueError(f'Unexpected historical task manifest lengths: {len(base)}, {len(subset)}')
    wanted = {int(i) for i in args.task_ids.split(',') if i.strip()}
    eligible = {int(r.task_id) for r in base.itertuples() if r.modality in MODALITIES}
    if wanted - eligible:
        raise ValueError(f'Task IDs not brain tasks: {sorted(wanted - eligible)}')
    selected = base[base.modality.isin(MODALITIES)]
    if wanted:
        selected = selected[selected.task_id.isin(wanted)]
    jobs = []
    for row in selected.itertuples(index=False):
        expression = locate(row.expression, roots)
        sample_manifest = locate(row.manifest, roots)
        original_a = json.loads(locate(row.A_run_manifest, roots).read_text())
        orientation = str(row.orientation)
        if original_a['orientation'] != orientation:
            raise ValueError(f'Orientation mismatch in task {row.task_id}')
        historical_pairs = ((str(original_a['group_A']), str(original_a['group_B'])),
                            (str(row.group_a), str(row.group_b)))
        if set(historical_pairs[0]) != set(historical_pairs[1]):
            raise ValueError(f'Group mismatch in task {row.task_id}: {historical_pairs}')
        common = dict(base_task_id=int(row.task_id), modality=str(row.modality),
                      contrast_id=str(row.contrast_id), expression=expression,
                      manifest=sample_manifest, orientation=orientation,
                      group_a=str(row.group_a), group_b=str(row.group_b),
                      complete=yes(row.global_complete_features),
                      expected_n=int(row.A_n), expected_features=int(row.A_n_features),
                      fdr=float(original_a['fdr']))
        if args.kind in ('all', 'b'):
            reference_dir = locate(row.B_outdir, roots)
            original_b = json.loads((reference_dir / 'run_manifest.json').read_text())
            if (str(original_b['group_A']), str(original_b['group_B'])) != historical_pairs[1]:
                raise ValueError(f'B orientation does not match Stage11 task {row.task_id}')
            if original_b.get('continuous_covariates') != [] or original_b.get('categorical_covariates') != []:
                raise ValueError(f'B model not intercept-only, task {row.task_id}')
            jobs.append(dict(common, kind='b', job_id=f'B_{int(row.task_id):02d}',
                             continuous='', categorical='',
                             contrast=str(original_b['contrast']),
                             reference=reference_result(reference_dir),
                             original_manifest=original_b))
        if args.kind in ('all', 'subset'):
            sub = subset[(subset.base_task_id == row.task_id) & (subset.modality == row.modality)]
            expected_count = 2 ** len({'dlpfc': ['age', 'rin', 'pmi', 'sex', 'sequencing_batch'],
                                       'tmt': ['age', 'pmi', 'sex', 'batch']}[row.modality]) - 2
            if len(sub) != expected_count:
                raise ValueError(f'Unexpected subset count for {row.task_id}: {len(sub)}')
            for item in sub.itertuples(index=False):
                ref_result = locate(item.result_path, roots)
                original_manifest_path = ref_result.parent / 'run_manifest.json'
                original = json.loads(original_manifest_path.read_text())
                if (str(original['group_A']), str(original['group_B'])) != historical_pairs[1]:
                    raise ValueError(f'Subset orientation mismatch {item.subset_task_id}')
                continuous = covars(item.continuous)
                categorical = covars(item.categorical)
                if original['continuous_covariates'] != ([v for v in continuous.split(',') if v]):
                    raise ValueError(f'Continuous mismatch subset {item.subset_task_id}')
                if original['categorical_covariates'] != ([v for v in categorical.split(',') if v]):
                    raise ValueError(f'Categorical mismatch subset {item.subset_task_id}')
                jobs.append(dict(common, kind='subset',
                                 job_id=f'S_{int(item.subset_task_id):03d}',
                                 continuous=continuous, categorical=categorical,
                                 contrast=str(item.contrast_name),
                                 reference=ref_result, original_manifest=original))
    return jobs[:args.max_jobs] if args.max_jobs else jobs


def command(script, job, outdir):
    cmd = [sys.executable, str(script), '--expression', str(job['expression']),
           '--manifest', str(job['manifest']), '--outdir', str(outdir),
           '--contrast-name', job['contrast'], '--group-a', job['group_a'],
           '--group-b', job['group_b'], '--orientation', job['orientation'],
           '--fdr', str(job['fdr'])]
    if job['continuous']:
        cmd += ['--continuous', job['continuous']]
    if job['categorical']:
        cmd += ['--categorical', job['categorical']]
    if job['complete']:
        cmd.append('--global-complete-features')
    return cmd


def compare_csv(reference, actual, rtol, atol):
    a = pd.read_csv(reference, low_memory=False)
    b = pd.read_csv(actual, low_memory=False)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {'status': 'fail', 'reason': 'schema or row-count mismatch'}
    if 'feature_id' not in a.columns or a.feature_id.duplicated().any() or b.feature_id.duplicated().any():
        return {'status': 'fail', 'reason': 'missing or duplicated feature_id'}
    a = a.set_index('feature_id').sort_index()
    b = b.set_index('feature_id').sort_index()
    if not a.index.equals(b.index):
        return {'status': 'fail', 'reason': 'feature universe mismatch'}
    bad = {}
    max_delta = 0.0
    allowed_direction_flips = []
    # Sign labels of residualized rank statistics are undefined at numerical zero.
    # This exception never changes the numeric comparison or any FDR checks.
    near_zero_threshold = 1e-10
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            match = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
            if not match.all():
                bad[col] = int((~match).sum())
            finite = np.isfinite(xv) & np.isfinite(yv)
            if finite.any():
                max_delta = max(max_delta, float(np.max(np.abs(xv[finite] - yv[finite]))))
        elif col == 'direction':
            ax = x.fillna('<NA>').astype(str)
            by = y.fillna('<NA>').astype(str)
            disagree = ax.ne(by)
            if disagree.any():
                if 'partial_rank_statistic' not in a or 'partial_rank_statistic' not in b:
                    bad[col] = int(disagree.sum())
                    continue
                ar = pd.to_numeric(a['partial_rank_statistic'], errors='coerce').to_numpy(dtype=float)
                br = pd.to_numeric(b['partial_rank_statistic'], errors='coerce').to_numpy(dtype=float)
                near_zero = (np.isfinite(ar) & np.isfinite(br)
                             & (np.abs(ar) <= near_zero_threshold)
                             & (np.abs(br) <= near_zero_threshold))
                # Only an opposite sign-label pair with the same phenotype prefix
                # qualifies; arbitrary label changes still fail.
                suffix_up = '_associated_higher_rank'
                suffix_down = '_associated_lower_rank'
                flip = np.array([
                    (u.endswith(suffix_up) and v == u[:-len(suffix_up)] + suffix_down)
                    or (u.endswith(suffix_down) and v == u[:-len(suffix_down)] + suffix_up)
                    for u, v in zip(ax.to_numpy(), by.to_numpy())
                ], dtype=bool)
                allowed = disagree.to_numpy() & near_zero & flip
                rejected = disagree.to_numpy() & ~allowed
                allowed_direction_flips = a.index[allowed].astype(str).tolist()
                if rejected.any():
                    bad[col] = int(rejected.sum())
        else:
            match = x.fillna('<NA>').astype(str).equals(y.fillna('<NA>').astype(str))
            if not match:
                bad[col] = 'non-numeric mismatch'
    return {'status': 'pass' if not bad else 'fail', 'n_features': len(a),
            'max_numeric_abs_difference': max_delta, 'discrepancies': bad,
            'near_zero_direction_exception_threshold': near_zero_threshold,
            'near_zero_direction_exception_count': len(allowed_direction_flips),
            'near_zero_direction_exception_feature_ids': allowed_direction_flips}


def check_manifest(reference, generated):
    # Exact historical model metadata should reproduce, including group direction.
    return {k: {'expected': reference.get(k), 'generated': generated.get(k)}
            for k in sorted(set(reference) | set(generated))
            if reference.get(k) != generated.get(k)}


def verify(job, outdir, rtol, atol):
    expected_file = outdir / f"{job['contrast']}_Regular_AREA_all_results.csv"
    manifest = outdir / 'run_manifest.json'
    if not expected_file.is_file() or not manifest.is_file():
        return {'status': 'fail', 'error': 'missing generated result or manifest'}
    results = compare_csv(job['reference'], expected_file, rtol, atol)
    manifest_diffs = check_manifest(job['original_manifest'], json.loads(manifest.read_text()))
    return {'status': 'pass' if results['status'] == 'pass' and not manifest_diffs else 'fail',
            'comparison': results, 'manifest_differences': manifest_diffs}


def write_report(path, summary):
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')


def main(argv=None):
    a = parse_args(argv)
    for attr in ('release_root', 'research_root', 'reference_root', 'output_root'):
        setattr(a, attr, getattr(a, attr).expanduser().resolve())
    script = a.release_root / 'upstream/portable_scripts/run_regular_area_binary.py'
    if not script.is_file():
        raise SystemExit(f'Missing AREA executable: {script}')
    if a.max_jobs < 0 or a.rtol < 0 or a.atol < 0:
        raise SystemExit('Invalid negative numeric argument')
    for protected in (a.release_root, a.research_root, a.reference_root):
        if a.output_root == protected or protected in a.output_root.parents:
            raise SystemExit(f'Unsafe output inside input/source tree: {a.output_root}')
    jobs = make_jobs(a)
    if not jobs:
        raise SystemExit('No jobs selected')
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise SystemExit('Duplicated job IDs')
    if a.output_root.exists() and not a.resume:
        raise SystemExit('Output directory exists; use --resume or a fresh destination')
    if a.resume and not a.output_root.is_dir():
        raise SystemExit('--resume requires an existing output directory')
    print(f'Stage 11 brain sensitivity: {len(jobs)} jobs', flush=True)
    print('B:', sum(j['kind'] == 'b' for j in jobs),
          'subsets:', sum(j['kind'] == 'subset' for j in jobs), flush=True)
    print('Output:', a.output_root, flush=True)
    if not a.execute:
        print('PREFLIGHT ONLY; no outputs created', flush=True)
        return 0
    a.output_root.mkdir(parents=True, exist_ok=True)
    report_path = a.output_root / 'stage11_sensitivity_report.json'
    report = {'configuration': {'task_ids': a.task_ids, 'kind': a.kind,
                                'max_jobs': a.max_jobs, 'rtol': a.rtol, 'atol': a.atol},
              'jobs': []}
    for i, job in enumerate(jobs, 1):
        dest = a.output_root / job['job_id']
        item = {'job_id': job['job_id'], 'base_task_id': job['base_task_id'],
                'kind': job['kind'], 'modality': job['modality'],
                'contrast_id': job['contrast_id'],
                'reference': str(job['reference']), 'output_dir': str(dest)}
        if a.resume and dest.is_dir():
            previous = verify(job, dest, a.rtol, a.atol)
            if previous['status'] == 'pass':
                item.update(previous)
                item['execution'] = 'previously_verified'
                report['jobs'].append(item)
                print(f'[{i}/{len(jobs)}] {job["job_id"]} resume verified', flush=True)
                write_report(report_path, report)
                continue
            # Fail-closed: never overwrite a failed or partial run silently.
            item.update(previous)
            item['error'] = 'Existing result did not verify; inspect or select a fresh output root'
            report['jobs'].append(item)
            write_report(report_path, report)
            print(f'[{i}/{len(jobs)}] {job["job_id"]} FAIL existing result', flush=True)
            if a.stop_on_failure:
                break
            continue
        cmd = command(script, job, dest)
        print(f'[{i}/{len(jobs)}] {job["job_id"]}: {job["contrast_id"]} {job["modality"]}', flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            (a.output_root / f'{job["job_id"]}.log').write_text(proc.stdout + '\n' + proc.stderr)
            item['exit_code'] = proc.returncode
            item['execution'] = 'executed'
            if proc.returncode:
                item['status'] = 'fail'
                item['error'] = f'AREA returned exit code {proc.returncode}'
            else:
                item.update(verify(job, dest, a.rtol, a.atol))
        except Exception as exc:
            item['status'] = 'fail'
            item['error'] = repr(exc)
        report['jobs'].append(item)
        write_report(report_path, report)
        print('   ', item['status'].upper(), flush=True)
        if item['status'] != 'pass' and a.stop_on_failure:
            break
    passed = sum(j['status'] == 'pass' for j in report['jobs'])
    print(f'Passed {passed}/{len(jobs)}; report {report_path}', flush=True)
    return 0 if passed == len(jobs) else 1


if __name__ == '__main__':
    sys.exit(main())

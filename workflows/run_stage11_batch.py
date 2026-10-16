#!/usr/bin/env python3
"""Re-run brain AREA models using original model script and verify frozen results.

No source or reference file is modified. Default mode is read-only preflight.
Requires pandas and numpy in the selected Python environment.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

REQUIRED = {'task_id', 'modality', 'manifest', 'expression', 'orientation',
            'group_a', 'group_b', 'A_continuous', 'A_categorical',
            'global_complete_features', 'A_result', 'A_run_manifest',
            'contrast_id'}
NUM_RTOL = 1e-10
NUM_ATOL = 1e-12


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, required=True)
    p.add_argument('--research-root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--task-ids', default='', help='Optional comma-separated selection from the 12 brain tasks')
    return p.parse_args(argv)


def is_true(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def localize(original, roots):
    original = str(original)
    if '/rosmap_multiomic_slurm/' in original:
        relative = original.split('/rosmap_multiomic_slurm/', 1)[1]
    else:
        raise ValueError('Unrecognized HPC input path: ' + original)
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError('Could not locate '+original+' under '+', '.join(map(str, roots)))


def contrast_name(task):
    # Recover exact original --contrast-name from historical run manifest, not ad hoc translation.
    return str(task['model_manifest']['contrast'])


def prepare_tasks(frame, roots, ids):
    if REQUIRED - set(frame.columns):
        raise ValueError('Missing task columns: '+str(sorted(REQUIRED-set(frame.columns))))
    chosen = frame.loc[frame.modality.isin(['dlpfc','tmt'])].copy()
    if ids:
        eligible = set(chosen.task_id.astype(int))
        if not ids <= eligible:
            raise ValueError('Requested task IDs not eligible for brain execution: '+str(ids-eligible))
        chosen = chosen.loc[chosen.task_id.astype(int).isin(ids)]
    jobs=[]
    for row in chosen.to_dict('records'):
        manifest_path=localize(row['A_run_manifest'], roots)
        metadata=json.loads(manifest_path.read_text())
        original_a = str(metadata.get('group_A', ''))
        original_b = str(metadata.get('group_B', ''))
        task_a = str(row['group_a'])
        task_b = str(row['group_b'])

        if not original_a or not original_b or original_a == original_b:
            raise ValueError('Invalid original model groups in task '+str(row['task_id']))

        same = (original_a, original_b) == (task_a, task_b)
        reversed_groups = (original_a, original_b) == (task_b, task_a)

        if not same and not reversed_groups:
            raise ValueError('Unrecognized group disagreement in task '+str(row['task_id']))

        if reversed_groups and not (
            str(row['family']) == 'CERAD'
            and int(row['task_id']) in (16, 17, 18)
        ):
            raise ValueError('Unexpected group reversal in task '+str(row['task_id']))

        row['group_orientation'] = (
            'reversed_in_stage11_task' if reversed_groups else 'matching'
        )
        if metadata.get('orientation') != str(row['orientation']):
            raise ValueError('Orientation disagreement in task '+str(row['task_id']))
        if list(metadata.get('continuous_covariates', [])) != split_covariates(row['A_continuous']):
            raise ValueError('Continuous covariates disagree in task '+str(row['task_id']))
        if list(metadata.get('categorical_covariates', [])) != split_covariates(row['A_categorical']):
            raise ValueError('Categorical covariates disagree in task '+str(row['task_id']))
        row.update(model_manifest=metadata,
                   expression_local=localize(row['expression'],roots),
                   sample_manifest_local=localize(row['manifest'],roots),
                   results_reference=localize(row['A_result'],roots),
                   manifest_reference=manifest_path)
        jobs.append(row)
    return jobs


def split_covariates(value):
    return [x.strip() for x in str(value).split(',') if x.strip() and x.strip().lower()!='nan']


def build_command(script, job, outdir):
    args=[sys.executable, str(script),
          '--expression',str(job['expression_local']),
          '--manifest',str(job['sample_manifest_local']),
          '--outdir',str(outdir),
          '--contrast-name',contrast_name(job),
          '--group-a',str(job['model_manifest']['group_A']), '--group-b',str(job['model_manifest']['group_B']),
          '--orientation',str(job['orientation']),
          '--continuous',','.join(split_covariates(job['A_continuous'])),
          '--categorical',','.join(split_covariates(job['A_categorical'])),
          '--fdr',str(job['model_manifest']['fdr'])]
    if is_true(job['global_complete_features']):
        args.append('--global-complete-features')
    if job['model_manifest'].get('block'):
        # Refuse to assume block CLI semantics for unscripted model variant.
        raise ValueError('Blocked model requires explicit CLI inspection')
    return args


def compare_manifest(expected, actual):
    return {key: {'reference': expected.get(key), 'generated':actual.get(key)}
            for key in sorted(set(expected)|set(actual)) if expected.get(key)!=actual.get(key)}


def compare_results(reference, generated):
    a=pd.read_csv(reference)
    b=pd.read_csv(generated)
    report={'reference_rows':len(a),'generated_rows':len(b), 'status':'fail'}
    if list(a.columns)!=list(b.columns):
        report['error']='column mismatch';return report
    if 'feature_id' not in a.columns:
        report['error']='missing feature_id';return report
    if a.feature_id.isna().any() or b.feature_id.isna().any() or not a.feature_id.is_unique or not b.feature_id.is_unique:
        report['error']='null/duplicate feature IDs';return report
    if set(a.feature_id)!=set(b.feature_id):
        report['error']='feature ID set mismatch';return report
    b=b.set_index('feature_id').loc[a.feature_id].reset_index()
    a=a.reset_index(drop=True)
    discrepancies={}
    metrics={}
    for column in a.columns:
        if column=='feature_id':continue
        x,y=a[column],b[column]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv=x.to_numpy(dtype=float);yv=y.to_numpy(dtype=float)
            close=np.isclose(xv,yv,rtol=NUM_RTOL,atol=NUM_ATOL,equal_nan=True)
            finite=np.isfinite(xv)&np.isfinite(yv)
            metrics[column]={
                'close':bool(close.all()),
                'mismatched':int((~close).sum()),
                'max_abs_diff': float(np.max(np.abs(xv[finite]-yv[finite]))) if finite.any() else None,
                'exact':bool(np.array_equal(xv,yv,equal_nan=True))}
            if not close.all(): discrepancies[column]=int((~close).sum())
        else:
            same=x.fillna('<MISSING>').astype(str).eq(y.fillna('<MISSING>').astype(str))
            metrics[column]={'identical':bool(same.all()),'mismatched':int((~same).sum())}
            if not same.all():discrepancies[column]=int((~same).sum())
    report['metrics']=metrics
    report['discrepancies']=discrepancies
    report['status']='pass' if not discrepancies else 'fail'
    return report


def main(argv=None):
    a=parse_args(argv)
    roots=[a.reference_root.expanduser().resolve(),a.research_root.expanduser().resolve()]
    release=a.release_root.expanduser().resolve()
    script=release/'upstream/portable_scripts/run_regular_area_binary.py'
    taskfile=roots[0]/'results/stage11_covariate_stress/stage11_tasks.tsv'
    if not script.is_file() or not taskfile.is_file():
        raise SystemExit('Missing AREA script or task manifest')
    requested={int(x) for x in a.task_ids.split(',') if x.strip()}
    frame=pd.read_csv(taskfile,sep='\t')
    jobs=prepare_tasks(frame, roots, requested)
    if not jobs:raise SystemExit('No tasks selected')
    outroot=a.output_root.expanduser().resolve()
    # Prevent output anywhere inside a reference or research repository or release source.
    for source in (release, *roots):
        if outroot==source or source in outroot.parents:
            raise SystemExit('Unsafe output directory inside an input/source tree: '+str(outroot))
    if outroot.exists():raise SystemExit('Refusing existing output directory: '+str(outroot))
    print('AREA Stage 11 brain model execution plan')
    print('Task count:',len(jobs),'| modes:', ', '.join(sorted(set(j['modality'] for j in jobs))))
    for job in jobs:
        print('Task',job['task_id'],job['modality'],job['contrast_id'],
              'samples',job['model_manifest']['n'],
              'features',job['model_manifest']['n_features'],
              'global-complete',is_true(job['global_complete_features']))
        build_command(script,job,outroot/f"task_{int(job['task_id']):02d}")
    if not a.execute:
        print('PREFLIGHT ONLY: no models run and no files created; add --execute to run.')
        return 0
    outroot.mkdir(parents=True,exist_ok=False)
    summary=[]
    for job in jobs:
        taskid=int(job['task_id']);outdir=outroot/f'task_{taskid:02d}'
        cmd=build_command(script,job,outdir)
        print('Running task',taskid,job['contrast_id'],job['modality'],flush=True)
        proc=subprocess.run(cmd,capture_output=True,text=True)
        (outroot/f'task_{taskid:02d}.log').write_text(proc.stdout+'\n'+proc.stderr)
        item={'task_id':taskid,'modality':job['modality'],'contrast_id':str(job['contrast_id']),
              'exit_code':proc.returncode,'reference':str(job['results_reference']),
              'output_dir':str(outdir)}
        if proc.returncode==0:
            generated=outdir/job['results_reference'].name
            if not generated.is_file() or not (outdir/'run_manifest.json').is_file():
                item['status']='fail';item['error']='expected output missing'
            else:
                item['comparison']=compare_results(job['results_reference'],generated)
                item['manifest_differences']=compare_manifest(job['model_manifest'],json.loads((outdir/'run_manifest.json').read_text()))
                item['status']='pass' if item['comparison']['status']=='pass' and not item['manifest_differences'] else 'fail'
        else:
            item['status']='fail';item['error']=proc.stderr[-3000:]
        summary.append(item)
        print('Task',taskid,item['status'],flush=True)
        # Fail fast: do not proceed after an unexpected error or numerical mismatch.
        if item['status']!='pass':break
    report={'method':'Original AREA Python runner, model manifests and expression inputs',
            'note':'Input provenance is recorded in the task manifest. This is not raw-data reproduction.',
            'numerical_tolerance':{'rtol':NUM_RTOL,'atol':NUM_ATOL},
            'tasks':summary}
    (outroot/'stage11_batch_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('Report:',outroot/'stage11_batch_report.json')
    print('Passed:',sum(x['status']=='pass' for x in summary),'/',len(jobs))
    return 0 if len(summary)==len(jobs) and all(x['status']=='pass' for x in summary) else 1


if __name__=='__main__':
    sys.exit(main())

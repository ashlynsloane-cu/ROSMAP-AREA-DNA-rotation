#!/usr/bin/env python3
"""Reproduce six fully-adjusted historical monocyte Stage11 AREA models.

Input data remain outside Git. Preflight runs without creating files.
"""
import argparse
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_stage11_batch import compare_manifest, compare_results, is_true, split_covariates


def resolve_hpc(value, roots):
    value = str(value)
    marker = '/rosmap_multiomic_slurm/'
    if marker in value:
        tail = value.split(marker, 1)[1]
        for root in roots:
            p = root / tail
            if p.is_file():
                return p.resolve()
    p = Path(value).expanduser()
    if p.is_file():
        return p.resolve()
    raise FileNotFoundError(f'Unresolved historical reference: {value}')


def tasks(args):
    root = args.reference_root / 'results/stage11_covariate_stress'
    table = pd.read_csv(root / 'stage11_tasks.tsv', sep='\t')
    chosen = table[table['modality'].eq('monocyte')]
    if len(chosen) != 6:
        raise ValueError(f'Expected six monocyte tasks, found {len(chosen)}')
    jobs = []
    for t in chosen.to_dict('records'):
        ref_manifest = resolve_hpc(t['A_run_manifest'], [args.reference_root, args.research_root])
        model = json.loads(ref_manifest.read_text())
        if model['orientation'] != t['orientation'] or model.get('block'):
            raise ValueError('Unsupported orientation or blocking: task '+str(t['task_id']))
        if model['continuous_covariates'] != split_covariates(t['A_continuous']):
            raise ValueError('Continuous covariates mismatch: task '+str(t['task_id']))
        if model['categorical_covariates'] != split_covariates(t['A_categorical']):
            raise ValueError('Categorical covariates mismatch: task '+str(t['task_id']))
        pairs = ((str(t['group_a']), str(t['group_b'])),
                 (model['group_A'], model['group_B']))
        if set(pairs[0]) != set(pairs[1]):
            raise ValueError('Group names mismatch: task '+str(t['task_id']))
        manifest = resolve_hpc(t['manifest'], [args.research_root, args.reference_root])
        reference = resolve_hpc(t['A_result'], [args.reference_root, args.research_root])
        jobs.append(dict(row=t, model=model, sample_manifest=manifest,
                         reference=reference, ref_manifest=ref_manifest))
    return jobs


def command(script, expression, job, outdir):
    t, m = job['row'], job['model']
    cmd = [sys.executable, str(script), '--expression', str(expression),
           '--manifest', str(job['sample_manifest']), '--outdir', str(outdir),
           '--contrast-name', str(m['contrast']), '--group-a', str(m['group_A']),
           '--group-b', str(m['group_B']), '--orientation', str(t['orientation']),
           '--continuous', ','.join(m['continuous_covariates']),
           '--categorical', ','.join(m['categorical_covariates']),
           '--fdr', str(m['fdr'])]
    if is_true(t['global_complete_features']):
        cmd += ['--global-complete-features']
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release-root', 'research-root', 'reference-root', 'output-root', 'monocyte-expression'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--round-trip-expression-parser', action='store_true',
                   help='Run isolated AREA source copy with round-trip float parsing; original source remains unchanged')
    p.add_argument('--max-jobs', type=int, default=0)
    p.add_argument('--task-ids', type=str, default='', help='Comma-separated historical task IDs, e.g. 7,10,13,16')
    args = p.parse_args(argv)
    for key in ('release_root', 'research_root', 'reference_root', 'output_root', 'monocyte_expression'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    out = args.output_root
    for protected in (args.release_root, args.research_root, args.reference_root, args.monocyte_expression.parent):
        if out == protected or protected in out.parents:
            p.error('Output directory is inside source or authorized data tree: '+str(out))
    if out.exists():
        p.error('Output directory already exists; select a new one')
    script = args.release_root / 'upstream/portable_scripts/run_regular_area_binary.py'
    if not script.is_file() or not args.monocyte_expression.is_file():
        p.error('Missing AREA script or authorized monocyte expression matrix')
    jobs = tasks(args)
    if args.task_ids:
        requested = [int(i.strip()) for i in args.task_ids.split(',') if i.strip()]
        if len(requested) != len(set(requested)):
            p.error('Duplicate task IDs')
        available = {int(j['row']['task_id']) for j in jobs}
        if not set(requested) <= available:
            p.error(f'Unknown monocyte task ID(s): {sorted(set(requested)-available)}')
        jobs = [j for j in jobs if int(j['row']['task_id']) in set(requested)]
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    for job in jobs:
        taskid = int(job['row']['task_id'])
        # The historical model used a 14,756-gene x 554-sample matrix; verify universe before fitting.
        if int(job['model']['n_features']) != 14756:
            p.error(f'Unexpected monocyte feature universe for task {taskid}')
        print(f'Task {taskid}: {job["model"]["contrast"]} n={job["model"]["n"]} features=14756', flush=True)
    if not args.execute:
        print('PREFLIGHT ONLY. Use --execute to fit.')
        return 0
    parser_metadata = {'mode': 'historical_default'}
    patched_source = None
    if args.round_trip_expression_parser:
        original = script.read_text()
        # Locate the actual reader via AST, without depending on whitespace,
        # escaped-tab spelling, or CSV formatting style.
        tree = __import__('ast').parse(original)
        readers = [node for node in tree.body
                   if isinstance(node, (__import__('ast').FunctionDef, __import__('ast').AsyncFunctionDef))
                   and node.name == 'read_any']
        if len(readers) != 1:
            raise RuntimeError('Expected exactly one read_any function; no model executed')
        calls = [node for node in __import__('ast').walk(readers[0])
                 if isinstance(node, __import__('ast').Call)
                 and isinstance(node.func, __import__('ast').Attribute)
                 and isinstance(node.func.value, __import__('ast').Name)
                 and node.func.value.id == 'pd' and node.func.attr == 'read_csv']
        if len(calls) != 1:
            raise RuntimeError('Expected one pandas read_csv in read_any; no model executed')
        call = calls[0]
        if any(kw.arg == 'float_precision' for kw in call.keywords):
            raise RuntimeError('AREA reader already specifies float_precision; no model executed')
        # The original call is single-line. Require this before patching.
        if call.lineno != call.end_lineno:
            raise RuntimeError('Unexpected multiline AREA reader; no model executed')
        lines = original.splitlines(keepends=True)
        line = lines[call.lineno - 1]
        closing_col = call.end_col_offset - 1
        if line[closing_col] != ')':
            raise RuntimeError('Unexpected AREA reader syntax; no model executed')
        lines[call.lineno - 1] = (line[:closing_col]
                                  + ',float_precision="round_trip"'
                                  + line[closing_col:])
        patched_source = ''.join(lines)
        __import__('ast').parse(patched_source)
        parser_metadata = {
            'mode': 'round_trip',
            'original_script_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'patched_script_sha256': hashlib.sha256(patched_source.encode()).hexdigest(),
            'change': 'add float_precision=round_trip to read_any pandas CSV reader',
        }
    out.mkdir(parents=True)
    if patched_source is not None:
        script = out / 'run_regular_area_binary_round_trip.py'
        script.write_text(patched_source)
    results = []
    for job in jobs:
        taskid = int(job['row']['task_id'])
        directory = out / f'task_{taskid:02d}'
        run = subprocess.run(command(script, args.monocyte_expression, job, directory), capture_output=True, text=True)
        (out / f'task_{taskid:02d}.log').write_text(run.stdout+'\n'+run.stderr)
        item = {'task_id': taskid, 'exit_code': run.returncode}
        if run.returncode:
            item['status']='fail'; item['error']=run.stderr[-1500:]
        else:
            actual = directory / job['reference'].name
            if not actual.is_file() or not (directory/'run_manifest.json').is_file():
                item['status']='fail'; item['error']='Missing generated output'
            else:
                item['comparison'] = compare_results(job['reference'], actual)
                item['manifest_differences'] = compare_manifest(job['model'], json.loads((directory/'run_manifest.json').read_text()))
                item['status'] = 'pass' if item['comparison']['status']=='pass' and not item['manifest_differences'] else 'fail'
        results.append(item)
        print(f'Task {taskid}: {item["status"]}', flush=True)
        if item['status'] != 'pass':
            break
    report = {'method':'Stage11 monocyte fully adjusted AREA independent model replay',
              'note':'Model-ready counts input. Not a raw FASTQ-to-count reconstruction.',
              'parser':parser_metadata, 'tasks':results}
    (out/'stage11_monocyte_area_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed',sum(r['status']=='pass' for r in results),'/',len(jobs))
    return 0 if len(results)==len(jobs) and all(r['status']=='pass' for r in results) else 1

if __name__ == '__main__':
    sys.exit(main())

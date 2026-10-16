#!/usr/bin/env python3
"""Fresh phenotype-only DESeq2 (RNA) / limma (TMT) on primary ROSMAP manifests.

No archived results or original research checkout required. Inputs are authorized
raw gene counts and processed TMT abundance. Default is a read-only preflight.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

CONTRASTS = (("cogdx", "NCI_vs_AD"), ("cogdx", "NCI_vs_MCI"),
             ("cogdx", "MCI_vs_AD"), ("Braak", "0-2_vs_3-4"),
             ("Braak", "3-4_vs_5-6"), ("CERAD", "1-2_vs_3-4"))


def canonical(s):
    s = str(s).replace("–", "-").replace("—", "-").strip().lower()
    s = re.sub(r"^(braak|cerad)[ _]+", "", s)
    return re.sub(r"\s+", "", s)


def select(index, modality, axis, contrast):
    subset = index[(index.modality == modality) &
                   (index.axis == axis) & (index.contrast == contrast)]
    subset = subset[subset.cohort == ("DLPFC discovery" if modality == "dlpfc" else "TMT discovery")]
    # The cognitive stage includes both PURE and BROAD; choose PURE exclusively.
    if axis == "cogdx":
        subset = subset[subset.manifest.astype(str).str.contains("PURE_1_NCI_2_MCI_4_AD_exclude_3_5_6", regex=False)]
    if len(subset) != 1:
        raise ValueError(f"Expected one primary {modality} {axis} {contrast}, found {len(subset)}")
    row = subset.iloc[0]
    path = Path(row.manifest)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path)
    if not {'sample_id', 'analysis_group'}.issubset(manifest.columns):
        raise ValueError(f"Missing sample/group column in {path}")
    if len(manifest) != int(row.n) or manifest.sample_id.duplicated().any():
        raise ValueError(f"Cohort size or unique IDs mismatch: {path}")
    groups = list(manifest.analysis_group.dropna().astype(str).unique())
    expected = [canonical(x) for x in contrast.split('_vs_')]
    mapping = {canonical(x): x for x in groups}
    if len(mapping) != 2 or set(mapping) != set(expected):
        raise ValueError(f"Unexpected groups {groups} for {axis} {contrast}")
    # Frozen Stage11 phenotype-only convention: CERAD 3-4 reference to 1-2.
    if axis == 'CERAD':
        expected = expected[::-1]
    return row, (mapping[expected[0]], mapping[expected[1]])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests-root', required=True, type=Path)
    p.add_argument('--rna-counts', required=True, type=Path,
                   help='Unnormalized integer counts RDS/CSV/TSV (NOT normalized AREA expression)')
    p.add_argument('--tmt-expression', required=True, type=Path)
    p.add_argument('--outdir', required=True, type=Path)
    p.add_argument('--rscript', default='Rscript')
    p.add_argument('--rna-column-map', type=Path,
                   help='Provide if raw RNA count columns do not match manifest sample IDs')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    index_path = args.manifests_root / 'manifest_index.csv'
    sources = [index_path, args.rna_counts, args.tmt_expression]
    if args.rna_column_map:
        sources.append(args.rna_column_map)
    scripts = {'dlpfc': repo / 'scripts/multiomic/run_rna_deseq2_binary.R',
               'tmt': repo / 'scripts/multiomic/run_tmt_limma_binary.R'}
    for path in [*sources, *scripts.values()]:
        if not path.is_file():
            p.error(f'Required input missing: {path}')
    if args.rna_counts.resolve() == (repo / 'data').resolve():
        p.error('Raw counts must be a file, not directory')
    index = pd.read_csv(index_path)
    jobs = []
    for modality in ('dlpfc', 'tmt'):
        for axis, contrast in CONTRASTS:
            row, groups = select(index, modality, axis, contrast)
            manifest = Path(row.manifest)
            target = args.outdir / modality / axis / contrast
            label = f'{modality}_{axis}_{contrast}'
            cmd = [args.rscript, str(scripts[modality]),
                   '--manifest', str(manifest), '--outdir', str(target),
                   '--contrast-name', label, '--group-a', groups[0],
                   '--group-b', groups[1], '--continuous', '', '--categorical', '']
            if modality == 'dlpfc':
                cmd += ['--counts', str(args.rna_counts)]
                if args.rna_column_map:
                    cmd += ['--column-map', str(args.rna_column_map)]
            else:
                cmd += ['--matrix', str(args.tmt_expression), '--expected-features', '5211']
            jobs.append({'id': label, 'modality': modality, 'manifest': str(manifest),
                         'sample_count': int(row.n), 'groups': groups,
                         'output': str(target), 'command': cmd})
    print(f'FRESH_CONVENTIONAL_PREFLIGHT: {len(jobs)} jobs')
    for job in jobs:
        print(job['id'], f"n={job['sample_count']}", f"A={job['groups'][0]}", f"B={job['groups'][1]}")
    if not args.execute:
        print('PREFLIGHT_ONLY: add --execute to fit all 12')
        return 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {'input_hashes': {str(path): digest(path) for path in sources},
              'script_hashes': {k: digest(v) for k, v in scripts.items()}, 'jobs': []}
    report_path = args.outdir / 'fresh_conventional_models.json'
    for job in jobs:
        target = Path(job['output'])
        target.mkdir(parents=True, exist_ok=True)
        print('FITTING', job['id'], flush=True)
        with (target / 'execution.log').open('w') as log:
            result = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT,
                                    check=False)
        job_record = {k: v for k, v in job.items() if k != 'command'}
        job_record['returncode'] = result.returncode
        job_record['output_hashes'] = {p.name: digest(p) for p in target.glob('*.csv')}
        report['jobs'].append(job_record)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f"{job['id']} exited {result.returncode}; see {target / 'execution.log'}")
        if not (target / 'run_summary.csv').is_file() or len(job_record['output_hashes']) < 2:
            raise RuntimeError(f"Missing expected results for {job['id']}; see {target / 'execution.log'}")
    print(f'FRESH_CONVENTIONAL_COMPLETE: {len(jobs)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

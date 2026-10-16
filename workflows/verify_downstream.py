#!/usr/bin/env python3
"""Read-only verification of independently regenerated ROSMAP Stage 12 outputs.

Compares all top-level TSVs by SHA-256 and, if hashes differ, compares
literal TSV rows as multisets (preserving duplicate multiplicities).
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

STAGES = {
    '12A': 'stage12A_rna_protein_conjunction',
    '12B': 'stage12B_deseq2_limma_conjunction',
    '12C': 'stage12C_monocyte_followup',
    '12D': 'stage12D_biological_candidate_sets',
    '12E': 'stage12E_biological_evidence_matrix',
    '12I': 'stage12I_full_rank_gsea',
    '12J': 'stage12J_rrho',
    '12K': 'stage12K_convergence_architecture',
}
BASE = Path('results/stage12_brain_convergence')
EXCLUDED = {('12I', 'GENE_SET_LIBRARY_AUDIT.tsv')}


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def table_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle, delimiter='\t', strict=True)
        header = next(reader, None)
        if header is None:
            raise ValueError('Empty TSV')
        rows = []
        for lineno, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f'Line {lineno}: {len(row)} cells, expected {len(header)}')
            rows.append(tuple(row))
    return header, rows


def compare(reference, generated):
    record = {
        'reference_sha256': checksum(reference),
        'generated_sha256': checksum(generated),
    }
    if record['reference_sha256'] == record['generated_sha256']:
        record['status'] = 'exact'
        return record
    try:
        ref_cols, ref_rows = table_rows(reference)
        gen_cols, gen_rows = table_rows(generated)
        record['reference_rows'] = len(ref_rows)
        record['generated_rows'] = len(gen_rows)
        if ref_cols != gen_cols:
            record['status'] = 'different'
            record['reason'] = 'columns differ'
        elif Counter(ref_rows) == Counter(gen_rows):
            record['status'] = 'content_equivalent'
            if ref_rows == gen_rows:
                record['reason'] = 'same ordered rows; different byte serialization'
            else:
                record['reason'] = 'same TSV row multiset, different row ordering'
        else:
            record['status'] = 'different'
            record['reason'] = 'row contents/multiplicities differ'
            old_only = Counter(ref_rows) - Counter(gen_rows)
            new_only = Counter(gen_rows) - Counter(ref_rows)
            record['reference_only_row_instances'] = sum(old_only.values())
            record['generated_only_row_instances'] = sum(new_only.values())
    except (UnicodeError, csv.Error, ValueError) as exc:
        record['status'] = 'different'
        record['reason'] = f'Unable to compare TSV: {exc}'
    return record


def verify(reference_root, generated_root):
    records = []
    for stage, dirname in STAGES.items():
        ref_dir = reference_root / BASE / dirname
        gen_dir = generated_root / BASE / dirname
        refs = sorted(ref_dir.glob('*.tsv'))
        if not refs:
            records.append({'stage': stage, 'status': 'missing_reference_stage',
                            'path': str(ref_dir)})
            continue
        for ref in refs:
            entry = {'stage': stage, 'filename': ref.name,
                     'reference_path': str(ref),
                     'generated_path': str(gen_dir / ref.name)}
            candidate = gen_dir / ref.name
            if (stage, ref.name) in EXCLUDED:
                entry['status'] = 'reference_derived_excluded'
            elif not candidate.is_file():
                entry['status'] = 'missing_generated'
            else:
                entry.update(compare(ref, candidate))
            records.append(entry)
        for generated in sorted(gen_dir.glob('*.tsv')):
            if not (ref_dir / generated.name).is_file():
                records.append({'stage': stage, 'filename': generated.name,
                                'status': 'extra_generated', 'generated_path': str(generated)})
    totals = Counter(rec['status'] for rec in records)
    return {
        'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'reference_root': str(reference_root),
        'generated_root': str(generated_root),
        'python_version': platform.python_version(),
        'status_counts': dict(sorted(totals.items())),
        'independently_verified_tsvs': totals['exact'] + totals['content_equivalent'],
        'records': records,
        'scope_note': 'Top-level TSV outputs only; not execution provenance, plots, gzip, TXT or source-data validation.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--generated-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='Write JSON report outside source/reference trees')
    args = parser.parse_args()
    ref = args.reference_root.expanduser().resolve(strict=True)
    gen = args.generated_root.expanduser().resolve(strict=True)
    if ref == gen:
        parser.error('Reference and generated roots must differ.')
    report = verify(ref, gen)
    for stage in STAGES:
        subset = [r for r in report['records'] if r['stage'] == stage]
        counts = Counter(r['status'] for r in subset)
        print(f'{stage}: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))
    print('TOTAL:', report['status_counts'])
    print('Independently verified TSVs:', report['independently_verified_tsvs'])
    if args.report:
        destination = args.report.expanduser().resolve()
        if destination == ref or destination == gen or ref in destination.parents or gen in destination.parents:
            parser.error('Report must be outside reference and generated roots.')
        if destination.exists():
            parser.error('Report path already exists; refusing to overwrite.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Report:', destination)
    errors = {'different', 'missing_generated', 'missing_reference_stage', 'extra_generated'}
    return 1 if any(report['status_counts'].get(status, 0) for status in errors) else 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Stream-validate authorized ROSMAP monocyte model matrices without exposing donors."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_matrices(raw, normalized, expected_samples=554):
    stats = {'samples': None, 'features': 0, 'raw_noninteger_values': 0,
             'raw_negative_values': 0, 'normalized_negative_values': 0,
             'nonfinite_values': 0, 'duplicate_feature_ids': 0,
             'feature_id_order_identical': True,
             'sample_column_order_identical': True}
    seen = set()
    with raw.open(newline='') as fa, normalized.open(newline='') as fb:
        ra, rb = csv.reader(fa, delimiter='\t'), csv.reader(fb, delimiter='\t')
        ha, hb = next(ra), next(rb)
        if len(ha) < 2 or ha[0] != 'Geneid' or hb[0] != 'Geneid':
            raise ValueError('Both matrices must have Geneid as first column')
        if len(ha) != len(set(ha)) or len(hb) != len(set(hb)):
            raise ValueError('Duplicate sample-column headers')
        stats['samples'] = len(ha)-1
        if len(ha) != len(hb) or stats['samples'] != expected_samples:
            raise ValueError('Sample count mismatch or unexpected sample count')
        if ha != hb:
            stats['sample_column_order_identical'] = False
            raise ValueError('Matrices do not have identical sample headers and order')
        sentinel = object()
        from itertools import zip_longest
        for a, b in zip_longest(ra, rb, fillvalue=sentinel):
            if a is sentinel or b is sentinel:
                raise ValueError('Matrix feature row-count mismatch')
            if len(a) != len(ha) or len(b) != len(hb):
                raise ValueError(f'Malformed matrix row {stats["features"]+2}')
            if a[0] != b[0]:
                stats['feature_id_order_identical'] = False
                raise ValueError(f'Feature order mismatch at row {stats["features"]+2}')
            if not a[0]:
                raise ValueError('Empty feature ID')
            if a[0] in seen:
                stats['duplicate_feature_ids'] += 1
            seen.add(a[0])
            stats['features'] += 1
            for aval, bval in zip(a[1:], b[1:]):
                try:
                    av, bv = float(aval), float(bval)
                except ValueError:
                    raise ValueError(f'Non-numeric matrix value at row {stats["features"]+1}') from None
                if not math.isfinite(av) or not math.isfinite(bv):
                    stats['nonfinite_values'] += 1
                    continue
                if av < 0:
                    stats['raw_negative_values'] += 1
                if bv < 0:
                    stats['normalized_negative_values'] += 1
                if not av.is_integer():
                    stats['raw_noninteger_values'] += 1
    if stats['features'] == 0 or stats['duplicate_feature_ids']:
        raise ValueError('Empty feature universe or duplicate feature IDs')
    if any(stats[k] for k in ('raw_noninteger_values','raw_negative_values', 'normalized_negative_values','nonfinite_values')):
        raise ValueError('Invalid count/normalized values: ' + json.dumps(stats))
    return stats, ha[1:]


def check_metadata(path, sample_names):
    with path.open(newline='') as fh:
        rd = csv.DictReader(fh, delimiter='\t')
        headers = rd.fieldnames or []
        sample_col = next((c for c in ('specimen','sample_id','specimenID','monocyte_specimen_id') if c in headers), None)
        participant_col = next((c for c in ('individualID','individual_id') if c in headers), None)
        if not sample_col or not participant_col:
            raise ValueError('Missing metadata sample/participant identifier columns')
        n = 0
        specimens = set()
        for row in rd:
            n += 1
            v = row[sample_col].strip()
            if not v or v in specimens:
                raise ValueError('Missing/duplicate metadata sample identifier')
            specimens.add(v)
    observed = set(sample_names)
    exact = len(observed & specimens)
    return {'rows': n, 'sample_identifier_column': sample_col,
            'participant_identifier_column': participant_col,
            'exact_matrix_metadata_sample_matches': exact,
            'full_exact_match': observed == specimens,
            'sample_mapping_status': 'PASS' if observed == specimens else 'NEEDS_MAPPING_VERIFICATION'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-counts', type=Path, required=True)
    ap.add_argument('--normalized-counts', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--expected-samples', type=int, default=554)
    args = ap.parse_args()
    for p in (args.raw_counts, args.normalized_counts, args.metadata):
        if not p.is_file():
            ap.error('Missing required input: ' + str(p))
    if args.output.resolve() in (p.resolve() for p in (args.raw_counts, args.normalized_counts,args.metadata)):
        ap.error('Cannot overwrite input')
    report = {'schema_version': 1, 'validated_input_level': 'processed_model_matrices_NOT_raw_sequencing',
              'raw_to_figure_complete': False, 'checks': {}, 'files': {}}
    for key,p in [('raw_counts',args.raw_counts),('normalized_counts',args.normalized_counts),('metadata',args.metadata)]:
        report['files'][key] = {'sha256': sha256(p), 'bytes':p.stat().st_size}
    stats, names = check_matrices(args.raw_counts, args.normalized_counts, args.expected_samples)
    report['checks']['matrix'] = stats
    report['checks']['metadata'] = check_metadata(args.metadata, names)
    report['status'] = 'PASS' if report['checks']['metadata']['full_exact_match'] else 'MATRIX_PASS_METADATA_MAPPING_UNVERIFIED'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Monocyte matrix integrity: PASS',stats['features'],'features x',stats['samples'],'samples')
    print('Metadata sample mapping:',report['checks']['metadata']['sample_mapping_status'])
    print('Report:',args.output)
    if report['status'] != 'PASS':
        print('Matrix accepted; metadata sample mapping must be established before model execution.')
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

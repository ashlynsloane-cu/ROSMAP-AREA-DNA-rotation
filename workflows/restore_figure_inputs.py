#!/usr/bin/env python3
"""Restore missing canonical historical figure inputs from an authorized research tree.

Only SHA-256 verified inputs under Git-ignored results/ or data/frozen_inputs/ can be linked.
Nothing is deleted, overwritten, or copied into Git by this helper.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'results':
        raise ValueError('Only repository-relative results/ paths are eligible: ' + value)
    return p


def canonical_records(repo):
    with (repo / 'config/canonical_figure_inputs.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('workspace') == 'repo':
                yield row['source_id'], safe_relative(row['relative_path']), row['sha256']


def additional_records(repo):
    path = repo / 'finalization_audit/repository_freeze/additional_figure_inputs.json'
    if path.is_file():
        data = json.loads(path.read_text())
        for name, spec in data.items():
            rel = Path(spec['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Unsafe figure input path: {rel}')
            if rel.parts and (rel.parts[0] == 'results' or
                              rel.parts[:2] == ('data', 'frozen_inputs')):
                yield name, rel, spec['sha256']


def ignored(repo, rel):
    result = subprocess.run(['git', 'check-ignore', '-q', '--', str(rel)],
                            cwd=repo, capture_output=True, check=False)
    return result.returncode == 0

def run(repo, research, frozen=None, execute=False):
    records = list(canonical_records(repo)) + list(additional_records(repo))
    missing = []
    unsafe = []
    linked = 0
    for label, rel, expected in records:
        target = repo / rel
        if not ignored(repo, rel):
            print('BLOCKED_NOT_GIT_IGNORED', label, rel)
            unsafe.append(label)
            continue
        if target.is_file():
            if digest(target) == expected:
                print('OK', label, rel)
            else:
                print('HASH_MISMATCH_EXISTING', label, rel)
                unsafe.append(label)
            continue
        if target.exists() or target.is_symlink():
            print('BLOCKED_EXISTING_PATH', label, rel)
            unsafe.append(label)
            continue
        roots = [research] + ([frozen] if frozen else [])
        candidates = []
        for root in roots:
            direct = root / rel
            if direct.is_file():
                candidates.append(direct)
            if rel.parts[:2] == ('data', 'frozen_inputs'):
                stripped = Path(*rel.parts[2:])
                if (root / stripped).is_file():
                    candidates.append(root / stripped)
            # Search narrowly within expected file trees when direct paths differ.
            candidates.extend(p for p in root.rglob(rel.name) if p.is_file())
        matches = [p for p in dict.fromkeys(candidates) if digest(p) == expected]
        if not matches:
            print('NOT_FOUND_HASH_MATCH', label, rel)
            missing.append(label)
            continue
        source = matches[0].resolve()
        if not any(source.is_relative_to(root.resolve()) for root in roots):
            print('BLOCKED_OUTSIDE_RESEARCH', label)
            unsafe.append(label)
            continue
        print('LINK' if execute else 'WOULD_LINK', label, rel, '<-', source)
        if execute:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            linked += 1
    print(f'SUMMARY records={len(records)} linked={linked} unresolved={len(missing)} unsafe={len(unsafe)}')
    return int(bool(missing or unsafe))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--research-root', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, default=None)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    repo, research = a.repo_root.resolve(), a.research_root.resolve()
    if not (repo / '.git').exists() or not research.is_dir() or repo == research:
        p.error('Expected distinct existing release Git and authorized research directories')
    frozen = a.frozen_root.expanduser().resolve() if a.frozen_root else None
    if frozen is not None and not frozen.is_dir():
        p.error('Frozen root does not exist')
    return run(repo, research, frozen, a.execute)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(2)

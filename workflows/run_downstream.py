#!/usr/bin/env python3
"""Reproduce ROSMAP Stage 12 analyses from provisioned model results.

This workflow is intentionally downstream-only. It preserves the unmodified
portable sources, uses historical restricted GMT files for 12I, and refuses
existing output roots. It does not provide OS-level filesystem confinement.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

STAGES = [
    ('12A', 'stage12A_rna_protein_conjunction.py', 'stage12A_rna_protein_conjunction'),
    ('12B', 'stage12B_deseq2_limma_conjunction.py', 'stage12B_deseq2_limma_conjunction'),
    ('12C', 'stage12C_monocyte_followup.py', 'stage12C_monocyte_followup'),
    ('12D', 'stage12D_assemble_biological_candidate_sets.py', 'stage12D_biological_candidate_sets'),
    ('12E', 'stage12E_build_biological_evidence_matrix.py', 'stage12E_biological_evidence_matrix'),
    ('12I', 'stage12I_full_rank_gsea_leading_edge.py', 'stage12I_full_rank_gsea'),
    ('12J', 'stage12J_rrho_crossomic_concordance.py', 'stage12J_rrho'),
    ('12K', 'stage12K_formalize_convergence_architecture.py', 'stage12K_convergence_architecture'),
]
BASE = Path('results/stage12_brain_convergence')
GMT_NAMES = (
    'REAC__Reactome_Pathways_2024__restricted_4903.gmt',
    'GO_BP__GO_Biological_Process_2026__restricted_4903.gmt',
    'HALLMARK__MSigDB_Hallmark_2020__restricted_4903.gmt',
)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def runtime_with_frozen_gmts(source, frozen):
    """Replace only load_gene_sets to avoid mutable remote Enrichr libraries."""
    code = source.read_text(encoding='utf-8')
    tree = ast.parse(code)
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_gene_sets']
    if len(definitions) != 1:
        raise RuntimeError('Expected exactly one load_gene_sets function; source changed.')
    replacement = textwrap.dedent('''\
    def load_gene_sets(universe: set[str], args):
        from pathlib import Path
        import os
        import shutil
        import pandas as pd

        original = Path(os.environ['ROSMAP_FROZEN_GSEA_REFERENCE'])
        expected = (
            'REAC__Reactome_Pathways_2024__restricted_4903.gmt',
            'GO_BP__GO_Biological_Process_2026__restricted_4903.gmt',
            'HALLMARK__MSigDB_Hallmark_2020__restricted_4903.gmt',
        )
        combined = {}
        for filename in expected:
            source = original / 'gene_sets' / filename
            target = GENESET_DIR / filename
            shutil.copyfile(source, target)
            short = filename.split('__', 1)[0]
            with source.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    cells = line.rstrip('\\r\\n').split('\\t')
                    if len(cells) < 3:
                        raise ValueError(f'Invalid GMT line {filename}:{line_number}')
                    term, genes = cells[0], cells[2:]
                    if not term.startswith(short + '||'):
                        raise ValueError(f'Unexpected prefix: {term}')
                    if term in combined:
                        raise ValueError(f'Duplicate term: {term}')
                    if len(genes) != len(set(genes)) or not set(genes).issubset(universe):
                        raise ValueError(f'Gene universe or duplicate mismatch: {term}')
                    if not args.min_size <= len(genes) <= args.max_size:
                        raise ValueError(f'Out-of-range gene set: {term}')
                    combined[term] = genes
        if not combined:
            raise RuntimeError('No frozen gene sets loaded.')
        # Raw Enrichr library counts cannot be re-derived from restricted GMTs.
        # The historical audit is explicitly reference-derived metadata.
        audit = pd.read_csv(original / 'GENE_SET_LIBRARY_AUDIT.tsv', sep='\\t')
        if len(audit) != 3:
            raise ValueError('Unexpected historical gene-set audit length.')
        if 'gmt_path' in audit:
            audit['gmt_path'] = audit['gmt_path'].map(
                lambda value: str(GENESET_DIR / Path(str(value)).name)
            )
        return combined, audit
    ''')
    lines = code.splitlines(keepends=True)
    node = definitions[0]
    lines[node.lineno - 1:node.end_lineno] = [replacement]
    modified = ''.join(lines)
    ast.parse(modified)
    return modified


def check_layout(args):
    paths = (args.release_root, args.research_root, args.reference_root)
    for root in paths:
        if not root.is_dir():
            raise ValueError(f'Expected directory missing: {root}')
    if not (args.release_root / 'upstream/portable_scripts').is_dir():
        raise ValueError('portable_scripts not found in release root')
    tasks = args.reference_root / 'results/stage11_covariate_stress/stage11_tasks.tsv'
    if not tasks.is_file():
        raise ValueError(f'Stage 11 tasks missing: {tasks}')
    ref_gsea = args.reference_root / BASE / 'stage12I_full_rank_gsea'
    if not (ref_gsea / 'GENE_SET_LIBRARY_AUDIT.tsv').is_file():
        raise ValueError('Historical GSEA library audit missing')
    for filename in GMT_NAMES:
        if not (ref_gsea / 'gene_sets' / filename).is_file():
            raise ValueError(f'Frozen GMT missing: {filename}')
    for stage, script, _ in STAGES:
        if not (args.release_root / 'upstream/portable_scripts' / script).is_file():
            raise ValueError(f'Missing portable {stage} source: {script}')
    return tasks, ref_gsea


def versions():
    entries = {'python': sys.version.split()[0]}
    for package in ('gseapy', 'numpy', 'pandas', 'scipy'):
        try:
            entries[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            entries[package] = None
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--analysis-root', type=Path,
                        help='Optional, unused by Stage 12; defaults to research root.')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='MUST NOT exist. Created only with --execute.')
    parser.add_argument('--execute', action='store_true',
                        help='Without this flag, preflight only; no writes.')
    parser.add_argument('--monocyte-expression', type=Path,
                        help='Optional for Stage 11 only; unused by downstream runner.')
    args = parser.parse_args(argv)
    for name in ('release_root', 'research_root', 'reference_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.analysis_root = (args.analysis_root.expanduser().resolve() if args.analysis_root
                          else args.research_root)
    args.output_root = args.output_root.expanduser().resolve()
    tasks, ref_gsea = check_layout(args)
    print('ROSMAP downstream-only execution plan')
    print('Stages:', ', '.join(stage for stage, _, _ in STAGES))
    print('Output:', args.output_root)
    print('Python environment:', versions())
    print('Historical gene sets:', *[(p, digest(ref_gsea / 'gene_sets' / p)) for p in GMT_NAMES], sep='\n  ')
    if args.output_root.exists():
        parser.error('Output root exists: refusing overwrite or reuse.')
    for protected in (args.release_root, args.research_root, args.reference_root,
                      args.analysis_root):
        if args.output_root == protected or protected in args.output_root.parents or args.output_root in protected.parents:
            parser.error('Output root must be outside all source/reference roots.')
    if not args.execute:
        print('PREFLIGHT ONLY: no files created; add --execute to run.')
        return 0
    if versions().get('gseapy') != '1.3.0':
        parser.error('GSEApy 1.3.0 required for recorded reproduction configuration.')
    args.output_root.mkdir(parents=True, exist_ok=False)
    root = args.output_root
    dest_tasks = root / 'results/stage11_covariate_stress/stage11_tasks.tsv'
    dest_tasks.parent.mkdir(parents=True, exist_ok=True)
    dest_tasks.symlink_to(tasks)
    manifest = {
        'schema_version': 1,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'downstream_from_saved_models',
        'reference_derived': ['stage12I/GENE_SET_LIBRARY_AUDIT.tsv'],
        'environment': versions(),
        'paths': {key: str(getattr(args, key)) for key in
                  ('release_root', 'research_root', 'reference_root', 'analysis_root', 'output_root')},
        'frozen_gmt_sha256': {name: digest(ref_gsea / 'gene_sets' / name) for name in GMT_NAMES},
        'stages': [],
    }
    env = os.environ.copy()
    env.update({
        'ROSMAP_RESEARCH_ROOT': str(args.research_root),
        'ROSMAP_OFFLINE_ROOT': str(root),
        'ROSMAP_ANALYSIS_ROOT': str(args.analysis_root),
        'ROSMAP_FROZEN_GSEA_REFERENCE': str(ref_gsea),
        'PYTHONUNBUFFERED': '1',
    })
    if args.monocyte_expression:
        env['ROSMAP_MONOCYTE_EXPRESSION'] = str(args.monocyte_expression.expanduser().resolve())
    try:
        for stage, filename, dirname in STAGES:
            script = args.release_root / 'upstream/portable_scripts' / filename
            run_script = script
            if stage == '12I':
                run_script = root / 'stage12I_frozen_runtime.py'
                run_script.write_text(runtime_with_frozen_gmts(script, ref_gsea), encoding='utf-8')
            log = root / f'{stage}_execution.log'
            command = [sys.executable, str(run_script)]
            if stage == '12I':
                command.extend(['--permutations', '2000', '--seed', '42',
                                '--threads', '4', '--min-size', '10', '--max-size', '500'])
            print(f'Running {stage}: {filename}', flush=True)
            with log.open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=root, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, check=False)
            record = {'stage': stage, 'script': str(script), 'script_sha256': digest(script),
                      'exit_code': result.returncode, 'log': str(log),
                      'output_dir': str(root / BASE / dirname)}
            manifest['stages'].append(record)
            if result.returncode:
                raise RuntimeError(f'{stage} failed; see {log}')
        # import sibling verifier without depending on PYTHONPATH
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from verify_downstream import verify
        report = verify(args.reference_root, root)
        report_path = root / 'downstream_validation.json'
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        manifest['validation_counts'] = report['status_counts']
        manifest['independently_verified_tsvs'] = report['independently_verified_tsvs']
        print('Verification:', report['status_counts'])
        failures = {'different', 'missing_generated', 'missing_reference_stage', 'extra_generated'}
        if any(report['status_counts'].get(key, 0) for key in failures):
            raise RuntimeError(f'Validation discrepancies detected; inspect {report_path}')
        return 0
    except Exception as exc:
        manifest['error'] = str(exc)
        print(f'FAILED: {exc}', file=sys.stderr)
        return 1
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (root / 'execution_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())

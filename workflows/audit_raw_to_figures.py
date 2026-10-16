#!/usr/bin/env python3
"""Read-only dependency inventory for ROSMAP raw-to-eight-figure handoff.

Never reads sample-level content. Writes only into a new output directory.
This is a GATE/AUDIT, not a raw preprocessing implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime, timezone

TARGETS = {
    'RNA raw counts': ['ROSMAP_raw_counts_global_gene_universe.rds', 'ROSMAP_raw_counts_all_expressed_features.rds'],
    'RNA normalized AREA': ['ROSMAP_AREA_normalized_counts_all_samples.csv'],
    'TMT corrected input': ['C2.median_polish_corrected_log2(abundanceRatioCenteredOnMedianOfBatchMediansPerProtein)-8817x400.csv'],
    'Monocyte model matrix': ['monocyte_sizefactor_normalized_counts_area_554.tsv'],
    'Stage11 task manifest': ['stage11_tasks.tsv'],
    'Stage11 subset manifest': ['covariate_subset_tasks.tsv'],
    'Stage11 Shapley results': ['EXACT_SHAPLEY_BY_DATASET.tsv'],
    'Stage12 biology matrix': ['MASTER_BRAIN_BIOLOGICAL_EVIDENCE.tsv'],
}
CODE_SUFFIXES={'.py','.R','.r','.sh','.yaml','.yml','.toml','.json'}
PRODUCER_TERMS = re.compile(r'preprocess|normaliz|median.polish|batch.correct|monocyte|manifest|stage10|stage9|stage7|figure|phospho',re.I)

def files(root, allowed_suffixes=None):
    for p in root.rglob('*'):
        if any(x in p.parts for x in ('.git','.venv','__pycache__','.mypy_cache','node_modules')):
            continue
        if p.is_file() and (allowed_suffixes is None or p.suffix in allowed_suffixes):
            yield p

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--release-root',type=Path,required=True)
    a.add_argument('--research-root',type=Path,required=True)
    a.add_argument('--reference-root',type=Path,required=True)
    a.add_argument('--raw-root',type=Path,help='optional authorized ROSMAP input directory')
    a.add_argument('--output-root',type=Path,required=True)
    args=a.parse_args()
    roots={key:getattr(args,key).expanduser().resolve() for key in ('release_root','research_root','reference_root')}
    if args.raw_root:roots['raw_root']=args.raw_root.expanduser().resolve()
    output=args.output_root.expanduser().resolve()
    if output.exists():a.error('Output directory already exists; choose a new destination')
    for key,root in roots.items():
        if not root.is_dir():a.error(f'{key} missing: {root}')
        if output == root or root in output.parents:a.error('Output cannot be within source directory: '+str(root))
    entries={}
    for key,root in roots.items():
        print('Indexing filenames:',key,flush=True)
        entries[key]={}
        for p in files(root):
            for category,names in TARGETS.items():
                if p.name in names:
                    entries[key].setdefault(category,[]).append(str(p.relative_to(root)))
    candidate_scripts=[]
    for source_key in ('research_root','release_root'):
        root=roots[source_key]
        for p in files(root,CODE_SUFFIXES):
            # Filename-based candidate inventory; not an inferred execution DAG.
            rel=str(p.relative_to(root))
            if PRODUCER_TERMS.search(p.name) and (p.suffix in {'.py','.R','.r','.sh'}):
                candidate_scripts.append({'source':source_key,'path':rel})
    figure_manifest=[]
    release=roots['release_root']
    manifest_candidates=[release/'finalization_audit/repository_freeze/canonical_figure_manifest.csv',release/'config/canonical_figure_manifest.csv']
    for path in manifest_candidates:
        if path.is_file():
            with path.open(newline='') as fh:figure_manifest=list(csv.DictReader(fh))
            break
    fig_status=[]
    for row in figure_manifest:
        script=release/row.get('script','')
        fig_status.append({'figure':row.get('figure'), 'script':row.get('script'), 'script_exists':script.is_file(), 'reproduction_test':row.get('reproduction_test','unknown')})
    known_runners={name:(release/'workflows'/name).is_file() for name in (
      'run_stage11_batch.py','run_stage11_sensitivity_batch.py',
      'run_stage11_conventional_batch.py','run_downstream.py','run_main_figures.py')}
    gates={
       'original_raw_inputs_mapped_to_preprocessing':'NOT_VERIFIED',
       'rna_raw_to_normalized_reproduced':'NOT_VERIFIED',
       'tmt_source_to_5211_model_matrix_reproduced':'NOT_VERIFIED',
       'monocyte_raw_to_model_matrix_reproduced':'NOT_VERIFIED',
       'metadata_to_manifest_reproduced':'NOT_VERIFIED',
       'stage7_9_10_inputs_reproduced':'NOT_VERIFIED',
       'all_eight_figures_from_regenerated_inputs':'NOT_VERIFIED',
       'independent_clean_clone_raw_to_figures':'NOT_VERIFIED'}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Non-sensitive filename-only inventory; not proof of raw-to-figure reproducibility',
        'roots':{k:str(v) for k,v in roots.items()},'target_artifact_locations':entries,
        'candidate_scripts_filename_only':sorted(candidate_scripts,key=lambda x:(x['source'],x['path'])),
        'figure_manifest':fig_status,'runners':known_runners,'acceptance_gates':gates}
    output.mkdir(parents=True)
    (output/'integration_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# ROSMAP raw-to-figure integration gate','',
       'This inventory is based on filenames; it is NOT an executable preprocessing DAG or a validation of raw input lineage.','',
       '## Located artifacts (local only; do not commit donor data or absolute paths)']
    for cat in TARGETS:
        lines.append(f'### {cat}')
        any_found=False
        for key,found in entries.items():
            for path in found.get(cat,[]):
                any_found=True;lines.append(f'- `{key}/{path}`')
        if not any_found:lines.append('- NOT LOCATED')
    lines += ['','## Existing runners']
    for name,exists in known_runners.items():lines.append(f'- {name}: {"present" if exists else "MISSING"}')
    lines += ['','## Figure manifest']
    if not fig_status:lines.append('- NOT LOCATED')
    for row in fig_status:lines.append(f'- Figure {row["figure"]}: {row["script"]} (script exists={row["script_exists"]}; status={row["reproduction_test"]})')
    lines += ['','## Required independent acceptance tests']
    for k,v in gates.items():lines.append(f'- [ ] {k}: {v}')
    lines += ['','## Candidate source scripts','',
      'Review the JSON inventory for filename-derived scripts. No candidate is proven to be an input producer without inspecting code and executing it.']
    (output/'integration_gate.md').write_text('\n'.join(lines)+'\n')
    print('Wrote:',output/'integration_inventory.json')
    print('Wrote:',output/'integration_gate.md')
    print('Candidate scripts:',len(candidate_scripts))
    print('Figure manifest entries:',len(fig_status))
    print('ATTENTION: raw-to-figure gates remain unverified.')

if __name__=='__main__':main()

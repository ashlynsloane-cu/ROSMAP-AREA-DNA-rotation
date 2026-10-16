#!/usr/bin/env python3
"""Register one REAL upstream producer in a portable raw-to-figure plan.

Does not run a producer, synthesize data, or mark a stage reproduced.
Placeholders: {release_root}, {raw_root}, {output_root}.
"""
import argparse
import json
from pathlib import Path
import re
import sys

STAGES = ('rna_preprocess','tmt_preprocess','monocyte_preprocess','metadata_manifests',
          'area_models','conventional_models','stage7','stage9','stage10',
          'stage11_sensitivity','stage12','figures')
TOKENS = {'release_root','raw_root','output_root'}
BAD = re.compile(r'/home/[^/\s]+/|/Users/[^/\s]+/|stage11_offline_|data/frozen_inputs')


def connect(plan, name, command, inputs, outputs):
    if name not in STAGES:
        raise ValueError('Unknown stage')
    if not command or not outputs:
        raise ValueError('A real command and at least one nonempty output contract are required')
    for value in command + inputs + outputs:
        if BAD.search(value):
            raise ValueError('Machine-specific or frozen-data path: '+value)
        if set(re.findall(r'\{([^{}]+)\}', value)) - TOKENS:
            raise ValueError('Unknown placeholder: '+value)
    if any(not p.startswith('{output_root}/') or '..' in Path(p).parts for p in outputs):
        raise ValueError('Every declared output must be below {output_root}/')
    data=json.loads(Path(plan).read_text())
    stages=data['stages']
    if [x['id'] for x in stages]!=list(STAGES):
        raise ValueError('Unexpected stage configuration')
    for stage in stages:
        if stage['id']==name:
            stage.update(command=command, requires=inputs, produces=outputs)
            stage.pop('blocked_reason', None)
            break
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output-plan', type=Path, required=True)
    ap.add_argument('--stage', required=True, choices=STAGES)
    ap.add_argument('--command-json', required=True, help='JSON array of command argv strings')
    ap.add_argument('--requires-json', default='[]', help='JSON array of input paths')
    ap.add_argument('--produces-json', required=True, help='JSON array of output paths')
    a=ap.parse_args()
    if a.plan.resolve()==a.output_plan.resolve():
        ap.error('Write to a separate plan for review; do not overwrite source plan')
    try:
        command=json.loads(a.command_json)
        inputs=json.loads(a.requires_json)
        outputs=json.loads(a.produces_json)
        if not all(isinstance(x,list) and all(isinstance(v,str) for v in x) for x in (command,inputs,outputs)):
            raise ValueError('All three JSON fields must contain arrays of strings')
        data=connect(a.plan,a.stage,command,inputs,outputs)
    except (ValueError, KeyError) as exc:
        ap.error(str(exc))
    a.output_plan.parent.mkdir(parents=True,exist_ok=True)
    a.output_plan.write_text(json.dumps(data,indent=2)+'\n')
    print('Configured:',a.stage,'Output:',a.output_plan)
    print('No data generated. Run pipeline preflight, then --execute.')

if __name__=='__main__':
    main()

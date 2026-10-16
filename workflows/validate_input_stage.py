#!/usr/bin/env python3
"""Record and validate an authorized input file without modifying data.

This is an INPUT GATE, not a raw FASTQ/proteomics preprocessing algorithm.
The receipt is an integrity/provenance artifact produced in the isolated output root.
"""
import argparse
import hashlib
import json
from pathlib import Path


def receipt(source, result, min_bytes=1):
    source=Path(source).expanduser().resolve()
    result=Path(result).expanduser().resolve()
    if not source.is_file() or source.stat().st_size < min_bytes:
        raise ValueError(f"Missing or empty authorized input: {source}")
    if source == result:
        raise ValueError("Receipt must differ from source")
    h=hashlib.sha256()
    with source.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1<<20), b''):
            h.update(chunk)
    result.parent.mkdir(parents=True,exist_ok=True)
    payload={"type":"authorized_input_integrity_receipt", "schema_version":1,
             "source":str(source), "bytes":source.stat().st_size, "sha256":h.hexdigest(),
             "note":"Input validated but not transformed; original assay preprocessing not performed."}
    result.write_text(json.dumps(payload, indent=2)+'\n')
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args()
    print(json.dumps(receipt(a.source,a.receipt),indent=2))

if __name__=='__main__': main()

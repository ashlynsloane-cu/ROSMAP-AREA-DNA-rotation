import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validate_input_stage', ROOT / 'workflows/validate_input_stage.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class TestInputStage(unittest.TestCase):
    def test_receipt_matches_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'counts.tsv'; source.write_bytes(b'gene\tsample\ng1\t2\n')
            dest=root/'out/receipt.json'
            got=MODULE.receipt(source,dest)
            self.assertEqual(got['bytes'],source.stat().st_size)
            self.assertEqual(json.loads(dest.read_text())['sha256'],got['sha256'])
    def test_empty_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'empty';source.touch()
            with self.assertRaises(ValueError): MODULE.receipt(source,root/'receipt.json')
    def test_plan_has_only_four_connected_stages(self):
        cfg=json.loads((ROOT/'config/raw_to_figures_first_four.json').read_text())
        self.assertEqual([r['id'] for r in cfg['stages'] if r['command']],
          ['rna_preprocess','tmt_preprocess','monocyte_preprocess','metadata_manifests'])
        self.assertTrue(all(r['command'] is None for r in cfg['stages'][4:]))

if __name__=='__main__':unittest.main()

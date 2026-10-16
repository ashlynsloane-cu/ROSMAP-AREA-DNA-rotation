import importlib.util
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parents[1] / 'workflows/run_fresh_area_stage.py'
spec = importlib.util.spec_from_file_location('fresh_area_stage', P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class BridgeTests(unittest.TestCase):
    def test_absent_reports_are_not_valid(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            self.assertFalse(mod.valid_monocyte(p/'m.json', p/'x.tsv', p))
            self.assertFalse(mod.valid_brain(p/'b.json', p/'x.csv', p/'y.csv'))

    def test_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'x'
            p.write_text('abc')
            self.assertEqual(mod.digest(p), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

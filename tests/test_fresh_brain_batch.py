import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'workflows/run_fresh_brain_batch.py'
spec = importlib.util.spec_from_file_location('fresh_brain_batch', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FreshBrainBatchTest(unittest.TestCase):
    def test_braak_dash(self):
        self.assertEqual(module.groups('Braak', '0-2_vs_3-4', ['0–2', '3–4']), ('0–2', '3–4'))

    def test_cerad_direction(self):
        self.assertEqual(module.groups('CERAD', '1-2_vs_3-4', ['CERAD 1–2', 'CERAD 3–4']), ('CERAD 3–4', 'CERAD 1–2'))

    def test_reject_ambiguous(self):
        with self.assertRaises(ValueError):
            module.groups('Braak', '0-2_vs_3-4', ['0–2', '0-2'])

    def test_original_covariates(self):
        self.assertEqual(module.SPECS['dlpfc']['continuous'], 'age_numeric,rin_numeric,pmi_numeric')
        self.assertTrue(module.SPECS['tmt']['global_complete'])


if __name__ == '__main__':
    unittest.main()

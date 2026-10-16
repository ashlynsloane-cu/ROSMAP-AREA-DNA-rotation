import importlib.util
import tempfile
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'workflows/prepare_figure_handoff.py'
spec = importlib.util.spec_from_file_location('prepare_figure_handoff', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestHandoff(unittest.TestCase):
    def test_expected_count(self):
        with tempfile.TemporaryDirectory() as td:
            f = module.find_outputs(td)
            self.assertEqual(len(f), 30)
            self.assertFalse(any(item['present'] for item in f.values()))

    def test_csv_requires_real_file(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'models/brain_area/tmt/cogdx/NCI_vs_AD'
            d.mkdir(parents=True)
            self.assertFalse(module.find_outputs(td)['area/tmt/cogdx/NCI_vs_AD']['present'])
            (d / 'results.csv').write_text('feature,pvalue\na,0.1\n')
            self.assertTrue(module.find_outputs(td)['area/tmt/cogdx/NCI_vs_AD']['present'])

if __name__ == '__main__':
    unittest.main()

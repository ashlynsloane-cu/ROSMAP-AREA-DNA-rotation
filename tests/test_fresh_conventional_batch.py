import importlib.util
from pathlib import Path
import tempfile
import unittest
import pandas as pd

SPEC = importlib.util.spec_from_file_location('fresh', Path(__file__).resolve().parents[1] / 'workflows/run_fresh_conventional_batch.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

class ConventionalTests(unittest.TestCase):
    def test_canonical(self):
        self.assertEqual(MOD.canonical('CERAD 1–2'), '1-2')
        self.assertEqual(MOD.canonical('Braak_3-4'), '3-4')

    def test_cerad_orientation(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / 'cohort.csv'
            pd.DataFrame({'sample_id':['a','b'], 'analysis_group':['CERAD_1-2','CERAD_3-4']}).to_csv(manifest, index=False)
            index = pd.DataFrame([{'modality':'tmt','axis':'CERAD','contrast':'1-2_vs_3-4',
                'cohort':'TMT discovery','manifest':str(manifest),'n':2}])
            _, groups = MOD.select(index, 'tmt', 'CERAD', '1-2_vs_3-4')
            self.assertEqual(groups, ('CERAD_3-4','CERAD_1-2'))

    def test_duplicate_rejected(self):
        index = pd.DataFrame([{'modality':'dlpfc','axis':'cogdx','contrast':'NCI_vs_AD',
            'cohort':'DLPFC discovery','manifest':'/tmp/PURE_1_NCI_2_MCI_4_AD_exclude_3_5_6/manifest.csv','n':2}] * 2)
        with self.assertRaises(ValueError):
            MOD.select(index, 'dlpfc', 'cogdx', 'NCI_vs_AD')

if __name__ == '__main__':
    unittest.main()

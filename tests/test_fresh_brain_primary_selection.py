import importlib.util
from pathlib import Path
import unittest
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'workflows/run_fresh_brain_batch.py'
spec = importlib.util.spec_from_file_location('brain_primary', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrimarySelectionTest(unittest.TestCase):
    def test_pure_selected_regardless_of_order(self):
        rows = []
        for role, definition in [('sensitivity', 'BROAD: includes intermediate'),
                                 ('primary', 'PURE: 1=NCI, 2=MCI, 4=AD')]:
            rows.append(dict(modality='dlpfc', cohort='DLPFC discovery', axis='cogdx',
                             contrast='NCI_vs_AD', role=role, phenotype_definition=definition,
                             manifest=definition))
        index = pd.DataFrame(rows)
        self.assertTrue(module.select(index, 'dlpfc', 'cogdx', 'NCI_vs_AD').manifest.startswith('PURE:'))
        self.assertTrue(module.select(index.iloc[::-1], 'dlpfc', 'cogdx', 'NCI_vs_AD').manifest.startswith('PURE:'))

    def test_duplicates_fail_closed(self):
        entry = dict(modality='tmt', cohort='TMT discovery', axis='cogdx',
                     contrast='MCI_vs_AD', role='primary', phenotype_definition='PURE: X')
        with self.assertRaises(ValueError):
            module.select(pd.DataFrame([entry, entry]), 'tmt', 'cogdx', 'MCI_vs_AD')

    def test_pathology_unique(self):
        entry = dict(modality='dlpfc', cohort='DLPFC discovery', axis='Braak',
                     contrast='0-2_vs_3-4', role='pathology', phenotype_definition='braak')
        self.assertEqual(module.select(pd.DataFrame([entry]), 'dlpfc', 'Braak', '0-2_vs_3-4').role,
                         'pathology')

if __name__ == '__main__':
    unittest.main()

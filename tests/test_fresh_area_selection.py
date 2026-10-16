import importlib.util
import unittest
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / 'workflows'


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FreshAreaSelectionTests(unittest.TestCase):
    def test_monocyte_six_expected(self):
        mod = load('run_fresh_monocyte_all_area')
        frame = pd.DataFrame([{'axis': a, 'contrast': c, 'cohort': cohort,
                               'role': 'primary', 'modality': 'monocyte',
                               'manifest': f'{a}/{c}.csv', 'n': 2,
                               'group_A': 'A', 'group_B': 'B'}
                              for a, c, cohort in mod.CONTRASTS])
        self.assertEqual(len(mod.select_rows(frame)), 6)

    def test_missing_monocyte_rejected(self):
        mod = load('run_fresh_monocyte_all_area')
        frame = pd.DataFrame([{'axis': a, 'contrast': c, 'cohort': cohort,
                               'role': 'primary', 'modality': 'monocyte'}
                              for a, c, cohort in mod.CONTRASTS[:-1]])
        with self.assertRaises(ValueError):
            mod.select_rows(frame)

    def test_brain_discovery_selection(self):
        mod = load('run_fresh_brain_area')
        frame = pd.DataFrame([{'axis': 'cogdx', 'contrast': 'NCI_vs_AD',
                               'cohort': 'DLPFC discovery', 'role': 'primary',
                               'modality': 'dlpfc'}])
        result = mod.select_manifest(frame, 'dlpfc', 'NCI_vs_AD', 'cogdx')
        self.assertEqual(result['modality'], 'dlpfc')


if __name__ == '__main__':
    unittest.main()

"""Regression tests for index display labels and model group identities."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / 'workflows/run_fresh_monocyte_all_area.py'
spec = importlib.util.spec_from_file_location('fresh_model_labels_unicode', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class UnicodePathologyGroupTests(unittest.TestCase):
    def test_braak_en_dash_in_index(self):
        row = {'axis': 'Braak', 'contrast': '0-2_vs_3-4',
               'group_A': '0–2', 'group_B': '3–4'}
        tab = pd.DataFrame({'analysis_group': ['0-2', '3-4']})
        self.assertEqual(module.resolve_groups(row, tab), ('0-2', '3-4'))

    def test_braak_en_dash_in_manifest(self):
        row = {'axis': 'Braak', 'contrast': '3-4_vs_5-6',
               'group_A': '3–4', 'group_B': '5–6'}
        tab = pd.DataFrame({'analysis_group': ['3–4', '5–6']})
        self.assertEqual(module.resolve_groups(row, tab), ('3–4', '5–6'))

    def test_cerad_index_en_dash_and_prefix(self):
        row = {'axis': 'CERAD', 'contrast': '1-2_vs_3-4',
               'group_A': 'CERAD_1–2', 'group_B': 'CERAD_3–4'}
        tab = pd.DataFrame({'analysis_group': ['CERAD_1-2', 'CERAD_3-4']})
        self.assertEqual(module.resolve_groups(row, tab), ('CERAD_1-2', 'CERAD_3-4'))

    def test_rejects_swapped_index_labels(self):
        row = {'axis': 'Braak', 'contrast': '0-2_vs_3-4',
               'group_A': '3–4', 'group_B': '0–2'}
        with self.assertRaisesRegex(ValueError, 'conflict'):
            module.resolve_groups(row, pd.DataFrame({'analysis_group': ['0-2', '3-4']}))

    def test_rejects_ambiguous_manifest_aliases(self):
        row = {'axis': 'Braak', 'contrast': '0-2_vs_3-4',
               'group_A': '0–2', 'group_B': '3–4'}
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            module.resolve_groups(row, pd.DataFrame({'analysis_group': ['0-2', '0–2', '3-4']}))


if __name__ == '__main__':
    unittest.main()

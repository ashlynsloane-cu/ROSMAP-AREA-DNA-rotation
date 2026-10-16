"""Regression tests for pathology axis prefixes in freshly built index."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'workflows' / 'run_fresh_monocyte_all_area.py'
spec = importlib.util.spec_from_file_location('fresh_monocyte_all', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CeradLabels(unittest.TestCase):
    def test_index_prefixed_with_space(self):
        row = {'axis': 'CERAD', 'contrast': '1-2_vs_3-4',
               'group_A': 'CERAD 1–2', 'group_B': 'CERAD 3–4'}
        manifest = pd.DataFrame({'analysis_group': ['1-2', '3-4']})
        self.assertEqual(module.resolve_groups(row, manifest), ('1-2', '3-4'))

    def test_manifest_prefixed_with_space(self):
        row = {'axis': 'CERAD', 'contrast': '1-2_vs_3-4',
               'group_A': 'CERAD 1–2', 'group_B': 'CERAD 3–4'}
        manifest = pd.DataFrame({'analysis_group': ['CERAD 1–2', 'CERAD 3–4']})
        self.assertEqual(module.resolve_groups(row, manifest),
                         ('CERAD 1–2', 'CERAD 3–4'))

    def test_reversed_labels_fail(self):
        row = {'axis': 'CERAD', 'contrast': '1-2_vs_3-4',
               'group_A': 'CERAD 3–4', 'group_B': 'CERAD 1–2'}
        manifest = pd.DataFrame({'analysis_group': ['1-2', '3-4']})
        with self.assertRaises(ValueError):
            module.resolve_groups(row, manifest)


if __name__ == '__main__':
    unittest.main()

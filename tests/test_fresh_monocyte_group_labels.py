import importlib.util
import unittest
from pathlib import Path
import pandas as pd

file = Path(__file__).resolve().parents[1] / "workflows/run_fresh_monocyte_all_area.py"
spec = importlib.util.spec_from_file_location("fresh_six_groups", file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class TestManifestGroupResolution(unittest.TestCase):
    def test_braak_prefix_is_display_only(self):
        row = {"axis": "Braak", "contrast": "0-2_vs_3-4", "group_A": "Braak_0-2", "group_B": "Braak_3-4"}
        frame = pd.DataFrame({"analysis_group": ["0-2", "3-4"]})
        self.assertEqual(mod.resolve_groups(row, frame), ("0-2", "3-4"))

    def test_braak_plain_index_labels(self):
        row = {"axis": "Braak", "contrast": "3-4_vs_5-6", "group_A": "3-4", "group_B": "5-6"}
        frame = pd.DataFrame({"analysis_group": ["3-4", "5-6"]})
        self.assertEqual(mod.resolve_groups(row, frame), ("3-4", "5-6"))

    def test_cerad_labels(self):
        row = {"axis": "CERAD", "contrast": "1-2_vs_3-4", "group_A": "CERAD_1-2", "group_B": "CERAD_3-4"}
        frame = pd.DataFrame({"analysis_group": ["CERAD_1-2", "CERAD_3-4"]})
        self.assertEqual(mod.resolve_groups(row, frame), ("CERAD_1-2", "CERAD_3-4"))

    def test_rejects_swapped_groups(self):
        row = {"axis": "Braak", "contrast": "0-2_vs_3-4", "group_A": "Braak_3-4", "group_B": "Braak_0-2"}
        frame = pd.DataFrame({"analysis_group": ["0-2", "3-4"]})
        with self.assertRaisesRegex(ValueError, "conflict"):
            mod.resolve_groups(row, frame)

    def test_rejects_wrong_manifest_groups(self):
        row = {"axis": "Braak", "contrast": "0-2_vs_3-4", "group_A": "0-2", "group_B": "3-4"}
        frame = pd.DataFrame({"analysis_group": ["0-2", "5-6"]})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            mod.resolve_groups(row, frame)

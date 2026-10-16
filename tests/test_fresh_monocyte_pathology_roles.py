import importlib.util
import unittest
from pathlib import Path
import pandas as pd

script = Path(__file__).resolve().parents[1] / "workflows" / "run_fresh_monocyte_all_area.py"
spec = importlib.util.spec_from_file_location("monocyte_fresh", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestPathologySelection(unittest.TestCase):
    def make_rows(self):
        return pd.DataFrame([{
            "axis": axis, "contrast": contrast, "cohort": cohort,
            "modality": "monocyte", "role": "primary" if axis == "cogdx" else "followup",
            "manifest": f"{axis}/{contrast}.csv", "n": 2,
            "group_A": "A", "group_B": "B"
        } for axis, contrast, cohort in module.CONTRASTS])

    def test_nonprimary_pathology_allowed(self):
        self.assertEqual(len(module.select_rows(self.make_rows())), 6)

    def test_duplicate_pathology_rejected(self):
        frame = self.make_rows()
        frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            module.select_rows(frame)

    def test_nonprimary_cognitive_rejected(self):
        frame = self.make_rows()
        frame.loc[0, "role"] = "followup"
        with self.assertRaisesRegex(ValueError, "missing"):
            module.select_rows(frame)

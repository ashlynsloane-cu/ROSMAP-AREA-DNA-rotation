import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflows'))
from check_handoff_release import unsafe_path

class TestHandoffSafety(unittest.TestCase):
    def test_blocks_data(self):
        for name in ['data/x.tsv', 'results/x.csv', 'workflows/__pycache__/a.pyc', '.venv-x/lib/a.py', 'a.rds', 'x/monocyte_raw_counts_554.tsv']:
            with self.subTest(name=name):
                self.assertTrue(unsafe_path(name))
    def test_allows_source(self):
        for name in ['workflows/run_stage11_batch.py', 'docs/LAB_HANDOFF_FINAL_CHECKLIST.md', 'tests/test_check_handoff_release.py']:
            with self.subTest(name=name):
                self.assertFalse(unsafe_path(name))

if __name__ == '__main__':
    unittest.main()

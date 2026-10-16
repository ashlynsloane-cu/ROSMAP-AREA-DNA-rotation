import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).resolve().parent.parent / 'workflows/run_stage11_conventional_batch.py'
spec = importlib.util.spec_from_file_location('conventional_batch', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_historical_resolution_missing(self):
        self.assertIsNone(m.resolve_historical('/rosmap_multiomic_slurm/unavailable.csv', [Path('/tmp')]))

    def test_compare_reorder_and_roundoff(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a.csv', Path(d)/'b.csv'
            pd.DataFrame({'feature_id': ['x', 'y'], 'pvalue': [.1, .2], 'fdr_significant': [False, True]}).to_csv(a, index=False)
            pd.DataFrame({'feature_id': ['y', 'x'], 'pvalue': [.2, .1 + 1e-12], 'fdr_significant': [True, False]}).to_csv(b, index=False)
            self.assertEqual(m.compare_table(a,b,1e-7,1e-10)['status'],'pass')

    def test_compare_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a.csv', Path(d)/'b.csv'
            pd.DataFrame({'feature_id':['x'], 'padj':[.1]}).to_csv(a,index=False)
            pd.DataFrame({'feature_id':['x'], 'padj':[.4]}).to_csv(b,index=False)
            self.assertEqual(m.compare_table(a,b,1e-7,1e-10)['status'],'fail')

    def test_summary_ignores_localized_paths_but_checks_counts(self):
        with tempfile.TemporaryDirectory() as d:
            a,b = Path(d)/'a.csv',Path(d)/'b.csv'
            pd.DataFrame([{'counts_file':'/old/count.rds','group_A':'NCI','n':40}]).to_csv(a,index=False)
            pd.DataFrame([{'counts_file':'/new/count.rds','group_A':'NCI','n':40}]).to_csv(b,index=False)
            self.assertEqual(m.compare_summary(a,b,1e-7,1e-10)['status'],'pass')
            pd.DataFrame([{'counts_file':'/new/count.rds','group_A':'AD','n':40}]).to_csv(b,index=False)
            self.assertEqual(m.compare_summary(a,b,1e-7,1e-10)['status'],'fail')

if __name__=='__main__':
    unittest.main()

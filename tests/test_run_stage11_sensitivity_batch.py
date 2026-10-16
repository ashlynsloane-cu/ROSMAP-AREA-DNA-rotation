import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location('runner', Path(__file__).resolve().parent.parent / 'workflows' / 'run_stage11_sensitivity_batch.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_covars(self):
        self.assertEqual(m.covars(float('nan')), '')
        self.assertEqual(m.covars('age, sex'), 'age,sex')

    def test_boolean(self):
        self.assertTrue(m.yes(True))
        self.assertFalse(m.yes('false'))

    def test_command(self):
        job = {'expression': Path('/tmp/expression'), 'manifest': Path('/tmp/meta'),
               'contrast': 'subset', 'group_a': 'CERAD_3-4', 'group_b': 'CERAD_1-2',
               'orientation': 'samples_by_features', 'fdr': .05,
               'continuous': 'age_numeric', 'categorical': '', 'complete': False}
        cmd = m.command(Path('/tmp/script'), job, Path('/tmp/out'))
        self.assertEqual(cmd[cmd.index('--group-a') + 1], 'CERAD_3-4')
        self.assertIn('--continuous', cmd)
        self.assertNotIn('--categorical', cmd)
        self.assertNotIn('--global-complete-features', cmd)
        job['complete'] = True
        self.assertIn('--global-complete-features', m.command(Path('/tmp/script'), job, Path('/tmp/out')))

    def test_comparison_row_order_and_roundoff(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / 'a.csv', Path(t) / 'b.csv'
            pd.DataFrame({'feature_id': ['g1', 'g2'], 'pvalue': [.1, .2], 'fdr_significant': [True, False]}).to_csv(a, index=False)
            pd.DataFrame({'feature_id': ['g2', 'g1'], 'pvalue': [.2, .1 + 1e-15], 'fdr_significant': [False, True]}).to_csv(b, index=False)
            self.assertEqual(m.compare_csv(a, b, 1e-9, 1e-12)['status'], 'pass')
            pd.DataFrame({'feature_id': ['g2', 'g1'], 'pvalue': [.2, .3], 'fdr_significant': [False, True]}).to_csv(b, index=False)
            self.assertEqual(m.compare_csv(a, b, 1e-9, 1e-12)['status'], 'fail')

    def test_near_zero_direction_flip_is_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.csv', Path(temp) / 'b.csv'
            pd.DataFrame({
                'feature_id': ['ENSG00000168827.10'],
                'partial_rank_statistic': [-1.989520e-13],
                'pvalue': [1.0], 'fdr_significant': [False],
                'direction': ['MCI_associated_lower_rank'],
            }).to_csv(a, index=False)
            pd.DataFrame({
                'feature_id': ['ENSG00000168827.10'],
                'partial_rank_statistic': [2.273737e-13],
                'pvalue': [1.0], 'fdr_significant': [False],
                'direction': ['MCI_associated_higher_rank'],
            }).to_csv(b, index=False)
            outcome = m.compare_csv(a, b, 1e-9, 1e-12)
            self.assertEqual(outcome['status'], 'pass')
            self.assertEqual(outcome['near_zero_direction_exception_count'], 1)
            self.assertEqual(outcome['near_zero_direction_exception_feature_ids'],
                             ['ENSG00000168827.10'])

    def test_meaningful_direction_reversal_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.csv', Path(temp) / 'b.csv'
            pd.DataFrame({
                'feature_id': ['g1'], 'partial_rank_statistic': [1.5],
                'direction': ['MCI_associated_higher_rank'],
            }).to_csv(a, index=False)
            pd.DataFrame({
                'feature_id': ['g1'], 'partial_rank_statistic': [1.5],
                'direction': ['MCI_associated_lower_rank'],
            }).to_csv(b, index=False)
            outcome = m.compare_csv(a, b, 1e-9, 1e-12)
            self.assertEqual(outcome['status'], 'fail')
            self.assertEqual(outcome['discrepancies']['direction'], 1)

    def test_invalid_near_zero_direction_label_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.csv', Path(temp) / 'b.csv'
            pd.DataFrame({
                'feature_id': ['g1'], 'partial_rank_statistic': [1e-14],
                'direction': ['MCI_associated_higher_rank'],
            }).to_csv(a, index=False)
            pd.DataFrame({
                'feature_id': ['g1'], 'partial_rank_statistic': [-1e-14],
                'direction': ['INVALID'],
            }).to_csv(b, index=False)
            outcome = m.compare_csv(a, b, 1e-9, 1e-12)
            self.assertEqual(outcome['status'], 'fail')

    def test_manifest_strict(self):
        self.assertEqual(m.check_manifest({'group_A': 'A'}, {'group_A': 'B'})['group_A']['generated'], 'B')


if __name__ == '__main__':
    unittest.main()

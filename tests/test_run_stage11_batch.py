import importlib.util
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import numpy as np

P=Path(__file__).resolve().parents[1]/'workflows/run_stage11_batch.py'
spec=importlib.util.spec_from_file_location('stage11',P)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_boolean(self):
        self.assertTrue(m.is_true(True))
        self.assertTrue(m.is_true('1'))
        self.assertFalse(m.is_true('False'))
    def test_covariates(self):
        self.assertEqual(m.split_covariates(' age , rin '),['age','rin'])
        self.assertEqual(m.split_covariates(float('nan')),[])
    def test_global_complete_command(self):
        j={'expression_local':Path('/tmp/expr'),'sample_manifest_local':Path('/tmp/meta'),
           'model_manifest':{'contrast':'test','fdr':0.05,'block':None,'group_A':'NCI','group_B':'AD'},
           'group_a':'NCI','group_b':'AD','orientation':'features_by_samples',
           'A_continuous':'age','A_categorical':'sex',
           'global_complete_features':True}
        self.assertIn('--global-complete-features',m.build_command(Path('/tmp/area.py'),j,Path('/tmp/out')))
        j['global_complete_features']=False
        self.assertNotIn('--global-complete-features',m.build_command(Path('/tmp/area.py'),j,Path('/tmp/out')))
    def test_feature_reordering_and_rounding(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.csv';b=Path(t)/'b.csv'
            pd.DataFrame({'feature_id':['x','y'],'pvalue':[.1,.2], 'fdr_significant':[False,True]}).to_csv(a,index=False)
            pd.DataFrame({'feature_id':['y','x'],'pvalue':[.2, .1+1e-15], 'fdr_significant':[True,False]}).to_csv(b,index=False)
            self.assertEqual(m.compare_results(a,b)['status'],'pass')
    def test_detect_difference(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.csv';b=Path(t)/'b.csv'
            pd.DataFrame({'feature_id':['x'],'pvalue':[.1], 'direction':['up']}).to_csv(a,index=False)
            pd.DataFrame({'feature_id':['x'],'pvalue':[.2], 'direction':['down']}).to_csv(b,index=False)
            self.assertEqual(m.compare_results(a,b)['status'],'fail')
    def test_detect_missing_feature(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.csv';b=Path(t)/'b.csv'
            pd.DataFrame({'feature_id':['x'],'pvalue':[.1]}).to_csv(a,index=False)
            pd.DataFrame({'feature_id':['y'],'pvalue':[.1]}).to_csv(b,index=False)
            self.assertEqual(m.compare_results(a,b)['status'],'fail')
    def test_manifest_diff(self):
        self.assertEqual(m.compare_manifest({'n':10},{'n':10}),{})
        self.assertIn('n',m.compare_manifest({'n':10},{'n':9}))

if __name__=='__main__':unittest.main()

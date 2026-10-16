import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('connector',HERE/'workflows/connect_pipeline_stage.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class ConnectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.plan=Path(self.tmp.name)/'plan.json'
        self.plan.write_text(json.dumps({'stages':[{'id':s,'command':None,'requires':[],'produces':[]} for s in mod.STAGES]}))
    def tearDown(self):
        self.tmp.cleanup()
    def test_register_command(self):
        v=mod.connect(self.plan,'rna_preprocess',['python3','{release_root}/scripts/producer.py'],
                      ['{raw_root}/rna.rds'],['{output_root}/rna/model.csv'])
        self.assertEqual(v['stages'][0]['command'][0],'python3')
    def test_no_absolute_homes(self):
        with self.assertRaises(ValueError):
            mod.connect(self.plan,'stage10',['python3','/home/someuser/foo.py'],[],['{output_root}/x.tsv'])
    def test_no_untracked_output(self):
        with self.assertRaises(ValueError):
            mod.connect(self.plan,'stage10',['python3','script.py'],[],['{release_root}/data/out.tsv'])
    def test_no_faux_success(self):
        with self.assertRaises(ValueError):
            mod.connect(self.plan,'figures',['python3','figures.py'],[],[])

if __name__=='__main__': unittest.main()

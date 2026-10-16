import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'workflows' / 'run_raw_to_figures.py'
spec = importlib.util.spec_from_file_location('raw_runner', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PLAN = json.loads((SOURCE.parents[1] / 'config/raw_to_figures_plan.json').read_text())


class GateTests(unittest.TestCase):
    def test_plan_valid(self):
        mod.check_config(PLAN)

    def test_unconfigured_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for d in ('repo', 'raw'):
                (base / d).mkdir()
            result = mod.run(PLAN, {'release_root': base / 'repo',
                                     'raw_root': base / 'raw',
                                     'output_root': base / 'output'}, False)
            self.assertFalse(result['raw_to_figure_complete'])
            self.assertEqual(result['stages'][0]['status'], 'BLOCKED')

    def test_rejects_research_root_output(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'repo').mkdir()
            (base / 'raw').mkdir()
            with self.assertRaises(ValueError):
                mod.run(PLAN, {'release_root': base / 'repo',
                               'raw_root': base / 'raw',
                               'output_root': base / 'raw' / 'result'})

    def test_producer_missing_input_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'repo').mkdir()
            (base / 'raw').mkdir()
            plan = json.loads(json.dumps(PLAN))
            plan['stages'][0].update(command=['python3', '--version'],
                                     requires=['{raw_root}/not-there.csv'],
                                     produces=['{output_root}/rna.csv'])
            result = mod.run(plan, {'release_root': base / 'repo',
                                    'raw_root': base / 'raw',
                                    'output_root': base / 'output'}, False)
            self.assertIn('Missing required input', result['stages'][0]['reason'])


if __name__ == '__main__':
    unittest.main()

import importlib.util
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parents[1] / 'workflows/restore_figure_inputs.py'
spec = importlib.util.spec_from_file_location('restore', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestRestore(unittest.TestCase):
    def test_safe(self):
        self.assertEqual(m.safe_relative('results/a.tsv'), Path('results/a.tsv'))
        for x in ('../results/a.tsv', '/tmp/results/a.tsv', 'data/a.tsv', 'results/../../x'):
            with self.assertRaises(ValueError):
                m.safe_relative(x)

    def test_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'
            p.write_bytes(b'abc')
            self.assertEqual(m.digest(p), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


if __name__ == '__main__':
    unittest.main()

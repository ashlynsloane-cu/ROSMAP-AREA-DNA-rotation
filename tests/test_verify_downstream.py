import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'workflows' / 'verify_downstream.py'
spec = importlib.util.spec_from_file_location('verify_downstream', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.a = root / 'original.tsv'
        self.b = root / 'generated.tsv'

    def check(self, old, new, status):
        self.a.write_text(old, encoding='utf-8')
        self.b.write_text(new, encoding='utf-8')
        self.assertEqual(module.compare(self.a, self.b)['status'], status)

    def test_exact(self):
        self.check('a\tb\n1\t2\n', 'a\tb\n1\t2\n', 'exact')

    def test_reordered_rows(self):
        self.check('a\n1\n2\n', 'a\n2\n1\n', 'content_equivalent')

    def test_duplicate_counts_preserved(self):
        self.check('a\n1\n1\n2\n', 'a\n1\n2\n2\n', 'different')

    def test_different_numbers(self):
        self.check('a\n0.05\n', 'a\n0.06\n', 'different')

    def test_different_header(self):
        self.check('a\n1\n', 'b\n1\n', 'different')

    def test_serialization_change(self):
        self.check('a\n1\n', 'a\r\n1\r\n', 'content_equivalent')

    def test_malformed_row(self):
        self.check('a\tb\n1\t2\n', 'a\tb\n1\n', 'different')


if __name__ == '__main__':
    unittest.main()

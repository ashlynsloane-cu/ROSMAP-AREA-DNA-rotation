import csv
import tempfile
import unittest
from pathlib import Path
from workflows.validate_monocyte_inputs import check_matrices,check_metadata

class MonocyteGateTests(unittest.TestCase):
    def test_full_check(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,m=(Path(d)/name for name in ('a.tsv','b.tsv','m.tsv'))
            a.write_text('Geneid\tSample_1\tSample_2\ng1\t0\t3\ng2\t5\t2\n')
            b.write_text('Geneid\tSample_1\tSample_2\ng1\t0.0\t4.1\ng2\t7.2\t2.2\n')
            m.write_text('specimen\tindividualID\nSample_1\ti1\nSample_2\ti2\n')
            stats,names=check_matrices(a,b,2)
            self.assertEqual(stats['features'],2)
            self.assertTrue(check_metadata(m,names)['full_exact_match'])
    def test_misaligned(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=(Path(d)/name for name in ('a.tsv','b.tsv'))
            a.write_text('Geneid\tSample_1\ng1\t3\n')
            b.write_text('Geneid\tSample_1\ng2\t3\n')
            with self.assertRaises(ValueError):check_matrices(a,b,1)
    def test_fractional_raw(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=(Path(d)/name for name in ('a.tsv','b.tsv'))
            a.write_text('Geneid\tSample_1\ng1\t3.2\n')
            b.write_text('Geneid\tSample_1\ng1\t3\n')
            with self.assertRaises(ValueError):check_matrices(a,b,1)

if __name__=='__main__':unittest.main()

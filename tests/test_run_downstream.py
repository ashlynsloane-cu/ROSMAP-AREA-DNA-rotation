"""Regression check: generated Stage 12I script must parse and compile."""
import ast
from pathlib import Path
import tempfile
import unittest
import sys

WORKFLOWS = Path(__file__).resolve().parents[1] / 'workflows'
sys.path.insert(0, str(WORKFLOWS))
from run_downstream import runtime_with_frozen_gmts


class FrozenGmtRuntimeTests(unittest.TestCase):
    def test_generated_runtime_compiles(self):
        source_text = (
            'from pathlib import Path\n'
            'import pandas as pd\n'
            'GENESET_DIR = Path("/tmp/sets")\n'
            'def load_gene_sets(universe, args):\n'
            '    raise RuntimeError("remote library lookup")\n'
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.py'
            path.write_text(source_text, encoding='utf-8')
            generated = runtime_with_frozen_gmts(path, Path(folder))
            ast.parse(generated)
            compile(generated, '<generated-stage12I>', 'exec')
            self.assertIn('import shutil', generated)
            self.assertIn("ROSMAP_FROZEN_GSEA_REFERENCE", generated)


if __name__ == '__main__':
    unittest.main()

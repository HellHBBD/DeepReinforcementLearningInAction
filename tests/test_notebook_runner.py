"""Exercise the CI runner's exit status, including IPython's shell-command behavior."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import nbformat


class NotebookRunner(unittest.TestCase):
    def test_runner_reports_success_and_propagates_failures(self):
        runner = pathlib.Path(__file__).resolve().parents[1] / 'scripts/check_notebooks.py'
        cases = [
            ('answer = 6 * 7\nassert answer == 42', True),
            ('raise ValueError("intentional failure")', False),
            ('import sys\n!{sys.executable} -c "raise SystemExit(7)"', False),
            ('losses = [float("nan")]', False),
        ]
        for source, expected in cases:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                (root / 'scripts').mkdir()
                shutil.copy2(runner, root / 'scripts/check_notebooks.py')
                nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]),
                               root / 'example.ipynb')
                manifest = [{'path': 'example.ipynb', 'smoke_through': 0,
                             'boundary_sha256': hashlib.sha256(source.encode()).hexdigest()}]
                (root / 'scripts/notebooks.json').write_text(json.dumps(manifest))
                run = subprocess.run([sys.executable, str(root / 'scripts/check_notebooks.py')],
                                     capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode == 0, expected, run.stdout + run.stderr)
                report = json.loads((root / 'artifacts/notebooks/summary.json').read_text())
                self.assertEqual(report[0]['success'], expected)

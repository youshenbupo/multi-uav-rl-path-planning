"""Smoke test for the MATLAB reference-data inspection command."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MatlabReferenceInspectorTests(unittest.TestCase):
    """Ensure exported MATLAB fixtures can be inspected from Python."""

    def test_inspector_reports_files_fields_and_numeric_health(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/inspect_matlab_reference.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        for label in (
            "MATLAB reference files:",
            "reference_cases.mat",
            "Fields:",
            "Shape:",
            "Dtype:",
            "Range:",
            "NaN count:",
            "Inf count:",
        ):
            self.assertIn(label, completed.stdout)


if __name__ == "__main__":
    unittest.main()

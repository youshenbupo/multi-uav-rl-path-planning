"""Smoke tests for the phase-two Python project scaffold."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectScaffoldTests(unittest.TestCase):
    """Validate the minimal project layout and environment report interface."""

    def test_required_directories_and_package_exist(self) -> None:
        required_paths = (
            "configs",
            "data/regression",
            "data/expert",
            "data/scenarios",
            "multiuav/core",
            "multiuav/geometry",
            "multiuav/evaluation",
            "multiuav/expert",
            "multiuav/envs",
            "multiuav/learning",
            "multiuav/safety",
            "multiuav/data",
            "scripts",
            "tools/matlab",
        )
        for relative_path in required_paths:
            self.assertTrue((PROJECT_ROOT / relative_path).is_dir(), relative_path)

        self.assertTrue((PROJECT_ROOT / "multiuav/__init__.py").is_file())

    def test_environment_check_reports_required_capabilities(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/check_environment.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        for label in (
            "Python:",
            "NumPy:",
            "PyTorch:",
            "CUDA available:",
            "GPU:",
            "Gymnasium:",
            "CVXPY:",
            "OSQP:",
        ):
            self.assertIn(label, completed.stdout)

    def test_environment_file_pins_verified_runtime_without_pyg(self) -> None:
        environment_file = (PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8")

        self.assertIn("name: multiuav_rl", environment_file)
        self.assertIn("python=3.11.15", environment_file)
        self.assertIn("torch==2.13.0+cu130", environment_file)
        self.assertIn("torchvision==0.28.0+cu130", environment_file)
        self.assertNotIn("torch-geometric", environment_file.lower())


if __name__ == "__main__":
    unittest.main()

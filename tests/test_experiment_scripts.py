"""Command-level contracts for the single Phase-14 experiment interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_phase14_entry_points_share_mainline_switches() -> None:
    """Every public command exposes the same reproducibility and ablation controls."""
    required_options = {
        "--config",
        "--seed",
        "--device",
        "--num-uavs",
        "--scenario",
        "--checkpoint",
        "--render",
        "--use-expert-pretrain",
        "--use-graph",
        "--use-hierarchy",
        "--use-cbf",
    }
    for script_name in ("train.py", "evaluate.py", "run_benchmark.py", "run_ablation.py"):
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert all(option in completed.stdout for option in required_options)

"""Tests for reproducible Phase-14 experiment artifacts and method availability."""

from __future__ import annotations

import json
from pathlib import Path

from multiuav.experiments.metrics import SeedResult
from multiuav.experiments.outputs import ExperimentOutput
from multiuav.experiments.registry import MethodRegistry
from multiuav.experiments.spec import ExperimentSpec


def test_experiment_writer_preserves_raw_seed_result_and_statistical_summary(
    tmp_path: Path,
) -> None:
    """Every requested seed remains in JSONL while summaries include uncertainty."""
    spec = ExperimentSpec(name="smoke", seeds=(3, 5))
    layout = ExperimentOutput.create(tmp_path, spec, metadata={"device": "cpu"})
    layout.write_raw_result(seed=3, episode=0, result={"success_rate": 1.0, "energy_proxy": 3.0})
    summary = layout.write_summary(
        [
            SeedResult(seed=3, metrics={"success_rate": 1.0, "energy_proxy": 3.0}),
            SeedResult(seed=5, metrics={"success_rate": 0.0, "energy_proxy": 5.0}),
        ]
    )

    raw_path = layout.root / "raw_results" / "seed_3.jsonl"
    assert json.loads(raw_path.read_text(encoding="utf-8").strip())["seed"] == 3
    assert summary["success_rate"]["mean"] == 0.5
    assert summary["success_rate"]["confidence_interval_95"] is not None
    assert (layout.root / "summary.csv").is_file()
    assert (layout.root / "environment.json").is_file()


def test_unavailable_external_baseline_is_explicit() -> None:
    """Unimplemented external baselines cannot silently produce invented values."""
    method = MethodRegistry().resolve("orca")

    assert method.availability == "unavailable"
    assert "not implemented" in method.reason

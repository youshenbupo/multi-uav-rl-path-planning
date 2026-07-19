"""Tests for reproducible Phase-14 experiment artifacts and method availability."""

from __future__ import annotations

import json
from pathlib import Path

from multiuav.experiments.metrics import SeedResult
from multiuav.experiments.outputs import ExperimentOutput
from multiuav.experiments.registry import MethodRegistry
from multiuav.experiments.runner import evaluate_goal_controller
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


def test_dynamic_communication_evaluation_writes_auditable_seed_metrics(tmp_path: Path) -> None:
    """The common evaluator exercises moving obstacles, delayed communication, and CBF."""
    spec = ExperimentSpec(name="dynamic_smoke", seeds=(13,), num_uavs=3, device="cpu")
    layout = ExperimentOutput.create(tmp_path, spec, metadata={"device": "cpu"})
    result = evaluate_goal_controller(spec, layout, episodes_per_seed=1, max_steps=4)

    assert result[0].seed == 13
    assert "CBF_intervention_rate" in result[0].metrics
    raw = (layout.root / "raw_results" / "seed_13.jsonl").read_text(encoding="utf-8")
    assert '"dynamic_obstacle_count": 1' in raw
    assert '"communication_delay_steps": 1' in raw

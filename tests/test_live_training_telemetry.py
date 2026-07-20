"""Durable CBF telemetry contracts for interrupted learned-policy training."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    ("family", "config_name"),
    (
        ("mappo", "dynamic_mappo_smoke.yaml"),
        ("graph_mappo", "dynamic_graph_smoke.yaml"),
    ),
)
def test_training_persists_cbf_telemetry_after_each_update(
    tmp_path: Path, family: str, config_name: str
) -> None:
    """A forced interruption after an update must still leave inspectable CBF evidence."""
    root = Path(__file__).parents[1]
    telemetry_path = tmp_path / f"{family}_live_telemetry.json"
    telemetry_path.write_text(
        json.dumps({"initial_evaluation_cbf": {"emergency_fallback_count": 7}}),
        encoding="utf-8",
    )
    if family == "mappo":
        from multiuav.learning.runner import MAPPOExperiment, load_mappo_experiment_config

        config = replace(
            load_mappo_experiment_config(root / "configs/experiments" / config_name),
            rollout_length=1,
            total_steps=3,
        )
        experiment = MAPPOExperiment(config, device=torch.device("cpu"))
    else:
        from multiuav.learning.graph_runner import (
            GraphMAPPOExperiment,
            load_graph_experiment_config,
        )

        config = replace(
            load_graph_experiment_config(root / "configs/experiments" / config_name),
            rollout_length=1,
            total_steps=3,
        )
        experiment = GraphMAPPOExperiment(config, device=torch.device("cpu"))

    experiment.train(telemetry_path=telemetry_path)
    experiment.close()

    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert telemetry["total_transitions"] == 3
    assert telemetry["cbf"]["decision_count"] == 1
    assert "emergency_events" in telemetry["cbf"]
    assert telemetry["initial_evaluation_cbf"]["emergency_fallback_count"] == 7

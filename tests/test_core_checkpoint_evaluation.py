"""Regression contracts for uniform learned-policy checkpoint evaluation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import multiuav.experiments.core_evaluation as core_evaluation
from multiuav.experiments.core_evaluation import (
    evaluate_graph_checkpoint,
    evaluate_mappo_checkpoint,
)
from multiuav.experiments.outputs import ExperimentOutput
from multiuav.experiments.spec import ExperimentSpec


def test_checkpoint_evaluation_factory_identifies_the_real_controller(tmp_path: Path) -> None:
    """A reproducible output must name its exact checkpoint/controller, never a smoke policy."""
    checkpoint = tmp_path / "saved_policy.pt"
    checkpoint.write_bytes(b"fixture")
    spec = ExperimentSpec(
        name="identified_checkpoint",
        seeds=(71,),
        num_uavs=3,
        device="cpu",
        checkpoint=checkpoint,
    )

    factory = getattr(core_evaluation, "create_checkpoint_evaluation_output", None)
    assert factory is not None
    output = factory(
        tmp_path,
        spec,
        controller="mappo_checkpoint",
        config_path=Path("configs/rl/dynamic_mappo_baseline.yaml"),
    )

    metadata = json.loads((output.root / "environment.json").read_text(encoding="utf-8"))
    assert metadata["controller"] == "mappo_checkpoint"
    assert metadata["checkpoint"] == str(checkpoint)


@pytest.mark.parametrize(
    ("family", "config_name", "checkpoint_name", "evaluator"),
    (
        (
            "mappo",
            "dynamic_mappo_smoke.yaml",
            "mappo_fixture.pt",
            evaluate_mappo_checkpoint,
        ),
        (
            "graph_mappo",
            "dynamic_graph_smoke.yaml",
            "graph_fixture.pt",
            evaluate_graph_checkpoint,
        ),
    ),
)
def test_checkpoint_evaluators_write_the_same_episode_metric_contract(
    tmp_path: Path,
    family: str,
    config_name: str,
    checkpoint_name: str,
    evaluator: object,
) -> None:
    """MLP and graph checkpoints must use real actors and retain comparable JSONL records."""
    root = Path(__file__).parents[1]
    config_path = root / "configs/experiments" / config_name
    checkpoint = tmp_path / checkpoint_name
    if family == "mappo":
        from multiuav.learning.mappo import save_checkpoint
        from multiuav.learning.runner import MAPPOExperiment, load_mappo_experiment_config

        config = replace(
            load_mappo_experiment_config(config_path), num_envs=1, total_steps=3
        )
        fixture = MAPPOExperiment(config, device=torch.device("cpu"))
        save_checkpoint(checkpoint, fixture.trainer, step=0)
        fixture.close()
    else:
        from multiuav.learning.graph_mappo import save_graph_checkpoint
        from multiuav.learning.graph_runner import (
            GraphMAPPOExperiment,
            load_graph_experiment_config,
        )

        config = replace(
            load_graph_experiment_config(config_path), num_envs=1, total_steps=3
        )
        fixture = GraphMAPPOExperiment(config, device=torch.device("cpu"))
        save_graph_checkpoint(checkpoint, fixture.trainer, step=0)
        fixture.close()

    spec = ExperimentSpec(
        name=f"{family}_checkpoint_smoke",
        seeds=(43,),
        num_uavs=3,
        device="cpu",
        checkpoint=checkpoint,
        use_cbf=True,
    )
    output = ExperimentOutput.create(tmp_path, spec, metadata={"fixture_checkpoint": True})

    results = evaluator(
        spec,
        output,
        config_path=config_path,
        episodes_per_seed=1,
        max_steps=2,
    )

    expected_metrics = {
        "success_rate",
        "collision_rate",
        "minimum_separation_mean",
        "temporal_conflict_count",
        "energy_proxy",
        "decision_latency",
        "CBF_intervention_rate",
        "CBF_emergency_fallback_rate",
        "CBF_mean_solve_time_seconds",
    }
    assert expected_metrics <= results[0].metrics.keys()
    raw = json.loads((output.root / "raw_results" / "seed_43.jsonl").read_text(encoding="utf-8"))
    assert raw["controller"] == f"{family}_checkpoint"
    assert raw["checkpoint"] == str(checkpoint)
    assert raw["scenario"] == "within_distribution"
    assert raw["cbf"]["decision_count"] == 2

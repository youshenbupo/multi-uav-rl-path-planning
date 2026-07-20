"""Fair dynamic-world contract for the non-graph MLP-MAPPO control arm."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from multiuav.learning.networks import SharedGaussianActor
from multiuav.learning.runner import MAPPOExperiment, load_mappo_experiment_config


def test_mlp_mappo_uses_shared_dynamic_world_and_execution_cbf() -> None:
    """The MLP arm must use no graph encoder while matching the graph arms' world semantics."""
    root = Path(__file__).parents[1]
    config = replace(
        load_mappo_experiment_config(root / "configs/experiments/dynamic_mappo_smoke.yaml"),
        rollout_length=1,
        total_steps=3,
    )
    experiment = MAPPOExperiment(config, device=torch.device("cpu"))

    experiment._step_environments(np.zeros((1, 3, 3), dtype=np.float32))

    assert isinstance(experiment.trainer.actor, SharedGaussianActor)
    assert len(experiment.environments[0].scenario.dynamic_obstacles) == 1
    assert experiment.environments[0].config.communication_delay_steps == 1
    assert experiment.cbf_telemetry.decision_count == 1
    assert experiment.cbf_adapter is not None

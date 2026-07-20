"""Fair-information contracts for dynamic-world graph comparison arms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from multiuav.learning.graph_runner import GraphMAPPOExperiment, load_graph_experiment_config


def test_graph_arms_share_dynamic_communication_but_differ_only_in_information_model() -> None:
    """Raw, predictive, and uncertainty arms must consume the same delivered packets."""
    root = Path(__file__).parents[1]
    base = load_graph_experiment_config(root / "configs/rl/dynamic_graph_baseline.yaml")
    raw = GraphMAPPOExperiment(
        replace(base, graph_mode="distance_graph", rollout_length=1, total_steps=3),
        device=torch.device("cpu"),
    )
    predictive = GraphMAPPOExperiment(
        replace(base, graph_mode="predictive_graph", rollout_length=1, total_steps=3),
        device=torch.device("cpu"),
    )
    complete = GraphMAPPOExperiment(
        replace(base, graph_mode="uncertainty_predictive_graph", rollout_length=1, total_steps=3),
        device=torch.device("cpu"),
    )
    actions = np.tile(np.array([0.5, 0.0, 0.0]), (1, 3, 1))
    for experiment in (raw, predictive, complete):
        experiment._step_environments(actions)

    raw_graph = raw._build_graph()
    predictive_graph = predictive._build_graph()
    complete_graph = complete._build_graph()

    assert raw.environments[0].config == predictive.environments[0].config
    assert predictive.environments[0].config == complete.environments[0].config
    assert raw.cbf_adapter is not None
    assert raw.cbf_telemetry.decision_count == 1
    assert not torch.equal(raw_graph.edge_features, predictive_graph.edge_features)
    assert torch.count_nonzero(predictive_graph.edge_features[..., -1]) == 0
    assert torch.count_nonzero(complete_graph.edge_features[..., -1]) > 0

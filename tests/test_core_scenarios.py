"""Executable semantics for the six fixed paper evaluation scenarios."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

import multiuav.experiments.core_evaluation as core_evaluation
from multiuav.learning.runner import MAPPOExperiment, load_mappo_experiment_config


def test_core_scenarios_are_explicit_environment_perturbations() -> None:
    """Scenario labels must map to distinct communication/dynamic-world conditions."""
    root = Path(__file__).parents[1]
    base = load_mappo_experiment_config(root / "configs/experiments/dynamic_mappo_smoke.yaml")
    apply = getattr(core_evaluation, "apply_core_scenario", None)
    assert apply is not None

    nominal = apply(base, "nominal")
    delayed = apply(base, "delay_only")
    loss = apply(base, "loss_only")
    dynamic = apply(base, "dynamic_only")
    combined = apply(base, "combined")
    ood = apply(base, "ood_communication_obstacle")

    assert nominal.dynamic_obstacle_enabled is False
    assert nominal.communication_delay_steps == 0
    assert nominal.communication_drop_probability == 0.0
    assert delayed.communication_delay_steps > 0 and delayed.communication_drop_probability == 0.0
    assert loss.communication_delay_steps == 0 and loss.communication_drop_probability > 0.0
    assert dynamic.dynamic_obstacle_enabled is True
    assert dynamic.communication_delay_steps == 0 and dynamic.communication_drop_probability == 0.0
    assert combined.dynamic_obstacle_enabled is True
    assert combined.communication_delay_steps > 0 and combined.communication_drop_probability > 0.0
    assert ood.communication_delay_steps > combined.communication_delay_steps
    assert ood.communication_drop_probability > combined.communication_drop_probability
    assert ood.dynamic_obstacle_velocity_scale > combined.dynamic_obstacle_velocity_scale


def test_ood_dynamic_obstacle_speed_perturbation_has_a_valid_twenty_step_trajectory() -> None:
    """The OOD velocity perturbation must be executable, not only a config label."""
    root = Path(__file__).parents[1]
    base = load_mappo_experiment_config(root / "configs/experiments/dynamic_mappo_smoke.yaml")
    ood = core_evaluation.apply_core_scenario(base, "ood_communication_obstacle")

    experiment = MAPPOExperiment(replace(ood, num_envs=1), device=torch.device("cpu"))
    try:
        obstacle = experiment.environments[0].scenario.dynamic_obstacles[0]
        end = obstacle.initial_center + 20 * obstacle.velocity
    finally:
        experiment.close()

    assert obstacle.velocity[1] == 3.0
    assert 0.0 <= obstacle.initial_center[1] <= 100.0
    assert 0.0 <= end[1] <= 100.0

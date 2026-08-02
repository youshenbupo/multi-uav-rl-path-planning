"""Validated core 3-UAV mainline experiment matrix with no shared training artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CoreMethod:
    """One independently trained learned control arm in the paper-facing matrix."""

    name: str
    family: str
    config: Path
    artifact_prefix: str
    graph_mode: str | None


@dataclass(frozen=True)
class CoreProtocol:
    """Fixed seeds, scenarios, and independent arm identities for the first experiment gate."""

    seeds: tuple[int, ...]
    initial_num_uavs: int
    later_num_uavs: tuple[int, ...]
    training_transitions_per_seed: int
    evaluation_episodes_per_seed: int
    scenarios: tuple[str, ...]
    methods: tuple[CoreMethod, ...]


def load_core_protocol(path: Path) -> CoreProtocol:
    """Load the paper-facing protocol and reject missing or unfair comparison declarations."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Core protocol must be a YAML mapping.")
    seeds = _integer_tuple(payload.get("seeds"), "seeds")
    later_num_uavs = _integer_tuple(payload.get("later_num_uavs"), "later_num_uavs")
    scenarios = _string_tuple(payload.get("scenarios"), "scenarios")
    methods_value = payload.get("methods")
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Core protocol requires exactly five unique seeds.")
    if set(scenarios) != {
        "nominal",
        "delay_only",
        "loss_only",
        "dynamic_only",
        "combined",
        "ood_communication_obstacle",
    }:
        raise ValueError("Core protocol must declare the six fixed paper scenarios.")
    if not isinstance(methods_value, list):
        raise ValueError("Core protocol methods must be a list.")
    methods = tuple(_method(item, path.parents[2]) for item in methods_value)
    names = tuple(method.name for method in methods)
    prefixes = tuple(method.artifact_prefix for method in methods)
    if names != (
        "mlp_mappo",
        "raw_graph_mappo",
        "predictive_graph_mappo",
        "uncertainty_predictive_graph_mappo",
    ):
        raise ValueError("Core protocol must declare the four required learned comparison arms.")
    if tuple(method.graph_mode for method in methods) != (
        None,
        "mappo",
        "predictive_graph",
        "uncertainty_predictive_graph",
    ):
        raise ValueError("Core protocol graph modes must match the audited comparison arms.")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("Every core arm must have its own artifact prefix.")
    initial_num_uavs = payload.get("initial_num_uavs")
    transitions = payload.get("training_transitions_per_seed")
    evaluation_episodes = payload.get("evaluation_episodes_per_seed")
    if initial_num_uavs != 3 or later_num_uavs != (5, 8):
        raise ValueError("Core protocol must begin at 3 UAVs then scale to 5 and 8.")
    if not isinstance(transitions, int) or transitions < 1:
        raise ValueError("training_transitions_per_seed must be positive.")
    if not isinstance(evaluation_episodes, int) or evaluation_episodes < 1:
        raise ValueError("evaluation_episodes_per_seed must be positive.")
    return CoreProtocol(
        seeds=seeds,
        initial_num_uavs=initial_num_uavs,
        later_num_uavs=later_num_uavs,
        training_transitions_per_seed=transitions,
        evaluation_episodes_per_seed=evaluation_episodes,
        scenarios=scenarios,
        methods=methods,
    )


def _method(value: object, root: Path) -> CoreMethod:
    if not isinstance(value, Mapping):
        raise ValueError("Each core method must be a mapping.")
    name = value.get("name")
    family = value.get("family")
    config_value = value.get("config")
    prefix = value.get("artifact_prefix")
    graph_mode = value.get("graph_mode")
    if not isinstance(name, str) or not name:
        raise ValueError("Core method name is required.")
    if not isinstance(family, str) or not family:
        raise ValueError("Core method family is required.")
    if not isinstance(config_value, str) or not config_value:
        raise ValueError("Core method config is required.")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Core method artifact_prefix is required.")
    if value.get("use_cbf") is not True:
        raise ValueError("Every core comparison arm must use the shared CBF.")
    if family not in {"mappo", "graph_mappo"}:
        raise ValueError("Core method family is unsupported.")
    if family == "mappo" and graph_mode is not None:
        raise ValueError("MLP-MAPPO must not declare a graph mode.")
    if family == "graph_mappo" and graph_mode not in {
        "mappo",
        "distance_graph",
        "predictive_graph",
        "uncertainty_predictive_graph",
    }:
        raise ValueError("Graph comparison arm requires a supported graph mode.")
    config = root / config_value
    if not config.is_file():
        raise ValueError("Core method configuration file is missing.")
    selected_graph_mode = graph_mode if isinstance(graph_mode, str) else None
    return CoreMethod(name, family, config, prefix, selected_graph_mode)


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{name} must be a list of integers.")
    return tuple(value)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a list of nonempty strings.")
    return tuple(value)

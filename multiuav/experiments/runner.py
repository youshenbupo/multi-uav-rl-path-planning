"""Deterministic dynamic-world evaluation shared by every Phase-14 command."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from multiuav.core.models import DynamicCylinder, Scenario
from multiuav.envs.multi_uav_env import EnvironmentConfig, MultiUAVParallelEnv
from multiuav.experiments.metrics import SeedResult
from multiuav.experiments.outputs import ExperimentOutput
from multiuav.experiments.spec import ExperimentSpec
from multiuav.learning.conflict_graph import ConflictGraphBuilder, GraphBuildConfig
from multiuav.learning.graph_runner import build_graph_from_environments, make_graph_scenario
from multiuav.safety import CBFConfig, NormalizedActionCBFAdapter, OSQPSafetyFilter


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested neural-inference device without silently ignoring CUDA."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def create_experiment_output(spec: ExperimentSpec, output_root: Path) -> ExperimentOutput:
    """Persist the exact dynamic-world semantics and resolved neural device before a run."""
    device = resolve_device(spec.device)
    configuration = dynamic_evaluation_config()
    metadata: dict[str, Any] = {
        "resolved_device": str(device),
        "controller": "goal_directed_semantic_smoke",
        "dynamic_obstacles": [
            {
                "identifier": obstacle.identifier,
                "initial_center": obstacle.initial_center.tolist(),
                "velocity": obstacle.velocity.tolist(),
                "radius": obstacle.radius,
                "height": obstacle.height,
            }
            for obstacle in dynamic_evaluation_scenario(spec.num_uavs).dynamic_obstacles
        ],
        "communication": {
            "enabled": configuration.communication_enabled,
            "range": configuration.communication_range,
            "delay_steps": configuration.communication_delay_steps,
            "drop_probability": configuration.communication_drop_probability,
            "max_staleness_steps": configuration.communication_max_staleness_steps,
        },
        "metric_limitations": {
            "expert_gap": "unavailable without a matched CA-HGALO expert reference rollout",
            "generalization_gap": "unavailable without paired within/out-of-distribution runs",
        },
    }
    return ExperimentOutput.create(output_root, spec, metadata=metadata)


def dynamic_evaluation_scenario(num_uavs: int) -> Scenario:
    """Create the standard evaluation world with one deterministic crossing obstacle."""
    base = make_graph_scenario(num_uavs=num_uavs)
    return replace(
        base,
        dynamic_obstacles=(
            DynamicCylinder(
                identifier="crossing_0",
                initial_center=np.array([22.0, 50.0, 30.0]),
                velocity=np.array([0.0, 2.0, 0.0]),
                radius=3.0,
                height=20.0,
            ),
        ),
    )


def dynamic_evaluation_config() -> EnvironmentConfig:
    """Use fixed dynamic-obstacle and delayed/lossy communication semantics."""
    return EnvironmentConfig(
        dt=1.0,
        max_steps=20,
        max_horizontal_speed=8.0,
        max_vertical_speed=6.0,
        normalize_actions=True,
        goal_radius=5.0,
        collision_distance=6.0,
        severe_clearance_shortfall=8.0,
        severe_threat_penetration=5.0,
        max_neighbors=3,
        max_dynamic_obstacles=1,
        communication_enabled=True,
        communication_range=45.0,
        communication_delay_steps=1,
        communication_drop_probability=0.10,
        communication_max_staleness_steps=3,
    )


def evaluate_goal_controller(
    spec: ExperimentSpec,
    output: ExperimentOutput,
    *,
    episodes_per_seed: int = 4,
    max_steps: int | None = None,
) -> list[SeedResult]:
    """Run an auditable deterministic controller through graph, communication, and CBF paths.

    This is a semantic integration evaluator, not a replacement for a trained checkpoint.
    Learned-policy checkpoint evaluation is intentionally kept separate until a compatible
    staged checkpoint has been produced by the training command.
    """
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive.")
    device = resolve_device(spec.device)
    scenario = dynamic_evaluation_scenario(spec.num_uavs)
    configuration = dynamic_evaluation_config()
    if max_steps is not None:
        configuration = replace(configuration, max_steps=max_steps)
    graph_builder = ConflictGraphBuilder(
        GraphBuildConfig(
            communication_radius=configuration.communication_range,
            risk_distance=scenario.safe_separation * 1.5,
            prediction_horizon=5.0,
            top_k_neighbors=min(3, spec.num_uavs - 1),
            current_distance_edges=True,
            predicted_conflict_edges=True,
            self_loops=True,
        )
    )
    results: list[SeedResult] = []
    for seed in spec.seeds:
        records = [
            _run_episode(
                scenario,
                configuration,
                graph_builder,
                device,
                seed=seed + episode,
                use_graph=spec.use_graph,
                use_hierarchy=spec.use_hierarchy,
                use_cbf=spec.use_cbf,
            )
            for episode in range(episodes_per_seed)
        ]
        for episode, record in enumerate(records):
            output.write_raw_result(seed=seed, episode=episode, result=record)
        results.append(SeedResult(seed=seed, metrics=_aggregate_episode_records(records)))
    output.write_summary(results)
    return results


def _run_episode(
    scenario: Scenario,
    configuration: EnvironmentConfig,
    graph_builder: ConflictGraphBuilder,
    device: torch.device,
    *,
    seed: int,
    use_graph: bool,
    use_hierarchy: bool,
    use_cbf: bool,
) -> dict[str, float | int | str]:
    environment = MultiUAVParallelEnv(scenario, configuration)
    environment.reset(seed=seed)
    adapter = (
        NormalizedActionCBFAdapter(OSQPSafetyFilter(CBFConfig(max_solve_time_seconds=0.1)))
        if use_cbf
        else None
    )
    path_length = 0.0
    energy_proxy = 0.0
    minimum_separations: list[float] = []
    decision_latencies: list[float] = []
    corrections: list[float] = []
    emergency_count = 0
    conflict_count = 0
    terrain_violation = 0
    threat_violation = 0
    reason = "max_steps"
    while environment.agents:
        started = time.perf_counter()
        proposed = _goal_actions(environment, graph_builder, device, use_graph, use_hierarchy)
        if adapter is not None:
            safe_actions, decision = adapter.filter_normalized(environment._snapshot(), proposed)
            corrections.append(decision.intervention_norm)
            emergency_count += int(decision.emergency_fallback_used)
        else:
            safe_actions = proposed
        decision_latencies.append(time.perf_counter() - started)
        previous = environment.positions.copy()
        _, _, terminations, truncations, infos = environment.step(
            {
                agent: safe_actions[environment.agent_name_mapping[agent]].astype(np.float32)
                for agent in environment.agents
            }
        )
        path_length += float(np.linalg.norm(environment.positions - previous, axis=1).sum())
        energy_proxy += float(np.square(safe_actions).sum())
        separation = environment.minimum_separation
        minimum_separations.append(separation)
        conflict_count += int(separation < scenario.safe_separation)
        reason = str(infos["uav_0"]["termination_reason"])
        costs = infos["uav_0"]["safety_costs"]
        terrain_violation += int(costs["terrain_violation_cost"] > 0.0)
        threat_violation += int(costs["threat_violation_cost"] > 0.0)
        if all(terminations.values()) or all(truncations.values()):
            break
    return {
        "termination_reason": reason,
        "success": float(reason == "all_arrived"),
        "collision": float(reason == "collision"),
        "terrain_violation": float(reason == "terrain_violation" or terrain_violation > 0),
        "threat_violation": float(reason == "threat_violation" or threat_violation > 0),
        "path_length": path_length,
        "mission_time": float(environment.step_count * configuration.dt),
        "minimum_separation": float(min(minimum_separations, default=np.inf)),
        "temporal_conflict_count": conflict_count,
        "energy_proxy": energy_proxy,
        "decision_latency": float(np.mean(decision_latencies, dtype=float)),
        "cbf_intervention": float(np.count_nonzero(corrections)),
        "cbf_mean_correction": float(np.mean(corrections, dtype=float)) if corrections else 0.0,
        "cbf_emergency_count": emergency_count,
        "dynamic_obstacle_count": len(scenario.dynamic_obstacles),
        "communication_delay_steps": configuration.communication_delay_steps,
        "communication_drop_probability": configuration.communication_drop_probability,
    }


def _goal_actions(
    environment: MultiUAVParallelEnv,
    graph_builder: ConflictGraphBuilder,
    device: torch.device,
    use_graph: bool,
    use_hierarchy: bool,
) -> np.ndarray:
    goals = np.asarray([mission.goal for mission in environment.scenario.missions], dtype=float)
    direction = goals - environment.positions
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    actions = direction / np.maximum(norms, 1e-8)
    actions[~environment.active_mask] = 0.0
    if not use_graph:
        return np.asarray(actions, dtype=float)
    graph = build_graph_from_environments(graph_builder, [environment], device=device)
    adjacency = graph.adjacency[0].detach().cpu().numpy().astype(bool)
    np.fill_diagonal(adjacency, False)
    if use_hierarchy:
        for index in range(len(actions)):
            if adjacency[index, :index].any():
                actions[index] *= 0.25
    return np.asarray(actions, dtype=float)


def _aggregate_episode_records(records: list[dict[str, float | int | str]]) -> dict[str, float]:
    """Map raw retained records to the required finite cross-seed summary metrics."""
    def mean(name: str) -> float:
        return float(np.mean([float(record[name]) for record in records], dtype=float))

    separations = np.asarray([float(record["minimum_separation"]) for record in records])
    return {
        "success_rate": mean("success"),
        "collision_rate": mean("collision"),
        "terrain_violation_rate": mean("terrain_violation"),
        "threat_violation_rate": mean("threat_violation"),
        "mean_path_length": mean("path_length"),
        "mission_time": mean("mission_time"),
        "minimum_separation_mean": float(np.mean(separations)),
        "minimum_separation_5th_percentile": float(np.percentile(separations, 5.0)),
        "temporal_conflict_count": mean("temporal_conflict_count"),
        "energy_proxy": mean("energy_proxy"),
        "decision_latency": mean("decision_latency"),
        "CBF_intervention_rate": float(
            np.mean([float(record["cbf_intervention"]) > 0 for record in records])
        ),
        "CBF_mean_correction": mean("cbf_mean_correction"),
        "CBF_emergency_count": mean("cbf_emergency_count"),
    }

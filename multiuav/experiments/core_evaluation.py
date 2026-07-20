"""Uniform per-episode evaluation for independently trained core comparison arms.

The evaluator deliberately loads the requested learned checkpoint and never
substitutes a goal controller.  MLP-MAPPO and GraphMAPPO therefore produce the
same raw-record and seed-summary metric contracts before their results are
compared in the paper protocol.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from multiuav.envs.multi_uav_env import MultiUAVParallelEnv
from multiuav.experiments.metrics import SeedResult
from multiuav.experiments.outputs import ExperimentOutput
from multiuav.experiments.runner import _aggregate_episode_records
from multiuav.experiments.spec import ExperimentSpec
from multiuav.safety import SafetyFilterDecision, SafetyFilterTelemetry

_ActionFunction = Callable[[dict[str, np.ndarray], MultiUAVParallelEnv], np.ndarray]


def create_checkpoint_evaluation_output(
    base_directory: Path,
    spec: ExperimentSpec,
    *,
    controller: str,
    config_path: Path,
) -> ExperimentOutput:
    """Create an auditable learned-checkpoint result directory before evaluation starts."""
    if controller not in {"mappo_checkpoint", "graph_mappo_checkpoint"}:
        raise ValueError("controller must identify a supported learned checkpoint family.")
    return ExperimentOutput.create(
        base_directory,
        spec,
        metadata={
            "controller": controller,
            "checkpoint": str(_checkpoint(spec)),
            "training_config": str(config_path),
            "device": spec.device,
            "num_uavs": spec.num_uavs,
            "scenario": spec.scenario,
            "actor_information": "own_truth_and_delivered_packets_only",
            "cbf_execution": "cpu_osqp_when_enabled",
        },
    )


def apply_core_scenario(config: Any, scenario: str) -> Any:
    """Apply one fixed paper scenario without changing the learned-policy architecture."""
    combined = "combined" if scenario == "within_distribution" else scenario
    common = {"communication_enabled": True}
    if combined == "nominal":
        return replace(
            config,
            **common,
            communication_delay_steps=0,
            communication_drop_probability=0.0,
            dynamic_obstacle_enabled=False,
        )
    if combined == "delay_only":
        return replace(
            config,
            **common,
            communication_delay_steps=max(config.communication_delay_steps, 1),
            communication_drop_probability=0.0,
            dynamic_obstacle_enabled=False,
        )
    if combined == "loss_only":
        return replace(
            config,
            **common,
            communication_delay_steps=0,
            communication_drop_probability=max(config.communication_drop_probability, 0.1),
            dynamic_obstacle_enabled=False,
        )
    if combined == "dynamic_only":
        return replace(
            config,
            **common,
            communication_delay_steps=0,
            communication_drop_probability=0.0,
            dynamic_obstacle_enabled=True,
        )
    if combined == "combined":
        return replace(
            config,
            **common,
            communication_delay_steps=max(config.communication_delay_steps, 1),
            communication_drop_probability=max(config.communication_drop_probability, 0.1),
            dynamic_obstacle_enabled=True,
        )
    if combined == "ood_communication_obstacle":
        return replace(
            config,
            **common,
            communication_delay_steps=max(config.communication_delay_steps + 2, 2),
            communication_drop_probability=min(
                max(config.communication_drop_probability + 0.2, 0.3), 0.8
            ),
            communication_max_staleness_steps=max(config.communication_max_staleness_steps + 2, 2),
            communication_uncertainty_growth_per_step=(
                max(config.communication_uncertainty_growth_per_step, 1.0) * 1.5
            ),
            dynamic_obstacle_enabled=True,
            dynamic_obstacle_velocity_scale=max(config.dynamic_obstacle_velocity_scale, 1.0) * 1.5,
        )
    raise ValueError(f"Unknown core evaluation scenario: {scenario}")


def evaluate_mappo_checkpoint(
    spec: ExperimentSpec,
    output: ExperimentOutput,
    *,
    config_path: Path,
    episodes_per_seed: int,
    max_steps: int | None = None,
) -> list[SeedResult]:
    """Evaluate a saved non-graph MAPPO actor and retain every completed episode."""
    from multiuav.learning.mappo import load_checkpoint
    from multiuav.learning.runner import MAPPOExperiment, load_mappo_experiment_config

    _validate_request(spec, output, episodes_per_seed)
    base_config = load_mappo_experiment_config(config_path)

    def make_experiment(seed: int) -> Any:
        return MAPPOExperiment(
            replace(
                apply_core_scenario(base_config, spec.scenario),
                seed=seed,
                num_envs=1,
                num_uavs=spec.num_uavs,
                cbf_enabled=spec.use_cbf,
            ),
            device=torch.device(spec.device),
        )

    def action_for(experiment: Any) -> _ActionFunction:
        def choose(
            observations: dict[str, np.ndarray], environment: MultiUAVParallelEnv
        ) -> np.ndarray:
            current = experiment._observations_for_environment(observations, environment)
            with torch.no_grad():
                actions = (
                    experiment.trainer.actor.act(
                        experiment.trainer.observation_normalizer.normalize(current),
                        deterministic=True,
                    )
                    .cpu()
                    .numpy()
                )
            return np.asarray(actions, dtype=np.float32)

        return choose

    return _evaluate_checkpoint(
        spec,
        output,
        make_experiment=make_experiment,
        load=lambda experiment: load_checkpoint(
            _checkpoint(spec), experiment.trainer, map_location=torch.device(spec.device)
        ),
        action_for=action_for,
        controller="mappo_checkpoint",
        episodes_per_seed=episodes_per_seed,
        max_steps=max_steps,
    )


def evaluate_graph_checkpoint(
    spec: ExperimentSpec,
    output: ExperimentOutput,
    *,
    config_path: Path,
    episodes_per_seed: int,
    max_steps: int | None = None,
) -> list[SeedResult]:
    """Evaluate a saved graph-MAPPO actor and retain every completed episode."""
    from multiuav.learning.graph_mappo import load_graph_checkpoint
    from multiuav.learning.graph_runner import GraphMAPPOExperiment, load_graph_experiment_config

    _validate_request(spec, output, episodes_per_seed)
    base_config = load_graph_experiment_config(config_path)

    def make_experiment(seed: int) -> Any:
        return GraphMAPPOExperiment(
            replace(
                apply_core_scenario(base_config, spec.scenario),
                seed=seed,
                num_envs=1,
                num_uavs=spec.num_uavs,
                cbf_enabled=spec.use_cbf,
            ),
            device=torch.device(spec.device),
        )

    def action_for(experiment: Any) -> _ActionFunction:
        def choose(
            observations: dict[str, np.ndarray], environment: MultiUAVParallelEnv
        ) -> np.ndarray:
            node_features = experiment._node_features_for([(observations, environment)])
            graph = experiment._build_graph_for([environment])
            with torch.no_grad():
                actions, _, _ = experiment.trainer.actor.sample(
                    node_features,
                    graph.edge_features,
                    graph.adjacency,
                    graph.node_mask,
                    deterministic=True,
                )
            return np.asarray(actions[0].cpu().numpy(), dtype=np.float32)

        return choose

    return _evaluate_checkpoint(
        spec,
        output,
        make_experiment=make_experiment,
        load=lambda experiment: load_graph_checkpoint(
            _checkpoint(spec), experiment.trainer, map_location=torch.device(spec.device)
        ),
        action_for=action_for,
        controller="graph_mappo_checkpoint",
        episodes_per_seed=episodes_per_seed,
        max_steps=max_steps,
    )


def _evaluate_checkpoint(
    spec: ExperimentSpec,
    output: ExperimentOutput,
    *,
    make_experiment: Callable[[int], Any],
    load: Callable[[Any], int],
    action_for: Callable[[Any], _ActionFunction],
    controller: str,
    episodes_per_seed: int,
    max_steps: int | None,
) -> list[SeedResult]:
    """Run a single learned-policy family through the common JSONL metric path."""
    seed_results: list[SeedResult] = []
    runtime_seeds: list[dict[str, object]] = []
    for seed in spec.seeds:
        experiment = make_experiment(seed)
        try:
            completed_step = load(experiment)
            experiment.trainer.actor.eval()
            records: list[dict[str, float | int | str]] = []
            episode_cbf: list[dict[str, object]] = []
            choose_action = action_for(experiment)
            for episode in range(episodes_per_seed):
                record, cbf = _run_learned_episode(
                    experiment,
                    choose_action,
                    seed=seed + 10_000 + episode,
                    max_steps=max_steps,
                )
                records.append(record)
                raw_record: dict[str, object] = {
                    **record,
                    "controller": controller,
                    "checkpoint": str(_checkpoint(spec)),
                    "checkpoint_completed_step": completed_step,
                    "scenario": spec.scenario,
                    "cbf": cbf,
                }
                output.write_raw_result(seed=seed, episode=episode, result=raw_record)
                episode_cbf.append(cbf)
            seed_results.append(SeedResult(seed=seed, metrics=_aggregate_episode_records(records)))
            runtime_seeds.append(
                {"seed": seed, "checkpoint_step": completed_step, "episodes": episode_cbf}
            )
        finally:
            experiment.close()
    output.write_summary(seed_results)
    output.write_runtime_telemetry(
        {
            "controller": controller,
            "checkpoint": str(_checkpoint(spec)),
            "per_seed": runtime_seeds,
        }
    )
    return seed_results


def _run_learned_episode(
    experiment: Any,
    choose_action: _ActionFunction,
    *,
    seed: int,
    max_steps: int | None,
) -> tuple[dict[str, float | int | str], dict[str, object]]:
    """Run one deterministic learned-policy episode with isolated CPU CBF telemetry."""
    configuration = experiment.environments[0].config
    if max_steps is not None:
        configuration = replace(configuration, max_steps=max_steps)
    environment = MultiUAVParallelEnv(experiment.environments[0].scenario, configuration)
    observations, _ = environment.reset(seed=seed)
    adapter = experiment._make_cbf_adapter() if experiment.config.cbf_enabled else None
    telemetry = SafetyFilterTelemetry()
    total_return = 0.0
    path_length = 0.0
    energy_proxy = 0.0
    minimum_separations: list[float] = []
    decision_latencies: list[float] = []
    corrections: list[float] = []
    conflict_count = 0
    terrain_violation = 0
    threat_violation = 0
    reason = "max_steps"
    while environment.agents:
        started = time.perf_counter()
        proposed = choose_action(observations, environment)
        safe_actions, decision = _filter_action(adapter, environment, proposed)
        if decision is not None:
            corrections.append(decision.intervention_norm)
            telemetry.record(decision, context=_cbf_context(environment, decision))
        decision_latencies.append(time.perf_counter() - started)
        previous = environment.positions.copy()
        observations, rewards, terminations, truncations, infos = environment.step(
            {
                agent: safe_actions[environment.agent_name_mapping[agent]].astype(np.float32)
                for agent in environment.possible_agents
            }
        )
        total_return += float(sum(rewards.values()))
        path_length += float(np.linalg.norm(environment.positions - previous, axis=1).sum())
        energy_proxy += float(np.square(safe_actions).sum())
        separation = float(environment.minimum_separation)
        minimum_separations.append(separation)
        conflict_count += int(separation < environment.scenario.safe_separation)
        reason = str(infos["uav_0"]["termination_reason"])
        costs = infos["uav_0"]["safety_costs"]
        terrain_violation += int(costs["terrain_violation_cost"] > 0.0)
        threat_violation += int(costs["threat_violation_cost"] > 0.0)
        if all(terminations.values()) or all(truncations.values()):
            break
    cbf = telemetry.as_dict()
    return (
        {
            "termination_reason": reason,
            "success": float(reason == "all_arrived"),
            "collision": float(reason == "collision"),
            "terrain_violation": float(reason == "terrain_violation" or terrain_violation > 0),
            "threat_violation": float(reason == "threat_violation" or threat_violation > 0),
            "episode_return": total_return,
            "path_length": path_length,
            "mission_time": float(environment.step_count * configuration.dt),
            "minimum_separation": float(min(minimum_separations, default=0.0)),
            "temporal_conflict_count": conflict_count,
            "energy_proxy": energy_proxy,
            "decision_latency": float(np.mean(decision_latencies, dtype=float)),
            "cbf_intervention": float(telemetry.intervention_count),
            "cbf_mean_correction": float(np.mean(corrections, dtype=float)) if corrections else 0.0,
            "cbf_emergency_count": telemetry.emergency_fallback_count,
            "cbf_decision_count": telemetry.decision_count,
            "cbf_intervention_rate": (
                telemetry.intervention_count / telemetry.decision_count
                if telemetry.decision_count
                else 0.0
            ),
            "cbf_emergency_fallback_rate": (
                telemetry.emergency_fallback_count / telemetry.decision_count
                if telemetry.decision_count
                else 0.0
            ),
            "cbf_mean_solve_time_seconds": (
                telemetry.total_solve_time / telemetry.decision_count
                if telemetry.decision_count
                else 0.0
            ),
            "cbf_total_solve_time_seconds": float(telemetry.total_solve_time),
            "dynamic_obstacle_count": len(environment.scenario.dynamic_obstacles),
            "communication_delay_steps": configuration.communication_delay_steps,
            "communication_drop_probability": configuration.communication_drop_probability,
        },
        cbf,
    )


def _filter_action(
    adapter: Any | None, environment: MultiUAVParallelEnv, proposed: np.ndarray
) -> tuple[np.ndarray, SafetyFilterDecision | None]:
    """Apply the common execution shield without backpropagating through OSQP."""
    if adapter is None:
        return np.asarray(proposed, dtype=float), None
    safe, decision = adapter.filter_normalized(environment._snapshot(), proposed)
    return safe, decision


def _cbf_context(
    environment: MultiUAVParallelEnv, decision: SafetyFilterDecision
) -> dict[str, object]:
    """Keep sufficient failure context for later CBF diagnosis without hidden actor inputs."""
    snapshot = environment._snapshot()
    return {
        "environment_step": environment.step_count,
        "positions": environment.positions.tolist(),
        "requested_velocities": decision.u_rl.tolist(),
        "velocities": snapshot.velocities.tolist(),
        "knowledge_valid": [state.valid.tolist() for state in snapshot.knowledge_states],
        "knowledge_uncertainty": [
            state.position_uncertainty.tolist() for state in snapshot.knowledge_states
        ],
        "dynamic_obstacle_centers": (
            snapshot.dynamic_world.centers.tolist() if snapshot.dynamic_world is not None else []
        ),
    }


def _validate_request(
    spec: ExperimentSpec, output: ExperimentOutput, episodes_per_seed: int
) -> None:
    if output.spec != spec:
        raise ValueError("Evaluation output must be created from the supplied experiment spec.")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive.")
    _checkpoint(spec)


def _checkpoint(spec: ExperimentSpec) -> Path:
    if spec.checkpoint is None:
        raise ValueError("Learned-policy checkpoint evaluation requires spec.checkpoint.")
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {spec.checkpoint}")
    return spec.checkpoint

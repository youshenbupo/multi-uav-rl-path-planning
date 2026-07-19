"""Closed-loop metric contracts for behavior-cloning evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch

from multiuav.data.bc_dataset import load_bc_scenario
from multiuav.envs.multi_uav_env import EnvironmentConfig, MultiUAVParallelEnv
from multiuav.learning.behavior_cloning import ExpertLowLevelPolicy
from multiuav.learning.conflict_graph import ConflictGraphBuilder, GraphBuildConfig


@dataclass(frozen=True)
class BCRolloutMetrics:
    """Required aggregate BC closed-loop metrics over held-out expert missions."""

    arrival_rate: float
    collision_rate: float
    minimum_separation: float
    path_length: float
    expert_path_deviation: float
    episode_count: int


def evaluate_bc_low_checkpoint(
    checkpoint: Path,
    episodes: list[tuple[Path, str]],
    *,
    device: torch.device,
) -> BCRolloutMetrics:
    """Run deterministic low-BC actions in reconstructed held-out environments."""
    if not episodes:
        raise ValueError("Closed-loop BC evaluation requires at least one held-out episode.")
    payload: dict[str, Any] = torch.load(checkpoint, map_location=device, weights_only=False)
    config: dict[str, Any] = payload["config"]
    stats = _load_stats(checkpoint.parent / "normalization_stats.json", device)
    policy = ExpertLowLevelPolicy(
        node_feature_dim=len(stats["mean"]),
        edge_feature_dim=15,
        embedding_dim=int(config["embedding_dim"]),
        num_heads=int(config["graph_heads"]),
        num_layers=int(config["graph_layers"]),
    ).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    graph_builder = ConflictGraphBuilder(_graph_config())
    results = [
        _rollout_episode(
            policy,
            graph_builder,
            stats,
            shard=shard,
            episode_id=episode_id,
            max_neighbors=int(config["max_neighbors"]),
            max_horizontal_speed=float(config["max_horizontal_speed"]),
            max_vertical_speed=float(config["max_vertical_speed"]),
            device=device,
        )
        for shard, episode_id in episodes
    ]
    return BCRolloutMetrics(
        arrival_rate=float(np.mean([result["arrived"] for result in results])),
        collision_rate=float(np.mean([result["collision"] for result in results])),
        minimum_separation=float(np.min([result["minimum_separation"] for result in results])),
        path_length=float(np.mean([result["path_length"] for result in results])),
        expert_path_deviation=float(
            np.mean([result["expert_path_deviation"] for result in results])
        ),
        episode_count=len(results),
    )


def _rollout_episode(
    policy: ExpertLowLevelPolicy,
    graph_builder: ConflictGraphBuilder,
    stats: dict[str, torch.Tensor],
    *,
    shard: Path,
    episode_id: str,
    max_neighbors: int,
    max_horizontal_speed: float,
    max_vertical_speed: float,
    device: torch.device,
) -> dict[str, float | bool]:
    scenario = load_bc_scenario(shard, episode_id)
    with h5py.File(shard, "r") as handle:
        expert_positions = np.asarray(handle[f"episodes/{episode_id}/positions"], dtype=float)
    environment = MultiUAVParallelEnv(
        scenario,
        EnvironmentConfig(
            dt=1.0,
            max_steps=expert_positions.shape[0],
            max_horizontal_speed=max_horizontal_speed,
            max_vertical_speed=max_vertical_speed,
            normalize_actions=True,
            max_neighbors=max_neighbors,
        ),
    )
    observations, _ = environment.reset(seed=0)
    positions = [environment.positions.copy()]
    terminal_reason = "none"
    while environment.agents:
        actions = _policy_actions(policy, graph_builder, environment, observations, stats, device)
        observations, _, terminated, truncated, infos = environment.step(actions)
        positions.append(environment.positions.copy())
        terminal_reason = next(iter(infos.values()))["termination_reason"]
        if all(terminated.values()) or all(truncated.values()):
            break
    path = np.asarray(positions)
    expert_resampled = _resample_expert(expert_positions, path.shape[0])
    path_length = np.linalg.norm(np.diff(path, axis=0), axis=2).sum(axis=0).mean()
    deviation = np.linalg.norm(path - expert_resampled, axis=2).mean()
    return {
        "arrived": terminal_reason == "all_arrived",
        "collision": terminal_reason == "collision",
        "minimum_separation": float(environment.minimum_separation),
        "path_length": float(path_length),
        "expert_path_deviation": float(deviation),
    }


def _policy_actions(
    policy: ExpertLowLevelPolicy,
    graph_builder: ConflictGraphBuilder,
    environment: MultiUAVParallelEnv,
    observations: dict[str, np.ndarray],
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, np.ndarray]:
    names = environment.possible_agents
    nodes = torch.as_tensor(
        np.asarray([observations[name] for name in names]), device=device
    ).unsqueeze(0)
    normalized_nodes = (nodes - stats["mean"]) / stats["std"]
    positions = torch.as_tensor(
        environment.positions, dtype=torch.float32, device=device
    ).unsqueeze(0)
    velocities = torch.as_tensor(
        environment.velocities, dtype=torch.float32, device=device
    ).unsqueeze(0)
    goals = torch.as_tensor(
        np.asarray([mission.goal for mission in environment.scenario.missions]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    active = torch.as_tensor(environment.active_mask, dtype=torch.bool, device=device).unsqueeze(0)
    graph = graph_builder.build(
        positions=positions, velocities=velocities, goals=goals, active_mask=active
    )
    commands = torch.zeros((*normalized_nodes.shape[:2], 6), device=device)
    commands[..., 0] = 1.0
    with torch.no_grad():
        actions = policy(
            normalized_nodes, graph.edge_features, graph.adjacency, active, commands
        )[0]
    return {name: actions[environment.agent_name_mapping[name]].cpu().numpy() for name in names}


def _load_stats(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mean": torch.as_tensor(payload["node_mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(payload["node_std"], dtype=torch.float32, device=device),
    }


def _graph_config() -> GraphBuildConfig:
    return GraphBuildConfig(100.0, 70.0, 5.0, 3, True, True, True)


def _resample_expert(positions: np.ndarray, count: int) -> np.ndarray:
    indices = np.linspace(0, positions.shape[0] - 1, count)
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    alpha = (indices - lower)[:, None, None]
    return cast(np.ndarray, positions[lower] * (1.0 - alpha) + positions[upper] * alpha)

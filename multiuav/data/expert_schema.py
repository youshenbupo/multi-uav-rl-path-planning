"""Versioned HDF5 storage schema for variable-length multi-UAV expert episodes."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from multiuav.data.expert_dataset import COST_COMPONENT_NAMES, ExpertEpisode

SCHEMA_VERSION = "multiuav_expert_episode_v1"


def write_episode(path: Path, episode_id: str, episode: ExpertEpisode) -> None:
    """Create or replace one HDF5 episode group without fixed edge padding."""
    if not episode_id or "/" in episode_id:
        raise ValueError("episode_id must be a non-empty HDF5 path component.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        episodes = handle.require_group("episodes")
        if episode_id in episodes:
            del episodes[episode_id]
        group = episodes.create_group(episode_id)
        _write_episode_group(group, episode)


def _write_episode_group(group: h5py.Group, episode: ExpertEpisode) -> None:
    group.attrs["scenario_id"] = episode.scenario_id
    group.attrs["seed"] = episode.seed
    group.attrs["num_uavs"] = len(episode.repaired_trajectories)
    group.attrs["dt"] = episode.dt
    group.attrs["nominal_speed"] = episode.nominal_speed
    group.attrs["max_speed"] = episode.max_speed
    group.attrs["source"] = episode.source
    group.attrs["trajectory_layout"] = "flattened_values_with_offsets"
    group.attrs["dynamic_graph_layout"] = "one_group_per_time_step; no edge padding"
    group.attrs["high_level_action_names"] = json.dumps(
        {"continue": 0, "wait_yield": 1, "climb": 2, "descend": 3, "shift_left": 4,
         "shift_right": 5, "local_subgoal": 6, "assigned_delay": 7, "conflict_priority": 8},
        sort_keys=True,
    )
    _write_scenario(group, episode)
    _write_trajectories(group, "raw_waypoints", episode.raw_trajectories)
    _write_trajectories(group, "repaired_waypoints", episode.repaired_trajectories)
    _write_dataset(group, "positions", episode.positions)
    _write_dataset(group, "velocities", episode.velocities)
    _write_dataset(group, "timestamps", episode.timestamps)
    _write_dataset(group, "active_mask", episode.active_mask)
    _write_dataset(group, "start_delays", episode.start_delays)
    _write_dataset(group, "low_level_actions", episode.low_level_actions)
    _write_dataset(group, "low_level_action_mask", episode.low_level_action_mask)
    _write_dataset(group, "low_level_clip_ratio", episode.low_level_clip_ratio)
    _write_dataset(group, "high_level_actions", episode.high_level_actions)
    _write_dataset(group, "high_level_action_mask", episode.high_level_action_mask)
    if episode.convergence_history is not None:
        _write_dataset(group, "convergence_history", episode.convergence_history)
    cost_values = np.asarray(
        [getattr(episode.evaluation.cost_breakdown, name) for name in COST_COMPONENT_NAMES],
        dtype=float,
    )
    cost = _write_dataset(group, "cost_breakdown", cost_values)
    cost.attrs["component_names"] = json.dumps(COST_COMPONENT_NAMES)
    _write_dataset(
        group, "strict_success", np.asarray(episode.evaluation.strict_success, dtype=bool)
    )
    _write_dataset(
        group, "minimum_separation", np.asarray(episode.evaluation.minimum_separation, dtype=float)
    )
    _write_dataset(
        group,
        "temporal_conflict_count",
        np.asarray(episode.evaluation.temporal_conflict_count, dtype=np.int64),
    )
    _write_dynamic_graph(group, episode)
    _write_log(group, "schedule_log", episode.schedule_log)
    _write_log(group, "spatial_repair_log", episode.spatial_repair_log)


def _write_scenario(group: h5py.Group, episode: ExpertEpisode) -> None:
    scenario_group = group.create_group("scenario")
    _write_dataset(scenario_group, "terrain_x", episode.scenario.terrain.x_grid)
    _write_dataset(scenario_group, "terrain_y", episode.scenario.terrain.y_grid)
    _write_dataset(scenario_group, "terrain_heights", episode.scenario.terrain.heights)
    threats = np.asarray(
        [
            [threat.center_x, threat.center_y, threat.radius, threat.height]
            for threat in episode.scenario.threats
        ],
        dtype=float,
    ).reshape(-1, 4)
    _write_dataset(scenario_group, "threats", threats)
    _write_dataset(
        scenario_group,
        "starts",
        np.asarray([mission.start for mission in episode.scenario.missions], dtype=float),
    )
    _write_dataset(
        scenario_group,
        "goals",
        np.asarray([mission.goal for mission in episode.scenario.missions], dtype=float),
    )
    scenario_group.attrs["parameters"] = json.dumps(
        {
            "min_clearance": episode.scenario.min_clearance,
            "max_clearance": episode.scenario.max_clearance,
            "target_clearance": episode.scenario.target_clearance,
            "max_turn_degrees": episode.scenario.max_turn_degrees,
            "safe_separation": episode.scenario.safe_separation,
            "threat_margin": episode.scenario.threat_margin,
            "cruise_speed": episode.scenario.cruise_speed,
            "collision_samples": episode.scenario.collision_samples,
            "large_penalty": episode.scenario.large_penalty,
            "spatial_collision_weight": episode.scenario.spatial_collision_weight,
            "time_collision_weight": episode.scenario.time_collision_weight,
            "sync_weight": episode.scenario.sync_weight,
            "world_x": episode.scenario.world_x,
            "world_y": episode.scenario.world_y,
            "world_z": episode.scenario.world_z,
            "weights": episode.scenario.weights,
        },
        sort_keys=True,
    )


def _write_trajectories(group: h5py.Group, name: str, trajectories: tuple[object, ...]) -> None:
    paths = [np.asarray(getattr(trajectory, "points"), dtype=float) for trajectory in trajectories]
    offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(path) for path in paths])
    values = np.vstack(paths) if paths else np.empty((0, 3), dtype=float)
    waypoint_group = group.create_group(name)
    _write_dataset(waypoint_group, "values", values)
    _write_dataset(waypoint_group, "offsets", offsets)


def _write_dynamic_graph(group: h5py.Group, episode: ExpertEpisode) -> None:
    graph_group = group.create_group("dynamic_graph")
    graph_group.attrs["node_feature_names"] = json.dumps(
        ["x", "y", "z", "vx", "vy", "vz", "active"]
    )
    graph_group.attrs["edge_feature_names"] = json.dumps(
        ["dx", "dy", "dz", "distance", "shortfall"]
    )
    steps = graph_group.create_group("steps")
    for time_index, step in enumerate(episode.dynamic_graph):
        step_group = steps.create_group(f"{time_index:06d}")
        _write_dataset(step_group, "node_features", step.node_features)
        _write_dataset(step_group, "edge_index", step.edge_index)
        _write_dataset(step_group, "edge_features", step.edge_features)
        _write_dataset(step_group, "edge_mask", step.edge_mask)
        _write_dataset(step_group, "conflict_labels", step.conflict_labels)


def _write_log(group: h5py.Group, name: str, entries: tuple[dict[str, object], ...]) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    payload = np.asarray([json.dumps(entry, sort_keys=True) for entry in entries], dtype=object)
    group.create_dataset(name, data=payload, dtype=string_dtype)


def _write_dataset(group: h5py.Group, name: str, values: np.ndarray) -> h5py.Dataset:
    array = np.asarray(values)
    kwargs: dict[str, object] = {}
    if array.ndim > 0 and array.size > 0:
        kwargs["compression"] = "gzip"
        kwargs["shuffle"] = True
    return group.create_dataset(name, data=array, **kwargs)

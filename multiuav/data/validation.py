"""Strict HDF5 expert-episode validation, including evaluator cost replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from multiuav.core.models import CylindricalThreat, Scenario, TerrainMap, Trajectory, UAVMission
from multiuav.evaluation.multi_uav import MultiUAVEvaluator


@dataclass(frozen=True)
class DatasetValidationReport:
    """Validation result suitable for scripts and CI assertions."""

    is_valid: bool
    episode_count: int
    issues: tuple[str, ...]


def validate_expert_dataset(path: Path) -> DatasetValidationReport:
    """Check numerical health, shapes, trajectory consistency, and evaluator replay."""
    issues: list[str] = []
    with h5py.File(path, "r") as handle:
        if "episodes" not in handle:
            return DatasetValidationReport(False, 0, ("Missing /episodes group.",))
        episode_ids = sorted(handle["episodes"].keys())
        for episode_id in episode_ids:
            _validate_episode(handle[f"episodes/{episode_id}"], episode_id, issues)
    return DatasetValidationReport(not issues, len(episode_ids), tuple(issues))


def _validate_episode(group: h5py.Group, episode_id: str, issues: list[str]) -> None:
    required = (
        "scenario",
        "raw_waypoints",
        "repaired_waypoints",
        "positions",
        "velocities",
        "timestamps",
        "active_mask",
        "start_delays",
        "low_level_actions",
        "low_level_action_mask",
        "low_level_clip_ratio",
        "high_level_actions",
        "high_level_action_mask",
        "cost_breakdown",
        "strict_success",
        "minimum_separation",
        "temporal_conflict_count",
        "dynamic_graph",
        "schedule_log",
        "spatial_repair_log",
    )
    for name in required:
        if name not in group:
            issues.append(f"{episode_id}: missing {name}.")
    if any(name not in group for name in required):
        return
    positions = np.asarray(group["positions"], dtype=float)
    velocities = np.asarray(group["velocities"], dtype=float)
    timestamps = np.asarray(group["timestamps"], dtype=float)
    active_mask = np.asarray(group["active_mask"], dtype=bool)
    actions = np.asarray(group["low_level_actions"], dtype=float)
    action_mask = np.asarray(group["low_level_action_mask"], dtype=bool)
    clip_ratio = np.asarray(group["low_level_clip_ratio"], dtype=float)
    high_actions = np.asarray(group["high_level_actions"], dtype=np.int16)
    high_mask = np.asarray(group["high_level_action_mask"], dtype=bool)
    num_uavs = int(group.attrs["num_uavs"])
    expected_shape = (len(timestamps), num_uavs, 3)
    if positions.shape != expected_shape:
        issues.append(f"{episode_id}: positions shape {positions.shape} != {expected_shape}.")
        return
    for name, values, shape in (
        ("velocities", velocities, expected_shape),
        ("low_level_actions", actions, expected_shape),
        ("active_mask", active_mask, expected_shape[:2]),
        ("low_level_action_mask", action_mask, expected_shape[:2]),
        ("low_level_clip_ratio", clip_ratio, expected_shape[:2]),
        ("high_level_actions", high_actions, expected_shape[:2]),
        ("high_level_action_mask", high_mask, expected_shape[:2]),
    ):
        if values.shape != shape:
            issues.append(f"{episode_id}: {name} shape {values.shape} != {shape}.")
    _check_finite(episode_id, positions, "positions", issues)
    _check_finite(episode_id, velocities, "velocities", issues)
    _check_finite(episode_id, actions, "low_level_actions", issues)
    _check_finite(episode_id, clip_ratio, "low_level_clip_ratio", issues)
    if "convergence_history" in group:
        convergence_history = np.asarray(group["convergence_history"], dtype=float)
        if convergence_history.ndim != 1 or len(convergence_history) == 0:
            issues.append(f"{episode_id}: convergence_history must be a nonempty vector.")
        _check_finite(episode_id, convergence_history, "convergence_history", issues)
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0.0):
        issues.append(f"{episode_id}: timestamps are not strictly increasing.")
    max_speed = float(group.attrs["max_speed"])
    if np.any(np.linalg.norm(actions, axis=2) > max_speed + 1e-8):
        issues.append(f"{episode_id}: low-level action exceeds max_speed.")
    if np.any((clip_ratio < 0.0) | (clip_ratio > 1.0)):
        issues.append(f"{episode_id}: invalid low-level clip ratio.")
    if np.any(high_mask & (high_actions < 0)):
        issues.append(f"{episode_id}: masked high-level label is missing.")
    if np.any(~high_mask & (high_actions != -1)):
        issues.append(f"{episode_id}: unmasked high-level label was fabricated.")
    _validate_trajectories_and_cost(group, episode_id, issues)
    _validate_dynamic_graph(group, episode_id, len(timestamps), num_uavs, issues)


def _validate_trajectories_and_cost(group: h5py.Group, episode_id: str, issues: list[str]) -> None:
    try:
        scenario = _scenario_from_group(group["scenario"])
        raw_paths = _trajectories_from_group(group["raw_waypoints"])
        repaired_paths = _trajectories_from_group(group["repaired_waypoints"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        issues.append(f"{episode_id}: invalid scenario or trajectory storage: {error}")
        return
    if len(raw_paths) != len(scenario.missions) or len(repaired_paths) != len(scenario.missions):
        issues.append(f"{episode_id}: trajectory count does not match scenario missions.")
        return
    for index, (raw, repaired, mission) in enumerate(
        zip(raw_paths, repaired_paths, scenario.missions, strict=True)
    ):
        for name, path in (("raw", raw), ("repaired", repaired)):
            if not np.allclose(path.points[0], mission.start, rtol=1e-6, atol=1e-8):
                issues.append(f"{episode_id}: {name} path {index} start does not match mission.")
            if not np.allclose(path.points[-1], mission.goal, rtol=1e-6, atol=1e-8):
                issues.append(f"{episode_id}: {name} path {index} goal does not match mission.")
    evaluator = MultiUAVEvaluator(scenario)
    start_delays = np.asarray(group["start_delays"], dtype=float)
    replay = evaluator.evaluate(repaired_paths, start_delays)
    stored = np.asarray(group["cost_breakdown"], dtype=float)
    if stored.shape != (8,) or not np.allclose(stored[-1], replay.total_cost, rtol=1e-6, atol=1e-8):
        issues.append(f"{episode_id}: cost cannot be reproduced by Python evaluator.")
    if bool(np.asarray(group["strict_success"])) != replay.strict_success:
        issues.append(f"{episode_id}: strict_success differs from evaluator replay.")
    stored_separation = float(np.asarray(group["minimum_separation"]))
    if not np.isclose(stored_separation, replay.minimum_separation, rtol=1e-6, atol=1e-8):
        issues.append(f"{episode_id}: minimum separation differs from evaluator replay.")
    if int(np.asarray(group["temporal_conflict_count"])) != replay.temporal_conflict_count:
        issues.append(f"{episode_id}: temporal conflict count differs from evaluator replay.")


def _scenario_from_group(group: h5py.Group) -> Scenario:
    parameters = json.loads(_text(group.attrs["parameters"]))
    terrain = TerrainMap(
        x_grid=np.asarray(group["terrain_x"], dtype=float),
        y_grid=np.asarray(group["terrain_y"], dtype=float),
        heights=np.asarray(group["terrain_heights"], dtype=float),
    )
    threats = tuple(CylindricalThreat(*row) for row in np.asarray(group["threats"], dtype=float))
    missions = tuple(
        UAVMission(start=start, goal=goal)
        for start, goal in zip(
            np.asarray(group["starts"], dtype=float),
            np.asarray(group["goals"], dtype=float),
            strict=True,
        )
    )
    return Scenario(terrain=terrain, threats=threats, missions=missions, **parameters)


def _trajectories_from_group(group: h5py.Group) -> tuple[Trajectory, ...]:
    values = np.asarray(group["values"], dtype=float)
    offsets = np.asarray(group["offsets"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("invalid flattened trajectory offsets")
    return tuple(
        Trajectory(values[offsets[index] : offsets[index + 1]]) for index in range(len(offsets) - 1)
    )


def _validate_dynamic_graph(
    group: h5py.Group,
    episode_id: str,
    time_count: int,
    num_uavs: int,
    issues: list[str],
) -> None:
    steps = group["dynamic_graph/steps"]
    if len(steps) != time_count:
        issues.append(f"{episode_id}: dynamic graph step count does not match timestamps.")
        return
    for step_id in sorted(steps.keys()):
        step = steps[step_id]
        nodes = np.asarray(step["node_features"], dtype=float)
        index = np.asarray(step["edge_index"], dtype=np.int64)
        features = np.asarray(step["edge_features"], dtype=float)
        mask = np.asarray(step["edge_mask"], dtype=bool)
        labels = np.asarray(step["conflict_labels"], dtype=bool)
        if nodes.shape != (num_uavs, 7):
            issues.append(f"{episode_id}/{step_id}: invalid node feature shape.")
        if index.shape[0] != 2 or features.shape != (index.shape[1], 5):
            issues.append(f"{episode_id}/{step_id}: invalid variable edge storage.")
        if mask.shape != (index.shape[1],) or labels.shape != (index.shape[1],):
            issues.append(f"{episode_id}/{step_id}: invalid edge mask or labels.")
        _check_finite(episode_id, nodes, f"dynamic_graph/{step_id}/node_features", issues)
        _check_finite(episode_id, features, f"dynamic_graph/{step_id}/edge_features", issues)


def _check_finite(episode_id: str, values: np.ndarray, name: str, issues: list[str]) -> None:
    if not np.isfinite(values).all():
        issues.append(f"{episode_id}: {name} contains NaN or Inf.")


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

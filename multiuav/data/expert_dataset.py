"""Build in-memory expert episodes with explicit masks and variable graph steps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import EvaluationResult, Scenario, Trajectory
from multiuav.data.matlab_import import ImportedExpertCase
from multiuav.evaluation.multi_uav import MultiUAVEvaluator
from multiuav.geometry.trajectories import path_length, resample_path

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int16]

HIGH_LEVEL_ACTIONS = {
    "continue": 0,
    "wait_yield": 1,
    "climb": 2,
    "descend": 3,
    "shift_left": 4,
    "shift_right": 5,
    "local_subgoal": 6,
    "assigned_delay": 7,
    "conflict_priority": 8,
}
COST_COMPONENT_NAMES = (
    "j_len",
    "j_alt",
    "j_turn",
    "j_thr",
    "j_spa",
    "j_tmp",
    "j_sync",
    "total_cost",
)


@dataclass(frozen=True)
class DynamicGraphStep:
    """Variable-edge graph at a single environment time step."""

    node_features: FloatArray
    edge_index: NDArray[np.int64]
    edge_features: FloatArray
    edge_mask: BoolArray
    conflict_labels: BoolArray


@dataclass(frozen=True)
class ExpertEpisode:
    """Episode data ready for one HDF5 group with fixed dynamics and masked labels."""

    scenario_id: str
    seed: int
    scenario: Scenario
    nominal_speed: float
    max_speed: float
    dt: float
    source: str
    raw_trajectories: tuple[Trajectory, ...]
    repaired_trajectories: tuple[Trajectory, ...]
    positions: FloatArray
    velocities: FloatArray
    timestamps: FloatArray
    active_mask: BoolArray
    start_delays: FloatArray
    low_level_actions: FloatArray
    low_level_action_mask: BoolArray
    low_level_clip_ratio: FloatArray
    high_level_actions: IntArray
    high_level_action_mask: BoolArray
    convergence_history: FloatArray | None
    dynamic_graph: tuple[DynamicGraphStep, ...]
    schedule_log: tuple[dict[str, Any], ...]
    spatial_repair_log: tuple[dict[str, Any], ...]
    evaluation: EvaluationResult


class ExpertDataset:
    """Read-only, lightweight access to HDF5 expert episodes."""

    def __init__(self, path: str | Any) -> None:
        self.path = path

    def episode_ids(self) -> tuple[str, ...]:
        """Return episode identifiers in deterministic lexical order."""
        import h5py

        with h5py.File(self.path, "r") as handle:
            return tuple(sorted(handle["episodes"].keys()))

    def load_episode(self, episode_id: str) -> LoadedExpertEpisode:
        """Load primary fixed-shape arrays; dynamic graphs stay on disk by design."""
        import h5py

        with h5py.File(self.path, "r") as handle:
            group = handle[f"episodes/{episode_id}"]
            return LoadedExpertEpisode(
                positions=np.asarray(group["positions"], dtype=float),
                velocities=np.asarray(group["velocities"], dtype=float),
                timestamps=np.asarray(group["timestamps"], dtype=float),
                active_mask=np.asarray(group["active_mask"], dtype=bool),
                low_level_actions=np.asarray(group["low_level_actions"], dtype=float),
                high_level_actions=np.asarray(group["high_level_actions"], dtype=np.int16),
            )


@dataclass(frozen=True)
class LoadedExpertEpisode:
    """Fixed-shape arrays most training loaders consume eagerly."""

    positions: FloatArray
    velocities: FloatArray
    timestamps: FloatArray
    active_mask: BoolArray
    low_level_actions: FloatArray
    high_level_actions: IntArray


def build_expert_episode(
    imported: ImportedExpertCase,
    *,
    dt: float,
    max_speed: float | None = None,
    schedule_log: Sequence[object] | None = None,
    spatial_repair_log: Sequence[object] | None = None,
    convergence_history: FloatArray | None = None,
) -> ExpertEpisode:
    """Convert complete trajectories to time series, graphs, actions, masks, and metrics."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    action_speed = imported.nominal_speed if max_speed is None else max_speed
    if action_speed <= 0.0:
        raise ValueError("max_speed must be positive.")
    if len(imported.repaired_trajectories) != len(imported.scenario.missions):
        raise ValueError("Scenario missions and repaired trajectories must have equal counts.")
    schedules = _normalise_logs(imported.schedule_log if schedule_log is None else schedule_log)
    raw_repairs = imported.spatial_repair_log if spatial_repair_log is None else spatial_repair_log
    repairs = _normalise_logs(raw_repairs)
    positions, velocities, timestamps, active_mask = _time_series(
        imported.repaired_trajectories,
        imported.start_delays,
        imported.nominal_speed,
        dt,
    )
    low_actions, low_mask, clip_ratio = _low_level_actions(
        positions, active_mask, dt, action_speed
    )
    high_actions, high_mask = _high_level_actions(
        timestamps,
        active_mask,
        imported.start_delays,
        imported.repaired_trajectories,
        schedules,
        repairs,
        imported.nominal_speed,
    )
    dynamic_graph = _dynamic_graph(positions, velocities, active_mask, imported.scenario)
    evaluation = MultiUAVEvaluator(imported.scenario).evaluate(
        imported.repaired_trajectories, imported.start_delays
    )
    return ExpertEpisode(
        scenario_id=imported.scenario_id,
        seed=imported.seed,
        scenario=imported.scenario,
        nominal_speed=imported.nominal_speed,
        max_speed=action_speed,
        dt=dt,
        source=imported.source,
        raw_trajectories=imported.raw_trajectories,
        repaired_trajectories=imported.repaired_trajectories,
        positions=positions,
        velocities=velocities,
        timestamps=timestamps,
        active_mask=active_mask,
        start_delays=imported.start_delays.copy(),
        low_level_actions=low_actions,
        low_level_action_mask=low_mask,
        low_level_clip_ratio=clip_ratio,
        high_level_actions=high_actions,
        high_level_action_mask=high_mask,
        convergence_history=(
            None if convergence_history is None else np.asarray(convergence_history, dtype=float)
        ),
        dynamic_graph=dynamic_graph,
        schedule_log=schedules,
        spatial_repair_log=repairs,
        evaluation=evaluation,
    )


def _time_series(
    trajectories: Sequence[Trajectory],
    start_delays: FloatArray,
    speed: float,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    durations = np.asarray([path_length(trajectory.points) / speed for trajectory in trajectories])
    if start_delays.shape != durations.shape:
        raise ValueError("start_delays must provide one value per trajectory.")
    final_time = float(np.max(start_delays + durations))
    time_count = max(2, int(np.ceil(final_time / dt)) + 1)
    timestamps = np.arange(time_count, dtype=float) * dt
    positions = np.empty((time_count, len(trajectories), 3), dtype=float)
    active_mask = np.zeros((time_count, len(trajectories)), dtype=bool)
    for uav_index, (trajectory, delay, duration) in enumerate(
        zip(trajectories, start_delays, durations, strict=True)
    ):
        progress = np.clip((timestamps - delay) / max(duration, 1e-9), 0.0, 1.0)
        samples = resample_path(trajectory.points, time_count)
        source_fractions = np.linspace(0.0, 1.0, time_count)
        positions[:, uav_index, :] = np.column_stack(
            [np.interp(progress, source_fractions, samples[:, axis]) for axis in range(3)]
        )
        active_mask[:, uav_index] = (timestamps >= delay) & (timestamps <= delay + duration)
    velocities = np.zeros_like(positions)
    velocities[:-1] = np.diff(positions, axis=0) / dt
    velocities[~active_mask] = 0.0
    return positions, velocities, timestamps, active_mask


def _low_level_actions(
    positions: FloatArray,
    active_mask: BoolArray,
    dt: float,
    max_speed: float,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    actions = np.zeros_like(positions)
    mask = np.zeros(active_mask.shape, dtype=bool)
    mask[:-1] = active_mask[:-1] & active_mask[1:]
    desired = np.diff(positions, axis=0) / dt
    magnitudes = np.linalg.norm(desired, axis=2)
    ratios = np.ones(active_mask.shape, dtype=float)
    desired_ratios = np.minimum(1.0, max_speed / np.maximum(magnitudes, 1e-12))
    actions[:-1] = desired * desired_ratios[..., None]
    actions[~mask] = 0.0
    ratios[:-1] = np.where(mask[:-1], desired_ratios, 1.0)
    return actions, mask, ratios


def _high_level_actions(
    timestamps: FloatArray,
    active_mask: BoolArray,
    start_delays: FloatArray,
    trajectories: Sequence[Trajectory],
    schedule_log: Sequence[Mapping[str, Any]],
    repair_log: Sequence[Mapping[str, Any]],
    speed: float,
) -> tuple[IntArray, BoolArray]:
    actions = np.full(active_mask.shape, -1, dtype=np.int16)
    mask = np.zeros(active_mask.shape, dtype=bool)
    for entry in schedule_log:
        uav_index = int(entry["delayed_uav"])
        if 0 <= uav_index < active_mask.shape[1]:
            actions[0, uav_index] = HIGH_LEVEL_ACTIONS["assigned_delay"]
            mask[0, uav_index] = True
    for entry in repair_log:
        uav_index = int(entry["moved_uav"])
        if not 0 <= uav_index < active_mask.shape[1]:
            continue
        fraction = float(entry.get("critical_fraction", 0.5))
        duration = path_length(trajectories[uav_index].points) / speed
        event_time = start_delays[uav_index] + fraction * duration
        time_index = int(np.argmin(np.abs(timestamps - event_time)))
        altitude_offset = float(entry.get("altitude_offset", 0.0))
        action_name = "climb" if altitude_offset >= 0.0 else "descend"
        actions[time_index, uav_index] = HIGH_LEVEL_ACTIONS[action_name]
        mask[time_index, uav_index] = True
    return actions, mask


def _dynamic_graph(
    positions: FloatArray,
    velocities: FloatArray,
    active_mask: BoolArray,
    scenario: Scenario,
) -> tuple[DynamicGraphStep, ...]:
    steps: list[DynamicGraphStep] = []
    num_uavs = positions.shape[1]
    for time_index in range(positions.shape[0]):
        node_features = np.concatenate(
            [
                positions[time_index],
                velocities[time_index],
                active_mask[time_index, :, None].astype(float),
            ],
            axis=1,
        )
        edges: list[tuple[int, int]] = []
        features: list[list[float]] = []
        labels: list[bool] = []
        for first in range(num_uavs - 1):
            if not active_mask[time_index, first]:
                continue
            for second in range(first + 1, num_uavs):
                if not active_mask[time_index, second]:
                    continue
                delta = positions[time_index, second] - positions[time_index, first]
                distance = float(np.linalg.norm(delta))
                shortfall = max(0.0, scenario.safe_separation - distance)
                edges.append((first, second))
                features.append([*delta.tolist(), distance, shortfall])
                labels.append(shortfall > 0.0)
        edge_index = (
            np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
        )
        edge_features = np.asarray(features, dtype=float).reshape(-1, 5)
        edge_mask = np.ones(len(edges), dtype=bool)
        steps.append(
            DynamicGraphStep(
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                edge_mask=edge_mask,
                conflict_labels=np.asarray(labels, dtype=bool),
            )
        )
    return tuple(steps)


def _normalise_logs(entries: Sequence[object]) -> tuple[dict[str, Any], ...]:
    normalised: list[dict[str, Any]] = []
    for entry in entries:
        value = entry if isinstance(entry, Mapping) else vars(entry)
        normalised.append(_json_ready(value))
    return tuple(normalised)


def _json_ready(value: object) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value

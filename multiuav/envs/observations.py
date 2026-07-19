"""Local observations and centralized state for the Phase-8 multi-UAV environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import Scenario
from multiuav.geometry.terrain import terrain_height

FloatArray = NDArray[np.float64]

BASE_LOCAL_OBSERVATION_SIZE = 17


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Immutable numerical state shared by observation, reward, and cost modules."""

    scenario: Scenario
    positions: FloatArray
    velocities: FloatArray
    active_mask: NDArray[np.bool_]
    previous_goal_distances: FloatArray
    step_count: int
    max_steps: int
    max_horizontal_speed: float
    max_vertical_speed: float
    boundary_clipped: NDArray[np.bool_]


def local_observation_size(max_neighbors: int) -> int:
    """Return the stable local feature width for a configured neighbour budget."""
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be nonnegative.")
    return BASE_LOCAL_OBSERVATION_SIZE + 6 * max_neighbors


def build_local_observation(
    snapshot: EnvironmentSnapshot, uav_index: int, max_neighbors: int
) -> NDArray[np.float32]:
    """Build a normalized, padded local observation in documented feature order."""
    scenario = snapshot.scenario
    spans = _world_spans(scenario)
    point = snapshot.positions[uav_index]
    velocity = snapshot.velocities[uav_index]
    goal = scenario.missions[uav_index].goal
    goal_relative = goal - point
    clearance = float(terrain_height(scenario.terrain, point[:2])[0])
    clearance = point[2] - clearance
    threat = _nearest_threat_features(scenario, point, spans)
    neighbour_features = _neighbour_features(snapshot, uav_index, max_neighbors, spans)
    own_position = _normalise_position(point, scenario)
    own_velocity = _normalise_velocity(
        velocity, snapshot.max_horizontal_speed, snapshot.max_vertical_speed
    )
    normalised_goal = goal_relative / spans
    goal_distance = np.array([np.linalg.norm(goal_relative) / np.linalg.norm(spans)], dtype=float)
    remaining_time = np.array(
        [max(0.0, snapshot.max_steps - snapshot.step_count) / snapshot.max_steps], dtype=float
    )
    values = np.concatenate(
        (
            own_position,
            own_velocity,
            normalised_goal,
            goal_distance,
            np.array([clearance / spans[2]], dtype=float),
            threat,
            neighbour_features,
            remaining_time,
            np.array([0.0], dtype=float),
        )
    )
    expected = local_observation_size(max_neighbors)
    if values.shape != (expected,):
        raise RuntimeError("Local observation layout invariant was violated.")
    return np.asarray(np.nan_to_num(values), dtype=np.float32)


def build_centralized_state(snapshot: EnvironmentSnapshot) -> NDArray[np.float32]:
    """Build a flat centralized-critic state without exposing mutable environment arrays."""
    scenario = snapshot.scenario
    spans = _world_spans(scenario)
    uav_rows = np.concatenate(
        (
            np.asarray([_normalise_position(point, scenario) for point in snapshot.positions]),
            np.asarray(
                [
                    _normalise_velocity(
                        velocity, snapshot.max_horizontal_speed, snapshot.max_vertical_speed
                    )
                    for velocity in snapshot.velocities
                ]
            ),
            np.asarray(
                [_normalise_position(mission.goal, scenario) for mission in scenario.missions]
            ),
            snapshot.active_mask.astype(float).reshape(-1, 1),
        ),
        axis=1,
    ).reshape(-1)
    terrain_values = np.asarray(scenario.terrain.heights, dtype=float)
    clearances = np.asarray(
        [
            point[2] - float(terrain_height(scenario.terrain, point[:2])[0])
            for point in snapshot.positions[snapshot.active_mask]
        ],
        dtype=float,
    )
    if len(clearances) == 0:
        clearances = np.zeros(1, dtype=float)
    terrain_summary = np.array(
        [
            terrain_values.min() / spans[2],
            terrain_values.max() / spans[2],
            clearances.min() / spans[2],
            clearances.mean() / spans[2],
        ],
        dtype=float,
    )
    threat_rows = np.asarray(
        [
            [
                threat.center_x / spans[0],
                threat.center_y / spans[1],
                threat.radius / min(spans[0], spans[1]),
                threat.height / spans[2],
            ]
            for threat in scenario.threats
        ],
        dtype=float,
    ).reshape(-1)
    minimum_distance, conflict_count = pairwise_conflict_summary(snapshot)
    conflict_summary = np.array(
        [minimum_distance / np.linalg.norm(spans), conflict_count], dtype=float
    )
    values = np.concatenate((uav_rows, terrain_summary, threat_rows, conflict_summary))
    expected = 10 * len(scenario.missions) + 4 + 4 * len(scenario.threats) + 2
    if values.shape != (expected,):
        raise RuntimeError("Centralized state layout invariant was violated.")
    return np.asarray(np.nan_to_num(values), dtype=np.float32)


def pairwise_conflict_summary(snapshot: EnvironmentSnapshot) -> tuple[float, int]:
    """Return finite minimum active separation and the current safe-separation edge count."""
    active_points = snapshot.positions[snapshot.active_mask]
    spans = _world_spans(snapshot.scenario)
    if len(active_points) < 2:
        return float(np.linalg.norm(spans)), 0
    minimum_distance = np.inf
    conflict_count = 0
    for index, point in enumerate(active_points[:-1]):
        distances = np.linalg.norm(active_points[index + 1 :] - point, axis=1)
        minimum_distance = min(minimum_distance, float(distances.min()))
        conflict_count += int(np.count_nonzero(distances < snapshot.scenario.safe_separation))
    return float(minimum_distance), conflict_count


def _world_spans(scenario: Scenario) -> FloatArray:
    return np.asarray(
        [
            scenario.world_x[1] - scenario.world_x[0],
            scenario.world_y[1] - scenario.world_y[0],
            scenario.world_z[1] - scenario.world_z[0],
        ],
        dtype=float,
    )


def _normalise_position(point: FloatArray, scenario: Scenario) -> FloatArray:
    lower = np.array([scenario.world_x[0], scenario.world_y[0], scenario.world_z[0]], dtype=float)
    return cast(FloatArray, (np.asarray(point, dtype=float) - lower) / _world_spans(scenario))


def _normalise_velocity(
    velocity: FloatArray, max_horizontal_speed: float, max_vertical_speed: float
) -> FloatArray:
    values = np.asarray(velocity, dtype=float).copy()
    values[:2] /= max_horizontal_speed
    values[2] /= max_vertical_speed
    return values


def _nearest_threat_features(
    scenario: Scenario, point: FloatArray, spans: FloatArray
) -> FloatArray:
    if not scenario.threats:
        return np.zeros(4, dtype=float)
    candidates: list[tuple[float, FloatArray, float]] = []
    for threat in scenario.threats:
        offset = np.asarray([threat.center_x - point[0], threat.center_y - point[1]], dtype=float)
        signed_radial = float(np.linalg.norm(offset) - threat.radius)
        candidates.append((signed_radial, offset, threat.height))
    signed_radial, offset, height = min(candidates, key=lambda item: item[0])
    return np.array(
        [
            offset[0] / spans[0],
            offset[1] / spans[1],
            signed_radial / min(spans[:2]),
            (point[2] - height) / spans[2],
        ],
        dtype=float,
    )


def _neighbour_features(
    snapshot: EnvironmentSnapshot, uav_index: int, max_neighbors: int, spans: FloatArray
) -> FloatArray:
    current = snapshot.positions[uav_index]
    candidates = [
        index for index, active in enumerate(snapshot.active_mask) if index != uav_index and active
    ]
    candidates.sort(key=lambda index: float(np.linalg.norm(snapshot.positions[index] - current)))
    values: list[FloatArray] = []
    for index in candidates[:max_neighbors]:
        relative_position = (snapshot.positions[index] - current) / spans
        relative_velocity = _normalise_velocity(
            snapshot.velocities[index] - snapshot.velocities[uav_index],
            snapshot.max_horizontal_speed,
            snapshot.max_vertical_speed,
        )
        values.append(np.concatenate((relative_position, relative_velocity)))
    while len(values) < max_neighbors:
        values.append(np.zeros(6, dtype=float))
    return np.concatenate(values) if values else np.empty(0, dtype=float)

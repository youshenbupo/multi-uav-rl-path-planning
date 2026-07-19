"""Local observations and centralized state for the Phase-8 multi-UAV environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import Scenario
from multiuav.envs.communication import AgentKnowledgeState
from multiuav.envs.dynamic_world import DynamicWorldState
from multiuav.geometry.terrain import terrain_height

FloatArray = NDArray[np.float64]

BASE_LOCAL_OBSERVATION_SIZE = 17
DYNAMIC_OBSTACLE_FEATURE_SIZE = 9


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
    dynamic_world: DynamicWorldState | None = None
    knowledge_states: tuple[AgentKnowledgeState, ...] = ()


def local_observation_size(max_neighbors: int, max_dynamic_obstacles: int = 0) -> int:
    """Return the stable local feature width for a configured neighbour budget."""
    if max_neighbors < 0 or max_dynamic_obstacles < 0:
        raise ValueError("Observation entity budgets must be nonnegative.")
    return (
        BASE_LOCAL_OBSERVATION_SIZE
        + DYNAMIC_OBSTACLE_FEATURE_SIZE * max_dynamic_obstacles
        + 6 * max_neighbors
    )


def build_local_observation(
    snapshot: EnvironmentSnapshot,
    uav_index: int,
    max_neighbors: int,
    max_dynamic_obstacles: int = 0,
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
    dynamic_obstacle_features = _dynamic_obstacle_features(
        snapshot, uav_index, max_dynamic_obstacles, spans
    )
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
            dynamic_obstacle_features,
            neighbour_features,
            remaining_time,
            np.array([0.0], dtype=float),
        )
    )
    expected = local_observation_size(max_neighbors, max_dynamic_obstacles)
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
    dynamic_rows = _dynamic_centralized_rows(snapshot, spans)
    minimum_distance, conflict_count = pairwise_conflict_summary(snapshot)
    conflict_summary = np.array(
        [minimum_distance / np.linalg.norm(spans), conflict_count], dtype=float
    )
    values = np.concatenate(
        (uav_rows, terrain_summary, threat_rows, dynamic_rows, conflict_summary)
    )
    dynamic_count = len(snapshot.dynamic_world.obstacles) if snapshot.dynamic_world else 0
    expected = 10 * len(scenario.missions) + 4 + 4 * len(scenario.threats) + 8 * dynamic_count + 2
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
    positions = snapshot.positions
    velocities = snapshot.velocities
    if snapshot.knowledge_states:
        knowledge = snapshot.knowledge_states[uav_index]
        positions = knowledge.positions
        velocities = knowledge.velocities
        candidates = [
            index
            for index, known in enumerate(knowledge.valid)
            if index != uav_index and known and snapshot.active_mask[index]
        ]
    else:
        candidates = [
            index
            for index, active in enumerate(snapshot.active_mask)
            if index != uav_index and active
        ]
    candidates.sort(key=lambda index: float(np.linalg.norm(positions[index] - current)))
    values: list[FloatArray] = []
    for index in candidates[:max_neighbors]:
        relative_position = (positions[index] - current) / spans
        relative_velocity = _normalise_velocity(
            velocities[index] - snapshot.velocities[uav_index],
            snapshot.max_horizontal_speed,
            snapshot.max_vertical_speed,
        )
        values.append(np.concatenate((relative_position, relative_velocity)))
    while len(values) < max_neighbors:
        values.append(np.zeros(6, dtype=float))
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _dynamic_obstacle_features(
    snapshot: EnvironmentSnapshot,
    uav_index: int,
    max_dynamic_obstacles: int,
    spans: FloatArray,
) -> FloatArray:
    if max_dynamic_obstacles == 0:
        return np.empty(0, dtype=float)
    if snapshot.dynamic_world is None:
        return np.zeros(DYNAMIC_OBSTACLE_FEATURE_SIZE * max_dynamic_obstacles, dtype=float)
    point = snapshot.positions[uav_index]
    velocity = snapshot.velocities[uav_index]
    candidates = list(enumerate(snapshot.dynamic_world.obstacles))
    centers = snapshot.dynamic_world.centers
    candidates.sort(key=lambda item: float(np.linalg.norm(centers[item[0]] - point)))
    rows: list[FloatArray] = []
    for index, obstacle in candidates[:max_dynamic_obstacles]:
        relative_position = centers[index] - point
        relative_velocity = obstacle.velocity - velocity
        relative_speed_squared = float(relative_velocity @ relative_velocity)
        time_to_cpa = 0.0
        if relative_speed_squared > 1e-12:
            time_to_cpa = max(
                0.0, -float(relative_position @ relative_velocity) / relative_speed_squared
            )
        rows.append(
            np.concatenate(
                (
                    relative_position / spans,
                    _normalise_velocity(
                        relative_velocity,
                        snapshot.max_horizontal_speed,
                        snapshot.max_vertical_speed,
                    ),
                    np.array(
                        [
                            obstacle.radius / min(spans[:2]),
                            time_to_cpa / max(snapshot.max_steps, 1),
                            1.0,
                        ],
                        dtype=float,
                    ),
                )
            )
        )
    while len(rows) < max_dynamic_obstacles:
        rows.append(np.zeros(DYNAMIC_OBSTACLE_FEATURE_SIZE, dtype=float))
    return np.concatenate(rows)


def _dynamic_centralized_rows(snapshot: EnvironmentSnapshot, spans: FloatArray) -> FloatArray:
    if snapshot.dynamic_world is None or not snapshot.dynamic_world.obstacles:
        return np.empty(0, dtype=float)
    return np.asarray(
        [
            [
                *(_normalise_position(center, snapshot.scenario)),
                *(_normalise_velocity(
                    obstacle.velocity,
                    snapshot.max_horizontal_speed,
                    snapshot.max_vertical_speed,
                )),
                obstacle.radius / min(spans[:2]),
                obstacle.height / spans[2],
            ]
            for center, obstacle in zip(
                snapshot.dynamic_world.centers, snapshot.dynamic_world.obstacles, strict=True
            )
        ],
        dtype=float,
    ).reshape(-1)

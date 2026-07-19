"""Performance rewards and explicit safety costs for the multi-UAV environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from multiuav.envs.observations import EnvironmentSnapshot
from multiuav.geometry.terrain import terrain_height


@dataclass(frozen=True)
class SafetyCosts:
    """Safety signals that remain separate from the scalar learning reward."""

    inter_uav_collision_cost: float = 0.0
    terrain_violation_cost: float = 0.0
    threat_violation_cost: float = 0.0
    dynamic_obstacle_violation_cost: float = 0.0
    boundary_violation_cost: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Return serializable named costs for PettingZoo infos."""
        return asdict(self)


@dataclass(frozen=True)
class PerformanceReward:
    """Attributable performance components whose sum is the agent reward."""

    progress: float
    goal_arrival: float
    path_efficiency: float
    smoothness: float
    energy_proxy: float
    time_penalty: float

    @property
    def total(self) -> float:
        """Return the scalar reward consumed by the future policy optimizer."""
        return float(sum(asdict(self).values()))

    def as_dict(self) -> dict[str, float]:
        """Return serializable component values for PettingZoo infos."""
        return asdict(self)


def compute_safety_costs(snapshot: EnvironmentSnapshot, uav_index: int) -> SafetyCosts:
    """Compute the four required safety costs from the current state only."""
    scenario = snapshot.scenario
    point = snapshot.positions[uav_index]
    collision_cost = 0.0
    if snapshot.active_mask[uav_index]:
        for index, other in enumerate(snapshot.positions):
            if index == uav_index or not snapshot.active_mask[index]:
                continue
            shortfall = max(0.0, scenario.safe_separation - float(np.linalg.norm(other - point)))
            collision_cost += shortfall**2 / max(scenario.safe_separation**2, 1e-9)
    ground = float(terrain_height(scenario.terrain, point[:2])[0])
    terrain_shortfall = max(0.0, scenario.min_clearance - (point[2] - ground))
    threat_cost = 0.0
    for threat in scenario.threats:
        if point[2] > threat.height:
            continue
        radial_penetration = max(
            0.0,
            threat.radius
            - float(np.linalg.norm(point[:2] - np.array([threat.center_x, threat.center_y]))),
        )
        threat_cost += radial_penetration**2 / max(threat.radius**2, 1e-9)
    dynamic_obstacle_cost = 0.0
    if snapshot.dynamic_world is not None:
        for center, obstacle in zip(
            snapshot.dynamic_world.centers, snapshot.dynamic_world.obstacles, strict=True
        ):
            if abs(point[2] - center[2]) > obstacle.height / 2.0:
                continue
            radial_penetration = max(
                0.0, obstacle.radius - float(np.linalg.norm(point[:2] - center[:2]))
            )
            dynamic_obstacle_cost += radial_penetration**2 / max(obstacle.radius**2, 1e-9)
    return SafetyCosts(
        inter_uav_collision_cost=float(collision_cost),
        terrain_violation_cost=float(terrain_shortfall**2),
        threat_violation_cost=float(threat_cost),
        dynamic_obstacle_violation_cost=float(dynamic_obstacle_cost),
        boundary_violation_cost=float(snapshot.boundary_clipped[uav_index]),
    )


def compute_performance_reward(
    snapshot: EnvironmentSnapshot,
    uav_index: int,
    *,
    newly_arrived: bool,
    previous_velocity: np.ndarray,
    weights: dict[str, float],
) -> PerformanceReward:
    """Compute dense performance reward without embedding safety costs in it."""
    scenario = snapshot.scenario
    mission = scenario.missions[uav_index]
    point = snapshot.positions[uav_index]
    velocity = snapshot.velocities[uav_index]
    current_distance = float(np.linalg.norm(mission.goal - point))
    initial_distance = max(float(np.linalg.norm(mission.goal - mission.start)), 1e-6)
    progress = (
        weights["progress"]
        * (snapshot.previous_goal_distances[uav_index] - current_distance)
        / initial_distance
    )
    goal_arrival = weights["goal_arrival"] if newly_arrived else 0.0
    speed = float(np.linalg.norm(velocity))
    direction = mission.goal - point
    alignment = 0.0
    if speed > 1e-9 and np.linalg.norm(direction) > 1e-9:
        alignment = float(np.dot(velocity, direction) / (speed * np.linalg.norm(direction)))
    path_efficiency = weights["path_efficiency"] * max(0.0, alignment)
    smoothness = (
        -weights["smoothness"]
        * float(np.linalg.norm(velocity - previous_velocity))
        / max(snapshot.max_horizontal_speed, snapshot.max_vertical_speed)
    )
    energy = -weights["energy"] * (
        np.linalg.norm(velocity[:2]) ** 2 / snapshot.max_horizontal_speed**2
        + velocity[2] ** 2 / snapshot.max_vertical_speed**2
    )
    return PerformanceReward(
        progress=float(progress),
        goal_arrival=float(goal_arrival),
        path_efficiency=float(path_efficiency),
        smoothness=float(smoothness),
        energy_proxy=float(energy),
        time_penalty=-weights["time"],
    )

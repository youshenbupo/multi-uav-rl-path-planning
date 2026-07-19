"""Deterministic feasibility-oriented geometric repairs for encoded candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import Scenario, Trajectory
from multiuav.geometry.terrain import terrain_height
from multiuav.geometry.trajectories import turn_angles

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GeometricRepairInfo:
    """Counts kept separate from coordination logs for optimizer diagnostics."""

    boundary_count: int = 0
    terrain_count: int = 0
    threat_count: int = 0
    turn_count: int = 0


class GeometricRepairer:
    """Repair individual candidate waypoints before the shared evaluator scores them."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def repair_waypoint(
        self, point: FloatArray, uav_index: int
    ) -> tuple[FloatArray, GeometricRepairInfo]:
        """Clamp bounds, terrain clearance, and cylinder margins deterministically."""
        del uav_index  # Per-UAV layering is a separate legacy migration surface.
        repaired = np.asarray(point, dtype=float).copy()
        boundary_count = 0
        terrain_count = 0
        threat_count = 0
        original_xy = repaired[:2].copy()
        repaired[0] = np.clip(repaired[0], *self.scenario.world_x)
        repaired[1] = np.clip(repaired[1], *self.scenario.world_y)
        if not np.allclose(original_xy, repaired[:2]):
            boundary_count += 1
        ground = float(terrain_height(self.scenario.terrain, repaired[:2])[0])
        lower = ground + self.scenario.min_clearance
        upper = min(ground + self.scenario.max_clearance, self.scenario.world_z[1])
        previous_z = repaired[2]
        repaired[2] = np.clip(repaired[2], lower, upper)
        if repaired[2] != previous_z:
            terrain_count += 1
        for threat in self.scenario.threats:
            center = np.array([threat.center_x, threat.center_y], dtype=float)
            offset = repaired[:2] - center
            distance = float(np.linalg.norm(offset))
            required_distance = threat.radius + self.scenario.threat_margin
            if distance < required_distance and 0.0 <= repaired[2] <= threat.height:
                direction = offset / distance if distance > 1e-9 else np.array([1.0, 0.0])
                repaired[:2] = center + required_distance * direction
                repaired[0] = np.clip(repaired[0], *self.scenario.world_x)
                repaired[1] = np.clip(repaired[1], *self.scenario.world_y)
                threat_count += 1
        return repaired, GeometricRepairInfo(boundary_count, terrain_count, threat_count, 0)

    def repair_trajectory(
        self, trajectory: Trajectory, uav_index: int
    ) -> tuple[Trajectory, GeometricRepairInfo]:
        """Repair non-terminal waypoints and apply one deterministic turn-smoothing pass."""
        points = np.asarray(trajectory.points, dtype=float).copy()
        info = GeometricRepairInfo()
        for index in range(1, len(points) - 1):
            points[index], waypoint_info = self.repair_waypoint(points[index], uav_index)
            info = _merge_info(info, waypoint_info)
        max_turn = np.deg2rad(self.scenario.max_turn_degrees)
        for index, angle in enumerate(turn_angles(points), start=1):
            if angle <= max_turn:
                continue
            candidate = 0.5 * (points[index - 1] + points[index + 1])
            points[index], waypoint_info = self.repair_waypoint(candidate, uav_index)
            info = _merge_info(info, waypoint_info)
            info = GeometricRepairInfo(
                info.boundary_count,
                info.terrain_count,
                info.threat_count,
                info.turn_count + 1,
            )
        return Trajectory(points), info


def _merge_info(first: GeometricRepairInfo, second: GeometricRepairInfo) -> GeometricRepairInfo:
    return GeometricRepairInfo(
        first.boundary_count + second.boundary_count,
        first.terrain_count + second.terrain_count,
        first.threat_count + second.threat_count,
        first.turn_count + second.turn_count,
    )

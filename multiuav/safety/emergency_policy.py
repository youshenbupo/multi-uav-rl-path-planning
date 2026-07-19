"""Bounded non-RL fallback commands for CBF solver failure and numerical anomalies."""

from __future__ import annotations

from typing import cast

import numpy as np

from multiuav.envs.observations import EnvironmentSnapshot
from multiuav.geometry.terrain import terrain_height


class EmergencyPolicy:
    """Generate hover or conservative repulsion velocities without using the rejected action."""

    def safe_actions(self, snapshot: EnvironmentSnapshot) -> np.ndarray:
        """Return one physical desired velocity per UAV, with inactive rows fixed at zero."""
        positions = np.asarray(snapshot.positions, dtype=float)
        active = np.asarray(snapshot.active_mask, dtype=bool)
        actions = np.zeros_like(positions, dtype=float)
        for index, point in enumerate(positions):
            if not active[index]:
                continue
            direction = self._risk_direction(snapshot, index)
            actions[index] = self._clip(direction, snapshot)
        return actions

    def _risk_direction(self, snapshot: EnvironmentSnapshot, index: int) -> np.ndarray:
        point = snapshot.positions[index]
        direction = np.zeros(3, dtype=float)
        for other, other_point in enumerate(snapshot.positions):
            if other == index or not snapshot.active_mask[other]:
                continue
            relative = point - other_point
            distance = float(np.linalg.norm(relative))
            if distance < snapshot.scenario.safe_separation:
                direction += self._unit(relative, fallback_axis=index) * (
                    snapshot.scenario.safe_separation - distance
                )
        for threat in snapshot.scenario.threats:
            if point[2] > threat.height:
                continue
            relative_xy = point[:2] - np.array([threat.center_x, threat.center_y], dtype=float)
            distance = float(np.linalg.norm(relative_xy))
            safe_radius = threat.radius + snapshot.scenario.threat_margin
            if distance < safe_radius:
                direction[:2] += self._unit(relative_xy, fallback_axis=index) * (
                    safe_radius - distance
                )
        if snapshot.dynamic_world is not None:
            for center, obstacle in zip(
                snapshot.dynamic_world.centers, snapshot.dynamic_world.obstacles, strict=True
            ):
                if abs(point[2] - center[2]) > obstacle.height / 2.0:
                    continue
                relative_xy = point[:2] - center[:2]
                distance = float(np.linalg.norm(relative_xy))
                safe_radius = obstacle.radius + snapshot.scenario.threat_margin
                if distance < safe_radius:
                    direction[:2] += self._unit(relative_xy, fallback_axis=index) * (
                        safe_radius - distance
                    )
        ground = float(terrain_height(snapshot.scenario.terrain, point[:2])[0])
        if point[2] - ground < snapshot.scenario.min_clearance:
            direction[2] += snapshot.scenario.min_clearance - (point[2] - ground)
        bounds = (snapshot.scenario.world_x, snapshot.scenario.world_y, snapshot.scenario.world_z)
        for axis, (lower, upper) in enumerate(bounds):
            margin = min(5.0, 0.1 * (upper - lower))
            if point[axis] < lower + margin:
                direction[axis] += lower + margin - point[axis]
            if point[axis] > upper - margin:
                direction[axis] -= point[axis] - (upper - margin)
        return direction

    def _clip(self, direction: np.ndarray, snapshot: EnvironmentSnapshot) -> np.ndarray:
        result = np.nan_to_num(np.asarray(direction, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        horizontal_norm = float(np.linalg.norm(result[:2]))
        if horizontal_norm > snapshot.max_horizontal_speed:
            result[:2] *= snapshot.max_horizontal_speed / horizontal_norm
        result[2] = float(
            np.clip(result[2], -snapshot.max_vertical_speed, snapshot.max_vertical_speed)
        )
        return cast(np.ndarray, result)

    def _unit(self, vector: np.ndarray, *, fallback_axis: int) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            return np.array(np.asarray(vector, dtype=float) / norm, dtype=float)
        fallback = np.zeros_like(vector, dtype=float)
        fallback[fallback_axis % len(vector)] = 1.0
        return fallback

"""Deterministic local Gaussian spatial conflict repair."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import Scenario, Trajectory
from multiuav.expert.conflict_graph import ConflictGraph, build_conflict_graph
from multiuav.geometry.terrain import terrain_height
from multiuav.geometry.trajectories import resample_path

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialRepairLogEntry:
    """Auditable record of one local, feasibility-restored path displacement."""

    iteration: int
    uav_pair: tuple[int, int]
    moved_uav: int
    conflict_center: FloatArray
    critical_fraction: float
    lateral_offset: float
    altitude_offset: float
    minimum_distance_before: float
    minimum_distance_after: float
    conflict_weight_before: float
    conflict_weight_after: float
    feasibility_repairs: tuple[str, ...]


@dataclass(frozen=True)
class SpatialRepairResult:
    """Repaired trajectories, graph metrics, and complete repair log."""

    repaired_trajectories: tuple[Trajectory, ...]
    graph_before: ConflictGraph
    graph_after: ConflictGraph
    log: tuple[SpatialRepairLogEntry, ...]
    unresolved: bool


class SpatialRepairer:
    """Move one conflict participant along a local horizontal normal per round."""

    def __init__(
        self,
        scenario: Scenario,
        nominal_speed: float | None = None,
        *,
        maximum_iterations: int = 3,
        gaussian_sigma: float = 0.18,
        lateral_gain: float = 1.1,
        altitude_step: float = 12.0,
    ) -> None:
        if maximum_iterations < 1 or gaussian_sigma <= 0.0 or lateral_gain <= 0.0:
            raise ValueError("Spatial-repair parameters must be positive.")
        self.scenario = scenario
        self.nominal_speed = scenario.cruise_speed if nominal_speed is None else nominal_speed
        if self.nominal_speed <= 0.0:
            raise ValueError("nominal_speed must be positive.")
        self.maximum_iterations = maximum_iterations
        self.gaussian_sigma = gaussian_sigma
        self.lateral_gain = lateral_gain
        self.altitude_step = altitude_step

    def repair(self, trajectories: Sequence[Trajectory]) -> SpatialRepairResult:
        """Repair the worst graph edge repeatedly, preserving endpoint missions."""
        repaired = [
            Trajectory(np.asarray(trajectory.points, dtype=float).copy())
            for trajectory in trajectories
        ]
        if not repaired:
            raise ValueError("At least one trajectory is required.")
        graph_before = build_conflict_graph(repaired, self.scenario, self.nominal_speed)
        graph = graph_before
        entries: list[SpatialRepairLogEntry] = []

        for iteration in range(1, self.maximum_iterations + 1):
            pair = graph.worst_conflict_pair
            if pair is None:
                break
            interval = graph.interval_for_pair(pair)
            if interval is None:
                break
            moved_uav = self._select_mover(pair)
            fraction = interval.critical_fractions[1 if moved_uav == pair[1] else 0]
            center = self._conflict_center(
                repaired[pair[0]].points, repaired[pair[1]].points, interval
            )
            candidate, repairs = self._shift_path(
                repaired[moved_uav].points,
                self._other_path(repaired, pair, moved_uav),
                fraction,
                interval.max_shortfall,
                moved_uav,
                iteration,
            )
            candidate_paths = repaired.copy()
            candidate_paths[moved_uav] = Trajectory(candidate)
            updated_graph = build_conflict_graph(candidate_paths, self.scenario, self.nominal_speed)
            if updated_graph.total_conflict_weight >= graph.total_conflict_weight - 1e-9:
                break
            entries.append(
                SpatialRepairLogEntry(
                    iteration=iteration,
                    uav_pair=pair,
                    moved_uav=moved_uav,
                    conflict_center=center,
                    critical_fraction=fraction,
                    lateral_offset=self.lateral_gain * interval.max_shortfall,
                    altitude_offset=self._altitude_sign(moved_uav) * self.altitude_step,
                    minimum_distance_before=interval.minimum_distance,
                    minimum_distance_after=_pair_minimum(updated_graph, pair),
                    conflict_weight_before=graph.total_conflict_weight,
                    conflict_weight_after=updated_graph.total_conflict_weight,
                    feasibility_repairs=repairs,
                )
            )
            repaired = candidate_paths
            graph = updated_graph

        return SpatialRepairResult(
            repaired_trajectories=tuple(repaired),
            graph_before=graph_before,
            graph_after=graph,
            log=tuple(entries),
            unresolved=graph.worst_conflict_pair is not None,
        )

    def _select_mover(self, pair: tuple[int, int]) -> int:
        """Use the legacy parity tie-break: first odd MATLAB index keeps priority."""
        first, second = pair
        return second if (first + 1) % 2 else first

    def _other_path(
        self,
        trajectories: Sequence[Trajectory],
        pair: tuple[int, int],
        moved_uav: int,
    ) -> FloatArray:
        other_index = pair[1] if moved_uav == pair[0] else pair[0]
        return trajectories[other_index].points

    def _conflict_center(
        self,
        first_path: FloatArray,
        second_path: FloatArray,
        interval: object,
    ) -> FloatArray:
        critical_fractions = getattr(interval, "critical_fractions")
        first = _sample_at_fraction(first_path, critical_fractions[0])
        second = _sample_at_fraction(second_path, critical_fractions[1])
        return (first + second) / 2.0

    def _shift_path(
        self,
        path: FloatArray,
        other_path: FloatArray,
        critical_fraction: float,
        shortfall: float,
        moved_uav: int,
        iteration: int,
    ) -> tuple[FloatArray, tuple[str, ...]]:
        candidate = path.copy()
        other_point = _sample_at_fraction(other_path, critical_fraction)
        offset = self.lateral_gain * shortfall * (1.0 + 0.08 * (iteration - 1))
        altitude = (
            self._altitude_sign(moved_uav) * self.altitude_step * (1.0 + 0.05 * (iteration - 1))
        )
        repairs: set[str] = set()
        waypoint_count = len(candidate)
        for index in range(1, waypoint_count - 1):
            waypoint_fraction = index / (waypoint_count - 1)
            weight = float(
                np.exp(
                    -((waypoint_fraction - critical_fraction) ** 2) / (2.0 * self.gaussian_sigma**2)
                )
            )
            if weight < 1e-3:
                continue
            tangent = candidate[index + 1, :2] - candidate[index - 1, :2]
            tangent_norm = float(np.linalg.norm(tangent))
            normal = (
                np.array([1.0, 0.0]) if tangent_norm < 1e-9 else np.array([-tangent[1], tangent[0]])
            )
            normal /= max(float(np.linalg.norm(normal)), 1e-9)
            push = candidate[index, :2] - other_point[:2]
            if float(np.dot(normal, push)) < 0.0:
                normal = -normal
            candidate[index, :2] += weight * offset * normal
            candidate[index, 2] += weight * altitude
            candidate[index], local_repairs = self._restore_feasibility(candidate[index], normal)
            repairs.update(local_repairs)
        return candidate, tuple(sorted(repairs))

    def _restore_feasibility(
        self, point: FloatArray, fallback_normal: FloatArray
    ) -> tuple[FloatArray, tuple[str, ...]]:
        repaired = point.copy()
        changes: list[str] = []
        old_xy = repaired[:2].copy()
        repaired[0] = np.clip(repaired[0], *self.scenario.world_x)
        repaired[1] = np.clip(repaired[1], *self.scenario.world_y)
        if not np.allclose(old_xy, repaired[:2]):
            changes.append("boundary")
        ground = float(terrain_height(self.scenario.terrain, repaired[:2])[0])
        lower = ground + self.scenario.min_clearance
        upper = min(ground + self.scenario.max_clearance, self.scenario.world_z[1])
        old_z = repaired[2]
        repaired[2] = np.clip(repaired[2], lower, upper)
        if repaired[2] != old_z:
            changes.append("terrain")
        for threat in self.scenario.threats:
            radial = repaired[:2] - np.array([threat.center_x, threat.center_y])
            radius = float(np.linalg.norm(radial))
            limit = threat.radius + self.scenario.threat_margin
            if radius < limit and 0.0 <= repaired[2] <= threat.height:
                direction = radial / radius if radius > 1e-9 else fallback_normal
                repaired[:2] = np.array([threat.center_x, threat.center_y]) + limit * direction
                repaired[0] = np.clip(repaired[0], *self.scenario.world_x)
                repaired[1] = np.clip(repaired[1], *self.scenario.world_y)
                changes.append("threat")
        return repaired, tuple(changes)

    @staticmethod
    def _altitude_sign(moved_uav: int) -> float:
        return -1.0 if (moved_uav + 1) % 2 == 0 else 1.0


def _sample_at_fraction(path: FloatArray, fraction: float) -> FloatArray:
    samples = resample_path(path, 1001)
    index = int(round(np.clip(fraction, 0.0, 1.0) * (len(samples) - 1)))
    return cast(FloatArray, np.asarray(samples[index], dtype=float))


def _pair_minimum(graph: ConflictGraph, pair: tuple[int, int]) -> float:
    interval = graph.interval_for_pair(pair)
    return np.inf if interval is None else interval.minimum_distance

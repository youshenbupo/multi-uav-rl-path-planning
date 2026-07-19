"""MATLAB-compatible, strict multi-UAV path evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multiuav.core.models import (
    ConstraintResult,
    CostBreakdown,
    EvaluationResult,
    Scenario,
    Trajectory,
)
from multiuav.geometry.terrain import clearance
from multiuav.geometry.trajectories import (
    path_length,
    resample_path,
    segment_to_cylinder_distance,
    turn_angles,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SingleUAVMetrics:
    """Unweighted legacy single-UAV terms and strict-constraint violations."""

    path_length: float
    altitude_penalty: float
    turn_penalty: float
    threat_penalty: float
    cost: float
    travel_time: float
    max_turn_degrees: float
    terrain_violation_count: int
    turn_violation: float
    turn_violation_count: int
    threat_violation: float
    threat_violation_count: int


class MultiUAVEvaluator:
    """Evaluate fixed Cartesian trajectories without applying legacy repairs."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def evaluate(
        self,
        trajectories: Sequence[Trajectory],
        start_delays: ArrayLike | None = None,
        delay_profiles: ArrayLike | None = None,
    ) -> EvaluationResult:
        """Return cost components plus strict feasibility for a path set.

        Paths are deliberately evaluated as supplied. The MATLAB evaluator can
        optionally invoke scheduling and repair heuristics; those are deferred
        migration modules, not hidden side effects of this stable interface.
        """
        paths = tuple(np.asarray(trajectory.points, dtype=float) for trajectory in trajectories)
        if len(paths) != len(self.scenario.missions):
            raise ValueError("One trajectory is required for every scenario mission.")
        if not paths:
            raise ValueError("At least one trajectory is required.")

        single_metrics = tuple(self._evaluate_single(path) for path in paths)
        travel_times = np.asarray([metric.travel_time for metric in single_metrics], dtype=float)
        delays, profiles = _schedule_arrays(
            len(paths), self.scenario.collision_samples, start_delays, delay_profiles
        )
        spatial_penalty, minimum_spatial, spatial_conflicts, _ = self._spatial_metrics(paths)
        temporal_weight, temporal_count, minimum_temporal, _ = self._temporal_metrics(
            paths, travel_times, delays, profiles
        )
        effective_arrivals = delays + profiles[-1] + travel_times
        sync_penalty = self.scenario.sync_weight * float(
            np.square(effective_arrivals - np.mean(effective_arrivals)).sum()
        )

        weights = self.scenario.weights
        j_len = weights["length"] * sum(metric.path_length for metric in single_metrics)
        j_alt = weights["altitude"] * sum(metric.altitude_penalty for metric in single_metrics)
        j_turn = weights["turn"] * sum(metric.turn_penalty for metric in single_metrics)
        j_thr = weights["threat"] * sum(metric.threat_penalty for metric in single_metrics)
        j_spa = spatial_penalty
        j_tmp = self.scenario.time_collision_weight * temporal_weight
        j_sync = sync_penalty
        total_cost = j_len + j_alt + j_turn + j_thr + j_spa + j_tmp + j_sync
        breakdown = CostBreakdown(
            j_len=j_len,
            j_alt=j_alt,
            j_turn=j_turn,
            j_thr=j_thr,
            j_spa=j_spa,
            j_tmp=j_tmp,
            j_sync=j_sync,
            total_cost=total_cost,
        )
        details = ConstraintResult(
            terrain_feasible=all(metric.terrain_violation_count == 0 for metric in single_metrics),
            threat_feasible=all(metric.threat_violation_count == 0 for metric in single_metrics),
            turning_feasible=all(metric.turn_violation_count == 0 for metric in single_metrics),
            spatial_feasible=spatial_conflicts == 0,
            temporal_feasible=temporal_count == 0,
        )
        strict_success = all(
            (
                details.terrain_feasible,
                details.threat_feasible,
                details.turning_feasible,
                details.spatial_feasible,
                details.temporal_feasible,
            )
        )
        minimum_separation = min(minimum_spatial, minimum_temporal)
        return EvaluationResult(
            total_cost=total_cost,
            cost_breakdown=breakdown,
            strict_success=strict_success,
            minimum_separation=float(minimum_separation),
            temporal_conflict_count=temporal_count,
            constraint_details=details,
        )

    def evaluate_single(self, trajectory: Trajectory) -> SingleUAVMetrics:
        """Evaluate one fixed path with the legacy single-UAV objective terms."""
        return self._evaluate_single(np.asarray(trajectory.points, dtype=float))

    def _evaluate_single(self, path: FloatArray) -> SingleUAVMetrics:
        path_distance = path_length(path)
        travel_time = path_distance / max(self.scenario.cruise_speed, 1e-6)
        path_clearance = clearance(path, self.scenario.terrain)
        altitude_penalty = float(
            np.square(np.maximum(0.0, self.scenario.min_clearance - path_clearance)).sum()
            + np.square(np.maximum(0.0, path_clearance - self.scenario.max_clearance)).sum()
            + 0.15 * np.square(path_clearance - self.scenario.target_clearance).sum()
        )
        terrain_violation_count = int(
            np.count_nonzero(
                (path_clearance < self.scenario.min_clearance)
                | (path_clearance > self.scenario.max_clearance)
            )
        )
        angles = turn_angles(path)
        max_turn = np.deg2rad(self.scenario.max_turn_degrees)
        excessive = angles > max_turn
        turn_penalty = float(15.0 * np.square(angles[~excessive]).sum())
        turn_penalty += float(
            excessive.sum() * self.scenario.large_penalty
            + 1000.0 * np.square(np.rad2deg(angles[excessive] - max_turn)).sum()
        )
        turn_violation = float((angles[excessive] - max_turn).sum())

        threat_penalty = 0.0
        threat_violation = 0.0
        threat_violation_count = 0
        for first, second in zip(path[:-1], path[1:], strict=True):
            for threat in self.scenario.threats:
                cylinder = np.array(
                    [threat.center_x, threat.center_y, threat.radius, threat.height], dtype=float
                )
                minimum_distance = segment_to_cylinder_distance(first, second, cylinder, 12)
                if minimum_distance < 0.0:
                    threat_violation += abs(minimum_distance)
                    threat_violation_count += 1
                    threat_penalty += self.scenario.large_penalty + 2000.0 * abs(minimum_distance)
                elif minimum_distance < self.scenario.threat_margin:
                    threat_penalty += 600.0 * (self.scenario.threat_margin - minimum_distance) ** 2

        weights = self.scenario.weights
        total_cost = (
            weights["length"] * path_distance
            + weights["altitude"] * altitude_penalty
            + weights["turn"] * turn_penalty
            + weights["threat"] * threat_penalty
        )
        return SingleUAVMetrics(
            path_length=path_distance,
            altitude_penalty=altitude_penalty,
            turn_penalty=turn_penalty,
            threat_penalty=threat_penalty,
            cost=total_cost,
            travel_time=travel_time,
            max_turn_degrees=float(np.rad2deg(angles).max(initial=0.0)),
            terrain_violation_count=terrain_violation_count,
            turn_violation=turn_violation,
            turn_violation_count=int(excessive.sum()),
            threat_violation=threat_violation,
            threat_violation_count=threat_violation_count,
        )

    def _spatial_metrics(self, paths: Sequence[FloatArray]) -> tuple[float, float, int, float]:
        penalty = 0.0
        minimum_distance = np.inf
        conflict_count = 0
        violation = 0.0
        samples = self.scenario.collision_samples
        for index, first_path in enumerate(paths[:-1]):
            first_samples = resample_path(first_path, samples)
            for second_path in paths[index + 1 :]:
                second_samples = resample_path(second_path, samples)
                distances = np.linalg.norm(first_samples - second_samples, axis=1)
                pair_minimum = float(distances.min())
                minimum_distance = min(minimum_distance, pair_minimum)
                conflict_count += int(np.count_nonzero(distances < self.scenario.safe_separation))
                if pair_minimum < self.scenario.safe_separation:
                    shortfall = self.scenario.safe_separation - pair_minimum
                    violation += shortfall
                    penalty += self.scenario.large_penalty + (
                        self.scenario.spatial_collision_weight * shortfall**2
                    )
        return penalty, float(minimum_distance), conflict_count, violation

    def _temporal_metrics(
        self,
        paths: Sequence[FloatArray],
        travel_times: FloatArray,
        start_delays: FloatArray,
        delay_profiles: FloatArray,
    ) -> tuple[float, int, float, float]:
        total_weight = 0.0
        total_count = 0
        minimum_distance = np.inf
        total_shortfall = 0.0
        samples = self.scenario.collision_samples
        sample_fractions = np.linspace(0.0, 1.0, samples)
        for index, first_path in enumerate(paths[:-1]):
            first_samples = resample_path(first_path, samples)
            first_times = start_delays[index] + sample_fractions * travel_times[index]
            first_times = first_times + delay_profiles[:, index]
            for other_index, second_path in enumerate(paths[index + 1 :], start=index + 1):
                second_samples = resample_path(second_path, samples)
                second_times = (
                    start_delays[other_index] + sample_fractions * travel_times[other_index]
                )
                second_times = second_times + delay_profiles[:, other_index]
                overlap_start = max(first_times[0], second_times[0])
                overlap_end = min(first_times[-1], second_times[-1])
                if overlap_end <= overlap_start:
                    continue
                common_times = np.linspace(overlap_start, overlap_end, samples)
                synced_first = _interpolate_path(first_times, first_samples, common_times)
                synced_second = _interpolate_path(second_times, second_samples, common_times)
                distances = np.linalg.norm(synced_first - synced_second, axis=1)
                shortfall = np.maximum(0.0, self.scenario.safe_separation - distances)
                minimum_distance = min(minimum_distance, float(distances.min()))
                total_weight += float(np.square(shortfall).sum())
                total_count += int(np.count_nonzero(shortfall > 0.0))
                total_shortfall += float(shortfall.sum())
        return total_weight, total_count, float(minimum_distance), total_shortfall


def _schedule_arrays(
    num_uavs: int,
    samples: int,
    start_delays: ArrayLike | None,
    delay_profiles: ArrayLike | None,
) -> tuple[FloatArray, FloatArray]:
    delays = (
        np.zeros(num_uavs, dtype=float)
        if start_delays is None
        else np.asarray(start_delays, dtype=float)
    )
    delays = delays.reshape(-1)
    if len(delays) != num_uavs:
        raise ValueError("start_delays must provide one value per trajectory.")
    profiles = (
        np.zeros((samples, num_uavs), dtype=float)
        if delay_profiles is None
        else np.asarray(delay_profiles, dtype=float)
    )
    if profiles.shape != (samples, num_uavs):
        raise ValueError("delay_profiles must have shape (collision_samples, num_trajectories).")
    return delays, profiles


def _interpolate_path(times: FloatArray, points: FloatArray, query_times: FloatArray) -> FloatArray:
    """Interpolate each Cartesian coordinate on MATLAB's time axis convention."""
    monotonic_times = np.maximum.accumulate(times)
    for index in range(1, len(monotonic_times)):
        if monotonic_times[index] <= monotonic_times[index - 1]:
            monotonic_times[index] = monotonic_times[index - 1] + 1e-6
    return np.column_stack(
        [np.interp(query_times, monotonic_times, points[:, axis]) for axis in range(3)]
    )

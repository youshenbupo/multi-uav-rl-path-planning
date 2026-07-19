"""Deterministic temporal conflict graphs for fixed multi-UAV trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multiuav.core.models import Scenario, Trajectory
from multiuav.geometry.trajectories import path_length, resample_path

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ConflictInterval:
    """Time and path-fraction bounds of one pairwise temporal conflict."""

    uav_pair: tuple[int, int]
    start_time: float
    end_time: float
    critical_time: float
    start_fractions: tuple[float, float]
    end_fractions: tuple[float, float]
    critical_fractions: tuple[float, float]
    minimum_distance: float
    max_shortfall: float
    total_shortfall: float
    conflict_weight: float
    sample_count: int


@dataclass(frozen=True)
class ConflictGraph:
    """Sparse conflict graph with deterministic, zero-based UAV identifiers."""

    node_ids: IntArray
    edge_index: IntArray
    edge_weights: FloatArray
    conflict_intervals: tuple[ConflictInterval, ...]
    minimum_pairwise_distances: FloatArray
    worst_conflict_pair: tuple[int, int] | None
    node_conflict_loads: FloatArray
    node_conflict_counts: IntArray
    total_conflict_weight: float
    total_conflict_count: int
    total_shortfall: float
    travel_times: FloatArray

    def interval_for_pair(self, uav_pair: tuple[int, int]) -> ConflictInterval | None:
        """Return an interval using an order-insensitive UAV-pair lookup."""
        canonical_pair = tuple(sorted(uav_pair))
        return next(
            (
                interval
                for interval in self.conflict_intervals
                if interval.uav_pair == canonical_pair
            ),
            None,
        )


def build_conflict_graph(
    trajectories: Sequence[Trajectory],
    scenario: Scenario,
    nominal_speed: float | None = None,
    start_delays: ArrayLike | None = None,
    delay_profiles: ArrayLike | None = None,
) -> ConflictGraph:
    """Build MATLAB-compatible temporal conflicts for synchronized trajectories.

    The edge weight is the MATLAB ``edgeWeights`` quantity: cumulative squared
    safe-separation shortfall. ``total_shortfall`` retains the unsquared value.
    """
    paths = tuple(np.asarray(trajectory.points, dtype=float) for trajectory in trajectories)
    if not paths:
        raise ValueError("At least one trajectory is required.")
    speed = scenario.cruise_speed if nominal_speed is None else nominal_speed
    if speed <= 0.0:
        raise ValueError("nominal_speed must be positive.")
    samples = scenario.collision_samples
    delays, profiles = _schedule_arrays(len(paths), samples, start_delays, delay_profiles)
    travel_times = np.asarray([path_length(path) / speed for path in paths], dtype=float)
    node_ids = np.arange(len(paths), dtype=np.int64)
    node_loads = np.zeros(len(paths), dtype=float)
    node_counts = np.zeros(len(paths), dtype=np.int64)
    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    pairwise_minimums: list[float] = []
    intervals: list[ConflictInterval] = []
    fractions = np.linspace(0.0, 1.0, samples)

    for first_index, first_path in enumerate(paths[:-1]):
        first_samples = resample_path(first_path, samples)
        first_times = delays[first_index] + fractions * travel_times[first_index]
        first_times = first_times + profiles[:, first_index]
        for second_index, second_path in enumerate(paths[first_index + 1 :], start=first_index + 1):
            second_samples = resample_path(second_path, samples)
            second_times = delays[second_index] + fractions * travel_times[second_index]
            second_times = second_times + profiles[:, second_index]
            interval = _pair_interval(
                first_index,
                second_index,
                first_samples,
                second_samples,
                first_times,
                second_times,
                fractions,
                scenario.safe_separation,
            )
            if interval is None:
                continue
            edge_pairs.append(interval.uav_pair)
            edge_weights.append(interval.conflict_weight)
            pairwise_minimums.append(interval.minimum_distance)
            intervals.append(interval)
            node_loads[first_index] += interval.conflict_weight
            node_loads[second_index] += interval.conflict_weight
            node_counts[first_index] += interval.sample_count
            node_counts[second_index] += interval.sample_count

    pair_array = (
        np.asarray(edge_pairs, dtype=np.int64).T if edge_pairs else np.empty((2, 0), dtype=np.int64)
    )
    weight_array = np.asarray(edge_weights, dtype=float)
    minimum_array = np.asarray(pairwise_minimums, dtype=float)
    worst_pair = _worst_pair(edge_pairs, edge_weights)
    return ConflictGraph(
        node_ids=node_ids,
        edge_index=pair_array,
        edge_weights=weight_array,
        conflict_intervals=tuple(intervals),
        minimum_pairwise_distances=minimum_array,
        worst_conflict_pair=worst_pair,
        node_conflict_loads=node_loads,
        node_conflict_counts=node_counts,
        total_conflict_weight=float(weight_array.sum()),
        total_conflict_count=int(sum(interval.sample_count for interval in intervals)),
        total_shortfall=float(sum(interval.total_shortfall for interval in intervals)),
        travel_times=travel_times,
    )


def _pair_interval(
    first_index: int,
    second_index: int,
    first_samples: FloatArray,
    second_samples: FloatArray,
    first_times: FloatArray,
    second_times: FloatArray,
    fractions: FloatArray,
    safe_separation: float,
) -> ConflictInterval | None:
    overlap_start = max(first_times[0], second_times[0])
    overlap_end = min(first_times[-1], second_times[-1])
    if overlap_end <= overlap_start:
        return None
    common_times = np.linspace(overlap_start, overlap_end, len(fractions))
    synced_first = _interpolate_path(first_times, first_samples, common_times)
    synced_second = _interpolate_path(second_times, second_samples, common_times)
    distances = np.linalg.norm(synced_first - synced_second, axis=1)
    shortfalls = np.maximum(0.0, safe_separation - distances)
    mask = shortfalls > 0.0
    if not np.any(mask):
        return None
    conflict_indices = np.flatnonzero(mask)
    critical_index = int(np.argmax(shortfalls))
    first_conflict = int(conflict_indices[0])
    last_conflict = int(conflict_indices[-1])
    return ConflictInterval(
        uav_pair=(first_index, second_index),
        start_time=float(common_times[first_conflict]),
        end_time=float(common_times[last_conflict]),
        critical_time=float(common_times[critical_index]),
        start_fractions=(
            _fraction_at_time(first_times, fractions, common_times[first_conflict]),
            _fraction_at_time(second_times, fractions, common_times[first_conflict]),
        ),
        end_fractions=(
            _fraction_at_time(first_times, fractions, common_times[last_conflict]),
            _fraction_at_time(second_times, fractions, common_times[last_conflict]),
        ),
        critical_fractions=(
            _fraction_at_time(first_times, fractions, common_times[critical_index]),
            _fraction_at_time(second_times, fractions, common_times[critical_index]),
        ),
        minimum_distance=float(distances.min()),
        max_shortfall=float(shortfalls.max()),
        total_shortfall=float(shortfalls.sum()),
        conflict_weight=float(np.square(shortfalls).sum()),
        sample_count=int(mask.sum()),
    )


def _schedule_arrays(
    num_uavs: int,
    samples: int,
    start_delays: ArrayLike | None,
    delay_profiles: ArrayLike | None,
) -> tuple[FloatArray, FloatArray]:
    delays = (
        np.zeros(num_uavs, dtype=float)
        if start_delays is None
        else np.asarray(start_delays, dtype=float).reshape(-1)
    )
    if delays.shape != (num_uavs,):
        raise ValueError("start_delays must provide one delay per trajectory.")
    profiles = (
        np.zeros((samples, num_uavs), dtype=float)
        if delay_profiles is None
        else np.asarray(delay_profiles, dtype=float)
    )
    if profiles.shape != (samples, num_uavs):
        raise ValueError("delay_profiles must have shape (collision_samples, num_trajectories).")
    return delays, profiles


def _interpolate_path(times: FloatArray, points: FloatArray, query_times: FloatArray) -> FloatArray:
    monotonic_times = np.maximum.accumulate(times.copy())
    for index in range(1, len(monotonic_times)):
        if monotonic_times[index] <= monotonic_times[index - 1]:
            monotonic_times[index] = monotonic_times[index - 1] + 1e-6
    return np.column_stack(
        [np.interp(query_times, monotonic_times, points[:, axis]) for axis in range(3)]
    )


def _fraction_at_time(times: FloatArray, fractions: FloatArray, query_time: float) -> float:
    monotonic_times = np.maximum.accumulate(times.copy())
    for index in range(1, len(monotonic_times)):
        if monotonic_times[index] <= monotonic_times[index - 1]:
            monotonic_times[index] = monotonic_times[index - 1] + 1e-6
    return float(np.clip(np.interp(query_time, monotonic_times, fractions), 0.0, 1.0))


def _worst_pair(
    edge_pairs: Sequence[tuple[int, int]], edge_weights: Sequence[float]
) -> tuple[int, int] | None:
    if not edge_pairs:
        return None
    maximum = max(edge_weights)
    candidates = [
        pair for pair, weight in zip(edge_pairs, edge_weights, strict=True) if weight == maximum
    ]
    return min(candidates)

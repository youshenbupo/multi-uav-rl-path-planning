"""Deterministic takeoff-delay scheduling driven by temporal conflict graphs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import Scenario, Trajectory
from multiuav.expert.conflict_graph import ConflictGraph, build_conflict_graph

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScheduleLogEntry:
    """One deterministic delay action and its observed graph improvement."""

    iteration: int
    uav_pair: tuple[int, int]
    delayed_uav: int
    delay_increment: float
    total_delay: float
    conflict_weight_before: float
    conflict_weight_after: float
    selection_reason: str


@dataclass(frozen=True)
class ScheduleResult:
    """Final scheduling state, including an explicit unresolved outcome."""

    start_delays: FloatArray
    delay_profiles: FloatArray
    final_graph: ConflictGraph
    log: tuple[ScheduleLogEntry, ...]
    unresolved: bool


class TemporalScheduler:
    """Apply bounded takeoff delays to progressively remove temporal conflicts."""

    def __init__(
        self,
        scenario: Scenario,
        nominal_speed: float | None = None,
        *,
        maximum_delay: float = 45.0,
        delay_step: float = 3.0,
        delay_buffer: float = 1.5,
        maximum_iterations: int = 18,
    ) -> None:
        if maximum_delay < 0.0 or delay_step <= 0.0 or maximum_iterations < 1:
            raise ValueError(
                "Scheduling limits must be non-negative and permit at least one iteration."
            )
        self.scenario = scenario
        self.nominal_speed = scenario.cruise_speed if nominal_speed is None else nominal_speed
        if self.nominal_speed <= 0.0:
            raise ValueError("nominal_speed must be positive.")
        self.maximum_delay = maximum_delay
        self.delay_step = delay_step
        self.delay_buffer = delay_buffer
        self.maximum_iterations = maximum_iterations

    def schedule(self, trajectories: Sequence[Trajectory]) -> ScheduleResult:
        """Schedule bounded start delays, rebuilding the conflict graph each round."""
        paths = tuple(trajectories)
        if not paths:
            raise ValueError("At least one trajectory is required.")
        delays = np.zeros(len(paths), dtype=float)
        profiles = np.zeros((self.scenario.collision_samples, len(paths)), dtype=float)
        graph = build_conflict_graph(paths, self.scenario, self.nominal_speed, delays, profiles)
        entries: list[ScheduleLogEntry] = []

        for iteration in range(1, self.maximum_iterations + 1):
            if graph.worst_conflict_pair is None:
                break
            pair = graph.worst_conflict_pair
            interval = graph.interval_for_pair(pair)
            if interval is None:
                break
            delayed_uav, reason = self._select_delay_target(pair, graph, delays)
            base_step = max(self.delay_step, self.scenario.safe_separation / self.nominal_speed)
            requested_increment = max(
                base_step,
                interval.max_shortfall / self.nominal_speed + self.delay_buffer,
            )
            remaining = self.maximum_delay - delays[delayed_uav]
            increment = min(requested_increment, max(0.0, remaining))
            if increment <= 1e-9:
                break
            before_weight = graph.total_conflict_weight
            delays[delayed_uav] += increment
            updated_graph = build_conflict_graph(
                paths, self.scenario, self.nominal_speed, delays, profiles
            )
            entries.append(
                ScheduleLogEntry(
                    iteration=iteration,
                    uav_pair=pair,
                    delayed_uav=delayed_uav,
                    delay_increment=float(increment),
                    total_delay=float(delays.sum()),
                    conflict_weight_before=before_weight,
                    conflict_weight_after=updated_graph.total_conflict_weight,
                    selection_reason=reason,
                )
            )
            graph = updated_graph

        return ScheduleResult(
            start_delays=delays,
            delay_profiles=profiles,
            final_graph=graph,
            log=tuple(entries),
            unresolved=graph.worst_conflict_pair is not None,
        )

    def _select_delay_target(
        self,
        pair: tuple[int, int],
        graph: ConflictGraph,
        delays: FloatArray,
    ) -> tuple[int, str]:
        """Mirror MATLAB's tie sequence, including its column-major edge order."""
        first, second = pair[1], pair[0]
        first_load = graph.node_conflict_loads[first]
        second_load = graph.node_conflict_loads[second]
        if first_load > second_load + 1e-9:
            return first, "higher conflict load"
        if second_load > first_load + 1e-9:
            return second, "higher conflict load"
        if delays[first] < delays[second] - 1e-9:
            return first, "equal conflict load; lower accumulated delay"
        if delays[second] < delays[first] - 1e-9:
            return second, "equal conflict load; lower accumulated delay"
        first_time = graph.travel_times[first]
        second_time = graph.travel_times[second]
        if first_time >= second_time:
            return first, "equal conflict load and delay; longer-or-equal travel time"
        return second, "equal conflict load and delay; longer travel time"

"""Unified rule-based conflict coordinator; intentionally not a learned policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import EvaluationResult, Scenario, Trajectory
from multiuav.evaluation.multi_uav import MultiUAVEvaluator
from multiuav.expert.conflict_graph import ConflictGraph, build_conflict_graph
from multiuav.expert.spatial_repair import SpatialRepairer, SpatialRepairLogEntry
from multiuav.expert.temporal_scheduler import ScheduleLogEntry, TemporalScheduler

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoordinationResult:
    """Complete output of schedule-and-repair coordination for fixed paths."""

    repaired_trajectories: tuple[Trajectory, ...]
    start_delays: FloatArray
    conflict_graph: ConflictGraph
    schedule_log: tuple[ScheduleLogEntry, ...]
    repair_log: tuple[SpatialRepairLogEntry, ...]
    evaluation_before: EvaluationResult
    evaluation_after: EvaluationResult
    unresolved: bool


class ConflictAwareCoordinator:
    """Coordinate fixed trajectories through temporal delays and local repair."""

    def __init__(self, scenario: Scenario, nominal_speed: float | None = None) -> None:
        self.scenario = scenario
        self.nominal_speed = scenario.cruise_speed if nominal_speed is None else nominal_speed
        if self.nominal_speed <= 0.0:
            raise ValueError("nominal_speed must be positive.")

    def coordinate(self, trajectories: Sequence[Trajectory]) -> CoordinationResult:
        """Return the required paths, delays, graph, logs, and before/after costs."""
        paths = tuple(trajectories)
        evaluator = MultiUAVEvaluator(self.scenario)
        evaluation_before = evaluator.evaluate(paths)
        schedule = TemporalScheduler(self.scenario, self.nominal_speed).schedule(paths)
        repair = SpatialRepairer(self.scenario, self.nominal_speed).repair(paths)
        final_graph = build_conflict_graph(
            repair.repaired_trajectories,
            self.scenario,
            self.nominal_speed,
            schedule.start_delays,
            schedule.delay_profiles,
        )
        evaluation_after = evaluator.evaluate(
            repair.repaired_trajectories,
            schedule.start_delays,
            schedule.delay_profiles,
        )
        return CoordinationResult(
            repaired_trajectories=repair.repaired_trajectories,
            start_delays=schedule.start_delays,
            conflict_graph=final_graph,
            schedule_log=schedule.log,
            repair_log=repair.log,
            evaluation_before=evaluation_before,
            evaluation_after=evaluation_after,
            unresolved=(
                schedule.unresolved
                or repair.unresolved
                or final_graph.worst_conflict_pair is not None
            ),
        )

"""MATLAB-regression tests for rule-based multi-UAV conflict coordination."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.io import loadmat

from multiuav.core.models import CylindricalThreat, Scenario, TerrainMap, Trajectory, UAVMission
from multiuav.evaluation.multi_uav import MultiUAVEvaluator
from multiuav.expert.conflict_graph import build_conflict_graph
from multiuav.expert.coordinator import ConflictAwareCoordinator
from multiuav.expert.spatial_repair import SpatialRepairer
from multiuav.expert.temporal_scheduler import TemporalScheduler


def _reference() -> dict[str, object]:
    fixture = Path(__file__).resolve().parents[1] / "data/regression/matlab/reference_cases.mat"
    return loadmat(fixture, simplify_cells=True)["referenceCases"]["evaluation"]


def _flat_scenario() -> Scenario:
    terrain = TerrainMap(
        x_grid=np.array([0.0, 1000.0]),
        y_grid=np.array([0.0, 1000.0]),
        heights=np.zeros((2, 2)),
    )
    mission = UAVMission(start=np.zeros(3), goal=np.ones(3))
    return Scenario(
        terrain=terrain,
        threats=(),
        missions=(mission, mission),
        safe_separation=50.0,
        cruise_speed=20.0,
        collision_samples=21,
    )


def _reference_trajectories(reference: dict[str, object]) -> tuple[Trajectory, ...]:
    paths = reference["synchronized_paths"]
    return tuple(Trajectory(np.asarray(path, dtype=float)) for path in paths)


class ConflictCoordinationTests(unittest.TestCase):
    """Lock down deterministic graph, delay, repair, and coordinator behavior."""

    def test_conflict_graph_matches_matlab_synchronized_fixture(self) -> None:
        reference = _reference()
        graph = build_conflict_graph(_reference_trajectories(reference), _flat_scenario(), 20.0)

        self.assertEqual((0, 1), graph.worst_conflict_pair)
        assert_allclose(graph.edge_weights, [reference["synchronized_conflict_weight"]])
        assert_allclose(graph.minimum_pairwise_distances, [10.0])
        self.assertEqual(graph.edge_index.tolist(), [[0], [1]])
        self.assertEqual(graph.conflict_intervals[0].sample_count, 21)
        assert_allclose(graph.node_conflict_loads, [33600.0, 33600.0])

    def test_conflict_graph_represents_an_empty_graph(self) -> None:
        trajectories = (
            Trajectory(np.array([[0, 0, 80], [100, 0, 80]], float)),
            Trajectory(np.array([[0, 500, 80], [100, 500, 80]], float)),
        )
        graph = build_conflict_graph(trajectories, _flat_scenario(), 20.0)

        self.assertEqual(graph.edge_index.shape, (2, 0))
        self.assertEqual(graph.edge_weights.shape, (0,))
        self.assertIsNone(graph.worst_conflict_pair)

    def test_temporal_scheduler_matches_matlab_delay_target_and_amount(self) -> None:
        reference = _reference()
        result = TemporalScheduler(_flat_scenario(), nominal_speed=20.0, delay_step=5.0).schedule(
            _reference_trajectories(reference)
        )

        assert_allclose(result.start_delays, reference["schedule_after_start_delays"])
        self.assertFalse(result.unresolved)
        self.assertEqual(result.final_graph.edge_index.shape, (2, 0))
        self.assertEqual(len(result.log), 1)
        entry = result.log[0]
        self.assertEqual(entry.uav_pair, (0, 1))
        self.assertEqual(entry.delayed_uav, 1)
        self.assertEqual(entry.delay_increment, 5.0)
        self.assertIn("equal", entry.selection_reason)

    def test_scheduler_recomputes_conflicts_and_reports_unresolved(self) -> None:
        base = _flat_scenario()
        scenario = replace(base, missions=(base.missions[0],) * 3)
        trajectories = tuple(
            Trajectory(np.array([[100.0, y, 100.0], [900.0, y, 100.0]]))
            for y in (300.0, 310.0, 320.0)
        )
        resolved = TemporalScheduler(
            scenario,
            nominal_speed=20.0,
            maximum_delay=20.0,
            delay_step=1.0,
        ).schedule(trajectories)
        unresolved = TemporalScheduler(
            scenario,
            nominal_speed=20.0,
            maximum_delay=0.0,
            delay_step=1.0,
        ).schedule(trajectories)

        self.assertGreaterEqual(len(resolved.log), 2)
        self.assertFalse(resolved.unresolved)
        self.assertTrue(unresolved.unresolved)
        self.assertEqual(len(unresolved.log), 0)

    def test_spatial_repair_reports_matlab_conflict_center_and_improves_weight(self) -> None:
        reference = _reference()
        trajectories = _reference_trajectories(reference)
        baseline = build_conflict_graph(trajectories, _flat_scenario(), 20.0)
        result = SpatialRepairer(_flat_scenario()).repair(trajectories)

        self.assertGreaterEqual(len(result.log), 1)
        assert_allclose(result.log[0].conflict_center, [100.0, 305.0, 100.0])
        self.assertEqual(result.graph_after.worst_conflict_pair, (0, 1))
        self.assertLess(result.graph_after.total_conflict_weight, baseline.total_conflict_weight)
        assert_allclose(result.graph_after.minimum_pairwise_distances, [10.0])
        evaluator = MultiUAVEvaluator(_flat_scenario())
        python_before = evaluator.evaluate(trajectories).total_cost
        python_after = evaluator.evaluate(result.repaired_trajectories).total_cost
        self.assertLess(python_after, python_before)
        self.assertLess(
            reference["spatial_repair_fixed_cost_after"],
            reference["spatial_repair_fixed_cost_before"],
        )

    def test_coordinator_returns_the_required_complete_artifact(self) -> None:
        reference = _reference()
        result = ConflictAwareCoordinator(_flat_scenario(), nominal_speed=20.0).coordinate(
            _reference_trajectories(reference)
        )

        self.assertEqual(len(result.repaired_trajectories), 2)
        self.assertEqual(result.start_delays.shape, (2,))
        self.assertIsNotNone(result.conflict_graph)
        self.assertIsInstance(result.schedule_log, tuple)
        self.assertIsInstance(result.repair_log, tuple)
        self.assertGreater(result.evaluation_before.total_cost, 0.0)
        self.assertGreater(result.evaluation_after.total_cost, 0.0)

    def test_spatial_repair_reapplies_point_threat_feasibility(self) -> None:
        reference = _reference()
        scenario = replace(
            _flat_scenario(),
            threats=(CylindricalThreat(500.0, 305.0, 1.0, 200.0),),
        )
        result = SpatialRepairer(scenario).repair(_reference_trajectories(reference))

        self.assertGreaterEqual(len(result.log), 1)
        self.assertIn("threat", result.log[0].feasibility_repairs)
        self.assertTrue(
            MultiUAVEvaluator(scenario)
            .evaluate(result.repaired_trajectories)
            .constraint_details.threat_feasible
        )


if __name__ == "__main__":
    unittest.main()

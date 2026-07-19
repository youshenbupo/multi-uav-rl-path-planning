"""Tests for the strict multi-UAV evaluator and its MATLAB timing convention."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.io import loadmat

from multiuav.core.models import (
    CylindricalThreat,
    Scenario,
    TerrainMap,
    Trajectory,
    UAVMission,
)
from multiuav.evaluation.multi_uav import MultiUAVEvaluator


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


class MultiUAVEvaluatorTests(unittest.TestCase):
    """Validate individual, spatial, and synchronized temporal components."""

    def test_evaluate_returns_all_components_and_strict_failure(self) -> None:
        paths = (
            Trajectory(np.array([[100, 300, 100], [500, 300, 100], [900, 300, 100]], float)),
            Trajectory(np.array([[100, 310, 100], [500, 310, 100], [900, 310, 100]], float)),
        )
        result = MultiUAVEvaluator(_flat_scenario()).evaluate(paths)

        self.assertFalse(result.strict_success)
        self.assertEqual(10.0, result.minimum_separation)
        self.assertEqual(21, result.temporal_conflict_count)
        self.assertGreater(result.cost_breakdown.j_spa, 1e6)
        self.assertGreater(result.cost_breakdown.j_tmp, 0.0)
        self.assertEqual(result.total_cost, result.cost_breakdown.total_cost)

    def test_synchronized_fixture_matches_matlab_conflict_graph(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "data/regression/matlab/reference_cases.mat"
        evaluation = loadmat(fixture, simplify_cells=True)["referenceCases"]["evaluation"]
        paths = tuple(
            Trajectory(np.asarray(path, dtype=float)) for path in evaluation["synchronized_paths"]
        )
        result = MultiUAVEvaluator(_flat_scenario()).evaluate(paths)

        assert_allclose(
            result.minimum_separation,
            evaluation["synchronized_min_distance"],
            rtol=1e-6,
            atol=1e-8,
        )
        assert_allclose(
            result.cost_breakdown.j_tmp / 1800.0,
            evaluation["synchronized_conflict_weight"],
            rtol=1e-6,
            atol=1e-8,
        )
        self.assertEqual(result.temporal_conflict_count, evaluation["synchronized_conflict_count"])

    def test_single_uav_terms_match_exported_matlab_evaluator(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "data/regression/matlab/reference_cases.mat"
        reference = loadmat(fixture, simplify_cells=True)["referenceCases"]
        geometry = reference["geometry"]
        base_scenario = reference["scenarios"][1]
        terrain = TerrainMap(
            x_grid=np.asarray(geometry["terrain_x"], dtype=float),
            y_grid=np.asarray(geometry["terrain_y"], dtype=float),
            heights=np.asarray(geometry["terrain_z_grid"], dtype=float),
        )
        threats = tuple(CylindricalThreat(*row) for row in base_scenario["threats"])
        mission = UAVMission(start=np.zeros(3), goal=np.ones(3))
        evaluator = MultiUAVEvaluator(Scenario(terrain, threats, (mission,)))
        decoded_path = Trajectory(np.asarray(geometry["decoded_path"], dtype=float))
        result = evaluator.evaluate_single(decoded_path)
        expected = reference["evaluation"]["single_uav_breakdown"][0]
        assert_allclose(
            [
                result.path_length,
                result.altitude_penalty,
                result.turn_penalty,
                result.threat_penalty,
                result.cost,
                result.travel_time,
                result.max_turn_degrees,
            ],
            expected,
            rtol=1e-6,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()

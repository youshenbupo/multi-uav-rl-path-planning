"""Deterministic module and integration tests for the restored HGALO optimizer."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from multiuav.core.models import Scenario, TerrainMap, UAVMission
from multiuav.expert.alo import alo_local_exploitation, levy_flight_step
from multiuav.expert.ca_hgalo import HGALOConfig, HGALOPlanner
from multiuav.expert.encoding import build_planning_problem, decode_candidate
from multiuav.expert.gwo import gwo_guided_update
from multiuav.expert.scenarios import load_scenario


def _scenario() -> Scenario:
    terrain = TerrainMap(
        x_grid=np.array([0.0, 1000.0]),
        y_grid=np.array([0.0, 1000.0]),
        heights=np.zeros((2, 2)),
    )
    return Scenario(
        terrain=terrain,
        threats=(),
        missions=(
            UAVMission(np.array([0.0, 0.0, 80.0]), np.array([600.0, 0.0, 80.0])),
            UAVMission(np.array([0.0, 300.0, 80.0]), np.array([600.0, 300.0, 80.0])),
        ),
        safe_separation=50.0,
        collision_samples=21,
    )


class HGALOOptimizerTests(unittest.TestCase):
    """Cover encoding, stochastic steps under explicit seeds, and greedy evolution."""

    def test_seed_encoding_decodes_to_mission_endpoints(self) -> None:
        problem = build_planning_problem(_scenario(), num_waypoints=3)
        paths = decode_candidate(problem.seed, problem)

        self.assertEqual(len(paths), 2)
        for path, mission in zip(paths, _scenario().missions, strict=True):
            assert_allclose(path.points[0], mission.start)
            assert_allclose(path.points[-1], mission.goal)

    def test_s2_yaml_problem_seed_matches_exported_matlab_seed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        _, scenario, num_waypoints = load_scenario(
            root / "configs/scenarios/s2.yaml",
            root / "data/regression/matlab/reference_cases.mat",
        )
        problem = build_planning_problem(scenario, num_waypoints)
        from scipy.io import loadmat

        expected = loadmat(
            root / "data/regression/matlab/reference_cases.mat", simplify_cells=True
        )["referenceCases"]["geometry"]["decode_candidate"]
        assert_allclose(problem.seed, expected, rtol=1e-6, atol=1e-8)

    def test_gwo_and_alo_steps_are_seeded_and_bounded(self) -> None:
        current = np.array([0.2, 0.8, 0.4])
        alpha = np.array([0.9, 0.7, 0.6])
        beta = np.array([0.8, 0.5, 0.3])
        delta = np.array([0.4, 0.3, 0.2])
        lower = np.zeros(3)
        upper = np.ones(3)
        first_rng = np.random.default_rng(123)
        second_rng = np.random.default_rng(123)
        first = gwo_guided_update(current, alpha, beta, delta, 2, 10, lower, upper, first_rng)
        second = gwo_guided_update(current, alpha, beta, delta, 2, 10, lower, upper, second_rng)

        assert_allclose(first, second, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all((first >= lower) & (first <= upper)))
        first_levy = levy_flight_step(3, 1.5, np.random.default_rng(7))
        second_levy = levy_flight_step(3, 1.5, np.random.default_rng(7))
        assert_allclose(first_levy, second_levy)
        local = alo_local_exploitation(
            first,
            alpha,
            2,
            10,
            lower,
            upper,
            np.random.default_rng(9),
            HGALOConfig(pop_size=4, max_iter=3),
        )
        self.assertTrue(np.all((local >= lower) & (local <= upper)))

    def test_small_seeded_run_has_monotonic_global_elite_history_and_logs(self) -> None:
        config = HGALOConfig(pop_size=6, max_iter=4, seed=42)
        result = HGALOPlanner(_scenario(), num_waypoints=3, config=config).run()

        self.assertEqual(result.convergence_history.shape, (4,))
        self.assertTrue(np.all(np.diff(result.convergence_history) <= 1e-8))
        self.assertEqual(len(result.best_trajectories), 2)
        self.assertEqual(len(result.generation_stats), 4)
        self.assertIsNotNone(result.coordination.conflict_graph)


if __name__ == "__main__":
    unittest.main()

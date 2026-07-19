"""Behavioral tests for the Phase-13 execution-only CBF safety layer."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multiuav.core.models import DynamicCylinder, Scenario, TerrainMap, UAVMission
from multiuav.envs.communication import AgentKnowledgeState
from multiuav.envs.dynamic_world import DynamicWorldState
from multiuav.envs.observations import EnvironmentSnapshot
from multiuav.safety.action_adapter import NormalizedActionCBFAdapter
from multiuav.safety.cbf_constraints import CBFConfig, CBFConstraintBuilder, load_cbf_config
from multiuav.safety.qp_filter import OSQPSafetyFilter
from scripts.test_cbf_scenarios import run_scenarios


class CBFSafetyTests(unittest.TestCase):
    """Exercise externally observable barrier and filter behavior."""

    def test_pair_constraint_uses_opposite_joint_control_coefficients(self) -> None:
        snapshot = _snapshot(np.array([[20.0, 50.0, 40.0], [40.0, 50.0, 40.0]]))

        rows = CBFConstraintBuilder(CBFConfig()).build(snapshot)
        separation = next(row for row in rows if row.kind == "uav_separation")

        self.assertTrue(np.allclose(separation.coefficients[:3], -separation.coefficients[3:]))
        self.assertLess(separation.barrier, 0.0)

    def test_pairwise_margin_grows_with_delivered_information_uncertainty(self) -> None:
        snapshot = _snapshot(np.array([[20.0, 50.0, 40.0], [60.0, 50.0, 40.0]]))
        knowledge = AgentKnowledgeState(
            positions=snapshot.positions,
            velocities=np.zeros((2, 3)),
            valid=np.array([True, True]),
            ages=np.array([0, 2]),
            predicted_positions=snapshot.positions,
            position_uncertainty=np.array([0.0, 4.0]),
        )
        uncertain_snapshot = replace(snapshot, knowledge_states=(knowledge, knowledge))
        config = CBFConfig(communication_uncertainty_margin_gain=1.0)

        certain_row = next(
            row
            for row in CBFConstraintBuilder(config).build(snapshot)
            if row.kind == "uav_separation"
        )
        uncertain_row = next(
            row
            for row in CBFConstraintBuilder(config).build(uncertain_snapshot)
            if row.kind == "uav_separation"
        )

        self.assertLess(uncertain_row.barrier, certain_row.barrier)

    def test_joint_qp_changes_a_head_on_command_before_separation_is_lost(self) -> None:
        snapshot = _snapshot(np.array([[33.0, 50.0, 40.0], [67.0, 50.0, 40.0]]))
        requested = np.array([[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]])

        decision = OSQPSafetyFilter(CBFConfig(max_solve_time_seconds=0.2)).filter(
            snapshot, requested
        )
        separation = next(
            row
            for row in CBFConstraintBuilder(CBFConfig(max_solve_time_seconds=0.2)).build(snapshot)
            if row.kind == "uav_separation"
        )

        self.assertFalse(decision.emergency_fallback_used)
        self.assertIn("solved", decision.solver_status)
        self.assertGreater(decision.intervention_norm, 0.0)
        self.assertGreaterEqual(
            float(separation.coefficients @ decision.u_safe.reshape(-1)) + decision.slack_value,
            separation.lower - 1e-4,
        )

    def test_dynamic_obstacle_row_accounts_for_relative_velocity(self) -> None:
        snapshot = _dynamic_snapshot()

        rows = CBFConstraintBuilder(CBFConfig()).build(snapshot)
        obstacle_row = next(row for row in rows if row.kind == "dynamic_cylinder_crossing")

        self.assertTrue(np.allclose(obstacle_row.coefficients[:3], [20.0, 0.0, 0.0]))
        self.assertEqual(obstacle_row.barrier, 0.0)
        self.assertEqual(obstacle_row.lower, 20.0)

    def test_qp_reports_dynamic_constraint_count(self) -> None:
        decision = OSQPSafetyFilter(CBFConfig(max_solve_time_seconds=0.2)).filter(
            _dynamic_snapshot(), np.zeros((1, 3), dtype=float)
        )

        self.assertFalse(decision.emergency_fallback_used)
        self.assertEqual(decision.dynamic_constraint_count, 1)

    def test_infeasible_qp_uses_bounded_emergency_action_not_raw_action(self) -> None:
        snapshot = _snapshot(np.array([[2.0, 50.0, 5.0], [20.0, 50.0, 5.0]]))
        requested = np.array([[-20.0, 0.0, -10.0], [20.0, 0.0, 0.0]])
        solver = SimpleNamespace(
            setup=lambda **_: None,
            solve=lambda: SimpleNamespace(info=SimpleNamespace(status="primal infeasible"), x=None),
        )

        with patch("multiuav.safety.qp_filter.osqp.OSQP", return_value=solver):
            decision = OSQPSafetyFilter(CBFConfig(max_solve_time_seconds=0.2)).filter(
                snapshot, requested
            )

        self.assertTrue(decision.emergency_fallback_used)
        self.assertEqual(decision.solver_status, "primal infeasible")
        self.assertFalse(np.allclose(decision.u_safe, requested))
        self.assertLessEqual(np.linalg.norm(decision.u_safe[0, :2]), snapshot.max_horizontal_speed)
        self.assertLessEqual(abs(decision.u_safe[0, 2]), snapshot.max_vertical_speed)

    def test_normalized_adapter_preserves_action_contract_after_filtering(self) -> None:
        snapshot = _snapshot(np.array([[33.0, 50.0, 40.0], [67.0, 50.0, 40.0]]))
        normalized = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

        filtered, decision = NormalizedActionCBFAdapter(
            OSQPSafetyFilter(CBFConfig(max_solve_time_seconds=0.2))
        ).filter_normalized(snapshot, normalized)

        self.assertEqual(filtered.shape, normalized.shape)
        self.assertTrue(np.all(np.abs(filtered) <= 1.0))
        self.assertGreater(decision.intervention_norm, 0.0)

    def test_required_cbf_scenario_runner_reports_all_nine_cases(self) -> None:
        reports = run_scenarios(CBFConfig(max_solve_time_seconds=0.2))

        self.assertEqual(
            set(reports),
            {
                "head_on",
                "crossing",
                "three_way_convergence",
                "near_terrain",
                "near_cylindrical_threat",
                "initially_unsafe",
                "multiple_conflicts",
                "forced_infeasible_qp",
                "low_risk_preservation",
            },
        )
        self.assertTrue(all(report["passed"] for report in reports.values()))

    def test_cbf_yaml_defines_direct_solver_safety_settings(self) -> None:
        config = load_cbf_config(Path("configs/safety/cbf.yaml"))

        self.assertEqual(config.horizontal_speed_polygon_sides, 16)
        self.assertGreater(config.slack_penalty, 0.0)


def _snapshot(positions: np.ndarray) -> EnvironmentSnapshot:
    terrain = TerrainMap(
        x_grid=np.array([0.0, 100.0]),
        y_grid=np.array([0.0, 100.0]),
        heights=np.zeros((2, 2)),
    )
    scenario = Scenario(
        terrain=terrain,
        threats=(),
        missions=tuple(
            UAVMission(start=point.copy(), goal=np.array([90.0, 50.0, 40.0]))
            for point in positions
        ),
        safe_separation=30.0,
        min_clearance=10.0,
        world_x=(0.0, 100.0),
        world_y=(0.0, 100.0),
        world_z=(0.0, 100.0),
    )
    count = len(positions)
    return EnvironmentSnapshot(
        scenario=scenario,
        positions=positions.astype(float),
        velocities=np.zeros_like(positions, dtype=float),
        active_mask=np.ones(count, dtype=bool),
        previous_goal_distances=np.ones(count),
        step_count=0,
        max_steps=100,
        max_horizontal_speed=20.0,
        max_vertical_speed=10.0,
        boundary_clipped=np.zeros(count, dtype=bool),
    )


def _dynamic_snapshot() -> EnvironmentSnapshot:
    terrain = TerrainMap(
        x_grid=np.array([0.0, 100.0]),
        y_grid=np.array([0.0, 100.0]),
        heights=np.zeros((2, 2)),
    )
    obstacle = DynamicCylinder(
        identifier="crossing",
        initial_center=np.array([0.0, 50.0, 40.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        radius=5.0,
        height=80.0,
    )
    scenario = Scenario(
        terrain=terrain,
        threats=(),
        missions=(
            UAVMission(
                start=np.array([10.0, 50.0, 40.0]), goal=np.array([90.0, 50.0, 40.0])
            ),
        ),
        safe_separation=30.0,
        min_clearance=10.0,
        threat_margin=5.0,
        world_x=(0.0, 100.0),
        world_y=(0.0, 100.0),
        world_z=(0.0, 100.0),
        dynamic_obstacles=(obstacle,),
    )
    return EnvironmentSnapshot(
        scenario=scenario,
        positions=np.array([[10.0, 50.0, 40.0]]),
        velocities=np.zeros((1, 3)),
        active_mask=np.array([True]),
        previous_goal_distances=np.ones(1),
        step_count=0,
        max_steps=100,
        max_horizontal_speed=20.0,
        max_vertical_speed=10.0,
        boundary_clipped=np.zeros(1, dtype=bool),
        dynamic_world=DynamicWorldState((obstacle,), dt=1.0),
    )


if __name__ == "__main__":
    unittest.main()

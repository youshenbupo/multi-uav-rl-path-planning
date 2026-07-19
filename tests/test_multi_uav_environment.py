"""Phase-8 multi-UAV environment MVP behavior tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from multiuav.core.models import (
    CylindricalThreat,
    DynamicCylinder,
    Scenario,
    TerrainMap,
    UAVMission,
)
from multiuav.envs.dynamics import SingleIntegrator3D
from multiuav.envs.multi_uav_env import EnvironmentConfig, MultiUAVParallelEnv
from multiuav.envs.wrappers import EpisodeTrajectoryRecorder
from scripts.visualize_environment_episode import build_goal_directed_actions


def _scenario(
    missions: tuple[UAVMission, ...],
    *,
    terrain_height: float = 0.0,
    threats: tuple[CylindricalThreat, ...] = (),
    dynamic_obstacles: tuple[DynamicCylinder, ...] = (),
) -> Scenario:
    """Create a small deterministic world for environment unit tests."""
    terrain = TerrainMap(
        x_grid=np.array([0.0, 100.0]),
        y_grid=np.array([0.0, 100.0]),
        heights=np.full((2, 2), terrain_height),
    )
    return Scenario(
        terrain=terrain,
        threats=threats,
        missions=missions,
        min_clearance=10.0,
        max_clearance=80.0,
        target_clearance=30.0,
        safe_separation=20.0,
        threat_margin=5.0,
        world_x=(0.0, 100.0),
        world_y=(0.0, 100.0),
        world_z=(0.0, 100.0),
        dynamic_obstacles=dynamic_obstacles,
    )


def _config(**overrides: object) -> EnvironmentConfig:
    """Return compact values so each test reaches its intended condition quickly."""
    values: dict[str, object] = {
        "dt": 1.0,
        "max_steps": 20,
        "max_horizontal_speed": 10.0,
        "max_vertical_speed": 10.0,
        "normalize_actions": True,
        "goal_radius": 2.0,
        "collision_distance": 5.0,
        "severe_clearance_shortfall": 4.0,
        "severe_threat_penetration": 3.0,
        "max_neighbors": 2,
    }
    values.update(overrides)
    return EnvironmentConfig(**values)


def _one_mission(start: list[float], goal: list[float]) -> tuple[UAVMission, ...]:
    return (UAVMission(np.asarray(start, dtype=float), np.asarray(goal, dtype=float)),)


class EnvironmentTests(unittest.TestCase):
    """Verify all required Phase-8 physics, interface, safety, and rendering behavior."""

    def test_single_integrator_clips_normalized_action_and_world_bounds(self) -> None:
        dynamics = SingleIntegrator3D(dt=1.0, max_horizontal_speed=10.0, max_vertical_speed=3.0)
        result = dynamics.advance(
            np.array([95.0, 95.0, 99.0]),
            np.array([1.0, 1.0, 1.0]),
            ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0)),
        )

        assert_allclose(result.velocity, [10.0 / np.sqrt(2.0), 10.0 / np.sqrt(2.0), 3.0])
        assert_allclose(result.position, [100.0, 100.0, 100.0])
        self.assertTrue(result.boundary_clipped)

    def test_reset_and_step_return_parallel_api_shapes_and_central_state(self) -> None:
        env = MultiUAVParallelEnv(_scenario(_one_mission([10, 10, 30], [90, 10, 30])), _config())
        observations, infos = env.reset(seed=7)

        self.assertEqual(set(observations), {"uav_0"})
        self.assertEqual(observations["uav_0"].shape, (29,))
        self.assertEqual(observations["uav_0"].dtype, np.float32)
        self.assertTrue(env.observation_space("uav_0").contains(observations["uav_0"]))
        self.assertEqual(env.state().shape, (16,))
        self.assertIn("central_state", infos["uav_0"])

        next_observations, rewards, terminations, truncations, next_infos = env.step(
            {"uav_0": np.zeros(3, dtype=np.float32)}
        )

        self.assertEqual(set(next_observations), {"uav_0"})
        self.assertEqual(set(rewards), {"uav_0"})
        self.assertFalse(terminations["uav_0"])
        self.assertFalse(truncations["uav_0"])
        self.assertEqual(
            set(next_infos["uav_0"]["safety_costs"]),
            {
                "inter_uav_collision_cost",
                "terrain_violation_cost",
                "threat_violation_cost",
                "dynamic_obstacle_violation_cost",
                "boundary_violation_cost",
            },
        )

    def test_fixed_seed_reproduces_reset_and_transition(self) -> None:
        scenario = _scenario(
            (
                UAVMission(np.array([10.0, 10.0, 30.0]), np.array([90.0, 10.0, 30.0])),
                UAVMission(np.array([10.0, 40.0, 30.0]), np.array([90.0, 40.0, 30.0])),
            )
        )
        first = MultiUAVParallelEnv(scenario, _config())
        second = MultiUAVParallelEnv(scenario, _config())
        initial_first, _ = first.reset(seed=123)
        initial_second, _ = second.reset(seed=123)
        actions = {agent: np.array([0.5, 0.0, 0.0], dtype=np.float32) for agent in first.agents}
        transition_first = first.step(actions)
        transition_second = second.step(actions)

        for agent in first.possible_agents:
            assert_allclose(initial_first[agent], initial_second[agent])
            assert_allclose(transition_first[0][agent], transition_second[0][agent])
            self.assertEqual(transition_first[1][agent], transition_second[1][agent])

    def test_straight_goal_flight_and_zero_action(self) -> None:
        scenario = _scenario(_one_mission([10, 10, 30], [90, 10, 30]))
        env = MultiUAVParallelEnv(scenario, _config())
        env.reset(seed=0)
        env.step({"uav_0": np.array([1.0, 0.0, 0.0], dtype=np.float32)})
        assert_allclose(env.positions[0], [20.0, 10.0, 30.0])
        env.step({"uav_0": np.zeros(3, dtype=np.float32)})
        assert_allclose(env.positions[0], [20.0, 10.0, 30.0])
        assert_allclose(env.velocities[0], [0.0, 0.0, 0.0])

    def test_opposing_uavs_trigger_collision_with_separate_cost(self) -> None:
        scenario = _scenario(
            (
                UAVMission(np.array([40.0, 50.0, 30.0]), np.array([60.0, 50.0, 30.0])),
                UAVMission(np.array([60.0, 50.0, 30.0]), np.array([40.0, 50.0, 30.0])),
            )
        )
        env = MultiUAVParallelEnv(scenario, _config())
        env.reset(seed=0)
        _, _, terminations, _, infos = env.step(
            {
                "uav_0": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "uav_1": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            }
        )

        self.assertTrue(all(terminations.values()))
        self.assertEqual(infos["uav_0"]["termination_reason"], "collision")
        self.assertGreater(infos["uav_0"]["safety_costs"]["inter_uav_collision_cost"], 0.0)

    def test_terrain_and_cylindrical_threat_violations_terminate(self) -> None:
        terrain_env = MultiUAVParallelEnv(
            _scenario(_one_mission([10, 10, 15], [90, 10, 15]), terrain_height=10.0),
            _config(),
        )
        terrain_env.reset(seed=0)
        _, _, terrain_terms, _, terrain_infos = terrain_env.step(
            {"uav_0": np.array([0.0, 0.0, -1.0], dtype=np.float32)}
        )
        self.assertTrue(terrain_terms["uav_0"])
        self.assertEqual(terrain_infos["uav_0"]["termination_reason"], "terrain_violation")

        threat = CylindricalThreat(center_x=50.0, center_y=50.0, radius=10.0, height=80.0)
        threat_env = MultiUAVParallelEnv(
            _scenario(_one_mission([40, 50, 30], [90, 50, 30]), threats=(threat,)), _config()
        )
        threat_env.reset(seed=0)
        _, _, threat_terms, _, threat_infos = threat_env.step(
            {"uav_0": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        )
        self.assertTrue(threat_terms["uav_0"])
        self.assertEqual(threat_infos["uav_0"]["termination_reason"], "threat_violation")

    def test_environment_advances_dynamic_obstacle_once_per_step(self) -> None:
        obstacle = DynamicCylinder(
            identifier="crossing",
            initial_center=np.array([30.0, 20.0, 30.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            radius=3.0,
            height=40.0,
        )
        env = MultiUAVParallelEnv(
            _scenario(
                _one_mission([10, 10, 30], [90, 10, 30]), dynamic_obstacles=(obstacle,)
            ),
            _config(max_dynamic_obstacles=1),
        )
        env.reset(seed=4)

        _, _, _, _, infos = env.step({"uav_0": np.zeros(3, dtype=np.float32)})

        self.assertEqual(env.dynamic_world.step_count, 1)
        assert_allclose(infos["uav_0"]["dynamic_obstacles"][0]["center"], [31.0, 20.0, 30.0])

    def test_dynamic_cylinder_violation_has_its_own_cost_and_termination_reason(self) -> None:
        obstacle = DynamicCylinder(
            identifier="blocking",
            initial_center=np.array([20.0, 10.0, 30.0]),
            velocity=np.zeros(3),
            radius=5.0,
            height=60.0,
        )
        env = MultiUAVParallelEnv(
            _scenario(
                _one_mission([10, 10, 30], [90, 10, 30]), dynamic_obstacles=(obstacle,)
            ),
            _config(max_dynamic_obstacles=1),
        )
        env.reset(seed=4)

        _, _, terminations, _, infos = env.step(
            {"uav_0": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        )

        self.assertTrue(terminations["uav_0"])
        self.assertEqual(infos["uav_0"]["termination_reason"], "dynamic_obstacle_violation")
        self.assertGreater(infos["uav_0"]["safety_costs"]["dynamic_obstacle_violation_cost"], 0.0)

    def test_delayed_communication_keeps_hidden_neighbor_out_of_local_observation(self) -> None:
        env = MultiUAVParallelEnv(
            _scenario(
                (
                    UAVMission(np.array([10.0, 10.0, 30.0]), np.array([90.0, 10.0, 30.0])),
                    UAVMission(np.array([30.0, 10.0, 30.0]), np.array([90.0, 10.0, 30.0])),
                )
            ),
            _config(
                max_neighbors=1,
                communication_enabled=True,
                communication_delay_steps=1,
                communication_max_staleness_steps=2,
            ),
        )
        observations, _ = env.reset(seed=4)

        self.assertTrue(np.allclose(observations["uav_0"][15:21], 0.0))
        observations, _, _, _, _ = env.step(
            {agent: np.zeros(3, dtype=np.float32) for agent in env.agents}
        )

        self.assertFalse(np.allclose(observations["uav_0"][15:21], 0.0))

    def test_all_arrived_and_variable_uav_counts(self) -> None:
        for count in (1, 2, 4):
            missions = tuple(
                UAVMission(
                    np.array([10.0, 10.0 + 20.0 * index, 30.0]),
                    np.array([10.0, 10.0 + 20.0 * index, 30.0]),
                )
                for index in range(count)
            )
            env = MultiUAVParallelEnv(_scenario(missions), _config())
            observations, _ = env.reset(seed=0)
            _, _, terminations, _, infos = env.step(
                {agent: np.zeros(3, dtype=np.float32) for agent in env.agents}
            )

            self.assertEqual(len(observations), count)
            self.assertEqual(env.state().shape, (10 * count + 6,))
            self.assertTrue(all(terminations.values()))
            self.assertEqual(infos["uav_0"]["termination_reason"], "all_arrived")

    def test_max_steps_truncates_and_numerical_state_terminates(self) -> None:
        scenario = _scenario(_one_mission([10, 10, 30], [90, 10, 30]))
        truncated = MultiUAVParallelEnv(scenario, _config(max_steps=1))
        truncated.reset(seed=0)
        _, _, truncation_terms, truncations, truncation_infos = truncated.step(
            {"uav_0": np.zeros(3, dtype=np.float32)}
        )
        self.assertFalse(truncation_terms["uav_0"])
        self.assertTrue(truncations["uav_0"])
        self.assertEqual(truncation_infos["uav_0"]["termination_reason"], "max_steps")

        numerical = MultiUAVParallelEnv(scenario, _config())
        numerical.reset(seed=0)
        numerical.positions[0, 0] = np.nan
        _, _, numerical_terms, _, numerical_infos = numerical.step(
            {"uav_0": np.zeros(3, dtype=np.float32)}
        )
        self.assertTrue(numerical_terms["uav_0"])
        self.assertEqual(numerical_infos["uav_0"]["termination_reason"], "numerical_error")

    def test_recorder_retains_paths_and_renders_required_figures(self) -> None:
        env = EpisodeTrajectoryRecorder(
            MultiUAVParallelEnv(_scenario(_one_mission([10, 10, 30], [30, 10, 30])), _config())
        )
        env.reset(seed=0)
        env.step({"uav_0": np.array([1.0, 0.0, 0.0], dtype=np.float32)})
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "episode.png"
            env.save_summary(output)

            self.assertEqual(len(env.positions_history), 2)
            self.assertTrue(np.isfinite(env.minimum_separation_history).all())
            self.assertTrue(output.is_file())

    def test_goal_directed_visualization_policy_respects_normalized_action_space(self) -> None:
        environment = MultiUAVParallelEnv(
            _scenario(_one_mission([10, 10, 30], [90, 10, 40])), _config()
        )
        environment.reset(seed=0)

        actions = build_goal_directed_actions(environment)

        self.assertEqual(set(actions), {"uav_0"})
        self.assertTrue(environment.action_space("uav_0").contains(actions["uav_0"]))
        self.assertGreater(actions["uav_0"][0], 0.0)


if __name__ == "__main__":
    unittest.main()

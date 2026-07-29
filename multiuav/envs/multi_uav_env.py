"""PettingZoo parallel multi-UAV environment using the Phase-8 single integrator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from multiuav.core.models import Scenario
from multiuav.envs.communication import CommunicationChannel, CommunicationConfig
from multiuav.envs.dynamic_world import DynamicWorldState, validate_dynamic_trajectories
from multiuav.envs.dynamics import SingleIntegrator3D
from multiuav.envs.observations import (
    EnvironmentSnapshot,
    build_centralized_state,
    build_local_observation,
    local_observation_size,
    pairwise_conflict_summary,
)
from multiuav.envs.rewards import (
    PerformanceReward,
    SafetyCosts,
    compute_performance_reward,
    compute_safety_costs,
)
from multiuav.geometry.terrain import terrain_height


@dataclass(frozen=True)
class EnvironmentConfig:
    """Explicit, serializable Phase-8 environment constants."""

    dt: float = 1.0
    max_steps: int = 200
    max_horizontal_speed: float = 35.0
    max_vertical_speed: float = 12.0
    normalize_actions: bool = True
    goal_radius: float = 20.0
    collision_distance: float = 20.0
    severe_clearance_shortfall: float = 20.0
    severe_threat_penetration: float = 10.0
    max_neighbors: int = 3
    max_dynamic_obstacles: int = 0
    communication_enabled: bool = False
    communication_range: float = float("inf")
    communication_delay_steps: int = 0
    communication_drop_probability: float = 0.0
    communication_max_staleness_steps: int = 0
    communication_uncertainty_growth_per_step: float = 0.0
    rewards: dict[str, float] = field(
        default_factory=lambda: {
            "progress": 1.0,
            "goal_arrival": 25.0,
            "path_efficiency": 0.1,
            "smoothness": 0.05,
            "energy": 0.02,
            "time": 0.01,
        }
    )
    safety_costs: dict[str, float] = field(
        default_factory=lambda: {
            "collision": 1.0,
            "terrain": 1.0,
            "threat": 1.0,
            "dynamic_obstacle": 1.0,
            "boundary": 1.0,
        }
    )

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.max_steps < 1:
            raise ValueError("dt must be positive and max_steps must be at least one.")
        if self.max_horizontal_speed <= 0.0 or self.max_vertical_speed <= 0.0:
            raise ValueError("Speed limits must be positive.")
        if self.goal_radius < 0.0 or self.collision_distance < 0.0:
            raise ValueError("Goal and collision radii must be nonnegative.")
        if self.severe_clearance_shortfall < 0.0 or self.severe_threat_penetration < 0.0:
            raise ValueError("Severe safety thresholds must be nonnegative.")
        if self.max_neighbors < 0 or self.max_dynamic_obstacles < 0:
            raise ValueError("Observation entity budgets must be nonnegative.")
        self.communication_config()
        required_rewards = {
            "progress",
            "goal_arrival",
            "path_efficiency",
            "smoothness",
            "energy",
            "time",
        }
        if set(self.rewards) != required_rewards:
            raise ValueError("Reward mapping has missing or unsupported keys.")
        required_costs = {"collision", "terrain", "threat", "dynamic_obstacle", "boundary"}
        if set(self.safety_costs) != required_costs:
            raise ValueError("Safety-cost mapping has missing or unsupported keys.")

    def communication_config(self) -> CommunicationConfig:
        """Project serializable flat settings to the communication runtime configuration."""
        return CommunicationConfig(
            enabled=self.communication_enabled,
            range=self.communication_range,
            delay_steps=self.communication_delay_steps,
            drop_probability=self.communication_drop_probability,
            max_staleness_steps=self.communication_max_staleness_steps,
            prediction_dt=self.dt,
            uncertainty_growth_per_step=self.communication_uncertainty_growth_per_step,
        )


def load_environment_config(path: Path) -> EnvironmentConfig:
    """Load the checked-in YAML environment configuration with strict key validation."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Environment YAML must contain a mapping.")
    return EnvironmentConfig(**dict(payload))


class MultiUAVParallelEnv(ParallelEnv):
    """Simultaneous-action multi-UAV environment with centralized critic state."""

    metadata = {"name": "multiuav_parallel_v0", "render_modes": []}

    def __init__(self, scenario: Scenario, config: EnvironmentConfig) -> None:
        self.scenario = scenario
        self.config = config
        self.possible_agents = [f"uav_{index}" for index in range(len(scenario.missions))]
        if not self.possible_agents:
            raise ValueError("A multi-UAV environment needs at least one mission.")
        self.agent_name_mapping = {agent: index for index, agent in enumerate(self.possible_agents)}
        self.agents: list[str] = []
        self.dynamics = SingleIntegrator3D(
            dt=config.dt,
            max_horizontal_speed=config.max_horizontal_speed,
            max_vertical_speed=config.max_vertical_speed,
            normalize_actions=config.normalize_actions,
        )
        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.positions = np.empty((len(self.possible_agents), 3), dtype=float)
        self.velocities = np.zeros_like(self.positions)
        self.active_mask = np.ones(len(self.possible_agents), dtype=bool)
        self.previous_goal_distances = np.zeros(len(self.possible_agents), dtype=float)
        self.boundary_clipped = np.zeros(len(self.possible_agents), dtype=bool)
        self.step_count = 0
        validate_dynamic_trajectories(
            scenario.dynamic_obstacles,
            dt=config.dt,
            max_steps=config.max_steps,
            world_x=scenario.world_x,
            world_y=scenario.world_y,
            world_z=scenario.world_z,
        )
        self.dynamic_world = DynamicWorldState(scenario.dynamic_obstacles, config.dt)
        self.communication_channel: CommunicationChannel | None = None

    @cache
    def observation_space(self, agent: str) -> spaces.Box:
        """Return the stable per-agent local observation space."""
        self._validate_agent_name(agent)
        observation_size = local_observation_size(
            self.config.max_neighbors, self.config.max_dynamic_obstacles
        )
        shape = (observation_size,)
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    @cache
    def action_space(self, agent: str) -> spaces.Box:
        """Return normalized or physical three-dimensional desired-velocity actions."""
        self._validate_agent_name(agent)
        if self.config.normalize_actions:
            low = np.full(3, -1.0, dtype=np.float32)
            high = np.full(3, 1.0, dtype=np.float32)
        else:
            low = np.array(
                [
                    -self.config.max_horizontal_speed,
                    -self.config.max_horizontal_speed,
                    -self.config.max_vertical_speed,
                ],
                dtype=np.float32,
            )
            high = -low
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def state_space(self) -> spaces.Box:
        """Return the fixed centralized-state space for this fixed scenario."""
        size = (
            10 * len(self.possible_agents)
            + 4
            + 4 * len(self.scenario.threats)
            + 8 * self.config.max_dynamic_obstacles
            + 2
        )
        return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset missions, masks, and deterministic random state for a new parallel episode."""
        del options
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.positions = np.asarray(
            [mission.start for mission in self.scenario.missions], dtype=float
        )
        self.velocities = np.zeros_like(self.positions)
        goals = np.asarray([mission.goal for mission in self.scenario.missions], dtype=float)
        self.previous_goal_distances = np.linalg.norm(goals - self.positions, axis=1)
        self.active_mask = self.previous_goal_distances > self.config.goal_radius
        self.boundary_clipped = np.zeros(len(self.possible_agents), dtype=bool)
        self.step_count = 0
        self.dynamic_world = DynamicWorldState(self.scenario.dynamic_obstacles, self.config.dt)
        if self.config.communication_enabled:
            self.communication_channel = CommunicationChannel(
                self.config.communication_config(), self.np_random
            )
            self.communication_channel.reset(len(self.possible_agents))
            self.communication_channel.broadcast(
                positions=self.positions,
                velocities=self.velocities,
                active_mask=self.active_mask,
                step=self.step_count,
            )
            self.communication_channel.deliver(step=self.step_count)
        else:
            self.communication_channel = None
        snapshot = self._snapshot()
        observations = self._observations(snapshot)
        return observations, self._infos(snapshot, "none", {}, {})

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Apply simultaneous desired velocities and return standard Parallel API dictionaries."""
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        current_agents = self.agents[:]
        if set(actions) != set(current_agents):
            raise ValueError("Actions must provide exactly one value for every live agent.")
        old_velocities = self.velocities.copy()
        old_active = self.active_mask.copy()
        self.boundary_clipped = np.zeros(len(self.possible_agents), dtype=bool)
        for agent in current_agents:
            index = self.agent_name_mapping[agent]
            raw_action = np.asarray(actions[agent], dtype=np.float32)
            if not self.action_space(agent).contains(raw_action):
                raise ValueError(f"Action for {agent} is outside its declared action space.")
            if not old_active[index]:
                self.velocities[index] = 0.0
                continue
            result = self.dynamics.advance(
                self.positions[index], np.asarray(raw_action, dtype=float), self._world_bounds
            )
            self.positions[index] = result.position
            self.velocities[index] = result.velocity
            self.boundary_clipped[index] = result.boundary_clipped
        self.step_count += 1
        goals = np.asarray([mission.goal for mission in self.scenario.missions], dtype=float)
        current_goal_distances = np.linalg.norm(goals - self.positions, axis=1)
        newly_arrived = old_active & (current_goal_distances <= self.config.goal_radius)
        self.active_mask = old_active & ~newly_arrived
        self.dynamic_world = self.dynamic_world.advance()
        if (
            self.communication_channel is not None
            and np.isfinite(self.positions).all()
            and np.isfinite(self.velocities).all()
        ):
            self.communication_channel.broadcast(
                positions=self.positions,
                velocities=self.velocities,
                active_mask=self.active_mask,
                step=self.step_count,
            )
            self.communication_channel.deliver(step=self.step_count)
        safety_snapshot = self._snapshot(active_mask=old_active)
        safety_costs = {
            agent: compute_safety_costs(safety_snapshot, self.agent_name_mapping[agent])
            for agent in current_agents
        }
        snapshot = self._snapshot()
        rewards: dict[str, float] = {}
        performance_rewards: dict[str, PerformanceReward] = {}
        for agent in current_agents:
            index = self.agent_name_mapping[agent]
            performance = compute_performance_reward(
                snapshot,
                index,
                newly_arrived=bool(newly_arrived[index]),
                previous_velocity=old_velocities[index],
                weights=self.config.rewards,
            )
            performance_rewards[agent] = performance
            rewards[agent] = performance.total
        reason = self._termination_reason(safety_snapshot)
        is_terminal = reason != "none"
        is_truncated = not is_terminal and self.step_count >= self.config.max_steps
        if is_truncated:
            reason = "max_steps"
        terminations = {agent: is_terminal for agent in current_agents}
        truncations = {agent: is_truncated for agent in current_agents}
        observations = self._observations(snapshot)
        infos = self._infos(snapshot, reason, safety_costs, performance_rewards)
        self.previous_goal_distances = current_goal_distances
        if is_terminal or is_truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Return a copy-safe centralized critic state for all potential UAVs."""
        return build_centralized_state(
            self._snapshot(), max_dynamic_obstacles=self.config.max_dynamic_obstacles
        )

    @property
    def minimum_separation(self) -> float:
        """Return the finite minimum active-UAV separation of the current state."""
        return pairwise_conflict_summary(self._snapshot())[0]

    @property
    def _world_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return self.scenario.world_x, self.scenario.world_y, self.scenario.world_z

    def _snapshot(self, active_mask: np.ndarray | None = None) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            scenario=self.scenario,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            active_mask=self.active_mask.copy() if active_mask is None else active_mask.copy(),
            previous_goal_distances=self.previous_goal_distances.copy(),
            step_count=self.step_count,
            max_steps=self.config.max_steps,
            max_horizontal_speed=self.config.max_horizontal_speed,
            max_vertical_speed=self.config.max_vertical_speed,
            boundary_clipped=self.boundary_clipped.copy(),
            dynamic_world=self.dynamic_world,
            knowledge_states=(
                tuple(
                    self.communication_channel.knowledge_for(index, step=self.step_count)
                    for index in range(len(self.possible_agents))
                )
                if self.communication_channel is not None
                else ()
            ),
        )

    def _observations(self, snapshot: EnvironmentSnapshot) -> dict[str, np.ndarray]:
        return {
            agent: build_local_observation(
                snapshot, self.agent_name_mapping[agent], self.config.max_neighbors
                , self.config.max_dynamic_obstacles
            )
            for agent in self.possible_agents
        }

    def _infos(
        self,
        snapshot: EnvironmentSnapshot,
        reason: str,
        safety_costs: dict[str, SafetyCosts],
        performance_rewards: dict[str, PerformanceReward],
    ) -> dict[str, dict[str, Any]]:
        state = build_centralized_state(
            snapshot, max_dynamic_obstacles=self.config.max_dynamic_obstacles
        )
        default_cost = SafetyCosts()
        return {
            agent: {
                "central_state": state.copy(),
                "active": bool(snapshot.active_mask[self.agent_name_mapping[agent]]),
                "safety_costs": safety_costs.get(agent, default_cost).as_dict(),
                "performance_reward": performance_rewards.get(
                    agent,
                    PerformanceReward(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ).as_dict(),
                "termination_reason": reason,
                "minimum_separation": self.minimum_separation,
                "dynamic_obstacles": self._dynamic_obstacle_info(snapshot),
                "communication": self._communication_info(snapshot, self.agent_name_mapping[agent]),
            }
            for agent in self.possible_agents
        }

    def _termination_reason(self, safety_snapshot: EnvironmentSnapshot) -> str:
        if not np.isfinite(self.positions).all() or not np.isfinite(self.velocities).all():
            return "numerical_error"
        active_points = self.positions[safety_snapshot.active_mask]
        if len(active_points) >= 2:
            for index, point in enumerate(active_points[:-1]):
                if np.any(
                    np.linalg.norm(active_points[index + 1 :] - point, axis=1)
                    < self.config.collision_distance
                ):
                    return "collision"
        for point in self.positions[safety_snapshot.active_mask]:
            ground = float(terrain_height(self.scenario.terrain, point[:2])[0])
            if (
                point[2] - ground
                < self.scenario.min_clearance - self.config.severe_clearance_shortfall
            ):
                return "terrain_violation"
            for threat in self.scenario.threats:
                if point[2] <= threat.height:
                    signed_radial = float(
                        np.linalg.norm(point[:2] - np.array([threat.center_x, threat.center_y]))
                        - threat.radius
                    )
                    if signed_radial < -self.config.severe_threat_penetration:
                        return "threat_violation"
            if safety_snapshot.dynamic_world is not None:
                for center, obstacle in zip(
                    safety_snapshot.dynamic_world.centers,
                    safety_snapshot.dynamic_world.obstacles,
                    strict=True,
                ):
                    if abs(point[2] - center[2]) > obstacle.height / 2.0:
                        continue
                    signed_radial = float(np.linalg.norm(point[:2] - center[:2]) - obstacle.radius)
                    if signed_radial < -self.config.severe_threat_penetration:
                        return "dynamic_obstacle_violation"
        if not np.any(self.active_mask):
            return "all_arrived"
        return "none"

    def _validate_agent_name(self, agent: str) -> None:
        if agent not in self.agent_name_mapping:
            raise ValueError(f"Unknown UAV agent: {agent}")

    def _dynamic_obstacle_info(self, snapshot: EnvironmentSnapshot) -> list[dict[str, Any]]:
        """Return serializable current moving-obstacle truth for experiment audit."""
        if snapshot.dynamic_world is None:
            return []
        return [
            {
                "identifier": obstacle.identifier,
                "center": center.tolist(),
                "velocity": obstacle.velocity.tolist(),
                "radius": obstacle.radius,
                "height": obstacle.height,
            }
            for center, obstacle in zip(
                snapshot.dynamic_world.centers, snapshot.dynamic_world.obstacles, strict=True
            )
        ]

    def _communication_info(
        self, snapshot: EnvironmentSnapshot, index: int
    ) -> dict[str, list[int] | list[bool]]:
        """Expose received-neighbour validity and age without replacing actor observations."""
        if not snapshot.knowledge_states:
            return {"valid": [], "ages": []}
        knowledge = snapshot.knowledge_states[index]
        return {"valid": knowledge.valid.tolist(), "ages": knowledge.ages.tolist()}

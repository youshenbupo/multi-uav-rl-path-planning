"""PettingZoo rollout collection, training orchestration, and deterministic MAPPO evaluation."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from multiuav.core.models import CylindricalThreat, Scenario, TerrainMap, UAVMission
from multiuav.envs.multi_uav_env import EnvironmentConfig, MultiUAVParallelEnv
from multiuav.learning.graph_runner import make_graph_scenario
from multiuav.learning.mappo import MAPPOConfig, MAPPOTrainer, save_checkpoint
from multiuav.learning.networks import CentralizedCritic, SharedGaussianActor
from multiuav.learning.rollout_buffer import RolloutBuffer
from multiuav.learning.telemetry import write_live_training_telemetry
from multiuav.safety import (
    CBFConfig,
    NormalizedActionCBFAdapter,
    OSQPSafetyFilter,
    SafetyFilterTelemetry,
)


@dataclass(frozen=True)
class MAPPOExperimentConfig:
    """YAML-backed configuration for basic MAPPO training and evaluation."""

    seed: int
    num_envs: int
    num_uavs: int
    rollout_length: int
    total_steps: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_ratio: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    batch_size: int
    ppo_epochs: int
    hidden_dims: tuple[int, ...]
    evaluation_interval: int
    checkpoint_interval: int
    normalize_rewards: bool
    obstacle: bool
    communication_enabled: bool = False
    communication_delay_steps: int = 0
    communication_drop_probability: float = 0.0
    communication_max_staleness_steps: int = 0
    communication_uncertainty_growth_per_step: float = 0.0
    dynamic_obstacle_enabled: bool = False
    dynamic_obstacle_velocity_scale: float = 1.0
    cbf_enabled: bool = False
    cbf_slack_penalty: float = 1_000.0
    cbf_max_iterations: int = 20_000
    cbf_communication_uncertainty_margin_gain: float = 0.0
    cbf_max_communication_uncertainty_margin: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.num_envs,
                self.num_uavs,
                self.rollout_length,
                self.total_steps,
                self.batch_size,
                self.ppo_epochs,
                self.evaluation_interval,
                self.checkpoint_interval,
            )
            < 1
        ):
            raise ValueError("MAPPO count fields must be positive.")
        if self.num_uavs < 2:
            raise ValueError("MAPPO requires at least two UAVs.")
        if not self.hidden_dims or any(value < 1 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths.")
        if (
            self.cbf_slack_penalty <= 0.0
            or self.cbf_max_iterations < 1
            or self.dynamic_obstacle_velocity_scale <= 0.0
        ):
            raise ValueError("CBF slack penalty and iteration budget must be positive.")

    def optimizer_config(self) -> MAPPOConfig:
        """Project runner fields to the trainer-only optimizer configuration."""
        return MAPPOConfig(
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_ratio=self.clip_ratio,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            batch_size=self.batch_size,
            ppo_epochs=self.ppo_epochs,
            normalize_rewards=self.normalize_rewards,
        )

    def cbf_config(self) -> CBFConfig:
        """Use the same execution-only CBF configuration as every learned comparison arm."""
        return CBFConfig(
            max_solve_time_seconds=0.1,
            slack_penalty=self.cbf_slack_penalty,
            max_iterations=self.cbf_max_iterations,
            communication_uncertainty_margin_gain=(
                self.cbf_communication_uncertainty_margin_gain
            ),
            max_communication_uncertainty_margin=(
                self.cbf_max_communication_uncertainty_margin
            ),
        )


def load_mappo_experiment_config(path: Path) -> MAPPOExperimentConfig:
    """Load a complete experiment configuration from YAML without implicit defaults."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("MAPPO configuration must be a YAML mapping.")
    payload = dict(values)
    payload["hidden_dims"] = tuple(payload["hidden_dims"])
    return MAPPOExperimentConfig(**payload)


def make_empty_two_uav_scenario() -> Scenario:
    """Return a deterministic, obstacle-free two-UAV curriculum scenario."""
    return _two_uav_scenario(threats=())


def make_cylinder_two_uav_scenario() -> Scenario:
    """Return the same curriculum scenario with one simple cylindrical obstacle."""
    return _two_uav_scenario(
        threats=(CylindricalThreat(center_x=50.0, center_y=50.0, radius=10.0, height=70.0),)
    )


class MAPPOExperiment:
    """Own vector-by-list environments, policy collection, training, logging, and evaluation."""

    def __init__(
        self,
        config: MAPPOExperimentConfig,
        *,
        device: torch.device,
        log_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.device = device
        _seed_everything(config.seed)
        scenario = self._scenario()
        environment_config = EnvironmentConfig(
            dt=1.0,
            max_steps=20,
            max_horizontal_speed=8.0,
            max_vertical_speed=6.0,
            normalize_actions=True,
            goal_radius=5.0,
            collision_distance=6.0 if self._uses_dynamic_protocol() else 8.0,
            severe_clearance_shortfall=8.0,
            severe_threat_penetration=5.0,
            max_neighbors=min(3, config.num_uavs - 1),
            max_dynamic_obstacles=int(config.dynamic_obstacle_enabled),
            communication_enabled=config.communication_enabled,
            communication_range=45.0,
            communication_delay_steps=config.communication_delay_steps,
            communication_drop_probability=config.communication_drop_probability,
            communication_max_staleness_steps=config.communication_max_staleness_steps,
            communication_uncertainty_growth_per_step=(
                config.communication_uncertainty_growth_per_step
            ),
        )
        self.environments = [
            MultiUAVParallelEnv(scenario, environment_config) for _ in range(config.num_envs)
        ]
        self.observations = [
            environment.reset(seed=config.seed + index)[0]
            for index, environment in enumerate(self.environments)
        ]
        observation_dim = self.environments[0].observation_space("uav_0").shape[0]
        state_dim = self.environments[0].state_space.shape[0]
        self.trainer = MAPPOTrainer(
            SharedGaussianActor(observation_dim, 3, config.hidden_dims),
            CentralizedCritic(state_dim, config.hidden_dims),
            config.optimizer_config(),
            device=device,
        )
        self.cbf_adapter = self._make_cbf_adapter() if config.cbf_enabled else None
        self.cbf_telemetry = SafetyFilterTelemetry()
        self.last_evaluation_cbf_telemetry = SafetyFilterTelemetry()
        self.writer = SummaryWriter(log_dir=str(log_dir)) if log_dir is not None else None
        self.total_transitions = 0
        self.reset_counts = [0 for _ in self.environments]
        self.episode_returns = np.zeros(config.num_envs, dtype=float)
        self.episode_path_lengths = np.zeros(config.num_envs, dtype=float)
        self.episode_minimum_separation = np.full(config.num_envs, np.inf, dtype=float)
        self.completed_returns: list[float] = []
        self.completed_successes: list[float] = []
        self.completed_collisions: list[float] = []
        self.completed_path_lengths: list[float] = []
        self.completed_minimum_separations: list[float] = []

    def collect_rollout(self) -> tuple[RolloutBuffer, dict[str, float]]:
        """Collect one full synchronous `[T,E,N]` on-policy rollout from PettingZoo environments."""
        first_environment = self.environments[0]
        buffer = RolloutBuffer(
            rollout_length=self.config.rollout_length,
            num_envs=self.config.num_envs,
            num_agents=self.config.num_uavs,
            observation_dim=first_environment.observation_space("uav_0").shape[0],
            state_dim=first_environment.state_space.shape[0],
            action_dim=3,
            device=self.device,
        )
        raw_observation_batches: list[torch.Tensor] = []
        for _ in range(self.config.rollout_length):
            observations = self._observation_tensor()
            raw_observation_batches.append(
                observations.reshape(-1, observations.shape[-1]).detach().cpu()
            )
            states = self._state_tensor()
            with torch.no_grad():
                normalised_observations = self.trainer.observation_normalizer.normalize(
                    observations
                )
                actions, log_probabilities, _ = self.trainer.actor.sample(
                    normalised_observations.reshape(-1, observations.shape[-1]), deterministic=False
                )
                actions = actions.reshape(self.config.num_envs, self.config.num_uavs, 3)
                log_probabilities = log_probabilities.reshape(
                    self.config.num_envs, self.config.num_uavs
                )
                values = self.trainer.critic(states.reshape(-1, states.shape[-1])).reshape(
                    self.config.num_envs, self.config.num_uavs
                )
            transition = self._step_environments(actions.detach().cpu().numpy())
            next_states = transition["next_states"]
            with torch.no_grad():
                next_values = self.trainer.critic(
                    next_states.reshape(-1, next_states.shape[-1])
                ).reshape(self.config.num_envs, self.config.num_uavs)
            rewards = transition["rewards"]
            if self.config.normalize_rewards:
                self.trainer.reward_normalizer.update(rewards.reshape(-1, 1).detach().cpu())
                rewards = self.trainer.reward_normalizer.normalize(rewards)
            buffer.add(
                # PPO must evaluate each stored action in exactly the normalized
                # observation coordinates that produced its old log probability.
                observations=normalised_observations,
                states=states,
                actions=actions,
                log_probabilities=log_probabilities,
                rewards=rewards,
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                values=values,
                next_values=next_values,
            )
        self.trainer.observation_normalizer.update(torch.cat(raw_observation_batches, dim=0))
        buffer.compute_returns_and_advantages(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
        )
        return buffer, self._rollout_metrics()

    def train(
        self, *, checkpoint_dir: Path | None = None, telemetry_path: Path | None = None
    ) -> list[dict[str, float]]:
        """Run on-policy updates until the configured transition budget is reached."""
        records: list[dict[str, float]] = []
        started = time.perf_counter()
        while self.total_transitions < self.config.total_steps:
            buffer, rollout_metrics = self.collect_rollout()
            update_metrics = self.trainer.update(buffer.flatten())
            increment = self.config.rollout_length * self.config.num_envs * self.config.num_uavs
            self.total_transitions += increment
            record = {
                **rollout_metrics,
                **update_metrics,
                "fps": increment / max(time.perf_counter() - started, 1e-9),
            }
            records.append(record)
            self._log(record)
            if telemetry_path is not None:
                write_live_training_telemetry(
                    telemetry_path,
                    total_transitions=self.total_transitions,
                    cbf=self.cbf_telemetry,
                )
            if (
                checkpoint_dir is not None
                and self.total_transitions % self.config.checkpoint_interval == 0
            ):
                save_checkpoint(
                    checkpoint_dir / f"mappo_step_{self.total_transitions}.pt",
                    self.trainer,
                    step=self.total_transitions,
                )
            if self.total_transitions % self.config.evaluation_interval == 0:
                evaluation = self.evaluate(episodes=4)
                self._log({f"evaluation/{name}": value for name, value in evaluation.items()})
                if telemetry_path is not None:
                    write_live_training_telemetry(
                        telemetry_path,
                        total_transitions=self.total_transitions,
                        cbf=self.cbf_telemetry,
                        extra={
                            "last_interval_evaluation": evaluation,
                            "last_interval_evaluation_cbf": (
                                self.last_evaluation_cbf_telemetry.as_dict()
                            ),
                        },
                    )
        return records

    def evaluate(self, *, episodes: int) -> dict[str, float]:
        """Run deterministic actor-mean evaluation on fresh, explicitly seeded environments."""
        returns: list[float] = []
        successes: list[float] = []
        collisions: list[float] = []
        path_lengths: list[float] = []
        separations: list[float] = []
        cbf_decisions = 0
        cbf_interventions = 0
        cbf_fallbacks = 0
        cbf_solve_times: list[float] = []
        evaluation_telemetry = SafetyFilterTelemetry()
        scenario = self._scenario()
        environment_config = self.environments[0].config
        evaluation_cbf_adapter = self._make_cbf_adapter() if self.config.cbf_enabled else None
        for episode in range(episodes):
            environment = MultiUAVParallelEnv(scenario, environment_config)
            observations, _ = environment.reset(seed=self.config.seed + 10_000 + episode)
            total_return = 0.0
            path_length = 0.0
            minimum_separation = np.inf
            reason = "max_steps"
            while environment.agents:
                current_observations = self._observations_for_environment(observations, environment)
                with torch.no_grad():
                    actions = (
                        self.trainer.actor.act(
                            self.trainer.observation_normalizer.normalize(current_observations),
                            deterministic=True,
                        )
                        .cpu()
                        .numpy()
                    )
                previous_positions = environment.positions.copy()
                if evaluation_cbf_adapter is not None:
                    snapshot = environment._snapshot()
                    actions, decision = evaluation_cbf_adapter.filter_normalized(snapshot, actions)
                    cbf_decisions += 1
                    cbf_interventions += int(decision.intervention_norm > 0.0)
                    cbf_fallbacks += int(decision.emergency_fallback_used)
                    cbf_solve_times.append(decision.solve_time)
                    evaluation_telemetry.record(
                        decision,
                        context={
                            "episode": episode,
                            "environment_step": environment.step_count,
                            "positions": environment.positions.tolist(),
                            "requested_velocities": decision.u_rl.tolist(),
                            "velocities": snapshot.velocities.tolist(),
                            "knowledge_valid": [
                                state.valid.tolist() for state in snapshot.knowledge_states
                            ],
                            "knowledge_uncertainty": [
                                state.position_uncertainty.tolist()
                                for state in snapshot.knowledge_states
                            ],
                            "dynamic_obstacle_centers": (
                                snapshot.dynamic_world.centers.tolist()
                                if snapshot.dynamic_world is not None
                                else []
                            ),
                        },
                    )
                observations, rewards, terminations, truncations, infos = environment.step(
                    {
                        agent: actions[environment.agent_name_mapping[agent]]
                        for agent in environment.possible_agents
                    }
                )
                total_return += float(sum(rewards.values()))
                path_length += float(
                    np.linalg.norm(environment.positions - previous_positions, axis=1).sum()
                )
                minimum_separation = min(minimum_separation, environment.minimum_separation)
                reason = infos["uav_0"]["termination_reason"]
                if all(terminations.values()) or all(truncations.values()):
                    break
            returns.append(total_return)
            successes.append(float(reason == "all_arrived"))
            collisions.append(float(reason == "collision"))
            path_lengths.append(path_length)
            separations.append(float(minimum_separation))
        cbf_intervention_rate = cbf_interventions / cbf_decisions if cbf_decisions else 0.0
        cbf_fallback_rate = cbf_fallbacks / cbf_decisions if cbf_decisions else 0.0
        cbf_mean_solve_time = float(np.mean(cbf_solve_times)) if cbf_solve_times else 0.0
        self.last_evaluation_cbf_telemetry = evaluation_telemetry
        return {
            "episode_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "collision_rate": float(np.mean(collisions)),
            "mean_path_length": float(np.mean(path_lengths)),
            "minimum_separation": float(np.mean(separations)),
            "CBF_intervention_rate": cbf_intervention_rate,
            "CBF_emergency_fallback_rate": cbf_fallback_rate,
            "CBF_mean_solve_time_seconds": cbf_mean_solve_time,
        }

    def close(self) -> None:
        """Flush TensorBoard data if the experiment owns a writer."""
        if self.writer is not None:
            self.writer.close()

    def _step_environments(self, actions: np.ndarray) -> dict[str, torch.Tensor]:
        rewards = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=np.float32)
        terminated = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=bool)
        truncated = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=bool)
        next_states: list[np.ndarray] = []
        for environment_index, environment in enumerate(self.environments):
            previous_positions = environment.positions.copy()
            filtered_actions = actions[environment_index]
            if self.cbf_adapter is not None:
                snapshot = environment._snapshot()
                filtered_actions, decision = self.cbf_adapter.filter_normalized(
                    snapshot, filtered_actions
                )
                self.cbf_telemetry.record(
                    decision,
                    context={
                        "environment_index": environment_index,
                        "environment_step": environment.step_count,
                        "positions": environment.positions.tolist(),
                        "requested_velocities": decision.u_rl.tolist(),
                        "velocities": snapshot.velocities.tolist(),
                        "knowledge_valid": [
                            state.valid.tolist() for state in snapshot.knowledge_states
                        ],
                        "knowledge_uncertainty": [
                            state.position_uncertainty.tolist()
                            for state in snapshot.knowledge_states
                        ],
                        "dynamic_obstacle_centers": (
                            snapshot.dynamic_world.centers.tolist()
                            if snapshot.dynamic_world is not None
                            else []
                        ),
                    },
                )
            next_observations, reward_dict, terminal_dict, truncation_dict, infos = (
                environment.step(
                    {
                        agent: filtered_actions[environment.agent_name_mapping[agent]]
                        for agent in environment.possible_agents
                    }
                )
            )
            rewards[environment_index] = [
                reward_dict[agent] for agent in environment.possible_agents
            ]
            terminated[environment_index] = [
                terminal_dict[agent] for agent in environment.possible_agents
            ]
            truncated[environment_index] = [
                truncation_dict[agent] for agent in environment.possible_agents
            ]
            next_states.append(environment.state())
            self.episode_returns[environment_index] += float(rewards[environment_index].sum())
            self.episode_path_lengths[environment_index] += float(
                np.linalg.norm(environment.positions - previous_positions, axis=1).sum()
            )
            self.episode_minimum_separation[environment_index] = min(
                self.episode_minimum_separation[environment_index], environment.minimum_separation
            )
            is_done = bool(
                terminated[environment_index].all() or truncated[environment_index].all()
            )
            if is_done:
                reason = infos["uav_0"]["termination_reason"]
                self.completed_returns.append(float(self.episode_returns[environment_index]))
                self.completed_successes.append(float(reason == "all_arrived"))
                self.completed_collisions.append(float(reason == "collision"))
                self.completed_path_lengths.append(
                    float(self.episode_path_lengths[environment_index])
                )
                self.completed_minimum_separations.append(
                    float(self.episode_minimum_separation[environment_index])
                )
                self.reset_counts[environment_index] += 1
                self.observations[environment_index], _ = environment.reset(
                    seed=self.config.seed
                    + environment_index
                    + 1000 * self.reset_counts[environment_index]
                )
            else:
                self.observations[environment_index] = next_observations
        next_state_array = np.asarray(next_states)
        repeated_next_states = np.repeat(next_state_array[:, None, :], self.config.num_uavs, axis=1)
        return {
            "rewards": torch.as_tensor(rewards, device=self.device),
            "terminated": torch.as_tensor(terminated, device=self.device),
            "truncated": torch.as_tensor(truncated, device=self.device),
            "next_states": torch.as_tensor(repeated_next_states, device=self.device),
        }

    def _observation_tensor(self) -> torch.Tensor:
        values = [
            self._observations_for_environment(observations, environment).cpu().numpy()
            for observations, environment in zip(self.observations, self.environments, strict=True)
        ]
        return torch.as_tensor(np.asarray(values), device=self.device)

    def _state_tensor(self) -> torch.Tensor:
        states = np.asarray([environment.state() for environment in self.environments])
        repeated = np.repeat(states[:, None, :], self.config.num_uavs, axis=1)
        return torch.as_tensor(repeated, device=self.device)

    def _observations_for_environment(
        self, observations: dict[str, np.ndarray], environment: MultiUAVParallelEnv
    ) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray([observations[agent] for agent in environment.possible_agents]),
            dtype=torch.float32,
            device=self.device,
        )

    def _rollout_metrics(self) -> dict[str, float]:
        completed = max(len(self.completed_returns), 1)
        return {
            "episode_return": float(np.mean(self.completed_returns))
            if self.completed_returns
            else 0.0,
            "success_rate": float(np.sum(self.completed_successes) / completed),
            "collision_rate": float(np.sum(self.completed_collisions) / completed),
            "mean_path_length": float(np.mean(self.completed_path_lengths))
            if self.completed_path_lengths
            else 0.0,
            "minimum_separation": float(np.mean(self.completed_minimum_separations))
            if self.completed_minimum_separations
            else float(np.mean(self.episode_minimum_separation)),
        }

    def _log(self, metrics: dict[str, float]) -> None:
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.total_transitions)

    def _uses_dynamic_protocol(self) -> bool:
        return self.config.num_uavs != 2 or self.config.dynamic_obstacle_enabled

    def _scenario(self) -> Scenario:
        if self._uses_dynamic_protocol():
            return make_graph_scenario(
                num_uavs=self.config.num_uavs,
                obstacle=self.config.obstacle,
                dynamic_obstacle=self.config.dynamic_obstacle_enabled,
                dynamic_obstacle_velocity_scale=self.config.dynamic_obstacle_velocity_scale,
            )
        return (
            make_cylinder_two_uav_scenario()
            if self.config.obstacle
            else make_empty_two_uav_scenario()
        )

    def _make_cbf_adapter(self) -> NormalizedActionCBFAdapter:
        return NormalizedActionCBFAdapter(OSQPSafetyFilter(self.config.cbf_config()))


def _two_uav_scenario(*, threats: tuple[CylindricalThreat, ...]) -> Scenario:
    terrain = TerrainMap(
        x_grid=np.array([0.0, 100.0]),
        y_grid=np.array([0.0, 100.0]),
        heights=np.zeros((2, 2), dtype=float),
    )
    missions = (
        UAVMission(np.array([10.0, 25.0, 30.0]), np.array([35.0, 25.0, 30.0])),
        UAVMission(np.array([10.0, 75.0, 30.0]), np.array([35.0, 75.0, 30.0])),
    )
    return Scenario(
        terrain=terrain,
        threats=threats,
        missions=missions,
        min_clearance=10.0,
        max_clearance=80.0,
        target_clearance=30.0,
        safe_separation=15.0,
        threat_margin=5.0,
        world_x=(0.0, 100.0),
        world_y=(0.0, 100.0),
        world_z=(0.0, 100.0),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

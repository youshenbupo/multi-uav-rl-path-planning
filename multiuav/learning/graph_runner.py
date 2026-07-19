"""Variable-size graph MAPPO rollout collection, training, and deterministic evaluation."""

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
from multiuav.learning.bc_finetuning import BCFineTuneSchedule
from multiuav.learning.conflict_graph import (
    EDGE_FEATURE_DIMENSION,
    ConflictGraph,
    ConflictGraphBuilder,
    GraphBuildConfig,
)
from multiuav.learning.graph_mappo import (
    GraphMAPPOConfig,
    GraphMAPPOTrainer,
    save_graph_checkpoint,
)
from multiuav.learning.graph_networks import (
    ConflictPredictionHead,
    GraphActor,
    GraphCentralizedCritic,
)
from multiuav.learning.graph_rollout_buffer import GraphRolloutBuffer


@dataclass(frozen=True)
class GraphExperimentConfig:
    """Complete YAML configuration for an ordinary, variable-UAV GraphMAPPO run."""

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
    embedding_dim: int
    graph_heads: int
    graph_layers: int
    evaluation_interval: int
    checkpoint_interval: int
    graph_mode: str
    communication_radius: float
    risk_distance: float
    prediction_horizon: float
    top_k_neighbors: int
    auxiliary_horizon: int
    aux_conflict_coef: float
    aux_distance_coef: float
    obstacle: bool

    def __post_init__(self) -> None:
        count_values = (
            self.num_envs,
            self.num_uavs,
            self.rollout_length,
            self.total_steps,
            self.batch_size,
            self.ppo_epochs,
            self.embedding_dim,
            self.graph_heads,
            self.graph_layers,
            self.evaluation_interval,
            self.checkpoint_interval,
            self.auxiliary_horizon,
        )
        if min(count_values) < 1:
            raise ValueError("Graph experiment count fields must be positive.")
        if self.num_uavs < 2:
            raise ValueError("GraphMAPPO requires at least two UAVs.")
        if self.graph_mode not in {"mappo", "distance_graph", "predictive_graph"}:
            raise ValueError("graph_mode must be mappo, distance_graph, or predictive_graph.")
        if self.embedding_dim % self.graph_heads != 0:
            raise ValueError("embedding_dim must be divisible by graph_heads.")

    def optimizer_config(self) -> GraphMAPPOConfig:
        """Project experiment settings to the trainer-only configuration."""
        return GraphMAPPOConfig(
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_ratio=self.clip_ratio,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            batch_size=self.batch_size,
            ppo_epochs=self.ppo_epochs,
            aux_conflict_coef=self.aux_conflict_coef,
            aux_distance_coef=self.aux_distance_coef,
        )

    def graph_build_config(self) -> GraphBuildConfig:
        """Set the required no-graph/distance/predictive edge ablation mode."""
        return GraphBuildConfig(
            communication_radius=self.communication_radius,
            risk_distance=self.risk_distance,
            prediction_horizon=self.prediction_horizon,
            top_k_neighbors=0 if self.graph_mode == "mappo" else self.top_k_neighbors,
            current_distance_edges=self.graph_mode != "mappo",
            predicted_conflict_edges=self.graph_mode == "predictive_graph",
            self_loops=True,
        )


@dataclass(frozen=True)
class _GraphStepTransition:
    """Typed transition output between collection and the next centralized-value call."""

    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_node_features: torch.Tensor
    next_graph: ConflictGraph


def load_graph_experiment_config(path: Path) -> GraphExperimentConfig:
    """Load a complete GraphMAPPO YAML document with no implicit settings."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("GraphMAPPO configuration must be a YAML mapping.")
    return GraphExperimentConfig(**dict(values))


def make_graph_scenario(*, num_uavs: int, obstacle: bool = False) -> Scenario:
    """Create reproducible parallel missions for any supported graph size."""
    if num_uavs < 2:
        raise ValueError("num_uavs must be at least two.")
    terrain = TerrainMap(
        x_grid=np.array([0.0, 100.0]),
        y_grid=np.array([0.0, 100.0]),
        heights=np.zeros((2, 2), dtype=float),
    )
    lanes = np.linspace(15.0, 85.0, num_uavs)
    missions = tuple(
        UAVMission(np.array([10.0, lane, 30.0]), np.array([35.0, lane, 30.0])) for lane in lanes
    )
    threats = (
        (CylindricalThreat(center_x=50.0, center_y=50.0, radius=8.0, height=70.0),)
        if obstacle
        else ()
    )
    return Scenario(
        terrain=terrain,
        threats=threats,
        missions=missions,
        min_clearance=10.0,
        max_clearance=80.0,
        target_clearance=30.0,
        safe_separation=8.0,
        world_x=(0.0, 100.0),
        world_y=(0.0, 100.0),
        world_z=(0.0, 100.0),
    )


class GraphMAPPOExperiment:
    """Manage vectorized graph rollouts while retaining native PettingZoo environments."""

    def __init__(
        self,
        config: GraphExperimentConfig,
        *,
        device: torch.device,
        log_dir: Path | None = None,
        bc_low_checkpoint: Path | None = None,
        bc_schedule: BCFineTuneSchedule | None = None,
    ) -> None:
        self.config = config
        self.device = device
        _seed_everything(config.seed)
        scenario = make_graph_scenario(num_uavs=config.num_uavs, obstacle=config.obstacle)
        environment_config = EnvironmentConfig(
            dt=1.0,
            max_steps=20,
            max_horizontal_speed=8.0,
            max_vertical_speed=6.0,
            normalize_actions=True,
            goal_radius=5.0,
            collision_distance=6.0,
            severe_clearance_shortfall=8.0,
            severe_threat_penetration=5.0,
            max_neighbors=min(3, config.num_uavs - 1),
        )
        self.environments = [
            MultiUAVParallelEnv(scenario, environment_config) for _ in range(config.num_envs)
        ]
        self.observations = [
            environment.reset(seed=config.seed + environment_index)[0]
            for environment_index, environment in enumerate(self.environments)
        ]
        node_feature_dim = self.environments[0].observation_space("uav_0").shape[0]
        self.graph_builder = ConflictGraphBuilder(config.graph_build_config())
        self.trainer = GraphMAPPOTrainer(
            GraphActor(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=EDGE_FEATURE_DIMENSION,
                embedding_dim=config.embedding_dim,
                num_heads=config.graph_heads,
                num_layers=config.graph_layers,
                action_dim=3,
            ),
            GraphCentralizedCritic(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=EDGE_FEATURE_DIMENSION,
                embedding_dim=config.embedding_dim,
                num_heads=config.graph_heads,
                num_layers=config.graph_layers,
            ),
            ConflictPredictionHead(
                embedding_dim=config.embedding_dim, edge_feature_dim=EDGE_FEATURE_DIMENSION
            ),
            config.optimizer_config(),
            device=device,
        )
        if (bc_low_checkpoint is None) != (bc_schedule is None):
            raise ValueError("BC checkpoint and fine-tuning schedule must be provided together.")
        if bc_low_checkpoint is not None and bc_schedule is not None:
            self.trainer.initialize_from_bc(bc_low_checkpoint, bc_schedule)
        self.writer = SummaryWriter(log_dir=str(log_dir)) if log_dir is not None else None
        self.total_transitions = 0
        self.episode_ids = np.zeros(config.num_envs, dtype=np.int64)
        self.episode_returns = np.zeros(config.num_envs, dtype=float)
        self.episode_path_lengths = np.zeros(config.num_envs, dtype=float)
        self.episode_minimum_separation = np.full(config.num_envs, np.inf, dtype=float)
        self.completed_returns: list[float] = []
        self.completed_successes: list[float] = []
        self.completed_collisions: list[float] = []
        self.completed_path_lengths: list[float] = []
        self.completed_minimum_separations: list[float] = []

    def collect_rollout(self) -> tuple[GraphRolloutBuffer, dict[str, float]]:
        """Collect one full synchronous graph rollout without flattening agent axes."""
        node_feature_dim = self.environments[0].observation_space("uav_0").shape[0]
        buffer = GraphRolloutBuffer(
            rollout_length=self.config.rollout_length,
            num_envs=self.config.num_envs,
            num_agents=self.config.num_uavs,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=EDGE_FEATURE_DIMENSION,
            action_dim=3,
            device=self.device,
        )
        for _ in range(self.config.rollout_length):
            node_features = self._node_features()
            graph = self._build_graph()
            positions = self._positions_tensor()
            episode_ids = torch.as_tensor(self.episode_ids.copy(), device=self.device)
            with torch.no_grad():
                actions, log_probabilities, _ = self.trainer.actor.sample(
                    node_features,
                    graph.edge_features,
                    graph.adjacency,
                    graph.node_mask,
                    deterministic=False,
                )
                values = (
                    self.trainer.critic(
                        node_features, graph.edge_features, graph.adjacency, graph.node_mask
                    )
                    .unsqueeze(dim=-1)
                    .expand(-1, self.config.num_uavs)
                )
            transition = self._step_environments(actions.cpu().numpy())
            with torch.no_grad():
                next_values = (
                    self.trainer.critic(
                        transition.next_node_features,
                        transition.next_graph.edge_features,
                        transition.next_graph.adjacency,
                        transition.next_graph.node_mask,
                    )
                    .unsqueeze(dim=-1)
                    .expand(-1, self.config.num_uavs)
                )
            buffer.add(
                node_features=node_features,
                edge_features=graph.edge_features,
                adjacency=graph.adjacency,
                active_mask=graph.node_mask,
                positions=positions,
                episode_ids=episode_ids,
                actions=actions,
                log_probabilities=log_probabilities,
                rewards=transition.rewards,
                terminated=transition.terminated,
                truncated=transition.truncated,
                values=values,
                next_values=next_values,
            )
        buffer.compute_returns_and_advantages(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
        )
        return buffer, self._rollout_metrics()

    def train(self, *, checkpoint_dir: Path | None = None) -> list[dict[str, float]]:
        """Run graph PPO updates to the configured total individual-agent transition count."""
        records: list[dict[str, float]] = []
        started = time.perf_counter()
        while self.total_transitions < self.config.total_steps:
            buffer, rollout_metrics = self.collect_rollout()
            update_metrics = self.trainer.update(
                buffer.flatten(
                    horizon=self.config.auxiliary_horizon,
                    conflict_distance=self.config.risk_distance,
                )
            )
            increment = self.config.rollout_length * self.config.num_envs * self.config.num_uavs
            self.total_transitions += increment
            record = {
                **rollout_metrics,
                **update_metrics,
                "fps": increment / max(time.perf_counter() - started, 1e-9),
            }
            records.append(record)
            self._log(record)
            if (
                checkpoint_dir is not None
                and self.total_transitions % self.config.checkpoint_interval == 0
            ):
                save_graph_checkpoint(
                    checkpoint_dir / f"graph_mappo_step_{self.total_transitions}.pt",
                    self.trainer,
                    step=self.total_transitions,
                )
            if self.total_transitions % self.config.evaluation_interval == 0:
                evaluation = self.evaluate(episodes=4)
                self._log({f"evaluation/{name}": value for name, value in evaluation.items()})
        return records

    def evaluate(self, *, episodes: int) -> dict[str, float]:
        """Evaluate deterministic mean actions on fresh fixed-seed graph environments."""
        returns: list[float] = []
        successes: list[float] = []
        collisions: list[float] = []
        path_lengths: list[float] = []
        separations: list[float] = []
        scenario = make_graph_scenario(num_uavs=self.config.num_uavs, obstacle=self.config.obstacle)
        environment_config = self.environments[0].config
        for episode in range(episodes):
            environment = MultiUAVParallelEnv(scenario, environment_config)
            observations, _ = environment.reset(seed=self.config.seed + 10_000 + episode)
            total_return = 0.0
            path_length = 0.0
            minimum_separation = np.inf
            reason = "max_steps"
            while environment.agents:
                node_features = self._node_features_for([(observations, environment)])
                graph = self._build_graph_for([environment])
                with torch.no_grad():
                    actions, _, _ = self.trainer.actor.sample(
                        node_features,
                        graph.edge_features,
                        graph.adjacency,
                        graph.node_mask,
                        deterministic=True,
                    )
                previous_positions = environment.positions.copy()
                observations, rewards, terminations, truncations, infos = environment.step(
                    {
                        agent: actions[0, environment.agent_name_mapping[agent]].cpu().numpy()
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
        return {
            "episode_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "collision_rate": float(np.mean(collisions)),
            "mean_path_length": float(np.mean(path_lengths)),
            "minimum_separation": float(np.mean(separations)),
        }

    def close(self) -> None:
        """Flush optional TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def _step_environments(self, actions: np.ndarray) -> _GraphStepTransition:
        rewards = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=np.float32)
        terminated = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=bool)
        truncated = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=bool)
        next_observation_pairs: list[tuple[dict[str, np.ndarray], MultiUAVParallelEnv]] = []
        next_environments: list[MultiUAVParallelEnv] = []
        for environment_index, environment in enumerate(self.environments):
            previous_positions = environment.positions.copy()
            next_observations, reward_dict, terminal_dict, truncation_dict, infos = (
                environment.step(
                    {
                        agent: actions[environment_index, environment.agent_name_mapping[agent]]
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
            next_observation_pairs.append((next_observations, environment))
            next_environments.append(environment)
            self.episode_returns[environment_index] += float(rewards[environment_index].sum())
            self.episode_path_lengths[environment_index] += float(
                np.linalg.norm(environment.positions - previous_positions, axis=1).sum()
            )
            self.episode_minimum_separation[environment_index] = min(
                self.episode_minimum_separation[environment_index], environment.minimum_separation
            )
            done = bool(terminated[environment_index].all() or truncated[environment_index].all())
            if done:
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
                self.episode_ids[environment_index] += 1
                self.episode_returns[environment_index] = 0.0
                self.episode_path_lengths[environment_index] = 0.0
                self.episode_minimum_separation[environment_index] = np.inf
                self.observations[environment_index], _ = environment.reset(
                    seed=self.config.seed
                    + environment_index
                    + 1000 * int(self.episode_ids[environment_index])
                )
            else:
                self.observations[environment_index] = next_observations
        return _GraphStepTransition(
            rewards=torch.as_tensor(rewards, device=self.device),
            terminated=torch.as_tensor(terminated, device=self.device),
            truncated=torch.as_tensor(truncated, device=self.device),
            next_node_features=self._node_features_for(next_observation_pairs),
            next_graph=self._build_graph_for(next_environments),
        )

    def _node_features(self) -> torch.Tensor:
        return self._node_features_for(list(zip(self.observations, self.environments, strict=True)))

    def _node_features_for(
        self, observations_and_environments: list[tuple[dict[str, np.ndarray], MultiUAVParallelEnv]]
    ) -> torch.Tensor:
        values = [
            np.asarray([observations[agent] for agent in environment.possible_agents])
            for observations, environment in observations_and_environments
        ]
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)

    def _build_graph(self) -> ConflictGraph:
        return self._build_graph_for(self.environments)

    def _build_graph_for(self, environments: list[MultiUAVParallelEnv]) -> ConflictGraph:
        positions = torch.as_tensor(
            np.asarray([environment.positions for environment in environments]),
            dtype=torch.float32,
            device=self.device,
        )
        velocities = torch.as_tensor(
            np.asarray([environment.velocities for environment in environments]),
            dtype=torch.float32,
            device=self.device,
        )
        goals = torch.as_tensor(
            np.asarray(
                [
                    [mission.goal for mission in environment.scenario.missions]
                    for environment in environments
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        active_mask = torch.as_tensor(
            np.asarray([environment.active_mask for environment in environments]),
            dtype=torch.bool,
            device=self.device,
        )
        return self.graph_builder.build(
            positions=positions, velocities=velocities, goals=goals, active_mask=active_mask
        )

    def _positions_tensor(self) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray([environment.positions for environment in self.environments]),
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

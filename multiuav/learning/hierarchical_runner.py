"""PettingZoo collection for staged hierarchical graph MAPPO training."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from multiuav.envs.multi_uav_env import EnvironmentConfig, MultiUAVParallelEnv
from multiuav.learning.conflict_graph import (
    EDGE_FEATURE_DIMENSION,
    ConflictGraph,
    ConflictGraphBuilder,
    GraphBuildConfig,
)
from multiuav.learning.gae import compute_gae
from multiuav.learning.graph_runner import build_graph_from_environments, make_graph_scenario
from multiuav.learning.hierarchical_mappo import (
    HierarchicalMAPPOConfig,
    HierarchicalMAPPOTrainer,
    HighLevelBatch,
    LowLevelBatch,
)
from multiuav.learning.hierarchical_policy import (
    CoordinationAction,
    HierarchicalConfig,
    HierarchicalPolicy,
    HighLevelActionContext,
    HighLevelActor,
    HighLevelCritic,
    LowLevelActor,
    LowLevelCritic,
)
from multiuav.learning.hierarchical_rollout_buffer import compute_duration_aware_gae
from multiuav.safety import (
    CBFConfig,
    NormalizedActionCBFAdapter,
    OSQPSafetyFilter,
    SafetyFilterTelemetry,
)


@dataclass(frozen=True)
class HierarchicalExperimentConfig:
    """All settings needed for a deterministic staged hierarchical experiment."""

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
    high_interval: int
    communication_radius: float
    risk_distance: float
    prediction_horizon: float
    top_k_neighbors: int
    enable_high_level_policy: bool
    enable_local_subgoal: bool
    enable_priority: bool
    enable_delay: bool
    enable_altitude_maneuver: bool
    allow_joint_finetune: bool
    communication_enabled: bool = False
    communication_delay_steps: int = 0
    communication_drop_probability: float = 0.0
    communication_max_staleness_steps: int = 0
    communication_uncertainty_growth_per_step: float = 0.0
    graph_uncertainty_scale: float = 1.0
    graph_uncertainty_risk_gain: float = 0.0
    dynamic_obstacle_enabled: bool = False
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
                self.embedding_dim,
                self.graph_heads,
                self.graph_layers,
                self.high_interval,
            )
            < 1
        ):
            raise ValueError("Hierarchical experiment count fields must be positive.")
        if self.num_uavs < 2 or self.embedding_dim % self.graph_heads != 0:
            raise ValueError("num_uavs must be >=2 and embedding_dim divisible by graph_heads.")
        if self.cbf_slack_penalty <= 0.0 or self.cbf_max_iterations < 1:
            raise ValueError("CBF slack penalty and iteration budget must be positive.")

    def graph_config(self) -> GraphBuildConfig:
        """Use predictive graph features for both hierarchy levels."""
        return GraphBuildConfig(
            communication_radius=self.communication_radius,
            risk_distance=self.risk_distance,
            prediction_horizon=self.prediction_horizon,
            top_k_neighbors=self.top_k_neighbors,
            current_distance_edges=True,
            predicted_conflict_edges=True,
            self_loops=True,
            uncertainty_scale=self.graph_uncertainty_scale,
            uncertainty_risk_gain=self.graph_uncertainty_risk_gain,
        )

    def policy_config(self) -> HierarchicalConfig:
        """Project experiment toggles to policy state semantics."""
        return HierarchicalConfig(
            high_interval=self.high_interval,
            enable_high_level_policy=self.enable_high_level_policy,
            enable_local_subgoal=self.enable_local_subgoal,
            enable_priority=self.enable_priority,
            enable_delay=self.enable_delay,
            enable_altitude_maneuver=self.enable_altitude_maneuver,
        )

    def trainer_config(self) -> HierarchicalMAPPOConfig:
        """Project experiment optimizer values to staged trainer settings."""
        return HierarchicalMAPPOConfig(
            learning_rate=self.learning_rate,
            clip_ratio=self.clip_ratio,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            batch_size=self.batch_size,
            ppo_epochs=self.ppo_epochs,
            allow_joint_finetune=self.allow_joint_finetune,
        )


def load_hierarchical_experiment_config(path: Path) -> HierarchicalExperimentConfig:
    """Load one complete hierarchical experiment YAML mapping."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Hierarchical MAPPO configuration must be a YAML mapping.")
    return HierarchicalExperimentConfig(**dict(values))


@dataclass
class _PendingHighSample:
    """One graph-level high action accumulating rewards until its next boundary."""

    node_features: Tensor
    edge_features: Tensor
    adjacency: Tensor
    active_mask: Tensor
    actions: Tensor
    log_probabilities: Tensor
    values: Tensor
    discounted_rewards: Tensor
    durations: Tensor
    terminated: Tensor


@dataclass(frozen=True)
class _StepTransition:
    """Typed result of a synchronized vector-environment low-level step."""

    rewards: Tensor
    terminated: Tensor
    next_node_features: Tensor
    next_graph: ConflictGraph


class HierarchicalMAPPOExperiment:
    """Collect rule-high or learned-high rollouts over the existing graph environment."""

    def __init__(self, config: HierarchicalExperimentConfig, *, device: torch.device) -> None:
        self.config = config
        self.device = device
        _seed_everything(config.seed)
        scenario = make_graph_scenario(
            num_uavs=config.num_uavs, dynamic_obstacle=config.dynamic_obstacle_enabled
        )
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
            max_dynamic_obstacles=int(config.dynamic_obstacle_enabled),
            communication_enabled=config.communication_enabled,
            communication_range=config.communication_radius,
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
        self.policy_state = HierarchicalPolicy(
            config.policy_config(),
            num_envs=config.num_envs,
            num_agents=config.num_uavs,
            device=device,
        )
        node_feature_dim = self.environments[0].observation_space("uav_0").shape[0]
        self.trainer = HierarchicalMAPPOTrainer(
            high_actor=HighLevelActor(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=EDGE_FEATURE_DIMENSION,
                embedding_dim=config.embedding_dim,
                num_heads=config.graph_heads,
                num_layers=config.graph_layers,
            ),
            high_critic=HighLevelCritic(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=EDGE_FEATURE_DIMENSION,
                embedding_dim=config.embedding_dim,
                num_heads=config.graph_heads,
                num_layers=config.graph_layers,
            ),
            low_actor=LowLevelActor(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=EDGE_FEATURE_DIMENSION,
                embedding_dim=config.embedding_dim,
                num_heads=config.graph_heads,
                num_layers=config.graph_layers,
                high_interval=config.high_interval,
            ),
            low_critic=LowLevelCritic(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=EDGE_FEATURE_DIMENSION,
                embedding_dim=config.embedding_dim,
                num_heads=config.graph_heads,
                num_layers=config.graph_layers,
                high_interval=config.high_interval,
            ),
            config=config.trainer_config(),
            device=device,
        )
        self.graph_builder = ConflictGraphBuilder(config.graph_config())
        self.cbf_adapter = (
            NormalizedActionCBFAdapter(
                OSQPSafetyFilter(
                    CBFConfig(
                        max_solve_time_seconds=0.1,
                        slack_penalty=config.cbf_slack_penalty,
                        max_iterations=config.cbf_max_iterations,
                        communication_uncertainty_margin_gain=(
                            config.cbf_communication_uncertainty_margin_gain
                        ),
                        max_communication_uncertainty_margin=(
                            config.cbf_max_communication_uncertainty_margin
                        ),
                    )
                )
            )
            if config.cbf_enabled
            else None
        )
        self.cbf_telemetry = SafetyFilterTelemetry()
        self.total_transitions = 0

    def train(self, *, stage: str) -> list[dict[str, float]]:
        """Run only the selected staged optimizer until the configured transition budget."""
        if stage not in {"low", "high", "joint"}:
            raise ValueError("stage must be low, high, or joint.")
        if stage == "joint" and not self.config.allow_joint_finetune:
            raise RuntimeError("Joint fine-tuning is disabled by configuration.")
        records: list[dict[str, float]] = []
        started = time.perf_counter()
        while self.total_transitions < self.config.total_steps:
            low_batch, high_batch, rollout_metrics = self.collect_rollout(stage=stage)
            if stage == "low":
                update_metrics = self.trainer.train_low(low_batch)
            elif stage == "high":
                update_metrics = self.trainer.train_high(high_batch)
            else:
                update_metrics = self.trainer.train_joint(low_batch, high_batch)
            increment = self.config.rollout_length * self.config.num_envs * self.config.num_uavs
            self.total_transitions += increment
            records.append(
                {
                    **rollout_metrics,
                    **update_metrics,
                    "fps": increment / max(time.perf_counter() - started, 1e-9),
                }
            )
        return records

    def collect_rollout(
        self, *, stage: str
    ) -> tuple[LowLevelBatch, HighLevelBatch, dict[str, float]]:
        """Collect aligned low/high data using a rule high level or learned high level."""
        if stage not in {"low", "high", "joint"}:
            raise ValueError("stage must be low, high, or joint.")
        low_storage: dict[str, list[Tensor]] = {
            "node_features": [],
            "edge_features": [],
            "adjacency": [],
            "active_mask": [],
            "actions": [],
            "old_log_probabilities": [],
            "old_values": [],
            "rewards": [],
            "terminated": [],
            "next_values": [],
            "context_actions": [],
            "context_one_hot": [],
            "context_remaining": [],
            "context_subgoal": [],
        }
        pending: list[_PendingHighSample | None] = [None for _ in self.environments]
        high_samples: list[_PendingHighSample] = []
        high_next_values: list[Tensor] = []
        for _ in range(self.config.rollout_length):
            node_features = self._node_features()
            graph = self._build_graph()
            proposals, proposal_log_probabilities, proposal_values = self._propose_high(
                stage, node_features, graph
            )
            high_actions, boundaries = self.policy_state.begin_low_step(proposals, graph.node_mask)
            context = self.policy_state.context()
            self._open_high_samples(
                pending,
                high_samples,
                high_next_values,
                boundaries,
                node_features,
                graph,
                high_actions,
                proposal_log_probabilities,
                proposal_values,
            )
            with torch.no_grad():
                low_actions, low_log_probabilities, _ = self.trainer.low_actor.sample(
                    node_features,
                    graph.edge_features,
                    graph.adjacency,
                    graph.node_mask,
                    context,
                    deterministic=False,
                )
                low_values = (
                    self.trainer.low_critic(
                        node_features,
                        graph.edge_features,
                        graph.adjacency,
                        graph.node_mask,
                        context,
                    )
                    .unsqueeze(dim=-1)
                    .expand(-1, self.config.num_uavs)
                )
            transition = self._step(low_actions.cpu().numpy())
            self._accumulate_high_rewards(pending, transition.rewards, graph.node_mask)
            self.policy_state.finish_low_step(transition.next_graph.node_mask)
            next_context = self.policy_state.context()
            with torch.no_grad():
                next_low_values = (
                    self.trainer.low_critic(
                        transition.next_node_features,
                        transition.next_graph.edge_features,
                        transition.next_graph.adjacency,
                        transition.next_graph.node_mask,
                        next_context,
                    )
                    .unsqueeze(dim=-1)
                    .expand(-1, self.config.num_uavs)
                )
                next_high_values = (
                    self.trainer.high_critic(
                        transition.next_node_features,
                        transition.next_graph.edge_features,
                        transition.next_graph.adjacency,
                        transition.next_graph.node_mask,
                    )
                    .unsqueeze(dim=-1)
                    .expand(-1, self.config.num_uavs)
                )
            close_mask = (self.policy_state.remaining_steps == 0) | transition.terminated
            self._close_high_samples(
                pending,
                high_samples,
                high_next_values,
                close_mask,
                transition.terminated,
                next_high_values,
            )
            low_storage["node_features"].append(node_features)
            low_storage["edge_features"].append(graph.edge_features)
            low_storage["adjacency"].append(graph.adjacency)
            low_storage["active_mask"].append(graph.node_mask)
            low_storage["actions"].append(low_actions)
            low_storage["old_log_probabilities"].append(low_log_probabilities)
            low_storage["old_values"].append(low_values)
            low_storage["rewards"].append(transition.rewards)
            low_storage["terminated"].append(transition.terminated)
            low_storage["next_values"].append(next_low_values)
            low_storage["context_actions"].append(context.actions)
            low_storage["context_one_hot"].append(context.action_one_hot)
            low_storage["context_remaining"].append(context.remaining_steps)
            low_storage["context_subgoal"].append(context.local_subgoal_offsets)
        self._close_unfinished(pending, high_samples, high_next_values)
        low_batch = self._low_batch(low_storage)
        high_batch = self._high_batch(high_samples, high_next_values)
        return low_batch, high_batch, {"high_boundaries": float(len(high_samples))}

    def _propose_high(
        self, stage: str, node_features: Tensor, graph: ConflictGraph
    ) -> tuple[Tensor, Tensor, Tensor]:
        if stage == "low":
            return (
                self._rule_high_actions(graph),
                torch.zeros_like(graph.node_mask, dtype=torch.float32),
                torch.zeros_like(graph.node_mask, dtype=torch.float32),
            )
        with torch.no_grad():
            actions, log_probabilities, _ = self.trainer.high_actor.sample(
                node_features,
                graph.edge_features,
                graph.adjacency,
                graph.node_mask,
                deterministic=False,
            )
            values = (
                self.trainer.high_critic(
                    node_features, graph.edge_features, graph.adjacency, graph.node_mask
                )
                .unsqueeze(dim=-1)
                .expand(-1, self.config.num_uavs)
            )
        return actions, log_probabilities, values

    def _rule_high_actions(self, graph: ConflictGraph) -> Tensor:
        shortfall = graph.edge_features[..., 9] > 0.0
        risk = shortfall.any(dim=-1) & graph.node_mask
        agent_indices = torch.arange(self.config.num_uavs, device=self.device).view(1, -1)
        yielding = risk & (agent_indices % 2 == 1)
        return torch.where(
            yielding,
            torch.full_like(
                agent_indices.expand_as(graph.node_mask), int(CoordinationAction.WAIT_OR_YIELD)
            ),
            torch.full_like(
                agent_indices.expand_as(graph.node_mask), int(CoordinationAction.CONTINUE)
            ),
        )

    def _open_high_samples(
        self,
        pending: list[_PendingHighSample | None],
        high_samples: list[_PendingHighSample],
        high_next_values: list[Tensor],
        boundaries: Tensor,
        node_features: Tensor,
        graph: ConflictGraph,
        actions: Tensor,
        log_probabilities: Tensor,
        values: Tensor,
    ) -> None:
        for environment_index in range(self.config.num_envs):
            if not bool(boundaries[environment_index].any()):
                continue
            if pending[environment_index] is not None:
                raise RuntimeError("A high-level sample remained open at its next boundary.")
            active = graph.node_mask[environment_index]
            pending[environment_index] = _PendingHighSample(
                node_features=node_features[environment_index].detach().clone(),
                edge_features=graph.edge_features[environment_index].detach().clone(),
                adjacency=graph.adjacency[environment_index].detach().clone(),
                active_mask=active.detach().clone(),
                actions=actions[environment_index].detach().clone(),
                log_probabilities=log_probabilities[environment_index].detach().clone(),
                values=values[environment_index].detach().clone(),
                discounted_rewards=torch.zeros(self.config.num_uavs, device=self.device),
                durations=torch.zeros(self.config.num_uavs, dtype=torch.long, device=self.device),
                terminated=torch.zeros(self.config.num_uavs, dtype=torch.bool, device=self.device),
            )

    def _accumulate_high_rewards(
        self, pending: list[_PendingHighSample | None], rewards: Tensor, active_mask: Tensor
    ) -> None:
        for environment_index, sample in enumerate(pending):
            if sample is None:
                continue
            discount = torch.pow(
                torch.full_like(sample.durations, self.config.gamma, dtype=torch.float32),
                sample.durations.to(dtype=torch.float32),
            )
            active = active_mask[environment_index] & sample.active_mask
            sample.discounted_rewards += (
                discount * rewards[environment_index] * active.to(dtype=torch.float32)
            )
            sample.durations += active.to(dtype=torch.long)

    def _close_high_samples(
        self,
        pending: list[_PendingHighSample | None],
        high_samples: list[_PendingHighSample],
        high_next_values: list[Tensor],
        close_mask: Tensor,
        terminated: Tensor,
        next_values: Tensor,
    ) -> None:
        for environment_index, sample in enumerate(pending):
            if sample is None or not bool(close_mask[environment_index].any()):
                continue
            sample.terminated |= terminated[environment_index]
            high_samples.append(sample)
            high_next_values.append(next_values[environment_index].detach().clone())
            pending[environment_index] = None

    def _close_unfinished(
        self,
        pending: list[_PendingHighSample | None],
        high_samples: list[_PendingHighSample],
        high_next_values: list[Tensor],
    ) -> None:
        for environment_index, sample in enumerate(pending):
            if sample is None:
                continue
            high_samples.append(sample)
            high_next_values.append(torch.zeros_like(sample.values))
            pending[environment_index] = None

    def _low_batch(self, storage: dict[str, list[Tensor]]) -> LowLevelBatch:
        def flatten(name: str) -> Tensor:
            values = torch.stack(storage[name], dim=0)
            return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])

        rewards = torch.stack(storage["rewards"], dim=0)
        values = torch.stack(storage["old_values"], dim=0)
        next_values = torch.stack(storage["next_values"], dim=0)
        terminated = torch.stack(storage["terminated"], dim=0)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        return LowLevelBatch(
            node_features=flatten("node_features"),
            edge_features=flatten("edge_features"),
            adjacency=flatten("adjacency"),
            active_mask=flatten("active_mask"),
            context=HighLevelActionContext(
                actions=flatten("context_actions"),
                action_one_hot=flatten("context_one_hot"),
                remaining_steps=flatten("context_remaining"),
                local_subgoal_offsets=flatten("context_subgoal"),
                priority_scores=torch.zeros_like(flatten("context_remaining"), dtype=torch.float32),
                suggested_delays=torch.zeros_like(
                    flatten("context_remaining"), dtype=torch.float32
                ),
            ),
            actions=flatten("actions"),
            old_log_probabilities=flatten("old_log_probabilities"),
            old_values=flatten("old_values"),
            returns=returns.reshape(-1, self.config.num_uavs),
            advantages=advantages.reshape(-1, self.config.num_uavs),
        )

    def _high_batch(
        self, samples: list[_PendingHighSample], next_values: list[Tensor]
    ) -> HighLevelBatch:
        if not samples:
            raise RuntimeError(
                "A hierarchical rollout must contain at least one high-level sample."
            )
        rewards = torch.stack([sample.discounted_rewards for sample in samples])
        values = torch.stack([sample.values for sample in samples])
        terminals = torch.stack([sample.terminated for sample in samples])
        durations = torch.stack([sample.durations.clamp_min(1) for sample in samples])
        next_value_tensor = torch.stack(next_values)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        for agent_index in range(self.config.num_uavs):
            agent_advantages, agent_returns = compute_duration_aware_gae(
                rewards=rewards[:, agent_index],
                values=values[:, agent_index],
                next_values=next_value_tensor[:, agent_index],
                terminated=terminals[:, agent_index],
                durations=durations[:, agent_index],
                gamma_low=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            advantages[:, agent_index] = agent_advantages
            returns[:, agent_index] = agent_returns
        return HighLevelBatch(
            node_features=torch.stack([sample.node_features for sample in samples]),
            edge_features=torch.stack([sample.edge_features for sample in samples]),
            adjacency=torch.stack([sample.adjacency for sample in samples]),
            active_mask=torch.stack([sample.active_mask for sample in samples]),
            actions=torch.stack([sample.actions for sample in samples]),
            old_log_probabilities=torch.stack([sample.log_probabilities for sample in samples]),
            old_values=values,
            returns=returns,
            advantages=advantages,
        )

    def _step(self, actions: np.ndarray) -> _StepTransition:
        rewards = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=np.float32)
        terminated = np.zeros((self.config.num_envs, self.config.num_uavs), dtype=bool)
        next_observations: list[tuple[dict[str, np.ndarray], MultiUAVParallelEnv]] = []
        for environment_index, environment in enumerate(self.environments):
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
                        "environment_step": snapshot.step_count,
                        "positions": snapshot.positions.tolist(),
                        "velocities": snapshot.velocities.tolist(),
                        "requested_velocities": decision.u_rl.tolist(),
                        "knowledge_uncertainty": [
                            state.position_uncertainty.tolist()
                            for state in snapshot.knowledge_states
                        ],
                        "knowledge_valid": [
                            state.valid.tolist() for state in snapshot.knowledge_states
                        ],
                        "dynamic_obstacle_centers": (
                            snapshot.dynamic_world.centers.tolist()
                            if snapshot.dynamic_world is not None
                            else []
                        ),
                    },
                )
            observation, reward_dict, terminal_dict, truncation_dict, _ = environment.step(
                {
                    agent: filtered_actions[environment.agent_name_mapping[agent]]
                    for agent in environment.possible_agents
                }
            )
            rewards[environment_index] = [
                reward_dict[agent] for agent in environment.possible_agents
            ]
            terminated[environment_index] = [
                terminal_dict[agent] or truncation_dict[agent]
                for agent in environment.possible_agents
            ]
            next_observations.append((observation, environment))
            if terminated[environment_index].all():
                self.observations[environment_index], _ = environment.reset(
                    seed=self.config.seed + 1000 + environment_index
                )
            else:
                self.observations[environment_index] = observation
        return _StepTransition(
            rewards=torch.as_tensor(rewards, device=self.device),
            terminated=torch.as_tensor(terminated, device=self.device),
            next_node_features=self._node_features_for(next_observations),
            next_graph=self._build_graph(),
        )

    def _node_features(self) -> Tensor:
        return self._node_features_for(list(zip(self.observations, self.environments, strict=True)))

    def _node_features_for(
        self, pairs: list[tuple[dict[str, np.ndarray], MultiUAVParallelEnv]]
    ) -> Tensor:
        values = [
            np.asarray([observations[agent] for agent in environment.possible_agents])
            for observations, environment in pairs
        ]
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)

    def _build_graph(self) -> ConflictGraph:
        return build_graph_from_environments(
            self.graph_builder, self.environments, device=self.device
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

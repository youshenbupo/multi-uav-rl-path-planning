"""Two-time-scale action state and policy primitives for hierarchical MAPPO."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from multiuav.learning.graph_networks import (
    DenseGraphAttentionEncoder,
    GraphActor,
    GraphCentralizedCritic,
)


class CoordinationAction(IntEnum):
    """Discrete coordination commands emitted by the high-level policy."""

    CONTINUE = 0
    WAIT_OR_YIELD = 1
    SHIFT_LEFT = 2
    SHIFT_RIGHT = 3
    CLIMB = 4
    DESCEND = 5


@dataclass(frozen=True)
class HierarchicalConfig:
    """Explicit ablation controls and high-level action-hold duration."""

    high_interval: int
    enable_high_level_policy: bool
    enable_local_subgoal: bool
    enable_priority: bool
    enable_delay: bool
    enable_altitude_maneuver: bool

    def __post_init__(self) -> None:
        if self.high_interval < 1:
            raise ValueError("high_interval must be positive.")


@dataclass(frozen=True)
class HighLevelActionContext:
    """Per-agent held coordination state consumed by every low-level decision."""

    actions: Tensor
    action_one_hot: Tensor
    remaining_steps: Tensor
    local_subgoal_offsets: Tensor
    priority_scores: Tensor
    suggested_delays: Tensor


class HierarchicalPolicy:
    """Own high-level action persistence independently from learned network weights."""

    def __init__(
        self,
        config: HierarchicalConfig,
        *,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ) -> None:
        if min(num_envs, num_agents) < 1:
            raise ValueError("num_envs and num_agents must be positive.")
        self.config = config
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.actions = torch.full(
            (num_envs, num_agents),
            int(CoordinationAction.CONTINUE),
            dtype=torch.long,
            device=device,
        )
        self.remaining_steps = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)

    def begin_low_step(self, proposals: Tensor, active_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Replace commands only at high boundaries and return action/boundary masks."""
        self._validate(proposals, active_mask)
        boundaries = active_mask & (self.remaining_steps == 0)
        normalized = self._normalize_actions(proposals.to(device=self.device))
        self.actions = torch.where(boundaries, normalized, self.actions)
        self.remaining_steps = torch.where(
            boundaries,
            torch.full_like(self.remaining_steps, self.config.high_interval),
            self.remaining_steps,
        )
        self._clear_inactive(active_mask)
        return self.actions.clone(), boundaries

    def finish_low_step(self, active_mask: Tensor) -> None:
        """Consume one low-level time unit and immediately clear ended agents."""
        if active_mask.shape != (self.num_envs, self.num_agents) or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean with shape [num_envs,num_agents].")
        self.remaining_steps = torch.where(
            active_mask,
            (self.remaining_steps - 1).clamp_min(0),
            self.remaining_steps,
        )
        self._clear_inactive(active_mask)

    def context(self) -> HighLevelActionContext:
        """Return held commands and optional-field tensors with disabled values fixed to zero."""
        action_one_hot = torch.nn.functional.one_hot(
            self.actions, num_classes=len(CoordinationAction)
        ).to(dtype=torch.float32)
        zeros_vector = torch.zeros(self.num_envs, self.num_agents, 3, device=self.device)
        zeros_scalar = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        return HighLevelActionContext(
            actions=self.actions.clone(),
            action_one_hot=action_one_hot,
            remaining_steps=self.remaining_steps.clone(),
            local_subgoal_offsets=zeros_vector,
            priority_scores=zeros_scalar,
            suggested_delays=zeros_scalar,
        )

    def _normalize_actions(self, proposals: Tensor) -> Tensor:
        if not self.config.enable_high_level_policy:
            return torch.full_like(proposals, int(CoordinationAction.CONTINUE))
        if torch.any(proposals < int(CoordinationAction.CONTINUE)) or torch.any(
            proposals > int(CoordinationAction.DESCEND)
        ):
            raise ValueError("High-level action proposals must be in [0,5].")
        if not self.config.enable_altitude_maneuver:
            altitude = (proposals == int(CoordinationAction.CLIMB)) | (
                proposals == int(CoordinationAction.DESCEND)
            )
            proposals = torch.where(
                altitude,
                torch.full_like(proposals, int(CoordinationAction.CONTINUE)),
                proposals,
            )
        return proposals

    def _clear_inactive(self, active_mask: Tensor) -> None:
        self.actions = torch.where(
            active_mask,
            self.actions,
            torch.full_like(self.actions, int(CoordinationAction.CONTINUE)),
        )
        self.remaining_steps = torch.where(
            active_mask, self.remaining_steps, torch.zeros_like(self.remaining_steps)
        )

    def _validate(self, proposals: Tensor, active_mask: Tensor) -> None:
        expected = (self.num_envs, self.num_agents)
        if proposals.shape != expected or proposals.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "proposals must be an integer tensor with shape [num_envs,num_agents]."
            )
        if active_mask.shape != expected or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean with shape [num_envs,num_agents].")


class HighLevelActor(nn.Module):
    """Shared categorical coordination policy over predictive graph node embeddings."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.encoder = DenseGraphAttentionEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.logits = nn.Linear(embedding_dim, len(CoordinationAction))

    def sample(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        active_mask: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample held-command proposals for active nodes only."""
        distribution = self._distribution(node_features, edge_features, adjacency, active_mask)
        actions = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        mask = active_mask.to(dtype=torch.float32)
        actions = torch.where(
            active_mask, actions, torch.full_like(actions, int(CoordinationAction.CONTINUE))
        )
        return actions, distribution.log_prob(actions) * mask, distribution.entropy() * mask

    def evaluate_actions(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        active_mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate stored categorical high actions with inactive values masked to zero."""
        distribution = self._distribution(node_features, edge_features, adjacency, active_mask)
        mask = active_mask.to(dtype=torch.float32)
        return distribution.log_prob(actions) * mask, distribution.entropy() * mask

    def _distribution(
        self, node_features: Tensor, edge_features: Tensor, adjacency: Tensor, active_mask: Tensor
    ) -> Categorical:
        embeddings = self.encoder(node_features, edge_features, adjacency, active_mask)
        return Categorical(logits=self.logits(embeddings))


class HighLevelCritic(nn.Module):
    """Centralized graph critic at the coordination-policy time scale."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.encoder = DenseGraphAttentionEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.value = nn.Linear(embedding_dim, 1)

    def forward(
        self, node_features: Tensor, edge_features: Tensor, adjacency: Tensor, active_mask: Tensor
    ) -> Tensor:
        """Return one centralized value per graph state."""
        embeddings = self.encoder(node_features, edge_features, adjacency, active_mask)
        mask = active_mask.unsqueeze(dim=-1).to(dtype=embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return cast(Tensor, self.value(pooled).squeeze(dim=-1))


class LowLevelActor(nn.Module):
    """Graph Gaussian velocity policy conditioned on the held high-level context."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        high_interval: int,
    ) -> None:
        super().__init__()
        if high_interval < 1:
            raise ValueError("high_interval must be positive.")
        self.high_interval = high_interval
        self.actor = GraphActor(
            node_feature_dim=node_feature_dim + len(CoordinationAction) + 1 + 3,
            edge_feature_dim=edge_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            action_dim=3,
        )

    def sample(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        active_mask: Tensor,
        context: HighLevelActionContext,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample normalized velocity actions conditioned on held coordination state."""
        conditioned = self._condition(node_features, context)
        actions, log_probabilities, entropy = self.actor.sample(
            conditioned, edge_features, adjacency, active_mask, deterministic=deterministic
        )
        return self._gate_wait(actions, log_probabilities, entropy, context.actions)

    def evaluate_actions(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        active_mask: Tensor,
        context: HighLevelActionContext,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate low actions with the same wait gate used at collection time."""
        conditioned = self._condition(node_features, context)
        log_probabilities, entropy = self.actor.evaluate_actions(
            conditioned, edge_features, adjacency, active_mask, actions
        )
        wait_mask = context.actions == int(CoordinationAction.WAIT_OR_YIELD)
        zero = torch.zeros_like(log_probabilities)
        return torch.where(wait_mask, zero, log_probabilities), torch.where(
            wait_mask, zero, entropy
        )

    def _condition(self, node_features: Tensor, context: HighLevelActionContext) -> Tensor:
        return condition_low_node_features(node_features, context, high_interval=self.high_interval)

    @staticmethod
    def _gate_wait(
        actions: Tensor, log_probabilities: Tensor, entropy: Tensor, commands: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        wait_mask = commands == int(CoordinationAction.WAIT_OR_YIELD)
        return (
            torch.where(wait_mask.unsqueeze(dim=-1), torch.zeros_like(actions), actions),
            torch.where(wait_mask, torch.zeros_like(log_probabilities), log_probabilities),
            torch.where(wait_mask, torch.zeros_like(entropy), entropy),
        )


class LowLevelCritic(nn.Module):
    """Centralized low-level value network conditioned on the held high action."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        high_interval: int,
    ) -> None:
        super().__init__()
        self.high_interval = high_interval
        self.critic = GraphCentralizedCritic(
            node_feature_dim=node_feature_dim + len(CoordinationAction) + 1 + 3,
            edge_feature_dim=edge_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        active_mask: Tensor,
        context: HighLevelActionContext,
    ) -> Tensor:
        """Predict one centralized low-level value per graph state."""
        conditioned = condition_low_node_features(
            node_features, context, high_interval=self.high_interval
        )
        return cast(Tensor, self.critic(conditioned, edge_features, adjacency, active_mask))


def condition_low_node_features(
    node_features: Tensor, context: HighLevelActionContext, *, high_interval: int
) -> Tensor:
    """Append held coordination fields to local node features for low-level networks."""
    if high_interval < 1 or context.actions.shape != node_features.shape[:2]:
        raise ValueError("High-level context must align with node features and positive interval.")
    remaining = (context.remaining_steps / high_interval).to(dtype=node_features.dtype)
    return torch.cat(
        (
            node_features,
            context.action_one_hot.to(dtype=node_features.dtype),
            remaining.unsqueeze(dim=-1),
            context.local_subgoal_offsets.to(dtype=node_features.dtype),
        ),
        dim=-1,
    )

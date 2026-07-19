"""Whole-graph rollout storage and future conflict-target construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from multiuav.learning.gae import compute_gae


@dataclass(frozen=True)
class GraphRolloutBatch:
    """Flattened `[time,environment]` graph samples for GraphMAPPO updates."""

    node_features: Tensor
    edge_features: Tensor
    adjacency: Tensor
    active_mask: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor
    auxiliary_conflict: Tensor
    auxiliary_minimum_distance: Tensor
    auxiliary_mask: Tensor


def build_future_conflict_targets(
    *,
    positions: Tensor,
    active_mask: Tensor,
    episode_ids: Tensor,
    horizon: int,
    conflict_distance: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build masked pair targets from future in-episode recorded positions."""
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [T,E,N,3].")
    time_steps, num_envs, num_agents, _ = positions.shape
    if active_mask.shape != (time_steps, num_envs, num_agents):
        raise ValueError("active_mask must have shape [T,E,N].")
    if episode_ids.shape != (time_steps, num_envs):
        raise ValueError("episode_ids must have shape [T,E].")
    if horizon < 1 or conflict_distance <= 0.0:
        raise ValueError("horizon and conflict_distance must be positive.")
    minimum_distance = torch.full(
        (time_steps, num_envs, num_agents, num_agents),
        float("inf"),
        dtype=positions.dtype,
        device=positions.device,
    )
    target_mask = torch.zeros_like(minimum_distance, dtype=torch.bool)
    current_pairs = active_mask[:, :, :, None] & active_mask[:, :, None, :]
    for offset in range(1, horizon + 1):
        if offset >= time_steps:
            break
        future_positions = positions[offset:]
        relative_positions = future_positions[:, :, None, :, :] - future_positions[:, :, :, None, :]
        distances = torch.linalg.vector_norm(relative_positions, dim=-1)
        future_pairs = active_mask[offset:, :, :, None] & active_mask[offset:, :, None, :]
        same_episode = (episode_ids[offset:] == episode_ids[:-offset])[:, :, None, None]
        valid = current_pairs[:-offset] & future_pairs & same_episode
        previous = minimum_distance[:-offset]
        minimum_distance[:-offset] = torch.where(
            valid, torch.minimum(previous, distances), previous
        )
        target_mask[:-offset] |= valid
    diagonal = torch.eye(num_agents, dtype=torch.bool, device=positions.device).view(
        1, 1, num_agents, num_agents
    )
    target_mask &= ~diagonal
    minimum_distance = torch.where(
        target_mask, minimum_distance, torch.zeros_like(minimum_distance)
    )
    conflict = target_mask & (minimum_distance < conflict_distance)
    return conflict, minimum_distance, target_mask


class GraphRolloutBuffer:
    """Strict fixed-shape graph rollout storage indexed as `[T,E,N,...]`."""

    def __init__(
        self,
        *,
        rollout_length: int,
        num_envs: int,
        num_agents: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        if (
            min(
                rollout_length, num_envs, num_agents, node_feature_dim, edge_feature_dim, action_dim
            )
            < 1
        ):
            raise ValueError("Graph rollout dimensions must be positive.")
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        prefix = (rollout_length, num_envs, num_agents)
        self.node_features = torch.zeros(*prefix, node_feature_dim, device=device)
        self.edge_features = torch.zeros(*prefix, num_agents, edge_feature_dim, device=device)
        self.adjacency = torch.zeros(*prefix, num_agents, dtype=torch.bool, device=device)
        self.active_mask = torch.zeros(*prefix, dtype=torch.bool, device=device)
        self.positions = torch.zeros(*prefix, 3, device=device)
        self.episode_ids = torch.zeros(rollout_length, num_envs, dtype=torch.long, device=device)
        self.actions = torch.zeros(*prefix, action_dim, device=device)
        self.log_probabilities = torch.zeros(*prefix, device=device)
        self.rewards = torch.zeros(*prefix, device=device)
        self.terminated = torch.zeros(*prefix, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(*prefix, dtype=torch.bool, device=device)
        self.values = torch.zeros(*prefix, device=device)
        self.next_values = torch.zeros(*prefix, device=device)
        self.advantages = torch.zeros(*prefix, device=device)
        self.returns = torch.zeros(*prefix, device=device)
        self.index = 0

    def add(
        self,
        *,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        active_mask: Tensor,
        positions: Tensor,
        episode_ids: Tensor,
        actions: Tensor,
        log_probabilities: Tensor,
        rewards: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        values: Tensor,
        next_values: Tensor,
    ) -> None:
        """Store one complete `[E,N]` graph transition."""
        if self.index >= self.rollout_length:
            raise RuntimeError("Graph rollout buffer is already full.")
        expected = (self.num_envs, self.num_agents)
        self._copy(
            self.node_features[self.index], node_features, (*expected, self.node_features.shape[-1])
        )
        self._copy(
            self.edge_features[self.index],
            edge_features,
            (*expected, self.num_agents, self.edge_features.shape[-1]),
        )
        self._copy(self.adjacency[self.index], adjacency, (*expected, self.num_agents))
        self._copy(self.active_mask[self.index], active_mask, expected)
        self._copy(self.positions[self.index], positions, (*expected, 3))
        self._copy(self.episode_ids[self.index], episode_ids, (self.num_envs,))
        self._copy(self.actions[self.index], actions, (*expected, self.actions.shape[-1]))
        self._copy(self.log_probabilities[self.index], log_probabilities, expected)
        self._copy(self.rewards[self.index], rewards, expected)
        self._copy(self.terminated[self.index], terminated, expected)
        self._copy(self.truncated[self.index], truncated, expected)
        self._copy(self.values[self.index], values, expected)
        self._copy(self.next_values[self.index], next_values, expected)
        self.index += 1

    def compute_returns_and_advantages(self, *, gamma: float, gae_lambda: float) -> None:
        """Compute per-agent GAE from the filled graph rollout."""
        if self.index != self.rollout_length:
            raise RuntimeError("GAE requires a full graph rollout.")
        self.advantages, self.returns = compute_gae(
            rewards=self.rewards,
            values=self.values,
            next_values=self.next_values,
            terminated=self.terminated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def flatten(self, *, horizon: int, conflict_distance: float) -> GraphRolloutBatch:
        """Flatten time/environment axes while preserving each graph's node axes."""
        if self.index != self.rollout_length:
            raise RuntimeError("Only full graph rollouts can be flattened.")
        conflict, minimum_distance, auxiliary_mask = build_future_conflict_targets(
            positions=self.positions,
            active_mask=self.active_mask,
            episode_ids=self.episode_ids,
            horizon=horizon,
            conflict_distance=conflict_distance,
        )
        sample_count = self.rollout_length * self.num_envs
        return GraphRolloutBatch(
            node_features=self.node_features.reshape(sample_count, self.num_agents, -1),
            edge_features=self.edge_features.reshape(
                sample_count, self.num_agents, self.num_agents, -1
            ),
            adjacency=self.adjacency.reshape(sample_count, self.num_agents, self.num_agents),
            active_mask=self.active_mask.reshape(sample_count, self.num_agents),
            actions=self.actions.reshape(sample_count, self.num_agents, -1),
            old_log_probabilities=self.log_probabilities.reshape(sample_count, self.num_agents),
            old_values=self.values.reshape(sample_count, self.num_agents),
            returns=self.returns.reshape(sample_count, self.num_agents),
            advantages=self.advantages.reshape(sample_count, self.num_agents),
            auxiliary_conflict=conflict.reshape(sample_count, self.num_agents, self.num_agents),
            auxiliary_minimum_distance=minimum_distance.reshape(
                sample_count, self.num_agents, self.num_agents
            ),
            auxiliary_mask=auxiliary_mask.reshape(sample_count, self.num_agents, self.num_agents),
        )

    def _copy(self, destination: Tensor, source: Tensor, expected_shape: tuple[int, ...]) -> None:
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"Expected tensor shape {expected_shape}, received {tuple(source.shape)}."
            )
        destination.copy_(source.to(device=self.device, dtype=destination.dtype))

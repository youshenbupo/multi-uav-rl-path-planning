"""Validated fixed-shape `[time, environment, UAV]` storage for MAPPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from multiuav.learning.gae import compute_gae


@dataclass(frozen=True)
class RolloutBatch:
    """Flattened samples ready for shuffled PPO minibatches."""

    observations: Tensor
    states: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor


class RolloutBuffer:
    """Store a complete on-policy rollout with strict tensor-shape invariants."""

    def __init__(
        self,
        *,
        rollout_length: int,
        num_envs: int,
        num_agents: int,
        observation_dim: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        if min(rollout_length, num_envs, num_agents, observation_dim, state_dim, action_dim) < 1:
            raise ValueError("Rollout dimensions must all be positive.")
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        prefix = (rollout_length, num_envs, num_agents)
        self.observations = torch.zeros(*prefix, observation_dim, device=device)
        self.states = torch.zeros(*prefix, state_dim, device=device)
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
        observations: Tensor,
        states: Tensor,
        actions: Tensor,
        log_probabilities: Tensor,
        rewards: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        values: Tensor,
        next_values: Tensor,
    ) -> None:
        """Store one `[environment, UAV]` transition after exact shape validation."""
        if self.index >= self.rollout_length:
            raise RuntimeError("Rollout buffer is already full.")
        expected = (self.num_envs, self.num_agents)
        self._copy(
            self.observations[self.index], observations, (*expected, self.observations.shape[-1])
        )
        self._copy(self.states[self.index], states, (*expected, self.states.shape[-1]))
        self._copy(self.actions[self.index], actions, (*expected, self.actions.shape[-1]))
        self._copy(self.log_probabilities[self.index], log_probabilities, expected)
        self._copy(self.rewards[self.index], rewards, expected)
        self._copy(self.terminated[self.index], terminated, expected)
        self._copy(self.truncated[self.index], truncated, expected)
        self._copy(self.values[self.index], values, expected)
        self._copy(self.next_values[self.index], next_values, expected)
        self.index += 1

    def compute_returns_and_advantages(self, *, gamma: float, gae_lambda: float) -> None:
        """Compute GAE for exactly the filled prefix of the current rollout."""
        if self.index != self.rollout_length:
            raise RuntimeError("GAE requires a full rollout.")
        self.advantages, self.returns = compute_gae(
            rewards=self.rewards,
            values=self.values,
            next_values=self.next_values,
            terminated=self.terminated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def flatten(self) -> RolloutBatch:
        """Flatten `[T,E,N,...]` into independent shared-policy samples."""
        if self.index != self.rollout_length:
            raise RuntimeError("Only full rollouts can be flattened.")
        sample_count = self.rollout_length * self.num_envs * self.num_agents
        return RolloutBatch(
            observations=self.observations.reshape(sample_count, -1),
            states=self.states.reshape(sample_count, -1),
            actions=self.actions.reshape(sample_count, -1),
            old_log_probabilities=self.log_probabilities.reshape(sample_count),
            old_values=self.values.reshape(sample_count),
            returns=self.returns.reshape(sample_count),
            advantages=self.advantages.reshape(sample_count),
        )

    def clear(self) -> None:
        """Reset the insertion cursor without reallocating tensors."""
        self.index = 0

    def _copy(self, destination: Tensor, source: Tensor, expected_shape: tuple[int, ...]) -> None:
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"Expected tensor shape {expected_shape}, received {tuple(source.shape)}."
            )
        destination.copy_(source.to(device=self.device, dtype=destination.dtype))

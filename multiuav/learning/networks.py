"""Shared Gaussian actor and centralized critic networks for basic MAPPO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Normal

_SQUASH_EPSILON = 1e-6


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    """Build a compact ReLU MLP with explicit positive layer dimensions."""
    dimensions = (input_dim, *hidden_dims, output_dim)
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("MLP dimensions must all be positive.")
    layers: list[nn.Module] = []
    for index, (first, second) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        layers.append(nn.Linear(first, second))
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SharedGaussianActor(nn.Module):
    """One parameter-sharing Gaussian policy applied independently to every UAV observation."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.network = _mlp(observation_dim, hidden_dims, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, observations: Tensor) -> Normal:
        """Return an independent-dimension Normal action distribution."""
        mean = self.network(observations)
        scale = self.log_std.exp().expand_as(mean)
        return Normal(mean, scale)

    def sample(self, observations: Tensor, *, deterministic: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Return bounded action plus summed log probability and entropy per sample."""
        distribution = self.distribution(observations)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        actions = raw_action.tanh()
        log_probability = self._squashed_log_probability(distribution, raw_action, actions)
        entropy = distribution.entropy().sum(dim=-1)
        return actions, log_probability, entropy

    def act(self, observations: Tensor, *, deterministic: bool) -> Tensor:
        """Return only the bounded environment action for execution."""
        actions, _, _ = self.sample(observations, deterministic=deterministic)
        return actions

    def evaluate_actions(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate stored bounded actions under the current Gaussian policy."""
        distribution = self.distribution(observations)
        bounded_actions = actions.clamp(-1.0 + _SQUASH_EPSILON, 1.0 - _SQUASH_EPSILON)
        raw_actions = 0.5 * (torch.log1p(bounded_actions) - torch.log1p(-bounded_actions))
        return (
            self._squashed_log_probability(distribution, raw_actions, bounded_actions),
            distribution.entropy().sum(dim=-1),
        )

    @staticmethod
    def _squashed_log_probability(
        distribution: Normal, raw_actions: Tensor, actions: Tensor
    ) -> Tensor:
        """Evaluate a tanh-squashed Gaussian with its change-of-variables correction."""
        correction = torch.log(1.0 - actions.square() + _SQUASH_EPSILON)
        return cast(Tensor, (distribution.log_prob(raw_actions) - correction).sum(dim=-1))


class CentralizedCritic(nn.Module):
    """Value network that consumes only the environment's centralized state."""

    def __init__(self, state_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.network = _mlp(state_dim, hidden_dims, 1)

    def forward(self, states: Tensor) -> Tensor:
        """Return one scalar value per centralized state row."""
        return cast(Tensor, self.network(states).squeeze(dim=-1))

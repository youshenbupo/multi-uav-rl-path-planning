"""Generalized advantage estimation with explicit true-terminal bootstrap masking."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gae(
    *,
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Return advantages and returns, bootstrapping time limits but never true terminals."""
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("Rewards, values, and next_values must have identical shapes.")
    if terminated.shape != rewards.shape:
        raise ValueError("terminated must match rewards shape.")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1].")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = (~terminated[index]).to(dtype=rewards.dtype)
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        carry = delta + gamma * gae_lambda * bootstrap * carry
        advantages[index] = carry
    return advantages, advantages + values

"""Independent high-level reward accumulation and duration-aware GAE utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HighLevelRecord:
    """One completed high-level decision with its actual low-step duration."""

    environment_index: int
    agent_index: int
    action: int
    log_probability: float
    value: float
    discounted_reward: float
    duration: int
    terminated: bool
    next_value: float
    low_start_step: int


@dataclass
class _PendingHighRecord:
    """Mutable reward accumulator while a held high-level command is active."""

    action: int
    log_probability: float
    value: float
    discounted_reward: float
    duration: int
    low_start_step: int


def compute_duration_aware_gae(
    *,
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    durations: Tensor,
    gamma_low: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE where each record spans its own positive low-level duration."""
    if not (
        rewards.shape == values.shape == next_values.shape == terminated.shape == durations.shape
    ):
        raise ValueError("High-level GAE tensors must have identical shapes.")
    if rewards.ndim != 1:
        raise ValueError("Duration-aware GAE accepts one chronological decision sequence.")
    if torch.any(durations < 1):
        raise ValueError("High-level durations must be positive.")
    if not 0.0 <= gamma_low <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma_low and gae_lambda must be in [0,1].")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.shape[0] - 1, -1, -1):
        discount = gamma_low ** int(durations[index].item())
        bootstrap = (~terminated[index]).to(dtype=rewards.dtype)
        delta = rewards[index] + discount * bootstrap * next_values[index] - values[index]
        carry = delta + discount * gae_lambda * bootstrap * carry
        advantages[index] = carry
    return advantages, advantages + values


class HierarchicalRolloutBuffer:
    """Track high-level records aligned to the low-level environment-step clock."""

    def __init__(self, *, num_envs: int, num_agents: int, gamma: float) -> None:
        if min(num_envs, num_agents) < 1 or not 0.0 <= gamma <= 1.0:
            raise ValueError("Buffer dimensions must be positive and gamma must be in [0,1].")
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.gamma = gamma
        self._pending: list[list[_PendingHighRecord | None]] = [
            [None for _ in range(num_agents)] for _ in range(num_envs)
        ]
        self._records: list[HighLevelRecord] = []

    def open_high(
        self,
        *,
        actions: Tensor,
        log_probabilities: Tensor,
        values: Tensor,
        boundary_mask: Tensor,
        low_step: int,
    ) -> None:
        """Open a record exactly for active high-policy boundaries."""
        expected = (self.num_envs, self.num_agents)
        for name, value in {
            "actions": actions,
            "log_probabilities": log_probabilities,
            "values": values,
            "boundary_mask": boundary_mask,
        }.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape [num_envs,num_agents].")
        if boundary_mask.dtype != torch.bool or low_step < 0:
            raise ValueError("boundary_mask must be boolean and low_step nonnegative.")
        for environment_index in range(self.num_envs):
            for agent_index in range(self.num_agents):
                if not bool(boundary_mask[environment_index, agent_index]):
                    continue
                if self._pending[environment_index][agent_index] is not None:
                    raise RuntimeError("A high-level record is already open at this boundary.")
                self._pending[environment_index][agent_index] = _PendingHighRecord(
                    action=int(actions[environment_index, agent_index].item()),
                    log_probability=float(log_probabilities[environment_index, agent_index].item()),
                    value=float(values[environment_index, agent_index].item()),
                    discounted_reward=0.0,
                    duration=0,
                    low_start_step=low_step,
                )

    def add_low_rewards(self, rewards: Tensor) -> None:
        """Accumulate one low-step reward for every currently held high command."""
        if rewards.shape != (self.num_envs, self.num_agents):
            raise ValueError("rewards must have shape [num_envs,num_agents].")
        for environment_index in range(self.num_envs):
            for agent_index in range(self.num_agents):
                pending = self._pending[environment_index][agent_index]
                if pending is None:
                    continue
                pending.discounted_reward += self.gamma**pending.duration * float(
                    rewards[environment_index, agent_index].item()
                )
                pending.duration += 1

    def close_high(self, *, close_mask: Tensor, terminated: Tensor, next_values: Tensor) -> None:
        """Close selected held commands using their actual accumulated duration."""
        expected = (self.num_envs, self.num_agents)
        if (
            close_mask.shape != expected
            or terminated.shape != expected
            or next_values.shape != expected
        ):
            raise ValueError("High close tensors must have shape [num_envs,num_agents].")
        if close_mask.dtype != torch.bool or terminated.dtype != torch.bool:
            raise ValueError("close_mask and terminated must be boolean.")
        for environment_index in range(self.num_envs):
            for agent_index in range(self.num_agents):
                if not bool(close_mask[environment_index, agent_index]):
                    continue
                pending = self._pending[environment_index][agent_index]
                if pending is None:
                    raise RuntimeError("Cannot close a high-level record that is not open.")
                if pending.duration < 1:
                    raise RuntimeError("High-level record cannot close before one low-level step.")
                self._records.append(
                    HighLevelRecord(
                        environment_index=environment_index,
                        agent_index=agent_index,
                        action=pending.action,
                        log_probability=pending.log_probability,
                        value=pending.value,
                        discounted_reward=pending.discounted_reward,
                        duration=pending.duration,
                        terminated=bool(terminated[environment_index, agent_index]),
                        next_value=float(next_values[environment_index, agent_index].item()),
                        low_start_step=pending.low_start_step,
                    )
                )
                self._pending[environment_index][agent_index] = None

    def records(self) -> tuple[HighLevelRecord, ...]:
        """Return immutable completed records in closure order."""
        return tuple(self._records)

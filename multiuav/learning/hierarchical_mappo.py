"""Separate staged PPO optimizers and checkpoints for hierarchical graph policies."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn, optim

from multiuav.learning.bc_finetuning import (
    BCFineTuneSchedule,
    BCTransferReport,
    load_shape_compatible_bc_state,
    set_encoder_trainable,
)
from multiuav.learning.hierarchical_policy import (
    CoordinationAction,
    HighLevelActionContext,
    HighLevelActor,
    HighLevelCritic,
    LowLevelActor,
    LowLevelCritic,
)


@dataclass(frozen=True)
class HierarchicalMAPPOConfig:
    """Shared PPO settings plus an explicit joint-finetuning guard."""

    learning_rate: float
    clip_ratio: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    batch_size: int
    ppo_epochs: int
    allow_joint_finetune: bool

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.max_grad_norm, self.batch_size, self.ppo_epochs) <= 0:
            raise ValueError("Hierarchical PPO optimizer settings must be positive.")
        if min(self.clip_ratio, self.entropy_coef, self.value_coef) < 0.0:
            raise ValueError("Hierarchical PPO coefficients must be nonnegative.")


@dataclass(frozen=True)
class LowLevelBatch:
    """Whole-graph low-level PPO samples with held high-action context."""

    node_features: Tensor
    edge_features: Tensor
    adjacency: Tensor
    active_mask: Tensor
    context: HighLevelActionContext
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor


@dataclass(frozen=True)
class HighLevelBatch:
    """Whole-graph high-level PPO samples at coordination boundaries."""

    node_features: Tensor
    edge_features: Tensor
    adjacency: Tensor
    active_mask: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor


@dataclass(frozen=True)
class HierarchicalBCTransferReport:
    """Auditable low/high reports for a hierarchy initialized from BC checkpoints."""

    low: BCTransferReport | None
    high: BCTransferReport | None


class HierarchicalMAPPOTrainer:
    """Train low and high policies separately before any explicitly enabled joint fine-tuning."""

    def __init__(
        self,
        *,
        high_actor: HighLevelActor,
        high_critic: HighLevelCritic,
        low_actor: LowLevelActor,
        low_critic: LowLevelCritic,
        config: HierarchicalMAPPOConfig,
        device: torch.device,
    ) -> None:
        self.high_actor = high_actor.to(device)
        self.high_critic = high_critic.to(device)
        self.low_actor = low_actor.to(device)
        self.low_critic = low_critic.to(device)
        self.config = config
        self.device = device
        self.low_optimizer = optim.Adam(
            [*self.low_actor.parameters(), *self.low_critic.parameters()], lr=config.learning_rate
        )
        self.high_optimizer = optim.Adam(
            [*self.high_actor.parameters(), *self.high_critic.parameters()], lr=config.learning_rate
        )
        self.bc_schedule: BCFineTuneSchedule | None = None
        self.bc_update_count = 0

    def initialize_from_bc(
        self,
        *,
        low_checkpoint: Path | None,
        high_checkpoint: Path | None,
        schedule: BCFineTuneSchedule,
    ) -> HierarchicalBCTransferReport:
        """Transfer compatible encoders and restart both staged PPO optimizers at BC rate.

        The low actor adds high-command context before its graph encoder.  Therefore
        only BC tensors whose remapped target key and shape match are copied; this
        deliberately leaves the widened low node encoder random while reusing
        compatible attention layers.
        """
        low_report = (
            load_shape_compatible_bc_state(
                self.low_actor,
                low_checkpoint,
                source_prefix="encoder.",
                target_prefix="actor.encoder.",
            )
            if low_checkpoint is not None
            else None
        )
        high_report = (
            load_shape_compatible_bc_state(self.high_actor, high_checkpoint)
            if high_checkpoint is not None
            else None
        )
        self.bc_schedule = schedule
        self.bc_update_count = 0
        self._apply_bc_encoder_schedule()
        self.low_optimizer = optim.Adam(
            [*self.low_actor.parameters(), *self.low_critic.parameters()],
            lr=schedule.ppo_learning_rate,
        )
        self.high_optimizer = optim.Adam(
            [*self.high_actor.parameters(), *self.high_critic.parameters()],
            lr=schedule.ppo_learning_rate,
        )
        return HierarchicalBCTransferReport(low=low_report, high=high_report)

    def train_low(self, batch: LowLevelBatch) -> dict[str, float]:
        """Stage A update: optimize only graph-conditioned velocity policy and critic."""
        self._apply_bc_encoder_schedule()
        sample_count = batch.node_features.shape[0]
        metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": []}
        advantages = _normalize_advantages(
            batch.advantages.to(self.device), batch.active_mask.to(self.device)
        )
        for index in _minibatches(
            sample_count, self.config.batch_size, self.config.ppo_epochs, self.device
        ):
            context = _slice_context(batch.context, index, self.device)
            active_mask = batch.active_mask[index].to(self.device)
            learning_mask = active_mask & (context.actions != int(CoordinationAction.WAIT_OR_YIELD))
            log_probabilities, entropy = self.low_actor.evaluate_actions(
                batch.node_features[index].to(self.device),
                batch.edge_features[index].to(self.device),
                batch.adjacency[index].to(self.device),
                active_mask,
                context,
                batch.actions[index].to(self.device),
            )
            values = (
                self.low_critic(
                    batch.node_features[index].to(self.device),
                    batch.edge_features[index].to(self.device),
                    batch.adjacency[index].to(self.device),
                    active_mask,
                    context,
                )
                .unsqueeze(dim=-1)
                .expand_as(batch.old_values[index].to(self.device))
            )
            policy_loss = _clipped_loss(
                log_probabilities,
                batch.old_log_probabilities[index].to(self.device),
                advantages[index],
                learning_mask,
                self.config.clip_ratio,
            )
            value_loss = _value_loss(
                values,
                batch.old_values[index].to(self.device),
                batch.returns[index].to(self.device),
                active_mask,
                self.config.clip_ratio,
            )
            entropy_mean = _masked_mean(entropy, learning_mask)
            total_loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy_mean
            )
            self.low_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                [*self.low_actor.parameters(), *self.low_critic.parameters()],
                self.config.max_grad_norm,
            )
            self.low_optimizer.step()
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy_mean.detach().cpu()))
        self.bc_update_count += 1
        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def train_high(self, batch: HighLevelBatch) -> dict[str, float]:
        """Stage B update: low policy remains untouched while coordination policy learns."""
        self._apply_bc_encoder_schedule()
        sample_count = batch.node_features.shape[0]
        metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": []}
        advantages = _normalize_advantages(
            batch.advantages.to(self.device), batch.active_mask.to(self.device)
        )
        for index in _minibatches(
            sample_count, self.config.batch_size, self.config.ppo_epochs, self.device
        ):
            active_mask = batch.active_mask[index].to(self.device)
            log_probabilities, entropy = self.high_actor.evaluate_actions(
                batch.node_features[index].to(self.device),
                batch.edge_features[index].to(self.device),
                batch.adjacency[index].to(self.device),
                active_mask,
                batch.actions[index].to(self.device),
            )
            values = (
                self.high_critic(
                    batch.node_features[index].to(self.device),
                    batch.edge_features[index].to(self.device),
                    batch.adjacency[index].to(self.device),
                    active_mask,
                )
                .unsqueeze(dim=-1)
                .expand_as(batch.old_values[index].to(self.device))
            )
            policy_loss = _clipped_loss(
                log_probabilities,
                batch.old_log_probabilities[index].to(self.device),
                advantages[index],
                active_mask,
                self.config.clip_ratio,
            )
            value_loss = _value_loss(
                values,
                batch.old_values[index].to(self.device),
                batch.returns[index].to(self.device),
                active_mask,
                self.config.clip_ratio,
            )
            entropy_mean = _masked_mean(entropy, active_mask)
            total_loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy_mean
            )
            self.high_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                [*self.high_actor.parameters(), *self.high_critic.parameters()],
                self.config.max_grad_norm,
            )
            self.high_optimizer.step()
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy_mean.detach().cpu()))
        self.bc_update_count += 1
        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def train_joint(self, low_batch: LowLevelBatch, high_batch: HighLevelBatch) -> dict[str, float]:
        """Run explicit optional joint fine-tuning only when configuration enables it."""
        if not self.config.allow_joint_finetune:
            raise RuntimeError("Joint fine-tuning is disabled by configuration.")
        return {
            **{f"low/{name}": value for name, value in self.train_low(low_batch).items()},
            **{f"high/{name}": value for name, value in self.train_high(high_batch).items()},
        }

    def _apply_bc_encoder_schedule(self) -> None:
        """Freeze both transferred policy encoders until the configured staged-update count."""
        if self.bc_schedule is None:
            return
        trainable = not self.bc_schedule.encoder_frozen(update_index=self.bc_update_count)
        set_encoder_trainable(self.high_actor, trainable=trainable)
        set_encoder_trainable(self.low_actor.actor, trainable=trainable)


def _minibatches(
    sample_count: int, batch_size: int, ppo_epochs: int, device: torch.device
) -> Iterator[Tensor]:
    for _ in range(ppo_epochs):
        order = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            yield order[start : start + batch_size]


def _slice_context(
    context: HighLevelActionContext, index: Tensor, device: torch.device
) -> HighLevelActionContext:
    return HighLevelActionContext(
        actions=context.actions[index].to(device),
        action_one_hot=context.action_one_hot[index].to(device),
        remaining_steps=context.remaining_steps[index].to(device),
        local_subgoal_offsets=context.local_subgoal_offsets[index].to(device),
        priority_scores=context.priority_scores[index].to(device),
        suggested_delays=context.suggested_delays[index].to(device),
    )


def _normalize_advantages(advantages: Tensor, mask: Tensor) -> Tensor:
    valid = advantages[mask]
    normalized = (advantages - valid.mean()) / (valid.std(unbiased=False) + 1e-8)
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def _clipped_loss(
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_ratio: float,
) -> Tensor:
    ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -_masked_mean(torch.minimum(unclipped, clipped), mask)


def _value_loss(
    values: Tensor, old_values: Tensor, returns: Tensor, mask: Tensor, clip_ratio: float
) -> Tensor:
    clipped_values = old_values + (values - old_values).clamp(-clip_ratio, clip_ratio)
    return 0.5 * _masked_mean(
        torch.maximum((values - returns).square(), (clipped_values - returns).square()), mask
    )


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def save_hierarchical_checkpoint(
    path: Path, trainer: HierarchicalMAPPOTrainer, *, stage: str, step: int
) -> None:
    """Persist both policy levels, separate optimizers, config, and RNG state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "high_actor": trainer.high_actor.state_dict(),
            "high_critic": trainer.high_critic.state_dict(),
            "low_actor": trainer.low_actor.state_dict(),
            "low_critic": trainer.low_critic.state_dict(),
            "high_optimizer": trainer.high_optimizer.state_dict(),
            "low_optimizer": trainer.low_optimizer.state_dict(),
            "config": asdict(trainer.config),
            "stage": stage,
            "step": step,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_hierarchical_checkpoint(
    path: Path, trainer: HierarchicalMAPPOTrainer, *, map_location: torch.device
) -> tuple[str, int]:
    """Restore both hierarchy levels and return training stage plus completed step count."""
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    trainer.high_actor.load_state_dict(payload["high_actor"])
    trainer.high_critic.load_state_dict(payload["high_critic"])
    trainer.low_actor.load_state_dict(payload["low_actor"])
    trainer.low_critic.load_state_dict(payload["low_critic"])
    trainer.high_optimizer.load_state_dict(payload["high_optimizer"])
    trainer.low_optimizer.load_state_dict(payload["low_optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    cuda_rng = payload.get("torch_cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    return str(payload["stage"]), int(payload["step"])

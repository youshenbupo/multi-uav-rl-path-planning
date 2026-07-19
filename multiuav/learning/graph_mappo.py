"""PPO optimization for graph-encoded multi-UAV policies and auxiliary predictions."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as functional

from multiuav.learning.bc_finetuning import (
    BCFineTuneSchedule,
    BCTransferReport,
    GraphImitationBatch,
    load_shape_compatible_bc_state,
    masked_imitation_loss,
    set_encoder_trainable,
)
from multiuav.learning.graph_networks import (
    ConflictPredictionHead,
    GraphActor,
    GraphCentralizedCritic,
)
from multiuav.learning.graph_rollout_buffer import GraphRolloutBatch


@dataclass(frozen=True)
class GraphMAPPOConfig:
    """Optimizer and auxiliary-task weights for graph MAPPO."""

    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_ratio: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    batch_size: int
    ppo_epochs: int
    aux_conflict_coef: float
    aux_distance_coef: float

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.max_grad_norm, self.batch_size, self.ppo_epochs) <= 0:
            raise ValueError("Graph MAPPO optimizer settings must be positive.")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0,1].")
        if min(self.clip_ratio, self.entropy_coef, self.value_coef) < 0.0:
            raise ValueError("PPO loss coefficients must be nonnegative.")
        if min(self.aux_conflict_coef, self.aux_distance_coef) < 0.0:
            raise ValueError("Auxiliary loss coefficients must be nonnegative.")


class GraphMAPPOTrainer:
    """Optimize graph actor, centralized critic, and optional auxiliary edge head."""

    def __init__(
        self,
        actor: GraphActor,
        critic: GraphCentralizedCritic,
        auxiliary: ConflictPredictionHead,
        config: GraphMAPPOConfig,
        *,
        device: torch.device,
    ) -> None:
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.auxiliary = auxiliary.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters(), *self.auxiliary.parameters()],
            lr=config.learning_rate,
        )
        self.bc_schedule: BCFineTuneSchedule | None = None
        self.update_count = 0

    def initialize_from_bc(
        self, checkpoint: Path, schedule: BCFineTuneSchedule
    ) -> BCTransferReport:
        """Load compatible BC actor state and apply its lower-rate encoder schedule."""
        report = load_shape_compatible_bc_state(self.actor, checkpoint)
        normalization_path = checkpoint.parent / "normalization_stats.json"
        if normalization_path.is_file():
            stats = json.loads(normalization_path.read_text(encoding="utf-8"))
            self.actor.set_node_normalization(
                torch.tensor(stats["node_mean"], dtype=torch.float32, device=self.device),
                torch.tensor(stats["node_std"], dtype=torch.float32, device=self.device),
            )
        self.bc_schedule = schedule
        set_encoder_trainable(self.actor, trainable=not schedule.encoder_frozen(update_index=0))
        self.optimizer = optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters(), *self.auxiliary.parameters()],
            lr=schedule.ppo_learning_rate,
        )
        return report

    def update(
        self, batch: GraphRolloutBatch, *, imitation_batch: GraphImitationBatch | None = None
    ) -> dict[str, float]:
        """Run PPO plus an optional strict-mask low-level BC auxiliary objective."""
        if self.bc_schedule is not None:
            set_encoder_trainable(
                self.actor,
                trainable=not self.bc_schedule.encoder_frozen(update_index=self.update_count),
            )
        node_features = batch.node_features.to(self.device)
        edge_features = batch.edge_features.to(self.device)
        adjacency = batch.adjacency.to(self.device)
        active_mask = batch.active_mask.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probabilities = batch.old_log_probabilities.to(self.device)
        old_values = batch.old_values.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = self._normalize_advantages(batch.advantages.to(self.device), active_mask)
        auxiliary_conflict = batch.auxiliary_conflict.to(self.device)
        auxiliary_minimum_distance = batch.auxiliary_minimum_distance.to(self.device)
        auxiliary_mask = batch.auxiliary_mask.to(self.device)
        imitation = self._prepare_imitation_batch(imitation_batch)
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "aux_conflict_loss": [],
            "aux_distance_loss": [],
            "imitation_loss": [],
        }
        sample_count = node_features.shape[0]
        for _ in range(self.config.ppo_epochs):
            order = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                masks = active_mask[index]
                new_log_probabilities, entropy = self.actor.evaluate_actions(
                    node_features[index],
                    edge_features[index],
                    adjacency[index],
                    masks,
                    actions[index],
                )
                policy_loss, ratio, clip_fraction = self._policy_loss(
                    new_log_probabilities,
                    old_log_probabilities[index],
                    advantages[index],
                    masks,
                )
                graph_values = self.critic(
                    node_features[index], edge_features[index], adjacency[index], masks
                )
                values = graph_values.unsqueeze(dim=-1).expand_as(old_values[index])
                clipped_values = old_values[index] + (values - old_values[index]).clamp(
                    -self.config.clip_ratio, self.config.clip_ratio
                )
                value_loss = 0.5 * _masked_mean(
                    torch.maximum(
                        (values - returns[index]).square(),
                        (clipped_values - returns[index]).square(),
                    ),
                    masks,
                )
                embeddings = self.actor.encode(
                    node_features[index], edge_features[index], adjacency[index], masks
                )
                conflict_logits, predicted_distances = self.auxiliary(
                    embeddings, edge_features[index]
                )
                target_mask = auxiliary_mask[index]
                conflict_loss = _masked_mean(
                    functional.binary_cross_entropy_with_logits(
                        conflict_logits,
                        auxiliary_conflict[index].to(dtype=conflict_logits.dtype),
                        reduction="none",
                    ),
                    target_mask,
                )
                distance_loss = _masked_mean(
                    functional.smooth_l1_loss(
                        predicted_distances, auxiliary_minimum_distance[index], reduction="none"
                    ),
                    target_mask,
                )
                imitation_loss = torch.zeros((), device=self.device)
                if imitation is not None:
                    imitation_indices = (
                        torch.arange(index.numel(), device=self.device) + start
                    ) % imitation.sample_count
                    predicted_actions, _, _ = self.actor.sample(
                        imitation.node_features[imitation_indices],
                        imitation.edge_features[imitation_indices],
                        imitation.adjacency[imitation_indices],
                        imitation.active_mask[imitation_indices],
                        deterministic=True,
                    )
                    imitation_loss = masked_imitation_loss(
                        predicted_actions,
                        imitation.expert_actions[imitation_indices],
                        imitation.expert_action_mask[imitation_indices],
                    )
                entropy_mean = _masked_mean(entropy, masks)
                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                    + self.config.aux_conflict_coef * conflict_loss
                    + self.config.aux_distance_coef * distance_loss
                    + self._imitation_coefficient * imitation_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    [
                        *self.actor.parameters(),
                        *self.critic.parameters(),
                        *self.auxiliary.parameters(),
                    ],
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
                metrics["value_loss"].append(float(value_loss.detach().cpu()))
                metrics["entropy"].append(float(entropy_mean.detach().cpu()))
                metrics["approx_kl"].append(
                    float(
                        _masked_mean(old_log_probabilities[index] - new_log_probabilities, masks)
                        .detach()
                        .cpu()
                    )
                )
                metrics["clip_fraction"].append(float(clip_fraction.detach().cpu()))
                metrics["aux_conflict_loss"].append(float(conflict_loss.detach().cpu()))
                metrics["aux_distance_loss"].append(float(distance_loss.detach().cpu()))
                metrics["imitation_loss"].append(float(imitation_loss.detach().cpu()))
        with torch.no_grad():
            predicted = self.critic(node_features, edge_features, adjacency, active_mask)
            expanded_predicted = predicted.unsqueeze(dim=-1).expand_as(returns)
            valid_returns = returns[active_mask]
            valid_errors = (returns - expanded_predicted)[active_mask]
            variance = torch.var(valid_returns, unbiased=False)
            explained_variance = 1.0 - torch.var(valid_errors, unbiased=False) / (variance + 1e-8)
        self.update_count += 1
        return {
            **{name: float(np.mean(values)) for name, values in metrics.items()},
            "explained_variance": float(explained_variance.detach().cpu()),
        }

    @property
    def _imitation_coefficient(self) -> float:
        """Expose zero unless an explicit BC fine-tuning schedule enables imitation."""
        return 0.0 if self.bc_schedule is None else self.bc_schedule.imitation_coef

    def _prepare_imitation_batch(
        self, batch: GraphImitationBatch | None
    ) -> GraphImitationBatch | None:
        """Validate that a requested BC objective has real, mask-bearing source data."""
        if self._imitation_coefficient == 0.0:
            return None
        if batch is None:
            raise ValueError("A nonzero imitation coefficient requires an expert action batch.")
        if batch.sample_count < 1:
            raise ValueError("Imitation batch must contain at least one graph state.")
        if batch.node_features.shape[-1] != self.actor.encoder.node_feature_dim:
            raise ValueError("Imitation node feature dimension does not match the graph actor.")
        if batch.edge_features.shape[-1] != self.actor.encoder.edge_feature_dim:
            raise ValueError("Imitation edge feature dimension does not match the graph actor.")
        return batch.to(self.device)

    def _normalize_advantages(self, advantages: Tensor, active_mask: Tensor) -> Tensor:
        valid = advantages[active_mask]
        normalized = (advantages - valid.mean()) / (valid.std(unbiased=False) + 1e-8)
        return torch.where(active_mask, normalized, torch.zeros_like(normalized))

    def _policy_loss(
        self,
        new_log_probabilities: Tensor,
        old_log_probabilities: Tensor,
        advantages: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ratio = torch.exp(new_log_probabilities - old_log_probabilities)
        unclipped = ratio * advantages
        clipped = (
            ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
        )
        policy_loss = -_masked_mean(torch.minimum(unclipped, clipped), mask)
        clip_fraction = _masked_mean(
            (torch.abs(ratio - 1.0) > self.config.clip_ratio).to(dtype=ratio.dtype), mask
        )
        return policy_loss, ratio, clip_fraction


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean only valid entries, returning differentiable zero for an empty mask."""
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def save_graph_checkpoint(path: Path, trainer: GraphMAPPOTrainer, *, step: int) -> None:
    """Persist graph models, optimizer, configuration, and all relevant RNG state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": trainer.actor.state_dict(),
            "critic": trainer.critic.state_dict(),
            "auxiliary": trainer.auxiliary.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "config": asdict(trainer.config),
            "step": step,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_graph_checkpoint(
    path: Path, trainer: GraphMAPPOTrainer, *, map_location: torch.device
) -> int:
    """Restore a graph trainer checkpoint and return its completed transition count."""
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    trainer.actor.load_state_dict(payload["actor"])
    trainer.critic.load_state_dict(payload["critic"])
    trainer.auxiliary.load_state_dict(payload["auxiliary"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["step"])

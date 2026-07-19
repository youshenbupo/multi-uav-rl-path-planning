"""Masked behavior-cloning losses and graph policy heads for expert supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from multiuav.learning.graph_networks import DenseGraphAttentionEncoder


@dataclass(frozen=True)
class LowBCLoss:
    """Independently observable normalized velocity supervision terms."""

    normalized_mse: Tensor
    direction: Tensor
    speed: Tensor

    def total(
        self, *, normalized_mse_coef: float, direction_coef: float, speed_coef: float
    ) -> Tensor:
        """Return the explicitly weighted training scalar."""
        if min(normalized_mse_coef, direction_coef, speed_coef) < 0.0:
            raise ValueError("Low BC loss coefficients must be nonnegative.")
        return (
            normalized_mse_coef * self.normalized_mse
            + direction_coef * self.direction
            + speed_coef * self.speed
        )


@dataclass(frozen=True)
class HighBCOutput:
    """Categorical maneuver and optional continuous coordination predictions."""

    maneuver_logits: Tensor
    delay: Tensor
    priority: Tensor
    local_subgoal: Tensor


@dataclass(frozen=True)
class HighBCTargets:
    """Authoritative high-level targets and independent masks for each field."""

    actions: Tensor
    action_mask: Tensor
    delay: Tensor
    delay_mask: Tensor
    priority: Tensor
    priority_mask: Tensor
    local_subgoal: Tensor
    local_subgoal_mask: Tensor


@dataclass(frozen=True)
class HighBCLoss:
    """Sparse high-level supervision terms; absent authoritative targets remain None."""

    classification: Tensor | None
    delay: Tensor | None
    priority: Tensor | None
    local_subgoal: Tensor | None
    label_count: int


class ExpertHighLevelPolicy(nn.Module):
    """Graph encoder with nine schema maneuver classes and optional coordination heads."""

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
        self.maneuver_head = nn.Linear(embedding_dim, 9)
        self.delay_head = nn.Linear(embedding_dim, 1)
        self.priority_head = nn.Linear(embedding_dim, 1)
        self.local_subgoal_head = nn.Linear(embedding_dim, 3)

    def forward(
        self, node_features: Tensor, edge_features: Tensor, adjacency: Tensor, active_mask: Tensor
    ) -> HighBCOutput:
        """Predict high coordination fields and zero all inactive-UAV output rows."""
        embeddings = self.encoder(node_features, edge_features, adjacency, active_mask)
        output_mask = active_mask.unsqueeze(dim=-1).to(dtype=embeddings.dtype)
        return HighBCOutput(
            maneuver_logits=self.maneuver_head(embeddings) * output_mask,
            delay=self.delay_head(embeddings) * output_mask,
            priority=self.priority_head(embeddings) * output_mask,
            local_subgoal=self.local_subgoal_head(embeddings) * output_mask,
        )


def high_bc_loss(output: HighBCOutput, targets: HighBCTargets) -> HighBCLoss:
    """Compute only source-authoritative high-level supervision; never infer continue."""
    _validate_high_targets(output, targets)
    label_count = int(targets.action_mask.sum())
    classification = None
    if label_count:
        labels = targets.actions[targets.action_mask]
        classification = functional.cross_entropy(
            output.maneuver_logits[targets.action_mask], labels
        )
    return HighBCLoss(
        classification=classification,
        delay=_masked_regression(output.delay, targets.delay, targets.delay_mask),
        priority=_masked_regression(output.priority, targets.priority, targets.priority_mask),
        local_subgoal=_masked_regression(
            output.local_subgoal, targets.local_subgoal, targets.local_subgoal_mask
        ),
        label_count=label_count,
    )


class ExpertLowLevelPolicy(nn.Module):
    """MAPPO-compatible graph encoder with a separately conditioned velocity head."""

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
        self.command_projection = nn.Linear(6, embedding_dim)
        self.action_head = nn.Linear(2 * embedding_dim, 3)

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        active_mask: Tensor,
        high_commands: Tensor,
    ) -> Tensor:
        """Return bounded normalized velocity fractions for active UAVs only."""
        if high_commands.shape != (*node_features.shape[:2], 6):
            raise ValueError("High command context must have shape [B,N,6].")
        embeddings = self.encoder(node_features, edge_features, adjacency, active_mask)
        conditioned = torch.cat((embeddings, self.command_projection(high_commands)), dim=-1)
        actions = self.action_head(conditioned).tanh()
        return cast(Tensor, actions * active_mask.unsqueeze(dim=-1).to(dtype=actions.dtype))


def low_bc_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> LowBCLoss:
    """Compute strict masked MSE, direction cosine, and speed-magnitude losses."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("Low BC prediction and target must both have shape [B,N,3].")
    if mask.shape != prediction.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Low BC mask must be a boolean [B,N] tensor.")
    mask_float = mask.unsqueeze(dim=-1).to(dtype=prediction.dtype)
    valid_count = mask_float.sum().clamp_min(1.0)
    normalized_mse = ((prediction - target).square() * mask_float).sum() / (
        valid_count * prediction.shape[-1]
    )
    predicted_speed = torch.linalg.vector_norm(prediction, dim=-1)
    target_speed = torch.linalg.vector_norm(target, dim=-1)
    direction_mask = mask & (target_speed > 1e-6)
    direction = _masked_mean(
        1.0 - functional.cosine_similarity(prediction, target, dim=-1), direction_mask
    )
    speed = _masked_mean((predicted_speed - target_speed).square(), mask)
    return LowBCLoss(normalized_mse=normalized_mse, direction=direction, speed=speed)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average a `[B,N]` tensor without treating an empty mask as an error/label."""
    if values.shape != mask.shape:
        raise ValueError("Masked values and mask must share shape [B,N].")
    mask_float = mask.to(dtype=values.dtype)
    return (values * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def _masked_regression(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor | None:
    """Return no metric/loss when no authoritative target exists for one output head."""
    if not bool(mask.any()):
        return None
    expanded_mask = mask.unsqueeze(dim=-1).to(dtype=prediction.dtype)
    return ((prediction - target).square() * expanded_mask).sum() / (
        expanded_mask.sum() * prediction.shape[-1]
    )


def _validate_high_targets(output: HighBCOutput, targets: HighBCTargets) -> None:
    """Check fixed graph axes and ensure only masked labels enter categorical loss."""
    batch_shape = output.maneuver_logits.shape[:2]
    if output.maneuver_logits.ndim != 3 or output.maneuver_logits.shape[-1] != 9:
        raise ValueError("High maneuver logits must have shape [B,N,9].")
    if targets.actions.shape != batch_shape or targets.action_mask.shape != batch_shape:
        raise ValueError("High action target and mask must have shape [B,N].")
    masks = (
        targets.action_mask,
        targets.delay_mask,
        targets.priority_mask,
        targets.local_subgoal_mask,
    )
    if any(mask.shape != batch_shape or mask.dtype != torch.bool for mask in masks):
        raise ValueError("Every high-level target mask must be boolean with shape [B,N].")
    expected = ((output.delay, targets.delay, 1), (output.priority, targets.priority, 1),
                (output.local_subgoal, targets.local_subgoal, 3))
    for prediction, target, width in expected:
        if prediction.shape != (*batch_shape, width) or target.shape != prediction.shape:
            raise ValueError("High-level continuous target shape does not match its output head.")
    labels = targets.actions[targets.action_mask]
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= 9):
        raise ValueError("Masked high action labels must be in the schema range [0, 8].")

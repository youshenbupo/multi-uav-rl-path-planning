"""Explicit BC-to-MAPPO encoder fine-tuning schedule primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BCFineTuneSchedule:
    """Freeze a transferred graph encoder, then release it for lower-rate PPO updates."""

    freeze_encoder_updates: int
    ppo_learning_rate: float
    imitation_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.freeze_encoder_updates < 0 or self.ppo_learning_rate <= 0.0:
            raise ValueError("BC fine-tuning freeze count and PPO learning rate are invalid.")
        if self.imitation_coef < 0.0:
            raise ValueError("BC imitation coefficient must be nonnegative.")

    def encoder_frozen(self, *, update_index: int) -> bool:
        """Return whether the encoder remains frozen before a zero-based PPO update."""
        if update_index < 0:
            raise ValueError("PPO update index must be nonnegative.")
        return update_index < self.freeze_encoder_updates


@dataclass(frozen=True)
class BCTransferReport:
    """Auditable result of loading only shape-compatible BC parameters."""

    loaded_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


@dataclass(frozen=True)
class GraphImitationBatch:
    """Authoritative low-level expert targets aligned to dense graph-policy inputs."""

    node_features: Tensor
    edge_features: Tensor
    adjacency: Tensor
    active_mask: Tensor
    expert_actions: Tensor
    expert_action_mask: Tensor

    def __post_init__(self) -> None:
        if self.node_features.ndim != 3:
            raise ValueError("Imitation node features must have shape [B,N,D].")
        batch_size, node_count, _ = self.node_features.shape
        if self.edge_features.shape[:3] != (batch_size, node_count, node_count):
            raise ValueError("Imitation edge features must align to the node graph axes.")
        if self.adjacency.shape != (batch_size, node_count, node_count):
            raise ValueError("Imitation adjacency must have shape [B,N,N].")
        if self.active_mask.shape != (batch_size, node_count):
            raise ValueError("Imitation active mask must have shape [B,N].")
        if self.expert_actions.shape != (batch_size, node_count, 3):
            raise ValueError("Expert actions must have shape [B,N,3].")
        if self.expert_action_mask.shape != (batch_size, node_count):
            raise ValueError("Expert action mask must have shape [B,N].")
        if self.adjacency.dtype != torch.bool or self.active_mask.dtype != torch.bool:
            raise ValueError("Imitation graph masks must be boolean.")
        if self.expert_action_mask.dtype != torch.bool:
            raise ValueError("Expert action mask must be boolean.")
        if torch.any(self.expert_action_mask & ~self.active_mask):
            raise ValueError("Expert action labels must refer only to active UAVs.")

    @property
    def sample_count(self) -> int:
        """Return the number of graph states available for auxiliary imitation."""
        return int(self.node_features.shape[0])

    def to(self, device: torch.device) -> GraphImitationBatch:
        """Move every immutable batch tensor to the MAPPO device."""
        return GraphImitationBatch(
            node_features=self.node_features.to(device),
            edge_features=self.edge_features.to(device),
            adjacency=self.adjacency.to(device),
            active_mask=self.active_mask.to(device),
            expert_actions=self.expert_actions.to(device),
            expert_action_mask=self.expert_action_mask.to(device),
        )


def load_shape_compatible_bc_state(
    module: nn.Module,
    checkpoint: Path,
    *,
    source_prefix: str = "",
    target_prefix: str = "",
) -> BCTransferReport:
    """Copy only shape-compatible tensors, optionally remapping an explicit module prefix."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["state_dict"]
    target = module.state_dict()
    compatible: dict[str, Tensor] = {}
    loaded_source_keys: list[str] = []
    for source_key, value in source.items():
        if source_prefix and not source_key.startswith(source_prefix):
            continue
        suffix = source_key[len(source_prefix) :] if source_prefix else source_key
        target_key = f"{target_prefix}{suffix}"
        if target_key in target and target[target_key].shape == value.shape:
            compatible[target_key] = value
            loaded_source_keys.append(source_key)
    module.load_state_dict(compatible, strict=False)
    return BCTransferReport(
        loaded_keys=tuple(sorted(loaded_source_keys)),
        skipped_keys=tuple(sorted(key for key in source if key not in loaded_source_keys)),
    )


def set_encoder_trainable(module: nn.Module, *, trainable: bool) -> None:
    """Freeze or release a graph-policy encoder while leaving action/value heads trainable."""
    encoder = getattr(module, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("BC fine-tuning target must expose an nn.Module encoder.")
    for parameter in encoder.parameters():
        parameter.requires_grad = trainable


def masked_imitation_loss(
    predicted_actions: Tensor, expert_actions: Tensor, mask: Tensor
) -> Tensor:
    """Compute optional low-level behavior-cloning loss without using invalid expert rows."""
    if predicted_actions.shape != expert_actions.shape or predicted_actions.shape[-1] != 3:
        raise ValueError("Imitation actions must share shape [B,N,3].")
    if mask.shape != predicted_actions.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Imitation mask must be boolean with shape [B,N].")
    weights = mask.unsqueeze(dim=-1).to(dtype=predicted_actions.dtype)
    return ((predicted_actions - expert_actions).square() * weights).sum() / (
        weights.sum().clamp_min(1.0) * predicted_actions.shape[-1]
    )

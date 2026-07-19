"""Deterministic strict-mask behavior-cloning training and artifact writing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, optim

from multiuav.data.bc_dataset import (
    BCSplitConfig,
    BCSupervision,
    BehaviorCloningManifest,
    build_bc_manifest,
    load_bc_supervision,
)
from multiuav.learning.behavior_cloning import (
    ExpertHighLevelPolicy,
    ExpertLowLevelPolicy,
    high_bc_loss,
    low_bc_loss,
)
from multiuav.learning.conflict_graph import GraphBuildConfig


@dataclass(frozen=True)
class BehaviorCloningConfig:
    """Complete, serializable pretraining configuration."""

    shards: tuple[Path, ...]
    split: BCSplitConfig
    epochs: int
    learning_rate: float
    embedding_dim: int
    graph_heads: int
    graph_layers: int
    max_neighbors: int
    max_horizontal_speed: float
    max_vertical_speed: float
    normalized_mse_coef: float
    direction_coef: float
    speed_coef: float

    def __post_init__(self) -> None:
        if min(
            self.epochs,
            self.learning_rate,
            self.embedding_dim,
            self.graph_heads,
            self.graph_layers,
            self.max_horizontal_speed,
            self.max_vertical_speed,
        ) <= 0:
            raise ValueError(
                "BC epochs, dimensions, learning rate, and speed limits must be positive."
            )
        if self.embedding_dim % self.graph_heads != 0 or self.max_neighbors < 0:
            raise ValueError("BC graph dimensions or neighbour count are invalid.")
        if min(self.normalized_mse_coef, self.direction_coef, self.speed_coef) < 0.0:
            raise ValueError("BC low loss coefficients must be nonnegative.")


def train_behavior_cloning(
    config: BehaviorCloningConfig, *, output_directory: Path, device: torch.device
) -> dict[str, float | int | None]:
    """Train episode-batched low/high BC and write the five required artifacts."""
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = build_bc_manifest(config.shards, split=config.split)
    graph_config = _graph_config()
    loaded = [
        (
            row,
            load_bc_supervision(
                Path(row["source_file"]),
                row["episode_id"],
                max_neighbors=config.max_neighbors,
                max_horizontal_speed=config.max_horizontal_speed,
                max_vertical_speed=config.max_vertical_speed,
                graph_config=graph_config,
            ),
        )
        for row in manifest["episodes"]
    ]
    train = [sample for row, sample in loaded if row["split"] == "train"]
    validation = [sample for row, sample in loaded if row["split"] == "validation"]
    if not train:
        raise ValueError("Episode split produced no low-level training episodes.")
    mean, std = _fit_node_normalization(train)
    reference = train[0].features
    low_policy = ExpertLowLevelPolicy(
        node_feature_dim=reference.node_features.shape[-1],
        edge_feature_dim=reference.edge_features.shape[-1],
        embedding_dim=config.embedding_dim,
        num_heads=config.graph_heads,
        num_layers=config.graph_layers,
    ).to(device)
    high_policy = ExpertHighLevelPolicy(
        node_feature_dim=reference.node_features.shape[-1],
        edge_feature_dim=reference.edge_features.shape[-1],
        embedding_dim=config.embedding_dim,
        num_heads=config.graph_heads,
        num_layers=config.graph_layers,
    ).to(device)
    low_optimizer = optim.Adam(low_policy.parameters(), lr=config.learning_rate)
    high_optimizer = optim.Adam(high_policy.parameters(), lr=config.learning_rate)
    low_losses: list[float] = []
    high_label_count = 0
    for _ in range(config.epochs):
        for supervision in train:
            tensors = _to_device(supervision, mean, std, device)
            commands = _continue_commands(tensors["node_features"])
            prediction = low_policy(
                tensors["node_features"],
                tensors["edge_features"],
                tensors["adjacency"],
                tensors["active_mask"],
                commands,
            )
            low_terms = low_bc_loss(prediction, tensors["low_actions"], tensors["low_action_mask"])
            low_total = low_terms.total(
                normalized_mse_coef=config.normalized_mse_coef,
                direction_coef=config.direction_coef,
                speed_coef=config.speed_coef,
            )
            low_optimizer.zero_grad()
            low_total.backward()
            torch.nn.utils.clip_grad_norm_(low_policy.parameters(), 1.0)
            low_optimizer.step()
            low_losses.append(float(low_total.detach().cpu()))
            output = high_policy(
                tensors["node_features"],
                tensors["edge_features"],
                tensors["adjacency"],
                tensors["active_mask"],
            )
            targets = _empty_optional_high_targets(tensors)
            high_terms = high_bc_loss(output, targets)
            high_label_count += high_terms.label_count
            if high_terms.classification is not None:
                high_optimizer.zero_grad()
                high_terms.classification.backward()
                torch.nn.utils.clip_grad_norm_(high_policy.parameters(), 1.0)
                high_optimizer.step()
    validation_low_loss = _evaluate_low(
        low_policy,
        validation,
        mean=mean,
        std=std,
        config=config,
        device=device,
    )
    _write_artifacts(
        output_directory,
        config=config,
        manifest=manifest,
        mean=mean,
        std=std,
        low_policy=low_policy,
        high_policy=high_policy,
    )
    return {
        "train_low_loss": float(np.mean(low_losses)),
        "high_label_count": high_label_count,
        "validation_low_loss": validation_low_loss,
        "validation_high_classification": None,
    }


def _graph_config() -> GraphBuildConfig:
    return GraphBuildConfig(
        communication_radius=100.0,
        risk_distance=70.0,
        prediction_horizon=5.0,
        top_k_neighbors=3,
        current_distance_edges=True,
        predicted_conflict_edges=True,
        self_loops=True,
    )


def _fit_node_normalization(samples: list[BCSupervision]) -> tuple[Tensor, Tensor]:
    values = torch.cat(
        [sample.features.node_features[sample.features.active_mask] for sample in samples], dim=0
    )
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-6)


def _evaluate_low(
    policy: ExpertLowLevelPolicy,
    samples: list[BCSupervision],
    *,
    mean: Tensor,
    std: Tensor,
    config: BehaviorCloningConfig,
    device: torch.device,
) -> float | None:
    """Return an episode-disjoint validation loss, or None for an empty partition."""
    if not samples:
        return None
    losses: list[float] = []
    policy.eval()
    with torch.no_grad():
        for sample in samples:
            tensors = _to_device(sample, mean, std, device)
            prediction = policy(
                tensors["node_features"],
                tensors["edge_features"],
                tensors["adjacency"],
                tensors["active_mask"],
                _continue_commands(tensors["node_features"]),
            )
            terms = low_bc_loss(prediction, tensors["low_actions"], tensors["low_action_mask"])
            losses.append(
                float(
                    terms.total(
                        normalized_mse_coef=config.normalized_mse_coef,
                        direction_coef=config.direction_coef,
                        speed_coef=config.speed_coef,
                    )
                    .cpu()
                )
            )
    policy.train()
    return float(np.mean(losses))


def _to_device(
    sample: BCSupervision, mean: Tensor, std: Tensor, device: torch.device
) -> dict[str, Tensor]:
    features = sample.features
    normalized_nodes = (features.node_features - mean) / std
    return {
        "node_features": normalized_nodes.to(device),
        "edge_features": features.edge_features.to(device),
        "adjacency": features.adjacency.to(device),
        "active_mask": features.active_mask.to(device),
        "low_actions": sample.low_actions.to(device),
        "low_action_mask": sample.low_action_mask.to(device),
        "high_actions": sample.high_actions.to(device),
        "high_action_mask": sample.high_action_mask.to(device),
    }


def _continue_commands(node_features: Tensor) -> Tensor:
    commands = torch.zeros(
        (*node_features.shape[:2], 6), dtype=node_features.dtype, device=node_features.device
    )
    commands[..., 0] = 1.0
    return commands


def _empty_optional_high_targets(tensors: dict[str, Tensor]) -> Any:
    from multiuav.learning.behavior_cloning import HighBCTargets

    shape = tensors["high_actions"].shape
    zeros = torch.zeros((*shape, 1), device=tensors["high_actions"].device)
    return HighBCTargets(
        actions=tensors["high_actions"],
        action_mask=tensors["high_action_mask"],
        delay=zeros,
        delay_mask=torch.zeros(shape, dtype=torch.bool, device=zeros.device),
        priority=zeros,
        priority_mask=torch.zeros(shape, dtype=torch.bool, device=zeros.device),
        local_subgoal=torch.zeros((*shape, 3), device=zeros.device),
        local_subgoal_mask=torch.zeros(shape, dtype=torch.bool, device=zeros.device),
    )


def _write_artifacts(
    output_directory: Path,
    *,
    config: BehaviorCloningConfig,
    manifest: BehaviorCloningManifest,
    mean: Tensor,
    std: Tensor,
    low_policy: ExpertLowLevelPolicy,
    high_policy: ExpertHighLevelPolicy,
) -> None:
    checkpoint_config = asdict(config)
    torch.save(
        {"state_dict": low_policy.state_dict(), "config": checkpoint_config},
        output_directory / "bc_low_level.pt",
    )
    torch.save(
        {"state_dict": high_policy.state_dict(), "config": checkpoint_config},
        output_directory / "bc_high_level.pt",
    )
    (output_directory / "normalization_stats.json").write_text(
        json.dumps({"node_mean": mean.tolist(), "node_std": std.tolist()}, indent=2),
        encoding="utf-8",
    )
    config_data = asdict(config)
    config_data["shards"] = [str(path) for path in config.shards]
    (output_directory / "training_config.yaml").write_text(
        yaml.safe_dump(config_data), encoding="utf-8"
    )
    (output_directory / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

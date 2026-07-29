"""Pure-PyTorch dense masked graph-attention encoder primitives."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Normal

_SQUASH_EPSILON = 1e-6


class DenseGraphAttentionEncoder(nn.Module):
    """Permutation-equivariant edge-conditioned attention over padded UAV nodes."""

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
        if min(node_feature_dim, edge_feature_dim, embedding_dim, num_heads, num_layers) < 1:
            raise ValueError("Graph encoder dimensions and layer count must be positive.")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.embedding_dim = embedding_dim
        self.node_encoder = nn.Linear(node_feature_dim, embedding_dim)
        self.layers = nn.ModuleList(
            [
                _DenseGraphAttentionLayer(
                    embedding_dim=embedding_dim,
                    edge_feature_dim=edge_feature_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """Encode `[B,N,*]` dense graph tensors and zero all inactive-node outputs."""
        self._validate_inputs(node_features, edge_features, adjacency, node_mask)
        hidden = self.node_encoder(node_features)
        hidden = hidden * node_mask.unsqueeze(dim=-1).to(dtype=hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, edge_features, adjacency, node_mask)
        return cast(Tensor, hidden * node_mask.unsqueeze(dim=-1).to(dtype=hidden.dtype))

    def _validate_inputs(
        self, node_features: Tensor, edge_features: Tensor, adjacency: Tensor, node_mask: Tensor
    ) -> None:
        if node_features.ndim != 3 or node_features.shape[-1] != self.node_feature_dim:
            raise ValueError("node_features must have shape [B,N,node_feature_dim].")
        batch_size, node_count, _ = node_features.shape
        if edge_features.shape != (batch_size, node_count, node_count, self.edge_feature_dim):
            raise ValueError("edge_features must have shape [B,N,N,edge_feature_dim].")
        if adjacency.shape != (batch_size, node_count, node_count) or adjacency.dtype != torch.bool:
            raise ValueError("adjacency must be a boolean tensor with shape [B,N,N].")
        if node_mask.shape != (batch_size, node_count) or node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be a boolean tensor with shape [B,N].")


class _DenseGraphAttentionLayer(nn.Module):
    """One edge-conditioned attention/residual/normalization block."""

    def __init__(self, *, embedding_dim: int, edge_feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.edge_score = nn.Linear(edge_feature_dim, num_heads, bias=False)
        self.edge_value = nn.Linear(edge_feature_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.first_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.second_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, hidden: Tensor, edge_features: Tensor, adjacency: Tensor, node_mask: Tensor
    ) -> Tensor:
        batch_size, node_count, _ = hidden.shape
        scores = self.edge_score(edge_features)
        valid_edges = self._valid_edges(adjacency, node_mask)
        masked_scores = scores.masked_fill(~valid_edges.unsqueeze(dim=-1), -1e9)
        weights = torch.softmax(masked_scores, dim=2)
        weights = weights * valid_edges.unsqueeze(dim=-1).to(dtype=weights.dtype)
        edge_values = self.edge_value(edge_features).reshape(
            batch_size, node_count, node_count, self.num_heads, self.head_dim
        )
        attended = torch.sum(weights.unsqueeze(dim=-1) * edge_values, dim=2).reshape(
            batch_size, node_count, self.embedding_dim
        )
        mask = node_mask.unsqueeze(dim=-1).to(dtype=hidden.dtype)
        hidden = self.first_norm(hidden + self.output(attended)) * mask
        return cast(Tensor, self.second_norm(hidden + self.feed_forward(hidden)) * mask)

    @staticmethod
    def _valid_edges(adjacency: Tensor, node_mask: Tensor) -> Tensor:
        """Ensure every active node can attend to itself, even if graph edges are empty."""
        batch_size, node_count = node_mask.shape
        receiver_active = node_mask[:, :, None]
        diagonal = torch.eye(node_count, dtype=torch.bool, device=adjacency.device).expand(
            batch_size, -1, -1
        )
        return (adjacency & receiver_active) | (diagonal & receiver_active)


class GraphActor(nn.Module):
    """Parameter-sharing squashed Gaussian actor operating on graph nodes."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive.")
        self.encoder = DenseGraphAttentionEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.node_feature_mean: Tensor
        self.node_feature_std: Tensor
        self.register_buffer("node_feature_mean", torch.zeros(node_feature_dim))
        self.register_buffer("node_feature_std", torch.ones(node_feature_dim))
        self.policy = nn.Linear(embedding_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def encode(
        self, node_features: Tensor, edge_features: Tensor, adjacency: Tensor, node_mask: Tensor
    ) -> Tensor:
        """Return encoded node embeddings for actions or auxiliary predictions."""
        normalized_nodes = (node_features - self.node_feature_mean) / self.node_feature_std
        return cast(Tensor, self.encoder(normalized_nodes, edge_features, adjacency, node_mask))

    def set_node_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Install train-split statistics when importing a BC-trained encoder."""
        if mean.shape != self.node_feature_mean.shape or std.shape != self.node_feature_std.shape:
            raise ValueError(
                "Node normalization statistics do not match GraphActor feature dimension."
            )
        if torch.any(std <= 0.0):
            raise ValueError("Node normalization standard deviations must be positive.")
        self.node_feature_mean.copy_(mean.to(self.node_feature_mean))
        self.node_feature_std.copy_(std.to(self.node_feature_std))

    def sample(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample or mean-act per active graph node."""
        embeddings = self.encode(node_features, edge_features, adjacency, node_mask)
        distribution = self._distribution(embeddings)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = raw_actions.tanh()
        log_probabilities = _squashed_log_probability(distribution, raw_actions, actions)
        entropy = distribution.entropy().sum(dim=-1)
        mask = node_mask.to(dtype=actions.dtype)
        return (
            actions * mask.unsqueeze(dim=-1),
            log_probabilities * mask,
            entropy * mask,
        )

    def evaluate_actions(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate saved normalized actions in the same graph coordinates."""
        embeddings = self.encode(node_features, edge_features, adjacency, node_mask)
        distribution = self._distribution(embeddings)
        bounded_actions = actions.clamp(-1.0 + _SQUASH_EPSILON, 1.0 - _SQUASH_EPSILON)
        raw_actions = 0.5 * (torch.log1p(bounded_actions) - torch.log1p(-bounded_actions))
        mask = node_mask.to(dtype=actions.dtype)
        return (
            _squashed_log_probability(distribution, raw_actions, bounded_actions) * mask,
            distribution.entropy().sum(dim=-1) * mask,
        )

    def _distribution(self, embeddings: Tensor) -> Normal:
        mean = self.policy(embeddings)
        return Normal(mean, self.log_std.exp().expand_as(mean))


class GraphCentralizedCritic(nn.Module):
    """Centralized critic from a masked, permutation-invariant graph summary."""

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
        self.value = nn.Linear(embedding_dim, 1)

    def forward(
        self, node_features: Tensor, edge_features: Tensor, adjacency: Tensor, node_mask: Tensor
    ) -> Tensor:
        """Predict one value per graph after masked mean aggregation."""
        embeddings = self.encoder(node_features, edge_features, adjacency, node_mask)
        mask = node_mask.unsqueeze(dim=-1).to(dtype=embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return cast(Tensor, self.value(pooled).squeeze(dim=-1))


class ConflictPredictionHead(nn.Module):
    """Predict future conflict probability and nonnegative minimum distance per edge."""

    def __init__(self, *, embedding_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.edge_encoder = nn.Linear(edge_feature_dim, embedding_dim)
        self.output = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 2)
        )
        self.distance_activation = nn.Softplus()

    def forward(self, node_embeddings: Tensor, edge_features: Tensor) -> tuple[Tensor, Tensor]:
        """Return `[B,N,N]` conflict logits and nonnegative distance predictions."""
        batch_size, node_count, embedding_dim = node_embeddings.shape
        if edge_features.shape[:3] != (batch_size, node_count, node_count):
            raise ValueError("edge_features must align with node_embeddings graph axes.")
        first = node_embeddings[:, :, None, :].expand(-1, -1, node_count, -1)
        second = node_embeddings[:, None, :, :].expand(-1, node_count, -1, -1)
        encoded_edges = self.edge_encoder(edge_features)
        output = self.output(torch.cat((first, second, encoded_edges), dim=-1))
        return output[..., 0], self.distance_activation(output[..., 1])


def _squashed_log_probability(distribution: Normal, raw_actions: Tensor, actions: Tensor) -> Tensor:
    """Evaluate tanh-squashed Gaussian log probabilities with Jacobian correction."""
    correction = torch.log(1.0 - actions.square() + _SQUASH_EPSILON)
    return cast(Tensor, (distribution.log_prob(raw_actions) - correction).sum(dim=-1))

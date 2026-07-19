"""Predictive dense conflict-graph construction from batched UAV kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

EDGE_FEATURE_DIMENSION = 15


@dataclass(frozen=True)
class GraphBuildConfig:
    """Numerical and sparse-connectivity settings for one predictive graph."""

    communication_radius: float
    risk_distance: float
    prediction_horizon: float
    top_k_neighbors: int
    current_distance_edges: bool
    predicted_conflict_edges: bool
    self_loops: bool
    position_scale: float = 100.0
    velocity_scale: float = 20.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if (
            min(
                self.communication_radius,
                self.risk_distance,
                self.prediction_horizon,
                self.position_scale,
                self.velocity_scale,
                self.epsilon,
            )
            <= 0.0
        ):
            raise ValueError("Graph distance, horizon, scales, and epsilon must be positive.")
        if self.top_k_neighbors < 0:
            raise ValueError("top_k_neighbors must be nonnegative.")


@dataclass(frozen=True)
class ConflictGraph:
    """Fixed-rank graph tensors using receiver-neighbour edge order `[B,i,j,*]`."""

    edge_features: Tensor
    adjacency: Tensor
    node_mask: Tensor


def compute_cpa_features(
    *,
    relative_positions: Tensor,
    relative_velocities: Tensor,
    prediction_horizon: float,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return clipped CPA time and separation for arbitrary trailing-3 tensors."""
    if relative_positions.shape != relative_velocities.shape or relative_positions.shape[-1] != 3:
        raise ValueError("CPA inputs must have matching trailing dimension 3.")
    if prediction_horizon <= 0.0 or epsilon <= 0.0:
        raise ValueError("prediction_horizon and epsilon must be positive.")
    denominator = relative_velocities.square().sum(dim=-1) + epsilon
    time_to_cpa = -(relative_positions * relative_velocities).sum(dim=-1) / denominator
    time_to_cpa = time_to_cpa.clamp(0.0, prediction_horizon)
    distance_at_cpa = torch.linalg.vector_norm(
        relative_positions + time_to_cpa.unsqueeze(dim=-1) * relative_velocities,
        dim=-1,
    )
    return time_to_cpa, distance_at_cpa


class ConflictGraphBuilder:
    """Build a sparse, masked dense graph without any learned parameters."""

    def __init__(self, config: GraphBuildConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        positions: Tensor,
        velocities: Tensor,
        goals: Tensor,
        active_mask: Tensor,
    ) -> ConflictGraph:
        """Construct predictive edge features and sparse adjacency for `[B,N,3]` inputs."""
        self._validate_inputs(positions, velocities, goals, active_mask)
        relative_positions = positions[:, None, :, :] - positions[:, :, None, :]
        relative_velocities = velocities[:, None, :, :] - velocities[:, :, None, :]
        current_distance = torch.linalg.vector_norm(relative_positions, dim=-1)
        time_to_cpa, distance_at_cpa = compute_cpa_features(
            relative_positions=relative_positions,
            relative_velocities=relative_velocities,
            prediction_horizon=self.config.prediction_horizon,
            epsilon=self.config.epsilon,
        )
        pair_active = active_mask[:, :, None] & active_mask[:, None, :]
        communication_available = current_distance <= self.config.communication_radius
        predicted_shortfall = (self.config.risk_distance - distance_at_cpa).clamp_min(
            0.0
        ) / self.config.risk_distance
        relative_goal_direction = self._relative_goal_direction(positions, goals)
        edge_features = torch.cat(
            (
                relative_positions / self.config.position_scale,
                relative_velocities / self.config.velocity_scale,
                (current_distance / self.config.position_scale).unsqueeze(dim=-1),
                (time_to_cpa / self.config.prediction_horizon).unsqueeze(dim=-1),
                (distance_at_cpa / self.config.position_scale).unsqueeze(dim=-1),
                predicted_shortfall.unsqueeze(dim=-1),
                relative_goal_direction,
                communication_available.to(dtype=positions.dtype).unsqueeze(dim=-1),
                pair_active.to(dtype=positions.dtype).unsqueeze(dim=-1),
            ),
            dim=-1,
        )
        adjacency = self._adjacency(
            current_distance=current_distance,
            distance_at_cpa=distance_at_cpa,
            pair_active=pair_active,
        )
        return ConflictGraph(
            edge_features=edge_features,
            adjacency=adjacency,
            node_mask=active_mask,
        )

    def _adjacency(
        self,
        *,
        current_distance: Tensor,
        distance_at_cpa: Tensor,
        pair_active: Tensor,
    ) -> Tensor:
        batch_size, node_count, _ = current_distance.shape
        diagonal = torch.eye(node_count, dtype=torch.bool, device=current_distance.device).expand(
            batch_size, -1, -1
        )
        non_self_active = pair_active & ~diagonal
        adjacency = torch.zeros_like(pair_active)
        if self.config.current_distance_edges:
            adjacency |= non_self_active & (current_distance <= self.config.communication_radius)
        if self.config.predicted_conflict_edges:
            adjacency |= non_self_active & (distance_at_cpa <= self.config.risk_distance)
        adjacency |= self._top_k_edges(current_distance, non_self_active)
        if self.config.self_loops:
            adjacency |= pair_active & diagonal
        return adjacency

    def _top_k_edges(self, current_distance: Tensor, candidates: Tensor) -> Tensor:
        """Select at most k nearest active non-self neighbours per receiver node."""
        if self.config.top_k_neighbors == 0:
            return torch.zeros_like(candidates)
        node_count = current_distance.shape[-1]
        neighbour_count = min(self.config.top_k_neighbors, max(node_count - 1, 0))
        if neighbour_count == 0:
            return torch.zeros_like(candidates)
        masked_distances = current_distance.masked_fill(~candidates, float("inf"))
        _, nearest_indices = torch.topk(masked_distances, k=neighbour_count, dim=-1, largest=False)
        selected = torch.zeros_like(candidates)
        selected.scatter_(dim=-1, index=nearest_indices, value=True)
        return selected & candidates

    def _relative_goal_direction(self, positions: Tensor, goals: Tensor) -> Tensor:
        own_goal_vectors = goals - positions
        norms = torch.linalg.vector_norm(own_goal_vectors, dim=-1, keepdim=True)
        directions = own_goal_vectors / norms.clamp_min(self.config.epsilon)
        directions = torch.where(
            norms > self.config.epsilon, directions, torch.zeros_like(directions)
        )
        return directions[:, None, :, :] - directions[:, :, None, :]

    @staticmethod
    def _validate_inputs(
        positions: Tensor, velocities: Tensor, goals: Tensor, active_mask: Tensor
    ) -> None:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [B,N,3].")
        if velocities.shape != positions.shape or goals.shape != positions.shape:
            raise ValueError("velocities and goals must match positions shape [B,N,3].")
        if active_mask.shape != positions.shape[:2] or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a boolean tensor with shape [B,N].")

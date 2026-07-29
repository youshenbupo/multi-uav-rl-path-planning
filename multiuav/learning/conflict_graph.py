"""Predictive dense conflict-graph construction from batched UAV kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

EDGE_FEATURE_DIMENSION = 16


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
    uncertainty_scale: float = 1.0
    uncertainty_risk_gain: float = 0.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if (
            min(
                self.communication_radius,
                self.risk_distance,
                self.prediction_horizon,
                self.position_scale,
                self.velocity_scale,
                self.uncertainty_scale,
                self.epsilon,
            )
            <= 0.0
        ):
            raise ValueError("Graph distance, horizon, scales, and epsilon must be positive.")
        if self.top_k_neighbors < 0:
            raise ValueError("top_k_neighbors must be nonnegative.")
        if self.uncertainty_risk_gain < 0.0:
            raise ValueError("uncertainty_risk_gain must be nonnegative.")


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
                torch.zeros_like(current_distance).unsqueeze(dim=-1),
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

    def build_from_knowledge(
        self,
        *,
        positions: Tensor,
        received_positions: Tensor,
        received_velocities: Tensor,
        goals: Tensor,
        active_mask: Tensor,
        knowledge_valid: Tensor,
        knowledge_ages: Tensor,
        predicted_positions: Tensor | None = None,
        knowledge_uncertainty: Tensor | None = None,
    ) -> ConflictGraph:
        """Build actor edges only from each receiver's delivered neighbour state."""
        self._validate_inputs(positions, positions, goals, active_mask)
        expected_pair_shape = (*positions.shape[:2], positions.shape[1])
        if received_positions.shape != (*expected_pair_shape, 3):
            raise ValueError("received_positions must have shape [B,N,N,3].")
        if received_velocities.shape != received_positions.shape:
            raise ValueError("received_velocities must match received_positions.")
        if knowledge_valid.shape != expected_pair_shape or knowledge_valid.dtype != torch.bool:
            raise ValueError("knowledge_valid must be a boolean tensor with shape [B,N,N].")
        if knowledge_ages.shape != expected_pair_shape:
            raise ValueError("knowledge_ages must have shape [B,N,N].")
        if predicted_positions is None:
            predicted_positions = received_positions
        if predicted_positions.shape != received_positions.shape:
            raise ValueError("predicted_positions must match received_positions.")
        if knowledge_uncertainty is None:
            knowledge_uncertainty = torch.zeros_like(knowledge_ages, dtype=positions.dtype)
        if knowledge_uncertainty.shape != expected_pair_shape:
            raise ValueError("knowledge_uncertainty must have shape [B,N,N].")
        if not torch.isfinite(knowledge_uncertainty).all() or (knowledge_uncertainty < 0.0).any():
            raise ValueError("knowledge_uncertainty must be finite and nonnegative.")
        receiver_indices = torch.arange(positions.shape[1], device=positions.device)
        receiver_velocities = received_velocities[:, receiver_indices, receiver_indices, :]
        relative_positions = predicted_positions - positions[:, :, None, :]
        relative_velocities = received_velocities - receiver_velocities[:, :, None, :]
        current_distance = torch.linalg.vector_norm(relative_positions, dim=-1)
        time_to_cpa, distance_at_cpa = compute_cpa_features(
            relative_positions=relative_positions,
            relative_velocities=relative_velocities,
            prediction_horizon=self.config.prediction_horizon,
            epsilon=self.config.epsilon,
        )
        receiver_active = knowledge_valid[:, receiver_indices, receiver_indices]
        actor_visible_pairs = receiver_active[:, :, None].expand_as(knowledge_valid)
        known_pair = actor_visible_pairs & knowledge_valid
        normalized_uncertainty = knowledge_uncertainty.to(dtype=positions.dtype)
        normalized_uncertainty = normalized_uncertainty / self.config.uncertainty_scale
        normalized_uncertainty = torch.where(
            known_pair, normalized_uncertainty, torch.zeros_like(normalized_uncertainty)
        )
        risk_adjusted_cpa_distance = (
            distance_at_cpa - self.config.uncertainty_risk_gain * knowledge_uncertainty
        ).clamp_min(0.0)
        predicted_shortfall = (self.config.risk_distance - risk_adjusted_cpa_distance).clamp_min(
            0.0
        ) / self.config.risk_distance
        relative_goal_direction = torch.zeros_like(relative_positions)
        normalized_age = knowledge_ages.to(dtype=positions.dtype).clamp_min(0.0)
        normalized_age = normalized_age / self.config.prediction_horizon
        normalized_age = torch.where(known_pair, normalized_age, torch.zeros_like(normalized_age))
        edge_features = torch.cat(
            (
                relative_positions / self.config.position_scale,
                relative_velocities / self.config.velocity_scale,
                (current_distance / self.config.position_scale).unsqueeze(dim=-1),
                (time_to_cpa / self.config.prediction_horizon).unsqueeze(dim=-1),
                (distance_at_cpa / self.config.position_scale).unsqueeze(dim=-1),
                predicted_shortfall.unsqueeze(dim=-1),
                relative_goal_direction,
                known_pair.to(dtype=positions.dtype).unsqueeze(dim=-1),
                normalized_age.unsqueeze(dim=-1),
                normalized_uncertainty.unsqueeze(dim=-1),
            ),
            dim=-1,
        )
        adjacency = self._adjacency(
            current_distance=current_distance,
            distance_at_cpa=risk_adjusted_cpa_distance,
            pair_active=actor_visible_pairs,
            knowledge_available=known_pair,
        )
        return ConflictGraph(
            edge_features=edge_features, adjacency=adjacency, node_mask=active_mask
        )

    def _adjacency(
        self,
        *,
        current_distance: Tensor,
        distance_at_cpa: Tensor,
        pair_active: Tensor,
        knowledge_available: Tensor | None = None,
    ) -> Tensor:
        batch_size, node_count, _ = current_distance.shape
        diagonal = torch.eye(node_count, dtype=torch.bool, device=current_distance.device).expand(
            batch_size, -1, -1
        )
        available_pairs = pair_active if knowledge_available is None else knowledge_available
        non_self_active = available_pairs & ~diagonal
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

    def _relative_goal_direction_from_knowledge(
        self, positions: Tensor, received_positions: Tensor, goals: Tensor
    ) -> Tensor:
        receiver_goal_vectors = goals - positions
        receiver_norms = torch.linalg.vector_norm(receiver_goal_vectors, dim=-1, keepdim=True)
        receiver_directions = receiver_goal_vectors / receiver_norms.clamp_min(self.config.epsilon)
        receiver_directions = torch.where(
            receiver_norms > self.config.epsilon,
            receiver_directions,
            torch.zeros_like(receiver_directions),
        )
        sender_goal_vectors = goals[:, None, :, :] - received_positions
        sender_norms = torch.linalg.vector_norm(sender_goal_vectors, dim=-1, keepdim=True)
        sender_directions = sender_goal_vectors / sender_norms.clamp_min(self.config.epsilon)
        sender_directions = torch.where(
            sender_norms > self.config.epsilon,
            sender_directions,
            torch.zeros_like(sender_directions),
        )
        return sender_directions - receiver_directions[:, :, None, :]

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

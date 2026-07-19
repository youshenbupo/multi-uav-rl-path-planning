"""Conversion boundary between normalized environment actions and physical CBF velocities."""

from __future__ import annotations

import numpy as np

from multiuav.envs.observations import EnvironmentSnapshot
from multiuav.safety.qp_filter import OSQPSafetyFilter, SafetyFilterDecision


class NormalizedActionCBFAdapter:
    """Apply a physical-velocity CBF filter while preserving the environment's `[-1,1]` API."""

    def __init__(self, safety_filter: OSQPSafetyFilter) -> None:
        self.safety_filter = safety_filter

    def filter_normalized(
        self, snapshot: EnvironmentSnapshot, normalized_actions: np.ndarray
    ) -> tuple[np.ndarray, SafetyFilterDecision]:
        """Filter normalized `[N,3]` actions and return normalized safe actions plus telemetry."""
        actions = np.asarray(normalized_actions, dtype=float)
        expected = np.asarray(snapshot.positions).shape
        if actions.shape != expected:
            raise ValueError(f"Normalized CBF actions must have shape {expected}.")
        clipped = np.clip(np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        scales = np.array(
            [
                snapshot.max_horizontal_speed,
                snapshot.max_horizontal_speed,
                snapshot.max_vertical_speed,
            ],
            dtype=float,
        )
        decision = self.safety_filter.filter(snapshot, clipped * scales)
        safe_normalized = np.clip(decision.u_safe / scales, -1.0, 1.0)
        safe_normalized[~snapshot.active_mask] = 0.0
        return safe_normalized, decision

"""Replaceable kinematic models for the multi-UAV reinforcement-learning environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
WorldBounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class DynamicsResult:
    """One transition result with commanded velocity and boundary status."""

    position: FloatArray
    velocity: FloatArray
    boundary_clipped: bool


class DynamicsModel(Protocol):
    """Minimal boundary for later replacement by a double-integrator model."""

    def advance(
        self, position: FloatArray, action: FloatArray, world_bounds: WorldBounds
    ) -> DynamicsResult:
        """Return the next bounded position and applied velocity."""


@dataclass(frozen=True)
class SingleIntegrator3D:
    """Three-dimensional single integrator with independent horizontal/vertical limits."""

    dt: float
    max_horizontal_speed: float
    max_vertical_speed: float
    normalize_actions: bool = True

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.max_horizontal_speed <= 0.0 or self.max_vertical_speed <= 0.0:
            raise ValueError("Dynamics dt and speed limits must be positive.")

    def advance(
        self, position: FloatArray, action: FloatArray, world_bounds: WorldBounds
    ) -> DynamicsResult:
        """Apply ``p_next = p + dt * v`` after speed and world-bound clipping."""
        point = np.asarray(position, dtype=float).reshape(3)
        command = np.asarray(action, dtype=float).reshape(3)
        velocity = self._velocity_from_action(command)
        proposed = point + self.dt * velocity
        lower = np.asarray([axis[0] for axis in world_bounds], dtype=float)
        upper = np.asarray([axis[1] for axis in world_bounds], dtype=float)
        bounded = np.clip(proposed, lower, upper)
        return DynamicsResult(
            position=np.asarray(bounded, dtype=float),
            velocity=np.asarray(velocity, dtype=float),
            boundary_clipped=not np.allclose(proposed, bounded, rtol=0.0, atol=1e-12),
        )

    def _velocity_from_action(self, action: FloatArray) -> FloatArray:
        command = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize_actions:
            command = np.clip(command, -1.0, 1.0)
            command = command * np.array(
                [self.max_horizontal_speed, self.max_horizontal_speed, self.max_vertical_speed]
            )
        horizontal_norm = float(np.linalg.norm(command[:2]))
        if horizontal_norm > self.max_horizontal_speed:
            command[:2] *= self.max_horizontal_speed / horizontal_norm
        command[2] = float(np.clip(command[2], -self.max_vertical_speed, self.max_vertical_speed))
        return np.asarray(command, dtype=float)

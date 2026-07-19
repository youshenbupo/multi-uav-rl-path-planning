"""Deterministic constant-velocity obstacle world state."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from multiuav.core.models import DynamicCylinder, FloatArray


@dataclass(frozen=True)
class DynamicWorldState:
    """Immutable obstacle definitions plus the current discrete simulation step."""

    obstacles: tuple[DynamicCylinder, ...]
    dt: float
    step_count: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt < 0.0:
            raise ValueError("Dynamic world dt must be finite and nonnegative.")
        if self.step_count < 0:
            raise ValueError("Dynamic world step_count must be nonnegative.")
        identifiers = [obstacle.identifier for obstacle in self.obstacles]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Dynamic cylinder identifiers must be unique.")

    @property
    def centers(self) -> FloatArray:
        """Return current centers in stable obstacle order as an independent array."""
        if not self.obstacles:
            return np.empty((0, 3), dtype=float)
        return np.asarray(
            [
                obstacle.initial_center + self.step_count * self.dt * obstacle.velocity
                for obstacle in self.obstacles
            ],
            dtype=float,
        )

    def advance(self) -> DynamicWorldState:
        """Advance exactly one environment step without mutating obstacle definitions."""
        return replace(self, step_count=self.step_count + 1)


def validate_dynamic_trajectories(
    obstacles: tuple[DynamicCylinder, ...],
    *,
    dt: float,
    max_steps: int,
    world_x: tuple[float, float],
    world_y: tuple[float, float],
    world_z: tuple[float, float],
) -> None:
    """Reject constant-velocity paths whose centers leave the configured world horizon."""
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("Dynamic trajectory dt must be finite and nonnegative.")
    if max_steps < 0:
        raise ValueError("Dynamic trajectory max_steps must be nonnegative.")
    bounds = (world_x, world_y, world_z)
    for bound in bounds:
        if len(bound) != 2 or not np.isfinite(bound).all() or bound[0] > bound[1]:
            raise ValueError("Dynamic trajectory world bounds must be finite ordered pairs.")
    for obstacle in obstacles:
        start = obstacle.initial_center
        end = start + max_steps * dt * obstacle.velocity
        for axis, (lower, upper) in enumerate(bounds):
            if min(start[axis], end[axis]) < lower or max(start[axis], end[axis]) > upper:
                raise ValueError(
                    f"Dynamic cylinder '{obstacle.identifier}' leaves world bounds on axis {axis}."
                )

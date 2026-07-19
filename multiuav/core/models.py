"""Typed, NumPy-backed data structures for multi-UAV evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DynamicCylinder:
    """Finite-height vertical cylinder moving at a constant three-dimensional velocity."""

    identifier: str
    initial_center: FloatArray
    velocity: FloatArray
    radius: float
    height: float

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("Dynamic cylinder identifier must be nonempty.")
        initial_center = _immutable_vector(self.initial_center, "initial_center")
        velocity = _immutable_vector(self.velocity, "velocity")
        if self.radius <= 0.0 or not np.isfinite(self.radius):
            raise ValueError("Dynamic cylinder radius must be finite and positive.")
        if self.height <= 0.0 or not np.isfinite(self.height):
            raise ValueError("Dynamic cylinder height must be finite and positive.")
        object.__setattr__(self, "initial_center", initial_center)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class CylindricalThreat:
    """Vertical cylinder represented as ``[x, y, radius, height]``."""

    center_x: float
    center_y: float
    radius: float
    height: float


@dataclass(frozen=True)
class TerrainMap:
    """Rectilinear terrain grids whose rows represent Y and columns represent X."""

    x_grid: FloatArray
    y_grid: FloatArray
    heights: FloatArray


@dataclass(frozen=True)
class UAVMission:
    """Start and goal points for one vehicle."""

    start: FloatArray
    goal: FloatArray


@dataclass(frozen=True)
class Scenario:
    """Immutable scenario parameters used by geometry and evaluation."""

    terrain: TerrainMap
    threats: tuple[CylindricalThreat, ...]
    missions: tuple[UAVMission, ...]
    min_clearance: float = 35.0
    max_clearance: float = 140.0
    target_clearance: float = 80.0
    max_turn_degrees: float = 65.0
    safe_separation: float = 70.0
    threat_margin: float = 25.0
    cruise_speed: float = 35.0
    collision_samples: int = 45
    large_penalty: float = 1e6
    spatial_collision_weight: float = 1200.0
    time_collision_weight: float = 1800.0
    sync_weight: float = 0.35
    world_x: tuple[float, float] = (0.0, 1000.0)
    world_y: tuple[float, float] = (0.0, 1000.0)
    world_z: tuple[float, float] = (0.0, 320.0)
    dynamic_obstacles: tuple[DynamicCylinder, ...] = ()
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "length": 1.0,
            "altitude": 0.55,
            "turn": 1.0,
            "threat": 1.2,
            "sync": 0.35,
        }
    )


@dataclass(frozen=True)
class Trajectory:
    """Ordered Cartesian ``(N, 3)`` waypoints for a single UAV."""

    points: FloatArray


@dataclass(frozen=True)
class CostBreakdown:
    """The seven requested objective components and aggregate total."""

    j_len: float
    j_alt: float
    j_turn: float
    j_thr: float
    j_spa: float
    j_tmp: float
    j_sync: float
    total_cost: float


@dataclass(frozen=True)
class ConstraintResult:
    """Strict feasibility flags and aggregate violation metrics."""

    terrain_feasible: bool
    threat_feasible: bool
    turning_feasible: bool
    spatial_feasible: bool
    temporal_feasible: bool


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluator output consumed by future environments and tests."""

    total_cost: float
    cost_breakdown: CostBreakdown
    strict_success: bool
    minimum_separation: float
    temporal_conflict_count: int
    constraint_details: ConstraintResult


def _immutable_vector(value: FloatArray, name: str) -> FloatArray:
    """Return a finite read-only three-vector for immutable model records."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"Dynamic cylinder {name} must be a finite vector with shape [3].")
    copied = vector.copy()
    copied.setflags(write=False)
    return copied

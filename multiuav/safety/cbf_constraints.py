"""Linear control-barrier rows for the Phase-8 three-dimensional single integrator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from multiuav.core.models import TerrainMap
from multiuav.envs.observations import EnvironmentSnapshot
from multiuav.geometry.terrain import terrain_height


@dataclass(frozen=True)
class CBFConfig:
    """Numerical and geometric constants for execution-only barrier construction."""

    alpha: float = 1.0
    terrain_gradient_epsilon: float = 0.25
    threat_vertical_influence: float = 0.0
    horizontal_speed_polygon_sides: int = 16
    slack_penalty: float = 1_000.0
    max_solve_time_seconds: float = 0.02

    def __post_init__(self) -> None:
        if self.alpha <= 0.0 or self.terrain_gradient_epsilon <= 0.0:
            raise ValueError("CBF alpha and terrain gradient epsilon must be positive.")
        if self.threat_vertical_influence < 0.0 or self.slack_penalty <= 0.0:
            raise ValueError("CBF threat influence and slack penalty are invalid.")
        if self.horizontal_speed_polygon_sides < 4 or self.max_solve_time_seconds <= 0.0:
            raise ValueError("CBF speed polygon and solve-time settings are invalid.")


@dataclass(frozen=True)
class CBFConstraintRow:
    """One lower-bounded linear CBF row: ``coefficients @ u >= lower``."""

    kind: str
    barrier: float
    coefficients: np.ndarray
    lower: float


class CBFConstraintBuilder:
    """Build joint-control CBF rows only for currently active UAVs."""

    def __init__(self, config: CBFConfig) -> None:
        self.config = config

    def build(self, snapshot: EnvironmentSnapshot) -> tuple[CBFConstraintRow, ...]:
        """Create pairwise, terrain, threat, and world-boundary barrier rows."""
        positions = np.asarray(snapshot.positions, dtype=float)
        active_mask = np.asarray(snapshot.active_mask, dtype=bool)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("CBF positions must have shape [N,3].")
        if active_mask.shape != (len(positions),):
            raise ValueError("CBF active mask must have shape [N].")
        if not np.isfinite(positions).all():
            raise ValueError("CBF cannot construct constraints from non-finite positions.")
        rows: list[CBFConstraintRow] = []
        for first in range(len(positions)):
            if not active_mask[first]:
                continue
            for second in range(first + 1, len(positions)):
                if active_mask[second]:
                    rows.append(self._separation_row(snapshot, first, second))
            rows.append(self._terrain_row(snapshot, first))
            rows.extend(self._threat_rows(snapshot, first))
            rows.extend(self._boundary_rows(snapshot, first))
        return tuple(rows)

    def _separation_row(
        self, snapshot: EnvironmentSnapshot, first: int, second: int
    ) -> CBFConstraintRow:
        relative = snapshot.positions[first] - snapshot.positions[second]
        safe_distance = snapshot.scenario.safe_separation
        barrier = float(relative @ relative - safe_distance**2)
        coefficients = np.zeros(3 * len(snapshot.positions), dtype=float)
        coefficients[3 * first : 3 * first + 3] = 2.0 * relative
        coefficients[3 * second : 3 * second + 3] = -2.0 * relative
        return self._row("uav_separation", barrier, coefficients)

    def _terrain_row(self, snapshot: EnvironmentSnapshot, index: int) -> CBFConstraintRow:
        point = snapshot.positions[index]
        terrain = snapshot.scenario.terrain
        height = float(terrain_height(terrain, point[:2])[0])
        gradient = self._terrain_gradient(terrain, point[:2])
        barrier = float(point[2] - height - snapshot.scenario.min_clearance)
        coefficients = np.zeros(3 * len(snapshot.positions), dtype=float)
        coefficients[3 * index : 3 * index + 3] = np.array(
            [-gradient[0], -gradient[1], 1.0], dtype=float
        )
        return self._row("terrain_clearance", barrier, coefficients)

    def _threat_rows(self, snapshot: EnvironmentSnapshot, index: int) -> list[CBFConstraintRow]:
        point = snapshot.positions[index]
        rows: list[CBFConstraintRow] = []
        for threat_index, threat in enumerate(snapshot.scenario.threats):
            if point[2] > threat.height + self.config.threat_vertical_influence:
                continue
            relative = point[:2] - np.array([threat.center_x, threat.center_y], dtype=float)
            safe_radius = threat.radius + snapshot.scenario.threat_margin
            barrier = float(relative @ relative - safe_radius**2)
            coefficients = np.zeros(3 * len(snapshot.positions), dtype=float)
            coefficients[3 * index : 3 * index + 2] = 2.0 * relative
            rows.append(self._row(f"cylindrical_threat_{threat_index}", barrier, coefficients))
        return rows

    def _boundary_rows(self, snapshot: EnvironmentSnapshot, index: int) -> list[CBFConstraintRow]:
        point = snapshot.positions[index]
        axes = (snapshot.scenario.world_x, snapshot.scenario.world_y, snapshot.scenario.world_z)
        rows: list[CBFConstraintRow] = []
        for axis, (lower_bound, upper_bound) in enumerate(axes):
            lower_coefficients = np.zeros(3 * len(snapshot.positions), dtype=float)
            lower_coefficients[3 * index + axis] = 1.0
            rows.append(
                self._row(
                    f"world_lower_{axis}", float(point[axis] - lower_bound), lower_coefficients
                )
            )
            upper_coefficients = np.zeros(3 * len(snapshot.positions), dtype=float)
            upper_coefficients[3 * index + axis] = -1.0
            rows.append(
                self._row(
                    f"world_upper_{axis}", float(upper_bound - point[axis]), upper_coefficients
                )
            )
        return rows

    def _row(self, kind: str, barrier: float, coefficients: np.ndarray) -> CBFConstraintRow:
        return CBFConstraintRow(
            kind=kind,
            barrier=barrier,
            coefficients=coefficients,
            lower=-self.config.alpha * barrier,
        )

    def _terrain_gradient(self, terrain: TerrainMap, point_xy: np.ndarray) -> np.ndarray:
        epsilon = self.config.terrain_gradient_epsilon
        query_x = np.asarray(point_xy, dtype=float)
        query_y = np.asarray(point_xy, dtype=float)
        query_x[0] += epsilon
        query_y[1] += epsilon
        base_height = float(terrain_height(terrain, point_xy)[0])
        gradient_x = (float(terrain_height(terrain, query_x)[0]) - base_height) / epsilon
        gradient_y = (float(terrain_height(terrain, query_y)[0]) - base_height) / epsilon
        return np.array([gradient_x, gradient_y], dtype=float)


def load_cbf_config(path: Path) -> CBFConfig:
    """Load all CBF solver settings from one explicit YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("CBF configuration must contain a YAML mapping.")
    return CBFConfig(**dict(payload))

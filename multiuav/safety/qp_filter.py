"""Joint direct-OSQP control-barrier safety filter for physical desired velocities."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from multiuav.envs.observations import EnvironmentSnapshot
from multiuav.safety.cbf_constraints import CBFConfig, CBFConstraintBuilder, CBFConstraintRow
from multiuav.safety.emergency_policy import EmergencyPolicy

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyFilterDecision:
    """Per-step CBF filtering telemetry required for execution and evaluation audit."""

    u_rl: np.ndarray
    u_safe: np.ndarray
    intervention_norm: float
    active_constraint_count: int
    dynamic_constraint_count: int
    slack_value: float
    solver_status: str
    solve_time: float
    emergency_fallback_used: bool
    emergency_reason: str | None = None


class OSQPSafetyFilter:
    """Solve a warm-started joint velocity/slack QP without any PPO gradient path."""

    def __init__(
        self, config: CBFConfig, *, emergency_policy: EmergencyPolicy | None = None
    ) -> None:
        self.config = config
        self.builder = CBFConstraintBuilder(config)
        self.emergency_policy = (
            emergency_policy if emergency_policy is not None else EmergencyPolicy()
        )
        self._last_solution: np.ndarray | None = None

    def filter(
        self, snapshot: EnvironmentSnapshot, desired_velocities: np.ndarray
    ) -> SafetyFilterDecision:
        """Filter physical `[N,3]` desired velocities, never using them after failure."""
        requested = np.asarray(desired_velocities, dtype=float)
        expected = np.asarray(snapshot.positions).shape
        if requested.shape != expected:
            raise ValueError(f"CBF requested velocities must have shape {expected}.")
        if not np.isfinite(requested).all():
            return self._emergency(
                snapshot,
                np.nan_to_num(requested),
                "nonfinite_requested_velocity",
                0,
                0,
                0.0,
            )
        try:
            rows = self.builder.build(snapshot)
            dynamic_constraint_count = sum(
                row.kind.startswith("dynamic_cylinder_") for row in rows
            )
            solution, status, solve_time = self._solve(snapshot, requested, rows)
        except (ValueError, FloatingPointError, RuntimeError) as error:
            return self._emergency(
                snapshot,
                requested,
                f"solver_exception:{type(error).__name__}",
                0,
                0,
                0.0,
            )
        if solution is None:
            return self._emergency(
                snapshot, requested, status, len(rows), dynamic_constraint_count, solve_time
            )
        controls = solution[: 3 * len(requested)].reshape(requested.shape)
        slacks = solution[3 * len(requested) :]
        if not np.isfinite(controls).all() or not np.isfinite(slacks).all():
            return self._emergency(
                snapshot,
                requested,
                "nonfinite_solver_output",
                len(rows),
                dynamic_constraint_count,
                solve_time,
            )
        self._last_solution = solution.copy()
        return SafetyFilterDecision(
            u_rl=requested.copy(),
            u_safe=controls,
            intervention_norm=float(np.linalg.norm(controls - requested)),
            active_constraint_count=len(rows),
            dynamic_constraint_count=dynamic_constraint_count,
            slack_value=float(np.max(slacks, initial=0.0)),
            solver_status=status,
            solve_time=solve_time,
            emergency_fallback_used=False,
        )

    def _solve(
        self,
        snapshot: EnvironmentSnapshot,
        requested: np.ndarray,
        rows: tuple[CBFConstraintRow, ...],
    ) -> tuple[np.ndarray | None, str, float]:
        control_count = requested.size
        row_count = len(rows)
        variable_count = control_count + row_count
        quadratic = sparse.diags(
            np.concatenate(
                (
                    np.ones(control_count, dtype=float),
                    np.full(row_count, self.config.slack_penalty, dtype=float),
                )
            ),
            format="csc",
        )
        linear = np.concatenate((-requested.reshape(-1), np.zeros(row_count, dtype=float)))
        matrix, lower, upper = self._constraint_matrix(snapshot, rows, control_count, row_count)
        solver = osqp.OSQP()
        started = time.perf_counter()
        solver.setup(
            P=quadratic,
            q=linear,
            A=matrix,
            l=lower,
            u=upper,
            verbose=False,
            polishing=False,
            warm_starting=True,
            max_iter=20_000,
            eps_abs=1e-4,
            eps_rel=1e-4,
            time_limit=self.config.max_solve_time_seconds,
        )
        if self._last_solution is not None and self._last_solution.shape == (variable_count,):
            solver.warm_start(x=self._last_solution)
        result = solver.solve()
        elapsed = time.perf_counter() - started
        status = str(result.info.status).lower()
        if status != "solved" or elapsed > self.config.max_solve_time_seconds:
            failure_status = (
                status if elapsed <= self.config.max_solve_time_seconds else "solve_time_limit"
            )
            return None, failure_status, elapsed
        if result.x is None:
            return None, "missing_solver_solution", elapsed
        return np.asarray(result.x, dtype=float), status, elapsed

    def _constraint_matrix(
        self,
        snapshot: EnvironmentSnapshot,
        rows: tuple[CBFConstraintRow, ...],
        control_count: int,
        row_count: int,
    ) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
        cbf_coefficients = (
            np.vstack([row.coefficients for row in rows])
            if rows
            else np.empty((0, control_count), dtype=float)
        )
        cbf_lower = np.array([row.lower for row in rows], dtype=float)
        cbf_matrix = sparse.hstack(
            (sparse.csc_matrix(cbf_coefficients), sparse.eye(row_count, format="csc")), format="csc"
        )
        velocity_matrix, velocity_lower, velocity_upper = self._velocity_constraints(
            snapshot, control_count, row_count
        )
        slack_matrix = sparse.hstack(
            (sparse.csc_matrix((row_count, control_count)), sparse.eye(row_count, format="csc")),
            format="csc",
        )
        return (
            sparse.vstack((cbf_matrix, velocity_matrix, slack_matrix), format="csc"),
            np.concatenate((cbf_lower, velocity_lower, np.zeros(row_count, dtype=float))),
            np.concatenate(
                (
                    np.full(row_count, np.inf),
                    velocity_upper,
                    np.full(row_count, np.inf),
                )
            ),
        )

    def _velocity_constraints(
        self, snapshot: EnvironmentSnapshot, control_count: int, row_count: int
    ) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
        matrices: list[np.ndarray] = []
        lowers: list[float] = []
        uppers: list[float] = []
        sides = self.config.horizontal_speed_polygon_sides
        polygon_limit = snapshot.max_horizontal_speed * np.cos(np.pi / sides)
        for index, is_active in enumerate(snapshot.active_mask):
            if not is_active:
                for axis in range(3):
                    row = np.zeros(control_count, dtype=float)
                    row[3 * index + axis] = 1.0
                    matrices.append(row)
                    lowers.append(0.0)
                    uppers.append(0.0)
                continue
            for side in range(sides):
                angle = 2.0 * np.pi * side / sides
                row = np.zeros(control_count, dtype=float)
                row[3 * index : 3 * index + 2] = (np.cos(angle), np.sin(angle))
                matrices.append(row)
                lowers.append(-np.inf)
                uppers.append(polygon_limit)
            vertical = np.zeros(control_count, dtype=float)
            vertical[3 * index + 2] = 1.0
            matrices.append(vertical)
            lowers.append(-snapshot.max_vertical_speed)
            uppers.append(snapshot.max_vertical_speed)
        return sparse.hstack(
            (
                sparse.csc_matrix(np.asarray(matrices, dtype=float)),
                sparse.csc_matrix((len(matrices), row_count)),
            ),
            format="csc",
        ), np.asarray(lowers, dtype=float), np.asarray(uppers, dtype=float)

    def _emergency(
        self,
        snapshot: EnvironmentSnapshot,
        requested: np.ndarray,
        reason: str,
        active_constraint_count: int,
        dynamic_constraint_count: int,
        solve_time: float,
    ) -> SafetyFilterDecision:
        safe = self.emergency_policy.safe_actions(snapshot)
        LOGGER.warning("CBF filter emergency fallback: %s", reason)
        return SafetyFilterDecision(
            u_rl=np.asarray(requested, dtype=float).copy(),
            u_safe=safe,
            intervention_norm=float(np.linalg.norm(safe - np.nan_to_num(requested))),
            active_constraint_count=active_constraint_count,
            dynamic_constraint_count=dynamic_constraint_count,
            slack_value=0.0,
            solver_status=reason,
            solve_time=solve_time,
            emergency_fallback_used=True,
            emergency_reason=reason,
        )

"""Run the nine deterministic Phase-13 CBF safety scenarios and print telemetry JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiuav.core.models import CylindricalThreat, Scenario, TerrainMap, UAVMission  # noqa: E402
from multiuav.envs.observations import EnvironmentSnapshot  # noqa: E402
from multiuav.safety.cbf_constraints import CBFConfig, load_cbf_config  # noqa: E402
from multiuav.safety.qp_filter import OSQPSafetyFilter, SafetyFilterDecision  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build deterministic scenario-runner options; OSQP itself is a CPU solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/safety/cbf.yaml")
    parser.add_argument("--scenario", default="all", choices=("all", *_scenario_names()))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--max-solve-time-seconds", type=float)
    return parser


def run_scenarios(config: CBFConfig | None = None) -> dict[str, dict[str, object]]:
    """Execute all required deterministic states and return JSON-serializable evidence."""
    active_config = config if config is not None else CBFConfig(max_solve_time_seconds=0.2)
    reports: dict[str, dict[str, object]] = {}
    for name, snapshot, requested, predicate, force_failure in _scenario_cases():
        safety_filter: OSQPSafetyFilter
        if force_failure:
            safety_filter = _ForcedFailureFilter(active_config)
        else:
            safety_filter = OSQPSafetyFilter(active_config)
        decision = safety_filter.filter(snapshot, requested)
        bounded = _within_limits(decision.u_safe, snapshot)
        passed = bool(np.isfinite(decision.u_safe).all() and bounded and predicate(decision))
        reports[name] = {
            "passed": passed,
            "u_rl": decision.u_rl.tolist(),
            "u_safe": decision.u_safe.tolist(),
            "intervention_norm": decision.intervention_norm,
            "active_constraint_count": decision.active_constraint_count,
            "dynamic_constraint_count": decision.dynamic_constraint_count,
            "slack_value": decision.slack_value,
            "solver_status": decision.solver_status,
            "solve_time": decision.solve_time,
            "emergency_fallback_used": decision.emergency_fallback_used,
        }
    return reports


def main() -> None:
    """Print selected scenario evidence and fail loudly if any required invariant is false."""
    arguments = build_parser().parse_args()
    del arguments.device  # OSQP direct QP remains intentionally execution-side CPU work.
    config = load_cbf_config(arguments.config)
    if arguments.max_solve_time_seconds is not None:
        config = replace(config, max_solve_time_seconds=arguments.max_solve_time_seconds)
    reports = run_scenarios(config)
    selected = (
        reports
        if arguments.scenario == "all"
        else {arguments.scenario: reports[arguments.scenario]}
    )
    print(json.dumps(selected, indent=2))
    if not all(bool(report["passed"]) for report in selected.values()):
        raise SystemExit("At least one CBF scenario invariant failed.")


class _ForcedFailureFilter(OSQPSafetyFilter):
    """Deterministically prove the emergency path is not a raw-RL fallback."""

    def _solve(
        self,
        snapshot: EnvironmentSnapshot,
        requested: np.ndarray,
        rows: tuple[object, ...],
    ) -> tuple[np.ndarray | None, str, float, int, float, float]:
        del snapshot, requested, rows
        return None, "forced_infeasible_qp", 0.0, 0, 0.0, 0.0


def _scenario_cases() -> tuple[
    tuple[
        str,
        EnvironmentSnapshot,
        np.ndarray,
        Callable[[SafetyFilterDecision], bool],
        bool,
    ],
    ...,
]:
    head_on = _snapshot(np.array([[33.0, 50.0, 40.0], [67.0, 50.0, 40.0]]))
    crossing = _snapshot(np.array([[38.0, 38.0, 40.0], [62.0, 62.0, 40.0]]))
    convergence = _snapshot(
        np.array([[34.0, 50.0, 40.0], [66.0, 50.0, 40.0], [50.0, 77.7, 40.0]])
    )
    terrain = _snapshot(np.array([[50.0, 50.0, 9.0]]))
    threat = _snapshot(
        np.array([[84.0, 50.0, 20.0]]),
        threats=(CylindricalThreat(center_x=50.0, center_y=50.0, radius=10.0, height=70.0),),
    )
    unsafe = _snapshot(np.array([[44.0, 50.0, 40.0], [56.0, 50.0, 40.0]]))
    conflicts = _snapshot(
        np.array([[20.0, 50.0, 9.0], [42.0, 50.0, 9.0], [31.0, 70.0, 9.0]]),
        threats=(CylindricalThreat(center_x=50.0, center_y=50.0, radius=6.0, height=70.0),),
    )
    low_risk = _snapshot(np.array([[15.0, 20.0, 50.0], [85.0, 80.0, 50.0]]))
    return (
        ("head_on", head_on, np.array([[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]]), _intervenes, False),
        (
            "crossing",
            crossing,
            np.array([[14.0, 14.0, 0.0], [-14.0, -14.0, 0.0]]),
            _intervenes,
            False,
        ),
        (
            "three_way_convergence",
            convergence,
            np.array([[15.0, 0.0, 0.0], [-15.0, 0.0, 0.0], [0.0, -15.0, 0.0]]),
            _intervenes,
            False,
        ),
        ("near_terrain", terrain, np.array([[0.0, 0.0, -8.0]]), _climbs, False),
        (
            "near_cylindrical_threat",
            threat,
            np.array([[-15.0, 0.0, 0.0]]),
            _moves_away_from_threat,
            False,
        ),
        ("initially_unsafe", unsafe, np.zeros((2, 3)), _intervenes, False),
        ("multiple_conflicts", conflicts, np.zeros((3, 3)), _safe_multiple_conflict_result, False),
        ("forced_infeasible_qp", terrain, np.array([[0.0, 0.0, -8.0]]), _uses_emergency, True),
        (
            "low_risk_preservation",
            low_risk,
            np.array([[5.0, 2.0, 1.0], [-5.0, -2.0, -1.0]]),
            _preserves_low_risk,
            False,
        ),
    )


def _snapshot(
    positions: np.ndarray, *, threats: tuple[CylindricalThreat, ...] = ()
) -> EnvironmentSnapshot:
    terrain = TerrainMap(
        x_grid=np.array([0.0, 100.0]),
        y_grid=np.array([0.0, 100.0]),
        heights=np.zeros((2, 2)),
    )
    scenario = Scenario(
        terrain=terrain,
        threats=threats,
        missions=tuple(
            UAVMission(start=point.copy(), goal=np.array([90.0, 90.0, 40.0])) for point in positions
        ),
        safe_separation=30.0,
        min_clearance=10.0,
        world_x=(0.0, 100.0),
        world_y=(0.0, 100.0),
        world_z=(0.0, 100.0),
    )
    count = len(positions)
    return EnvironmentSnapshot(
        scenario=scenario,
        positions=positions.astype(float),
        velocities=np.zeros_like(positions, dtype=float),
        active_mask=np.ones(count, dtype=bool),
        previous_goal_distances=np.ones(count),
        step_count=0,
        max_steps=100,
        max_horizontal_speed=20.0,
        max_vertical_speed=10.0,
        boundary_clipped=np.zeros(count, dtype=bool),
    )


def _within_limits(actions: np.ndarray, snapshot: EnvironmentSnapshot) -> bool:
    return bool(
        np.all(np.linalg.norm(actions[:, :2], axis=1) <= snapshot.max_horizontal_speed + 1e-3)
        and np.all(np.abs(actions[:, 2]) <= snapshot.max_vertical_speed + 1e-3)
    )


def _intervenes(decision: SafetyFilterDecision) -> bool:
    return not decision.emergency_fallback_used and decision.intervention_norm > 1e-6


def _climbs(decision: SafetyFilterDecision) -> bool:
    return not decision.emergency_fallback_used and decision.u_safe[0, 2] > 0.0


def _moves_away_from_threat(decision: SafetyFilterDecision) -> bool:
    return not decision.emergency_fallback_used and decision.u_safe[0, 0] > 0.0


def _safe_multiple_conflict_result(decision: SafetyFilterDecision) -> bool:
    """Accept either a QP solution or its explicit bounded emergency safety fallback."""
    return bool(np.isfinite(decision.slack_value))


def _uses_emergency(decision: SafetyFilterDecision) -> bool:
    return decision.emergency_fallback_used and decision.solver_status == "forced_infeasible_qp"


def _preserves_low_risk(decision: SafetyFilterDecision) -> bool:
    return not decision.emergency_fallback_used and decision.intervention_norm < 1e-4


def _scenario_names() -> tuple[str, ...]:
    return (
        "head_on",
        "crossing",
        "three_way_convergence",
        "near_terrain",
        "near_cylindrical_threat",
        "initially_unsafe",
        "multiple_conflicts",
        "forced_infeasible_qp",
        "low_risk_preservation",
    )


if __name__ == "__main__":
    main()

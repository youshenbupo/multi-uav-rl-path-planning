"""Replay one recorded CBF emergency event with explicit numerical solver settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiuav.envs.communication import AgentKnowledgeState  # noqa: E402
from multiuav.envs.dynamic_world import DynamicWorldState  # noqa: E402
from multiuav.envs.observations import EnvironmentSnapshot  # noqa: E402
from multiuav.learning.graph_runner import make_graph_scenario  # noqa: E402
from multiuav.safety import CBFConfig, OSQPSafetyFilter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Expose every numerical setting varied by a recorded-QP replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--telemetry-key", default="cbf")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--max-solve-time-seconds", type=float, default=0.1)
    parser.add_argument("--slack-penalty", type=float, default=1_000.0)
    parser.add_argument("--uncertainty-margin-gain", type=float, default=0.5)
    parser.add_argument("--max-uncertainty-margin", type=float, default=5.0)
    return parser


def build_snapshot_and_requested(event: dict[str, Any]) -> tuple[EnvironmentSnapshot, np.ndarray]:
    """Rebuild the CBF-relevant dynamic snapshot saved for one emergency event."""
    context = event.get("context")
    if not isinstance(context, dict):
        raise ValueError("Emergency event has no replay context.")
    positions = np.asarray(context["positions"], dtype=float)
    velocities = np.asarray(context["velocities"], dtype=float)
    requested = np.asarray(context["requested_velocities"], dtype=float)
    valid = np.asarray(context["knowledge_valid"], dtype=bool)
    uncertainty = np.asarray(context["knowledge_uncertainty"], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or requested.shape != positions.shape:
        raise ValueError("Recorded positions and requested velocities must both have shape [N, 3].")
    if velocities.shape != positions.shape or valid.shape != positions.shape[:1] * 2:
        raise ValueError("Recorded CBF context has inconsistent state dimensions.")
    if uncertainty.shape != valid.shape:
        raise ValueError(
            "Recorded communication uncertainty shape does not match valid-mask shape."
        )
    scenario = make_graph_scenario(num_uavs=len(positions), dynamic_obstacle=True)
    dynamic_world = DynamicWorldState(
        scenario.dynamic_obstacles, dt=1.0, step_count=int(context["environment_step"])
    )
    recorded_centers = np.asarray(context["dynamic_obstacle_centers"], dtype=float)
    if not np.allclose(dynamic_world.centers, recorded_centers):
        raise ValueError(
            "Recorded dynamic-obstacle centers cannot be reconstructed from the event step."
        )
    knowledge = tuple(
        AgentKnowledgeState(
            positions=np.zeros_like(positions),
            velocities=np.zeros_like(positions),
            valid=valid[receiver],
            ages=np.zeros(len(positions), dtype=int),
            predicted_positions=np.zeros_like(positions),
            position_uncertainty=uncertainty[receiver],
        )
        for receiver in range(len(positions))
    )
    return (
        EnvironmentSnapshot(
            scenario=scenario,
            positions=positions,
            velocities=velocities,
            active_mask=np.ones(len(positions), dtype=bool),
            previous_goal_distances=np.ones(len(positions), dtype=float),
            step_count=int(context["environment_step"]),
            max_steps=20,
            max_horizontal_speed=8.0,
            max_vertical_speed=6.0,
            boundary_clipped=np.zeros(len(positions), dtype=bool),
            dynamic_world=dynamic_world,
            knowledge_states=knowledge,
        ),
        requested,
    )


def main() -> None:
    """Load one training artifact event and print a direct-OSQP replay result."""
    arguments = build_parser().parse_args()
    telemetry = json.loads(arguments.telemetry.read_text(encoding="utf-8"))
    selected_telemetry = telemetry.get(arguments.telemetry_key)
    if not isinstance(selected_telemetry, dict):
        raise ValueError("telemetry-key does not select a retained CBF telemetry mapping.")
    events = selected_telemetry.get("emergency_events")
    if not isinstance(events, list):
        raise ValueError("Selected telemetry has no emergency-event list.")
    if not 0 <= arguments.event_index < len(events):
        raise ValueError("event-index is outside the retained emergency-event list.")
    snapshot, requested = build_snapshot_and_requested(events[arguments.event_index])
    decision = OSQPSafetyFilter(
        CBFConfig(
            max_solve_time_seconds=arguments.max_solve_time_seconds,
            max_iterations=arguments.max_iterations,
            slack_penalty=arguments.slack_penalty,
            communication_uncertainty_margin_gain=arguments.uncertainty_margin_gain,
            max_communication_uncertainty_margin=arguments.max_uncertainty_margin,
        )
    ).filter(snapshot, requested)
    print(
        json.dumps(
            {
                "solver_status": decision.solver_status,
                "solver_iterations": decision.solver_iterations,
                "primal_residual": decision.primal_residual,
                "dual_residual": decision.dual_residual,
                "solve_time_seconds": decision.solve_time,
                "emergency_fallback_used": decision.emergency_fallback_used,
                "intervention_norm": decision.intervention_norm,
                "slack_value": decision.slack_value,
                "active_constraint_count": decision.active_constraint_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

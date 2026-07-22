"""Replay retained CBF emergency events with an explicit unchanged solver protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiuav.safety import CBFConfig, OSQPSafetyFilter  # noqa: E402
from scripts.diagnose_cbf_failure import build_snapshot_and_requested  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Require explicit source telemetry and a never-overwritten JSONL destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "telemetry",
        type=Path,
        nargs="+",
        help="Retained telemetry JSON or JSONL files.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--max-solve-time-seconds", type=float, default=0.1)
    parser.add_argument("--slack-penalty", type=float, default=100.0)
    parser.add_argument("--uncertainty-margin-gain", type=float, default=0.5)
    parser.add_argument("--max-uncertainty-margin", type=float, default=5.0)
    return parser


def _event_mappings(value: Any, pointer: str = "$") -> list[tuple[str, dict[str, Any]]]:
    """Find every telemetry mapping that retains an emergency-event list."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        events = value.get("emergency_events")
        if isinstance(events, list):
            for index, event in enumerate(events):
                if isinstance(event, dict):
                    found.append((f"{pointer}.emergency_events[{index}]", event))
        for key, nested in value.items():
            if key != "emergency_events":
                found.extend(_event_mappings(nested, f"{pointer}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_event_mappings(nested, f"{pointer}[{index}]"))
    return found


def _payloads_from_telemetry(path: Path) -> list[tuple[str, Any]]:
    """Load one JSON document or every nonblank JSONL record with stable pointers."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".jsonl":
        return [("$", json.loads(text))]
    return [
        (f"$[{line_index}]", json.loads(line))
        for line_index, line in enumerate(text.splitlines())
        if line.strip()
    ]


def _replay(event: dict[str, Any], arguments: argparse.Namespace) -> dict[str, object]:
    """Directly replay one saved request using exactly the reported mainline values."""
    snapshot, requested = build_snapshot_and_requested(event)
    decision = OSQPSafetyFilter(
        CBFConfig(
            max_solve_time_seconds=arguments.max_solve_time_seconds,
            max_iterations=arguments.max_iterations,
            slack_penalty=arguments.slack_penalty,
            communication_uncertainty_margin_gain=arguments.uncertainty_margin_gain,
            max_communication_uncertainty_margin=arguments.max_uncertainty_margin,
        )
    ).filter(snapshot, requested)
    return {
        "solver_status": decision.solver_status,
        "solver_iterations": decision.solver_iterations,
        "primal_residual": decision.primal_residual,
        "dual_residual": decision.dual_residual,
        "solve_time_seconds": decision.solve_time,
        "emergency_fallback_used": decision.emergency_fallback_used,
        "intervention_norm": decision.intervention_norm,
        "slack_value": decision.slack_value,
        "active_constraint_count": decision.active_constraint_count,
    }


def main() -> None:
    """Persist one JSONL replay record for every retained event and a compact summary."""
    arguments = build_parser().parse_args()
    output_path = arguments.output_jsonl
    summary_path = output_path.with_suffix(".summary.json")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("Replay output already exists; choose a unique attempt identity.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with output_path.open("x", encoding="utf-8") as handle:
        for telemetry_path in arguments.telemetry:
            for base_pointer, payload in _payloads_from_telemetry(telemetry_path):
                for pointer, event in _event_mappings(payload, base_pointer):
                    record: dict[str, object] = {
                        "telemetry": str(telemetry_path),
                        "event_pointer": pointer,
                        "recorded_solver_status": event.get("solver_status"),
                        "recorded_solver_iterations": event.get("solver_iterations"),
                    }
                    try:
                        record["replay"] = _replay(event, arguments)
                    except (KeyError, TypeError, ValueError) as error:
                        record["replay_error"] = f"{type(error).__name__}: {error}"
                    records.append(record)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "telemetry_files": [str(path) for path in arguments.telemetry],
        "solver_protocol": {
            "max_iterations": arguments.max_iterations,
            "max_solve_time_seconds": arguments.max_solve_time_seconds,
            "slack_penalty": arguments.slack_penalty,
            "uncertainty_margin_gain": arguments.uncertainty_margin_gain,
            "max_uncertainty_margin": arguments.max_uncertainty_margin,
        },
        "event_count": len(records),
        "replay_error_count": sum("replay_error" in record for record in records),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"jsonl": str(output_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

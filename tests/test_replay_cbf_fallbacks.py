"""Regression tests for retained CBF event discovery."""

from scripts.replay_cbf_fallbacks import _event_mappings


def test_event_mappings_does_not_replay_last_interval_compatibility_alias_twice() -> None:
    """The append-only interval record is canonical when the legacy alias matches it."""
    event = {"solver_status": "solve_time_limit", "decision_index": 61}
    interval_cbf = {"emergency_events": [event]}
    telemetry = {
        "interval_evaluations": [
            {"total_transitions": 98_304, "cbf": interval_cbf}
        ],
        "last_interval_evaluation_cbf": interval_cbf,
    }

    found = _event_mappings(telemetry)

    assert found == [
        ("$.interval_evaluations[0].cbf.emergency_events[0]", event)
    ]

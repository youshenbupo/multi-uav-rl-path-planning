"""Regression coverage for replaying persisted CBF failure context."""

from __future__ import annotations

import numpy as np

from scripts.diagnose_cbf_failure import build_snapshot_and_requested


def test_diagnostic_reconstructs_dynamic_cbf_snapshot_from_event_context() -> None:
    """A retained event must recreate its moving-obstacle and uncertainty inputs exactly."""
    event = {
        "context": {
            "environment_step": 1,
            "positions": [[10.0, 15.0, 30.0], [10.0, 85.0, 30.0]],
            "velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "requested_velocities": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "knowledge_valid": [[True, True], [True, True]],
            "knowledge_uncertainty": [[0.0, 1.0], [2.0, 0.0]],
            "dynamic_obstacle_centers": [[22.0, 52.0, 30.0]],
        }
    }

    snapshot, requested = build_snapshot_and_requested(event)

    np.testing.assert_allclose(snapshot.dynamic_world.centers, [[22.0, 52.0, 30.0]])
    np.testing.assert_allclose(requested, event["context"]["requested_velocities"])
    assert snapshot.knowledge_states[1].position_uncertainty[0] == 2.0

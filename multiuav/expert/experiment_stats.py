"""Stable aggregation for resumable HGALO reconstruction experiment records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ALGORITHM = "HGALO_PYTHON_COORDINATED_RECONSTRUCTION"
PROVENANCE = "Available MATLAB HGALO operators; CA-HGALO MATLAB entrypoint unavailable."


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSON Lines records produced by the resumable batch script."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_records(records: Sequence[Mapping[str, Any]], *, scenario_id: str) -> dict[str, Any]:
    """Use the most recent result per seed and calculate reproducible aggregate metrics."""
    latest_by_seed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        latest_by_seed[int(record["seed"])] = record
    selected = [latest_by_seed[seed] for seed in sorted(latest_by_seed)]
    if not selected:
        raise ValueError("At least one experiment record is required.")

    costs = np.asarray([record["best_cost"] for record in selected], dtype=float)
    separations = np.asarray([record["minimum_separation"] for record in selected], dtype=float)
    conflicts = np.asarray([record["temporal_conflict_count"] for record in selected], dtype=float)
    runtimes = np.asarray([record["runtime_seconds"] for record in selected], dtype=float)
    histories = [np.asarray(record["convergence_history"], dtype=float) for record in selected]
    if any(history.ndim != 1 or not np.isfinite(history).all() for history in histories):
        raise ValueError("Each convergence history must be a finite one-dimensional sequence.")
    if len({len(history) for history in histories}) != 1:
        raise ValueError("Convergence histories must have a common length.")
    convergence = np.vstack(histories)
    return {
        "algorithm": ALGORITHM,
        "provenance": PROVENANCE,
        "scenario_id": scenario_id,
        "seed_count": len(selected),
        "best_cost": float(costs.min()),
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std()),
        "success_rate": float(np.mean([record["strict_success"] for record in selected])),
        "mean_minimum_separation": float(separations.mean()),
        "mean_temporal_conflict_count": float(conflicts.mean()),
        "mean_runtime_seconds": float(runtimes.mean()),
        "mean_convergence_history": convergence.mean(axis=0).tolist(),
        "std_convergence_history": convergence.std(axis=0).tolist(),
        "records": [dict(record) for record in selected],
    }

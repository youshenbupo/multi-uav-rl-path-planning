"""Create a descriptive seed-level paired summary from checkpoint JSONL files.

The script deliberately treats a seed-scenario cell (its 20 evaluation episodes)
as one observation.  It reports no p-values, confidence intervals, or method
claims; those require a preregistered analysis decision beyond this data audit.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

EPISODES_PER_CELL = 20
METRIC_FIELDS = (
    "success",
    "collision",
    "terrain_violation",
    "threat_violation",
    "episode_return",
    "path_length",
    "mission_time",
    "minimum_separation",
    "temporal_conflict_count",
    "energy_proxy",
    "decision_latency",
    "cbf_emergency_count",
    "cbf_emergency_fallback_rate",
    "cbf_intervention_rate",
    "cbf_mean_correction",
    "cbf_mean_solve_time_seconds",
)


def _load_cells(root: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Load and validate one twenty-episode JSONL file per seed-scenario cell."""
    cells: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for jsonl_path in sorted(root.rglob("*.jsonl")):
        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != EPISODES_PER_CELL:
            raise ValueError(
                f"{jsonl_path} must contain {EPISODES_PER_CELL} nonblank records, "
                f"got {len(records)}"
            )
        first = records[0]
        if not isinstance(first.get("seed"), int) or not isinstance(first.get("scenario"), str):
            raise ValueError(f"{jsonl_path} lacks integer seed or string scenario")
        key = (first["seed"], first["scenario"])
        if key in cells:
            raise ValueError(f"Duplicate seed-scenario cell {key} under {root}")
        expected_episodes = list(range(EPISODES_PER_CELL))
        episodes = sorted(record.get("episode") for record in records)
        if episodes != expected_episodes:
            raise ValueError(f"{jsonl_path} must contain episode indices {expected_episodes}")
        if any(
            record.get("seed") != key[0] or record.get("scenario") != key[1]
            for record in records
        ):
            raise ValueError(f"{jsonl_path} mixes seed or scenario identities")
        cells[key] = records
    return cells


def _cell_mean(records: list[dict[str, Any]], metric: str) -> float:
    values: list[float] = []
    for record in records:
        value = record.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Metric {metric!r} is not numeric in every record")
        values.append(float(value))
    return statistics.fmean(values)


def _sample_std(values: list[float]) -> float:
    """Return zero for a one-seed descriptive group instead of an undefined value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_paired_roots(reference_root: Path, treatment_root: Path) -> dict[str, Any]:
    """Summarize matched cells using trained seeds, never raw episodes, as units."""
    reference_cells = _load_cells(reference_root)
    treatment_cells = _load_cells(treatment_root)
    if set(reference_cells) != set(treatment_cells):
        raise ValueError("Reference and treatment cell identities differ")
    if not reference_cells:
        raise ValueError("No paired JSONL cells found")

    scenarios = sorted({scenario for _, scenario in reference_cells})
    scenario_summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        keys = sorted(key for key in reference_cells if key[1] == scenario)
        task_metrics: dict[str, dict[str, float]] = {}
        cbf_diagnostic_metrics: dict[str, dict[str, float]] = {}
        for metric in METRIC_FIELDS:
            try:
                reference_values = [_cell_mean(reference_cells[key], metric) for key in keys]
                treatment_values = [_cell_mean(treatment_cells[key], metric) for key in keys]
            except ValueError:
                continue
            deltas = [
                treatment - reference
                for reference, treatment in zip(reference_values, treatment_values)
            ]
            summary = {
                "reference_seed_mean": statistics.fmean(reference_values),
                "reference_seed_sample_std": _sample_std(reference_values),
                "treatment_seed_mean": statistics.fmean(treatment_values),
                "treatment_seed_sample_std": _sample_std(treatment_values),
                "paired_delta_treatment_minus_reference_mean": statistics.fmean(deltas),
                "paired_delta_sample_std": _sample_std(deltas),
            }
            if metric.startswith("cbf_"):
                cbf_diagnostic_metrics[metric] = summary
            else:
                task_metrics[metric] = summary
        if not task_metrics and not cbf_diagnostic_metrics:
            raise ValueError(f"No shared numeric metrics for scenario {scenario!r}")
        scenario_summaries.append(
            {
                "scenario": scenario,
                "seed_count": len(keys),
                "episodes_per_seed": EPISODES_PER_CELL,
                "seed_ids": [key[0] for key in keys],
                "task_metrics": task_metrics,
                "cbf_diagnostic_metrics": cbf_diagnostic_metrics,
            }
        )

    return {
        "reference_root": str(reference_root),
        "treatment_root": str(treatment_root),
        "independent_unit": "trained_seed",
        "episode_records_are_not_independent_training_replicates": True,
        "scenario_summaries": scenario_summaries,
        "claim_boundary": (
            "descriptive paired seed-cell summary only; "
            "no significance or method-effect claim"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for a provenance-preserving paired summary artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--treatment-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    """Validate roots and serialize the descriptive paired summary."""
    args = build_parser().parse_args()
    summary = summarize_paired_roots(args.reference_root, args.treatment_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

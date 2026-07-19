"""Run one seeded HGALO reconstruction and persist expert artifacts and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Execute one explicit seed and save JSON diagnostics plus an HDF5 expert episode."""
    from multiuav.data.expert_dataset import build_expert_episode
    from multiuav.data.expert_schema import write_episode
    from multiuav.data.matlab_import import ImportedExpertCase
    from multiuav.expert.ca_hgalo import HGALOPlanner
    from multiuav.expert.scenarios import load_hgalo_config, load_scenario

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=PROJECT_ROOT / "configs/scenarios/s1.yaml")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/expert/ca_hgalo.yaml"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "data/regression/matlab/reference_cases.mat",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data/expert/ca_hgalo_run.json",
    )
    parser.add_argument(
        "--expert-output",
        type=Path,
        default=PROJECT_ROOT / "data/expert/ca_hgalo_experts.h5",
    )
    args = parser.parse_args()
    scenario_id, scenario, num_waypoints = load_scenario(args.scenario, args.reference)
    config = load_hgalo_config(args.config, args.seed)
    started = time.perf_counter()
    result = HGALOPlanner(scenario, num_waypoints, config).run()
    runtime_seconds = time.perf_counter() - started
    imported = ImportedExpertCase(
        scenario_id=scenario_id,
        seed=result.seed,
        scenario=scenario,
        raw_trajectories=result.raw_best_trajectories,
        repaired_trajectories=result.best_trajectories,
        start_delays=result.coordination.start_delays,
        nominal_speed=scenario.cruise_speed,
        source=result.provenance,
        schedule_log=result.coordination.schedule_log,
        spatial_repair_log=result.coordination.repair_log,
    )
    episode = build_expert_episode(
        imported,
        dt=args.dt,
        schedule_log=result.coordination.schedule_log,
        spatial_repair_log=result.coordination.repair_log,
        convergence_history=result.convergence_history,
    )
    episode_id = f"{scenario_id}_seed_{result.seed}"
    write_episode(args.expert_output, episode_id, episode)
    report = _report_payload(result, episode_id, runtime_seconds)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Algorithm: {result.algorithm}")
    print(f"Provenance: {result.provenance}")
    print(f"Seed: {result.seed}")
    print(f"Final cost: {episode.evaluation.total_cost:.6f}")
    print(f"Strict success: {episode.evaluation.strict_success}")
    print(f"Runtime seconds: {runtime_seconds:.3f}")
    print(f"Report: {args.report}")
    print(f"Expert episode: {args.expert_output}#{episode_id}")


def _report_payload(result: Any, episode_id: str, runtime_seconds: float) -> dict[str, Any]:
    evaluation = result.coordination.evaluation_after
    graph = result.coordination.conflict_graph
    return {
        "algorithm": result.algorithm,
        "provenance": result.provenance,
        "seed": result.seed,
        "episode_id": episode_id,
        "runtime_seconds": runtime_seconds,
        "best_candidate": result.best_candidate.tolist(),
        "final_trajectories": [
            trajectory.points.tolist() for trajectory in result.best_trajectories
        ],
        "cost_breakdown": _json_ready(asdict(evaluation.cost_breakdown)),
        "strict_success": evaluation.strict_success,
        "minimum_separation": evaluation.minimum_separation,
        "temporal_conflict_count": evaluation.temporal_conflict_count,
        "conflict_graph": {
            "edge_index": graph.edge_index.tolist(),
            "edge_weights": graph.edge_weights.tolist(),
            "worst_conflict_pair": graph.worst_conflict_pair,
        },
        "schedule_log": _json_ready([asdict(entry) for entry in result.coordination.schedule_log]),
        "spatial_repair_log": _json_ready(
            [asdict(entry) for entry in result.coordination.repair_log]
        ),
        "convergence_history": result.convergence_history.tolist(),
        "generation_stats": _json_ready([asdict(stat) for stat in result.generation_stats]),
    }


def _json_ready(value: object) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    main()

"""Run seeded HGALO reconstruction experiments and write HDF5 expert demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run the requested number of explicit seeds and emit aggregate statistics."""
    from multiuav.data.expert_dataset import build_expert_episode
    from multiuav.data.expert_schema import write_episode
    from multiuav.data.matlab_import import ImportedExpertCase
    from multiuav.expert.ca_hgalo import HGALOPlanner
    from multiuav.expert.experiment_stats import summarize_records
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
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/expert/ca_hgalo_experts.h5"
    )
    parser.add_argument(
        "--summary", type=Path, default=PROJECT_ROOT / "data/expert/ca_hgalo_summary.json"
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=PROJECT_ROOT / "data/expert/ca_hgalo_records.jsonl",
        help="JSON Lines file flushed after every completed seed.",
    )
    args = parser.parse_args()
    if args.seed_count < 1:
        raise ValueError("seed-count must be positive.")
    scenario_id, scenario, num_waypoints = load_scenario(args.scenario, args.reference)
    records: list[dict[str, float | int | bool | list[float]]] = []
    args.records.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("a", encoding="utf-8") as record_file:
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            config = load_hgalo_config(args.config, seed)
            started = time.perf_counter()
            result = HGALOPlanner(scenario, num_waypoints, config).run()
            runtime = time.perf_counter() - started
            imported = ImportedExpertCase(
                scenario_id=scenario_id,
                seed=seed,
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
                convergence_history=result.convergence_history,
            )
            write_episode(args.output, f"{scenario_id}_seed_{seed}", episode)
            record = {
                "seed": seed,
                "best_cost": episode.evaluation.total_cost,
                "strict_success": episode.evaluation.strict_success,
                "minimum_separation": episode.evaluation.minimum_separation,
                "temporal_conflict_count": episode.evaluation.temporal_conflict_count,
                "runtime_seconds": runtime,
                "convergence_history": result.convergence_history.tolist(),
            }
            records.append(record)
            record_file.write(json.dumps(record) + "\n")
            record_file.flush()
            print(f"seed={seed} cost={episode.evaluation.total_cost:.6f} runtime={runtime:.3f}s")
    summary = summarize_records(records, scenario_id=scenario_id)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

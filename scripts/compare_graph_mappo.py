"""Run and summarize the required Phase-10 MAPPO/distance/predictive graph matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class GraphComparisonConfig:
    """Validated finite matrix for the required Phase-10 ablation comparison."""

    seed: int
    num_uavs: tuple[int, ...]
    graph_modes: tuple[str, ...]
    total_steps: int
    evaluation_episodes: int

    def __post_init__(self) -> None:
        if self.num_uavs != (3, 5, 8):
            raise ValueError("Phase-10 comparison must include exactly 3, 5, and 8 UAVs.")
        if self.graph_modes != ("mappo", "distance_graph", "predictive_graph"):
            raise ValueError("Comparison must include mappo, distance_graph, predictive_graph.")
        if min(self.total_steps, self.evaluation_episodes) < 1:
            raise ValueError("total_steps and evaluation_episodes must be positive.")


def load_comparison_config(path: Path) -> GraphComparisonConfig:
    """Load a strict graph-ablation experiment matrix from YAML."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Graph comparison configuration must be a YAML mapping.")
    payload: dict[str, Any] = dict(values)
    payload["num_uavs"] = tuple(payload["num_uavs"])
    payload["graph_modes"] = tuple(payload["graph_modes"])
    return GraphComparisonConfig(**payload)


def build_parser() -> argparse.ArgumentParser:
    """Build explicit comparator paths and optional safe runtime overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/experiments/graph_comparison.yaml"
    )
    parser.add_argument(
        "--base-config", type=Path, default=PROJECT_ROOT / "configs/rl/graph_mappo.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/rl/graph_comparison"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--total-steps", type=int)
    return parser


def main() -> None:
    """Execute each required cell and serialize raw and grouped actual metrics."""
    from multiuav.learning.graph_mappo import save_graph_checkpoint
    from multiuav.learning.graph_runner import GraphMAPPOExperiment, load_graph_experiment_config

    args = build_parser().parse_args()
    matrix = load_comparison_config(args.config)
    if args.total_steps is not None and args.total_steps < 1:
        raise ValueError("--total-steps must be positive.")
    base = load_graph_experiment_config(args.base_config)
    total_steps = args.total_steps if args.total_steps is not None else matrix.total_steps
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for count_index, num_uavs in enumerate(matrix.num_uavs):
        for graph_mode in matrix.graph_modes:
            seed = matrix.seed + 100 * count_index
            config = replace(
                base,
                seed=seed,
                num_uavs=num_uavs,
                graph_mode=graph_mode,
                total_steps=total_steps,
            )
            output = args.output_dir / f"{graph_mode}_{num_uavs}uav"
            experiment = GraphMAPPOExperiment(config, device=device, log_dir=output / "tensorboard")
            initial = experiment.evaluate(episodes=matrix.evaluation_episodes)
            updates = experiment.train(checkpoint_dir=output / "checkpoints")
            final = experiment.evaluate(episodes=matrix.evaluation_episodes)
            checkpoint = output / "checkpoints" / "graph_mappo_final.pt"
            save_graph_checkpoint(checkpoint, experiment.trainer, step=experiment.total_transitions)
            experiment.close()
            records.append(
                {
                    "seed": seed,
                    "num_uavs": num_uavs,
                    "graph_mode": graph_mode,
                    "total_steps": experiment.total_transitions,
                    "initial_evaluation": initial,
                    "final_evaluation": final,
                    "last_update": updates[-1] if updates else {},
                    "checkpoint": str(checkpoint),
                }
            )
    aggregate = _aggregate(records)
    (args.output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"records": records, "aggregate": aggregate}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"records": records, "aggregate": aggregate}, indent=2))


def _aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Produce one transparent metric row per comparison cell without significance claims."""
    summary: list[dict[str, object]] = []
    for record in records:
        final = record["final_evaluation"]
        if not isinstance(final, dict):
            raise ValueError("Comparison record final_evaluation must be a mapping.")
        summary.append(
            {
                "num_uavs": record["num_uavs"],
                "graph_mode": record["graph_mode"],
                "episode_return": final["episode_return"],
                "success_rate": final["success_rate"],
                "collision_rate": final["collision_rate"],
                "minimum_separation": final["minimum_separation"],
            }
        )
    return summary


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()

"""Load and deterministically evaluate a Phase-10 graph MAPPO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build checkpoint evaluation arguments with explicit graph overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/rl/graph_mappo.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-uavs", type=int, choices=(3, 5, 8))
    parser.add_argument("--graph-mode", choices=("mappo", "distance_graph", "predictive_graph"))
    parser.add_argument("--with-cylinder", action="store_true")
    return parser


def main() -> None:
    """Restore an exact graph checkpoint and print machine-readable evaluation metrics."""
    from multiuav.learning.graph_mappo import load_graph_checkpoint
    from multiuav.learning.graph_runner import GraphMAPPOExperiment, load_graph_experiment_config

    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")
    config = load_graph_experiment_config(args.config)
    config = replace(config, obstacle=args.with_cylinder or config.obstacle)
    if args.num_uavs is not None:
        config = replace(config, num_uavs=args.num_uavs)
    if args.graph_mode is not None:
        config = replace(config, graph_mode=args.graph_mode)
    device = _resolve_device(args.device)
    experiment = GraphMAPPOExperiment(config, device=device)
    step = load_graph_checkpoint(args.checkpoint, experiment.trainer, map_location=device)
    metrics = experiment.evaluate(episodes=args.episodes)
    experiment.close()
    print(json.dumps({"checkpoint_step": step, "metrics": metrics}, indent=2))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()

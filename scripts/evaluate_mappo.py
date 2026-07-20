"""Evaluate a saved Phase-9 MAPPO checkpoint using deterministic Gaussian means."""

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
    """Build the checkpoint-evaluation command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/rl/mappo_baseline.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "data/rl/mappo_baseline/checkpoints/mappo_final.pt",
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-uavs", type=int, choices=(3, 5, 8))
    parser.add_argument("--with-cylinder", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/rl/mappo_baseline/evaluation.json"
    )
    return parser


def main() -> None:
    """Restore a checkpoint and write deterministic evaluation metrics to JSON."""
    from multiuav.learning.mappo import load_checkpoint
    from multiuav.learning.runner import MAPPOExperiment, load_mappo_experiment_config

    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive.")
    config = load_mappo_experiment_config(args.config)
    if args.with_cylinder:
        config = replace(config, obstacle=True)
    if args.num_uavs is not None:
        config = replace(config, num_uavs=args.num_uavs)
    device = _resolve_device(args.device)
    experiment = MAPPOExperiment(config, device=device)
    step = load_checkpoint(args.checkpoint, experiment.trainer, map_location=device)
    metrics = experiment.evaluate(episodes=args.episodes)
    payload = {"checkpoint_step": step, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    experiment.close()
    print(json.dumps(payload, indent=2))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()

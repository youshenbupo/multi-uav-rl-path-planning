"""Train the Phase-9 basic MAPPO baseline on the empty two-UAV curriculum first."""

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
    """Build a parser with explicit portable paths and no hidden training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/rl/mappo_baseline.yaml"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/rl/mappo_baseline")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-uavs", type=int, choices=(3, 5, 8))
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--with-cylinder", action="store_true")
    return parser


def main() -> None:
    """Run training, write TensorBoard/checkpoints, and save an auditable JSON summary."""
    from multiuav.learning.mappo import save_checkpoint
    from multiuav.learning.runner import MAPPOExperiment, load_mappo_experiment_config

    args = build_parser().parse_args()
    config = load_mappo_experiment_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.with_cylinder:
        config = replace(config, obstacle=True)
    if args.num_uavs is not None:
        config = replace(config, num_uavs=args.num_uavs)
    if args.total_steps is not None:
        if args.total_steps < 1:
            raise ValueError("--total-steps must be positive.")
        config = replace(config, total_steps=args.total_steps)
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = MAPPOExperiment(config, device=device, log_dir=args.output_dir / "tensorboard")
    initial = experiment.evaluate(episodes=8)
    initial_cbf_telemetry = experiment.last_evaluation_cbf_telemetry.as_dict()
    records = experiment.train(checkpoint_dir=args.output_dir / "checkpoints")
    final = experiment.evaluate(episodes=8)
    final_cbf_telemetry = experiment.last_evaluation_cbf_telemetry.as_dict()
    checkpoint = args.output_dir / "checkpoints" / "mappo_final.pt"
    save_checkpoint(checkpoint, experiment.trainer, step=experiment.total_transitions)
    summary = {
        "config": str(args.config),
        "device": str(device),
        "num_uavs": config.num_uavs,
        "dynamic_obstacle_enabled": config.dynamic_obstacle_enabled,
        "initial_evaluation": initial,
        "initial_evaluation_cbf": initial_cbf_telemetry,
        "final_evaluation": final,
        "final_evaluation_cbf": final_cbf_telemetry,
        "update_count": len(records),
        "last_update": records[-1] if records else {},
        "checkpoint": str(checkpoint),
        "cbf": experiment.cbf_telemetry.as_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    experiment.close()
    print(json.dumps(summary, indent=2))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()

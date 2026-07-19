"""Train one explicit Stage-A, Stage-B, or enabled joint hierarchical MAPPO phase."""

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
    """Build portable staged-training arguments with no hidden stage selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/rl/hierarchical_mappo.yaml"
    )
    parser.add_argument("--stage", required=True, choices=("low", "high", "joint"))
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/rl/hierarchical_mappo"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--load-checkpoint", type=Path)
    return parser


def main() -> None:
    """Load optional predecessor weights, train the requested stage, and checkpoint it."""
    from multiuav.learning.hierarchical_mappo import (
        load_hierarchical_checkpoint,
        save_hierarchical_checkpoint,
    )
    from multiuav.learning.hierarchical_runner import (
        HierarchicalMAPPOExperiment,
        load_hierarchical_experiment_config,
    )

    args = build_parser().parse_args()
    config = load_hierarchical_experiment_config(args.config)
    if args.total_steps is not None:
        if args.total_steps < 1:
            raise ValueError("--total-steps must be positive.")
        config = replace(config, total_steps=args.total_steps)
    device = _resolve_device(args.device)
    experiment = HierarchicalMAPPOExperiment(config, device=device)
    loaded: tuple[str, int] | None = None
    if args.load_checkpoint is not None:
        loaded = load_hierarchical_checkpoint(
            args.load_checkpoint, experiment.trainer, map_location=device
        )
    records = experiment.train(stage=args.stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"hierarchical_{args.stage}_final.pt"
    save_hierarchical_checkpoint(
        checkpoint, experiment.trainer, stage=args.stage, step=experiment.total_transitions
    )
    summary = {
        "stage": args.stage,
        "loaded_checkpoint": loaded,
        "total_transitions": experiment.total_transitions,
        "update_count": len(records),
        "last_update": records[-1] if records else {},
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / f"{args.stage}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()

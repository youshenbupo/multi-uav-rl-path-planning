"""Restore a hierarchical checkpoint and collect finite rollout data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build explicit checkpoint and stage arguments for hierarchical evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/rl/hierarchical_mappo.yaml"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=("low", "high", "joint"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def main() -> None:
    """Restore hierarchy state and report finite one-rollout diagnostics for the chosen stage."""
    from multiuav.learning.hierarchical_mappo import load_hierarchical_checkpoint
    from multiuav.learning.hierarchical_runner import (
        HierarchicalMAPPOExperiment,
        load_hierarchical_experiment_config,
    )

    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    experiment = HierarchicalMAPPOExperiment(
        load_hierarchical_experiment_config(args.config), device=device
    )
    loaded_stage, loaded_step = load_hierarchical_checkpoint(
        args.checkpoint, experiment.trainer, map_location=device
    )
    low_batch, high_batch, metrics = experiment.collect_rollout(stage=args.stage)
    print(
        json.dumps(
            {
                "checkpoint_stage": loaded_stage,
                "checkpoint_step": loaded_step,
                "low_samples": int(low_batch.actions.shape[0]),
                "high_samples": int(high_batch.actions.shape[0]),
                "metrics": metrics,
            },
            indent=2,
        )
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()

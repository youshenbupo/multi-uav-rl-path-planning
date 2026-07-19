"""Train the Phase-10 graph MAPPO ablations on a deterministic multi-UAV curriculum."""

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
    """Build portable training arguments without hidden experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/rl/graph_mappo.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/rl/graph_mappo")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-uavs", type=int, choices=(3, 5, 8))
    parser.add_argument("--graph-mode", choices=("mappo", "distance_graph", "predictive_graph"))
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--with-cylinder", action="store_true")
    parser.add_argument("--bc-low-checkpoint", type=Path)
    parser.add_argument("--freeze-encoder-updates", type=int, default=0)
    parser.add_argument("--fine-tune-learning-rate", type=float)
    parser.add_argument("--imitation-coef", type=float, default=0.0)
    return parser


def main() -> None:
    """Train, checkpoint, evaluate, and serialize an auditable graph-run summary."""
    from multiuav.learning.bc_finetuning import BCFineTuneSchedule
    from multiuav.learning.graph_mappo import save_graph_checkpoint
    from multiuav.learning.graph_runner import GraphMAPPOExperiment, load_graph_experiment_config

    args = build_parser().parse_args()
    config = load_graph_experiment_config(args.config)
    config = replace(config, obstacle=args.with_cylinder or config.obstacle)
    if args.num_uavs is not None:
        config = replace(config, num_uavs=args.num_uavs)
    if args.graph_mode is not None:
        config = replace(config, graph_mode=args.graph_mode)
    if args.total_steps is not None:
        if args.total_steps < 1:
            raise ValueError("--total-steps must be positive.")
        config = replace(config, total_steps=args.total_steps)
    device = _resolve_device(args.device)
    if args.bc_low_checkpoint is None and (
        args.freeze_encoder_updates != 0
        or args.fine_tune_learning_rate is not None
        or args.imitation_coef != 0.0
    ):
        raise ValueError("BC fine-tuning flags require --bc-low-checkpoint.")
    schedule = (
        BCFineTuneSchedule(
            freeze_encoder_updates=args.freeze_encoder_updates,
            ppo_learning_rate=(
                args.fine_tune_learning_rate
                if args.fine_tune_learning_rate is not None
                else config.learning_rate
            ),
            imitation_coef=args.imitation_coef,
        )
        if args.bc_low_checkpoint is not None
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = GraphMAPPOExperiment(
        config,
        device=device,
        log_dir=args.output_dir / "tensorboard",
        bc_low_checkpoint=args.bc_low_checkpoint,
        bc_schedule=schedule,
    )
    initial = experiment.evaluate(episodes=8)
    records = experiment.train(checkpoint_dir=args.output_dir / "checkpoints")
    final = experiment.evaluate(episodes=8)
    checkpoint = args.output_dir / "checkpoints" / "graph_mappo_final.pt"
    save_graph_checkpoint(checkpoint, experiment.trainer, step=experiment.total_transitions)
    summary = {
        "config": str(args.config),
        "graph_mode": config.graph_mode,
        "num_uavs": config.num_uavs,
        "device": str(device),
        "initial_evaluation": initial,
        "final_evaluation": final,
        "update_count": len(records),
        "last_update": records[-1] if records else {},
        "checkpoint": str(checkpoint),
        "bc_low_checkpoint": str(args.bc_low_checkpoint) if args.bc_low_checkpoint else None,
        "bc_fine_tune": {
            "freeze_encoder_updates": schedule.freeze_encoder_updates,
            "ppo_learning_rate": schedule.ppo_learning_rate,
            "imitation_coef": schedule.imitation_coef,
        }
        if schedule is not None
        else None,
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

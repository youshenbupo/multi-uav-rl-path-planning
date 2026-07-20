"""Train the Phase-14 single mainline experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    """Parse the common experiment contract before staged training dispatch."""
    from multiuav.experiments.cli import build_mainline_parser, spec_from_args
    from multiuav.experiments.runner import create_experiment_output, resolve_device
    from multiuav.learning.bc_finetuning import BCFineTuneSchedule
    from multiuav.learning.hierarchical_mappo import (
        load_hierarchical_checkpoint,
        save_hierarchical_checkpoint,
    )
    from multiuav.learning.hierarchical_runner import (
        HierarchicalMAPPOExperiment,
        load_hierarchical_experiment_config,
    )

    parser = build_mainline_parser("training")
    parser.add_argument("--stage", choices=("low", "high", "joint"), default="low")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--bc-low-checkpoint", type=Path)
    parser.add_argument("--bc-high-checkpoint", type=Path)
    parser.add_argument("--bc-freeze-encoder-updates", type=int, default=2)
    parser.add_argument("--bc-fine-tune-learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    spec = spec_from_args(args)
    output = create_experiment_output(
        spec, args.output_root, controller="hierarchical_training"
    )
    configuration = load_hierarchical_experiment_config(args.config)
    configuration = replace(
        configuration,
        seed=spec.seeds[0],
        num_uavs=spec.num_uavs,
        total_steps=args.total_steps if args.total_steps is not None else configuration.total_steps,
        communication_enabled=True,
        dynamic_obstacle_enabled=True,
        cbf_enabled=spec.use_cbf,
    )
    experiment = HierarchicalMAPPOExperiment(configuration, device=resolve_device(spec.device))
    if spec.use_expert_pretrain:
        if args.bc_low_checkpoint is None or args.bc_high_checkpoint is None:
            parser.error(
                "--use-expert-pretrain requires --bc-low-checkpoint and --bc-high-checkpoint."
            )
        experiment.trainer.initialize_from_bc(
            low_checkpoint=args.bc_low_checkpoint,
            high_checkpoint=args.bc_high_checkpoint,
            schedule=BCFineTuneSchedule(
                freeze_encoder_updates=args.bc_freeze_encoder_updates,
                ppo_learning_rate=args.bc_fine_tune_learning_rate,
            ),
        )
    if spec.checkpoint is not None:
        load_hierarchical_checkpoint(
            spec.checkpoint, experiment.trainer, map_location=experiment.device
        )
    records = experiment.train(stage=args.stage)
    checkpoint = output.root / "checkpoints" / f"hierarchical_{args.stage}_final.pt"
    save_hierarchical_checkpoint(
        checkpoint, experiment.trainer, stage=args.stage, step=experiment.total_transitions
    )
    output.write_runtime_telemetry(
        {
            "command": sys.argv[1:],
            "configuration": {
                "filename": args.config.name,
                "sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            },
            "git_revision": _git_revision(),
            "training": {
                "stage": args.stage,
                "total_transitions": experiment.total_transitions,
                "update_count": len(records),
            },
            "cbf_configuration": {
                "enabled": configuration.cbf_enabled,
                "slack_penalty": configuration.cbf_slack_penalty,
                "max_iterations": configuration.cbf_max_iterations,
                "uncertainty_margin_gain": (
                    configuration.cbf_communication_uncertainty_margin_gain
                ),
                "max_uncertainty_margin": configuration.cbf_max_communication_uncertainty_margin,
            },
            "cbf": experiment.cbf_telemetry.as_dict(),
        }
    )
    print(
        json.dumps(
            {"output": str(output.root), "checkpoint": str(checkpoint), "updates": len(records)}
        )
    )


def _git_revision() -> str:
    """Return the source revision that produced a training artifact when available."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


if __name__ == "__main__":
    main()

"""Evaluate one independently trained core-protocol checkpoint into retained JSONL results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Expose every evaluation-relevant input; no controller fallback is available."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("mappo", "graph_mappo"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-uavs", type=int, choices=(3, 5, 8), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--scenario",
        choices=(
            "nominal",
            "delay_only",
            "loss_only",
            "dynamic_only",
            "combined",
            "ood_communication_obstacle",
        ),
        default="combined",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--without-cbf", action="store_true")
    return parser


def main() -> None:
    """Load exactly one trained policy and evaluate it on the requested seed."""
    from multiuav.experiments.core_evaluation import (
        create_checkpoint_evaluation_output,
        evaluate_graph_checkpoint,
        evaluate_mappo_checkpoint,
    )
    from multiuav.experiments.spec import ExperimentSpec

    args = build_parser().parse_args()
    spec = ExperimentSpec(
        name=args.experiment_name,
        seeds=(args.seed,),
        num_uavs=args.num_uavs,
        device=args.device,
        checkpoint=args.checkpoint,
        method=args.family,
        scenario=args.scenario,
        use_graph=args.family == "graph_mappo",
        use_cbf=not args.without_cbf,
    )
    controller = "mappo_checkpoint" if args.family == "mappo" else "graph_mappo_checkpoint"
    output = create_checkpoint_evaluation_output(
        args.output_dir, spec, controller=controller, config_path=args.config
    )
    evaluator = evaluate_mappo_checkpoint if args.family == "mappo" else evaluate_graph_checkpoint
    results = evaluator(
        spec,
        output,
        config_path=args.config,
        episodes_per_seed=args.episodes,
        max_steps=args.max_steps,
    )
    print(output.root)
    print(results[0].metrics)


if __name__ == "__main__":
    main()

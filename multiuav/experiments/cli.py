"""Shared command-line contract for the Phase-14 single experiment mainline."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiuav.experiments.spec import ExperimentSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rl" / "hierarchical_mappo.yaml"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


def build_mainline_parser(command: str) -> argparse.ArgumentParser:
    """Create the invariant CLI shared by training and all evaluation modes."""
    parser = argparse.ArgumentParser(
        description=f"Phase-14 {command}: BC-initialized hierarchical graph MAPPO with CBF."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-uavs", type=int, default=3)
    parser.add_argument("--scenario", default="within_distribution")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--use-expert-pretrain", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--use-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-hierarchy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cbf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--experiment-name", default=f"phase14_{command}")
    return parser


def spec_from_args(args: argparse.Namespace, *, method: str = "full_method") -> ExperimentSpec:
    """Convert only explicit CLI values into the shared serializable run specification."""
    seeds = tuple(args.seed) if args.seed is not None else (20260719,)
    return ExperimentSpec(
        name=args.experiment_name,
        seeds=seeds,
        method=method,
        device=args.device,
        num_uavs=args.num_uavs,
        scenario=args.scenario,
        checkpoint=args.checkpoint,
        render=args.render,
        use_expert_pretrain=args.use_expert_pretrain,
        use_graph=args.use_graph,
        use_hierarchy=args.use_hierarchy,
        use_cbf=args.use_cbf,
    )

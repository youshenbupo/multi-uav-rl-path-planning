"""Benchmark registered Phase-14 methods on the shared experiment interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    """Parse the common experiment contract before registered baseline dispatch."""
    from multiuav.experiments.cli import build_mainline_parser, spec_from_args
    from multiuav.experiments.registry import MethodRegistry
    from multiuav.experiments.runner import (
        create_experiment_output,
        evaluate_hierarchical_checkpoint,
    )

    parser = build_mainline_parser("benchmark")
    parser.add_argument("--methods", nargs="+", default=("full_method",))
    parser.add_argument("--episodes-per-seed", type=int, default=4)
    args = parser.parse_args()
    unsupported = [
        method.name
        for method in (MethodRegistry().resolve(name) for name in args.methods)
        if method.availability != "available"
    ]
    if unsupported:
        parser.error(f"Requested unavailable external baselines: {', '.join(unsupported)}")
    if tuple(args.methods) != ("full_method",):
        parser.error("Unified benchmark execution currently supports only full_method.")
    spec = spec_from_args(args)
    if spec.checkpoint is None:
        parser.error(
            "full_method requires --checkpoint; no smoke-controller substitution is allowed."
        )
    output = create_experiment_output(spec, args.output_root)
    results = evaluate_hierarchical_checkpoint(
        spec,
        output,
        config_path=args.config,
        episodes_per_seed=args.episodes_per_seed,
    )
    print(json.dumps({"output": str(output.root), "seeds": [result.seed for result in results]}))


if __name__ == "__main__":
    main()

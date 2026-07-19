"""Run Phase-14 ablations on the same single mainline interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    """Parse the common experiment contract before ablation dispatch."""
    from multiuav.experiments.cli import build_mainline_parser, spec_from_args
    from multiuav.experiments.runner import create_experiment_output, evaluate_goal_controller

    parser = build_mainline_parser("ablation")
    parser.add_argument("--ablations", nargs="+", default=("none",))
    parser.add_argument("--episodes-per-seed", type=int, default=4)
    args = parser.parse_args()
    spec = spec_from_args(args)
    output = create_experiment_output(spec, args.output_root)
    results = evaluate_goal_controller(spec, output, episodes_per_seed=args.episodes_per_seed)
    print(
        json.dumps(
            {
                "output": str(output.root),
                "ablations": args.ablations,
                "seeds": [result.seed for result in results],
            }
        )
    )


if __name__ == "__main__":
    main()

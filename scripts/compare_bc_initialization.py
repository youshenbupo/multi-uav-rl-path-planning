"""Inspect or validate reproducible random/BC/MAPPO initialization comparison arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build an interface that resolves only YAML-defined comparison modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/experiments/bc_comparison.yaml"
    )
    parser.add_argument(
        "--mode",
        choices=("random", "low_bc_only", "high_bc_only", "full_bc", "bc_mappo_finetune"),
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Fail when the selected arm's required BC checkpoints have not been trained yet.",
    )
    return parser


def main() -> None:
    """Print resolved settings, making absent artifacts visible instead of inventing results."""
    from multiuav.learning.bc_comparison import load_bc_comparison_config

    arguments = build_parser().parse_args()
    config = load_bc_comparison_config(arguments.config)
    modes = (arguments.mode,) if arguments.mode is not None else config.modes
    rows: list[dict[str, object]] = []
    missing: list[Path] = []
    for mode in modes:
        resolved = config.resolve(mode)
        checkpoints = config.checkpoint_paths(mode)
        present = {str(path): path.is_file() for path in checkpoints}
        missing.extend(path for path in checkpoints if not path.is_file())
        rows.append(
            {
                "mode": resolved.mode,
                "use_low_checkpoint": resolved.use_low_checkpoint,
                "use_high_checkpoint": resolved.use_high_checkpoint,
                "use_mappo_finetune": resolved.use_mappo_finetune,
                "freeze_encoder_updates": config.freeze_encoder_updates
                if resolved.use_mappo_finetune
                else None,
                "ppo_learning_rate": config.ppo_learning_rate
                if resolved.use_mappo_finetune
                else None,
                "imitation_coef": config.imitation_coef if resolved.use_mappo_finetune else None,
                "checkpoint_artifacts": present,
            }
        )
    if arguments.require_artifacts and missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Selected comparison arm requires missing BC artifacts: {names}")
    print(json.dumps({"seed": config.seed, "arms": rows}, indent=2))


if __name__ == "__main__":
    main()

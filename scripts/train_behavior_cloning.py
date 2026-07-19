"""Train strict-mask low/high behavior cloning from validated HDF5 expert shards."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Expose every reproducibility-sensitive behavior-cloning launch option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/rl/behavior_cloning.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/rl/behavior_cloning"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> None:
    """Load YAML, train the selected verified shards, and print artifact metrics."""
    from multiuav.data.bc_dataset import BCSplitConfig
    from multiuav.learning.bc_trainer import BehaviorCloningConfig, train_behavior_cloning

    args = build_parser().parse_args()
    payload = _load_yaml(args.config)
    split_payload = payload.pop("split")
    if not isinstance(split_payload, dict):
        raise ValueError("Behavior-cloning YAML split field must be a mapping.")
    config = BehaviorCloningConfig(
        shards=tuple((PROJECT_ROOT / str(path)).resolve() for path in payload.pop("shards")),
        split=BCSplitConfig(**split_payload),
        **payload,
    )
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive.")
        config = replace(config, epochs=args.epochs)
    device = _resolve_device(args.device)
    result = train_behavior_cloning(config, output_directory=args.output_dir, device=device)
    print(json.dumps({"device": str(device), **result}, indent=2))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Behavior-cloning YAML must contain a mapping.")
    return dict(payload)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()

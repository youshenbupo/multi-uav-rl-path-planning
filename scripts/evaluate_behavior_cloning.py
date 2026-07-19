"""Run deterministic closed-loop low-BC evaluation on manifest test episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    from multiuav.learning.bc_evaluation import evaluate_bc_low_checkpoint

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = [
        (Path(row["source_file"]), row["episode_id"])
        for row in manifest["episodes"]
        if row["split"] == "test"
    ]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.device == "cuda":
        device = torch.device("cuda")
    metrics = evaluate_bc_low_checkpoint(args.checkpoint, episodes, device=device)
    print(json.dumps(vars(metrics), indent=2))


if __name__ == "__main__":
    main()

"""Write the checkpoint- and scenario-specific core evaluation command manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Serialize 120 independent evaluation commands without running them."""
    from multiuav.experiments.core_jobs import materialize_core_evaluation_jobs
    from multiuav.experiments.core_protocol import load_core_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/core_3uav_mainline.yaml",
    )
    parser.add_argument(
        "--training-output-root", type=Path, default=PROJECT_ROOT / "outputs/core_3uav"
    )
    parser.add_argument(
        "--evaluation-output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/core_3uav_evaluations",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/core_3uav/core_evaluation_jobs.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    protocol = load_core_protocol(args.protocol)
    jobs = materialize_core_evaluation_jobs(
        protocol,
        project_root=PROJECT_ROOT,
        training_output_root=args.training_output_root,
        evaluation_output_root=args.evaluation_output_root,
        device=args.device,
    )
    payload = [
        {
            "method": job.method,
            "seed": job.seed,
            "scenario": job.scenario,
            "checkpoint": str(job.checkpoint),
            "output_directory": str(job.output_directory),
            "command": list(job.command),
        }
        for job in jobs
    ]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(jobs)} core evaluation jobs to {args.manifest}")


if __name__ == "__main__":
    main()

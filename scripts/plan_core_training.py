"""Write the immutable 3-UAV five-seed core training job manifest without starting GPU jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiuav.experiments.core_jobs import materialize_core_training_jobs  # noqa: E402
from multiuav.experiments.core_protocol import load_core_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/core_3uav_mainline.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/core_3uav")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "outputs/core_3uav/jobs.json"
    )
    arguments = parser.parse_args()
    protocol = load_core_protocol(arguments.protocol)
    jobs = materialize_core_training_jobs(
        protocol,
        project_root=PROJECT_ROOT,
        output_root=arguments.output_root,
        device=arguments.device,
    )
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(
        json.dumps(
            {
                "protocol": str(arguments.protocol),
                "jobs": [
                    {
                        "method": job.method,
                        "seed": job.seed,
                        "config": str(job.config),
                        "output_directory": str(job.output_directory),
                        "command": list(job.command),
                    }
                    for job in jobs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(arguments.manifest), "job_count": len(jobs)}))


if __name__ == "__main__":
    main()

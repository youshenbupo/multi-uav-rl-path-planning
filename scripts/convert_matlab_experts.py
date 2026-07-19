"""Convert the deterministic MATLAB reference fixture into an HDF5 expert episode."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Parse CLI options and convert one reference case into one HDF5 group."""
    from multiuav.data.expert_dataset import build_expert_episode
    from multiuav.data.expert_schema import write_episode
    from multiuav.data.matlab_import import import_matlab_reference
    from multiuav.expert.coordinator import ConflictAwareCoordinator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/regression/matlab/reference_cases.mat",
        help="MATLAB reference_cases.mat input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/expert/matlab_experts.h5",
        help="HDF5 output path.",
    )
    parser.add_argument(
        "--episode-id", default="matlab_simple", help="Output episode group identifier."
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Uniform temporal sampling interval.")
    parser.add_argument(
        "--max-speed", type=float, default=None, help="Optional low-level action speed cap."
    )
    parser.add_argument(
        "--coordinate-python",
        action="store_true",
        help=(
            "Attach auditable Python rule-coordinator logs; absent by default for MATLAB-only data."
        ),
    )
    args = parser.parse_args()
    imported = import_matlab_reference(args.input)
    if args.coordinate_python:
        coordinator = ConflictAwareCoordinator(imported.scenario, imported.nominal_speed)
        coordinated = coordinator.coordinate(imported.raw_trajectories)
        imported = replace(
            imported,
            repaired_trajectories=coordinated.repaired_trajectories,
            start_delays=coordinated.start_delays,
            source="python_rule_coordinator_replay",
            schedule_log=coordinated.schedule_log,
            spatial_repair_log=coordinated.repair_log,
        )
    episode = build_expert_episode(imported, dt=args.dt, max_speed=args.max_speed)
    write_episode(args.output, args.episode_id, episode)
    print(f"Wrote {args.episode_id!r} to {args.output}")
    print(f"High-level labels present: {int(episode.high_level_action_mask.sum())}")


if __name__ == "__main__":
    main()

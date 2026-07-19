"""Render one expert episode as 3D and top-view trajectory plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Render positions during active flight to a single PNG artifact."""
    from multiuav.data.expert_dataset import ExpertDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data/expert/matlab_experts.h5",
    )
    parser.add_argument("--episode-id", default="matlab_simple")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/expert/matlab_simple_overview.png",
    )
    args = parser.parse_args()
    episode = ExpertDataset(args.dataset).load_episode(args.episode_id)
    figure = plt.figure(figsize=(12, 5))
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_top = figure.add_subplot(1, 2, 2)
    for uav_index in range(episode.positions.shape[1]):
        active = episode.active_mask[:, uav_index]
        path = episode.positions[active, uav_index]
        axis_3d.plot(path[:, 0], path[:, 1], path[:, 2], label=f"UAV {uav_index}")
        axis_top.plot(path[:, 0], path[:, 1], label=f"UAV {uav_index}")
    axis_3d.set(xlabel="X", ylabel="Y", zlabel="Z", title="Expert trajectory (3D)")
    axis_top.set(xlabel="X", ylabel="Y", title="Expert trajectory (top view)")
    axis_3d.legend()
    axis_top.legend()
    axis_top.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""Roll out a deterministic goal-directed policy and render a Phase-8 environment episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_goal_directed_actions(environment: object) -> dict[str, np.ndarray]:
    """Return bounded, non-learning actions for an environment visualization rollout."""
    positions = np.asarray(getattr(environment, "positions"), dtype=float)
    scenario = getattr(environment, "scenario")
    config = getattr(environment, "config")
    actions: dict[str, np.ndarray] = {}
    for agent in getattr(environment, "agents"):
        index = getattr(environment, "agent_name_mapping")[agent]
        direction = np.asarray(scenario.missions[index].goal - positions[index], dtype=float)
        action = np.zeros(3, dtype=np.float32)
        horizontal_norm = float(np.linalg.norm(direction[:2]))
        if config.normalize_actions:
            if horizontal_norm > 1e-9:
                action[:2] = direction[:2] / horizontal_norm
            action[2] = float(
                np.clip(direction[2] / max(config.max_vertical_speed, 1e-9), -1.0, 1.0)
            )
        else:
            action[:2] = (
                direction[:2]
                / max(horizontal_norm, 1e-9)
                * min(horizontal_norm, config.max_horizontal_speed)
            )
            action[2] = float(
                np.clip(direction[2], -config.max_vertical_speed, config.max_vertical_speed)
            )
        actions[agent] = action
    return actions


def main() -> None:
    """Load a scenario/config, run a bounded goal-directed rollout, and save its diagnostics."""
    from multiuav.envs.multi_uav_env import MultiUAVParallelEnv, load_environment_config
    from multiuav.envs.wrappers import EpisodeTrajectoryRecorder
    from multiuav.expert.scenarios import load_scenario

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=PROJECT_ROOT / "configs/scenarios/s1.yaml")
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/env/multi_uav_mvp.yaml",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "data/regression/matlab/reference_cases.mat",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/expert/environment_episode_overview.png",
    )
    args = parser.parse_args()
    _, scenario, _ = load_scenario(args.scenario, args.reference)
    environment = EpisodeTrajectoryRecorder(
        MultiUAVParallelEnv(scenario, load_environment_config(args.environment_config))
    )
    environment.reset(seed=args.seed)
    while environment.agents:
        environment.step(build_goal_directed_actions(environment))
    environment.save_summary(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

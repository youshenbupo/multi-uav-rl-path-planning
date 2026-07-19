"""Non-invasive centralized-state and rendering helpers for the multi-UAV environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from multiuav.envs.multi_uav_env import MultiUAVParallelEnv


class CentralizedStateWrapper:
    """Expose an unchanged environment with a convenience centralized-state accessor."""

    def __init__(self, environment: MultiUAVParallelEnv) -> None:
        self.environment = environment

    def reset(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        return self.environment.reset(*args, **kwargs)

    def step(self, actions: dict[str, np.ndarray]) -> tuple[Any, Any, Any, Any, Any]:
        return self.environment.step(actions)

    def state(self) -> np.ndarray:
        return self.environment.state()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)


class EpisodeTrajectoryRecorder(CentralizedStateWrapper):
    """Record positions and separations without changing the wrapped environment dynamics."""

    def __init__(self, environment: MultiUAVParallelEnv) -> None:
        super().__init__(environment)
        self.positions_history: list[np.ndarray] = []
        self._minimum_separation_history: list[float] = []

    def reset(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        observations, infos = super().reset(*args, **kwargs)
        self.positions_history = [self.environment.positions.copy()]
        self._minimum_separation_history = [self.environment.minimum_separation]
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]) -> tuple[Any, Any, Any, Any, Any]:
        transition = super().step(actions)
        self.positions_history.append(self.environment.positions.copy())
        self._minimum_separation_history.append(self.environment.minimum_separation)
        return transition

    @property
    def minimum_separation_history(self) -> np.ndarray:
        """Return finite history samples as a new float array."""
        return np.asarray(self._minimum_separation_history, dtype=float)

    def save_summary(self, path: Path) -> None:
        """Render the required 3D, top-view, and separation-curve figure."""
        if not self.positions_history:
            raise RuntimeError("reset must be called before rendering an episode summary.")
        render_episode_summary(
            np.asarray(self.positions_history, dtype=float), self.minimum_separation_history, path
        )


def render_episode_summary(
    positions: np.ndarray, minimum_separation: np.ndarray, path: Path
) -> None:
    """Save a three-panel episode diagnostic image without opening a GUI window."""
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape [steps, num_uavs, 3].")
    figure = plt.figure(figsize=(15, 4.5))
    axis_3d = figure.add_subplot(1, 3, 1, projection="3d")
    axis_top = figure.add_subplot(1, 3, 2)
    axis_separation = figure.add_subplot(1, 3, 3)
    for index in range(positions.shape[1]):
        path_points = positions[:, index]
        axis_3d.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], label=f"UAV {index}")
        axis_top.plot(path_points[:, 0], path_points[:, 1], label=f"UAV {index}")
    axis_3d.set(xlabel="X", ylabel="Y", zlabel="Z", title="Trajectory (3D)")
    axis_top.set(xlabel="X", ylabel="Y", title="Trajectory (top view)")
    axis_top.set_aspect("equal", adjustable="box")
    axis_separation.plot(np.arange(len(minimum_separation)), minimum_separation)
    axis_separation.set(xlabel="Step", ylabel="Minimum separation", title="Minimum separation")
    axis_3d.legend()
    axis_top.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)

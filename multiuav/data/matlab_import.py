"""Import deterministic MATLAB reference cases as explicit expert-episode inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from multiuav.core.models import Scenario, TerrainMap, Trajectory, UAVMission

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImportedExpertCase:
    """A MATLAB source case without invented schedule or high-level labels."""

    scenario_id: str
    seed: int
    scenario: Scenario
    raw_trajectories: tuple[Trajectory, ...]
    repaired_trajectories: tuple[Trajectory, ...]
    start_delays: FloatArray
    nominal_speed: float
    source: str
    schedule_log: tuple[object, ...] = ()
    spatial_repair_log: tuple[object, ...] = ()


def import_matlab_reference(path: Path) -> ImportedExpertCase:
    """Load the exported flat synchronized fixture as one honest expert episode.

    The restored MATLAB reference contains final schedule values and repair
    paths, but not per-action histories. Therefore both log collections are
    intentionally empty and downstream high-level label masks remain false.
    """
    reference = loadmat(path, simplify_cells=True)["referenceCases"]
    evaluation = reference["evaluation"]
    raw_paths = _trajectory_tuple(evaluation["synchronized_paths"])
    repaired_paths = _trajectory_tuple(evaluation["spatial_repair_paths_after"])
    starts = tuple(path.points[0].copy() for path in raw_paths)
    goals = tuple(path.points[-1].copy() for path in raw_paths)
    terrain = TerrainMap(
        x_grid=np.array([0.0, 1000.0]),
        y_grid=np.array([0.0, 1000.0]),
        heights=np.zeros((2, 2), dtype=float),
    )
    missions = tuple(UAVMission(start=start, goal=goal) for start, goal in zip(starts, goals))
    travel_times = np.asarray(evaluation["synchronized_travel_times"], dtype=float)
    speed = _nominal_speed(raw_paths, travel_times)
    scenario = Scenario(
        terrain=terrain,
        threats=(),
        missions=missions,
        safe_separation=50.0,
        cruise_speed=speed,
        collision_samples=21,
    )
    metadata_path = path.with_name("metadata.json")
    seed = 0
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        seed = int(float(metadata.get("random_seed", 0)))
    return ImportedExpertCase(
        scenario_id="matlab_synchronized_flat",
        seed=seed,
        scenario=scenario,
        raw_trajectories=raw_paths,
        repaired_trajectories=repaired_paths,
        start_delays=np.asarray(evaluation["schedule_after_start_delays"], dtype=float),
        nominal_speed=speed,
        source="matlab_reference_without_action_logs",
    )


def _trajectory_tuple(value: object) -> tuple[Trajectory, ...]:
    paths = np.asarray(value, dtype=object)
    if paths.ndim == 3:
        return tuple(
            Trajectory(np.asarray(paths[index], dtype=float)) for index in range(paths.shape[0])
        )
    return tuple(Trajectory(np.asarray(path, dtype=float)) for path in paths.reshape(-1))


def _nominal_speed(paths: tuple[Trajectory, ...], travel_times: FloatArray) -> float:
    distances = np.asarray(
        [np.linalg.norm(np.diff(trajectory.points, axis=0), axis=1).sum() for trajectory in paths],
        dtype=float,
    )
    valid = travel_times > 1e-9
    if not np.any(valid):
        raise ValueError("MATLAB reference does not provide positive synchronized travel times.")
    speeds = distances[valid] / travel_times[valid]
    if not np.allclose(speeds, speeds[0], rtol=1e-6, atol=1e-8):
        raise ValueError("MATLAB synchronized paths do not share one nominal speed.")
    return float(speeds[0])

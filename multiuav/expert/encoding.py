"""MATLAB-compatible spherical candidate representation for fixed-waypoint planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import Scenario, Trajectory
from multiuav.expert.geometric_repair import GeometricRepairer, GeometricRepairInfo
from multiuav.geometry.trajectories import resample_path

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class UAVEncodingMeta:
    """Zero-based candidate indices for one UAV's length, azimuth, and elevation values."""

    length_indices: IntArray
    azimuth_indices: IntArray
    elevation_indices: IntArray
    base_step: float


@dataclass(frozen=True)
class PlanningProblem:
    """Candidate bounds, seed, and metadata shared by all optimizer operators."""

    scenario: Scenario
    num_waypoints: int
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    seed: FloatArray
    metadata: tuple[UAVEncodingMeta, ...]

    @property
    def dimensions(self) -> int:
        """Return the flattened candidate dimension."""
        return len(self.seed)


def build_planning_problem(scenario: Scenario, num_waypoints: int) -> PlanningProblem:
    """Mirror MATLAB planning bounds and its straight-line spherical seed."""
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be positive.")
    dimension = len(scenario.missions) * num_waypoints * 3
    lower = np.zeros(dimension, dtype=float)
    upper = np.zeros(dimension, dtype=float)
    seed = np.zeros(dimension, dtype=float)
    metadata: list[UAVEncodingMeta] = []
    cursor = 0
    for mission in scenario.missions:
        delta = mission.goal - mission.start
        planar_distance = float(np.hypot(delta[0], delta[1]))
        total_distance = float(np.linalg.norm(delta))
        base_step = total_distance / (num_waypoints + 1)
        nominal_azimuth = float(np.arctan2(delta[1], delta[0]))
        nominal_elevation = float(np.arctan2(delta[2], max(planar_distance, 1e-6)))
        lengths = np.arange(cursor, cursor + num_waypoints * 3, 3, dtype=np.int64)
        azimuths = lengths + 1
        elevations = lengths + 2
        lower[lengths] = 0.55 * base_step
        upper[lengths] = 1.65 * base_step
        lower[azimuths] = nominal_azimuth - np.pi / 1.8
        upper[azimuths] = nominal_azimuth + np.pi / 1.8
        lower[elevations] = max(-np.pi / 9.0, nominal_elevation - np.pi / 10.0)
        upper[elevations] = min(np.pi / 5.0, nominal_elevation + np.pi / 10.0)
        seed[lengths] = base_step
        seed[azimuths] = nominal_azimuth
        seed[elevations] = nominal_elevation
        metadata.append(UAVEncodingMeta(lengths, azimuths, elevations, base_step))
        cursor += num_waypoints * 3
    return PlanningProblem(scenario, num_waypoints, lower, upper, seed, tuple(metadata))


def project_candidate(candidate: FloatArray, problem: PlanningProblem) -> FloatArray:
    """Perform the bounded projection shared by initialization and mutation operators."""
    values = np.asarray(candidate, dtype=float).reshape(-1)
    if values.shape != problem.seed.shape:
        raise ValueError("Candidate shape does not match planning problem dimension.")
    return cast(
        FloatArray,
        np.asarray(np.clip(values, problem.lower_bounds, problem.upper_bounds), dtype=float),
    )


def chaotic_initialize(
    population_size: int, problem: PlanningProblem, rng: np.random.Generator
) -> FloatArray:
    """Mirror MATLAB logistic-map initialization and retain the straight seed as elite row 0."""
    if population_size < 1:
        raise ValueError("population_size must be positive.")
    chaos = rng.random((population_size, problem.dimensions))
    for index in range(1, problem.dimensions):
        chaos[:, index] = 4.0 * chaos[:, index - 1] * (1.0 - chaos[:, index - 1])
    population = problem.lower_bounds + chaos * (problem.upper_bounds - problem.lower_bounds)
    population[0] = problem.seed
    return population


def decode_candidate(
    candidate: FloatArray,
    problem: PlanningProblem,
    repairer: GeometricRepairer | None = None,
) -> tuple[Trajectory, ...]:
    """Decode length/azimuth/elevation values to repaired Cartesian trajectories."""
    values = project_candidate(candidate, problem)
    active_repairer = GeometricRepairer(problem.scenario) if repairer is None else repairer
    trajectories: list[Trajectory] = []
    for uav_index, (mission, meta) in enumerate(
        zip(problem.scenario.missions, problem.metadata, strict=True)
    ):
        current = mission.start.copy()
        points = [current.copy()]
        for length, azimuth, elevation in zip(
            values[meta.length_indices],
            values[meta.azimuth_indices],
            values[meta.elevation_indices],
            strict=True,
        ):
            direction = np.array(
                [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]
            )
            current, _ = active_repairer.repair_waypoint(current + length * direction, uav_index)
            points.append(current.copy())
        points.append(mission.goal.copy())
        trajectories.append(Trajectory(np.vstack(points)))
    return tuple(trajectories)


def encode_trajectories(
    trajectories: tuple[Trajectory, ...], problem: PlanningProblem
) -> FloatArray:
    """Encode repaired paths back into bounded spherical candidate coordinates."""
    if len(trajectories) != len(problem.metadata):
        raise ValueError("Trajectory count does not match planning problem metadata.")
    candidate = problem.seed.copy()
    for trajectory, meta in zip(trajectories, problem.metadata, strict=True):
        points = resample_path(trajectory.points, problem.num_waypoints + 2)
        segments = np.diff(points, axis=0)[: problem.num_waypoints]
        lengths = np.linalg.norm(segments, axis=1)
        planar = np.linalg.norm(segments[:, :2], axis=1)
        candidate[meta.length_indices] = lengths
        candidate[meta.azimuth_indices] = np.arctan2(segments[:, 1], segments[:, 0])
        candidate[meta.elevation_indices] = np.arctan2(segments[:, 2], np.maximum(planar, 1e-9))
    return project_candidate(candidate, problem)


def repair_candidate(
    candidate: FloatArray, problem: PlanningProblem
) -> tuple[FloatArray, tuple[Trajectory, ...], tuple[GeometricRepairInfo, ...]]:
    """Decode, geometrically repair, and re-encode a bounded candidate."""
    repairer = GeometricRepairer(problem.scenario)
    decoded = decode_candidate(candidate, problem, repairer)
    repaired: list[Trajectory] = []
    information: list[GeometricRepairInfo] = []
    for index, trajectory in enumerate(decoded):
        path, info = repairer.repair_trajectory(trajectory, index)
        repaired.append(path)
        information.append(info)
    repaired_tuple = tuple(repaired)
    return encode_trajectories(repaired_tuple, problem), repaired_tuple, tuple(information)

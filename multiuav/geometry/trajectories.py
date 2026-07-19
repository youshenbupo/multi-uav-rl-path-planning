"""MATLAB-compatible Cartesian trajectory geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def decode_spherical_path(
    start: ArrayLike,
    lengths: ArrayLike,
    azimuths: ArrayLike,
    elevations: ArrayLike,
) -> FloatArray:
    """Integrate MATLAB's length/azimuth/elevation steps without repair logic."""
    current = np.asarray(start, dtype=float).reshape(3)
    lengths_array = np.asarray(lengths, dtype=float).reshape(-1)
    azimuth_array = np.asarray(azimuths, dtype=float).reshape(-1)
    elevation_array = np.asarray(elevations, dtype=float).reshape(-1)
    if not (len(lengths_array) == len(azimuth_array) == len(elevation_array)):
        raise ValueError("Lengths, azimuths, and elevations must have equal lengths.")
    points = [current.copy()]
    for length, azimuth, elevation in zip(
        lengths_array, azimuth_array, elevation_array, strict=True
    ):
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        current = current + length * direction
        points.append(current.copy())
    return np.vstack(points)


def path_length(path: ArrayLike) -> float:
    """Return the sum of Euclidean segment lengths."""
    points = _path_array(path)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_path(path: ArrayLike, sample_count: int) -> FloatArray:
    """Resample a polyline uniformly in cumulative arc length, including endpoints."""
    points = _path_array(path)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(lengths)))
    queries = np.linspace(0.0, arc_length[-1], sample_count)
    sampled = np.empty((sample_count, 3), dtype=float)
    for index, query in enumerate(queries):
        segment_index = int(np.searchsorted(arc_length, query, side="right") - 1)
        if segment_index >= len(lengths):
            sampled[index] = points[-1]
            continue
        span = arc_length[segment_index + 1] - arc_length[segment_index]
        alpha = 0.0 if span == 0.0 else (query - arc_length[segment_index]) / span
        sampled[index] = (1.0 - alpha) * points[segment_index] + alpha * points[segment_index + 1]
    return sampled


def turn_angles(path: ArrayLike) -> FloatArray:
    """Return interior turn angles in radians."""
    segments = np.diff(_path_array(path), axis=0)
    angles: list[float] = []
    for first, second in zip(segments[:-1], segments[1:], strict=True):
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        cosine = 1.0 if denominator == 0.0 else float(np.dot(first, second) / denominator)
        angles.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.asarray(angles, dtype=float)


def segment_to_cylinder_distance(
    start: ArrayLike,
    end: ArrayLike,
    cylinder: ArrayLike,
    sample_count: int,
) -> float:
    """Match MATLAB's sampled signed distance to a finite vertical cylinder."""
    first = np.asarray(start, dtype=float).reshape(3)
    second = np.asarray(end, dtype=float).reshape(3)
    center_x, center_y, radius, height = np.asarray(cylinder, dtype=float).reshape(4)
    distances: list[float] = []
    for fraction in np.linspace(0.0, 1.0, sample_count):
        point = first + fraction * (second - first)
        radial = np.hypot(point[0] - center_x, point[1] - center_y) - radius
        vertical = max(0.0, -point[2], point[2] - height)
        if radial <= 0.0 and 0.0 <= point[2] <= height:
            distances.append(-min(-radial, height - point[2]))
        else:
            distances.append(float(np.hypot(max(radial, 0.0), vertical)))
    return min(distances)


def point_to_segment_distance(point: ArrayLike, start: ArrayLike, end: ArrayLike) -> float:
    """Return Euclidean distance from a point to a finite three-dimensional segment."""
    query = np.asarray(point, dtype=float).reshape(3)
    first = np.asarray(start, dtype=float).reshape(3)
    second = np.asarray(end, dtype=float).reshape(3)
    direction = second - first
    squared_length = float(np.dot(direction, direction))
    if squared_length == 0.0:
        return float(np.linalg.norm(query - first))
    fraction = np.clip(np.dot(query - first, direction) / squared_length, 0.0, 1.0)
    return float(np.linalg.norm(query - (first + fraction * direction)))


def _path_array(path: ArrayLike) -> FloatArray:
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("A path must have shape (N, 3) with at least two points.")
    return points

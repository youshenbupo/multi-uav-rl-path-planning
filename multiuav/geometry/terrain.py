"""Terrain interpolation and trajectory-derived terrain metrics."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multiuav.core.models import TerrainMap

FloatArray = NDArray[np.float64]


def terrain_height(terrain: TerrainMap, xy: ArrayLike) -> FloatArray:
    """Linearly interpolate terrain heights, returning zero outside the MATLAB grid.

    MATLAB's ``interp2(..., "linear")`` returns ``NaN`` beyond the grid and
    ``core.terrain_height`` subsequently changes that value to zero.
    """
    query = np.asarray(xy, dtype=float)
    if query.shape == (2,):
        query = query.reshape(1, 2)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("Terrain queries must have shape (2,) or (N, 2).")
    x_values = _axis_values(terrain.x_grid, "x_grid")
    y_values = _axis_values(terrain.y_grid, "y_grid")
    heights = np.asarray(terrain.heights, dtype=float)
    if heights.shape != (len(y_values), len(x_values)):
        raise ValueError("Terrain heights must have shape (len(y_grid), len(x_grid)).")
    query_x = query[:, 0]
    query_y = query[:, 1]
    inside = (
        (query_x >= x_values[0])
        & (query_x <= x_values[-1])
        & (query_y >= y_values[0])
        & (query_y <= y_values[-1])
    )
    values = np.zeros(len(query), dtype=float)
    if not np.any(inside):
        return cast(FloatArray, values)
    x_inside = query_x[inside]
    y_inside = query_y[inside]
    x_index = np.clip(np.searchsorted(x_values, x_inside, side="right") - 1, 0, len(x_values) - 2)
    y_index = np.clip(np.searchsorted(y_values, y_inside, side="right") - 1, 0, len(y_values) - 2)
    x_fraction = (x_inside - x_values[x_index]) / (x_values[x_index + 1] - x_values[x_index])
    y_fraction = (y_inside - y_values[y_index]) / (y_values[y_index + 1] - y_values[y_index])
    lower = (1.0 - x_fraction) * heights[y_index, x_index] + x_fraction * heights[
        y_index, x_index + 1
    ]
    upper = (1.0 - x_fraction) * heights[y_index + 1, x_index] + x_fraction * heights[
        y_index + 1, x_index + 1
    ]
    values[inside] = (1.0 - y_fraction) * lower + y_fraction * upper
    return cast(FloatArray, values)


def clearance(path: ArrayLike, terrain: TerrainMap) -> FloatArray:
    """Return each Cartesian waypoint's height above interpolated terrain."""
    points = _path_array(path)
    return points[:, 2] - terrain_height(terrain, points[:, :2])


def finite_difference_speeds(path: ArrayLike, time_points: ArrayLike) -> FloatArray:
    """Estimate each segment speed from Cartesian finite differences."""
    points = _path_array(path)
    times = np.asarray(time_points, dtype=float).reshape(-1)
    if len(times) != len(points):
        raise ValueError("time_points must provide one timestamp per path point.")
    deltas = np.diff(times)
    if np.any(deltas <= 0.0):
        raise ValueError("time_points must be strictly increasing.")
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / deltas
    return cast(FloatArray, np.asarray(speeds, dtype=float))


def _axis_values(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if np.allclose(array, array[0:1, :]):
            array = array[0, :]
        elif np.allclose(array, array[:, 0:1]):
            array = array[:, 0]
    array = array.reshape(-1)
    if len(array) < 2 or np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must contain at least two increasing grid values.")
    return array


def _path_array(path: ArrayLike) -> FloatArray:
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("A path must have shape (N, 3) with at least two points.")
    return points

"""Additional geometry coverage, including the exported MATLAB terrain grid."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.io import loadmat

from multiuav.core.models import TerrainMap
from multiuav.geometry.terrain import clearance, finite_difference_speeds, terrain_height
from multiuav.geometry.trajectories import (
    path_length,
    point_to_segment_distance,
    resample_path,
    segment_to_cylinder_distance,
    turn_angles,
)


class ExtendedGeometryTests(unittest.TestCase):
    """Validate remaining evaluator geometry primitives."""

    def test_terrain_height_uses_matlab_grid_orientation(self) -> None:
        terrain = TerrainMap(
            x_grid=np.array([0.0, 10.0]),
            y_grid=np.array([0.0, 20.0]),
            heights=np.array([[0.0, 10.0], [20.0, 30.0]]),
        )
        assert_allclose(terrain_height(terrain, [5.0, 10.0]), [15.0])
        assert_allclose(terrain_height(terrain, [-1.0, 10.0]), [0.0])

    def test_clearance_speed_and_point_segment_distance(self) -> None:
        terrain = TerrainMap(
            x_grid=np.array([0.0, 1.0]),
            y_grid=np.array([0.0, 1.0]),
            heights=np.zeros((2, 2)),
        )
        path = np.array([[0.0, 0.0, 4.0], [3.0, 4.0, 4.0], [6.0, 4.0, 4.0]])
        assert_allclose(clearance(path, terrain), [4.0, 4.0, 4.0])
        assert_allclose(finite_difference_speeds(path, [0.0, 2.0, 3.0]), [2.5, 3.0])
        self.assertEqual(4.0, point_to_segment_distance([3, 4, 0], [0, 0, 0], [6, 0, 0]))

    def test_terrain_queries_match_exported_matlab_reference(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "data/regression/matlab/reference_cases.mat"
        geometry = loadmat(fixture, simplify_cells=True)["referenceCases"]["geometry"]
        terrain = TerrainMap(
            x_grid=np.asarray(geometry["terrain_x"], dtype=float),
            y_grid=np.asarray(geometry["terrain_y"], dtype=float),
            heights=np.asarray(geometry["terrain_z_grid"], dtype=float),
        )
        queries = np.asarray(geometry["terrain_query_xy"], dtype=float)
        expected = np.asarray(geometry["terrain_query_z"], dtype=float)
        assert_allclose(terrain_height(terrain, queries), expected, rtol=1e-6, atol=1e-8)

    def test_exported_matlab_geometry_operations_match(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "data/regression/matlab/reference_cases.mat"
        geometry = loadmat(fixture, simplify_cells=True)["referenceCases"]["geometry"]
        path = np.asarray(geometry["decoded_path"], dtype=float)
        assert_allclose(path_length(path), geometry["path_length_total"], rtol=1e-6, atol=1e-8)
        assert_allclose(
            resample_path(path, 11), geometry["resampled_path_11"], rtol=1e-6, atol=1e-8
        )
        assert_allclose(turn_angles(path), geometry["turn_angles_rad"], rtol=1e-6, atol=1e-8)
        for index, expected in enumerate(geometry["cylinder_min_distances"]):
            segment = geometry["cylinder_segments"][:, :, index]
            assert_allclose(
                segment_to_cylinder_distance(segment[0], segment[1], geometry["cylinder"], 101),
                expected,
                rtol=1e-6,
                atol=1e-8,
            )


if __name__ == "__main__":
    unittest.main()

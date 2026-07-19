"""Deterministic geometry tests for MATLAB-compatible primitives."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose

from multiuav.geometry.trajectories import (
    decode_spherical_path,
    path_length,
    resample_path,
    segment_to_cylinder_distance,
    turn_angles,
)


class GeometryTests(unittest.TestCase):
    """Exercise core geometric behavior before evaluator integration."""

    def test_decodes_spherical_steps_to_cartesian_points(self) -> None:
        path = decode_spherical_path(
            start=np.array([0.0, 0.0, 0.0]),
            lengths=np.array([2.0, 3.0]),
            azimuths=np.array([0.0, np.pi / 2]),
            elevations=np.array([0.0, 0.0]),
        )
        assert_allclose(path, [[0, 0, 0], [2, 0, 0], [2, 3, 0]], rtol=1e-6, atol=1e-8)

    def test_length_resampling_and_turns_use_arc_length(self) -> None:
        path = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
        self.assertEqual(4.0, path_length(path))
        assert_allclose(
            resample_path(path, 5),
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0]],
        )
        assert_allclose(turn_angles(path), [np.pi / 2])

    def test_segment_to_cylinder_distance_uses_matlab_signed_convention(self) -> None:
        cylinder = np.array([0.0, 0.0, 5.0, 20.0])
        self.assertEqual(
            -5.0,
            segment_to_cylinder_distance([-10, 0, 10], [10, 0, 10], cylinder, 101),
        )
        self.assertEqual(
            5.0,
            segment_to_cylinder_distance([-10, 10, 10], [10, 10, 10], cylinder, 101),
        )

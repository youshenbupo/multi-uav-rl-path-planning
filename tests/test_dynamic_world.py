"""Tests for deterministic moving-obstacle world semantics."""

from __future__ import annotations

import numpy as np
import pytest

from multiuav.core.models import DynamicCylinder
from multiuav.envs.dynamic_world import DynamicWorldState, validate_dynamic_trajectories


def test_dynamic_world_advances_constant_velocity_without_mutating_definition() -> None:
    """Advancing a world changes only the derived obstacle center at the next step."""
    obstacle = DynamicCylinder(
        identifier="crossing",
        initial_center=np.array([10.0, 5.0, 20.0]),
        velocity=np.array([2.0, 0.0, 0.0]),
        radius=3.0,
        height=40.0,
    )
    world = DynamicWorldState((obstacle,), dt=0.5)

    advanced = world.advance()

    np.testing.assert_allclose(world.centers[0], [10.0, 5.0, 20.0])
    np.testing.assert_allclose(advanced.centers[0], [11.0, 5.0, 20.0])
    assert advanced.step_count == 1


def test_dynamic_cylinder_rejects_invalid_geometry() -> None:
    """A dynamic obstacle must have finite vectors and strictly positive geometry."""
    with pytest.raises(ValueError, match="radius"):
        DynamicCylinder(
            identifier="invalid",
            initial_center=np.zeros(3),
            velocity=np.zeros(3),
            radius=0.0,
            height=1.0,
        )


def test_dynamic_trajectory_rejects_world_boundary_escape() -> None:
    """Constant-velocity scenarios fail before an obstacle can leave the world."""
    obstacle = DynamicCylinder(
        identifier="escape",
        initial_center=np.array([9.0, 5.0, 20.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        radius=1.0,
        height=10.0,
    )

    with pytest.raises(ValueError, match="escape"):
        validate_dynamic_trajectories(
            (obstacle,),
            dt=1.0,
            max_steps=2,
            world_x=(0.0, 10.0),
            world_y=(0.0, 10.0),
            world_z=(0.0, 30.0),
        )

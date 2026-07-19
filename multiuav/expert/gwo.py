"""Seeded Grey Wolf Optimizer guidance used by the restored HGALO optimizer."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gwo_guided_update(
    current: FloatArray,
    alpha: FloatArray,
    beta: FloatArray,
    delta: FloatArray,
    iteration: int,
    maximum_iterations: int,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Apply the MATLAB GWO update using an explicit NumPy random generator."""
    if iteration < 1 or maximum_iterations < 1:
        raise ValueError("iteration and maximum_iterations must be positive.")
    coefficient = 2.0 - 2.0 * (iteration - 1) / max(maximum_iterations - 1, 1)
    proposals: list[FloatArray] = []
    for leader in (alpha, beta, delta):
        a_values = 2.0 * coefficient * rng.random(current.shape) - coefficient
        c_values = 2.0 * rng.random(current.shape)
        distance = np.abs(c_values * leader - current)
        proposals.append(leader - a_values * distance)
    candidate = np.mean(proposals, axis=0)
    return cast(FloatArray, np.asarray(np.clip(candidate, lower_bounds, upper_bounds), dtype=float))

"""Seeded ALO-style local exploitation and Levy-flight primitives."""

from __future__ import annotations

from math import gamma
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from multiuav.expert.ca_hgalo import HGALOConfig

FloatArray = NDArray[np.float64]


def levy_flight_step(dimensions: int, beta: float, rng: np.random.Generator) -> FloatArray:
    """Generate the Mantegna Levy step used by MATLAB ``levy_flight_step``."""
    if dimensions < 1 or not 0.0 < beta <= 2.0:
        raise ValueError("dimensions must be positive and beta must be in (0, 2].")
    sigma = (
        gamma(1.0 + beta)
        * np.sin(np.pi * beta / 2.0)
        / (gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    numerator = rng.normal(size=dimensions) * sigma
    denominator = np.abs(rng.normal(size=dimensions)) ** (1.0 / beta)
    return cast(FloatArray, np.asarray(numerator / denominator, dtype=float))


def alo_local_exploitation(
    current: FloatArray,
    elite: FloatArray,
    iteration: int,
    maximum_iterations: int,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    rng: np.random.Generator,
    config: HGALOConfig,
) -> FloatArray:
    """Apply MATLAB's elite attraction, adaptive local scale, and optional Levy step."""
    scale = config.local_scale0 * max(
        0.02, (1.0 - iteration / max(maximum_iterations, 1)) ** config.local_decay
    )
    span = upper_bounds - lower_bounds
    step = rng.normal(size=current.shape) * span * scale
    if config.use_levy_flight:
        step += levy_flight_step(len(current), config.levy_beta, rng) * span * (0.02 * scale)
    attraction = 0.3 * rng.random(current.shape) * (elite - current)
    return cast(
        FloatArray,
        np.asarray(np.clip(current + attraction + step, lower_bounds, upper_bounds), dtype=float),
    )

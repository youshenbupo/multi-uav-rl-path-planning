"""Validated scale profiles for reproducible multi-UAV experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from multiuav.experiments.spec import SUPPORTED_UAV_COUNTS


@dataclass(frozen=True)
class ScaleProfile:
    """Resource and communication settings selected by a supported team size."""

    num_uavs: int
    num_envs: int
    rollout_length: int
    communication_radius: float

    def __post_init__(self) -> None:
        if self.num_uavs not in SUPPORTED_UAV_COUNTS:
            raise ValueError(f"Unsupported UAV count: {self.num_uavs}.")
        if self.num_envs < 1 or self.rollout_length < 1 or self.communication_radius <= 0.0:
            raise ValueError("Scale profile resource values must be positive.")


def load_scale_profiles(path: Path) -> tuple[ScaleProfile, ...]:
    """Load every required team-size profile in strict ascending order."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("profiles"), list):
        raise ValueError("Scale profile YAML must contain a profiles list.")
    profiles = tuple(
        ScaleProfile(**dict(item)) for item in payload["profiles"] if isinstance(item, Mapping)
    )
    if tuple(profile.num_uavs for profile in profiles) != SUPPORTED_UAV_COUNTS:
        raise ValueError(f"Profiles must exactly cover {SUPPORTED_UAV_COUNTS} in order.")
    return profiles

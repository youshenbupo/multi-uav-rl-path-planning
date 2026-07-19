"""Verified HDF5 episode indexing for strict-mask behavior cloning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from multiuav.core.models import CylindricalThreat, Scenario, TerrainMap, UAVMission
from multiuav.envs.observations import (
    EnvironmentSnapshot,
    build_local_observation,
    local_observation_size,
)
from multiuav.learning.conflict_graph import ConflictGraphBuilder, GraphBuildConfig

BCSplit = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class BCSplitConfig:
    """Deterministic, episode-only train/validation/test allocation settings."""

    seed: int
    train_fraction: float
    validation_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between zero and one.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one.")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction plus validation_fraction must be less than one.")


class ManifestEpisode(TypedDict):
    """JSON-compatible source and split metadata for one complete episode."""

    source_file: str
    episode_id: str
    episode_key: str
    scenario_id: str
    seed: int
    num_uavs: int
    split: BCSplit
    low_label_count: int
    high_label_count: int


class BehaviorCloningManifest(TypedDict):
    """Stable manifest written beside behavior-cloning training artifacts."""

    schema_version: str
    split_seed: int
    episodes: list[ManifestEpisode]


@dataclass(frozen=True)
class LoadedBCEpisode:
    """Unmodified action labels and masks loaded from one authoritative HDF5 group."""

    source_file: Path
    episode_id: str
    split: BCSplit
    high_actions: NDArray[np.int16]
    high_action_mask: NDArray[np.bool_]
    low_action_mask: NDArray[np.bool_]
    raw_edge_mask: NDArray[np.bool_]

    @property
    def high_label_count(self) -> int:
        """Return only source-authoritative high labels, never inferred classes."""
        return int(self.high_action_mask.sum())


@dataclass(frozen=True)
class BCPolicyFeatures:
    """MAPPO-compatible local observations and predictive graph tensors from expert state."""

    node_features: Tensor
    edge_features: Tensor
    adjacency: Tensor
    active_mask: Tensor


@dataclass(frozen=True)
class BCSupervision:
    """One full expert episode with MAPPO-aligned inputs and strict source targets."""

    features: BCPolicyFeatures
    low_actions: Tensor
    low_action_mask: Tensor
    high_actions: Tensor
    high_action_mask: Tensor
    raw_edge_mask: Tensor
    max_speed: float


def build_bc_manifest(paths: Sequence[Path], *, split: BCSplitConfig) -> BehaviorCloningManifest:
    """Index verified HDF5 shards and allocate complete episodes deterministically."""
    canonical_paths = _canonical_verified_paths(paths)
    episodes: list[ManifestEpisode] = []
    seen_keys: set[str] = set()
    for path in canonical_paths:
        with h5py.File(path, "r") as handle:
            if "episodes" not in handle:
                raise ValueError(f"Verified expert shard has no /episodes group: {path}")
            for episode_id in sorted(handle["episodes"].keys()):
                group = handle[f"episodes/{episode_id}"]
                episode_key = f"{path.name}::{episode_id}"
                if episode_key in seen_keys:
                    raise ValueError(f"Duplicate source episode key: {episode_key}")
                seen_keys.add(episode_key)
                scenario_id = _required_text_attribute(group, "scenario_id")
                episode_seed = _required_integer_attribute(group, "seed")
                num_uavs = _required_integer_attribute(group, "num_uavs")
                low_mask = np.asarray(group["low_level_action_mask"], dtype=bool)
                high_mask = np.asarray(group["high_level_action_mask"], dtype=bool)
                episodes.append(
                    {
                        "source_file": str(path),
                        "episode_id": episode_id,
                        "episode_key": episode_key,
                        "scenario_id": scenario_id,
                        "seed": episode_seed,
                        "num_uavs": num_uavs,
                        "split": _episode_split(scenario_id, episode_seed, split),
                        "low_label_count": int(low_mask.sum()),
                        "high_label_count": int(high_mask.sum()),
                    }
                )
    if not episodes:
        raise ValueError("No expert episodes were found in the supplied verified HDF5 shards.")
    return {"schema_version": "bc_manifest_v1", "split_seed": split.seed, "episodes": episodes}


def load_bc_episode(path: Path, episode_id: str, *, split: BCSplit) -> LoadedBCEpisode:
    """Load masks and high labels without replacing unavailable labels with continue."""
    canonical_path = _canonical_verified_paths([path])[0]
    with h5py.File(canonical_path, "r") as handle:
        group = handle[f"episodes/{episode_id}"]
        high_actions = np.asarray(group["high_level_actions"], dtype=np.int16)
        high_mask = np.asarray(group["high_level_action_mask"], dtype=bool)
        low_mask = np.asarray(group["low_level_action_mask"], dtype=bool)
        raw_edge_mask = _load_raw_edge_mask(group, time_count=high_actions.shape[0])
    if high_actions.shape != high_mask.shape or low_mask.shape != high_mask.shape:
        raise ValueError("BC action arrays and masks must share the fixed [T,N] layout.")
    if np.any(high_actions[~high_mask] != -1):
        raise ValueError("Unavailable high labels must be represented as -1.")
    return LoadedBCEpisode(
        source_file=canonical_path,
        episode_id=episode_id,
        split=split,
        high_actions=high_actions,
        high_action_mask=high_mask,
        low_action_mask=low_mask,
        raw_edge_mask=raw_edge_mask,
    )


def build_bc_policy_features(
    *,
    scenario: Scenario,
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    active_mask: NDArray[np.bool_],
    max_neighbors: int,
    max_horizontal_speed: float,
    max_vertical_speed: float,
    graph_config: GraphBuildConfig,
) -> BCPolicyFeatures:
    """Recreate the Phase-8 observations and Phase-10 graph at each expert state."""
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [T,N,3].")
    if velocities.shape != positions.shape or active_mask.shape != positions.shape[:2]:
        raise ValueError("velocities and active_mask must align with positions.")
    time_count, num_uavs, _ = positions.shape
    if num_uavs != len(scenario.missions):
        raise ValueError("Expert state UAV count must match scenario missions.")
    if min(time_count, max_horizontal_speed, max_vertical_speed) <= 0:
        raise ValueError("Expert feature inputs must have positive time count and speed limits.")
    goals = np.asarray([mission.goal for mission in scenario.missions], dtype=float)
    node_features = np.empty(
        (time_count, num_uavs, local_observation_size(max_neighbors)), dtype=np.float32
    )
    for time_index in range(time_count):
        current_positions = positions[time_index]
        snapshot = EnvironmentSnapshot(
            scenario=scenario,
            positions=current_positions,
            velocities=velocities[time_index],
            active_mask=active_mask[time_index],
            previous_goal_distances=np.linalg.norm(goals - current_positions, axis=1),
            step_count=time_index,
            max_steps=time_count,
            max_horizontal_speed=max_horizontal_speed,
            max_vertical_speed=max_vertical_speed,
            boundary_clipped=np.zeros(num_uavs, dtype=bool),
        )
        node_features[time_index] = np.asarray(
            [build_local_observation(snapshot, index, max_neighbors) for index in range(num_uavs)]
        )
    builder = ConflictGraphBuilder(graph_config)
    graph = builder.build(
        positions=torch.as_tensor(positions, dtype=torch.float32),
        velocities=torch.as_tensor(velocities, dtype=torch.float32),
        goals=torch.as_tensor(
            np.repeat(goals[None, :, :], time_count, axis=0), dtype=torch.float32
        ),
        active_mask=torch.as_tensor(active_mask, dtype=torch.bool),
    )
    return BCPolicyFeatures(
        node_features=torch.as_tensor(node_features, dtype=torch.float32),
        edge_features=graph.edge_features,
        adjacency=graph.adjacency,
        active_mask=graph.node_mask,
    )


def load_bc_supervision(
    path: Path,
    episode_id: str,
    *,
    max_neighbors: int,
    max_horizontal_speed: float,
    max_vertical_speed: float,
    graph_config: GraphBuildConfig,
) -> BCSupervision:
    """Load a verified episode as normalized low targets plus never-inferred high labels."""
    canonical_path = _canonical_verified_paths([path])[0]
    with h5py.File(canonical_path, "r") as handle:
        group = handle[f"episodes/{episode_id}"]
        positions = np.asarray(group["positions"], dtype=float)
        velocities = np.asarray(group["velocities"], dtype=float)
        active_mask = np.asarray(group["active_mask"], dtype=bool)
        low_actions = np.asarray(group["low_level_actions"], dtype=float)
        low_mask = np.asarray(group["low_level_action_mask"], dtype=bool)
        high_actions = np.asarray(group["high_level_actions"], dtype=np.int64)
        high_mask = np.asarray(group["high_level_action_mask"], dtype=bool)
        max_speed = _required_float_attribute(group, "max_speed")
        raw_edges = _load_raw_edge_mask(group, time_count=positions.shape[0])
        scenario = _scenario_from_group(group)
    if min(max_speed, max_horizontal_speed, max_vertical_speed) <= 0.0:
        raise ValueError("Expert and environment speed limits must be positive.")
    if low_actions.shape != positions.shape or low_mask.shape != positions.shape[:2]:
        raise ValueError("Low expert actions/mask must align with positions [T,N,3]/[T,N].")
    if high_actions.shape != low_mask.shape or high_mask.shape != low_mask.shape:
        raise ValueError("High expert labels/mask must align with low action mask [T,N].")
    features = build_bc_policy_features(
        scenario=scenario,
        positions=positions,
        velocities=velocities,
        active_mask=active_mask,
        max_neighbors=max_neighbors,
        max_horizontal_speed=max_horizontal_speed,
        max_vertical_speed=max_vertical_speed,
        graph_config=graph_config,
    )
    return BCSupervision(
        features=features,
        low_actions=torch.as_tensor(
            np.clip(low_actions / max_speed, -1.0, 1.0), dtype=torch.float32
        ),
        low_action_mask=torch.as_tensor(low_mask, dtype=torch.bool),
        high_actions=torch.as_tensor(high_actions, dtype=torch.long),
        high_action_mask=torch.as_tensor(high_mask, dtype=torch.bool),
        raw_edge_mask=torch.as_tensor(raw_edges, dtype=torch.bool),
        max_speed=max_speed,
    )


def load_bc_scenario(path: Path, episode_id: str) -> Scenario:
    """Load the immutable environment scenario stored with one verified expert episode."""
    canonical_path = _canonical_verified_paths([path])[0]
    with h5py.File(canonical_path, "r") as handle:
        return _scenario_from_group(handle[f"episodes/{episode_id}"])


def _canonical_verified_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved = tuple(sorted((Path(path).resolve() for path in paths), key=lambda value: str(value)))
    if not resolved:
        raise ValueError("At least one verified expert HDF5 shard is required.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same verified expert HDF5 shard was supplied more than once.")
    for path in resolved:
        if path.suffix != ".h5" or "verified" not in path.name or not path.is_file():
            raise ValueError(f"Expected an existing verified .h5 expert shard, got: {path}")
    return resolved


def _episode_split(scenario_id: str, episode_seed: int, split: BCSplitConfig) -> BCSplit:
    payload = f"{split.seed}:{scenario_id}:{episode_seed}".encode()
    fraction = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    if fraction < split.train_fraction:
        return "train"
    if fraction < split.train_fraction + split.validation_fraction:
        return "validation"
    return "test"


def _required_attribute(group: h5py.Group, name: str) -> object:
    if name not in group.attrs:
        raise ValueError(f"Expert episode {group.name} lacks required {name!r} attribute.")
    return group.attrs[name]


def _required_text_attribute(group: h5py.Group, name: str) -> str:
    value = _required_attribute(group, name)
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _required_integer_attribute(group: h5py.Group, name: str) -> int:
    """Read one scalar integer attribute without accepting strings or arrays."""
    value = _required_attribute(group, name)
    if not isinstance(value, int | np.integer):
        raise ValueError(f"Expert episode {group.name} has non-integer {name!r} attribute.")
    return int(value)


def _required_float_attribute(group: h5py.Group, name: str) -> float:
    """Read one scalar numerical attribute without accepting strings or arrays."""
    value = _required_attribute(group, name)
    if not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"Expert episode {group.name} has non-numeric {name!r} attribute.")
    return float(value)


def _load_raw_edge_mask(group: h5py.Group, *, time_count: int) -> NDArray[np.bool_]:
    """Densify only source-authoritative directed variable graph edges for auditing."""
    num_uavs = _required_integer_attribute(group, "num_uavs")
    values = np.zeros((time_count, num_uavs, num_uavs), dtype=bool)
    if "dynamic_graph" not in group or "steps" not in group["dynamic_graph"]:
        raise ValueError(f"Expert episode {group.name} lacks dynamic graph steps.")
    steps = group["dynamic_graph/steps"]
    if len(steps) != time_count:
        raise ValueError("Dynamic graph step count must equal fixed time-series length.")
    for time_index, step_name in enumerate(sorted(steps.keys())):
        step = steps[step_name]
        edge_index = np.asarray(step["edge_index"], dtype=np.int64)
        edge_mask = np.asarray(step["edge_mask"], dtype=bool)
        if edge_index.shape != (2, edge_mask.size):
            raise ValueError("Dynamic graph edge_index and edge_mask shapes are inconsistent.")
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_uavs):
            raise ValueError("Dynamic graph edge index lies outside the episode UAV range.")
        values[time_index, edge_index[0, edge_mask], edge_index[1, edge_mask]] = True
    return values


def _scenario_from_group(group: h5py.Group) -> Scenario:
    """Rebuild the immutable project scenario stored alongside one expert episode."""
    scenario_group = group["scenario"]
    terrain = TerrainMap(
        x_grid=np.asarray(scenario_group["terrain_x"], dtype=float),
        y_grid=np.asarray(scenario_group["terrain_y"], dtype=float),
        heights=np.asarray(scenario_group["terrain_heights"], dtype=float),
    )
    threat_values = np.asarray(scenario_group["threats"], dtype=float)
    threats = tuple(CylindricalThreat(*map(float, row)) for row in threat_values)
    starts = np.asarray(scenario_group["starts"], dtype=float)
    goals = np.asarray(scenario_group["goals"], dtype=float)
    parameters_value = scenario_group.attrs["parameters"]
    parameters_text = (
        parameters_value.decode("utf-8")
        if isinstance(parameters_value, bytes)
        else str(parameters_value)
    )
    parameters = json.loads(parameters_text)
    return Scenario(
        terrain=terrain,
        threats=threats,
        missions=tuple(
            UAVMission(start=start, goal=goal) for start, goal in zip(starts, goals, strict=True)
        ),
        **parameters,
    )

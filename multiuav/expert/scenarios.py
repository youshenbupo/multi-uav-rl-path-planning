"""Load MATLAB-exported S1--S4 terrain, terminals, threats, and YAML parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat

from multiuav.core.models import CylindricalThreat, Scenario, TerrainMap, UAVMission

_SCENARIO_INDEX = {
    "low_threat_3uav": 0,
    "base_5uav": 1,
    "high_threat_5uav": 2,
    "high_threat_8uav": 3,
}


def load_scenario(config_path: Path, reference_path: Path) -> tuple[str, Scenario, int]:
    """Load one YAML scenario and authoritative terrain/terminal values from MATLAB export."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Scenario configuration must be a mapping: {config_path}")
    scenario_key = str(config["scenario_key"])
    if scenario_key not in _SCENARIO_INDEX:
        raise ValueError(f"Unknown MATLAB scenario key: {scenario_key}")
    reference = loadmat(reference_path, simplify_cells=True)["referenceCases"]
    geometry = reference["geometry"]
    exported_scenario = reference["scenarios"][_SCENARIO_INDEX[scenario_key]]
    terrain = TerrainMap(
        x_grid=np.asarray(geometry["terrain_x"], dtype=float),
        y_grid=np.asarray(geometry["terrain_y"], dtype=float),
        heights=np.asarray(geometry["terrain_z_grid"], dtype=float),
    )
    starts = np.asarray(exported_scenario["starts"], dtype=float)
    goals = np.asarray(exported_scenario["goals"], dtype=float)
    threats = tuple(
        CylindricalThreat(*row) for row in np.asarray(exported_scenario["threats"], dtype=float)
    )
    missions = tuple(
        UAVMission(start=start, goal=goal) for start, goal in zip(starts, goals, strict=True)
    )
    scenario = Scenario(
        terrain=terrain,
        threats=threats,
        missions=missions,
        min_clearance=float(config["min_clearance"]),
        max_clearance=float(config["max_clearance"]),
        target_clearance=float(config["target_clearance"]),
        max_turn_degrees=float(config["max_turn_degrees"]),
        safe_separation=float(config["safe_separation"]),
        threat_margin=float(config["threat_margin"]),
        cruise_speed=float(config["cruise_speed"]),
        collision_samples=int(config["collision_samples"]),
        world_x=_bounds(config["world_x"]),
        world_y=_bounds(config["world_y"]),
        world_z=_bounds(config["world_z"]),
    )
    return str(config["scenario_id"]), scenario, int(config["num_waypoints"])


def load_hgalo_config(path: Path, seed: int | None = None) -> Any:
    """Load YAML values into ``HGALOConfig`` without scattering optimizer constants."""
    from multiuav.expert.ca_hgalo import HGALOConfig

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"HGALO configuration must be a mapping: {path}")
    mapping = {
        "pop_size": raw["population_size"],
        "max_iter": raw["max_iterations"],
        "seed": raw["seed"] if seed is None else seed,
        "use_chaos_init": raw["use_chaos_init"],
        "use_levy_flight": raw["use_levy_flight"],
        "use_elite_local_search": raw["use_elite_local_search"],
        "use_opposition_learning": raw["use_opposition_learning"],
        "use_feasibility_repair": raw["use_feasibility_repair"],
        "use_gwo_guidance": raw["use_gwo_guidance"],
        "use_alo_local": raw["use_alo_local"],
        "local_prob": raw["local_probability"],
        "top_k_ratio": raw["top_k_ratio"],
        "local_scale0": raw["local_scale0"],
        "local_decay": raw["local_decay"],
        "levy_beta": raw["levy_beta"],
        "elite_preserve": raw["elite_preserve"],
        "greedy_selection": raw["greedy_selection"],
        "use_deb_feasibility_rules": raw["use_deb_feasibility_rules"],
        "use_feasibility_preserving_mutation": raw["use_feasibility_preserving_mutation"],
        "adaptive_threat_penalty": raw["adaptive_threat_penalty"],
        "adaptive_threat_alpha": raw["adaptive_threat_alpha"],
        "adaptive_threat_min_scale": raw["adaptive_threat_min_scale"],
        "adaptive_threat_max_scale": raw["adaptive_threat_max_scale"],
    }
    return HGALOConfig(**mapping)


def _bounds(values: Any) -> tuple[float, float]:
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 2:
        raise ValueError("World bounds must contain exactly two values.")
    return bounds[0], bounds[1]

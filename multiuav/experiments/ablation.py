"""Strict ablation-manifest loading without substituting missing checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

SUPPORTED_ABLATIONS = frozenset(
    {
        "expert_pretrain",
        "cpa",
        "graph_encoder",
        "hierarchy",
        "high_level_delay",
        "altitude_maneuver",
        "cbf",
        "auxiliary_prediction",
        "curriculum",
    }
)


@dataclass(frozen=True)
class AblationEntry:
    """One trained ablation artifact and the components disabled during its training."""

    name: str
    checkpoint: Path | None
    disabled: tuple[str, ...]
    availability: str
    reason: str


@dataclass(frozen=True)
class AblationManifest:
    """Ordered manifest retained beside every resolved experiment matrix."""

    entries: tuple[AblationEntry, ...]


def load_ablation_manifest(path: Path) -> AblationManifest:
    """Load a manifest and mark missing artifacts unavailable without guessing results."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("entries"), list):
        raise ValueError("Ablation manifest must contain an entries list.")
    entries = tuple(_entry_from_mapping(item, path.parent) for item in payload["entries"])
    names = tuple(entry.name for entry in entries)
    if not entries or len(set(names)) != len(names):
        raise ValueError("Ablation manifest entries must be nonempty and have unique names.")
    return AblationManifest(entries=entries)


def _entry_from_mapping(value: object, parent: Path) -> AblationEntry:
    if not isinstance(value, Mapping):
        raise ValueError("Each ablation entry must be a mapping.")
    name = value.get("name")
    disabled_value = value.get("disabled", [])
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise ValueError("Each ablation entry requires a nonempty path-safe name.")
    if not isinstance(disabled_value, list) or not all(
        isinstance(item, str) for item in disabled_value
    ):
        raise ValueError("Ablation disabled must be a list of strings.")
    disabled = tuple(disabled_value)
    unknown = set(disabled) - SUPPORTED_ABLATIONS
    if unknown:
        raise ValueError(f"Unsupported ablations: {sorted(unknown)}.")
    checkpoint_value = value.get("checkpoint")
    if checkpoint_value is None:
        return AblationEntry(name, None, disabled, "unavailable", "checkpoint is not declared")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Ablation checkpoint must be a nonempty path string or null.")
    checkpoint = (parent / checkpoint_value).resolve()
    if not checkpoint.is_file():
        return AblationEntry(name, checkpoint, disabled, "unavailable", "missing checkpoint")
    return AblationEntry(name, checkpoint, disabled, "available", "trained checkpoint available")

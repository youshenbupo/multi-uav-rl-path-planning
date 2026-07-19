"""Validated experiment definitions for the required BC/MAPPO initialization comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

ComparisonMode = Literal[
    "random", "low_bc_only", "high_bc_only", "full_bc", "bc_mappo_finetune"
]
_REQUIRED_MODES = frozenset(
    {"random", "low_bc_only", "high_bc_only", "full_bc", "bc_mappo_finetune"}
)


@dataclass(frozen=True)
class ResolvedBCComparison:
    """Explicit checkpoint and optimization choices for one fair initialization arm."""

    mode: ComparisonMode
    use_low_checkpoint: bool
    use_high_checkpoint: bool
    use_mappo_finetune: bool


@dataclass(frozen=True)
class BCComparisonConfig:
    """YAML-owned definition of all Phase-12 initialization comparison arms."""

    seed: int
    modes: tuple[ComparisonMode, ...]
    bc_checkpoint_dir: Path
    freeze_encoder_updates: int
    ppo_learning_rate: float
    imitation_coef: float

    def __post_init__(self) -> None:
        if set(self.modes) != _REQUIRED_MODES or len(self.modes) != len(_REQUIRED_MODES):
            raise ValueError("BC comparison config must list every required mode exactly once.")
        if self.freeze_encoder_updates < 0 or self.ppo_learning_rate <= 0.0:
            raise ValueError("BC comparison fine-tuning schedule is invalid.")
        if self.imitation_coef < 0.0:
            raise ValueError("BC comparison imitation coefficient must be nonnegative.")

    def resolve(self, mode: str) -> ResolvedBCComparison:
        """Resolve a named arm without silently substituting an unavailable policy level."""
        if mode not in _REQUIRED_MODES:
            raise ValueError(f"Unsupported BC comparison mode: {mode}.")
        selected = cast(ComparisonMode, mode)
        return ResolvedBCComparison(
            mode=selected,
            use_low_checkpoint=selected in {"low_bc_only", "full_bc", "bc_mappo_finetune"},
            use_high_checkpoint=selected in {"high_bc_only", "full_bc", "bc_mappo_finetune"},
            use_mappo_finetune=selected == "bc_mappo_finetune",
        )

    def checkpoint_paths(self, mode: str) -> tuple[Path, ...]:
        """Return exactly the artifacts needed by one arm, preserving relative YAML paths."""
        resolved = self.resolve(mode)
        paths: list[Path] = []
        if resolved.use_low_checkpoint:
            paths.append(self.bc_checkpoint_dir / "bc_low_level.pt")
        if resolved.use_high_checkpoint:
            paths.append(self.bc_checkpoint_dir / "bc_high_level.pt")
        return tuple(paths)


def load_bc_comparison_config(path: Path) -> BCComparisonConfig:
    """Load and validate the complete comparison YAML rather than accepting ad-hoc flags."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("BC comparison configuration must be a YAML mapping.")
    values = dict(raw)
    modes = values.get("modes")
    if not isinstance(modes, list) or not all(isinstance(mode, str) for mode in modes):
        raise ValueError("BC comparison modes must be a list of strings.")
    values["modes"] = tuple(cast(ComparisonMode, mode) for mode in modes)
    checkpoint_dir = values.get("bc_checkpoint_dir")
    if not isinstance(checkpoint_dir, str):
        raise ValueError("bc_checkpoint_dir must be a path string.")
    values["bc_checkpoint_dir"] = Path(checkpoint_dir)
    return BCComparisonConfig(**values)

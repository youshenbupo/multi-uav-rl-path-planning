"""Strict serializable inputs shared by train, evaluation, benchmark, and ablation commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_UAV_COUNTS = (3, 5, 8, 12, 16)


@dataclass(frozen=True)
class ExperimentSpec:
    """One reproducible experiment request with no hidden command-line settings."""

    name: str
    seeds: tuple[int, ...]
    method: str = "full_method"
    device: str = "auto"
    num_uavs: int = 3
    scenario: str = "within_distribution"
    checkpoint: Path | None = None
    render: bool = False
    use_expert_pretrain: bool = True
    use_graph: bool = True
    use_hierarchy: bool = True
    use_cbf: bool = True

    def __post_init__(self) -> None:
        if not self.name or any(part in self.name for part in ("/", "\\")):
            raise ValueError("Experiment name must be a nonempty directory name.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Experiment seeds must be nonempty and unique.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Experiment device must be auto, cpu, or cuda.")
        if self.num_uavs not in SUPPORTED_UAV_COUNTS:
            raise ValueError(f"num_uavs must be one of {SUPPORTED_UAV_COUNTS}.")

    def as_dict(self) -> dict[str, object]:
        """Return a YAML/JSON-safe mapping with checkpoint paths converted to text."""
        values = asdict(self)
        values["checkpoint"] = str(self.checkpoint) if self.checkpoint is not None else None
        values["seeds"] = list(self.seeds)
        return values

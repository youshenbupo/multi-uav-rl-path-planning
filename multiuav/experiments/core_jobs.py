"""Materialize independent core-protocol training jobs before any long GPU execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multiuav.experiments.core_protocol import CoreProtocol


@dataclass(frozen=True)
class CoreTrainingJob:
    """One non-shareable method/seed training job with an explicit output directory."""

    method: str
    seed: int
    config: Path
    output_directory: Path
    command: tuple[str, ...]


def materialize_core_training_jobs(
    protocol: CoreProtocol, *, project_root: Path, output_root: Path, device: str
) -> tuple[CoreTrainingJob, ...]:
    """Create deterministic CLI contracts for every independently trained 3-UAV core arm."""
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto.")
    jobs: list[CoreTrainingJob] = []
    for method in protocol.methods:
        for seed in protocol.seeds:
            output_directory = output_root / f"{method.artifact_prefix}_seed_{seed}"
            command: tuple[str, ...]
            if method.family == "mappo":
                command = (
                    "python",
                    "scripts/train_mappo.py",
                    "--config",
                    str(method.config.relative_to(project_root)),
                    "--device",
                    device,
                    "--seed",
                    str(seed),
                    "--num-uavs",
                    str(protocol.initial_num_uavs),
                    "--total-steps",
                    str(protocol.training_transitions_per_seed),
                    "--output-dir",
                    str(output_directory),
                )
            else:
                assert method.graph_mode is not None
                command = (
                    "python",
                    "scripts/train_graph_mappo.py",
                    "--config",
                    str(method.config.relative_to(project_root)),
                    "--device",
                    device,
                    "--seed",
                    str(seed),
                    "--num-uavs",
                    str(protocol.initial_num_uavs),
                    "--graph-mode",
                    method.graph_mode,
                    "--total-steps",
                    str(protocol.training_transitions_per_seed),
                    "--output-dir",
                    str(output_directory),
                )
            jobs.append(
                CoreTrainingJob(
                    method=method.name,
                    seed=seed,
                    config=method.config,
                    output_directory=output_directory,
                    command=command,
                )
            )
    return tuple(jobs)

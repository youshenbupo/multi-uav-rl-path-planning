"""Independent artifacts and commands for every core protocol seed/arm pair."""

from pathlib import Path

import multiuav.experiments.core_jobs as core_jobs
from multiuav.experiments.core_jobs import materialize_core_training_jobs
from multiuav.experiments.core_protocol import load_core_protocol


def test_core_job_matrix_has_one_unique_artifact_per_method_and_seed() -> None:
    root = Path(__file__).parents[1]
    protocol = load_core_protocol(root / "configs/experiments/core_3uav_mainline.yaml")
    jobs = materialize_core_training_jobs(
        protocol, project_root=root, output_root=root / "outputs/core_3uav", device="cuda"
    )

    assert len(jobs) == 20
    assert len({job.output_directory for job in jobs}) == 20
    assert all("--total-steps" in job.command for job in jobs)
    assert all("--seed" in job.command for job in jobs)
    assert {job.method for job in jobs} == {method.name for method in protocol.methods}


def test_core_evaluation_job_matrix_preserves_training_checkpoint_and_scenario_identity() -> None:
    """Every paper condition must evaluate its own seed checkpoint, not a shared substitute."""
    root = Path(__file__).parents[1]
    protocol = load_core_protocol(root / "configs/experiments/core_3uav_mainline.yaml")
    materialize = getattr(core_jobs, "materialize_core_evaluation_jobs", None)
    assert materialize is not None

    jobs = materialize(
        protocol,
        project_root=root,
        training_output_root=root / "outputs/core_3uav",
        evaluation_output_root=root / "outputs/core_3uav_evaluations",
        device="cuda",
    )

    assert len(jobs) == 120
    assert len({job.output_directory for job in jobs}) == 120
    assert {job.scenario for job in jobs} == set(protocol.scenarios)
    assert all("--checkpoint" in job.command and "--scenario" in job.command for job in jobs)
    assert all("checkpoints" in str(job.checkpoint) for job in jobs)

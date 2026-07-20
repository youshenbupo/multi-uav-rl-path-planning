"""Independent artifacts and commands for every core protocol seed/arm pair."""

from pathlib import Path

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

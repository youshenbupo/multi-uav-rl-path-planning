"""Tests for reproducible Phase-14 experiment artifacts and method availability."""

from __future__ import annotations

import json
from pathlib import Path

from multiuav.experiments.ablation import load_ablation_manifest
from multiuav.experiments.metrics import SeedResult
from multiuav.experiments.outputs import ExperimentOutput
from multiuav.experiments.registry import MethodRegistry
from multiuav.experiments.runner import evaluate_goal_controller, evaluate_hierarchical_checkpoint
from multiuav.experiments.scales import load_scale_profiles
from multiuav.experiments.spec import ExperimentSpec


def test_experiment_writer_preserves_raw_seed_result_and_statistical_summary(
    tmp_path: Path,
) -> None:
    """Every requested seed remains in JSONL while summaries include uncertainty."""
    spec = ExperimentSpec(name="smoke", seeds=(3, 5))
    layout = ExperimentOutput.create(tmp_path, spec, metadata={"device": "cpu"})
    layout.write_raw_result(seed=3, episode=0, result={"success_rate": 1.0, "energy_proxy": 3.0})
    summary = layout.write_summary(
        [
            SeedResult(seed=3, metrics={"success_rate": 1.0, "energy_proxy": 3.0}),
            SeedResult(seed=5, metrics={"success_rate": 0.0, "energy_proxy": 5.0}),
        ]
    )

    raw_path = layout.root / "raw_results" / "seed_3.jsonl"
    assert json.loads(raw_path.read_text(encoding="utf-8").strip())["seed"] == 3
    assert summary["success_rate"]["mean"] == 0.5
    assert summary["success_rate"]["confidence_interval_95"] is not None
    assert (layout.root / "summary.csv").is_file()
    assert (layout.root / "environment.json").is_file()


def test_unavailable_external_baseline_is_explicit() -> None:
    """Unimplemented external baselines cannot silently produce invented values."""
    method = MethodRegistry().resolve("orca")

    assert method.availability == "unavailable"
    assert "not implemented" in method.reason


def test_registry_does_not_claim_unified_dynamic_adapter_for_legacy_methods() -> None:
    """Existing legacy scripts stay unavailable until they can run the shared dynamic protocol."""
    method = MethodRegistry().resolve("predictive_graph_mappo")

    assert method.availability == "unavailable"
    assert "dynamic-world adapter" in method.reason


def test_dynamic_communication_evaluation_writes_auditable_seed_metrics(tmp_path: Path) -> None:
    """The common evaluator exercises moving obstacles, delayed communication, and CBF."""
    spec = ExperimentSpec(name="dynamic_smoke", seeds=(13,), num_uavs=3, device="cpu")
    layout = ExperimentOutput.create(tmp_path, spec, metadata={"device": "cpu"})
    result = evaluate_goal_controller(spec, layout, episodes_per_seed=1, max_steps=4)

    assert result[0].seed == 13
    assert "CBF_intervention_rate" in result[0].metrics
    assert (layout.root / "figures" / "seed_13_episode_0.png").is_file()
    raw = (layout.root / "raw_results" / "seed_13.jsonl").read_text(encoding="utf-8")
    assert '"dynamic_obstacle_count": 1' in raw
    assert '"communication_delay_steps": 1' in raw


def test_checkpoint_evaluation_runs_hierarchy_cbf_and_writes_episode_figure(tmp_path: Path) -> None:
    """A compatible fixture checkpoint traverses the learned hierarchy evaluation path."""
    from dataclasses import replace

    import torch

    from multiuav.learning.hierarchical_mappo import save_hierarchical_checkpoint
    from multiuav.learning.hierarchical_runner import (
        HierarchicalMAPPOExperiment,
        load_hierarchical_experiment_config,
    )

    config_path = Path(__file__).parents[1] / "configs/rl/hierarchical_mappo.yaml"
    config = replace(
        load_hierarchical_experiment_config(config_path),
        seed=41,
        num_envs=1,
        total_steps=1,
    )
    fixture = HierarchicalMAPPOExperiment(config, device=torch.device("cpu"))
    checkpoint = tmp_path / "fixture_random_policy.pt"
    save_hierarchical_checkpoint(checkpoint, fixture.trainer, stage="low", step=0)
    spec = ExperimentSpec(
        name="checkpoint_smoke", seeds=(43,), num_uavs=3, device="cpu", checkpoint=checkpoint
    )
    output = ExperimentOutput.create(tmp_path, spec, metadata={"fixture_checkpoint": True})

    results = evaluate_hierarchical_checkpoint(
        spec, output, config_path=config_path, episodes_per_seed=1, max_steps=3
    )

    assert "CBF_intervention_rate" in results[0].metrics
    assert (output.root / "figures" / "seed_43_episode_0.png").is_file()
    raw = (output.root / "raw_results" / "seed_43.jsonl").read_text(encoding="utf-8")
    assert '"controller": "hierarchical_checkpoint"' in raw


def test_scale_profiles_define_every_required_uav_count() -> None:
    """The scale matrix is an executable source for every supported team size."""
    path = Path(__file__).parents[1] / "configs/experiments/scale_profiles.yaml"
    profiles = load_scale_profiles(path)

    assert tuple(profile.num_uavs for profile in profiles) == (3, 5, 8, 12, 16)
    assert all(profile.communication_radius > 0.0 for profile in profiles)


def test_ablation_manifest_marks_missing_checkpoint_as_unavailable(tmp_path: Path) -> None:
    """Ablation matrix entries cannot be evaluated without their own trained artifact."""
    manifest_path = tmp_path / "ablations.yaml"
    manifest_path.write_text(
        "entries:\n  - name: no_cbf\n    checkpoint: missing_hierarchy.pt\n    disabled: [cbf]\n",
        encoding="utf-8",
    )

    manifest = load_ablation_manifest(manifest_path)

    assert manifest.entries[0].availability == "unavailable"
    assert "missing checkpoint" in manifest.entries[0].reason

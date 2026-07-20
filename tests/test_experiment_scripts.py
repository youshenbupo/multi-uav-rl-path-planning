"""Command-level contracts for the single Phase-14 experiment interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from multiuav.learning.hierarchical_mappo import save_hierarchical_checkpoint
from multiuav.learning.hierarchical_runner import (
    HierarchicalMAPPOExperiment,
    load_hierarchical_experiment_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_phase14_entry_points_share_mainline_switches() -> None:
    """Every public command exposes the same reproducibility and ablation controls."""
    required_options = {
        "--config",
        "--seed",
        "--device",
        "--num-uavs",
        "--scenario",
        "--checkpoint",
        "--render",
        "--use-expert-pretrain",
        "--use-graph",
        "--use-hierarchy",
        "--use-cbf",
    }
    for script_name in ("train.py", "evaluate.py", "run_benchmark.py", "run_ablation.py"):
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert all(option in completed.stdout for option in required_options)


def test_evaluate_script_uses_checkpoint_evaluator_for_compatible_weights(tmp_path: Path) -> None:
    """The public evaluator executes checkpoint weights rather than replacing them."""
    from dataclasses import replace

    config_path = PROJECT_ROOT / "configs/rl/hierarchical_mappo.yaml"
    config = replace(load_hierarchical_experiment_config(config_path), num_envs=1, total_steps=1)
    fixture = HierarchicalMAPPOExperiment(config, device=torch.device("cpu"))
    checkpoint = tmp_path / "fixture.pt"
    save_hierarchical_checkpoint(checkpoint, fixture.trainer, stage="low", step=0)

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--seed",
            "59",
            "--episodes-per-seed",
            "1",
            "--max-steps",
            "2",
            "--output-root",
            str(tmp_path),
            "--experiment-name",
            "checkpoint_cli",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "checkpoint_cli" in completed.stdout
    assert (tmp_path / "checkpoint_cli" / "figures" / "seed_59_episode_0.png").is_file()
    metadata = json.loads(
        (tmp_path / "checkpoint_cli" / "environment.json").read_text(encoding="utf-8")
    )
    assert metadata["controller"] == "hierarchical_checkpoint"


def test_benchmark_rejects_checkpointless_learned_full_method(tmp_path: Path) -> None:
    """A benchmark must not substitute a smoke controller for the learned full method."""
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_benchmark.py"),
            "--methods",
            "full_method",
            "--device",
            "cpu",
            "--episodes-per-seed",
            "1",
            "--output-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "requires --checkpoint" in completed.stderr


def test_checkpointless_evaluation_is_recorded_as_semantic_smoke(tmp_path: Path) -> None:
    """A no-checkpoint run must not be labelled as the learned full method."""
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--device",
            "cpu",
            "--seed",
            "71",
            "--episodes-per-seed",
            "1",
            "--max-steps",
            "2",
            "--output-root",
            str(tmp_path),
            "--experiment-name",
            "semantic_smoke_cli",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "semantic_smoke_cli" in completed.stdout
    config = yaml.safe_load((tmp_path / "semantic_smoke_cli" / "config.yaml").read_text())
    assert config["method"] == "semantic_smoke"


def test_training_script_records_controller_and_cbf_telemetry(tmp_path: Path) -> None:
    """A real training invocation must retain CBF fallback evidence for later exclusion analysis."""
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config",
            str(PROJECT_ROOT / "configs" / "experiments" / "mainline_smoke.yaml"),
            "--device",
            "cpu",
            "--no-use-expert-pretrain",
            "--stage",
            "low",
            "--total-steps",
            "4",
            "--output-root",
            str(tmp_path),
            "--experiment-name",
            "training_telemetry_cli",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "training_telemetry_cli" in completed.stdout
    output = tmp_path / "training_telemetry_cli"
    metadata = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    telemetry = json.loads((output / "runtime_telemetry.json").read_text(encoding="utf-8"))
    assert metadata["controller"] == "hierarchical_training"
    assert telemetry["training"]["update_count"] == 1
    assert telemetry["cbf_configuration"]["slack_penalty"] == 100.0
    assert telemetry["cbf"]["decision_count"] == 4
    assert sum(telemetry["cbf"]["solver_status_counts"].values()) == 4


def test_ablation_script_writes_missing_artifact_resolution_without_metrics(tmp_path: Path) -> None:
    """Untrained ablation arms produce an auditable resolution instead of synthetic results."""
    manifest = PROJECT_ROOT / "configs/experiments/ablation_manifest.yaml"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_ablation.py"),
            "--manifest",
            str(manifest),
            "--output-root",
            str(tmp_path),
            "--experiment-name",
            "ablation_resolution",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "unavailable" in completed.stdout
    resolution = json.loads(
        (tmp_path / "ablation_resolution" / "ablation_resolution.json").read_text(encoding="utf-8")
    )
    assert all(entry["availability"] == "unavailable" for entry in resolution["entries"])
    assert not (tmp_path / "ablation_resolution" / "summary.json").exists()


def test_ablation_script_evaluates_available_arm_checkpoint(tmp_path: Path) -> None:
    """A manifest arm with a compatible checkpoint receives its own evaluated output."""
    from dataclasses import replace

    config_path = PROJECT_ROOT / "configs/experiments/mainline_smoke.yaml"
    config = replace(load_hierarchical_experiment_config(config_path), num_envs=1, total_steps=1)
    fixture = HierarchicalMAPPOExperiment(config, device=torch.device("cpu"))
    checkpoint = tmp_path / "fixture.pt"
    save_hierarchical_checkpoint(checkpoint, fixture.trainer, stage="low", step=0)
    manifest = tmp_path / "available_manifest.yaml"
    manifest.write_text(
        "entries:\n"
        "  - name: full_method\n"
        f"    checkpoint: {checkpoint.as_posix()}\n"
        "    disabled: []\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_ablation.py"),
            "--manifest",
            str(manifest),
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--seed",
            "73",
            "--episodes-per-seed",
            "1",
            "--output-root",
            str(tmp_path),
            "--experiment-name",
            "available_ablation",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "available" in completed.stdout
    assert (tmp_path / "available_ablation" / "full_method" / "summary.json").is_file()

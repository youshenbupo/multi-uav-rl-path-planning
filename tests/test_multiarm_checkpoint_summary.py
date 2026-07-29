"""Tests for descriptive multi-arm checkpoint-evaluation summaries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.summarize_multiarm_checkpoint_evaluations import summarize_multiarm_roots


def _write_cell(
    root: Path,
    *,
    prefix: str,
    seed: int,
    scenario: str,
    success: float,
) -> None:
    cell = root / f"{prefix}_seed_{seed}_{scenario}" / "raw_results"
    cell.mkdir(parents=True)
    records = [
        {
            "seed": seed,
            "scenario": scenario,
            "episode": episode,
            "success": success,
        }
        for episode in range(20)
    ]
    (cell / f"seed_{seed}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_multiarm_summary_uses_trained_seed_cells_for_each_arm(tmp_path: Path) -> None:
    roots = {"mlp": tmp_path / "mlp", "graph": tmp_path / "graph"}
    for arm, root in roots.items():
        _write_cell(
            root,
            prefix=arm,
            seed=11,
            scenario="nominal",
            success=0.0 if arm == "mlp" else 1.0,
        )
        _write_cell(
            root,
            prefix=arm,
            seed=12,
            scenario="nominal",
            success=1.0,
        )

    summary = summarize_multiarm_roots(roots)

    assert summary["independent_unit"] == "trained_seed"
    assert summary["episode_records_are_not_independent_training_replicates"] is True
    scenario = summary["scenario_summaries"][0]
    assert scenario["seed_count"] == 2
    assert scenario["episodes_per_seed"] == 20
    assert scenario["metrics"]["success"]["mlp"]["seed_mean"] == pytest.approx(0.5)
    assert scenario["metrics"]["success"]["graph"]["seed_mean"] == pytest.approx(1.0)


def test_multiarm_summary_rejects_unmatched_cell_identities(tmp_path: Path) -> None:
    mlp = tmp_path / "mlp"
    graph = tmp_path / "graph"
    _write_cell(mlp, prefix="mlp", seed=11, scenario="nominal", success=0.0)
    _write_cell(graph, prefix="graph", seed=12, scenario="nominal", success=1.0)

    with pytest.raises(ValueError, match="cell identities differ"):
        summarize_multiarm_roots({"mlp": mlp, "graph": graph})


def test_multiarm_summary_selects_each_arm_by_cell_prefix(tmp_path: Path) -> None:
    combined_root = tmp_path / "all_arms"
    for prefix, success in (("mlp", 0.0), ("graph", 1.0)):
        _write_cell(
            combined_root,
            prefix=prefix,
            seed=11,
            scenario="nominal",
            success=success,
        )

    summary = summarize_multiarm_roots(
        {"mlp": combined_root, "graph": combined_root},
        arm_prefixes={"mlp": "mlp_", "graph": "graph_"},
    )

    metrics = summary["scenario_summaries"][0]["metrics"]["success"]
    assert metrics["mlp"]["seed_mean"] == 0.0
    assert metrics["graph"]["seed_mean"] == 1.0


def test_multiarm_summary_script_is_directly_executable() -> None:
    script = Path(__file__).parents[1] / "scripts" / "summarize_multiarm_checkpoint_evaluations.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "NAME=ROOT" in result.stdout

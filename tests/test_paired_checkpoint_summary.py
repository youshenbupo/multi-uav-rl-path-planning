"""Tests for seed-level paired checkpoint-evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_paired_checkpoint_evaluations import summarize_paired_roots


def _write_cell(
    root: Path,
    *,
    prefix: str,
    seed: int,
    scenario: str,
    success: float,
    energy_proxy: float,
) -> None:
    cell = root / f"{prefix}_seed_{seed}_{scenario}" / "raw_results"
    cell.mkdir(parents=True)
    records = [
        {
            "seed": seed,
            "scenario": scenario,
            "episode": episode,
            "success": success,
            "energy_proxy": energy_proxy + episode,
        }
        for episode in range(20)
    ]
    (cell / f"seed_{seed}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_summarize_paired_roots_uses_seed_cells_not_episode_pseudoreplicates(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    treatment = tmp_path / "treatment"
    for seed, reference_success, treatment_success in (
        (11, 0.0, 1.0),
        (12, 1.0, 1.0),
    ):
        _write_cell(
            reference,
            prefix="full",
            seed=seed,
            scenario="nominal",
            success=reference_success,
            energy_proxy=10.0,
        )
        _write_cell(
            treatment,
            prefix="ablation",
            seed=seed,
            scenario="nominal",
            success=treatment_success,
            energy_proxy=20.0,
        )

    summary = summarize_paired_roots(reference, treatment)

    assert summary["independent_unit"] == "trained_seed"
    scenario = summary["scenario_summaries"][0]
    assert scenario["scenario"] == "nominal"
    assert scenario["seed_count"] == 2
    assert scenario["episodes_per_seed"] == 20
    success = scenario["metrics"]["success"]
    assert success["reference_seed_mean"] == pytest.approx(0.5)
    assert success["treatment_seed_mean"] == pytest.approx(1.0)
    assert success["paired_delta_treatment_minus_reference_mean"] == pytest.approx(0.5)
    assert success["paired_delta_sample_std"] == pytest.approx(0.5**0.5)
    energy = scenario["metrics"]["energy_proxy"]
    assert energy["paired_delta_treatment_minus_reference_mean"] == pytest.approx(10.0)


def test_summarize_paired_roots_rejects_missing_seed_scenario_cell(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    treatment = tmp_path / "treatment"
    _write_cell(
        reference,
        prefix="full",
        seed=11,
        scenario="nominal",
        success=0.0,
        energy_proxy=10.0,
    )

    with pytest.raises(ValueError, match="cell identities differ"):
        summarize_paired_roots(reference, treatment)

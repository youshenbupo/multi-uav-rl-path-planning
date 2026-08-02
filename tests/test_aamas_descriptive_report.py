"""Tests for the bounded AAMAS descriptive report artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_aamas_descriptive_report import (
    ARM_LABELS,
    CBF_METRICS,
    SCENARIOS,
    TASK_METRICS,
    build_artifact,
)


def _three_summary() -> dict[str, Any]:
    def arm_values(metric_index: int) -> dict[str, dict[str, float]]:
        return {
            arm: {"seed_mean": 0.1 * (index + 1) + metric_index, "seed_sample_std": 0.01}
            for index, arm in enumerate(ARM_LABELS)
        }

    return {
        "scenario_summaries": [
            {
                "scenario": scenario,
                "seed_count": 5,
                "episodes_per_seed": 20,
                "task_metrics": {
                    metric: arm_values(index) for index, metric in enumerate(TASK_METRICS)
                },
                "cbf_diagnostic_metrics": {
                    metric: arm_values(index) for index, metric in enumerate(CBF_METRICS)
                },
            }
            for scenario in SCENARIOS
        ]
    }


def _paired_summary(delta: float) -> dict[str, Any]:
    def metric_values(metric_index: int) -> dict[str, float]:
        return {
            "reference_seed_mean": float(metric_index),
            "reference_seed_sample_std": 0.1,
            "treatment_seed_mean": float(metric_index) + delta,
            "treatment_seed_sample_std": 0.2,
            "paired_delta_treatment_minus_reference_mean": delta,
            "paired_delta_sample_std": 0.3,
        }

    return {
        "scenario_summaries": [
            {
                "scenario": scenario,
                "seed_count": 5,
                "episodes_per_seed": 20,
                "task_metrics": {
                    metric: metric_values(index) for index, metric in enumerate(TASK_METRICS)
                },
                "cbf_diagnostic_metrics": {
                    metric: metric_values(index) for index, metric in enumerate(CBF_METRICS)
                },
            }
            for scenario in SCENARIOS
        ]
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_retains_complete_scenarios_metrics_and_claim_boundary(tmp_path: Path) -> None:
    three = tmp_path / "three.json"
    five = tmp_path / "five.json"
    eight = tmp_path / "eight.json"
    _write(three, _three_summary())
    _write(five, _paired_summary(-0.2))
    _write(eight, _paired_summary(0.1))

    artifact, provenance = build_artifact(
        three,
        five,
        eight,
        Path("outputs/report/provenance.json"),
        "2026-08-02T00:00:00Z",
        "deadbeef",
    )

    datasets = artifact["snapshot"]["datasets"]
    assert len(datasets["three_uav_success"]) == 6 * 4
    assert len(datasets["three_uav_task"]) == 6 * 11 * 4
    assert len(datasets["three_uav_cbf"]) == 6 * 5 * 4
    assert len(datasets["scale_success"]) == 6 * 2
    assert len(datasets["scale_task"]) == 6 * 11 * 2
    assert len(datasets["scale_cbf"]) == 6 * 5 * 2
    assert artifact["package_info"]["claim_boundary"] == "descriptive_only"
    assert [table["id"] for table in artifact["manifest"]["tables"]] == [
        "table_three_success",
        "table_scale_success",
    ]
    assert [chart["id"] for chart in artifact["manifest"]["charts"]] == [
        "chart_three_uav_success",
        "chart_scale_full_success",
    ]
    sources = {source["id"]: source for source in artifact["manifest"]["sources"]}
    assert sources["source_scale_success"]["query"]["sql"] == "SELECT * FROM scale_success"
    assert provenance["paired_delta_direction"].startswith("full_minus")
    assert provenance["omitted_visuals"][0]["replacement"].startswith("Exact compact table")
    assert "significance" in provenance["claim_boundary"]["prohibited"]


def test_report_rejects_incomplete_scenario_matrix(tmp_path: Path) -> None:
    three = tmp_path / "three.json"
    five = tmp_path / "five.json"
    eight = tmp_path / "eight.json"
    payload = _three_summary()
    payload["scenario_summaries"].pop()
    _write(three, payload)
    _write(five, _paired_summary(0.0))
    _write(eight, _paired_summary(0.0))

    with pytest.raises(ValueError, match="six canonical scenarios"):
        build_artifact(
            three,
            five,
            eight,
            Path("outputs/report/provenance.json"),
            "2026-08-02T00:00:00Z",
            "deadbeef",
        )

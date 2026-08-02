"""Tests for the portable AAMAS descriptive PDF report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader

from scripts.build_aamas_descriptive_pdf import build_pdf_report
from scripts.build_aamas_descriptive_report import ARM_LABELS, CBF_METRICS, SCENARIOS, TASK_METRICS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _three_summary() -> dict[str, Any]:
    def arm_values(metric_index: int) -> dict[str, dict[str, float]]:
        return {
            arm: {
                "seed_mean": 0.05 * (index + 1) + 0.001 * metric_index,
                "seed_sample_std": 0.01,
            }
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
            "reference_seed_mean": 0.2 + 0.001 * metric_index,
            "reference_seed_sample_std": 0.1,
            "treatment_seed_mean": 0.2 + delta + 0.001 * metric_index,
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


def test_pdf_report_is_parseable_complete_and_source_backed(tmp_path: Path) -> None:
    three = tmp_path / "three.json"
    five = tmp_path / "five.json"
    eight = tmp_path / "eight.json"
    pdf = tmp_path / "report.pdf"
    provenance = tmp_path / "provenance.json"
    _write(three, _three_summary())
    _write(five, _paired_summary(-0.2))
    _write(eight, _paired_summary(0.1))

    result = build_pdf_report(
        three,
        five,
        eight,
        pdf,
        provenance,
        generated_at="2026-08-02T00:00:00Z",
        git_revision="deadbeef",
    )

    assert pdf.read_bytes().startswith(b"%PDF-")
    reader = PdfReader(pdf)
    assert len(reader.pages) >= 4
    assert reader.metadata is not None
    assert reader.metadata.title == "Post-isolation multi-UAV descriptive evidence"
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "DESCRIPTIVE ONLY" in text
    assert "3-UAV success by method and scenario" in text
    assert "5/8-UAV paired success evidence" in text
    assert "No superiority, formal safety, component causality" in text
    for label in ("Nominal", "Delay only", "Loss only", "Dynamic only", "Combined"):
        assert label in text
    assert "OOD comm. + obstacle" in text

    sidecar = json.loads(provenance.read_text(encoding="utf-8"))
    assert sidecar["pdf_sha256"] == result["pdf_sha256"]
    assert sidecar["page_count"] == len(reader.pages)
    assert sidecar["claim_boundary"] == "descriptive_only"
    assert set(sidecar["inputs"]) == {"three_uav", "five_uav", "eight_uav"}
    assert sidecar["visible_success_rows"] == {"three_uav": 24, "scale_paired": 12}
    assert sidecar["builder"]["path"].endswith("scripts/build_aamas_descriptive_pdf.py")
    assert len(sidecar["builder"]["sha256"]) == 64
    assert sidecar["charts"]["scale_success"]["series"] == [
        "no_uncertainty_seed_mean",
        "full_seed_mean",
    ]


def test_pdf_report_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    occupied = tmp_path / "report.pdf"
    occupied.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_pdf_report(
            tmp_path / "three.json",
            tmp_path / "five.json",
            tmp_path / "eight.json",
            occupied,
            tmp_path / "provenance.json",
            generated_at="2026-08-02T00:00:00Z",
            git_revision="deadbeef",
        )


def test_pdf_report_supports_direct_script_execution() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_aamas_descriptive_pdf.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--three-uav-summary" in completed.stdout

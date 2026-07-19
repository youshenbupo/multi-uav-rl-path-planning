"""Auditable experiment directory, raw-result, and summary writers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from multiuav.experiments.metrics import SeedResult, SummaryMetric, aggregate_seed_results
from multiuav.experiments.spec import ExperimentSpec


@dataclass(frozen=True)
class ExperimentOutput:
    """Fixed layout that retains configs, metadata, raw seed records, and summaries."""

    root: Path
    spec: ExperimentSpec

    @classmethod
    def create(
        cls, base_directory: Path, spec: ExperimentSpec, *, metadata: dict[str, Any]
    ) -> ExperimentOutput:
        """Create every required output directory and persist immutable run context."""
        root = base_directory / spec.name
        for name in ("checkpoints", "tensorboard", "raw_results", "figures", "logs"):
            (root / name).mkdir(parents=True, exist_ok=True)
        (root / "config.yaml").write_text(
            yaml.safe_dump(spec.as_dict(), sort_keys=True), encoding="utf-8"
        )
        (root / "environment.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        (root / "README.md").write_text(
            "# Experiment output\n\nRaw seed results are retained under `raw_results/`.\n",
            encoding="utf-8",
        )
        return cls(root=root, spec=spec)

    def write_raw_result(self, *, seed: int, episode: int, result: dict[str, Any]) -> None:
        """Append one never-deleted episode record to its requested seed JSONL file."""
        if seed not in self.spec.seeds or episode < 0:
            raise ValueError("Raw result must use a requested seed and nonnegative episode.")
        payload = {"seed": seed, "episode": episode, **result}
        raw_result_path = self.root / "raw_results" / f"seed_{seed}.jsonl"
        with raw_result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_summary(self, results: list[SeedResult]) -> dict[str, SummaryMetric]:
        """Write JSON and CSV summaries while preserving every requested seed result."""
        if {result.seed for result in results} != set(self.spec.seeds):
            raise ValueError("Summary must contain exactly the requested seed set.")
        summary = aggregate_seed_results(results)
        (self.root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        with (self.root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("metric", "count", "mean", "std", "ci95_lower", "ci95_upper"),
            )
            writer.writeheader()
            for metric, values in summary.items():
                interval = values["confidence_interval_95"]
                writer.writerow(
                    {
                        "metric": metric,
                        "count": values["count"],
                        "mean": values["mean"],
                        "std": values["std"],
                        "ci95_lower": interval[0] if interval is not None else "",
                        "ci95_upper": interval[1] if interval is not None else "",
                    }
                )
        return summary

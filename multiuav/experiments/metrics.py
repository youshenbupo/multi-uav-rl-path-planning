"""Seed-preserving metric aggregation without outlier deletion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class SeedResult:
    """One complete aggregate record for a requested random seed."""

    seed: int
    metrics: dict[str, float]


class SummaryMetric(TypedDict):
    """One statistically aggregated metric written to the experiment summary."""

    count: int
    mean: float
    std: float
    confidence_interval_95: list[float] | None


def aggregate_seed_results(results: list[SeedResult]) -> dict[str, SummaryMetric]:
    """Compute mean, sample standard deviation, and two-sided 95% t intervals."""
    if not results:
        raise ValueError("At least one seed result is required for aggregation.")
    if len({result.seed for result in results}) != len(results):
        raise ValueError("Seed results must have unique seeds.")
    metric_names = sorted(set().union(*(result.metrics for result in results)))
    summary: dict[str, SummaryMetric] = {}
    for name in metric_names:
        values = np.asarray(
            [result.metrics[name] for result in results if name in result.metrics],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Metric '{name}' contains a non-finite seed value.")
        count = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        interval: list[float] | None = None
        if count > 1:
            half_width = float(t.ppf(0.975, df=count - 1) * std / np.sqrt(count))
            interval = [mean - half_width, mean + half_width]
        summary[name] = {
            "count": count,
            "mean": mean,
            "std": std,
            "confidence_interval_95": interval,
        }
    return summary

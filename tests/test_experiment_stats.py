"""Tests for deterministic aggregation of resumable HGALO experiment records."""

from __future__ import annotations

import unittest

from multiuav.expert.experiment_stats import summarize_records


class ExperimentStatsTests(unittest.TestCase):
    """Exercise the persisted-record aggregation boundary without planner mocks."""

    def test_summary_uses_latest_record_for_each_seed_and_averages_curves(self) -> None:
        records = [
            {
                "seed": 7,
                "best_cost": 12.0,
                "strict_success": False,
                "minimum_separation": 10.0,
                "temporal_conflict_count": 1,
                "runtime_seconds": 2.0,
                "convergence_history": [20.0, 12.0],
            },
            {
                "seed": 8,
                "best_cost": 8.0,
                "strict_success": True,
                "minimum_separation": 20.0,
                "temporal_conflict_count": 0,
                "runtime_seconds": 4.0,
                "convergence_history": [16.0, 8.0],
            },
            {
                "seed": 7,
                "best_cost": 10.0,
                "strict_success": True,
                "minimum_separation": 14.0,
                "temporal_conflict_count": 0,
                "runtime_seconds": 3.0,
                "convergence_history": [18.0, 10.0],
            },
        ]

        summary = summarize_records(records, scenario_id="s1")

        self.assertEqual(summary["seed_count"], 2)
        self.assertEqual([record["seed"] for record in summary["records"]], [7, 8])
        self.assertEqual(summary["best_cost"], 8.0)
        self.assertEqual(summary["mean_cost"], 9.0)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["mean_convergence_history"], [17.0, 9.0])


if __name__ == "__main__":
    unittest.main()

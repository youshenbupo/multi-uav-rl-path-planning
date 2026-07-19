"""Regression checks for the phase-one MATLAB legacy audit documents."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESTORED_SOURCE_ROOT = PROJECT_ROOT.parent / "legacy_hgalo" / "HGALO_恢复源码"
DOCS_ROOT = PROJECT_ROOT / "docs"


class LegacyAuditDocumentTests(unittest.TestCase):
    """Ensure the phase-one audit remains tied to observable source evidence."""

    def test_inventory_documents_the_observed_legacy_structure(self) -> None:
        inventory_path = DOCS_ROOT / "legacy_inventory.md"
        inventory = inventory_path.read_text(encoding="utf-8")

        self.assertIn("333", inventory)
        self.assertIn("81", inventory)
        self.assertIn("run_benchmark_formal.m", inventory)
        self.assertIn("run_HGALO_planner.m", inventory)
        self.assertIn("CA-HGALO", inventory)
        self.assertIn("not present", " ".join(inventory.lower().split()))
        self.assertEqual(81, len(list(RESTORED_SOURCE_ROOT.rglob("*.m"))))

    def test_mapping_covers_required_migration_modules(self) -> None:
        mapping_path = DOCS_ROOT / "matlab_python_mapping.md"
        mapping = mapping_path.read_text(encoding="utf-8")

        required_source_files = (
            "build_demo_environment.m",
            "terrain_height.m",
            "decode_path_for_uav.m",
            "resample_path.m",
            "single_uav_cost.m",
            "build_conflict_graph.m",
            "resolve_conflict_schedule.m",
            "repair_spatial_conflicts.m",
            "pso_baseline.m",
            "gwo_baseline.m",
            "run_HGALO_planner.m",
        )
        for source_file in required_source_files:
            self.assertIn(source_file, mapping)

        self.assertIn("MATLAB regression data", mapping)
        self.assertIn("1-based", mapping)
        self.assertIn("radian", mapping.lower())


if __name__ == "__main__":
    unittest.main()

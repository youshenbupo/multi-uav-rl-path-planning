"""HDF5 expert-episode schema, conversion, and validation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from multiuav.data.expert_dataset import ExpertDataset, build_expert_episode
from multiuav.data.expert_schema import write_episode
from multiuav.data.matlab_import import import_matlab_reference
from multiuav.data.validation import validate_expert_dataset
from multiuav.expert.coordinator import ConflictAwareCoordinator


class ExpertDatasetTests(unittest.TestCase):
    """Verify fixed-shape dynamics and variable-shape graph storage without padding errors."""

    def setUp(self) -> None:
        self.fixture = (
            Path(__file__).resolve().parents[1] / "data/regression/matlab/reference_cases.mat"
        )

    def test_matlab_episode_writes_required_hdf5_fields_and_masks_unknown_labels(self) -> None:
        imported = import_matlab_reference(self.fixture)
        episode = build_expert_episode(imported, dt=1.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experts.h5"
            write_episode(output, "matlab_simple", episode)
            with h5py.File(output, "r") as handle:
                group = handle["episodes/matlab_simple"]
                self.assertEqual(group.attrs["scenario_id"], "matlab_synchronized_flat")
                self.assertEqual(group["positions"].shape[1:], (2, 3))
                self.assertEqual(group["velocities"].shape, group["positions"].shape)
                self.assertEqual(group["active_mask"].shape, group["positions"].shape[:2])
                self.assertEqual(group["low_level_actions"].shape, group["positions"].shape)
                self.assertFalse(np.any(group["high_level_action_mask"][...]))
                self.assertEqual(group["raw_waypoints/offsets"].shape, (3,))
                self.assertIn("steps", group["dynamic_graph"])
                self.assertIn("schedule_log", group)
                self.assertIn("spatial_repair_log", group)

    def test_dataset_reader_and_validator_recompute_cost_and_shapes(self) -> None:
        imported = import_matlab_reference(self.fixture)
        episode = build_expert_episode(imported, dt=1.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experts.h5"
            write_episode(output, "matlab_simple", episode)
            report = validate_expert_dataset(output)
            reader = ExpertDataset(output)

            self.assertTrue(report.is_valid, report.issues)
            self.assertEqual(reader.episode_ids(), ("matlab_simple",))
            loaded = reader.load_episode("matlab_simple")
            self.assertEqual(loaded.positions.shape[1:], (2, 3))
            self.assertTrue(np.all(np.diff(loaded.timestamps) > 0.0))
            self.assertLessEqual(np.max(np.linalg.norm(loaded.low_level_actions, axis=2)), 20.0)

    def test_low_level_actions_are_clipped_and_record_the_ratio(self) -> None:
        imported = import_matlab_reference(self.fixture)
        episode = build_expert_episode(imported, dt=1.0, max_speed=5.0)

        self.assertTrue(np.any(episode.low_level_clip_ratio < 1.0))
        speeds = np.linalg.norm(episode.low_level_actions, axis=2)
        self.assertLessEqual(np.max(speeds), 5.0 + 1e-8)

    def test_python_coordinator_logs_create_masked_high_level_labels(self) -> None:
        imported = import_matlab_reference(self.fixture)
        result = ConflictAwareCoordinator(imported.scenario, imported.nominal_speed).coordinate(
            imported.raw_trajectories
        )
        episode = build_expert_episode(
            imported,
            dt=1.0,
            schedule_log=result.schedule_log,
            spatial_repair_log=result.repair_log,
        )

        self.assertGreater(np.count_nonzero(episode.high_level_action_mask), 0)
        self.assertTrue(np.all(episode.high_level_actions[episode.high_level_action_mask] >= 0))

    def test_validator_rejects_nonfinite_optional_convergence_history(self) -> None:
        imported = import_matlab_reference(self.fixture)
        episode = build_expert_episode(
            imported, dt=1.0, convergence_history=np.array([100.0, 50.0])
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "experts.h5"
            write_episode(output, "matlab_simple", episode)
            with h5py.File(output, "r+") as handle:
                handle["episodes/matlab_simple/convergence_history"][1] = np.nan
            report = validate_expert_dataset(output)

        self.assertFalse(report.is_valid)
        self.assertIn("convergence_history contains NaN or Inf.", report.issues[0])


if __name__ == "__main__":
    unittest.main()

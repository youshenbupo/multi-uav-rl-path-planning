"""TDD coverage for strict-mask expert behavior-cloning data and policies."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from multiuav.core.models import Scenario, TerrainMap, UAVMission
from multiuav.data.bc_dataset import (
    BCSplitConfig,
    build_bc_manifest,
    build_bc_policy_features,
    load_bc_episode,
    load_bc_supervision,
)
from multiuav.learning.bc_comparison import load_bc_comparison_config
from multiuav.learning.bc_evaluation import BCRolloutMetrics
from multiuav.learning.bc_finetuning import (
    BCFineTuneSchedule,
    GraphImitationBatch,
    load_shape_compatible_bc_state,
    masked_imitation_loss,
)
from multiuav.learning.bc_trainer import BehaviorCloningConfig, train_behavior_cloning
from multiuav.learning.behavior_cloning import (
    ExpertHighLevelPolicy,
    ExpertLowLevelPolicy,
    HighBCOutput,
    HighBCTargets,
    high_bc_loss,
    low_bc_loss,
)
from multiuav.learning.conflict_graph import EDGE_FEATURE_DIMENSION, GraphBuildConfig
from multiuav.learning.graph_mappo import GraphMAPPOConfig, GraphMAPPOTrainer
from multiuav.learning.graph_networks import (
    ConflictPredictionHead,
    GraphActor,
    GraphCentralizedCritic,
)
from multiuav.learning.graph_rollout_buffer import GraphRolloutBatch
from scripts.compare_bc_initialization import build_parser as build_bc_comparison_parser
from scripts.train_behavior_cloning import build_parser as build_behavior_cloning_parser


class BehaviorCloningTests(unittest.TestCase):
    """Exercise public BC interfaces using a small, authoritative-mask fixture."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temporary_directory.name) / "s1_verified_experts.h5"
        with h5py.File(self.dataset_path, "w") as handle:
            episodes = handle.create_group("episodes")
            for seed in range(3):
                episode = episodes.create_group(f"episode_{seed}")
                episode.attrs["scenario_id"] = "s1_low_threat_3uav"
                episode.attrs["seed"] = seed
                episode.attrs["num_uavs"] = 2
                episode.create_dataset(
                    "high_level_actions", data=np.array([[0, -1]], dtype=np.int16)
                )
                episode.create_dataset("high_level_action_mask", data=np.array([[True, False]]))
                episode.create_dataset("low_level_action_mask", data=np.ones((1, 2), dtype=bool))
                dynamic_graph = episode.create_group("dynamic_graph")
                graph_step = dynamic_graph.create_group("steps").create_group("000000")
                graph_step.create_dataset("edge_index", data=np.array([[0], [1]], dtype=np.int64))
                graph_step.create_dataset("edge_mask", data=np.array([True], dtype=bool))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_manifest_assigns_each_entire_episode_to_exactly_one_split(self) -> None:
        manifest = build_bc_manifest(
            [self.dataset_path],
            split=BCSplitConfig(seed=17, train_fraction=0.6, validation_fraction=0.2),
        )

        rows = manifest["episodes"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["episode_key"] for row in rows}), 3)
        self.assertTrue({row["split"] for row in rows} <= {"train", "validation", "test"})
        self.assertEqual(sum(int(row["low_label_count"]) for row in rows), 6)

    def test_unlabelled_high_action_remains_negative_and_masked(self) -> None:
        episode = load_bc_episode(self.dataset_path, "episode_0", split="train")

        self.assertEqual(int(episode.high_actions[0, 1]), -1)
        self.assertFalse(bool(episode.high_action_mask[0, 1]))
        self.assertEqual(episode.high_label_count, 1)

    def test_source_variable_graph_edge_mask_is_retained_without_symmetrising(self) -> None:
        episode = load_bc_episode(self.dataset_path, "episode_0", split="train")

        self.assertEqual(episode.raw_edge_mask.shape, (1, 2, 2))
        self.assertTrue(bool(episode.raw_edge_mask[0, 0, 1]))
        self.assertFalse(bool(episode.raw_edge_mask[0, 1, 0]))

    def test_policy_features_reconstruct_environment_observation_and_predictive_graph(self) -> None:
        terrain = TerrainMap(
            x_grid=np.array([0.0, 100.0]),
            y_grid=np.array([0.0, 100.0]),
            heights=np.zeros((2, 2)),
        )
        scenario = Scenario(
            terrain=terrain,
            threats=(),
            missions=(
                UAVMission(start=np.array([0.0, 0.0, 20.0]), goal=np.array([80.0, 0.0, 20.0])),
                UAVMission(start=np.array([0.0, 20.0, 20.0]), goal=np.array([80.0, 20.0, 20.0])),
            ),
            world_x=(0.0, 100.0),
            world_y=(0.0, 100.0),
            world_z=(0.0, 100.0),
        )
        features = build_bc_policy_features(
            scenario=scenario,
            positions=np.array([[[0.0, 0.0, 20.0], [0.0, 20.0, 20.0]]]),
            velocities=np.array([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            active_mask=np.array([[True, True]]),
            max_neighbors=1,
            max_horizontal_speed=8.0,
            max_vertical_speed=6.0,
            graph_config=GraphBuildConfig(
                communication_radius=30.0,
                risk_distance=12.0,
                prediction_horizon=5.0,
                top_k_neighbors=1,
                current_distance_edges=True,
                predicted_conflict_edges=True,
                self_loops=True,
            ),
        )

        self.assertEqual(features.node_features.shape, (1, 2, 23))
        self.assertEqual(features.edge_features.shape, (1, 2, 2, EDGE_FEATURE_DIMENSION))
        self.assertTrue(torch.equal(features.active_mask, torch.tensor([[True, True]])))
        self.assertTrue(bool(features.adjacency[0, 0, 1]))

    def test_low_bc_loss_ignores_invalid_action_rows_and_reports_all_terms(self) -> None:
        result = low_bc_loss(
            prediction=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            target=torch.tensor([[[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            mask=torch.tensor([[True, False]]),
        )

        self.assertAlmostEqual(float(result.normalized_mse), 1.0 / 12.0, places=6)
        self.assertAlmostEqual(float(result.direction), 0.0, places=6)
        self.assertAlmostEqual(float(result.speed), 0.25, places=6)

    def test_low_policy_conditions_velocity_on_high_command_and_masks_inactive_uav(self) -> None:
        policy = ExpertLowLevelPolicy(
            node_feature_dim=4,
            edge_feature_dim=EDGE_FEATURE_DIMENSION,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
        actions = policy(
            node_features=torch.zeros((1, 2, 4)),
            edge_features=torch.zeros((1, 2, 2, EDGE_FEATURE_DIMENSION)),
            adjacency=torch.eye(2, dtype=torch.bool).unsqueeze(0),
            active_mask=torch.tensor([[True, False]]),
            high_commands=torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6]]),
        )

        self.assertEqual(actions.shape, (1, 2, 3))
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.equal(actions[0, 1], torch.zeros(3)))

    def test_high_bc_never_relabels_unavailable_minus_one_as_continue(self) -> None:
        output = HighBCOutput(
            maneuver_logits=torch.tensor([[[20.0] + [0.0] * 8, [0.0, 0.0, 20.0] + [0.0] * 6]]),
            delay=torch.zeros((1, 2, 1)),
            priority=torch.zeros((1, 2, 1)),
            local_subgoal=torch.zeros((1, 2, 3)),
        )
        targets = HighBCTargets(
            actions=torch.tensor([[-1, 2]]),
            action_mask=torch.tensor([[False, True]]),
            delay=torch.zeros((1, 2, 1)),
            delay_mask=torch.zeros((1, 2), dtype=torch.bool),
            priority=torch.zeros((1, 2, 1)),
            priority_mask=torch.zeros((1, 2), dtype=torch.bool),
            local_subgoal=torch.zeros((1, 2, 3)),
            local_subgoal_mask=torch.zeros((1, 2), dtype=torch.bool),
        )

        result = high_bc_loss(output, targets)

        self.assertEqual(result.label_count, 1)
        self.assertLess(float(result.classification), 1e-5)
        self.assertIsNone(result.delay)
        self.assertIsNone(result.priority)
        self.assertIsNone(result.local_subgoal)

    def test_high_policy_exposes_all_schema_and_optional_continuous_heads(self) -> None:
        policy = ExpertHighLevelPolicy(
            node_feature_dim=4,
            edge_feature_dim=EDGE_FEATURE_DIMENSION,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
        output = policy(
            node_features=torch.zeros((1, 2, 4)),
            edge_features=torch.zeros((1, 2, 2, EDGE_FEATURE_DIMENSION)),
            adjacency=torch.eye(2, dtype=torch.bool).unsqueeze(0),
            active_mask=torch.tensor([[True, False]]),
        )

        self.assertEqual(output.maneuver_logits.shape, (1, 2, 9))
        self.assertEqual(output.delay.shape, (1, 2, 1))
        self.assertEqual(output.priority.shape, (1, 2, 1))
        self.assertEqual(output.local_subgoal.shape, (1, 2, 3))
        self.assertTrue(torch.equal(output.maneuver_logits[0, 1], torch.zeros(9)))

    def test_verified_hdf5_episode_loads_mapppo_aligned_supervision(self) -> None:
        path = Path("data/expert/s1_30seed_verified_experts.h5")
        supervision = load_bc_supervision(
            path,
            "s1_low_threat_3uav_seed_0",
            max_neighbors=3,
            max_horizontal_speed=35.0,
            max_vertical_speed=35.0,
            graph_config=GraphBuildConfig(
                communication_radius=100.0,
                risk_distance=70.0,
                prediction_horizon=5.0,
                top_k_neighbors=2,
                current_distance_edges=True,
                predicted_conflict_edges=True,
                self_loops=True,
            ),
        )

        self.assertEqual(supervision.features.node_features.shape[:2], (42, 3))
        self.assertEqual(supervision.features.edge_features.shape[-1], EDGE_FEATURE_DIMENSION)
        self.assertEqual(supervision.low_actions.shape, (42, 3, 3))
        self.assertTrue(torch.all(supervision.low_actions.abs() <= 1.0))
        self.assertEqual(int(supervision.high_action_mask.sum()), 0)

    def test_training_writes_all_required_behavior_cloning_artifacts(self) -> None:
        output_directory = Path(self.temporary_directory.name) / "bc_output"
        result = train_behavior_cloning(
            BehaviorCloningConfig(
                shards=(Path("data/expert/s1_30seed_verified_experts.h5"),),
                split=BCSplitConfig(seed=2, train_fraction=0.7, validation_fraction=0.15),
                epochs=1,
                learning_rate=1e-3,
                embedding_dim=8,
                graph_heads=2,
                graph_layers=1,
                max_neighbors=3,
                max_horizontal_speed=35.0,
                max_vertical_speed=35.0,
                normalized_mse_coef=1.0,
                direction_coef=0.2,
                speed_coef=0.1,
            ),
            output_directory=output_directory,
            device=torch.device("cpu"),
        )

        for name in (
            "bc_low_level.pt",
            "bc_high_level.pt",
            "normalization_stats.json",
            "training_config.yaml",
            "dataset_manifest.json",
        ):
            self.assertTrue((output_directory / name).is_file())
        self.assertTrue(np.isfinite(result["train_low_loss"]))
        self.assertEqual(int(result["high_label_count"]), 0)
        self.assertTrue(np.isfinite(result["validation_low_loss"]))

    def test_training_cli_exposes_reproducible_config_and_epoch_override(self) -> None:
        args = build_behavior_cloning_parser().parse_args(
            ["--config", "configs/rl/behavior_cloning.yaml", "--epochs", "2", "--device", "cpu"]
        )

        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.device, "cpu")

    def test_closed_loop_metric_contract_has_all_required_fields(self) -> None:
        metrics = BCRolloutMetrics(
            arrival_rate=0.5,
            collision_rate=0.0,
            minimum_separation=4.0,
            path_length=10.0,
            expert_path_deviation=2.0,
            episode_count=2,
        )

        self.assertEqual(metrics.episode_count, 2)
        self.assertGreater(metrics.minimum_separation, 0.0)

    def test_bc_finetune_schedule_freezes_then_releases_encoder(self) -> None:
        schedule = BCFineTuneSchedule(freeze_encoder_updates=2, ppo_learning_rate=1e-4)

        self.assertTrue(schedule.encoder_frozen(update_index=0))
        self.assertTrue(schedule.encoder_frozen(update_index=1))
        self.assertFalse(schedule.encoder_frozen(update_index=2))

    def test_bc_transfer_loads_only_shape_compatible_state(self) -> None:
        source = torch.nn.Linear(2, 3)
        target = torch.nn.Linear(2, 2)
        checkpoint = Path(self.temporary_directory.name) / "transfer.pt"
        torch.save({"state_dict": source.state_dict()}, checkpoint)

        report = load_shape_compatible_bc_state(target, checkpoint)

        self.assertEqual(report.loaded_keys, ())
        self.assertEqual(set(report.skipped_keys), {"weight", "bias"})

    def test_bc_encoder_transfers_into_matching_graph_actor(self) -> None:
        source = ExpertLowLevelPolicy(
            node_feature_dim=4,
            edge_feature_dim=EDGE_FEATURE_DIMENSION,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
        target = GraphActor(
            node_feature_dim=4,
            edge_feature_dim=EDGE_FEATURE_DIMENSION,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            action_dim=3,
        )
        checkpoint = Path(self.temporary_directory.name) / "bc_encoder.pt"
        torch.save({"state_dict": source.state_dict()}, checkpoint)

        report = load_shape_compatible_bc_state(target, checkpoint)

        self.assertIn("encoder.node_encoder.weight", report.loaded_keys)
        self.assertTrue(
            torch.equal(target.encoder.node_encoder.weight, source.encoder.node_encoder.weight)
        )

    def test_imitation_loss_ignores_invalid_expert_action_rows(self) -> None:
        loss = masked_imitation_loss(
            torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            torch.tensor([[True, False]]),
        )

        self.assertAlmostEqual(float(loss), 1.0 / 3.0, places=6)

    def test_comparison_config_exposes_all_five_required_modes(self) -> None:
        config = load_bc_comparison_config(Path("configs/experiments/bc_comparison.yaml"))
        parser = build_bc_comparison_parser()
        args = parser.parse_args(["--mode", "bc_mappo_finetune"])

        self.assertEqual(
            set(config.modes),
            {"random", "low_bc_only", "high_bc_only", "full_bc", "bc_mappo_finetune"},
        )
        self.assertTrue(config.resolve("full_bc").use_low_checkpoint)
        self.assertTrue(config.resolve("full_bc").use_high_checkpoint)
        self.assertTrue(config.resolve(args.mode).use_mappo_finetune)

    def test_graph_mappo_consumes_optional_masked_imitation_batch(self) -> None:
        """A nonzero configured coefficient must produce a reported BC auxiliary loss."""
        actor = GraphActor(
            node_feature_dim=4,
            edge_feature_dim=EDGE_FEATURE_DIMENSION,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            action_dim=3,
        )
        trainer = GraphMAPPOTrainer(
            actor=actor,
            critic=GraphCentralizedCritic(
                node_feature_dim=4,
                edge_feature_dim=EDGE_FEATURE_DIMENSION,
                embedding_dim=8,
                num_heads=2,
                num_layers=1,
            ),
            auxiliary=ConflictPredictionHead(
                embedding_dim=8, edge_feature_dim=EDGE_FEATURE_DIMENSION
            ),
            config=GraphMAPPOConfig(
                learning_rate=1e-3,
                gamma=0.99,
                gae_lambda=0.95,
                clip_ratio=0.2,
                entropy_coef=0.0,
                value_coef=0.5,
                max_grad_norm=1.0,
                batch_size=1,
                ppo_epochs=1,
                aux_conflict_coef=0.0,
                aux_distance_coef=0.0,
            ),
            device=torch.device("cpu"),
        )
        checkpoint = Path(self.temporary_directory.name) / "low_bc.pt"
        torch.save({"state_dict": actor.state_dict()}, checkpoint)
        trainer.initialize_from_bc(
            checkpoint,
            BCFineTuneSchedule(
                freeze_encoder_updates=1, ppo_learning_rate=1e-4, imitation_coef=0.5
            ),
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in trainer.actor.encoder.parameters())
        )
        node_features = torch.zeros((1, 2, 4))
        edge_features = torch.zeros((1, 2, 2, EDGE_FEATURE_DIMENSION))
        adjacency = torch.eye(2, dtype=torch.bool).unsqueeze(0)
        active_mask = torch.tensor([[True, False]])
        rollout = GraphRolloutBatch(
            node_features=node_features,
            edge_features=edge_features,
            adjacency=adjacency,
            active_mask=active_mask,
            actions=torch.zeros((1, 2, 3)),
            old_log_probabilities=torch.zeros((1, 2)),
            old_values=torch.zeros((1, 2)),
            returns=torch.zeros((1, 2)),
            advantages=torch.ones((1, 2)),
            auxiliary_conflict=torch.zeros((1, 2, 2), dtype=torch.bool),
            auxiliary_minimum_distance=torch.zeros((1, 2, 2)),
            auxiliary_mask=torch.zeros((1, 2, 2), dtype=torch.bool),
        )
        metrics = trainer.update(
            rollout,
            imitation_batch=GraphImitationBatch(
                node_features=node_features,
                edge_features=edge_features,
                adjacency=adjacency,
                active_mask=active_mask,
                expert_actions=torch.ones((1, 2, 3)),
                expert_action_mask=torch.tensor([[True, False]]),
            ),
        )

        self.assertIn("imitation_loss", metrics)
        self.assertGreater(metrics["imitation_loss"], 0.0)
        self.assertFalse(
            any(parameter.requires_grad for parameter in trainer.actor.encoder.parameters())
        )
        trainer.update(
            rollout,
            imitation_batch=GraphImitationBatch(
                node_features=node_features,
                edge_features=edge_features,
                adjacency=adjacency,
                active_mask=active_mask,
                expert_actions=torch.ones((1, 2, 3)),
                expert_action_mask=torch.tensor([[True, False]]),
            ),
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in trainer.actor.encoder.parameters())
        )


if __name__ == "__main__":
    unittest.main()

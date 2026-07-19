"""Tests for Phase-11 hierarchical action timing, buffers, and staged PPO."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from multiuav.learning.bc_finetuning import BCFineTuneSchedule
from multiuav.learning.behavior_cloning import ExpertHighLevelPolicy, ExpertLowLevelPolicy
from multiuav.learning.hierarchical_mappo import (
    HierarchicalMAPPOConfig,
    HierarchicalMAPPOTrainer,
    HighLevelBatch,
    LowLevelBatch,
    load_hierarchical_checkpoint,
    save_hierarchical_checkpoint,
)
from multiuav.learning.hierarchical_policy import (
    CoordinationAction,
    HierarchicalConfig,
    HierarchicalPolicy,
    HighLevelActionContext,
    HighLevelActor,
    HighLevelCritic,
    LowLevelActor,
    LowLevelCritic,
)
from multiuav.learning.hierarchical_rollout_buffer import (
    HierarchicalRolloutBuffer,
    compute_duration_aware_gae,
)
from multiuav.learning.hierarchical_runner import (
    HierarchicalExperimentConfig,
    HierarchicalMAPPOExperiment,
    load_hierarchical_experiment_config,
)
from scripts.evaluate_hierarchical_mappo import build_parser as build_hierarchical_evaluation_parser
from scripts.train_hierarchical_mappo import build_parser as build_hierarchical_training_parser


class HierarchicalPolicyTests(unittest.TestCase):
    """Exercise the high-level hold contract before learning-network integration."""

    def test_high_level_actions_hold_exactly_h_low_level_steps(self) -> None:
        policy = HierarchicalPolicy(
            HierarchicalConfig(
                high_interval=3,
                enable_high_level_policy=True,
                enable_local_subgoal=False,
                enable_priority=False,
                enable_delay=False,
                enable_altitude_maneuver=False,
            ),
            num_envs=1,
            num_agents=2,
            device=torch.device("cpu"),
        )
        proposals = torch.tensor(
            [[int(CoordinationAction.WAIT_OR_YIELD), int(CoordinationAction.SHIFT_LEFT)]]
        )
        active = torch.tensor([[True, True]])

        first_actions, first_boundary = policy.begin_low_step(proposals, active)
        policy.finish_low_step(active)
        second_actions, second_boundary = policy.begin_low_step(proposals + 1, active)
        policy.finish_low_step(active)
        third_actions, third_boundary = policy.begin_low_step(proposals + 1, active)
        policy.finish_low_step(active)
        fourth_actions, fourth_boundary = policy.begin_low_step(proposals + 1, active)

        self.assertTrue(first_boundary.all())
        self.assertFalse(second_boundary.any())
        self.assertFalse(third_boundary.any())
        self.assertTrue(fourth_boundary.all())
        self.assertTrue(torch.equal(first_actions, proposals))
        self.assertTrue(torch.equal(second_actions, proposals))
        self.assertTrue(torch.equal(third_actions, proposals))
        self.assertTrue(torch.equal(fourth_actions, proposals + 1))

    def test_hierarchical_runner_uses_delivered_knowledge_for_graph_edges(self) -> None:
        config = replace(
            load_hierarchical_experiment_config(Path("configs/rl/hierarchical_mappo.yaml")),
            num_envs=1,
            communication_enabled=True,
            communication_delay_steps=1,
            communication_max_staleness_steps=2,
        )
        experiment = HierarchicalMAPPOExperiment(config, device=torch.device("cpu"))

        graph = experiment._build_graph()

        expected_adjacency = torch.eye(config.num_uavs, dtype=torch.bool)
        self.assertTrue(torch.equal(graph.adjacency[0], expected_adjacency))

    def test_inactive_agent_resets_hold_and_disabled_context_is_zero(self) -> None:
        policy = HierarchicalPolicy(
            HierarchicalConfig(
                high_interval=4,
                enable_high_level_policy=True,
                enable_local_subgoal=False,
                enable_priority=False,
                enable_delay=False,
                enable_altitude_maneuver=False,
            ),
            num_envs=1,
            num_agents=2,
            device=torch.device("cpu"),
        )
        active = torch.tensor([[True, True]])
        policy.begin_low_step(
            torch.tensor([[int(CoordinationAction.CLIMB), int(CoordinationAction.CONTINUE)]]),
            active,
        )
        policy.finish_low_step(torch.tensor([[True, False]]))

        context = policy.context()

        self.assertEqual(context.actions[0, 1].item(), int(CoordinationAction.CONTINUE))
        self.assertEqual(context.remaining_steps[0, 1].item(), 0)
        self.assertTrue(
            torch.equal(
                context.local_subgoal_offsets, torch.zeros_like(context.local_subgoal_offsets)
            )
        )
        self.assertTrue(
            torch.equal(context.priority_scores, torch.zeros_like(context.priority_scores))
        )
        self.assertTrue(
            torch.equal(context.suggested_delays, torch.zeros_like(context.suggested_delays))
        )

    def test_high_and_low_level_actors_have_separate_action_contracts(self) -> None:
        torch.manual_seed(12)
        high = HighLevelActor(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
        )
        high_critic = HighLevelCritic(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
        )
        low = LowLevelActor(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            high_interval=3,
        )
        node_features = torch.randn(1, 2, 5)
        edge_features = torch.randn(1, 2, 2, 15)
        adjacency = torch.eye(2, dtype=torch.bool).unsqueeze(0)
        active = torch.tensor([[True, False]])
        high_actions, high_log_probabilities, high_entropy = high.sample(
            node_features, edge_features, adjacency, active, deterministic=False
        )
        context = policy_context(
            actions=high_actions,
            remaining_steps=torch.tensor([[3, 0]]),
        )
        low_actions, low_log_probabilities, low_entropy = low.sample(
            node_features, edge_features, adjacency, active, context, deterministic=False
        )

        self.assertEqual(high_actions.shape, (1, 2))
        self.assertEqual(high_log_probabilities.shape, (1, 2))
        self.assertEqual(high_entropy.shape, (1, 2))
        self.assertEqual(high_critic(node_features, edge_features, adjacency, active).shape, (1,))
        self.assertEqual(low_actions.shape, (1, 2, 3))
        self.assertEqual(low_log_probabilities.shape, (1, 2))
        self.assertEqual(low_entropy.shape, (1, 2))
        self.assertTrue(torch.equal(low_actions[~active], torch.zeros_like(low_actions[~active])))

    def test_high_buffer_aligns_boundaries_and_closes_early_episode(self) -> None:
        buffer = HierarchicalRolloutBuffer(num_envs=1, num_agents=2, gamma=0.9)
        active = torch.tensor([[True, True]])
        buffer.open_high(
            actions=torch.tensor([[0, 1]]),
            log_probabilities=torch.tensor([[0.1, 0.2]]),
            values=torch.tensor([[0.5, 0.6]]),
            boundary_mask=active,
            low_step=0,
        )
        buffer.add_low_rewards(torch.tensor([[1.0, 2.0]]))
        buffer.add_low_rewards(torch.tensor([[2.0, 3.0]]))
        buffer.close_high(
            close_mask=torch.tensor([[False, True]]),
            terminated=torch.tensor([[False, True]]),
            next_values=torch.tensor([[0.4, 0.0]]),
        )
        buffer.close_high(
            close_mask=torch.tensor([[True, False]]),
            terminated=torch.tensor([[False, False]]),
            next_values=torch.tensor([[0.4, 0.0]]),
        )

        records = buffer.records()

        self.assertEqual(len(records), 2)
        self.assertEqual({record.duration for record in records}, {2})
        self.assertEqual({record.low_start_step for record in records}, {0})
        self.assertEqual(
            [
                round(record.discounted_reward, 6)
                for record in sorted(records, key=lambda record: record.action)
            ],
            [2.8, 4.7],
        )
        self.assertEqual(sum(record.terminated for record in records), 1)

    def test_duration_aware_high_gae_uses_actual_duration_discount(self) -> None:
        advantages, returns = compute_duration_aware_gae(
            rewards=torch.tensor([2.8]),
            values=torch.tensor([0.5]),
            next_values=torch.tensor([0.4]),
            terminated=torch.tensor([False]),
            durations=torch.tensor([2]),
            gamma_low=0.9,
            gae_lambda=1.0,
        )

        self.assertAlmostEqual(advantages.item(), 2.624, places=6)
        self.assertAlmostEqual(returns.item(), 3.124, places=6)

    def test_stage_a_updates_low_only_and_stage_b_freezes_low_with_checkpoint(self) -> None:
        torch.manual_seed(51)
        high, high_critic, low, low_critic = hierarchical_networks()
        trainer = HierarchicalMAPPOTrainer(
            high_actor=high,
            high_critic=high_critic,
            low_actor=low,
            low_critic=low_critic,
            config=HierarchicalMAPPOConfig(
                learning_rate=3e-4,
                clip_ratio=0.2,
                entropy_coef=0.01,
                value_coef=0.5,
                max_grad_norm=0.5,
                batch_size=2,
                ppo_epochs=1,
                allow_joint_finetune=False,
            ),
            device=torch.device("cpu"),
        )
        low_batch, high_batch = hierarchical_batches()
        low_before = [parameter.detach().clone() for parameter in trainer.low_actor.parameters()]

        low_metrics = trainer.train_low(low_batch)
        low_after_stage_a = [
            parameter.detach().clone() for parameter in trainer.low_actor.parameters()
        ]
        high_metrics = trainer.train_high(high_batch)
        low_after_stage_b = [
            parameter.detach().clone() for parameter in trainer.low_actor.parameters()
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "hierarchical.pt"
            save_hierarchical_checkpoint(checkpoint, trainer, stage="high", step=11)
            restored = HierarchicalMAPPOTrainer(
                high_actor=hierarchical_networks()[0],
                high_critic=hierarchical_networks()[1],
                low_actor=hierarchical_networks()[2],
                low_critic=hierarchical_networks()[3],
                config=trainer.config,
                device=torch.device("cpu"),
            )
            stage, step = load_hierarchical_checkpoint(
                checkpoint, restored, map_location=torch.device("cpu")
            )

        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(low_before, low_after_stage_a)
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(low_after_stage_a, low_after_stage_b)
            )
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in low_metrics.values()))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in high_metrics.values()))
        self.assertEqual((stage, step), ("high", 11))

    def test_hierarchy_bc_transfer_remaps_only_compatible_low_encoder_layers(self) -> None:
        """Keep the widened low encoder safe while transferring compatible graph layers."""
        high, high_critic, low, low_critic = hierarchical_networks()
        trainer = HierarchicalMAPPOTrainer(
            high_actor=high,
            high_critic=high_critic,
            low_actor=low,
            low_critic=low_critic,
            config=HierarchicalMAPPOConfig(
                learning_rate=3e-4,
                clip_ratio=0.2,
                entropy_coef=0.01,
                value_coef=0.5,
                max_grad_norm=0.5,
                batch_size=2,
                ppo_epochs=1,
                allow_joint_finetune=False,
            ),
            device=torch.device("cpu"),
        )
        bc_high = ExpertHighLevelPolicy(
            node_feature_dim=5, edge_feature_dim=15, embedding_dim=16, num_heads=4, num_layers=1
        )
        bc_low = ExpertLowLevelPolicy(
            node_feature_dim=5, edge_feature_dim=15, embedding_dim=16, num_heads=4, num_layers=1
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            high_checkpoint = Path(temporary_directory) / "high.pt"
            low_checkpoint = Path(temporary_directory) / "low.pt"
            torch.save({"state_dict": bc_high.state_dict()}, high_checkpoint)
            torch.save({"state_dict": bc_low.state_dict()}, low_checkpoint)
            report = trainer.initialize_from_bc(
                low_checkpoint=low_checkpoint,
                high_checkpoint=high_checkpoint,
                schedule=BCFineTuneSchedule(freeze_encoder_updates=1, ppo_learning_rate=1e-4),
            )

        self.assertIn("encoder.node_encoder.weight", report.high.loaded_keys)
        self.assertIn("encoder.layers.0.edge_score.weight", report.low.loaded_keys)
        self.assertIn("encoder.node_encoder.weight", report.low.skipped_keys)
        self.assertFalse(
            any(parameter.requires_grad for parameter in trainer.high_actor.encoder.parameters())
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in trainer.low_actor.actor.encoder.parameters()
            )
        )
        self.assertEqual(trainer.high_optimizer.param_groups[0]["lr"], 1e-4)

    def test_rule_high_stage_collects_real_low_batch_and_trains_low_policy(self) -> None:
        config = HierarchicalExperimentConfig(
            seed=61,
            num_envs=1,
            num_uavs=3,
            rollout_length=4,
            total_steps=12,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=2,
            ppo_epochs=1,
            embedding_dim=16,
            graph_heads=4,
            graph_layers=1,
            high_interval=2,
            communication_radius=20.0,
            risk_distance=12.0,
            prediction_horizon=5.0,
            top_k_neighbors=2,
            enable_high_level_policy=True,
            enable_local_subgoal=False,
            enable_priority=False,
            enable_delay=False,
            enable_altitude_maneuver=False,
            allow_joint_finetune=False,
        )
        experiment = HierarchicalMAPPOExperiment(config, device=torch.device("cpu"))

        low_batch, high_batch, metrics = experiment.collect_rollout(stage="low")
        update = experiment.trainer.train_low(low_batch)

        self.assertEqual(low_batch.actions.shape, (4, 3, 3))
        self.assertGreater(high_batch.actions.shape[0], 0)
        self.assertIn("high_boundaries", metrics)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in update.values()))

    def test_hierarchical_yaml_and_cli_expose_staged_ablation_controls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load(
            (root / "configs/rl/hierarchical_mappo.yaml").read_text(encoding="utf-8")
        )
        config = load_hierarchical_experiment_config(root / "configs/rl/hierarchical_mappo.yaml")

        self.assertEqual(config.high_interval, 3)
        self.assertTrue(
            {
                "enable_high_level_policy",
                "enable_local_subgoal",
                "enable_priority",
                "enable_delay",
                "enable_altitude_maneuver",
                "allow_joint_finetune",
                "communication_enabled",
                "dynamic_obstacle_enabled",
                "cbf_enabled",
            }.issubset(payload)
        )
        self.assertEqual(
            build_hierarchical_training_parser().parse_args(["--stage", "low"]).stage, "low"
        )
        self.assertEqual(
            build_hierarchical_evaluation_parser()
            .parse_args(["--checkpoint", "checkpoint.pt", "--stage", "high"])
            .stage,
            "high",
        )

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for checkpoint map-location coverage"
    )
    def test_cuda_checkpoint_load_keeps_rng_state_on_cpu(self) -> None:
        """CUDA map_location must not move the CPU RNG-state tensor into CUDA."""
        trainer = HierarchicalMAPPOTrainer(
            high_actor=hierarchical_networks()[0],
            high_critic=hierarchical_networks()[1],
            low_actor=hierarchical_networks()[2],
            low_critic=hierarchical_networks()[3],
            config=hierarchical_trainer_config(),
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cpu_checkpoint.pt"
            save_hierarchical_checkpoint(checkpoint, trainer, stage="low", step=7)
            restored = HierarchicalMAPPOTrainer(
                high_actor=hierarchical_networks()[0],
                high_critic=hierarchical_networks()[1],
                low_actor=hierarchical_networks()[2],
                low_critic=hierarchical_networks()[3],
                config=trainer.config,
                device=torch.device("cuda"),
            )

            stage, step = load_hierarchical_checkpoint(
                checkpoint, restored, map_location=torch.device("cuda")
            )

        self.assertEqual((stage, step), ("low", 7))


def policy_context(*, actions: torch.Tensor, remaining_steps: torch.Tensor):
    """Create minimal context without a state machine for actor contract testing."""
    return HighLevelActionContext(
        actions=actions,
        action_one_hot=torch.nn.functional.one_hot(actions, num_classes=6).float(),
        remaining_steps=remaining_steps,
        local_subgoal_offsets=torch.zeros(1, 2, 3),
        priority_scores=torch.zeros(1, 2),
        suggested_delays=torch.zeros(1, 2),
    )


def hierarchical_networks() -> tuple[
    HighLevelActor, HighLevelCritic, LowLevelActor, LowLevelCritic
]:
    """Create small independent policy modules for staged-optimizer testing."""
    return (
        HighLevelActor(
            node_feature_dim=5, edge_feature_dim=15, embedding_dim=16, num_heads=4, num_layers=1
        ),
        HighLevelCritic(
            node_feature_dim=5, edge_feature_dim=15, embedding_dim=16, num_heads=4, num_layers=1
        ),
        LowLevelActor(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            high_interval=3,
        ),
        LowLevelCritic(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            high_interval=3,
        ),
    )


def hierarchical_trainer_config() -> HierarchicalMAPPOConfig:
    """Create the compact optimizer configuration used by checkpoint tests."""
    return HierarchicalMAPPOConfig(
        learning_rate=3e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        batch_size=2,
        ppo_epochs=1,
        allow_joint_finetune=False,
    )


def hierarchical_batches() -> tuple[LowLevelBatch, HighLevelBatch]:
    """Create finite graph tensors that exercise separate low/high PPO updates."""
    node_features = torch.randn(2, 2, 5)
    edge_features = torch.randn(2, 2, 2, 15)
    adjacency = torch.eye(2, dtype=torch.bool).repeat(2, 1, 1)
    active = torch.ones(2, 2, dtype=torch.bool)
    context = HighLevelActionContext(
        actions=torch.tensor([[0, 1], [2, 0]]),
        action_one_hot=torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 0]]), num_classes=6
        ).float(),
        remaining_steps=torch.tensor([[3, 3], [2, 2]]),
        local_subgoal_offsets=torch.zeros(2, 2, 3),
        priority_scores=torch.zeros(2, 2),
        suggested_delays=torch.zeros(2, 2),
    )
    return (
        LowLevelBatch(
            node_features=node_features,
            edge_features=edge_features,
            adjacency=adjacency,
            active_mask=active,
            context=context,
            actions=torch.zeros(2, 2, 3),
            old_log_probabilities=torch.zeros(2, 2),
            old_values=torch.zeros(2, 2),
            returns=torch.ones(2, 2),
            advantages=torch.tensor([[1.0, -1.0], [0.5, -0.5]]),
        ),
        HighLevelBatch(
            node_features=node_features,
            edge_features=edge_features,
            adjacency=adjacency,
            active_mask=active,
            actions=torch.tensor([[0, 1], [2, 0]]),
            old_log_probabilities=torch.zeros(2, 2),
            old_values=torch.zeros(2, 2),
            returns=torch.ones(2, 2),
            advantages=torch.tensor([[1.0, -1.0], [0.5, -0.5]]),
        ),
    )


if __name__ == "__main__":
    unittest.main()

"""Tests for Phase-10 predictive conflict-graph physics and masking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

from multiuav.envs.multi_uav_env import EnvironmentConfig, MultiUAVParallelEnv
from multiuav.learning.conflict_graph import (
    ConflictGraphBuilder,
    GraphBuildConfig,
    compute_cpa_features,
)
from multiuav.learning.graph_mappo import (
    GraphMAPPOConfig,
    GraphMAPPOTrainer,
    load_graph_checkpoint,
    save_graph_checkpoint,
)
from multiuav.learning.graph_networks import (
    ConflictPredictionHead,
    DenseGraphAttentionEncoder,
    GraphActor,
    GraphCentralizedCritic,
)
from multiuav.learning.graph_rollout_buffer import (
    GraphRolloutBatch,
    GraphRolloutBuffer,
    build_future_conflict_targets,
)
from multiuav.learning.graph_runner import (
    GraphExperimentConfig,
    GraphMAPPOExperiment,
    build_graph_from_environments,
    load_graph_experiment_config,
    make_graph_scenario,
)
from scripts.compare_graph_mappo import load_comparison_config
from scripts.evaluate_graph_mappo import build_parser as build_graph_evaluation_parser
from scripts.train_graph_mappo import build_parser as build_graph_training_parser


class PredictiveConflictGraphTests(unittest.TestCase):
    """Exercise the graph contracts before graph-policy integration exists."""

    def test_opposing_flight_has_exact_cpa_time_and_distance(self) -> None:
        time_to_cpa, distance_at_cpa = compute_cpa_features(
            relative_positions=torch.tensor([[10.0, 0.0, 0.0]]),
            relative_velocities=torch.tensor([[-2.0, 0.0, 0.0]]),
            prediction_horizon=8.0,
        )

        assert_allclose(time_to_cpa.numpy(), [5.0], atol=1e-6)
        assert_allclose(distance_at_cpa.numpy(), [0.0], atol=1e-6)

    def test_parallel_equal_velocity_has_finite_zero_time_cpa(self) -> None:
        time_to_cpa, distance_at_cpa = compute_cpa_features(
            relative_positions=torch.tensor([[10.0, 0.0, 0.0]]),
            relative_velocities=torch.tensor([[0.0, 0.0, 0.0]]),
            prediction_horizon=8.0,
        )

        self.assertTrue(torch.isfinite(time_to_cpa).all())
        self.assertTrue(torch.isfinite(distance_at_cpa).all())
        assert_allclose(time_to_cpa.numpy(), [0.0], atol=1e-6)
        assert_allclose(distance_at_cpa.numpy(), [10.0], atol=1e-6)

    def test_builder_uses_sparse_edges_and_removes_inactive_nodes(self) -> None:
        builder = ConflictGraphBuilder(
            GraphBuildConfig(
                communication_radius=3.0,
                risk_distance=2.0,
                prediction_horizon=5.0,
                top_k_neighbors=0,
                current_distance_edges=True,
                predicted_conflict_edges=True,
                self_loops=True,
            )
        )
        graph = builder.build(
            positions=torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [30.0, 0.0, 0.0]]]),
            velocities=torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            goals=torch.tensor([[[20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]]),
            active_mask=torch.tensor([[True, True, False]]),
        )

        self.assertEqual(graph.edge_features.shape, (1, 3, 3, 16))
        self.assertEqual(graph.adjacency.shape, (1, 3, 3))
        self.assertTrue(graph.adjacency[0, 0, 0])
        self.assertTrue(graph.adjacency[0, 1, 1])
        self.assertTrue(graph.adjacency[0, 0, 1])
        self.assertTrue(graph.adjacency[0, 1, 0])
        self.assertFalse(graph.adjacency[0, 2].any())
        self.assertFalse(graph.adjacency[0, :, 2].any())
        self.assertEqual(int(graph.adjacency.sum()), 4)

    def test_knowledge_builder_omits_edges_for_unreceived_neighbors(self) -> None:
        builder = ConflictGraphBuilder(
            GraphBuildConfig(
                communication_radius=100.0,
                risk_distance=20.0,
                prediction_horizon=5.0,
                top_k_neighbors=1,
                current_distance_edges=True,
                predicted_conflict_edges=True,
                self_loops=True,
            )
        )
        graph = builder.build_from_knowledge(
            positions=torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]),
            received_positions=torch.zeros((1, 2, 2, 3)),
            received_velocities=torch.zeros((1, 2, 2, 3)),
            goals=torch.tensor([[[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]),
            active_mask=torch.tensor([[True, True]]),
            knowledge_valid=torch.tensor([[[True, False], [False, True]]]),
            knowledge_ages=torch.tensor([[[0, -1], [-1, 0]]]),
        )

        self.assertTrue(graph.adjacency[0, 0, 0])
        self.assertTrue(graph.adjacency[0, 1, 1])
        self.assertFalse(graph.adjacency[0, 0, 1])
        self.assertFalse(graph.adjacency[0, 1, 0])

    def test_knowledge_graph_uses_predicted_position_and_uncertainty_risk(self) -> None:
        builder = ConflictGraphBuilder(
            GraphBuildConfig(
                communication_radius=100.0,
                risk_distance=5.0,
                prediction_horizon=5.0,
                top_k_neighbors=0,
                current_distance_edges=False,
                predicted_conflict_edges=True,
                self_loops=True,
                uncertainty_scale=10.0,
                uncertainty_risk_gain=1.0,
            )
        )
        predicted_positions = torch.tensor(
            [[[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]]
        )
        graph = builder.build_from_knowledge(
            positions=torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]),
            received_positions=torch.zeros((1, 2, 2, 3)),
            predicted_positions=predicted_positions,
            received_velocities=torch.tensor(
                [[[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
            ),
            goals=torch.tensor([[[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]),
            active_mask=torch.tensor([[True, True]]),
            knowledge_valid=torch.tensor([[[True, True], [False, True]]]),
            knowledge_ages=torch.tensor([[[0, 2], [-1, 0]]]),
            knowledge_uncertainty=torch.tensor([[[0.0, 3.0], [0.0, 0.0]]]),
        )

        self.assertEqual(graph.edge_features.shape[-1], 16)
        self.assertAlmostEqual(float(graph.edge_features[0, 0, 1, -1]), 0.3)
        self.assertTrue(graph.adjacency[0, 0, 1])

    def test_graph_runner_uses_environment_knowledge_when_communication_is_delayed(self) -> None:
        environment = MultiUAVParallelEnv(
            make_graph_scenario(num_uavs=3),
            EnvironmentConfig(
                max_steps=20,
                max_horizontal_speed=8.0,
                max_vertical_speed=6.0,
                goal_radius=5.0,
                collision_distance=6.0,
                severe_clearance_shortfall=8.0,
                severe_threat_penetration=5.0,
                max_neighbors=2,
                communication_enabled=True,
                communication_range=float("inf"),
                communication_delay_steps=1,
                communication_max_staleness_steps=2,
            ),
        )
        environment.reset(seed=3)
        builder = ConflictGraphBuilder(
            GraphBuildConfig(
                communication_radius=100.0,
                risk_distance=20.0,
                prediction_horizon=5.0,
                top_k_neighbors=2,
                current_distance_edges=True,
                predicted_conflict_edges=True,
                self_loops=True,
            )
        )

        graph = build_graph_from_environments(builder, [environment], device=torch.device("cpu"))

        self.assertTrue(torch.equal(graph.adjacency[0], torch.eye(3, dtype=torch.bool)))

    def test_graph_edges_ignore_current_peer_activity_not_in_delivered_knowledge(self) -> None:
        builder = ConflictGraphBuilder(
            GraphBuildConfig(
                communication_radius=100.0,
                risk_distance=20.0,
                prediction_horizon=5.0,
                top_k_neighbors=1,
                current_distance_edges=True,
                predicted_conflict_edges=True,
                self_loops=True,
            )
        )
        positions = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
        received_positions = torch.tensor(
            [[[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]]
        )
        common = {
            "positions": positions,
            "received_positions": received_positions,
            "received_velocities": torch.zeros_like(received_positions),
            "goals": torch.zeros_like(positions),
            "knowledge_valid": torch.ones(1, 2, 2, dtype=torch.bool),
            "knowledge_ages": torch.zeros(1, 2, 2, dtype=torch.int64),
        }

        active_peer_graph = builder.build_from_knowledge(
            active_mask=torch.tensor([[True, True]]), **common
        )
        stopped_peer_graph = builder.build_from_knowledge(
            active_mask=torch.tensor([[True, False]]), **common
        )

        self.assertTrue(
            torch.equal(active_peer_graph.adjacency[0, 0], stopped_peer_graph.adjacency[0, 0])
        )

    def test_graph_attention_preserves_shapes_and_handles_empty_neighbourhoods(self) -> None:
        torch.manual_seed(4)
        encoder = DenseGraphAttentionEncoder(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
        )
        node_features = torch.randn(2, 3, 5)
        edge_features = torch.randn(2, 3, 3, 15)
        adjacency = torch.zeros(2, 3, 3, dtype=torch.bool)
        node_mask = torch.tensor([[True, True, False], [True, False, True]])

        embeddings = encoder(node_features, edge_features, adjacency, node_mask)

        self.assertEqual(embeddings.shape, (2, 3, 16))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(
            torch.equal(embeddings[~node_mask], torch.zeros_like(embeddings[~node_mask]))
        )

    def test_graph_actor_action_ignores_unreceived_peer_node_features(self) -> None:
        torch.manual_seed(11)
        actor = GraphActor(
            node_feature_dim=5,
            edge_feature_dim=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            action_dim=3,
        )
        node_features = torch.tensor([[[1.0, 0.5, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 1.0]]])
        changed_peer_features = node_features.clone()
        changed_peer_features[0, 1] = torch.tensor([-9.0, 8.0, 7.0, -6.0, 5.0])
        edge_features = torch.zeros(1, 2, 2, 16)
        adjacency = torch.tensor([[[True, True], [False, True]]])
        node_mask = torch.tensor([[True, True]])

        original_actions, _, _ = actor.sample(
            node_features, edge_features, adjacency, node_mask, deterministic=True
        )
        changed_actions, _, _ = actor.sample(
            changed_peer_features, edge_features, adjacency, node_mask, deterministic=True
        )

        assert_allclose(
            original_actions[0, 0].detach().numpy(), changed_actions[0, 0].detach().numpy()
        )

    def test_graph_actor_action_ignores_current_peer_activity(self) -> None:
        torch.manual_seed(23)
        actor = GraphActor(
            node_feature_dim=5,
            edge_feature_dim=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            action_dim=3,
        )
        node_features = torch.tensor([[[1.0, 0.5, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 1.0]]])
        edge_features = torch.zeros(1, 2, 2, 16)
        edge_features[0, 0, 1, 0] = 1.0
        adjacency = torch.tensor([[[True, True], [False, True]]])

        active_actions, _, _ = actor.sample(
            node_features,
            edge_features,
            adjacency,
            torch.tensor([[True, True]]),
            deterministic=True,
        )
        stopped_peer_actions, _, _ = actor.sample(
            node_features,
            edge_features,
            adjacency,
            torch.tensor([[True, False]]),
            deterministic=True,
        )

        assert_allclose(
            active_actions[0, 0].detach().numpy(), stopped_peer_actions[0, 0].detach().numpy()
        )

    def test_graph_attention_is_permutation_equivariant(self) -> None:
        torch.manual_seed(19)
        encoder = DenseGraphAttentionEncoder(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
        ).eval()
        node_features = torch.randn(1, 4, 5)
        edge_features = torch.randn(1, 4, 4, 15)
        adjacency = torch.tensor(
            [
                [
                    [True, True, False, False],
                    [True, True, True, False],
                    [False, True, True, True],
                    [False, False, True, True],
                ]
            ]
        )
        node_mask = torch.ones(1, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])

        original = encoder(node_features, edge_features, adjacency, node_mask)
        permuted = encoder(
            node_features[:, permutation],
            edge_features[:, permutation][:, :, permutation],
            adjacency[:, permutation][:, :, permutation],
            node_mask[:, permutation],
        )

        assert_allclose(
            permuted.detach().numpy(), original[:, permutation].detach().numpy(), atol=1e-6
        )

    def test_graph_policy_and_auxiliary_heads_have_masked_shapes(self) -> None:
        torch.manual_seed(23)
        actor = GraphActor(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            action_dim=3,
        )
        critic = GraphCentralizedCritic(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
        )
        auxiliary = ConflictPredictionHead(embedding_dim=16, edge_feature_dim=15)
        node_features = torch.randn(2, 3, 5)
        edge_features = torch.randn(2, 3, 3, 15)
        adjacency = torch.eye(3, dtype=torch.bool).repeat(2, 1, 1)
        node_mask = torch.tensor([[True, True, False], [True, False, True]])

        actions, log_probabilities, entropy = actor.sample(
            node_features, edge_features, adjacency, node_mask, deterministic=False
        )
        evaluated_log_probabilities, evaluated_entropy = actor.evaluate_actions(
            node_features, edge_features, adjacency, node_mask, actions
        )
        embeddings = actor.encode(node_features, edge_features, adjacency, node_mask)
        conflict_logits, predicted_distances = auxiliary(embeddings, edge_features)
        values = critic(node_features, edge_features, adjacency, node_mask)

        self.assertEqual(actions.shape, (2, 3, 3))
        self.assertEqual(log_probabilities.shape, (2, 3))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(conflict_logits.shape, (2, 3, 3))
        self.assertEqual(predicted_distances.shape, (2, 3, 3))
        self.assertTrue(torch.all(predicted_distances >= 0.0))
        self.assertTrue(torch.equal(actions[~node_mask], torch.zeros_like(actions[~node_mask])))
        assert_allclose(
            log_probabilities.detach().numpy(),
            evaluated_log_probabilities.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        assert_allclose(entropy.detach().numpy(), evaluated_entropy.detach().numpy())

    def test_future_targets_exclude_reset_boundaries(self) -> None:
        positions = torch.tensor(
            [
                [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            ]
        )
        active_mask = torch.ones(3, 1, 2, dtype=torch.bool)
        episode_ids = torch.tensor([[0], [0], [1]])

        conflict, minimum_distance, target_mask = build_future_conflict_targets(
            positions=positions,
            active_mask=active_mask,
            episode_ids=episode_ids,
            horizon=2,
            conflict_distance=3.0,
        )

        self.assertTrue(target_mask[0, 0, 0, 1])
        self.assertFalse(conflict[0, 0, 0, 1])
        self.assertAlmostEqual(minimum_distance[0, 0, 0, 1].item(), 8.0)
        self.assertFalse(target_mask[1, 0].any())

    def test_graph_rollout_buffer_flattens_whole_graph_samples(self) -> None:
        buffer = GraphRolloutBuffer(
            rollout_length=2,
            num_envs=1,
            num_agents=2,
            node_feature_dim=5,
            edge_feature_dim=15,
            action_dim=3,
            device=torch.device("cpu"),
        )
        for step in range(2):
            buffer.add(
                node_features=torch.full((1, 2, 5), float(step)),
                edge_features=torch.full((1, 2, 2, 15), float(step)),
                adjacency=torch.eye(2, dtype=torch.bool).unsqueeze(0),
                active_mask=torch.ones(1, 2, dtype=torch.bool),
                positions=torch.tensor([[[float(step), 0.0, 0.0], [10.0, 0.0, 0.0]]]),
                episode_ids=torch.tensor([0]),
                actions=torch.zeros(1, 2, 3),
                log_probabilities=torch.zeros(1, 2),
                rewards=torch.ones(1, 2),
                terminated=torch.zeros(1, 2, dtype=torch.bool),
                truncated=torch.zeros(1, 2, dtype=torch.bool),
                values=torch.zeros(1, 2),
                next_values=torch.zeros(1, 2),
            )
        buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        batch = buffer.flatten(horizon=1, conflict_distance=3.0)

        self.assertEqual(batch.node_features.shape, (2, 2, 5))
        self.assertEqual(batch.edge_features.shape, (2, 2, 2, 15))
        self.assertEqual(batch.actions.shape, (2, 2, 3))
        self.assertEqual(batch.auxiliary_mask.shape, (2, 2, 2))
        self.assertTrue(torch.isfinite(batch.returns).all())

    def test_graph_mappo_combines_finite_ppo_and_auxiliary_losses(self) -> None:
        torch.manual_seed(31)
        actor = GraphActor(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            action_dim=3,
        )
        critic = GraphCentralizedCritic(
            node_feature_dim=5,
            edge_feature_dim=15,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
        )
        auxiliary = ConflictPredictionHead(embedding_dim=16, edge_feature_dim=15)
        trainer = GraphMAPPOTrainer(
            actor,
            critic,
            auxiliary,
            GraphMAPPOConfig(
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_ratio=0.2,
                entropy_coef=0.01,
                value_coef=0.5,
                max_grad_norm=0.5,
                batch_size=2,
                ppo_epochs=1,
                aux_conflict_coef=0.2,
                aux_distance_coef=0.1,
            ),
            device=torch.device("cpu"),
        )
        batch = GraphRolloutBatch(
            node_features=torch.randn(2, 2, 5),
            edge_features=torch.randn(2, 2, 2, 15),
            adjacency=torch.eye(2, dtype=torch.bool).repeat(2, 1, 1),
            active_mask=torch.ones(2, 2, dtype=torch.bool),
            actions=torch.zeros(2, 2, 3),
            old_log_probabilities=torch.zeros(2, 2),
            old_values=torch.zeros(2, 2),
            returns=torch.ones(2, 2),
            advantages=torch.tensor([[1.0, -1.0], [0.5, -0.5]]),
            auxiliary_conflict=torch.tensor(
                [[[False, True], [True, False]], [[False, False], [False, False]]]
            ),
            auxiliary_minimum_distance=torch.ones(2, 2, 2),
            auxiliary_mask=~torch.eye(2, dtype=torch.bool).repeat(2, 1, 1),
        )

        metrics = trainer.update(batch)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "graph_mappo.pt"
            save_graph_checkpoint(checkpoint, trainer, step=9)
            restored = GraphMAPPOTrainer(
                GraphActor(
                    node_feature_dim=5,
                    edge_feature_dim=15,
                    embedding_dim=16,
                    num_heads=4,
                    num_layers=1,
                    action_dim=3,
                ),
                GraphCentralizedCritic(
                    node_feature_dim=5,
                    edge_feature_dim=15,
                    embedding_dim=16,
                    num_heads=4,
                    num_layers=1,
                ),
                ConflictPredictionHead(embedding_dim=16, edge_feature_dim=15),
                trainer.config,
                device=torch.device("cpu"),
            )
            step = load_graph_checkpoint(checkpoint, restored, map_location=torch.device("cpu"))

        self.assertEqual(step, 9)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for map-location coverage")
    def test_cuda_graph_checkpoint_load_keeps_rng_state_on_cpu(self) -> None:
        """CUDA graph checkpoint loading must restore CPU RNG state from CPU tensors."""
        config = GraphMAPPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=2,
            ppo_epochs=1,
            aux_conflict_coef=0.2,
            aux_distance_coef=0.1,
        )
        source = GraphMAPPOTrainer(
            GraphActor(
                node_feature_dim=5,
                edge_feature_dim=15,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                action_dim=3,
            ),
            GraphCentralizedCritic(
                node_feature_dim=5,
                edge_feature_dim=15,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
            ),
            ConflictPredictionHead(embedding_dim=16, edge_feature_dim=15),
            config,
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cpu_graph_checkpoint.pt"
            save_graph_checkpoint(checkpoint, source, step=7)
            restored = GraphMAPPOTrainer(
                GraphActor(
                    node_feature_dim=5,
                    edge_feature_dim=15,
                    embedding_dim=16,
                    num_heads=4,
                    num_layers=1,
                    action_dim=3,
                ),
                GraphCentralizedCritic(
                    node_feature_dim=5,
                    edge_feature_dim=15,
                    embedding_dim=16,
                    num_heads=4,
                    num_layers=1,
                ),
                ConflictPredictionHead(embedding_dim=16, edge_feature_dim=15),
                config,
                device=torch.device("cuda"),
            )
            step = load_graph_checkpoint(checkpoint, restored, map_location=torch.device("cuda"))

        self.assertEqual(step, 7)

    def test_variable_uav_graph_runner_collects_and_evaluates_deterministically(self) -> None:
        self.assertEqual(len(make_graph_scenario(num_uavs=3).missions), 3)
        self.assertEqual(len(make_graph_scenario(num_uavs=5).missions), 5)
        self.assertEqual(len(make_graph_scenario(num_uavs=8).missions), 8)
        config = GraphExperimentConfig(
            seed=41,
            num_envs=1,
            num_uavs=3,
            rollout_length=4,
            total_steps=24,
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
            evaluation_interval=24,
            checkpoint_interval=24,
            graph_mode="predictive_graph",
            communication_radius=20.0,
            risk_distance=12.0,
            prediction_horizon=5.0,
            top_k_neighbors=2,
            auxiliary_horizon=2,
            aux_conflict_coef=0.1,
            aux_distance_coef=0.1,
            obstacle=False,
        )
        experiment = GraphMAPPOExperiment(config, device=torch.device("cpu"))

        buffer, _ = experiment.collect_rollout()
        update = experiment.trainer.update(
            buffer.flatten(horizon=config.auxiliary_horizon, conflict_distance=config.risk_distance)
        )
        first_evaluation = experiment.evaluate(episodes=2)
        second_evaluation = experiment.evaluate(episodes=2)

        self.assertEqual(buffer.actions.shape, (4, 1, 3, 3))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in update.values()))
        self.assertEqual(first_evaluation, second_evaluation)

    def test_graph_yaml_and_cli_expose_required_comparison_controls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load((root / "configs/rl/graph_mappo.yaml").read_text(encoding="utf-8"))
        config = load_graph_experiment_config(root / "configs/rl/graph_mappo.yaml")

        self.assertEqual(config.graph_mode, "predictive_graph")
        self.assertTrue(
            {
                "num_uavs",
                "graph_mode",
                "communication_radius",
                "risk_distance",
                "prediction_horizon",
                "top_k_neighbors",
                "auxiliary_horizon",
                "aux_conflict_coef",
                "aux_distance_coef",
            }.issubset(payload)
        )
        self.assertEqual(
            build_graph_training_parser()
            .parse_args(["--num-uavs", "5", "--graph-mode", "distance_graph"])
            .num_uavs,
            5,
        )
        self.assertEqual(
            build_graph_evaluation_parser()
            .parse_args(["--checkpoint", "checkpoint.pt", "--num-uavs", "8"])
            .num_uavs,
            8,
        )

    def test_comparison_matrix_covers_three_graph_modes_and_uav_counts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        comparison = load_comparison_config(root / "configs/experiments/graph_comparison.yaml")

        self.assertEqual(comparison.num_uavs, (3, 5, 8))
        self.assertEqual(comparison.graph_modes, ("mappo", "distance_graph", "predictive_graph"))


if __name__ == "__main__":
    unittest.main()

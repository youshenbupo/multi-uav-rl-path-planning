"""Numerical and persistence tests for the Phase-9 basic MAPPO baseline."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

from multiuav.learning.gae import compute_gae
from multiuav.learning.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    RunningMeanStd,
    clipped_policy_loss,
    load_checkpoint,
    save_checkpoint,
)
from multiuav.learning.networks import CentralizedCritic, SharedGaussianActor
from multiuav.learning.rollout_buffer import RolloutBuffer
from multiuav.learning.runner import MAPPOExperiment, MAPPOExperimentConfig
from scripts.evaluate_mappo import build_parser as build_evaluation_parser
from scripts.train_mappo import build_parser as build_training_parser


class MAPPOTests(unittest.TestCase):
    """Cover the required GAE, PPO, buffer, action, checkpoint, and seed contracts."""

    def test_gaussian_actor_log_probability_and_deterministic_action_contract(self) -> None:
        torch.manual_seed(7)
        actor = SharedGaussianActor(observation_dim=5, action_dim=3, hidden_dims=(16, 16))
        observations = torch.zeros(4, 5)
        distribution = actor.distribution(observations)
        actions, log_probability, entropy = actor.sample(observations, deterministic=False)
        evaluated_log_probability, evaluated_entropy = actor.evaluate_actions(observations, actions)

        self.assertEqual(actions.shape, (4, 3))
        self.assertEqual(log_probability.shape, entropy.shape)
        self.assertEqual(log_probability.shape, (4,))
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))
        assert_allclose(
            log_probability.detach().numpy(),
            evaluated_log_probability.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        assert_allclose(entropy.detach().numpy(), evaluated_entropy.detach().numpy())
        assert_allclose(
            actor.act(observations, deterministic=True).detach().numpy(),
            distribution.mean.tanh().detach().numpy(),
        )

    def test_gae_stops_bootstrap_at_true_terminal(self) -> None:
        advantages, returns = compute_gae(
            rewards=torch.tensor([[[1.0]], [[2.0]]]),
            values=torch.tensor([[[0.5]], [[0.4]]]),
            next_values=torch.tensor([[[0.4]], [[0.0]]]),
            terminated=torch.tensor([[[False]], [[True]]]),
            gamma=0.9,
            gae_lambda=1.0,
        )

        assert_allclose(advantages.squeeze().numpy(), [2.30, 1.6], rtol=1e-6, atol=1e-6)
        assert_allclose(returns.squeeze().numpy(), [2.80, 2.0], rtol=1e-6, atol=1e-6)

    def test_rollout_buffer_preserves_time_environment_agent_shapes(self) -> None:
        buffer = RolloutBuffer(
            rollout_length=2,
            num_envs=1,
            num_agents=2,
            observation_dim=3,
            state_dim=4,
            action_dim=3,
            device=torch.device("cpu"),
        )
        for step in range(2):
            buffer.add(
                observations=torch.full((1, 2, 3), float(step)),
                states=torch.full((1, 2, 4), float(step)),
                actions=torch.zeros(1, 2, 3),
                log_probabilities=torch.zeros(1, 2),
                rewards=torch.ones(1, 2),
                terminated=torch.zeros(1, 2, dtype=torch.bool),
                truncated=torch.zeros(1, 2, dtype=torch.bool),
                values=torch.zeros(1, 2),
                next_values=torch.zeros(1, 2),
            )
        buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        batch = buffer.flatten()

        self.assertEqual(batch.observations.shape, (4, 3))
        self.assertEqual(batch.states.shape, (4, 4))
        self.assertEqual(batch.actions.shape, (4, 3))
        self.assertEqual(batch.advantages.shape, (4,))
        self.assertTrue(torch.isfinite(batch.returns).all())

    def test_clipped_policy_loss_and_observation_normalizer_are_numerical(self) -> None:
        loss, ratio, clip_fraction = clipped_policy_loss(
            new_log_probabilities=torch.log(torch.tensor([1.4, 0.7])),
            old_log_probabilities=torch.zeros(2),
            advantages=torch.ones(2),
            clip_ratio=0.2,
        )
        normalizer = RunningMeanStd(shape=(2,))
        normalizer.update(torch.tensor([[1.0, 3.0], [3.0, 5.0]]))

        assert_allclose(ratio.numpy(), [1.4, 0.7], rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(loss.item(), -0.95, places=6)
        self.assertAlmostEqual(clip_fraction.item(), 1.0, places=6)
        assert_allclose(normalizer.mean.numpy(), [2.0, 4.0], rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(normalizer.normalize(torch.tensor([[2.0, 4.0]]))).all())

    def test_checkpoint_round_trip_and_seeded_actions_are_reproducible(self) -> None:
        config = MAPPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=4,
            ppo_epochs=1,
            normalize_rewards=False,
        )
        first = MAPPOTrainer(
            SharedGaussianActor(5, 3, (16, 16)),
            CentralizedCritic(4, (16, 16)),
            config,
            device=torch.device("cpu"),
        )
        first.observation_normalizer.update(torch.ones(2, 5))
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "mappo.pt"
            save_checkpoint(checkpoint, first, step=12)
            second = MAPPOTrainer(
                SharedGaussianActor(5, 3, (16, 16)),
                CentralizedCritic(4, (16, 16)),
                config,
                device=torch.device("cpu"),
            )
            step = load_checkpoint(checkpoint, second, map_location=torch.device("cpu"))

        self.assertEqual(step, 12)
        for first_parameter, second_parameter in zip(
            first.actor.parameters(), second.actor.parameters(), strict=True
        ):
            assert_allclose(first_parameter.detach().numpy(), second_parameter.detach().numpy())
        torch.manual_seed(99)
        first_action, _, _ = first.actor.sample(torch.zeros(1, 5), deterministic=False)
        torch.manual_seed(99)
        second_action, _, _ = second.actor.sample(torch.zeros(1, 5), deterministic=False)
        assert_allclose(first_action.detach().numpy(), second_action.detach().numpy())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for map-location coverage")
    def test_cuda_checkpoint_load_keeps_rng_state_on_cpu(self) -> None:
        """CUDA weight loading must not place the CPU RNG-state tensor on the GPU."""
        config = MAPPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=4,
            ppo_epochs=1,
            normalize_rewards=False,
        )
        source = MAPPOTrainer(
            SharedGaussianActor(5, 3, (16, 16)),
            CentralizedCritic(4, (16, 16)),
            config,
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cpu_checkpoint.pt"
            save_checkpoint(checkpoint, source, step=7)
            restored = MAPPOTrainer(
                SharedGaussianActor(5, 3, (16, 16)),
                CentralizedCritic(4, (16, 16)),
                config,
                device=torch.device("cuda"),
            )
            step = load_checkpoint(checkpoint, restored, map_location=torch.device("cuda"))

        self.assertEqual(step, 7)

    def test_runner_collects_parallel_rollout_and_evaluates_deterministically(self) -> None:
        config = MAPPOExperimentConfig(
            seed=17,
            num_envs=2,
            num_uavs=2,
            rollout_length=4,
            total_steps=16,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=8,
            ppo_epochs=1,
            hidden_dims=(16, 16),
            evaluation_interval=16,
            checkpoint_interval=16,
            normalize_rewards=False,
            obstacle=False,
        )
        experiment = MAPPOExperiment(config, device=torch.device("cpu"))

        buffer, rollout_metrics = experiment.collect_rollout()
        first_evaluation = experiment.evaluate(episodes=2)
        second_evaluation = experiment.evaluate(episodes=2)

        self.assertEqual(buffer.rewards.shape, (4, 2, 2))
        self.assertTrue(torch.isfinite(buffer.rewards).all())
        self.assertTrue(torch.isfinite(buffer.log_probabilities).all())
        self.assertIn("episode_return", rollout_metrics)
        self.assertIn("success_rate", first_evaluation)
        self.assertEqual(first_evaluation, second_evaluation)

    def test_update_uses_the_same_normalized_observations_as_action_sampling(self) -> None:
        """A changing observation normalizer must not make the first PPO ratio overflow."""
        config = MAPPOExperimentConfig(
            seed=20260718,
            num_envs=2,
            num_uavs=2,
            rollout_length=64,
            total_steps=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=128,
            ppo_epochs=1,
            hidden_dims=(16, 16),
            evaluation_interval=256,
            checkpoint_interval=256,
            normalize_rewards=False,
            obstacle=False,
        )
        experiment = MAPPOExperiment(config, device=torch.device("cpu"))

        buffer, _ = experiment.collect_rollout()
        metrics = experiment.trainer.update(buffer.flatten())

        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for parameter in experiment.trainer.actor.parameters()
            )
        )

    def test_two_baseline_updates_remain_finite_with_observation_normalization(self) -> None:
        """Statistics updates must not destabilize a later on-policy rollout."""
        config = MAPPOExperimentConfig(
            seed=20260718,
            num_envs=4,
            num_uavs=2,
            rollout_length=64,
            total_steps=1024,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=256,
            ppo_epochs=4,
            hidden_dims=(64, 64),
            evaluation_interval=1024,
            checkpoint_interval=1024,
            normalize_rewards=False,
            obstacle=False,
        )
        experiment = MAPPOExperiment(config, device=torch.device("cpu"))

        records = experiment.train()

        self.assertEqual(len(records), 2)
        self.assertTrue(
            all(math.isfinite(value) for record in records for value in record.values())
        )
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for parameter in experiment.trainer.actor.parameters()
            )
        )

    def test_cli_parsers_and_yaml_expose_all_required_baseline_settings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs/rl/mappo_baseline.yaml").read_text(encoding="utf-8")
        )
        required = {
            "seed",
            "num_envs",
            "num_uavs",
            "rollout_length",
            "total_steps",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_ratio",
            "entropy_coef",
            "value_coef",
            "max_grad_norm",
            "batch_size",
            "ppo_epochs",
            "hidden_dims",
            "evaluation_interval",
            "checkpoint_interval",
        }

        self.assertTrue(required.issubset(config))
        self.assertIsNotNone(build_training_parser().parse_args([]))
        self.assertIsNotNone(build_evaluation_parser().parse_args([]))


if __name__ == "__main__":
    unittest.main()

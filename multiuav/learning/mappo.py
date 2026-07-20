"""PPO optimization, normalization, and checkpoint persistence for basic MAPPO."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn, optim

from multiuav.learning.networks import CentralizedCritic, SharedGaussianActor
from multiuav.learning.rollout_buffer import RolloutBatch


@dataclass(frozen=True)
class MAPPOConfig:
    """All optimizer settings consumed by the basic MAPPO trainer."""

    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_ratio: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    batch_size: int
    ppo_epochs: int
    normalize_rewards: bool


class RunningMeanStd:
    """Serializable running statistics used for observation or reward normalization."""

    def __init__(self, shape: tuple[int, ...], *, epsilon: float = 0.0) -> None:
        self.mean = torch.zeros(shape)
        self.variance = torch.ones(shape)
        self.count = torch.tensor(float(epsilon))

    def update(self, values: Tensor) -> None:
        """Merge a batch using numerically stable parallel mean/variance updates."""
        batch = values.detach().to(dtype=torch.float32, device=torch.device("cpu"))
        if batch.ndim < 1 or tuple(batch.shape[1:]) != tuple(self.mean.shape):
            raise ValueError("RunningMeanStd batch has an incompatible feature shape.")
        batch_mean = batch.mean(dim=0)
        batch_variance = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(batch.shape[0]))
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        mean = self.mean + delta * batch_count / total_count
        aggregate = self.variance * self.count + batch_variance * batch_count
        aggregate += delta.square() * self.count * batch_count / total_count
        self.mean = mean
        self.variance = aggregate / total_count
        self.count = total_count

    def normalize(self, values: Tensor) -> Tensor:
        """Normalize values while retaining their caller-selected device."""
        mean = self.mean.to(values.device)
        variance = self.variance.to(values.device)
        return (values - mean) / torch.sqrt(variance + 1e-8)

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean, "variance": self.variance, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.mean = state["mean"].detach().cpu()
        self.variance = state["variance"].detach().cpu()
        self.count = state["count"].detach().cpu()


def clipped_policy_loss(
    *,
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    clip_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return PPO clipped surrogate loss, probability ratio, and clip fraction."""
    ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    clip_fraction = (torch.abs(ratio - 1.0) > clip_ratio).float().mean()
    return loss, ratio, clip_fraction


class MAPPOTrainer:
    """Own actor/critic optimization and normalization for the shared-policy baseline."""

    def __init__(
        self,
        actor: SharedGaussianActor,
        critic: CentralizedCritic,
        config: MAPPOConfig,
        *,
        device: torch.device,
    ) -> None:
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()], lr=config.learning_rate
        )
        self.observation_normalizer = RunningMeanStd((actor.observation_dim,))
        self.reward_normalizer = RunningMeanStd(())

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Run shuffled PPO minibatches and return finite aggregate training diagnostics."""
        advantages = batch.advantages.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        observations = batch.observations.to(self.device)
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probabilities = batch.old_log_probabilities.to(self.device)
        old_values = batch.old_values.to(self.device)
        returns = batch.returns.to(self.device)
        sample_count = len(observations)
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        for _ in range(self.config.ppo_epochs):
            order = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                new_log_probabilities, entropy = self.actor.evaluate_actions(
                    observations[index], actions[index]
                )
                policy_loss, ratio, clip_fraction = clipped_policy_loss(
                    new_log_probabilities=new_log_probabilities,
                    old_log_probabilities=old_log_probabilities[index],
                    advantages=advantages[index],
                    clip_ratio=self.config.clip_ratio,
                )
                values = self.critic(states[index])
                clipped_values = old_values[index] + (values - old_values[index]).clamp(
                    -self.config.clip_ratio, self.config.clip_ratio
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (values - returns[index]).square(),
                        (clipped_values - returns[index]).square(),
                    ).mean()
                )
                entropy_mean = entropy.mean()
                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    [*self.actor.parameters(), *self.critic.parameters()], self.config.max_grad_norm
                )
                self.optimizer.step()
                metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
                metrics["value_loss"].append(float(value_loss.detach().cpu()))
                metrics["entropy"].append(float(entropy_mean.detach().cpu()))
                metrics["approx_kl"].append(
                    float(
                        (old_log_probabilities[index] - new_log_probabilities).mean().detach().cpu()
                    )
                )
                metrics["clip_fraction"].append(float(clip_fraction.detach().cpu()))
        with torch.no_grad():
            predicted = self.critic(states)
            variance = torch.var(returns, unbiased=False)
            explained_variance = 1.0 - torch.var(returns - predicted, unbiased=False) / (
                variance + 1e-8
            )
        return {
            **{name: float(np.mean(values)) for name, values in metrics.items()},
            "explained_variance": float(explained_variance.detach().cpu()),
        }


def save_checkpoint(path: Path, trainer: MAPPOTrainer, *, step: int) -> None:
    """Persist models, optimizer, normalizers, configuration, step, and RNG state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": trainer.actor.state_dict(),
            "critic": trainer.critic.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "observation_normalizer": trainer.observation_normalizer.state_dict(),
            "reward_normalizer": trainer.reward_normalizer.state_dict(),
            "config": asdict(trainer.config),
            "step": step,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_checkpoint(path: Path, trainer: MAPPOTrainer, *, map_location: torch.device) -> int:
    """Restore an exact trainer state and return its completed environment-step count."""
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    trainer.actor.load_state_dict(payload["actor"])
    trainer.critic.load_state_dict(payload["critic"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.observation_normalizer.load_state_dict(payload["observation_normalizer"])
    trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if payload["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    return int(payload["step"])

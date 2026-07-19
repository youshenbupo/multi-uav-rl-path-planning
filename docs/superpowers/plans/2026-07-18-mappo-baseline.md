# Basic MAPPO Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a parameter-sharing Gaussian MAPPO baseline with centralized critic on the Phase-8 multi-UAV environment.

**Architecture:** Pure PyTorch MLP actor/critic, a tensor rollout buffer, functional GAE, and a runner that converts PettingZoo dictionaries to `[T,E,N]` batches. The actor consumes only local observations; the critic consumes the centralized state. All logging and checkpoint state are owned by the runner/trainer boundary.

**Tech Stack:** Python 3.11, PyTorch 2.13, NumPy, PettingZoo, Gymnasium, TensorBoard, PyYAML, unittest.

---

## File responsibilities

- Create `configs/rl/mappo_baseline.yaml`: training, model, normalization, checkpoint, and evaluation settings.
- Create `multiuav/learning/networks.py`: shared Gaussian actor and centralized critic.
- Create `multiuav/learning/gae.py`: terminal-aware GAE.
- Create `multiuav/learning/rollout_buffer.py`: fixed `[T,E,N]` storage and flattening.
- Create `multiuav/learning/mappo.py`: configuration, running normalization, PPO update, checkpoint functions.
- Create `multiuav/learning/runner.py`: scenarios, PettingZoo collection, logging, evaluation, and training.
- Create `scripts/train_mappo.py` and `scripts/evaluate_mappo.py`: explicit-path CLIs.
- Create `tests/test_mappo_baseline.py`: Phase-9 unit and smoke checks.

### Task 1: Network and Gaussian action contract

**Files:**
- Create: `multiuav/learning/networks.py`
- Test: `tests/test_mappo_baseline.py`

- [ ] Write failing tests for shape, summed log probability, entropy, and deterministic mean action.

```python
actor = SharedGaussianActor(observation_dim=5, action_dim=3, hidden_dims=(16, 16))
distribution = actor.distribution(torch.zeros(4, 5))
actions, log_prob, entropy = actor.sample(torch.zeros(4, 5), deterministic=False)
assert actions.shape == (4, 3)
assert log_prob.shape == entropy.shape == (4,)
assert torch.allclose(actor.act(obs, deterministic=True), distribution.mean.clamp(-1, 1))
```

- [ ] Run `python -m unittest tests.test_mappo_baseline.MAPPOTests.test_gaussian_actor_log_probability_contract -v`; expect missing-module failure.
- [ ] Implement an MLP helper, `SharedGaussianActor`, and `CentralizedCritic`. `distribution()` returns `Normal(mean, exp(log_std).expand_as(mean))`; `evaluate_actions()` returns `log_prob.sum(-1)` and `entropy.sum(-1)`.
- [ ] Re-run the actor test; expect pass.

### Task 2: GAE and rollout buffer

**Files:**
- Create: `multiuav/learning/gae.py`
- Create: `multiuav/learning/rollout_buffer.py`
- Test: `tests/test_mappo_baseline.py`

- [ ] Write a failing numerical GAE test with terminal and truncation masks.

```python
advantages, returns = compute_gae(
    rewards=torch.tensor([[[1.0]], [[2.0]]]),
    values=torch.tensor([[[0.5]], [[0.4]]]),
    next_values=torch.tensor([[[0.4]], [[0.0]]]),
    terminated=torch.tensor([[[False]], [[True]]]),
    gamma=0.9, gae_lambda=1.0,
)
assert_allclose(advantages.squeeze().numpy(), [2.30, 1.6])
```

- [ ] Run the GAE test; expect missing-module failure.
- [ ] Implement reverse-time GAE: `delta = r + gamma * (1-terminated) * next_value - value`; no terminal bootstrap, but time-limit truncation remains bootstrapped.
- [ ] Write a failing buffer-shape test that stores two `[E,N]` transitions and asserts a flattened batch `[T*E*N,...]`.
- [ ] Implement `RolloutBuffer.add`, `compute_returns_and_advantages`, `flatten`, and `clear`, with shape validation for observations, centralized state, actions, log probabilities, rewards, terminations, truncations, values, and next values.
- [ ] Re-run GAE and buffer tests; expect pass.

### Task 3: PPO loss, normalizers, and checkpoint

**Files:**
- Create: `multiuav/learning/mappo.py`
- Test: `tests/test_mappo_baseline.py`

- [ ] Write failing tests for exact PPO ratio/clipped surrogate, clipped critic loss, running observation normalization, and save/load round trip.

```python
loss, ratio, clip_fraction = clipped_policy_loss(
    new_log_prob=torch.log(torch.tensor([1.4, 0.7])),
    old_log_prob=torch.zeros(2),
    advantages=torch.ones(2),
    clip_ratio=0.2,
)
assert_allclose(ratio.numpy(), [1.4, 0.7])
assert_allclose(loss.item(), -0.95)
assert_allclose(clip_fraction.item(), 1.0)
```

- [ ] Run the selected tests; expect missing-module failure.
- [ ] Implement `RunningMeanStd`, `MAPPOConfig`, `MAPPOTrainer.update`, and checkpoint `save_checkpoint`/`load_checkpoint`. Trainer uses Adam, advantage normalization, policy ratio `exp(new-old)`, clipped value loss around old value, entropy coefficient, and `clip_grad_norm_`.
- [ ] Checkpoint payload includes actor, critic, optimizer, normalizers, step, config, Python/NumPy/PyTorch CPU/CUDA RNG states.
- [ ] Re-run PPO and checkpoint tests; expect pass.

### Task 4: Runner and deterministic evaluation

**Files:**
- Create: `multiuav/learning/runner.py`
- Test: `tests/test_mappo_baseline.py`

- [ ] Write failing test constructing two empty two-UAV environments, collecting a four-step rollout, and asserting finite `[4,2,2]` rewards/actions/log probabilities.
- [ ] Run the collection test; expect missing-module failure.
- [ ] Implement `make_empty_two_uav_scenario`, `make_cylinder_two_uav_scenario`, `MAPPOExperiment`, vectorized-by-list collection, environment reset after terminal/truncation, TensorBoard metric emission, deterministic evaluation, and scalar JSON result writing.
- [ ] Write a failing seed-reproducibility test comparing deterministic evaluation metrics from two restored trainers.
- [ ] Re-run runner tests; expect pass.

### Task 5: Training and evaluation CLIs

**Files:**
- Create: `scripts/train_mappo.py`
- Create: `scripts/evaluate_mappo.py`
- Create: `configs/rl/mappo_baseline.yaml`
- Test: `tests/test_mappo_baseline.py`

- [ ] Add a failing test that imports CLI parser builders and verifies every required YAML key is present.
- [ ] Implement YAML config fields: `seed`, `num_envs`, `num_uavs`, `rollout_length`, `total_steps`, `learning_rate`, `gamma`, `gae_lambda`, `clip_ratio`, `entropy_coef`, `value_coef`, `max_grad_norm`, `batch_size`, `ppo_epochs`, `hidden_dims`, `evaluation_interval`, and `checkpoint_interval`.
- [ ] Implement `train_mappo.py` with empty-first training and optional cylinder verification; implement `evaluate_mappo.py` loading a checkpoint and using deterministic actor means.
- [ ] Run CLI `--help` commands and the config test; expect pass.

### Task 6: Minimal training evidence and documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/progress.md`
- Modify: `docs/technical_decisions.md`
- Modify: `docs/known_issues.md`
- Test: `tests/test_mappo_baseline.py`

- [ ] Run a short empty two-UAV smoke training followed by checkpoint resume and deterministic evaluation; require finite reward/loss/KL/entropy metrics and an existing checkpoint.
- [ ] Run the configured empty two-UAV training and report first/final evaluation reward and success rate; rerun with one cylinder after the empty result is valid.
- [ ] Add the required TensorBoard metric names and commands to documentation; explicitly state that Phase 9 stops before GNN, hierarchy, expert pretraining, and CBF.
- [ ] Run `python -m unittest discover -s tests -v`, `python -m ruff check multiuav scripts tests`, and `python -m mypy multiuav scripts`; expect zero failures/errors.

## Self-review

- Every required Phase-9 file, unit test, YAML key, TensorBoard metric, training gate, checkpoint behavior, and deferred feature maps to a task.
- The `terminated` field is the sole GAE bootstrap mask; `truncated` is stored for reset/reporting only.
- Actor inputs remain local observations and critic inputs remain centralized states in every task.

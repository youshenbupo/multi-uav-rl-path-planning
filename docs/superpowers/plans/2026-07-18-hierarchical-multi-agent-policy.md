# Hierarchical Multi-Agent Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable, configurable high-level coordination policy over graph-conditioned low-level velocity MAPPO.

**Architecture:** A hold-state policy boundary separates discrete graph coordination actions from continuous graph velocity actions. Separate low/high rollout stores retain their own PPO statistics and use duration-aware high-level discounting. Training begins with a rule coordinator, then freezes the low level while learning the high level.

**Tech Stack:** Python 3.11, PyTorch distributions, PettingZoo, existing dense graph encoder, unittest, YAML, TensorBoard.

---

### Task 1: Hierarchical actions and timing

**Files:**
- Create: `multiuav/learning/hierarchical_policy.py`
- Create: `tests/test_hierarchical_policy.py`

- [x] Write failing tests for `H=3` command persistence, boundary positions, inactive reset, optional-field zeroing, and wait-action velocity gate.
- [x] Implement action enum, configuration, `HighLevelActionState`, deterministic rule coordinator, and `HierarchicalPolicy` hold transition.
- [x] Implement graph `HighLevelActor`, centralized high critic, and conditioned `LowLevelActor`; verify action/log-probability shapes and masks.

### Task 2: Two-scale rollout and discounting

**Files:**
- Create: `multiuav/learning/hierarchical_rollout_buffer.py`
- Modify: `tests/test_hierarchical_policy.py`

- [x] Write failing tests for low/high buffer alignment, closed early episodes, and a two-step high reward with expected `gamma_low` weighting and `gamma_low^duration` bootstrap factor.
- [x] Implement low-step storage, high-boundary opening/closing, discount accumulation, actual-duration high GAE, and fixed-rank flattened batches.
- [x] Verify all buffer tests pass without resetting or padding valid high records across episode boundaries.

### Task 3: Staged hierarchical PPO and checkpoints

**Files:**
- Create: `multiuav/learning/hierarchical_mappo.py`
- Modify: `tests/test_hierarchical_policy.py`

- [x] Write failing tests for finite Stage-A rule-high low update, frozen low parameters in Stage B, and checkpoint round-trip.
- [x] Implement separate PPO optimizers/GAE metrics for low and high policies, parameter freezing, joint-finetune guard, and checkpoint state.
- [x] Verify coefficient-free auxiliary loss is not added and no random joint initialization path exists.

### Task 4: Environment runner, configuration, and staged CLIs

**Files:**
- Create: `multiuav/learning/hierarchical_runner.py`
- Create: `configs/rl/hierarchical_mappo.yaml`
- Create: `scripts/train_hierarchical_mappo.py`
- Create: `scripts/evaluate_hierarchical_mappo.py`
- Modify: `tests/test_hierarchical_policy.py`

- [x] Write failing integration test for a short Stage-A rule-high low-training episode and deterministic evaluation.
- [x] Implement deterministic graph-environment collection with high boundaries, staged checkpoint loading, TensorBoard metrics, and CLI `--stage` choices `low`, `high`, `joint`.
- [x] Validate every ablation toggle, reject `joint` unless explicit configuration enables it, and preserve active masks through reset.

### Task 5: Documentation and final validation

**Files:**
- Modify: `README.md`
- Modify: `docs/progress.md`
- Modify: `docs/technical_decisions.md`
- Modify: `docs/known_issues.md`

- [x] Run a real Stage-A low-level training and Stage-B high-level training/checkpoint evaluation with their observed outputs stored under `data/rl`.
- [x] Document commands, limits, stage order, and exact observed results without claiming safety or generalization.
- [x] Run `python -m unittest discover -s tests -v`, `python -m ruff check .`, and `python -m mypy multiuav scripts`.

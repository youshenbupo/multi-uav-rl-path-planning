# Expert Behavior-Cloning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train strict-mask graph behavior-cloning policies from verified HDF5 experts and expose them for staged MAPPO initialization/fine tuning.

**Architecture:** A new HDF5 shard loader turns episode time steps and sparse graph groups into padded dense graph batches while retaining all validity masks. Separate graph-encoder low and high BC modules train against their own targets, checkpoints, and metrics. A small integration adapter transfers matching BC weights into existing MAPPO policies and controls encoder freezing and optional masked imitation loss.

**Tech Stack:** Python 3.11, h5py, NumPy, PyTorch, PettingZoo, YAML, unittest.

---

### Task 1: Verified HDF5 shard discovery and episode-level split

**Files:**
- Create: `multiuav/data/bc_dataset.py`
- Modify: `multiuav/data/__init__.py`
- Test: `tests/test_behavior_cloning.py`

- [ ] **Step 1: Write failing split/mask tests**

```python
def test_split_keeps_entire_episode_in_one_partition(self) -> None:
    manifest = build_bc_manifest([fixture_h5], seed=9, train_fraction=.6, validation_fraction=.2)
    episode_splits = {row["episode_key"]: row["split"] for row in manifest["episodes"]}
    self.assertEqual(len(episode_splits), 3)
    self.assertEqual({"train", "validation", "test"}, set(episode_splits.values()))

def test_unlabelled_high_actions_are_not_continue_targets(self) -> None:
    batch = next(iter(ExpertGraphDataset([fixture_h5], split="train").batches(batch_size=2)))
    self.assertFalse(batch.high_action_mask[0, 1])
    self.assertEqual(int(batch.high_actions[0, 1]), -1)
```

- [ ] **Step 2: Run the focused tests and confirm the import failure**

Run: `python -m unittest tests.test_behavior_cloning -v`

Expected: FAIL because `bc_dataset` is absent.

- [ ] **Step 3: Implement explicit shard/episode records and stable split allocation**

```python
@dataclass(frozen=True)
class ExpertEpisodeRecord:
    shard: Path
    episode_id: str
    episode_key: str
    scenario_id: str
    seed: int
    num_uavs: int
    split: Literal["train", "validation", "test"]

def build_bc_manifest(paths: Sequence[Path], *, seed: int,
                      train_fraction: float, validation_fraction: float) -> dict[str, object]:
    records = tuple(_read_episode_records(paths))
    return {"schema_version": "bc_manifest_v1", "split_seed": seed,
            "episodes": [_manifest_row(record, train_fraction, validation_fraction)
                         for record in records]}
```

Read `scenario_id`, `seed`, `num_uavs`, low and high mask counts from each HDF5
group. Reject duplicate canonical `(path, episode_id)` keys, non-verified file
names, fractions that do not sum to one, and any split that leaks an episode
key. Default discovery lists exactly the six verified part files in lexical
order, treating the un-suffixed verified file as part 01.

- [ ] **Step 4: Implement sample extraction and dense graph collation**

```python
@dataclass(frozen=True)
class ExpertGraphBatch:
    node_features: Tensor       # [B,N,local_observation_size]
    edge_features: Tensor       # [B,N,N,15], Phase-10 predictive graph
    adjacency: Tensor           # [B,N,N] bool
    active_mask: Tensor         # [B,N] bool
    raw_edge_mask: Tensor       # [B,N,N] bool, source HDF5 graph audit only
    low_actions: Tensor         # [B,N,3]
    low_action_mask: Tensor     # [B,N] bool
    high_actions: Tensor        # [B,N], -1 when unavailable
    high_action_mask: Tensor    # [B,N] bool
    max_speeds: Tensor          # [B,1,1]
```

Load each dynamic graph step into `raw_edge_mask`, preserving only source
`edge_mask=true` edges and verifying graph node activity agrees with the stored
active mask. Reconstruct `EnvironmentSnapshot` local observations and call
`ConflictGraphBuilder` with HDF5 positions, velocities, goals, and active mask
to make the 15-feature policy graph. Normalise only continuous local-observation
features using training-split statistics; preserve active and every boolean mask
exactly.

- [ ] **Step 5: Run focused tests and commit the self-contained dataset work**

Run: `python -m unittest tests.test_behavior_cloning -v`

Expected: PASS for split isolation, sparse labels, variable UAV padding, and
edge masking.

### Task 2: Low-level graph behavior cloning

**Files:**
- Create: `multiuav/learning/behavior_cloning.py`
- Modify: `tests/test_behavior_cloning.py`

- [ ] **Step 1: Write failing masked-loss tests**

```python
def test_low_bc_loss_ignores_invalid_actions_and_has_three_terms(self) -> None:
    loss = low_bc_loss(prediction, target, action_mask)
    self.assertAlmostEqual(float(loss.normalized_mse), expected_mse, places=6)
    self.assertAlmostEqual(float(loss.direction), expected_direction, places=6)
    self.assertAlmostEqual(float(loss.speed), expected_speed, places=6)
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m unittest tests.test_behavior_cloning.BehaviorCloningTests.test_low_bc_loss_ignores_invalid_actions_and_has_three_terms -v`

Expected: FAIL because the loss is absent.

- [ ] **Step 3: Implement the low policy and loss**

```python
class ExpertLowLevelPolicy(nn.Module):
    def forward(self, node_features: Tensor, edge_features: Tensor,
                adjacency: Tensor, active_mask: Tensor, high_commands: Tensor) -> Tensor:
        embeddings = self.encoder(node_features, edge_features, adjacency, active_mask)
        conditioned = torch.cat((embeddings, high_commands), dim=-1)
        return self.action_head(conditioned).tanh()

def low_bc_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> LowBCLoss:
    valid = mask & (target.norm(dim=-1) > 1e-6)
    return LowBCLoss(normalized_mse=_masked_mean((prediction - target).square(), mask),
                      direction=_masked_mean(1 - F.cosine_similarity(prediction, target, dim=-1), valid),
                      speed=_masked_mean((prediction.norm(dim=-1) - target.norm(dim=-1)).square(), mask))
```

Use an explicit all-continue command one-hot vector only for low BC samples
whose HDF5 high command is unavailable. This is input context, not a high label
or high-loss target. Convert normalized predictions back to per-episode physical
speed only for RMSE reporting.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_behavior_cloning -v`

Expected: PASS for masked MSE, cosine, magnitude, action bounds, and inactive
node zeroing.

### Task 3: Strict high-level behavior cloning

**Files:**
- Modify: `multiuav/learning/behavior_cloning.py`
- Modify: `tests/test_behavior_cloning.py`

- [ ] **Step 1: Write failing strict-label tests**

```python
def test_high_bc_never_uses_unlabelled_minus_one_as_continue(self) -> None:
    result = high_bc_loss(logits, labels=torch.tensor([[-1, 2]]),
                          action_mask=torch.tensor([[False, True]]))
    self.assertEqual(result.label_count, 1)
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m unittest tests.test_behavior_cloning.BehaviorCloningTests.test_high_bc_never_uses_unlabelled_minus_one_as_continue -v`

Expected: FAIL because the high policy is absent.

- [ ] **Step 3: Implement categorical and masked optional-regression heads**

```python
class ExpertHighLevelPolicy(nn.Module):
    def forward(self, node_features: Tensor, edge_features: Tensor,
                adjacency: Tensor, active_mask: Tensor) -> HighBCOutput:
        embeddings = self.encoder(node_features, edge_features, adjacency, active_mask)
        return HighBCOutput(self.maneuver(embeddings), self.delay(embeddings),
                            self.priority(embeddings), self.subgoal(embeddings))

def high_bc_loss(output: HighBCOutput, targets: HighBCTargets) -> HighBCLoss:
    if int(targets.action_mask.sum()) == 0:
        return HighBCLoss.zero(label_count=0)
    selected_logits = output.maneuver_logits[targets.action_mask]
    selected_labels = targets.actions[targets.action_mask]
    return HighBCLoss(classification=F.cross_entropy(selected_logits, selected_labels),
                      delay=_masked_mse(output.delay, targets.delay, targets.delay_mask),
                      priority=_masked_mse(output.priority, targets.priority, targets.priority_mask),
                      subgoal=_masked_mse(output.subgoal, targets.subgoal, targets.subgoal_mask))
```

The schema's nine label classes remain distinct. For transfer to Phase 11,
only encoder weights and the six overlapping maneuver rows are eligible; the
non-overlapping delay/priority/subgoal semantic heads remain BC-only unless
their authoritative targets exist.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_behavior_cloning -v`

Expected: PASS for label count, no-label `null` metrics, classifier output
shape, and masked regression heads.

### Task 4: Trainer, artifacts, and real pretraining CLI

**Files:**
- Create: `multiuav/learning/bc_trainer.py`
- Create: `configs/rl/behavior_cloning.yaml`
- Create: `scripts/train_behavior_cloning.py`
- Modify: `tests/test_behavior_cloning.py`

- [ ] **Step 1: Write failing artifact/checkpoint tests**

```python
def test_training_writes_required_artifacts(self) -> None:
    result = train_behavior_cloning(config, output_dir)
    for name in ("bc_low_level.pt", "bc_high_level.pt", "normalization_stats.json",
                 "training_config.yaml", "dataset_manifest.json"):
        self.assertTrue((output_dir / name).is_file())
```

- [ ] **Step 2: Run focused test and confirm it fails**

Run: `python -m unittest tests.test_behavior_cloning.BehaviorCloningTests.test_training_writes_required_artifacts -v`

Expected: FAIL because the trainer is absent.

- [ ] **Step 3: Implement deterministic BC training**

```python
@dataclass(frozen=True)
class BehaviorCloningConfig:
    shards: tuple[Path, ...]
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    embedding_dim: int
    graph_heads: int
    graph_layers: int
    normalized_mse_coef: float
    direction_coef: float
    speed_coef: float
```

Fit normalization from train low-action rows only. Save serializable model
metadata, normalization statistics, source manifest, exact configuration, and
both checkpoints even when high label count is zero; the high checkpoint stores
its count and null validation metrics. Reject configs with absent train low
labels or incompatible graph dimensions.

- [ ] **Step 4: Run a short CPU low/high BC training**

Run: `python scripts/train_behavior_cloning.py --config configs/rl/behavior_cloning.yaml --device cpu --epochs 2 --output-dir data/rl/bc_smoke`

Expected: five named artifacts, finite low metrics, and explicit high label
counts/null split metrics where the split has no labels.

### Task 5: Closed-loop expert-holdout evaluation

**Files:**
- Create: `multiuav/learning/bc_evaluation.py`
- Create: `scripts/evaluate_behavior_cloning.py`
- Modify: `tests/test_behavior_cloning.py`

- [ ] **Step 1: Write failing closed-loop metric test**

```python
def test_closed_loop_reports_required_metrics(self) -> None:
    metrics = evaluate_bc_checkpoint(low_checkpoint, test_records, device=torch.device("cpu"))
    self.assertEqual(set(metrics), {"arrival_rate", "collision_rate", "minimum_separation",
                                    "path_length", "expert_path_deviation", "episode_count"})
```

- [ ] **Step 2: Implement HDF5-scenario reconstruction and deterministic rollout**

```python
def scenario_from_episode(record: ExpertEpisodeRecord) -> Scenario:
    with h5py.File(record.shard, "r") as handle:
        return _read_scenario(handle[f"episodes/{record.episode_id}"])

def evaluate_bc_checkpoint(checkpoint: Path, records: Sequence[ExpertEpisodeRecord],
                           *, device: torch.device) -> dict[str, float | int]:
    return _aggregate_episode_metrics(_rollout_one_episode(checkpoint, record, device)
                                      for record in records)
```

Use the environment's normalized-action contract and each HDF5 episode's
mission geometry. Resample expert positions onto evaluated times before
computing path deviation; exclude inactive/padded rows. Do not repair policy
trajectories or use CBF.

- [ ] **Step 3: Run evaluator test and a real held-out smoke evaluation**

Run: `python scripts/evaluate_behavior_cloning.py --checkpoint data/rl/bc_smoke/bc_low_level.pt --manifest data/rl/bc_smoke/dataset_manifest.json --device cpu`

Expected: finite required metrics and explicit episode count.

### Task 6: MAPPO initialization, encoder schedule, and comparison interface

**Files:**
- Create: `multiuav/learning/bc_finetuning.py`
- Modify: `multiuav/learning/graph_mappo.py`
- Modify: `multiuav/learning/hierarchical_mappo.py`
- Create: `configs/experiments/bc_comparison.yaml`
- Create: `scripts/compare_bc_initialization.py`
- Modify: `tests/test_behavior_cloning.py`

- [ ] **Step 1: Write failing transfer/freeze tests**

```python
def test_bc_transfer_freezes_then_unfreezes_graph_encoder(self) -> None:
    schedule = BCFineTuneSchedule(freeze_encoder_updates=2, ppo_learning_rate=1e-4)
    apply_bc_initialization(trainer, checkpoint, schedule, mode="low_bc_only")
    self.assertFalse(any(parameter.requires_grad for parameter in trainer.actor.encoder.parameters()))
    schedule_after_update(trainer, update_index=2)
    self.assertTrue(all(parameter.requires_grad for parameter in trainer.actor.encoder.parameters()))
```

- [ ] **Step 2: Implement transfer and optional imitation configuration**

```python
@dataclass(frozen=True)
class BCFineTuneSchedule:
    freeze_encoder_updates: int
    ppo_learning_rate: float
    imitation_coef: float = 0.0

def apply_bc_initialization(trainer: GraphMAPPOTrainer | HierarchicalMAPPOTrainer,
                            checkpoint: Path, schedule: BCFineTuneSchedule,
                            mode: Literal["random", "low_bc_only", "high_bc_only", "full_bc",
                                          "bc_mappo_finetune"]) -> TransferReport:
    report = load_shape_compatible_state(trainer, checkpoint, mode)
    set_encoder_trainable(trainer, trainable=schedule.freeze_encoder_updates == 0)
    reset_ppo_optimizers(trainer, learning_rate=schedule.ppo_learning_rate)
    return report
```

Load only explicitly shape-compatible tensors, report every loaded/skipped key,
and rebuild PPO optimizers at the requested fine-tuning rate. `imitation_coef`
defaults to zero and, when enabled, applies only a correctly masked expert low
action loss to a supplied BC batch. It does not invent high labels.

- [ ] **Step 3: Implement comparison configuration validation and run tests**

Run: `python -m unittest tests.test_behavior_cloning -v`

Expected: PASS for five comparison modes, freeze/unfreeze timing, lower learning
rate, shape-safe loading, and masked auxiliary loss.

### Task 7: Documentation and final validation

**Files:**
- Modify: `README.md`
- Modify: `docs/progress.md`
- Modify: `docs/technical_decisions.md`
- Modify: `docs/known_issues.md`

- [ ] **Step 1: Record actual smoke results and sparse-label limitation**

Document the exact shard list, episode splits, high-label counts, artifact
paths, training/evaluation commands, and closed-loop metrics. State that the
current sparse high labels do not establish a learned high-coordination benefit.

- [ ] **Step 2: Run final checks**

Run: `python -m unittest discover -s tests -p 'test_*.py' -v`

Expected: all tests pass.

Run: `python -m ruff check .`

Expected: `All checks passed!`

Run: `python -m mypy multiuav scripts`

Expected: `Success: no issues found`.

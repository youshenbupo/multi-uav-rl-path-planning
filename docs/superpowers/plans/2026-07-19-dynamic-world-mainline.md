# Dynamic World Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the single CA-HGALO -> BC -> predictive graph hierarchical MAPPO -> CBF-QP research path in a dynamic, communication-impaired world with reproducible experiments.

**Architecture:** Add deterministic dynamic-world and communication modules below the PettingZoo environment.  Expose actor knowledge separately from centralized truth, consume it in the predictive graph and hierarchy, and pass the same world snapshot to CBF.  A compact experiment package owns method selection, metrics, output writing, and the four public scripts while reusing existing trainers and checkpoints.

**Tech Stack:** Python 3.11, NumPy, PettingZoo/Gymnasium, PyTorch/CUDA, OSQP, PyYAML, pytest, Ruff, mypy, Matplotlib.

---

## Planned file structure

- Create `multiuav/envs/dynamic_world.py`: immutable moving-cylinder definitions, deterministic state advance, and validation.
- Create `multiuav/envs/communication.py`: packet queue and per-UAV delivered-knowledge state.
- Modify `multiuav/core/models.py`: add `DynamicCylinder` and attach static dynamic-obstacle definitions to `Scenario`.
- Modify `multiuav/envs/multi_uav_env.py`, `observations.py`, `rewards.py`: advance world state, expose knowledge safely, and account for dynamic violations.
- Modify `multiuav/learning/conflict_graph.py`, `graph_runner.py`, `hierarchical_runner.py`: construct graph features from delivered knowledge and support all scale profiles.
- Modify `multiuav/safety/cbf_constraints.py`: add moving-obstacle relative-velocity barriers.
- Create `multiuav/experiments/{spec,scenarios,metrics,outputs,registry,runner}.py`: unified public experiment contract.
- Create `scripts/{train,evaluate,run_benchmark,run_ablation}.py`: thin entry points over `multiuav.experiments`.
- Create `configs/experiments/mainline_smoke.yaml` and `configs/experiments/scale_profiles.yaml`.
- Create `docs/{training,evaluation,reproducibility}.md`.
- Create or extend focused tests under `tests/` for every new module and integration boundary.

### Task 1: Deterministic dynamic-world data and stepping

**Files:**
- Create: `tests/test_dynamic_world.py`
- Create: `multiuav/envs/dynamic_world.py`
- Modify: `multiuav/core/models.py`
- Modify: `multiuav/envs/__init__.py`

- [ ] **Step 1: Write the failing dynamic-obstacle test**

```python
def test_dynamic_world_advances_constant_velocity_without_mutating_definition() -> None:
    obstacle = DynamicCylinder("crossing", np.array([10.0, 5.0, 20.0]), np.array([2.0, 0.0, 0.0]), 3.0, 40.0)
    world = DynamicWorldState((obstacle,), dt=0.5, step_count=0)

    advanced = world.advance()

    np.testing.assert_allclose(advanced.centers[0], [11.0, 5.0, 20.0])
    np.testing.assert_allclose(world.centers[0], [10.0, 5.0, 20.0])
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_dynamic_world.py::test_dynamic_world_advances_constant_velocity_without_mutating_definition -v`

Expected: FAIL because `DynamicCylinder` and `DynamicWorldState` do not exist.

- [ ] **Step 3: Implement minimal immutable dynamic-world types**

```python
@dataclass(frozen=True)
class DynamicCylinder:
    identifier: str
    initial_center: FloatArray
    velocity: FloatArray
    radius: float
    height: float

@dataclass(frozen=True)
class DynamicWorldState:
    obstacles: tuple[DynamicCylinder, ...]
    dt: float
    step_count: int = 0

    @property
    def centers(self) -> FloatArray:
        return np.asarray([o.initial_center + self.step_count * self.dt * o.velocity for o in self.obstacles])

    def advance(self) -> "DynamicWorldState":
        return replace(self, step_count=self.step_count + 1)
```

Validate identifiers, finite 3-vectors, positive geometry, and finite nonnegative `dt`.  Extend `Scenario` with `dynamic_obstacles: tuple[DynamicCylinder, ...] = ()` and export the public types.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest tests/test_dynamic_world.py -v`

Expected: PASS, including invalid geometry and out-of-bound trajectory validation tests.

- [ ] **Step 5: Commit**

```bash
git add multiuav/core/models.py multiuav/envs/dynamic_world.py multiuav/envs/__init__.py tests/test_dynamic_world.py
git commit -m "feat: add deterministic dynamic obstacle world"
```

### Task 2: Communication delivery model and actor knowledge

**Files:**
- Create: `tests/test_communication.py`
- Create: `multiuav/envs/communication.py`
- Modify: `multiuav/envs/observations.py`

- [ ] **Step 1: Write the failing delivery test**

```python
def test_delay_hides_neighbour_until_delivery_step() -> None:
    channel = CommunicationChannel(CommunicationConfig(enabled=True, range=100.0, delay_steps=2, drop_probability=0.0, max_staleness_steps=3), np.random.default_rng(7))
    channel.reset(2)
    channel.broadcast(np.array([[0., 0., 0.], [5., 0., 0.]]), np.zeros((2, 3)), np.array([True, True]), step=0)

    assert not channel.knowledge_for(0, step=1).valid[1]
    channel.deliver(step=2)
    assert channel.knowledge_for(0, step=2).valid[1]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_communication.py::test_delay_hides_neighbour_until_delivery_step -v`

Expected: FAIL because the communication module is missing.

- [ ] **Step 3: Implement packet queue and delivered state**

Create typed `CommunicationConfig`, `AgentKnowledgeState`, private queued packets, and `CommunicationChannel`.  `broadcast` uses pairwise range, one RNG draw per in-range directed packet, and immutable copied state; `deliver` applies due packets; `knowledge_for` returns own truth plus only fresh delivered neighbors with age and validity.  Add deterministic drop and stale-expiry tests.  Keep existing local observation layout unchanged when communication is disabled.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest tests/test_communication.py -v`

Expected: PASS with delay, loss, range, stale expiry, and seeded repeatability coverage.

- [ ] **Step 5: Commit**

```bash
git add multiuav/envs/communication.py multiuav/envs/observations.py tests/test_communication.py
git commit -m "feat: add delayed and lossy UAV communication"
```

### Task 3: Integrate world and communication into environment semantics

**Files:**
- Modify: `tests/test_multi_uav_environment.py`
- Modify: `tests/test_dynamic_world.py`
- Modify: `multiuav/envs/multi_uav_env.py`
- Modify: `multiuav/envs/observations.py`
- Modify: `multiuav/envs/rewards.py`
- Modify: `configs/env/multi_uav_mvp.yaml`

- [ ] **Step 1: Write failing integration tests**

```python
def test_environment_advances_dynamic_obstacles_once_per_step() -> None:
    env = make_dynamic_env()
    env.reset(seed=4)
    _, _, _, _, infos = env.step(zero_actions(env))
    np.testing.assert_allclose(infos["uav_0"]["dynamic_obstacles"][0]["center"], [11.0, 5.0, 20.0])

def test_dynamic_cylinder_penetration_is_a_named_safety_cost() -> None:
    snapshot = snapshot_inside_dynamic_cylinder()
    assert compute_safety_costs(snapshot, 0).dynamic_obstacle_violation_cost > 0.0
```

- [ ] **Step 2: Run the two tests and verify RED**

Run: `pytest tests/test_multi_uav_environment.py -k "dynamic" -v`

Expected: FAIL because the environment has no dynamic-world state or dynamic safety cost.

- [ ] **Step 3: Integrate state without hidden double-advance**

Initialize world/channel during `reset`; on every `step`, deliver previously queued packets, step UAV dynamics, advance dynamic world once, broadcast new active-UAV state, then build snapshot/observations/infos.  Add `dynamic_world`, `communication_channel`, and actor knowledge to `EnvironmentSnapshot`.  Extend safety costs and environment config with named dynamic-obstacle and communication settings.  Add nearest dynamic-obstacle features and validity masks only when enabled; update `local_observation_size` and observation-space assertions accordingly.

- [ ] **Step 4: Run regression and focused tests**

Run: `pytest tests/test_multi_uav_environment.py tests/test_dynamic_world.py tests/test_communication.py -v`

Expected: PASS; static configurations keep their existing observation width and results.

- [ ] **Step 5: Commit**

```bash
git add multiuav/envs configs/env/multi_uav_mvp.yaml tests/test_multi_uav_environment.py tests/test_dynamic_world.py
git commit -m "feat: integrate dynamic world into multi-UAV environment"
```

### Task 4: Dynamic-obstacle CBF constraints and telemetry

**Files:**
- Modify: `tests/test_cbf_safety.py`
- Modify: `multiuav/safety/cbf_constraints.py`
- Modify: `multiuav/safety/qp_filter.py`
- Modify: `multiuav/safety/action_adapter.py`
- Modify: `scripts/test_cbf_scenarios.py`

- [ ] **Step 1: Write the failing moving-obstacle barrier test**

```python
def test_dynamic_obstacle_row_accounts_for_relative_velocity() -> None:
    row = CBFConstraintBuilder(CBFConfig()).build(dynamic_snapshot())[0]
    assert row.kind == "dynamic_cylinder_crossing"
    assert row.relative_velocity_offset is not None
    assert row.lower == pytest.approx(-row.config_alpha_times_barrier + row.relative_velocity_offset)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_cbf_safety.py -k dynamic_obstacle_row -v`

Expected: FAIL because dynamic obstacle rows are absent.

- [ ] **Step 3: Implement moving-cylinder barrier rows**

Extend `CBFConstraintRow` with a finite scalar lower-bound offset or precomputed lower value.  For vertically overlapping cylinders, construct `h = ||p_xy-o_xy||^2 - (radius + threat_margin)^2` and enforce `2(p-o)^T u >= -alpha*h + 2(p-o)^T v_obstacle`.  Preserve static rows and all QP fallback rules.  Add dynamic constraint count to `SafetyFilterDecision` and normalized adapter telemetry.

- [ ] **Step 4: Run CBF tests and scenario script**

Run: `pytest tests/test_cbf_safety.py -v; python scripts/test_cbf_scenarios.py --scenario dynamic_crossing`

Expected: PASS; the dynamic scenario has finite solution or explicit emergency telemetry and never executes unfiltered requested action on failure.

- [ ] **Step 5: Commit**

```bash
git add multiuav/safety scripts/test_cbf_scenarios.py tests/test_cbf_safety.py
git commit -m "feat: filter actions against moving obstacles"
```

### Task 5: Communication-aware predictive graph and hierarchy scales

**Files:**
- Modify: `tests/test_predictive_conflict_graph.py`
- Modify: `tests/test_hierarchical_policy.py`
- Modify: `multiuav/learning/conflict_graph.py`
- Modify: `multiuav/learning/graph_runner.py`
- Modify: `multiuav/learning/hierarchical_runner.py`
- Modify: `configs/rl/graph_mappo.yaml`
- Modify: `configs/rl/hierarchical_mappo.yaml`
- Create: `configs/experiments/scale_profiles.yaml`

- [ ] **Step 1: Write failing graph-knowledge tests**

```python
def test_predictive_graph_omits_edge_for_unreceived_neighbor() -> None:
    graph = ConflictGraphBuilder(config).build_from_knowledge(knowledge_with_missing_neighbor())
    assert graph.adjacency[0, 1] == 0.0

def test_scale_profile_supports_sixteen_uavs() -> None:
    assert resolve_scale_profile(16).num_uavs == 16
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_predictive_conflict_graph.py -k "knowledge or scale" -v`

Expected: FAIL because graph construction has no knowledge input and profiles do not exist.

- [ ] **Step 3: Extend graph inputs and runner configuration**

Add a dedicated `build_from_knowledge` path that maps invalid or stale neighbor data to absent edges and adds bounded age/validity features while retaining `build` for static callers.  Extend runner configs with dynamic-world and communication fields, create deterministic scale profiles 3/5/8/12/16, and remove CLI choices that reject valid profiles.  Ensure actor graph uses knowledge while critic remains explicitly documented CTDE truth.

- [ ] **Step 4: Run graph and hierarchy suites**

Run: `pytest tests/test_predictive_conflict_graph.py tests/test_hierarchical_policy.py -v`

Expected: PASS with existing static graph tests unchanged and knowledge masking covered.

- [ ] **Step 5: Commit**

```bash
git add multiuav/learning configs/rl configs/experiments/scale_profiles.yaml tests/test_predictive_conflict_graph.py tests/test_hierarchical_policy.py
git commit -m "feat: use delivered communication in predictive graphs"
```

### Task 6: Reproducible experiment kernel, metrics, and method registry

**Files:**
- Create: `tests/test_experiment_framework.py`
- Create: `multiuav/experiments/__init__.py`
- Create: `multiuav/experiments/spec.py`
- Create: `multiuav/experiments/metrics.py`
- Create: `multiuav/experiments/outputs.py`
- Create: `multiuav/experiments/registry.py`
- Create: `multiuav/experiments/runner.py`

- [ ] **Step 1: Write failing output and unavailable-method tests**

```python
def test_experiment_writer_preserves_seed_raw_result_and_summary(tmp_path: Path) -> None:
    layout = ExperimentOutput.create(tmp_path, ExperimentSpec(name="smoke", seeds=(3, 5)))
    layout.write_raw_result(seed=3, episode=0, result=episode_result())
    summary = layout.write_summary([seed_result(3), seed_result(5)])
    assert (layout.root / "raw_results" / "seed_3.jsonl").exists()
    assert summary["success_rate"]["confidence_interval_95"] is not None

def test_unavailable_external_baseline_is_explicit() -> None:
    assert MethodRegistry().resolve("orca").availability == "unavailable"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_experiment_framework.py -v`

Expected: FAIL because the experiments package is missing.

- [ ] **Step 3: Implement the serializable experiment contract**

Implement frozen `ExperimentSpec`, strict YAML loading, device resolution, metadata collection, output layout, JSONL raw seed records, CSV/JSON summary, Student-t interval calculation, and no-outlier-deletion aggregation.  Implement method registry entries for the eight specified existing methods plus `astar`, `rrt_star`, and `orca` unavailable adapters.  Controller selection must reject checkpointless learned methods and reject `--use-cbf` only for methods that cannot produce action arrays.  Include required metrics plus dynamic and communication metrics.

- [ ] **Step 4: Run framework tests and static analysis**

Run: `pytest tests/test_experiment_framework.py -v; ruff check multiuav/experiments; mypy multiuav/experiments`

Expected: PASS with no lint or type errors.

- [ ] **Step 5: Commit**

```bash
git add multiuav/experiments tests/test_experiment_framework.py
git commit -m "feat: add reproducible experiment framework"
```

### Task 7: Public scripts, scenario profiles, figures, and documentation

**Files:**
- Create: `tests/test_experiment_scripts.py`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `scripts/run_benchmark.py`
- Create: `scripts/run_ablation.py`
- Create: `configs/experiments/mainline_smoke.yaml`
- Create: `docs/training.md`
- Create: `docs/evaluation.md`
- Create: `docs/reproducibility.md`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI contract tests**

```python
@pytest.mark.parametrize("script", ["train.py", "evaluate.py", "run_benchmark.py", "run_ablation.py"])
def test_public_script_exposes_required_common_arguments(script: str) -> None:
    completed = subprocess.run([sys.executable, f"scripts/{script}", "--help"], text=True, capture_output=True, check=True)
    for flag in ("--config", "--seed", "--device", "--num-uavs", "--scenario", "--checkpoint", "--render", "--use-expert-pretrain", "--use-graph", "--use-hierarchy", "--use-cbf"):
        assert flag in completed.stdout
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_experiment_scripts.py -v`

Expected: FAIL because these scripts do not exist.

- [ ] **Step 3: Implement thin CLI adapters and docs**

Put common flag parsing in one importable helper.  Each script resolves an `ExperimentSpec` and delegates to the experiment runner; no training or metric logic is duplicated in scripts.  `run_ablation.py` supports each required flag disabling pretraining, CPA, graph encoder, hierarchy, high-level delay, altitude maneuver, CBF, auxiliary prediction, and curriculum.  The smoke YAML enables a 3-UAV dynamic crossing, one-step communication delay, graph, hierarchy, and CBF, writes a PNG figure, and records CUDA availability.  Document exact CUDA and CPU commands plus the OSQP CPU caveat.

- [ ] **Step 4: Run CLI tests and smoke help commands**

Run: `pytest tests/test_experiment_scripts.py -v; python scripts/train.py --help; python scripts/evaluate.py --help`

Expected: PASS; all four scripts expose the common contract.

- [ ] **Step 5: Commit**

```bash
git add scripts configs/experiments/mainline_smoke.yaml docs README.md tests/test_experiment_scripts.py
git commit -m "feat: add unified training and evaluation commands"
```

### Task 8: End-to-end smoke validation and final quality gates

**Files:**
- Modify: `tests/test_experiment_framework.py`
- Modify: `docs/progress.md`

- [ ] **Step 1: Write the failing full-mainline smoke test**

```python
def test_mainline_smoke_runs_graph_hierarchy_cbf_and_writes_artifacts(tmp_path: Path) -> None:
    result = run_smoke_mainline(tmp_path, device=available_device())
    assert result["capabilities"]["dynamic_obstacles"]
    assert result["metrics"]["cbf_intervention_rate"] >= 0.0
    assert (tmp_path / "figures" / "episode_overview.png").exists()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_experiment_framework.py::test_mainline_smoke_runs_graph_hierarchy_cbf_and_writes_artifacts -v`

Expected: FAIL until the runner binds environment, graph/hierarchy action generation, CBF, and output writer.

- [ ] **Step 3: Bind the minimal end-to-end runner**

Use a compatible small BC checkpoint if present; otherwise create an explicit test fixture checkpoint from the matching policy architecture, never silently claim an expert checkpoint was loaded.  Run a bounded 3-UAV episode with dynamic world and delayed communication, graph/hierarchy inference, normalized-action CBF filtering, raw result writing, and Matplotlib episode figure generation.  Record whether CUDA was actually selected.

- [ ] **Step 4: Run all required verification**

Run:

```bash
pytest
ruff check .
mypy multiuav scripts
python scripts/evaluate.py --config configs/experiments/mainline_smoke.yaml --device cuda --use-expert-pretrain --use-graph --use-hierarchy --use-cbf
```

Expected: all tests, lint, and typing pass; the smoke run writes every required output subdirectory and reports the actual resolved device.

- [ ] **Step 5: Update phase progress and commit**

```bash
git add tests/test_experiment_framework.py docs/progress.md
git commit -m "test: verify dynamic world mainline smoke run"
```

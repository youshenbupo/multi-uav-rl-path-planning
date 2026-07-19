# Uncertainty-Aware Communication Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make communication staleness causally affect multi-UAV graph coordination and CBF safety margins without leaking neighbour truth to actors.

**Architecture:** Each delivered packet retains immutable source position, velocity, and age. At observation time, an actor dead-reckons the received neighbour state and derives a deterministic, age-growing scalar uncertainty bound. The graph uses the predicted state, age, and uncertainty to compute risk-adjusted CPA features and adjacency. The centralized simulation CBF keeps its true-state constraint geometry, but increases the pairwise separation margin according to the two actors' delivered-information uncertainty; this distinction is documented and measured.

**Tech Stack:** Python 3.11, NumPy, PyTorch, PettingZoo, CVXPY/OSQP, pytest, YAML.

---

## File map

- Modify: `multiuav/envs/communication.py` — packet prediction and immutable uncertainty bounds.
- Modify: `multiuav/envs/multi_uav_env.py` — serialize uncertainty model and pass environment `dt`.
- Modify: `multiuav/envs/observations.py` — expose predicted neighbour state, age, and uncertainty without truth leakage.
- Modify: `multiuav/learning/conflict_graph.py` — add uncertainty feature and risk-adjusted CPA adjacency.
- Modify: `multiuav/learning/graph_runner.py` — construct actor graphs from predicted received data.
- Modify: `multiuav/safety/cbf_constraints.py` — uncertainty-sensitive pairwise separation margin.
- Modify: `tests/test_communication.py`, `tests/test_predictive_conflict_graph.py`, `tests/test_cbf_safety.py`, `tests/test_multi_uav_environment.py` — regression and non-leakage tests.
- Modify: `configs/experiments/mainline_smoke.yaml`, `configs/experiments/dynamic_world_mainline.yaml`, `docs/phase14_experiments.md`, `docs/known_issues.md` — protocol and limitations.
- Create: `docs/aamas2027_research_brief.md` — claim boundaries, notation, literature-search protocol, and required evidence.

### Task 1: Add deterministic packet prediction and uncertainty bounds

**Files:**
- Modify: `multiuav/envs/communication.py`
- Modify: `multiuav/envs/multi_uav_env.py`
- Test: `tests/test_communication.py`

- [x] **Step 1: Write failing communication tests.**

```python
def test_knowledge_dead_reckons_a_delivered_packet_and_grows_uncertainty() -> None:
    channel = CommunicationChannel(
        CommunicationConfig(enabled=True, delay_steps=0, max_staleness_steps=3,
                            prediction_dt=0.5, uncertainty_growth_per_step=2.0),
        np.random.default_rng(0),
    )
    channel.reset(2)
    channel.broadcast(positions=np.array([[0., 0., 0.], [5., 0., 0.]]),
                      velocities=np.array([[0., 0., 0.], [4., 0., 0.]]),
                      active_mask=np.array([True, True]), step=0)
    channel.deliver(step=0)
    knowledge = channel.knowledge_for(0, step=2)
    np.testing.assert_allclose(knowledge.predicted_positions[1], [9., 0., 0.])
    assert knowledge.position_uncertainty[1] == 4.0
```

- [x] **Step 2: Run the targeted test and confirm it fails because the fields/configuration do not exist.**

Run: `conda run -n multiuav_rl python -m pytest tests/test_communication.py -q`

- [x] **Step 3: Extend the immutable communication contract.**

Add `prediction_dt: float` and `uncertainty_growth_per_step: float` to `CommunicationConfig`; reject non-positive `prediction_dt` and negative growth. Extend `AgentKnowledgeState` with read-only `[N,3]` `predicted_positions` and `[N]` `position_uncertainty`. In `knowledge_for`, preserve the original delivered position, compute `received + age * prediction_dt * received_velocity` only for valid neighbours, and set `uncertainty_growth_per_step * age`; invalid rows remain zero.

- [x] **Step 4: Project environment configuration to the channel.**

Add flat `communication_uncertainty_growth_per_step: float = 0.0` to `EnvironmentConfig`, validate it, and construct `CommunicationConfig(prediction_dt=self.dt, uncertainty_growth_per_step=...)` in `communication_config`.

- [x] **Step 5: Run communication and environment tests.**

Run: `conda run -n multiuav_rl python -m pytest tests/test_communication.py tests/test_multi_uav_environment.py -q`
Expected: all tests pass.

### Task 2: Make observations and graph construction uncertainty-aware

**Files:**
- Modify: `multiuav/envs/observations.py`
- Modify: `multiuav/learning/conflict_graph.py`
- Modify: `multiuav/learning/graph_runner.py`
- Test: `tests/test_predictive_conflict_graph.py`
- Test: `tests/test_multi_uav_environment.py`

- [x] **Step 1: Write failing graph tests.**

```python
def test_knowledge_graph_uses_dead_reckoned_position_and_uncertainty_risk() -> None:
    graph = builder.build_from_knowledge(
        positions=positions, received_positions=received_positions,
        received_velocities=received_velocities, predicted_positions=predicted_positions,
        knowledge_uncertainty=torch.tensor([[[0.0, 4.0], [0.0, 0.0]]]),
        goals=goals, active_mask=active, knowledge_valid=valid, knowledge_ages=ages,
    )
    assert graph.edge_features.shape[-1] == 16
    assert graph.edge_features[0, 0, 1, -1] > 0.0
```

- [x] **Step 2: Change the explicit graph tensor contract.**

Set `EDGE_FEATURE_DIMENSION = 16`. Add `uncertainty_scale` and `uncertainty_risk_gain` to `GraphBuildConfig` with finite non-negative validation. `build_from_knowledge` must accept `predicted_positions` and `knowledge_uncertainty`, use predicted positions for relative position, CPA, relative goal direction, and current-distance features, append normalized uncertainty as feature 16, and derive risk-adjusted CPA separation as:

```python
risk_adjusted_cpa_distance = (distance_at_cpa - uncertainty_risk_gain * knowledge_uncertainty).clamp_min(0.0)
predicted_shortfall = (risk_distance - risk_adjusted_cpa_distance).clamp_min(0.0) / risk_distance
```

Use that risk-adjusted distance for predicted-conflict edges. The non-knowledge `build` path supplies zero uncertainty so its semantics are unchanged except for the zero-valued final feature.

- [x] **Step 3: Preserve fair local observations.**

Replace the hard-coded neighbour width `6` with a named neighbour feature width `8`. For delivered neighbours, append normalized age and normalized uncertainty to relative position/velocity. Padded neighbour rows are eight zeros. Update `local_observation_size` and add a test proving an actor observation changes when only a delivered packet age changes, while the actor cannot observe an invalid neighbour.

- [x] **Step 4: Connect environment knowledge to graph tensors.**

In `build_graph_from_environments`, batch `knowledge.predicted_positions` and `knowledge.position_uncertainty` and pass them to `build_from_knowledge`. Update graph-network construction and all affected test assertions to use `EDGE_FEATURE_DIMENSION`, never literal `15`.

- [x] **Step 5: Run graph and actor-observation tests.**

Run: `conda run -n multiuav_rl python -m pytest tests/test_predictive_conflict_graph.py tests/test_multi_uav_environment.py -q`
Expected: all tests pass and no actor graph edge can be created from an invalid packet.

### Task 3: Apply communication uncertainty to CBF safety margins

**Files:**
- Modify: `multiuav/safety/cbf_constraints.py`
- Test: `tests/test_cbf_safety.py`

- [x] **Step 1: Write a failing margin test.**

```python
def test_pairwise_cbf_margin_increases_with_delivered_information_uncertainty() -> None:
    certain = CBFConstraintBuilder(CBFConfig(communication_uncertainty_margin_gain=2.0)).build(certain_snapshot)
    uncertain = CBFConstraintBuilder(CBFConfig(communication_uncertainty_margin_gain=2.0)).build(uncertain_snapshot)
    assert uncertain_row.barrier < certain_row.barrier
```

- [x] **Step 2: Add bounded, explicit CBF configuration.**

Add `communication_uncertainty_margin_gain: float = 0.0` and `max_communication_uncertainty_margin: float = 0.0` to `CBFConfig`; validate both are non-negative and cap the applied margin when the maximum is positive. In `_separation_row`, compute the pair uncertainty as the maximum of valid directed knowledge `i→j` and `j→i`; absent knowledge contributes zero. Set `safe_distance = scenario.safe_separation + applied_margin` before computing the barrier.

- [x] **Step 3: Document the safety-boundary semantics in code.**

State in the builder docstring that the simulated joint CBF uses central truth for constraint geometry, while communication-derived uncertainty only tightens its margin. Do not present this as a decentralized CBF guarantee.

- [x] **Step 4: Run CBF tests.**

Run: `conda run -n multiuav_rl python -m pytest tests/test_cbf_safety.py -q`
Expected: prior deterministic rows remain unchanged at zero gain; uncertainty produces a stricter row.

### Task 4: Add paper-facing research and experiment artifacts

**Files:**
- Create: `docs/aamas2027_research_brief.md`
- Modify: `configs/experiments/mainline_smoke.yaml`
- Modify: `configs/experiments/dynamic_world_mainline.yaml`
- Modify: `docs/phase14_experiments.md`
- Modify: `docs/known_issues.md`

- [x] **Step 1: Record the claim boundary and notation.**

Write the research brief with: research question; `p_ij^rec`, `v_ij^rec`, packet age `a_ij`; dead-reckoned `p_hat_ij`; uncertainty `sigma_ij`; risk-adjusted CPA; CBF margin; three contributions; actor/critic/safety information boundaries; claims that need multi-seed evidence; and claims that must never be made.

- [ ] **Step 2: Define only the core comparison matrix.**

Set the protocol to initial 3/5/8 UAV, five seeds, and scenarios nominal, delay, loss, dynamic, combined, and OOD. Require separately trained artifacts for MLP-MAPPO, naive graph, predictive graph without uncertainty, and full method. Do not claim external A*/RRT*/ORCA/CA-HGALO results without compatible implementations.

- [x] **Step 3: Make the smoke configuration exercise the new mechanism.**

Set a nonzero communication uncertainty growth, graph uncertainty scale/risk gain, and bounded CBF uncertainty margin in the smoke protocol. Keep it explicitly labelled as a plumbing test.

- [x] **Step 4: Update limitations.**

State that no trained dynamic-world comparison currently exists, new observation and edge dimensions invalidate old checkpoints, and uncertainty bounds are a model assumption requiring sensitivity analysis rather than a calibrated sensor-error estimate.

### Task 5: Verify, commit, and prepare the next experimental gate

**Files:**
- Modify: `docs/progress.md`

- [x] **Step 1: Run static checks and the full test suite.**

Run:

```powershell
conda run -n multiuav_rl python -m pytest -q
conda run -n multiuav_rl ruff check .
conda run -n multiuav_rl mypy multiuav scripts
```

Expected: all tests pass, Ruff exits 0, and mypy exits 0.

- [x] **Step 2: Run a bounded CUDA integration smoke test.**

Use `mainline_smoke.yaml`, CUDA, and a new experiment name. Verify `environment.json` records `resolved_device: cuda`, communication uncertainty settings, CBF settings, and no mistaken performance claim.

- [ ] **Step 3: Update evidence and commit.**

Add a concise progress entry naming the implemented mechanism and verification output. Commit only source, tests, configs, and documentation; do not commit generated runs, checkpoints, or caches. Push the active `codex/phase14-dynamic-world` branch after tests pass.

## Plan self-review

- Scope coverage: Tasks 1–3 implement every causal link in the locked paper narrative; Task 4 fixes the scientific protocol; Task 5 verifies and records it.
- Deliberate exclusions: CA-HGALO fidelity, external baseline adapters, long training, and manuscript drafting remain later goal stages because they cannot validate this mechanism before it exists.
- Compatibility: existing checkpoints are intentionally treated as incompatible after the observation/edge-width change. No fallback or silent weight reuse is allowed.

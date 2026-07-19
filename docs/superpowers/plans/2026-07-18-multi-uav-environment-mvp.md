# Multi-UAV Environment MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a PettingZoo Parallel API multi-UAV single-integrator environment with separate performance rewards and safety costs, centralized state, visualization, and Phase 8 tests.

**Architecture:** `dynamics.py`, `observations.py`, and `rewards.py` are pure NumPy modules. `multi_uav_env.py` owns episode state and PettingZoo lifecycle; `wrappers.py` records renderable episode data without affecting the environment. Existing immutable `Scenario`, terrain, and threat models are reused.

**Tech Stack:** Python 3.11, NumPy, Gymnasium 1.3, PettingZoo 1.26, PyYAML, Matplotlib, unittest, ruff, mypy.

---

## File responsibilities

- Create `configs/env/multi_uav_mvp.yaml`: all environment constants.
- Create `multiuav/envs/dynamics.py`: normalized action decoding and one-step dynamics protocol.
- Create `multiuav/envs/observations.py`: local observation and centralized-state builders.
- Create `multiuav/envs/rewards.py`: performance reward and separate safety-cost functions.
- Create `multiuav/envs/multi_uav_env.py`: `ParallelEnv` state, lifecycle, spaces, termination.
- Create `multiuav/envs/wrappers.py`: centralized-state and trajectory-recorder wrappers.
- Create `scripts/visualize_environment_episode.py`: deterministic rollout/render entry point.
- Create `tests/test_multi_uav_environment.py`: all Phase 8 required behavior.
- Modify `multiuav/envs/__init__.py`, `README.md`, `docs/progress.md`, `docs/technical_decisions.md`, and `docs/known_issues.md` only after verification.

## Locked data contracts

`EnvironmentConfig` is a frozen dataclass loaded by
`load_environment_config(path: Path)`. Tests may construct it with overrides,
but all executable scripts load the checked-in YAML. Its hard terminal
thresholds are `collision_distance`, `severe_clearance_shortfall`, and
`severe_threat_penetration`.

For `K = max_neighbors`, the local vector dimension is exactly `17 + 6K`:
`own_position[3], own_velocity[3], goal_relative[3], goal_distance[1],
terrain_clearance[1], nearest_threat[4], neighbours[K, 6], remaining_time[1],
high_level_placeholder[1]`. Missing threat and neighbour slots are zeros. The
central state dimension is `10N + 4 + 4T + 2`, where `N` is the fixed mission
count and `T` is the fixed scenario threat count. It contains `N` rows of
`position[3], velocity[3], goal[3], active[1]`, terrain min/max height plus
minimum/mean active clearance, threat `(x,y,radius,height)` rows, then minimum
separation and conflict count.

### Task 1: Lock YAML configuration and dynamics contract

**Files:**
- Create: `configs/env/multi_uav_mvp.yaml`
- Create: `multiuav/envs/dynamics.py`
- Test: `tests/test_multi_uav_environment.py`

- [ ] Write the failing dynamics tests for normalized horizontal/vertical clipping and bounds.

```python
def test_single_integrator_clips_normalized_action_and_world_bounds(self) -> None:
    dynamics = SingleIntegrator3D(dt=1.0, max_horizontal_speed=10.0, max_vertical_speed=3.0)
    result = dynamics.advance(np.array([995.0, 995.0, 319.0]), np.array([1.0, 1.0, 1.0]), world)
    assert_allclose(result.velocity, [10.0 / np.sqrt(2.0), 10.0 / np.sqrt(2.0), 3.0])
    assert result.boundary_clipped
```

- [ ] Run `python -m unittest tests.test_multi_uav_environment.EnvironmentTests.test_single_integrator_clips_normalized_action_and_world_bounds -v`; expect import failure before implementation.
- [ ] Implement `DynamicsResult`, `DynamicsModel`, and `SingleIntegrator3D.advance()` with the formula `position + dt * velocity`; decode normalized action before jointly clipping XY and independently clipping Z.
- [ ] Add `EnvironmentConfig` and `load_environment_config(path: Path)` using `yaml.safe_load`; reject nonpositive dynamics values, negative thresholds, and invalid reward mappings.
- [ ] Add YAML fields `dt: 1.0`, `max_steps: 200`, `max_horizontal_speed: 35.0`, `max_vertical_speed: 12.0`, `normalize_actions: true`, `goal_radius: 20.0`, `collision_distance: 20.0`, `severe_clearance_shortfall: 20.0`, `severe_threat_penetration: 10.0`, and explicit reward/severity fields.
- [ ] Re-run the single test; expect pass.

### Task 2: Implement pure observation and reward/cost modules

**Files:**
- Create: `multiuav/envs/observations.py`
- Create: `multiuav/envs/rewards.py`
- Test: `tests/test_multi_uav_environment.py`

- [ ] Write failing tests asserting `float32` fixed observation shape, zero neighbour padding, and the four named safety costs.

```python
observation = build_local_observation(snapshot, uav_index=0, max_neighbors=2)
self.assertEqual(observation.dtype, np.float32)
self.assertEqual(observation.shape, (29,))
self.assertEqual(set(costs), {
    "inter_uav_collision_cost", "terrain_violation_cost",
    "threat_violation_cost", "boundary_violation_cost",
})
```

- [ ] Run both tests; expect missing-module errors.
- [ ] Implement `EnvironmentSnapshot`, `build_local_observation()`, and `build_centralized_state()` with named feature constants and normalized finite `float32` output. Assert local dimension `17 + 6K` and central dimension `10N + 4 + 4T + 2` in tests.
- [ ] Implement `SafetyCosts`, `compute_safety_costs()`, and `compute_performance_reward()`; keep safety costs out of the scalar reward.
- [ ] Re-run both tests; expect pass.

### Task 3: Implement PettingZoo environment lifecycle

**Files:**
- Create: `multiuav/envs/multi_uav_env.py`
- Modify: `multiuav/envs/__init__.py`
- Test: `tests/test_multi_uav_environment.py`

- [ ] Write failing reset and step tests.

```python
observations, infos = env.reset(seed=7)
self.assertEqual(set(observations), set(env.possible_agents))
self.assertTrue(env.observation_space("uav_0").contains(observations["uav_0"]))
next_obs, rewards, terms, truncs, infos = env.step(zero_actions(env))
self.assertEqual(set(rewards), set(env.possible_agents))
self.assertIn("central_state", infos["uav_0"])
```

- [ ] Run these tests; expect import failure.
- [ ] Implement constructor spaces, `possible_agents`, `agents`, cached `observation_space()`/`action_space()`, `reset()`, `step()`, `state()`, and deterministic NumPy seeding.
- [ ] On every step, construct a snapshot, update active masks/positions/velocities, expose component reward/cost/termination data in infos, clear `agents` only after global termination/truncation.
- [ ] Re-run reset and step tests; expect pass.

### Task 4: Cover required flight, failure, completion, and cardinality behavior

**Files:**
- Modify: `tests/test_multi_uav_environment.py`
- Modify: `multiuav/envs/multi_uav_env.py`
- Test: `tests/test_multi_uav_environment.py`

- [ ] Write failing tests for fixed-seed reset/step equivalence, goal-directed straight flight, zero action, opposing collision, terrain violation, cylindrical threat violation, all-arrived termination, and 1/2/4-UAV scenarios.
- [ ] Run each test individually and confirm the expected missing semantics, not test setup errors.
- [ ] Implement only the termination and active-mask logic needed for those tests: termination reason precedence is numerical error, collision below `collision_distance`, terrain below `min_clearance - severe_clearance_shortfall`, threat penetration beyond `severe_threat_penetration`, all arrived; max-step is truncation. Boundary clipping creates a cost but not a global terminal event.
- [ ] Re-run `python -m unittest tests.test_multi_uav_environment -v`; expect all Phase 8 cases pass.

### Task 5: Add wrappers and artifact visualization

**Files:**
- Create: `multiuav/envs/wrappers.py`
- Create: `scripts/visualize_environment_episode.py`
- Modify: `tests/test_multi_uav_environment.py`

- [ ] Write failing test that `EpisodeTrajectoryRecorder` produces one initial position plus each transition and finite minimum-separation history.
- [ ] Run the recorder test; expect import failure.
- [ ] Implement `CentralizedStateWrapper` and `EpisodeTrajectoryRecorder`; add `render_episode_summary()` producing 3D, top-view, and separation-curve subplots.
- [ ] Implement the script with explicit YAML/seed/output arguments and a deterministic goal-seeking controller only for visualization.
- [ ] Run the recorder test and script; inspect the generated PNG.

### Task 6: Integrate documentation and verify the stage gate

**Files:**
- Modify: `README.md`
- Modify: `docs/progress.md`
- Modify: `docs/technical_decisions.md`
- Modify: `docs/known_issues.md`

- [ ] Document the PettingZoo API, YAML command, local/central state contract, separate safety costs, and no-MAPPO scope.
- [ ] Run `python -m unittest discover -s tests -v`; expect all existing and environment tests pass.
- [ ] Run `python -m ruff check multiuav scripts tests` and `python -m mypy multiuav scripts`; expect zero errors.
- [ ] Run the visualization command and visually inspect the saved PNG.
- [ ] Record the precise verification evidence and state that Phase 8 stops before MAPPO.

## Self-review

- Each Phase 8 required file, behavior, test category, and visualization has a dedicated task.
- The plan introduces neither MAPPO nor any forbidden Phase 9 component.
- Names and data contracts are consistent across dynamics, snapshot, environment, wrappers, and tests.

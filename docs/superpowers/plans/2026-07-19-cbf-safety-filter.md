# CBF Safety Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an execution-only direct-OSQP CBF filter for multi-UAV desired velocities with bounded emergency fallback and auditable telemetry.

**Architecture:** Build typed linear CBF rows from an immutable scenario snapshot, solve all active UAV controls jointly with per-row slack, and isolate non-solved handling in an emergency module. An adapter converts between the environment's normalized actions and physical velocities without adding any differentiable dependency to PPO.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse matrices, OSQP, PettingZoo, unittest, YAML.

---

### Task 1: Constraint representation and pairwise/geometry CBF rows

**Files:**
- Create: `multiuav/safety/cbf_constraints.py`
- Create: `configs/safety/cbf.yaml`
- Test: `tests/test_cbf_safety.py`

- [ ] **Step 1: Write failing geometry-row tests**

```python
def test_pair_constraint_contains_opposite_joint_control_coefficients(self) -> None:
    rows = CBFConstraintBuilder(config).build(snapshot)
    pair = next(row for row in rows if row.kind == "uav_separation")
    self.assertTrue(np.allclose(pair.coefficients[:3], -pair.coefficients[3:6]))
```

- [ ] **Step 2: Run `python -m unittest tests.test_cbf_safety.CBFSafetyTests.test_pair_constraint_contains_opposite_joint_control_coefficients -v` and confirm import failure.**

- [ ] **Step 3: Implement `CBFConfig`, `CBFConstraintRow`, and `CBFConstraintBuilder`.**

```python
row = CBFConstraintRow(kind="uav_separation", barrier=h,
    coefficients=coefficients, lower=-alpha * h, slackable=True)
```

Implement pair separation, terrain clearance using finite-difference terrain gradient, vertical-gated cylinder radial clearance, and six signed world-boundary barriers. Reject non-finite state data and retain only active-UAV rows.

- [ ] **Step 4: Run focused tests for rows, terrain, threats, boundaries, and inactive masks.**

### Task 2: Direct OSQP joint filter and emergency behavior

**Files:**
- Create: `multiuav/safety/qp_filter.py`
- Create: `multiuav/safety/emergency_policy.py`
- Modify: `multiuav/safety/__init__.py`
- Test: `tests/test_cbf_safety.py`

- [ ] **Step 1: Write failing tests for safe correction, slack telemetry, warm start, and forced solver failure.**

```python
decision = safety_filter.filter(snapshot, rl_velocity)
self.assertFalse(decision.emergency_fallback_used)
self.assertGreater(decision.intervention_norm, 0.0)
```

- [ ] **Step 2: Run the focused test and confirm the missing filter failure.**

- [ ] **Step 3: Implement OSQP matrix assembly and solve handling.**

Build `z=[u_0,...,u_n,s_0,...,s_m]`, diagonal quadratic objective, CBF lower rows, slack non-negativity, polygonal horizontal-speed rows, and vertical bounds. Cache/warm-start the previous solution only when shape matches. Treat only solved statuses within `max_solve_time_seconds` and finite output as normal success.

- [ ] **Step 4: Implement bounded `EmergencyPolicy`.**

```python
direction = pair_repulsion + threat_repulsion + boundary_repulsion + terrain_ascent
return clip_velocity(direction if np.linalg.norm(direction) else np.zeros(3))
```

Never return the raw requested control from a QP failure. Emit structured logging and a typed `SafetyFilterDecision` containing all required per-step fields.

- [ ] **Step 5: Run focused solver/fallback tests.**

### Task 3: Normalized-action adapter and scenario script

**Files:**
- Create: `multiuav/safety/action_adapter.py`
- Create: `scripts/test_cbf_scenarios.py`
- Test: `tests/test_cbf_safety.py`

- [ ] **Step 1: Write a failing adapter test proving normalized input/output matches physical filter limits.**

```python
filtered, decision = adapter.filter_normalized(snapshot, normalized_actions)
self.assertTrue(np.all(np.abs(filtered) <= 1.0))
```

- [ ] **Step 2: Implement conversion between normalized action dictionaries and physical joint velocity arrays.**

Only the adapter invokes the filter; PPO actors remain unchanged. Inactive UAV actions become zero.

- [ ] **Step 3: Implement the deterministic scenario CLI.**

```powershell
python scripts/test_cbf_scenarios.py --scenario all --device cpu
```

It creates the nine required synthetic states, runs the filter, prints JSON telemetry, and exits nonzero only when a scenario invariant fails.

- [ ] **Step 4: Run the complete CBF test module and script.**

### Task 4: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/progress.md`
- Modify: `docs/technical_decisions.md`
- Modify: `docs/known_issues.md`

- [ ] **Step 1: Record exact QP semantics, first-run telemetry, and execution-only boundary.**

- [ ] **Step 2: Run final checks.**

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
python -m ruff check .
python -m mypy multiuav scripts
```

Expected: all tests, lint, and types pass. State explicitly that a short safety-filter scenario test is not a trained-policy safety claim.

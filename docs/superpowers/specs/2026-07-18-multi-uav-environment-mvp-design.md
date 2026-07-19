# Multi-UAV Reinforcement-Learning Environment MVP Design

## Scope

Phase 8 builds only a deterministic, testable multi-agent environment. It does
not introduce MAPPO, GNNs, hierarchical actions, expert pretraining, CBF, or
any change under `legacy_hgalo`.

## Interface

`MultiUAVParallelEnv` implements PettingZoo's native `ParallelEnv` interface.
`reset(seed, options)` returns dictionaries of local observations and infos;
`step(actions)` returns observations, scalar rewards, terminations,
truncations, and infos keyed by the live agents. `possible_agents` is fixed at
construction as `uav_0` through `uav_{N-1}`. `state()` exposes a stable,
flat `float32` centralized-critic vector. A small wrapper exposes the same
central state without changing PettingZoo lifecycle semantics.

Reached UAVs remain live until the episode's global terminal condition, but
their `active_mask` becomes false and their motion is zeroed. This keeps
centralized state shape fixed while making completed vehicles inert.

## Configuration and scenario boundary

`configs/env/multi_uav_mvp.yaml` contains all dynamic and reward constants:
`dt`, `max_steps`, horizontal/vertical speed limits, normalized-action flag,
goal radius, severity thresholds, observation neighbour count, and reward
weights. The environment accepts the existing immutable `Scenario`; scenario
loading stays in `multiuav.expert.scenarios` and uses no absolute path in core
environment code.

## Dynamics

`DynamicsModel` is a narrow protocol taking position, desired velocity and
`dt`, returning next position, applied velocity and a boundary-clipped flag.
`SingleIntegrator3D` performs `p_next = p + dt * v`. Action normalization maps
each action component from `[-1, 1]`; XY is jointly limited to the configured
horizontal speed and Z is independently limited to the vertical speed. The
same protocol leaves a later double-integrator replacement localized to
`dynamics.py`.

## Observations and central state

Each local `float32` observation has a documented fixed layout:

1. normalized own position and applied velocity;
2. normalized goal-relative vector and scalar goal distance;
3. local terrain clearance;
4. nearest threat relative XY, signed radial clearance, and height relation;
5. nearest configured neighbours' relative position and velocity, padded with
   zeros;
6. normalized remaining time and a zero high-level-command placeholder.

`state()` concatenates all normalized UAV position/velocity/goal rows, the
active mask, terrain height/clearance summary, padded threat summary, and
pairwise conflict-summary values (minimum separation and current conflict
count). It is therefore independent of observation implementation details.

## Reward, cost, and terminal semantics

Performance reward is the sum of per-UAV progress, one-time goal arrival,
path-efficiency, smoothness, energy-proxy, and time terms. Safety is not folded
into a single collision penalty: each `infos[agent]["safety_costs"]` contains
`inter_uav_collision_cost`, `terrain_violation_cost`,
`threat_violation_cost`, and `boundary_violation_cost`.

All agents terminate when every UAV reaches its goal or an unrecoverable
collision/serious terrain/threat violation/numeric failure occurs. The episode
truncates only at `max_steps`. `infos[agent]["termination_reason"]` is one of
`all_arrived`, `collision`, `terrain_violation`, `threat_violation`,
`numerical_error`, `max_steps`, or `none`.

## Validation and visualization

Unit tests cover the ten explicit phase requirements: reset/step shape,
seeded reproducibility, straight flight, zero action, opposing collision,
terrain and cylinder-threat violation, all-arrived termination, and variable
UAV count. A trajectory recorder wrapper accumulates positions and minimum
separation; its renderer produces 3D, top-view, and separation-curve PNGs.

## Design self-review

- No placeholder requirements remain; all required fields and terminal causes
  have an owner module.
- PettingZoo agent lifecycle, active-mask semantics, and centralized-state
  shape are consistent.
- The scope deliberately stops before all Phase 9 training features.

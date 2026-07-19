# Dynamic World Mainline Design

## Goal

Implement one reproducible research path for multi-UAV planning:

```text
CA-HGALO demonstrations -> behavior-cloning initialization
-> predictive-conflict-graph hierarchical MAPPO -> CBF-QP action filter
-> dynamic-world evaluation
```

Behavior cloning is an initialization stage, not a separate final method.  The only
trainable final policy is hierarchical predictive-graph MAPPO; CBF filters its
actions before the environment executes them.

## Scope

The dynamic-world layer adds moving cylindrical obstacles and delayed/lost
inter-agent communication to the existing static 3-D environment.  Every
component that consumes world state will use the same definitions: environment
stepping, local actor observations, conflict-graph construction, CBF constraints,
evaluation, metrics, experiment records, and smoke tests.

The deliverable also supplies the Phase-14 unified `train.py`, `evaluate.py`,
`run_benchmark.py`, and `run_ablation.py` entry points, reproducible output
directories, scale profiles for 3/5/8/12/16 UAVs, and the required documentation.

## Non-goals

- No fabricated dynamic-obstacle or communication results.
- No new algorithm branch competing with the mainline above.
- A*, RRT*, and ORCA are not implemented in this phase.  The benchmark registry
  will report them as unavailable adapters instead of inventing values.
- The CBF solver remains direct OSQP on CPU.  Neural inference and training use
  the configured PyTorch device, normally CUDA when available.

## World model

### Dynamic obstacles

`DynamicCylinder` is a finite-height vertical cylinder with stable identifier,
radius, height, initial center, and constant three-dimensional velocity.  Its
center at simulation step `k` is `initial_center + k * dt * velocity`.  Scenario
validation requires that its generated trajectory remains inside the terrain
boundary for the configured episode horizon; there is no hidden reflection or
teleportation rule.

The environment owns a `DynamicWorldState`, advances it exactly once per call to
`MultiUAVParallelEnv.step`, and exposes an immutable snapshot after the advance.
Collision and safety-cost calculation treats a UAV inside a cylinder's horizontal
radius and vertical span as a dynamic-obstacle violation.  Local observations
include the nearest configured number of dynamic obstacles as relative position,
velocity, radius, and time-to-closest-approach features.  Empty slots are
zero-filled and paired with a validity mask so fixed policy dimensions remain
stable.

### Communication channel

`CommunicationConfig` has `enabled`, `range`, `delay_steps`, `drop_probability`,
and `max_staleness_steps`.  At each step every active UAV broadcasts its true
kinematic state.  A message outside range is not sent; an in-range message is
dropped using the environment RNG or enqueued for `current_step + delay_steps`.
At delivery the receiver replaces only the sender's previously delivered state.

An actor receives its own true state plus the freshest delivered neighbor states,
their ages, and validity masks.  A missing or too-stale neighbor is represented as
invalid rather than as a current position.  The graph actor and high-level policy
must use this knowledge view, never the environment's hidden global truth.  The
centralized critic may use full state during training under CTDE, but evaluation
and deployed action selection use only delivered communication.

Communication is disabled by default with zero delay and zero loss, preserving
the existing static-environment behavior.  A fixed reset seed reproduces both
motion and all packet outcomes.

## Conflict graph and policy integration

The predictive conflict graph is built from an `AgentKnowledgeState` rather than
from raw environment positions.  Edge CPA features use delivered position and
velocity; an edge is omitted when either endpoint knowledge is invalid.  Node and
edge features carry staleness/validity, allowing hierarchical MAPPO to learn
whether a predicted conflict is based on fresh or old information.

The high-level policy remains responsible for subgoal, priority, delay, and
altitude decisions.  The low-level policy consumes the high-level context and the
communication-aware graph to propose normalized velocity actions.  Existing BC
checkpoints are used only to initialize compatible low-level encoder/actor
weights; training then follows the existing staged low/high/joint hierarchy.

## CBF integration

`CBFConstraintBuilder` receives the same dynamic-world snapshot as the
environment.  For a UAV position `p`, obstacle predicted center `o`, and obstacle
velocity `v_o`, the horizontal relative-velocity barrier is based on
`2(p-o)^T(u-v_o) + alpha*h >= 0`, where `h` is squared clearance minus squared
combined radius.  Vertical overlap gates the obstacle constraint as it does for
static cylindrical threats.  The QP still jointly filters all active UAV actions,
keeps speed bounds, and falls back to the existing emergency policy for
non-solved, non-finite, or time-limited results.

Every filtered step records solver status, correction norm, emergency use, and
dynamic-obstacle constraint count.  CBF remains a safety layer only: it does not
provide observations or modify rewards invisibly.

## Unified experiment interface

`ExperimentSpec` is the single serializable input to all four scripts.  It records
method, seed, device, UAV count, scenario profile, checkpoint paths, component
flags, environment dynamics, communication configuration, and output name.  Each
script accepts at least:

```text
--config --seed --device --num-uavs --scenario --checkpoint --render
--use-expert-pretrain --use-graph --use-hierarchy --use-cbf
```

The method registry contains: straight-line greedy, Python CA-HGALO, rule conflict
coordinator, MAPPO, predictive graph MAPPO, hierarchical graph MAPPO,
BC-initialized hierarchical graph MAPPO, and the full CBF method.  The first three
are controller adapters; learned methods load an explicit compatible checkpoint.
Unavailable external methods return a structured `unavailable` benchmark record.

Scale profiles are defined for 3, 5, 8, 12, and 16 UAVs.  The runner supports
within-distribution and small-to-large zero-shot evaluation, unseen maps, unseen
threat density, dynamic obstacles, and communication impairment.  A configuration
cannot claim a scenario capability that its world model does not enable.

## Results and reproducibility

Each run creates:

```text
outputs/<experiment_name>/
  config.yaml  environment.json  checkpoints/  tensorboard/
  raw_results/  summary.csv  summary.json  figures/  logs/  README.md
```

Raw data keeps every requested seed and every failed episode.  Aggregation reports
mean, sample standard deviation, and 95% Student-t confidence intervals without
dropping outliers.  Required metrics are success rate, collision rate, terrain and
threat violation rates, mean path length, mission time, mean and fifth-percentile
minimum separation, temporal conflict count, energy proxy, decision latency, CBF
intervention rate, CBF mean correction, CBF emergency count, expert gap, and
generalization gap.  Dynamic-obstacle and communication violations are additionally
recorded so their contribution is auditable.

`environment.json` records Git commit, Python, PyTorch, CUDA, GPU, seed, command,
resolved configuration, device, and enabled capabilities.  Documentation in
`docs/training.md`, `docs/evaluation.md`, and `docs/reproducibility.md` describes
the supported commands and caveats.

## Error handling and compatibility

Invalid obstacle trajectories, negative delay, unsupported scale, incompatible
checkpoint method, or a requested GPU without CUDA raise clear errors before an
experiment begins.  Existing static scenarios use disabled dynamic obstacles and
communication; their tests must retain their current behavior.  Rendering is an
optional adapter and will be reported unavailable when no renderer is configured.

## Verification

Tests will cover deterministic obstacle advancement, dynamic collision accounting,
delivered-versus-hidden communication observations, packet loss determinism,
communication-aware graph masking, relative-velocity CBF constraints, QP fallback,
output metadata, per-seed aggregation, and unavailable baseline reporting.  The
final smoke run will load a compatible BC checkpoint, execute graph and hierarchy
inference on CUDA when available, filter actions through CBF, produce raw results
and a figure, then run `pytest`, `ruff check .`, and `mypy multiuav`.

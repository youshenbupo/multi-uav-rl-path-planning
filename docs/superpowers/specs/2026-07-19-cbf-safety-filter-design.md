# Phase-13 CBF Safety Filter Design

## Goal

Filter a simultaneous RL desired-velocity command for the Phase-8
three-dimensional single-integrator environment without putting a QP in PPO
backpropagation. The filter is used only at execution and evaluation time.

## Chosen architecture

At every step, one direct-OSQP problem jointly optimizes the velocity of every
active UAV. This is necessary because the pairwise barrier derivative depends
on both `u_i` and `u_j`. Inactive UAV velocities are fixed to zero and do not
create pair constraints.

`CBFConstraintBuilder` converts the current `Scenario`, positions and active
mask into linear rows. `OSQPSafetyFilter` owns a warm-startable solver and
returns a typed decision record. `EmergencyPolicy` provides a finite,
bounded, conservative output whenever the QP is unavailable, exceeds its
configured wall-time budget, returns a non-solved status, or yields non-finite
data. The raw RL command is never used as a failure fallback.

## Optimization problem

For active controls `u` and nonnegative per-CBF slack `s`, minimize

`0.5 ||u - u_rl||^2 + 0.5 * slack_penalty * ||s||^2`.

Each barrier is expressed as `a*u + alpha*h + s >= 0`. For a pair,
`h_ij = ||p_i-p_j||^2 - d_safe^2` and the row contains
`2*(p_i-p_j)` for `u_i` and its negative for `u_j`. Terrain clearance uses
`h = z - terrain(x,y) - min_clearance` and a finite-difference terrain
gradient. Cylinder safety uses squared horizontal distance outside
`radius + threat_margin` while the vehicle is within the configured vertical
threat influence height. Lower/upper world boundaries use their signed axis
distances as barriers.

OSQP sees only linear constraints. Horizontal speed is represented by a
configurable inscribed regular polygon (16 sides by default), vertical speed
is bounded directly, and every slack is bounded below by zero. The inscribed
polygon is conservative and therefore cannot violate the circular horizontal
speed bound.

## Failure behavior and telemetry

The filter records a status, solve duration, active-row count, maximum slack,
intervention norm, and whether emergency logic was used. A timeout or bad QP
status invokes emergency behavior: safe hover in a non-risk state, otherwise a
bounded sum of repulsive pair/threat/boundary directions plus terrain ascent.
Its output is always clipped to the same horizontal/vertical limits. The
emergency event includes a reason in the decision record and Python logging.

## Integration and verification

The environment remains mathematically unchanged. A small execution adapter
accepts normalized or physical action dictionaries, filters physical desired
velocities jointly, and returns normalized actions where required. This keeps
the safety layer out of the PPO loss graph. Tests cover head-on, crossing,
three-way convergence, terrain, threat, initially unsafe, conflicting
constraints, forced QP failure, and low-risk action preservation. The scenario
script produces readable telemetry without claiming policy performance.

# Technical Decisions

## 2026-07-16: treat recovered HGALO as a read-only legacy baseline

The active MATLAB package is `legacy_hgalo/HGALO_恢复源码`. It contains HGALO
but no standalone CA-HGALO entrypoint. New work remains in
`reinforcement_learning`; any later MATLAB export/wrapper belongs in
`reinforcement_learning/tools/matlab`.

## 2026-07-16: no RL implementation before evaluator validation

The phase-one source audit confirms that the legacy evaluator couples geometry,
cost, synchronization, scheduling, and optional repair. Python implementation
must validate these lower layers against MATLAB fixtures before starting MAPPO.

## 2026-07-16: use the official CUDA 13.0 PyTorch wheel runtime

The development machine has NVIDIA driver 591.74 reporting CUDA 13.1 and an
RTX 5060 Laptop GPU. The PyTorch official CUDA 13.0 wheel index offered
`torch==2.13.0+cu130` and matching `torchvision==0.28.0+cu130`; both import and
use the GPU successfully. No standalone CUDA Toolkit or PyTorch Geometric was
installed.

## 2026-07-16: retain deterministic MATLAB fixture provenance

The regression fixture uses MATLAB R2025a, seed `20260716`, source-level
functions, and explicit metadata. The exporter stores all seven single-UAV
cost terms alongside aggregate cost terms. CA-HGALO output is represented only
by an unavailable marker because replacing it with HGALO would be false data.

## 2026-07-16: keep fixed-path evaluation pure

`MultiUAVEvaluator.evaluate` evaluates trajectories supplied by its caller and
reports strict feasibility; it does not silently repair a path or select a
takeoff schedule. This reproduces the MATLAB objective and conflict metrics
that are directly testable while keeping later environment actions and policy
outputs observable. Optional legacy repair and scheduler modules will have
separate interfaces and regression gates.

## 2026-07-16: make coordination decisions deterministic and auditable

Conflict graphs use explicit zero-based node IDs and deterministic public-edge
ordering. Scheduler tie-breaking preserves the legacy MATLAB column-major
worst-edge orientation so the exported two-UAV fixture delays UAV index 1 by
five seconds. The coordinator remains a rule-based expert module and has no
learned state, GNN, MAPPO, or CBF dependency.

## 2026-07-16: encode expert data without inventing labels or padding graphs

Expert episodes use HDF5 groups, fixed `[T,N,*]` arrays with masks for temporal
data, flattened waypoint arrays with offsets for variable paths, and one HDF5
group per dynamic graph step for variable edge counts. The MATLAB importer
marks high-level labels unavailable when action logs are absent. This preserves
provenance and lets future Python CA-HGALO logs fill the same schema directly.

## 2026-07-16: reconstruct HGALO without relabeling it as CA-HGALO

The recovered MATLAB implementation exposes HGALO operators, not an
independent CA-HGALO runner or output. `multiuav.expert.ca_hgalo.HGALOPlanner`
therefore ports the documented optimizer sequence and applies the audited
Python conflict coordinator once to the final elite. Outputs carry the explicit
`HGALO_PYTHON_COORDINATED_RECONSTRUCTION` algorithm and provenance strings.
This supports reproducible Python expert data while preventing a false
cross-language CA-HGALO validation claim.

## 2026-07-18: use native PettingZoo Parallel API for the environment MVP

The environment uses PettingZoo's simultaneous-action `ParallelEnv` contract
instead of inventing a multi-agent Gym dictionary convention. `state()` is a
separate fixed centralized-critic surface; agent-local observations remain
decentralized. Reached UAVs remain represented by an active mask and accept
inert actions until a global terminal event, preserving stable centralized-state
shape for the later shared-actor training stage.

## 2026-07-18: keep performance reward and safety costs independently observable

The scalar reward contains progress, arrival, efficiency, smoothness, energy,
and time components only. Inter-UAV collision, terrain, threat, and boundary
costs are named entries in `info[agent]["safety_costs"]`. This prevents safety
semantics from being hidden in one large collision penalty and leaves a clean
interface for later constrained-learning or CBF work without adding either in
Phase 8.

## 2026-07-18: use rollout-stable normalized inputs and tanh-squashed Gaussian actions

The basic MAPPO actor is a shared Gaussian network whose samples are mapped
through `tanh` to the environment's normalized action Box. Its log probability
uses the corresponding change-of-variables correction, avoiding the invalid
likelihood semantics of hard clipping a Gaussian sample. Observation statistics
remain fixed during a rollout; the buffer stores the exact normalized tensor
used to generate each action, and statistics update only after collection. This
keeps old and new PPO log probabilities in the same coordinates and prevented
the observed ratio overflow/NaN failure.

## 2026-07-18: make the Phase-9 curriculum discretely reachable

The basic two-UAV MAPPO curriculum uses parallel 25 m missions, a 20-step
horizon, 8 m/s horizontal limit, and a 5 m arrival radius. The radius avoids a
discrete-step fly-by where a directed policy passes within 3--5 m of the goal
without an arrival event. It is a verification curriculum only; it does not
change the Phase-8 general environment defaults or assert an obstacle-avoidance
benchmark.

## 2026-07-18: use dense masked PyTorch graph attention before PyTorch Geometric

Phase 10 represents each padded graph as node tensors `[B,N,D]`, edge tensors
`[B,N,N,E]`, adjacency `[B,N,N]`, and an active mask `[B,N]`. Attention is
implemented directly with tensor projections and score masking. Every active
node receives a forced self-loop before softmax, while inactive output rows are
zeroed. This keeps no-neighbour cases finite, supports variable team sizes, and
preserves permutation equivariance without adding PyTorch Geometric.

## 2026-07-18: keep predictive supervision distinct from PPO

The graph rollout stores positions and episode IDs and derives auxiliary targets
only from later samples of the same episode. Binary conflict and Smooth L1
minimum-distance terms are masked on invalid/self pairs and weighted by two
explicit configuration coefficients. The PPO policy/value/entropy objective is
unchanged; setting both auxiliary coefficients to zero removes the extra
supervision without changing graph policy interfaces.

## 2026-07-18: separate hierarchical PPO time scales at action boundaries

The high coordination actor emits a categorical command only at an active-UAV
boundary and `HierarchicalPolicy` holds that command for exactly `H` low steps.
The low Gaussian velocity actor is conditioned on the held command and its
remaining fraction. Low and high batches never share action likelihoods,
values, advantages, or optimizers. A high record accumulates low rewards as
`sum(gamma_low^k * r_k)` and uses `gamma_low^duration` when bootstrapping GAE;
therefore an early terminal interval receives its true shorter discount rather
than an assumed `H`-step one. This makes stage-wise freezing auditable and
avoids introducing a joint objective until explicitly enabled.

## 2026-07-18: preserve expert uncertainty through BC and transfer only compatible tensors

Phase 12 uses HDF5 action masks as the sole authority for BC supervision.
Unknown high commands stay negative and excluded from categorical loss; an
all-continue vector is only low-policy context, never an invented target.
The HDF5 raw graph is retained for audit, while policy inputs are reconstructed
as the Phase-8 local observation and Phase-10 predictive graph. Node
normalization is fit on the training split only and imported with a transferred
BC encoder so the deployed GraphActor operates in the same coordinates.

BC-to-MAPPO transfers require exact key/shape compatibility and report every
loaded or skipped tensor. The hierarchical low actor has widened node input due
to held high-command context, so its node encoder is intentionally skipped;
compatible graph-attention layers may transfer by explicit prefix remapping.
The BC auxiliary action loss is enabled only with a supplied expert batch and
its own boolean action mask. This avoids accidental use of on-policy samples,
padded UAV rows, or unknown high-level labels as expert data.

## 2026-07-19: solve CBF constraints jointly with direct OSQP outside PPO

For `p_dot = u`, the pairwise barrier derivative contains both `u_i` and
`u_j`; separate per-UAV QPs cannot generally satisfy the same pair row. Phase
13 therefore solves a single active-fleet QP with all physical desired
velocities and one nonnegative slack per CBF row. The filter is an execution
adapter, not a policy-network layer and not a differentiable PPO operation.

OSQP handles linear constraints only. Maximum horizontal speed is therefore a
conservative inscribed regular polygon and maximum vertical speed a box bound.
Terrain gradient is computed from the existing interpolator using finite
differences. A non-solved or over-budget result invokes bounded emergency
behavior instead of the raw requested action; slack makes contradictory or
already unsafe geometry observable rather than silently pretending it is
feasible.

# AAMAS 2027 method and readiness assessment

**Assessment date:** 2026-08-02
**Status:** internally auditable descriptive package; not yet externally
submission-ready and never authorized for external submission.

## Research claim under test

The project tests whether delivered-packet prediction and age-dependent
uncertainty can provide a useful coordination representation under delayed and
lossy communication, and whether the same uncertainty can adapt an online CBF
execution margin in dynamic-obstacle simulation.

The intended mechanism chain is fixed:

> communication staleness, delay, and loss -> neighbour-state prediction
> uncertainty -> uncertainty-aware predictive interaction graph -> coordinated
> action -> uncertainty-adaptive CBF margin -> dynamic-obstacle execution.

The current evidence supports implementation and descriptive evaluation of
that chain. It does not support superiority, formal safety, calibrated
uncertainty, real-time, or robust-generalization claims.

## Information boundary

For receiver `i` and sender `j`, a successfully delivered packet stores
position `p_ij^rec`, velocity `v_ij^rec`, and source step `t_ij^rec`. At step
`t`, the packet age and prediction are

`a_ij = t - t_ij^rec`

`p_hat_ij = p_ij^rec + a_ij * dt * v_ij^rec`.

The scalar robustness bound is

`sigma_ij = gamma_sigma * a_ij`.

An actor has exact current state only for itself. A neighbour is available only
through a delivered packet whose age has not exceeded the staleness limit.
Invalid or expired packets create no neighbour edge and contribute only padded
zeros. The actor never receives the sender's current activity flag, private
observation, or goal. A previously delivered packet persists until staleness
expiry even if the sender later becomes inactive.

`sigma_ij` is a deterministic age proxy. It has not been calibrated against a
physical sensor or motion-error distribution and must not be described as a
probabilistic standard deviation.

## Policy representations

All actors output a normalized three-dimensional desired velocity from a
shared squashed-Gaussian policy.

### MLP MAPPO

The MLP actor consumes its local observation: own state and goal, local terrain
and obstacle features, and up to three delivered-packet neighbour slots. It
does not consume current neighbour truth. During training, its value network
uses the environment's complete centralized state, repeated once per agent.
Thus the MLP arm is CTDE: centralized information affects value learning but
not actor execution.

### GraphMAPPO modes

The graph actor uses a residual encoding of the receiver's own node plus
edge-conditioned attention. Peer node features are deliberately excluded from
attention key/value computation after the actor-isolation repair. The current
GraphMAPPO runner passes the same communication-limited node/edge tensors to
its graph value estimator; it does not inject hidden current neighbour truth
into that critic.

The evaluated 3-UAV graph modes are:

- `mappo`: self loops only and no neighbour edge. This is the historical
  "raw GraphMAPPO" label, but it is not a raw neighbour-state graph.
- `predictive_graph`: delivered neighbour positions are propagated by packet
  age; age is exposed, while the graph uncertainty feature and uncertainty
  risk gain are zero.
- `uncertainty_predictive_graph`: prediction, age, uncertainty, and
  uncertainty-adjusted risk are all active.

For a directed receiver-to-sender edge, the 16 features are predicted relative
position (3), delivered relative velocity (3), distance, CPA time, CPA
distance, separation shortfall, three fixed-zero sender-goal slots, validity,
age, and uncertainty. Candidate edges are retained when the neighbour is
within the 45 m communication radius, has risk-adjusted CPA distance within
12 m, or belongs to the two nearest valid delivered neighbours. Self loops are
always present.

For the uncertainty-aware mode,

`d_CPA^risk = max(0, d_CPA - kappa_sigma * sigma_ij)`.

The configured uncertainty feature scale is 10 and risk gain is 1. Prediction
horizon is 5 steps.

## Training objective

Both MAPPO implementations use shared policies, GAE, and clipped PPO. The core
hyperparameters are learning rate `3e-4`, discount `0.99`, GAE lambda `0.95`,
clip ratio `0.2`, entropy coefficient `0.01`, value coefficient `0.5`, gradient
norm cap `0.5`, minibatch size `64`, and two PPO epochs. Graph models use a
64-dimensional embedding, four attention heads, and two graph layers.

The paper-facing runs request 100,000 transitions. Collection stops only at a
complete synchronous rollout boundary, so retained totals are 100,032 for the
3-UAV arms, 100,160 for the 5-UAV full method, 100,080 for the 5-UAV joint
ablation, and 100,224 for both 8-UAV arms. These are protocol consequences, not
post-hoc training-budget changes.

## Execution-side CBF

The policy action is filtered by a joint CPU OSQP quadratic program using
centralized simulator geometry. It constrains pairwise separation, terrain,
static threats, moving cylinders, world bounds, and an approximate horizontal
speed polygon. Pairwise clearance is enlarged by

`d_safe^CBF = d_safe + min(m_max, kappa_CBF * max(sigma_ij, sigma_ji))`.

For the full method, `kappa_CBF = 0.5` and `m_max = 5`. The optimization
minimizes action correction plus slack cost with slack penalty `100`, maximum
iterations `20,000`, absolute and relative tolerances `1e-4`, and a `0.1 s`
solve-time limit. These are the core-run values; the generic
`configs/safety/cbf.yaml` is not the source of truth for these runs. No safety
parameter may be silently changed for training, evaluation, or replay.

Only an exact OSQP `solved` status received within the time limit is accepted.
Any inaccurate, iteration-limit, timeout, or error outcome invokes a bounded
emergency hover/repulsion controller and retains the full context for replay.
Slack, fallback, and discrete integration mean this implementation is an
execution shield in simulation, not a formal end-to-end or decentralized-CBF
safety guarantee.

## Comparisons and causal limits

The 3-UAV four-arm comparison holds the uncertainty-adaptive CBF settings fixed
across MLP, self-loop graph, predictive graph, and uncertainty-aware graph.
Consequently, its predictive-versus-uncertainty-aware contrast isolates graph
use of the age bound more closely, but it does not remove uncertainty from the
complete system.

The independently trained 5/8-UAV ablation keeps packet-age prediction but sets
both graph uncertainty risk gain and the CBF uncertainty margin gain/cap to
zero. It is therefore a joint uncertainty-pathway ablation. It cannot identify
whether any difference is caused by graph risk, the CBF margin, or their
interaction. Graph-only and CBF-only independent arms remain an open causal
gate.

## Evaluation evidence

Every eligible arm has five independently trained seeds (`20260719`--
`20260723`). Each final checkpoint has 20 retained episodes in each of six
conditions: nominal, delay only, loss only, dynamic obstacle only, combined,
and an out-of-distribution communication/obstacle condition. Eligible roots
contain 30 seed-scenario cells and 600 raw episode records per arm.

The frozen descriptive rule first averages the 20 episodes within a
seed-scenario cell and then reports the mean and sample standard deviation over
the five trained seeds. The 5/8-UAV paired delta is full minus independently
trained no-uncertainty for the same seed. No p-values, confidence intervals,
rankings, or multiple-comparison decisions are present.

The values are mixed. For example, full-minus-no-uncertainty success deltas are
`-0.20` for 5-UAV delay-only, `-0.40` for 8-UAV delay-only, and `-0.09` for
8-UAV loss-only; collision deltas are zero for all six 5/8-UAV scenarios, and
return and minimum-separation deltas change sign. These examples prevent a
one-sided narrative; the complete JSON summaries are authoritative.

## Readiness decision

Completed:

- post-isolation actor-boundary repair and test coverage;
- five eligible seeds for every currently reportable arm;
- complete six-scenario evaluation matrices and raw JSONL;
- all-source CBF fallback replay with retained source/context records;
- independently validated descriptive aggregation;
- an explicit decision to keep the already viewed five-seed package
  descriptive and add no retrospective hypothesis tests;
- full primary-source review of DACOM and DHCG and a source-tiered comparison
  with graph-CBF and decentralized-UAV prior work.

Open before external submission readiness:

1. Train graph-only and CBF-only uncertainty ablations if the manuscript seeks
   component-level causal claims.
2. Obtain and fully read the closed IEEE UAV paper before making paper-specific
   communication or safety comparisons, and broaden systematic literature
   coverage before any novelty or "first" language.
3. Build every manuscript table and figure directly from eligible raw roots,
   keeping all six scenarios and unfavorable cells visible.
4. Resolve or explicitly waive the unrelated legacy MATLAB inventory drift
   after a separate ownership audit.
5. Resolve the portable-report runtime layout blocker or use another fully
   validated delivery surface; never distribute the failed HTML attempts.
6. Perform a final manuscript-to-artifact audit. External submission requires
   explicit user authorization and is outside the current task.

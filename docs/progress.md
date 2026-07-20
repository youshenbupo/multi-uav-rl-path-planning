# Project Progress

## Phase 1 — MATLAB CA-HGALO/HGALO audit (completed 2026-07-16)

Source evidence was catalogued in `legacy_inventory.md` and
`matlab_python_mapping.md`. No MATLAB files were modified, no MATLAB job was
run, and no reinforcement-learning, GNN, or CBF implementation was added.

The phase-one stop gate is active. Do not start phase two until explicitly
requested.

## Phase 2 — Python environment and project scaffold (completed 2026-07-16)

Created the `multiuav_rl` Conda environment with Python 3.11.15, a pinned
`environment.yml`, the required package/directory structure, static-tool
configuration, and `scripts/check_environment.py`. Verified torch 2.13.0 with
the CUDA 13.0 runtime can use the NVIDIA GeForce RTX 5060 Laptop GPU.

No path-planning, MAPPO, GNN, or CBF algorithm was implemented. The phase-two
stop gate is active.

## Phase 3 — MATLAB regression reference export (completed 2026-07-16)

Executed the deterministic MATLAB exporter under MATLAB R2025a and wrote
`data/regression/matlab/reference_cases.mat` plus `metadata.json`. The export
covers geometry, S1--S4 parameters, evaluation terms, conflict scheduling, and
space-repair trajectories; a Python inspection script reports field names,
shapes, types, numeric ranges, and NaN/Inf counts.

CA-HGALO algorithm-level output remains explicitly unavailable because no
CA-HGALO source entrypoint exists in the restored MATLAB baseline. No Python
algorithm migration was started. The phase-three stop gate is active.

## Phase 4 - typed geometry and strict evaluator (completed 2026-07-16)

Implemented typed scenario, terrain, threat, mission, trajectory, cost, and
constraint models; MATLAB-compatible terrain interpolation, spherical-step
geometry, resampling, turn and cylinder distances, clearance, and finite-
difference speed; plus `MultiUAVEvaluator.evaluate` for the seven objective
components and strict feasibility result.

The complete test suite passes with direct MATLAB fixture comparisons for the
terrain grid/queries, resampling, cylinder distances, single-UAV seven-term
cost, and synchronized temporal conflict metrics. Optional MATLAB path repair
and conflict-aware scheduling remain deferred, and no MAPPO, GNN, or CBF code
has been added. The phase-four stop gate is active.

## Phase 5 - rule-based cooperative conflict handling (completed 2026-07-16)

Implemented a deterministic temporal conflict graph, bounded multi-round
takeoff-delay scheduler, local Gaussian spatial repairer, and
`ConflictAwareCoordinator`. Every action carries an auditable log; exhausted
delay budgets return an explicit unresolved result.

MATLAB regression covers the synchronized conflict edge, squared edge weight,
delay target and amount, repair-center convention, repair conflict trend, and
fixed-path cost direction. The MATLAB exporter now retains before/after
fixed-path repair costs. No GNN, MAPPO, or CBF code has been added. The
phase-five stop gate is active.

## Phase 6 - CA-HGALO expert episode data pipeline (completed 2026-07-16)

Implemented MATLAB-reference import, versioned HDF5 episode groups, flattened
variable-length waypoints, fixed-shape time series, per-step variable graph
groups, speed-clipped low-level actions, high-level label masks, logs, reader,
validator, converter, and 3D/top-view visualizer. MATLAB-only episodes keep
their high-level masks false because the available source exports no action
histories; Python rule-coordinator replays can add attributable labels.

The generated HDF5 sample passes finite/shape/time/speed/endpoint/graph and
Python evaluator cost replay validation, and its 3D and top-view trajectory
plot was rendered successfully. No GNN, MAPPO, or CBF code has been added. The
phase-six stop gate is active.

## Phase 7 - HGALO reconstruction expert planner (completed 2026-07-16)

Implemented the available MATLAB HGALO operators in `multiuav.expert`: chaotic
initialization, spherical candidate encoding, GWO guidance, ALO local search
with Lévy steps, bounded projection, feasibility-oriented geometry repair,
greedy survivor selection, global elite tracking, and final rule-based
temporal/spatial conflict coordination. YAML exposes S1--S4 plus the recovered
MATLAB defaults. Every run stores raw/final trajectories, evaluator components,
graphs, scheduling/repair logs, high-level labels, and convergence history.

Deterministic operator/encoding tests, a 5-seed trial, and a 30-seed S1 trial
passed. The 30-seed result has best cost `4009.8338`, mean cost `259041.7705`,
strict-success rate `76.67%`, mean minimum separation `75.9837`, and zero mean
temporal conflicts; all six five-episode HDF5 shards pass validator replay.
This remains a Python reconstruction: the restored MATLAB archive has no
independent CA-HGALO entrypoint or CA-HGALO regression output. No RL, GNN, or
CBF implementation has been added. The phase-seven stop gate is active.

## Phase 8 - multi-UAV reinforcement-learning environment MVP (completed 2026-07-18)

Implemented a native PettingZoo Parallel API environment with Gymnasium Box
spaces, a configurable three-dimensional single-integrator model, normalized
velocity actions, separate horizontal/vertical limits, and world-boundary
clipping. It exposes fixed-layout local observations and a finite centralized
critic state containing UAV rows, active masks, terrain/threat summaries, and
conflict summary fields.

Performance reward components (progress, arrival, efficiency, smoothness,
energy, time) are recorded separately from collision, terrain, threat, and
boundary safety costs. Explicit global terminal reasons cover all-arrived,
collision, severe terrain/threat violation, numerical error, and max-step
truncation. A trajectory recorder renders 3D, top-view, and minimum-separation
figures.

The 10 required environment tests and PettingZoo `parallel_api_test` pass,
alongside the existing suite and static checks. No MAPPO, GNN, conflict
prediction, hierarchical policy, expert pretraining, or CBF has been added.
The phase-eight stop gate is active.

## Phase 9 - basic MAPPO baseline (completed 2026-07-18)

Implemented a parameter-sharing tanh-squashed Gaussian actor for decentralized
execution, centralized critic, fixed-shape on-policy buffer, GAE, PPO ratio and
value clipping, entropy regularization, gradient clipping, configurable reward
normalization, and serializable observation statistics. The rollout stores the
exact normalized observations used during sampling; statistics update only
between rollouts so PPO ratios remain finite and semantically consistent.

The reproducible empty two-UAV curriculum (`seed=20260718`, 32,768
transitions) improved deterministic evaluation return from `-1.4780` to
`52.1214` and success rate from `0.0` to `1.0`, with zero collision rate. The
same baseline with one cylinder improved return from `-1.6700` to `52.2801`
and success from `0.0` to `1.0`, again with zero collision rate. Checkpoint
reload reproduced both evaluations. TensorBoard records the requested episode,
safety, loss, KL, clipping, explained-variance, and FPS metrics.

No GNN, conflict prediction, hierarchical policy, expert pretraining, or CBF
has been introduced. The phase-nine stop gate is active.

## Phase 10 - predictive spatiotemporal conflict graph (completed 2026-07-18)

Implemented a pure-PyTorch dense predictive conflict graph with configurable
communication/risk radii, top-k neighbourhoods, current-distance and CPA-risk
edges, self-loops, and inactive-node masking. Each directed edge records
relative position/velocity, current distance, finite-horizon CPA time and
distance, predicted shortfall, relative goal direction, communication, and
active-pair state. The masked multi-head graph attention encoder uses edge
conditioning, residual paths, LayerNorm, and safe self-attention for nodes with
no neighbours.

GraphMAPPO supports shared graph actors, centralized masked graph critics,
variable agent counts, and an independently weighted auxiliary edge head for
future conflict classification and minimum-distance regression. Its rollout
labels exclude cross-reset futures. The 12 graph-specific tests cover CPA,
parallel stability, sparse masking, no-neighbour finiteness, variable `N`,
attention shapes, permutation equivariance, PPO/auxiliary losses, checkpoints,
and the 3/5/8-UAV runner.

The required seed-matched 3x3 comparison ran at 3,072 requested transitions
per cell (5-UAV cells reached 3,200 due to whole-rollout increments). Results
are preserved under `data/rl/graph_comparison_seed_matched`. At this small,
single-seed budget no stable method ranking is claimed: all cells have zero
success rate; current-distance graphs improved the 3- and 5-UAV return/safety
measurements, while 8-UAV predictive and distance graphs still collided in
evaluation. No hierarchy or CBF has been introduced. The phase-ten stop gate is
active.

## Phase 11 - hierarchical multi-agent policy (completed 2026-07-18)

Implemented a graph-conditioned two-time-scale MAPPO policy. The high policy
selects a discrete coordination command for each active UAV every configurable
`H` low steps and holds it through the interval; the low actor then receives
local node features plus command one-hot encoding, remaining hold fraction, and
reserved local-subgoal fields before emitting continuous normalized velocity.
Both levels retain separate log probabilities, values, GAE, PPO optimizers, and
checkpoint state. High returns and bootstrap discounts use each action's actual
duration, including early episode termination.

Stage A uses a fixed predictive-conflict rule coordinator and updates only the
low policy. Stage B restores that checkpoint, freezes the low policy, and
updates only the learned graph high policy. Stage C is a guarded optional joint
path. The YAML ablates the high policy, local subgoal, priority, delay, and
altitude fields; the first trainable version keeps continuous coordination
fields zero-valued by default.

Short CPU smoke runs are stored under `data/rl/hierarchical_stage_a` and
`data/rl/hierarchical_stage_b`: Stage A completed 384 transitions and two
updates (final low policy loss `-0.04769`); restored Stage B completed another
384 transitions and two high-only updates (final high policy loss `0.12750`).
Checkpoint restoration then collected 64 low samples and 26 high samples.
These runs verify the staged execution path only, not task success,
generalization, or safety. The phase-eleven stop gate is active.

## Phase 12 - expert behavior cloning and MAPPO initialization (completed 2026-07-18)

Implemented strict-mask HDF5 loading for the six validated expert shards.
Episode-level deterministic splits prevent leakage; loader supervision retains
variable UAV active masks, source directed-edge masks, low action masks, and
unavailable high labels as `-1` rather than reclassifying them as `continue`.
Policy features reconstruct Phase-8 local observations and the Phase-10
15-feature predictive graph from each expert state.

Separate graph BC policies train a normalized low-level desired velocity head
(MSE, direction-cosine, and speed-magnitude terms) and a high-level nine-class
maneuver/optional continuous-head policy. A one-epoch CPU run using seed
`20260718` wrote all five required artifacts to `data/rl/behavior_cloning`:
training low loss `0.1753701`, validation low loss `0.0569111`, and six
authoritative high labels. The validation high classification metric is null:
that split has no authoritative high label.

Closed-loop evaluation of that low checkpoint over its seven episode-disjoint
test records produced arrival rate `0.0`, collision rate `0.0`, minimum
separation `221.3437`, total mean path length `222.8935`, and expert path
deviation `540.3143`. These are smoke measurements, not a performance claim;
zero arrivals and large deviation show that one epoch is not a useful trained
controller.

Graph MAPPO and the hierarchy now load only auditable shape-compatible BC
encoder tensors, reset PPO optimizers to the configured fine-tuning rate, and
freeze/release transferred encoders by update count. Graph MAPPO accepts an
optional correctly masked expert-action auxiliary loss. A 64-transition CPU
BC-to-Graph-MAPPO smoke run completed one update with the BC checkpoint,
two-update freeze schedule, and PPO LR `1e-4`; it verifies the actual entry
path but is far too short to establish learning. The five-arm YAML interface
validates random, low-only, high-only, full-BC, and BC+MAPPO settings. No CBF
has been added. The phase-twelve stop gate is active.

## Phase 13 - CBF safety filter (completed 2026-07-19)

Implemented a joint direct-OSQP safety QP for the Phase-8 three-dimensional
single integrator. At each execution step it minimizes distance from the RL
desired velocity plus per-row slack penalty, while enforcing CBF rows for
active UAV-pair separation, terrain clearance, vertical-gated cylindrical
threat clearance, and all world bounds. A conservative 16-sided polygon bounds
horizontal velocity; vertical velocity uses explicit bounds. The solver is
warm-started when the state dimension is unchanged and rejects time-limit,
non-solved, non-finite, or exceptional results.

The returned decision includes requested/safe velocities, intervention norm,
active constraint count, maximum slack, solver status/time, and an emergency
fallback flag. Failure never passes through raw RL velocity: the fallback
hovers when safe or applies bounded pair/threat/boundary repulsion and terrain
ascent. The normalized-action adapter is explicit and does not enter PPO
backpropagation.

`scripts/test_cbf_scenarios.py --scenario all` passed all nine requested
deterministic cases: head-on, crossing, three-way convergence, terrain,
cylindrical threat, initially unsafe state, multiple conflicts, forced QP
infeasibility, and low-risk preservation. The slowest constructed multiple-
conflict QP took approximately `0.0313 s`, below the checked-in `0.1 s` limit.
This is execution-path evidence, not a guarantee of safety for an arbitrary
learned policy or an initially unrecoverable state. The phase-thirteen stop
gate is active.

## Phase 14 - dynamic-world mainline (implementation completed 2026-07-19)

The environment now has deterministic constant-velocity cylinders and seeded
range/delay/drop/staleness communication. Actor graph construction uses only
delivered neighbour knowledge while centralized training state retains truth;
the CBF adds moving-cylinder relative-velocity rows. The hierarchy training
entrypoint enables these world semantics and the execution-side CBF.

The unified experiment kernel preserves each raw seed episode, 95% Student-t
summary intervals, CUDA selection metadata, and per-episode trajectory PNGs.
`evaluate.py --checkpoint` now executes compatible high- and low-level
hierarchy weights in this dynamic world rather than substituting a rule policy.
A CUDA smoke command resolved the local RTX 5060 device and wrote its output
artifacts. Its no-checkpoint controller is explicitly labelled a semantic smoke
controller, not a learned result. The full component-removal matrix is declared
in an ablation manifest: each arm requires its own checkpoint and missing
artifacts are reported unavailable instead of receiving copied metrics.

A separate CUDA run loaded the existing BC low/high artifacts, trained the
dynamic hierarchy for one 12-transition update, then restored the resulting
checkpoint through `scripts/evaluate.py` for a bounded three-step episode.
The checkpoint evaluator resolved CUDA and wrote raw metrics plus a PNG. Its
success rate was `0.0` and the CBF recorded two emergency fallbacks, so this is
strictly an end-to-end execution check, not a performance, safety, or learning
claim. The implementation phase is complete; sufficiently long multi-seed
training and comparison runs remain future experimental evidence, not missing
software functionality.

## AAMAS 2027 - uncertainty-calibrated coordination (implementation baseline 2026-07-19)

The active paper mechanism now treats a valid delayed packet as an immutable received state
plus an actor-side dead-reckoned position and deterministic age-growing uncertainty bound.
Invalid packets remain absent from actor observations and graph edges. Local neighbour rows
now include relative predicted position/velocity, packet age, and uncertainty; predictive
graph edges add an uncertainty feature and use uncertainty-tightened CPA risk for conflict
adjacency.

The simulated joint CBF still builds geometry rows from centralized truth, but now increases
the inter-UAV separation margin from the worst valid directed communication uncertainty. This
is recorded as a centralized risk-aware shield, not a decentralized CBF guarantee.
`docs/aamas2027_research_brief.md` fixes the research question, equations, required evidence,
and claims prohibited before multi-seed experiments.

The complete regression suite passed `129` tests, Ruff, and mypy in the `multiuav_rl`
environment. A fresh `--device cuda` no-BC 12-transition smoke run completed one real PPO
update and persisted uncertainty settings in
`outputs/aamas_uncertainty_cuda_smoke_v2/environment.json`. It reported an OSQP
maximum-iteration emergency fallback, so the run is plumbing evidence only and highlights a
CBF tuning/diagnosis task before formal training.

## AAMAS 2027 - CBF auditability and numerical diagnosis (2026-07-20)

Training outputs now distinguish `hierarchical_training` from semantic-smoke and checkpoint
controllers, record the training command, source-config SHA-256, Git revision, resolved CBF
configuration, and JSON telemetry for every CBF decision. The telemetry retains solver-status
counts, solve times, iteration counts, intervention/fallback rates, and the CBF-relevant state
of each emergency event. `scripts/diagnose_cbf_failure.py` independently reconstructs a retained
dynamic-world emergency event and replays the direct OSQP solve with explicit numerical settings.

The earlier four-decision CUDA smoke fallback was reproduced from its saved artifact: with the
previous slack penalty of 1000 the QP reached 20,000 iterations with nontrivial residuals. The
same event solved with a penalty of 100 in 7,100 iterations. The mainline hierarchical and smoke
YAML files therefore set `cbf_slack_penalty: 100.0` while retaining the explicit 20,000 iteration
cap. A fresh CUDA 12-transition training smoke on the same semantics recorded 4/4 solved CBF
decisions, zero emergency fallbacks, maximum 7,950 iterations, and maximum CPU QP time about
0.0095 seconds. This validates traceability and a local numerical improvement only; multi-seed,
long-horizon and stress-scenario fallback analysis remains mandatory before any safety claim.

## AAMAS 2027 - dynamic GraphMAPPO comparison executor (2026-07-20)

The ordinary raw-packet graph, predictive graph without uncertainty, and uncertainty-aware
predictive graph now share one GraphMAPPO dynamic-world executor. Each arm uses the same seeded
three-UAV scenario, delayed/lossy communication, moving cylinder, rewards, action bounds and
execution-side CPU CBF; only graph-state prediction, information-age and uncertainty inputs
change. `configs/rl/dynamic_graph_baseline.yaml` is the common long-run protocol and
`configs/experiments/dynamic_graph_smoke.yaml` is a bounded plumbing configuration.

The complete uncertainty arm completed a CUDA smoke run with one PPO update. Its retained CBF
telemetry reports four solved decisions and zero fallbacks. This verifies the fair-executor path,
not relative policy quality. At that point, MLP-MAPPO had not yet been attached to the dynamic
executor; the subsequent entry records that adapter separately.

## AAMAS 2027 - dynamic MLP-MAPPO control arm (2026-07-20)

The non-graph baseline now uses the original shared Gaussian MLP actor and centralized MLP critic
with the same three-UAV dynamic-world, delayed/lossy communication, normalized action interface,
and execution-only CBF configuration as the graph arms. Its new common long-run configuration is
`configs/rl/dynamic_mappo_baseline.yaml`. A one-update CUDA smoke ran through the actual MLP
policy and saved CBF training telemetry with four solved training decisions and zero training
fallbacks. The evaluation episode stream separately recorded a `solved inaccurate` CBF fallback;
evaluation metrics now include CBF intervention rate, fallback rate, and mean QP time for both MLP
and graph methods. These are diagnostic smoke observations only, and preclude any safety claim.

The retained MLP evaluation event was replayed with
`scripts/diagnose_cbf_failure.py --telemetry-key final_evaluation_cbf`.
Raising the 20,000-iteration cap to 100,000 reached the 0.1-second time limit,
while a slack penalty of 10 solved only by accepting a large slack (about 75).
The mainline penalty remains 100; no numerical setting was silently relaxed to
hide the fallback. The event is now a reproducible candidate for the paper's
QP-failure analysis.

## AAMAS 2027 - uniform core-checkpoint evaluator (2026-07-20)

The four paper-facing learned arms now have a common checkpoint evaluator:
MLP-MAPPO and all GraphMAPPO variants execute their actual saved actor through
the same seeded dynamic environment, CPU-side CBF shield, per-episode JSONL
schema, and seed-level statistical summary. Every raw record identifies the
controller family, exact checkpoint, checkpoint step, termination reason,
communication condition, dynamic-obstacle count, path/energy/separation/conflict
metrics, decision latency, and full CBF telemetry. CBF intervention, emergency
fallback, and QP time are aggregated by CBF decision count rather than by a
coarser episode indicator.

`scripts/evaluate_core_checkpoint.py` evaluates exactly one independently
trained checkpoint and refuses a missing checkpoint; it never falls back to a
goal controller. The six paper scenarios are executable semantics: nominal,
delay-only, loss-only, dynamic-only, combined, and an OOD condition that
increases delay, loss/staleness/uncertainty growth, and moving-obstacle speed.
`scripts/plan_core_evaluation.py` materializes the 120 independent
method/seed/scenario evaluation commands, each tied to its matching training
checkpoint. The associated tests use random saved MLP and graph actors as
plumbing fixtures only; no performance claim or core experiment result exists
yet.

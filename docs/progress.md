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

## AAMAS 2027 - CBF live-telemetry diagnosis gate (2026-07-20)

The first attempted formal MLP seed was deliberately stopped after its initial
untrained-policy evaluation emitted repeated `solve_time_limit` fallbacks. Its
partial checkpoint directory is retained with `ABORTED.json` and is explicitly
ineligible for the five-seed comparison. This was a diagnosis gate, not an
experiment result.

Training now atomically writes `live_training_telemetry.json` after every PPO
update. It retains training CBF telemetry plus initial and interval-evaluation
telemetry, so an interrupted job preserves every emergency event and replay
context. A fresh controlled CUDA diagnostic of 9,216 transitions with the same
seed found 16/160 initial-evaluation fallbacks (10%), 0/3,072 training-rollout
fallbacks, and 0/160 final-evaluation fallbacks. The initial failures arise
from the untrained actor's QP requests and are retained; no slack penalty,
solver limit, or numerical tolerance was relaxed to suppress them. This is a
numerical/telemetry diagnosis only, not a learned-policy performance result.

## AAMAS 2027 - checkpoint evaluation schema repair (2026-07-20)

The first full 3-UAV MLP-MAPPO seed (`20260719`) completed 100,032 transitions
on CUDA and produced its final checkpoint. Its training telemetry retained one
emergency fallback across 33,344 CBF decisions, 16 fallbacks in the separate
initial untrained-policy evaluation, and zero in the final interval evaluation.
These are retained diagnostic telemetry, not performance results.

Attempting the required six-scenario evaluator exposed two reproducible
checkpoint portability defects: changing from a dynamic to a nominal scenario
changed the learned actor/critic feature widths, and CUDA checkpoint loading
moved CPU RNG-state tensors to GPU. The evaluator now pins both local-observation
and centralized-state dynamic-obstacle capacities to the training schema while
allowing scenario semantics to remove the actual obstacle. MAPPO and GraphMAPPO
loaders now restore CPU and CUDA RNG state from CPU tensors after CUDA weight
loading. Regression tests cover nominal evaluation for both learned families
and CUDA map-location loading for both checkpoint formats.

The first repaired nominal evaluation was written to
`outputs/core_3uav_evaluations/core_3uav_mlp_mappo_seed_20260719_nominal_schemafix_rngfix_rerun2`.
Its single-seed, single-condition metrics are retained solely as raw protocol
evidence; the remaining five scenarios, four additional seeds, and all three
other comparison arms are required before any comparative statement.

## AAMAS 2027 - related-work evidence gate (2026-07-20)

`docs/related_work_matrix.md` now records the first source-verified entries
without turning titles into novelty claims. The AAAI landing-page abstract for
DACOM and the deposited IJCAI abstract for DHCG were reviewed; the AAMAS 2026
neural graph-CBF paper is retained as metadata-only pending primary-paper
inspection. The current paper text may describe a testable uncertainty-prediction
and CBF-margin design, but cannot claim to be the first combination until the
remaining primary-paper comparisons are complete.

## AAMAS 2027 - executable OOD obstacle perturbation (2026-07-20)

The six-scenario protocol's OOD obstacle-speed setting was initially
non-executable: the old constant-velocity trajectory ran from y=50 to y=110 at
1.5x speed and correctly failed the world-boundary validator. The scenario
generator now derives the crossing obstacle's initial y-coordinate from the
fixed 20-step horizon and a terminal y=90. Thus the in-distribution 1.0x case
is unchanged (y=50 to y=90), while the 1.5x OOD case is a valid faster y=30 to
y=90 trajectory. A regression test instantiates the OOD environment and checks
its velocity and endpoint. The prior failed OOD evaluator directory is retained
as invalid; the repaired raw result uses the `_trajectoryfix_rerun1` identity.

## AAMAS 2027 - third independent MLP seed retained (2026-07-22)

The third 3-UAV MLP-MAPPO seed (`20260721`) completed 100,032 CUDA transitions
and produced `mappo_final.pt`. Its retained training telemetry has 3 emergency
fallbacks in 33,344 CBF decisions, 11 initial untrained-policy-evaluation
fallbacks, and zero final-evaluation fallbacks. Replaying one retained training
event at the unchanged mainline slack penalty of 100 reached the 20,000-iteration
limit with status `solved inaccurate`, zero slack, and an emergency fallback.
This is retained as a numerical failure case; no solver limit, slack penalty, or
tolerance was changed. Three of five MLP seeds are now trained, but no main
comparison or statistical conclusion exists until the remaining seeds, methods,
and six-scenario evaluations are complete.

## AAMAS 2027 - fourth independent MLP seed retained (2026-07-22)

After confirming that no prior train/evaluation process was alive, the fourth
independent 3-UAV MLP-MAPPO training job completed on CUDA.  The exact command
was `D:\anaconda3\envs\multiuav_rl\python.exe scripts/train_mappo.py --config
configs/rl/dynamic_mappo_baseline.yaml --device cuda --seed 20260722 --num-uavs
3 --total-steps 100000 --output-dir outputs/core_3uav/core_3uav_mlp_mappo_seed_20260722`.
It ran at Git revision `7749876e89e0a487da43df2f46f03380e34d9039`, with
`configs/rl/dynamic_mappo_baseline.yaml` SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46` and the
verified `multiuav_rl` CUDA runtime (PyTorch `2.13.0+cu130`, RTX 5060 Laptop
GPU).  OSQP CBF remained CPU-side under the unchanged mainline configuration.

The run reached 100,032 transitions (1,042 PPO updates) and produced
`outputs/core_3uav/core_3uav_mlp_mappo_seed_20260722/checkpoints/mappo_final.pt`.
Its retained raw records are `summary.json`, `live_training_telemetry.json`,
`launcher_stderr.log`, and all checkpoint/tensorboard files in that same output
directory.  CBF telemetry records 1 emergency fallback in 33,344 training
decisions, 13 in 160 initial untrained-policy evaluation decisions, and 0 in
96 final-evaluation decisions.  These are execution and traceability facts,
not performance, safety, or comparative evidence.  The seed still needs all
six 20-episode uniform checkpoint evaluations, and every retained fallback
must be replayed and summarized before interpreting any aggregate statistic.

The next concrete action is the serial, same-protocol CUDA training of seed
`20260723`; GraphMAPPO training remains blocked until the fifth MLP seed and
the MLP unified evaluation matrix are complete.

## AAMAS 2027 - fifth independent MLP seed retained (2026-07-22)

The fifth independent 3-UAV MLP-MAPPO training job completed serially after
seed `20260722` exited; no MLP jobs were run concurrently.  The exact command
was `D:\anaconda3\envs\multiuav_rl\python.exe scripts/train_mappo.py --config
configs/rl/dynamic_mappo_baseline.yaml --device cuda --seed 20260723 --num-uavs
3 --total-steps 100000 --output-dir outputs/core_3uav/core_3uav_mlp_mappo_seed_20260723`.
It used Git revision `7749876e89e0a487da43df2f46f03380e34d9039` and the same
unchanged configuration SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46` as the
preceding formal MLP seeds.  Learned-network execution used the verified CUDA
runtime; the OSQP CBF remained CPU-side.

The run reached 100,032 transitions (1,042 PPO updates) and produced
`outputs/core_3uav/core_3uav_mlp_mappo_seed_20260723/checkpoints/mappo_final.pt`.
The raw `summary.json`, `live_training_telemetry.json`, launcher logs,
checkpoints, and TensorBoard data are retained in its output directory.  The
saved CBF telemetry records zero emergency fallbacks in 33,344 training
decisions, 160 initial untrained-policy-evaluation decisions, and 88 final
evaluation decisions.  This verifies completion and traceability for the fifth
independent MLP training artifact only; it neither establishes a low fallback
rate nor a learned-policy or safety result.

All five prescribed MLP checkpoints now exist.  The next concrete action is to
run only the missing cells of the uniform six-scenario, 20-episode evaluator
for each checkpoint, retaining every JSONL attempt identity and excluding the
already documented incomplete seed-20260719 evaluator directories from any
later aggregation.  GraphMAPPO work remains deferred until that MLP matrix and
the CBF fallback replay summary are complete.

## AAMAS 2027 - seed 20260720 unified MLP evaluation completed (2026-07-22)

The checkpoint
`outputs/core_3uav/core_3uav_mlp_mappo_seed_20260720/checkpoints/mappo_final.pt`
now has one validated 20-episode evaluation artifact for every fixed scenario:
`nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, and
`ood_communication_obstacle`.  The five newly required calls used the same
`evaluate_core_checkpoint.py --family mappo --config
configs/rl/dynamic_mappo_baseline.yaml --seed 20260720 --num-uavs 3 --episodes
20 --device cuda` contract at Git revision
`7749876e89e0a487da43df2f46f03380e34d9039`, varying only `--scenario` and the
matching `core_3uav_mlp_mappo_seed_20260720_<scenario>` experiment name.  The
CUDA policy actor and CPU OSQP CBF path were retained unchanged.

For each of the six directories under `outputs/core_3uav_evaluations/`,
`summary.json` exists and `raw_results/seed_20260720.jsonl` contains exactly 20
records.  Those JSONL files, runtime telemetry, configuration snapshots, and
summaries are the verification evidence.  This completes an evaluation row,
not a statistical result: it proves the protocol ran and retained raw data, but
does not prove MLP performance, safety, or generalization.  The next concrete
action is to execute the six same-contract evaluations for seed `20260721`;
the complete five-seed matrix and fallback replay remain required before any
aggregation or GraphMAPPO run.

## AAMAS 2027 - seed 20260721 unified MLP evaluation completed (2026-07-22)

The independently trained checkpoint
`outputs/core_3uav/core_3uav_mlp_mappo_seed_20260721/checkpoints/mappo_final.pt`
was evaluated serially in all six fixed scenarios with the unchanged uniform
contract: `evaluate_core_checkpoint.py --family mappo --config
configs/rl/dynamic_mappo_baseline.yaml --seed 20260721 --num-uavs 3 --episodes
20 --device cuda`, Git revision `7749876e89e0a487da43df2f46f03380e34d9039`.
The matching `core_3uav_mlp_mappo_seed_20260721_<scenario>` directories retain
the six scenario identities, CUDA actor execution, and CPU OSQP CBF protocol.

Each directory has `summary.json` and exactly 20 records in
`raw_results/seed_20260721.jsonl`; runtime telemetry and config/environment
snapshots are retained alongside them.  This verifies the row's completeness
and raw-result preservation only.  It does not support a seed-level or
five-seed performance, safety, or generalization claim.  Next, run the six
same-contract evaluations for checkpoint seed `20260722`, then seed
`20260723`, before any aggregation or GraphMAPPO work.

## AAMAS 2027 - seed 20260722 unified MLP evaluation completed (2026-07-22)

The independently trained checkpoint
`outputs/core_3uav/core_3uav_mlp_mappo_seed_20260722/checkpoints/mappo_final.pt`
now has six serial, 20-episode evaluations under the unchanged MLP checkpoint
contract at Git revision `7749876e89e0a487da43df2f46f03380e34d9039`.  Each
command used the same actor/CBF split and
`evaluate_core_checkpoint.py --family mappo --config
configs/rl/dynamic_mappo_baseline.yaml --seed 20260722 --num-uavs 3 --episodes
20 --device cuda`, with only the fixed scenario and matching experiment name
changed.

All six `core_3uav_mlp_mappo_seed_20260722_<scenario>` directories under
`outputs/core_3uav_evaluations/` have `summary.json` and exactly 20 records in
`raw_results/seed_20260722.jsonl`; their runtime telemetry and configuration
snapshots remain retained.  This is verified raw-data completeness, not an
outcome claim.  Seed `20260723` is the final missing MLP evaluation row; run it
before fallback aggregation or any GraphMAPPO training.

## AAMAS 2027 - five-seed MLP six-scenario evaluation matrix complete (2026-07-22)

The final checkpoint
`outputs/core_3uav/core_3uav_mlp_mappo_seed_20260723/checkpoints/mappo_final.pt`
was evaluated in all six prescribed scenarios using the same fixed MLP command
contract (`--episodes 20`, `--device cuda`, actor checkpoint only, CPU OSQP CBF)
at Git revision `7749876e89e0a487da43df2f46f03380e34d9039`.  The five-seed MLP
matrix now contains 30 eligible scenario directories and 600 retained episode
records: each selected directory has `summary.json` plus exactly 20 lines in
`raw_results/seed_<seed>.jsonl`.

The selected seed-20260719 nominal and OOD artifacts remain respectively
`core_3uav_mlp_mappo_seed_20260719_nominal_schemafix_rngfix_rerun2` and
`core_3uav_mlp_mappo_seed_20260719_ood_communication_obstacle_trajectoryfix_rerun1`.
The earlier nominal, nominal-schemafix-rerun1, and OOD directories are retained
invalid attempts and are explicitly excluded from any aggregation.  All
remaining seed/scenario identities use their standard matching directory under
`outputs/core_3uav_evaluations/`.

This establishes a complete, raw-data-preserving MLP evaluation protocol, not
a comparison result or safety conclusion.  The next action is to replay and
summarize every retained MLP training and evaluation CBF emergency fallback at
unchanged solver settings, recording exclusions and raw diagnostic locations;
only then may the three GraphMAPPO arms begin their matching independent runs.

## AAMAS 2027 - retained CBF fallback replay audit (2026-07-22)

`scripts/replay_cbf_fallbacks.py` is a diagnostic-only, append-safe replay
utility. It replays retained contexts with the unchanged mainline solver
protocol: 20,000 iterations, 0.1-second solve limit, slack penalty 100.0,
uncertainty-margin gain 0.5, and uncertainty-margin cap 5.0. It refuses to
overwrite an existing JSONL/summary identity. Its CLI load check and replay
runs used Git revision `7749876e89e0a487da43df2f46f03380e34d9039`.

The five completed MLP training summaries were replayed into
`outputs/cbf_diagnostics/core_mlp_5seed_replay_20260722.jsonl` and its sibling
summary. All 47 retained formal events replayed without a context error. Their
recorded statuses were 33 `solve_time_limit`, 7 `solved inaccurate`, and 7
`maximum iterations reached`; the same-parameter replay returned 12 `solved`,
9 `solved inaccurate`, 19 `solve_time_limit`, and 7 `maximum iterations
reached`, with 35 replayed emergency fallbacks. This is an event-level numerical
audit, not a fallback-rate, safety, or policy-quality result. The 30 uniform
checkpoint evaluations retain zero emergency events.

Historical diagnostic artifacts were separately retained in
`outputs/cbf_diagnostics/historic_cbf_replay_20260722.jsonl` and its sibling
summary: 18 of 21 events replayed, while three legacy records were marked
nonreplayable rather than repaired or discarded. The reasons were a missing
`requested_velocities` field in `cbf_diagnosis_context`, no saved context in
`cbf_diagnosis_residuals`, and missing `velocities` in the
`dynamic_mlp_mappo_cuda_smoke_v2` final-evaluation event. The aborted training
directory and all invalid evaluator attempts remain retained and excluded; no
slack, iteration limit, or tolerance changed during this audit.

This proves only that the formal MLP corpus is complete and its retained
fallback contexts can be replayed under the frozen protocol. Comparative method
advantage, safety guarantees, fallback-rate claims, and generalization remain
unproven. Next, run static/regression checks for the replay utility, commit the
tooling and documentation without `.gitignore`, then begin the first matched
raw-packet GraphMAPPO seed.

Verification of the replay utility passed at the same revision: `python -m
ruff check scripts/replay_cbf_fallbacks.py` and `python -m mypy
scripts/replay_cbf_fallbacks.py` both passed, and the full suite passed `147`
tests with `28` existing OSQP dependency deprecation warnings. The retained
test stdout/stderr evidence is
`outputs/cbf_diagnostics/pytest_replay_tool_20260722_stdout.log` and
`outputs/cbf_diagnostics/pytest_replay_tool_20260722_stderr.log`.

## AAMAS 2027 - first raw-packet GraphMAPPO seed retained (2026-07-22)

The first independent raw-packet GraphMAPPO job completed on CUDA with the
exact command `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_baseline.yaml
--device cuda --seed 20260719 --num-uavs 3 --graph-mode distance_graph
--total-steps 100000 --output-dir
outputs/core_3uav/core_3uav_raw_graph_seed_20260719`.  It used Git revision
`231cb8f322031db6aba8fabe41b5ddb35f3ef74e` and configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`; learned
network execution was CUDA and OSQP CBF stayed CPU-side.

The job reached 100,032 transitions (1,042 updates) and wrote
`checkpoints/graph_mappo_final.pt`, `summary.json`, live telemetry, launcher
logs, checkpoints, and TensorBoard data under its named output directory.
Retained CBF telemetry has 1 emergency fallback in 33,344 training decisions,
0 in 160 initial-evaluation decisions, and 0 in 112 final-evaluation decisions.
This proves one artifact's completion and traceability only, not graph quality,
safety, or a comparison against MLP.  The next action is serial training of raw
graph seed `20260720` with unchanged protocol; all five raw-graph checkpoints
and six-scenario JSONL evaluations remain required before any aggregation.

## AAMAS 2027 - second raw-packet GraphMAPPO seed retained (2026-07-22)

The independently trained raw-packet GraphMAPPO seed `20260720` completed the
same 100,000-transition CUDA command and frozen graph configuration as seed
`20260719`, at Git revision `961927b45dae3a5e0372c0263de536fe6d237c24`.  Its
output is `outputs/core_3uav/core_3uav_raw_graph_seed_20260720`, including
`checkpoints/graph_mappo_final.pt`, summary, live telemetry, launcher logs, and
TensorBoard data.  It reached 100,032 transitions and 1,042 updates.

CBF telemetry retains zero emergency fallbacks in 33,344 training decisions,
160 initial-evaluation decisions, and 104 final-evaluation decisions.  This is
completion evidence for a second independent raw-graph artifact only, not a
performance, safety, or fallback-rate result.  Next, train seed `20260721`
serially with the unchanged protocol; do not aggregate or start evaluation
claims until all five raw-graph seeds and their uniform scenario JSONL exist.

## AAMAS 2027 - third raw-packet GraphMAPPO seed retained (2026-07-23)

Raw-packet GraphMAPPO seed `20260721` completed the unchanged 100,000-step
CUDA protocol and produced
`outputs/core_3uav/core_3uav_raw_graph_seed_20260721/checkpoints/graph_mappo_final.pt`
at the current formal graph configuration.  The output directory retains its
summary, live telemetry, launcher logs, checkpoints, and TensorBoard data.
It reached 100,032 transitions and 1,042 updates.  Retained CBF telemetry is
1 emergency fallback in 33,344 training decisions, 0 in 160 initial-evaluation
decisions, and 0 in the final evaluation.  This remains seed-level traceability
only, not a comparative or safety result.  Next is serial seed `20260722` under
the same frozen protocol; no solver parameter is changed because of the event.

## AAMAS 2027 - fourth raw-packet GraphMAPPO seed retained (2026-07-23)

Raw-packet GraphMAPPO seed `20260722` completed the unchanged CUDA protocol at
100,032 transitions and 1,042 updates.  Its retained artifact is
`outputs/core_3uav/core_3uav_raw_graph_seed_20260722/checkpoints/graph_mappo_final.pt`,
with summary, live telemetry, launcher logs, checkpoints, and TensorBoard data
preserved in the same directory.  CBF telemetry records 3 emergency fallbacks
in 33,344 training decisions and 0 in both 160 initial-evaluation decisions and
the final evaluation.  These events are retained diagnostics, not a rate or
method conclusion; the frozen solver values were not changed.  Next: train
serial seed `20260723`, then evaluate all five raw-graph final checkpoints in
the six-scenario 20-episode protocol.

## AAMAS 2027 - raw-packet GraphMAPPO five-seed training complete (2026-07-23)

The fifth independent raw-packet GraphMAPPO seed `20260723` completed 100,032
CUDA transitions and 1,042 updates.  Its retained final checkpoint is
`outputs/core_3uav/core_3uav_raw_graph_seed_20260723/checkpoints/graph_mappo_final.pt`;
summary, live telemetry, launcher logs, checkpoints, and TensorBoard data are
kept with it.  At Git revision `13fceb826cb2bfc9b2fc929ea7054f69152b7561`, CBF
telemetry records 2 emergency fallbacks in 33,344 training decisions, 0 in 160
initial-evaluation decisions, and 0 in 110 final-evaluation decisions.

All five raw-packet graph checkpoints now exist, but this is not a method
comparison, safety result, or fallback-rate claim.  The next action is the
uniform evaluator for every raw-graph final checkpoint, six fixed scenarios and
20 episodes each, preserving every JSONL and then replaying raw-graph fallbacks
at unchanged solver settings before starting the predictive graph arm.

## AAMAS 2027 - raw-packet GraphMAPPO unified evaluation and CBF replay complete (2026-07-23)

All five independently trained raw-packet GraphMAPPO final checkpoints now
have the frozen six-scenario checkpoint evaluation matrix: seeds `20260719`
through `20260723`, scenarios `nominal`, `delay_only`, `loss_only`,
`dynamic_only`, `combined`, and `ood_communication_obstacle`, with 20 episodes
per condition.  The evaluator command for every cell was
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/evaluate_core_checkpoint.py
--family graph_mappo --config configs/rl/dynamic_graph_baseline.yaml
--checkpoint <seed-output>/checkpoints/graph_mappo_final.pt --output-dir
outputs/core_3uav_evaluations --experiment-name
core_3uav_raw_graph_seed_<seed>_<scenario> --seed <seed> --num-uavs 3
--episodes 20 --scenario <scenario> --device cuda`.  It used the frozen graph
configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D` at
pre-documentation Git revision `13fceb826cb2bfc9b2fc929ea7054f69152b7561`;
network inference ran on CUDA and OSQP CBF remained CPU-side.

The 30 named directories under `outputs/core_3uav_evaluations/` each retain a
`summary.json` and `raw_results/seed_<seed>.jsonl`; an explicit completeness
check verified exactly 20 JSONL records in every cell (600 records total).
No attempt directory was overwritten.  These are raw within-method records,
not a five-seed aggregate, cross-method comparison, safety guarantee, or
generalization conclusion.

The unchanged CBF replay command was
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/replay_cbf_fallbacks.py
--output-jsonl outputs/cbf_diagnostics/raw_graph_5seed_replay_20260723.jsonl
<five training summaries> <thirty evaluation summaries>`.  Its companion
`outputs/cbf_diagnostics/raw_graph_5seed_replay_20260723.summary.json` records
the frozen protocol (20,000 iterations, 0.1 seconds, slack penalty 100,
uncertainty-margin gain 0.5, cap 5.0), seven retained events, and zero replay
errors.  All seven sources were training summaries; two replays retained an
emergency fallback (`maximum iterations reached` and `solved inaccurate`),
while the other five solved.  No CBF setting was changed.  The next concrete
action is to train the predictive-without-uncertainty graph arm independently
with the same five seeds, then apply the same six-scenario protocol.

## AAMAS 2027 - first predictive-without-uncertainty GraphMAPPO seed retained (2026-07-23)

The first independently trained predictive-without-uncertainty GraphMAPPO job
completed its 100,000-transition CUDA budget using
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed 20260719
--num-uavs 3 --graph-mode predictive_graph --total-steps 100000 --output-dir
outputs/core_3uav/core_3uav_predictive_graph_seed_20260719`.  It was launched
from Git revision `faca19ac96e52013f0b436a240938b0e9b0ad3da` with configuration
SHA-256 `1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
The summary explicitly identifies `uses_predicted_knowledge: true` and
`uses_uncertainty: false`; learned execution used CUDA and OSQP CBF remained
CPU-side.  The retained final checkpoint is
`outputs/core_3uav/core_3uav_predictive_graph_seed_20260719/checkpoints/graph_mappo_final.pt`.
Its summary, CBF event context, and independent launcher stdout/stderr logs
remain in the named output path.

The final summary records 1,042 updates, one retained training emergency
fallback in 33,344 decisions (`solved inaccurate` at 20,000 iterations), zero
fallbacks in 160 initial-evaluation decisions, and zero in 124 final-evaluation
decisions.  No solver setting was altered.  This is seed-local completion and
traceability evidence only; it proves neither a method effect nor a safety or
fallback-rate claim.  Next: launch seed `20260720` serially with the same
frozen command pattern, retain all telemetry, and defer the six-scenario
evaluation until all five predictive-without-uncertainty checkpoints exist.

## AAMAS 2027 - second predictive-without-uncertainty GraphMAPPO seed retained (2026-07-23)

Independent seed `20260720` completed the same frozen 100,000-transition CUDA
command as seed `20260719`, changing only `--seed 20260720` and the output path
`outputs/core_3uav/core_3uav_predictive_graph_seed_20260720`.  It was launched
from Git revision `024d537` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.  The
retained `summary.json`, final checkpoint
`checkpoints/graph_mappo_final.pt`, CBF contexts, and unique launcher logs all
remain under that output name.  It reached 1,042 updates; the summary confirms
predictive knowledge is enabled and uncertainty input is disabled, with CUDA
for learned computation and CPU-side OSQP CBF.

Training CBF telemetry retains 3 emergency fallbacks in 33,344 decisions: one
`solve_time_limit` and two `solved inaccurate` events; all retain their solver
and environment contexts.  Initial evaluation has 0 in 160 decisions and
final evaluation 0 in 132 decisions.  The frozen slack penalty, iteration cap,
time cap, and tolerances were not changed.  This remains seed-level artifact
evidence, not a cross-method or fallback-rate conclusion.  Next: serially run
seed `20260721` with the same command, preserving its independent telemetry;
do not begin six-scenario evaluation before all five final checkpoints exist.

## AAMAS 2027 - third predictive-without-uncertainty GraphMAPPO seed retained (2026-07-23)

Independent seed `20260721` completed the same frozen 100,000-transition CUDA
protocol, launched at Git revision `035552f` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.  Its
final checkpoint is
`outputs/core_3uav/core_3uav_predictive_graph_seed_20260721/checkpoints/graph_mappo_final.pt`;
the same output directory retains `summary.json`, the sole fallback context,
and distinct launcher stdout/stderr logs.  The summary reports 1,042 updates,
predictive delivered-packet knowledge enabled, uncertainty input disabled, CUDA
for learned execution, and CPU-side OSQP CBF.

It retains one training emergency fallback in 33,344 decisions: `solved
inaccurate` at the frozen 20,000-iteration cap.  Initial evaluation has zero
fallbacks in 160 decisions and final evaluation zero in 109 decisions.  No
slack, iteration, timing, or tolerance value changed.  This proves only the
third seed's traceable completion, not comparative performance, a safety
property, or a fallback-rate estimate.  Next: launch seed `20260722` serially
under this unchanged command pattern; retain all event contexts and postpone
the six-scenario evaluator until five final checkpoints exist.

## AAMAS 2027 - fourth predictive-without-uncertainty GraphMAPPO seed retained (2026-07-23)

Independent seed `20260722` completed the unchanged 100,000-transition CUDA
protocol, launched at Git revision `69c79a3` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.  Its
final checkpoint is
`outputs/core_3uav/core_3uav_predictive_graph_seed_20260722/checkpoints/graph_mappo_final.pt`;
the output directory preserves its summary, full CBF fallback context, and
unique launcher logs.  The summary records 1,042 updates and confirms the
predictive-but-no-uncertainty information model, CUDA learned execution, and
CPU-side OSQP CBF.

Training retains one emergency fallback in 33,344 decisions: `solve_time_limit`
after 14,107 solver iterations at the frozen 0.1-second limit.  Initial
evaluation retains zero in 160 decisions and final evaluation zero in 153
decisions.  No solver setting changed.  This is traceable seed-level completion
only, not a comparative, safety, or fallback-rate result.  Next: train final
seed `20260723` serially with the identical frozen protocol, then verify all
five checkpoints before beginning the six-scenario evaluation matrix.

## AAMAS 2027 - predictive-without-uncertainty GraphMAPPO five-seed training complete (2026-07-23)

The fifth independent seed `20260723` completed the same 100,000-transition
CUDA command, launched at Git revision `87e6dd6` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.  Its
retained final checkpoint is
`outputs/core_3uav/core_3uav_predictive_graph_seed_20260723/checkpoints/graph_mappo_final.pt`;
its summary, three full training-fallback contexts, and unique launcher logs
remain in the seed output directory.  The summary confirms the arm uses
predicted delivered-packet knowledge without uncertainty input, CUDA learned
execution, and CPU-side OSQP CBF; it reached 1,042 updates.

The fifth seed has 3 retained training emergency fallbacks in 33,344 decisions:
one `maximum iterations reached` and two `solve_time_limit` events.  Initial
evaluation retains zero in 160 decisions and final evaluation zero in 123.
Across the five seed summaries, retained training counts are 1, 3, 1, 1, and 3
for seeds `20260719`--`20260723`; these are preserved diagnostics, not a
cross-seed fallback-rate conclusion.  No safety solver parameter changed.

All five independent predictive-without-uncertainty checkpoints now exist, but
this is not yet a method comparison, safety result, or generalization claim.
The next action is the frozen evaluator matrix for every checkpoint and each of
the six scenarios, 20 episodes per cell, preserving JSONL and checking all 600
records before CBF replay under unchanged settings.

## AAMAS 2027 - predictive-without-uncertainty unified evaluation and CBF replay complete (2026-07-23)

Every predictive-without-uncertainty final checkpoint now has the frozen
six-scenario checkpoint matrix: seeds `20260719`--`20260723`, scenarios
`nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, and
`ood_communication_obstacle`, at 20 episodes per cell.  Every invocation used
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe
scripts/evaluate_core_checkpoint.py --family graph_mappo --config
configs/rl/dynamic_graph_baseline.yaml --checkpoint
<seed-output>/checkpoints/graph_mappo_final.pt --output-dir
outputs/core_3uav_evaluations --experiment-name
core_3uav_predictive_graph_seed_<seed>_<scenario> --seed <seed> --num-uavs 3
--episodes 20 --scenario <scenario> --device cuda`, at pre-documentation Git
revision `869c767` and frozen configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
Network inference used CUDA and OSQP CBF remained CPU-side.

The 30 named directories under `outputs/core_3uav_evaluations/` each retain a
`summary.json` and `raw_results/seed_<seed>.jsonl`; a completeness check found
exactly 20 JSONL records in every condition (600 records total).  No directory
was overwritten.  The seed-20260720 OOD raw JSONL retains an evaluation
`solve_time_limit` event in episode 15; it remains included rather than
excluded.  These are raw within-method records, not a cross-method comparison,
safety guarantee, or generalization result.

`scripts/replay_cbf_fallbacks.py` now explicitly accepts JSONL as well as JSON
telemetry, preserving each source-line pointer.  `ruff`, `mypy`, and the full
test suite passed (`147 passed`, with 28 existing OSQP deprecation warnings).
The final all-source replay is
`outputs/cbf_diagnostics/predictive_graph_5seed_replay_20260723_jsonlfix_rerun2.jsonl`
with its companion summary: it reads the five training summaries, 30 evaluation
summaries, and 30 evaluation JSONL files under unchanged values (20,000
iterations, 0.1 seconds, penalty 100, uncertainty gain 0.5, cap 5.0).  It
retains 10 source events: nine replay without an error (two emergency fallback
replays), while the OOD evaluation event is an explicit `replay_error` because
its retained center-only dynamic-obstacle state cannot reconstruct the full
trajectory at the recorded step.  No data were imputed and no solver setting
changed.  The prior
`...predictive_graph_5seed_replay_20260723_jsonlfix.jsonl` attempt is retained
and excluded from this all-source accounting because a PowerShell path-expression
error passed zero raw JSONL files; it is not overwritten.  Next: train the
uncertainty-aware predictive graph arm independently with the same five seeds.

## AAMAS 2027 - first uncertainty-aware predictive GraphMAPPO seed retained (2026-07-23)

The first independent uncertainty-aware predictive GraphMAPPO job completed
the 100,000-transition CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed 20260719
--num-uavs 3 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260719`.
It was launched at Git revision `63d3f7f` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.  The
summary confirms both predicted delivered-packet knowledge and uncertainty
input are enabled; learned execution used CUDA and OSQP CBF stayed CPU-side.
The retained final checkpoint is `checkpoints/graph_mappo_final.pt` in the
named output directory, alongside summary, event context, and launcher logs.

The summary records 1,042 updates and one retained training emergency fallback
in 33,344 decisions: `solved inaccurate` at 20,000 iterations.  Initial
evaluation has zero fallbacks in 160 decisions and final evaluation zero in 127
decisions.  No solver setting changed.  This documents seed-local completion
only, not a comparison, safety result, or fallback-rate estimate.  Next: run
seed `20260720` serially with the same frozen protocol, retaining every event;
postpone six-scenario evaluation until all five final checkpoints exist.

## AAMAS 2027 - second uncertainty-aware predictive GraphMAPPO seed retained (2026-07-23)

Independent seed `20260720` completed the frozen 100,000-transition CUDA
command `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_baseline.yaml
--device cuda --seed 20260720 --num-uavs 3 --graph-mode
uncertainty_predictive_graph --total-steps 100000 --output-dir
outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260720`.
It was launched at Git revision `d0de541` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
The retained output directory contains `summary.json`, the final checkpoint
`checkpoints/graph_mappo_final.pt`, live telemetry, TensorBoard data, and the
dedicated launcher stdout/stderr logs.  Its summary identifies
`uncertainty_predictive_graph`; learned network work ran on CUDA while OSQP
CBF remained CPU-side.

The completed run records 1,042 updates.  Training telemetry retains one
emergency fallback in 33,344 decisions: `solved inaccurate` at decision index
25,156 with the unchanged 20,000-iteration cap (the full solver and
environment context remains in `summary.json`).  Initial evaluation retains
zero fallbacks in 160 decisions; final interval evaluation retains zero in 154
decisions.  The frozen slack penalty, solve-time limit, iteration cap, and
tolerances were not changed.

This is second-seed artifact and traceability evidence only: it does not prove
a method effect, safety result, generalization result, or cross-seed
fallback-rate estimate.  Next, verify no legacy training process remains and
run seed `20260721` serially with the identical command; defer all
six-scenario evaluations and aggregation until all five uncertainty-aware
predictive final checkpoints are complete.

## AAMAS 2027 - third uncertainty-aware predictive GraphMAPPO seed retained (2026-07-23)

Independent seed `20260721` completed the same frozen 100,000-transition CUDA
command as the preceding uncertainty-aware seeds, changing only `--seed
20260721` and the output directory:
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed 20260721
--num-uavs 3 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260721`.
It was launched at Git revision `23c9da0` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
The output directory retains `summary.json`, final checkpoint
`checkpoints/graph_mappo_final.pt`, live telemetry, TensorBoard data, and
dedicated launcher stdout/stderr logs.  The summary confirms
`uses_predicted_knowledge: true`, `uses_uncertainty: true`, three UAVs, and
CUDA network execution; OSQP CBF remained CPU-side.

The summary records 1,042 updates and one retained training emergency fallback
in 33,344 CBF decisions: `maximum iterations reached` at decision index 4,994,
after the unchanged 20,000 iteration cap (solve time 0.0335165 seconds,
primal residual 0.00643271, dual residual 0.0003267405).  Its complete
position, requested-velocity, delivered-knowledge, uncertainty, and dynamic
obstacle context remains in `summary.json`; the prior dedicated stderr notice
is retained too.  Initial evaluation has zero fallbacks in 160 decisions and
final interval evaluation zero in 160 decisions.  The slack penalty,
iteration cap, solve-time limit, and tolerances were not changed.

This is third-seed completion and traceability evidence, not a performance,
safety, generalization, or cross-seed fallback-rate result.  The next action
is to confirm no Python training process remains, then launch seed `20260722`
serially with the identical frozen command.  Do not start the six-scenario
matrix or aggregate results until seeds `20260719`--`20260723` all have final
checkpoints.

## AAMAS 2027 - fourth uncertainty-aware predictive GraphMAPPO seed retained (2026-07-23)

Independent seed `20260722` completed the frozen 100,000-transition CUDA
command `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_baseline.yaml
--device cuda --seed 20260722 --num-uavs 3 --graph-mode
uncertainty_predictive_graph --total-steps 100000 --output-dir
outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260722`.
It was launched at Git revision `195d03f` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
The output directory preserves `summary.json`, final checkpoint
`checkpoints/graph_mappo_final.pt`, live telemetry, TensorBoard data, and
dedicated launcher stdout/stderr logs.  The summary confirms three UAVs,
`uses_predicted_knowledge: true`, `uses_uncertainty: true`, CUDA network work,
and the CPU-side OSQP CBF path.

The final summary records 1,042 updates and one retained training emergency
fallback in 33,344 CBF decisions: `solve_time_limit` at decision index 681
under the unchanged 0.1-second limit (0.100514 seconds, 6,942 iterations,
primal residual 0.0418170, dual residual 1.4419138).  Its complete initial
position, requested-velocity, delivered-knowledge, uncertainty, and dynamic
obstacle context remains in `summary.json`; the dedicated stderr notice is
also retained.  Initial evaluation retains zero fallbacks in 160 decisions and
final interval evaluation zero in 122 decisions.  No slack penalty, solve-time
limit, iteration cap, tolerance, or other safety setting changed.

This establishes a fourth traceable seed artifact only; it is not a
performance, safety, generalization, or cross-seed fallback-rate result.  Next,
verify that no Python training process remains, then launch the final seed
`20260723` serially with this identical frozen protocol.  Six-scenario
evaluation and any aggregation remain deferred until all five final
checkpoints exist.

## AAMAS 2027 - five uncertainty-aware predictive GraphMAPPO checkpoints retained (2026-07-23)

The final independent seed `20260723` completed the frozen 100,000-transition
CUDA command `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_baseline.yaml
--device cuda --seed 20260723 --num-uavs 3 --graph-mode
uncertainty_predictive_graph --total-steps 100000 --output-dir
outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260723`.
It was launched at Git revision `b09be13` with configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
The named output directory retains the final
`checkpoints/graph_mappo_final.pt`, `summary.json`, live telemetry,
TensorBoard data, and dedicated launcher stdout/stderr logs.  Its summary
confirms three UAVs, `uses_predicted_knowledge: true`,
`uses_uncertainty: true`, CUDA network work, and CPU-side OSQP CBF.

This run has 1,042 updates and one retained training emergency fallback in
33,344 CBF decisions: `solve_time_limit` at decision index 6,257 under the
unchanged 0.1-second limit (0.1007939 seconds, 6,394 iterations, primal
residual 0.0640996, dual residual 1.0995986).  The complete initial-position,
requested-velocity, delivered-knowledge, uncertainty, and dynamic-obstacle
context remains in `summary.json` and the dedicated stderr notification is
retained.  Initial evaluation has zero fallbacks in 160 decisions, and final
interval evaluation has zero in 158 decisions.  No slack penalty, solve-time
limit, iteration cap, tolerance, or other safety parameter changed.

All five independent 3-UAV uncertainty-aware predictive checkpoints for seeds
`20260719`--`20260723` now exist.  This is completion and traceability
evidence for the training phase only; it does not establish a method effect,
safety property, generalization result, or fallback-rate estimate.  Next, run
the frozen six-scenario evaluator for every final checkpoint (20 episodes per
cell, unique directory and raw JSONL per cell), validate all 600 JSONL records,
then replay every retained training and evaluation CBF event under unchanged
solver settings.  Aggregate or cross-method conclusions remain prohibited
until those steps are complete.

## AAMAS 2027 - uncertainty-aware predictive unified evaluation and CBF replay complete (2026-07-26)

All five uncertainty-aware predictive final checkpoints now have the frozen
six-scenario matrix: seeds `20260719`--`20260723`, scenarios `nominal`,
`delay_only`, `loss_only`, `dynamic_only`, `combined`, and
`ood_communication_obstacle`, with 20 episodes per cell.  Every invocation
used `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
scripts/evaluate_core_checkpoint.py --family graph_mappo --config
configs/rl/dynamic_graph_baseline.yaml --checkpoint
<seed-output>/checkpoints/graph_mappo_final.pt --output-dir
outputs/core_3uav_evaluations --experiment-name
core_3uav_uncertainty_predictive_graph_seed_<seed>_<scenario> --seed <seed>
--num-uavs 3 --episodes 20 --scenario <scenario> --device cuda`, serially, at
pre-evaluation Git revision `4bea499` and frozen configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
Network inference used CUDA and OSQP CBF remained CPU-side.  The serial
launcher logs are
`outputs/core_3uav_evaluations/uncertainty_predictive_graph_5seed_eval_20260726_launcher_stdout.log`
and `_launcher_stderr.log`.

The 30 uniquely named directories under `outputs/core_3uav_evaluations/` each
retain `summary.json` and `raw_results/seed_<seed>.jsonl`.  An explicit
completeness audit found exactly 20 nonblank JSONL records in every cell (600
records total); no target directory existed before launch and none was
overwritten.  A direct scan of all 600 retained raw episode records found no
evaluation `cbf.emergency_events`.  This is raw-protocol evidence only, not a
safety claim, fallback-rate estimate, generalization result, or cross-method
comparison.

The append-safe all-source CBF replay is
`outputs/cbf_diagnostics/uncertainty_predictive_graph_5seed_replay_20260726.jsonl`
with companion
`outputs/cbf_diagnostics/uncertainty_predictive_graph_5seed_replay_20260726.summary.json`.
It consumes the five training summaries, 30 evaluation summaries, and 30
evaluation JSONL files under unchanged values: 20,000 iterations, 0.1-second
solve limit, slack penalty 100.0, uncertainty-margin gain 0.5, and cap 5.0.
It retains five source training events and records zero replay errors; the
source status/replay status pairs are `solved inaccurate` to `solved
inaccurate` (seed 20260719), `solved inaccurate` to `solved` (20260720),
`maximum iterations reached` to the same status (20260721), and
`solve_time_limit` to `solved` (20260722 and 20260723).  Emergency-fallback
flags in the replay output remain retained diagnostics, not a fallback-rate or
safety conclusion.

The replay executable itself succeeded.  Two subsequent read-only PowerShell
post-processing attempts incorrectly assumed companion names ending in
`_summary.json` and `.jsonl.summary.json`; both failed after the valid replay
had already been written, created no additional result artifact, and did not
overwrite the valid JSONL or its actual `.summary.json` companion.  They are
recorded here as excluded post-processing errors.  No solver setting or raw
telemetry was changed.

The matched 3-UAV five-seed/evaluation/replay protocol is now complete for all
four learned arms, but this does not license comparative, statistical, or
method-effect claims without a prespecified aggregation analysis.  Next:
inspect the frozen 5/8-UAV independent-training and ablation protocol, retain
its separate artifacts, and keep literature novelty statements gated on
primary-paper reading.

## AAMAS 2027 - frozen 5/8-UAV uncertainty-aware scale configurations validated (2026-07-26)

Two scale-specific mainline configurations now materialize the existing fixed
profiles in `configs/experiments/scale_profiles.yaml` without changing the
3-UAV protocol: `configs/rl/dynamic_graph_5uav.yaml` (five UAVs, two
environments, rollout length 32; SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`) and
`configs/rl/dynamic_graph_8uav.yaml` (eight UAVs, two environments, rollout
length 24; SHA-256
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`).  Both
retain delayed/lossy packet semantics, uncertainty-aware predictive graph
mode, dynamic obstacle enablement, CUDA network execution, CPU-side OSQP CBF,
and the frozen CBF penalty 100.0/20,000-iteration/0.1-second/margin-gain-0.5/
margin-cap-5.0 protocol.

Initial loading exposed that the new 8-UAV file omitted
`dynamic_obstacle_enabled: true`; no training or evaluation had started, so
the file was corrected before any artifact was generated.  The final configs
loaded with their intended UAV count, environment count, rollout length,
dynamic-obstacle, communication, CBF, and uncertainty-predictive settings.
`pytest -q tests/test_dynamic_graph_baselines.py
tests/test_core_checkpoint_evaluation.py` passed (4 tests; 7 existing OSQP
deprecation warnings); Ruff and mypy both passed for
`scripts/train_graph_mappo.py` and `multiuav/learning/graph_runner.py`.

This validates configuration semantics only, not scalable performance, safety,
or generalization.  Next: independently train the first 5-UAV
uncertainty-aware predictive seed with the frozen 5-UAV configuration, retain
all CBF events, and continue serially before beginning the corresponding 8-UAV
matrix or independent ablation arms.

## AAMAS 2027 - first 5-UAV uncertainty-aware predictive seed retained (2026-07-26)

The first independent 5-UAV uncertainty-aware predictive GraphMAPPO job
completed the 100,000-transition CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260719
--num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260719`.
It was launched at Git revision `7d6486e` with configuration SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
Its output directory retains `summary.json`, final checkpoint
`checkpoints/graph_mappo_final.pt`, TensorBoard data, and live telemetry.  The
summary confirms five UAVs, predicted delivered-packet knowledge and
uncertainty input, CUDA network execution, and CPU-side OSQP CBF.

The run reached 100,160 transitions and 313 updates.  Retained CBF telemetry
has zero emergency fallbacks in 20,032 training decisions, zero in 160 initial
evaluation decisions, and zero in 160 final evaluation decisions; no safety
parameter changed.  This seed-local diagnostic artifact does not prove a
fallback rate, safety property, scalability result, or method effect.

The intended launcher stdout/stderr redirection files did not materialize,
although the live training process, `live_training_telemetry.json`, summary,
checkpoint, and TensorBoard outputs are retained.  A subsequent read-only
`Get-Item` check for those absent log files returned nonzero and wrote no data;
this is an observability/post-processing gap, not an invalidated training
artifact.  Do not recreate or overwrite the missing logs.  Next: verify no
Python process remains and train seed `20260720` serially with the identical
5-UAV frozen configuration, retaining all outputs before any scale evaluation,
aggregation, 8-UAV run, or ablation claim.

## AAMAS 2027 - second 5-UAV uncertainty-aware predictive seed retained (2026-07-26)

The second independent 5-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260720
--num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260720`.
It was launched at Git revision `8899c15` with configuration SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
Its output directory retains `summary.json`,
`checkpoints/graph_mappo_final.pt`, TensorBoard data, and
`live_training_telemetry.json`; dedicated launcher stdout/stderr are retained
as `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260720_launcher_stdout.log`
and `_launcher_stderr.log`.  The summary verifies five UAVs, predicted
delivered-packet knowledge with uncertainty, and CUDA network execution; OSQP
CBF remained CPU-side.

The retained final telemetry reaches 100,160 transitions and 313 updates.  It
records three training CBF emergency fallbacks in 20,032 decisions: `solved
inaccurate` at decision indices 14,582 and 14,652 (each at the unchanged
20,000-iteration cap), and `solve_time_limit` at 17,555 (0.10062640000251122
seconds, 11,265 iterations, primal residual 0.04024062089868315, and dual
residual 0.0004940578132866836).  The complete state, delivered-packet, and
dynamic-obstacle contexts remain in the telemetry and summary.  Initial and
final evaluation telemetry each retain zero emergency fallbacks in 160
decisions.  No slack penalty, iteration cap, solve-time limit, tolerance, or
other safety parameter changed.

This is one seed-local execution record, not a fallback-rate estimate, safety
property, scalability result, or method-effect claim.  Next: after confirming
no Python training process remains, train seed `20260721` serially with the
identical frozen configuration; do not begin 5-UAV evaluation, 8-UAV work, or
any ablation arm until all five 5-UAV seeds are retained and documented.

## AAMAS 2027 - third 5-UAV uncertainty-aware predictive seed retained (2026-07-26)

The third independent 5-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260721
--num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260721`.
It was launched at Git revision `7c0fbc8` with configuration SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
Its output directory retains `summary.json`,
`checkpoints/graph_mappo_final.pt`, TensorBoard data, and
`live_training_telemetry.json`, while its dedicated launcher stdout/stderr are
retained at
`outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260721_launcher_stdout.log`
and `_launcher_stderr.log`.  The final summary verifies five UAVs, predicted
delivered-packet knowledge with uncertainty, CUDA network execution, and
CPU-side OSQP CBF.

It reached 100,160 transitions and 313 updates.  The training CBF telemetry
retains two `solved inaccurate` emergency fallbacks in 20,032 decisions, both
at the unchanged 20,000-iteration cap: decision 4,753 (environment 0, step
16; 0.05340159998741001 seconds; primal/dual residuals
0.0026409692851873355/0.00011929081440663525) and decision 6,341
(environment 0, step 10; 0.028405999997630715 seconds; primal/dual residuals
0.001457085484388434/0.000030369510219408637).  Full positions, velocities,
delivered-packet validity/uncertainty, and dynamic-obstacle contexts are
retained in the final summary and live telemetry.  Initial and final
evaluation telemetry each retain zero emergency fallbacks in 160 decisions.
No slack penalty, iteration cap, solve-time limit, tolerance, or other safety
parameter changed.

This is a seed-local diagnostic artifact, not a fallback-rate estimate, safety
property, scalability result, or method-effect claim.  Next: confirm no Python
training process remains, then train seed `20260722` serially using the
identical frozen configuration; keep 5-UAV evaluation, 8-UAV work, and
ablation arms deferred until the five-seed 5-UAV training set is complete.

## AAMAS 2027 - fourth 5-UAV uncertainty-aware predictive seed retained (2026-07-26)

The fourth independent 5-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260722
--num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260722`.
It was launched at Git revision `0310300` with configuration SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
`summary.json`, `checkpoints/graph_mappo_final.pt`, TensorBoard output, live
telemetry, and dedicated launcher stdout/stderr are retained under the seed
directory and sibling `...seed_20260722_launcher_{stdout,stderr}.log` paths.
The final summary verifies five UAVs, uncertainty-aware predicted delivered
packet knowledge, CUDA network execution, and CPU-side OSQP CBF.

It reached 100,160 transitions and 313 updates.  Training telemetry retains
two emergency fallbacks in 20,032 CBF decisions: `maximum iterations reached`
at decision 4,167 (environment 0 step 3; 0.049038999975891784 seconds;
primal/dual residuals 0.005824136094528052/0.0000491939606585269) and `solved
inaccurate` at decision 9,017 (environment 0 step 8;
0.051416900008916855 seconds; primal/dual residuals
0.0014887963841878208/0.00000338793323094724).  Their full positions,
velocities, delivered-packet validity/uncertainty, and dynamic-obstacle
contexts remain in the summary and live telemetry.  Initial and final
evaluation telemetry each retain zero emergency fallbacks in 160 decisions.
No slack penalty, iteration cap, solve-time limit, tolerance, or other safety
parameter changed.

This seed-local diagnostic artifact does not establish a fallback-rate,
safety, scalability, or method-effect result.  Next: confirm no Python process
remains and train final seed `20260723` serially under the identical frozen
configuration.  Defer 5-UAV evaluation, 8-UAV work, and ablation arms until
all five 5-UAV training artifacts are retained and documented.

## AAMAS 2027 - fifth 5-UAV uncertainty-aware predictive seed retained (2026-07-26)

The final independent 5-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260723
--num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260723`.
It was launched at Git revision `31ff7cb` with configuration SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
Its output retains `summary.json`, `checkpoints/graph_mappo_final.pt`,
TensorBoard data, live telemetry, and dedicated launcher stdout/stderr under
the seed directory and sibling `...seed_20260723_launcher_{stdout,stderr}.log`
paths.  The final summary verifies five UAVs, uncertainty-aware predicted
delivered-packet knowledge, CUDA network execution, and CPU-side OSQP CBF.

The run reached 100,160 transitions and 313 updates.  Its final
`summary.json` and live telemetry retain one training `solved inaccurate`
emergency fallback in 20,032 CBF decisions: decision 7,308, environment 1
step 13, obstacle center `[22.0, 76.0, 30.0]`, unchanged 20,000-iteration
cap, 0.04082990001188591 seconds, and primal/dual residuals
0.0002440071509879367/0.0000122640447944475.  Initial and final evaluation
telemetry each retain zero emergency fallbacks in 160 decisions.  The
dedicated stderr independently retains a `solve_time_limit` notification and
a `solved inaccurate` notification; the latter corresponds to the retained
summary event, whereas the former is not separately represented in final
telemetry.  Both raw sources are preserved as an explicit event-accounting
discrepancy, with no imputation, merging, exclusion, or safety-parameter
change.

All five independent 5-UAV uncertainty-aware predictive training artifacts
for seeds `20260719`--`20260723` now exist.  This completes only their
training-artifact phase; it does not establish a scale-level result, fallback
rate, safety property, or method effect.  Next: use the frozen checkpoint
evaluator to run every seed across six scenarios for 20 episodes per unique
cell, retain raw JSONL, validate completeness, and replay all CBF evidence
without changing solver values.  Only then may 8-UAV work or separately
trained ablations begin.

## AAMAS 2027 - 5-UAV uncertainty-aware predictive evaluation and CBF replay retained (2026-07-26)

All five retained 5-UAV uncertainty-aware predictive checkpoints now have the
frozen six-scenario matrix: seeds `20260719`--`20260723`, scenarios
`nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, and
`ood_communication_obstacle`, 20 episodes per unique cell.  A hidden serial
launcher ran each invocation as
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe
scripts/evaluate_core_checkpoint.py --family graph_mappo --config
configs/rl/dynamic_graph_5uav.yaml --checkpoint
outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_<seed>/checkpoints/graph_mappo_final.pt
--output-dir outputs/core_5uav_evaluations --experiment-name
core_5uav_uncertainty_predictive_graph_seed_<seed>_<scenario> --seed <seed>
--num-uavs 5 --episodes 20 --scenario <scenario> --device cuda`, serially, at
launch revision `bbb871a` and configuration SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
Network inference used CUDA and OSQP CBF remained CPU-side.  Launcher output
is retained in
`outputs/core_5uav_evaluations/uncertainty_predictive_graph_5uav_5seed_eval_20260726_launcher_stdout.log`
and `_launcher_stderr.log`; stderr contains the retained harmless PowerShell
first-module CLIXML notice.

The post-run audit found exactly 30 uniquely named directories, each with
`summary.json` and `raw_results/seed_<seed>.jsonl`.  Every JSONL file has
exactly 20 nonblank, JSON-parseable records: 600 total, zero malformed cells.
A direct scan of all 600 raw records found zero evaluation
`cbf.emergency_events`.  This is completeness and raw-protocol evidence only,
not a safety, fallback-rate, performance, generalization, or scalability
claim.

The append-safe replay output is
`outputs/cbf_diagnostics/uncertainty_predictive_graph_5uav_5seed_replay_20260726.jsonl`
with companion
`outputs/cbf_diagnostics/uncertainty_predictive_graph_5uav_5seed_replay_20260726.summary.json`.
It consumed five training summaries, 30 evaluation summaries, and 30 raw
evaluation JSONL files under unchanged values: 20,000 iterations, 0.1-second
solve limit, slack penalty 100.0, uncertainty-margin gain 0.5, and cap 5.0.
The valid replay retains eight source events and zero replay errors.  It cannot
replay the separately retained seed-`20260723` stderr-only `solve_time_limit`
notification because that notification lacks a distinct telemetry event
context; that unresolved accounting discrepancy remains reported rather than
imputed or excluded.

The replay executable itself succeeded.  A later read-only post-processing
check incorrectly sought a companion named `.jsonl.summary.json`, then exited
nonzero after the valid JSONL and actual `.summary.json` companion already
existed.  It created no new result, overwrote nothing, and is retained as an
excluded post-processing error.  No solver, safety margin, or raw telemetry
changed.  The five-seed 5-UAV training/evaluation/replay protocol is complete
as an artifact set, but it does not license a method comparison or any scale
claim.  Next: begin the independently trained 8-UAV uncertainty-aware
predictive five-seed protocol, then its matched matrix/replay, before any
separately trained key ablation.

## AAMAS 2027 - first 8-UAV uncertainty-aware predictive seed retained (2026-07-26)

The first independent 8-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_8uav.yaml --device cuda --seed 20260719
--num-uavs 8 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260719`.
It was launched at Git revision `07d0f47` with configuration SHA-256
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
The final output retains `summary.json`,
`checkpoints/graph_mappo_final.pt`, TensorBoard data, and
`live_training_telemetry.json`; it verifies eight UAVs, predicted
delivered-packet knowledge with uncertainty, CUDA network execution, and
CPU-side OSQP CBF.

It reached 100,224 transitions and 261 updates.  Training CBF telemetry
retains one `solved inaccurate` emergency fallback in 12,528 decisions at
decision 4,222 (environment 1 step 10; obstacle center `[22.0, 70.0, 30.0]`;
unchanged 20,000-iteration cap; 0.05533530001412146 seconds; primal/dual
residuals 0.0011686880877244399/0.0000192246889227154).  Full eight-UAV
positions, velocities, delivered-packet validity/uncertainty, and obstacle
context remain in final telemetry and the summary.  Initial and final
evaluation telemetry each retain zero emergency fallbacks in 160 decisions.
No slack penalty, iteration cap, solve-time limit, tolerance, uncertainty
margin, or other safety parameter changed.

The intended dedicated launcher stdout/stderr logs did not materialize because
the parent output directory did not exist at launch.  A subsequent read-only
`Get-Content` check of the absent stderr path and a later 55-second read-only
completion check both exited nonzero; neither wrote or invalidated an
experiment artifact.  An earlier preflight import used nonexistent
`load_graph_mappo_config`, failed before launch, and likewise created no
experiment artifact; correct loading subsequently validated the frozen config.
Do not recreate missing logs or repeat the seed.  This is a seed-local
diagnostic artifact, not a fallback-rate, safety, scalability, or method-effect
claim.  Next: after confirming no Python process remains, train seed
`20260720` serially with the identical 8-UAV configuration, preserving every
event before any 8-UAV evaluation or ablation task.

## AAMAS 2027 - second 8-UAV uncertainty-aware predictive seed retained (2026-07-26)

The second independent 8-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_8uav.yaml --device cuda --seed 20260720
--num-uavs 8 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260720`.
It was launched at Git revision `a99f7bb` with configuration SHA-256
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
Its output directory retains `summary.json`,
`checkpoints/graph_mappo_final.pt`, TensorBoard data, live telemetry, and
dedicated launcher stdout/stderr.  The final summary confirms eight UAVs,
predicted delivered-packet knowledge with uncertainty, CUDA network execution,
and CPU-side OSQP CBF.

It reached 100,224 transitions and 261 updates.  Final training telemetry
retains sixteen `solve_time_limit` CBF emergency fallbacks in 12,528 decisions
at indices `282, 1121, 1721, 2641, 3922, 3924, 3961, 3962, 3964, 4162, 5881,
5882, 6001, 8042, 8044, 11681`; their complete eight-UAV contexts and exact
residuals remain in `summary.json` and `live_training_telemetry.json`.  The
maximum retained solve time is 0.10106859999359585 seconds and the maximum
observed iterations are 17,750; both are descriptive telemetry, not altered
settings.  Initial and final evaluation telemetry each retain zero emergency
fallbacks in 160 decisions.  The dedicated stderr retains 32 emergency
notification lines, which do not map one-to-one to the sixteen telemetry
events.  Both raw sources are preserved as an explicit event-accounting
discrepancy: they are not merged, imputed, or silently excluded.

No slack penalty, iteration cap, solve-time limit, tolerance, uncertainty
margin, or other safety parameter changed.  This is seed-local diagnostic
evidence only, not a fallback-rate, safety, performance, scalability, or
method-effect result.  Next: verify no Python process remains, then train seed
`20260721` serially under the identical frozen 8-UAV configuration; defer
8-UAV evaluation and all ablations until five base seeds are retained.

## AAMAS 2027 - third 8-UAV uncertainty-aware predictive seed retained (2026-07-26)

The third independent 8-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_8uav.yaml --device cuda --seed 20260721
--num-uavs 8 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260721`.
It was launched at Git revision `2fb5a0e` with configuration SHA-256
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
Its output directory retains the final checkpoint, summary, TensorBoard data,
live telemetry, and dedicated launcher logs; the summary verifies eight UAVs,
predicted delivered-packet knowledge with uncertainty, CUDA network execution,
and CPU-side OSQP CBF.

It reached 100,224 transitions and 261 updates.  Final training telemetry
retains five `solve_time_limit` emergency fallbacks in 12,528 CBF decisions at
indices `3441, 5081, 5082, 5084, 5122`; full eight-UAV contexts, solve times,
iterations, and residuals remain in the summary and telemetry.  Initial and
final evaluation telemetry each retain zero emergency fallbacks in 160
decisions.  Dedicated stderr retains six notification lines, rather than a
one-to-one mapping to the five telemetry events.  Preserve this explicit raw
source discrepancy without merging, imputation, exclusion, or protocol change.
No slack penalty, iteration cap, solve-time limit, tolerance, uncertainty
margin, or other safety parameter changed.

This is a seed-local diagnostic artifact, not a fallback-rate, safety,
performance, scalability, or method-effect result.  Next: verify no Python
process remains, train seed `20260722` serially with the identical frozen
8-UAV protocol, and defer evaluation/ablations until all five base seeds are
retained.

## AAMAS 2027 - fourth 8-UAV uncertainty-aware predictive seed retained (2026-07-26)

The fourth independent 8-UAV uncertainty-aware predictive GraphMAPPO job
completed the frozen CUDA command for seed `20260722` with
`configs/rl/dynamic_graph_8uav.yaml`, `--device cuda`, `--num-uavs 8`,
`--graph-mode uncertainty_predictive_graph`, 100,000 requested steps, and
output `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260722`.
It launched at revision `056b6d3` with configuration SHA-256
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
Final checkpoint, summary, TensorBoard, live telemetry, and launcher logs are
retained; CUDA network execution and CPU-side OSQP CBF remain verified.

It reached 100,224 transitions and 261 updates.  Final telemetry retains two
`solved inaccurate` training fallbacks in 12,528 CBF decisions: indices 10,468
and 11,541, each at the unchanged 20,000-iteration cap (0.04474149999441579
and 0.036341500002890825 seconds).  Full contexts remain in summary and live
telemetry.  Initial/final evaluation telemetry each retain zero fallbacks in
160 decisions.  Stderr has four notification lines, rather than a one-to-one
mapping to the two telemetry events; preserve both sources without merging,
imputation, exclusion, or parameter change.  This is seed-local diagnostics,
not a scale, safety, fallback-rate, performance, or method-effect result.
Next: serially train final seed `20260723` under the identical protocol.

## AAMAS 2027 - fifth 8-UAV uncertainty-aware predictive seed retained (2026-07-27)

The final independent 8-UAV uncertainty-aware predictive job completed the
frozen CUDA command for seed `20260723`, launched at revision `fd6142c` with
`configs/rl/dynamic_graph_8uav.yaml` SHA-256
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
Its final checkpoint, summary, TensorBoard, telemetry, and launcher logs are
retained under
`outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260723`.
It reached 100,224 transitions and 261 updates with CUDA network execution and
CPU-side OSQP CBF.

Final training telemetry retains one `solved inaccurate` event in 12,528 CBF
decisions (index 4,279; unchanged 20,000-iteration cap; full context retained).
Initial training-script evaluation retains nine `solve_time_limit` events in
160 decisions; final training-script evaluation retains zero in 160 decisions.
The launcher stderr retains ten notifications, which are not mapped one-to-one
to telemetry.  Preserve all raw sources and this discrepancy without merging,
imputation, exclusion, or any parameter change.  This completes the 8-UAV
five-seed training artifact set only; it is not a scale, safety, fallback-rate,
performance, or method-effect conclusion.  Next: run the frozen six-scenario
20-episode evaluation matrix and unchanged-protocol CBF replay for all five
8-UAV checkpoints before any ablation.

## AAMAS 2027 - 8-UAV uncertainty-aware predictive evaluation and CBF replay retained (2026-07-27)

The frozen five-seed 8-UAV checkpoint matrix completed serially at launcher revision `f90471f`. Each final checkpoint from `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260719` through `..._20260723` was evaluated by `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/evaluate_core_checkpoint.py --family graph_mappo --config configs/rl/dynamic_graph_8uav.yaml --checkpoint <seed-dir>/checkpoints/graph_mappo_final.pt --output-dir outputs/core_8uav_evaluations --experiment-name core_8uav_uncertainty_predictive_graph_seed_<seed>_<scenario> --seed <seed> --num-uavs 8 --episodes 20 --scenario <scenario> --device cuda`, for seeds `20260719`--`20260723` and scenarios `nominal`, `delay`, `loss`, `dynamic`, `combined`, and `ood`. The frozen configuration SHA-256 is `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`. CUDA was used for learned-network inference; OSQP CBF stayed CPU-side.

Outputs remain under `outputs/core_8uav_evaluations/`; dedicated serial launcher logs are `uncertainty_predictive_graph_8uav_5seed_eval_20260727_launcher_stdout.log` and `_launcher_stderr.log`. A read-only audit verified 30 cell directories, 30 `summary.json` files, and 30 raw JSONL files. Each JSONL contains exactly 20 nonblank parseable episode records: 600 records total, zero parse errors, and zero malformed-count cells. The preceding launcher-construction attempt had a PowerShell `-or` syntax error before executing any evaluator; it created only the empty output root, no cells, logs, or results, and remains a retained, excluded invalid attempt.

All training summaries, 30 evaluation summaries, and 30 raw JSONL files were then supplied unchanged to `scripts/replay_cbf_fallbacks.py` with the frozen protocol `--max-iterations 20000 --max-solve-time-seconds 0.1 --slack-penalty 100.0 --uncertainty-margin-gain 0.5 --max-uncertainty-margin 5.0`. The outputs `outputs/cbf_diagnostics/uncertainty_predictive_graph_8uav_5seed_replay_20260727.jsonl` and its companion `.summary.json` retain 200 source events across 65 telemetry files. Of these, 116 replayed; their replay statuses are 87 `solve_time_limit`, 26 `solved`, 2 `solved inaccurate`, and 1 `maximum iterations reached`. The remaining 84 are explicit `replay_error` records: `ValueError: Recorded dynamic-obstacle centers cannot be reconstructed from the event step.` They are preserved rather than imputed, excluded, or used to alter CBF values. The replay is diagnostic evidence only; it establishes neither a fallback rate nor safety, performance, scalability, or method-effect claims.

Before the next task, no Python process was running. Next: inspect the existing independently trained critical-ablation protocol and launch only an arm whose actor-observation contract, training configuration, and output path are explicitly frozen; no shared full-method checkpoint may be relabelled as an ablation.

## AAMAS 2027 - principal no-uncertainty ablation protocol frozen (2026-07-27)

The research brief defines the principal controlled ablation as the same delayed,
lossy delivered-packet predictive graph with `gamma_sigma = 0` and
`kappa_sigma = kappa_CBF = 0`. To make this executable without reusing a full
checkpoint, the independently trainable 5-UAV and 8-UAV profiles are
`configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml` and
`configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`.
Relative to the corresponding frozen full profiles, the only mechanism changes
are `graph_mode: predictive_graph`, `graph_uncertainty_risk_gain: 0.0`, and the
two CBF uncertainty-margin fields set to `0.0`; communication delay/loss,
delivered-packet prediction, dynamic obstacles, slack penalty 100, 20,000
iterations, and all optimization/scale values remain matched. The CBF change is
an explicit causal ablation, not a fallback-rate response or covert solver
tuning. `predictive_graph` causes the graph runner to pass zero uncertainty to
actor graph construction; it does not substitute neighbour truth.

No ablation training, evaluation, or result exists yet. The first pending job
is the independently trained 5-UAV seed `20260719`, with a distinct output
directory and CUDA neural execution; OSQP remains CPU-side. It must be verified
and documented before any serial next seed or six-scenario evaluation.

Configuration projection verified both profiles as `uses_predicted_knowledge:
true`, `uses_uncertainty: false`, graph uncertainty-risk gain `0.0`, and both
CBF uncertainty margins `0.0`. The actor/packet and CBF checks passed with
`39 passed` (`tests/test_predictive_conflict_graph.py`,
`tests/test_multi_uav_environment.py`, `tests/test_cbf_safety.py`), retaining
only 11 existing OSQP `PendingDeprecationWarning`s. `ruff check multiuav scripts
tests` passed, and `mypy --explicit-package-bases multiuav scripts` passed for
101 source files. The initial mypy invocation without explicit package bases
failed before analysis because `scripts/diagnose_cbf_failure.py` was discovered
under two module names; the successful explicit-package-bases invocation is the
authoritative static-check result.

The frozen protocol/configuration record was committed and pushed on
`codex/phase14-dynamic-world` as `b30f69f` (`docs: freeze uncertainty ablation
protocol`).

## AAMAS 2027 - first 5-UAV principal no-uncertainty ablation seed retained (2026-07-27)

The first independently trained 5-UAV principal no-uncertainty ablation
completed the frozen CUDA command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
--config configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml
--device cuda --seed 20260719 --num-uavs 5 --graph-mode predictive_graph
--total-steps 100000 --output-dir
outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260719`.
It launched at revision `b919a81` with configuration SHA-256
`6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`.
The final checkpoint, `summary.json`, `live_training_telemetry.json`,
TensorBoard data, and dedicated launcher stdout/stderr are retained under the
distinct output root. The final summary verifies five UAVs, CUDA network
execution, predicted delivered-packet knowledge, `uses_uncertainty: false`,
and CPU-side OSQP CBF; it therefore neither reuses the full-arm checkpoint nor
exposes neighbour truth to the actor.

It reached 100,080 transitions and 417 updates. Training telemetry retains one
`solve_time_limit` emergency fallback in 20,016 CBF decisions (decision index
4,756; 0.10070149996317923 seconds; 14,524 iterations; primal residual
0.0020588788764825863; dual residual 6.888436801064233e-05). The full position,
requested-velocity, delivered-knowledge, uncertainty, and dynamic-obstacle
context remains in the summary/telemetry. Initial and final training-script
evaluations each retain zero emergency fallbacks in 160 decisions; launcher
stderr has one matching notification line. CBF slack penalty, iteration cap,
solve-time limit, and tolerances were not changed.

No cell-level evaluation, replay, performance, safety, fallback-rate,
scalability, or method-effect conclusion follows from this one seed. No invalid
training attempt was created. With no Python process remaining, next: launch
seed `20260720` serially under the identical 5-UAV ablation protocol, retaining
its distinct raw telemetry before any evaluation or 8-UAV ablation launch.

## AAMAS 2027 - second 5-UAV principal no-uncertainty ablation seed retained (2026-07-27)

Seed `20260720` completed the identical independent CUDA command at launch
revision `2402d8b`, configuration SHA-256
`6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`, and
output `outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260720`.
Final checkpoint, summary, live telemetry, TensorBoard, and launcher logs are
retained. The summary verifies five UAVs, `predictive_graph`, CUDA network
execution, delivered-packet prediction, and disabled uncertainty graph input.
It reached 100,080 transitions and 417 updates. Training telemetry retains zero
emergency fallbacks in 20,016 CBF decisions; initial and final evaluations each
retain zero in 160. The launcher stderr nevertheless has one emergency-notice
line, so both raw sources are preserved as an explicit discrepancy rather than
merged or discarded. No solver/safety setting changed. This one seed proves no
performance, safety, scale, fallback-rate, or ablation effect. Next: document
and serially launch seed `20260721` with the same frozen protocol.

## AAMAS 2027 - third 5-UAV principal no-uncertainty ablation seed retained (2026-07-27)

Seed `20260721` completed independently at revision `d49401a`, frozen config
SHA-256 `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`, and
output `outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260721`.
Checkpoint, summary, telemetry, TensorBoard, and launcher logs are retained.
It reached 100,080 CUDA transitions and 417 updates. Training telemetry retains
one `solved inaccurate` fallback in 20,016 decisions (index 7,458; 20,000
iterations; 0.04322150000371039 seconds; full context retained); initial/final
evaluations retain zero in 160 each and stderr has one matching notice. No CBF
setting changed. This is single-seed diagnostics only, not a performance,
safety, fallback-rate, scale, or ablation conclusion. Next: launch seed 20260722
serially after documentation and a no-process preflight.

## AAMAS 2027 - fourth 5-UAV principal no-uncertainty ablation seed retained (2026-07-27)

Seed `20260722` completed independently with the same frozen 5-UAV predictive
no-uncertainty protocol and distinct output
`outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260722`.
Final checkpoint, summary, live telemetry, TensorBoard, and launcher logs are
retained. It reached 100,080 CUDA transitions and 417 updates. Training
telemetry retains one `solve_time_limit` emergency fallback in 20,016 decisions;
initial/final short evaluations remain separate and must be checked from the
retained summary. No safety parameter changed. This is seed-local diagnostics,
not a performance, safety, fallback-rate, scale, or ablation conclusion. Next:
document and serially launch seed 20260723.

## AAMAS 2027 - fifth 5-UAV principal no-uncertainty ablation seed retained (2026-07-27)

Seed `20260723` completed the frozen independent 5-UAV CUDA protocol at
`outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260723`.
Final checkpoint, summary, telemetry, TensorBoard, and launcher logs are
retained. It reached 100,080 transitions and 417 updates; training telemetry
retains two `solved inaccurate` emergency fallbacks in 20,016 decisions while
20,014 are `solved`. No CBF parameter changed. The five independently trained
5-UAV ablation seeds now exist, but no performance, safety, fallback-rate,
scale, or ablation effect is established until the matched six-scenario,
20-episode-per-cell evaluation and unchanged-protocol replay are retained.

## AAMAS 2027 - interrupted 5-UAV no-uncertainty evaluation attempt retained and excluded (2026-07-29)

The fourth launcher attempt used the frozen evaluator command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/evaluate_core_checkpoint.py --family graph_mappo --config configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml --checkpoint <seed-dir>/checkpoints/graph_mappo_final.pt --output-dir outputs/core_5uav_ablation_evaluations_rerun3 --experiment-name core_5uav_predictive_no_uncertainty_seed_${seed}_${scenario} --seed <seed> --num-uavs 5 --episodes 20 --scenario <canonical-scenario> --device cuda`, serially over the five frozen checkpoints and six canonical scenarios. It used CUDA for network inference and CPU OSQP, with no CBF parameter change. At revision `6516395`, the retained launcher stdout/stderr and all artifacts are under `outputs/core_5uav_ablation_evaluations_rerun3/`.

After interruption, read-only inspection found no running Python process, 24 cell directories, 24 `summary.json` files, and 24 raw JSONL files: all six scenarios for seeds `20260719`--`20260722`, with no partial cell directory and zero-byte launcher stderr. This is not the required 30-cell matrix and is therefore retained but excluded wholesale from aggregation, replay input selection, and any result statement. The next evaluation must start in a new root and complete a fresh 30-cell / 600-record audit; do not reuse or overwrite this aborted root.

## AAMAS 2027 - 5-UAV principal no-uncertainty evaluation and CBF replay retained (2026-07-29)

After a zero-Python-process preflight, the fresh serial launcher in `outputs/core_5uav_ablation_evaluations_rerun4/launcher.ps1` ran at revision `10a9c8e`. For each seed `20260719`--`20260723` and canonical scenario `nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, or `ood_communication_obstacle`, it invoked `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/evaluate_core_checkpoint.py --family graph_mappo --config configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml --checkpoint <seed-dir>/checkpoints/graph_mappo_final.pt --output-dir outputs/core_5uav_ablation_evaluations_rerun4 --experiment-name core_5uav_predictive_no_uncertainty_seed_${seed}_${scenario} --seed <seed> --num-uavs 5 --episodes 20 --scenario <scenario> --device cuda`. The frozen configuration SHA-256 is `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`; learned-network inference used CUDA and the OSQP CBF remained CPU-side.

Read-only post-run auditing verified exactly the 30 expected cell identities, 30 `summary.json` files, and 30 raw JSONL files beneath `outputs/core_5uav_ablation_evaluations_rerun4/`. Every JSONL contains exactly 20 nonblank parseable episode records: 600 records total, zero parse/identity errors, zero-byte launcher stderr, and no residual Python process. The prior roots `core_5uav_ablation_evaluations/`, `_rerun1/`, `_rerun2/`, and `_rerun3/` remain retained and excluded; only rerun4 is eligible for this ablation's future paired aggregation.

The unchanged-protocol diagnostic command `scripts/replay_cbf_fallbacks.py --output-jsonl outputs/cbf_diagnostics/predictive_no_uncertainty_5uav_5seed_replay_20260729.jsonl --max-iterations 20000 --max-solve-time-seconds 0.1 --slack-penalty 100.0 --uncertainty-margin-gain 0.0 --max-uncertainty-margin 0.0 <5 training summaries> <30 rerun4 summaries> <30 rerun4 JSONL files>` retained all 65 source telemetry files. Its JSONL and companion `.summary.json` contain five source fallback events (two recorded `solve_time_limit`, three recorded `solved inaccurate`) and zero replay errors; four replays solved, one was `solved inaccurate`, and one replay used the emergency fallback. The 30 valid evaluation cells supplied no additional emergency event. These are diagnostic/reproducibility records only: they do not establish safety, a fallback rate, performance, scalability, or an ablation effect, and no CBF parameter was changed.

## AAMAS 2027 - first 8-UAV principal no-uncertainty ablation seed retained (2026-07-29)

Seed `20260719` completed the frozen independent CUDA command at launch revision `0f6e011`: `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml --device cuda --seed 20260719 --num-uavs 8 --graph-mode predictive_graph --total-steps 100000 --output-dir outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260719`. The frozen configuration SHA-256 is `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`; it retains delivered-packet prediction (`uses_predicted_knowledge: true`) while disabling uncertainty inputs/margins (`uses_uncertainty: false`, both relevant gains/max margin zero), with CUDA network execution and CPU OSQP.

Distinct final checkpoint, summary, live telemetry, TensorBoard, and launcher stdout/stderr are retained beneath `outputs/core_8uav_ablation/`. The final summary records 100,224 transitions and 261 updates. Training telemetry retains 182 emergency fallbacks in 12,528 CBF decisions (177 `solve_time_limit`, four `solved inaccurate`, and one `maximum iterations reached`); the initial script evaluation retains 15 `solve_time_limit` fallbacks in 160 decisions, while the final script evaluation retains zero in 160. No slack, iteration cap, solve-time limit, tolerance, observation contract, or controller was changed. This seed-local diagnostic evidence establishes neither safety, a fallback rate, performance, scalability, nor an ablation effect. Next: after verifying no Python process, launch seed `20260720` serially with the identical frozen configuration and a distinct output path; preserve all raw context.

## AAMAS 2027 - second 8-UAV principal no-uncertainty ablation seed retained (2026-07-29)

Seed `20260720` completed independently at launch revision `96267be` with the same frozen 8-UAV predictive-no-uncertainty command and configuration SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`, retaining CUDA network execution and CPU OSQP. Its distinct root `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260720` contains final checkpoint, summary, live telemetry, TensorBoard, and retained launcher stdout/stderr.

The final summary records 100,224 transitions and 261 updates, with predicted delivered-packet knowledge enabled and uncertainty input disabled. Training retains 191 emergency fallbacks in 12,528 decisions (190 `solve_time_limit`, one `maximum iterations reached`); initial script evaluation has zero in 160; final script evaluation has 13 `solve_time_limit` fallbacks in 160. All 204 contexts and launcher logs are preserved. No CBF slack, iteration limit, time limit, tolerance, policy contract, or controller changed. This remains single-seed diagnostic evidence, not a safety, fallback-rate, performance, scalability, or ablation-effect conclusion. Next: after a zero-process preflight, train seed `20260721` serially with the same frozen protocol.

## AAMAS 2027 - third 8-UAV principal no-uncertainty ablation seed retained (2026-07-29)

Seed `20260721` completed independently at launch revision `08427a2` using the unchanged frozen 8-UAV predictive-no-uncertainty configuration SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`. The distinct root `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260721` retains final checkpoint, summary, live telemetry, TensorBoard, and launcher stdout/stderr; it reached 100,224 transitions and 261 updates with CUDA network execution, CPU OSQP, predicted delivered-packet knowledge, and uncertainty input disabled.

Training telemetry retains 139 emergency fallbacks in 12,528 decisions (138 `solve_time_limit`, one `solved inaccurate`); initial/final script evaluations retain 13/10 `solve_time_limit` fallbacks in 160 decisions each. All 162 event contexts are retained. No safety parameter, observation contract, or controller changed. This is a seed-local diagnostic, not evidence for safety, a fallback rate, performance, scalability, or a causal ablation effect. Next: after zero-process preflight, launch seed `20260722` serially under the same frozen protocol.

## AAMAS 2027 - fourth 8-UAV principal no-uncertainty ablation seed retained (2026-07-29)

Seed `20260722` completed independently at launch revision `59e9984` with the same frozen configuration SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`. Its distinct artifact root `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260722` retains final checkpoint, summary, live telemetry, TensorBoard, and launcher stdout/stderr. It reached 100,224 transitions and 261 updates with predicted delivered-packet knowledge, uncertainty input disabled, CUDA network execution, and CPU OSQP.

Training telemetry retains 218 emergency fallbacks in 12,528 decisions (217 `solve_time_limit`, one `solved inaccurate`); initial/final script evaluations retain 5/11 `solve_time_limit` fallbacks in 160 decisions each. All 234 contexts are retained. No safety or controller parameter changed. This single-seed diagnostic does not establish safety, a fallback rate, performance, scalability, or a causal ablation effect. Next: after zero-process preflight, serially train final seed `20260723` with this frozen protocol.

## AAMAS 2027 - fifth 8-UAV principal no-uncertainty ablation seed retained (2026-07-29)

Seed `20260723` completed the frozen independent 8-UAV CUDA command at launch revision `278129e` using configuration SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`. The distinct final checkpoint, summary, live telemetry, TensorBoard, and launcher stdout/stderr are retained under `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260723`. It reached 100,224 transitions and 261 updates with delivered-packet predictive knowledge, disabled uncertainty input, CUDA neural execution, and CPU OSQP.

Training telemetry retains three emergency fallbacks in 12,528 decisions (one `maximum iterations reached`, two `solved inaccurate`); initial script evaluation has eight `solve_time_limit` fallbacks in 160 decisions and final script evaluation has zero in 160. All 11 contexts are retained. No CBF parameter, observation contract, or controller changed. The five independent 8-UAV ablation final checkpoints now exist, but no performance, safety, fallback-rate, scalability, or ablation-effect conclusion is established until a fresh six-scenario 20-episode matrix and unchanged all-source replay are audited. Next: evaluate all five checkpoints serially into a new root, retaining JSONL.

## AAMAS 2027 - 8-UAV principal no-uncertainty evaluation and CBF replay retained (2026-07-29)

At revision `408a115`, after a corrected zero-training-process preflight, the retained launcher `outputs/core_8uav_ablation_evaluations_20260729/launcher.ps1` serially evaluated each final checkpoint from `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260719` through `..._20260723`. It used `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/evaluate_core_checkpoint.py --family graph_mappo --config configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml --checkpoint <seed-dir>/checkpoints/graph_mappo_final.pt --output-dir outputs/core_8uav_ablation_evaluations_20260729 --experiment-name core_8uav_predictive_no_uncertainty_seed_${seed}_${scenario} --seed <seed> --num-uavs 8 --episodes 20 --scenario <scenario> --device cuda` for canonical scenarios `nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, and `ood_communication_obstacle`. The config SHA-256 is `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`; neural inference used CUDA and OSQP remained CPU-side.

A read-only audit verified exactly the 30 expected seed-scenario directories, 30 `summary.json` files, and 30 raw JSONL files. Each JSONL has exactly 20 parseable nonblank episode records: 600 records total, zero parse/identity errors, no remaining evaluator process, and zero-byte launcher stderr. The initial rendering typo occurred before an evaluator launched and created no result cell; it is retained as a preflight failure only. This valid root is eligible for later paired aggregation, but no performance, safety, scale, or ablation conclusion is made here.

The unchanged-protocol command `scripts/replay_cbf_fallbacks.py --output-jsonl outputs/cbf_diagnostics/predictive_no_uncertainty_8uav_5seed_replay_20260729.jsonl --max-iterations 20000 --max-solve-time-seconds 0.1 --slack-penalty 100.0 --uncertainty-margin-gain 0.0 --max-uncertainty-margin 0.0 <5 training summaries> <30 valid summaries> <30 valid JSONL files>` retains all 65 source files. Its JSONL and companion `.summary.json` retain 808 source events (797 recorded `solve_time_limit`, eight `solved inaccurate`, three `maximum iterations reached`) with zero replay errors. Replay statuses are 793 `solved`, five `solved inaccurate`, nine `solve_time_limit`, and one `maximum iterations reached`; 15 replay events used emergency fallback. Preserve all records and do not alter CBF values. These diagnostics establish neither a fallback rate nor safety, performance, scalability, or a causal ablation effect. Next: audit the matching full-method 8-UAV matrix and define a paired invalid-attempt-excluding aggregation before any comparison statement.

## AAMAS 2027 - matched 8-UAV full-method/ablation input audit retained (2026-07-29)

Read-only audit of `outputs/core_8uav_evaluations/` (uncertainty-aware full method) and `outputs/core_8uav_ablation_evaluations_20260729/` (principal no-uncertainty ablation) verified exactly 30 expected seed-scenario directories, 30 summaries, 30 JSONL files, and 600 parseable records per arm, with no missing/unexpected identities or parse/count errors. Raw records expose common `seed`, `scenario`, and `episode` fields, so each arm has the same five seed identifiers, six scenario identifiers, and 20 indexed episodes per cell.

This establishes data eligibility, not a comparison result. Any later inferential aggregation must first summarize each 20-episode seed-scenario cell, then treat the five independently trained seeds as the independent replicates for each scenario; it must not pseudo-replicate the 100 episode records as 100 independently trained policies. It must exclude every invalid/interrupted evaluation root previously recorded in `known_issues.md`, retain raw per-episode JSONL, report missing/error cells explicitly, and distinguish evaluation metrics from diagnostic CBF replay outcomes. Next: implement or audit a read-only paired-summary artifact under these rules before writing a method-effect claim.

## AAMAS 2027 - descriptive paired 8-UAV summary artifact retained (2026-07-29)

Added `scripts/summarize_paired_checkpoint_evaluations.py` and its test `tests/test_paired_checkpoint_summary.py`. The read-only tool validates one 20-episode JSONL per seed-scenario cell, rejects unmatched cell identities, summarizes each metric within a cell, and then reports only descriptive paired seed-level values (mean and sample standard deviation) across the five trained seeds. It explicitly writes `independent_unit: trained_seed`, a pseudo-replication guard, and a no-significance/no-method-effect claim boundary. It neither trains, evaluates, tunes, nor changes a controller/CBF setting.

The verified command `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/summarize_paired_checkpoint_evaluations.py --reference-root outputs/core_8uav_evaluations --treatment-root outputs/core_8uav_ablation_evaluations_20260729 --output-json outputs/paired_summaries/8uav_full_vs_no_uncertainty_20260729.json` retained a six-scenario, five-seed, 20-episodes-per-seed descriptive artifact. It is input provenance and descriptive reporting only, not a statistical significance, safety, performance, scalability, or causal-ablation conclusion. TDD evidence: the new tests first failed because the module did not exist, then passed (2 passed). Full Ruff passed; `mypy --explicit-package-bases multiuav scripts` passed 102 source files. Full pytest retained one pre-existing unrelated failure (`tests/test_legacy_audit.py`: observed restored MATLAB `.m` count 83 while test/doc expect 81), with 148 tests passing and 28 existing OSQP warnings. Next: resolve or explicitly quarantine that legacy-inventory drift separately, and do not use the paired descriptive artifact as a submission claim until the complete multi-arm analysis plan and literature evidence are ready.

## AAMAS 2027 - official IJCAI DHCG source retrieval retained (2026-07-29)

At revision `7a065bd`, primary-source inspection followed DOI
`10.24963/ijcai.2023/24` to the official IJCAI proceedings page and reviewed its
metadata and abstract. The page identifies the paper, proceedings pages
208--216, and official PDF `https://www.ijcai.org/proceedings/2023/0024.pdf`.
The abstract supports only a high-level record: DHCG learns message-based
directed acyclic communication topologies end-to-end, applies an acyclicity
constraint/reward and a projection, and reports policy/value variants on listed
cooperative MARL benchmarks. It does not by itself establish packet timing,
predicted neighbour-state uncertainty features, dynamic-obstacle UAV settings,
or CBF behavior.

The browser's two official-PDF download attempts (one navigate-mode attempt
and one ordinary download into ignored `tmp/pdfs/`) each timed out and produced
no local file. The retained command record is
`C:\\Users\\yun96\\.claude\\skills\\gstack\\browse\\dist\\browse.exe goto https://doi.org/10.24963/ijcai.2023/24`,
followed by `download https://www.ijcai.org/proceedings/2023/0024.pdf` first
with `--navigate` and then to `tmp/pdfs/liu_dhcg_ijcai2023.pdf`; no PDF parsing
or full-text inference was performed. The failed retrieval is recorded in `docs/related_work_matrix.md` and
`docs/known_issues.md`; no experimental output, policy, CBF configuration, or
claim boundary changed. Next: obtain the official PDF or an author-provided
primary copy and read it before making any DHCG method-difference or novelty
statement.

## AAMAS 2027 - official IEEE multi-UAV source retrieval block retained (2026-07-29)

At revision `fe6be1f`, primary-source inspection used
`C:\\Users\\yun96\\.claude\\skills\\gstack\\browse\\dist\\browse.exe goto https://doi.org/10.1109/LCSYS.2021.3138941`.
The DOI resolved to official IEEE Xplore document `9663555`, but the page
returned `Unusual Traffic Detected (Error 418)` before presenting an abstract
or PDF. No download, PDF parsing, source inference, experimental output, policy,
or CBF configuration change occurred. The evidence ledger therefore remains
metadata-only and records the block in `docs/related_work_matrix.md` and
`docs/known_issues.md`. Next: obtain an authorized IEEE or author-provided
primary copy; do not infer communication, obstacle, or safety assumptions from
the title or metadata.

## AAMAS 2027 - descriptive four-arm 3-UAV summary artifact retained (2026-07-29)

At revision `2c9b901`, added the read-only
`scripts/summarize_multiarm_checkpoint_evaluations.py` and
`tests/test_multiarm_checkpoint_summary.py`. The tool accepts named evaluation
roots and an optional experiment-directory prefix per arm. It validates every
selected JSONL cell has 20 nonblank records with one seed/scenario identity and
episode indices 0--19, requires exact arm-to-arm cell identity matching, then
aggregates episodes within each seed-scenario cell and reports only seed-level
means and sample standard deviations. It writes the explicit boundaries
`independent_unit: trained_seed` and no significance, method-effect, safety, or
causal claim; it neither trains/evaluates a policy nor changes CBF values. Its
per-scenario output separates task-level fields (`task_metrics`) from
solver-shield fields (`cbf_diagnostic_metrics`) so CBF fallback/intervention
diagnostics cannot be silently folded into a task-performance result.

The retained command was
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts\\summarize_multiarm_checkpoint_evaluations.py --arm mlp_mappo=outputs\\core_3uav_evaluations --cell-prefix mlp_mappo=core_3uav_mlp_mappo_ --arm raw_graph_mappo=outputs\\core_3uav_evaluations --cell-prefix raw_graph_mappo=core_3uav_raw_graph_ --arm predictive_graph_no_uncertainty=outputs\\core_3uav_evaluations --cell-prefix predictive_graph_no_uncertainty=core_3uav_predictive_graph_ --arm uncertainty_predictive_graph=outputs\\core_3uav_evaluations --cell-prefix uncertainty_predictive_graph=core_3uav_uncertainty_predictive_graph_ --output-json outputs\\paired_summaries\\3uav_four_arm_descriptive_20260729.json`.
Its output has four named arms, six scenarios, five matched trained seeds, and
20 episodes per seed-scenario cell. For MLP, the two empty original directories
`core_3uav_mlp_mappo_seed_20260719_nominal` and
`..._ood_communication_obstacle` remain excluded; the existing valid
`..._nominal_schemafix_rngfix_rerun2` and
`..._ood_communication_obstacle_trajectoryfix_rerun1` JSONL cells supply those
two identities. The three GraphMAPPO prefixes each select 30 valid cells / 600
records. A first read-only PowerShell audit command had an empty-pipe parse
error before it read/wrote an artifact; the corrected audit found the two empty
MLP directories and no issue in the other three arms.

TDD evidence: the initial missing-module test failed, the direct-CLI test then
failed before the import-path repair, and the prefix-selection test failed
before that behavior existed. After minimal implementation, paired and
multi-arm target tests passed (6 passed), Ruff passed, and
`mypy --explicit-package-bases multiuav scripts` passed 103 sources. Full
pytest remains an unrelated retained failure: 152 passed, 1 failed because the
legacy audit asserts 81 restored MATLAB files while 83 are present, with 28
existing OSQP warnings. This artifact is descriptive input provenance only;
it proves neither performance, significance, safety, scale, a fallback rate,
nor a causal four-arm comparison. The final output has 11 task metrics and five
CBF diagnostic metrics per scenario in distinct fields. Next: define a
publication-facing analysis plan that keeps CBF diagnostics distinct and
excludes every invalid root.

## AAMAS 2027 - descriptive paired 5-UAV full/ablation summary retained (2026-07-29)

At revision `cb82ef6`, the existing read-only paired-summary command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts\\summarize_paired_checkpoint_evaluations.py --reference-root outputs\\core_5uav_evaluations --treatment-root outputs\\core_5uav_ablation_evaluations_rerun4 --output-json outputs\\paired_summaries\\5uav_full_vs_no_uncertainty_20260729.json`
validated both roots and wrote the retained descriptive artifact. The reference
is the uncertainty-aware predictive-graph five-UAV matrix; treatment is the
independently trained predictive-no-uncertainty matrix under frozen config
`configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`
(SHA-256 `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`).
Network evaluation used CUDA and OSQP remained CPU-side; no policy or CBF
setting changed.

The validator accepted six scenarios, five common trained seeds, and 20 indexed
episodes per seed-scenario cell: 30 matched cells / 600 raw JSONL records per
arm. Only `outputs/core_5uav_ablation_evaluations_rerun4/` is selected for the
ablation. The retained invalid/excluded roots are
`outputs/core_5uav_ablation_evaluations/`, `_rerun1/`, `_rerun2/`, and the
interrupted 24-cell `_rerun3/`; none was read by this command. The separate
all-source ablation CBF replay remains
`outputs/cbf_diagnostics/predictive_no_uncertainty_5uav_5seed_replay_20260729.jsonl`
with its companion summary and unchanged solver values. This new JSON summary
is a seed-level descriptive paired artifact only, not a significance,
performance, safety, fallback-rate, scalability, or causal ablation result.
Next: retain this input provenance in the publication analysis plan and extend
the same invalid-root discipline to every remaining scale/ablation comparison.

## AAMAS 2027 - paired summaries separate task and CBF diagnostic fields (2026-07-29)

At revision `292c2e7`, refined the existing read-only
`scripts/summarize_paired_checkpoint_evaluations.py` and its test so every
scenario has independent `task_metrics` and `cbf_diagnostic_metrics` objects.
The tool still validates 20 records/cell and exact paired identities, aggregates
only within seed-scenario cells, and uses trained seed as the independent unit.
It still emits no significance or method-effect claim and does not train,
evaluate, or change CBF parameters. This change prevents CBF fallback,
intervention, correction, and solve-time fields from being silently presented
as task-performance metrics.

The retained regeneration commands were
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts\\summarize_paired_checkpoint_evaluations.py --reference-root outputs\\core_5uav_evaluations --treatment-root outputs\\core_5uav_ablation_evaluations_rerun4 --output-json outputs\\paired_summaries\\5uav_full_vs_no_uncertainty_20260729.json`
and
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts\\summarize_paired_checkpoint_evaluations.py --reference-root outputs\\core_8uav_evaluations --treatment-root outputs\\core_8uav_ablation_evaluations_20260729 --output-json outputs\\paired_summaries\\8uav_full_vs_no_uncertainty_20260729.json`.
Both outputs retain six scenarios, five common trained seeds, 20 episodes per
seed-scenario cell, 11 task fields, and five CBF diagnostic fields. The prior
five-UAV invalid-root exclusions and the 8-UAV valid-root selection remain
unchanged; all original JSONL and separate all-source CBF replays remain raw
evidence. Target paired/multi-arm tests passed (6), Ruff passed, and
`mypy --explicit-package-bases multiuav scripts` passed 103 sources. Full
pytest remains 152 passed / 1 unrelated legacy-inventory failure (83 MATLAB
files observed vs 81 asserted) / 28 OSQP warnings. This proves reporting-field
separation only, not performance, significance, safety, fallback rate, scale,
or a causal effect. Next: write a publication analysis plan that names eligible
artifacts, exclusions, and the no-claim boundary before any interpretation.

## AAMAS 2027 - publication-facing analysis plan retained (2026-07-29)

At revision `019065b`, added `docs/aamas2027_analysis_plan.md`. It fixes the
operational reporting scope for the validated 3-UAV four-arm and 5/8-UAV paired
artifacts: eligible roots, every known invalid-root exclusion, three exact
regeneration commands, configuration hashes where applicable, five-seed
independent unit, 20-episode within-cell aggregation, separate task/CBF fields,
and raw CBF replay provenance. It also records the current 152 passed / one
legacy-inventory failure / 28 OSQP warning verification state and the remaining
literature, inference, safety-claim, and external-submission gates. It changes
no experiment, policy, packet-observation contract, CBF setting, or result.
The document is an operational retrospective plan rather than a
preregistration; it establishes no statistical, safety, performance,
scalability, novelty, or causal conclusion. Next: use it to audit any future
manuscript table/figure and continue primary-source full-text retrieval.

## AAMAS 2027 - Neural Graph CBF official extended abstract reviewed (2026-07-29)

At revision `6cfc789`, primary-source verification followed the official AAMAS
2026 proceedings table of contents to Deng et al.'s page-3232 entry and its
official PDF `https://ifaamas.org/Proceedings/aamas2026/pdfs/KRTJ4225.pdf`.
The PDF completed after a browser download timeout; its SHA-256 is
`0AE5CA03242EDE2EB5E60822729EE6B6279F4FA4C14B836E411658260E09D29A`.
Text extraction and bundled-runtime `pypdf` verify an unencrypted three-page
extended abstract containing method, experiment, and reference sections.

The reviewed source describes a frozen CTDE MAPPO reference policy plus a
GNN-parameterized graph-CBF additive safety correction. It uses pointwise CBF
QP actions for supervised action matching, barrier validity/invariance losses,
and modified MPE Simple Spread boundary/collision evaluation. This establishes a
direct neural graph-CBF comparator, but the short text does not document packet
delay/loss/staleness, delivered-packet prediction uncertainty, or dynamic
obstacles. It therefore supports no first-combination, safety-guarantee, or
method-superiority claim for this project.

The document-generation commands were the gstack browser download of the
official proceedings PDF and `pdftotext`; the bundled Python verification
confirmed page count and the presence of method/experiments/references text.
The system Poppler wrappers could not render PNGs because their target path was
missing, and browser PDF navigation returned to a proceedings screenshot; this
visual-layout limitation is retained in `docs/known_issues.md`. No code,
experiment, CBF parameter, or actor/critic observation contract changed. Next:
continue retrieving the remaining primary PDFs and inspect a longer Deng version
if one becomes available before writing a stronger comparison.

## AAMAS 2027 - actor-information isolation repair (2026-07-29)

Starting from revision `55abb87`, a static audit found three actor-information
violations: disabling communication still supplied neighbour truth through a
channel fallback; current sender activity could erase an already delivered
packet; and GraphMAPPO attention let a receiver action depend on another
node's private local observation and current activity. The repair changes only
the actor-information protocol: a disabled channel yields padded empty
neighbour rows; received packets remain visible until their configured
staleness expiry; graph edges use receiver-local self truth plus delivered
knowledge; peer node features and peer activity cannot affect a receiver
action. Neighbour-goal direction was removed from actor edge features because
neighbour goals are not packet payloads. The critic/execution-side CBF
interface and all CBF numerical values are unchanged; neural operations remain
PyTorch/CUDA-compatible and OSQP remains CPU-side.

The test-first red evidence was the new no-communication local-observation,
delivered-packet persistence, and Graph Actor peer-feature/activity/edge tests.
After repair, `D:\\anaconda3\\envs\\multiuav_rl\\python.exe -m pytest -q
tests\\test_predictive_conflict_graph.py tests\\test_dynamic_graph_baselines.py
tests\\test_multi_uav_environment.py tests\\test_communication.py` passed 39
tests (three existing OSQP `PendingDeprecationWarning`s); Ruff passed and
`mypy --explicit-package-bases multiuav scripts` passed 103 sources.

The complete `D:\\anaconda3\\envs\\multiuav_rl\\python.exe -m pytest -q`
run after the repair has 157 passed and one retained unrelated failure:
`tests/test_legacy_audit.py::LegacyAuditDocumentTests::test_inventory_documents_the_observed_legacy_structure`
still expects 81 restored MATLAB files while 83 are present. It also emits 28
existing OSQP warnings. The legacy inventory/test was not edited because it is
outside this protocol repair.

This proves regression coverage for the stated information boundaries, not
learning, task performance, safety, fallback rate, or scalability. All prior
MLP and GraphMAPPO checkpoints, six-scenario JSONL, paired/multi-arm summaries,
and CBF replays are retained but protocol-ineligible because they were produced
before this repair; the explicit root-level exclusions are in
`docs/aamas2027_analysis_plan.md`. No training/evaluation process was launched
and no output was deleted or overwritten. Next: commit the isolation repair,
run the full suite, then begin a new-root five-seed 3-UAV MLP rerun on CUDA
before any GraphMAPPO rerun or reporting.

The repair was committed locally as `0e048cd` (`fix: isolate actor information
from peer truth`). The required push command
`git push github.com:youshenbupo/multi-uav-rl-path-planning.git
codex/phase14-dynamic-world` failed before any remote update with
`Permission denied (publickey)`. No credentials, remote configuration, or
history were changed. Restore authorized SSH access, then push this commit and
the documentation-only follow-up; `.gitignore` remains user-owned and unstaged.

## AAMAS 2027 - post-isolation MLP rerun started (2026-07-29)

After a zero-training-process preflight and CUDA check (PyTorch
`2.13.0+cu130`, RTX 5060 Laptop GPU), the sole serial first rerun was launched
from local revision `00a970f`:
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts\\train_mappo.py
--config configs\\rl\\dynamic_mappo_baseline.yaml --device cuda --seed
20260719 --num-uavs 3 --total-steps 100000 --output-dir
outputs\\core_3uav_post_actor_isolation\\core_3uav_mlp_mappo_seed_20260719`.
The frozen configuration SHA-256 is
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`.
Network training/inference uses CUDA and OSQP remains CPU-side under unchanged
slack/iteration/time/tolerance settings. At launch verification, PID `44928`
was alive; raw stdout/stderr are
`outputs/core_3uav_post_actor_isolation_seed_20260719.stdout.log` and `.stderr.log`,
and live telemetry is under the new output root. The stderr already contains a
retained `solved inaccurate` emergency-fallback notice; do not tune or restart
in response. This is an active training attempt, not a result. Do not start
another training/evaluation process until it naturally exits; then audit final
checkpoint, summary, telemetry, and every raw fallback before launching seed
`20260720`.

## AAMAS 2027 - post-isolation MLP seed 20260719 retained (2026-07-29)

The sole post-isolation 3-UAV MLP seed `20260719` naturally completed at
100,032 transitions and 1,042 updates under the exact CUDA command and frozen
configuration recorded above (training revision `00a970f`; config SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`). Its
final checkpoint is
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260719/checkpoints/mappo_final.pt`;
the summary, TensorBoard data, and raw live telemetry remain in the same new
root, with launcher logs in `outputs/core_3uav_post_actor_isolation_seed_20260719.*.log`.

Training telemetry retains three emergency events across 33,344 decisions
(one `solved inaccurate`, two `maximum iterations reached`); its initial and
final short script evaluations retain zero fallbacks over 160 and 98 decisions
respectively. The append-safe read-only replay command
`D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts\\replay_cbf_fallbacks.py
--output-jsonl outputs\\cbf_diagnostics\\post_actor_isolation_mlp_seed_20260719_replay_20260729.jsonl
--max-iterations 20000 --max-solve-time-seconds 0.1 --slack-penalty 100.0
--uncertainty-margin-gain 0.5 --max-uncertainty-margin 5.0
outputs\\core_3uav_post_actor_isolation\\core_3uav_mlp_mappo_seed_20260719\\live_training_telemetry.json`
retains all three source events with zero replay errors. Under exactly those
unchanged values, one replay remains `solved inaccurate` with emergency
fallback and two replay as `solved`; the latter do not authorize changing
slack, iteration limits, tolerances, or time limits.

This proves only a completed, auditable first independent post-isolation
training attempt and its diagnostic replay. It does not establish learning,
performance, safety, a fallback rate, or a valid five-seed comparison. No
JSONL checkpoint evaluation has started. Next: after confirming no process and
an absent target path, launch seed `20260720` serially with the same frozen
protocol.

## AAMAS 2027 - post-isolation MLP seed 20260720 retained (2026-07-29)

After a zero-process preflight, the sole serial seed `20260720` completed under
the same frozen CUDA command/configuration as seed `20260719`, changing only
`--seed 20260720` and output root
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260720`
(training revision `a4e9d8d`; config SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`). It
naturally reached 100,032 transitions / 1,042 updates on CUDA. Its final
checkpoint, summary, TensorBoard, and raw telemetry are retained in that root;
the unique launcher logs are
`outputs/core_3uav_post_actor_isolation_seed_20260720.stdout.log` and
`.stderr.log`.

Training telemetry retains one `maximum iterations reached` emergency fallback
in 33,344 decisions; initial/final script evaluations retain zero fallbacks in
160/82 decisions. The append-safe replay
`outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260720_replay_20260729.jsonl`
and companion summary use unchanged 20,000-iteration, 0.1-second,
penalty-100, uncertainty-gain-0.5/cap-5 values. It retains the one source event
with zero replay errors; the replay also reaches `maximum iterations reached`
and uses emergency fallback. Preserve it without numerical tuning. This is a
second independent auditable training attempt, not a performance, safety,
fallback-rate, or five-seed conclusion. Next: serial seed `20260721` under the
same protocol; no checkpoint evaluation starts yet.

## AAMAS 2027 - telemetry-lock repair and invalid MLP seed 20260721 attempt (2026-07-29)

The first post-isolation seed `20260721` attempt at revision `2122b14` stopped
at 91,776 transitions without a final checkpoint/summary. Its complete retained
root is
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260721/`,
with its launcher logs in the matching top-level `outputs` paths and a new
`ABORTED.json` exclusion marker. The direct stderr cause is
`PermissionError [WinError 5]` while atomically replacing
`live_training_telemetry.json`; all partial checkpoints and telemetry, including
eight initial-evaluation `solved inaccurate` fallback contexts, are preserved
but invalid for every aggregation.

The minimal repair in `multiuav/learning/telemetry.py` retains atomic replace
but retries a transient `PermissionError` at most five times with a 50ms delay;
it does not alter rollout, actor, CBF, slack, iteration, tolerance, or time
settings. TDD evidence: the new transient-lock test failed first at the old
single replacement, then passed after the bounded retry. Target telemetry tests
passed 3 with two existing OSQP warnings; Ruff and explicit-package mypy passed
103 sources. The full suite now has 158 passed and one retained unrelated
legacy-inventory failure (83 MATLAB files observed vs 81 asserted), plus 28
existing OSQP warnings. This validates telemetry persistence under the mocked
transient lock, not training performance or a solver property. Next: commit the
repair/documentation, preserve the invalid root, and rerun seed `20260721`
under a unique `telemetryretry1` root before seeds `20260722`/`20260723`.

## AAMAS 2027 - post-isolation MLP seed 20260721 retry retained (2026-07-29)

The fresh retry command used seed `20260721`, CUDA, unchanged frozen
configuration SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`, revision
`0b7d402`, and unique root
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260721_telemetryretry1`.
It naturally completed 100,032 transitions / 1,042 updates, with final
checkpoint, summary, TensorBoard and raw telemetry retained. This is the only
valid `20260721` post-isolation candidate; the earlier same-seed directory
remains invalid and untouched.

Training and final script evaluation retain zero CBF fallback events over
33,344/103 decisions. The initial untrained-policy evaluation retains eight
`solved inaccurate` fallback contexts over 160 decisions. The full-source
replay at
`outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260721_telemetryretry1_replay_20260729.jsonl`
and companion summary preserves all eight sources with zero replay errors under
unchanged 20,000-iteration, 0.1-second, penalty-100,
uncertainty-gain-0.5/cap-5 values. Seven replays return `maximum iterations
reached`, one remains `solved inaccurate`, and all eight use emergency fallback.
These diagnostic outcomes must not trigger a numerical change. This establishes
only a third completed independent post-isolation MLP training artifact; no
performance, safety, fallback-rate, or five-seed conclusion exists. Next:
serially train seed `20260722` under a new root; do not begin evaluations yet.

## AAMAS 2027 - post-isolation MLP seed 20260722 retained (2026-07-29)

The sole serial CUDA seed `20260722` at revision `73112f6` used the unchanged
frozen configuration SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46` and root
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260722`.
It naturally completed 100,032 transitions / 1,042 updates with final
checkpoint, summary, TensorBoard, raw telemetry, and unique launcher logs.
Training/final script evaluation retain zero fallbacks in 33,344/89 decisions;
initial untrained-policy evaluation retains 13 events in 160 decisions.

The complete per-seed source replay is
`outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260722_replay_20260729.jsonl`
plus summary. Its unchanged 20,000-iteration, 0.1-second, penalty-100,
uncertainty-gain-0.5/cap-5 protocol retains 13 events with zero replay errors:
nine recorded `solve_time_limit` and four `maximum iterations reached`; replay
has ten `solve_time_limit` and three `maximum iterations reached`, all 13 with
emergency fallback. No CBF value changed. This is diagnostic evidence for a
fourth independent seed only, not a performance, safety, or fallback-rate
conclusion. Next: launch final MLP seed `20260723` serially before any checkpoint
evaluation.

## AAMAS 2027 - five post-isolation 3-UAV MLP seeds complete (2026-07-29)

The five valid independent CUDA training artifacts are now
`...seed_20260719`, `...seed_20260720`,
`...seed_20260721_telemetryretry1`, `...seed_20260722`, and
`...seed_20260723` beneath `outputs/core_3uav_post_actor_isolation/`. Each
reached 100,032 transitions / 1,042 updates with its own final checkpoint,
summary, TensorBoard and raw telemetry under frozen config SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`. The
earlier `seed_20260721` directory with `ABORTED.json` remains invalid/excluded,
not a sixth seed.

The all-source five-seed training/initial/final telemetry replay is
`outputs/cbf_diagnostics/post_actor_isolation_mlp_5seed_training_replay_20260729.jsonl`
plus summary. With unchanged 20,000 iterations, 0.1 seconds, penalty 100,
uncertainty gain 0.5 and cap 5.0, it retains all 26 source events from five
telemetry files with zero replay errors: recorded 10 `solved inaccurate`, 7
`maximum iterations reached`, and 9 `solve_time_limit`; replay 2 `solved
inaccurate`, 3 `solved`, 11 `maximum iterations reached`, and 10
`solve_time_limit`, with 23 replay fallbacks. All source and replay JSONL stay
raw; no parameter was changed. This establishes only training/replay
provenance, not a performance, safety, fallback-rate, significance, or baseline
claim. Next: serially evaluate every final checkpoint in six canonical scenarios
at 20 episodes/cell into a new root, retaining JSONL before GraphMAPPO work.

## AAMAS 2027 - post-isolation MLP seed 20260719 six-scenario matrix retained (2026-07-29)

At revision `e625af5`, the final checkpoint
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260719/checkpoints/mappo_final.pt`
was evaluated on CUDA with CPU OSQP under the unchanged frozen configuration
SHA-256 `794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`.
The command form was `D:\anaconda3\envs\multiuav_rl\python.exe
scripts\evaluate_core_checkpoint.py --family mappo --config
configs\rl\dynamic_mappo_baseline.yaml --checkpoint <above> --output-dir
outputs\core_3uav_post_actor_isolation_evaluations --experiment-name <cell>
--seed 20260719 --num-uavs 3 --episodes 20 --scenario <scenario> --device cuda`.
The six distinct `<cell>` directories cover `nominal`, `delay_only`,
`loss_only`, `dynamic_only`, `combined`, and `ood_communication_obstacle`.

Each cell under `outputs/core_3uav_post_actor_isolation_evaluations/` retains
its `summary.json` and `raw_results/seed_20260719.jsonl`. Direct parsing
verified 20 records in each JSONL, fixed identity `(seed=20260719, scenario=<cell
scenario>)`, six expected scenarios, and hence 120 retained raw episode
records. The `nominal` launcher logs are
`outputs/core_3uav_post_actor_isolation_eval_seed_20260719_nominal.stdout.log`
and `.stderr.log`; subsequent scenario artifacts retain their per-cell output
directories. This proves only protocol-complete evaluation provenance for one
checkpoint. It does not establish performance, safety, fallback rate,
significance, or a cross-seed conclusion. Next: perform the same no-overwrite,
20-episode six-scenario matrix for the remaining four valid MLP checkpoints,
then replay all retained evaluation fallback sources without changing CBF
parameters.

## AAMAS 2027 - post-isolation MLP seed 20260720 six-scenario matrix retained (2026-07-29)

At revision `2b232ac`, the final checkpoint
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260720/checkpoints/mappo_final.pt`
was evaluated with the same CUDA actor / CPU OSQP command form, frozen
configuration SHA-256, six canonical scenarios, distinct output cells, and 20
episodes per cell recorded above for seed `20260719`. The output root is
`outputs/core_3uav_post_actor_isolation_evaluations/`, with cell names
`core_3uav_post_actor_isolation_mlp_mappo_seed_20260720_<scenario>`.

Every cell retains a parseable `summary.json` and
`raw_results/seed_20260720.jsonl`. Direct parsing verified 20 records per cell,
fixed `(seed=20260720, scenario=<cell scenario>)` identity, all six expected
scenario names, and 120 total raw records. This makes a second matched
checkpoint matrix available; it does not establish performance, safety,
fallback rate, significance, a baseline comparison, or a five-seed conclusion.
Next: repeat the no-overwrite matrix for valid retry checkpoint
`20260721_telemetryretry1`, then seeds `20260722` and `20260723`, before an
all-source CBF fallback replay under unchanged numerical settings.

## AAMAS 2027 - post-isolation MLP seed 20260721 retry six-scenario matrix retained (2026-07-29)

At revision `5c81524`, the only valid seed-`20260721` checkpoint,
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260721_telemetryretry1/checkpoints/mappo_final.pt`,
received the matched CUDA actor / CPU OSQP six-scenario, 20-episode-per-cell
evaluation under the unchanged frozen configuration SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`. Its
separate cells are
`outputs/core_3uav_post_actor_isolation_evaluations/core_3uav_post_actor_isolation_mlp_mappo_seed_20260721_telemetryretry1_<scenario>`.

All six `raw_results/seed_20260721.jsonl` files and their `summary.json` files
were parsed: each JSONL has exactly 20 records with fixed numeric seed 20260721
and the corresponding canonical scenario, totaling 120 records. The aborted
pre-retry `...seed_20260721/` root was neither overwritten nor used. This
documents a third valid matched matrix only; it is not evidence of performance,
safety, fallback rate, significance, a baseline comparison, or a five-seed
result. Next: perform the no-overwrite six-scenario matrix for seeds `20260722`
and `20260723`, then replay retained training and evaluation fallback sources
under unchanged CBF settings.

## AAMAS 2027 - post-isolation MLP seed 20260722 six-scenario matrix retained (2026-07-29)

At revision `fd0b487`, final checkpoint
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260722/checkpoints/mappo_final.pt`
completed the same frozen-config SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`, CUDA
actor / CPU OSQP, six-canonical-scenario, 20-episode-per-cell protocol. The
six new cells are
`outputs/core_3uav_post_actor_isolation_evaluations/core_3uav_post_actor_isolation_mlp_mappo_seed_20260722_<scenario>`.

Every raw `raw_results/seed_20260722.jsonl` and its co-located `summary.json`
is parseable; direct validation found 20 records with fixed seed/scenario
identity per cell and all six expected scenarios (120 records). This adds a
fourth matched matrix only. No performance, safety, fallback-rate,
significance, baseline, or five-seed result is proven. Next: evaluate final
seed `20260723` with the exact no-overwrite protocol, then retain a combined
CBF source replay without changing any solver parameter.

## AAMAS 2027 - five post-isolation MLP checkpoint matrices complete (2026-07-29)

At revision `f757da3`, final checkpoint
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260723/checkpoints/mappo_final.pt`
completed the final matched CUDA actor / CPU OSQP six-scenario evaluation under
the unchanged frozen configuration SHA-256
`794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`. Its
six cells are
`outputs/core_3uav_post_actor_isolation_evaluations/core_3uav_post_actor_isolation_mlp_mappo_seed_20260723_<scenario>`.

Direct raw-artifact validation now covers valid tags `20260719`, `20260720`,
`20260721_telemetryretry1` (numeric seed `20260721`), `20260722`, and
`20260723`: all 30 expected cells have a parseable `summary.json` and a
parseable 20-record per-seed JSONL with fixed seed/scenario identity. This is
600 retained raw episodes (five checkpoints × six scenarios × 20 episodes).
The interrupted pre-retry `20260721` root and every historical pre-actor-repair
output remain excluded. The full matrix establishes evaluation completeness and
provenance only; no performance, safety, fallback-rate, significance, or
baseline result has yet been calculated or claimed. Next: replay every retained
training and evaluation CBF fallback context under unchanged solver settings,
retain source/replay JSONL, and explicitly record zero-event cells as such.

## AAMAS 2027 - post-isolation MLP CBF all-source replay retained (2026-07-29)

At revision `9b3fd56`, the no-overwrite command
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\replay_cbf_fallbacks.py
--output-jsonl outputs\cbf_diagnostics\post_actor_isolation_mlp_5seed_training_and_evaluation_replay_20260729.jsonl
--max-iterations 20000 --max-solve-time-seconds 0.1 --slack-penalty 100
--uncertainty-margin-gain 0.5 --max-uncertainty-margin 5.0 <35 retained
telemetry files>` completed on CPU OSQP. Inputs were five valid training
`live_training_telemetry.json` files plus all 30 valid evaluation
`runtime_telemetry.json` files; the preflight found 26 training and one
evaluation event. The output JSONL and companion summary retain all 35 source
paths, the unchanged numerical protocol, and 27 records.

Recorded statuses are 10 `solved inaccurate`, seven `maximum iterations
reached`, and 10 `solve_time_limit`. Of 26 successful replays, statuses are
three `solved inaccurate`, three `solved`, 11 `maximum iterations reached`, and
nine `solve_time_limit`, with 23 replay emergency fallbacks. One source event
is explicitly unreplayed, not dropped: it is the `solve_time_limit` at
`$.per_seed[0].episodes[2].emergency_events[0]` in
`...seed_20260719_ood_communication_obstacle/runtime_telemetry.json`; replay
reports `ValueError: Recorded dynamic-obstacle centers cannot be reconstructed
from the event step`. Preserve this raw source/replay evidence and do not
change slack, iteration limit, tolerance, solve-time, or uncertainty-margin
settings. This is diagnostic/provenance evidence, not a performance, safety,
fallback-rate, significance, or baseline claim. Next: begin fresh matching
GraphMAPPO training only after a zero-process/new-root preflight; retain this
unreplayable OOD context as an open telemetry-fidelity issue.

## AAMAS 2027 - raw GraphMAPPO seed 20260719 launcher-aborted attempt retained (2026-07-29)

The first fresh raw-graph attempt used revision `a908886`, CUDA, seed `20260719`,
three UAVs, `configs/rl/dynamic_graph_baseline.yaml` SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`, graph
mode `mappo`, and `--total-steps 100000` in
`outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260719/`.
The PowerShell wrapper used `ErrorActionPreference=Stop`; a normal CBF fallback
diagnostic emitted to native stderr (`solved inaccurate`) was elevated to a
terminating `NativeCommandError`. The child is no longer active and the root
has only 10,848 transitions, three interval checkpoints (3,072/6,144/9,216),
raw telemetry, and TensorBoard; no final checkpoint or summary exists.

`ABORTED.json` marks this complete root as preserved and excluded from every
training, evaluation, CBF replay, aggregation, and paper claim. This is a
launcher/telemetry-stream handling failure, not a CBF parameter result; no CBF
value changed. Next: retry the same numeric seed in a unique
`launcherretry1` root with process-level stdout/stderr redirection that keeps
native diagnostics out of the invoking PowerShell error stream, then audit only
a natural 100k completion.

## AAMAS 2027 - post-isolation raw GraphMAPPO seed 20260719 retry retained (2026-07-29)

The only valid raw-graph seed-`20260719` artifact is the unique retry root
`outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260719_launcherretry1/`.
At launch revision `461951c`, it used
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
--config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed 20260719
--num-uavs 3 --graph-mode mappo --total-steps 100000 --output-dir <root>`;
the frozen config SHA-256 is
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
Process-level stdout/stderr redirection avoided the first launcher's erroneous
stderr-as-fatal handling. It naturally completed on CUDA at 100,032 transitions
(the expected 32-step rollout boundary beyond the requested 100k) and 1,042
updates, with `summary.json`, final `graph_mappo_final.pt`, TensorBoard, raw
telemetry, and separate launcher logs retained.

The raw training telemetry has two `solved inaccurate` emergency fallbacks over
33,344 decisions; the initial and final in-script evaluations each retain zero
fallbacks over 160 decisions. Its all-source per-seed replay is
`outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260719_launcherretry1_replay_20260729.jsonl`
plus companion summary. It retains two source events with zero replay errors
under unchanged 20,000-iteration, 0.1-second, penalty-100,
uncertainty-gain-0.5/cap-5 CPU-OSQP settings: one replay is `solved inaccurate`
and falls back, one is `solved`. The preceding non-retry raw-graph root remains
invalid/excluded via its `ABORTED.json`. This establishes one valid independent
raw-graph training/replay artifact only, not a performance, safety,
fallback-rate, significance, baseline, or five-seed conclusion. Next: after a
zero-process/absent-path preflight, serially train raw-graph seed `20260720` in
a new root before any raw-graph checkpoint evaluation.

## AAMAS 2027 - post-isolation raw GraphMAPPO seed 20260720 retained (2026-07-29)

At revision `cf68ed3`, the serial CUDA command for seed `20260720` reused the
unchanged `dynamic_graph_baseline.yaml` SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`, three
UAVs, raw graph mode `mappo`, and 100k requested steps in
`outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260720/`.
Process-level stdout/stderr redirection wrote separate preserved launcher logs.
It naturally completed at the expected rollout-boundary 100,032 transitions /
1,042 updates with final checkpoint, summary, TensorBoard, and raw telemetry.

Training telemetry retains one `maximum iterations reached` CBF emergency
fallback in 33,344 decisions; initial and final in-script evaluations each have
zero fallbacks in 160 decisions. The unique per-seed replay at
`outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260720_replay_20260729.jsonl`
plus summary retains the source with zero replay errors under unchanged
20,000/0.1s/100/0.5/5.0 CPU-OSQP values; the replay is `solved` without
fallback. This is the second valid raw-graph training/replay artifact only, not
a performance, safety, fallback-rate, significance, baseline, or five-seed
conclusion. Next: new-root serial CUDA training for raw-graph seed `20260721`
before evaluation.

## AAMAS 2027 - post-isolation raw GraphMAPPO seed 20260721 retained (2026-07-29)

At revision `d07b637`, raw-graph mode `mappo` seed `20260721` used the unchanged
three-UAV dynamic-graph configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D` with
CUDA and process-level launcher output redirection. The unique root
`outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260721/`
naturally reached 100,032 transitions / 1,042 updates; final checkpoint,
summary, TensorBoard, raw telemetry, and stdout/stderr logs are retained.

Training retains one `solved inaccurate` CBF fallback in 33,344 decisions;
initial and final in-script evaluations are 0/160 and 0/160. Per-seed replay
at `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260721_replay_20260729.jsonl`
has one source, zero replay errors, and replays `solved` without fallback under
the unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP protocol. This is a third valid
raw-graph artifact only, not a performance, safety, fallback-rate,
significance, baseline, or five-seed conclusion. Next: new-root serial CUDA
training for raw-graph seed `20260722` before evaluation.

## AAMAS 2027 - post-isolation raw GraphMAPPO seed 20260722 retained; user-requested stop (2026-07-30)

At revision `d827ff3`, the serial CUDA raw-graph `mappo` command for seed
`20260722` used the unchanged three-UAV
`configs/rl/dynamic_graph_baseline.yaml` SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`, requested
100k steps, and process-level stdout/stderr redirection. The distinct root
`outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260722/`
naturally completed at 100,032 transitions / 1,042 updates. Final checkpoint,
summary, TensorBoard, raw telemetry, and both launcher logs are retained and
parsed; the actor device is CUDA and graph mode is `mappo`.

Training telemetry has zero CBF emergency fallbacks in 33,344 decisions;
initial and final in-script evaluations have zero fallbacks in 160 and 131
decisions, respectively. The unique zero-event replay at
`outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260722_replay_20260730.jsonl`
and companion summary retain the source and confirm `event_count=0`,
`replay_error_count=0` under unchanged 20,000-iteration, 0.1-second,
penalty-100, uncertainty-gain-0.5/cap-5 CPU-OSQP settings.

The user requested stopping after this current run. No seed `20260723`, no
GraphMAPPO evaluation, no other GraphMAPPO arm, no 5/8-UAV experiment, and no
new literature work was started. This leaves raw-graph seed 20260723 plus all
matched five-seed predictive/no-uncertainty and uncertainty-predictive graph
arms incomplete; no performance, safety, fallback-rate, significance, or
baseline conclusion exists. The earlier non-retry seed-20260719 root remains
invalid and excluded. Next when authorized: zero-process/new-root preflight,
then raw-graph seed `20260723`; preserve every existing artifact.

## AAMAS 2027 - authorized SSH push restored (2026-07-30)

After the user added the existing public key to GitHub, `ssh -T git@github.com`
authenticated as `youshenbupo`. The first push spelling from older notes,
`git push github.com codex/phase14-dynamic-world`, correctly failed because
`github.com` is not a configured Git remote. `git remote -v` verified the
authorized remote name is `origin` and targets
`git@github.com:youshenbupo/multi-uav-rl-path-planning.git`. The authorized
command `git push origin codex/phase14-dynamic-world` then succeeded, advancing
the remote branch from `55abb87` to `014c006`. `.gitignore` remained the sole
unstaged user-owned worktree change and was not staged or committed. This
establishes remote synchronization for the already committed history only; it
does not alter any experimental claim or permit external paper submission.

## Project documentation and safe cleanup audit (2026-07-30)

At the then-current worktree, the project guide was replaced with a
mainline-faithful `README.md`; `docs/aamas2027_method_and_readiness.md` records
the mathematical formulation, information boundary, worked examples, verified
artifacts, and paper-preparation gates; and `docs/next_agent_prompt.md`
provides a copyable continuation prompt. `docs/maintenance_cleanup_20260730.md`
records a cache-only cleanup audit and the exact manual cleanup commands.

The audit found only regenerable Python/tool caches (three top-level tool-cache
directories plus 12 `__pycache__` directories, approximately 48 MB). The
execution environment rejected the explicit deletion request before it ran.
Accordingly no files, experiment outputs, JSONL records, documentation,
configuration, data, or source files were deleted. This is a documentation and
maintenance outcome only; it adds no experimental evidence and preserves the
user-requested research stop point.

## AAMAS 2027 - post-isolation raw GraphMAPPO seed 20260723 retained (2026-07-31)

At revision `6bc5d1a`, the resumed serial CUDA command used
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py`
with `configs/rl/dynamic_graph_baseline.yaml` (SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`),
`--device cuda --seed 20260723 --num-uavs 3 --graph-mode mappo --total-steps
100000`, and the distinct output root
`outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260723/`.
`Start-Process` redirected native output to the separately retained
`outputs/core_3uav_post_actor_isolation_raw_graph_mappo_seed_20260723.stdout.log`
and `.stderr.log`, preventing ordinary CBF diagnostics from being promoted to
a PowerShell failure.

The job naturally completed with `total_transitions=100032` and
`update_count=1042`. `summary.json`, `live_training_telemetry.json`,
`checkpoints/graph_mappo_final.pt`, TensorBoard files, and both launcher logs
are retained and parsed. The summary identifies `graph_mode=mappo`, the raw
information model (no predicted knowledge or uncertainty features), three
UAVs, and CUDA. Training telemetry records 33,344 CBF decisions and zero
emergency events; initial and final in-script evaluation records also retain
zero emergency events. These seed-local diagnostics and short in-script
evaluations are provenance only, not performance, safety, fallback-rate,
significance, or baseline evidence.

The valid raw-graph training set is now five seeds: `20260719_launcherretry1`,
`20260720`, `20260721`, `20260722`, and `20260723`. The earlier non-retry
`20260719` root remains preserved through `ABORTED.json` and excluded. Next:
create the retained zero-event replay diagnostic for seed `20260723`, then run
the new-root, six-scenario, 20-episode CUDA-actor/CPU-OSQP checkpoint matrix
for all five valid raw-graph checkpoints before beginning any predictive arm.

## AAMAS 2027 - raw GraphMAPPO seed 20260723 zero-event CBF replay retained (2026-07-31)

The CPU OSQP replay command consumed only
`outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260723/live_training_telemetry.json`
and wrote the new, non-overwriting diagnostic
`outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260723_replay_20260731.jsonl`
with its companion `.summary.json`. It used the unchanged protocol: 20,000
maximum iterations, 0.1-second solve limit, slack penalty 100,
uncertainty-margin gain 0.5, and maximum uncertainty margin 5.0. The retained
summary identifies `event_count=0` and `replay_error_count=0`; the empty JSONL
is deliberate zero-event provenance, not a result deletion or a safety,
fallback-rate, performance, significance, or baseline claim. Next: preflight
new raw-graph evaluation paths, then execute the five-checkpoint six-scenario
matrix with 20 episodes per cell before an all-source raw-graph replay.

## AAMAS 2027 - raw GraphMAPPO evaluation launcher interruption retained (2026-07-31)

The first post-isolation raw-graph matrix launcher used the distinct root
`outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo/`
and the retained launcher/stdout/stderr paths named
`core_3uav_post_actor_isolation_raw_graph_mappo_evaluation_launcher_20260731`.
It serially began the five valid final checkpoints over the six canonical
20-episode CUDA-actor/CPU-OSQP scenarios. The launcher exited during
`core_3uav_post_actor_isolation_raw_graph_mappo_seed_20260720_ood_communication_obstacle`:
11 cells are complete, while that cell has eight raw records and no
`summary.json`; the remaining 18 cells do not exist. Its stderr is empty and
stdout ends immediately after announcing the partial cell, so the causal
mechanism is unknown.

`ABORTED.json` in that root records the exact state and exclusion. The entire
root—including its complete-looking JSONL files—remains raw forensic evidence
and is excluded from aggregation, replay inputs, and paper claims. No CBF or
evaluator parameter changed. Next: use a completely fresh root and launcher
paths for the full 30-cell rerun, then parse only that complete root before
all-source replay.

## AAMAS 2027 - post-isolation raw GraphMAPPO matrix and all-source replay complete (2026-07-31)

The valid rerun root
`outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo_rerun1/`
used five independently trained final checkpoints, the six canonical
scenarios, 20 episodes per cell, CUDA inference, and CPU OSQP. A per-cell
`Start-Process -Wait` launcher retained independent stdout/stderr and exit
codes. Direct parsing verified 30 directories, 30 parseable `summary.json`
files, 30 parseable seed JSONL files, exactly 600 records, fixed seed/scenario
identity, 30 zero exit codes, and no stderr output. The preceding non-rerun
root remains wholly excluded through its `ABORTED.json`.

The unchanged-protocol all-source replay is
`outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_5seed_training_and_evaluation_replay_20260731.jsonl`
plus companion summary. Its inputs are exactly five valid training telemetry
files and 30 valid rerun evaluation telemetry files. Under frozen
20,000-iteration, 0.1-second, penalty-100, uncertainty-gain-0.5/cap-5.0
CPU-OSQP values, it retains four source events, zero replay errors, three
solved/non-fallback replays, and one `solved inaccurate` replay that still uses
emergency fallback. This completes raw-graph provenance only; it does not prove
performance, safety, fallback rate, significance, or superiority. Next:
zero-process/new-root preflight, then serial five-seed `predictive_graph`
training under the same budget before any uncertainty-aware arm.

## AAMAS 2027 - first post-isolation predictive GraphMAPPO seed retained (2026-07-31)

At launch revision `3683c7b`, seed `20260719` used the frozen
`configs/rl/dynamic_graph_baseline.yaml` SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`,
three UAVs, `--graph-mode predictive_graph`, CUDA, 100k requested steps, and
the unique root
`outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260719/`.
It naturally completed at 100,032 transitions / 1,042 updates with final
checkpoint, summary, telemetry, TensorBoard and separate stdout/stderr logs.
The summary confirms delivered-packet prediction is enabled and uncertainty
features are disabled.

Training retained two emergency events in 33,344 decisions (`solve_time_limit`
and `solved inaccurate`); initial/final short evaluations retained zero events.
The unchanged-protocol replay at
`outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260719_replay_20260731.jsonl`
has two records and zero replay errors: one replay solved without fallback and
one remained `solved inaccurate` with emergency fallback. No solver value
changed. This is one-seed provenance only, not performance, safety,
fallback-rate, significance or comparison evidence. Next: serially train seed
`20260720` in a new root under the identical protocol.

## AAMAS 2027 - second post-isolation predictive GraphMAPPO seed retained (2026-07-31)

At launch revision `474bdce`, seed `20260720` ran serially with
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
--config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed 20260720
--num-uavs 3 --graph-mode predictive_graph --total-steps 100000 --output-dir
outputs\core_3uav_post_actor_isolation\core_3uav_predictive_graph_seed_20260720`.
The frozen configuration SHA-256 is
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
`Start-Process` retained native output separately at
`outputs/core_3uav_post_actor_isolation_predictive_graph_seed_20260720.stdout.log`
and `.stderr.log`; the latter is empty.

The job naturally completed at 100,032 transitions / 1,042 updates. Its
`summary.json`, `live_training_telemetry.json`, TensorBoard output and final
`checkpoints/graph_mappo_final.pt` remain under
`outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260720/`.
The parsed summary identifies three UAVs, `graph_mode=predictive_graph`,
`uses_predicted_knowledge=true`, `uses_uncertainty=false`, and CUDA. Training
telemetry retains zero CBF emergency events in 33,344 decisions; initial and
final in-script evaluation telemetry retain zero events in 160 and 152
decisions.

The append-safe diagnostic replay
`outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260720_replay_20260731.jsonl`
and companion summary consume only this seed's completed training telemetry.
They use the unchanged CPU-OSQP protocol (20,000 maximum iterations,
0.1-second solve limit, slack penalty 100, uncertainty-margin gain 0.5, and
maximum margin 5.0) and record `event_count=0`,
`replay_error_count=0`. The intentionally empty JSONL is retained zero-event
provenance. These seed-local records and short evaluations prove neither
performance, safety, fallback rate, significance, nor a method comparison.
Next: commit and push this documentation without staging the user's
`.gitignore`, then zero-process/absent-path preflight and serial CUDA training
of predictive seed `20260721` under the identical protocol.

## AAMAS 2027 - third post-isolation predictive GraphMAPPO seed retained (2026-07-31)

At launch revision `b69fcf1`, seed `20260721` ran serially with
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
--config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed 20260721
--num-uavs 3 --graph-mode predictive_graph --total-steps 100000 --output-dir
outputs\core_3uav_post_actor_isolation\core_3uav_predictive_graph_seed_20260721`.
The frozen configuration SHA-256 remained
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
`Start-Process` retained its native stdout/stderr separately at
`outputs/core_3uav_post_actor_isolation_predictive_graph_seed_20260721.stdout.log`
and `.stderr.log`; stderr is empty.

The process naturally completed at 100,032 transitions / 1,042 updates. Its
`summary.json`, `live_training_telemetry.json`, TensorBoard data and final
`checkpoints/graph_mappo_final.pt` remain under
`outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260721/`.
The parsed identity is three UAVs, `graph_mode=predictive_graph`,
`uses_predicted_knowledge=true`, `uses_uncertainty=false`, and CUDA. Training
retains zero CBF emergency events in 33,344 decisions; initial/final in-script
evaluations retain zero events in 160/160 decisions.

The no-overwrite replay
`outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260721_replay_20260731.jsonl`
and companion summary consume only this seed's completed training telemetry.
They use unchanged 20,000 maximum iterations, 0.1-second solve limit, slack
penalty 100, uncertainty-margin gain 0.5, and maximum margin 5.0 on CPU OSQP,
recording `event_count=0` and `replay_error_count=0`. The deliberately empty
JSONL is retained zero-event provenance, not a safety, performance,
fallback-rate, significance, or comparison result. Next: commit and push this
documentation without staging `.gitignore`, then perform a
zero-process/absent-path preflight and launch predictive seed `20260722`
serially under the identical protocol.

## AAMAS 2027 - first predictive GraphMAPPO seed 20260722 attempt interrupted and excluded (2026-08-01)

At revision `5d53757`, the first post-isolation seed-`20260722` attempt used
the frozen configuration SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D` and
the exact serial CUDA command
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
--config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed 20260722
--num-uavs 3 --graph-mode predictive_graph --total-steps 100000 --output-dir
outputs\core_3uav_post_actor_isolation\core_3uav_predictive_graph_seed_20260722`.
The distinct launcher logs are
`outputs/core_3uav_post_actor_isolation_predictive_graph_seed_20260722.stdout.log`
and `.stderr.log`.

The process exited before the requested budget at 84,672 transitions. The
last complete interval checkpoint is `checkpoints/graph_mappo_step_82944.pt`;
live telemetry, earlier checkpoints and TensorBoard files remain, but
`summary.json` and `checkpoints/graph_mappo_final.pt` do not exist. Both
launcher logs are empty. The causal mechanism is unknown: the immediately
preceding process-only monitor returned Windows exit code `1073807364`
(`0x40010004`), but this temporal adjacency is not treated as causal proof.
The retained telemetry SHA-256 is
`E21022A507B66F03F20AAEA0DEAC76EFA6DE042C2F50933C5778916286AB4E08`; it
contains zero emergency events over 28,224 training, 160 initial-evaluation,
and 77 latest-interval CBF decisions.

`ABORTED.json` now marks the entire root invalid. Its partial checkpoints,
telemetry, logs and TensorBoard data are preserved but excluded from training
eligibility, evaluation, CBF replay, aggregation, statistics and manuscript
claims. No training, CBF or safety parameter changed. Next: commit and push
this documentation without staging `.gitignore`, then use a zero-process and
absent-path preflight before retrying the same numeric seed in the unique
`core_3uav_predictive_graph_seed_20260722_launcherretry1` root with distinct
retry logs.

## AAMAS 2027 - predictive GraphMAPPO seed 20260722 valid retry retained (2026-08-01)

The first intended retry identity, `launcherretry1`, failed before child
creation because the active PowerShell launcher environment exposed both
case-distinct `Path` and `PATH` keys. `Start-Process` raised
`ArgumentException: Item has already been added. Key in dictionary: 'Path';
Key being added: 'PATH'`. No Python process or training root was created, but
its two zero-byte redirected logs and sidecar
`outputs/core_3uav_post_actor_isolation_predictive_graph_seed_20260722_launcherretry1.ABORTED.json`
are retained and excluded. `Get-ChildItem Env:` reproduced the duplicate-key
failure, and `-UseNewEnvironment` did not bypass it.

Root-cause testing then operated only inside the short-lived launcher process:
it captured both PATH values, merged their components with case-insensitive
de-duplication, removed both keys, and set one `Path`. Under that single change,
`Start-Process cmd.exe` exited 0. The same launcher-local normalization ran
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\check_environment.py`
with exit 0 and empty stderr, verifying Python 3.11.15, PyTorch
`2.13.0+cu130`, CUDA available, and the NVIDIA GeForce RTX 5060 Laptop GPU.
The retained probe logs are
`outputs/post_actor_isolation_start_process_env_probe_20260801.stdout.log` and
`.stderr.log`. No persistent system/user environment, repository file,
training protocol, or CBF parameter changed.

After a zero-process and absent-path preflight, the valid second retry launched
at revision `084f776` with the same frozen config SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`:
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
--config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed 20260722
--num-uavs 3 --graph-mode predictive_graph --total-steps 100000 --output-dir
outputs\core_3uav_post_actor_isolation\core_3uav_predictive_graph_seed_20260722_launcherretry2`.
Its distinct stdout/stderr use the matching top-level `launcherretry2` paths.

The process naturally completed at 100,032 transitions / 1,042 updates. Final
checkpoint, summary, live telemetry and TensorBoard are retained in the retry2
root; summary identity is three UAVs, `predictive_graph`, predicted delivered
knowledge enabled, uncertainty disabled, and CUDA. Training telemetry has zero
emergency events in 33,344 CBF decisions; initial/final short evaluations have
zero in 160/126. Retry2 stderr is empty. The append-safe zero-event replay
`outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260722_launcherretry2_replay_20260801.jsonl`
and companion summary use unchanged 20,000 iterations, 0.1 seconds, penalty
100, uncertainty gain 0.5 and cap 5.0 on CPU OSQP, recording
`event_count=0`, `replay_error_count=0`.

Only `launcherretry2` is eligible for the matched predictive arm. The original
84,672-transition root and retry1 prelaunch attempt remain preserved/excluded.
This is completion and diagnostic provenance only, not performance, safety,
fallback-rate, significance, or comparison evidence. Next: commit/push these
documents without staging `.gitignore`, then serially train final predictive
seed `20260723` after zero-process/absent-path preflight.

## AAMAS 2027 - fifth post-isolation predictive GraphMAPPO seed retained (2026-08-01)

At launch revision `249ffd5`, a zero-process and absent-path preflight preceded
the final serial three-UAV `predictive_graph` seed. The frozen configuration
SHA-256 remained
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`, and the
exact CUDA command was
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
--config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed 20260723
--num-uavs 3 --graph-mode predictive_graph --total-steps 100000 --output-dir
outputs\core_3uav_post_actor_isolation\core_3uav_predictive_graph_seed_20260723`.
Only the transient launcher process normalized the duplicate-case PATH entries;
no persistent environment, repository, training, evaluation, or CBF setting
changed. Distinct stdout/stderr logs use the matching top-level seed-20260723
paths.

The process naturally completed at 100,032 transitions / 1,042 updates. Its
summary confirms three UAVs, `predictive_graph`, delivered-packet prediction
enabled, uncertainty disabled, and CUDA. Final checkpoint, summary, live
telemetry, TensorBoard data, and launcher logs are retained under
`outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260723/`.
The final checkpoint SHA-256 is
`666898D014CADA85ECF118116CD9E7E6634B5820AEC3452626A06CC37467D0A7`; live
telemetry SHA-256 is
`BC79C49915BED01AA5517E40E7EBEC1F84033BC748797296D95CF7752ABF9100`.

Training retained one `maximum iterations reached` emergency fallback in
33,344 CBF decisions; initial/final short evaluations retained zero in 160/103
decisions. The sole stderr line is the matching fallback warning. The
append-safe CPU-OSQP replay at
`outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260723_replay_20260801.jsonl`
plus companion summary retains the one source event with zero replay errors
under unchanged 20,000 iterations, 0.1 seconds, slack penalty 100, uncertainty
gain 0.5, and cap 5.0. The replay status is `solved` without emergency fallback;
this diagnostic variability does not authorize solver tuning or a safety,
performance, fallback-rate, significance, or comparison claim.

The eligible predictive roots are now seeds `20260719`, `20260720`, `20260721`,
`20260722_launcherretry2`, and `20260723`; the first seed-20260722 root and its
retry1 prelaunch attempt remain excluded. Five-seed training provenance is
complete, but the arm is not evidence-complete until all five final checkpoints
finish the six canonical scenarios at 20 episodes per cell and an all-source
training/evaluation replay is retained. Next: commit/push these documents
without staging `.gitignore`, then preflight an entirely new predictive
evaluation root and run the 30 cells serially before starting
`uncertainty_predictive_graph`.

## AAMAS 2027 - post-isolation predictive GraphMAPPO evidence matrix retained (2026-08-01)

At revision `49e45f2`, a zero-process/new-root preflight verified all five
eligible predictive final checkpoints and the unchanged dynamic-graph config
SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
The retained launcher
`outputs/core_3uav_post_actor_isolation_predictive_graph_evaluation_launcher_20260801.ps1`
(SHA-256
`EAE8D16400E410CFEA593B3855E4C1419E32CCEE8589BE3EDA01B67F73E943C5`)
normalized duplicate-case PATH entries only inside its transient process, then
used per-cell `Start-Process -Wait` execution. It evaluated the five eligible
final checkpoints (`20260719`, `20260720`, `20260721`,
`20260722_launcherretry2`, and `20260723`) across `nominal`, `delay_only`,
`loss_only`, `dynamic_only`, `combined`, and
`ood_communication_obstacle`, with 20 episodes per cell, CUDA network
inference, and CPU OSQP. The distinct, non-overwriting root is
`outputs/core_3uav_post_actor_isolation_evaluations_20260801_predictive_graph/`;
launcher stdout/stderr and every cell's stdout/stderr are retained.

The launcher naturally exited 0. Direct content parsing verified exactly 30
directories, 30 parseable summaries, 30 parseable JSONL files and 600 episode
records. All records have the expected numeric seed, scenario, eligible
checkpoint, `graph_mappo_checkpoint` controller and completed checkpoint step
100,032. All 30 environment records identify three UAVs, CUDA,
`own_truth_and_delivered_packets_only`, and `cpu_osqp_when_enabled`. There are
30 zero exit-code lines, 30 empty cell stderr files, and empty launcher stderr.
Across the 600 episodes, runtime telemetry retains 11,223 CBF decisions and
zero emergency fallback events.

The append-safe all-source replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_5seed_training_and_evaluation_replay_20260801.jsonl`
plus companion summary. Its inputs are exactly five eligible training
telemetry files and the 30 valid evaluation runtime telemetry files. Under the
unchanged 20,000-iteration, 0.1-second, penalty-100, uncertainty-gain-0.5 and
cap-5.0 CPU-OSQP protocol, it retains three source events and zero replay
errors. Recorded statuses are one each of `solve_time_limit`, `solved
inaccurate`, and `maximum iterations reached`; replay statuses are two
`solved` and one `solved inaccurate`, with the latter retaining emergency
fallback. JSONL SHA-256 is
`58832392FAA584823FFE588727501A49FC7CD01020B036AD78A634D1A691F816` and summary
SHA-256 is
`F0B0F8D61B6CDA4C254AB54185CC39D7B87496FBC5366545CA497E2BFC64DEF7`.

This closes the post-isolation three-UAV predictive-without-uncertainty arm's
training, evaluation, and CBF-provenance gate. It does not itself establish
performance, safety, fallback rate, significance, robustness, or superiority.
Next: commit/push the three documents without staging `.gitignore`, then
perform a zero-process/new-root preflight and begin matched five-seed
`uncertainty_predictive_graph` training serially, starting with seed
`20260719`; no 5/8-UAV or cross-method claim precedes that arm's complete
matrix and replay.

## AAMAS 2027 - first uncertainty-aware seed attempt interrupted and excluded (2026-08-01)

After the predictive-arm gate and documentation push, a zero-process and
absent-path preflight at revision `6d6ba24` launched the first matched
three-UAV `uncertainty_predictive_graph` attempt with frozen config SHA-256
`1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`:
`D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
--config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed 20260719
--num-uavs 3 --graph-mode uncertainty_predictive_graph --total-steps 100000
--output-dir
outputs\core_3uav_post_actor_isolation\core_3uav_uncertainty_predictive_graph_seed_20260719`.
The transient launcher-only PATH normalization was unchanged. Separate
top-level stdout/stderr logs were created and retained.

The process did not complete naturally: it exited with signed code
`-1073741510` (Windows `0xC000013A`) at 78,624 transitions. The last complete
checkpoint is `checkpoints/graph_mappo_step_76800.pt`; no final checkpoint or
`summary.json` exists. Both launcher logs are empty, so the causal mechanism is
unknown. The interrupt-style exit code is retained as evidence but does not
identify what issued the interruption. Read-only monitoring during the run is
not asserted as causal.

The preserved live telemetry SHA-256 is
`89CD48936C924B6AE17F15E33227B7F6A28EC7B34C68C1963A018C8A32B95B76`.
It contains zero emergency events over 26,208 training CBF decisions, 160
initial-evaluation decisions, and 71 last-interval decisions. These partial
zero-event observations are ineligible and are not replayed or interpreted as
a safety/fallback result. The retained parseable
`outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260719/ABORTED.json`
(SHA-256
`585A2CB501693A19CBA0589FB548D0D6EFAFA2212824A7152E747FB50341B71E`)
excludes the whole root from training eligibility, evaluation, CBF replay,
aggregation, statistics, and manuscript claims. No experiment or CBF setting
changed. Next: commit/push these documents without staging `.gitignore`, then
retry numeric seed `20260719` only in a distinct `launcherretry1` root after a
zero-process/absent-path preflight; preserve this interrupted root forever.

## AAMAS 2027 - uncertainty seed retry1 interrupted; control-event diagnosis retained (2026-08-01)

After the invalid first attempt was documented and pushed, a zero-process and
absent-path preflight at revision `c3b59fa` launched numeric seed `20260719`
from scratch under identical config/training/CBF values in the distinct root
`outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260719_launcherretry1/`.
It again received signed exit code `-1073741510` (`0xC000013A`), this time at
13,344 transitions. The last complete checkpoint is step 12,288; no final
checkpoint or summary exists, and stdout/stderr are empty. Live telemetry
SHA-256
`A70594390834830E0D3DA7ABD38CEA66B172728A85A2340644486113E054C384`
retains zero emergency events over 4,448 training, 160 initial-evaluation and
80 last-interval CBF decisions. These incomplete zero-event counts are not
replayed or used as results.

The parseable retry1 `ABORTED.json` SHA-256 is
`03A80527E4B3B7E82F7202DA668218B6097559BAF3239B77E21AEA7288CC6D8D`; it
excludes the entire retry1 root from training eligibility, evaluation, replay,
aggregation, statistics, and manuscript claims. Both invalid roots remain
preserved, and neither will be overwritten, resumed, or deleted.

Systematic root-cause investigation found the following. The installed Windows
SDK identifies `0xC000013A` as `STATUS_CONTROL_C_EXIT`; repository Python has no
`SIGINT`, control-C, `KeyboardInterrupt`, or self-signal path; and Application/
System event-log inspection found no Python, CUDA, NVIDIA, application-error,
or driver warning in the failure window. A 90-second PowerShell child process
survived two parallel read-only shell calls and exited 0. A stronger retained
probe using the same Conda Python, CUDA initialization on the RTX 5060,
`Start-Process -Wait`, separate logs, and a 240-second lifetime also survived
three parallel read-only shell calls and naturally exited 0 after 246 seconds,
with empty stderr. The probe artifacts are
`outputs/diagnostics/cuda_process_lifetime_probe_20260801.py` and matching
stdout/stderr; script SHA-256 is
`CAB32F367DC4C74D44799C14F20CDB380C8D72AF7B13E57D4E2D3120BE083A1E`.

These checks rule out a deterministic training-script self-exit, generic
`Start-Process`, Python/CUDA initialization, arbitrary parallel shell access,
and a simple 180-second lifetime cap. They do not identify the external sender
of the control event. The remaining narrow environmental hypothesis is an
interaction between sustained real training load and parallel shell monitoring
inside the shared tool process tree. The single minimal next test is a fresh
`launcherretry2` full run with no parallel shell commands while it is active:
only wait on its existing execution cell. If that also returns `0xC000013A`,
stop further retries and treat the launch architecture as blocked pending user
direction; do not change model, training, or CBF settings.

## AAMAS 2027 - first valid uncertainty-aware seed retained after monitored-launch diagnosis (2026-08-01)

At revision `f21566e`, a zero-process/absent-path preflight verified both prior
invalid roots remained preserved, then launched numeric seed `20260719` from
scratch in the unique
`outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260719_launcherretry2/`
root. The exact frozen command differed from the two invalid attempts only by
the output identity. During the active 1,152-second run, no parallel shell
process was started; the existing execution cell was observed only through its
wait operation. The process naturally exited 0 rather than returning the prior
`STATUS_CONTROL_C_EXIT`.

The completed summary and live telemetry verify 100,032 transitions / 1,042
updates, three UAVs, `graph_mode=uncertainty_predictive_graph`, delivered-
packet prediction enabled, uncertainty enabled, and CUDA. Final checkpoint,
summary, live telemetry, TensorBoard and distinct retry2 stdout/stderr are
retained; stderr is empty. Final checkpoint SHA-256 is
`F0EDD4AADAC090E6FA0B740770EF4381D05F053756DB9BF4D898A2E957D9FC61` and live
telemetry SHA-256 is
`CE5F616B11B356B046FC63C0603324498D1E05FEFF1C1EBF91903E73E3CB1953`.

Training telemetry retains zero emergency events over 33,344 CBF decisions;
initial and final in-script evaluations retain zero over 160 and 96 decisions.
The no-overwrite zero-event replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260719_launcherretry2_replay_20260801.jsonl`
plus companion summary uses unchanged 20,000 iterations, 0.1 seconds, slack
penalty 100, uncertainty gain 0.5 and cap 5.0 on CPU OSQP, recording
`event_count=0` and `replay_error_count=0`. Its deliberately empty JSONL
SHA-256 is the empty-file hash
`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`; summary
SHA-256 is
`F83AC9DAEEC599FF065A9B81A1A605931EA843DAE325A387009D2F0AB20BDA57`.

Only `launcherretry2` is eligible for this numeric seed. The non-retry and
retry1 roots remain invalid and excluded. The successful one-variable run
supports an operational constraint—do not launch parallel shell commands while
a long training cell is active, and perform artifact/process audits only after
that cell exits—but does not prove the exact external control-event source.
This is one-seed completion/provenance only, not performance, safety,
fallback-rate, significance, robustness, or comparison evidence. Next:
commit/push these documents without staging `.gitignore`, then preflight and
launch uncertainty seed `20260720` serially in a new root under the same
no-parallel-shell waiting discipline and frozen protocol.

## AAMAS 2027 - second valid post-isolation uncertainty-aware seed retained (2026-08-01)

At revision `9f6c5f1`, a zero-process/absent-path preflight launched independent
seed `20260720` under the same frozen three-UAV CUDA command, changing only the
numeric seed and the unique root
`outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260720/`.
The active run was observed only through its existing execution-cell wait
handle; no parallel shell command ran. It naturally exited 0 after about 595
seconds and reached 100,032 transitions / 1,042 updates.

The summary verifies `uncertainty_predictive_graph`, delivered-packet
prediction and uncertainty both enabled, three UAVs, and CUDA. Final checkpoint,
summary, live telemetry, TensorBoard and distinct stdout/stderr are retained;
stderr is empty. Final checkpoint SHA-256 is
`F07979DDA3540A399D80ABD0604A30DE512EFFC934E510643EBF1C5FD7899D2E` and live
telemetry SHA-256 is
`CDCAD0AC73962BF9B94E3C3BC5F24E1E052CA61A51C2E0FF95F6DF3AB4F1C432`.
Training, initial and final evaluations retain zero CBF emergency events over
33,344, 160 and 131 decisions respectively.

The append-safe replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260720_replay_20260801.jsonl`
plus companion summary uses unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP values
and records `event_count=0`, `replay_error_count=0`; the empty JSONL is
deliberate zero-event provenance. This is the second valid seed artifact only,
not a performance, safety, fallback-rate, significance, robustness, or
comparison result. Next: commit/push these documents without `.gitignore`,
then preflight and serially launch seed `20260721` with the same frozen command
and wait-only active-run discipline.

## AAMAS 2027 - third valid post-isolation uncertainty-aware seed retained (2026-08-01)

At revision `30e164d`, seed `20260721` used the frozen three-UAV CUDA
`uncertainty_predictive_graph` command in the unique root
`outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260721/`.
Only the active execution-cell wait handle was used during the run. It naturally
exited 0 after about 1,422 seconds and completed 100,032 transitions / 1,042
updates. Summary identity confirms delivered-packet prediction and uncertainty
enabled, three UAVs and CUDA. Final checkpoint, summary, live telemetry,
TensorBoard and separate stdout/stderr are retained.

Training records one `maximum iterations reached` emergency fallback over
33,344 CBF decisions; initial/final evaluations record zero over 160/160.
Stderr contains only the matching fallback warning. Final checkpoint SHA-256 is
`092AC439646CFBBC12270FA5A12D97C85A92E5BD43AFC4BF07BD16DAD23BE909` and live
telemetry SHA-256 is
`F62A2229D171775FF531618EA874B9DC4E4694B34CAECB86EABE101389B601B0`.
The unique replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260721_replay_20260801.jsonl`
plus summary retains the one source event with zero replay errors under
unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP values. It replays `solved` without
fallback; this does not authorize solver tuning or any safety/performance
claim. This is third-seed provenance only. Next: commit/push without staging
`.gitignore`, then preflight and launch seed `20260722` serially with the same
wait-only active-run discipline.

## AAMAS 2027 - fourth valid post-isolation uncertainty-aware seed retained (2026-08-01)

At revision `67407cc`, independent seed `20260722` ran in the fresh root
`outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260722/`
under the frozen three-UAV CUDA uncertainty-aware command and wait-only active
monitoring. It naturally exited 0 after about 1,401 seconds and completed
100,032 transitions / 1,042 updates. Identity confirms
`uncertainty_predictive_graph`, delivered-packet prediction true, uncertainty
true, three UAVs and CUDA. Final checkpoint, summary, telemetry, TensorBoard and
separate logs are retained.

Training retains one `solved inaccurate` emergency fallback over 33,344 CBF
decisions; initial/final evaluations are zero-event over 160/147. Stderr is the
matching one-line warning. Final checkpoint SHA-256 is
`CCEDE56C3F88B4CF6AAB1E4BC00E6162FE6E8E36A43CB6C329097469585CC00C`; telemetry
SHA-256 is
`6FF5F852845B25DD70C6D457AC5D076B9EBB4268E9005390B6A80A418BB954B1`.
The append-safe replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260722_replay_20260801.jsonl`
plus summary retains the source with zero errors under unchanged values; replay
remains `solved inaccurate` with emergency fallback. Preserve this failure mode
without tuning. This is fourth-seed provenance only. Next: commit/push without
`.gitignore`, then run final seed `20260723` serially with the same protocol and
wait-only discipline before any evaluation.

## AAMAS 2027 - post-isolation uncertainty-aware five-seed training complete (2026-08-01)

At revision `0f5a971`, final independent seed `20260723` ran in the fresh root
`outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260723/`
under the frozen three-UAV CUDA command and wait-only active-run discipline. It
naturally exited 0 after about 1,335 seconds and completed 100,032 transitions /
1,042 updates. Summary identity confirms prediction and uncertainty enabled,
three UAVs and CUDA. Final checkpoint, summary, telemetry, TensorBoard and
separate logs are retained.

Training records one `solved inaccurate` emergency fallback over 33,344 CBF
decisions; initial/final evaluations are zero over 160/160. Stderr is only the
matching warning. Final checkpoint SHA-256 is
`BE80BFAC6F621D46455C01985A1024035351CB6674E7794DDAD582BB031E3550`; telemetry
SHA-256 is
`2494F404B5D7C4E96C138BE39D1FAD2D80A7A2ED8909BCCA9C2A2E05FAEBA267`.
The unique replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260723_replay_20260801.jsonl`
plus summary retains the event with zero errors under unchanged values and
replays `solved` without fallback.

The eligible uncertainty-aware roots are now `20260719_launcherretry2`,
`20260720`, `20260721`, `20260722`, and `20260723`; the non-retry and retry1
seed-20260719 roots remain preserved/excluded. Five-seed training provenance is
complete, but the arm is not evidence-complete until all five final checkpoints
finish the six canonical 20-episode scenarios and the all-source replay is
retained. No performance, safety, fallback-rate, significance, robustness or
comparison claim follows yet. Next: commit/push documents without `.gitignore`,
then preflight a wholly new evaluation root and run all 30 cells serially with
CUDA inference, CPU OSQP and per-cell logs before all-source replay.

## AAMAS 2027 - post-isolation uncertainty-aware evidence matrix retained (2026-08-01)

At revision `c9bff6e`, a zero-process/new-root preflight verified the five
eligible final checkpoints and unchanged dynamic-graph configuration. The
retained launcher
`outputs/core_3uav_post_actor_isolation_uncertainty_predictive_graph_evaluation_launcher_20260801.ps1`
(SHA-256
`2941E7A898C7B4F877C09E5716F25642CFDC16B38AD79C2975FB93B49EA8DD0C`)
used the transient single-PATH normalization, per-cell `Start-Process -Wait`,
and no parallel shell monitoring. It evaluated the eligible checkpoints
`20260719_launcherretry2`, `20260720`, `20260721`, `20260722`, and `20260723`
over `nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, and
`ood_communication_obstacle`, with 20 episodes per cell, CUDA network
inference and CPU OSQP. The unique root is
`outputs/core_3uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`.

The launcher naturally exited 0 after about 333 seconds. Direct content audit
found zero errors: exactly 30 directories, 30 parseable summaries, 30 parseable
JSONL files, and 600 episode records with correct numeric seed, scenario,
eligible checkpoint, controller and 100,032-step identity. All environment
records identify three UAVs, CUDA, actor own truth plus delivered packets only,
and CPU OSQP. All 30 cell exit codes are zero; 30 cell stderr files and launcher
stderr are empty. Evaluation telemetry retains 10,855 CBF decisions and zero
emergency events.

The append-safe 35-input replay is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5seed_training_and_evaluation_replay_20260801.jsonl`
plus summary, using exactly five eligible training and 30 valid evaluation
telemetry files. Under unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP values it
retains three source events and zero replay errors. Recorded statuses are one
`maximum iterations reached` and two `solved inaccurate`; replay statuses are
two `solved` and one `solved inaccurate`, with one replay emergency fallback.
JSONL SHA-256 is
`3371F6BCF08237C0E1BE75F241EDC66092E178D209DA493C69A2200890DE5084`; summary
SHA-256 is
`8BE10BCD9EEAC4EE0FCCF694D40CDC63559CAAA8A34624CFCD0D1E2770FC0090`.

This closes the post-isolation three-UAV uncertainty-aware training,
evaluation, and CBF-provenance gate and therefore completes the prescribed
three-UAV four-arm artifact matrix. It does not by itself establish
performance, safety, fallback rate, significance, robustness, generalization,
or superiority. Next: commit/push these documents without `.gitignore`, then
read-only audit the existing post-isolation 5/8-UAV and independently trained
ablation artifacts/protocol before launching anything new; fill only genuinely
missing matched evidence in fresh roots, then proceed to statistical and
primary-literature gates.

## AAMAS 2027 - post-isolation 5/8-UAV scale and ablation gap audited (2026-08-01)

After closing the post-isolation three-UAV four-arm artifact matrix, a read-only
inventory checked the current scale/full-method and independent no-uncertainty
ablation roots against the actor-information repair boundary. Existing roots
`outputs/core_5uav/`, `outputs/core_5uav_evaluations/`,
`outputs/core_5uav_ablation/`, the retained 5-UAV ablation evaluation attempts,
`outputs/core_8uav/`, `outputs/core_8uav_evaluations/`,
`outputs/core_8uav_ablation/`, and
`outputs/core_8uav_ablation_evaluations_20260729/` all predate the repair. Per
the protocol-wide invalidation already recorded in `known_issues.md`, they and
their paired summaries/replays remain forensic-only and cannot enter task,
safety, scale, ablation, or manuscript aggregation.

No top-level 5/8-UAV post-actor-isolation root exists. Therefore the matched
scale/ablation evidence is genuinely missing rather than partially reusable.
The frozen full-method configs still match their documented hashes:
`configs/rl/dynamic_graph_5uav.yaml` =
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32` and
`configs/rl/dynamic_graph_8uav.yaml` =
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
The independently trained no-uncertainty configs also match:
5-UAV `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`
and 8-UAV `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`.

`docs/aamas2027_analysis_plan.md` now identifies the four completed
post-isolation 3-UAV evaluation roots as eligible raw inputs but intentionally
leaves the descriptive/inferential artifact empty until a separately
prespecified independent-seed analysis is run. Its 5/8-UAV rows correctly
remain `None` pending new data. Next: commit/push this audit without staging
`.gitignore`, then begin the 5-UAV full uncertainty-aware arm with seed
`20260719`, 100k CUDA transitions, frozen 5-UAV config, a fresh
`outputs/core_5uav_post_actor_isolation/` seed root, CPU OSQP, retained CBF
telemetry, and wait-only active-run monitoring. Complete its five-seed matrix
and replay before independently trained no-uncertainty 5-UAV ablations, then
repeat the matched protocol at 8 UAV.

## AAMAS 2027 - first post-isolation 5-UAV full-method seed retained (2026-08-01)

At revision `4944c4d`, a zero-process/absent-path preflight launched independent
seed `20260719` with frozen config `configs/rl/dynamic_graph_5uav.yaml` SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`, five
UAVs, CUDA, `uncertainty_predictive_graph`, and 100k requested steps in the new
root
`outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260719/`.
Only the execution-cell wait handle was used while active. It naturally exited
0 after about 328 seconds and reached the expected rollout boundary 100,160
transitions / 313 updates. Summary identity confirms prediction and uncertainty
enabled, five UAVs and CUDA. Final checkpoint, summary, telemetry, TensorBoard
and separate logs are retained.

Training records two `solved inaccurate` emergency fallbacks over 20,032 CBF
decisions; initial/final evaluations are zero over 160/160. Final checkpoint
SHA-256 is
`82C623A9EF653CDB5D425D61E883C2DA9EDDFA7BB75FE41A842D6AFA6C4E2CAC`; telemetry
SHA-256 is
`1676AC77F08C7A8AB5E60E76574A1512A6A19609C11034007A980D69EA28F1DB`.
The unique replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260719_replay_20260801.jsonl`
plus summary retains both sources with zero errors under unchanged values; one
replay remains `solved inaccurate` with emergency fallback and one is `solved`
without fallback. This is first-seed scale provenance only, not performance,
safety, fallback-rate or generalization evidence. Next: commit/push documents
without `.gitignore`, then preflight and run seed `20260720` serially under the
same 5-UAV full-method and wait-only protocol.

## AAMAS 2027 - second post-isolation 5-UAV full-method seed retained (2026-08-01)

At revision `4fc64aa`, independent seed `20260720` used the unchanged frozen
5-UAV full-method command and fresh
`outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260720/`
root. With wait-only monitoring it naturally exited 0 after about 599 seconds
at 100,160 transitions / 313 updates. Identity verifies prediction and
uncertainty enabled, five UAVs and CUDA. Final checkpoint SHA-256 is
`82366DBB9B4B866646791A4D0B026223FA253708EB387F03AF8D00B52AE5859C`; telemetry
SHA-256 is
`1E78610C498A69A501F29A46E30CD10B5EEBF3F0C1C00DC72AB7D4E316EF31B2`.

Training records one `solved inaccurate` fallback in 20,032 CBF decisions;
initial/final are zero over 160/160. Its unique replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260720_replay_20260801.jsonl`
has one source, zero errors and remains `solved inaccurate` with emergency
fallback under unchanged values. This is seed-local provenance only. Next:
commit/push without `.gitignore`, then preflight and run 5-UAV full seed
`20260721` serially under the same protocol.

## AAMAS 2027 - interval-evaluation telemetry audit repair and invalid 5-UAV seed 20260721 (2026-08-01)

The post-isolation 5-UAV full-method seed `20260721` process launched from
revision `3836319` and naturally exited 0 at 100,160 transitions / 313 updates
under frozen `configs/rl/dynamic_graph_5uav.yaml` SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
Its final checkpoint, summary, live telemetry, TensorBoard data and distinct
stdout/stderr remain in
`outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260721/`
and the matching top-level log paths. The checkpoint SHA-256 is
`7658AAD2E4FF2A5CC0034E0AACBD916866179827261D41867317796815108A8B` and
telemetry SHA-256 is
`2E8800E091EF030FC9F388F845C379C61D7B97DEFDB8F0D0624D64047F49428D`.

Despite natural completion, this root is not audit-eligible. Its stderr retains
five CBF fallback notices: one `maximum iterations reached` and four
`solve_time_limit`. Final telemetry retains the one training event in 20,032
training decisions, zero events in the 160-decision initial evaluation, and
only the final zero-event interval evaluation (80 decisions); the four earlier
interval-evaluation fallback contexts were overwritten by later assignments to
`last_interval_evaluation_cbf`. They therefore cannot be reconstructed or
replayed. `ABORTED.json` marks the entire root, including its final checkpoint,
ineligible for evaluation, replay, aggregation, statistics and manuscript use.
No CBF setting changed, and no incomplete one-event replay was created.

The defect was reproduced test-first: the new real GraphMAPPO smoke test failed
with `KeyError: 'interval_evaluations'`. The minimal backward-compatible repair
adds an explicit append operation to `write_live_training_telemetry` and makes
the graph runner retain every interval record as
`interval_evaluations[{total_transitions, evaluation, cbf}]`, while preserving
the existing `last_interval_evaluation` and `last_interval_evaluation_cbf`
fields. The exact test then passed. Broader graph/telemetry/checkpoint tests
passed 27; Ruff passed; and explicit-package mypy passed 103 source files.
The full suite recorded 159 passed and one unrelated legacy-audit failure: the
test expects 81 MATLAB files beneath the no-longer-present
`legacy_hgalo/HGALO_恢复源码` path, while the current workspace contains
`legacy_hgalo/HGALO_code`, so the tested path yields zero. That legacy issue was
not modified. The suite emitted 190 existing OSQP deprecation warnings.

This repair establishes prospective audit fidelity only; it recovers none of
the four missing seed-20260721 contexts and proves no performance, safety,
fallback-rate, scale or method result. Next: commit/push the code, test and
three documents without staging `.gitignore`; then zero-process/absent-path
preflight and rerun numeric seed `20260721` from scratch in the unique
`core_5uav_uncertainty_predictive_graph_seed_20260721_intervaltelemetryretry1`
root. Use only the active execution-cell wait handle. Eligibility requires
natural completion and a one-to-one audit between all retained interval/training
events and stderr notices before replay.

## AAMAS 2027 - valid post-isolation 5-UAV full seed 20260721 telemetry retry retained (2026-08-01)

After commit `a645024` was pushed, a zero-process/CUDA/config-hash and
absent-path preflight launched numeric seed `20260721` from scratch in the
unique root
`outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260721_intervaltelemetryretry1/`.
The command retained frozen `configs/rl/dynamic_graph_5uav.yaml` SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`,
five UAVs, `uncertainty_predictive_graph`, 100k requested steps, CUDA neural
execution and CPU OSQP. While active, only the original execution-cell wait
handle was used. The process naturally exited 0 after about 316 seconds at
100,160 transitions / 313 updates.

Final checkpoint, summary, telemetry, TensorBoard and distinct logs are
retained. Summary identity confirms predicted delivered-packet knowledge and
uncertainty enabled, five UAVs and CUDA. Checkpoint SHA-256 is
`7658AAD2E4FF2A5CC0034E0AACBD916866179827261D41867317796815108A8B` and
live telemetry SHA-256 is
`361B38F493BD1A220D2B2E24B936BB8A69B51068A99A1F7B06795F3F6C519159`.
The telemetry has six append-only interval records at transitions 15,360,
30,720, 46,080, 61,440, 76,800 and 92,160. Each interval retains its CBF
`emergency_events` array; all six are zero-event over 80 decisions each.
Initial/final evaluations are zero-event over 160/160 decisions. Training has
one `maximum iterations reached` event over 20,032 decisions, exactly matching
the sole stderr fallback notice. This closes the earlier event-accounting gap
for the retry; the non-retry root remains invalid through `ABORTED.json`.

The no-overwrite replay is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260721_intervaltelemetryretry1_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 iterations, 0.1 seconds, slack
penalty 100, uncertainty gain 0.5 and cap 5.0 on CPU OSQP, it retains one source
event, zero replay errors, and remains `maximum iterations reached` with
emergency fallback. JSONL SHA-256 is
`6EC8D55348D6F44D0351C5EC4B0C85998F655734B248A2AB2B74D72E7108CD9D`;
summary SHA-256 is
`6627159DAAA487F7E7CB360D8A55536A5DE2BE7A6DC67A346163A45B930B386A`.
No CBF value changed.

Only `intervaltelemetryretry1` is eligible for numeric seed `20260721`.
This is seed-local scale and audit provenance, not performance, safety,
fallback-rate, generalization or method evidence. Next: commit/push the three
documents without `.gitignore`, then zero-process/absent-path preflight and run
5-UAV full seed `20260722` serially under the same frozen protocol and
wait-only discipline. Its telemetry must preserve all interval histories before
replay or eligibility.

## AAMAS 2027 - valid post-isolation 5-UAV full seed 20260722 retained (2026-08-01)

At launch revision `b16f055`, a zero-process/CUDA/config-hash and absent-path
preflight launched seed `20260722` in the unique
`outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260722/`
root. It used frozen `configs/rl/dynamic_graph_5uav.yaml` SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`,
five UAVs, `uncertainty_predictive_graph`, CUDA neural execution and CPU OSQP.
Only the active execution-cell wait handle was used. The process naturally
exited 0 after about 308 seconds at 100,160 transitions / 313 updates, with
correct prediction/uncertainty/five-UAV/CUDA identity.

Final checkpoint SHA-256 is
`EF75464219A872CE1BBE6534CA55444FD35D86B94E6DC072435D26DADD33EC5B` and
live telemetry SHA-256 is
`6873C3C7C007793FFA7D61054356F1B75325A7A82DF66C73E03E13F8797A5508`.
Training, initial and final records are zero-event over 20,032/160/160 CBF
decisions. All six append-only interval evaluations are retained. The interval
at 61,440 transitions contains one `solved inaccurate` fallback over its 80
decisions; the other five intervals are zero-event. That one source exactly
matches the sole stderr fallback notice, providing a real long-run validation
that the telemetry repair preserves an earlier interval context after later
interval writes.

The unique replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260722_replay_20260801.jsonl`
plus summary recursively locates the event at
`$.interval_evaluations[3].cbf.emergency_events[0]`. Under unchanged
20,000/0.1s/100/0.5/5.0 CPU-OSQP values, it has one source, zero replay errors
and replays `solved` without fallback, with retained slack about 263.39. JSONL
SHA-256 is
`C2C29FE2DF6B3BD564DA280718234D5D10F28ED64899C13C746DA002EF29718C` and
summary SHA-256 is
`97E9B5374BD9F70823D0808103A3EBF56A2D541292C786B7A3D027FB170AB3FE`.
No CBF value changed; replay variability is diagnostic only.

This is a fourth eligible five-UAV full-method seed, not performance, safety,
fallback-rate, scale or generalization evidence. Next: commit/push the three
documents without `.gitignore`, then zero-process/absent-path preflight and run
final seed `20260723` serially under the same frozen protocol and wait-only
discipline. Audit all interval/stderr contexts and replay before beginning the
five-seed six-scenario matrix.

## AAMAS 2027 - post-isolation 5-UAV full-method five-seed training complete (2026-08-01)

At launch revision `d93e281`, final seed `20260723` ran under frozen
`configs/rl/dynamic_graph_5uav.yaml` SHA-256
`28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`
in the unique
`outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260723/`
root. The CUDA neural/CPU-OSQP process was observed only through its wait
handle and naturally exited 0 after about 316 seconds at 100,160 transitions /
313 updates. Identity confirms five UAVs, prediction and uncertainty enabled,
and CUDA. Final checkpoint SHA-256 is
`F88ACA68DF279DFEEA63F8E6DDFB030C152D0F71149279BBC79A6BD62AD9BAD3`;
telemetry SHA-256 is
`ED93223A7AE0D0E9379DDED394BBAD7D13AE719ED58FA495B7692045DF621D24`.

Training has zero events over 20,032 CBF decisions. All six retained interval
evaluations are zero-event over 80 decisions each; initial/final are zero over
160/160, and stderr has zero fallback notices. The required zero-event replay
is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260723_replay_20260801.jsonl`
plus companion summary, recording `event_count=0` and
`replay_error_count=0` under unchanged 20,000/0.1s/100/0.5/5.0 values. Its
empty JSONL SHA-256 is
`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`;
summary SHA-256 is
`E48D9BE26ADE532BC92C070E55C8B949B8C376E2A520C471FB1564D2273F53D3`.

The five eligible full-method roots are `20260719`, `20260720`,
`20260721_intervaltelemetryretry1`, `20260722`, and `20260723`. A retrospective
audit of pre-fix seeds `20260719` and `20260720` found respectively 2 and 1
retained training events and exactly 2 and 1 stderr notices, with zero
initial/last-interval/final events; thus there is no evidence of an unretained
fallback in those roots, although they predate the prospective full interval
history. The non-retry seed-20260721 root remains excluded because it has four
extra stderr notices without contexts.

This completes five-seed training and per-seed replay provenance only, not a
scale, safety, fallback-rate, performance or generalization result. Next:
commit/push the three documents without `.gitignore`, then preflight a wholly
new 5-UAV full-method evaluation root. Serially run all five eligible final
checkpoints over the six canonical scenarios at 20 episodes/cell, CUDA
inference and CPU OSQP, retain per-cell logs/JSONL, audit 30 cells/600 records,
then run the unchanged 35-input all-source replay before the independent
5-UAV no-uncertainty arm.

## AAMAS 2027 - post-isolation 5-UAV full-method evaluation and all-source replay complete (2026-08-01)

At revision `2c5ea51`, the retained launcher
`outputs/core_5uav_post_actor_isolation_uncertainty_predictive_graph_evaluation_launcher_20260801.ps1`
(SHA-256
`9741AF560483948B76C28E8233043F36564779D2F5BC5B66009F4B0897B04279`)
used transient single-PATH normalization and per-cell `Start-Process -Wait`.
It serially evaluated eligible seeds `20260719`, `20260720`,
`20260721_intervaltelemetryretry1`, `20260722` and `20260723` over all six
canonical scenarios at 20 episodes/cell, CUDA neural inference and CPU OSQP.
The unique root is
`outputs/core_5uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`.
The invalid non-retry seed-20260721 checkpoint was never used.

The launcher naturally exited 0 after about 387 seconds. Independent content
audit verified 30 directories, 30 summaries, 30 JSONL files, 30 runtime
telemetry files, 30 environment files and exactly 600 records. Seed, scenario,
episode, eligible checkpoint, `graph_mappo_checkpoint` controller and completed
step 100,160 identities all match. Every environment record confirms five UAVs,
CUDA, `own_truth_and_delivered_packets_only`, and
`cpu_osqp_when_enabled`. All 30 cell exit codes are zero; all 30 cell stderr
files and launcher stderr are empty. Evaluation telemetry contains 12,000 CBF
decisions and zero emergency events.

The append-safe 35-input replay is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_5seed_training_and_evaluation_replay_20260801.jsonl`
plus companion summary. It uses exactly five eligible training telemetry files
and 30 valid evaluation telemetry files under unchanged
20,000/0.1s/100/0.5/5.0 CPU-OSQP values. It retains five source events, zero
replay errors and three replay fallbacks. Recorded statuses are four `solved
inaccurate` and one `maximum iterations reached`; replay statuses are two
`solved inaccurate`, two `solved`, and one `maximum iterations reached`.
JSONL SHA-256 is
`F06D9871683718C3EA4DA6A1B07070B270B361CC580B63E25A5607BA92808206`;
summary SHA-256 is
`1A29D2D36C3C2249546108463360FFB0A3E602A2CD943D6D6A31196DD2041A57`.
No CBF value changed.

This closes the post-isolation 5-UAV full-method training, evaluation and CBF
provenance gate only; it proves no scale, safety, fallback-rate, performance,
generalization or method effect. Next: commit/push the four updated documents
without `.gitignore`, then preflight and start independent post-isolation
5-UAV `predictive_graph` no-uncertainty seed `20260719` from scratch under
`configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`, CUDA,
CPU OSQP, a fresh root and wait-only monitoring. Complete its five seeds,
matrix and replay before any 8-UAV work.

## AAMAS 2027 - first post-isolation 5-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `bedeb3a`, a zero-process/CUDA/config-hash and absent-path
preflight launched independent seed `20260719` from scratch under frozen
`configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`
SHA-256
`6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`.
The unique root is
`outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260719/`.
The command used five UAVs, `--graph-mode predictive_graph`, 100k requested
steps and CUDA neural execution; OSQP stayed CPU-side. The zero graph-risk and
CBF uncertainty-margin gains/cap are the prespecified no-uncertainty causal
ablation, while slack penalty 100, maximum 20,000 iterations and the 0.1-second
replay limit remain frozen. Only the active execution-cell wait handle was used.
The process naturally exited 0 after about 300 seconds at 100,080 transitions /
417 updates.

The final summary verifies `uses_predicted_knowledge=true`,
`uses_uncertainty=false`, five UAVs and CUDA. Final checkpoint SHA-256 is
`A196F0C8BF0A8C8E3A062C51333EB80F959CACE8D0AF1AA2DFEDBAF8E35B4755`;
summary SHA-256 is
`1EA5A4A21DBE98D07D37A0913B0A3237880C3A4DDD1651FF4CCC9EBA2E3976A8`;
live telemetry SHA-256 is
`78BA3E51B75CA76C2F3952DBC19ED2D98E9C014D791F69DB0FD38637AA55848D`.
Training retains two `solved inaccurate` events and one `maximum iterations
reached` event over 20,016 CBF decisions. Initial/final evaluations are
zero-event over 160/160 decisions. Six append-only interval evaluations are
retained at transitions 15,360 through 92,160; each has 80 CBF decisions and
zero events. The three training events exactly match the three stderr notices,
so no event-accounting discrepancy is observed.

The append-safe CPU-OSQP replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260719_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 maximum iterations, 0.1-second
solve limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0, it
retains three source events and zero replay errors. Recorded statuses are two
`solved inaccurate` and one `maximum iterations reached`; replay statuses are
one `solved inaccurate` with emergency fallback and two `solved` without
fallback. JSONL SHA-256 is
`FE08ACF17764EE3F30BFC4E94FACF64F441A5BB4B39BEA494FA073CD7FD1AE4A`;
summary SHA-256 is
`32EB0636F418F26018B6F4166559F776EE66C55BAA78EC1916FB444C7DB3B219`.
No CBF value changed.

This is first-seed post-isolation ablation provenance only, not evidence of a
scale, safety, fallback-rate, performance, generalization or causal method
effect. Next: commit/push these four documents without `.gitignore`, then
zero-process/absent-path preflight and run seed `20260720` serially under the
same frozen command and wait-only discipline. Complete five valid seeds, their
six-scenario matrix and all-source replay before post-isolation 8-UAV work.

## AAMAS 2027 - second post-isolation 5-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `a0fbafd`, the preflight verified no Python process, the
frozen config SHA-256
`6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`, and
absent seed-`20260720` root, stdout, stderr and replay paths. The first
orchestration call was mistakenly given a one-second shell timeout and returned
124 before Python child creation. Immediate process/path audit found zero Python
processes and no root or log. The retained excluded sidecar is
`outputs/core_5uav_post_actor_isolation_ablation_seed_20260720_prelaunch_timeout_20260801.ABORTED.json`
(SHA-256
`CC4BA555F79151EF1F3BB22876D970414DB9887FE0D726293F6749F444C74AB1`).
It records a no-result prelaunch failure and is not a seed attempt for any
training, replay, aggregation, statistics or manuscript use.

Because all canonical paths remained absent, the actual independent seed then
launched once with the same frozen five-UAV `predictive_graph` command in
`outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260720/`.
Only its execution-cell wait handle was used while active. It naturally exited
0 after about 718.5 seconds at 100,080 transitions / 417 updates. Summary
identity verifies predicted delivered-packet knowledge enabled, uncertainty
disabled, five UAVs and CUDA; OSQP remained CPU-side. Final checkpoint SHA-256
is `577324344D9150020A6100D56C68D3AE7E4E52A0BB34013B2CAC6C555175E043`;
summary SHA-256 is
`A9EBA6C7A9A569323FB0E051CCDF2592699A9610FBAA6B0458EC60F6748B8D40`;
telemetry SHA-256 is
`00D731D151147F16CCCA488F3FFDF0607445329792009CFE797E3A85DB91975C`.

Training, initial and final records have zero emergency events over
20,016/160/160 CBF decisions. All six append-only interval evaluations at
15,360 through 92,160 transitions retain 80 decisions and zero events each;
stderr is empty. The append-safe zero-event replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260720_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 maximum iterations, 0.1-second
solve limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0, it
records `event_count=0` and `replay_error_count=0`. The deliberately empty JSONL
SHA-256 is
`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`;
summary SHA-256 is
`18652CA0835CC0F05BC285279D286BCA049BFF95AB7ED271D619E8371E0108C2`.
No CBF value changed.

This is second-seed post-isolation ablation provenance only, not a scale,
safety, fallback-rate, performance, generalization or causal method effect.
Next: commit/push these four documents without `.gitignore`, then
zero-process/absent-path preflight and run seed `20260721` serially under the
same frozen command and wait-only discipline. Seeds 21--23, the six-scenario
matrix and all-source replay remain before post-isolation 8-UAV work.

## AAMAS 2027 - third post-isolation 5-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `91d7b42`, a zero-process/config-hash/absent-path preflight
launched independent seed `20260721` in the unique
`outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260721/`
root. It used frozen config SHA-256
`6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`,
five UAVs, `predictive_graph`, 100k requested steps, CUDA neural execution and
CPU OSQP. Only its execution-cell wait handle was used while active. It
naturally exited 0 after about 719.7 seconds at 100,080 transitions / 417
updates.

Summary identity verifies delivered-packet prediction enabled, uncertainty
disabled, five UAVs and CUDA. Final checkpoint SHA-256 is
`1703BCC62F16CEAB1EC527532DE21D18440CE7113174B4EA2C565AC5B7AFB7B0`;
summary SHA-256 is
`28A92FD8D817054A89E35C0F08A77DADAE4E6E916A0D923FDDA86EF17C8C5851`;
telemetry SHA-256 is
`8315F8C6052BA80575695DF7EA3302385F4E08AE8AA94DB18EBA1481B04D290F`.
Training retains one `solve_time_limit` and one `maximum iterations reached`
event over 20,016 decisions. Initial/final evaluations are zero-event over
160/160 decisions. All six append-only intervals at transitions 15,360 through
92,160 retain 80 decisions and zero events each. The two source events exactly
match the two stderr notices.

The append-safe replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260721_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 maximum iterations, 0.1-second
solve limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0, it has
two sources and zero errors. The recorded time-limit event replays `solved`
without fallback; the recorded maximum-iterations event remains maximum
iterations with fallback. JSONL SHA-256 is
`BF8981935C820794D61682CC0FBE2DAB2BEAA4AF024359C87FAE861822BCCAB7`;
summary SHA-256 is
`2DBA62589172951A8EF2266BA5E318FAF1D64E78091C3F9D8ADE7CF0CBC3455B`.
No CBF value changed.

This is third-seed post-isolation ablation provenance only, not a scale,
safety, fallback-rate, performance, generalization or causal method effect.
Next: commit/push these four documents without `.gitignore`, then preflight and
run seed `20260722` serially under the same frozen command and wait-only
discipline. Seeds 22/23, the six-scenario matrix and all-source replay remain
before post-isolation 8-UAV work.

## AAMAS 2027 - fourth post-isolation 5-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `7f1cc72`, a zero-process/config-hash/absent-path preflight
launched independent seed `20260722` in the unique
`outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260722/`
root. It used frozen config SHA-256
`6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`,
five UAVs, `predictive_graph`, 100k requested steps, CUDA neural execution and
CPU OSQP. Only its execution-cell wait handle was used while active. It
naturally exited 0 after about 737.8 seconds at 100,080 transitions / 417
updates.

Summary identity verifies delivered-packet prediction enabled, uncertainty
disabled, five UAVs and CUDA. Final checkpoint SHA-256 is
`0A10275FC2663816FFC61121154D3358540C741C4789AC16E8770A2613A71D51`;
summary SHA-256 is
`9DE6F575CD014E0FAC9C1E17B0AB54A447972EBDFC90953439D2AB921FBEE6B8`;
telemetry SHA-256 is
`DF89E4A41B15B23A038AA1B44F0DEB17878E7DDB54E63AAB6710611A38CEF70F`.
Training and final evaluation retain zero events over 20,016 and 160 CBF
decisions. Initial evaluation retains eight `solve_time_limit` events over 160
decisions. All six append-only interval evaluations at transitions 15,360
through 92,160 retain 80 decisions and zero events each. The eight source
events exactly match the eight stderr notices.

The append-safe replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260722_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 maximum iterations, 0.1-second
solve limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0, it has
eight sources and zero errors. Four recorded time-limit events replay `solved`
without fallback with slack about 451.735; four remain `solve_time_limit` with
fallback. JSONL SHA-256 is
`0E710F7DC071C8EF0344A8A52111C0443A6AFD4D4D99B260F204B4B36A3AD08E`;
summary SHA-256 is
`E11E626CC4D414938F4C897F4C135027E54E7D3F5D0B89935C1F75C1E9648E4A`.
No CBF value changed.

This is fourth-seed post-isolation ablation provenance only, not a scale,
safety, fallback-rate, performance, generalization or causal method effect.
Next: commit/push these four documents without `.gitignore`, then preflight and
run final seed `20260723` serially under the same frozen command and wait-only
discipline. The six-scenario matrix and 35-input all-source replay remain before
post-isolation 8-UAV work.

## AAMAS 2027 - post-isolation 5-UAV no-uncertainty five-seed training complete (2026-08-01)

At launch revision `a304367`, zero-process/CUDA/config-hash and absent-path
preflight launched independent seed `20260723` once in the unique
`outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260723/`
root. It used frozen config SHA-256
`6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`,
five UAVs, `predictive_graph`, 100k requested steps, CUDA neural execution and
CPU OSQP. Only the active execution-cell handle was awaited.

The cell reported code 1 after about 722.2 seconds even though training
completed normally. PowerShell 5 had converted redirected native stderr into a
`NativeCommandError`; a no-state Python probe explicitly calling `sys.exit(0)`
reproduced host code 1 when stderr was redirected. Independent file audit found
a complete parseable stdout JSON, summary, telemetry, final checkpoint, all six
interval checkpoints, 100,080 transitions / 417 updates and no `ABORTED.json`.
Thus this is a launcher-envelope false failure, not an interrupted attempt and
not grounds for a rerun.

Summary identity verifies predicted delivered-packet knowledge enabled,
uncertainty disabled, five UAVs and CUDA. Final checkpoint SHA-256 is
`FE76A928143EE956176B3A627C4EBB632CDC5FC8BED5142DB93B365B3FD14C22`;
summary SHA-256 is
`18279BBC231CAF4EA75CF20DD6A4047C474359BD92381E1CD3CE5D6EF62C9D9C`;
telemetry SHA-256 is
`F8CCA2EEED98AD89EC3FE8E13C4A08798147697CD0B422AD8DDAE59B4838151F`.
Training retains one `solved inaccurate` and one `maximum iterations reached`
event over 20,016 decisions. Initial/final evaluations are zero-event over
160/160 decisions; all six append-only interval evaluations at transitions
15,360 through 92,160 retain 80 decisions and zero events each. Both source
messages remain in stderr, with the first embedded in the PowerShell envelope
and the second emitted as a plain warning.

The append-safe replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260723_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 maximum iterations, 0.1-second
solve limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0, it has
two sources and zero errors. The recorded solved-inaccurate source replays
`solved` without fallback with slack about 10.056; the recorded
maximum-iterations source remains maximum iterations with fallback. JSONL
SHA-256 is
`AE2CCC96255F03864D1C2A9F1D673CD506610C5BD11D54EF8B9E65C29A2FEFEF`;
summary SHA-256 is
`517E5A90BD6809ADE5D25D7857879F0E19230955AB9608784010DEBF28E8A6C6`.
No CBF value changed.

Eligible roots now cover independent seeds `20260719`--`20260723`; the
seed-20260720 zero-artifact prelaunch-timeout sidecar remains excluded. This
closes five-seed training/per-seed replay provenance only, not a scale, safety,
fallback-rate, performance, generalization or causal method effect. Next:
commit/push these four documents without `.gitignore`, then preflight and run a
fresh 30-cell/600-record six-scenario final-checkpoint evaluation. After its
identity audit, run the unchanged 35-input all-source replay before any
post-isolation 8-UAV work.

## AAMAS 2027 - post-isolation 5-UAV no-uncertainty evidence gate complete (2026-08-01)

At launch revision `77b602d`, a zero-process/five-checkpoint/config-hash and
absent-path preflight retained the serial launcher
`outputs/core_5uav_post_actor_isolation_predictive_graph_evaluation_launcher_20260801.ps1`.
Its SHA-256 is
`505E353E04F25FFE610FA01731F470ECA772A196FE26AC0998BED1358750C9BF`.
The launcher naturally exited 0 after about 387.4 seconds and created the unique
`outputs/core_5uav_post_actor_isolation_evaluations_20260801_predictive_graph/`
root. Each eligible independent final checkpoint for seeds `20260719` through
`20260723` was evaluated serially under frozen
`configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml` across
`nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined` and
`ood_communication_obstacle`, with 20 episodes per cell, CUDA inference and CPU
OSQP.

Independent recursive audit verified exactly 30 cell directories, 30 summaries,
30 raw JSONL files, 30 runtime telemetry files, 30 environment files and 600
records. Every seed, scenario, episode index 0--19, eligible checkpoint,
`graph_mappo_checkpoint` controller and checkpoint completed step 100,080
matches. Environment records uniformly retain five UAVs, CUDA,
`own_truth_and_delivered_packets_only` and `cpu_osqp_when_enabled`. All 30 cell
exit codes are zero, all 30 cell stderr files are empty, and launcher stderr is
empty. Evaluation telemetry totals 11,680 CBF decisions and zero emergency
events. Launcher stdout SHA-256 is
`7225C125BBFFCF9513BC7A2AD5D16740FDBB0A0CA3151761FB0D79E64E844806`;
empty launcher stderr SHA-256 is
`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`.

The append-safe all-source replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_5seed_training_and_evaluation_replay_20260801.jsonl`
plus companion summary. Its exact 35 inputs are the five eligible training
telemetry files and the 30 valid evaluation runtime telemetry files. Under
unchanged 20,000 maximum iterations, 0.1-second solve limit, slack penalty 100,
uncertainty-margin gain 0.0 and cap 0.0, it retains 15 source events, zero replay
errors and three replay fallbacks. Recorded statuses are three `solved
inaccurate`, three `maximum iterations reached` and nine `solve_time_limit`;
replay statuses are 12 `solved` without fallback, one `solved inaccurate` with
fallback and two `maximum iterations reached` with fallback. JSONL SHA-256 is
`E15440B4142BF3E95B42B91CE62564DB391E8EA350AF7FD5FC4114238EEA8C6B`;
summary SHA-256 is
`9D5EE189C3F32A3F1A88A64F7AF6F87EEFBF7073726B30308A459E65A6A97062`.
No CBF value changed.

This closes post-isolation 5-UAV independent no-uncertainty training,
evaluation and CBF provenance only. It does not establish a scale, safety,
fallback-rate, performance, generalization or causal method effect. No fresh
paired/descriptive analysis is yet specified. Next: commit/push these four
documents without `.gitignore`, then audit frozen 8-UAV inputs/exclusions and
start post-isolation 8-UAV full uncertainty-aware training from scratch. Its
five seeds, matrix and replay precede the independently trained 8-UAV
no-uncertainty arm.

## AAMAS 2027 - first post-isolation 8-UAV full-method seed retained (2026-08-01)

At launch revision `d44fbf8`, the preflight confirmed no Python process, CUDA,
frozen `configs/rl/dynamic_graph_8uav.yaml` SHA-256
`9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`,
and absent post-isolation seed/log/replay paths. Historical full, evaluation,
ablation and ablation-evaluation roots remain preserved but protocol-ineligible.
Independent seed `20260719` launched once in the unique
`outputs/core_8uav_post_actor_isolation/core_8uav_uncertainty_predictive_graph_seed_20260719/`
root. A `Start-Process` wrapper retained the actual child exit code while only
the execution-cell handle was awaited. It naturally exited 0 after about 337.4
seconds at 100,224 transitions / 261 updates.

Summary identity verifies eight UAVs, `uncertainty_predictive_graph`, predicted
delivered-packet knowledge and uncertainty enabled, and CUDA; OSQP remained
CPU-side. Final checkpoint SHA-256 is
`5037D298FE85832EC94A3BA4F400D614B623DE9E27E59F91BDA3EADDBCB1E334`;
summary SHA-256 is
`F65A7F8666237001BE24A441C3C04C838C23A663BF3989F271359359F1737B18`;
telemetry SHA-256 is
`204E143E3D75B3F6E8631C9A1AEBBC0728D2E5952C11E4A7EF8D01D854544898`.
Training retains two `maximum iterations reached` and two `solved inaccurate`
events over 12,528 decisions. Initial/final evaluations are zero-event over
160/160 decisions. All 32 append-only interval evaluations at transitions
3,072 through 98,304 are retained; transition 55,296 has one
`solve_time_limit` event and the other 31 have zero events over 80 decisions
each. The five source events exactly match the five stderr warnings.

The append-safe replay is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_seed_20260719_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 maximum iterations, 0.1-second
solve limit, slack penalty 100, uncertainty-margin gain 0.5 and cap 5.0, it has
five sources and zero errors. Both maximum-iterations events remain emergency
fallbacks; both solved-inaccurate events and the interval time-limit event
replay `solved` without fallback. JSONL SHA-256 is
`23D491F1D5C0A4495E857A88786D64FE942BBB350E6A14738897CDB4029BE12D`;
summary SHA-256 is
`656407B82BD09AF3BBF71CBAB4F4D8A386C00E064A13B835CE0103EBD7E08253`.
No CBF value changed.

This is first-seed post-isolation 8-UAV provenance only, not a scale, safety,
fallback-rate, performance or generalization result. Next: commit/push these
four documents without `.gitignore`, then preflight and run seed `20260720`
serially under the identical frozen command. Four more valid full-method seeds,
their six-scenario matrix and all-source replay remain before independent
8-UAV no-uncertainty training.

## AAMAS 2027 - second post-isolation 8-UAV full-method seed retained (2026-08-01)

At launch revision `d0f1484`, the zero-process/config-hash/absent-path preflight
launched independent seed `20260720` once in the unique
`outputs/core_8uav_post_actor_isolation/core_8uav_uncertainty_predictive_graph_seed_20260720/`
root. It used frozen `configs/rl/dynamic_graph_8uav.yaml`, CUDA neural execution,
CPU OSQP and a `Start-Process` real-child-exit wrapper. Only the execution-cell
handle was awaited. It naturally exited 0 after about 335.9 seconds at 100,224
transitions / 261 updates.

Identity verifies eight UAVs, `uncertainty_predictive_graph`, delivered-packet
prediction and uncertainty enabled, and CUDA. Final checkpoint SHA-256 is
`C0A52742DD3F80BF1DDB76E13EC84FBF861813E28A6832C0B80193E6201E7847`;
summary SHA-256 is
`B1B71629CAABF879121015E829CB7BC9A7FF334D6E77921A8CF2F13A96C415F2`;
telemetry SHA-256 is
`8882B7761396C836D16847F2D910A2C254F2D362E24D2D9F8E4044B6EDC37ACF`.
Training retains one `maximum iterations reached` event over 12,528 decisions.
All 32 interval evaluations are retained; transition 92,160 has one `solved
inaccurate` event and the other 31 have zero events over 80 decisions each.
Initial/final evaluations are zero-event over 160/160 decisions. The two source
events exactly match stderr.

The append-safe replay is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_seed_20260720_replay_20260801.jsonl`
plus summary. Under unchanged 20,000 maximum iterations, 0.1-second solve limit,
slack penalty 100, uncertainty-margin gain 0.5 and cap 5.0, it has two sources
and zero errors. Maximum iterations remains an emergency fallback; the interval
solved-inaccurate event replays `solved` without fallback with slack about
503.416. JSONL SHA-256 is
`27ED2A4FFEDA74AC1D71AB136307060A6279A5A2196CD811EBCA4A5882519068`;
summary SHA-256 is
`3EE6B9F5FDB660FDD5FDB224510AF0D4DD66CB2AF7F19B3B1AB33BB43E5E16D8`.
No CBF value changed.

This is second-seed provenance only, not a scale, safety, fallback-rate,
performance or generalization result. Next: commit/push these four documents
without `.gitignore`, then preflight and run seed `20260721` serially under the
same frozen command. Three further full-method seeds, the six-scenario matrix
and all-source replay remain before independent 8-UAV no-uncertainty work.

## AAMAS 2027 - third post-isolation 8-UAV full-method seed retained (2026-08-01)

At launch revision `ec21477`, zero-process/config-hash/absent-path preflight
launched seed `20260721` once in the unique
`outputs/core_8uav_post_actor_isolation/core_8uav_uncertainty_predictive_graph_seed_20260721/`
root. The frozen 8-UAV CUDA/CPU-OSQP command used the real-child-exit wrapper
and wait-only monitoring. It naturally exited 0 after about 358.3 seconds at
100,224 transitions / 261 updates.

Identity verifies eight UAVs, `uncertainty_predictive_graph`, delivered-packet
prediction and uncertainty enabled, and CUDA. Final checkpoint SHA-256 is
`FA68BD69DF940CCCEE704876EB4C210B4315A8FFF30E896553B9975CE1E5ABFA`;
summary SHA-256 is
`EDC55CA4895645E1B3EE38FBB400A4BC1313CB7F01C9F30026862984E0DAC4A6`;
telemetry SHA-256 is
`4C0BF2066F53DC075895B923072BAEDA9F4D811C58BF3918A27C02DE9B7AD882`.
Training retains one `solved inaccurate` event over 12,528 decisions. Initial
evaluation retains one `solve_time_limit` over 160 decisions; final evaluation
is zero-event over 160. All 32 interval evaluations are retained: transition
6,144 contains one `solved inaccurate`, transition 9,216 contains four
`maximum iterations reached`, and the other 30 contain zero events over 80
decisions each. The seven sources exactly match seven stderr warnings.

The append-safe replay is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_seed_20260721_replay_20260801.jsonl`
plus summary. Under unchanged 20,000 maximum iterations, 0.1-second solve limit,
slack penalty 100, uncertainty-margin gain 0.5 and cap 5.0, it has seven sources
and zero errors. Training solved-inaccurate, initial time-limit and interval
solved-inaccurate replay `solved` without fallback; the four interval
maximum-iterations sources remain fallbacks. JSONL SHA-256 is
`E8B59625E3A7D2E1699BB21817594598B0ED42FE21E8C6C3E554B0B149212B29`;
summary SHA-256 is
`6F26DBECF5DE5476F39D56A817495C3514C6FA23A057A3CE43584E5941EA69C7`.
No CBF value changed.

This is third-seed provenance only, not a scale, safety, fallback-rate,
performance or generalization result. Next: commit/push these four documents
without `.gitignore`, then preflight and run seed `20260722` serially. Seeds
22/23, the six-scenario matrix and all-source replay remain before independent
8-UAV no-uncertainty work.

## AAMAS 2027 - fourth post-isolation 8-UAV full-method seed retained (2026-08-01)

At launch revision `aa1d691`, zero-process/config-hash/absent-path preflight
launched seed `20260722` once in the unique
`outputs/core_8uav_post_actor_isolation/core_8uav_uncertainty_predictive_graph_seed_20260722/`
root under the same frozen CUDA/CPU-OSQP, real-child-exit and wait-only
protocol. It naturally exited 0 after about 326.1 seconds at 100,224
transitions / 261 updates.

Identity verifies eight UAVs, `uncertainty_predictive_graph`, prediction and
uncertainty enabled, and CUDA. Final checkpoint SHA-256 is
`6FEDFF756141676241DF75E2630588D72C321873FD69144B6A7F2177AD8E85A0`;
summary SHA-256 is
`F495994B39E0B432159FDC2A4CB242A52F6F71999CAFED3F036DE74187AA18FC`;
telemetry SHA-256 is
`F14093318F94C186772E6386B7E33B2A126E6C845DC3A4D146C23F62BF1D5CEA`.
Training is zero-event over 12,528 decisions; all 32 retained intervals are
zero-event over 80 decisions each; final evaluation is zero-event over 160.
Initial evaluation retains one `solve_time_limit` over 160 decisions, exactly
matching the sole stderr warning.

Replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_seed_20260722_replay_20260801.jsonl`
plus summary has one source and zero errors under unchanged
20,000/0.1s/100/0.5/5.0 values. The time-limit source replays `solved` without
fallback with slack about 430.832. JSONL SHA-256 is
`AA24B00493DF20A5680A05805A5D7D37C477F2FC1EA4466DA1180D472C4613BC`;
summary SHA-256 is
`C8673ABFCFB40A266A8602AB4FBDD1105AF0432D0B3FA64F7B08B5A809EEF61D`.
No CBF value changed.

This is fourth-seed provenance only, not a scale, safety, fallback-rate,
performance or generalization result. Next: commit/push these four documents
without `.gitignore`, then preflight and run final seed `20260723` serially.
The six-scenario matrix and all-source replay remain before independent 8-UAV
no-uncertainty work.

## AAMAS 2027 - post-isolation 8-UAV full-method five-seed training complete (2026-08-01)

At launch revision `40a1dbb`, zero-process/config-hash/absent-path preflight
launched final independent seed `20260723` once in the unique
`outputs/core_8uav_post_actor_isolation/core_8uav_uncertainty_predictive_graph_seed_20260723/`
root under the frozen CUDA/CPU-OSQP, real-child-exit and wait-only protocol. It
naturally exited 0 after about 331.2 seconds at 100,224 transitions / 261
updates.

Identity verifies eight UAVs, `uncertainty_predictive_graph`, prediction and
uncertainty enabled, and CUDA. Final checkpoint SHA-256 is
`5895CE4CD953D2044A04C16790D0B313476A4432D2BA34068D63A5325559D9D7`;
summary SHA-256 is
`FC6B9575AE5FB44E993F920888207759F08DBFA5417D2ED6FF9FA9AB08C74BCA`;
telemetry SHA-256 is
`BDCA1A381ADF1F563E52A9C4D7EAC427C1010E478561346E82B9F0EEE60726E5`.
Training retains one `solved inaccurate` event over 12,528 decisions;
initial/final evaluations are zero-event over 160/160 and all 32 interval
evaluations are zero-event over 80 decisions each. The source exactly matches
the sole stderr warning.

Replay
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_seed_20260723_replay_20260801.jsonl`
plus summary has one source and zero errors under unchanged
20,000/0.1s/100/0.5/5.0 values. It remains `solved inaccurate` with emergency
fallback. JSONL SHA-256 is
`8E6B6128054F2C7B4F784E13CD779A314C56A3039E6F1B1773729DED8CAAB16D`;
summary SHA-256 is
`DA631FACCE63BCEE0AFC4CE3D3AE90DD9131BC3335F4E286E6B82A33C9FB46A5`.
No CBF value changed.

Eligible post-isolation full-method roots now cover independent seeds
`20260719`--`20260723`. This closes five-seed training/per-seed replay
provenance only, not a scale, safety, fallback-rate, performance or
generalization result. Next: commit/push these four documents without
`.gitignore`, then preflight and run a fresh 30-cell/600-record six-scenario
evaluation. After its identity audit, run the unchanged 35-input all-source
replay before independent 8-UAV no-uncertainty work.

## AAMAS 2027 - post-isolation 8-UAV full-method evidence gate complete (2026-08-01)

At matrix launch revision `81913bf`, five-checkpoint/config-hash/zero-process
and absent-path preflight retained
`outputs/core_8uav_post_actor_isolation_uncertainty_predictive_graph_evaluation_launcher_20260801.ps1`.
Its SHA-256 is
`062271DBCF034D3D9BE47062B15FF8DEC2C54C408ED7C10FEA605729343E8FFB`.
The serial launcher naturally exited 0 after about 460.6 seconds and created
the unique
`outputs/core_8uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`
root. It evaluated eligible seeds `20260719`--`20260723` across `nominal`,
`delay_only`, `loss_only`, `dynamic_only`, `combined` and
`ood_communication_obstacle`, with 20 episodes per cell, CUDA inference and CPU
OSQP.

Independent recursive audit verified exactly 30 cell directories, 30 summaries,
30 raw JSONL files, 30 runtime telemetry files, 30 environment files and 600
records. Seed, scenario, episode indices 0--19, eligible checkpoint,
`graph_mappo_checkpoint` and checkpoint completed step 100,224 all match.
Environment records uniformly retain eight UAVs, CUDA,
`own_truth_and_delivered_packets_only` and `cpu_osqp_when_enabled`. All 30 cell
exit codes are zero and all cell/launcher stderr files are empty. Evaluation
telemetry totals 12,000 CBF decisions and zero emergency events. Launcher stdout
SHA-256 is
`F8B3F5F46914AF09EA2F64CACDB841C3D525F4368204E4F6CA2438C5D0B2A208`;
empty stderr SHA-256 is
`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`.

The append-safe all-source replay is
`outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_5seed_training_and_evaluation_replay_20260801.jsonl`
plus summary. Its exact 35 inputs are five eligible training telemetry files and
30 valid evaluation runtime telemetry files. Under unchanged 20,000 maximum
iterations, 0.1-second solve limit, slack penalty 100, uncertainty-margin gain
0.5 and cap 5.0, it retains 16 training sources, zero errors and ten replay
fallbacks. Recorded statuses are seven `maximum iterations reached`, three
`solve_time_limit` and six `solved inaccurate`; replay statuses are three
maximum-iterations fallbacks, six time-limit fallbacks, one solved-inaccurate
fallback and six solved non-fallback decisions. This all-source run differs
from the combined separate per-seed replay outcomes because diagnostic solver
timing/order is variable. Both are retained; no CBF value is changed and no
result is selected for tuning. JSONL SHA-256 is
`205B1A2E4EA56A862E4E195E3119AA922E39CC5741628A78B2671D814174BAF2`;
summary SHA-256 is
`A5D8B27BB5CDC89174E4E168A8D02BF53068DDF03ACD60F69F2E520D419FD2B3`.

This closes post-isolation 8-UAV full-method training, evaluation and CBF
provenance only, not a scale, safety, fallback-rate, performance or
generalization result. Next: commit/push these four documents without
`.gitignore`, then audit the frozen 8-UAV no-uncertainty config and launch
independent seed `20260719` from scratch. Complete its five seeds, matrix and
replay before any paired/descriptive analysis.

## AAMAS 2027 - first post-isolation 8-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `a8485d6`, the preflight confirmed no Python process, CUDA,
frozen
`configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml` SHA-256
`F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`,
and absent post-isolation seed/log/replay paths. Historical ablation training
and evaluation roots remain preserved but protocol-ineligible. Independent
seed `20260719` launched once in the unique
`outputs/core_8uav_post_actor_isolation_ablation/core_8uav_predictive_no_uncertainty_seed_20260719/`
root. The real-child-exit wrapper and wait-only monitoring were used. It
naturally exited 0 after about 334.9 seconds at 100,224 transitions / 261
updates.

Identity verifies eight UAVs, `predictive_graph`, prediction enabled,
uncertainty disabled and CUDA; OSQP remained CPU-side. Final checkpoint SHA-256
is `E97C95BF620677D0E2D12DBA8017C1C595134F12A8BA1E710EE5616B1928BF43`;
summary SHA-256 is
`C28DDA23052C96572A2FA059C9DEFFBFC55D1C997677AAA533D8DCF7F6827853`;
telemetry SHA-256 is
`26AE0CFDA3F99A2D0AC1229FFB8E4FC12BB07B9834ABFBF23CDC7B6B085454B8`.
Training retains one `solved inaccurate` event over 12,528 decisions;
initial/final evaluations are zero-event over 160/160 and all 32 retained
intervals are zero-event over 80 decisions each. The source exactly matches the
sole stderr warning.

Replay
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260719_replay_20260801.jsonl`
plus summary has one source and zero errors under unchanged 20,000 maximum
iterations, 0.1-second solve limit, slack penalty 100, uncertainty-margin gain
0.0 and cap 0.0. It replays `solved` without fallback with slack about 0.107.
JSONL SHA-256 is
`CBC79913C06BEF6C04008FCF8DEC0A2138E557493016E97710D796D62E346779`;
summary SHA-256 is
`42A181E6A882A311B1BFB9F2C60CE4A0EE3AF1FF6AA5F0DD24B9305149E3DF86`.
No CBF value changed.

This is first-seed post-isolation ablation provenance only, not a causal,
scale, safety, fallback-rate, performance or generalization result. Next:
commit/push these four documents without `.gitignore`, then preflight and run
seed `20260720` serially. Four further seeds, the six-scenario matrix and
all-source replay remain before any paired/descriptive analysis.

## AAMAS 2027 - second post-isolation 8-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `999cedf`, a zero-process/CUDA/config-hash and absent-path
preflight launched independent seed `20260720` once in the unique
`outputs/core_8uav_post_actor_isolation_ablation/core_8uav_predictive_no_uncertainty_seed_20260720/`
root. It used frozen
`configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`
SHA-256
`F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`,
eight UAVs, `predictive_graph`, 100k requested steps, CUDA neural execution and
CPU OSQP. The transient launcher normalized duplicate-case PATH entries and
retained the real child exit code; only the execution-cell wait handle was used
while active. It naturally exited 0 after about 807.2 seconds at 100,224
transitions / 261 updates.

Summary identity verifies delivered-packet prediction enabled, uncertainty
disabled, eight UAVs and CUDA. Final checkpoint SHA-256 is
`B49EED820696E1487CA9EC98992F9750C3AD012875A04951E8DCCA3C27F02FE3`;
summary SHA-256 is
`F80509CED072F1BC98AD33BA6F0ED77CAB02650555A7C4CA1E8F6019A145F9B5`;
telemetry SHA-256 is
`3D9D38CAAA4267AA10AC16C1D81654774C24FCB69D531B61741D133BA28B4695`;
stdout/stderr SHA-256 values are respectively
`D63C538944DCAC998C3A7A536C0BCBB01BF754B185234A4C708CF2AC5F212AA3`
and
`A184D64488D900C69471FE3A5CB8AA6114789571C0A58A652CBACCB22CFDE440`.

Training retains 275 emergency events over 12,528 CBF decisions: 273
`solve_time_limit`, one `solved inaccurate` and one `maximum iterations
reached`. Initial/final evaluations retain 11/14 time-limit events over 160/160
decisions. All 32 append-only interval evaluations are retained; 24 intervals
contain 157 time-limit events and eight are zero-event. These 457 complete
sources exactly match stderr's 455 time-limit, one solved-inaccurate and one
maximum-iterations notices. No event-accounting discrepancy or `ABORTED.json`
exists. No slack, iteration cap, solve-time limit, tolerance, uncertainty
margin, observation contract or controller changed.

The append-safe replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260720_replay_20260801.jsonl`
plus companion summary. Under unchanged 20,000 maximum iterations, 0.1-second
solve limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0, it
retains all 457 unique source pointers and zero errors. Replay statuses are 118
`solved`, 337 `solve_time_limit`, one `solved inaccurate` and one `maximum
iterations reached`; 339 replays use emergency fallback. JSONL SHA-256 is
`D0D9186F7FB8DDD9EB85A09FF0FE783108137FC0E95BCE54765067E171BBE58E`;
summary SHA-256 is
`E363E2278D58D76072528AD207276DF566EBAB9B19F4D07910D2A3229B48CB8D`.

This large seed-local numerical cluster is diagnostic provenance only. It does
not establish a causal, scale, safety, fallback-rate, performance or
generalization result and does not justify changing any CBF value. Two valid
8-UAV no-uncertainty seeds now exist. Next: commit/push these four documents
without `.gitignore`, then zero-process/absent-path preflight and run seed
`20260721` serially under the identical frozen command and wait-only discipline.
Seeds 21--23, the six-scenario matrix and 35-input all-source replay remain
before any paired/descriptive analysis.

## AAMAS 2027 - third post-isolation 8-UAV no-uncertainty seed and replay alias fix retained (2026-08-01)

At launch revision `9a3ec69`, a zero-process/CUDA/config-hash and absent-path
preflight launched independent seed `20260721` once in the unique
`outputs/core_8uav_post_actor_isolation_ablation/core_8uav_predictive_no_uncertainty_seed_20260721/`
root. It used frozen
`configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`
SHA-256
`F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`,
eight UAVs, `predictive_graph`, CUDA neural execution and CPU OSQP. The
real-child-exit wrapper and wait-only monitoring were retained. It naturally
exited 0 after about 805.5 seconds at 100,224 transitions / 261 updates.

Identity verifies delivered-packet prediction enabled, uncertainty disabled,
eight UAVs and CUDA. Final checkpoint SHA-256 is
`16408CCAE3DF1BD06CA24593FB51B2A1BD48A5A1911824FE63872725867985FC`;
summary SHA-256 is
`221A9DECAB1AB3872A3074027398C5EE5F89D69A2EECB937601E3E1171938185`;
telemetry SHA-256 is
`0D67E4655A76B829332E7FA9E5A34E6F0327346B3687E0F8226085359B2BF148`;
stdout/stderr SHA-256 values are respectively
`338995C9C7E46AF7C19F9C3FACAF98E904254CC6718EA5D8749FF3CABD803A36`
and
`BA2E50BC5C52F8C9E049A2489D07D2DF75F357F8CEDC06AE964A397D49424CE3`.

Training retains 271 `solve_time_limit` events over 12,528 decisions;
initial/final evaluations retain 11/8 more over 160/160 decisions. All 32
append-only interval evaluations are retained; 26 intervals contain 156
time-limit events and six are zero-event. All 446 complete sources are
time-limit events and exactly match stderr's 446 notices. No event-accounting
gap or training `ABORTED.json` exists. No CBF value, observation contract or
controller changed.

The initial replay attempt at
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260721_replay_20260801.jsonl`
incorrectly wrote 450 records. Root-cause tracing showed that the last interval
at transition 98,304 retains four events in canonical
`interval_evaluations[31].cbf`, while backward-compatible
`last_interval_evaluation_cbf` contains the exact same four mappings. The
generic recursive walker replayed both paths. The invalid JSONL/summary remain
untouched and are explicitly excluded by sidecar
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260721_replay_20260801.ABORTED.json`
(SHA-256
`3F9DB96DE88BF33FAEF0F7C2E8F6019AECE882814845D299A389B8865C9B622B`).

A test-first minimal repair in `scripts/replay_cbf_fallbacks.py` skips
`last_interval_evaluation_cbf` only when append-only history exists and its
last `cbf` mapping is exactly equal to the alias. The regression test first
failed with the duplicate pointer, then passed. The corrected append-safe
output is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260721_replay_aliasdedup_rerun1_20260801.jsonl`
plus summary. Under unchanged 20,000 maximum iterations, 0.1-second solve
limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0, it retains
446 records, 446 unique canonical pointers and zero errors. Replay statuses are
133 `solved` and 313 `solve_time_limit`; the latter 313 use emergency fallback.
JSONL SHA-256 is
`4872E77BDEA291E4532115DE8BE673D8B283321B3F63BCDAA89F5A5DEECC7920`;
summary SHA-256 is
`FC8DB6766D714F9D3436434651875017D6197BEA15E9C7D06ADE5530D9D20E83`.

Verification: the focused regression test passed; full pytest recorded 160
passed, one existing unrelated `tests/test_legacy_audit.py` failure because its
obsolete restored-source path contains zero `.m` files while it expects 81,
and 190 existing OSQP warnings. Ruff passed, and mypy with explicit package
bases passed for the two changed/test files. This fix changes event discovery
only, not training, CBF or results. Three valid 8-UAV no-uncertainty seeds now
exist, but this remains diagnostic provenance rather than causal, scale,
safety, fallback-rate, performance or generalization evidence. Next:
commit/push the script, test and four documents without `.gitignore`, then
preflight and run seed `20260722` serially. Seeds 22/23, the six-scenario matrix
and 35-input all-source replay remain before paired/descriptive analysis.

## AAMAS 2027 - fourth post-isolation 8-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `0d61b16`, the preflight confirmed branch/remote equality,
no Python process, CUDA on the RTX 5060, frozen
`configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`
SHA-256
`F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`,
and absent seed-20260722 training/log/replay targets. The first launcher call
failed before child creation because PowerShell `Start-Process` rejected
case-colliding inherited `Path`/`PATH` keys. It produced no root and only the
two zero-byte original logs. These are preserved with
`outputs/core_8uav_post_actor_isolation_ablation_seed_20260722_launcher.ABORTED.json`
(SHA-256
`E2586A9CB1E58FAD389BEE189A6F795EFB5BCAC9A2544BAFE65DF1AA3317AD98`)
and are excluded.

The next unique `retry1` attempt entered Python, but the direct PowerShell
wrapper used terminating native-error handling. Its first expected CBF stderr
notice was promoted to `NativeCommandError`, interrupting the job before any
checkpoint or summary. No Python process remained. The partial root, its
TensorBoard directory, zero-byte logs and
`outputs/core_8uav_post_actor_isolation_ablation/core_8uav_predictive_no_uncertainty_seed_20260722_retry1/ABORTED.json`
(SHA-256
`EC8B97CC8F4B05092E8221CA3112B3B655584735DC6A2DC5B31B9D9E9177F110`)
remain preserved and excluded. Neither launcher failure is counted as a valid
seed or overwritten.

The transparent `cmd.exe` stderr-redirection replacement then launched the
unchanged Python command into the fresh eligible
`outputs/core_8uav_post_actor_isolation_ablation/core_8uav_predictive_no_uncertainty_seed_20260722_retry2/`
root. It naturally exited 0 after 802.7 seconds at 100,224 transitions / 261
updates. Identity verifies eight UAVs, `predictive_graph`, delivered-packet
prediction enabled, uncertainty disabled and CUDA neural execution; OSQP
remained CPU-side. Final checkpoint, summary and live telemetry SHA-256 values
are respectively
`7D96BFD79D4406F7C9D4FF8746240524FC72489B1B40C0194423F34E9B014391`,
`A5F3184E9927057BCCD2F2B4D98C5272D2F313127503B2F00A91644489435A0A`
and
`647A9C67C987A036B141EF92DAB823EE6DD9EDAB06B58B40AEBF343332D9FDF3`.
The unique stdout/stderr SHA-256 values are
`5C8C8A9CF8460A90809FE7B506B114021C3721D377E05CF6D19642B3E4C078B5`
and
`FB0EA4A4DD6CCE3AA8ADD2C8D3B0F42A4B8C869871467C22FFB782E70587DD25`.
There are 33 checkpoint files and no training `ABORTED.json`.

Training retains 306 `solve_time_limit` events over 12,528 decisions;
initial/final evaluations retain 14/16 more over 160/160 decisions. All 32
append-only intervals are present; 15 contain 93 time-limit events. The 429
canonical sources exactly match stderr's 429 notices. The equal
`last_interval_evaluation_cbf` compatibility alias is not double counted by
the repaired event walker.

Replay
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260722_replay_20260801.jsonl`
plus summary uses the unchanged 20,000 maximum iterations, 0.1-second solve
limit, slack penalty 100, uncertainty-margin gain 0.0 and cap 0.0. It retains
429 records, 429 unique canonical pointers and zero errors; source groups are
306 training, 14 initial, 93 interval and 16 final. Replay statuses are 124
`solved` and 305 `solve_time_limit`; all 305 time-limit results use emergency
fallback. JSONL SHA-256 is
`790DC7B083CEB18E18C31C3A5D14B686E07BA85A8891EA1F0803B32F10C11BAA`;
summary SHA-256 is
`1F55F1444D5449BC06B57E72CDB29B4614755F7B13E60757CBDAEE29F7CE284C`.

No configuration, observation contract, CBF value or controller changed. This
is seed-local diagnostic provenance, not causal, scale, safety, fallback-rate,
performance or generalization evidence. Four valid 8-UAV no-uncertainty seeds
now exist. Next: commit/push these four documents without `.gitignore`, then
run a zero-process/config-hash/absent-path preflight and train seed `20260723`
serially under the identical frozen protocol. The six-scenario matrix and
35-input all-source replay remain after that final seed.

## AAMAS 2027 - fifth post-isolation 8-UAV no-uncertainty seed retained (2026-08-01)

At launch revision `34f9303`, a zero-process/CUDA/config-hash/absent-path
preflight launched independent seed `20260723` once in the unique
`outputs/core_8uav_post_actor_isolation_ablation/core_8uav_predictive_no_uncertainty_seed_20260723/`
root. The command used frozen
`configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`
SHA-256
`F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`,
eight UAVs, `predictive_graph`, CUDA neural execution and CPU OSQP. The
transparent stderr wrapper returned the real child code. Training naturally
exited 0 after 822.9 seconds at 100,224 transitions / 261 updates.

Identity verifies delivered-packet prediction enabled, uncertainty disabled,
eight UAVs and CUDA. The 33 checkpoint files include the final checkpoint with
SHA-256
`27EC107FF41B2C3248EBE4737D6A118064F90C821F5BF4F2C8B5294B1A74C35D`.
Summary and live telemetry SHA-256 values are
`61CE5AFB8F02B81F8FBF75917EE23C3A8B7E62A5119E387384460B125E2C3AAE`
and
`296DDD53EB412A3AAD09C5C79ADA822E10DCF52E923B8039ED32372DA0B26FB7`;
stdout/stderr SHA-256 values are
`BCC80D828AF3EF0EA72FC5C6711D3C80DF6ECCA43D2FF3BFCF91DF869A1F843F`
and
`8850D5AA4AFA7EA0E182B4425CEF9BF841A18E1B626A9C9714FD496319F5A5D8`.
No training `ABORTED.json` exists.

Training retains 301 emergency events over 12,528 decisions; initial/final
evaluations retain 13/6 over 160/160. All 32 append-only intervals are present;
23 contain 137 events. Across all 457 canonical sources, 456 are
`solve_time_limit` and one is `maximum iterations reached`; stderr has exactly
the same 456/1 notices. The equal last-interval compatibility alias is not
double counted.

Replay
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260723_replay_20260801.jsonl`
plus summary uses unchanged 20,000 maximum iterations, 0.1-second limit, slack
penalty 100, uncertainty-margin gain 0.0 and cap 0.0. It contains 457 records,
457 unique pointers and zero errors, grouped as 301 training, 13 initial, 137
interval and six final. Replay statuses are 129 `solved`, 326
`solve_time_limit`, one `solved inaccurate` and one `maximum iterations
reached`; 328 use emergency fallback. JSONL/summary SHA-256 values are
`EC7C2D0717D1486A2145CBC7D5CD5A44BE545B099093276996D8969D4E584500`
and
`988D31FB83C5FDEE0706168456B3728778956CABFFD09473828E3EAB2C368A92`.

No CBF value, observation contract or controller changed. Five eligible
post-isolation 8-UAV no-uncertainty trainings now exist, but this closes only
training/per-seed replay provenance. It establishes no causal, scale, safety,
fallback-rate, performance or generalization result. Next: commit/push these
four documents without `.gitignore`, then verify all five checkpoint identities,
no Python process and absent fresh matrix/replay targets. Run a new 5-by-6
canonical evaluation matrix at 20 episodes per cell, audit all 600 records, and
then replay the five training plus 30 evaluation telemetry inputs.

## AAMAS 2027 - post-isolation 8-UAV no-uncertainty evidence gate complete (2026-08-01)

At matrix launch revision `c47aaf7`, all five eligible final checkpoints were
re-audited at 100,224 transitions / 261 updates with eight UAVs,
`predictive_graph`, CUDA, prediction enabled, uncertainty disabled and no
training `ABORTED.json`. Seed `20260722` correctly maps to its eligible
`retry2` root. Fresh matrix, launcher-log and all-source replay targets were
absent, and no Python process existed.

Retained launcher
`outputs/core_8uav_post_actor_isolation_predictive_graph_evaluation_launcher_20260801.ps1`
(SHA-256
`3E27C3EA6434AC8925BBC396C0BB1C81B871A0268CDF35B7B3EEF47333E419E9`)
serially evaluated the five seeds across `nominal`, `delay_only`, `loss_only`,
`dynamic_only`, `combined` and `ood_communication_obstacle`, 20 episodes per
cell, into the unique
`outputs/core_8uav_post_actor_isolation_evaluations_20260801_predictive_graph/`
root. It naturally exited 0 after 403.7 seconds. Launcher stdout SHA-256 is
`98B46602F539489FA3333A1B3AE1A315899741DC8A4915768795E74830D17F68`;
empty stderr SHA-256 is
`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`.

Independent content audit verifies 30 directories, 30 summaries, 30 raw JSONL
files, 30 runtime telemetry files, 30 environment files and exactly 600
records. Every seed/scenario pair has episode indices 0--19, the correct
eligible checkpoint, completed step 100,224 and `graph_mappo_checkpoint`
controller. All environment records confirm eight UAVs, CUDA,
`own_truth_and_delivered_packets_only` and `cpu_osqp_when_enabled`. All 30 cell
exits are zero; all 30 cell stderr files and launcher stderr are empty.
Evaluation telemetry contains 11,337 actual CBF decisions and zero emergency
events.

The append-safe all-source replay is
`outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_5seed_training_and_evaluation_replay_20260801.jsonl`
plus summary. It takes exactly five eligible training and 30 valid evaluation
telemetry files in fixed seed/scenario order under unchanged 20,000 maximum
iterations, 0.1-second solve limit, slack penalty 100, uncertainty-margin gain
0.0 and cap 0.0. It retains 1,790 records, 1,790 unique
`telemetry + event_pointer` sources and zero errors; all events come from
training because evaluation has none. Recorded statuses are 1,786
`solve_time_limit`, two `solved inaccurate` and two `maximum iterations
reached`. Replay statuses are 460 `solved`, 1,326 `solve_time_limit`, two
`solved inaccurate` and two `maximum iterations reached`; 1,330 replays use
fallback. JSONL/summary SHA-256 values are
`EF08BBE377BDF8B49D74DD07F70CE23CFD8932D613CBE09C06B45864924A9B3F`
and
`EE40FE1550535C79B799DA8C29C640ACF8FDF2B0E5EA3A63DDDFF36370620894`.

The five separate per-seed replays sum to 1,285 fallbacks, 45 fewer than the
fixed-order all-source run. Both are preserved as solver timing/order
diagnostic provenance; neither is selected, averaged or used to justify CBF
tuning. No CBF value, observation contract or controller changed. This closes
the post-isolation 8-UAV no-uncertainty training/evaluation/replay gate only.
It establishes no causal, scale, safety, fallback-rate, performance or
generalization result. Next: commit/push these four documents without
`.gitignore`, then specify and independently verify the fresh post-isolation
paired/descriptive analysis using seed as the inferential unit while retaining
all historical summaries as exclusions.

## AAMAS 2027 - post-isolation descriptive aggregation frozen before results (2026-08-01)

At parent revision `67b40e0`, read-only preflight verified that all eight
eligible post-isolation evaluation roots used by the planned 3-UAV four-arm and
5/8-UAV full-vs-no-uncertainty summaries each contain 30 cell directories, 30
raw JSONL files and 600 records. The three new descriptive output targets did
not exist; HEAD matched the remote and `.gitignore` remained the only dirty
user file.

Before inspecting any new aggregate values, `docs/aamas2027_analysis_plan.md`
now freezes the independent unit, input roots, arm/delta direction, complete
metric list, output names and claim boundary. Every seed--scenario cell is
first averaged over its 20 episodes; only the five independently trained seeds
are descriptive units. Outputs contain seed means/sample standard deviations
and, for 5/8 UAV, paired full-minus-no-uncertainty deltas. They contain no
p-values, confidence intervals, multiple-comparison decisions, rankings or
superiority labels. All six scenarios and every supported metric must remain,
and CBF fields stay separate diagnostic telemetry.

Next: commit/push this prospective plan and the three synchronized documents
without `.gitignore`. Only after that commit may the three new descriptive
artifacts be generated with the existing validated summary tools, independently
audited against eligible roots, documented and pushed as provenance records.

## AAMAS 2027 - post-isolation descriptive aggregation generated and independently validated (2026-08-02)

The plan was committed and pushed first at revision `030b50d`. A first targeted
pytest call produced one pass and eight setup errors because the sandbox denied
pytest temporary/cache directories; it generated no analysis output and did
not reach the affected test bodies. The identical rerun with filesystem access
passed all nine targeted summary/evaluator tests with four existing OSQP
deprecation warnings.

The unchanged summary tools then generated three unique artifacts and matching
stdout/stderr logs. The 3-UAV four-arm JSON SHA-256 is
`B3CB5F017A241DAF29C366DA9592624076A2FE44519F51809D5AE50251CF2F3E`;
the 5-UAV full-minus-no-uncertainty JSON is
`6661DBDDF678039581ED7FEC0D84595A641C25B87C6877DCEA95C3ADD5359120`;
the 8-UAV counterpart is
`E487CD7C16EAA285025917848AD51268BDC11228174ED17EFB395DC740144C61`.
Each stdout has the same hash as its JSON and each stderr is empty.

An independent validator that did not import the summary modules re-read all
4,800 eligible raw episode records. It verified 30 cells/600 records per root,
unique complete seed--scenario--episode identities, numeric/range validity,
matching cross-arm cell sets, all six scenarios, all 16 metrics and the absence
of inferential/ranking fields. It recomputed every seed-cell mean, sample
standard deviation and 5/8-UAV full-minus-no-uncertainty paired delta; every
number matched within `1e-12`. Stdout and JSON were identical.

The validated values are mixed rather than uniformly favorable. This blocks a
superiority narrative and reinforces the frozen claim boundary. Full details,
eligible inputs, artifact hashes, methodology and the explicit 13-artifact
`ABORTED` registry are in
`docs/post_actor_isolation_descriptive_validation.md`. The rating is **Share
with caveats** internally, not submission-ready evidence by itself. Next:
commit/push the five documents without `.gitignore`, then continue the
primary-paper full-text evidence and manuscript/reproducibility gates.

## AAMAS 2027 - primary-paper evidence gate advanced (2026-08-02)

At parent revision `f302b71`, the required gstack browser was used for all web
access. The AAAI and IJCAI official proceedings pages yielded the official
nine-page DACOM and DHCG PDFs. Both files passed `%PDF-` magic checks, were
text-extracted, rendered page by page, and fully reviewed. Their retained paths
and SHA-256 values are:

- `outputs/literature_primary_sources_20260802/Yuan_et_al_2023_DACOM_AAAI_official.pdf`:
  `E186B67D9558F117F6C83BBC75A38D6616825BA257DE6739A685E962DAA11EC9`;
- `outputs/literature_primary_sources_20260802/Liu_et_al_2023_DHCG_IJCAI_official.pdf`:
  `310BAA7F8446EEF14A69D9516046627DBE8ED2EF3A564A5E1CE70BDFF6FCD652`.

The AAMAS 2026 official graph-CBF extended abstract was downloaded again to
`outputs/literature_primary_sources_20260802/Deng_et_al_2026_neural_graph_CBF_AAMAS_extended_abstract_official.pdf`.
Its SHA-256
`0AE5CA03242EDE2EB5E60822729EE6B6279F4FA4C14B836E411658260E09D29A`
matches the earlier review record. OpenAlex exposes no longer repository
version, so its evidence tier remains extended abstract.

The IEEE UAV paper remains full-text blocked. IEEE Xplore returned HTTP 418;
the Crossref publisher PDF link and IEEE stamp PDF path also produced no file.
OpenAlex marks it closed access and reports no repository full text. Crossref,
ORCID and OpenAlex metadata/abstract were reviewed, but no method-difference
claim is permitted for communication assumptions, obstacle dynamics,
uncertainty or safety mechanism.

`docs/related_work_matrix.md` now records section-level DACOM/DHCG evidence,
source tiers, exact hashes, the failed IEEE paths, and a non-exhaustive claim
boundary. The review supports narrow paper-specific distinctions only; it does
not support a “first”, exhaustive novelty, formal safety, superiority or robust
generalization claim. Next: commit/push the literature ledger and synchronized
records without `.gitignore`, then repair the stale/corrupted method-readiness
document and expand the reproducibility/manuscript package against the current
post-isolation evidence.

## AAMAS 2027 - method, reproducibility, and manuscript package repaired (2026-08-02)

At parent revision `9ac4032`, the stale/garbled landing and protocol documents
were replaced with a clean evidence-bounded package: `README.md`,
`docs/aamas2027_method_and_readiness.md`, `docs/reproducibility.md`,
`docs/training.md`, and `docs/evaluation.md`. The research brief and analysis
plan were synchronized, and `docs/aamas2027_manuscript_package.md` was added
with a restrained title/abstract draft, section jobs, result-language gate,
figure/table provenance contract, limitations, and remaining submission gates.

Targeted implementation review established a method distinction that the old
documents blurred. The MLP actor is communication-limited but its critic uses
complete centralized environment state. The GraphMAPPO actor and graph critic
both receive the communication-limited graph tensors; peer private node
features/current truth are not injected. The execution-side CBF separately
uses centralized simulator geometry. The new method document also states that
the historical raw-graph label is actually `graph_mode=mappo`, a self-loop-only
graph, and distinguishes the 3-UAV graph-uncertainty comparison from the joint
graph+CBF no-uncertainty ablation at 5/8 UAV.

The checked-in protocol manifest incorrectly named the raw arm
`distance_graph`, even though all eligible post-isolation training used
`mappo`. It now records `mappo` and the actual artifact prefixes. A test-first
minimal loader repair accepts the already supported CLI mode and then locks the
four expected graph-mode identities. No model, checkpoint, output, CBF value,
or completed experiment was changed.

Fresh verification passed Ruff, mypy with `--explicit-package-bases` over 103
source files, `git diff --check`, and 28 focused protocol/communication/
predictive-graph/replay tests. Full pytest finished with 160 passes and one
unrelated legacy audit failure plus 190 OSQP dependency warnings. On this host,
the failure expects 81 MATLAB files in
`D:\yolo\multiuav\legacy_hgalo\HGALO_恢复源码`, but that external sibling path is
absent and zero files are observed; an earlier environment snapshot had
observed 83. The RL package does not create or modify that external source.

Final process audit found no Python process. `nvidia-smi` showed desktop,
Unity/player, system, and application processes but no Python training process.
`.gitignore` remains the only user-owned unrelated edit and must stay unstaged.
The resulting research package is internally auditable descriptive evidence,
not yet external-submission-ready. Next gates are a prospective inferential
decision, graph-only/CBF-only ablations if component causality is desired,
traceable all-scenario tables/figures, broader/full-text literature evidence,
and a final manuscript-to-artifact audit. External submission remains
prohibited without explicit user authorization.

## AAMAS 2027 - descriptive report generator retained; HTML runtime blocker audited (2026-08-02)

At parent revision `2e29747`, a test-first standard-library generator was added
as `scripts/build_aamas_descriptive_report.py`. It validates all six scenarios,
five trained seeds, 20 episodes per seed-scenario cell, four 3-UAV arms, all 11
task metrics and all five CBF diagnostic metrics before emitting a bounded
canonical report artifact and provenance sidecar. Two focused tests cover full
row retention, source SQL, renderer-facing chart identities, claim boundaries,
and rejection of an incomplete scenario matrix. Ruff, mypy and both tests pass.

The report design followed the data-visualization and technical-report
contracts: neutral chart titles; all scenarios retained; no ranking; task and
CBF data separated; full-minus-no-uncertainty direction fixed; charts adjacent
to explanatory text; and every native chart/table backed by a bounded snapshot
source with artifact-snapshot SQL. The final canonical attempt is
`outputs/aamas2027_descriptive_report_20260802_retry8/artifact.json` (SHA-256
`492BFA4E615258DB36DF7856E63E19BF2CC2CCB9A1426718C1FE8126C8AD0521`)
plus `provenance.json` (SHA-256
`41188D92131A5F5041D10BEBBFDCB21823D76C47C3F9F8DA796D42A84E2967C1`).

No HTML is eligible. Eight unique attempts plus one separated diagnostic root
are preserved with `ABORTED.json`. Early attempts exposed schema/source errors;
later attempts exposed horizontal-bar overflow, signed-bar reader fallback and
finally a portable-runtime desktop layout defect. In retry8, both native charts
rendered in the enhanced reader, but the official Chromium verifier rejected
page-level horizontal overflow. The generated runtime uses a sticky
`width:100vw` header with viewport-relative negative margins on a vertically
scrolling long report. Its failure screenshot SHA-256 is
`64C00A935D75751D38C78A6A36D843B561F713F566AA6993189835D4E26A0DC8`.
Generated HTML was never patched, no failed file is presented as valid, and
the upstream runtime must be fixed or a different fully validated surface used.

`docs/aamas2027_statistical_decision.md` also closes the current inferential
gate: the already viewed five-seed cohort remains descriptive. No retrospective
p-values, intervals, significance labels or superiority claims will be added.
With five paired seed differences, a conventional exact two-sided sign-flip
test has minimum attainable p-value `2/32 = 0.0625`; more importantly, no
endpoint/hypothesis/multiplicity plan preceded value inspection. Any future
confirmatory cohort needs a committed plan before new seed results are viewed.

Final verification passes Ruff, mypy over 104 source files, and three focused
report/core-protocol tests. Full pytest is 162 passed with one unrelated legacy
audit failure and 190 OSQP dependency warnings. The failure is unchanged: the
external sibling MATLAB source path is absent, so the test observes zero files
while asserting 81.

Next: commit/push the generator, tests and synchronized documentation without
`.gitignore`. The remaining paper gates are the portable surface blocker,
component-only ablations if causal component claims are retained, broader
full-text literature coverage, and a final manuscript-to-artifact audit.

## AAMAS 2027 - validated portable PDF evidence surface retained (2026-08-02)

At parent revision `a310097`, the blocked portable HTML was replaced by an
independent static-PDF surface rather than patched. A test-first
`scripts/build_aamas_descriptive_pdf.py` now validates the three frozen summary
schemas, refuses overwrite, creates a six-page landscape-A4 PDF with source and
generator provenance, reopens the result, and enforces required text before
writing the sidecar. `environment.yml` pins ReportLab, pdfplumber and pypdf.
`.gitattributes` marks PDFs as binary; staged and raw blob IDs match, so Git
line-ending conversion cannot invalidate their recorded hashes.

The first direct-script launch failed before argument parsing because its
package import was not available in script execution mode. It produced no PDF;
an explicit launcher `ABORTED.json` is retained. A failing CLI regression test
was added before the dual execution-mode import repair. The next unique root,
`retry1`, produced a structurally clean six-page report, but its full-method-
only scale chart was empty because all 12 full-method success means were zero.
That root and provenance remain with `ABORTED.json`. A failing provenance
contract test was added before changing the chart to show both independently
trained arm means.

The eligible unique root is
`output/pdf/aamas2027_descriptive_report_20260802_retry2/`. Its PDF SHA-256 is
`C37B1C931C9031D89FAAC568428800BB1AC264ECC96C821D742F1F92C25A6F09`;
provenance SHA-256 is
`96149F4BA7815A43B64FAB9511F1221E0855559A06E06A2D37B0E7BACF163AFA`;
the generation-time builder SHA-256 is
`E86B4459C3CB90A8639334E46425815174E249579269028C62E0904B85C8CDD3`.
Its provenance repeats the three already validated input hashes and records
six scenarios, five seeds, 20 episodes per seed-scenario cell, 24 visible
3-UAV success rows, 12 visible paired scale rows, all 11 task metrics, all five
CBF metrics, full-minus-no-uncertainty direction, descriptive-only claims and
no external-submission authorization.

Independent checks reopened the PDF with pypdf and pdfplumber, matched the PDF
and builder hashes, found one 25-row table on page 3 and one 13-row table on
page 5, and confirmed one 841.89 by 595.276 point page size. `pdfinfo` reports
six pages, PDF 1.4, no encryption, no JavaScript and no form. Poppler rendered
all six pages at 144 DPI. Complete visual inspection found no clipping,
overlap, black boxes, missing glyphs or unreadable labels. Its `Symbol` and
`ArialUnicode` missing-display-font diagnostics had no visible effect and are
retained as non-blocking renderer provenance.

This closes the portable evidence-surface gate only. It does not make any
failed HTML eligible and does not close manuscript-specific figure/table
traceability, component-only causal ablations, primary-literature breadth or
the final manuscript-to-artifact integrity audit. No experiment, CBF value or
claim boundary changed.

Final verification passes Ruff, mypy over 105 source files and six focused
PDF/report/protocol tests. Full pytest is 165 passed with one unchanged legacy
MATLAB inventory failure and 190 OSQP dependency warnings. The external sibling
source path is still absent, so the test observes zero `.m` files while
asserting 81; no RL or PDF test failed.

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

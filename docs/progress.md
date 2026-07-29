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
causal claim; it neither trains/evaluates a policy nor changes CBF values.

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
nor a causal four-arm comparison. Next: define a publication-facing analysis
plan that keeps CBF diagnostics distinct and excludes every invalid root.

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

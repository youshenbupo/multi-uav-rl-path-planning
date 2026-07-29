# Known Issues

- The active restored baseline has no standalone CA-HGALO source, entrypoint,
  or CA-HGALO regression artifact. This blocks source-faithful CA-HGALO expert
  demonstration and regression until authoritative material is provided.
- MATLAB execution was not verified during phase one; existing artifacts were
  inventoried only.
- `rg.exe` could not execute in this Windows environment because access was
  denied. PowerShell recursive enumeration and `Select-String` were used for
  the audit instead.
- The MATLAB source does not formally label its shared distance unit or publish
  a Python-ready tensor schema.
- The local browser timed out loading PyTorch's interactive installation page.
  The selected wheel versions were instead verified against the official
  `download.pytorch.org/whl/cu130` index and by a local CUDA/GPU import check.
- The regression bundle has no fixed-seed CA-HGALO output because the restored
  baseline provides no CA-HGALO entrypoint. It contains an explicit unavailable
  marker rather than a substituted HGALO result.
- The phase-four evaluator intentionally evaluates supplied Cartesian paths
  without invoking the legacy optional repair or conflict-aware scheduling
  heuristics. Exact regression of those mutating, feature-gated modules remains
  a later migration task; their absence is not treated as an evaluator pass.
- Phase-five spatial repair is a deterministic local expert heuristic. It
  preserves endpoint missions and validates changed waypoints, but it does not
  claim exact path-by-path equivalence with MATLAB's optional multi-candidate,
  coupled, and cluster repair variants.
- The restored MATLAB reference contains schedule outcomes and repaired paths,
  not action-by-action schedule or spatial-repair logs. Converted MATLAB-only
  episodes therefore intentionally contain no high-level labels; use a logged
  Python coordinator replay or a future authoritative CA-HGALO log source.
- Phase seven ports the available HGALO operator surface and composes the
  independently tested Python coordinator only after optimization. It is named
  `HGALO_PYTHON_COORDINATED_RECONSTRUCTION`, not MATLAB-verified CA-HGALO;
  authoritative CA-HGALO source/output is still required for cross-language
  algorithm-level regression.
- The optimizer geometric repair is feasibility-oriented (bounds, terrain,
  cylindrical threats, and one turn pass). It does not yet reproduce every
  optional legacy corridor, segment-threat, or coupled repair branch.
- The Phase-8 environment is deliberately a kinematic single-integrator MVP.
  It has no acceleration, attitude, actuator-lag, wind, sensor-noise, or
  vehicle-specific energy model. `DynamicsModel` isolates that limitation so a
  later double-integrator replacement does not alter the PettingZoo contract.
- Environment safety costs are explicit diagnostics, not constraints enforced
  by a CBF or a constrained RL optimizer. Those later mechanisms remain outside
  the Phase-8 stop boundary.
- The Phase-9 MAPPO result is a small, fixed-seed two-UAV verification
  curriculum, including one off-route cylinder. It demonstrates the baseline
  training path but is not evidence of generalization to denser conflicts,
  active obstacle avoidance, larger teams, or unseen terrain.
- Phase-10's required 3/5/8-UAV comparison uses one matched seed and a short
  rollout-aligned budget. It is an execution/ablation check, not a statistical
  performance claim. All evaluated cells had zero task success; the 8-UAV
  distance and predictive graph cells still collided. Longer multi-seed,
  crossing-traffic and active-obstacle curricula are required before claiming a
  conflict-graph benefit.
- Phase 11 has only short 384-transition CPU smoke runs for Stage A and Stage
  B. They establish command holding, staged freezing, checkpoint restoration,
  and finite PPO updates; they do not establish a high-level coordination
  benefit, safety, task success, or generalization.
- The first hierarchical version reserves local-subgoal offsets, priority
  scores, and suggested delays as configurable zero-valued fields. Learning
  those continuous coordination outputs is intentionally deferred; no expert
  cloning or CBF shield has been added.
- Phase-12 BC training has only six authoritative high maneuver labels across
  the six verified shards. The high checkpoint is an auditable artifact, but
  this data volume cannot establish a learned high-level coordination benefit;
  its optional delay, priority, and local-subgoal heads have no authoritative
  targets in the current schema.
- The recorded Phase-12 one-epoch BC smoke run has zero held-out arrivals and
  large expert-path deviation. It validates data/mask/training/evaluation
  plumbing only, not controller quality, robustness, or safety.
- The BC-to-Graph-MAPPO smoke run has one PPO update. Encoder transfer,
  normalization import, optimizer restart, and freezing are exercised, but a
  multi-seed, adequately long fine-tuning comparison is still required. The
  comparison resolver deliberately does not fabricate results for the five
  initialization arms.
- The Phase-13 CBF is a first-order, execution-only single-integrator filter.
  It does not model acceleration, attitude, delay, uncertainty, moving threats,
  wind, or actuator limits. A slack-positive result means the stated CBF rows
  could not all be satisfied exactly; it is not a proof of recovery from an
  already unsafe state.
- Direct OSQP is intentionally CPU-side. The neural policy may train or infer
  on CUDA, but this small joint QP has no GPU execution path in the current
  implementation. Larger fleets require timing studies and possibly a sparse
  or decomposed safety-solver design before real-time claims are made.
- The Phase-14 CUDA smoke run uses a deterministic goal-directed semantic
  controller when no hierarchy checkpoint is supplied. It verifies dynamic
  environment, communication, graph, CBF, output, and GPU selection plumbing,
  but it is not an algorithm-performance measurement. Learned evaluation now
  requires an explicit compatible checkpoint.
- The complete multi-seed trained-baseline and ablation matrix has not been
  executed. In particular, no claim is made yet for expert gap or
  generalization gap because matched CA-HGALO references and paired
  within/out-of-distribution runs are unavailable.
- The unified dynamic-world benchmark currently executes only a compatible
  hierarchical `full_method` checkpoint. Existing straight-line, CA-HGALO,
  MAPPO, graph-MAPPO, and hierarchy scripts are deliberately marked unavailable
  in this unified protocol until each has a comparable dynamic-world adapter.
- A dynamic GraphMAPPO training executor exists for the raw-packet,
  predictive-without-uncertainty, and uncertainty-aware predictive graph arms.
  MLP-MAPPO now shares the dynamic environment, CBF execution path, and the
  separate core-checkpoint evaluator, but neither family is yet wired into the
  legacy public benchmark registry or executed as the five-seed protocol.
- The dynamic MLP-MAPPO adapter is now implemented and its CUDA smoke completed,
  but it exposed an evaluation-side `solved inaccurate` CBF fallback. Training
  and evaluation now record separate fallback rates, yet the five-seed CBF
  stress analysis required before any safety statement remains outstanding.
- The retained MLP evaluation fallback was replayed from its saved dynamic
  context. Increasing OSQP from 20,000 to 100,000 iterations reached the
  configured 0.1-second solve limit instead of a reliable solution. Reducing
  the slack penalty from 100 to 10 did solve the replay, but only with a large
  slack value (about 75); that setting is deliberately not adopted as a safety
  fix. Formal experiments must retain and analyze these events rather than
  silently changing the numerical penalty to reduce the fallback count.
- The planned ablation manifest is present, but its nine separately trained
  checkpoint artifacts have not yet been produced. The ablation command writes
  unavailable resolution records until an arm-specific artifact is supplied.
- The AAMAS uncertainty model currently uses a deterministic age-growth bound,
  not an empirically calibrated sensor or motion-error distribution. Its growth,
  graph risk gain, and CBF margin cap require sensitivity analysis. The CBF
  remains centralized in simulation: true geometry builds constraints while
  communication uncertainty only tightens its clearance margin.
- The first 12-transition CUDA uncertainty smoke run produced one CBF
  `maximum iterations reached` fallback in four decisions with a slack penalty
  of 1000. A persisted-event replay showed the same QP converges with penalty
  100 before the 20,000-iteration budget; the checked-in mainline configuration
  now uses 100 and a subsequent four-decision CUDA smoke had zero fallbacks.
  This is a narrow numerical regression check, not evidence of low fallback
  rate, safety, or real-time performance under trained policies. Formal
  experiments must report all retained emergency events and rates across every
  seed and scenario.
- Introducing age and uncertainty features changes local observation width from
  six to eight values per neighbour and graph edge width from 15 to 16. Existing
  BC and hierarchical checkpoints are intentionally incompatible and cannot be
  silently reused for the new method.
- The four learned comparison arms share a checkpoint evaluator.  The 3-UAV
  MLP, raw-packet GraphMAPPO, and predictive-without-uncertainty each now have
  five-seed, 30-cell matrices (20 episodes per cell).  The uncertainty-aware
  predictive graph arm now also has its five independent checkpoints, matched
  30-cell matrix, 600-record JSONL completeness evidence, and all-source CBF
  replay.  Those raw artifacts still do not license a cross-method comparison,
  statistical claim, or protocol-wide fallback claim; prespecified aggregation
  and the required 5/8-UAV independent-training and ablation work remain.
- The 5-UAV and 8-UAV uncertainty-aware predictive configuration files now
  encode the fixed scale profiles and have passed loading/tests/static checks,
  but no scale artifact exists yet.  The first load caught and corrected a
  missing 8-UAV dynamic-obstacle enablement before execution; that pre-run
  configuration defect must not be treated as a result.  Five independent
  seeds, six-scenario JSONL matrices, CBF replay, and separately trained
  critical ablations remain required at each scale before scalability or
  ablation claims.
- The first retained 5-UAV uncertainty-aware predictive seed has final
  checkpoint, summary, TensorBoard, and live telemetry, but its intended
  launcher stdout/stderr redirection files did not materialize.  This limits
  launcher-level observability only; do not regenerate those missing files or
  mistake the one seed's CBF telemetry for a scale-level fallback or safety
  result.
- The second retained 5-UAV uncertainty-aware predictive seed has complete
  summary, checkpoint, TensorBoard, live telemetry, and dedicated launcher
  logs.  Its 20,032 training CBF decisions retain three emergency fallbacks
  (two `solved inaccurate` at the unchanged 20,000-iteration cap and one
  `solve_time_limit` at the unchanged 0.1-second limit); initial and final
  training-script evaluations each retain zero in 160 decisions.  All event
  contexts remain in
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260720/`.
  These seed-local diagnostics neither establish a 5-UAV fallback rate nor a
  safety/scalability result, and they must not motivate any safety-parameter
  change.
- The third retained 5-UAV uncertainty-aware predictive seed has complete
  summary, checkpoint, TensorBoard, live telemetry, and dedicated launcher
  logs.  Its 20,032 training CBF decisions retain two `solved inaccurate`
  emergency fallbacks at the unchanged 20,000-iteration cap; initial and final
  training-script evaluations each retain zero in 160 decisions.  All raw
  contexts remain in
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260721/`.
  This seed-local record cannot establish a scale-level fallback rate, safety,
  or scalability result and does not authorize a safety-parameter change.
- The fourth retained 5-UAV uncertainty-aware predictive seed has complete
  summary, checkpoint, TensorBoard, live telemetry, and dedicated launcher
  logs.  Its 20,032 training CBF decisions retain one `maximum iterations
  reached` event and one `solved inaccurate` event, both at the unchanged
  20,000-iteration cap; initial and final training-script evaluations each
  retain zero in 160 decisions.  Full raw contexts remain in
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260722/`.
  These are seed-local diagnostics only and do not license CBF tuning or a
  safety/scalability/fallback-rate conclusion.
- The fifth retained 5-UAV uncertainty-aware predictive seed has complete
  summary, checkpoint, TensorBoard, live telemetry, and dedicated launcher
  logs.  Its final summary/telemetry retain one `solved inaccurate` event in
  20,032 training CBF decisions, while the retained stderr also contains a
  `solve_time_limit` notification that is not separately represented by the
  final telemetry.  Both sources must remain preserved and be reported as an
  event-accounting discrepancy; do not merge, impute, silently exclude, or
  tune the CBF solver in response.  The five-seed 5-UAV evaluation matrix and
  all-source CBF replay are now retained, but neither permits a scale-level
  fallback, safety, or performance claim.
- The 5-UAV uncertainty-aware predictive five-seed evaluation matrix and CBF
  replay are now retained (30 valid 20-episode cells; 600 parseable raw JSONL
  records; eight replayed source events and zero replay errors), but they are
  not evidence of a scale-level fallback rate, safety, performance, or method
  effect.  The seed-`20260723` stderr-only `solve_time_limit` notification
  remains outside replay because no distinct telemetry context exists; retain
  it explicitly rather than imputing or excluding it.  A read-only
  post-processing check after the successful replay used the wrong companion
  filename (`.jsonl.summary.json` rather than `.summary.json`) and failed;
  neither valid replay artifact was overwritten.  The independent 8-UAV
  five-seed/matrix/replay protocol and separately trained critical ablations
  remain required before any scalability or ablation conclusion.
- The first retained 8-UAV uncertainty-aware predictive seed has final
  checkpoint, summary, TensorBoard, and live telemetry, but its intended
  launcher stdout/stderr did not materialize because the parent output folder
  was absent at launch.  The retained telemetry has one `solved inaccurate`
  training fallback in 12,528 decisions; initial and final training-script
  evaluations each have zero in 160 decisions.  The complete context is under
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260719/`.
  A failed pre-launch read-only import and two subsequent read-only observability
  checks created no experiment result.  Do not regenerate the missing logs,
  treat this one seed as a scale-level result, or tune CBF parameters in
  response.
- The second retained 8-UAV uncertainty-aware predictive seed has complete
  checkpoint, summary, TensorBoard, live telemetry, and launcher logs.  Its
  final telemetry retains sixteen `solve_time_limit` events in 12,528 training
  CBF decisions, while retained stderr has 32 notification lines that do not
  map one-to-one to those events.  Preserve both raw sources under
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260720/`;
  do not merge, impute, drop, or use this single seed for a scale-level
  conclusion.  The observed numerical events do not authorize CBF tuning.
- The third retained 8-UAV uncertainty-aware predictive seed has complete
  checkpoint, summary, TensorBoard, telemetry, and launcher logs.  Its final
  telemetry retains five `solve_time_limit` events in 12,528 training CBF
  decisions, while stderr retains six notification lines.  Preserve both raw
  sources in
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260721/`
  and their count discrepancy.  This one seed is not a scale-level conclusion
  and does not authorize numerical or safety-parameter tuning.
- The fourth retained 8-UAV uncertainty-aware predictive seed has two final
  telemetry `solved inaccurate` events in 12,528 training CBF decisions but
  four stderr notification lines.  Preserve the unmerged raw sources in
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260722/`.
  This single seed does not establish a scale-level result and does not
  authorize CBF tuning.
- The final retained 8-UAV seed has one training telemetry event, nine initial
  training-script evaluation events, and ten stderr notifications that are not
  one-to-one mapped.  Preserve all sources under
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260723/`;
  do not merge/impute/drop them or tune CBF.  The five-seed 8-UAV evaluation
  matrix and all-source replay remain required before any scale conclusion.
- The completed 8-UAV five-seed evaluation matrix retains 30 valid
  20-episode cells and 600 parseable raw JSONL records, but it is not yet a
  scale-level safety, fallback-rate, performance, or method-effect result.
  Its unchanged-protocol CBF replay retains 200 source events in
  `outputs/cbf_diagnostics/uncertainty_predictive_graph_8uav_5seed_replay_20260727.jsonl`:
  116 replayed and 84 are explicit `replay_error` records because the logged
  dynamic-obstacle centers cannot be reconstructed from the event step. Do not
  impute, drop, aggregate away, or tune CBF values in response. The initial
  evaluation-launch PowerShell `-or` syntax error occurred before any evaluator
  ran; its empty-root-only attempt is retained and excluded from all counts.
- The principal no-uncertainty ablation is protocol-frozen but has no training
  artifact yet. It must be independently trained from scratch using
  `dynamic_graph_{5,8}uav_predictive_no_uncertainty_ablation.yaml`; no full-arm
  checkpoint, altered controller, or unlabelled CBF parameter change may stand
  in for it. The deliberate zero uncertainty-margin fields isolate the stated
  `kappa_CBF = 0` ablation and must never be described as solver tuning.
- The initial 5-UAV ablation evaluation launcher is invalid and excluded. It
  created only an empty nominal cell directory before rejecting the unsupported
  scenario alias `delay`; raw stderr is retained under
  `outputs/core_5uav_ablation_evaluations/`. No JSONL or summary was emitted.
  A fresh output root with canonical scenario names is required; do not
  overwrite this invalid attempt.
- The subsequent fresh-root launcher is also invalid/excluded: a PowerShell
  interpolation error omitted seed numbers from experiment names, causing
  potential cross-seed directory collisions. It was stopped after detection;
  all created directories, JSONL, stdout, and stderr under
  `outputs/core_5uav_ablation_evaluations_rerun1/` are retained and must not be
  aggregated. A corrected launcher must use an unambiguous `${seed}` boundary
  and a new output root.
- A third 5-UAV evaluation-launch attempt is invalid and excluded.  Its inline
  PowerShell quoting stripped the intended string assignment, so evaluation
  argument validation failed before a cell could run.  Preserve its launcher
  stderr under `outputs/core_5uav_ablation_evaluations_rerun2/`; it contains no
  valid cell, JSONL, or summary and must not be overwritten or aggregated.
- The currently running fourth launcher uses a retained local `.ps1` script
  with `"core_5uav_predictive_no_uncertainty_seed_${seed}_${scenario}"` and a
  fresh `outputs/core_5uav_ablation_evaluations_rerun3/` root.  It remains
  in-progress until a read-only audit establishes all 30 seed-scenario cells,
  20 raw records per cell, and parseable JSONL; do not treat partial artifacts
  as results.
- The fourth 5-UAV no-uncertainty evaluation launcher was interrupted after 24
  complete cells (all six canonical scenarios for seeds `20260719`--`20260722`)
  and before seed `20260723`; no Python process remained on the 2026-07-29
  audit. Its zero-byte launcher stderr, stdout, summaries, and JSONL under
  `outputs/core_5uav_ablation_evaluations_rerun3/` are retained, but the root
  is incomplete and excluded wholesale from aggregation and CBF replay. Start
  a fresh root for a complete matrix; do not reuse or overwrite these cells.
- The fresh rerun4 root is the only valid 5-UAV principal no-uncertainty
  evaluation matrix: it has the audited 30 cells and 600 raw records. The
  earlier evaluation roots (initial, rerun1, rerun2, and interrupted rerun3)
  remain retained and explicitly excluded. Its all-source diagnostic replay
  retains five training fallback events with zero replay errors; this does not
  justify a safety, fallback-rate, performance, scalability, or causal-ablation
  claim, and must not motivate any CBF setting change.
- The first 8-UAV principal no-uncertainty seed retains 182 training CBF
  emergency fallbacks in 12,528 decisions and 15 in its 160-decision initial
  script evaluation, despite zero in its final 160-decision script evaluation.
  Preserve all summary/telemetry/launcher sources under
  `outputs/core_8uav_ablation/`; this one seed cannot establish a fallback
  rate, safety, performance, scale, or ablation effect. Do not relax slack,
  iteration/time limits, tolerances, or any CBF setting in response.
- The second 8-UAV principal no-uncertainty seed retains 191 training emergency
  fallbacks in 12,528 decisions and 13 in its final 160-decision script
  evaluation (zero in its initial 160). Preserve all 204 event contexts and
  launcher logs under `outputs/core_8uav_ablation/`; this is not a multi-seed
  rate or method conclusion and must not trigger CBF parameter relaxation.
- The third 8-UAV principal no-uncertainty seed retains 139 training emergency
  fallbacks in 12,528 decisions, plus 13/10 in its initial/final 160-decision
  script evaluations. Preserve all 162 event contexts and launcher logs under
  `outputs/core_8uav_ablation/`; no multi-seed rate or causal conclusion is
  justified and no CBF parameter may be relaxed in response.
- The fourth 8-UAV principal no-uncertainty seed retains 218 training emergency
  fallbacks in 12,528 decisions and 5/11 in its initial/final script
  evaluations. Preserve all 234 event contexts and logs under
  `outputs/core_8uav_ablation/`; do not infer a rate or causal effect and do
  not alter slack, limits, or tolerances in response.
- The final 8-UAV principal no-uncertainty seed retains three training fallback
  contexts (one `maximum iterations reached`, two `solved inaccurate`) and
  eight initial-evaluation `solve_time_limit` contexts, although its final
  script evaluation has zero. Preserve all 11 events under
  `outputs/core_8uav_ablation/`; neither this seed nor all five unevaluated
  checkpoints establish a fallback rate, safety, performance, scale, or causal
  ablation effect. Do not alter CBF values.
- The first retained 5-UAV principal no-uncertainty ablation seed has one
  `solve_time_limit` training CBF emergency fallback in 20,016 decisions,
  while its initial and final short evaluations each have zero in 160. Preserve
  the full context under
  `outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260719/`;
  it is a single-seed diagnostic and cannot establish a fallback rate, safety,
  performance, scale, or ablation effect. Do not tune solver values in response.
- The second retained 5-UAV principal no-uncertainty seed has zero CBF emergency
  events in its training telemetry and initial/final short evaluations, but its
  launcher stderr contains one emergency-notice line. Preserve both raw sources
  under `outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260720/`;
  do not merge, impute, discard, or tune based on this discrepancy.
- The OOD scenario increases the deterministic obstacle's velocity scale and
  communication degradation. It is a controlled simulation stress condition,
  not a calibrated physical motion, sensor, or radio model.
- Initial untrained-policy evaluations can produce CBF `solve_time_limit`
  fallbacks even while subsequent training rollouts and a short trained
  diagnostic evaluation have none. These events are now persisted immediately;
  formal reports must distinguish initial, interval, and final evaluation CBF
  rates and retain all events rather than averaging them away.
- The first core checkpoint evaluator invocation exposed feature-schema and CUDA
  RNG-map-location defects before it emitted any raw episode record. Both are
  covered by regression tests and fixed for subsequent runs. The two incomplete
  evaluator directories are retained as failed attempts and must never be
  included in statistical aggregation; only the `_schemafix_rngfix_rerun2`
  nominal directory contains a valid repaired result for seed `20260719`.
- The initial OOD obstacle-speed evaluator directory is also invalid: its
  1.5x constant-velocity trajectory left the world boundary before any episode
  could start. The generator and a regression test now enforce a valid
  twenty-step horizon, and only the `_trajectoryfix_rerun1` OOD directory is
  eligible for later aggregation.
- The completed seed `20260722` retains 13 CBF emergency fallbacks during the
  initial untrained-policy evaluation and 1 during 33,344 training decisions;
  its final interval evaluation has zero fallbacks.  These seed-local diagnostic
  counts do not establish a fallback rate or safety property.  Their raw contexts
  remain in `outputs/core_3uav/core_3uav_mlp_mappo_seed_20260722/` and must be
  replayed without changing the checked-in slack penalty, iteration cap, or
  tolerance.  Its six required checkpoint-evaluation JSONL artifacts now exist,
  but this seed-level diagnostic remains ineligible for performance or safety
  claims without the matched GraphMAPPO matrices.
- The five 3-UAV MLP training checkpoints and the 30-cell uniform evaluation
  matrix now exist, but no performance comparison is available: the three
  GraphMAPPO method families have not yet received their matched independent
  five-seed training/evaluation matrices.  Training-script initial/interval/
  final evaluation fields remain separate from the 20-episode checkpoint JSONL
  records.  Formal MLP fallbacks have been replayed without changing solver
  values, but a cross-method aggregate fallback claim remains unavailable.
- The raw-packet GraphMAPPO five-seed training and 30-cell evaluation matrix
  are now retained, with 600 valid episode JSONL records.  Its seven training
  CBF emergency events replayed under the frozen 20,000-iteration/0.1-second/
  penalty-100 protocol without replay errors; two still used emergency
  fallback.  This does not establish a fallback rate, safety property, or
  method comparison, and the two remaining predictive graph arms/matrices must remain
  independently trained rather than inferred from these artifacts.
- The predictive-without-uncertainty matrix retains one OOD evaluation CBF
  `solve_time_limit` event in its raw JSONL.  JSONL-aware replay locates it but
  cannot reconstruct the complete dynamic-obstacle trajectory from the
  center-only stored event state, so it is retained as an explicit
  `replay_error` in
  `outputs/cbf_diagnostics/predictive_graph_5seed_replay_20260723_jsonlfix_rerun2.jsonl`.
  The earlier `...jsonlfix.jsonl` replay attempt omitted raw JSONL due to a
  path-expression error and is retained/excluded, not overwritten.  This
  telemetry gap must be addressed prospectively without altering current CBF
  solver values; it does not permit imputation or exclusion of the event.
- Three older diagnostic CBF events are nonreplayable because their legacy
  telemetry lacks `requested_velocities`, a complete context, or `velocities`.
  They are retained as explicit `replay_error` JSONL records in
  `outputs/cbf_diagnostics/historic_cbf_replay_20260722.jsonl`, not silently
  imputed. Future telemetry must retain the complete replay context from the
  first event without altering the frozen CBF solver protocol.

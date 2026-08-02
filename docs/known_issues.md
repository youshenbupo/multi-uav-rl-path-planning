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
- The valid 8-UAV no-uncertainty five-seed matrix and its all-source replay are
  complete, but the initial identity-render preflight failed before it ran any
  evaluator and must remain distinguished from the valid 30-cell root. The
  valid replay retains 808 events with zero replay errors. Neither these raw
  diagnostics nor one method arm alone establish a safety/fallback rate,
  performance, scale, or ablation effect; retain all raw JSONL and do not tune
  CBF in response.
- The matched 8-UAV full-method and valid no-uncertainty matrices have
  identical five-seed/six-scenario/20-episode cell coverage, but no paired
  statistical summary has yet been approved. Future aggregation must use the
  five independent trained seeds—not the 100 per-scenario episodes—as the
  inferential repetition unit, and must exclude all retained invalid roots.
- The new 8-UAV paired-summary tool intentionally reports descriptive
  seed-level values only; it does not justify significance, safety,
  performance, scalability, or causal-ablation claims. The full pytest suite
  currently has a retained unrelated legacy-audit failure: the restored source
  tree contains 83 `.m` files while `legacy_inventory.md` and
  `tests/test_legacy_audit.py` still assert 81. The paired-summary tests,
  full Ruff, and explicit-package mypy pass; do not silently edit the legacy
  inventory/test without a separate audit of the two additional files.
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
- The official IJCAI proceedings page for Liu et al.'s DHCG paper is accessible
  and its abstract has been recorded, but two browser downloads of its official
  PDF timed out without producing a local file. The paper therefore remains
  full-text-unreviewed: do not infer its timing assumptions, graph inputs,
  uncertainty handling, dynamic-obstacle setting, or safety mechanism from the
  abstract. Obtain the official PDF or an author-provided primary copy before
  asserting a distinction or novelty gap.
- The Thumiger--Deghat IEEE Control Systems Letters DOI resolves to IEEE Xplore,
  but the official page returned `Unusual Traffic Detected` (HTTP 418) before
  the abstract or PDF was accessible. It remains metadata-only; do not infer
  its communication assumptions, dynamic-obstacle model, or safety mechanism.
  Obtain an authorized IEEE or author-provided primary copy rather than
  bypassing the access restriction.
- The four-arm 3-UAV descriptive summary includes only JSONL cells selected by
  explicit arm prefixes. Two original MLP seed-20260719 directories are empty
  invalid attempts and remain excluded; their documented valid schema/RNG and
  OOD-trajectory rerun cells are the selected identities. The summary is not
  inferential and must not be used for significance, method-effect, safety,
  fallback-rate, scalability, or causal claims. Full pytest still has the
  unrelated legacy MATLAB inventory failure (83 observed files vs 81 asserted);
  do not alter that inventory/test without a separate source audit. Its five
  CBF fields are explicitly diagnostic rather than task metrics; neither their
  values nor task fields may be converted to a cross-arm safety/fallback claim
  without a separately defined analysis and all retained replay context.
- The 5-UAV full-versus-no-uncertainty descriptive summary selects only
  `outputs/core_5uav_evaluations/` and the valid ablation rerun4 root. The
  initial, rerun1, rerun2, and interrupted rerun3 ablation roots remain retained
  and excluded. The seed-level paired JSON is not inferential and its CBF fields
  must not replace or be merged with the separately retained all-source CBF
  replay; no safety, fallback-rate, scale, or causal-ablation statement follows.
- The 5-UAV and 8-UAV paired JSON summaries now isolate CBF fields in
  `cbf_diagnostic_metrics`, separate from `task_metrics`. This controls report
  structure only: it does not make the CBF records an independent safety or
  fallback-rate analysis, and the raw JSONL plus all-source replay remain the
  evidence for any future diagnostic work.
- `docs/aamas2027_analysis_plan.md` is an operational retrospective reporting
  plan, not a preregistered inferential protocol. Its five-seed aggregation and
  exclusions are mandatory for current descriptive outputs, but any p-value,
  confidence interval, superiority, safety, or causal statement requires a
  separately specified and reviewed procedure before manuscript use.
- Deng et al.'s AAMAS 2026 official three-page extended abstract is now read,
  but it is not a longer archival paper and its visual PDF rendering could not
  be completed because the local Poppler wrappers point to a missing target
  path. Text extraction and page/section checks support content review, but do
  not use this source to infer omitted communication, uncertainty, or dynamic-
  obstacle details, or to validate its claimed safety guarantees. Inspect a
  longer version if one is located before a stronger comparison.
- **Protocol-wide actor-information invalidation (2026-07-29):** a static audit
  found that the pre-repair no-communication environment exposed neighbour
  truth, current sender activity could alter delivered-neighbour visibility,
  and GraphMAPPO attention could pass a peer's private local observation or
  current activity into a receiver action. The repair has targeted tests, but
  every learned checkpoint and downstream artifact generated before it is
  protocol-ineligible: `outputs/core_3uav_evaluations/`,
  `outputs/core_5uav_evaluations/`,
  `outputs/core_5uav_ablation_evaluations_rerun4/`,
  `outputs/core_8uav_evaluations/`,
  `outputs/core_8uav_ablation_evaluations_20260729/`, all corresponding
  historical paired/multi-arm JSON, and every historical CBF replay. Preserve
  all of them unmodified as forensic evidence; exclude them from every task,
  CBF-rate, safety, scale, ablation, or manuscript aggregation. Retrain and
  evaluate into distinct post-isolation roots, beginning with five 3-UAV MLP
  seeds, before any claim.
- This protocol fix intentionally changes GraphMAPPO attention parameter
  shapes and removes neighbour-goal direction from actor edge features.
  Historical GraphMAPPO checkpoints must not be force-loaded or treated as
  compatible. The repaired no-communication graph fallback is self-only, so
  legacy smoke configurations with communication disabled remain executable
  without manufacturing neighbour state; this validates plumbing only.
- The first post-isolation 3-UAV MLP seed (`20260719`) retains three training
  CBF emergency events in 33,344 decisions (one `solved inaccurate`, two
  `maximum iterations reached`). Its all-source-for-this-seed replay at
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260719_replay_20260729.jsonl`
  has zero replay errors under the unchanged 20,000-iteration, 0.1-second,
  penalty-100, uncertainty-gain-0.5/cap-5 protocol; one replay still falls
  back. Keep all contexts and do not tune CBF values. One new seed and its
  short in-script evaluations cannot establish any performance, safety, or
  fallback-rate statement; four matched post-isolation MLP seeds and their
  six-scenario JSONL evaluations remain required.
- Post-isolation 3-UAV MLP seed `20260720` retains one training
  `maximum iterations reached` emergency fallback in 33,344 decisions; its
  source context and unchanged-protocol replay are retained at
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260720_replay_20260729.jsonl`
  with zero replay errors. The replay also falls back. Do not change slack,
  iteration, tolerance, or solve-time settings; this single seed supplies no
  performance, safety, or fallback-rate evidence.
- The first post-isolation MLP seed `20260721` directory is an invalid,
  preserved interrupted attempt. It stopped at 91,776 transitions because
  Windows denied the atomic telemetry-file replacement, has no final checkpoint
  or summary, and carries `ABORTED.json` under
  `outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260721/`.
  Its partial checkpoints, all telemetry, and eight initial-evaluation
  `solved inaccurate` fallback contexts remain raw evidence but are excluded
  from every training, evaluation, CBF, and manuscript aggregation. The
  bounded-retry telemetry repair must be used only with a fresh retry root;
  never overwrite this attempt or change CBF settings.
- The valid post-isolation `20260721_telemetryretry1` MLP training artifact
  retains eight initial-untrained-policy `solved inaccurate` fallback contexts
  (0/33,344 training and 0/103 final-evaluation events). Its all-source replay
  under unchanged values is
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260721_telemetryretry1_replay_20260729.jsonl`:
  zero replay errors, seven `maximum iterations reached`, one `solved
  inaccurate`, and eight replay emergency fallbacks. Preserve both source and
  replay records; do not treat the short initial evaluation, a single seed, or
  a replay status as a fallback-rate/safety result or tuning justification.
- Valid post-isolation MLP seed `20260722` has 13 initial-evaluation CBF
  emergency events (zero training/final-evaluation events). The retained
  all-source replay at
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260722_replay_20260729.jsonl`
  has zero replay errors but all 13 replays use fallback (ten
  `solve_time_limit`, three `maximum iterations reached`). Preserve the raw
  events and do not alter the unchanged solver protocol or infer a rate/safety
  property from this seed-local diagnostic.
- The five valid post-isolation MLP training seeds have a single all-source
  training/initial/final CBF replay at
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_5seed_training_replay_20260729.jsonl`.
  It retains 26 source events from five telemetry files and zero replay errors
  under unchanged values, but 23 replay decisions use emergency fallback. This
  is provenance only: it is not an episode-level evaluation fallback rate and
  cannot support safety, performance, significance, or baseline claims. The
  invalid pre-retry `20260721` root is excluded from this source list.
- The completed six-scenario evaluation matrix for valid post-isolation MLP
  seed `20260719` is preserved at
  `outputs/core_3uav_post_actor_isolation_evaluations/core_3uav_post_actor_isolation_mlp_mappo_seed_20260719_<scenario>/`.
  It has 20 raw JSONL episodes per canonical scenario (120 episodes total), but
  only one checkpoint has this matrix so far. Treat all per-cell summaries and
  fallback records as raw provenance, not performance, safety, fallback-rate,
  significance, baseline, or multi-seed evidence. The other four valid MLP
  checkpoints still require matched cells; do not aggregate them with any
  protocol-ineligible historical output.
- A second matched matrix, for valid post-isolation MLP seed `20260720`, is
  retained in sibling `...seed_20260720_<scenario>/` cells with six parseable
  20-episode JSONL files (120 records). It is still incomplete evidence: the
  valid retry `20260721_telemetryretry1`, `20260722`, and `20260723`
  checkpoints require the same matrix, and no aggregation may include the
  invalid pre-retry `20260721` directory or legacy protocol-ineligible outputs.
- The third matrix is associated solely with valid retry root
  `core_3uav_mlp_mappo_seed_20260721_telemetryretry1`; its output cells carry
  numeric seed `20260721` and six 20-record JSONL files. The original aborted
  `...seed_20260721/` directory remains untouched and excluded. Two valid
  checkpoints (`20260722`, `20260723`) still lack matched cells, so no
  performance, safety, fallback-rate, significance, baseline, or five-seed
  statement is available.
- Valid MLP seed `20260722` now has the same six-cell, 120-record raw matrix,
  but `20260723` is still required to complete the planned five-seed grid. Do
  not infer any aggregate outcome or tune CBF based on these four checkpoint
  matrices; all legacy/ineligible outputs and the aborted pre-retry 20260721
  root remain excluded.
- The planned 3-UAV post-isolation MLP evaluation grid is now complete (five
  valid checkpoints × six canonical scenarios × 20 episodes = 600 raw JSONL
  records) under `outputs/core_3uav_post_actor_isolation_evaluations/`. This
  completeness does not by itself constitute a computed aggregate, uncertainty
  interval, significance test, safety/fallback-rate result, or baseline
  comparison. All source fallback contexts must still be replayed and retained
  under unchanged CBF values; invalid pre-retry and historical protocol-leaking
  artifacts remain excluded.
- The all-source valid MLP replay at
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_5seed_training_and_evaluation_replay_20260729.jsonl`
  retains all 27 events from five training and 30 evaluation telemetry files,
  but one OOD evaluation event cannot be reconstructed: seed `20260719`, OOD
  communication-obstacle, episode 2, has no dynamic-obstacle centers at the
  saved event step. Its `ValueError` is retained in the replay JSONL and counts
  toward `replay_error_count=1`; it must not be omitted, counted as successful,
  or used to justify a CBF numerical change. Diagnose telemetry fidelity before
  interpreting replay rates.
- The first fresh raw GraphMAPPO seed `20260719` directory is invalid and
  preserved at
  `outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260719/`.
  A PowerShell `ErrorActionPreference=Stop` wrapper treated the routine CBF
  `solved inaccurate` stderr diagnostic as fatal at 10,848 transitions. Its
  `ABORTED.json`, partial checkpoints, telemetry, TensorBoard, and empty
  launcher stderr log must remain untouched and excluded. A retry must use a
  unique root and process-level native-output redirection; do not change CBF
  settings or overwrite the invalid root.
- Valid raw GraphMAPPO seed `20260719` is only the
  `...seed_20260719_launcherretry1/` root. It retains two training
  `solved inaccurate` emergency fallbacks in 33,344 decisions; the companion
  replay has zero errors but one emergency fallback. Preserve all source and
  replay records and do not change CBF values. One valid raw-graph seed, its
  short in-script evaluations, or the replay outcome cannot support a
  performance, safety, fallback-rate, significance, or baseline statement.
- Valid raw GraphMAPPO seed `20260720` retains one training `maximum iterations
  reached` fallback in 33,344 decisions. Its unmodified-protocol replay is
  retained with zero replay errors and a `solved` non-fallback decision. This
  single diagnostic cannot justify safety-solver tuning or any outcome claim;
  preserve it alongside the prior invalid root and valid 20260719 retry.
- Valid raw GraphMAPPO seed `20260721` retains one training `solved inaccurate`
  fallback in 33,344 decisions. Its source/replay JSONL preserves a zero-error
  replay which is `solved` without fallback under unchanged safety values. This
  remains a seed-local diagnostic, not a reason to change CBF settings or make
  an outcome claim.
- The raw-graph `mappo` sequence is intentionally paused by the user's
  2026-07-30 stop request after valid seed `20260722`. It has only four valid
  seeds (20260719 retry, 20260720, 20260721, 20260722), so seed `20260723` and
  all checkpoint evaluations remain required. The zero-event seed-20260722
  replay at
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260722_replay_20260730.jsonl`
  proves only retained zero-event diagnostic provenance—not a safety or
  fallback-rate claim. Do not aggregate this incomplete arm or launch a new
  task without resuming the documented sequence.
- SSH authentication and remote push were restored on 2026-07-30: use remote
  name `origin`, not the historical nonexistent alias `github.com`. The remote
  branch `codex/phase14-dynamic-world` was synchronized through `014c006`.
  This resolves the prior push-access blocker but does not resolve the research
  gates or authorize any external paper submission.
- A 2026-07-30 cache-only cleanup audit found only regenerable `.mypy_cache`,
  `.pytest_cache`, `.ruff_cache`, and 12 Python `__pycache__` directories
  (about 48 MB total). The environment rejected the explicit cache deletion
  command before execution. No research or project file was removed; the
  exact inventory and user-runnable cleanup command are retained in
  `docs/maintenance_cleanup_20260730.md`. Do not bypass this protection or
  delete `outputs/`, `data/`, `tmp/`, raw JSONL, aborted roots, or logs under
  the guise of cleanup.
- Raw GraphMAPPO seed `20260723` now naturally completed in the unique
  post-isolation root with 100,032 transitions and 1,042 updates on CUDA. Its
  training telemetry has zero emergency events in 33,344 decisions, and its
  short initial/final in-script evaluations also retain zero events. This is
  seed-local diagnostic provenance, not a safety, fallback-rate, performance,
  significance, or method-comparison result. Preserve its full telemetry,
  checkpoint, TensorBoard and stdout/stderr logs; create and retain the
  required zero-event replay record before its checkpoint evaluation. The
  initial non-retry raw-graph seed `20260719` remains invalid/excluded, and no
  raw-graph aggregate may be claimed until all five final checkpoints have the
  complete six-scenario JSONL matrix and all-source replay.
- The required seed-`20260723` zero-event training replay is retained at
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260723_replay_20260731.jsonl`
  and companion summary. It records `event_count=0` and
  `replay_error_count=0` under the unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP
  protocol. An empty JSONL here is an intentional diagnostic record, not an
  omitted result, and cannot support a safety or fallback-rate conclusion.
- The first post-isolation raw GraphMAPPO five-checkpoint evaluator batch is
  invalid and retained at
  `outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo/`.
  It stopped during seed `20260720` OOD after 11 complete cells plus eight raw
  records without a summary in the twelfth; 18 cells were never created.
  `ABORTED.json`, the launcher, and stdout/stderr preserve the unknown-cause
  interruption (stderr is empty). Exclude the whole root, including the 11
  valid-looking cells, from all raw-graph aggregation, CBF replay inputs and
  manuscript claims. A new-root full rerun is required; do not overwrite this
  root or change CBF/evaluator parameters in response.
- The valid raw GraphMAPPO rerun matrix is complete under
  `outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo_rerun1/`
  with 30 cells and 600 identity-checked JSONL records. Its all-source replay
  retains four events and zero replay errors; one replay remains `solved
  inaccurate` with emergency fallback. These within-arm artifacts are
  diagnostic/provenance only and cannot support performance, safety,
  fallback-rate, significance, or comparison claims. The invalid first matrix
  must remain excluded, and the two predictive GraphMAPPO arms still require
  independently trained matching five-seed matrices before cross-method use.
- Post-isolation `predictive_graph` seed `20260719` is valid at 100,032 CUDA
  transitions / 1,042 updates and retains two training CBF events. Its
  unchanged replay has zero errors but one replay still uses emergency
  fallback. Preserve the complete telemetry/logs and do not tune CBF based on
  this seed. Four matching predictive seeds plus their six-scenario matrix and
  all-source replay remain required before any outcome or cross-method claim.
- Post-isolation `predictive_graph` seed `20260720` is valid at 100,032 CUDA
  transitions / 1,042 updates. Its training telemetry has zero CBF emergency
  events in 33,344 decisions, and its initial/final short evaluations also
  retain zero events. The deliberate zero-event replay at
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260720_replay_20260731.jsonl`
  plus companion summary records `event_count=0` and
  `replay_error_count=0` under unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP
  values. This is seed-local diagnostic provenance only, not evidence of
  performance, safety, fallback rate, significance, or a comparison. Three
  further predictive seeds, their complete six-scenario matrix, and the
  all-source replay remain required before the uncertainty-aware arm or any
  cross-method use.
- Post-isolation `predictive_graph` seed `20260721` is valid at 100,032 CUDA
  transitions / 1,042 updates. Training retains zero CBF emergency events in
  33,344 decisions, while initial/final short evaluations also retain zero in
  160/160 decisions. Its append-safe zero-event replay at
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260721_replay_20260731.jsonl`
  and companion summary records `event_count=0` and
  `replay_error_count=0` under unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP
  values. This is seed-local diagnostic provenance only and cannot support a
  performance, safety, fallback-rate, significance, or comparison claim. Two
  matching predictive seeds, the complete six-scenario matrix, and all-source
  replay remain required before the uncertainty-aware arm or cross-method use.
- The first post-isolation `predictive_graph` seed-`20260722` attempt is
  invalid and permanently retained at
  `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260722/`.
  It stopped at 84,672 transitions after the 82,944 checkpoint and has no
  final checkpoint or summary. Both launcher logs are empty, so the cause is
  unknown; a neighboring monitor invocation returned Windows code
  `1073807364` (`0x40010004`) but does not prove causality. Its partial
  telemetry and checkpoints remain under `ABORTED.json` exclusion and must not
  enter CBF replay, evaluation, aggregation, statistics, or manuscript claims.
  Retry the same numeric seed only in a distinct `launcherretry1` root with
  unchanged training and CBF settings.
- The intended `predictive_graph` seed-`20260722_launcherretry1` launch is a
  separate invalid prelaunch attempt: `Start-Process` encountered duplicate
  case-distinct `Path`/`PATH` keys in the launcher process environment before
  creating Python. Its zero-byte stdout/stderr and sidecar
  `outputs/core_3uav_post_actor_isolation_predictive_graph_seed_20260722_launcherretry1.ABORTED.json`
  are preserved and excluded. `-UseNewEnvironment` reproduced the same error.
  A process-local, nonpersistent merge to one de-duplicated `Path` was verified
  first with `cmd.exe` exit 0 and then with the dedicated Conda environment:
  PyTorch `2.13.0+cu130`, CUDA true, RTX 5060, empty stderr. This is a launcher
  environment repair only and did not change repository, experiment, or CBF
  configuration.
- Only post-isolation `predictive_graph` seed-`20260722_launcherretry2` is
  valid. It naturally reached 100,032 CUDA transitions / 1,042 updates, with
  zero CBF emergency events over 33,344 training, 160 initial-evaluation and
  126 final-evaluation decisions. Its zero-event replay at
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260722_launcherretry2_replay_20260801.jsonl`
  plus companion summary records `event_count=0`,
  `replay_error_count=0` under unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP
  settings. Preserve/exclude the original root and retry1 prelaunch attempt;
  this valid seed remains provenance only. Final predictive seed `20260723`,
  the six-scenario matrix and all-source replay remain required.
- Post-isolation `predictive_graph` seed `20260723` is valid at 100,032 CUDA
  transitions / 1,042 updates. Its training telemetry retains one `maximum
  iterations reached` fallback in 33,344 decisions, while initial/final short
  evaluations retain zero in 160/103 decisions. The unchanged-protocol replay
  at
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260723_replay_20260801.jsonl`
  plus companion summary records one source event, zero replay errors, and a
  `solved` replay without fallback. This status change under identical numeric
  settings is diagnostic variability, not permission to tune the CBF and not a
  safety or fallback-rate conclusion. The predictive five-seed training set is
  complete, but all six scenarios for all five final checkpoints and their
  all-source replay remain required. The invalid original seed-20260722 root
  and retry1 prelaunch attempt remain permanently excluded; do not start the
  uncertainty-aware arm before the predictive evaluation/replay gate closes.
- The post-isolation `predictive_graph` arm now has five eligible trainings and
  a complete valid evaluation root at
  `outputs/core_3uav_post_actor_isolation_evaluations_20260801_predictive_graph/`:
  30 identity-checked cells and 600 raw JSONL records, with 11,223 evaluation
  CBF decisions and zero recorded evaluation fallbacks. Its 35-input all-source
  replay at
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_5seed_training_and_evaluation_replay_20260801.jsonl`
  retains three training source events, zero replay errors, and one
  `solved inaccurate` replay that still uses emergency fallback. These are
  within-arm provenance facts, not performance, safety, fallback-rate,
  significance, robustness, or comparison claims. Preserve all invalid
  seed-20260722 attempts outside the eligible input set. The matched
  `uncertainty_predictive_graph` five-seed training/evaluation/replay arm is
  still required before any cross-method use, followed by prescribed 5/8-UAV
  and independently trained ablation evidence.
- The first post-isolation `uncertainty_predictive_graph` seed-`20260719`
  attempt is invalid and permanently retained at
  `outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260719/`.
  Its process exited with Windows code `0xC000013A` at 78,624 transitions,
  after the last complete 76,800-step checkpoint, with empty stdout/stderr and
  no final checkpoint or summary. The issuing cause is unknown; do not infer
  one from the interrupt-style code or adjacent read-only monitoring. Partial
  telemetry has zero events over 26,208 training, 160 initial, and 71 latest
  interval CBF decisions, but `ABORTED.json` excludes the entire root from
  replay, evaluation, aggregation, statistics, and claims. Do not overwrite,
  resume, or delete it. Retry the numeric seed only in a unique
  `launcherretry1` root under identical training and CBF settings before
  continuing the remaining uncertainty-aware seeds.
- The first `uncertainty_predictive_graph` seed-`20260719` retry is also
  invalid and retained in the distinct `launcherretry1` root. It exited with
  the same Windows `STATUS_CONTROL_C_EXIT` (`0xC000013A`) at 13,344
  transitions, with last complete checkpoint 12,288, empty stdout/stderr, and
  no final checkpoint/summary. Its partial zero-event telemetry is excluded by
  its own `ABORTED.json` and cannot enter replay or any result. Local Windows
  headers confirm the exit-code name; event logs show no Python/CUDA/NVIDIA or
  application error, and repository code has no self-signal path. Generic
  90-second PowerShell and 240-second Conda+CUDA lifetime probes both survived
  multiple parallel shell checks and exited 0, so the exact external control
  event source remains unresolved. Preserve both invalid training roots and
  the retained probe artifacts. Test only one fresh `launcherretry2` while
  issuing no parallel shell command; if it repeats, stop retrying and request
  direction rather than tuning or changing the research protocol.
- The fresh `uncertainty_predictive_graph` seed-`20260719_launcherretry2` run is
  valid: it naturally completed 100,032 CUDA transitions / 1,042 updates when
  the active execution cell was monitored only through `wait`, with no parallel
  shell processes. It has zero emergency events over 33,344 training, 160
  initial and 96 final CBF decisions, empty stderr, and a retained zero-event
  replay at
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260719_launcherretry2_replay_20260801.jsonl`.
  This establishes a safe operational mitigation, not the exact source of the
  earlier external `STATUS_CONTROL_C_EXIT` signals. For subsequent long
  trainings, do not start any parallel shell health check while the launch cell
  is active; use only its wait handle, then audit after exit. Preserve/exclude
  the non-retry and retry1 roots. Four further valid uncertainty-aware seeds,
  their complete six-scenario matrix and all-source replay are still required
  before any outcome or cross-method claim.
- Post-isolation `uncertainty_predictive_graph` seed `20260720` is valid at
  100,032 CUDA transitions / 1,042 updates. Training, initial and final CBF
  records are zero-event over 33,344, 160 and 131 decisions; stderr is empty.
  Its retained zero-event replay at
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260720_replay_20260801.jsonl`
  plus summary records `event_count=0`, `replay_error_count=0` under unchanged
  values. It completed under the wait-only active-run discipline, which must
  remain in force. This is seed-local provenance only. Three further valid
  seeds, the full six-scenario matrix and all-source replay remain required.
- Post-isolation `uncertainty_predictive_graph` seed `20260721` is valid at
  100,032 CUDA transitions / 1,042 updates. Training retains one `maximum
  iterations reached` fallback in 33,344 decisions; initial/final records are
  0/160 and 0/160. Its unchanged-protocol replay at
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260721_replay_20260801.jsonl`
  has one source, zero errors, and replays `solved` without fallback. This
  diagnostic variability is not permission to tune CBF and is not an outcome
  claim. Two further seeds plus the complete evaluation/replay gate remain.
- Post-isolation `uncertainty_predictive_graph` seed `20260722` is valid at
  100,032 CUDA transitions / 1,042 updates. Training retains one `solved
  inaccurate` fallback in 33,344 decisions; initial/final are 0/160 and 0/147.
  Its replay at
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260722_replay_20260801.jsonl`
  has one source, zero errors, and remains `solved inaccurate` with emergency
  fallback under unchanged values. Preserve this raw/replay evidence; do not
  tune the CBF. Final seed `20260723`, the 30-cell evaluation matrix and
  all-source replay remain required.
- Post-isolation `uncertainty_predictive_graph` seed `20260723` is valid at
  100,032 CUDA transitions / 1,042 updates. Training retains one `solved
  inaccurate` fallback in 33,344 decisions; initial/final are 0/160 and 0/160.
  Its unchanged replay at
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260723_replay_20260801.jsonl`
  has one source, zero errors and replays `solved` without fallback. Five valid
  trainings now exist (`20260719_launcherretry2`, `20260720`--`20260723`), but
  the arm still lacks its 30-cell evaluation matrix and all-source replay.
  Preserve both invalid seed-20260719 attempts and keep all claims gated.
- The post-isolation three-UAV uncertainty-aware arm now has a valid 30-cell,
  600-record evaluation root at
  `outputs/core_3uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`.
  Its evaluation telemetry has 10,855 decisions and zero emergency events. The
  35-input all-source replay at
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5seed_training_and_evaluation_replay_20260801.jsonl`
  retains three training events, zero replay errors and one replay fallback.
  This closes within-arm provenance, not an outcome, safety, fallback-rate or
  superiority claim. Both interrupted seed-20260719 roots remain excluded.
  Before new scale/ablation work, audit existing 5/8-UAV artifacts against the
  post-isolation actor-information boundary and independent-training rule;
  never reuse stale or mismatched evidence merely because directories exist.
- Read-only post-3-UAV audit confirms every existing 5/8-UAV full-method,
  no-uncertainty ablation, evaluation, paired summary and CBF replay artifact
  predates actor-information isolation and remains protocol-ineligible. No
  post-isolation 5/8-UAV output root currently exists. Frozen full and ablation
  config hashes still match their documented values, so the missing evidence
  must be regenerated from scratch in distinct roots; it cannot be repaired by
  re-aggregation. Required order is 5-UAV full five seeds/evaluation/replay,
  independently trained 5-UAV no-uncertainty five seeds/evaluation/replay, then
  the matching two 8-UAV arms. Preserve all historical roots and invalid
  attempts, and apply wait-only monitoring to long active jobs.
- The first valid post-isolation 5-UAV full-method seed is `20260719` under
  `outputs/core_5uav_post_actor_isolation/`, complete at 100,160 CUDA
  transitions / 313 updates. Training retains two `solved inaccurate`
  fallbacks in 20,032 decisions; initial/final are 0/160 and 0/160. Its replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260719_replay_20260801.jsonl`
  has two sources, zero errors, one replay fallback and one solved replay under
  frozen values. Do not infer a scale or safety result. Four further full seeds,
  their matrix/replay, and the independently trained no-uncertainty arm remain.
- Post-isolation 5-UAV full seed `20260720` is valid at 100,160 CUDA
  transitions / 313 updates. It has one `solved inaccurate` training fallback
  in 20,032 decisions, zero initial/final events, and a one-event replay that
  remains fallback with zero replay errors under frozen values. This is not a
  scale/safety claim. Three further full seeds and all downstream gates remain.
- The naturally completed post-isolation 5-UAV full seed `20260721` root is
  nevertheless invalid for audit and manuscript use. Its stderr contains five
  fallback notices (one `maximum iterations reached`, four
  `solve_time_limit`), while pre-fix telemetry retains only the one training
  event and the final zero-event interval evaluation. Repeated writes had
  overwritten four earlier interval-evaluation contexts. `ABORTED.json`
  excludes the root, checkpoint and all derived uses; do not replay the one
  retained event as a complete source set. The prospective append-only
  `interval_evaluations` repair has test-first coverage and does not recover
  lost contexts or change any CBF value. Rerun the numeric seed only from
  scratch in a new `intervaltelemetryretry1` root, then require stderr/event
  accounting agreement before eligibility.
- The current full pytest result after the interval-telemetry repair is 159
  passed and one unrelated legacy audit failure. The test still searches the
  absent `legacy_hgalo/HGALO_恢复源码` directory and therefore observes zero
  MATLAB files, whereas the current workspace directory is
  `legacy_hgalo/HGALO_code`; it expects 81. This supersedes the older observed
  83-versus-81 description for the present workspace state. Do not change the
  legacy inventory/test as part of the AAMAS telemetry repair; audit that rename
  separately.
- Only the post-isolation 5-UAV full-method
  `seed_20260721_intervaltelemetryretry1` root is eligible for numeric seed
  `20260721`. It completed at 100,160 CUDA transitions / 313 updates after the
  prospective fix and retains all six interval evaluations. Training has one
  `maximum iterations reached` event in 20,032 decisions; all interval,
  initial and final evaluations are zero-event, and the one retained source
  exactly matches the sole stderr warning. Its unchanged-protocol replay has
  zero errors and remains an emergency fallback. This does not establish a
  scale-level fallback rate or safety result and does not permit solver tuning.
  The non-retry naturally completed root remains permanently excluded because
  four of its contexts were lost.
- Post-isolation 5-UAV full seed `20260722` is eligible at 100,160 CUDA
  transitions / 313 updates. Its training, initial and final CBF streams are
  zero-event, while the append-only interval history retains one `solved
  inaccurate` fallback at transition 61,440. This source exactly matches the
  sole stderr notice and is recursively replayable; the unchanged replay solves
  without fallback but uses about 263.39 slack. Preserve the source/replay
  difference and do not use it for tuning or a safety/fallback-rate conclusion.
- Post-isolation 5-UAV full seed `20260723` is eligible at 100,160 CUDA
  transitions / 313 updates with zero events across training, all six retained
  intervals, initial/final evaluations and stderr. Its required zero-event
  replay is retained. The five eligible full-method seeds are now complete,
  but no scale/performance/safety conclusion is available before their complete
  30-cell matrix and all-source replay. Pre-fix eligible seeds 20260719/20 lack
  the new full interval history, but their retained training-event counts
  exactly equal their stderr notices (2/2 and 1/1) with zero events in the
  retained evaluation fields; no hidden fallback discrepancy is observed.
- The post-isolation 5-UAV full-method arm now has a valid 30-cell/600-record
  evaluation root and a 35-input all-source replay with five source events,
  zero replay errors and three replay fallbacks under frozen values. Evaluation
  contributes 12,000 decisions and zero events. These are eligible provenance,
  not a scale, safety, fallback-rate, performance or method-effect result. The
  independently trained post-isolation 5-UAV no-uncertainty arm is still
  incomplete, so no 5-UAV ablation comparison or paired statistic is available.
- The first independently trained post-isolation 5-UAV no-uncertainty seed is
  now valid at 100,080 CUDA transitions / 417 updates. Seed `20260719` retains
  two `solved inaccurate` and one `maximum iterations reached` training events
  over 20,016 decisions; initial/final evaluations and all six append-only
  interval evaluations are zero-event. The three source events exactly match
  stderr. Its unchanged 20,000/0.1s/100/0.0/0.0 replay has zero errors: one
  event remains `solved inaccurate` with fallback and two solve without
  fallback. Preserve the source/replay variability and do not tune it away.
  This one seed does not establish an ablation, scale, safety, fallback-rate,
  performance or generalization result; four further independently trained
  seeds, the 30-cell matrix and all-source replay remain required.
- Post-isolation 5-UAV no-uncertainty seed `20260720` is valid at 100,080 CUDA
  transitions / 417 updates with zero CBF emergency events across training,
  initial/final evaluations, all six retained interval evaluations and stderr.
  Its required zero-event replay has zero events/errors under frozen
  20,000/0.1s/100/0.0/0.0 values. An earlier orchestration call timed out after
  one second before Python child creation and produced no root/log; its distinct
  sidecar `...seed_20260720_prelaunch_timeout_20260801.ABORTED.json` is retained
  and excluded. Do not count that prelaunch failure as a training seed or hide
  it. Two valid seeds still establish no ablation, scale, safety, fallback-rate,
  performance or generalization result; seeds 21--23, the matrix and all-source
  replay remain required.
- Post-isolation 5-UAV no-uncertainty seed `20260721` is valid at 100,080 CUDA
  transitions / 417 updates. Training retains one `solve_time_limit` and one
  `maximum iterations reached` event over 20,016 decisions; initial/final and
  all six interval evaluations are zero-event, and the two sources exactly
  match stderr. Its unchanged replay has zero errors: the time-limit event
  solves without fallback, while maximum-iterations remains fallback. Preserve
  this variability without tuning. Three valid seeds still do not establish an
  ablation, scale, safety, fallback-rate, performance or generalization result;
  seeds 22/23, the matrix and all-source replay remain required.
- Post-isolation 5-UAV no-uncertainty seed `20260722` is valid at 100,080 CUDA
  transitions / 417 updates. Training, final evaluation and all six retained
  interval evaluations are zero-event; initial evaluation retains eight
  `solve_time_limit` events over 160 decisions, exactly matching the eight
  stderr notices. The unchanged-protocol replay has eight sources and zero
  errors: four replay `solved` without fallback and four remain
  `solve_time_limit` fallbacks. This nondeterministic replay split is retained
  as diagnostic provenance and is not permission to change CBF values. Four
  valid seeds still establish no ablation, scale, safety, fallback-rate,
  performance or generalization result; seed 23, the matrix and all-source
  replay remain required.
- Post-isolation 5-UAV no-uncertainty seed `20260723` is valid at 100,080 CUDA
  transitions / 417 updates despite the execution wrapper reporting code 1.
  This is a PowerShell 5 stderr-redirection artifact: the log's CBF warnings
  are wrapped as `NativeCommandError`, and a Python `sys.exit(0)` probe
  reproduced host code 1 whenever native stderr was redirected. Complete
  parseable stdout, final summary/checkpoint/telemetry, absent `ABORTED.json`,
  identities and counts independently prove normal completion. Training has
  one `solved inaccurate` and one `maximum iterations reached` event; all
  evaluation streams are zero-event. Its two-source replay has zero errors:
  solved-inaccurate becomes solved/non-fallback, while maximum-iterations
  remains fallback. Preserve the wrapper evidence and do not relaunch or tune.
  Five eligible no-uncertainty trainings now exist, but no method, scale,
  safety, fallback-rate, performance or generalization claim is allowed before
  the 30-cell matrix and 35-input replay are complete.
- The post-isolation 5-UAV independent no-uncertainty arm now has a valid
  30-cell/600-record evaluation root at
  `outputs/core_5uav_post_actor_isolation_evaluations_20260801_predictive_graph/`.
  All identities, episode indices, five-UAV/CUDA/actor-boundary/CPU-OSQP fields,
  30 zero exits and empty stderr files pass audit. Evaluation contains 11,680
  CBF decisions and zero emergency events. Its 35-input all-source replay has
  15 training sources, zero errors and three replay fallbacks under frozen
  values. This completes within-arm provenance only; it is not a causal,
  scale, safety, fallback-rate, performance or generalization result. A fresh
  prespecified paired/descriptive analysis is still absent, and historical
  paired summaries remain excluded. Post-isolation 8-UAV full and independent
  no-uncertainty arms remain to be regenerated from scratch.
- The first post-isolation 8-UAV full-method seed, `20260719`, is valid at
  100,224 CUDA transitions / 261 updates. Training retains two
  maximum-iterations and two solved-inaccurate events; append-only interval
  telemetry retains one additional time-limit event at transition 55,296.
  Initial/final and the other 31 intervals are zero-event, and all five sources
  exactly match stderr. Its unchanged-protocol replay has five sources, zero
  errors and two remaining maximum-iterations fallbacks; the other three solve
  without fallback. Preserve this diagnostic variability without tuning. One
  seed establishes no scale, safety, fallback-rate, performance or
  generalization result; four further full-method seeds, their matrix/replay,
  and the independent no-uncertainty arm remain required.
- Post-isolation 8-UAV full-method seed `20260720` is valid at 100,224 CUDA
  transitions / 261 updates. Training retains one maximum-iterations event;
  interval transition 92,160 retains one solved-inaccurate event. Initial,
  final and the other 31 intervals are zero-event, and both sources match
  stderr. Its unchanged replay has zero errors: maximum iterations remains a
  fallback while solved-inaccurate solves without fallback. Preserve the
  source/replay difference and do not tune. Two seeds establish no scale,
  safety, fallback-rate, performance or generalization result; seeds 21--23,
  the matrix/replay and independent no-uncertainty arm remain required.
- Post-isolation 8-UAV full-method seed `20260721` is valid at 100,224 CUDA
  transitions / 261 updates. Seven sources are fully retained and match stderr:
  one training solved-inaccurate, one initial time-limit, one interval
  solved-inaccurate at 6,144 and four interval maximum-iterations events at
  9,216. Its unchanged replay has zero errors; the first three solve without
  fallback and the four maximum-iterations sources remain fallbacks. Preserve
  this cluster without tuning. Three seeds establish no scale, safety,
  fallback-rate, performance or generalization result; seeds 22/23, the
  matrix/replay and independent no-uncertainty arm remain required.
- Post-isolation 8-UAV full-method seed `20260722` is valid at 100,224 CUDA
  transitions / 261 updates. Training, all intervals and final evaluation are
  zero-event; initial evaluation retains one time-limit event matching stderr.
  Its unchanged one-source replay has zero errors and solves without fallback.
  Preserve the diagnostic difference without tuning. Four seeds establish no
  scale, safety, fallback-rate, performance or generalization result; final
  seed 23, the matrix/replay and independent no-uncertainty arm remain.
- Post-isolation 8-UAV full-method seed `20260723` is valid at 100,224 CUDA
  transitions / 261 updates. Training retains one solved-inaccurate event;
  initial/final and all intervals are zero-event, and the source matches stderr.
  Its unchanged one-source replay has zero errors and remains a
  solved-inaccurate fallback. Five eligible full-method trainings now exist,
  but this establishes no scale, safety, fallback-rate, performance or
  generalization result. The 30-cell matrix/all-source replay and independently
  trained no-uncertainty arm remain required.
- The post-isolation 8-UAV full-method arm now has a valid 30-cell/600-record
  evaluation root at
  `outputs/core_8uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`.
  All identities, 30 zero exits and empty stderr files pass audit; evaluation
  contains 12,000 CBF decisions and zero events. Its 35-input all-source replay
  has 16 training sources, zero errors and ten fallbacks under frozen values.
  The all-source replay status mix differs from separate per-seed replays,
  demonstrating timing/order sensitivity of diagnostic OSQP reruns; preserve
  both sets and do not tune or select between them. This completes within-arm
  provenance only, not a scale, safety, fallback-rate, performance or
  generalization result. The independently trained post-isolation 8-UAV
  no-uncertainty arm remains entirely missing.
- The first independently trained post-isolation 8-UAV no-uncertainty seed,
  `20260719`, is valid at 100,224 CUDA transitions / 261 updates. Training
  retains one solved-inaccurate event; initial/final and all 32 intervals are
  zero-event, and the source matches stderr. Its unchanged 0.0/0.0-margin
  replay has one source, zero errors and solves without fallback. This is
  seed-local provenance only, not a causal, scale, safety, fallback-rate,
  performance or generalization result. Four further seeds, the matrix and
  all-source replay remain required.
- Post-isolation 8-UAV no-uncertainty seed `20260720` is valid at 100,224 CUDA
  transitions / 261 updates, but retains a large seed-local CBF numerical
  cluster. Training has 275 events over 12,528 decisions (273 time-limit, one
  solved-inaccurate and one maximum-iterations), initial/final evaluations have
  11/14 time-limit events, and 24 of 32 append-only interval evaluations retain
  another 157 time-limit events. All 457 contexts exactly match stderr and are
  replayed with zero errors under unchanged 20,000/0.1s/100/0.0/0.0 values;
  339 replays still use emergency fallback. Preserve every raw source and
  replay status without tuning, selective reporting or interpreting this one
  seed as a causal, scale, safety, fallback-rate, performance or generalization
  result. Seeds `20260721`--`20260723`, the valid 30-cell matrix and 35-input
  replay remain required.
- Post-isolation 8-UAV no-uncertainty seed `20260721` is valid at 100,224 CUDA
  transitions / 261 updates. Its 446 fully retained `solve_time_limit` sources
  comprise 271 training, 11 initial, 156 append-only interval and eight final
  events, exactly matching stderr. The first replay is invalid/excluded because
  recursive discovery counted the four final-interval mappings again through
  the equal `last_interval_evaluation_cbf` compatibility alias, yielding 450
  records. Its JSONL/summary and explicit `ABORTED.json` sidecar remain
  preserved. A test-first replay-tool fix skips only an exactly matching alias;
  the fresh `aliasdedup_rerun1` replay retains 446 unique pointers, zero errors
  and 313 replay fallbacks under unchanged 20,000/0.1s/100/0.0/0.0 values.
  Preserve both attempts and use only the corrected replay. This seed and its
  numerical cluster prove no causal, scale, safety, fallback-rate, performance
  or generalization result. Seeds `20260722`/`20260723`, the valid matrix and
  35-input replay remain required.
- Post-isolation 8-UAV no-uncertainty seed `20260722` has one eligible
  independent training, the `retry2` root, at 100,224 CUDA transitions / 261
  updates. Two earlier launcher attempts remain preserved and excluded. The
  first `Start-Process` call failed before Python child creation because the
  inherited Windows environment exposed case-colliding `Path`/`PATH` keys; it
  left two zero-byte logs and an explicit launcher `ABORTED` sidecar but no
  training root. The subsequent `retry1` direct-PowerShell wrapper was
  interrupted after an expected CBF stderr notice was promoted to terminating
  `NativeCommandError`; its partial root, zero-byte logs and `ABORTED.json`
  remain in place and are ineligible. The valid `retry2` used a transparent
  `cmd.exe` redirection wrapper without changing the Python command or protocol.
  It retains 429 time-limit sources: 306 training, 14 initial, 93 append-only
  interval and 16 final, exactly matching stderr. Its frozen
  20,000/0.1s/100/0.0/0.0 replay has 429 unique pointers, zero errors, 124
  solved records and 305 time-limit fallbacks. Preserve all three attempts and
  do not tune or select away the numerical cluster. Four valid seeds establish
  no causal, scale, safety, fallback-rate, performance or generalization
  result; seed `20260723`, the valid matrix and 35-input replay remain required.
- Post-isolation 8-UAV no-uncertainty seed `20260723` is valid at 100,224 CUDA
  transitions / 261 updates. Its 457 fully retained sources comprise 301
  training, 13 initial, 137 append-only interval and six final events; 456 are
  `solve_time_limit` and one is `maximum iterations reached`, exactly matching
  stderr. Its unchanged 20,000/0.1s/100/0.0/0.0 replay has 457 unique pointers
  and zero errors. Replay statuses are 129 solved, 326 time-limit, one
  solved-inaccurate and one maximum-iterations; 328 use emergency fallback.
  This finishes five eligible independent trainings but proves no causal,
  scale, safety, fallback-rate, performance or generalization result. A fresh
  30-cell/600-record evaluation matrix and 35-input all-source replay remain
  mandatory before any paired or descriptive analysis.
- The post-isolation 8-UAV independent no-uncertainty arm now has a valid
  30-cell/600-record evaluation root at
  `outputs/core_8uav_post_actor_isolation_evaluations_20260801_predictive_graph/`.
  All cell identities, episode indices, eligible checkpoints, completed step,
  eight-UAV/CUDA/actor-boundary/CPU-OSQP fields, 30 zero exits and empty stderr
  files pass audit. Evaluation contains 11,337 actual CBF decisions and zero
  emergency events. Its frozen 35-input all-source replay retains 1,790
  training sources, zero errors and 1,330 replay fallbacks. Separate per-seed
  replays sum to 1,285 fallbacks, so the 45-event difference is retained as
  diagnostic OSQP timing/order sensitivity; neither result may be selected or
  tuned away. This closes within-arm provenance only, not a causal, scale,
  safety, fallback-rate, performance or generalization claim. A fresh
  prespecified paired/descriptive analysis remains absent, and every historical
  paired summary remains excluded.
- The post-isolation descriptive aggregation was frozen before output
  generation at revision `67b40e0`, but it has only five independent trained
  seeds per arm. The planned artifacts therefore report seed-level means,
  sample standard deviations and, for 5/8 UAV, same-seed paired deltas only.
  They deliberately provide no p-values, confidence intervals, rankings or
  significance decisions and cannot support superiority, causality, safety or
  robust-generalization claims. All six scenarios and all supported metrics
  must be retained without post-hoc selection.

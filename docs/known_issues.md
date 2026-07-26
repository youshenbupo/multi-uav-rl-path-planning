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

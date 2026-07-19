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
- The planned ablation manifest is present, but its nine separately trained
  checkpoint artifacts have not yet been produced. The ablation command writes
  unavailable resolution records until an arm-specific artifact is supplied.

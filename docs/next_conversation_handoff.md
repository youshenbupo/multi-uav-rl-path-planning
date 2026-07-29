# AAMAS 2027 continuation handoff

**Snapshot date:** 2026-07-22.  This document is the authoritative starting
point for the next Codex conversation.  Inspect the worktree and running
processes before relying on any status below.

## Latest continuation update (2026-07-27, supersedes stale status below)

- **Actor-information protocol repair, 2026-07-29 (parent revision
  `55abb87`; commit/push still required):** a static audit found three
  violations of the mandatory actor boundary: (1) communication-disabled local
  observations exposed neighbour truth, (2) current sender activity could hide
  an already delivered packet, and (3) GraphMAPPO attention passed peer private
  local observations/current activity into receiver actions. The TDD repairs
  are in `multiuav/envs/communication.py`, `multi_uav_env.py`,
  `observations.py`, `learning/conflict_graph.py`, `graph_networks.py`, and
  `graph_runner.py`. Actor edges now contain only own truth plus delivered
  packets; delivered information survives until staleness expiry; peer node
  features/activity cannot change a receiver action; no-communication is
  self-only. Neighbour-goal direction was removed from actor edges. Critic/CBF
  interfaces and all CBF numerical parameters are unchanged; no training or
  evaluation was launched and no raw artifact was deleted.

  Red tests were observed for the no-channel leak, packet-erasure leak,
  Graph Actor peer-feature leak, Graph Actor peer-activity leak, and
  graph-edge peer-activity leak. Green verification:
  `D:\anaconda3\envs\multiuav_rl\python.exe -m pytest -q
  tests\test_predictive_conflict_graph.py tests\test_dynamic_graph_baselines.py
  tests\test_multi_uav_environment.py tests\test_communication.py` -> 39
  passed (three existing OSQP warnings); Ruff passed; explicit-package mypy
  passed 103 sources. The completed full pytest run is 157 passed / 1 retained unrelated legacy-inventory
  failure (83 observed MATLAB files versus 81 asserted) / 28 existing OSQP
  warnings; do not alter that inventory/test without a separate audit.

  This invalidates all pre-repair learned checkpoints and their downstream
  results, regardless of apparent JSONL completeness: historical
  `outputs/core_3uav_evaluations/`, `core_5uav_evaluations/`,
  `core_5uav_ablation_evaluations_rerun4/`, `core_8uav_evaluations/`,
  `core_8uav_ablation_evaluations_20260729/`, every paired/multi-arm summary,
  and all historical CBF replay JSONL. Retain them but exclude them from every
  analysis and paper claim, as documented in
  `docs/aamas2027_analysis_plan.md` and `docs/known_issues.md`. Existing Graph
  checkpoints are architecturally incompatible after the repair and must not
  be force-loaded. Next: run full pytest, commit/push while excluding the
  user-owned `.gitignore`, then start **new-root**, serial 3-UAV MLP five-seed
  100k CUDA training; only after valid MLP reruns/evaluations begin the three
  GraphMAPPO matching five-seed reruns.

- Neural Graph CBF is no longer metadata-only: the official AAMAS 2026
  proceedings entry and three-page extended abstract PDF were reviewed on
  2026-07-29 (SHA-256
  `0AE5CA03242EDE2EB5E60822729EE6B6279F4FA4C14B836E411658260E09D29A`). It
  uses a frozen MAPPO reference plus GNN graph-CBF additive correction trained
  with pointwise QP supervision and barrier losses on modified MPE boundary /
  collision constraints. It does not document the project mainline's packet
  staleness/delay/loss, predicted-packet uncertainty, or dynamic obstacles.
  Do not turn its abstract-level claims into a universal guarantee or a novelty
  statement; seek a longer version for a stronger comparison. Local Poppler
  rendering was unavailable, but the full text and page sections of the
  extended abstract were verified.

- Added `docs/aamas2027_analysis_plan.md` at revision `019065b`. It is the
  operational source of truth for eligible 3/5/8-UAV reporting inputs,
  invalid-root exclusions, five-seed aggregation, task/CBF separation,
  regeneration commands, CBF replay provenance, and remaining manuscript gates.
  It is retrospective and descriptive only, not a preregistration or a claim.
  Use it to audit future manuscript tables/figures before interpretation; next
  continue primary-source full-text retrieval without relaxing its evidence
  boundaries.

- Refined `scripts/summarize_paired_checkpoint_evaluations.py` so the retained
  5-UAV and 8-UAV paired outputs now split 11 `task_metrics` from five
  `cbf_diagnostic_metrics` per scenario. Their inputs, five trained seeds,
  six scenarios, 20 episodes/cell, invalid-root exclusions, and separate
  all-source CBF replays are unchanged. Target tests (6), Ruff, and mypy (103
  sources) pass; full pytest remains 152 passed / 1 unrelated legacy MATLAB
  inventory failure (83 vs 81) / 28 OSQP warnings. This is a reporting-boundary
  change, not a performance, safety, fallback-rate, scale, significance, or
  causal claim. Next: document a publication-facing analysis plan before
  interpreting any descriptive values.

- Generated the read-only 5-UAV full-method versus independently trained
  no-uncertainty descriptive paired artifact
  `outputs/paired_summaries/5uav_full_vs_no_uncertainty_20260729.json` at
  revision `cb82ef6`. The strict paired validator accepted both 30-cell / 600
  JSONL roots, six scenarios, five common training seeds, and 20 indexed
  episodes/cell. It uses only the valid no-uncertainty rerun4 root and excludes
  initial/rerun1/rerun2/interrupted-rerun3 roots. It is descriptive seed-level
  provenance only, not a significance, method-effect, safety, fallback-rate,
  scale, or causal claim; its CBF evidence stays separate in the existing
  all-source replay. Next: consolidate input/exclusion rules into a
  publication-facing analysis plan before interpreting any comparison values.

- Added `scripts/summarize_multiarm_checkpoint_evaluations.py` and
  `tests/test_multiarm_checkpoint_summary.py` for read-only, four-arm 3-UAV
  descriptive reporting. It accepts a named root plus cell-directory prefix,
  validates 20 episodes/cell and exact identities, and uses five trained seeds
  as the only independent unit. The retained artifact is
  `outputs/paired_summaries/3uav_four_arm_descriptive_20260729.json`; it covers
  MLP, raw graph, predictive graph without uncertainty, and uncertainty-aware
  predictive graph over six scenarios, five seeds, and 20 episodes/cell.
  MLP excludes the two empty seed-20260719 original directories and selects the
  documented schema/RNG and OOD-trajectory reruns instead. This is descriptive
  provenance, not a significance, method-effect, safety, fallback-rate, scale,
  or causal result. Its output separates 11 task metrics from five CBF
  diagnostic metrics per scenario. TDD target tests (6), Ruff, and mypy (103
  sources) pass.
  Full pytest remains 152 passed / 1 unrelated legacy MATLAB inventory failure
  (83 files observed vs 81 asserted) / 28 OSQP warnings. Next: prepare the
  publication-facing multi-arm analysis plan with strict CBF-diagnostic and
  invalid-root exclusions, then continue primary-source retrieval.

- Primary-source verification followed the Thumiger--Deghat UAV MARL DOI to
  official IEEE Xplore on 2026-07-29. The page returned `Unusual Traffic
  Detected` (HTTP 418) before abstract/PDF access. The ledger remains
  metadata-only; no statement may infer communication, dynamic-obstacle, or
  safety assumptions. Seek an authorized IEEE or author primary copy rather
  than bypassing the access restriction.

- Primary-source verification followed Liu et al.'s DHCG DOI to the official
  IJCAI proceedings page on 2026-07-29 and reviewed its metadata/abstract.
  The page exposed the official PDF, but two browser download attempts timed
  out and produced no local PDF. The ledger therefore remains
  full-text-unreviewed; it records only the official abstract's high-level
  description. Do not infer timing assumptions, graph inputs, uncertainty,
  dynamic-obstacle, or CBF properties. Seek the official PDF or author copy
  before making a DHCG difference or novelty statement.

- Primary-source verification also attempted the Neural Graph CBF DOI on
  2026-07-29. It resolved to the official ACM page, but Cloudflare human
  verification prevented access to abstract/PDF and network-idle timed out.
  The related-work ledger remains metadata-only and now records this retrieval
  block; no statement about method, assumptions, guarantees, or differences
  may cite this unread full text. Seek an official proceedings or author copy
  rather than bypassing the verification gate.

- Primary-source verification attempted DACOM's official AAAI PDF on
  2026-07-29 through the required browser flow. The landing-page abstract and
  official PDF link were accessible, but the browser download timed out and
  direct navigation reported a download start without producing a local file.
  No full-paper assertion was added: `docs/related_work_matrix.md` now records
  the failed retrieval and keeps DACOM explicitly full-text-unreviewed. Do not
  use it to support a novelty gap until an official or author-provided primary
  PDF can be read.

- Added and verified the read-only paired-summary artifact
  `scripts/summarize_paired_checkpoint_evaluations.py` with
  `tests/test_paired_checkpoint_summary.py`. It validates 20 records/cell and
  matched cell identities, aggregates episodes within each seed-scenario cell,
  and uses only the five trained seeds as independent descriptive units. The
  retained output
  `outputs/paired_summaries/8uav_full_vs_no_uncertainty_20260729.json` covers
  six scenarios × five seeds × 20 episodes and carries an explicit
  no-significance/no-method-effect claim boundary. TDD red/green evidence is
  retained: missing-module test failed first, then 2 tests passed. Full Ruff
  and `mypy --explicit-package-bases multiuav scripts` pass (102 sources).
  Full pytest has an unrelated retained legacy audit drift (83 observed `.m`
  files vs expected 81): 148 passed, 1 failed, 28 OSQP warnings. Do not mask
  it; a separate source inventory audit is needed before changing those legacy
  documents/tests. Next: continue primary-source literature verification and
  complete the remaining multi-arm reporting plan without turning descriptive
  values into causal or submission claims.

- Read-only audit confirms the matched 8-UAV full-method root
  `outputs/core_8uav_evaluations/` and valid ablation root
  `outputs/core_8uav_ablation_evaluations_20260729/` each have exact five-seed
  × six-scenario coverage: 30 directories, 30 summaries, 30 JSONL, and 600
  parseable records, with no identity/count errors. Raw records have common
  `seed`, `scenario`, and `episode` fields. This proves eligibility only.
  Before any method comparison, make a read-only paired summary that aggregates
  each 20-episode seed-scenario cell and uses the five independently trained
  seeds—not 100 episode records—as inferential replicates; explicitly exclude
  all invalid/interrupted roots and keep CBF diagnostic replay separate.

- The valid 8-UAV principal no-uncertainty five-seed matrix and all-source CBF
  replay are complete. The fresh serial root
  `outputs/core_8uav_ablation_evaluations_20260729/` was launched at revision
  `408a115` with frozen config SHA-256
  `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`, CUDA
  inference, CPU OSQP, five final checkpoints, six canonical scenarios, and
  20 episodes/cell. It has exactly 30 expected directories/summaries/JSONL and
  600 parseable records; launcher stderr is zero. The preceding identity-test
  interpolation error failed before evaluator start and made no result cell.
- `outputs/cbf_diagnostics/predictive_no_uncertainty_8uav_5seed_replay_20260729.jsonl`
  plus `.summary.json` apply unchanged 20,000 iterations, 0.1 seconds, slack
  penalty 100, and zero uncertainty margin gain/max margin over 65 sources.
  They retain 808 source events and zero replay errors; all raw records remain.
  These are not performance, safety, fallback-rate, scale, or ablation claims.
  Next: read-only audit the matching full-method 8-UAV valid matrix and define
  paired aggregation inputs/exclusions before any comparison; then continue
  primary-source literature verification.

- The first 8-UAV ablation evaluation preflight had a PowerShell interpolation
  typo while rendering a test identity; it failed before launching an evaluator
  or creating a result cell. The retained fresh root contains only its ignored
  launcher at that point. A corrected preflight validated
  `core_8uav_predictive_no_uncertainty_seed_20260719_nominal`, all five final
  checkpoints, and zero training processes, then launched the valid serial
  matrix at revision `408a115` under
  `outputs/core_8uav_ablation_evaluations_20260729/`. It uses the frozen 8-UAV
  no-uncertainty config, CUDA network inference, CPU OSQP, five seeds, six
  canonical scenarios, and 20 episodes/cell. At first health check two cells
  were complete, the third was active, and launcher stderr was zero. Do not
  start concurrent work. Audit 30 expected cells / summaries / JSONL and 600
  parseable records before replay; do not treat partial output as a result.

- Final independent 8-UAV principal no-uncertainty seed `20260723` is complete
  and documented. It launched at revision `278129e` under frozen config
  SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`;
  its final artifacts are retained in
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260723`.
  It reached 100,224 transitions/261 updates, with training/initial/final
  fallback counts 3/8/0 over 12,528/160/160 decisions; all 11 raw contexts and
  launcher logs remain. No Python training command remains and no CBF setting
  changed. All five 8-UAV independent checkpoints now exist. Next: launch a
  fresh serial six-scenario, 20-episode-per-cell evaluation matrix for every
  final checkpoint, retaining JSONL; audit it before all-source replay or any
  comparison claim.

- After a corrected zero-`python.exe`-training-process preflight, final 8-UAV
  principal no-uncertainty seed `20260723` launched at revision `278129e` with
  the frozen config SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`.
  The sole verified training command uses CUDA network execution and targets
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260723`.
  Live telemetry exists, final summary does not, and its retained launcher
  stderr has 392 bytes. Do not add work, restart, overwrite, or tune CBF while
  it runs. On completion, audit final checkpoint/summary/event contexts, write
  docs, then run the full five-seed 8-UAV six-scenario matrix and unchanged
  all-source replay before any ablation claim.

- Independent 8-UAV principal no-uncertainty seed `20260722` is complete and
  documented. It launched at revision `59e9984` with frozen config SHA-256
  `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`; all
  final artifacts are retained under
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260722`.
  It reached 100,224 transitions/261 updates; training/initial/final fallback
  counts are 218/5/11 across 12,528/160/160 decisions. All 234 contexts and
  launcher logs remain raw. No Python training command remains and no CBF
  setting changed. This is no result claim. Next: corrected zero-training-
  process preflight, then final serial seed `20260723`.

- Before seed `20260722`, a first command-line preflight was invalid because
  the PowerShell query matched its own command text; it launched nothing and
  created no seed artifact. A corrected preflight filtered `python.exe` plus
  the training command, found zero training processes, and launched the sole
  8-UAV principal no-uncertainty seed `20260722` at revision `59e9984` using
  frozen config SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`.
  The verified live command targets
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260722`;
  live telemetry exists, final summary does not, and retained stderr has 441
  bytes. Do not start another job or tune CBF. On natural completion, audit
  final checkpoint/summary/context before serially launching `20260723`.

- Independent 8-UAV principal no-uncertainty seed `20260721` is complete and
  documented. It ran at revision `08427a2` under the unchanged frozen config
  SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`;
  its distinct final artifacts remain under
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260721`.
  It reached 100,224 transitions/261 updates, with training/initial/final
  fallback counts 139/13/10 in 12,528/160/160 decisions. All 162 contexts and
  launcher logs are retained; no Python process remains and no CBF setting
  changed. This is not a performance or safety result. Next: zero-process
  preflight, then serial seed `20260722` under the identical protocol.

- A first no-process check immediately before the planned seed `20260721`
  launch transiently observed one Python process; the launcher was not run. A
  follow-up read-only check found none, and only then launched seed `20260721`
  serially at revision `08427a2` with the unchanged frozen config SHA-256
  `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73` into
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260721`.
  It is now the sole `multiuav_rl` CUDA job; live telemetry exists, final
  summary does not, and its retained launcher stderr has 736 bytes. Preserve
  that notice, do not restart or tune, and verify final artifacts/events before
  starting seed `20260722`.

- Independent 8-UAV principal no-uncertainty seed `20260720` is complete and
  documented. Launched at revision `96267be` with frozen config SHA-256
  `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`, it
  retains its final checkpoint, summary, telemetry, TensorBoard, and launcher
  logs under `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260720`.
  It reached 100,224 transitions/261 updates; training/initial/final script
  evaluation fallback counts are 191/0/13 over 12,528/160/160 decisions. All
  204 contexts remain raw. No Python process remains and no CBF setting
  changed. This is single-seed diagnostics only. Next: complete no-process
  preflight then launch seed `20260721` serially under the identical protocol.

- After a zero-Python-process preflight, the sole active CUDA job is independent
  8-UAV principal no-uncertainty seed `20260720`, launched at revision
  `96267be` with the same frozen config SHA-256
  `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73` and
  command pattern as seed `20260719`, but distinct output
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260720`.
  Its retained launcher logs are `..._seed_20260720_launcher_stdout.log` and
  `_stderr.log` under `outputs/core_8uav_ablation/`. Initial health found live
  telemetry, no final summary, one `multiuav_rl` Python process, and 245 bytes
  of preserved stderr; do not restart, overwrite, or alter CBF settings. On
  natural completion, verify final checkpoint/summary and every CBF context,
  document the seed, then serially launch `20260721`.

- The first independent 8-UAV principal no-uncertainty seed `20260719` is
  complete and documented. It ran at revision `0f6e011` using the frozen
  config SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`,
  CUDA network execution, CPU OSQP, predictive delivered-packet knowledge,
  and disabled uncertainty inputs/margins. Its distinct artifact root is
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260719`;
  final checkpoint, summary, telemetry, TensorBoard, and launcher logs are
  retained. It has 100,224 transitions/261 updates; training/initial/final
  script-evaluation fallback counts are 182/15/0 (12,528/160/160 decisions).
  Preserve all contexts and make no result claim or CBF change. No Python
  process remains. Next: launch seed `20260720` serially under the same config
  and a distinct output directory only after a no-process preflight.

- After a zero-Python-process preflight, the sole active CUDA job is the first
  independent 8-UAV principal no-uncertainty seed `20260719`, launched at
  revision `0f6e011`. Its retained ignored launcher is
  `outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260719_launcher.ps1`,
  which runs `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
  scripts/train_graph_mappo.py --config
  configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml
  --device cuda --seed 20260719 --num-uavs 8 --graph-mode predictive_graph
  --total-steps 100000 --output-dir
  outputs/core_8uav_ablation/core_8uav_predictive_no_uncertainty_seed_20260719`.
  Frozen config SHA-256 is
  `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`:
  predictive graph with risk gain 0 and CBF uncertainty-margin gain/max margin
  0; slack penalty 100, 20,000 iterations, and 0.1 seconds are unchanged.
  Initial health shows its output directory and live telemetry, no final
  summary, and zero-byte launcher stderr. CUDA is for the network, OSQP stays
  CPU-side. Do not start another Python job; verify final checkpoint, summary,
  and all CBF event context before documenting it and serially starting seed
  `20260720`.

- The valid 5-UAV principal no-uncertainty ablation evaluation matrix and its
  unchanged-protocol CBF replay are complete. At launch revision `10a9c8e`,
  rerun4 evaluated all five independent final checkpoints using frozen
  `configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`
  (SHA-256 `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`),
  CUDA network inference, CPU OSQP, six canonical scenarios, and 20 episodes
  per cell. `outputs/core_5uav_ablation_evaluations_rerun4/` contains exactly
  30 expected directories, 30 summaries, and 30 JSONL files; all 600 records
  parse, and launcher stderr is zero bytes. It is the sole eligible root;
  initial/rerun1/rerun2/rerun3 roots are retained but excluded.
- `outputs/cbf_diagnostics/predictive_no_uncertainty_5uav_5seed_replay_20260729.jsonl`
  and its `.summary.json` use unchanged 20,000 iterations, 0.1 seconds,
  slack penalty 100, zero uncertainty-margin gain, and zero maximum uncertainty
  margin over 65 telemetry sources (5 training summaries + 30 valid evaluation
  summaries + 30 valid JSONL files). It retains five training-source events,
  no evaluation-source events, and zero replay errors. Do not read any of this
  as a method, safety, performance, fallback-rate, or scalability conclusion;
  no CBF parameter changed. Next: audit the matching full-method 5-UAV data
  and define a paired, invalid-root-excluding aggregation plan before reporting
  any ablation comparison; then continue the frozen 8-UAV ablation protocol.

- A fresh, serial 5-UAV principal no-uncertainty evaluation matrix was launched
  on 2026-07-29 at revision `10a9c8e`, after a zero-Python-process preflight.
  The retained ignored launcher is
  `outputs/core_5uav_ablation_evaluations_rerun4/launcher.ps1`; its fixed
  command invokes the five final checkpoints through
  `scripts/evaluate_core_checkpoint.py` with frozen
  `configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`, 20
  episodes, CUDA network inference, and CPU OSQP for each of the six canonical
  scenarios. The validated identity template is
  `core_5uav_predictive_no_uncertainty_seed_${seed}_${scenario}`. Its stdout
  and stderr logs remain in rerun4. A health check found one `multiuav_rl`
  Python process, no cell artifact yet, and zero-byte stderr. Do not launch a
  concurrent job. On completion, audit 30 directories, 30 summaries, 30 JSONL,
  and 600 parseable records before treating the matrix as valid.

- A third 5-UAV ablation evaluation launcher is invalid/excluded. Its inline
  PowerShell quoting stripped the intended experiment-name string assignment,
  causing evaluator argument validation to fail before any cell ran. Preserve
  `outputs/core_5uav_ablation_evaluations_rerun2/` (launcher stderr only) and
  exclude it. The fourth launcher is active in the fresh
  `outputs/core_5uav_ablation_evaluations_rerun3/` root using a retained
  `.ps1` file and the verified name template
  `core_5uav_predictive_no_uncertainty_seed_${seed}_${scenario}`. That fourth
  attempt was interrupted after the 24 complete cells for seeds `20260719`--
  `20260722`; as of 2026-07-29 there is no Python process and rerun3 has 24
  directories / 24 summaries / 24 JSONL files with zero-byte launcher stderr.
  Preserve and exclude its entire root, including valid-looking raw cells, as
  an incomplete matrix. Launch a fresh full 30-cell root, then audit exactly
  30 cells / 30 summaries / 30 raw JSONL / 600 parseable records before replay.

- All five independently trained 5-UAV principal no-uncertainty ablation seeds
  `20260719`--`20260723` now have final checkpoints and summaries; the final
  seed reached 100,080 transitions/417 updates with two retained `solved
  inaccurate` training fallbacks in 20,016 decisions. No Python process
  remains. Next: run the six-scenario, 20-episode checkpoint matrix for every
  seed into a fresh evaluation root, retain JSONL, then replay all CBF events
  without changing solver values. No performance or ablation conclusion yet.
- The first evaluation launcher is invalid/excluded: it made an empty nominal
  directory and then stopped because `delay` is not a valid evaluator scenario
  name. Its stderr is preserved; it emitted no JSONL/summary. Launch a fresh
  30-cell root using exactly `nominal`, `delay_only`, `loss_only`,
  `dynamic_only`, `combined`, and `ood_communication_obstacle`.
- The corrected-root launcher is also invalid/excluded: its PowerShell string
  interpolation omitted seed identifiers from experiment names, risking
  cross-seed collisions. It was stopped; preserve every artifact under
  `outputs/core_5uav_ablation_evaluations_rerun1/` and exclude it. Before a new
  launch, validate one rendered command uses `${seed}` boundaries, then use a
  new root; no Python process remains.
- Seed `20260721` has completed; no Python process remains. Its independent
  5-UAV no-uncertainty ablation output is
  `outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260721`.
  It reached 100,080 transitions and 417 CUDA updates. Training telemetry has
  one `solved inaccurate` fallback in 20,016 decisions (index 7,458, 20,000
  iterations); initial/final short evaluations have zero in 160 each and stderr
  has one matching notice. Raw context is retained; no CBF setting changed and
  this is not a performance/safety/ablation claim. Next: document the completed
  seed in progress/known-issues, then launch seed `20260722` serially under the
  same config after a no-process preflight.
- Seed `20260722` has now completed at 100,080 transitions and 417 updates;
  its distinct output, checkpoint, summary, telemetry, TensorBoard, and logs
  are retained under `outputs/core_5uav_ablation/`. Training telemetry has one
  `solve_time_limit` emergency fallback in 20,016 decisions; do not change CBF
  values. Next: complete documentation and launch final seed `20260723`
  serially after confirming no Python process.

## Current continuation update (2026-07-27, supersedes stale status below)

- The sole active CUDA job is 5-UAV principal no-uncertainty ablation seed
  `20260721`, PID `50760` at first check. It launched at revision `d49401a`
  with the frozen configuration SHA-256
  `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`, output
  `outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260721`,
  CUDA learned-network execution, and CPU OSQP. First telemetry has 960
  transitions and zero emergencies; it is in-progress only. Do not start any
  second Python job or change CBF settings. Verify and document final artifacts
  before launching seed 20260722.
- The second independently trained 5-UAV principal no-uncertainty ablation
  seed `20260720` completed after no-Python-process preflight at revision
  `2402d8b`:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml
  --device cuda --seed 20260720 --num-uavs 5 --graph-mode predictive_graph
  --total-steps 100000 --output-dir
  outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260720`.
  Configuration SHA-256 is
  `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`.
  The distinct run directory and launcher stdout/stderr are retained under
  `outputs/core_5uav_ablation/`. It reached 100,080 transitions and 417 updates
  with zero training emergency fallback in 20,016 CBF decisions and zero in
  each initial/final 160-decision evaluation. Stderr has one emergency-notice
  line absent from telemetry; preserve this discrepancy without merging,
  imputation, exclusion, or CBF changes. No Python process remains. This is
  seed-local diagnostic evidence only. Next: launch seed `20260721` serially
  under the identical protocol and retain its distinct raw telemetry.
- The first independently trained 5-UAV principal no-uncertainty ablation
  seed `20260719` completed after a zero-Python-process preflight at revision
  `b919a81` with
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml
  --device cuda --seed 20260719 --num-uavs 5 --graph-mode predictive_graph
  --total-steps 100000 --output-dir
  outputs/core_5uav_ablation/core_5uav_predictive_no_uncertainty_seed_20260719`.
  Configuration SHA-256: `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`.
  The distinct output directory and retained launcher logs are under
  `outputs/core_5uav_ablation/`. It reached 100,080 transitions and 417 updates
  with CUDA network execution, predicted delivered-packet knowledge, and
  uncertainty graph input disabled. Training telemetry retains one
  `solve_time_limit` emergency event in 20,016 decisions (index 4,756;
  0.10070149996317923 seconds; 14,524 iterations); initial/final short
  evaluations retain zero in 160 each; stderr has one notification. The full
  event context is preserved. No Python process remains. This is seed-local
  diagnostics only, not a performance, safety, fallback-rate, or ablation
  conclusion. Next: launch seed `20260720` serially with this exact config and
  a distinct output directory; do not change CBF values or start evaluation.
- The 8-UAV uncertainty-aware predictive five-seed evaluation matrix and CBF
  replay are complete and retained. The serial CUDA evaluator used the five
  final checkpoints, frozen `configs/rl/dynamic_graph_8uav.yaml` SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`, six
  scenarios, and 20 episodes per condition. `outputs/core_8uav_evaluations/`
  contains exactly 30 cell directories, 30 summaries, and 30 JSONL files;
  all 600 episode records parse and every cell has 20 records. Launcher logs:
  `uncertainty_predictive_graph_8uav_5seed_eval_20260727_launcher_stdout.log`
  and `_launcher_stderr.log`. The launch revision was `f90471f`; neural
  inference was CUDA and OSQP CBF was CPU-side.
- CBF replay is retained at
  `outputs/cbf_diagnostics/uncertainty_predictive_graph_8uav_5seed_replay_20260727.jsonl`
  with companion `.summary.json`, over 5 training summaries + 30 evaluation
  summaries + 30 raw JSONL files under the unchanged 20,000-iteration,
  0.1-second, slack-penalty-100, uncertainty-margin-gain-0.5, max-margin-5.0
  protocol. It records 200 source events: 116 replayed and 84 explicit errors
  (`Recorded dynamic-obstacle centers cannot be reconstructed from the event
  step`). Preserve every error and raw event; do not tune, impute, merge, or
  make a safety/fallback-rate/scalability claim.
- The first 8-UAV evaluation launcher had a PowerShell `-or` syntax typo and
  failed before an evaluator ran, creating only an empty root. It is a
  retained, excluded invalid attempt. A subsequent read-only audit command
  also had a path-binding error and read no JSONL; the corrected audit above is
  the authoritative 30/600/zero-parse-error evidence.
- Preflight found no Python process. Next concrete action: inspect and freeze
  an existing critical-ablation arm that remains on the single communication-
  uncertainty-predictive-graph-to-CBF mainline, then launch its independent
  CUDA training serially. Never reuse the full-method checkpoint or stage
  `.gitignore`; do not start literature browsing until the required local
  `/browse` setup is explicitly authorized.
- The principal independently trained ablation is now frozen in
  `configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml` and
  `configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`.
  It preserves delayed/lossy delivered-packet prediction, dynamic obstacles,
  optimizer settings, CBF slack penalty 100, and iteration cap 20,000, while
  explicitly setting `graph_mode: predictive_graph`, graph uncertainty-risk
  gain 0, and both CBF uncertainty-margin fields 0. This implements the
  research brief's `gamma_sigma = kappa_sigma = kappa_CBF = 0` causal ablation,
  not covert CBF tuning. It has no results yet. After configuration checks and
  commit, launch only 5-UAV seed 20260719 on CUDA into a distinct
  `outputs/core_5uav_ablation/` directory, then verify/document it before the
  next serial seed.
- The two ablation profiles were loaded successfully as predicted-knowledge
  enabled and uncertainty-usage disabled, with zero graph risk gain and zero
  CBF uncertainty margins. Targeted actor-packet/graph/CBF tests passed
  (`39 passed`, 11 OSQP deprecation warnings); Ruff passed; mypy passed with
  `--explicit-package-bases` across 101 source files. A prior plain mypy
  invocation stopped at an existing duplicate module-discovery error for
  `scripts/diagnose_cbf_failure.py`; it produced no code/config change and the
  explicit-package-bases run is authoritative.
- Protocol/configuration record commit: `b30f69f` (`docs: freeze uncertainty
  ablation protocol`), pushed to `origin/codex/phase14-dynamic-world`. The only
  intentionally unstaged worktree file remains the user's `.gitignore`.

## Current continuation update (2026-07-26, supersedes stale status below)

- The frozen 8-UAV uncertainty-aware predictive evaluation matrix is active.
  After confirming no Python training process and all 30 cell paths absent, a
  hidden serial launcher began five final checkpoints across six scenarios, 20
  episodes per cell, using `evaluate_core_checkpoint.py --family graph_mappo
  --config configs/rl/dynamic_graph_8uav.yaml --checkpoint
  <8uav-seed>/checkpoints/graph_mappo_final.pt --output-dir
  outputs/core_8uav_evaluations --experiment-name
  core_8uav_uncertainty_predictive_graph_seed_<seed>_<scenario> --seed <seed>
  --num-uavs 8 --episodes 20 --scenario <scenario> --device cuda`, serially.
  Launch revision `f90471f`, config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`; CUDA
  is for network inference and OSQP remains CPU-side.  Launcher logs are
  `outputs/core_8uav_evaluations/uncertainty_predictive_graph_8uav_5seed_eval_20260727_launcher_stdout.log`
  and `_launcher_stderr.log`.  At health check three seed-20260719 cells had
  been created and a `multiuav_rl` CUDA evaluator was active; stderr contained
  only a retained PowerShell CLIXML first-module notice.  An earlier launcher
  construction failed before launch due to a PowerShell `-or` syntax typo; it
  created only the empty root directory, no cell/log/result, and is retained as
  an excluded invalid attempt.  Do not start ablations or make claims.  On
  completion audit all 30 cells/600 JSONL and replay unchanged-protocol CBF.
- Commit `0516060` records completed final 8-UAV seed `20260723`: final
  checkpoint/summary/telemetry/logs retained under its seed directory; 100,224
  transitions and 261 updates at revision `fd6142c`, frozen config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Training telemetry has one `solved inaccurate` event in 12,528 decisions
  (index 4,279); initial evaluation has nine `solve_time_limit` events in 160
  decisions; final evaluation has zero in 160.  Stderr has ten notifications,
  not a one-to-one mapping.  Preserve all sources without parameter changes.
  The 8-UAV five-seed training set is complete but not a result.  Next: verify
  the commit, run its frozen 30-cell six-scenario 20-episode matrix, audit 600
  raw JSONL records, and replay all CBF evidence before any ablation.
- After confirming no prior Python process and absent target paths, the sole
  active serial CUDA job is final 8-UAV uncertainty-aware predictive seed
  `20260723`, PID `54264` at health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_8uav.yaml --device cuda --seed 20260723
  --num-uavs 8 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260723`.
  It was launched at revision `fd6142c` with frozen config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Launcher logs, live telemetry, and TensorBoard output are retained.  At
  health check it had 8,064 transitions and no emergency event in 1,008
  training CBF decisions; this is in-progress only.  Do not start another job,
  change safety settings, or begin ablations.  On completion, verify the final
  checkpoint/summary and all events, then run the frozen six-scenario 20-episode
  8-UAV evaluation matrix before independent ablations.
- Commit `d27f115` records completed fourth 8-UAV uncertainty-aware predictive
  seed `20260722`: 100,224 transitions, 261 updates, final checkpoint,
  summary, TensorBoard, telemetry, and launcher logs retained under
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260722`.
  Launch revision is `056b6d3`, config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Telemetry retains two `solved inaccurate` events in 12,528 training CBF
  decisions (indices 10,468 and 11,541; unchanged 20,000-iteration cap);
  initial/final evaluation each retain zero in 160 decisions.  Stderr has four
  notifications, not a one-to-one mapping; preserve without merging, imputation,
  exclusion, or parameter changes.  This is seed-local diagnostics only.
  Before acting, verify the commit and no Python process, then train final seed
  `20260723` serially with the same 8-UAV protocol; do not begin 8-UAV
  evaluation or ablations first.
- After confirming no prior Python process and absent target paths, the sole
  active serial CUDA job is 8-UAV uncertainty-aware predictive seed
  `20260722`, PID `13344` at health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_8uav.yaml --device cuda --seed 20260722
  --num-uavs 8 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260722`.
  It was launched at revision `056b6d3` with frozen config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Launcher stdout/stderr, live telemetry, and TensorBoard output are retained.
  At health check it had 9,216 transitions and no emergency event in 1,152
  training CBF decisions; this is in-progress only.  Do not start another job,
  change safety settings, or begin ablations.  On completion, verify all
  artifacts/events, document the seed, then train final seed `20260723`
  serially before 8-UAV evaluation.
- Commit `6fe9d96` records completed third 8-UAV
  uncertainty-aware predictive seed `20260721`.  Its checkpoint, summary,
  TensorBoard data, live telemetry, and dedicated launcher logs are retained
  in `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260721`.
  It reached 100,224 transitions and 261 updates at launch revision `2fb5a0e`
  with frozen config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Final telemetry retains five `solve_time_limit` training fallbacks in 12,528
  decisions (indices 3441, 5081, 5082, 5084, 5122); initial and final
  evaluation telemetry each retain zero in 160 decisions.  Stderr has six
  notification lines, not a one-to-one event mapping; retain both sources
  without merging/imputation/exclusion or parameter changes.  This is
  seed-local diagnostic evidence only.  Before action, verify the
  documentation commit and no Python process, then train `20260722` serially
  with the same 8-UAV protocol; do not begin 8-UAV evaluation or ablations.
- After confirming no prior Python process and absent target paths, the sole
  active serial CUDA job is 8-UAV uncertainty-aware predictive seed
  `20260721`, PID `56460` at health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_8uav.yaml --device cuda --seed 20260721
  --num-uavs 8 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260721`.
  It was launched at revision `2fb5a0e` with frozen config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Dedicated launcher stdout/stderr, live telemetry, and TensorBoard data are
  retained.  At health check it had 4,224 transitions and no emergency event
  in 528 training CBF decisions; this is in-progress only.  Do not start
  another job, change safety settings, or begin ablations.  On completion,
  verify artifacts/events, document the seed, then train `20260722` and
  `20260723` serially before 8-UAV evaluation.
- Continued monitoring at 40,704 transitions retained four
  `solve_time_limit` telemetry events while stderr had five emergency-notification
  lines.  The latest retained event is decision 5,084, environment 1 step 1,
  obstacle `[22.0, 52.0, 30.0]`, unchanged 0.1-second limit,
  0.10099179999087937 seconds, 4,198 iterations, primal residual
  0.008861797552822449, and dual residual 0.10744201494920594.  Preserve both
  raw sources and their explicit count discrepancy; do not merge/impute/drop
  events or change the CBF protocol.  Telemetry SHA-256 at this observation is
  `D7E0859DE2D56684C547978F76279C085890D819AB87E8107693AB92AE22A5A9`.
- Commit `fea8558` records completed second 8-UAV
  uncertainty-aware predictive seed `20260720`.  Its checkpoint, summary,
  TensorBoard data, live telemetry, and dedicated launcher logs are retained
  in `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260720`.
  It reached 100,224 transitions and 261 updates at launch revision `a99f7bb`
  with frozen config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Final training telemetry retains sixteen `solve_time_limit` fallbacks in
  12,528 decisions (indices 282, 1121, 1721, 2641, 3922, 3924, 3961, 3962,
  3964, 4162, 5881, 5882, 6001, 8042, 8044, 11681); initial and final
  evaluation telemetry each retain zero in 160 decisions.  The dedicated stderr
  has 32 emergency-notification lines that do not map one-to-one to telemetry;
  preserve this discrepancy without merging, imputation, exclusion, or
  parameter change.  This is seed-local diagnostic evidence only.  Before
  action, verify the documentation commit and no Python process, then train
  seed `20260721` serially with the same 8-UAV protocol; do not begin 8-UAV
  evaluation or ablations first.
- After confirming no prior Python process remained and target paths were
  absent, the sole active serial CUDA job is 8-UAV uncertainty-aware predictive
  seed `20260720`, PID `35344` at health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_8uav.yaml --device cuda --seed 20260720
  --num-uavs 8 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260720`.
  It was launched at revision `a99f7bb` with config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Dedicated launcher stdout/stderr are retained at the sibling
  `...seed_20260720_launcher_{stdout,stderr}.log` paths.  At health check it
  had 3,840 transitions; its 480 training CBF decisions retained one
  `solve_time_limit` emergency fallback at decision 282, environment 1 step
  0, obstacle center `[22.0, 50.0, 30.0]`, unchanged 0.1-second limit,
  5,897 iterations, primal residual 0.051202473625902925, and dual residual
  0.9291498430633159.  The full context remains in live telemetry and stderr.
  This is an in-progress seed-local diagnostic only.  Do not start another
  training/evaluation job, modify any CBF setting, or start ablations.  On
  completion, verify final artifacts and every event, document the seed, then
  continue serially with seeds `20260721`--`20260723` before 8-UAV evaluation.
- Continued monitoring at 33,792 transitions retained ten `solve_time_limit`
  telemetry events (decision indices 282, 1121, 1721, 2641, 3922, 3924, 3961,
  3962, 3964, and 4162), each at the unchanged 0.1-second solve limit; full
  contexts and exact residuals/iterations remain in the raw telemetry.  The
  same retained stderr log had 21 emergency-notification lines at that point.
  Do not assume a one-to-one mapping, merge, or discard either raw source:
  preserve the explicit count discrepancy for final audit/replay, without
  altering the solver protocol.  Telemetry SHA-256 at this observation is
  `F8A908F884CBEED91BD79439CF0BD590C4F78CEAA7A668BC284F61D5331C596B`.
- At 58,368 transitions, telemetry had thirteen `solve_time_limit` events
  (adding indices 5881, 5882, and 6001 to the preceding ten) while stderr had
  28 notification lines.  The three added events used the same unchanged
  0.1-second limit and have complete raw contexts; their respective
  primal/dual residuals are 0.2613576275084453/10.318421165695543,
  0.21177180139506407/3.5120744819073146, and
  0.12085665689688246/1.1663214601318403.  Retain the telemetry snapshot
  (SHA-256 `8843C8CF240FD370556C50243EA724FB2A447F3AB6721DAD9255B34BD05CC770`)
  and raw stderr; these diagnostic observations do not justify a protocol
  change or numerical conclusion.
- Commit `2e5028e` records the completed first 8-UAV
  uncertainty-aware predictive seed `20260719`.  Its final checkpoint,
  summary, TensorBoard data, and live telemetry are retained in
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260719`.
  It reached 100,224 transitions and 261 updates at launch revision `07d0f47`
  with frozen config SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Training telemetry retains one `solved inaccurate` fallback in 12,528 CBF
  decisions (index 4,222 at the unchanged 20,000-iteration cap); initial and
  final evaluation telemetry each retain zero in 160 decisions.  The intended
  launcher logs remain absent and must not be recreated.  The known preflight
  import and subsequent read-only log/completion checks failed without
  generating or overwriting any experiment artifact; retain them as excluded
  observability errors.  This is seed-local diagnostic evidence only.  Before
  action, verify the documentation commit and no Python process, then train
  `20260720` serially with the identical 8-UAV protocol; do not start the
  8-UAV evaluation matrix or ablations first.
- The matched 8-UAV uncertainty-aware predictive protocol has begun with its
  first independent serial CUDA seed `20260719`, PID `43996` at health check.
  Command: `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
  scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_8uav.yaml
  --device cuda --seed 20260719 --num-uavs 8 --graph-mode
  uncertainty_predictive_graph --total-steps 100000 --output-dir
  outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260719`.
  It was launched at revision `07d0f47` with configuration SHA-256
  `9AACA64DFBEE765777652E2F22E771E566F0DA9045B9CBB05BA921789A6BC93C`.
  Correct pre-launch loading verified eight UAVs, two environments, rollout
  length 24, dynamic obstacles, and uncertainty use.  An earlier read-only
  preflight attempted to import the loader as `load_graph_mappo_config`, which
  does not exist; it failed before launch and created no experiment artifact.
  At health check the live telemetry and TensorBoard output existed, with 6,528
  transitions and no emergency event in 816 training CBF decisions.  The
  intended dedicated stdout/stderr paths did not materialize because the
  parent output directory was absent when the process was launched.  A later
  read-only `Get-Content` check of the absent stderr path therefore returned
  nonzero; do not restart, recreate, or overwrite the logs.  Retain live
  telemetry and final artifacts as available evidence.  This is an active
  seed-local observation only.  Do not start another training/evaluation job,
  modify CBF settings, or start ablations.  On completion, verify final
  checkpoint/summary and every retained event, document the seed, then proceed
  serially through `20260720`--`20260723` before the 8-UAV evaluation matrix.
- Continued seed `20260719` monitoring retained one `solved inaccurate` CBF
  emergency fallback at decision 4,222, environment 1 step 10, obstacle center
  `[22.0, 70.0, 30.0]`, unchanged 20,000-iteration cap, 0.05533530001412146
  seconds, primal residual 0.0011686880877244399, and dual residual
  0.0000192246889227154.  The full eight-UAV context is retained in
  `outputs/core_8uav/core_8uav_uncertainty_predictive_graph_seed_20260719/live_training_telemetry.json`
  (observed SHA-256
  `60BF79BA9759853D545FAE978F793E94E9B803D89EC065EF852B44C8BF385983` at
  43,776 transitions).  Preserve it and do not alter slack, solver limits,
  tolerances, uncertainty margin, or any safety setting; it is not a safety,
  performance, or fallback-rate conclusion.
- Commit `b3a9947` records the completed 5-UAV uncertainty-aware
  predictive five-seed training/evaluation/replay artifact set.  The five
  final checkpoints for seeds `20260719`--`20260723` each have all six frozen
  20-episode scenario cells under `outputs/core_5uav_evaluations/`, yielding
  30 directories and 600 raw JSONL records.  The completeness audit found
  exactly 20 parseable nonblank records in every cell.  The serial CUDA
  evaluator was launched at revision `bbb871a` with 5-UAV config SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`; its
  launcher logs are
  `outputs/core_5uav_evaluations/uncertainty_predictive_graph_5uav_5seed_eval_20260726_launcher_stdout.log`
  and `_launcher_stderr.log` (the latter retains a harmless first-module
  CLIXML notice).  Scan of all 600 raw records found zero evaluation CBF
  emergency events.  The valid all-source replay is
  `outputs/cbf_diagnostics/uncertainty_predictive_graph_5uav_5seed_replay_20260726.jsonl`
  plus `.summary.json`; it retains eight source events with zero replay errors
  under unchanged 20,000-iteration/0.1-second/penalty-100/uncertainty-gain-0.5/
  cap-5 values.  The seed-`20260723` stderr-only `solve_time_limit`
  notification remains an explicit unreplayed accounting discrepancy, not a
  basis for imputation/exclusion or parameter tuning.  A later read-only check
  used the wrong `.jsonl.summary.json` companion name and failed after the
  valid replay; it created no artifact and overwrote nothing.  This complete
  5-UAV artifact set is not a performance, safety, fallback-rate, scale, or
  method-effect result.  Next: verify this documentation commit, then start
  the independent 8-UAV uncertainty-aware predictive five-seed training
  protocol serially; defer ablations until its matched base artifacts exist.
- The frozen 5-UAV uncertainty-aware predictive checkpoint-evaluation matrix
  is active.  After confirming no training Python process and that all 30
  target directories were absent, a hidden serial launcher began all five
  final checkpoints (`20260719`--`20260723`) across `nominal`, `delay_only`,
  `loss_only`, `dynamic_only`, `combined`, and
  `ood_communication_obstacle`, 20 episodes per unique cell.  Each invocation
  uses `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
  scripts/evaluate_core_checkpoint.py --family graph_mappo --config
  configs/rl/dynamic_graph_5uav.yaml --checkpoint
  outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_<seed>/checkpoints/graph_mappo_final.pt
  --output-dir outputs/core_5uav_evaluations --experiment-name
  core_5uav_uncertainty_predictive_graph_seed_<seed>_<scenario> --seed <seed>
  --num-uavs 5 --episodes 20 --scenario <scenario> --device cuda`, serially.
  Launch revision is `bbb871a`; config SHA-256 is
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  CUDA is used for the learned network and OSQP CBF remains CPU-side.  The
  launcher logs are
  `outputs/core_5uav_evaluations/uncertainty_predictive_graph_5uav_5seed_eval_20260726_launcher_stdout.log`
  and `_launcher_stderr.log`.  Initial health evidence shows the first two
  completed unique cells for seed `20260719` (`nominal`, `delay_only`) and the
  active `multiuav_rl` CUDA evaluator process; stderr begins with a retained
  PowerShell CLIXML header.  This is a matrix-in-progress observation only; do
  not use early cells as results or start 8-UAV/ablation work.  On completion,
  audit exactly 30 cells and 600 raw JSONL records, retain failures/invalid
  attempts, then replay all CBF records under unchanged solver values.
- Commit `d3a3eee` records completed final 5-UAV
  uncertainty-aware predictive seed `20260723`.  Its checkpoint, summary,
  TensorBoard data, live telemetry, and dedicated launcher stdout/stderr are
  retained under
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260723`
  and sibling log paths.  It reached 100,160 transitions and 313 updates at
  launch revision `31ff7cb`, with frozen config SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Final telemetry retains one `solved inaccurate` training fallback in 20,032
  decisions (index 7,308 at the unchanged 20,000-iteration cap); initial and
  final evaluation telemetry each retain zero in 160 decisions.  Its stderr
  separately retains both a `solve_time_limit` notification and the `solved
  inaccurate` notification; do not merge or discard this unresolved
  event-accounting discrepancy.  No safety parameter changed.  The five
  independent 5-UAV training artifacts (seeds `20260719`--`20260723`) are now
  complete, but this does not support a scale/safety/fallback/performance
  conclusion.  Next: verify the documentation commit, then run the frozen
  six-scenario 20-episode checkpoint-evaluation matrix for all five seeds,
  retain and validate all raw JSONL, and replay all CBF evidence under the
  unchanged protocol before any 8-UAV or ablation task.
- Commit `de6f286` records completed fourth 5-UAV
  uncertainty-aware predictive seed `20260722`.  Its final checkpoint,
  summary, TensorBoard data, live telemetry, and dedicated launcher logs are
  retained in
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260722`
  and sibling launcher-log paths.  It reached 100,160 transitions and 313
  updates at launch revision `0310300` with frozen configuration SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Training telemetry retains two emergency fallback events in 20,032 CBF
  decisions: `maximum iterations reached` at index 4,167 and `solved
  inaccurate` at index 9,017, each at the unchanged 20,000-iteration cap;
  complete raw contexts are in `summary.json` and
  `live_training_telemetry.json`.  Initial and final evaluation telemetry each
  retain zero events in 160 decisions.  No safety setting changed.  This is
  seed-local diagnostic evidence only, not a performance, safety,
  scalability, or fallback-rate result.  Before acting, verify the commit and
  no Python process, then start final seed `20260723` serially with the same
  frozen command; do not begin evaluation, 8-UAV work, or ablations first.
- After confirming no prior Python process remained and target paths were
  absent, the sole active serial CUDA job is final 5-UAV uncertainty-aware
  predictive seed `20260723`, PID `9180` at its health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260723
  --num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260723`.
  It was launched at revision `31ff7cb` with configuration SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Its unique launcher logs are
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260723_launcher_stdout.log`
  and `_launcher_stderr.log`.  At health check it had 8,960 transitions,
  TensorBoard and live telemetry output, and no emergency events in 1,792
  training CBF decisions.  This is an in-progress observation, not a result.
  Do not start another job or change safety parameters.  On completion, verify
  final artifacts and every event, document the full five-seed training set,
  then run the frozen six-scenario 20-episode 5-UAV evaluation matrix before
  8-UAV work or ablations.
- During active seed `20260723` monitoring, the dedicated stderr log retained
  one `CBF filter emergency fallback: solve_time_limit` notification.  A
  subsequent raw-telemetry read at 42,560 transitions instead retained one
  `solved inaccurate` event at decision 7,308 (environment 1 step 13,
  obstacle `[22.0, 76.0, 30.0]`, unchanged 20,000-iteration cap,
  0.04082990001188591 seconds, primal residual 0.0002440071509879367, dual
  residual 0.0000122640447944475); its telemetry status counts were 8,511
  `solved` and one `solved inaccurate`.  Do not infer that the stderr
  `solve_time_limit` is the same event, or discard either record: preserve the
  raw stderr and
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260723/live_training_telemetry.json`
  (observed SHA-256
  `107BABDBC2A0D9AB89A101DF699826F5180771CE326A63C63266B409A6A80325`) and
  resolve the discrepancy only from final retained artifacts/replay.  Do not
  alter any solver or safety parameter.
- Commit `0310300` is pushed to `origin/codex/phase14-dynamic-world` and records completed third 5-UAV
  uncertainty-aware predictive seed `20260721`.  Its final checkpoint,
  summary, TensorBoard data, live telemetry, and dedicated launcher logs are
  retained under
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260721`
  and its sibling launcher-log paths.  It reached 100,160 transitions and 313
  updates at its launch revision `7c0fbc8` and the frozen 5-UAV configuration
  SHA-256 `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Its training telemetry retains two `solved inaccurate` emergency fallbacks
  in 20,032 decisions, both at the unchanged 20,000-iteration cap (indices
  4,753 and 6,341); complete contexts are in `summary.json` and
  `live_training_telemetry.json`.  Initial and final evaluation telemetry each
  have zero fallbacks in 160 decisions.  No safety setting changed.  This is
  seed-local diagnostic evidence only, not a performance, safety,
  scalability, or fallback-rate result.  Before acting, verify the commit and
  no remaining Python process, then start seed `20260722` serially using the
  identical command pattern; do not start any evaluation, 8-UAV, or ablation
  job first.
- After confirming no prior Python process remained and that all target paths
  were absent, the sole active serial CUDA job is 5-UAV uncertainty-aware
  predictive seed `20260722`, PID `43664` at its health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260722
  --num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260722`.
  It was launched at revision `0310300` with configuration SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Its unique launcher logs are
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260722_launcher_stdout.log`
  and `_launcher_stderr.log`.  At health check it had 960 transitions,
  TensorBoard and live telemetry output, and no emergency events in 192
  training CBF decisions; this is an in-progress observation, not a result.
  Do not start another training job or change any safety setting.  On
  completion, verify checkpoint/summary and retained events, document it, then
  run final seed `20260723` serially before any 5-UAV evaluation, 8-UAV run,
  ablation, or aggregate claim.
- Continued seed `20260722` monitoring retained one CBF emergency fallback:
  `maximum iterations reached` at decision 4,167, environment 0 step 3, with
  dynamic-obstacle center `[22.0, 56.0, 30.0]`, unchanged 20,000-iteration
  cap, 0.049038999975891784 seconds, primal residual 0.005824136094528052, and
  dual residual 0.0000491939606585269.  The full context remains in
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260722/live_training_telemetry.json`
  (observed SHA-256
  `D5718BBBCA5C5696C4756B340F10D73CB7664859CE3FED5EA9AD1247B28E6001` at
  33,920 transitions) and the retained stderr log.  Preserve the event and do
  not alter slack, iteration/solve-time limits, tolerances, or other safety
  settings; it is not a safety, performance, or fallback-rate conclusion.
- A second active-run event is retained at decision 9,017: `solved
  inaccurate`, environment 0 step 8, obstacle center `[22.0, 66.0, 30.0]`,
  unchanged 20,000-iteration cap, 0.051416900008916855 seconds, primal
  residual 0.0014887963841878208, and dual residual
  0.00000338793323094724.  At the later 52,160-transition observation,
  telemetry retains both events; the full raw context remains in the same
  telemetry file (observed SHA-256
  `1136A8182F233F664CE630E0AF1972FE6B773DD7F6E0193D1106572C0966B18A`) and
  stderr log.  This remains diagnostic-only and does not justify a solver or
  safety-margin change.
- Commit `7c0fbc8` is pushed to `origin/codex/phase14-dynamic-world`; it
  records the completed second 5-UAV uncertainty-aware predictive seed
  `20260720`.  Its final checkpoint, summary, TensorBoard data, live
  telemetry, and dedicated launcher stdout/stderr are retained under
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260720`
  and its sibling launcher-log paths.  It reached 100,160 transitions and 313
  updates.  Training telemetry retains three emergency fallbacks in 20,032
  CBF decisions: `solved inaccurate` at indices 14,582 and 14,652 (unchanged
  20,000-iteration cap), plus `solve_time_limit` at 17,555 (unchanged
  0.1-second limit); the full contexts are in `summary.json` and
  `live_training_telemetry.json`.  Initial and final evaluation telemetry each
  retain zero fallback events in 160 decisions.  No CBF safety parameter
  changed.  This is seed-local diagnostic evidence only, not a fallback-rate,
  safety, scalability, or method-effect result.
- After confirming no prior Python process remained and that all target paths
  were absent, the sole active serial CUDA job is 5-UAV uncertainty-aware
  predictive seed `20260721`, PID `38832` at its initial health check.
  Command: `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
  scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_5uav.yaml
  --device cuda --seed 20260721 --num-uavs 5 --graph-mode
  uncertainty_predictive_graph --total-steps 100000 --output-dir
  outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260721`.
  It was launched at revision `7c0fbc8` with configuration SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Its unique launcher logs are
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260721_launcher_stdout.log`
  and `_launcher_stderr.log`.  At the health check it had 1,920 transitions,
  live telemetry, and TensorBoard output; its 384 training CBF decisions had
  no emergency events.  This is an in-progress observation, not a result.
  Do not start any other training job, change CBF parameters, or begin 5-UAV
  evaluation/8-UAV/ablation work.  On completion, verify its final checkpoint
  and summary, document every retained event, then continue serially with
  seeds `20260722` and `20260723`.
- During continued seed `20260721` monitoring, live telemetry retained one CBF
  emergency fallback: `solved inaccurate` at decision 4,753, environment 0
  step 16, with the dynamic obstacle center `[22.0, 82.0, 30.0]`, unchanged
  20,000-iteration cap, 0.05340159998741001 seconds, primal residual
  0.0026409692851873355, and dual residual 0.00011929081440663525.  The full
  context remains in
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260721/live_training_telemetry.json`
  (observed SHA-256
  `03B7AA1FF5720593FA19401B4461767CEA9F865DD986EBF28294479D5B4AF4B7` at
  29,440 transitions) and the retained stderr log.  This active-run event is
  diagnostic only; do not alter slack, iteration limit, solve-time limit,
  tolerances, or any other CBF setting in response.
- Continued monitoring retained a second `solved inaccurate` emergency
  fallback at decision 6,341, environment 0 step 10, with obstacle center
  `[22.0, 70.0, 30.0]`, the unchanged 20,000-iteration cap,
  0.028405999997630715 seconds, primal residual 0.001457085484388434, and dual
  residual 0.0000303695102194086.  At the later 43,200-transition observation
  the telemetry retains two such events; complete contexts remain in the same
  raw telemetry file (observed SHA-256
  `1F73AE59A97741F8D5D5A393544E05513A4092CA8F81367DDEBAECAC795858FD`) and
  stderr log.  Preserve them without solver changes; this still provides no
  performance, safety, or fallback-rate conclusion.

## Current continuation update (2026-07-23, supersedes stale status below)

- Commit `8899c15` is pushed to `origin/codex/phase14-dynamic-world`; it
  records the first completed 5-UAV uncertainty-aware predictive seed
  `20260719`.  Its final checkpoint, summary, TensorBoard data, and live
  telemetry are retained under
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260719`.
  It reached 100,160 transitions and 313 updates; retained emergency-fallback
  counts are zero in 20,032 training decisions, 160 initial-evaluation
  decisions, and 160 final-evaluation decisions.  This is one seed-local
  diagnostic artifact, not a safety, performance, scalability, or
  fallback-rate result.  Its intended launcher stdout/stderr files remain
  absent and must not be recreated; live telemetry and final artifacts remain
  the available audit evidence.
- After confirming no Python training process remained, the sole active serial
  CUDA job is 5-UAV uncertainty-aware predictive seed `20260720`, PID `22948`
  at its initial health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260720
  --num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260720`.
  It was launched at revision `8899c15` with configuration SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Its unique launcher logs are present at
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260720_launcher_stdout.log`
  and `_launcher_stderr.log`.  Do not start another training job.  On
  completion, verify summary/checkpoint and every CBF event, document the seed,
  then continue serially through seeds `20260721`--`20260723` before 5-UAV
  evaluation, 8-UAV work, or ablation arms.
- During active seed `20260720` monitoring, live telemetry and dedicated stderr
  retained three CBF emergency events: `solved inaccurate` at decision indices
  14,582 and 14,652 (each at the unchanged 20,000-iteration cap), then
  `solve_time_limit` at decision 17,555 (0.1006264 seconds and 11,265
  iterations).  The process remained live; preserve all recorded contexts and
  final summary as authoritative.  Do not change slack, iteration cap,
  solve-time limit, tolerances, or any other safety setting in response.
- Commit `7d6486e` is pushed to `origin/codex/phase14-dynamic-world`; it
  freezes validated 5-UAV and 8-UAV uncertainty-aware predictive scale
  configurations.  The matched 3-UAV five-seed protocol is complete for MLP,
  raw-packet graph, predictive-without-uncertainty graph, and uncertainty-aware
  predictive graph: every arm has 30 unique 20-episode evaluation cells and
  retained raw JSONL.  The final uncertainty-aware all-source replay is
  `outputs/cbf_diagnostics/uncertainty_predictive_graph_5seed_replay_20260726.jsonl`
  plus `.summary.json`; it has five source training events and zero replay
  errors under unchanged solver values.  This is raw protocol evidence only,
  not a cross-method, safety, fallback-rate, or performance conclusion.
- After confirming no Python process remained, the sole active serial CUDA
  training job is the first independent 5-UAV uncertainty-aware predictive
  seed `20260719`, PID `17372` at its initial health check.  Command:
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_5uav.yaml --device cuda --seed 20260719
  --num-uavs 5 --graph-mode uncertainty_predictive_graph --total-steps 100000
  --output-dir
  outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260719`.
  It was launched at revision `7d6486e` with frozen configuration SHA-256
  `28DCB568D23FF993E5514D7E748DB249D4E68ED8A3523F3CDACA8AC223D27C32`.
  Its unique launcher logs are
  `outputs/core_5uav/core_5uav_uncertainty_predictive_graph_seed_20260719_launcher_stdout.log`
  and `_launcher_stderr.log`; output directory exists but no final summary or
  checkpoint yet.  Do not start another training job.  On completion, verify
  its checkpoint/summary and all CBF events, document the seed, then proceed
  serially through the remaining 5-UAV seeds before 5-UAV evaluation, 8-UAV
  work, or any independent ablation arm.
- The intended 5-UAV launcher stdout/stderr redirection paths had not
  materialized at the first health checks, while the live `multiuav_rl` Python
  process and its output-directory `live_training_telemetry.json` were present.
  This is a retained launch-observability gap, not grounds to restart or
  overwrite the active job.  Preserve its available output telemetry and record
  the final summary/checkpoint or any failure state exactly as produced.
- Commit `b09be13` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion evidence for uncertainty-aware predictive seed
  `20260722`.  Its final checkpoint and `summary.json` are present under
  `outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260722`.
  Final telemetry retains one training emergency fallback in 33,344 decisions
  (`solve_time_limit` at decision 681 under the unchanged 0.1-second limit),
  zero in 160 initial-evaluation decisions, and zero in 122 final-evaluation
  decisions.  The full event context is retained in the summary; this is
  seed-local diagnostic evidence, not a performance, safety, or cross-seed
  fallback-rate result.
- After confirming no Python training process remained, the sole active serial
  CUDA job is final seed `20260723`, PID `66004` at its initial health check.
  Its unchanged command is `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
  scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_baseline.yaml
  --device cuda --seed 20260723 --num-uavs 3 --graph-mode
  uncertainty_predictive_graph --total-steps 100000 --output-dir
  outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260723`.
  Its dedicated launcher logs are
  `outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260723_launcher_stdout.log`
  and `_launcher_stderr.log`; the output directory exists but final summary and
  checkpoint do not yet exist.  Do not start another training job.  On
  completion, verify final artifacts and all retained CBF events, document the
  fifth seed, then run the frozen six-scenario 20-episode evaluation matrix
  for all five checkpoints before any aggregate claim or CBF replay.
- During active seed `20260723` monitoring, its dedicated stderr log recorded
  `CBF filter emergency fallback: solve_time_limit`.  The process remained
  live after the notification.  Preserve the log and final summary as the
  authoritative full event record; do not change the solve-time limit, slack,
  iteration cap, tolerances, or any other safety setting.
- Commit `195d03f` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion evidence for uncertainty-aware predictive seed
  `20260721`.  Its final checkpoint and `summary.json` are present under
  `outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260721`.
  The retained final telemetry has one training emergency fallback in 33,344
  decisions (`maximum iterations reached` at decision 4,994 and the unchanged
  20,000-iteration cap), zero in 160 initial-evaluation decisions, and zero in
  160 final-evaluation decisions.  The summary preserves full event context;
  these are seed-local diagnostics only, not performance, safety, or
  cross-seed fallback-rate results.
- After confirming no Python training process remained, the sole active serial
  CUDA job is seed `20260722`, PID `68408` at its initial health check.  Its
  unchanged command is `D:\\anaconda3\\envs\\multiuav_rl\\python.exe
  scripts/train_graph_mappo.py --config configs/rl/dynamic_graph_baseline.yaml
  --device cuda --seed 20260722 --num-uavs 3 --graph-mode
  uncertainty_predictive_graph --total-steps 100000 --output-dir
  outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260722`.
  Its dedicated launcher logs are
  `outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260722_launcher_stdout.log`
  and `_launcher_stderr.log`; output directory exists, but no final summary or
  checkpoint yet.  Do not start another training job.  On completion, verify
  the final checkpoint/summary, document all CBF events, then start
  `20260723` serially with the same frozen protocol.
- During active seed `20260722` monitoring, its dedicated stderr log recorded
  `CBF filter emergency fallback: solve_time_limit`.  The training process
  remained live; preserve the log and final summary as the authoritative full
  event record.  Do not respond by changing the solve-time limit, slack,
  iteration cap, tolerances, or any other safety parameter.
- Commit `23c9da0` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion evidence for uncertainty-aware predictive seed
  `20260720`.  Its final checkpoint and `summary.json` are present under
  `outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260720`.
  The retained telemetry has 1 emergency fallback in 33,344 training CBF
  decisions (`solved inaccurate` at the unchanged 20,000-iteration cap), zero
  in 160 initial-evaluation decisions, and zero in 154 final-evaluation
  decisions.  This is seed-local diagnostic evidence only, not a performance,
  safety, or cross-seed fallback-rate result.
- After a fresh `Get-Process python,pythonw` check found no remaining training
  process, the sole active serial CUDA job is seed `20260721`, PID `66668` at
  the initial health check.  Its unchanged command is
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260721 --num-uavs 3 --graph-mode uncertainty_predictive_graph --total-steps
  100000 --output-dir
  outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260721`.
  Unique launcher logs are
  `outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260721_launcher_stdout.log`
  and `_launcher_stderr.log`; the output directory exists but there is not yet
  a final summary or checkpoint.  Do not start another training job.  On
  completion, verify the final checkpoint and summary, document every retained
  CBF fallback before launching seed `20260722`, then repeat for `20260723`.
- During active seed `20260721` monitoring, its dedicated stderr log recorded
  `CBF filter emergency fallback: maximum iterations reached`.  The process
  remained live after that notification; retain the log and final summary as
  authoritative for the complete count and context.  Do not change the
  iteration cap, tolerances, slack, or any other safety parameter in response.
- Commit `d0de541` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion of uncertainty-aware predictive seed `20260719`.  The
  sole active serial CUDA job is seed `20260720`, PID `18960` at launch,
  command
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260720 --num-uavs 3 --graph-mode uncertainty_predictive_graph --total-steps
  100000 --output-dir
  outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260720`.
  Its unique launcher logs end in `_seed_20260720_launcher_stdout.log` and
  `_launcher_stderr.log` under `outputs/core_3uav/`.  Verify final checkpoint
  and summary before acting, document each retained CBF event, then start
  `20260721`, `20260722`, and `20260723` serially with this unchanged protocol.
- Commit `e857715` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion of the predictive-without-uncertainty five-seed training,
  30-cell evaluation, JSONL-aware CBF replay, and explicit one-event telemetry
  gap.  The sole active serial CUDA job is the first uncertainty-aware
  predictive-graph seed `20260719`, PID `64184` at launch, command
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260719 --num-uavs 3 --graph-mode uncertainty_predictive_graph --total-steps
  100000 --output-dir
  outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260719`.
  Its unique launcher logs are
  `outputs/core_3uav/core_3uav_uncertainty_predictive_graph_seed_20260719_launcher_stdout.log`
  and `_launcher_stderr.log`.  Confirm final checkpoint/summary, document every
  CBF event, then run seeds `20260720`--`20260723` serially before any
  six-scenario evaluation.  Do not use the raw-packet/predictive matrices for
  a cross-method conclusion until this matched arm also completes.
- Commit `87e6dd6` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion of predictive-without-uncertainty seed `20260722`.  The
  sole active serial CUDA job is the fifth/final predictive-without-uncertainty
  seed `20260723`, PID `57804` at launch, command
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260723 --num-uavs 3 --graph-mode predictive_graph --total-steps 100000
  --output-dir outputs/core_3uav/core_3uav_predictive_graph_seed_20260723`.
  Its unique launcher logs are
  `outputs/core_3uav/core_3uav_predictive_graph_seed_20260723_launcher_stdout.log`
  and `_launcher_stderr.log`.  Verify final checkpoint/summary and preserve all
  CBF contexts before documenting completion.  Then run the six fixed
  20-episode scenarios for all five final checkpoints (unique directory per
  seed/scenario), replay all retained CBF events under unchanged solver values,
  and only then begin `uncertainty_predictive_graph`.
- Commit `69c79a3` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion of predictive-without-uncertainty seed `20260721`.  The
  sole active serial CUDA job is seed `20260722`, PID `60716` at launch,
  command
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260722 --num-uavs 3 --graph-mode predictive_graph --total-steps 100000
  --output-dir outputs/core_3uav/core_3uav_predictive_graph_seed_20260722`.
  Its unique launcher logs are
  `outputs/core_3uav/core_3uav_predictive_graph_seed_20260722_launcher_stdout.log`
  and `_launcher_stderr.log`.  Verify its final checkpoint and summary before
  acting, document all retained CBF events, then start `20260723` serially with
  the same frozen protocol.
- Commit `035552f` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion of predictive-without-uncertainty seed `20260720`.
  The one active serial CUDA job is seed `20260721`, PID `62592` at the first
  health check, command
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260721 --num-uavs 3 --graph-mode predictive_graph --total-steps 100000
  --output-dir outputs/core_3uav/core_3uav_predictive_graph_seed_20260721`.
  Its dedicated launcher logs end in `_seed_20260721_launcher_stdout.log` and
  `_launcher_stderr.log` under `outputs/core_3uav/`; retain rather than
  overwrite them.  On completion, verify final checkpoint/summary and record
  the retained CBF events before starting seed `20260722`, then `20260723`.
- Commit `024d537` is pushed to `origin/codex/phase14-dynamic-world`; it
  records the completed first predictive-without-uncertainty GraphMAPPO seed.
  The active serial CUDA job is now seed `20260720`, PID `28928` at the last
  health check, launched with
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260720 --num-uavs 3 --graph-mode predictive_graph --total-steps 100000
  --output-dir outputs/core_3uav/core_3uav_predictive_graph_seed_20260720`.
  Its unique launcher logs are
  `outputs/core_3uav/core_3uav_predictive_graph_seed_20260720_launcher_stdout.log`
  and `_launcher_stderr.log`.  Confirm the final checkpoint and summary before
  acting; do not overwrite those paths.  After documenting completion, launch
  `20260721` serially, then `20260722`, then `20260723` using the identical
  frozen protocol.
- Commit `faca19a` is pushed to `origin/codex/phase14-dynamic-world`; it
  records completion of the raw-packet GraphMAPPO five-seed 3-UAV matrix and
  CBF replay.  The user's `.gitignore` is the only intended dirty worktree
  file; never stage or alter it.
- Raw-packet GraphMAPPO seeds `20260719`--`20260723` are all complete at
  100,032 CUDA transitions.  Their six-scenario, 20-episode checkpoint
  evaluations are complete: 30 valid directories and 600 retained JSONL
  records under `outputs/core_3uav_evaluations/core_3uav_raw_graph_seed_*`.
  `outputs/cbf_diagnostics/raw_graph_5seed_replay_20260723.jsonl` retains all
  seven source events from the five training summaries; its companion summary
  records zero replay errors under the unchanged 20,000-iteration, 0.1-second,
  penalty-100, uncertainty-gain-0.5/cap-5 protocol.  Do not make a performance,
  safety, or fallback-rate claim from these within-method records.
- The next matched arm, predictive-without-uncertainty GraphMAPPO, has one
  active serial CUDA job.  At handoff its PID is `43872`, command
  `D:\\anaconda3\\envs\\multiuav_rl\\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260719 --num-uavs 3 --graph-mode predictive_graph --total-steps 100000
  --output-dir outputs/core_3uav/core_3uav_predictive_graph_seed_20260719`.
  Check the process and final checkpoint/summary before acting.  Its independent
  launcher logs are
  `outputs/core_3uav/core_3uav_predictive_graph_seed_20260719_launcher_stdout.log`
  and `_launcher_stderr.log`; do not overwrite them.  Once complete, document
  seed-local artifact and retained fallbacks, then run seeds `20260720`--
  `20260723` serially with the same command pattern.  Only after the five-seed
  training and six-scenario evaluation matrix is complete may the
  `uncertainty_predictive_graph` arm begin.

## Current continuation update (2026-07-22, supersedes stale status below)

- Commit `1a7800ccb383943a8d160d41856f1e1ff65a23e5` is pushed to
  `origin/codex/phase14-dynamic-world`.  It records the completed MLP protocol,
  CBF audit, and `scripts/replay_cbf_fallbacks.py`.  The only remaining dirty
  file is the user's unstaged `.gitignore`; never stage or alter it.
- MLP seeds 20260722 and 20260723 are complete at 100,032 CUDA transitions.
  Their training/initial/final CBF fallback counts are respectively 1/13/0 and
  0/0/0.  All five MLP final checkpoints now have all six 20-episode scenario
  evaluations: 30 valid directories and 600 raw JSONL records.  Seed-20260719
  must use the `nominal_schemafix_rngfix_rerun2` nominal and
  `ood_communication_obstacle_trajectoryfix_rerun1` OOD reruns; its three
  earlier failed directories remain retained and excluded.
- Formal MLP summaries retain 47 CBF fallback events.  They were replayed under
  unchanged mainline solver values into
  `outputs/cbf_diagnostics/core_mlp_5seed_replay_20260722.jsonl`, all without
  context errors.  Historical diagnostics have a separate 21-event replay
  JSONL with three explicit nonreplayable legacy records; do not impute them.
- The first matching raw-packet GraphMAPPO job is currently active and must be
  checked before taking action: PID `36556` at the last check, command
  `D:\anaconda3\envs\multiuav_rl\python.exe scripts/train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260719 --num-uavs 3 --graph-mode distance_graph --total-steps 100000
  --output-dir outputs/core_3uav/core_3uav_raw_graph_seed_20260719`.  At the
  last health check it had 39,648 transitions and zero training/initial/interval
  fallbacks.  Keep Graph jobs serial; complete raw graph five seeds and six
  scenarios, then predictive-without-uncertainty, then uncertainty-aware graph.
- Superseding the preceding raw-graph status: seed `20260719` completed at
  100,032 transitions and produced `graph_mappo_final.pt` at commit
  `231cb8f322031db6aba8fabe41b5ddb35f3ef74e`, with training/initial/final CBF
  fallbacks 1/0/0.  Its phase record is in `docs/progress.md`.  Seed `20260720`
  is now the sole active CUDA job in
  `outputs/core_3uav/core_3uav_raw_graph_seed_20260720`; verify its process and
  final summary before acting, then document it before starting seed 20260721.
- Superseding the preceding seed-20260720 status: it completed at 100,032
  transitions with training/initial/final CBF fallbacks 0/0/0.  Seed `20260721`
  is the sole active CUDA job in
  `outputs/core_3uav/core_3uav_raw_graph_seed_20260721`; the last launch PID
  was `57388`.  Confirm its final checkpoint and summary, document completion,
  and retain all fallback telemetry before serial seed `20260722` starts.
- Superseding the preceding seed-20260721 status: it completed at 100,032
  transitions with CBF fallbacks 1/0/0 (training/initial/final).  Seed
  `20260722` is now the only active raw-graph CUDA job in
  `outputs/core_3uav/core_3uav_raw_graph_seed_20260722`; verify final artifacts,
  document the phase, then continue serially with seed `20260723`.
- Superseding the preceding seed-20260722 status: it completed at 100,032
  transitions with CBF fallbacks 3/0/0 (training/initial/final).  Seed
  `20260723` is now the only raw-graph CUDA job in
  `outputs/core_3uav/core_3uav_raw_graph_seed_20260723`; verify final artifacts
  and document it before beginning the raw-graph six-scenario evaluations.
- Primary-paper browsing remains pending one-time local setup authorization:
  the required `/browse` skill reported `NEEDS_SETUP` for its browser binary.
  Do not substitute another browser or use metadata-only entries for novelty
  claims.  Once authorized, run the skill's setup and record only primary-source
  evidence in `docs/related_work_matrix.md`.
- Verified before the commit: Ruff and mypy for the new replay utility passed;
  full pytest passed 147 tests with 28 existing OSQP deprecation warnings.

## Fixed research mainline

> Communication staleness and packet loss create uncertainty about neighbour
> state.  Predictive interaction-graph features use that uncertainty to change
> distributed multi-UAV decisions, while an uncertainty-adaptive CBF margin
> provides execution-side safety in dynamic-obstacle worlds.

Working title: *Uncertainty-Calibrated Predictive Interaction Graphs for Safe
Multi-UAV Coordination under Intermittent Communication*.

Do not turn BC, hierarchy, dynamic obstacles, communication impairment, and CBF
into disconnected innovations.  BC/CA-HGALO may only be initialization or
auxiliary comparisons unless a verifiable original implementation exists.

## Repository and safety rules

- Repository: `D:\yolo\multiuav\reinforcement_learning`
- Branch: `codex/phase14-dynamic-world`
- Remote: `github.com:youshenbupo/multi-uav-rl-path-planning.git`
- CUDA is for learned networks; OSQP CBF remains CPU-side.
- Actor input: own truth plus delivered packets only.  Central critic may use
  true state.  Do not leak current neighbour truth to the actor.
- Do not delete aborted experiments, incomplete evaluator attempts, or raw
  JSONL.  Never hide a CBF fallback by changing slack, iteration budget, or
  tolerance without an explicit new protocol and a full report.
- `.gitignore` currently has a user-owned unstaged edit.  Never stage it unless
  the user explicitly asks.

## Completed and verified engineering work

- Communication packets carry raw delivered state, predicted state, age, valid
  mask, and uncertainty.  Predictive graph edges use predicted relative state,
  CPA, age, validity, and uncertainty.
- The simulated centralized CBF uses true geometry but tightens margins from
  packet uncertainty.  It is a risk-aware execution shield, **not** a
  decentralized safety guarantee.
- Four comparison arms exist: MLP-MAPPO, ordinary graph MAPPO, predictive graph
  without uncertainty, and uncertainty-aware predictive graph.
- Uniform checkpoint evaluation writes per-episode JSONL and supports exactly
  six scenarios: nominal, delay-only, loss-only, dynamic-only, combined, and
  `ood_communication_obstacle`.
- Checkpoint evaluator fixes are committed: dynamic-trained checkpoint feature
  layout remains fixed in non-dynamic scenarios, centralized critic state has a
  fixed obstacle capacity, and CUDA map-location restores RNG tensors on CPU.
- OOD 1.5x obstacle speed now has a valid 20-step trajectory.  Base training
  semantics remain y=50 to y=90; OOD is faster y=30 to y=90.
- Latest full verification before this handoff: Ruff and mypy passed; `147
  passed` pytest tests, with only 28 OSQP dependency deprecation warnings.

## Formal experiment status

### MLP 3-UAV training

| Seed | Status | Training CBF fallbacks / decisions | Initial / final evaluation fallbacks |
| --- | --- | ---: | ---: |
| 20260719 | complete, 100,032 transitions | 1 / 33,344 | 16 / 0 |
| 20260720 | complete, 100,032 transitions | 1 / 33,344 | 1 / 0 |
| 20260721 | complete, 100,032 transitions | 3 / 33,344 | 11 / 0 |
| 20260722 | not started | — | — |
| 20260723 | not started | — | — |

The first three are retained diagnostics, not performance evidence.  Seed
20260721 replay reached `solved inaccurate` at the 20,000-iteration cap with
zero slack and emergency fallback.  Seed 20260720 replay solved but used about
504 slack.  Keep these failure modes separate in later reporting.

### Valid unified evaluation artifacts so far

- Seed 20260719: valid nominal result is
  `..._nominal_schemafix_rngfix_rerun2`; delay-only, loss-only, dynamic-only,
  combined, and valid OOD (`..._trajectoryfix_rerun1`) exist.
- Earlier seed-20260719 nominal and OOD evaluator directories are incomplete
  failed attempts and must be excluded from aggregation.
- Seed 20260720: nominal evaluation exists.
- Seed 20260721: no unified six-scenario evaluation yet.

Some single-seed MLP conditions have zero success despite training-script short
evaluation success.  This is a research risk, not a conclusion: run all five
seeds and compare raw records before changing the method or claiming anything.

## Immediate next actions

1. Confirm no training process is alive.  Start MLP seeds `20260722` and
   `20260723` serially at 100,000 transitions with the same CUDA command
   pattern, retaining live telemetry.
2. Evaluate every completed MLP final checkpoint in every missing scenario with
   `scripts/evaluate_core_checkpoint.py`, 20 episodes each, and keep attempt
   identities unique.  Do not overwrite failed artifact directories.
3. Diagnose and summarize every training/evaluation CBF fallback with
   `scripts/diagnose_cbf_failure.py`; do not tune solver values merely to reduce
   counts.
4. Only after the MLP five-seed data are complete, start the three GraphMAPPO
   arms with the same seed/scenario/budget.  Then extend the four-arm protocol
   to 5 and 8 UAV.  Do not claim a baseline comparison from partial rows.
5. Train key ablations independently, never by disabling a module in a shared
   checkpoint.
6. Continue primary-source literature inspection.  The evidence ledger is
   `docs/related_work_matrix.md`; entries marked metadata-only cannot support
   method-difference claims.

## Mandatory documentation protocol

At the end of **every phase-level task**, create or update an in-repository
document before reporting completion.  The update must name:

1. exact artifact paths, commands/configuration, seed(s), and Git revision;
2. verification evidence and raw-result locations;
3. what is proven, what is not proven, and every failure/invalid attempt;
4. remaining risks and the concrete next action.

Use `docs/progress.md` for chronological verified progress,
`docs/known_issues.md` for unresolved risks and exclusions,
`docs/related_work_matrix.md` for literature evidence, and create a dedicated
handoff document when a conversation is ending.  Commit and push documentation
with the corresponding code/protocol change, excluding user-owned unrelated
files.

## Forbidden claims until evidence exists

- Do not describe smoke, short, or single-seed results as performance results.
- Do not claim superiority, significance, safety guarantees, real-time
  capability, or robust generalization before the prescribed five-seed data.
- Do not call a CA-HGALO substitute a source-faithful baseline.
- Do not imply the project is first to combine graph, delay, uncertainty, or
  CBF components until primary-paper comparisons support that claim.

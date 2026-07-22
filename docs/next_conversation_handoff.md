# AAMAS 2027 continuation handoff

**Snapshot date:** 2026-07-22.  This document is the authoritative starting
point for the next Codex conversation.  Inspect the worktree and running
processes before relying on any status below.

## Current continuation update (2026-07-23, supersedes stale status below)

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

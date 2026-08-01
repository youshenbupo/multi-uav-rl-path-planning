# AAMAS 2027 continuation handoff

**Snapshot date:** 2026-07-22.  This document is the authoritative starting
point for the next Codex conversation.  Inspect the worktree and running
processes before relying on any status below.

## Latest continuation update (2026-07-30, supersedes stale status below)

- **Second valid post-isolation 8-UAV full-method seed retained, 2026-08-01
  (launch revision `d0f1484`; documentation pending commit):** independent
  seed `20260720` ran once in
  `outputs/core_8uav_post_actor_isolation/core_8uav_uncertainty_predictive_graph_seed_20260720/`
  under the same frozen configuration, real-child-exit and wait-only protocol.
  It naturally exited 0 after about 335.9 seconds at 100,224 transitions / 261
  updates. Identity is eight UAVs, uncertainty-aware predictive graph, CUDA
  neural execution and CPU OSQP.

  Final checkpoint SHA-256 is
  `C0A52742DD3F80BF1DDB76E13EC84FBF861813E28A6832C0B80193E6201E7847`;
  summary SHA-256 is
  `B1B71629CAABF879121015E829CB7BC9A7FF334D6E77921A8CF2F13A96C415F2`;
  telemetry SHA-256 is
  `8882B7761396C836D16847F2D910A2C254F2D362E24D2D9F8E4044B6EDC37ACF`.
  Training retains one `maximum iterations reached` event over 12,528
  decisions. Of 32 append-only intervals, transition 92,160 retains one
  `solved inaccurate` event; the other 31, initial and final evaluations are
  zero-event. The two sources exactly match stderr.

  Replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_seed_20260720_replay_20260801.jsonl`
  plus summary has two sources and zero errors under unchanged
  20,000/0.1s/100/0.5/5.0 values. Maximum iterations remains a fallback; the
  interval solved-inaccurate event replays `solved` without fallback with slack
  about 503.416. JSONL SHA-256 is
  `27ED2A4FFEDA74AC1D71AB136307060A6279A5A2196CD811EBCA4A5882519068`;
  summary SHA-256 is
  `3EE6B9F5FDB660FDD5FDB224510AF0D4DD66CB2AF7F19B3B1AB33BB43E5E16D8`.
  No CBF value changed. Two seeds establish no scale, safety, fallback-rate,
  performance or generalization result.

  Next: commit/push these four documents without `.gitignore`, then preflight
  and run seed `20260721` under the identical frozen 8-UAV command. Three more
  valid full-method seeds, their matrix and all-source replay remain before the
  independently trained 8-UAV no-uncertainty arm.

- **First valid post-isolation 8-UAV full-method seed retained, 2026-08-01
  (launch revision `d44fbf8`; documentation pending commit):** after confirming
  all historical 8-UAV roots remain excluded and no post-isolation root existed,
  independent seed `20260719` ran once in
  `outputs/core_8uav_post_actor_isolation/core_8uav_uncertainty_predictive_graph_seed_20260719/`.
  `Start-Process` retained the real child exit code; the wait-only CUDA run
  naturally exited 0 after about 337.4 seconds at 100,224 transitions / 261
  updates. Identity is eight UAVs, `uncertainty_predictive_graph`, prediction
  and uncertainty enabled, and CUDA; OSQP remained CPU-side.

  Final checkpoint SHA-256 is
  `5037D298FE85832EC94A3BA4F400D614B623DE9E27E59F91BDA3EADDBCB1E334`;
  summary SHA-256 is
  `F65A7F8666237001BE24A441C3C04C838C23A663BF3989F271359359F1737B18`;
  telemetry SHA-256 is
  `204E143E3D75B3F6E8631C9A1AEBBC0728D2E5952C11E4A7EF8D01D854544898`.
  Training retains two `maximum iterations reached` and two `solved inaccurate`
  events over 12,528 decisions. Initial/final evaluations are zero-event over
  160/160 decisions. All 32 interval evaluations are append-only; transition
  55,296 retains one `solve_time_limit` event and the other 31 are zero-event
  over 80 decisions each. The five retained sources exactly match five raw
  stderr warnings.

  Replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_8uav_seed_20260719_replay_20260801.jsonl`
  plus summary has five sources and zero errors under unchanged
  20,000/0.1s/100/0.5/5.0 values. Both maximum-iterations sources remain
  fallbacks; the two solved-inaccurate sources and interval time-limit source
  replay `solved` without fallback. Replay JSONL SHA-256 is
  `23D491F1D5C0A4495E857A88786D64FE942BBB350E6A14738897CDB4029BE12D`;
  summary SHA-256 is
  `656407B82BD09AF3BBF71CBAB4F4D8A386C00E064A13B835CE0103EBD7E08253`.
  No CBF value changed. This is first-seed provenance only, not a scale,
  safety, fallback-rate, performance or generalization result.

  Next: commit/push these four documents without `.gitignore`, then perform a
  zero-process/config-hash/absent-path preflight and run seed `20260720` under
  the identical frozen 8-UAV full-method command and wait-only discipline.
  Four more valid seeds, the 30-cell matrix and all-source replay remain before
  independently trained 8-UAV no-uncertainty work.

- **Post-isolation 5-UAV independent no-uncertainty evidence gate is complete,
  2026-08-01 (matrix launch revision `77b602d`; documentation pending
  commit):** the retained launcher
  `outputs/core_5uav_post_actor_isolation_predictive_graph_evaluation_launcher_20260801.ps1`
  (SHA-256
  `505E353E04F25FFE610FA01731F470ECA772A196FE26AC0998BED1358750C9BF`)
  naturally exited 0 after about 387.4 seconds. It created the unique root
  `outputs/core_5uav_post_actor_isolation_evaluations_20260801_predictive_graph/`
  and serially evaluated eligible independent seeds `20260719`--`20260723`
  across the six canonical scenarios with 20 episodes per cell.

  Independent content audit verified 30 directories, 30 summaries, 30 raw
  JSONL files, 30 runtime telemetry files, 30 environment files and exactly 600
  records. Seed, scenario, episode indices 0--19, eligible checkpoint,
  `graph_mappo_checkpoint` controller and completed step 100,080 all match.
  Every environment record confirms five UAVs, CUDA,
  `own_truth_and_delivered_packets_only` and `cpu_osqp_when_enabled`. All 30
  cell exit codes are zero; all cell stderr files and launcher stderr are empty.
  Evaluation telemetry contains 11,680 CBF decisions and zero emergency events.
  Launcher stdout SHA-256 is
  `7225C125BBFFCF9513BC7A2AD5D16740FDBB0A0CA3151761FB0D79E64E844806`;
  empty stderr SHA-256 is
  `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`.

  The append-safe 35-input replay is
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_5seed_training_and_evaluation_replay_20260801.jsonl`
  plus companion summary. It uses exactly five eligible training telemetry
  files and 30 valid evaluation telemetry files under unchanged
  20,000/0.1s/100/0.0/0.0 CPU-OSQP values. It retains 15 source events, zero
  replay errors and three replay fallbacks. Recorded statuses are three
  `solved inaccurate`, three `maximum iterations reached` and nine
  `solve_time_limit`; replay statuses are 12 `solved`, one `solved inaccurate`
  and two `maximum iterations reached`. JSONL SHA-256 is
  `E15440B4142BF3E95B42B91CE62564DB391E8EA350AF7FD5FC4114238EEA8C6B`;
  summary SHA-256 is
  `9D5EE189C3F32A3F1A88A64F7AF6F87EEFBF7073726B30308A459E65A6A97062`.
  No CBF value changed.

  This closes the post-isolation 5-UAV no-uncertainty training, evaluation and
  CBF provenance gate only. It does not establish a causal method effect,
  scale, safety, fallback-rate, performance or generalization result, and no
  paired/descriptive analysis has yet been specified. Next: commit/push the
  four updated documents without `.gitignore`, then audit frozen 8-UAV configs,
  historical exclusions, eligible target absence and CUDA. Start the
  post-isolation 8-UAV full uncertainty-aware arm from scratch, five independent
  seeds followed by its matrix/replay, before the independently trained 8-UAV
  no-uncertainty arm.

- **Post-isolation 5-UAV independent no-uncertainty five-seed training is
  complete, 2026-08-01 (final seed launch revision `a304367`; documentation
  pending commit):** seed `20260723` ran once in the unique
  `outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260723/`
  root. The wait-only execution cell reported code 1 after about 722.2 seconds
  because PowerShell 5 converts redirected native stderr into a
  `NativeCommandError`; a no-state probe using Python `sys.exit(0)` reproduced
  the same host-code behavior. The training itself completed normally: stdout
  is a complete parseable final summary, and telemetry, summary and final
  checkpoint were atomically present at 100,080 transitions / 417 updates with
  no `ABORTED.json`. Treat the host code as a documented launcher-envelope
  artifact, not a failed or duplicate seed attempt.

  Identity is five UAVs, `predictive_graph`, prediction true, uncertainty false
  and CUDA; OSQP remained CPU-side. Final checkpoint SHA-256 is
  `FE76A928143EE956176B3A627C4EBB632CDC5FC8BED5142DB93B365B3FD14C22`;
  summary SHA-256 is
  `18279BBC231CAF4EA75CF20DD6A4047C474359BD92381E1CD3CE5D6EF62C9D9C`;
  telemetry SHA-256 is
  `F8CCA2EEED98AD89EC3FE8E13C4A08798147697CD0B422AD8DDAE59B4838151F`.
  Training retains one `solved inaccurate` and one `maximum iterations reached`
  event over 20,016 decisions. Initial/final and all six interval evaluations
  are zero-event over 160/160 and 80 decisions per interval. The two messages
  are both preserved in stderr; the first is inside PowerShell's error envelope
  and the second is plain stderr.

  Replay
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260723_replay_20260801.jsonl`
  plus summary has two sources and zero errors under unchanged
  20,000/0.1s/100/0.0/0.0 values. The solved-inaccurate source replays `solved`
  without fallback (slack about 10.056), while maximum-iterations remains a
  fallback. Replay JSONL SHA-256 is
  `AE2CCC96255F03864D1C2A9F1D673CD506610C5BD11D54EF8B9E65C29A2FEFEF`;
  summary SHA-256 is
  `517E5A90BD6809ADE5D25D7857879F0E19230955AB9608784010DEBF28E8A6C6`.

  Eligible no-uncertainty roots are seeds `20260719`--`20260723`; the separate
  seed-20260720 zero-artifact prelaunch-timeout sidecar remains excluded. This
  closes training/per-seed replay provenance only. Next: commit/push these four
  documents without `.gitignore`, then verify all five checkpoint identities,
  zero Python work and absent fresh matrix/replay targets. Run the 5-by-6
  canonical evaluation matrix at 20 episodes/cell, audit all 600 records, then
  replay all five training plus 30 evaluation telemetry inputs before any
  post-isolation 8-UAV work.

- **Fourth valid post-isolation 5-UAV independent no-uncertainty seed retained,
  2026-08-01 (launch revision `7f1cc72`; documentation pending commit):** seed
  `20260722` ran in the unique
  `outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260722/`
  root under the frozen configuration and wait-only discipline. It naturally
  exited 0 after about 737.8 seconds at 100,080 transitions / 417 updates.
  Identity is five UAVs, `predictive_graph`, prediction true, uncertainty false
  and CUDA; OSQP remained CPU-side. Final checkpoint SHA-256 is
  `0A10275FC2663816FFC61121154D3358540C741C4789AC16E8770A2613A71D51`;
  summary SHA-256 is
  `9DE6F575CD014E0FAC9C1E17B0AB54A447972EBDFC90953439D2AB921FBEE6B8`;
  telemetry SHA-256 is
  `DF89E4A41B15B23A038AA1B44F0DEB17878E7DDB54E63AAB6710611A38CEF70F`.

  Training and final evaluation are zero-event over 20,016 and 160 CBF
  decisions. Initial evaluation retains eight `solve_time_limit` events over
  160 decisions; all six interval evaluations are zero-event over 80 decisions
  each. The eight sources exactly match the eight stderr notices. The replay
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260722_replay_20260801.jsonl`
  plus summary has eight sources and zero errors under unchanged
  20,000/0.1s/100/0.0/0.0 values. Four sources replay `solved` without fallback
  and four remain `solve_time_limit` fallbacks. Replay JSONL SHA-256 is
  `0E710F7DC071C8EF0344A8A52111C0443A6AFD4D4D99B260F204B4B36A3AD08E`;
  summary SHA-256 is
  `E11E626CC4D414938F4C897F4C135027E54E7D3F5D0B89935C1F75C1E9648E4A`.
  This is fourth-seed provenance only and does not establish a method, scale,
  safety, fallback-rate, performance or generalization result.

  Next: commit/push these four documents without `.gitignore`, then preflight
  and run final seed `20260723` serially under the identical frozen command and
  wait-only discipline. The 30-cell matrix and 35-input all-source replay must
  follow before any post-isolation 8-UAV work.

- **Third valid post-isolation 5-UAV independent no-uncertainty seed retained,
  2026-08-01 (launch revision `91d7b42`; documentation pending commit):** seed
  `20260721` ran in the unique
  `outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260721/`
  root under the frozen config and wait-only discipline. It naturally exited 0
  after about 719.7 seconds at 100,080 transitions / 417 updates. Identity is
  five UAVs, `predictive_graph`, prediction true, uncertainty false and CUDA;
  OSQP stayed CPU-side. Final checkpoint SHA-256 is
  `1703BCC62F16CEAB1EC527532DE21D18440CE7113174B4EA2C565AC5B7AFB7B0`;
  telemetry SHA-256 is
  `8315F8C6052BA80575695DF7EA3302385F4E08AE8AA94DB18EBA1481B04D290F`.

  Training retains one `solve_time_limit` and one `maximum iterations reached`
  event over 20,016 decisions; initial/final and all six retained interval
  evaluations are zero-event. The two sources exactly match stderr. The replay
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260721_replay_20260801.jsonl`
  plus summary has two sources and zero errors under unchanged
  20,000/0.1s/100/0.0/0.0 values: the time-limit source replays `solved`
  without fallback, while the maximum-iterations source remains fallback.
  Replay JSONL SHA-256 is
  `BF8981935C820794D61682CC0FBE2DAB2BEAA4AF024359C87FAE861822BCCAB7`;
  summary SHA-256 is
  `2DBA62589172951A8EF2266BA5E318FAF1D64E78091C3F9D8ADE7CF0CBC3455B`.
  This is third-seed provenance only.

  Next: commit/push these four documents without `.gitignore`, then preflight
  and run seed `20260722` serially under the identical frozen command and
  wait-only discipline. Seeds 22/23, the 30-cell matrix and all-source replay
  remain before post-isolation 8-UAV work.

- **Second valid post-isolation 5-UAV independent no-uncertainty seed retained,
  2026-08-01 (launch revision `a0fbafd`; documentation pending commit):** a
  zero-process/config-hash/absent-path preflight preceded seed `20260720` in
  `outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260720/`.
  The first orchestration call used an erroneous one-second shell timeout and
  exited 124 before Python child creation; no root or logs were created. Its
  separate excluded record is
  `outputs/core_5uav_post_actor_isolation_ablation_seed_20260720_prelaunch_timeout_20260801.ABORTED.json`.
  The canonical paths remained absent, so the actual frozen CUDA command then
  launched once and was observed only through its execution-cell wait handle.
  It naturally exited 0 after about 718.5 seconds at 100,080 transitions / 417
  updates.

  Identity is five UAVs, `predictive_graph`, delivered-packet prediction true,
  uncertainty false and CUDA; OSQP remained CPU-side. Training, initial/final
  and all six retained interval evaluations are zero-event over respectively
  20,016, 160/160 and 80 decisions per interval; stderr is empty. Final
  checkpoint SHA-256 is
  `577324344D9150020A6100D56C68D3AE7E4E52A0BB34013B2CAC6C555175E043`;
  telemetry SHA-256 is
  `00D731D151147F16CCCA488F3FFDF0607445329792009CFE797E3A85DB91975C`.
  The zero-event replay
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260720_replay_20260801.jsonl`
  plus summary uses unchanged 20,000/0.1s/100/0.0/0.0 values and records zero
  events/errors. Empty JSONL SHA-256 is
  `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`;
  summary SHA-256 is
  `18652CA0835CC0F05BC285279D286BCA049BFF95AB7ED271D619E8371E0108C2`.
  This is second-seed provenance only.

  Next: commit/push these four documents without `.gitignore`, then preflight
  and run seed `20260721` serially under the identical frozen 5-UAV ablation
  command and wait-only discipline. Seeds 21--23, the 30-cell matrix and
  all-source replay remain before any post-isolation 8-UAV work.

- **First valid post-isolation 5-UAV independent no-uncertainty seed retained,
  2026-08-01 (launch revision `bedeb3a`; documentation pending commit):** seed
  `20260719` ran from scratch in the unique root
  `outputs/core_5uav_post_actor_isolation_ablation/core_5uav_predictive_no_uncertainty_seed_20260719/`
  under frozen config SHA-256
  `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`.
  The wait-only CUDA run naturally exited 0 after about 300 seconds at 100,080
  transitions / 417 updates. Identity is five UAVs, `predictive_graph`,
  delivered-packet prediction true, uncertainty false, and CUDA; OSQP remained
  CPU-side. Final checkpoint SHA-256 is
  `A196F0C8BF0A8C8E3A062C51333EB80F959CACE8D0AF1AA2DFEDBAF8E35B4755`
  and telemetry SHA-256 is
  `78BA3E51B75CA76C2F3952DBC19ED2D98E9C014D791F69DB0FD38637AA55848D`.

  Training retains two `solved inaccurate` events and one `maximum iterations
  reached` event over 20,016 CBF decisions. Initial/final evaluations are
  zero-event over 160/160 decisions, and all six append-only interval
  evaluations are zero-event over 80 decisions each. The three training events
  exactly match the three stderr notices. The unique replay
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_seed_20260719_replay_20260801.jsonl`
  plus summary uses unchanged 20,000/0.1s/100/0.0/0.0 values, has three sources
  and zero errors, and returns one `solved inaccurate` fallback plus two
  `solved` non-fallback decisions. Replay JSONL SHA-256 is
  `FE08ACF17764EE3F30BFC4E94FACF64F441A5BB4B39BEA494FA073CD7FD1AE4A`;
  summary SHA-256 is
  `32EB0636F418F26018B6F4166559F776EE66C55BAA78EC1916FB444C7DB3B219`.
  The zero uncertainty margins are the frozen causal ablation, not fallback
  tuning. This is one-seed provenance only.

  Next: commit/push these four documents without `.gitignore`, then confirm
  zero Python work and absent seed-`20260720` root/log/replay paths. Launch seed
  `20260720` serially under the identical frozen 5-UAV ablation command and
  wait-only discipline. Complete all five seeds and their matrix/all-source
  replay before starting post-isolation 8-UAV work.

- **Post-isolation 5-UAV full-method evidence gate is complete, 2026-08-01
  (matrix launch revision `2c5ea51`; documentation pending commit):** the
  unique root
  `outputs/core_5uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`
  contains 30 valid five-seed-by-six-scenario cells and 600 identity-checked
  records. CUDA, actor own-truth-plus-delivered-packets, CPU OSQP, checkpoint
  step 100,160, 30 zero exits and empty cell/launcher stderr were verified.
  Evaluation telemetry has 12,000 CBF decisions and zero events. Launcher SHA
  is `9741AF56...B04279`.

  The 35-input all-source replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_5seed_training_and_evaluation_replay_20260801.jsonl`
  has five source events, zero errors and three replay fallbacks under unchanged
  values. Recorded statuses are four solved-inaccurate and one
  maximum-iterations; replay statuses are two solved-inaccurate, two solved and
  one maximum-iterations. The invalid non-retry seed21 root/checkpoint remains
  excluded. This closes provenance only, not any scale/safety/result claim.

  Next: commit/push the four documents without `.gitignore`, then preflight a
  fresh post-isolation 5-UAV no-uncertainty root. Independently train seed
  `20260719` from scratch using frozen
  `configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`,
  `--graph-mode predictive_graph`, CUDA neural execution, CPU OSQP and
  wait-only monitoring. Complete five seeds and their matrix/replay before
  beginning matched 8-UAV work.

- **Post-isolation 5-UAV full five-seed training is complete, 2026-08-01
  (final seed launch revision `d93e281`; documentation pending commit):**
  seed `20260723` naturally completed at 100,160 transitions / 313 updates
  under the frozen five-UAV CUDA/wait-only protocol. Checkpoint SHA is
  `F88ACA68...9BAD3`; telemetry SHA is `ED93223A...21D24`. Training, six
  retained interval evaluations, initial/final evaluations and stderr are all
  zero-event. Its required zero-event replay has zero events/errors under
  unchanged values.

  Eligible roots are seeds `20260719`, `20260720`,
  `20260721_intervaltelemetryretry1`, `20260722`, and `20260723`. The pre-fix
  seed19/20 roots have 2/1 retained training events and exactly 2/1 stderr
  notices, with zero retained evaluation events, so no missing fallback is
  observed. The non-retry seed21 root remains excluded due four missing
  contexts. This closes only training/per-seed replay provenance.

  Next: commit/push these documents without `.gitignore`, then verify zero
  Python/GPU work, all five eligible checkpoints, config hash and absent target
  paths. Create a wholly new 5-UAV full evaluation root and serially run five
  checkpoints by six canonical scenarios, 20 episodes/cell, CUDA inference and
  CPU OSQP. Audit 30 summaries/JSONL and 600 records, then run the unchanged
  35-input all-source replay before any independently trained 5-UAV
  no-uncertainty work.

- **Fourth eligible post-isolation 5-UAV full seed retained, 2026-08-01
  (launch revision `b16f055`; documentation pending commit):** seed
  `20260722` naturally completed under the frozen 5-UAV CUDA command and
  wait-only monitoring at 100,160 transitions / 313 updates. Checkpoint SHA is
  `EF754642...3EC5B`; telemetry SHA is `6873C3C7...A5508`. Training,
  initial and final streams are zero-event over 20,032/160/160 decisions. All
  six interval evaluations are retained; transition 61,440 contains one
  `solved inaccurate` event, the other five contain zero, and the source count
  exactly matches the sole stderr warning.

  The unique replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260722_replay_20260801.jsonl`
  finds `$.interval_evaluations[3].cbf.emergency_events[0]`, has one source and
  zero errors, and replays solved/non-fallback with about 263.39 slack under
  unchanged values. This is diagnostic provenance and real-event validation of
  the interval-history fix, not a result or tuning justification.

  Next: commit/push the three documents without `.gitignore`, then preflight
  and run final 5-UAV full seed `20260723` in a unique root with the same
  frozen config and wait-only discipline. Audit interval/stderr accounting and
  replay all source events before any 30-cell evaluation.

- **Valid 5-UAV full seed `20260721` telemetry retry retained, 2026-08-01
  (launch revision `a645024`; documentation pending commit):** after the
  append-only telemetry repair was committed/pushed, numeric seed `20260721`
  ran from scratch in the unique `intervaltelemetryretry1` root with frozen
  5-UAV config, CUDA network execution, CPU OSQP and wait-only monitoring. It
  naturally exited 0 after about 316 seconds at 100,160 transitions / 313
  updates. The checkpoint hash is `7658AAD2...08A8B`; telemetry hash is
  `361B38F4...19159`.

  The repaired telemetry retains six interval-evaluation records at 15,360
  through 92,160 transitions, each with a CBF event array. All intervals are
  zero-event over 80 decisions each; initial/final are 0/160 and 0/160.
  Training retains one `maximum iterations reached` event in 20,032 decisions,
  exactly matching the sole stderr notice. Its unique unchanged-protocol replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260721_intervaltelemetryretry1_replay_20260801.jsonl`
  has one source, zero errors and remains maximum-iterations fallback. Only the
  retry is eligible; the naturally completed non-retry root remains excluded
  via `ABORTED.json` because four interval contexts were lost. No CBF value
  changed and no outcome claim follows.

  Next: commit/push these documents without `.gitignore`, then confirm zero
  Python/GPU training work and absent new paths. Launch 5-UAV full seed
  `20260722` in a unique root under the same frozen command and wait-only
  discipline. After natural exit, require complete interval-history/stderr
  accounting, then replay all events before seed `20260723`.

- **5-UAV full seed `20260721` completed but is audit-invalid; prospective
  interval telemetry repair verified, 2026-08-01 (launch revision `3836319`;
  repair documentation pending commit):** the frozen seed naturally exited 0
  at 100,160 transitions / 313 updates and retained its final checkpoint,
  summary and telemetry. However, stderr has five CBF fallback notices (one
  `maximum iterations reached`, four `solve_time_limit`), while the pre-fix
  telemetry retains only the one training context and the final zero-event
  interval evaluation. The graph runner overwrote
  `last_interval_evaluation_cbf`, permanently losing the four earlier interval
  contexts. The root now has `ABORTED.json`; preserve it and exclude its final
  checkpoint from evaluation, replay, aggregation, statistics and manuscript
  use. Do not generate an incomplete replay that appears exhaustive.

  TDD observed the new GraphMAPPO test fail with missing
  `interval_evaluations`, then pass after the minimal repair: the telemetry
  writer can append named list entries and the graph runner stores every
  `{total_transitions, evaluation, cbf}` interval record while retaining both
  old `last_*` fields. Relevant tests pass 27, Ruff passes, and mypy passes 103
  sources. Full pytest is 159 passed / one unrelated legacy-path failure / 190
  OSQP warnings; the current legacy tree is `legacy_hgalo/HGALO_code` while the
  old test targets `HGALO_恢复源码`, producing 0 versus expected 81. Do not fix
  that unrelated audit here.

  Next: commit/push code, test and these three documents without `.gitignore`.
  Then verify zero Python/GPU training work and absent retry paths, and rerun
  numeric seed `20260721` from scratch only in
  `outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260721_intervaltelemetryretry1/`
  with distinct logs, unchanged config/CBF values and wait-only monitoring.
  After natural exit, count all nested interval/training telemetry events and
  compare them with stderr before replay or eligibility. Continue seeds 22/23
  only after a valid retry.

- **Raw GraphMAPPO five-seed training is now complete, 2026-07-31 (revision
  `6bc5d1a` at launch; documentation pending commit):** after a zero-process
  and absent-path preflight, seed `20260723` ran serially with
  `D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
  --config configs/rl/dynamic_graph_baseline.yaml --device cuda --seed
  20260723 --num-uavs 3 --graph-mode mappo --total-steps 100000 --output-dir
  outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260723`.
  Native stdout/stderr were redirected to the distinct retained files
  `outputs/core_3uav_post_actor_isolation_raw_graph_mappo_seed_20260723.stdout.log`
  and `.stderr.log`. It naturally completed at 100,032 transitions / 1,042
  updates on CUDA, with final checkpoint, summary, telemetry and TensorBoard
  retained. Training telemetry has zero CBF emergency events in 33,344
  decisions; short initial/final in-script diagnostics are also zero-event.
  This is not a performance, safety, fallback-rate, significance or baseline
  result. The valid raw roots are `20260719_launcherretry1`, `20260720`,
  `20260721`, `20260722`, and `20260723`; the initial non-retry `20260719`
  root remains preserved/excluded through `ABORTED.json`.

  Before any predictive arm, first create the zero-event CBF replay record for
  seed `20260723`; then preflight distinct evaluation paths and run every
  valid raw-graph final checkpoint across `nominal`, `delay_only`, `loss_only`,
  `dynamic_only`, `combined`, and `ood_communication_obstacle`, with 20
  episodes each, CUDA network inference and CPU OSQP. Parse all 30 summaries
  and JSONL files (600 records), run an unchanged-protocol all-source replay,
  update the three documentation files, commit only those files, and push
  `origin/codex/phase14-dynamic-world`. Do not start `predictive_graph` until
  those gates are complete.

- **Raw GraphMAPPO seed `20260723` zero-event replay retained, 2026-07-31
  (documentation pending commit):**
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260723_replay_20260731.jsonl`
  and its companion summary replay only this seed's completed training telemetry
  with unchanged CPU-OSQP values (20,000 iterations, 0.1 seconds, penalty 100,
  uncertainty gain 0.5, cap 5.0). The summary records `event_count=0` and
  `replay_error_count=0`; the intentionally empty JSONL is retained diagnostic
  provenance, not a safety, fallback-rate, performance, or comparison result.
  Next: verify zero Python jobs and absent target cells, then sequentially run
  the five final raw-graph checkpoints over the six canonical 20-episode
  scenarios. Retain every JSONL and do not begin `predictive_graph` until the
  complete raw matrix and all-source replay have been documented and pushed.

- **Raw GraphMAPPO first evaluation batch invalid and retained, 2026-07-31
  (documentation pending commit):** the distinct
  `outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo/`
  batch was interrupted during seed `20260720` OOD. It has 11 complete cells,
  eight partial JSONL records with no summary in the twelfth cell, and no
  remaining 18 cells. The launcher emitted no stderr and stdout ends after the
  partial-cell launch; the cause is unknown. Its `ABORTED.json` makes the
  entire root—including complete-looking cells—ineligible for aggregation,
  replay and manuscript use. Preserve it; no CBF or evaluator setting changed.
  Next: document/commit/push this invalid attempt, then zero-process/preflight
  a new evaluation root and run all 30 cells serially before parsing results.

- **Post-isolation raw GraphMAPPO arm is now evidence-complete, 2026-07-31
  (documentation pending commit):** the eligible rerun root
  `outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo_rerun1/`
  has 30/30 directories, summaries and JSONL files, 600 parsed records with
  fixed seed/scenario identities, 30 zero evaluator exit codes, and empty
  per-cell stderr. The earlier non-rerun batch remains excluded by
  `ABORTED.json`. The all-source replay at
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_5seed_training_and_evaluation_replay_20260731.jsonl`
  uses five training plus 30 eligible evaluation telemetry files and unchanged
  20,000/0.1s/100/0.5/5.0 CPU-OSQP values. It retains four source events, zero
  replay errors, and one replay emergency fallback. This is not an outcome or
  safety claim.

  Next: commit/push these documents without staging `.gitignore`, confirm no
  `multiuav_rl` Python task and absent targets, then start the first independent
  three-UAV `predictive_graph` seed (`20260719`) at 100k CUDA steps with the
  frozen dynamic-graph configuration and a unique post-isolation root. Keep
  all five predictive seeds serial; do not start
  `uncertainty_predictive_graph` until their matrix and replay are complete.

- **First post-isolation `predictive_graph` seed complete, 2026-07-31
  (documentation pending commit):** seed `20260719` naturally reached 100,032
  CUDA transitions / 1,042 updates under the frozen dynamic-graph config in
  `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260719/`.
  Final checkpoint, summary, telemetry, TensorBoard and launcher logs are
  retained; prediction is enabled and uncertainty features are disabled.
  Training has two CBF emergency events, while initial/final short evaluations
  have zero. Its per-seed replay retains both events with zero errors and one
  replay emergency fallback under unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP
  values. This is not an outcome claim. Next: commit/push these documents,
  confirm zero training processes and absent paths, then launch seed
  `20260720` serially with the same command pattern.

- **Second post-isolation `predictive_graph` seed complete, 2026-07-31
  (documentation pending commit):** seed `20260720` naturally reached 100,032
  CUDA transitions / 1,042 updates under the same frozen dynamic-graph
  configuration in
  `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260720/`.
  Its final checkpoint, summary, live telemetry, TensorBoard output and
  separately redirected stdout/stderr logs are retained. The summary confirms
  three UAVs, delivered-packet prediction enabled, uncertainty features
  disabled, and CUDA learned-network execution. Training records zero CBF
  emergency events in 33,344 decisions; the initial and final short
  evaluations also retain zero events in 160 and 152 decisions. Launcher
  stderr is empty.

  The required zero-event replay is retained at
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260720_replay_20260731.jsonl`
  with its companion summary. It consumes this seed's training telemetry under
  unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP values and records
  `event_count=0`, `replay_error_count=0`. The empty JSONL is deliberate
  diagnostic provenance, not a performance, safety, fallback-rate,
  significance, or comparison result. Next: commit/push these documents
  without staging `.gitignore`, confirm zero training processes and absent
  paths, then launch predictive seed `20260721` serially under the identical
  protocol.

- **Third post-isolation `predictive_graph` seed complete, 2026-07-31
  (documentation pending commit):** seed `20260721` naturally reached 100,032
  CUDA transitions / 1,042 updates under the frozen dynamic-graph
  configuration in
  `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260721/`.
  Final checkpoint, summary, live telemetry, TensorBoard data and separately
  redirected stdout/stderr are retained. The summary confirms three UAVs,
  delivered-packet prediction enabled, uncertainty features disabled, and CUDA
  learned-network execution. Training has zero CBF emergency events in 33,344
  decisions; initial/final short evaluations also retain zero events in
  160/160 decisions. Launcher stderr is empty.

  Its required zero-event replay is
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260721_replay_20260731.jsonl`
  plus companion summary. Under unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP
  values it records `event_count=0`, `replay_error_count=0`. This is
  diagnostic provenance, not a performance, safety, fallback-rate,
  significance, or comparison result. Next: commit/push these documents
  without staging `.gitignore`, confirm zero training processes and absent
  paths, then launch predictive seed `20260722` serially under the identical
  protocol.

- **First post-isolation `predictive_graph` seed-`20260722` attempt is invalid
  and retained, detected 2026-08-01 (documentation pending commit):** the
  unique root
  `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260722/`
  stopped at 84,672 transitions, with its last complete interval checkpoint at
  82,944. It has live telemetry, partial checkpoints and TensorBoard data, but
  no `summary.json` or final checkpoint; separately redirected stdout/stderr
  are both empty. The causal mechanism is unknown. The immediately preceding
  process-only monitoring command returned Windows exit code `1073807364`
  (`0x40010004`), retained only as adjacent observability evidence rather than
  asserted as the cause. `ABORTED.json` records the exact command, revision,
  configuration hash, telemetry hash and exclusion. The partial telemetry has
  zero events in 28,224 training, 160 initial-evaluation and 77 latest-interval
  CBF decisions, but the whole root is ineligible for replay or any result.

  Next: commit/push this invalid-attempt documentation without staging
  `.gitignore`, then confirm zero training processes and absent retry paths.
  Rerun the same numeric seed with unchanged settings only in the unique
  `core_3uav_predictive_graph_seed_20260722_launcherretry1` root and distinct
  retry logs. Never overwrite or delete the invalid root.

- **Valid post-isolation `predictive_graph` seed-`20260722` retry completed,
  2026-08-01 (documentation pending commit):** the planned `launcherretry1`
  process never started because the current launcher environment contained
  both `Path` and `PATH`; PowerShell `Start-Process` failed before child
  creation with a duplicate-key `ArgumentException`. Its two zero-byte logs
  and the sidecar
  `outputs/core_3uav_post_actor_isolation_predictive_graph_seed_20260722_launcherretry1.ABORTED.json`
  are retained/excluded. Root-cause reproduction also made `Get-ChildItem
  Env:` fail. `-UseNewEnvironment` did not avoid the duplicate-key path. A
  launcher-process-only normalization captured and de-duplicated both values,
  removed both keys, and set one `Path`; `cmd.exe` then exited 0 and
  `scripts/check_environment.py` verified Python 3.11.15, PyTorch
  `2.13.0+cu130`, CUDA available on the RTX 5060, and empty stderr. No system,
  user, repository, training, or CBF configuration was changed.

  With zero Python processes and absent retry2 paths, seed `20260722` then ran
  serially under the unchanged frozen command in
  `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260722_launcherretry2/`.
  It naturally completed at 100,032 transitions / 1,042 updates with final
  checkpoint, summary, telemetry, TensorBoard and distinct retry2 logs. The
  summary confirms three UAVs, `predictive_graph`, predicted knowledge enabled,
  uncertainty disabled, and CUDA. Training has zero CBF emergency events in
  33,344 decisions; initial/final short evaluations have zero in 160/126.
  Retry2 stderr is empty. Its zero-event replay at
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260722_launcherretry2_replay_20260801.jsonl`
  plus summary records `event_count=0`, `replay_error_count=0` under unchanged
  20,000/0.1s/100/0.5/5.0 CPU-OSQP values. Only retry2 is eligible. This is
  provenance, not an outcome claim. Next: commit/push these documents without
  staging `.gitignore`, then zero-process/absent-path preflight and serial
  predictive seed `20260723`.

- **Post-isolation `predictive_graph` five-seed training is complete,
  2026-08-01 (seed-20260723 launch revision `249ffd5`; documentation pending
  commit):** after a zero-process/absent-path preflight, seed `20260723` used
  the frozen config SHA-256
  `1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`, three
  UAVs, CUDA, `--graph-mode predictive_graph`, 100k requested steps, and the
  unique root
  `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260723/`.
  It naturally completed at 100,032 transitions / 1,042 updates. Summary,
  final checkpoint, live telemetry, TensorBoard and distinct launcher logs are
  retained; identity confirms delivered-packet prediction enabled and
  uncertainty disabled. The checkpoint hash is
  `666898D014CADA85ECF118116CD9E7E6634B5820AEC3452626A06CC37467D0A7`, and
  telemetry hash is
  `BC79C49915BED01AA5517E40E7EBEC1F84033BC748797296D95CF7752ABF9100`.

  Training retains one `maximum iterations reached` CBF fallback in 33,344
  decisions; initial/final short evaluations retain zero in 160/103. The
  per-seed replay
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_seed_20260723_replay_20260801.jsonl`
  plus summary retains one event and zero errors under unchanged
  20,000/0.1s/100/0.5/5.0 CPU-OSQP settings; it replays `solved` without
  fallback. This is diagnostic provenance only. Eligible predictive roots are
  `20260719`, `20260720`, `20260721`, `20260722_launcherretry2`, and
  `20260723`; preserve/exclude the original seed-20260722 root and retry1
  prelaunch attempt.

  Next: update/commit/push the three documents without staging `.gitignore`,
  then confirm zero Python jobs and absent targets, create a wholly new
  predictive evaluation root, and serially evaluate all five final checkpoints
  across `nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, and
  `ood_communication_obstacle`, 20 episodes per cell, CUDA inference and CPU
  OSQP. Audit 30 summaries/JSONL and 600 records, then run the unchanged
  all-source replay. Do not begin `uncertainty_predictive_graph` first.

- **Post-isolation `predictive_graph` training/evaluation/replay gate is now
  complete, 2026-08-01 (matrix launch revision `49e45f2`; documentation
  pending commit):** the unique root
  `outputs/core_3uav_post_actor_isolation_evaluations_20260801_predictive_graph/`
  serially evaluated the five eligible final checkpoints over all six
  canonical scenarios at 20 episodes/cell, CUDA inference and CPU OSQP. The
  transient launcher-only PATH normalization was the only environment repair;
  no repository, evaluator, or CBF setting changed. The launcher exited 0 with
  empty stderr. Content audit found 30/30 directories, summaries and JSONL,
  exactly 600 records, correct seed/scenario/checkpoint/controller/100,032-step
  identity, 30 zero cell exits, and no cell stderr. Environment metadata in all
  cells confirms three UAVs, CUDA, actor own truth plus delivered packets only,
  and CPU OSQP. Evaluation telemetry records 11,223 CBF decisions and zero
  emergency events.

  The 35-input all-source replay
  `outputs/cbf_diagnostics/post_actor_isolation_predictive_graph_5seed_training_and_evaluation_replay_20260801.jsonl`
  plus summary uses the five eligible training telemetry files and all 30 valid
  evaluation telemetry files under unchanged 20,000/0.1s/100/0.5/5.0 values.
  It retains three source events and zero replay errors: recorded statuses are
  one `solve_time_limit`, one `solved inaccurate`, and one `maximum iterations
  reached`; replay gives two `solved` and one `solved inaccurate`, with one
  replay fallback. This closes only the predictive-without-uncertainty
  provenance gate, not any outcome or safety claim.

  Next: commit and push these documents without staging `.gitignore`; verify
  zero dedicated Python/GPU work and absent unique targets; then start matched
  three-UAV `uncertainty_predictive_graph` seed `20260719` at 100k CUDA steps
  with the frozen dynamic-graph configuration. Keep all five seeds serial,
  retain every fallback and invalid attempt, and do not begin its evaluation
  until all five valid trainings finish.

- **First post-isolation `uncertainty_predictive_graph` seed-`20260719`
  attempt is invalid and retained, 2026-08-01 (launch revision `6d6ba24`;
  documentation pending commit):** the frozen three-UAV CUDA command used the
  unique non-retry root
  `outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260719/`.
  It exited at 78,624 transitions with signed code `-1073741510` (Windows
  `0xC000013A`); the last complete checkpoint is step 76,800 and no final
  checkpoint or summary exists. Stdout/stderr are empty, so the cause remains
  unknown. The exit code does not identify the source of the interruption, and
  no causal claim is made about read-only monitoring.

  `ABORTED.json` preserves the command, revision, hashes and exclusions. The
  partial telemetry records zero events in 26,208 training, 160 initial and 71
  last-interval CBF decisions, but the entire root is ineligible for replay,
  evaluation, aggregation, statistics, or manuscript use. Do not overwrite,
  resume, delete, or reinterpret it. Next: commit/push the three documents
  without staging `.gitignore`; confirm zero dedicated Python work and absent
  retry targets; then rerun numeric seed `20260719` serially in the distinct
  `core_3uav_uncertainty_predictive_graph_seed_20260719_launcherretry1` root
  with unchanged configuration and CBF values. Only a naturally completed
  retry may enter the matched five-seed arm.

- **Uncertainty seed-20260719 retry1 is also invalid; systematic diagnosis is
  retained, 2026-08-01 (launch revision `c3b59fa`; documentation pending
  commit):** the distinct
  `core_3uav_uncertainty_predictive_graph_seed_20260719_launcherretry1` root
  stopped at 13,344 transitions with the same signed `-1073741510` /
  `0xC000013A` exit. Its last complete checkpoint is 12,288; final checkpoint
  and summary are absent; stdout/stderr are empty. `ABORTED.json` excludes the
  entire retry. Partial telemetry has zero events over 4,448 training, 160
  initial and 80 last-interval CBF decisions, but is not a replay/result input.

  Local Windows SDK headers name the code `STATUS_CONTROL_C_EXIT`. No relevant
  Windows Application/System error, Python signal code, CUDA/NVIDIA error, or
  stderr explanation was found. A 90-second PowerShell child and a retained
  240-second Conda+CUDA RTX-5060 probe both survived multiple parallel
  read-only shell checks and exited 0; thus generic `Start-Process`, Python,
  CUDA initialization, arbitrary concurrent shell access, and a simple
  180-second cap are ruled out. The external control-event sender remains
  unknown. The narrow remaining hypothesis is sustained real-training load
  interacting with parallel shell monitoring in the shared tool process tree.

  Next: commit/push these documents without `.gitignore`, preflight an absent
  `launcherretry2` root/logs and zero Python processes, then launch exactly one
  full identical retry from scratch. While it runs, call only `wait` on that
  execution cell: no parallel shell health checks. This is the one-variable
  hypothesis test. If retry2 also returns `0xC000013A`, do not attempt retry3;
  stop and request user direction without changing training or CBF values.

- **First valid post-isolation `uncertainty_predictive_graph` seed retained,
  2026-08-01 (retry2 launch revision `f21566e`; documentation pending
  commit):** the unique `seed_20260719_launcherretry2` root ran from scratch
  under the identical frozen command. While active, only the original
  execution-cell wait handle was used; no parallel shell process or artifact
  inspection ran. It naturally exited 0 after about 1,152 seconds and completed
  100,032 transitions / 1,042 updates. Summary identity is three UAVs,
  `uncertainty_predictive_graph`, delivered-packet prediction true,
  uncertainty true, and CUDA. Final checkpoint, summary, telemetry,
  TensorBoard and retry2 logs are retained; stderr is empty.

  Training, initial and final CBF records are respectively 0/33,344, 0/160 and
  0/96. The zero-event replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260719_launcherretry2_replay_20260801.jsonl`
  plus summary records `event_count=0`, `replay_error_count=0` under unchanged
  20,000/0.1s/100/0.5/5.0 CPU-OSQP values. Only retry2 is eligible; retain and
  exclude both earlier interrupted roots.

  The success supports, but does not uniquely prove, the shared-process-tree
  monitoring hypothesis. Mandatory operational rule for subsequent long jobs:
  after preflight and launch, call only `wait` on the existing execution cell;
  run no parallel shell health check until it exits, then perform one complete
  audit. Next: commit/push the three documents without `.gitignore`, then use
  that discipline for fresh uncertainty seed `20260720`. Four further seeds,
  all 30 evaluation cells and all-source replay remain before this arm closes.

- **Second valid post-isolation `uncertainty_predictive_graph` seed retained,
  2026-08-01 (launch revision `9f6c5f1`; documentation pending commit):** seed
  `20260720` ran in the unique matching root under the frozen three-UAV CUDA
  command. Only the existing execution-cell wait handle was used while active;
  it naturally exited 0 after about 595 seconds and completed 100,032
  transitions / 1,042 updates. Summary identity confirms delivered-packet
  prediction and uncertainty enabled. Final checkpoint, summary, telemetry,
  TensorBoard and logs are retained; stderr is empty. CBF records are 0/33,344
  training, 0/160 initial and 0/131 final. Its zero-event replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260720_replay_20260801.jsonl`
  plus summary records zero events/errors under frozen values.

  Next: commit/push these documents without `.gitignore`; then zero-process/
  absent-path preflight and launch seed `20260721` serially, again using no
  parallel shell while the run is active. Seeds `20260721`--`20260723`, the
  30-cell evaluation matrix, and all-source replay remain.

- **Third valid post-isolation `uncertainty_predictive_graph` seed retained,
  2026-08-01 (launch revision `30e164d`; documentation pending commit):** seed
  `20260721` ran under the frozen command in its unique root, with only wait
  calls while active. It naturally exited 0 after about 1,422 seconds and
  completed 100,032 transitions / 1,042 updates. Identity confirms prediction
  and uncertainty enabled, three UAVs, and CUDA. Training has one `maximum
  iterations reached` fallback in 33,344 decisions; initial/final are 0/160 and
  0/160. The matching one-event replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260721_replay_20260801.jsonl`
  has zero errors and replays `solved` without fallback under frozen values.

  Next: commit/push the documents without `.gitignore`, then preflight and run
  seed `20260722` serially using the same wait-only discipline. Seeds 22/23,
  all 30 evaluation cells and all-source replay remain.

- **Fourth valid post-isolation `uncertainty_predictive_graph` seed retained,
  2026-08-01 (launch revision `67407cc`; documentation pending commit):** seed
  `20260722` completed 100,032 transitions / 1,042 updates in its unique root,
  with wait-only monitoring and correct prediction/uncertainty/3-UAV/CUDA
  identity. Training records one `solved inaccurate` fallback in 33,344
  decisions; initial/final are 0/160 and 0/147. Its one-event replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260722_replay_20260801.jsonl`
  has zero errors and remains `solved inaccurate` with fallback under frozen
  values. Do not tune it away.

  Next: commit/push documents without `.gitignore`, then preflight and run final
  seed `20260723` using the same wait-only discipline. Only after its audit and
  replay may the complete five-checkpoint six-scenario evaluation begin.

- **Post-isolation `uncertainty_predictive_graph` five-seed training is now
  complete, 2026-08-01 (seed-20260723 launch revision `0f5a971`;
  documentation pending commit):** seed `20260723` completed 100,032
  transitions / 1,042 updates in its unique root under wait-only monitoring,
  with correct prediction/uncertainty/3-UAV/CUDA identity. Training records one
  `solved inaccurate` fallback in 33,344 decisions; initial/final are 0/160 and
  0/160. Its one-event replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_seed_20260723_replay_20260801.jsonl`
  has zero errors and replays `solved` without fallback under frozen values.

  Eligible roots are `20260719_launcherretry2`, `20260720`, `20260721`,
  `20260722`, and `20260723`; preserve/exclude the two earlier seed-20260719
  attempts. Next: commit/push documents without `.gitignore`, then preflight a
  completely new uncertainty-aware evaluation root and serially evaluate those
  five final checkpoints across all six canonical scenarios, 20 episodes per
  cell, CUDA inference and CPU OSQP. Audit 30 summaries/JSONL and 600 records,
  then run the unchanged 35-input all-source replay.

- **Post-isolation three-UAV uncertainty-aware evidence gate is complete,
  2026-08-01 (matrix launch revision `c9bff6e`; documentation pending
  commit):** the unique
  `outputs/core_3uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`
  root has 30/30 valid cells and 600 identity-checked episode JSONL records for
  five eligible checkpoints by six canonical scenarios. CUDA inference, CPU
  OSQP, actor own-truth-plus-delivered-packets metadata, 30 zero exits, and
  empty cell/launcher stderr were verified. Evaluation telemetry contains
  10,855 CBF decisions and zero emergency events.

  The 35-input all-source replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5seed_training_and_evaluation_replay_20260801.jsonl`
  plus summary has three source events and zero errors under frozen values.
  Sources are one `maximum iterations reached` and two `solved inaccurate`;
  replay gives two `solved`, one `solved inaccurate`, and one replay fallback.
  The two invalid seed-20260719 roots remain excluded. This closes the 3-UAV
  four-arm artifact matrix only, not any numerical or safety claim.

  Next: commit/push these documents without `.gitignore`, then perform a
  read-only inventory/audit of current post-isolation 5-UAV, 8-UAV and
  independently trained ablation artifacts against the current information
  boundary and matched protocol. Do not blindly rerun or accept older roots;
  identify exact missing valid cells/seeds first, preserve exclusions, and use
  new roots for any required work before statistics and literature claims.

- **Post-isolation 5/8-UAV gap audit complete, 2026-08-01 (documentation
  pending commit):** all current `core_5uav*` and `core_8uav*` training,
  evaluation, ablation, paired-summary and CBF-replay artifacts predate actor
  information isolation. They remain preserved forensic evidence but are
  ineligible for every result/scale/ablation/manuscript aggregation. No
  top-level post-isolation 5/8-UAV root exists. Full-method and independently
  trained no-uncertainty config hashes still match the four frozen values.

  `docs/aamas2027_analysis_plan.md` now lists the four eligible post-isolation
  3-UAV raw evaluation roots; no fresh multi-arm summary has yet been created.
  The 5/8 rows remain empty by design. Next: commit/push the four documentation
  files without `.gitignore`, then preflight and run 5-UAV full-method seed
  `20260719` under `configs/rl/dynamic_graph_5uav.yaml`, 100k CUDA steps,
  `--graph-mode uncertainty_predictive_graph`, CPU OSQP, and a fresh
  `outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260719`
  root. Use only the launch-cell wait handle while active. Complete five valid
  full-method seeds and their matrix/replay before starting the independently
  trained 5-UAV no-uncertainty arm, then repeat both matched arms at 8 UAV.

- **First valid post-isolation 5-UAV full-method seed retained, 2026-08-01
  (launch revision `4944c4d`; documentation pending commit):** seed `20260719`
  used frozen `dynamic_graph_5uav.yaml`, five UAVs, CUDA and
  `uncertainty_predictive_graph` in the new
  `outputs/core_5uav_post_actor_isolation/` root. With wait-only monitoring it
  naturally exited 0 after about 328 seconds at 100,160 transitions / 313
  updates. Training has two `solved inaccurate` fallbacks in 20,032 decisions;
  initial/final are 0/160 and 0/160. Its two-event replay
  `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_seed_20260719_replay_20260801.jsonl`
  has zero errors; one replay remains fallback and one solves.

  Next: commit/push documents without `.gitignore`, then preflight and run
  5-UAV full seed `20260720` with the identical frozen command and wait-only
  discipline. Four full seeds remain before evaluation or ablation training.

- **Second valid post-isolation 5-UAV full seed retained, 2026-08-01 (launch
  revision `4fc64aa`; documentation pending commit):** seed `20260720`
  completed 100,160 transitions / 313 updates with correct full-method identity
  and wait-only monitoring. Training has one `solved inaccurate` fallback in
  20,032 decisions; initial/final are 0/160 and 0/160. Its one-event replay
  remains `solved inaccurate` with fallback and zero errors under frozen values.
  Next: commit/push without `.gitignore`, then run seed `20260721` identically.

- **Documentation and safe-maintenance package, 2026-07-30:**
  read `README.md` for environment, entry-point, train/evaluate/replay, and
  reproducibility guidance; read `docs/aamas2027_method_and_readiness.md` for
  the detailed mathematical model and paper-readiness boundaries; and use the
  copyable `docs/next_agent_prompt.md` to start a successor conversation. A
  cache-only audit is retained in `docs/maintenance_cleanup_20260730.md`.
  Explicit deletion was blocked by the execution environment before running,
  so no file was removed. This is intentional preservation, not an incomplete
  experiment. The research sequence remains paused at raw GraphMAPPO seed
  `20260722`; on explicit resume, perform no new exploration and begin with
  zero-process/new-root preflight then raw GraphMAPPO seed `20260723`.

- **SSH authentication and push restored, 2026-07-30:** after the user added
  the existing public key to GitHub,
  `ssh -T git@github.com` authenticated as `youshenbupo`. The old command using
  remote name `github.com` failed because no such Git remote exists; `origin`
  is configured as `git@github.com:youshenbupo/multi-uav-rl-path-planning.git`.
  `git push origin codex/phase14-dynamic-world` succeeded and advanced the
  remote branch from `55abb87` to `014c006`. `.gitignore` was not staged. This
  restores Git
  synchronization only; research remains paused by the user-requested stop
  point and no external submission is authorized.

- **Post-isolation MLP seed `20260719` evaluation matrix completed,
  2026-07-29 (revision `e625af5`; documentation pending commit; push remains
  blocked):** the final checkpoint was evaluated in distinct new-root cells
  for `nominal`, `delay_only`, `loss_only`, `dynamic_only`, `combined`, and
  `ood_communication_obstacle`, each with `--episodes 20 --device cuda` using
  `scripts/evaluate_core_checkpoint.py`; OSQP remains CPU. All six cells are
  beneath `outputs/core_3uav_post_actor_isolation_evaluations/` with matching
  `summary.json` and raw `raw_results/seed_20260719.jsonl`. Direct parsing
  verified exactly 20 records/cell, constant `(seed=20260719, scenario=<cell>)`
  identity, and 120 total raw records. No numerical performance, safety,
  fallback-rate, significance, or baseline conclusion is permitted from this
  one checkpoint. Next: commit this documentation without staging `.gitignore`,
  confirm no Python job/absent targets, and perform the identical six-cell
  matrix for the remaining four valid MLP seeds; after all 30 cells, replay and
  summarize every retained CBF fallback source without changing CBF values.

- **Post-isolation MLP seed `20260720` evaluation matrix completed,
  2026-07-29 (revision `2b232ac`; documentation pending commit):** repeated
  the exact six canonical new-root CUDA/CPU-OSQP cells at 20 episodes/cell for
  the seed `20260720` final checkpoint. Each
  `...seed_20260720_<scenario>/raw_results/seed_20260720.jsonl` has a parseable
  20-record, fixed-identity JSONL and a parseable `summary.json`; all six
  scenario names are present (120 records). It supplies another protocol
  artifact, not a performance, safety, fallback-rate, significance, baseline,
  or multi-seed result. Next: commit docs without staging `.gitignore`, then
  preflight and evaluate valid retry `20260721_telemetryretry1` in the same
  six cells before seeds `20260722`/`20260723`.

- **Valid retry seed `20260721_telemetryretry1` evaluation matrix completed,
  2026-07-29 (revision `5c81524`; documentation pending commit):** evaluated
  only the retry checkpoint (never the aborted same-seed root) in six distinct
  canonical cells at 20 CUDA/CPU-OSQP episodes/cell. Each output under
  `outputs/core_3uav_post_actor_isolation_evaluations/` has parseable
  `summary.json` and a 20-record `raw_results/seed_20260721.jsonl` with fixed
  numeric seed/scenario identity, for 120 retained records. This is a third
  matching artifact, not an outcome claim. Next: commit documentation without
  staging `.gitignore`, preflight, and run seed `20260722` six-cell matrix.

- **Post-isolation MLP seed `20260722` evaluation matrix completed,
  2026-07-29 (revision `fd0b487`; documentation pending commit):** the matched
  six canonical CUDA-actor/CPU-OSQP cells at 20 episodes/cell are complete and
  preserve parseable `summary.json` plus 20 fixed-identity records/cell in
  `raw_results/seed_20260722.jsonl` (120 records). This is fourth-matrix
  provenance only. Next: commit documentation without staging `.gitignore`,
  then preflight/evaluate seed `20260723`; after its validation, conduct the
  all-source CBF fallback replay with unchanged safety parameters.

- **All five valid post-isolation 3-UAV MLP matrices completed,
  2026-07-29 (revision `f757da3`; documentation pending commit):** seed
  `20260723` completed the final six canonical CUDA-actor/CPU-OSQP cells. A
  direct audit across valid tags `20260719`, `20260720`,
  `20260721_telemetryretry1` (numeric seed 20260721), `20260722`, and
  `20260723` confirmed 30 expected cells, parseable summaries, fixed
  seed/scenario identity, and exactly 20 raw JSONL records/cell: 600 retained
  episodes. This proves matrix completeness/provenance only, not any outcome.
  The aborted pre-retry root and all historical protocol-ineligible outputs are
  excluded. Next: commit docs without staging `.gitignore`, inspect
  `replay_cbf_fallbacks.py --help`, construct a no-overwrite all-source replay
  input list from only these valid training/evaluation artifacts, and replay
  under unchanged CBF parameters before GraphMAPPO work.

- **All-source post-isolation MLP CBF replay completed, 2026-07-29 (revision
  `9b3fd56`; documentation pending commit):** the output
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_5seed_training_and_evaluation_replay_20260729.jsonl`
  plus summary cover five valid training `live_training_telemetry.json` and 30
  valid evaluation `runtime_telemetry.json` sources under explicitly unchanged
  20,000 / 0.1s / 100 / 0.5 / 5.0 CPU-OSQP values. It retains 27 source events
  (26 training, one evaluation): 26 replayed, 23 with emergency fallback; one
  OOD seed-20260719 event remains an explicit replay error because its recorded
  dynamic-obstacle centers cannot be reconstructed. Preserve it; do not tune
  CBF or call it a rate. Next: commit documentation without staging
  `.gitignore`, then zero-process/new-root preflight and launch only the first
  matching post-isolation GraphMAPPO training seed after inspecting the
  documented three-method configuration names.

- **Invalid raw GraphMAPPO seed `20260719` launcher attempt retained,
  2026-07-29 (revision `a908886`; documentation pending commit):** command was
  `train_graph_mappo.py --config configs/rl/dynamic_graph_baseline.yaml
  --device cuda --seed 20260719 --num-uavs 3 --graph-mode mappo --total-steps
  100000` under config SHA-256
  `1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
  The launch wrapper's `ErrorActionPreference=Stop` converted a normal CBF
  `solved inaccurate` native stderr diagnostic into `NativeCommandError`, so
  the child stopped at 10,848 transitions without final checkpoint/summary.
  `ABORTED.json` preserves/excludes the root and its 3,072/6,144/9,216 partial
  checkpoints, telemetry, and TensorBoard. No CBF setting changed. Next:
  commit docs without staging `.gitignore`; verify zero Python jobs and an
  absent distinct `...seed_20260719_launcherretry1` root, then retry with
  native stdout/stderr merged before PowerShell error handling.

- **Active raw GraphMAPPO seed `20260719` launcher retry (2026-07-29; commit
  `461951c`):** zero-process/new-root preflight passed, then
  `D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_graph_mappo.py
  --config configs\rl\dynamic_graph_baseline.yaml --device cuda --seed
  20260719 --num-uavs 3 --graph-mode mappo --total-steps 100000 --output-dir
  outputs\core_3uav_post_actor_isolation\core_3uav_raw_graph_mappo_seed_20260719_launcherretry1`
  was started through `Start-Process` with standard output/error redirected,
  avoiding the prior wrapper's stderr-as-fatal behavior. Python PID `17072` was
  alive at verification; retain only process/file-status checks while it runs,
  not live-telemetry reads. New logs are
  `outputs/core_3uav_post_actor_isolation_raw_graph_mappo_seed_20260719_launcherretry1.stdout.log`
  and `.stderr.log`; frozen config SHA-256 is
  `1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`. On
  natural exit, audit the final checkpoint/summary/transitions/updates and all
  fallback sources; document before launching the next seed. If it stops
  without final artifacts, preserve a second unique invalid attempt—never
  overwrite either root or change CBF settings.

- **Follow-up status, 2026-07-29 (commit `6df41d7`):** after the retry launch
  and documentation correction, PID `17072` remained the sole Python process
  in the dedicated `multiuav_rl` environment through repeated process-only
  checks; its CPU time rose from 109.75 to 297.25 seconds, so it was active and
  not yet ready for artifact audit. `.gitignore` remains the only unstaged
  worktree change. Do not inspect its live telemetry or launch another job.
  Next conversation: first check whether PID `17072` has exited, then audit the
  retry root for a 100k final checkpoint/summary before treating it as valid;
  otherwise continue process-only monitoring.

- **Valid post-isolation raw GraphMAPPO seed `20260719` retry completed,
  2026-07-29 (training revision `461951c`; documentation pending commit):**
  the `launcherretry1` root naturally completed on CUDA at 100,032 transitions
  / 1,042 updates using graph mode `mappo`, three UAVs, and frozen dynamic-graph
  config SHA-256 `1BA2CCE7E7699B97989AF4FFBFB3F26636398428113DCFEE3F798B58A681727D`.
  Final checkpoint, summary, TensorBoard, raw telemetry, and stdout/stderr logs
  exist and were parsed. Training retains two `solved inaccurate` fallbacks in
  33,344 decisions; initial/final in-script evaluations retain 0/160 and
  0/160. Per-seed replay at
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260719_launcherretry1_replay_20260729.jsonl`
  has two sources, zero errors, one replay fallback, under unchanged
  20,000/0.1s/100/0.5/5.0 CPU-OSQP values. Do not call this an outcome. The
  earlier non-retry root is invalid and excluded. Next: commit docs without
  staging `.gitignore`; confirm zero Python jobs and absent new output/log paths,
  then serially launch raw GraphMAPPO seed `20260720` with process-level stdio
  redirection before any Graph checkpoint evaluation.

- **Valid post-isolation raw GraphMAPPO seed `20260720` completed,
  2026-07-29 (revision `cf68ed3`; documentation pending commit):** serial CUDA
  raw-graph (`mappo`) training naturally completed at 100,032 transitions /
  1,042 updates under the unchanged dynamic-graph config SHA-256, with final
  checkpoint, summary, TensorBoard, telemetry, and launcher logs retained.
  Source telemetry has one `maximum iterations reached` fallback in 33,344
  training decisions and zero initial/final in-script fallback events (160 each).
  Its replay at
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260720_replay_20260729.jsonl`
  has one source, zero errors, and replays `solved` without fallback under
  unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP parameters. This is provenance,
  not an outcome claim. Next: commit docs without staging `.gitignore`, then
  zero-process/new-root preflight and train raw-graph seed `20260721` before
  any GraphMAPPO evaluation.

- **Active post-isolation raw GraphMAPPO seed `20260721` (2026-07-29; commit
  `d07b637`):** zero-process/new-root/CUDA preflight passed under the unchanged
  dynamic-graph config SHA-256, then `train_graph_mappo.py --device cuda --seed
  20260721 --num-uavs 3 --graph-mode mappo --total-steps 100000` started through
  process-level stdout/stderr redirection. PID `23164` was alive at verification.
  Target root is
  `outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260721/`;
  logs are `outputs/core_3uav_post_actor_isolation_raw_graph_mappo_seed_20260721.stdout.log`
  and `.stderr.log`. While active, use process/file-status checks only; do not
  read live telemetry or launch another job. On exit, audit 100,032 transitions,
  1,042 updates, CUDA identity, final checkpoint/summary, and fallback sources;
  document before seed `20260722`.

- **Valid post-isolation raw GraphMAPPO seed `20260721` completed,
  2026-07-29 (revision `d07b637`; documentation pending commit):** the CUDA
  raw-graph run completed at 100,032 transitions / 1,042 updates under the
  unchanged config, retaining final artifacts and logs. It has one training
  `solved inaccurate` fallback in 33,344 decisions and 0/160 initial/final
  events. Replay at
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260721_replay_20260729.jsonl`
  retains one source, zero errors, and a solved non-fallback replay using
  unchanged 20,000/0.1s/100/0.5/5.0 CPU-OSQP values. No outcome claim. Next:
  commit docs without staging `.gitignore`, then preflight/train raw-graph seed
  `20260722` serially before any evaluation.

- **Active post-isolation raw GraphMAPPO seed `20260722` (2026-07-29; commit
  `d827ff3`):** zero-process/new-root/CUDA preflight passed under the unchanged
  dynamic-graph config SHA-256. The CUDA command uses raw graph mode `mappo`,
  seed `20260722`, three UAVs, 100k requested steps, and process-level output
  redirection into
  `outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260722/`.
  PID `23264` was alive at verification; stdout/stderr logs use the corresponding
  `outputs/core_3uav_post_actor_isolation_raw_graph_mappo_seed_20260722.*.log`
  paths. Do not read live telemetry or start another job. On natural exit,
  audit the 100,032-transition/1,042-update final artifact and replay fallback
  sources before documenting/starting seed `20260723`.

- **Valid post-isolation raw GraphMAPPO seed `20260722` completed; stop point
  requested by user (2026-07-30; training revision `d827ff3`; documentation
  pending commit):** the serial CUDA raw-graph `mappo` job naturally reached
  100,032 transitions / 1,042 updates under the unchanged dynamic-graph config
  SHA-256. Its final checkpoint, summary, TensorBoard, raw telemetry, and
  launcher logs were parsed. CBF fallbacks are 0/33,344 training, 0/160 initial,
  and 0/131 final decisions. The retained zero-event replay is
  `outputs/cbf_diagnostics/post_actor_isolation_raw_graph_mappo_seed_20260722_replay_20260730.jsonl`
  plus summary (`event_count=0`, `replay_error_count=0`) under unchanged
  20,000/0.1s/100/0.5/5.0 CPU-OSQP values. The user explicitly requested
  stopping after the current run: do not start seed `20260723`, evaluations,
  other graph arms, scale-up, or literature work in this stop turn. This is a
  pause, not completion: raw graph still needs seed 20260723 and evaluations;
  predictive/no-uncertainty and uncertainty-predictive GraphMAPPO arms, 5/8 UAV,
  ablations, literature verification, and the authorized push remain open.
  On resume, first inspect process/worktree, then launch only raw-graph seed
  `20260723` after zero-process/new-root preflight.

- **Actor-information protocol repair, 2026-07-29 (commit `0e048cd`; push
  blocked):** a static audit found three
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
  be force-loaded. The full pytest result is recorded below. The push command
  `git push github.com:youshenbupo/multi-uav-rl-path-planning.git
  codex/phase14-dynamic-world` failed with `Permission denied (publickey)` and
  made no remote change. Do not alter credentials or `.gitignore`; restore
  authorized SSH access and push `0e048cd` plus the documentation follow-up.
  Then start **new-root**, serial 3-UAV MLP five-seed 100k CUDA training; only
  after valid MLP reruns/evaluations begin the three GraphMAPPO matching
  five-seed reruns.

- **Active serial rerun (do not start concurrent work):** post-isolation 3-UAV
  MLP seed `20260719` was launched after zero-process/CUDA preflight from local
  revision `00a970f` with
  `D:\anaconda3\envs\multiuav_rl\python.exe scripts\train_mappo.py --config
  configs\rl\dynamic_mappo_baseline.yaml --device cuda --seed 20260719
  --num-uavs 3 --total-steps 100000 --output-dir
  outputs\core_3uav_post_actor_isolation\core_3uav_mlp_mappo_seed_20260719`.
  Frozen config SHA-256:
  `794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`.
  PID `44928` was alive at launch verification; output stdout/stderr are
  `outputs/core_3uav_post_actor_isolation_seed_20260719.stdout.log` and
  `.stderr.log`, and live telemetry is in the new output directory. The
  retained stderr already contains `CBF filter emergency fallback: solved
  inaccurate`; do not tune CBF, restart, or discard it. Wait for natural exit,
  audit/checkpoint/document it, then serially launch seed `20260720`. This is
  an active attempt, not a performance/safety/fallback-rate result.

- **Post-isolation MLP seed `20260720` completed:** it naturally reached
  100,032 transitions / 1,042 CUDA updates at local revision `a4e9d8d` under
  the unchanged frozen configuration SHA-256
  `794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`.
  Its artifact root is
  `outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260720/`.
  Training retains one `maximum iterations reached` fallback across 33,344
  decisions; initial/final script evaluations retain zero fallbacks in 160/82
  decisions. The new all-source-for-seed replay
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260720_replay_20260729.jsonl`
  has one source event, zero replay errors, and the unchanged 20,000 / 0.1s /
  100 / 0.5 / 5.0 protocol; its replay also uses fallback. This is diagnostic
  only. Next: verify zero running jobs and absent target output, then serially
  launch seed `20260721`; do not start evaluation or tune CBF.

- **Invalid post-isolation MLP seed `20260721` attempt and telemetry fix:** the
  first `20260721` job at revision `2122b14` aborted at 91,776 transitions
  without final checkpoint/summary when `Path.replace` hit
  `PermissionError [WinError 5]` on `live_training_telemetry.json`. Preserve
  its root `outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260721/`,
  launcher logs, partial checkpoints, and eight initial-evaluation `solved
  inaccurate` contexts; `ABORTED.json` marks all of it excluded from every
  aggregation. Do not overwrite it. The new `telemetryretry1` run must use the
  committed bounded transient-lock retry in `multiuav/learning/telemetry.py`
  (five 50ms retries; atomic persistence otherwise unchanged), not a CBF
  parameter change. TDD red/green passed, Ruff/mypy passed 103 sources; full
  pytest is 158 passed / 1 retained legacy inventory failure / 28 OSQP warnings.
  Next: confirm zero jobs and absent unique retry path, then rerun `20260721`
  serially at 100k CUDA; only after valid completion continue seeds 22/23.

- **Valid post-isolation MLP seed `20260721` retry:** the new
  `...seed_20260721_telemetryretry1/` root at revision `0b7d402` completed
  100,032 CUDA transitions / 1,042 updates with final checkpoint/summary;
  it is the only valid same-seed artifact. Training/final script evaluation
  have 0/33,344 and 0/103 fallbacks. Initial untrained-policy evaluation retains
  eight `solved inaccurate` contexts in 160 decisions. Its all-source replay
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260721_telemetryretry1_replay_20260729.jsonl`
  has 8 sources, zero replay errors, seven `maximum iterations reached`, one
  `solved inaccurate`, and eight replay fallbacks under unchanged 20,000 / 0.1s
  / 100 / 0.5 / 5.0. This is diagnostic only; never tune CBF from it. The
  original `20260721` directory remains invalid. Next: zero-process/absent-path
  preflight then serially train `20260722` at 100k CUDA; no evaluation yet.

- **Post-isolation MLP seed `20260722` completed:** at revision `73112f6` it
  naturally reached 100,032 CUDA transitions / 1,042 updates in
  `outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260722/`.
  Training/final script evaluation have 0/33,344 and 0/89 fallbacks; initial
  evaluation retains 13 events in 160 decisions. The all-source replay
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260722_replay_20260729.jsonl`
  has 13 sources, zero replay errors, nine recorded time limits/four maximum
  iterations, and ten replay time limits/three maximum iterations; all replay
  events use fallback under unchanged 20,000 / 0.1s / 100 / 0.5 / 5.0.
  Diagnostic only: no tuning/claim. Next: zero-process and absent-path
  preflight, then final serial MLP seed `20260723`; do not begin evaluation.

- **Five valid post-isolation 3-UAV MLP training seeds are complete:**
  `20260719`, `20260720`, `20260721_telemetryretry1`, `20260722`, and
  `20260723` each reached 100,032 CUDA transitions / 1,042 updates with final
  checkpoint/summary under the frozen config hash
  `794C8537F9A879467854AFB657C6D47E6C8A9F3793497C70D28CD1F35B2D8C46`.
  The original `20260721` root stays invalid. The all-source replay
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_5seed_training_replay_20260729.jsonl`
  has 26 source events, zero replay errors, and 23 replay fallbacks under
  unchanged values; it is diagnostic provenance, not a rate/safety/result
  claim. Next: create a new evaluation root, then serially execute six
  canonical scenarios × 20 episodes for each matching final checkpoint, retain
  JSONL, and only then begin GraphMAPPO retraining.

- **Post-isolation MLP seed `20260719` completed:** the above process naturally
  exited at 100,032 transitions / 1,042 updates. Its final checkpoint,
  summary, TensorBoard data and `live_training_telemetry.json` are retained
  under `outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260719/`;
  launcher logs are in the corresponding top-level `outputs` paths. Training
  has three retained CBF emergency events in 33,344 decisions (one `solved
  inaccurate`, two `maximum iterations reached`); initial/final script
  evaluations record 0/160 and 0/98 fallbacks. The new replay
  `outputs/cbf_diagnostics/post_actor_isolation_mlp_seed_20260719_replay_20260729.jsonl`
  plus `.summary.json` has all three events and zero replay errors under
  unchanged 20,000 / 0.1s / 100 / 0.5 / 5.0 values; one replay still uses
  emergency fallback. This is diagnostic only. Before another job, confirm
  zero training processes and absent target path, then launch post-isolation
  MLP seed `20260720` serially under the same command with only seed/output
  path changed. Do not start checkpoint evaluation yet or alter CBF.

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

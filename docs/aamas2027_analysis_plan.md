# AAMAS 2027 analysis plan and evidence boundary

**Status:** protocol-invalidation update recorded 2026-07-29 from parent
revision `55abb87`. This is a retrospective provenance and reporting plan, not
a preregistration and not evidence of a method effect.

## Protocol invalidation (2026-07-29)

Static audit and test-first repairs found that the pre-update environment could
expose a neighbour through the no-communication fallback and its current
activity state; GraphMAPPO could additionally expose another node's local
self-observation through attention. These violate the actor boundary in the
scope below. The repaired protocol now gives an actor only own truth and
delivered packets, keeps a delivered packet until its configured staleness
expiry, prevents peer self-node features and current peer activity from
affecting an actor action, and uses an empty neighbour set when communication
is disabled. The centralized critic/execution-side CBF boundary is unchanged.

Consequently, every pre-update learned checkpoint, checkpoint evaluation,
paired/multi-arm summary, and CBF replay under the historical roots below is
retained for forensic provenance but **ineligible for every AAMAS result,
descriptive table, CBF-rate aggregation, or comparison**. This is a protocol
correction, not evidence about performance or safety. Do not delete, overwrite,
or regenerate the old records in place. Restart with distinct post-isolation
roots after the repaired code is committed.

The later post-isolation partial run
`outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260721/`
is separately invalid: it stopped at 91,776 transitions when a transient
Windows sharing violation blocked telemetry replacement, has `ABORTED.json`,
and lacks a final checkpoint/summary. Preserve it and exclude every partial
checkpoint, telemetry event, and launcher log. Its replacement must use a new
root after the bounded telemetry-retry repair; never overwrite it.

## Scope

Every included analysis must follow the single manuscript mainline:

> communication staleness, delay, and loss -> predicted neighbour-state
> uncertainty -> uncertainty-aware predictive interaction graph and coordinated
> action -> uncertainty-adaptive CBF safety margin -> dynamic-obstacle execution.

The actor may use only its own truth and delivered packets. Centralized critic
state is not an actor input. Neural-network training/inference uses CUDA; OSQP
CBF solving remains CPU-side. This plan does not authorize a controller change,
CBF tuning, a new imitation/hierarchical track, or external submission.

## Independent unit and aggregation rule

For every scenario and arm, first average the 20 retained episode records within
each `seed`--`scenario` JSONL cell. The five independently trained seeds are
the only descriptive repetition unit. Never count 100 episode records as 100
independent policies. Existing JSON summaries report seed means and sample
standard deviations only; they do not contain p-values, confidence intervals,
significance decisions, or superiority claims.

All summaries separate:

- `task_metrics`: task outcome, return, trajectory, separation, conflict,
  energy, and latency fields.
- `cbf_diagnostic_metrics`: emergency fallback, intervention, correction, and
  solve-time fields.

CBF diagnostic fields are not a safety guarantee or a substitute for retained
all-source CBF replay JSONL. Any future safety/fallback analysis must identify
the exact raw event sources and solver protocol separately.

## Eligible checkpoint-evaluation inputs

| Scale / comparison | Eligible raw inputs | Descriptive artifact | Explicit exclusions |
| --- | --- | --- | --- |
| 3-UAV four arms | Post-isolation roots: MLP `outputs/core_3uav_post_actor_isolation_evaluations/`; raw graph `outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo_rerun1/`; predictive no-uncertainty `outputs/core_3uav_post_actor_isolation_evaluations_20260801_predictive_graph/`; uncertainty-aware `outputs/core_3uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`. Each has five eligible checkpoints by six scenarios by 20 episodes. | None yet; a fresh prespecified aggregation using the independent seed as the inferential unit is still required. | Historical `outputs/core_3uav_evaluations/` and its paired summary remain protocol-ineligible. Preserve/exclude all documented interrupted/prelaunch roots, including MLP telemetry retry, raw/predictive failed roots, and the two uncertainty seed-20260719 interruptions. |
| 5-UAV full vs independent no-uncertainty | Full method: `outputs/core_5uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`; independent no-uncertainty: `outputs/core_5uav_post_actor_isolation_evaluations_20260801_predictive_graph/`. Each has five eligible independently trained checkpoints by six scenarios by 20 episodes, complete identity-audited raw JSONL and all-source replay provenance. | None; specify a fresh procedure using independent seed as the inferential unit before producing any paired/descriptive result. Historical paired JSON remains excluded. | Historical `outputs/core_5uav_evaluations/`, `outputs/core_5uav_ablation_evaluations_rerun4/`, initial/rerun1/rerun2/interrupted-rerun3 roots, the post-isolation non-retry full-method seed-20260721 root/checkpoint, and the zero-artifact seed-20260720 one-second prelaunch-timeout sidecar remain excluded. Seed-20260723's PowerShell stderr-redirection host code is a reproduced launcher-envelope artifact, not a failed training root. |
| 8-UAV full vs independent no-uncertainty | None until post-isolation retraining/evaluation. Historical `outputs/core_8uav_evaluations/` and `outputs/core_8uav_ablation_evaluations_20260729/` are retained only. | Historical `outputs/paired_summaries/8uav_full_vs_no_uncertainty_20260729.json` is excluded. | The preflight identity-render typo remains a no-result attempt; all historical evaluated cells are additionally protocol-ineligible. |

Each historical artifact has six canonical scenarios, five matched seed
identities, and 20 indexed episodes per seed-scenario cell. That proves only
historical completeness, not protocol eligibility or any result claim.

## Reproduction commands

The commands below are historical regeneration provenance only. Do not run them
to produce a reportable artifact; they aggregate protocol-ineligible inputs.
Future post-isolation commands must use new roots and an updated plan.

```powershell
D:\anaconda3\envs\multiuav_rl\python.exe scripts\summarize_multiarm_checkpoint_evaluations.py `
  --arm mlp_mappo=outputs\core_3uav_evaluations `
  --cell-prefix mlp_mappo=core_3uav_mlp_mappo_ `
  --arm raw_graph_mappo=outputs\core_3uav_evaluations `
  --cell-prefix raw_graph_mappo=core_3uav_raw_graph_ `
  --arm predictive_graph_no_uncertainty=outputs\core_3uav_evaluations `
  --cell-prefix predictive_graph_no_uncertainty=core_3uav_predictive_graph_ `
  --arm uncertainty_predictive_graph=outputs\core_3uav_evaluations `
  --cell-prefix uncertainty_predictive_graph=core_3uav_uncertainty_predictive_graph_ `
  --output-json outputs\paired_summaries\3uav_four_arm_descriptive_20260729.json

D:\anaconda3\envs\multiuav_rl\python.exe scripts\summarize_paired_checkpoint_evaluations.py `
  --reference-root outputs\core_5uav_evaluations `
  --treatment-root outputs\core_5uav_ablation_evaluations_rerun4 `
  --output-json outputs\paired_summaries\5uav_full_vs_no_uncertainty_20260729.json

D:\anaconda3\envs\multiuav_rl\python.exe scripts\summarize_paired_checkpoint_evaluations.py `
  --reference-root outputs\core_8uav_evaluations `
  --treatment-root outputs\core_8uav_ablation_evaluations_20260729 `
  --output-json outputs\paired_summaries\8uav_full_vs_no_uncertainty_20260729.json
```

The 5-UAV no-uncertainty input uses frozen
`configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`
(SHA-256 `6C42C3D5F957E3C8C21F496DCCA6D09F312AE95E816825EB9C301F539E22FA8B`).
The 8-UAV no-uncertainty input uses frozen
`configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`
(SHA-256 `F41E83D24EF472F7DAE981AC4D4228CD3B70C49AA18484CA1E0917BCEC105E73`).
Their CBF slack penalty, iteration cap, solve-time limit, and tolerances are not
changed by these commands.

## CBF diagnostic provenance

Keep raw replay events separate from task summaries. Current relevant retained
files include:

- `outputs/cbf_diagnostics/core_mlp_5seed_replay_20260722.jsonl`
- `outputs/cbf_diagnostics/raw_graph_5seed_replay_20260723.jsonl`
- `outputs/cbf_diagnostics/predictive_graph_5seed_replay_20260723_jsonlfix_rerun2.jsonl`
- `outputs/cbf_diagnostics/uncertainty_predictive_graph_5seed_replay_20260726.jsonl`
- `outputs/cbf_diagnostics/uncertainty_predictive_graph_5uav_5seed_replay_20260726.jsonl`
- `outputs/cbf_diagnostics/predictive_no_uncertainty_5uav_5seed_replay_20260729.jsonl`
- `outputs/cbf_diagnostics/uncertainty_predictive_graph_8uav_5seed_replay_20260727.jsonl`
- `outputs/cbf_diagnostics/predictive_no_uncertainty_8uav_5seed_replay_20260729.jsonl`
- `outputs/cbf_diagnostics/post_actor_isolation_uncertainty_predictive_graph_5uav_5seed_training_and_evaluation_replay_20260801.jsonl`
- `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_5uav_5seed_training_and_evaluation_replay_20260801.jsonl`

The final two 5-UAV files above are eligible post-isolation diagnostic
provenance; neither is a safety or fallback-rate result. All preceding
historical CBF files in the list remain protocol-ineligible for aggregate
reporting after the actor-boundary correction. Preserve them, their existing
nonreplayable records, and the predictive-graph dynamic-obstacle telemetry gap
as explicit forensic exclusions; no value is imputed.

## Manuscript gates still open

1. Do not convert the retained descriptive JSON into inferential or superiority
   language without a separately specified, reviewed statistical procedure and
   a check that it respects the five-seed independent unit.
2. Do not make a safety guarantee, CBF fallback-rate, real-time, or robust
   generalization claim from the current diagnostics.
3. Do not make a novelty/first-combination claim until the primary full texts
   recorded as unreviewed in `docs/related_work_matrix.md` are obtained and
   compared directly.
4. Complete a separate audit before editing the legacy MATLAB inventory/test;
   current full pytest is 152 passed, 1 legacy audit failure (83 observed `.m`
   files vs 81 asserted), and 28 OSQP deprecation warnings.
5. Before submission readiness, audit every manuscript figure/table against
   this file, raw JSONL provenance, CBF replay provenance, and documented
   invalid-root exclusions. External submission remains prohibited without
   explicit user authorization.

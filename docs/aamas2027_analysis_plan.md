# AAMAS 2027 analysis plan and evidence boundary

**Status:** operational reporting plan, recorded 2026-07-29 at revision
`019065b`. This is a retrospective provenance and reporting plan, not a
preregistration and not evidence of a method effect.

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
| 3-UAV four arms | `outputs/core_3uav_evaluations/` selected by the four prefixes `core_3uav_mlp_mappo_`, `core_3uav_raw_graph_`, `core_3uav_predictive_graph_`, and `core_3uav_uncertainty_predictive_graph_` | `outputs/paired_summaries/3uav_four_arm_descriptive_20260729.json` | Empty MLP seed-20260719 nominal/OOD originals. Use only the documented valid `nominal_schemafix_rngfix_rerun2` and `ood_communication_obstacle_trajectoryfix_rerun1` replacements. |
| 5-UAV full vs independent no-uncertainty | `outputs/core_5uav_evaluations/`; `outputs/core_5uav_ablation_evaluations_rerun4/` | `outputs/paired_summaries/5uav_full_vs_no_uncertainty_20260729.json` | `outputs/core_5uav_ablation_evaluations/`, `_rerun1/`, `_rerun2/`, and interrupted `_rerun3/`; none may enter aggregation. |
| 8-UAV full vs independent no-uncertainty | `outputs/core_8uav_evaluations/`; `outputs/core_8uav_ablation_evaluations_20260729/` | `outputs/paired_summaries/8uav_full_vs_no_uncertainty_20260729.json` | No alternate root is eligible. The preflight identity-render typo launched no evaluator and contributes no cell. |

Each listed artifact currently has six canonical scenarios, five matched seed
identities, and 20 indexed episodes per seed-scenario cell. This validates input
eligibility and descriptive aggregation scope only.

## Reproduction commands

Use the repository Python environment and do not modify input roots:

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

The known nonreplayable legacy records and the predictive-graph dynamic-obstacle
telemetry gap remain explicit errors/exclusions as recorded in
`docs/known_issues.md`; no value is imputed.

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

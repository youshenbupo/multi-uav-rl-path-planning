# Post-isolation descriptive aggregation validation

**Validation date:** 2026-08-02

**Frozen analysis-plan revision:** `030b50d`

**Overall assessment:** **Share with caveats** as an internal descriptive
artifact; **not ready for superiority, significance, causality, safety, or
robust-generalization claims**.

## Question and intended use

The validated artifacts summarize the post-actor-isolation checkpoint
evaluations without treating episode records as independent trained policies.
They support internal manuscript drafting and audit of the fixed research chain
only. They do not authorize external submission or an inferential claim.

## Dataset and grain

The source grain is `(arm, trained seed, scenario, episode)`. Every eligible
root has five independently trained seeds (`20260719`--`20260723`), six fixed
scenarios, and 20 indexed episodes per seed--scenario cell: 30 cells and 600
records per root. The 3-UAV four-arm artifact uses four roots (2,400 records);
the 5-UAV and 8-UAV paired artifacts use two roots each (1,200 records each),
for 4,800 independently re-read raw episode records in the validation pass.

Eligible inputs are exactly:

- 3-UAV MLP: `outputs/core_3uav_post_actor_isolation_evaluations/`
- 3-UAV raw graph: `outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo_rerun1/`
- 3-UAV predictive without uncertainty: `outputs/core_3uav_post_actor_isolation_evaluations_20260801_predictive_graph/`
- 3-UAV uncertainty-aware: `outputs/core_3uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`
- 5-UAV predictive without uncertainty: `outputs/core_5uav_post_actor_isolation_evaluations_20260801_predictive_graph/`
- 5-UAV uncertainty-aware: `outputs/core_5uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`
- 8-UAV predictive without uncertainty: `outputs/core_8uav_post_actor_isolation_evaluations_20260801_predictive_graph/`
- 8-UAV uncertainty-aware: `outputs/core_8uav_post_actor_isolation_evaluations_20260801_uncertainty_predictive_graph/`

## Methodology review

The aggregation matches the plan committed before output generation. Each
20-episode cell is averaged first. Only the five trained-seed cell means are
used for across-seed means and sample standard deviations. The 5-UAV and
8-UAV delta direction is full uncertainty-aware method minus independently
trained predictive-without-uncertainty method for the same seed. Task metrics
and CBF diagnostic metrics remain separated. No p-value, confidence interval,
multiple-comparison decision, ranking, or significance label is produced.

The existing summary-tool tests passed after a sandbox-only temporary-directory
failure was rerun with filesystem access: `9 passed`, with four existing OSQP
deprecation warnings. The first attempt had one pass and eight fixture setup
errors caused by denied access to pytest temporary/cache directories; it did
not exercise or invalidate the calculation logic and produced no summary
artifact.

## Data-quality and calculation checks

- Completeness: all eight roots have exactly 30 JSONL cells and 600 records.
- Uniqueness: every expected `(seed, scenario)` cell appears once; episode
  indices are exactly 0--19 with no duplicates or omissions.
- Consistency: all records within a cell agree on seed and scenario; paired and
  multi-arm roots have identical cell-key sets.
- Validity: all 16 frozen metrics are numeric in all 4,800 records. Rates are
  within `[0, 1]`; counts, time, distance, path, energy, latency, correction,
  and solve-time fields are nonnegative.
- Grain: raw episodes are never counted as independent policies. Output
  metadata states `trained_seed`, five seeds, and 20 episodes per cell.
- Independent recalculation: a validator that did not import either summary
  module recomputed every cell mean, sample standard deviation, and 5/8-UAV
  paired delta. Every output value matched within `1e-12` relative/absolute
  tolerance.
- Presentation integrity: all six scenarios and all 16 metrics remain present;
  recursive schema inspection found no p-value, confidence-interval, ranking,
  or significance field. Each stdout log parses to the exact same JSON as its
  saved artifact; all three stderr logs are empty.

## Validated artifacts

| Artifact | SHA-256 |
| --- | --- |
| `outputs/paired_summaries/post_actor_isolation_3uav_four_arm_descriptive_20260801.json` | `B3CB5F017A241DAF29C366DA9592624076A2FE44519F51809D5AE50251CF2F3E` |
| `outputs/paired_summaries/post_actor_isolation_5uav_full_vs_no_uncertainty_descriptive_20260801.json` | `6661DBDDF678039581ED7FEC0D84595A641C25B87C6877DCEA95C3ADD5359120` |
| `outputs/paired_summaries/post_actor_isolation_8uav_full_vs_no_uncertainty_descriptive_20260801.json` | `E487CD7C16EAA285025917848AD51268BDC11228174ED17EFB395DC740144C61` |

Each corresponding stdout log has the same hash as its JSON; every stderr log
has the empty-file SHA-256
`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`.

## Findings and analytical risks

The paired values are mixed rather than uniformly favorable to the full
method. As calculation spot checks, the full-minus-no-uncertainty success delta
is `-0.2` for 5-UAV delay-only and `-0.4` for 8-UAV delay-only; the 8-UAV
loss-only delta is `-0.09`. Collision deltas are zero in all six scenarios at
both scales. Return and minimum-separation deltas change sign across scenarios.
These observations are retained to prevent a one-sided narrative; they are not
endpoint selection or inferential conclusions. The complete JSON artifacts,
not these examples, are the authoritative descriptive record.

The principal limitation is five trained seeds. The outputs do not quantify
inferential uncertainty and cannot establish that any observed delta differs
from zero. Same-number seeds provide a blocking identity but not additional
replication. CBF fields are execution diagnostics; they are not safety
guarantees and do not replace the retained source-event/replay ledgers.

There is no temporal trend analysis: these are fixed experiment snapshots, not
a time series. Numerical variability in OSQP replay status/fallback counts is
already documented separately and is not reconciled by choosing a favorable
run.

## Explicit exclusion registry

The following retained `ABORTED` records and their associated attempts are
ineligible for every result, descriptive table, rate aggregation, or claim:

- `outputs/cbf_diagnostics/aborted_core_3uav_mlp_mappo_seed_20260719_attempt1/ABORTED.json`
- `outputs/cbf_diagnostics/post_actor_isolation_predictive_no_uncertainty_8uav_seed_20260721_replay_20260801.ABORTED.json`
- `outputs/core_3uav_post_actor_isolation/core_3uav_mlp_mappo_seed_20260721/ABORTED.json`
- `outputs/core_3uav_post_actor_isolation/core_3uav_predictive_graph_seed_20260722/ABORTED.json`
- `outputs/core_3uav_post_actor_isolation/core_3uav_raw_graph_mappo_seed_20260719/ABORTED.json`
- `outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260719/ABORTED.json`
- `outputs/core_3uav_post_actor_isolation/core_3uav_uncertainty_predictive_graph_seed_20260719_launcherretry1/ABORTED.json`
- `outputs/core_3uav_post_actor_isolation_evaluations_20260731_raw_graph_mappo/ABORTED.json`
- `outputs/core_3uav_post_actor_isolation_predictive_graph_seed_20260722_launcherretry1.ABORTED.json`
- `outputs/core_5uav_post_actor_isolation/core_5uav_uncertainty_predictive_graph_seed_20260721/ABORTED.json`
- `outputs/core_5uav_post_actor_isolation_ablation_seed_20260720_prelaunch_timeout_20260801.ABORTED.json`
- `outputs/core_8uav_post_actor_isolation_ablation/core_8uav_predictive_no_uncertainty_seed_20260722_retry1/ABORTED.json`
- `outputs/core_8uav_post_actor_isolation_ablation_seed_20260722_launcher.ABORTED.json`

Additionally, every learned checkpoint/evaluation/paired summary produced
before the actor-information isolation repair is protocol-ineligible. This
includes the historical `outputs/core_3uav_evaluations/`,
`outputs/core_5uav_evaluations/`, `outputs/core_8uav_evaluations/`, historical
5/8-UAV ablation-evaluation roots, and all pre-isolation files under
`outputs/paired_summaries/`. They remain forensic evidence only. An artifact is
eligible only when its exact root is listed in the eligible-input section or a
later documented plan explicitly adds it after the same audit discipline.

## Required caveats and next actions

1. Cite these summaries only as five-seed descriptive evidence, never as
   significance, superiority, causality, safety, or robust generalization.
2. Keep all scenarios and metrics visible when building manuscript tables;
   do not select only favorable cells.
3. Reconcile every table/figure cell back to the raw JSONL root and this report.
4. Keep CBF source/replay provenance separate from checkpoint-evaluation
   descriptive metrics.
5. Before external readiness, complete primary-paper full-text review and the
   manuscript/reproducibility audit. External submission remains prohibited.

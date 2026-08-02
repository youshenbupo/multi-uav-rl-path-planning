# AAMAS 2027 manuscript assembly package

**Purpose:** evidence-bounded drafting scaffold. This is not a finished paper,
an acceptance claim, or authorization to submit.

## Candidate title

**Communication-Uncertainty-Aware Predictive Graph Coordination with an
Adaptive CBF Execution Shield for Multi-UAV Systems**

Avoid "safe", "guaranteed", "robust", "calibrated", "real-time", or "first"
in the title unless later evidence directly establishes the corresponding
claim.

## Draft abstract

Delayed and lossy communication makes a multi-UAV controller act on stale
neighbour information, while collision avoidance must still be enforced during
execution around moving obstacles. We study an auditable coordination pipeline
that predicts each delivered neighbour packet forward using its timestamp and
velocity, derives a deterministic age-dependent uncertainty bound, injects the
bound into a predictive interaction graph, and uses the same quantity to adapt
the pairwise margin of an online control-barrier-function quadratic program.
The actor receives current truth only for itself and communication-delivered
information for neighbours. We evaluate independently trained policies with
five seeds under nominal, delayed, lossy, dynamic-obstacle, combined, and
out-of-distribution conditions at three, five, and eight UAVs. All raw episode
records and CBF fallback replay contexts are retained. The current descriptive
results are mixed across scale, scenario, and metric, so they support a
transparent characterization of the integrated mechanism rather than a claim
of uniform improvement. We discuss the centralized simulation shield,
uncalibrated age bound, limited seed count, and joint uncertainty ablation as
explicit limitations.

## Defensible contribution wording

The paper may currently claim that it:

1. implements a leakage-audited actor representation that predicts only
   delivered neighbour packets and removes expired packets from interaction;
2. propagates the same age-dependent bound into graph risk and an online CBF
   clearance margin;
3. provides independently trained 3/5/8-UAV evaluations across six fixed
   communication/obstacle conditions with retained episode and solver-event
   provenance; and
4. reports mixed five-seed descriptive evidence with explicit invalid-root and
   fallback-replay audits.

These are design and evaluation contributions. They are not novelty,
superiority, causality, formal-safety, or generalization claims.

## Section blueprint

### 1. Introduction

Motivate the coupled problem: communication impairment changes what agents
know, and an execution shield must act on the resulting uncertainty. State the
single mechanism chain and the exact evidence level. Do not open with a safety
guarantee or universal gap claim.

### 2. Related work

Organize by delayed communication, learned communication graphs, graph-CBF
MARL, and practical decentralized UAV avoidance. Use only distinctions recorded
in `docs/related_work_matrix.md`. DACOM and DHCG support narrow paper-specific
comparisons. The AAMAS graph-CBF source is only a three-page extended abstract,
and the IEEE UAV paper is metadata/abstract-only; retain those source-tier
caveats.

### 3. Problem formulation

Define the multi-agent state/action spaces, delivered-packet channel, packet
age and expiry, dynamic cylinder, local actor information, critic information,
and centralized execution-filter information separately. State that the age
bound is deterministic rather than statistically calibrated.

### 4. Method

Present packet prediction, uncertainty, edge features and risk-adjusted CPA,
edge-conditioned graph policy, MAPPO objective, uncertainty-adaptive CBF QP,
and emergency fallback. Explicitly distinguish the complete-state MLP critic
from the communication-limited graph critic and the centralized-geometry CBF.

### 5. Experimental protocol

Report five independently trained seeds, exact rollout-rounded transition
counts, six scenarios, 20 episodes per cell, independent 5/8-UAV ablation
training, raw JSONL retention, and exclusion rules. Say that seed-cell means are
the analysis units.

### 6. Results

Present all six scenarios and all prespecified metrics. Separate task outcomes
from CBF diagnostics. Lead with the mixed pattern, including unfavorable
success deltas and collision floor effects. Do not rank methods or interpret
descriptive standard deviations as inferential uncertainty.

### 7. Limitations and broader validity

Discuss five seeds; no inferential analysis; joint graph+CBF uncertainty
ablation at 5/8 UAV; no graph-only/CBF-only causal isolation; deterministic
uncertainty proxy; centralized simulator truth in the CBF; OSQP timing
sensitivity; slack/fallback/discrete-time limits; fixed simulated geometry;
and incomplete full-text access for one important comparator.

### 8. Reproducibility statement

Link every table/figure to eligible raw roots, config and Git hashes, exact
aggregation commands, and retained replay records. Include invalid/interrupted
attempts as exclusions rather than deleting them.

## Result-language gate

Allowed examples:

- "Across five trained seeds, the descriptive mean changed by ..."
- "The direction varied across scenarios."
- "No evaluation collision-rate difference was observed in these cells, where
  both arms were at the same floor."
- "CBF intervention and fallback fields are execution diagnostics."

Prohibited until new evidence exists:

- "significantly outperforms", "is safer", or "generalizes robustly";
- "guarantees collision avoidance" or "decentralized CBF";
- "uncertainty is calibrated";
- "real-time" based only on simulator decision latency or accepted QP limits;
- "first method" or any exhaustive novelty statement;
- a component-level graph or CBF causal claim from the joint 5/8-UAV ablation.

## Figure and table manifest

No figure or table is paper-ready until it has a sidecar ledger containing:

- eligible raw input roots and hashes;
- Git revision and config hashes;
- seed/scenario/episode counts;
- aggregation script and complete command;
- metric definition and delta direction;
- excluded roots that could otherwise be confused with the input;
- separate provenance for CBF diagnostics; and
- a wording check against the result-language gate above.

Recommended initial tables are the complete 3-UAV four-arm descriptive matrix
and complete 5/8-UAV full-minus-joint-ablation matrices. A compact figure may
show scenario-wise paired deltas only if it displays all six scenarios and
clearly labels five trained seeds and descriptive uncertainty.

## Remaining manuscript gates

1. Keep the current five-seed cohort descriptive as fixed in
   `docs/aamas2027_statistical_decision.md`; freeze a new plan before viewing
   any separately collected confirmatory cohort.
2. Decide whether to train graph-only and CBF-only independent ablations or
   keep claims explicitly at the integrated-pathway level.
3. Expand primary-source literature coverage and obtain the closed IEEE full
   text before stronger comparisons.
4. Use the validated PDF surface recorded in
   `docs/aamas2027_pdf_report_validation.md` as the portable descriptive view;
   still audit every manuscript-specific table/figure value and sidecar.
5. Complete a final consistency, reproducibility, and limitations review.
6. Obtain explicit user authorization before any external submission action.

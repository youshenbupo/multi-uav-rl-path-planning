# AAMAS 2027 research brief: uncertainty-aware coordination

## Locked question

Under delayed and lossy communication, can a multi-UAV policy coordinate more safely when
it converts each delivered neighbour packet into a predicted state with an age-dependent
uncertainty bound, and injects that bound into both predictive interaction graphs and an
execution-side CBF margin?

This is a **multi-agent coordination** question. Dynamic cylinders are evaluation semantics;
BC is a possible initialization; hierarchy is retained only if independently trained scale
experiments demonstrate a benefit.

## Information boundary and notation

For receiver `i` and sender `j`, a delivered packet contains immutable
`(p_ij^rec, v_ij^rec, t_ij^rec)`. At current step `t`, its age is
`a_ij = t - t_ij^rec`. The actor may use only a valid received packet and computes

`p_hat_ij = p_ij^rec + a_ij * dt * v_ij^rec`

`sigma_ij = gamma_sigma * a_ij`.

`gamma_sigma` is an explicit robustness-model parameter, not a calibrated physical sensor
standard deviation. Invalid packets create no actor graph edge and have zero padded features.
The MLP critic uses complete environment state under CTDE; the current graph critic instead
receives the same communication-limited node/edge representation as the graph actor. Neither
critic is an actor input. The simulated joint CBF uses true geometry to build constraints,
while `sigma_ij` may increase its pairwise clearance margin; this is not a decentralized-CBF
guarantee.

For CPA separation `d_CPA`, graph risk uses

`d_CPA^risk = max(0, d_CPA - kappa_sigma * sigma_ij)`.

For a CBF pairwise safety distance `d_safe`, the filter uses

`d_safe^CBF = d_safe + min(m_max, kappa_CBF * max(sigma_ij, sigma_ji))`.

## Method claim to test

The full method uses predicted delivered state, age, uncertainty, risk-adjusted CPA graph
features, and uncertainty-tightened CBF. At 3 UAV, the predictive-graph arm removes
uncertainty from graph risk while retaining the shared uncertainty-adaptive CBF. At 5 and 8
UAV, the independently trained comparison disables both graph uncertainty risk and the CBF
uncertainty margin. The latter is a joint pathway ablation, not separate causal isolation of
the graph and CBF components.

## Implemented contributions under evaluation

1. A delivery-aware interaction representation that predicts neighbour state from packet age
   and exposes a deterministic uncertainty bound without actor truth leakage.
2. An uncertainty-aware predictive graph that makes communication degradation alter
   both edge risk and graph connectivity.
3. A risk-adaptive execution-side CBF margin and a reproducible evaluation protocol for
   delayed/lossy communication plus dynamic obstacles.

These are implemented method contributions, not accepted novelty claims. Full primary-source
review of DACOM and DHCG now supports narrow paper-specific distinctions, while graph-CBF and
practical decentralized-UAV sources remain limited by short-version or full-text access.
Broader systematic coverage is still required before novelty or "first" language.

## Required evidence before paper claims

- Separately trained MLP-MAPPO, naive graph, predictive graph without uncertainty, and full
  method artifacts.
- Five fixed seeds for 3, 5, and 8 UAVs, with nominal, delay, loss, dynamic, combined, and
  out-of-distribution scenarios.
- Paired raw episode records with success, collision, minimum separation, temporal conflict,
  energy proxy, decision latency, CBF intervention, emergency fallback, and QP time.
- Independently trained graph-only and CBF-only uncertainty ablations if the paper seeks
  component-level causal claims. The completed 5/8-UAV ablation only identifies the joint
  uncertainty pathway.

## Claims that are prohibited until evidence exists

- No claim of statistically significant improvement, safety guarantee, real-time scalability,
  calibrated uncertainty, or generalization from smoke runs.
- No claim of a source-faithful CA-HGALO baseline or expert advantage without authoritative
  source/rollout artifacts.
- No substituted controller or checkpoint may stand in for an unavailable baseline.

## Literature evidence protocol

Continue searching primary venues and publisher pages for: `delayed communication multi-agent
reinforcement learning`, `lossy communication graph MARL`, `uncertainty-aware interaction
graph multi-agent`, `multi-UAV conflict resolution MARL`, and `control barrier function
multi-agent reinforcement learning`. Record venue, year, task assumptions, whether packet
age/uncertainty changes graph topology, whether CBF is decentralized, source tier, and the
exact paper-specific distinction. Do not make a method-difference claim without full text;
do not cite an item until its bibliographic identity and authoritative source are verified.
The live evidence ledger is `docs/related_work_matrix.md`.

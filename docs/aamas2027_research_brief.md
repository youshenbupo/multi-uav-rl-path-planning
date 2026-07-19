# AAMAS 2027 research brief: uncertainty-calibrated coordination

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
The centralized critic may use environment truth under CTDE. The simulated joint CBF uses
true geometry only to build constraints, while `sigma_ij` may increase its pairwise clearance
margin; this is not a decentralized-CBF guarantee.

For CPA separation `d_CPA`, graph risk uses

`d_CPA^risk = max(0, d_CPA - kappa_sigma * sigma_ij)`.

For a CBF pairwise safety distance `d_safe`, the filter uses

`d_safe^CBF = d_safe + min(m_max, kappa_CBF * max(sigma_ij, sigma_ji))`.

## Method claim to test

The full method uses predicted delivered state, age, uncertainty, risk-adjusted CPA graph
features, and uncertainty-tightened CBF. The principal controlled comparison is a predictive
graph using the same delivered packets but with `gamma_sigma = 0` and
`kappa_sigma = kappa_CBF = 0`.

## Proposed contributions, pending evidence

1. A delivery-aware interaction representation that predicts neighbour state from packet age
   and exposes a deterministic uncertainty bound without actor truth leakage.
2. An uncertainty-calibrated predictive graph that makes communication degradation alter
   both edge risk and graph connectivity.
3. A risk-adaptive execution-side CBF margin and a reproducible evaluation protocol for
   delayed/lossy communication plus dynamic obstacles.

These are implemented method contributions, not accepted novelty claims. A systematic
first-party literature search must establish distinction from delayed-communication MARL,
uncertainty-aware graph coordination, and CBF-shielded MARL before manuscript drafting.

## Required evidence before paper claims

- Separately trained MLP-MAPPO, naive graph, predictive graph without uncertainty, and full
  method artifacts.
- Five fixed seeds for 3, 5, and 8 UAVs, with nominal, delay, loss, dynamic, combined, and
  out-of-distribution scenarios.
- Paired raw episode records with success, collision, minimum separation, temporal conflict,
  energy proxy, decision latency, CBF intervention, emergency fallback, and QP time.
- Ablations trained independently for age, uncertainty, CPA prediction, graph, CBF, and any
  retained hierarchy/BC component.

## Claims that are prohibited until evidence exists

- No claim of statistically significant improvement, safety guarantee, real-time scalability,
  calibrated uncertainty, or generalization from smoke runs.
- No claim of a source-faithful CA-HGALO baseline or expert advantage without authoritative
  source/rollout artifacts.
- No substituted controller or checkpoint may stand in for an unavailable baseline.

## Literature-search protocol

Search primary venues and publisher pages for: `delayed communication multi-agent
reinforcement learning`, `lossy communication graph MARL`, `uncertainty-aware interaction
graph multi-agent`, `multi-UAV conflict resolution MARL`, and `control barrier function
multi-agent reinforcement learning`. Record venue, year, task assumptions, whether packet
age/uncertainty changes graph topology, whether CBF is decentralized, and the exact gap this
project can still claim. Do not cite an item until its title, authors, venue, and source page
are independently verified.

# Predictive Conflict Graph and GraphMAPPO Design

## Goal

Replace the Phase-9 actor/critic observation encoder with a pure-PyTorch,
permutation-equivariant predictive spatiotemporal conflict-graph encoder. This
phase adds neither a hierarchical policy nor a control-barrier-function filter.

## Graph contract

`ConflictGraphBuilder` receives batched position, velocity, goal, and active
mask tensors with shapes `[B,N,3]`, `[B,N,3]`, `[B,N,3]`, and `[B,N]`. It
returns a dense graph with node mask `[B,N]`, adjacency `[B,N,N]`, and edge
features `[B,N,N,15]`. Edge `(i,j)` is directed from receiver `i` to neighbour
`j`; all edge tensors use that same ordering.

For each ordered non-self pair it computes `delta_p = p_j-p_i`,
`delta_v = v_j-v_i`, clipped constant-velocity CPA time, CPA distance, and
the normalized predicted shortfall. The 15 features are relative position (3),
relative velocity (3), current distance, `t_cpa`, `d_cpa`, predicted shortfall,
unit relative goal direction (3), communication availability, and both-active.
Zero relative velocity is stabilized with `eps`; zero goal displacement has a
zero direction. All scalar distances and times are normalized by configurable
scales before neural encoding.

An edge exists only for two active nodes and one of: an enabled self-loop, a
current-distance edge inside the communication/risk radius, a predicted-conflict
edge inside risk distance within the finite prediction horizon, or membership
in the sender's configurable top-k nearest active neighbours. The builder never
creates an always-complete graph. The current-distance ablation disables only
predicted-conflict edges; the predictive graph enables them.

## Encoder and policies

`DenseGraphAttentionEncoder` projects node and edge features, then applies
masked multi-head dot-product attention over neighbour dimension `j`. Invalid
scores are masked before softmax and every valid node receives a self-loop, so
there is no all-masked softmax row. Each layer has residual projection,
LayerNorm, feed-forward residual, and active-mask zeroing. Shared projections
and masked aggregation make the encoder permutation equivariant: permuting UAV
nodes and all graph axes permutes node embeddings identically.

`GraphActor` maps one encoded node embedding to a squashed Gaussian action.
`GraphCentralizedCritic` computes a masked mean graph embedding and predicts a
single centralized value, replicated for active agents only when stored in the
PPO batch. `GraphMAPPOTrainer` retains the Phase-9 PPO, GAE, clipping,
normalization, checkpoint, and deterministic evaluation semantics while adding
an independently weighted auxiliary loss.

## Auxiliary targets

The graph encoder has an edge head for probability of an inter-UAV conflict in
the next configurable `H` recorded simulation steps and a nonnegative predicted
minimum-distance head. Rollout collection stores positions, active masks, and
episode-boundary information. After a rollout, targets are formed only from
future samples in the same episode; otherwise their labels are masked. Binary
cross entropy and Smooth L1 regression are averaged over valid non-self pairs
and multiplied by independent `aux_conflict_coef` and `aux_distance_coef`.
They are added after the PPO policy/value/entropy objective and logged
separately, so setting both coefficients to zero exactly recovers graph PPO.

## Training comparison

The graph runner supports 3, 5, and 8 UAV scenarios generated deterministically
from a seed, agent mask, and mission layout. For each count it runs three named
encoders with the same rollout, optimizer, seed family, and evaluation budget:
`mappo` (no graph), `distance_graph` (current-distance edges only), and
`predictive_graph` (CPA edges enabled). Results are written as JSON summaries
with scenario size, graph mode, seed, return, success, collision, separation,
and auxiliary metrics; the comparison script aggregates them without claiming
statistical significance from a single seed.

## Required verification

Tests cover exact head-aligned CPA values, parallel equal-velocity stability,
inactive nodes, no-neighbour finite output, variable `N`, attention tensor
shapes, and node permutation equivariance. Integration tests run an on-policy
GraphMAPPO update with auxiliary targets and assert finite losses/checkpoint
reload. Experiments execute all nine required mode/count combinations and
record their actual outputs.

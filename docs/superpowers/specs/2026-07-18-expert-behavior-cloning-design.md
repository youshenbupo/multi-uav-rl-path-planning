# Expert Behavior-Cloning Pretraining Design

## Goal

Train graph-conditioned low- and high-level behavior-cloning (BC) policies from
the validated HDF5 expert episodes in `data/expert`, then make their checkpoints
usable by the Phase-10 graph MAPPO and Phase-11 hierarchical MAPPO stacks. This
phase adds no CBF, safety shield, imitation-derived labels, or joint random
initialization path.

## Dataset boundary and split

The loader accepts an explicit list of verified HDF5 shards. The default uses
the six `s1_30seed_verified*_experts.h5` files as disjoint shards (the
un-suffixed file is part 01). Episode identifiers are namespaced by source file
to avoid collisions. Every split assignment is made once per episode from its
`scenario_id` and `seed`; all time steps and UAV rows from that episode remain
in the same split. The manifest stores source file, episode ID, scenario, seed,
UAV count, split, low-label count, and high-label count.

Variable UAV count is represented by per-batch padding and `active_mask`.
The loader reads the stored variable `edge_index`, `edge_features`, and
`edge_mask` into an audit tensor so raw graph validity remains observable.
For policy input and checkpoint compatibility, it reconstructs the existing
Phase-8 local observations from each stored scenario/state and builds the
existing Phase-10 15-feature predictive conflict graph from position, velocity,
goal, and active mask. Thus the BC encoder has exactly the node and edge widths
used by graph MAPPO; padded nodes and raw/staged edges remain invalid. No
synthetic expert edge labels or valid actions are introduced.

The high-level supervision rule is strict: `high_level_action_mask` is the
only authority for maneuver classification and optional targets. An unlabeled
entry is excluded from every high loss and metric; it is never recoded as
`continue`. The currently available verified shards contain sparse high labels,
so a split with zero such labels reports `null` high metrics and does not claim
high-policy quality.

## Networks and losses

`ExpertLowLevelPolicy` reuses the pure-PyTorch dense masked graph encoder. It
receives reconstructed Phase-8 local observations, a Phase-10 graph embedding,
and a held high-level command one-hot vector, and outputs a normalized desired
velocity. The MAPPO-compatible encoder itself consumes local observations and
the 15-feature predictive graph; the command is applied through a separate
conditioning projection so encoder state can transfer shape-safely.
The initial data has no dense authority for held commands, so its default
conditioning is an explicit all-continue one-hot context for low BC only; it is
not a target relabeling and does not affect high-level losses. It is controlled
by the training configuration and recorded in the manifest.

The low target is `low_level_actions / max_speed`, clipped to the environment's
normalized action range. Its masked loss is the weighted sum of normalized MSE,
one-minus direction cosine for nonzero target speeds, and normalized speed
magnitude error. Validation records velocity RMSE in physical units, direction
cosine, and a closed-loop endpoint rollout error.

`ExpertHighLevelPolicy` uses the same MAPPO-compatible graph encoder and has
independent heads for maneuver categorical logits, delay, priority, and
three-dimensional local subgoal. Only maneuver labels available in the schema
are optimized in the provided data; regression-head masks are false unless a
future authoritative HDF5 field carries that target. Their zero-count metrics
are `null`, not zero.

## Training, artifacts, and integration

Training uses deterministic episode-level split allocation, split-local
normalization derived from low-training samples only, early best-checkpoint
selection by validation low loss, and optional high checkpoint selection by
validation maneuver loss only when labelled validation samples exist. It writes
the required `bc_low_level.pt`, `bc_high_level.pt`,
`normalization_stats.json`, `training_config.yaml`, and `dataset_manifest.json`.

The closed-loop evaluator initializes the existing parallel environment with
each held-out episode's mission geometry and compares a deterministic BC policy
against expert trajectory samples. It reports arrival rate, collision rate,
minimum separation, path length, and masked expert-path deviation. Results are
diagnostics rather than a claim of feasibility equivalence.

MAPPO initialization accepts either or both BC checkpoints. It can freeze the
loaded graph encoder for a configured number of updates, then unfreeze it, use
a separate lower PPO fine-tuning learning rate, and optionally add a masked
imitation auxiliary loss on sampled expert batches. The default disables this
auxiliary term. Experiment configuration supports random initialization,
low-only BC, high-only BC, full BC, and BC plus MAPPO fine tuning.

## Verification

Tests cover shard de-duplication, episode-only split isolation, padded variable
UAV/edge masks, strict high-label masking, each low loss term, checkpoint and
normalization artifacts, closed-loop metrics, encoder freeze/unfreeze, and
comparison configuration validation. A short real low-BC run is required;
high-BC output must state its labelled sample count and cannot report a metric
for an empty split.

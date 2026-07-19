# Hierarchical Multi-Agent Policy Design

## Scope

Phase 11 adds a two-time-scale policy above the Phase-10 predictive conflict
graph: graph state feeds a discrete high-level coordination policy, which
conditions a low-level velocity policy. It does not add behavior cloning or a
CBF.

## Action and timing contract

The high level emits one categorical command per active UAV: `continue`,
`wait_or_yield`, `shift_left`, `shift_right`, `climb`, or `descend`. A
`HighLevelActionState` stores command, one-hot context, optional local-subgoal
offset, priority, delay, and remaining hold count as `[E,N,*]` tensors. At a
high boundary (`remaining == 0`) the policy emits a replacement command and
sets remaining to the configured interval `H`; after every low step remaining
decrements. Inactive or terminated UAVs are reset immediately and never emit a
stale command.

The first trainable configuration enables only categorical commands. Subgoal,
priority, delay, and altitude manoeuvre are explicit configuration flags and
default off. With optional fields off their context tensors are zero. The
`wait_or_yield` command gates its low-level action to zero; shift and altitude
commands contribute only configured local-subgoal offsets, leaving the low
level responsible for continuous `vx`, `vy`, `vz`.

## Networks

`HighLevelActor` uses the existing masked graph attention encoder and produces
six categorical logits per active node. `HighLevelCritic` masked-pools node
embeddings to a centralized graph value. `LowLevelActor` appends high-action
one-hot code, normalized remaining duration, and local-subgoal offset to each
local node observation before graph encoding, then emits a tanh-squashed
Gaussian velocity action. `HierarchicalPolicy` owns both actors and action
hold state; it is the only environment-facing policy surface.

## Rollouts and optimization

Low transitions are stored each simulator step. At every high action boundary,
the high buffer opens one transition; it accumulates low rewards as
`sum(gamma_low^k * reward_k)` until the command is replaced or the episode
ends. High GAE uses `gamma_low^duration` and the actual duration, not a fixed
interval after early termination. High and low PPO records retain separate
actions, log probabilities, values, rewards, terminals, and GAE tensors.

Stage A uses a deterministic rule coordinator derived from predictive graph
shortfall and trains only the low actor/critic. Stage B loads that low policy,
freezes it, and trains high actor/critic. Stage C is disabled by default and
only performs joint fine-tuning when explicitly enabled. The trainer logs each
stage separately and never starts both learned levels from random weights.

## Verification

Tests prove command persistence for exactly `H` steps, correct high boundaries,
early episode closure, high/low buffer alignment, duration-aware high discount,
checkpoint round-trip, and finite low-level optimization with the rule high
coordinator. YAML exposes toggles for high policy, subgoal, priority, delay,
and altitude manoeuvre.

# Basic MAPPO Baseline Design

## Scope

Phase 9 proves the ordinary multi-agent PPO training pipeline against the
Phase-8 PettingZoo environment. It includes neither a GNN, predicted conflict
graph, hierarchical action, expert pretraining, nor a CBF. The actor is shared
between all UAVs; each action consumes only that UAV's local observation. The
critic consumes only the fixed global `state()` vector.

## Data contract

The runner maintains a list of `MultiUAVParallelEnv` instances and treats each
fixed scenario agent as a `[environment, UAV]` training sample. Rollouts are
stored as tensors `[T, E, N, ...]` for local observation, global state, action,
pre-tanh-free Gaussian log probability, scalar performance reward, terminal
mask, truncation mask, old value, and next value. Safety costs are logged but
not added to the baseline reward.

GAE sets the bootstrap mask to zero for a true terminal event and retains the
critic next value for a time-limit truncation. An episode reset happens after
either condition, but the stored last transition still distinguishes the two.

## Networks and action distribution

`SharedGaussianActor` is an MLP mapping a local observation to a Normal mean
and one learned, state-independent action log-standard-deviation vector.
Sampling uses `Normal.sample()`; the resulting action is clipped to the
environment Box bounds and its independent-dimension log probabilities and
entropy are summed over the final action dimension. Evaluation uses the clipped
mean deterministically. `CentralizedCritic` is an independent MLP over the
global state.

This deliberately simple Gaussian baseline does not introduce a learned
high-level action or graph representation. The actor sees no centralized state
at execution time.

## Optimizer

Each update normalizes finite advantages, uses PPO's clipped probability-ratio
surrogate, clipped critic loss around the rollout value, entropy bonus, and
`clip_grad_norm_` on the actor/critic parameters. Observation normalization is
a serializable running mean/variance transform. Reward normalization is optional
and applies only to rollout rewards before GAE. The optimizer reports policy
loss, value loss, entropy, approximate KL, clip fraction, explained variance,
and finite-gradient checks.

## Checkpoint and evaluation

A checkpoint contains actor, critic, optimizer, normalization statistics,
training step, configuration, and Python/NumPy/PyTorch RNG states. Loading
restores all of them and permits deterministic evaluation. The evaluation CLI
uses actor means, no gradient calculation, and writes scalar metrics to JSON.

## Required training evidence

The runner first trains on a deterministic empty two-UAV scenario with
well-separated parallel missions. A later verification run uses the same
missions plus one simple cylinder. Each run records TensorBoard scalars:
episode return, success/collision rates, mean path length, minimum separation,
policy/value loss, entropy, approximate KL, clip fraction, explained variance,
and FPS. The report compares the first and final evaluation windows rather than
claiming a result from an unverified single episode.

## Design self-review

- All named Phase-9 modules, CLIs, YAML keys, losses, and test categories have
  a clear owner.
- The centralized/decentralized data boundary is explicit and does not leak
  global state to the actor.
- Terminal and time-limit bootstrap semantics are unambiguous.
- This scope ends before every feature deferred by the Phase-9 prompt.

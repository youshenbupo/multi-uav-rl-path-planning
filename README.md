# Multi-UAV Reinforcement Learning Path Planning

This project develops a staged Python research implementation for multi-UAV
spatiotemporal path planning in 3D terrain and threat environments. The target
research line is CA-HGALO expert demonstrations, a predictive spatiotemporal
conflict graph, hierarchical MAPPO, and a CBF safety filter. These components
are introduced only after their prerequisite validation stages.

## Create and activate the environment

```powershell
conda env create --file environment.yml
conda activate multiuav_rl
```

The checked-in environment file records the versions verified on the Windows
development machine. It intentionally does not install PyTorch Geometric.

## Environment check

```powershell
python scripts/check_environment.py
```

The report shows Python, NumPy, PyTorch, CUDA, GPU, Gymnasium, CVXPY, and OSQP
availability.

## Tests and static checks

```powershell
pytest
ruff check .
mypy multiuav scripts
```

No path-planning algorithm is implemented during the project-scaffold stage.

## Expert episode data

Convert the deterministic MATLAB fixture, validate the resulting HDF5 episode,
and render its 3D/top-view check:

```powershell
python scripts/convert_matlab_experts.py
python scripts/validate_expert_dataset.py data/expert/matlab_experts.h5
python scripts/visualize_expert_episode.py data/expert/matlab_experts.h5
```

See [the HDF5 schema](docs/expert_dataset_schema.md) for fixed time-series
shapes, variable-edge graph encoding, masks, labels, and validation rules.

## HGALO reconstruction expert runs

The restored MATLAB archive provides HGALO operators but no independent
CA-HGALO entrypoint. The Python planner therefore records the explicit
`HGALO_PYTHON_COORDINATED_RECONSTRUCTION` provenance rather than claiming a
MATLAB-verified CA-HGALO result.

```powershell
python scripts/run_ca_hgalo.py --scenario configs/scenarios/s1.yaml
python scripts/generate_expert_data.py --seed-start 0 --seed-count 5
python scripts/summarize_expert_runs.py --scenario-id s1_low_threat_3uav --records data/expert/ca_hgalo_records.jsonl --output data/expert/ca_hgalo_summary.json
```

Each completed seed writes its HDF5 episode and JSONL record before the next
seed begins. HDF5 episodes contain final/repaired trajectories, evaluator cost
components, conflict graph, scheduling and repair logs, high-level labels, and
the 55-generation optimizer convergence history.

## Multi-UAV environment MVP

Phase 8 provides `MultiUAVParallelEnv`, a native PettingZoo `ParallelEnv` with
Gymnasium `Box` spaces. It uses a configurable 3D single integrator and exposes
local observations plus `state()` for a future centralized critic. Performance
reward components and collision, terrain, threat, and boundary safety costs are
kept separate in each agent's `info` dictionary.

```powershell
python scripts/visualize_environment_episode.py `
  --scenario configs/scenarios/s1.yaml `
  --environment-config configs/env/multi_uav_mvp.yaml `
  --output data/expert/environment_episode_overview.png
```

The visualization contains a 3D trajectory, top view, and minimum-separation
curve. This stage intentionally contains no MAPPO, GNN, hierarchical policy,
expert pretraining, or CBF implementation.

## Basic MAPPO baseline

Phase 9 adds only a parameter-sharing Gaussian MAPPO baseline: decentralized
local-observation actor, centralized-state critic, GAE, PPO clipping, and
deterministic mean-action evaluation. It intentionally contains no GNN,
conflict predictor, hierarchy, expert pretraining, or CBF.

```powershell
python scripts/train_mappo.py --device cpu --output-dir data/rl/mappo_empty
python scripts/train_mappo.py --device cpu --with-cylinder --output-dir data/rl/mappo_cylinder
python scripts/evaluate_mappo.py --device cpu --checkpoint data/rl/mappo_empty/checkpoints/mappo_final.pt
```

TensorBoard records `episode_return`, `success_rate`, `collision_rate`,
`mean_path_length`, `minimum_separation`, `policy_loss`, `value_loss`,
`entropy`, `approx_kl`, `clip_fraction`, `explained_variance`, and `fps`.

## Predictive conflict-graph MAPPO

Phase 10 replaces the node encoder with a pure-PyTorch masked graph-attention
encoder. It can run node-only MAPPO, current-distance graphs, or finite-horizon
CPA predictive-conflict graphs without PyTorch Geometric. Auxiliary edge heads
predict in-episode future conflicts and minimum distance, with independently
configured weights.

```powershell
python scripts/train_graph_mappo.py --num-uavs 3 --graph-mode predictive_graph --device cpu
python scripts/evaluate_graph_mappo.py --num-uavs 3 --graph-mode predictive_graph --checkpoint data/rl/graph_mappo/checkpoints/graph_mappo_final.pt --device cpu
python scripts/compare_graph_mappo.py --device cpu
```

The required comparison runs `mappo`, `distance_graph`, and `predictive_graph`
for 3, 5, and 8 UAVs with matched seeds within each team size. It is an
ablation runner, not evidence of superiority from a single short run. Phase 10
does not add a hierarchical policy or CBF.

## Hierarchical graph MAPPO

Phase 11 adds a two-time-scale graph policy. Every `high_interval` low-level
steps, the high policy issues one of six discrete coordination commands:
continue, yield, shift left/right, climb, or descend. The command is held until
the next boundary. A conditioned low policy then emits normalized `vx`, `vy`,
and `vz` actions. Low- and high-level PPO retain separate log probabilities,
values, GAE, optimizers, and checkpoints; high rewards use the actual command
duration for discounting.

Train in order: first a low policy under the deterministic rule coordinator,
then restore that checkpoint and train only the high policy. Joint fine tuning
is rejected unless `allow_joint_finetune: true` is explicitly set.

```powershell
python scripts/train_hierarchical_mappo.py --stage low --device cpu --output-dir data/rl/hierarchical_stage_a
python scripts/train_hierarchical_mappo.py --stage high --device cpu --load-checkpoint data/rl/hierarchical_stage_a/hierarchical_low_final.pt --output-dir data/rl/hierarchical_stage_b
python scripts/evaluate_hierarchical_mappo.py --stage high --device cpu --checkpoint data/rl/hierarchical_stage_b/hierarchical_high_final.pt
```

`configs/rl/hierarchical_mappo.yaml` exposes the high interval and ablations
for the learned high policy, local subgoal offsets, priority scores, suggested
delays, and altitude maneuvers. The first implementation enables only the
discrete command; the continuous coordination fields are reserved zero-valued
extensions. It adds neither expert cloning nor a CBF safety layer.

## Strict-mask expert behavior cloning

Phase 12 trains separate low- and high-level graph BC policies from only the
six validated expert HDF5 shards. Splits are by complete episode, all source
activity/edge/action masks remain intact, and unavailable high labels remain
`-1` with a false mask. The low policy learns normalized velocity with MSE,
direction, and speed losses; the high policy never treats an unlabeled row as
the `continue` class.

```powershell
python scripts/train_behavior_cloning.py --config configs/rl/behavior_cloning.yaml --device cpu --output-dir data/rl/behavior_cloning
python scripts/evaluate_behavior_cloning.py --checkpoint data/rl/behavior_cloning/bc_low_level.pt --manifest data/rl/behavior_cloning/dataset_manifest.json --device cpu
python scripts/compare_bc_initialization.py --config configs/experiments/bc_comparison.yaml
python scripts/train_graph_mappo.py --device cpu --bc-low-checkpoint data/rl/behavior_cloning/bc_low_level.pt --freeze-encoder-updates 2 --fine-tune-learning-rate 0.0001
```

BC creates `bc_low_level.pt`, `bc_high_level.pt`, `normalization_stats.json`,
`training_config.yaml`, and `dataset_manifest.json`. Graph MAPPO imports only
shape-compatible tensors, applies the recorded train-split node normalization,
freezes its encoder for the configured initial updates, and supports a
strict-mask expert low-action auxiliary loss when a `GraphImitationBatch` is
provided. The comparison YAML defines `random`, `low_bc_only`, `high_bc_only`,
`full_bc`, and `bc_mappo_finetune`; the resolver shows missing artifacts rather
than presenting them as completed experiments. Phase 12 itself does not add a
CBF safety filter.

## CBF execution safety filter

Phase 13 adds an execution-only, joint direct-OSQP CBF filter for the
three-dimensional single-integrator environment. It jointly filters all active
UAV desired velocities for separation, terrain clearance, cylindrical threats,
world bounds, horizontal speed, and vertical speed. It is deliberately outside
PPO backpropagation. Failed, slow, non-finite, or non-solved QPs never return
the RL action: a bounded hover/repulsion emergency policy is used instead.

```powershell
python scripts/test_cbf_scenarios.py --scenario all
```

`configs/safety/cbf.yaml` owns CBF gain, slack penalty, solve-time limit, terrain
gradient step, and the conservative 16-sided horizontal-speed polygon. Each
decision records the requested and safe velocities, intervention norm, active
constraints, maximum slack, solver status/time, and emergency-fallback flag.
Direct OSQP runs on CPU; it solves a small execution-side QP and is independent
of the GPU used for neural-policy training.

# Phase-14 dynamic-world experiment protocol

The sole algorithmic mainline is **CA-HGALO demonstrations -> BC initialization
-> predictive-conflict-graph hierarchical MAPPO -> execution-side CBF-QP ->
dynamic-world evaluation**. BC is an initialization source, not a competing
final policy. Dynamic obstacles and delayed/lossy communication are environment
semantics shared by all applicable comparisons.

`scripts/train.py`, `scripts/evaluate.py`, `scripts/run_benchmark.py`, and
`scripts/run_ablation.py` share `--config`, `--seed`, `--device`, `--num-uavs`,
`--scenario`, `--checkpoint`, `--render`, `--use-expert-pretrain`,
`--use-graph`, `--use-hierarchy`, and `--use-cbf`. Neural policy training and
inference can use CUDA; the small OSQP CBF-QP intentionally remains CPU-side.

Training with the default BC initialization requires both validated BC
checkpoints explicitly:

```powershell
python scripts/train.py --device cuda --stage low `
  --bc-low-checkpoint data/rl/behavior_cloning/bc_low_level.pt `
  --bc-high-checkpoint data/rl/behavior_cloning/bc_high_level.pt
```

For a short integration smoke run (the goal-directed controller is labelled as
a semantic smoke controller; it is not a learned-policy result):

```powershell
python scripts/evaluate.py --device cuda --seed 20260719 `
  --episodes-per-seed 1 --max-steps 20 --experiment-name phase14_smoke
```

Each invocation creates `outputs/<experiment_name>/` with `config.yaml`,
`environment.json`, `checkpoints/`, `tensorboard/`, `raw_results/`,
`summary.csv`, `summary.json`, `figures/`, `logs/`, and `README.md`. Raw JSONL
records preserve every requested seed and failed episode. Aggregate summaries
use sample standard deviation plus two-sided 95% Student-t confidence
intervals; no outlier deletion is permitted.

The required scale and method matrix is in
`configs/experiments/dynamic_world_mainline.yaml`. It covers 3, 5, 8, 12, and
16 UAVs. The unified dynamic-world executor currently implements only the
checkpointed `full_method`; prior baseline scripts are registered as unavailable
until they receive comparable dynamic-world adapters. A*, RRT*, and ORCA are
also explicitly unavailable, so no method can silently receive invented results.

Reported finite metrics include success/collision/terrain/threat rates, path
length, mission time, mean and 5th-percentile separation, temporal conflicts,
energy proxy, decision latency, and CBF interventions/corrections/emergencies.
`expert_gap` and `generalization_gap` are recorded as unavailable in
`environment.json` until a matched CA-HGALO reference rollout and paired
within/out-of-distribution evaluation are supplied; they must not be fabricated.

# Phase-14 training

The only training line is BC-initialized hierarchical predictive-graph MAPPO.
The low and high BC checkpoints must be supplied explicitly; a missing
checkpoint is an error, never an implicit random initialization labelled as BC.

```powershell
python scripts/train.py --config configs/experiments/mainline_smoke.yaml --device cuda --stage low `
  --bc-low-checkpoint data/rl/behavior_cloning/bc_low_level.pt `
  --bc-high-checkpoint data/rl/behavior_cloning/bc_high_level.pt
```

Use `--no-use-expert-pretrain` only for the explicit no-BC ablation. PyTorch
uses CUDA when selected; CBF execution uses CPU OSQP by design.

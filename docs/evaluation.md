# Phase-14 evaluation

Use a compatible staged hierarchy checkpoint to run deterministic high-level
and low-level graph-policy inference in the dynamic obstacle and delayed/lossy
communication world:

```powershell
python scripts/evaluate.py --config configs/experiments/mainline_smoke.yaml `
  --checkpoint data/rl/hierarchical_mappo/hierarchical_low_final.pt `
  --device cuda --seed 20260719 --episodes-per-seed 4 --experiment-name phase14_eval
```

The evaluator loads the checkpoint, builds actor graphs from delivered packets,
filters normalized actions through CBF, preserves every raw episode, and writes
a top-view PNG for each episode. No-checkpoint execution is labelled a semantic
smoke controller and is not a learned-policy result.

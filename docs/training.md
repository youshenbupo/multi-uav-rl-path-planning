# Core training protocol

## Preflight

Training is serial. Before every launch, check `git status`, active Python
command lines, and `nvidia-smi`; verify that the new output directory does not
exist. Use the dedicated CUDA interpreter and leave OSQP on CPU. Never change
CBF slack, tolerances, iteration limit, solve-time limit, or margin settings to
make a run finish.

The five fixed independent seeds are `20260719`--`20260723`. Each run requests
100,000 transitions and finishes at the next complete rollout boundary.

## Command templates

MLP MAPPO, 3 UAV:

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' scripts/train_mappo.py `
  --config configs/rl/dynamic_mappo_baseline.yaml `
  --output-dir outputs/<new_unique_training_root> `
  --device cuda --seed <seed> --num-uavs 3 --total-steps 100000
```

GraphMAPPO, 3 UAV:

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' scripts/train_graph_mappo.py `
  --config configs/rl/dynamic_graph_baseline.yaml `
  --output-dir outputs/<new_unique_training_root> `
  --device cuda --seed <seed> --num-uavs 3 `
  --graph-mode <mappo|predictive_graph|uncertainty_predictive_graph> `
  --total-steps 100000
```

Use `mappo` for the historically named raw GraphMAPPO arm. It is a self-loop
graph with no neighbour edges; do not substitute `distance_graph` or describe
it as a raw neighbour-truth graph.

Full 5/8-UAV method:

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' scripts/train_graph_mappo.py `
  --config configs/rl/dynamic_graph_<5uav|8uav>.yaml `
  --output-dir outputs/<new_unique_training_root> `
  --device cuda --seed <seed> --num-uavs <5|8> `
  --graph-mode uncertainty_predictive_graph --total-steps 100000
```

Independent 5/8-UAV joint no-uncertainty ablation:

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' scripts/train_graph_mappo.py `
  --config configs/rl/dynamic_graph_<5uav|8uav>_predictive_no_uncertainty_ablation.yaml `
  --output-dir outputs/<new_unique_training_root> `
  --device cuda --seed <seed> --num-uavs <5|8> `
  --graph-mode predictive_graph --total-steps 100000
```

Do not initialize an ablation from full-method weights. Do not reuse a seed's
output root.

## Expected transition totals

| Arm | Environments x UAVs x rollout | Natural retained total |
| --- | --- | --- |
| 3-UAV arms | `1 x 3 x 32` | 100,032 |
| 5-UAV full | `2 x 5 x 32` | 100,160 |
| 5-UAV no-uncertainty | `2 x 5 x 24` | 100,080 |
| 8-UAV full and no-uncertainty | `2 x 8 x 24` | 100,224 |

## Completion decision

A run is eligible only if the process exits naturally, the expected final
transition checkpoint and summary exist, identity metadata match the requested
seed/config/mode/UAV count, telemetry is readable, and no `ABORTED.json` marks
the root. Document the result before launching evaluation. Preserve every
failed attempt and record why it is excluded.

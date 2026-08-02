# Core checkpoint evaluation and CBF replay

## Fixed scenario matrix

Evaluate only a naturally completed final checkpoint. For every independently
trained seed, run 20 deterministic episodes in each condition:

| Scenario | Communication | Dynamic obstacle |
| --- | --- | --- |
| `nominal` | delay 0, loss 0 | no |
| `delay_only` | delay at least 1 step, loss 0 | no |
| `loss_only` | delay 0, loss at least 0.1 | no |
| `dynamic_only` | delay 0, loss 0 | yes |
| `combined` | base delay/loss | yes |
| `ood_communication_obstacle` | delay 3, loss 0.3, staleness 5, uncertainty growth 1.5 | faster dynamic obstacle |

For the current base protocol, OOD increases delay by two steps, loss by 0.2,
maximum staleness by two, uncertainty growth by 1.5x, and obstacle speed by
1.5x. The fixed simulation uses `dt=1`, maximum 20 steps, horizontal speed
8, vertical speed 6, and one moving cylinder in dynamic conditions.

## Evaluation command

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' scripts/evaluate_core_checkpoint.py `
  --family <mappo|graph_mappo> `
  --config <training_config.yaml> `
  --checkpoint <final_checkpoint.pt> `
  --output-dir outputs/<new_unique_evaluation_root> `
  --experiment-name <stable_cell_name> `
  --seed <trained_seed> --num-uavs <3|5|8> `
  --episodes 20 --scenario <scenario> --device cuda
```

Do not pass `--without-cbf` for the core matrix. Use a new output root, retain
all JSONL, and verify exactly 20 unique episode indices plus checkpoint, seed,
scenario, graph-mode, and CBF identity after every cell.

## CBF fallback replay

Replay all training and evaluation telemetry sources for a completed arm. Pass
the exact CBF settings from that source; for the full method this includes
uncertainty margin gain `0.5` and cap `5`, while the independent 5/8-UAV joint
ablation uses zero for both.

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' scripts/replay_cbf_fallbacks.py `
  <telemetry_source_1> <telemetry_source_2> `
  --output-jsonl outputs/cbf_diagnostics/<new_unique_replay>.jsonl `
  --max-iterations 20000 --max-solve-time-seconds 0.1 `
  --slack-penalty 100 `
  --uncertainty-margin-gain <0.5|0.0> `
  --max-uncertainty-margin <5.0|0.0>
```

Preserve raw sources and full event context. A zero-event outcome still needs
a retained zero-event diagnostic. Do not infer that replay-count differences
caused by solver timing/order imply better or worse safety.

## Descriptive summaries

Use `scripts/summarize_multiarm_checkpoint_evaluations.py` for the 3-UAV four
arm matrix and `scripts/summarize_paired_checkpoint_evaluations.py` for the
5/8-UAV paired matrices. The inputs and exact validated output hashes are in
`docs/post_actor_isolation_descriptive_validation.md`.

The unit of analysis is the trained seed. First compute the mean of each metric
over 20 episodes within a seed-scenario cell; then compute the mean and sample
standard deviation over five seed-cell means. For paired scale comparisons,
use full minus no-uncertainty for the same seed. Keep all six scenarios, all
task metrics, and all CBF diagnostic metrics. Keep the task and CBF sections
separate.

Do not add episode-level tests, p-values, confidence intervals, rankings,
superiority labels, or selected favorable endpoints to the current descriptive
artifacts. An inferential analysis requires a separately reviewed plan before
the output exists.

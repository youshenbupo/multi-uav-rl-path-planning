# Expert Episode HDF5 Schema

Schema version: `multiuav_expert_episode_v1`.

Each episode is an independent group at `/episodes/<episode_id>`. Group
attributes identify the scenario, seed, temporal resolution, nominal speed,
action speed cap, source, and layout conventions.

## Fixed-shape time series

All time-series arrays share the same time index `T` and UAV index `N`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `positions` | `[T, N, 3]` | Repaired path positions at global timestamps |
| `velocities` | `[T, N, 3]` | Finite-difference desired velocities |
| `timestamps` | `[T]` | Strictly increasing seconds |
| `active_mask` | `[T, N]` | UAV is flying during this sample |
| `low_level_actions` | `[T, N, 3]` | Clipped desired velocity commands |
| `low_level_action_mask` | `[T, N]` | A next active position exists |
| `low_level_clip_ratio` | `[T, N]` | Applied speed ratio in `[0, 1]` |
| `high_level_actions` | `[T, N]` | Integer label, or `-1` when unknown |
| `high_level_action_mask` | `[T, N]` | Whether the high-level label is authoritative |
| `convergence_history` | `[I]`, optional | HGALO global-elite cost at each optimizer iteration |

The low-level command is `v_t = (p_(t+1)-p_t)/dt`, clipped to the stored
`max_speed`. The clipping ratio is stored rather than silently losing the
original command magnitude.

## Variable-size data

`raw_waypoints` and `repaired_waypoints` use `values: [W, 3]` plus
`offsets: [N+1]`; path `u` is `values[offsets[u]:offsets[u+1]]`.

Dynamic graph edges are deliberately not padded. Each
`dynamic_graph/steps/<time_index>` group stores:

| Field | Shape |
| --- | --- |
| `node_features` | `[N, 7]` (`position`, `velocity`, `active`) |
| `edge_index` | `[2, E_t]` |
| `edge_features` | `[E_t, 5]` (`delta`, distance, shortfall) |
| `edge_mask` | `[E_t]` |
| `conflict_labels` | `[E_t]` |

This one-group-per-step convention keeps `E_t` correct for every time step.

## High-level labels and provenance

Labels are `continue`, `wait_yield`, `climb`, `descend`, `shift_left`,
`shift_right`, `local_subgoal`, `assigned_delay`, and `conflict_priority`.
The name-to-integer mapping is stored in the episode attributes.

The restored MATLAB reference exports final schedule and repair results, not
action-by-action histories. Its converted episodes therefore have an all-false
`high_level_action_mask`; no labels are invented. The conversion script's
`--coordinate-python` option instead records labels sourced from the Python
rule coordinator and marks the episode source accordingly. Future CA-HGALO
logs can use the same `schedule_log` and `spatial_repair_log` string datasets.

## Validation

`scripts/validate_expert_dataset.py` checks finite values, time monotonicity,
all declared shapes, action speed limits, start/goal consistency, graph step
layouts, high-level mask integrity, and exact replay of stored cost, strict
success, minimum separation, and temporal conflict count with the Python
evaluator. When `convergence_history` is present, it must be a finite,
nonempty one-dimensional array; MATLAB-only reference episodes may omit it.

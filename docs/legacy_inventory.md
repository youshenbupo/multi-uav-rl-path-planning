# MATLAB Legacy Inventory

Audit date: 2026-07-16  
Audited root: `D:\yolo\multiuav\legacy_hgalo`  
Read-only rule: no file in the MATLAB baseline was changed.

## Scope and source status

The recursive inventory contains **333 files**. By extension: 94 PNG, **84
MATLAB `.m` files** (81 in the active restored source tree and 3 historical
export scripts), 25 PDF, 15 Markdown, 13 JSON, 12 CSV, 12 MJS, 11 MAT, 10
LOG, 8 AUX, 8 TEX, 6 ZIP, 5 each HTML/PPTX/CLS/MP4, 4 DRAWIO, 3 GZ, 2 PID, 2
TXT, and one each PY/JPG/extensionless/`.DS_Store`.

`HGALO_恢复源码/` is the only active MATLAB source package. Its `README.md`
states that it was reconstructed from the paper and historical material and
contains PSO, GWO, ALO, improved ALO, HGALO, old baselines, and ablations.
It also states that a standalone **CA-HGALO** algorithm entry is **not
present**. Therefore this audit does not infer CA-HGALO semantics or claim
that its output is available for regression.

## Complete directory overview

| Directory or artifact family | Contents and role |
| --- | --- |
| `HGALO_恢复源码/` | Active reconstructed MATLAB project: 81 `.m` files, `RESTORE_MANIFEST.md`, entry scripts, packages, and retained results. |
| `HGALO_恢复源码/+algorithms/` | PSO, GWO, ALO, improved ALO, HGALO, initialization, update, repair wrappers, and ablations. |
| `HGALO_恢复源码/+config/` | Default parameters and the four scenario builders. |
| `HGALO_恢复源码/+core/` | Environment, path encoding/decoding, evaluation, feasibility repair, conflict graph, and scheduling. |
| `HGALO_恢复源码/+experiments/` | Reproducible benchmark execution, progress persistence, CSV/Markdown tables, and postprocessing. |
| `HGALO_恢复源码/+utils/` | Clamp and roulette-wheel utilities. |
| `HGALO_恢复源码/+viz/` | Plots, explanation previews, and MPEG-4 video exports. |
| `HGALO_恢复源码/results/` | Retained `.mat`, `.csv`, `.png` benchmark and ablation artifacts; these are historical outputs, not newly rerun evidence. |
| `论文与LaTeX/` | Paper PDFs, TeX sources, figures, and notes; reference material, not an executable baseline. |
| `专利/` | Patent manuscripts and figures; reference material. |
| `历史归档/` | Archived templates, source-like root scripts, generated assets, videos, slides, old paper material, and temporary browser artifacts; not used by active entrypoints. |
| Root `README.md` and `journal_stage2_writing_notes.md` | Baseline provenance and writing notes. |

The command used for the complete recursive listing was:

```powershell
Get-ChildItem -LiteralPath D:\yolo\multiuav\legacy_hgalo -Recurse -File -Force
```

The active MATLAB source-file inventory is:

```text
+algorithms/alo_baseline.m
+algorithms/alo_improved.m
+algorithms/alo_improved_ablation_no_chaos.m
+algorithms/alo_improved_ablation_no_elite.m
+algorithms/alo_improved_ablation_no_layer_prior.m
+algorithms/alo_improved_ablation_no_levy.m
+algorithms/alo_improved_ablation_no_repair.m
+algorithms/alo_improved_ablation_no_sync.m
+algorithms/alo_improved_ablation_no_temporal.m
+algorithms/alo_local_exploitation.m
+algorithms/chaotic_initialize.m
+algorithms/evaluate_candidate.m
+algorithms/gwo_baseline.m
+algorithms/gwo_guided_update.m
+algorithms/hgalo_ablation_no_alo.m
+algorithms/hgalo_ablation_no_chaos.m
+algorithms/hgalo_ablation_no_gwo.m
+algorithms/hgalo_ablation_no_levy.m
+algorithms/hgalo_ablation_no_repair.m
+algorithms/initialize_hgalo_population.m
+algorithms/levy_flight_step.m
+algorithms/pso_baseline.m
+algorithms/repair_candidate_solution.m
+algorithms/run_HGALO_planner.m
+config/defaultConfig.m
+config/makeScenario.m
+core/apply_corridor_prior.m
+core/build_conflict_graph.m
+core/build_demo_environment.m
+core/build_planning_problem.m
+core/classify_failure_mode.m
+core/classify_temporal_conflict_topology.m
+core/conflict_window_detour.m
+core/decode_path_for_uav.m
+core/effective_corridor_spacing.m
+core/evaluate_multi_uav_cost.m
+core/final_conflict_window_detour.m
+core/min_distance_to_cylinder.m
+core/refine_spatial_conflicted_threat_paths.m
+core/repair_path.m
+core/repair_spatial_conflicts.m
+core/repair_threat_segments.m
+core/repair_waypoint.m
+core/resample_path.m
+core/resolve_conflict_schedule.m
+core/single_uav_cost.m
+core/terrain_height.m
+core/turn_violation_records.m
+core/uav_lane_offset.m
+core/uav_target_clearance.m
+experiments/postprocess_benchmark.m
+experiments/run_benchmark_suite.m
+experiments/run_records_to_table.m
+experiments/run_single_solver.m
+experiments/summary_rows_to_table.m
+experiments/write_benchmark_csv.m
+experiments/write_benchmark_markdown.m
+utils/clamp_value.m
+utils/roulette_wheel_selection.m
+viz/draw_cylinder.m
+viz/draw_uav_icon_2d.m
+viz/draw_uav_icon_3d.m
+viz/export_final_flight_video.m
+viz/export_ppt_intro_video.m
+viz/export_search_video.m
+viz/generate_explanation_previews.m
+viz/normalize_video_frame.m
+viz/plot_benchmark_summary.m
+viz/plot_failure_breakdown.m
+viz/plot_framework_scene.m
+viz/plot_metric_boxplots.m
+viz/plot_paths_and_convergence.m
+viz/write_single_run_summary.m
run_ablation.m
run_ablation_formal.m
run_ablation_hgalo.m
run_ablation_hgalo_formal.m
run_all_overnight_experiments.m
run_benchmark_formal.m
run_benchmark_hgalo_formal.m
run_multiuav_alo_demo.m
```

## Main entrypoints and call graph

| Entrypoint | Intended run | Solvers |
| --- | --- | --- |
| `run_multiuav_alo_demo.m` | Seeded one-off demo (`rng(42,"twister")`) | `alo_improved` |
| `run_benchmark_formal.m` | S1--S4 formal baseline benchmark | PSO, GWO, baseline ALO, improved ALO |
| `run_benchmark_hgalo_formal.m` | S1--S4 formal HGALO benchmark | Baselines plus HGALO |
| `run_ablation.m` | S3 improved-ALO ablation | Improved ALO and feature removals |
| `run_ablation_hgalo.m` | S3 HGALO ablation | Full HGALO and feature removals |
| `run_all_overnight_experiments.m` | Runs the four formal scripts and logs outcomes | Delegates to the preceding four entrypoints |

`run_benchmark_formal.m` and `run_benchmark_hgalo_formal.m` construct
configuration and scenario builders, then call
`experiments.run_benchmark_suite`. That function builds `env` and `problem`,
derives a stable seed per scenario/algorithm/run, calls
`experiments.run_single_solver`, and aggregates metrics. A solver calls
`core.evaluate_multi_uav_cost`; evaluation decodes every UAV path, computes
single-UAV and pairwise costs, optionally schedules/repairs conflicts, builds
the conflict graph, and classifies feasibility. Visualizers and experiment
writers consume the returned `result` and `details` structs.

```text
entry script
  -> config.defaultConfig / config.makeScenario
  -> core.build_demo_environment -> core.build_planning_problem
  -> experiments.run_benchmark_suite -> experiments.run_single_solver
  -> algorithms.{pso,gwo,alo,hgalo}
  -> algorithms.evaluate_candidate -> core.evaluate_multi_uav_cost
  -> decode_path_for_uav -> repair_waypoint -> repair_path
  -> single_uav_cost + resample_path + build_conflict_graph
  -> resolve_conflict_schedule / repair_spatial_conflicts (feature gated)
  -> experiment writers and visualization modules
```

## Scenario definitions

`config.makeScenario.m` defines the complete S1--S4 set; terrain and
terminal locations are generated by `core.build_demo_environment.m`.

| ID | Scenario key | UAVs | Interior waypoints | Safe separation | Threat margin |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | `low_threat_3uav` | 3 | 5 | 60 | inherited default 25 |
| S2 | `base_5uav` | 5 | 6 | inherited default 70 | inherited default 25 |
| S3 | `high_threat_5uav` | 5 | 7 | 75 | 35 |
| S4 | `high_threat_8uav` | 8 | 7 | 80 | 35 |

Low/base scenarios define five cylinders; high-threat scenarios define seven.
Each cylinder row is `[center_x, center_y, radius, height]`.

## Core modules verified by reading source

| Concern | MATLAB source and observed responsibility |
| --- | --- |
| Terrain and interpolation | `build_demo_environment.m` generates a `90 x 90` `peaks` grid rescaled to world XY and height; `terrain_height.m` uses linear `interp2`, falling back to zero out of grid. |
| Start/goal and threats | `build_demo_environment.m` chooses the first N terminal seeds, moves XY seeds out of cylinder radii plus a buffer, and computes terrain-relative terminal Z. |
| Spherical path representation | `build_planning_problem.m` allocates length/azimuth/elevation triples; `decode_path_for_uav.m` integrates them with `cos`/`sin`, clamps, applies corridor priors, and repairs. |
| Resampling and length | `resample_path.m` uses cumulative arc length and piecewise-linear interpolation; `single_uav_cost.m` sums Euclidean segment lengths. |
| Clearance and turn constraints | `single_uav_cost.m`, `repair_waypoint.m`, `repair_path.m`, and `turn_violation_records.m`. |
| Cylindrical threats | `min_distance_to_cylinder.m` samples segment points and returns signed in-cylinder distance or exterior distance; `single_uav_cost.m` applies margin and hard penalties. |
| Spatial and time conflicts | `evaluate_multi_uav_cost.m`, `repair_spatial_conflicts.m`, `build_conflict_graph.m`, and `resolve_conflict_schedule.m`. |
| Local repair | `repair_threat_segments.m`, `conflict_window_detour.m`, `final_conflict_window_detour.m`, and spatial refiners. |
| Solvers | `pso_baseline.m`, `gwo_baseline.m`, `alo_baseline.m`, `alo_improved.m`, and `run_HGALO_planner.m`. |
| Success, statistics, plots | `classify_failure_mode.m`, `run_benchmark_suite.m`, `postprocess_benchmark.m`, `write_*`, and `+viz/`. |

## Units, array conventions, and indexing hazards

The code does not declare a formal unit system. The following conventions are
directly evidenced by `defaultConfig.m`, `build_planning_problem.m`, and the
cost/scheduling functions:

| Quantity | Observed unit/convention |
| --- | --- |
| X/Y/Z, path length, clearances, radii, separations, threat margins | Unlabelled common distance unit; world XY is 0--1000 and Z is 0--320. Preserve it as a single explicit Python distance unit; do not label it metres without source evidence. |
| Time and speed | `travelTime = pathLength / cruiseSpeed`; `cruiseSpeed=35`, delays/buffers are named seconds (`scheduleDelayStepSec`, `scheduleBufferSec`). |
| Angles | Internally **radian**: `atan2`, `sin`, `cos`, and bounds using `pi`; `maxTurnDeg=65` is converted using `deg2rad`; reporting uses `rad2deg`. |
| Cost | Scalar weighted sum of length, altitude, turn, threat, spatial, temporal, sync, and schedule terms. Weights are dimensionless configuration values. |
| Path | `N x 3` numeric row matrix `[x,y,z]`; decoded paths have `numWaypoints + 2` rows. |
| Candidate | `1 x totalDim`, where `totalDim = numUavs * numWaypoints * 3`; each UAV block interleaves `[step_length, azimuth, elevation]`. |
| Population | `popSize x totalDim`; velocity/personal-best arrays in PSO share this layout. |
| Terrain | `90 x 90` matrices `env.X`, `env.Y`, `env.Z`. |
| Threats | `numThreats x 4` matrix `[x,y,radius,height]`. |
| Terminals | `numUavs x 3` start/goal matrices. |
| Per-UAV paths | MATLAB cell array `numUavs x 1`; Python should use `list[numpy.ndarray]`. |
| Conflict graph | Dense `numUavs x numUavs` matrices plus `numUavs x 1` node vectors; segment-delay profile is `collisionSamples x numUavs`. |

MATLAB is **1-based** and uses inclusive range endpoints. Array accesses such as
`path(i,:)`, `paths{u}`, and `meta(u)` must be converted deliberately to
Python's 0-based indexing. The MATLAB code generally represents points as row
vectors and stacks them by rows; blindly preserving MATLAB vector broadcasting
in NumPy/PyTorch is a migration risk.

## Randomness and outputs

Top-level scripts seed MATLAB's Twister generator with 11, 21, 31, 41, or 42.
`run_benchmark_suite.m` then sets each run's deterministic seed from a named
FNV-like 32-bit hash of `scenario|algorithm` plus `104729 * runIdx`, and stores
the policy as `stable_named_hash_v1`. Random calls occur in all population
solvers and initialization helpers. A Python port must expose and record an
explicit seed for every run.

Primary outputs are `.mat` workspaces/progress checkpoints, CSV/XLSX summaries,
PNG/PDF plots, Markdown/text summaries, and optional MPEG-4 videos. Existing
files under `results/` were only inventoried; they were not regenerated or
treated as newly validated results.

## Duplicate, archival, and uncertain material

| Finding | Classification | Migration handling |
| --- | --- | --- |
| `run_ablation_formal.m` runs `run_ablation.m`; `run_ablation_hgalo_formal.m` runs `run_ablation_hgalo.m`. | Thin aliases, not independent algorithms. | Preserve only one canonical Python command per experiment family. |
| `run_ablation.m` and `run_ablation_hgalo.m` share layout but target different solver families. | Similar names, intentionally different contents. | Keep separate experiment configurations. |
| Baseline/ablation modules with `no_*` suffixes. | Deliberate feature-removal variants. | Defer until matching baseline behavior is validated. |
| `历史归档/` source-like scripts and generated material. | Historical/archive content; not reached by active source entrypoints. | Do not port unless explicitly selected as a provenance source. |
| CA-HGALO source, algorithm entrypoint, and CA-HGALO MATLAB regression artifacts. | Missing from active restored source. | Block CA-HGALO behavior claims and regression until supplied. |
| Formal unit labels and a written tensor/data schema. | Not found in scanned active MATLAB source. | Define explicit Python schema in a later approved stage and flag any assumptions. |

## Current limitations

MATLAB was not invoked in this audit. The inventory and mapping are source-level
evidence only; no retained MAT/CSV output was revalidated. The local `rg.exe`
tool could not start because Windows returned “Access is denied”, so recursive
PowerShell enumeration and `Select-String` were used instead.

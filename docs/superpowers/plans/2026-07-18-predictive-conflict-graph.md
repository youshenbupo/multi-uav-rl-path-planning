# Predictive Conflict Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add predictive dense conflict graphs, a masked graph-attention MAPPO encoder, and reproducible 3/5/8-UAV comparison experiments.

**Architecture:** A graph builder produces fixed-shape dense tensors from simulator state. Graph networks consume those tensors with masks, while a graph-specific PPO runner stores graph observations and future-pair auxiliary targets. Existing Phase-9 MAPPO remains a no-graph comparator.

**Tech Stack:** Python 3.11, PyTorch, PettingZoo Parallel API, NumPy, PyYAML, unittest, TensorBoard.

---

### Task 1: Predictive edge physics and graph construction

**Files:**
- Create: `multiuav/learning/conflict_graph.py`
- Create: `tests/test_predictive_conflict_graph.py`

- [ ] Write failing tests for CPA: opposing points at `[0,0,0]` and `[10,0,0]` with velocities `[1,0,0]` and `[-1,0,0]` must yield `t_cpa=5`, `d_cpa=0`; equal parallel velocities must yield finite `t_cpa=0`, `d_cpa=10`.
- [ ] Implement `GraphBuildConfig`, `ConflictGraph`, `compute_cpa_features`, and `ConflictGraphBuilder.build(positions, velocities, goals, active_mask)` with `[B,N,*]` validation and 15 ordered edge features.
- [ ] Add failing and passing tests for communication/risk/top-k edge union, self-loops, inactive removal, and variable `N`.

### Task 2: Dense masked graph-attention encoder

**Files:**
- Create: `multiuav/learning/graph_networks.py`
- Modify: `tests/test_predictive_conflict_graph.py`

- [ ] Write failing tests that a `[B,N,D]` node tensor and `[B,N,N,E]` edge tensor produce `[B,N,H]` embeddings, no-neighbour active nodes are finite, inactive outputs are zero, and a node permutation produces the corresponding embedding permutation.
- [ ] Implement edge-conditioned multi-head attention with score masking, forced valid self-loop, `softmax(dim=-1)`, residual projection, `LayerNorm`, feed-forward residual, and mask zeroing.
- [ ] Implement `GraphActor`, `GraphCentralizedCritic`, and an edge-level `ConflictPredictionHead`; test actions/log probabilities, centralized values, and head shapes.

### Task 3: Graph PPO data and auxiliary supervision

**Files:**
- Create: `multiuav/learning/graph_rollout_buffer.py`
- Create: `multiuav/learning/graph_mappo.py`
- Modify: `tests/test_predictive_conflict_graph.py`

- [ ] Write failing tests for `[T,E,N,*]` graph batch flattening and future-H target masks that exclude a reset boundary.
- [ ] Implement a graph rollout buffer storing node features, edge features, adjacency, active masks, positions, episode IDs, PPO terms, and returns.
- [ ] Implement future conflict/minimum-distance target construction, masked BCE/Smooth L1 losses, and `GraphMAPPOTrainer.update`; test finite PPO/auxiliary losses with coefficients both zero and nonzero plus checkpoint round-trip.

### Task 4: Variable-UAV graph training runner and CLIs

**Files:**
- Create: `multiuav/learning/graph_runner.py`
- Create: `configs/rl/graph_mappo.yaml`
- Create: `scripts/train_graph_mappo.py`
- Create: `scripts/evaluate_graph_mappo.py`
- Modify: `tests/test_predictive_conflict_graph.py`

- [ ] Write failing tests that deterministic scenario generation accepts `N=3,5,8`, preserves agent masks, and a short graph rollout/update/evaluation is finite.
- [ ] Implement deterministic `N`-UAV mission generation, graph-state extraction, GraphMAPPO rollout/evaluation, TensorBoard graph/auxiliary scalar logging, YAML validation, and checkpoint persistence.
- [ ] Implement CLI graph-mode choices `distance_graph` and `predictive_graph`; reject invalid `N` and unknown modes with explicit errors.

### Task 5: Required comparator experiment and documentation

**Files:**
- Create: `scripts/compare_graph_mappo.py`
- Create: `configs/experiments/graph_comparison.yaml`
- Modify: `README.md`
- Modify: `docs/progress.md`
- Modify: `docs/technical_decisions.md`
- Modify: `docs/known_issues.md`

- [ ] Implement a comparison CLI that invokes the no-graph MAPPO adapter, distance graph, and predictive graph for 3/5/8 UAVs, writes one JSON record per run, and aggregates actual metrics by `(mode,num_uavs)`.
- [ ] Run all nine combinations, preserve their summaries under ignored `data/rl/graph_comparison`, and document commands/results without claiming causal superiority from one seed.
- [ ] Run `python -m unittest discover -s tests -v`, `python -m ruff check .`, and `python -m mypy multiuav scripts`; update progress only with observed outcomes.

# 多无人机强化学习路径规划

这是一个 Python/PyTorch 研究实现，用于在三维地形、威胁区域、动态障碍物，以及延迟/丢包通信下训练多无人机协同路径规划策略。当前 AAMAS 2027 主线是：

> 通信陈旧度、时延和丢包 → 邻机状态预测不确定性 → 不确定性感知预测交互图与协同行动 → 不确定性自适应 CBF 安全裕度 → 动态障碍环境中的安全执行。

项目仍在研究阶段。训练、评估和 CBF 回放的原始产物被保留以便审计；它们不能在未完成匹配多种子统计前被解读为性能、显著性或安全性结论。

## 重要约束

- Actor 只能访问自身真值和**已送达**的邻机通信包；禁止邻机当前真值泄漏。Centralized Critic 可使用联合真值状态。
- 神经网络训练与推理使用 CUDA；执行侧 OSQP CBF-QP 使用 CPU。
- `outputs/`、原始 JSONL、检查点、日志和中止尝试均是研究审计证据，**不要删除或覆盖**。
- 所有新训练必须使用新的输出目录；同一路径重复训练会被脚本拒绝或破坏可追溯性。
- 当前默认论文主线不包含模仿学习、层级策略或替代控制器。相关历史代码保留，但不应作为 AAMAS 主线实验启动。

## 快速开始

### 1. 创建环境

在 Windows PowerShell 中，从本目录执行：

```powershell
conda env create --file environment.yml
conda activate multiuav_rl
python scripts/check_environment.py
```

环境名为 `multiuav_rl`。`environment.yml` 锁定 Python 3.11、PyTorch CUDA 13.0 wheel、Gymnasium、PettingZoo、CVXPY/OSQP、Ruff 与 mypy。确认 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

在本项目当前协议中，CUDA 不可用时不应把计划中的主线神经网络训练降级为 CPU。

### 2. 基础质量检查

```powershell
python -m pytest -q tests\test_predictive_conflict_graph.py tests\test_dynamic_graph_baselines.py tests\test_multi_uav_environment.py tests\test_communication.py
ruff check multiuav scripts tests
mypy multiuav scripts
```

完整 `pytest` 历史上有一个独立的 legacy-inventory 失败（MATLAB 文件清单预期 81、当前观察到 83）；不要通过修改清单或测试来掩盖它。见 `docs/known_issues.md`。

## 当前主线的运行方式

所有示例均以 `D:\anaconda3\envs\multiuav_rl\python.exe` 明确使用专用环境。PowerShell 中可先设置：

```powershell
$py = 'D:\anaconda3\envs\multiuav_rl\python.exe'
```

### MLP MAPPO 对照组

下面命令仅是一个独立训练单元。完整实验需为五个匹配 seed 建立互不覆盖的输出根，并完成六个场景、每格 20 episode 的评估。

```powershell
& $py scripts\train_mappo.py `
  --config configs\rl\dynamic_mappo_baseline.yaml `
  --device cuda --seed 20260719 --num-uavs 3 --total-steps 100000 `
  --output-dir outputs\example_mlp_seed_20260719
```

训练通常因 32-step rollout 边界在 100,032 transitions 结束。不要把该边界行为误记为超额训练。

### GraphMAPPO 三个匹配方法

三个方法使用相同的动态世界配置，区别仅在 `--graph-mode`：

| 方法 | `--graph-mode` | 解释 |
| --- | --- | --- |
| Raw graph MAPPO | `mappo` | 不使用预测知识或不确定性图特征的图方法对照。 |
| Predictive graph | `predictive_graph` | 使用从已送达包推演出的邻机位置。 |
| Uncertainty-aware predictive graph | `uncertainty_predictive_graph` | 在预测边及 CBF 裕度中显式使用通信不确定性。 |

例如，启动不确定性感知主方法的一个新种子：

```powershell
& $py scripts\train_graph_mappo.py `
  --config configs\rl\dynamic_graph_baseline.yaml `
  --device cuda --seed 20260723 --num-uavs 3 `
  --graph-mode uncertainty_predictive_graph --total-steps 100000 `
  --output-dir outputs\example_uncertainty_graph_seed_20260723
```

不要复用 `outputs/core_3uav_post_actor_isolation/` 内已有路径；它们是正在积累的审计证据。

### 统一六场景评估

当前规范场景是：`nominal`、`delay_only`、`loss_only`、`dynamic_only`、`combined`、`ood_communication_obstacle`。每个检查点、每个场景独立运行 20 episode，并保留 JSONL：

```powershell
& $py scripts\evaluate_core_checkpoint.py `
  --family graph_mappo `
  --config configs\rl\dynamic_graph_baseline.yaml `
  --checkpoint outputs\example_uncertainty_graph_seed_20260723\checkpoints\graph_mappo_final.pt `
  --output-dir outputs\example_evaluations `
  --experiment-name example_uncertainty_graph_seed_20260723_combined `
  --seed 20260723 --num-uavs 3 --episodes 20 `
  --scenario combined --device cuda
```

执行器把神经网络推理放在 CUDA，但 OSQP CBF 保持 CPU。不要传递 `--without-cbf` 作为主线结果。

### CBF 回退重放

回放只诊断保存的紧急回退上下文；不得据此偷偷放宽 slack、迭代上限、容差或求解时间。

```powershell
& $py scripts\replay_cbf_fallbacks.py `
  --output-jsonl outputs\cbf_diagnostics\example_replay.jsonl `
  --max-iterations 20000 --max-solve-time-seconds 0.1 `
  --slack-penalty 100 --uncertainty-margin-gain 0.5 `
  --max-uncertainty-margin 5.0 `
  outputs\example_uncertainty_graph_seed_20260723\live_training_telemetry.json
```

输出 JSONL 与同名 `.summary.json` 均应保留。若目标已存在，换一个新名称，不要覆盖。

## 代码与脚本入口

| 目的 | 入口 | 关键实现 |
| --- | --- | --- |
| 环境与 GPU 检查 | `scripts/check_environment.py` | `environment.yml` |
| 训练 MLP MAPPO | `scripts/train_mappo.py` | `multiuav/learning/mappo.py`, `multiuav/learning/networks.py` |
| 训练 GraphMAPPO | `scripts/train_graph_mappo.py` | `multiuav/learning/graph_runner.py`, `graph_mappo.py`, `graph_networks.py` |
| 统一核心评估 | `scripts/evaluate_core_checkpoint.py` | `multiuav/evaluation/`, `multiuav/envs/` |
| 旧式单方法评估 | `scripts/evaluate_mappo.py`, `scripts/evaluate_graph_mappo.py` | 仅用于对应历史路径；主线评估使用统一入口 |
| CBF 回退重放 | `scripts/replay_cbf_fallbacks.py` | `multiuav/safety/qp_filter.py` |
| 场景级 CBF 检查 | `scripts/test_cbf_scenarios.py` | `multiuav/safety/cbf_constraints.py` |
| 训练/评估计划清单 | `scripts/plan_core_training.py`, `scripts/plan_core_evaluation.py` | `multiuav/experiments/` |
| 结果汇总 | `scripts/summarize_paired_checkpoint_evaluations.py`, `scripts/summarize_multiarm_checkpoint_evaluations.py` | 仅向完整、协议合格输入运行 |

核心模块：

- `multiuav/envs/multi_uav_env.py`：PettingZoo 并行环境、单积分器状态推进、通信调度、动态障碍状态与诊断信息。
- `multiuav/envs/communication.py`：确定性时延/丢包通信信道，以及 Actor 可见的 packet-based `AgentKnowledgeState`。
- `multiuav/envs/observations.py`：局部 Actor 观测和集中式 Critic 状态；局部观测不使用邻机当前真值。
- `multiuav/learning/conflict_graph.py`：CPA（closest point of approach）预测交互图及不确定性感知边特征。
- `multiuav/learning/graph_networks.py`：边条件注意力 Actor 和集中式 Critic。
- `multiuav/learning/graph_runner.py`：环境、图、PPO、动态障碍和 CBF 的训练编排。
- `multiuav/safety/cbf_constraints.py`、`multiuav/safety/qp_filter.py`：联合 OSQP CBF-QP、安全约束与紧急回退。

## 配置、输出与可复现性

- 主要配置：`configs/rl/dynamic_mappo_baseline.yaml` 和 `configs/rl/dynamic_graph_baseline.yaml`。
- 当前动态协议关键值：1-step delay、0.1 drop probability、最大陈旧度 3 step、每 step 不确定性增长 1.0、动态障碍启用、CBF penalty 100、最多 20,000 iterations、0.1 s solve limit、CBF 不确定性 gain 0.5/cap 5.0。
- 每次训练保留：`checkpoints/`、`summary.json`、`live_training_telemetry.json`、TensorBoard 和 launcher stdout/stderr 日志。
- 每个核心评估格保留：`raw_results/<seed>.jsonl`、`summary.json`、`runtime_telemetry.json`。
- 已中止/无效目录必须保留 `ABORTED.json`，并在统计、汇总和论文中明确排除。

详细的数学建模与当前证据状态在 [docs/aamas2027_method_and_readiness.md](docs/aamas2027_method_and_readiness.md)。继续实验前读取 [docs/next_agent_prompt.md](docs/next_agent_prompt.md) 和 [docs/next_conversation_handoff.md](docs/next_conversation_handoff.md)。

## Git 工作流

远程为 `origin`：

```powershell
git status --short
git push origin codex/phase14-dynamic-world
```

`.gitignore` 有用户自己的未提交改动：不要暂存、提交、还原或格式化该文件。

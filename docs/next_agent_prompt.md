# 给下一位智能体的继续提示词

将下面的文本完整复制到新对话。新智能体必须以仓库文件为事实来源，而不是依据聊天摘要猜测状态。

```text
继续推进 D:\yolo\multiuav\reinforcement_learning 的 AAMAS 2027 投稿就绪目标。

开始前必须完整阅读：
1. D:\yolo\multiuav\reinforcement_learning\docs\next_conversation_handoff.md
2. D:\yolo\multiuav\reinforcement_learning\docs\aamas2027_method_and_readiness.md
3. D:\yolo\multiuav\reinforcement_learning\docs\known_issues.md
4. D:\yolo\multiuav\reinforcement_learning\docs\progress.md

唯一论文主线：
通信陈旧度、时延和丢包
→ 邻机状态预测不确定性
→ 不确定性感知预测交互图与协同行动
→ 不确定性自适应 CBF 安全裕度
→ 动态障碍环境下的安全执行结果。

不可违反：
1. 不新开模仿学习、层级策略或替代控制器平行主线。
2. Actor 只能用自身真值和已送达通信包；Critic 可用集中真值；禁止邻机当前真值泄漏。
3. 神经网络训练/推理必须 CUDA；OSQP CBF 必须 CPU。
4. 不把短训练、单种子、烟雾测试或回放状态写成性能/安全/显著性结论。
5. 不伪造基线；不得以降低回退率为目的修改 CBF slack、迭代上限、容差或 solve-time。
6. 保留所有中止训练、无效评估、原始 JSONL、检查点、日志、回退事件；统计明确排除无效尝试。
7. 未经用户明确授权，不向任何外部投稿系统提交。
8. .gitignore 有用户自己的未提交改动：绝不暂存、提交、还原、格式化或覆盖它。
9. 每完成阶段任务，先更新 docs/progress.md、docs/known_issues.md 和 docs/next_conversation_handoff.md，再汇报；代码/协议改动必须跑适当 pytest、Ruff、mypy，并提交且推送 origin/codex/phase14-dynamic-world。

先做只读核查：git status、当前分支/HEAD、Get-Process python、最后 handoff 记录。不得假定上一轮状态仍然正确。

当前停点（先复核后行动）：
- MLP 五个有效 post-isolation 3-UAV 训练及 30 个六场景评估格（600 JSONL）已完成；全源 CBF 回放也已完成，含一个已记录的 OOD 动态障碍不可重放错误。
- raw GraphMAPPO (graph_mode=mappo) 已有四个有效种子：20260719_launcherretry1、20260720、20260721、20260722；初次 20260719 根无效且有 ABORTED.json，必须排除。
- 下一步是：确认无 Python 进程、目标和日志路径均不存在后，串行启动 raw GraphMAPPO 3-UAV seed 20260723，使用 configs/rl/dynamic_graph_baseline.yaml、--device cuda、--graph-mode mappo、--total-steps 100000 和唯一输出根。使用 Start-Process + stdout/stderr 重定向，避免 PowerShell 把正常 CBF stderr 诊断中止为错误。
- 该 seed 自然结束后，核验 100032 transitions、1042 updates、CUDA、最终 checkpoint/summary/telemetry/log；重放 CBF 源事件（即使零事件也留 zero-event replay）；先写文档再启动下一阶段。
- raw Graph 五种子完备后，做每检查点六场景 × 20 episode 的统一 JSONL 评估、全源 CBF 回放；然后运行 predictive_graph 和 uncertainty_predictive_graph 的匹配五种子训练与评估；随后 5/8 UAV 与独立关键消融。

所有训练使用：D:\anaconda3\envs\multiuav_rl\python.exe。
Git 远程名是 origin，不是 github.com；推送命令：
git push origin codex/phase14-dynamic-world
```

## 新对话启动检查表

1. 读取上面四份文档；
2. `git status --short` 确认 `.gitignore` 不会被暂存；
3. `Get-Process python` 确认没有并行作业；
4. 查看目标输出根和 stdout/stderr 是否存在，存在则绝不覆盖；
5. 训练/评估结束后解析原始 JSONL、summary、telemetry，并将无效尝试写入 `ABORTED.json`；
6. 每次文档提交后执行 `git push origin codex/phase14-dynamic-world`。

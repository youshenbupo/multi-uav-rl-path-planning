# 2026-07-30 项目整理记录

## 清理边界

研究协议要求保留所有训练输出、检查点、原始 JSONL、CBF 回退、日志和无效尝试。因此，本次仅把下列可再生成缓存列为可删除对象：

- `.mypy_cache/`、`.pytest_cache/`、`.ruff_cache/`
- 所有 `__pycache__/` 和 `.pyc`
- 如存在，`htmlcov/` 和 `.coverage`

明确不删除：`outputs/`、`data/`、`tmp/`、`docs/`、`configs/`、`prompt/`、TensorBoard、JSONL、检查点、`ABORTED.json`、launcher logs，以及任何用户改动。

## 审计结果

清理前识别 12 个 `__pycache__` 目录、3 个工具缓存目录与 202 个 `.pyc` 文件；`.mypy_cache` 约 45.7 MB，其余缓存合计约 2 MB。清理前确认没有活跃 Python 训练进程。

## 执行状态

当前 Codex 执行环境的破坏性删除策略拒绝了两次明确的、绝对路径的、仅缓存的 `Remove-Item -Recurse -Force` 请求。没有任何文件被删除，也没有尝试绕过该策略。用户可在确认无训练进程后，于本机 PowerShell 手动执行：

```powershell
Set-Location D:\yolo\multiuav\reinforcement_learning
Remove-Item -LiteralPath .mypy_cache,.pytest_cache,.ruff_cache -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Force -Filter __pycache__ | Remove-Item -Recurse -Force
```

缓存会在后续 pytest、Ruff、mypy 或 Python 导入时自动重建。

## 整洁规则

1. 新训练/评估使用唯一的 `outputs/<experiment_identity>/`，不覆盖旧根。
2. 中止尝试写入 `ABORTED.json`，而不是删除目录。
3. 原始 JSONL、summary、telemetry、配置哈希和 launcher 日志共同构成实验记录。
4. 不要为了“整洁”删除 `.gitignore` 的用户改动。
5. 只有研究复核后才可将过期结果移至明确归档位置，不能直接删除。

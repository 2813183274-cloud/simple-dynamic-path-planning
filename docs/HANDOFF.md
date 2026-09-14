# 项目换机交接

更新：2026-09-14。仓库阶段以 [进度文档](PROGRESS.md)为准；总体目标见 [设计文档](DESIGN.md)。

## 当前状态

- 第零至第三轮已完成并保留封存结果。
- 第四轮 P4-01 动作/制动诊断、P4-02 协议和 P4-03 v2 实现与回归已完成。
- 当前机制只把速度执行动作从硬裁剪改为 tanh；转向仍裁剪，16 维输入、原奖励、网络、场景和 PPO 超参数不变。
- 正式训练尚未启动，新最终测试集尚未生成。下一步为 P4-04：clip / speed_tanh 各 seed 1/2/3 从零训练。

## 新电脑恢复

```powershell
git clone https://github.com/2813183274-cloud/simple-dynamic-path-planning.git
cd simple-dynamic-path-planning
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B scripts/test_round_four.py
```

如果已有仓库，应先确认没有本地改动，再执行 `git pull --ff-only`。不要复制旧 `.venv`；重新安装依赖。`runs/` 中的大模型和日志通常不由 Git 携带，正式训练 P4-04 不依赖旧权重，但复核 P4-01/第三轮模型时需要从原电脑迁移相应 `runs/round3-*` 目录并核对哈希。

## 新任务提示词

```text
请先完整阅读 AGENTS.md、docs/HANDOFF.md、docs/PROGRESS.md、
docs/ROUND_FOUR_PROTOCOL_V2.md 和 docs/ROUND_FOUR_IMPLEMENTATION.md。
继续 P4-04；先检查实现封存和训练就绪状态。未经我明确要求，不训练、
不生成最终测试集、不修改旧轮次封存结果。每次改动后创建 Git commit。
```

## 关键命令与边界

```powershell
# 回归测试：零优化器更新
.\.venv\Scripts\python.exe -B scripts/test_round_four.py

# 仅检查某次训练是否就绪，不训练
.\.venv\Scripts\python.exe -B scripts/train_round_four.py --arm clip --seed 1

# 下面会正式训练，仅在明确批准 P4-04 后使用
.\.venv\Scripts\python.exe -B scripts/train_round_four.py --arm clip --seed 1 --execute
```

P4-04 必须完成六个规定运行，不能只保留最佳 seed。模型冻结后才能按协议生成 seed 5901/5902 的新最终数据；不得使用最终结果调参。Git 提交不等于大文件已迁移，换机前应分别核对 GitHub 提交和所需 `runs/` 目录。

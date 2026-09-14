# 项目换机交接

更新：2026-09-15。仓库阶段以 [进度文档](PROGRESS.md)为准；总体目标见 [设计文档](DESIGN.md)。

## 当前状态

- 第零至第三轮已完成并保留封存结果。
- 第四轮 P4-01 至 P4-05 已完成并验收；六次训练和 2520 个最终回合均已保存。
- 当前机制只把速度执行动作从硬裁剪改为 tanh；转向仍裁剪，16 维输入、原奖励、网络、场景和 PPO 超参数不变。
- tanh 同分布成功率点估计比 clip 高 3.33 pp，但 95% 区间含零；挑战点估计低 1.11 pp，且没有高风险阈值减速。下一研究方向待决策，不应继续重复 P4 或追加训练。

## 新电脑恢复

```powershell
git clone https://github.com/2813183274-cloud/simple-dynamic-path-planning.git
cd simple-dynamic-path-planning
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B scripts/test_round_four.py
```

如果已有仓库，应先确认没有本地改动，再执行 `git pull --ff-only`。不要复制旧 `.venv`；重新安装依赖。P4 六个 best 已随结果保存；`runs/` 中的完整训练逐步日志不由 Git 携带，复核训练过程或 P4-01/第三轮旧模型时仍需从原电脑迁移相应目录并核对哈希。

## 新任务提示词

```text
请先完整阅读 AGENTS.md、docs/HANDOFF.md、docs/PROGRESS.md、
docs/ROUND_FOUR_PROTOCOL_V2.md 和 docs/ROUND_FOUR_IMPLEMENTATION.md。
第四轮已完成。先核验 docs/ROUND_FOUR_ACCEPTANCE.md 和
results/round4/final_experiment/verification.json，再讨论下一研究方向。
未经我明确要求，不启动新训练、不修改任何封存结果。每次改动后创建 Git commit。
```

## 关键命令与边界

```powershell
# 回归测试：零优化器更新
.\.venv\Scripts\python.exe -B scripts/test_round_four.py

# 核验已完成结果的产物哈希
.\.venv\Scripts\python.exe -B -c "import json,hashlib,pathlib; r=pathlib.Path('results/round4/final_experiment'); m=json.load(open(r/'artifact_manifest.json')); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in m.items()); print(len(m),'P4 artifacts verified')"
```

P4 已完成，不能再用最终结果调参并覆盖本轮。六个 best 已复制到 `results/round4/final_experiment/models/`，可按方法封存哈希核对；完整训练逐步日志仍在原电脑 `runs/round4-*`，换机若要研究训练过程，应额外迁移这些目录。Git 提交不等于被忽略的完整 `runs/` 已迁移。

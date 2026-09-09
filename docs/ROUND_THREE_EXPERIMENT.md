# P3-05：正式训练与统一对照

日期：2026-09-09。本文记录 P3-05 实验，不代表 P3-06 论文贡献判断或整轮验收已经完成。

## 1. 执行边界

- 严格沿用 [协议 v1](ROUND_THREE_PROTOCOL.md) 和 [P3-04 实现封存](ROUND_THREE_IMPLEMENTATION.md)：base / instant / predictive 各 seed 1、2、3，从零训练；没有续训、调参、丢弃 seed 或延长预算。
- 每次请求 200000 步，PPO 完整 rollout 后实际 200704 步；总计 1806336 步。训练代码和依赖保持封存版本。
- 共用 60 场景 validation_D，按成功率、碰撞率、成功路径效率、共同原始奖励的固定顺序选 best；训练奖励不跨组直接比较。
- 全部训练完成后先封存九个模型与原 DWA h3-c1，再生成 seed 4901 的 240 场景常规集和 seed 4902 的 120 场景挑战集。四类会遇分别严格等额 60 / 30；360 个场景全部通过几何排重与可行性见证回放。
- 模型重复测试同一场景不增加独立场景数量；在线训练未保存全量初始几何，因此不能声称已穷尽与训练流逐场景排重。
- 没有修改原 16 维输入、中心距离线性安全项、前进权重 10、PPO 参数、动力学或 DWA 参数。没有改变根目录默认模型入口。

## 2. 最佳模型检查点

这些步数仅由共同验证集决定，不由最终测试决定。末次更新后也已评估。

| 组 | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| base | 90000 | 80000 | 150000 |
| instant | 200000 | 150000 | 200000 |
| predictive | 190000 | 160000 | 100000 |

每个模型完整 SHA-256、训练实际步数和验证成功率见 [模型封存](../results/round3/final_experiment/method_seal.json)。本轮 best 早于最后一步并不违反协议，也不能仅据此判断过拟合。

## 3. 最终结果

20 个方法/数据集组合、3600 回合全部完成，逐步轨迹通过速度、加速度、转向、位置积分、目标匀速运动、净空和奖励分项一致性检查。

| 方法 | 常规成功率：3 seed 均值 ± 样本 SD | 挑战成功率：3 seed 均值 ± 样本 SD |
|---|---:|---:|
| base | 72.36% ± 6.54 个百分点 | 58.06% ± 6.25 个百分点 |
| instant | 77.78% ± 1.27 个百分点 | 63.89% ± 6.31 个百分点 |
| predictive | 74.03% ± 3.64 个百分点 | 58.89% ± 9.94 个百分点 |
| DWA，单一确定性方法 | 239/240，99.58% | 118/120，98.33% |

主比较 predictive−instant 为 **−3.75 个百分点，配对层级 bootstrap 95% 区间 [−8.89，0.84]**。不支持预测项优于即时项，也不能由此断言预测项在所有情况下一定更差。predictive−base 为 +1.67 [−4.58，7.08]，instant−base 为 +5.42 [−1.11，12.92] 个百分点，区间也都包含零。

挑战集 predictive−instant 为 −5.00 [−13.33，4.44] 个百分点。它是辅助比较，不代替常规集主比较。所有区间按协议固定 10000 次、seed 5821，训练 seed 配对、会遇类型内场景配对；仅三个训练 seed，且辅助比较未作多重比较校正。seed 均值的 t 区间另见 `seed_statistics.csv`，不要与配对差区间混用。

### 3.1 安全与效率

| 数据 | 方法 | 静态碰撞率 | 动态碰撞率 | 成功条件路径效率 | 成功条件到达时间 |
|---|---|---:|---:|---:|---:|
| 常规 | base | 18.47% | 9.17% | 0.9636 | 27.83 s |
| 常规 | instant | 14.72% | 7.50% | 0.9583 | 27.95 s |
| 常规 | predictive | 17.64% | 8.33% | 0.9588 | 27.93 s |
| 挑战 | base | 27.50% | 14.44% | 0.9724 | 27.45 s |
| 挑战 | instant | 19.44% | 16.67% | 0.9652 | 27.63 s |
| 挑战 | predictive | 23.89% | 17.22% | 0.9676 | 27.61 s |

上述为各 seed 指标的均值。PPO 无超时或越界，全部失败为碰撞；新增惩罚没有通过停车超时掩盖失败，但也没有解决安全问题。成功条件效率与时间的存活样本不同，不能直接当作配对效率增益。平均净空、最差净空、原始奖励和完整计数均保存在逐 seed 表及逐场景文件。

DWA 无碰撞或越界，三次失败均为追越类超时：常规 ID 225，挑战 ID 101、105；全部如实保留，不调参消除这些失败。有限场景零碰撞不代表普遍安全。

### 3.2 机制与分类结果

- 三组 PPO 中，所有满足首次高风险定义的回合都没有在其后出现单步降速至少 0.1 m/s 的响应；延迟因此是无响应/删失，而不是 0 秒。常规 exposed/no-response 合计分别为 base 309/309、instant 275/275、predictive 296/296；挑战分别为 201/201、180/180、206/206。这些回合数不能当作独立训练 seed。
- 初始 2 s 后满速比例几乎为 100%：预测组三个 seed 在常规集分别为 100%、100%、99.93%，挑战均为 100%。因而本轮**没有达到预期的提前减速机制**。不能仅因轨迹或成功率变化声称学会减速。
- 新增项的平均每回合累计值：常规 instant −3.14、predictive −17.61；挑战分别 −3.38、−22.36。预测项产生更多累计惩罚，但没有带来稳定成功率收益；不能从此直接推出应继续加权。高风险速度、单步速度变化、角速度分布，以及成功/失败分组的原奖励和新增项累计/触发步数见各方法 `mechanism.json`。
- 类型差异明显：挑战集中 base / instant / predictive 的右交叉成功率分别为 62.22% / 48.89% / 41.11%，而左交叉为 60.00% / 71.11% / 72.22%。局部类型改善不能替代总体证据，也不支持所有会遇都受益；完整四类结果和探索性配对区间保留在 `by_type.csv`、`paired_comparisons.json`。
- 本轮只对固定权重、时域和预算下的这套实现下结论；不证明所有预测奖励都无效，不把负结果自动归因于训练步数不足。下一步 P3-06 整理研究结论和论文素材，是否开启新机制实验需另行决定，不能用本轮最终测试继续调参后仍称其未见。

结果图：[三组 seed 与 DWA 对照](../results/round3/final_experiment/success_comparison.png)；完整表：[自动汇总](../results/round3/final_experiment/SUMMARY.md)。图中柱是 seed 均值、黑点是三个 seed；DWA 不是三 seed 平均。

## 4. 文件入口与复核命令

所有正式运行保存在 `runs/round3-{base,instant,predictive}-seed{1,2,3}/`，未覆盖旧轮次。

| 查看内容 | 相对项目根目录的路径 |
|---|---|
| 最佳模型 | `runs/round3-<arm>-seed<seed>/models/best/best_model.zip` |
| 最佳步数及选模指标 | 同一 run 下 `models/best/selection_metrics.json` |
| 全部验证检查点成绩 | 同一 run 下 `logs/validation/evaluations.json` |
| 随机训练回合奖励曲线 | 同一 run 下 `results/training_reward_curve.png` |
| 完整训练步骤与状态 | 同一 run 下 `logs/training_steps.csv.gz`、`metadata.json` |
| 集中保存的训练证据副本 | `results/round3/final_experiment/training_records/` |
| 模型 / 数据封存 | `results/round3/final_experiment/method_seal.json`、`dataset_seal.json` |
| 逐场景与逐步轨迹 | `results/round3/final_experiment/final/<split>/<method>/episodes.csv`、`steps.csv.gz` |
| 成功 / 失败分组机制统计 | 同一评估目录下 `mechanism.json` |
| 失败示例 | 同一评估目录下 `first_failures.png`，按 ID 顺序取最多三个，不挑好看案例 |

奖励曲线是各组自身 shaped reward 的随机训练回合统计，不是固定测试成功率或收敛证明。没有标准差阴影。跨组性能以共同任务指标和 base return 比较。

```powershell
# 仅开发场景冒烟检查，不训练、不生成最终场景
.\.venv\Scripts\python.exe -B scripts/round_three_experiment.py smoke

# 已完成文件通过哈希验证后跳过；不重新训练
.\.venv\Scripts\python.exe -B scripts/round_three_experiment.py evaluate

# 从已完成评估重新计算统计与图表，不重新训练
.\.venv\Scripts\python.exe -B scripts/round_three_experiment.py summarize
```

`seal` 已经执行，不应再次执行或覆盖已有封存。训练入口拒绝覆盖同名 run；不要为重复命令删除正式产物。正式模型和大型训练日志在忽略 Git 的 `runs/` 中，Git 提交不等于模型备份。

实现与来源核验、最终产物哈希复核及 498 个历史产物不变检查见 [P3-05 核验记录](../results/round3/p3_05_verification.json)。旧结果未覆盖；本轮未改动已有训练实现的封存清单。

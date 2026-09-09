# Experiment Log

本文件用实验记录代替 Python 文件名承载历史。详细协议、哈希和限制以各轮验收文档为准；不确定信息不推测。

## Legacy / 初始基线

- 环境：单 USV、两个静态圆障碍、一个竖直运动并反弹的动态圆障碍；16 维中心距离观测；原线性奖励。
- 模型：根目录 `models/best/best_model.zip`，记录的最佳步数 180000，在 `test_scenarios_30.json` 上为 23/30。
- 限制：旧训练身份不完整，且该 30 场景集参与过开发；只作为兼容 benchmark，不作为新的独立最终结果。
- 证据：`models/best/selection_metrics.json`、`results/benchmark_30/`。

## Round 0：可信基线与评估契约

- 修改：run-id 隔离、来源/模型/数据哈希、固定验证选模、末次更新评估、奖励曲线和确定性复核。
- 训练：seed 1，请求 200000 步，实际 200704；最佳检查点 170000。
- 验证：23/30 成功、7/30 碰撞；这是验证成绩，不是独立测试。
- 证据：[ROUND_ZERO_ACCEPTANCE.md](ROUND_ZERO_ACCEPTANCE.md)。

## Round 1：阶段 A–D

- A：固定障碍半径；B：16 维下统一船体相对速度；C：约束随机二维会遇；D：混合静态布局及配对诊断。
- 动态会遇覆盖 head-on、左右交叉和 overtaking，风险与布局分层；训练仍是随机流，验证固定。
- 每阶段初次实验只有 seed 1，不能作为多 seed 稳定性证据。
- 证据：[ROUND_ONE_ACCEPTANCE.md](ROUND_ONE_ACCEPTANCE.md)、`configs/round1/`、`results/round1/`。

## Round 2：论文对照与统计

- 方法：C/D 各补足三个训练 seed；加入服从相同执行约束的 DWA，并只在验证数据调参。
- 最终数据：240 个常规、120 个挑战场景，四类会遇分层并封存。
- 结果：C/D 常规均值 69.58%/67.92%，挑战 65.00%/59.72%；DWA 240/240、120/120。D−C 配对区间包含零。
- 结论：不支持 D 稳定优于 C，也不支持 PPO 优于 DWA；C/D 是阶段整体方案对照，并非严格单因素消融。
- 证据：[ROUND_TWO_ACCEPTANCE.md](ROUND_TWO_ACCEPTANCE.md)、`results/round2/SUMMARY.md`。

## Round 3：即时/预测风险奖励

- 对照：base、instant（H=0）、predictive（H=4 s），权重 2、净空缓冲 2 m；其余输入、网络、原奖励、场景与预算不变。
- 训练：每组三 seed、每次实际 200704 步，共九个模型；共同 60 场景验证集独立选 best。
- 最终数据：新的 240 常规 + 120 挑战场景；模型先封存后生成，10 个方法共 3600 回合。
- 结果：常规 base/instant/predictive 为 72.36%/77.78%/74.03%，挑战为 58.06%/63.89%/58.89%；DWA 为 99.58%/98.33%。
- 主比较：predictive−instant = −3.75 pp，95% 配对层级 bootstrap 区间 [−8.89, 0.84]。
- 结论：不支持预测项的稳定收益；预定高风险减速响应没有出现。保留负结果，不据此加训或调权。
- 证据：[ROUND_THREE_ACCEPTANCE.md](ROUND_THREE_ACCEPTANCE.md)、[论文素材](ROUND_THREE_PAPER_MATERIALS.md)。

## 当前状态

第三轮已验收。P4-01 动作通道诊断仅为待确认候选，尚未实施。历史轮次脚本保留在原路径是为了满足模型与结果哈希复现，不应再通过复制它们创建 `v2/final` 版本。

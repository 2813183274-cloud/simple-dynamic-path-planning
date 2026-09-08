# P3-04 验收：预测风险奖励实现与测试

日期：2026-09-09。状态：**P3-04 已完成，P3-05 正式训练尚未启动。**

已实现 [协议 v1](ROUND_THREE_PROTOCOL.md) 中三组奖励、专用训练和开发评估入口，完成测试及实现封存。没有修改旧环境、旧训练脚本、既有 PPO 参数或旧封存结果；未生成新最终场景。验收证明实现符合约定，不证明预测奖励有效。

## 1. 实现入口

| 文件 | 职责 |
|---|---|
| `envs/risk_reward.py` | 纯观测风险计算与可选奖励包装层：base / instant / predictive |
| `round_three_utils.py` | 新增源码清单、协议/实现检查与第三轮模型身份认证 |
| `scripts/train_round_three.py` | 固定预算从零训练；共同 base reward 选模、分项日志、快照与失败记录 |
| `scripts/evaluate_round_three.py` | 共同评价函数与逐步动作/风险记录；CLI 仅允许 D 阶段验证或诊断数据 |
| `scripts/test_round_three.py` | 边界、环境等价、选模、来源认证及无优化器更新的集成检查 |
| `scripts/freeze_round_three.py` | 运行检查，记录 PPO 实际默认参数并创建不可覆盖的实现封存 |

原 `scripts/train.py` 与 legacy/A/B/C/D 行为保持原样；新奖励仅由第三轮专用入口显式启用。不能用旧训练入口声称已训练预测风险奖励。

### 奖励与日志

- 从同一 16 维观测计算 H=0 / 4 s 最近接近净空，不读取未来或见证路径。
- instant/predictive 权重均为 2、净空缓冲 2 m，额外项范围 [−2,0]；base 额外项为 0。
- 执行后计算新增项；任意终止或截断时额外项置 0，不改原终端奖励。
- 分别记录原奖励分量、`base_reward`、`risk_penalty`、`training_reward`、q_0/q_4、各实体净空/TCPA、实际速度与轨迹位置。
- Monitor 记录随机训练回合的原奖励、风险项和总奖励；曲线保留滚动均值，不新增标准差阴影。

### 训练与选模

仅接受三组和 seed 1/2/3，固定 CPU、请求 200,000 步、实际 200,704；拒绝覆盖和续训。源码、依赖或实际 PPO 参数与实现封存不符时拒绝启动。

共同 validation_D 的 60 场景按成功率、碰撞率、成功路径效率、平均 base reward 排序；shaped reward 不参与决胜，完全同分保留较早 best。训练结束继续评估最后一次参数更新。

快照覆盖新增奖励与入口。配置、模型、选模记录和训练日志均有哈希认证；不更新旧 `runs/latest_run.json`，避免旧评估默认入口误取新模型。

## 2. 已完成检查

- 静态迎面、动态接近/远离、零相对速度、H=0、边界、非法维度、NaN/越界与数值范围。
- 190 个固定动作转移中，三组与原环境的观测、状态、随机序列、原奖励、终止条件一致。
- 三组的成功、碰撞、超时额外项为 0，分项和正确；旋转/平移等价，风险函数不修改输入。
- shaped reward 改变不影响排序；base reward 改善可更新 best；完全相同时保留旧 best；结束回调记录最后参数版本。
- 源码清单或模型字节改变、未认证模型路径均被拒绝。
- 真实 PPO 仅初始化和开发回放，未训练；与既有 DWA 分别验证三组物理结果和共同回报一致。
- 临时目录中检查真实训练启动、快照与失败清理；学习被显式阻断。合成认证文件随临时目录删除，没有留下正式训练模型。
- 旧环境、实验流程及第一轮回归检查通过；既有第零/第一/第二轮与 P3-02 的 498 个文件哈希未变。该项只核对身份，不重新回放最终测试。

所有检查通过，正式 `runs/round3-*` 目录数量为 0，测试优化器更新次数为 0。没有第三轮训练成绩可报告。

## 3. 封存与使用

- [implementation_seal.json](../results/round3/implementation/implementation_seal.json)：新旧源码/协议、依赖与 PPO 实际配置。
- [test_report.json](../results/round3/implementation/test_report.json)：检查记录。

协议 JSON 的 `design_fixed_implementation_pending` 保留 P3-03 的历史状态，不回写原协议；当前实现状态以本验收和 implementation seal 为准。

就绪检查不会训练：

```powershell
.\.venv\Scripts\python.exe -B scripts\test_round_three.py
.\.venv\Scripts\python.exe -B scripts\train_round_three.py --arm predictive --seed 1
```

后续 P3-05 启动单个正式运行须显式添加 `--execute`。完整实验包含三组各三个 seed，不只训练预测组；已存在失败目录须先核查，不能静默删除并重试。

最终数据生成、模型封存和批量统计属于 P3-05 编排，应复用本轮评价函数，并另行记录新增编排/统计代码来源，不改本次奖励和选模语义。当前开发评估 CLI 不允许直接评估 test split。

实现封存不允许覆盖。后续如需修改清单中的源码或依赖，须说明原因并建立新实现版本，不能绕过哈希校验。

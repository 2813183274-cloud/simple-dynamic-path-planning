# USV RL Path Planning

一个用于硕士课题实验的二维无人艇（USV）动态路径规划项目。环境遵循 Gymnasium API，使用 Stable-Baselines3 PPO 学习连续线速度/角速度控制，并提供固定场景评估、DWA 对照、轨迹图和可追溯的分轮实验记录。

当前公开主线是阶段 D：两个固定半径静态圆障碍、一个二维匀速动态圆障碍、按会遇类型和风险分层随机生成训练场景。已完成实验保留原路径与哈希，根目录入口作为稳定门面，不改写历史算法。

## 快速开始

建议使用 Python 3.10 或 3.11。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 检查配置和实际命令，不训练
python train.py --config configs/main.json --run-id demo --dry-run

# 正式训练：默认阶段 D、seed 1、请求 200000 步
python train.py --config configs/main.json --run-id demo

# 用根目录兼容模型跑固定 30 场景，保存指标和每 3 场景一张图
python evaluate.py --config configs/main.json

# 指定认证 run
python evaluate.py --config configs/main.json --run-dir runs/demo --output-dir results/demo-benchmark

# 强制生成轨迹图，仍复用同一评估逻辑
python visualize.py --config configs/main.json --run-dir runs/demo --output-dir results/demo-plots
```

`train.py` 要求显式指定唯一 `--run-id`，防止覆盖模型。第一次默认评估输出到 `results/benchmark_30/`；再次运行需换目录或添加 `--overwrite`。

## 任务、运动学与障碍

地图为 100 m × 100 m。USV 需要从左侧起点到达右侧目标，同时避开两个静态障碍和一个动态障碍：

```text
x(t+1)     = x(t) + v(t+1) cos(theta(t)) dt
y(t+1)     = y(t) + v(t+1) sin(theta(t)) dt
theta(t+1) = theta(t) + omega(t+1) dt
```

二维动作位于 `[-1, 1]^2`，映射到目标线速度 `[0, 3] m/s` 和目标角速度 `[-π/4, π/4] rad/s`；实际速度还受线/角加速度限制。默认 `dt=0.2 s`，最多 600 步，到达阈值为 3 m。

阶段 D 每次训练 `reset()` 都会按 head-on、crossing-left、crossing-right、overtaking 及风险/静态布局约束重新采样；固定 seed 可复现随机序列。固定数据集通过 `reset(options={"scenario_index": i})` 精确加载。

## 16 维观测与 PPO

观测依次包含目标、两个静态障碍和动态障碍的归一化中心距离及相对方位 `sin/cos`（12 维），动态障碍相对速度（2 维），以及 USV 当前线/角速度（2 维）。阶段 D 的相对速度采用船体坐标和固定尺度；legacy 数据/模型保留旧语义，不能默认互换。

PPO 的策略和价值网络均为两层 64 单元 Tanh MLP。配置为学习率 `1e-4`、`n_steps=2048`、batch 64、10 epochs、`gamma=0.99`、`GAE λ=0.95`、clip 0.2、entropy 0.01。训练随机采样动作，评估使用确定性动作。

## 奖励函数

默认奖励在 [envs/dynamic_path_env.py](envs/dynamic_path_env.py) 的 `_calculate_reward` 中：

```text
10 × 距目标进展 − 0.05 时间代价
+ 静态中心距离 < 7 m 时的线性安全惩罚
+ 动态中心距离 < 8 m 时、权重 2 的线性安全惩罚
+ 单次终端奖励
```

成功 `+200`，静态碰撞 `−120`，动态碰撞 `−150`，越界 `−100`。碰撞使用半径之和，安全项使用中心距离。第三轮即时/预测风险项位于 [envs/risk_reward.py](envs/risk_reward.py)，属于已完成消融，不是默认奖励替换。

## 固定 30 场景 benchmark

[configs/test_scenarios_30.json](configs/test_scenarios_30.json) 是用户指定不可修改的固定 30 场景 benchmark，当前 SHA-256 为 `75bca781...fd2c757`。它曾参与历史开发/选模，适合回归和横向复测，但不能声称为未见最终测试。

- `configs/validation_scenarios_30.json`：legacy/default 训练选模。
- `configs/independent_test_scenarios_30.json`：legacy/default 独立 30 场景测试。
- `results/round2/`、`results/round3/final_experiment/`：后续论文实验封存的大样本最终数据与结果。

统一入口默认使用指定历史 benchmark，不重新生成场景。根目录兼容模型在其上为 23/30 成功、7 次静态碰撞；结果见 [results/benchmark_30](results/benchmark_30)。该模型的旧训练身份不完整，正式论文比较应使用 `runs/` 中认证模型。

## 指标与可视化

评估保存成功率、总/静态/动态碰撞率、越界率、超时率、平均奖励、步数、航行时间、路径长度、最终目标距离、最小静态/动态表面净空和成功条件路径效率。`nominal_initial_tcpa_dcpa.json` 给出初始名义 TCPA/DCPA：假设 USV 以最大速度直驶目标、动态障碍保持初始匀速；它不是策略实际轨迹的 CPA。

轨迹图包含起终点、静态/动态障碍、USV/动态障碍轨迹及最近动态会遇局部图。30 场景生成 10 张图，每张 3 行 × 2 列。只计算指标时使用 `--no-plots`。

## 项目结构

```text
├── train.py / evaluate.py / visualize.py   # 配置驱动公开入口
├── project_config.py                       # 配置、契约核对与旧核心转发
├── configs/main.json                       # 当前主线配置说明
├── envs/                                   # 唯一环境、场景与风险扩展
├── baselines/                              # DWA 对照
├── utils/metrics.py                        # 公开工作流附加指标
├── scripts/                                # 核心实现、测试和封存轮次工具
├── runs/                                   # 认证训练（Git 忽略）
├── models/                                 # legacy 兼容模型
├── results/                                # benchmark 与分轮正式结果
├── docs/                                   # 设计、开发、审计与实验历史
└── archives/                               # 早期散落产物归档
```

`scripts/round_*` 是已完成实验的复现入口，路径和内容被 SHA-256 封存；不要复制为新实验模板，也不要随意移动或合并。

## 配置与测试

[configs/main.json](configs/main.json) 展示环境、奖励、PPO、训练和评估参数。环境、奖励和 PPO 段是锁定契约，入口会与真实实现核对，不能只改 JSON 调参。训练预算、seed、设备和路径由入口读取，也可通过明确 CLI 参数覆盖。

```powershell
python scripts/test_public_workflow.py
python scripts/test_environment.py
python scripts/test_fixed_scenarios.py --scenario-file configs/test_scenarios_30.json
python scripts/test_dataset_splits.py
```

第一条包含 import/config、base 奖励逐步等价、三个固定场景和一次临时 PPO rollout；SB3 因 `n_steps=2048` 实际采样 2048 步，不保存模型。

## 文档导航

- [项目收口审计](docs/PROJECT_AUDIT.md)：职责、重复、KEEP/MERGE/ARCHIVE/DELETE 和算法风险。
- [实验日志](docs/EXPERIMENT_LOG.md)：各轮配置、结果与证据入口。
- [设计文档](docs/DESIGN.md)：总体目标和约束。
- [开发文档](docs/DEVELOPMENT.md)：详细复现与维护。
- [进度文档](docs/PROGRESS.md)：已完成轮次与待决策工作。

项目不覆盖真实船舶水动力、风浪流、传感器误差、COLREGs、多船交互或安全部署保证。仿真结果只能说明对应封存场景分布中的表现。

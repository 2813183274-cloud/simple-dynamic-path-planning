# Simple Dynamic Path Planning

一个结构精简、符合 Gymnasium API 的二维动态路径规划基线。智能体使用 Unicycle 运动学，在 100 m × 100 m 地图中避开两个固定圆形障碍物和一个上下反弹的动态圆形障碍物，并由 Stable-Baselines3 PPO 学习连续速度控制。

每次 `reset` 默认随机生成训练场景：起点位于左侧 `[5,15]×[35,65]`，目标位于右侧 `[85,95]×[35,65]`；两个静态障碍物分别位于中前段和中后段，动态障碍物从上方或下方竖直穿越。拒绝采样保证至少一个静态障碍物阻断起终点直线路径，并使动态障碍物按名义航速估算会进入航线的 8 m 安全走廊。所有随机数均来自 `self.np_random`，固定 seed 可复现。若需旧版固定地图，可构造 `DynamicPathPlanningEnv(randomize_scenario=False)`。

## 总体设计与状态转移

每个 `step` 依次执行：裁剪二维动作；映射为目标速度 `v_target∈[0,3]` 与 `ω_target∈[-π/4,π/4]`；以默认最大线加速度 `1.5 m/s²` 和最大角加速度 `π/4 rad/s²` 限制实际速度变化；按当前航向更新智能体位置并更新、归一化航向；更新动态障碍物并在考虑其半径的上下边界镜像反弹；累计实际位移；计算碰撞、越界、到达和超时；计算奖励；最后更新 `previous_goal_distance` 并返回观测。

在默认 `dt=0.2 s` 下，每步线速度最多变化 `0.3 m/s`，角速度最多变化 `π/20 rad/s`。构造环境时可通过 `linear_acceleration_max` 和 `angular_acceleration_max` 调整这两个正数限制。观测中的当前线速度和角速度是限幅后的实际执行速度，而不是 PPO 的目标速度。

到达、静态碰撞、动态碰撞和越界返回 `terminated=True`。仅达到 600 步且尚未发生终止事件时返回 `truncated=True`。若同一步存在多个事件，优先级为到达、静态障碍物 1、静态障碍物 2、动态障碍物、越界，保证终端奖励只加入一次。

## 固定的 16 维输入

输入始终为以下顺序，距离均除以地图对角线，方位均为目标物绝对方位减智能体航向后归一化的角度：

1. 目标距离、目标相对方位 `sin/cos`（0–2）
2. 静态障碍物 1 距离、相对方位 `sin/cos`（3–5）
3. 静态障碍物 2 距离、相对方位 `sin/cos`（6–8）
4. 动态障碍物距离、相对方位 `sin/cos`（9–11）
5. 世界坐标系相对速度 `(动态障碍物速度 - 智能体速度)/(3+1)`（12–13）
6. 当前归一化线速度与角速度（14–15）

静态障碍物不会按距离重排。返回前观测裁剪至 `[-1,1]` 并转换为 `float32`。

## PPO 网络

同一 16 维观测进入策略与价值分支。策略分支为 `64→64`，输出二维连续动作分布的均值，动作标准差由 PPO 学习；价值分支为 `64→64`，输出一个状态价值。二者均使用 Tanh。训练时策略可随机采样，评估严格使用 `deterministic=True`。两个动作分别控制线速度与角速度。

## 目录

```text
simple_dynamic_path_planning/
├── envs/
│   ├── __init__.py
│   ├── dynamic_path_env.py
│   └── scenario_dataset.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── test_environment.py
│   ├── test_fixed_scenarios.py
│   ├── generate_test_scenarios.py
│   └── manual_policy_test.py
├── configs/
├── models/
├── logs/
├── results/
├── requirements.txt
└── README.md
```

## 安装与运行

建议使用 Python 3.10 或 3.11。在项目根目录执行（Windows 与 Linux 命令相同）：

```bash
cd simple_dynamic_path_planning
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/test_environment.py
python scripts/manual_policy_test.py
python scripts/manual_policy_test.py --save results/manual_policy.png
python scripts/train.py --timesteps 500000 --seed 0
python scripts/generate_test_scenarios.py --num-scenarios 30 --seed 2026
python scripts/evaluate.py --scenario-file configs/test_scenarios_30.json
```

## 固定的 30 场景测试集

训练环境仍在每次 `reset()` 时随机生成场景。公平评估使用一次生成、永久复用的固定测试集；生成器以不读取障碍物的目标直达控制器确定风险和难度，不使用 PPO 结果筛选场景。

```bash
python scripts/generate_test_scenarios.py \
  --num-scenarios 30 \
  --seed 2026 \
  --output configs/test_scenarios_30.json

python scripts/test_fixed_scenarios.py \
  --scenario-file configs/test_scenarios_30.json

python scripts/evaluate.py \
  --model models/best/best_model.zip \
  --scenario-file configs/test_scenarios_30.json
```

测试集固定包含 5 个低动态风险、5 个静态主导、6 个向上交叉、6 个向下交叉、5 个混合风险和 3 个高风险场景。交叉场景的 easy/medium/hard 各占 2 个。JSON 保存场景哈希、生成配置和基线验证结果；固定测试模式通过 `reset(options={"scenario_index": i})` 精确加载，不添加随机扰动。

评估会保存：

```text
results/episode_results.csv
results/summary_metrics.json
results/metrics_by_scenario_type.csv
results/trajectory_sheets/scenarios_000_002_comparison.png
results/trajectory_sheets/scenarios_003_005_comparison.png
...（每3个场景一张，直到027-029）
```

为方便逐场景对照，每张图片包含 3 行测试结果：每行左侧是完整轨迹，右侧是同一场景的最近动态会遇局部图，即每张共 6 个子图。30 个场景总计生成 10 张对照图片。需要快速计算指标而不绘图时可向评估命令添加 `--no-plots`。

训练会生成定期检查点、`models/best/best_model.zip` 和 `models/final_model.zip`。评估默认优先加载最佳模型，不存在时加载最终模型，也可通过 `--model path/to/model.zip` 指定。

## 奖励

总奖励仅包含 `10 × 距离进展 - 0.05 + 两个静态安全惩罚 + 动态安全惩罚 + 单次终端奖励`。安全距离是中心距离；碰撞判定使用两物体半径之和。`info` 给出所有奖励分量、当前距离、整回合最小距离、实际路径长度和明确终止原因。

## 常见问题

- `ModuleNotFoundError`：确认已激活虚拟环境，并从本目录运行脚本后安装 `requirements.txt`。
- TensorBoard 或进度条依赖报错：确认 `tensorboard`、`tqdm`、`rich` 已安装。
- 找不到模型：先运行训练；或给评估脚本传入 `--model`。保存和默认加载路径已保持一致。
- 图形窗口不显示：在无桌面服务器上使用 `manual_policy_test.py --save results/manual_policy.png`。
- 训练效果差：先运行环境测试，检查观测顺序、坐标/角度、碰撞和奖励日志；一次只改变一个关键因素并记录前后指标。
- PPO 提示 rollout 被截断：这是固定 `n_steps=2048` 的正常采样行为，不等同于环境超时。

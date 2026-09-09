# USV 动态路径规划项目开发文档

## 1. 文档目的

本文说明如何搭建环境、运行训练和评估、定位产物、验证修改，以及如何避免模型和结果不兼容。本文描述的是当前代码可执行流程；尚未实现的工程改进会明确标注。

### 当前公开入口（项目收口后）

日常使用从根目录的三个稳定入口开始；它们读取 `configs/main.json`、核对锁定环境/PPO契约，再转发到现有核心实现，不复制训练或评估算法：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/main.json --run-id my-run
.\.venv\Scripts\python.exe evaluate.py --config configs/main.json
.\.venv\Scripts\python.exe visualize.py --config configs/main.json --model models/best/best_model.zip --output-dir results/my-plots
```

可先加 `--dry-run` 只看真实委托命令。默认评估使用用户指定的 `configs/test_scenarios_30.json` 历史 benchmark；该集合曾参与开发，入口明确允许 legacy split，但不把它称为未见最终测试。正式论文结果继续使用各轮封存数据。`scripts/train.py`、`scripts/evaluate.py` 是内部兼容核心；`scripts/round_*` 是已完成实验复现工具，不作为新实验入口模板。

主配置中的环境、奖励和 PPO 段是可见且可核对的锁定契约，不是任意调参覆盖。若数值与实现不同，入口拒绝运行。训练预算、seed、device 和路径可由配置或显式 CLI 参数选择。完整分类见 [项目收口审计](PROJECT_AUDIT.md)，实验沿革见 [实验日志](EXPERIMENT_LOG.md)。

### 第一轮入口（2026-09-07）

训练默认 `--stage D`，使用 `configs/round1/validation_D.json`，不是旧 30 场景验证集。A/B/C/D 的差异见 [第一轮验收记录](ROUND_ONE_ACCEPTANCE.md)。下文旧 `configs/validation_scenarios_30.json` 示例只适用于显式 `--stage legacy` 的历史实验。第一轮模型仍为 16 维，但 B/C/D 的速度编码与旧模型不同，不能交叉续训。

```powershell
.\.venv\Scripts\python.exe scripts\train.py --stage D --timesteps 200000 --seed 1 --run-id my-round1-D --device cpu
.\.venv\Scripts\python.exe scripts\evaluate.py --run-dir runs\my-round1-D --scenario-file configs\round1\validation_D.json --allow-non-test-split --output-dir results\my-round1-D-validation
.\.venv\Scripts\python.exe scripts\evaluate_counterfactual.py --run-dir runs\my-round1-D --scenario-file configs\round1\typical_cases.json --output-dir results\my-round1-D-paired
.\.venv\Scripts\python.exe scripts\test_round_one.py
```

上述均为验证/开发诊断，非最终测试。最终测试暂只有 `configs/round1/protocol.json` 中的封存协议。第一轮不要直接使用评估脚本的旧测试默认路径，它会因阶段不匹配而拒绝执行。

`--cross-stage-diagnostic` 仅用于显式共同开发协议比较：运动和几何来自数据集，观测编码来自模型训练阶段。此模式允许通过已保存源码快照核验的归档模型，不允许对 test split 使用，也不能据此跨阶段续训。场景源配置冻结后不应执行 `generate_round_one.py --rebuild-draft`；该选项仅用于冻结前开发，修改生成规则后应新建版本和运行目录。

## 2. 开发环境

建议使用 Python 3.10 或 3.11，并始终从项目根目录执行命令。

### 2.1 Windows PowerShell

```powershell
cd D:\PycharmProjects\simple-dynamic-path-planning
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后续示例直接调用虚拟环境解释器，不依赖终端是否已经激活环境。

### 2.2 Linux/macOS

```bash
cd /path/to/simple-dynamic-path-planning
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

当前核心依赖为 NumPy、Gymnasium、Stable-Baselines3、PyTorch、Matplotlib、Pandas、TensorBoard、tqdm 和 Rich。

## 3. 代码结构与修改边界

| 路径 | 作用 | 修改后的主要影响 |
|---|---|---|
| `envs/dynamic_path_env.py` | 环境动力学、观测、奖励、终止、随机场景 | 可能使所有旧模型失效 |
| `envs/scenario_dataset.py` | 固定场景生成和验证 | 可能改变基准测试语义或场景哈希 |
| `scripts/train.py` | PPO 参数、模型选优、训练曲线 | 改变训练过程和模型生命周期 |
| `scripts/evaluate.py` | 指标和轨迹图 | 可能只改变报告，不改变模型 |
| `scripts/test_environment.py` | 环境回归检查 | 应随环境契约同步更新 |
| `scripts/test_fixed_scenarios.py` | 固定数据集检查 | 应随场景 schema 同步更新 |
| `configs/validation_scenarios_30.json` | 固定验证集（seed 2026） | 只用于选模和调参 |
| `configs/independent_test_scenarios_30.json` | 独立测试集（seed 2027） | 只用于最终报告 |
| `configs/test_scenarios_30.json` | 遗留基准 | 已参与历史选模，不再视为独立测试 |
| `runs/<run_id>/` | 新训练的模型、日志、配置和结果 | 不同运行互相隔离 |
| `models/`、`logs/`、`results/` | 旧版遗留产物 | 可能缺少完整溯源信息 |

不要手工编辑固定场景 JSON 中的几何数据或哈希。需要新场景时应通过生成脚本创建，并执行固定场景检查。

## 4. 当前运行流程

### 4.1 环境检查

```powershell
.\.venv\Scripts\python.exe scripts\test_environment.py
```

该脚本检查：

- Gymnasium API 和 observation space；
- 16 维 `float32` 观测及有限值；
- 固定 seed 的可复现性；
- 随机场景几何范围；
- 速度/角速度加速度限幅；
- 静态和动态碰撞；
- 成功与超时语义；
- 当前线性安全奖励的数值。

### 4.2 固定场景检查

```powershell
.\.venv\Scripts\python.exe scripts\test_fixed_scenarios.py `
  --scenario-file configs\validation_scenarios_30.json

.\.venv\Scripts\python.exe scripts\test_dataset_splits.py
.\.venv\Scripts\python.exe scripts\test_experiment_workflow.py
```

它检查场景数量、类型配额、哈希、可通行性、运动方向、风险阈值，以及按索引重复加载的一致性。

### 4.3 手工策略冒烟测试

```powershell
.\.venv\Scripts\python.exe scripts\manual_policy_test.py `
  --seed 0 `
  --save results\manual_policy.png
```

该脚本使用朝向目标的简单控制器，不代表 PPO 性能，主要用于确认环境可以推进和渲染。

### 4.4 训练

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --timesteps 500000 `
  --seed 0 `
  --run-id ppo-seed0
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--timesteps` | `500000` | 请求的训练步数；PPO 可能按 rollout 长度略微超出 |
| `--seed` | `0` | Python/SB3/环境随机种子 |
| `--eval-freq` | `10000` | 固定场景验证间隔 |
| `--reward-smoothing-window` | `100` | 训练回合奖励滑动均值窗口 |
| `--validation-file` | `validation_scenarios_30.json` | 只允许 validation split |
| `--run-id` | 时间戳加 seed | 实验唯一目录名 |
| `--runs-dir` | `runs/` | 实验根目录 |
| `--resume` | 无 | 显式加载模型并续训 |
| `--device` | `auto` | 新训练与续训模型设备 |

不传 `--resume` 时创建新模型。续训必须显式指定已有运行和模型：

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --timesteps 200000 `
  --seed 0 `
  --run-id ppo-seed0 `
  --resume runs\ppo-seed0\models\final_model.zip
```

同一运行续训保留 Monitor 与验证历史、累计模型步数，并拒绝验证集哈希、源码契约、seed 或 PPO 配置不一致的组合。同一运行只接受自身最新认证的 `final_model.zip`；从认证的 best 模型分支继续训练时须使用新的 `--run-id`。旧目录模型、未认证检查点和缺少源码指纹的历史运行拒绝安全续训。恢复参数和优化器不意味着恢复完整随机数与环境状态，不保证与未中断训练逐位相同。

训练会写入：

```text
runs/<run_id>/config.json
runs/<run_id>/metadata.json
runs/<run_id>/models/checkpoints/ppo_dynamic_path_*_steps.zip
runs/<run_id>/models/best/best_model.zip
runs/<run_id>/models/best/selection_metrics.json
runs/<run_id>/models/final_model.zip
runs/<run_id>/logs/train_monitor.csv
runs/<run_id>/logs/validation/evaluations.json
runs/<run_id>/logs/tensorboard/PPO_*/...
runs/<run_id>/results/training_reward_curve.png
```

同一 `run_id` 不会被新训练静默覆盖。`metadata.json` 保存模型哈希、代码状态、依赖版本和观测/奖励 schema。

### 4.5 评估

显式指定模型，避免默认路径产生歧义：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --run-dir runs\ppo-seed0 `
  --scenario-file configs\independent_test_scenarios_30.json
```

不需要图片时：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --run-dir runs\ppo-seed0 `
  --scenario-file configs\independent_test_scenarios_30.json `
  --no-plots
```

`--run-dir` 优先加载该运行的最佳模型，不存在时加载最终模型，也可显式传 `--model`。默认只接受 `dataset_split=test`；诊断验证集时必须添加 `--allow-non-test-split`。非空输出目录不会被静默覆盖。

## 5. 如何读取训练和评估结果

### 5.1 最佳模型训练步数

查看：

```text
runs/<run_id>/models/best/selection_metrics.json
```

其中 `timesteps` 是累计模型步数；同时记录成功率、碰撞率、成功路径效率、平均奖励、验证集哈希和最佳模型哈希。

第零轮新增 `policy_updates`，记录 PPO 参数更新次数。训练结束会再次评估最终更新后的参数，因此同一 `timesteps` 可能有更新前后两条记录，不应按步数去重。

同一 `run_id` 安全续训时会恢复历史最佳标准，不会用续训初期模型覆盖已有更优模型。

### 5.2 固定验证曲线

查看：

```text
runs/<run_id>/logs/validation/evaluations.json
```

每条记录对应一次固定场景评估。`is_best=true` 表示它优于同一次脚本运行中此前的记录。

### 5.3 随机训练回合奖励

原始数据位于 `runs/<run_id>/logs/train_monitor.csv`，自动生成的图为 `runs/<run_id>/results/training_reward_curve.png`：

- 浅蓝线：每个随机训练回合奖励；
- 橙线：默认 100 回合滑动平均；
- 图中不包含标准差带。

随机回合难度不同，因此该曲线适合观察整体学习趋势，不适合单独判断最佳模型。

### 5.4 最终评估

- `summary_metrics.json`：整体成功、碰撞、Wilson 95% 区间、路径和净空；
- `evaluation_metadata.json`：模型哈希、数据集哈希、schema 和评估时间；
- `episode_results.csv`：逐场景结果和路径效率；
- `metrics_by_scenario_type.csv`：按场景类型比较；
- `trajectory_sheets/`：每 3 个场景一张图。

新评估结果可以独立核验来源；旧 `results/` 根目录下的遗留结果不具备这一保证。

## 6. 修改代码时的兼容性规则

### 6.1 必须重新训练的修改

以下任一修改都不应继续使用旧模型：

- 观测维度、顺序、归一化或坐标系；
- 动作含义或映射；
- 网络结构；
- 障碍物数量及其编码方式。

SB3 只能检查部分 shape 兼容性，无法识别“维度相同但语义已改变”的情况。

### 6.2 通常应重新训练并重新比较的修改

- 奖励项或权重；
- 碰撞、安全距离和终止规则；
- 随机场景生成分布；
- 动力学、时间步长或速度限制；
- PPO 超参数。

### 6.3 可以只重新评估的修改

- 指标汇总逻辑；
- CSV/JSON 字段；
- 轨迹图排版；
- 不改变环境返回值的日志增强。

即使只改评估，也应保留原结果，防止覆盖后无法对照。

## 7. 推荐开发流程

每次实验只改变一个主因素，按以下顺序执行：

1. 记录修改前模型、配置、seed 和结果；
2. 修改代码并同步更新测试；
3. 执行语法和环境检查；
4. 用少量步数做训练冒烟测试；
5. 正式运行多个 seed；
6. 仅用验证集选择模型；
7. 冻结方案后再运行独立测试集；
8. 保存模型、配置、日志、指标和图片的完整组合。

建议的最小检查命令：

```powershell
.\.venv\Scripts\python.exe -m py_compile envs\dynamic_path_env.py
.\.venv\Scripts\python.exe -m py_compile envs\scenario_dataset.py
.\.venv\Scripts\python.exe -m py_compile scripts\train.py
.\.venv\Scripts\python.exe -m py_compile scripts\evaluate.py
.\.venv\Scripts\python.exe scripts\test_environment.py
.\.venv\Scripts\python.exe scripts\test_fixed_scenarios.py
.\.venv\Scripts\python.exe scripts\test_dataset_splits.py
.\.venv\Scripts\python.exe scripts\test_experiment_workflow.py
```

## 8. 场景数据开发

重新生成当前格式的 30 场景：

```powershell
.\.venv\Scripts\python.exe scripts\generate_test_scenarios.py `
  --num-scenarios 30 `
  --seed 2026 `
  --split validation `
  --output configs\validation_scenarios_30.json
```

生成器以目标直达基线的最小净空划分难度。场景 JSON 包含：

- schema 和数据集名称；
- 数据集 seed 和生成配置；
- 起点、初始航向、目标；
- 静态障碍位置和半径；
- 动态障碍位置、速度和半径；
- 基线结果；
- 单场景哈希。

生成器强制恰好 30 个场景和固定配额，并在 schema v2 中写入 `dataset_split`。验证集和测试集必须使用不同 seed，并通过 `test_dataset_splits.py` 检查场景哈希不重叠。

## 9. 调试指南

### 9.1 OpenMP `libiomp5md.dll already initialized`

优先确认所有科学计算包来自同一个虚拟环境：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip show numpy torch
```

不要同时混用 Anaconda、系统 Python 和 `.venv` 的包。`KMP_DUPLICATE_LIB_OK=TRUE` 只是可能隐藏崩溃或错误结果的临时绕过方式，不应作为正式解决方案。

### 9.2 结果突然变差

依次检查：

1. 评估加载的是 `best_model` 还是 `final_model`；
2. 观测维度及语义是否与模型训练时一致；
3. `selection_metrics.json` 与结果文件是否来自同一轮；
4. 场景文件及其哈希是否变化；
5. 奖励、学习率、seed 和训练步数是否变化；
6. 分类型指标中是哪类场景退化。

### 9.3 “最佳模型还是很早的步数”

这是允许的：后续策略可能在随机训练奖励上继续提高，却在固定验证成功率或碰撞率上下降。应查看完整 `evaluations.json`，而不是默认最后一步最好。

### 9.4 上下绕行明显不对称

当前已观察到明显向上绕行偏置。排查时应使用上下镜像场景，并比较镜像前后的初始角速度、轨迹侧别和成功率。仅统计动态障碍向上/向下数量不足以判断策略是否对称。

## 10. 测试策略的不足与扩展建议

当前测试是可执行脚本，不是自动化测试套件。建议逐步增加：

- pytest 单元测试：观测编码、奖励每一分量、角度归一化、反弹和碰撞；
- 参数化边界测试：最大/最小半径、地图边缘、同时发生的终止事件；
- 确定性测试：相同 seed 和模型得到相同结果；
- 对称性测试：上下镜像场景的策略响应；
- 反事实测试：移除、冻结或反转动态障碍；
- 产物 schema 测试：指标包含模型与场景哈希；
- CI：语法、pytest 和固定小型冒烟评估。

## 11. 反事实诊断

运行配对诊断：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_counterfactual.py `
  --run-dir runs\ppo-seed0 `
  --scenario-file configs\validation_scenarios_30.json
```

脚本成对比较原场景、上下镜像、冻结动态障碍、反转动态速度和远置动态障碍，输出成功率、路径变化、绕行侧变化及初始动作。由于16维观测没有“障碍不存在”的编码，远置动态障碍只是移除障碍的近似代理，结果中会明确记录该限制。

配对诊断统一保持原场景的速度归一化尺度，避免冻结速度同时改变分母；这不改变训练环境的 16 维编码。第零轮基线与完整验收入口见 [ROUND_ZERO_ACCEPTANCE.md](ROUND_ZERO_ACCEPTANCE.md)。可使用 `scripts/test_round_zero.py` 检查末次更新选模和来源验证，使用 `scripts/freeze_round_zero.py --run-dir <运行目录>` 执行两次验证重放及开发诊断并生成产物哈希清单。

项目下一步计划和执行状态统一见 [PROGRESS.md](PROGRESS.md)，总体目标及设计约束见 [DESIGN.md](DESIGN.md)。本文负责开发和复现操作，不维护轮次待办。

## 12. 第二轮对照实验流程

本轮不修改环境、奖励、网络、训练脚本或旧结果。`baselines/dwa.py` 实现只接收 16 维观测的 DWA 风格局部规划器；`scripts/round_two_experiment.py` 负责验证集选参、冻结协议、最终评估；`scripts/summarize_round_two.py` 按冻结的统计方案读取结果，不重新训练或推理。

已完成的本轮直接查看 `results/round2/SUMMARY.md` 和 `docs/ROUND_TWO_ACCEPTANCE.md`。下面列出原始执行顺序，**不是要求在现有目录重跑**；调参和封存步骤遇到已存在的产物会拒绝覆盖。

```powershell
# 第一轮 seed 1 直接复用；以下四次均从零训练。
.\.venv\Scripts\python.exe scripts\train.py --stage C --timesteps 200000 --seed 2 --run-id round2-C-seed2
.\.venv\Scripts\python.exe scripts\train.py --stage C --timesteps 200000 --seed 3 --run-id round2-C-seed3
.\.venv\Scripts\python.exe scripts\train.py --stage D --timesteps 200000 --seed 2 --run-id round2-D-seed2
.\.venv\Scripts\python.exe scripts\train.py --stage D --timesteps 200000 --seed 3 --run-id round2-D-seed3
.\.venv\Scripts\python.exe scripts\test_round_two.py
.\.venv\Scripts\python.exe scripts\round_two_experiment.py tune
.\.venv\Scripts\python.exe scripts\round_two_experiment.py seal
.\.venv\Scripts\python.exe scripts\round_two_experiment.py evaluate
.\.venv\Scripts\python.exe scripts\summarize_round_two.py
```

`evaluate` 会校验源码、模型、数据集身份，跳过哈希一致的已完成结果；不会静默覆盖。未完成的输出目录需先人工核查并另行归档，再恢复该项；不要删除完成项或基于最终测试改参数。迁移机器时绝对模型路径需要新的、说明迁移关系的实验协议，不能直接编辑旧封存文件。

每个方法/测试集目录包含 `episodes.csv`、`by_type.csv`、`by_risk.json`、`summary.json`、全部轨迹 `trajectories.npz` 和 `complete.json` 哈希记录。为控制图像体积，仅绘制按 ID 排序的前三个失败，仍按每张三场景排版；其余轨迹均保留数值，可事后绘制但不能用来重新选模。零失败的方法没有失败图。

统计以 3 个训练 seed 为模型随机性单位，给出样本标准差和自由度 2 的 t 区间；配对分层 bootstrap 同时重采样 seed 与各会遇类型内的场景 ID，所有方法保持同场景配对。DWA 是一个确定性控制器，不虚构三份独立基线。单模型 Wilson 区间只是场景二项近似，不替代 seed 波动；固定分层设计的主要差值区间使用分层 bootstrap。仅三个 seed 时区间估计仍很不稳定，不能宣称训练已收敛或有普适显著优势。

## 13. 提交前检查清单

- [ ] 环境与固定场景检查通过；
- [ ] 观测、动作和奖励变更已同步到设计文档；
- [ ] 旧模型兼容性已明确；
- [ ] 没有把测试集用于选模或调参；
- [ ] 至少记录训练 seed、配置和场景哈希；
- [ ] 评估显式指定模型路径；
- [ ] 指标和轨迹图来自同一模型；
- [ ] 对失败场景做逐项检查，而非只看平均奖励；
- [ ] 保留修改前后完整产物以便复现。

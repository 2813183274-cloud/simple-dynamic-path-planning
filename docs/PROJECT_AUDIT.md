# 项目收口审计报告

审计日期：2026-09-09。审计基于提交 `f1cfbfc` 后的工作区，只读检查了 Git 状态、目录、27 个非快照 Python 源文件、命令入口、配置、模型/结果引用和既有封存约束。本报告先于结构重构形成。

## 1. 结论摘要

项目不是由多套 `env_v2/env_final` 构成；当前环境主实现只有 `DynamicPathPlanningEnv`。混乱主要来自三轮实验各自保留的训练、评估、封存和统计脚本，以及根目录缺少统一入口和机器可读的主线配置。

不能直接合并或移动这些轮次脚本：第零至第三轮的模型身份、实现封存和结果清单记录了源文件路径与 SHA-256。改写 `scripts/train.py`、`scripts/evaluate.py`、环境或场景模块，会让历史模型认证失败；移动 `round_*` 脚本会破坏复核命令。因此目标应是“一个对外主入口 + 原封不动的历史实验工具”，而不是物理删除可复现链。

当前真实主线为阶段 D 环境、16 维观测、原线性奖励和 PPO；第三轮风险奖励是独立 wrapper 消融，不是默认算法。固定 30 场景存在三个不同用途文件：`test_scenarios_30.json` 是历史固定 benchmark（已经用于历史选模，不能称为未见测试），`validation_scenarios_30.json` 用于旧主线选模，`independent_test_scenarios_30.json` 是旧主线独立测试。后两轮另有新的封存大样本测试集。文件名本身不足以表达这些数据治理差异。

## 2. 当前结构与职责

| 路径 | 职责 | 判定 |
|---|---|---|
| `envs/dynamic_path_env.py` | 唯一 USV Gymnasium 环境；reset/step、运动学、动作映射、16 维观测、障碍更新、碰撞/边界/目标、原奖励、基础渲染 | 主线 KEEP；已被历史来源封存 |
| `envs/scenario_dataset.py` | legacy 30 场景的读写、哈希、几何验证、生成与直行见证 | KEEP；职责偏多但不可在收口阶段拆分 |
| `envs/encounters.py` | A–D 会遇类型、CPA、几何约束和随机场景采样 | KEEP |
| `envs/round_one_dataset.py` | A–D 数据集序列化、生成与验证 | KEEP；名称带轮次但仍被 D 主线训练使用 |
| `envs/risk_reward.py` | 第三轮 base/instant/predictive 风险指标及奖励 wrapper | KEEP，实验扩展，不是默认奖励替代品 |
| `baselines/dwa.py` | 服从相同速度/加速度限制的 DWA 对照 | KEEP |
| `experiment_utils.py` | run-id、原子 JSON、哈希、Git/依赖/环境签名和旧模型认证 | KEEP |
| `round_three_utils.py` | 第三轮协议、实现与模型封存验证 | KEEP，但属于第三轮实验基础设施 |
| `scripts/train.py` | 正式通用 PPO 训练、Monitor、固定验证选模、续训认证、奖励曲线 | 当前训练核心 KEEP |
| `scripts/evaluate.py` | 通用固定场景推理、统一基础指标、CSV/JSON、每 3 场景绘图 | 当前评估与可视化核心 KEEP |
| `scripts/generate_test_scenarios.py` | legacy 固定数据集生成工具 | KEEP；不在日常主入口中突出 |
| `scripts/manual_policy_test.py` | 手工控制器和环境渲染冒烟 | KEEP，开发工具 |
| `scripts/test_environment.py` | 运动学、动作限制、终止与随机环境测试 | KEEP |
| `scripts/test_fixed_scenarios.py` | 固定数据复现和 reset 行为测试 | KEEP |
| `scripts/test_dataset_splits.py` | 30 场景验证/独立测试哈希排重 | KEEP |
| `scripts/test_experiment_workflow.py` | run 身份与工作流回归 | KEEP |
| `scripts/freeze_round_zero.py`, `test_round_zero.py` | 第零轮封存与复核 | ARCHIVE 语义；保留原路径以维持复现 |
| `scripts/generate_round_one.py`, `evaluate_round_one.py`, `test_round_one.py` | 第一轮数据、批量评价和回归 | ARCHIVE 语义；保留原路径 |
| `scripts/evaluate_counterfactual.py` | 配对动态响应诊断，复用通用绘图 | 有价值的分析工具 KEEP |
| `scripts/round_two_experiment.py`, `summarize_round_two.py`, `audit_round_two.py`, `test_round_two.py` | 第二轮 DWA、最终实验、统计、审计 | ARCHIVE 语义；封存依赖，保留原路径 |
| `scripts/train_round_three.py`, `evaluate_round_three.py`, `diagnose_round_three.py`, `freeze_round_three.py`, `round_three_experiment.py`, `test_round_three.py` | 第三轮消融训练、共同 base reward 评估、诊断、冻结、最终实验和回归 | ARCHIVE 语义；封存依赖，保留原路径 |
| `configs/test_scenarios_30.json` | 用户指定不可变的 30 场景历史 benchmark | KEEP；默认统一 benchmark，但必须标注历史用途 |
| `configs/validation_scenarios_30.json` | legacy/default 模型选优 | KEEP |
| `configs/independent_test_scenarios_30.json` | legacy/default 独立最终测试 | KEEP |
| `configs/round1/`, `configs/round3/` | 已完成轮次的数据与协议 | KEEP；不搬迁 |
| `models/` | 根目录旧默认模型兼容入口 | KEEP；身份不如 runs 完整，不覆盖 |
| `runs/` | 认证训练运行、best/final、日志和快照；Git 忽略 | KEEP，不清理 checkpoint |
| `results/round1..3/` | 论文实验、封存清单、逐场景数据和统计 | KEEP，不重排 |
| `archives/legacy-pre-round0-20260908.zip` | 已执行过的早期散落产物归档 | KEEP；恢复说明已有文档 |

## 3. 重复、耦合和命名问题

### 3.1 没有重复实现的核心逻辑

USV 运动学、动作映射、reset/step、观测、碰撞、边界、goal 和原奖励均只在 `dynamic_path_env.py` 实现。TCPA/DCPA 基础计算在 `encounters.py`，第三轮面向观测的有限时域风险在 `risk_reward.py`，两者用途不同，不应强行合并。PPO 构造只有 `scripts/train.py:create_model`，第三轮训练已复用它。通用绘图函数集中在 `scripts/evaluate.py`，第二、三轮会调用这些函数。

### 3.2 存在的重复或耦合

- `scripts/evaluate.py` 同时承担模型解析、回合执行、指标和绘图，427 行；`evaluate_round_three.py` 为额外逐步风险指标另有执行循环。两者输出契约不同，但基础 episode loop 重复。
- `round_two_experiment.py`、`round_three_experiment.py` 各自包含编排、审计、绘图和统计。它们属于封存实验，不适合作为新开发模板。
- 场景几何/CPA 分散在 `scenario_dataset.py`、`encounters.py` 与第三轮风险模块，存在数学概念相近但坐标和用途不同的实现；当前数值已封存，不能在无等价证明时合并。
- 超参数硬编码于环境和 `create_model`；各轮配置记录了实际值，但日常入口没有单一配置文件。直接把核心改为读取 YAML 会改变源哈希和模型认证。
- 各脚本自行建立 `ROOT` 并解析路径；方式基本一致，但没有公开的根入口。
- `round_one_dataset.py`、`round_three_utils.py` 的名称看似历史文件，实际上仍是对应认证流程依赖；“round”表示实验版本，不是可随意删除的旧副本。

## 4. KEEP / MERGE / ARCHIVE / DELETE

### KEEP

保留所有 `envs/`、`baselines/`、三个通用核心文件、`scripts/train.py`、`scripts/evaluate.py`、固定数据、模型、runs、results、现有文档和归档。根目录将新增统一入口与主线配置；旧入口仍可运行。

### MERGE

本轮只做入口层合并：统一配置加载、训练命令构造、评估命令构造和可视化开关。不会把已封存的 Python 文件物理合并。未来若解除历史认证路径约束，可在新版本中抽取 evaluation runner/metrics/plotting，但必须以快照兼容方式而非改写旧源码实施。

### ARCHIVE

第零至第三轮专用脚本、配置和结果在语义上归入“reproducibility archive”。由于路径和 SHA-256 被封存，它们继续留在 `scripts/`、`configs/round*`、`results/round*`，通过文档索引降级，不进行物理移动。既有 `archives/` 保留；不另复制一份造成更多冗余。

### DELETE

审计未发现可安全删除的非缓存 Python 文件或 checkpoint。项目源码树中的 `__pycache__` 若由测试生成可以删除，`.venv` 内缓存属于本地虚拟环境且已被 Git 忽略，不做递归清理。`results/round3/training-console/` 是正式训练控制台记录，暂保留。已有 Git 状态中的删除/修改来自先前工作，不能把它们误判为本任务授权删除。

## 5. 真实调用链（重构前）

训练：`scripts/train.py` → CLI 参数 → `DynamicPathPlanningEnv(stage=D)` → `Monitor` → `create_model(PPO)` → `FixedScenarioBestModelCallback` 读取固定验证集 → run 下 best/final/checkpoints/logs。

评估：`scripts/evaluate.py` → 解析 model/run → 模型身份检查 → 加载固定场景 → `DynamicPathPlanningEnv` → `PPO.load` → `run_scenario` → `build_summary/build_type_metrics` → CSV/JSON → `save_plot_sheets`。

第三轮是已完成的独立链：`train_round_three.py` + `RiskRewardWrapper`，最终由 `round_three_experiment.py` 统一封存和评估；它不是日常入口。

## 6. Potential algorithm issues（只报告，不修改）

1. `dynamic_path_env.py` 与 `train.py` 的环境/PPO参数硬编码，容易出现文档与运行参数漂移；历史结果受这些值影响。本轮通过外层配置声明和校验解决可发现性，不改数值来源。
2. PPO 确定性动作均值大量超过动作上界后被裁剪，P3-05 中策略几乎持续满速；可能限制减速行为，但不能据此断言训练梯度或失败唯一原因。历史结果已受影响，见第三轮验收。
3. 16 维观测没有边界距离；越界风险无法被直接表达。固定半径下中心距离足以确定当前圆形净空，但未来随机尺寸/形状不成立。
4. `test_scenarios_30.json` 已用于历史选模。按用户要求可作为固定 benchmark，但不能重新称为独立、未见最终测试；论文最终证据应使用封存的独立数据。
5. 第三轮风险预测假设 USV 保持当前速度/航向，目标匀速；它是奖励代理而非真实未来保证。P3-05 没有验证出预期减速收益。
6. `scenario_dataset.py` 既生成数据又含直行见证和几何统计，耦合偏高；拆分会改变历史源哈希，留待新主版本。

## 7. 目标结构（收口版）

```text
simple-dynamic-path-planning/
├── train.py                 # 配置驱动的公开训练入口
├── evaluate.py              # 配置驱动的固定 benchmark 入口
├── visualize.py             # 复用 evaluate 绘图的公开入口
├── configs/main.json        # 默认主线、PPO、路径与 benchmark 声明
├── envs/                    # 唯一环境及版本化场景/风险扩展
├── baselines/               # DWA
├── scripts/                 # 核心实现 + 已完成轮次复现工具
├── models/                  # 旧默认兼容模型，不覆盖
├── runs/                    # 认证运行（Git 忽略）
├── results/                 # 分轮正式结果，不重排
├── docs/
│   ├── PROJECT_AUDIT.md
│   ├── EXPERIMENT_LOG.md
│   └── ...
└── archives/                # 早期散落产物压缩归档
```

不创建只有十几行职责的 `agents/`、`utils/geometry.py` 等层；当前只有 PPO，`create_model` 已集中。根入口是稳定门面，内部已封存实现仍保留。

## 8. 小步实施计划

1. 提交本审计报告，作为修改前分类基线。
2. 新增不侵入旧核心的 JSON 配置读取与三个根入口；校验配置声明与环境/PPO实际常量一致，避免悄悄改变算法。
3. 增加入口/奖励/固定场景回归测试；先跑 import、环境、reward、固定场景和极少步训练 smoke，再酌情跑完整 30 场景。
4. 更新 README、DEVELOPMENT、EXPERIMENT_LOG 与最终变更报告。
5. 只删除本次测试在源码树产生的缓存；不移动封存源码、不删除 checkpoint、不修改三个 30 场景文件。

每一步单独 Git commit，不 push、不改写历史。工作区在本任务开始前已有大量修改/删除/未跟踪文件；提交必须按路径精确暂存，不能夹带无关状态。

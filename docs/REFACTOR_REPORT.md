# 项目收口最终变更报告

日期：2026-09-09。范围：系统审计、目录职责收口、统一公开入口、配置可见化、回归验证与文档整理。没有重新设计算法。

## 1. 原项目主要问题

- 根目录没有稳定的 train/evaluate/visualize 门面，使用者需要理解大量轮次脚本。
- 环境、奖励和 PPO 数值散落在实现中，已有 run 虽记录配置，但日常入口没有一个明确主配置。
- 已完成实验脚本与当前开发入口混放，文件名不能说明它是主线还是复现工具。
- 通用评估与第三轮风险评估有 episode loop/统计重复；第二、三轮编排脚本职责较重。
- `test_scenarios_30.json`、validation、independent test 和后续封存测试用途容易混淆。
- README 仍以第一轮为主要入口，目录树和第三轮完成状态过时。
- 工作区在本任务开始前已有大量修改、删除和未跟踪文件，不能用粗粒度清理或全量暂存。

完整逐文件职责和重复分析见 [PROJECT_AUDIT.md](PROJECT_AUDIT.md)。

## 2. 新目录结构

```text
├── train.py / evaluate.py / visualize.py  # 唯一公开入口
├── project_config.py                      # 配置读取、契约核对、委托
├── configs/main.json                      # 当前主线配置
├── envs/                                  # 唯一环境及场景/风险扩展
├── baselines/                             # DWA
├── utils/metrics.py                       # 初始名义 CPA 指标
├── scripts/                               # 核心、回归、历史复现工具
├── runs/ / models/ / results/             # 运行、模型、正式结果
├── docs/                                  # 设计、开发、审计、实验日志
└── archives/                              # 既有早期产物归档
```

没有机械创建 `agents/`、`dynamics.py`、`obstacles.py` 等薄层。当前环境只有一个类，继续拆分会增加接口且破坏来源哈希。

## 3. 保留的文件

保留全部环境、场景生成、奖励、PPO 核心、DWA、通用评估、回归、轮次脚本、模型、checkpoint、runs、results、配置和历史文档。`configs/test_scenarios_30.json` 内容与 SHA-256 均未改变。旧根模型继续作为 legacy 兼容默认，不被新模型覆盖。

## 4. 合并的内容

入口层完成逻辑合并：`configs/main.json` 统一声明环境/奖励/PPO/训练/评估信息；三个根入口统一调用 `project_config.py`，再委托既有 `scripts/train.py` 或 `scripts/evaluate.py`。可视化复用评估绘图，不另写 trajectory loop。原封存源码没有物理合并。

## 5. 进入 archive 的内容

没有新移动文件。第零至第三轮专用脚本、配置和结果在文档中被标为“复现档案”，但保留原路径，因为模型认证及 artifact manifest 固定记录了路径和字节哈希。既有早期散落输出已在此前进入 `archives/legacy-pre-round0-20260908.zip`。

## 6. 删除内容与原因

没有删除 checkpoint、结果、Python 源码或不确定文件。只清理本次测试在项目源码目录生成的 `__pycache__`；`.venv` 内缓存属于本地环境并已 Git 忽略，不递归清理。任务开始前 Git 已显示的删除项保持原状，没有据此扩大删除范围。

## 7. train.py 当前调用链

```text
train.py
→ load_config(configs/main.json)
→ verify_locked_environment（环境、reward、PPO 与实现一致）
→ scripts/train.py
→ DynamicPathPlanningEnv(stage D)
→ Monitor
→ create_model(PPO)
→ FixedScenarioBestModelCallback
→ runs/<run-id>/{best, final, checkpoints, logs, snapshot}
```

必须指定 `--run-id`，可用 `--dry-run`。训练场景仍在每次 reset 随机生成，没有变成固定测试场景训练。

## 8. evaluate.py 当前调用链

```text
evaluate.py
→ load_config + locked contract check
→ scripts/evaluate.py
→ resolve model/run + provenance check（认证 run 时）
→ load fixed 30-scene benchmark
→ DynamicPathPlanningEnv + PPO.load
→ run_scenario × 30
→ build_summary / build_type_metrics
→ CSV + JSON + save_plot_sheets
→ nominal_initial_tcpa_dcpa.json
```

`visualize.py` 走同一调用链并强制绘图。没有第二套绘图实现。

## 9. reward 组成

默认 reward 为：`10 × 距目标进展 − 0.05 + 静态线性安全项 + 动态线性安全项 + 单次终端项`。静态/动态安全中心距离阈值为 7/8 m；动态安全项权重 2。成功 +200，静态碰撞 −120，动态碰撞 −150，越界 −100。第三轮的 instant/predictive 项仍在独立 wrapper，未并入默认 reward。

## 10. config 关键参数

- environment：stage D、16 维输入、2 维动作、地图/dt/步数、半径、速度/加速度和到达阈值。
- reward：进展、时间、终端惩奖、安全距离、线性形状和动态权重。
- ppo：学习率、rollout、batch、epochs、gamma、GAE、clip、entropy、网络与激活。
- training：预算、seed、验证频率、曲线平滑窗口、device、runs 路径。
- evaluation：固定 benchmark、默认模型、输出、绘图和历史 benchmark 状态。

前三段是锁定契约：修改 JSON 不会静默改变算法，而会因与实现不一致而拒绝运行。

## 11. 发现的算法风险

- 确定性 PPO 速度动作存在明显上界裁剪和持续满速，第三轮未观察到预定风险后减速；原因尚不能唯一归于裁剪。
- 16 维观测不包含边界距离；未来若使用随机尺寸/形状，中心距离不再足以表达净空。
- 第三轮预测只是假设当前航向/速度与目标匀速的风险代理，不是未来真实轨迹。
- 根目录 legacy 模型身份不完整；本次 benchmark 输出明确记录 `legacy_training_identity_unverified=true`。
- 固定历史 benchmark 曾用于模型开发，不能包装成未见测试。
- 新增 CPA 文件是“初始名义 CPA”，不是策略实际轨迹 TCPA/DCPA；定义已写入产物。

均只报告或明确命名，没有在本次偷偷修复算法。

## 12. 已执行测试

- 四个公开模块编译/import 与三个根入口 dry-run。
- `scripts/test_public_workflow.py`：配置契约、benchmark SHA-256、base wrapper 逐步 reward/状态等价、三个固定场景、CPA 输出、临时 PPO rollout。
- `scripts/test_environment.py`：环境、动作/运动学和终止 smoke。
- `scripts/test_fixed_scenarios.py --scenario-file configs/test_scenarios_30.json`。
- `scripts/test_dataset_splits.py`：旧 validation 与 independent test 有效且不重叠。
- `evaluate.py --config configs/main.json`：完整 30 场景、指标和 10 张图。

## 13. 测试结果

以上全部通过。临时 PPO 测试请求 8 步，但 SB3 按 `n_steps=2048` 完成一个 rollout；仅在临时目录运行并删除，没有长期训练产物。完整 benchmark 得到 23/30 成功、7 次静态碰撞、0 动态碰撞/越界/超时，和已有选择记录一致。固定场景 SHA-256 仍为 `75bca781a36d768b8ca25ed8dac0bd5e1e6ad63a56c2f45ca73d5786cfd2c757`。

## 14. 未运行项目

没有重新运行九次正式训练、第二/三轮 3600 回合、所有历史封存脚本或 GPU 测试；这些已有哈希核验产物，重跑既耗时也不属于结构重构验证。未验证 Linux/macOS 命令，本机为 Windows。未重新生成任何固定场景。

## 15. 后续仍值得清理的地方

在不承担兼容迁移前，不建议继续物理清理。若未来建立明确的 v4 主版本，可复制历史源码到不可变 release/tag 后，再抽取统一 episode runner、trajectory schema 和 policy-trajectory TCPA/DCPA；必须用旧快照做数值回归，并建立新模型身份。P4-01 动作通道诊断仍是研究候选，与本次工程收口分开。

当前七个问题已有直接答案：环境在 `envs/dynamic_path_env.py`；reward 在同文件及实验 wrapper；PPO 从根 `train.py`；30 场景从根 `evaluate.py`；实验差异见 `EXPERIMENT_LOG.md`；最终配置见 `configs/main.json` 和各 run 的 config；端到端调用链见本文第 7–8 节。

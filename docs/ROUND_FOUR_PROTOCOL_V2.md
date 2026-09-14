# 第四轮协议 v2：有限执行接口与潜动作 PPO

2026-09-09。在任何正式训练、新最终数据生成之前修订；用户已确认继续。保留 [v1](ROUND_FOUR_PROTOCOL.md) 和 [接口阻断记录](ROUND_FOUR_IMPLEMENTATION.md)。本文件替代 v1 第 2 节的无界接口及第 4/6 节对应实现要求；其余实验分组、奖励、网络、初始化、预算、选模、指标、统计及数据冻结顺序全部继承 v1。机器入口为 `configs/round4/experiment_protocol_v2.json`，通过明确继承 v1 并覆盖接口字段解析。

## 接口契约

- 环境始终声明并接收二维 `Box(-1,1)` **执行动作 a**。环境包装层只增加诊断信息，不变换动作或改变奖励。
- 独立 `LatentActionPPO` 适配 SB3 2.8.0 收集流程：策略采样 z、value、潜高斯 log-prob；仅在 env.step 前执行一次 `a_v=clip(z_v)` 或 `tanh(z_v)`，`a_w=clip(z_w)`。rollout 始终保存 z，PPO 的原 train/evaluate_actions 不变。
- 保留原 rollout 中 episode-start、done、时间限制 terminal_observation 引导、GAE、回调中断及日志语义；仅支持当前二维连续动作、16 维观测、无 gSDE、无 squash_output 的协议。依赖版本和适配所依据的 SB3 源码字节哈希一起封存。
- 回调 `actions` 是原始 z，`clipped_actions` 是实际 a（兼容名称）；另存均值、标准差和动作诊断，禁止在训练回调中再次映射。正常评估/绘图使用 `model.predict()` 返回的 a 并直接 env.step；需要潜变量时使用单独的 `action_details()`，它同时返回 a 和原始数据。
- `model.predict()`、训练收集和评估复用同一个纯映射函数。`model.policy.predict()` 不是受支持的执行入口，不得用于第四轮推理。加载必须使用显式 `LatentActionPPO.load` 和来源认证，拒绝旧 PPO 文件、缺少 schema 的文件或错误组名；不能自动将旧模型升级。
- 两组保持相同策略类、参数形状、初始化和潜高斯解析熵，ent_coef=.01。变换后速度密度的 Jacobian 只用于数值核验；不参与潜空间概率/熵的计算。没有重复映射，也不将 a 当作高斯 z。
- clip 组也使用同一适配器，与原 SB3 的完整采样 rollout 逐项比较（动作、log-prob、奖励、done/episode-start、value、return、advantage）。两组测试保存加载后动作与轨迹一致；适配器没有新增可训练参数。

## 实施与封存

不修改旧核心或 site-packages，不复制整个历史训练系统。复用既有模型选择调度、候选排名、日志、绘图及风险诊断函数；仅新增必要的动作适配、第四轮运行身份和评估逻辑。正式入口默认只检查就绪，需要显式 `--execute`；本次禁止运行该选项。

P4-03 测试只允许采样、构造模型、临时保存加载以及 backward；以拦截 optimizer.step 和 PPO.train 的方式确认没有优化器更新。对时间限制终止、自然终止、立即减速计时、无响应、停滞、动作极值、无 NaN/Inf、来源篡改、末次选模和重放进行回归。正式结果仍需 P4-04 执行后才能讨论效果。

实现、测试报告、完整 PPO 运行参数、依赖版本、关键依赖源码及协议哈希形成独立封存清单。旧轮次封存和 P4-01 诊断产物不得被覆盖；协议再次变化须另立版本。

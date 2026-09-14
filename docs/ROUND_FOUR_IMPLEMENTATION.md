# P4-03 实现检查：协议 v1 被依赖接口阻断

日期：2026-09-09。状态：已启动，依赖检查阻断，未完成实现验收；无训练、无优化器更新，无新模型或最终数据。

## 复现证据

本地 Stable-Baselines3 2.8.0 在 `common/base_class.py` 第 198–201 行要求 Box 动作上下界均有限。协议 v1 规定的外层 `Box(-inf, inf, shape=(2,))` 会在 PPO 初始化时抛出：

```text
Continuous action space must have a finite lower and upper bound
```

复现入口（预期捕获该错误并输出阻断记录，退出码 0 表示复现成功，不表示协议可运行）：

```powershell
.\.venv\Scripts\python.exe -B scripts/check_round_four_action_space.py
```

检查没有执行 learn、optimizer.step、环境训练回合或保存模型。P4-02 检查了 squash_output 限制，但遗漏了基类的有限边界断言；这是前期接口设计遗漏，不是 tanh 数学定义或 PPO 本身不适用的证明。

## 为什么暂不继续

[协议 v1](ROUND_FOUR_PROTOCOL.md) 第 2 节明确要求：无界外层接口若不被依赖支持，暂停并修订协议，不能静默换实现。因此本次不删除依赖断言、不使用巨大有限边界冒充无界空间、不预裁剪 z，也不改 site-packages。P4-03 其他实现尚未开始，不能宣称测试全通过或实现已封存。

## 建议的 v2 修订（待确认，未实施）

保留“只改变速度映射”的研究问题和两组对照，但改用有限的执行动作接口及显式潜动作采样流程：自定义受版本约束的 PPO rollout 收集与推理适配，保存原始 z 及其高斯 log-prob，只把映射后的 a 交给环境；PPO 更新仍对保存的 z 计算比值和潜高斯熵。不能只改 predict 而忽略训练收集路径。

这需要新增受测试的算法适配层，而非原计划的单纯环境包装层。v2 必须明确训练/验证/加载后推理接口、回调看到的动作、终止引导、rollout 原始动作存储和双重映射防护；clip 组同时走适配层，与旧 SB3 做逐步等价回归，避免把适配差异混进主比较。两组奖励、网络、场景和预算无需因此变化。

确认修订方向后新建 v2 文档及机器配置，保留 v1 与本记录，再继续 P4-03。尚未消耗正式训练预算或使用新最终集。

## v2 后续结果（2026-09-14）

用户确认后已建立 [协议 v2](ROUND_FOUR_PROTOCOL_V2.md)，以有限执行动作接口和显式潜动作 PPO 适配替代 v1 的无界 Box。P4-03 已完成实现、回归与独立封存，v1 的失败记录继续保留。

- `LatentActionPPO` 只在环境执行前映射一次动作，rollout 保存原始高斯潜动作；环境仍接收二维 `[-1,1]` 动作。
- clip 组与本地 SB3 2.8.0 在自然终止和时间限制终止下的动作、log-prob、奖励、episode-start、value、return 和 advantage 精确一致。
- speed_tanh 组的潜空间概率比、解析熵、Jacobian 抵消、有限梯度、回调中断、保存加载、向量推理、逐步日志与绘图重放通过。
- 底层环境状态、原奖励和加速度限制未变；旧模型会被新版加载器拒绝，模型组名与动作 schema 不能被调用者覆盖。
- 测试拦截 `PPO.train` 与 `optimizer.step`，优化器更新为 0；未启动正式训练，未生成最终数据。

封存产物位于 `results/round4/implementation/`。下一步是 P4-04，必须由用户另行明确启动；训练入口默认仅检查就绪，只有 `--execute` 才会训练。

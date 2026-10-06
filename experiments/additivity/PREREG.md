# 预注册：E-B 能力可加性 —— 冻结认知核 + per-module 低秩旁路（bypass 臂）

> 写于任何 bypass 训练启动之前（2026-10-06 01:5x +08）。判据与配置在跑前冻结，跑完不改。

## 唯一待判问题
**冻结认知核 + 给新模块一条自己的低秩旁路适配器（per-module LoRA，不与任何老模块共享）
能否兼得「老模块零损伤」与「新模块学得好」。**

三档（新模块 = negation）：

| 档 | 认知核 | 适配 | 基线 |
|---|---|---|---|
| joint | 可训 | 无 | exact_match **0.9667**（dev-notes/12 §12.5，N5 seed42，文档值，不重跑） |
| frozen | 冻结 | 无 | exact_match **0.6383**（同上，R 臂，A′ 基座，文档值） |
| **bypass** | 冻结 | 每模块低秩旁路 | **本实验测** |

> 口径说明（如实记）：文档两档的基座各不相同（joint = N5 自己的联合基座；frozen = A′ 基座）。
> 本实验主档按委托指定用 `checkpoints/base_encoder.pt`（multitask_v2 拆分基座，4 张老卡也挂它），
> 因此额外跑**同配方 frozen 对照臂**（下称 F）作为配对基线；另跑一档 A′ 基座校准（F/B on A′）
> 用来把 0.6383 这个文档值接上（F-A′ ≈ 0.6383 才说明配方复现成功）。

## 配置（冻结）

- **旁路插入点（4 处）**：`encoder.blocks[0..2]` 的输出（残差流，每块之后）+ `encoder.norm` 的输出（喂给解码头之前）。形式统一：`h ← h + scale * (h @ Aᵀ @ Bᵀ)`，`A ∈ R^{r×d}` 小初始化 `N(0, 1/d)`，`B = 0` **精确零初始化** ⇒ 初始行为 = 冻结基座。
- **rank r = 16，alpha = 16（scale = alpha/r = 1.0）**，d = 128。可训旁路参数 = 4 × (r·d + d·r) = 4 × 4096 = **16384**（跑后以 `requires_grad` 实际打印值为准）。
- **gate 语义**：旁路开关是显式布尔；关 = 直接返回原张量（不加任何算术）⇒ 老模块路径严格不经过旁路（结构性保证）。
- **基座**：`checkpoints/base_encoder.pt` 全参数 `requires_grad=False`、常驻 `.eval()`（dropout 关）。
- **可训**：negation 解码头（`RobustARSliceDecoder`，2 层，与 `train_task_card.py` 同规格）+ 旁路（bypass 档）。
- **训练预算（与 N5/R 臂逐项对齐）**：16 epoch × 84 步 = **1344 步**，batch 64，negation 6000 样本，
  AdamW lr 1e-3 wd 1e-4，CosineAnnealing 1344 步，grad clip 1.0，`build_dataset(seed=DEFAULT_SEED)` +
  `random.Random(seed).shuffle`，val = 前 `max(200, n//10)` = 600。
- **seed：42、43**（两 seed 同号才下结论）。
- **配对保证**：同 seed 下先 `manual_seed(seed)` 建头、再用**独立 generator（seed+1000）**建旁路 A
  ⇒ F 与 B 档的头初始化、数据顺序逐位相同，唯一差异是旁路。

## 判据（跑前写死，不许事后放宽）

- **P1（老模块无损）**：4 张老卡 `exact_match` 的 Δ（挂新卡 vs 不挂新卡）≤ **各自**噪声带：
  pronoun 2.83pt / relation 1.39pt / sentiment 0.41pt / person 0.33pt。**不许用统一阈值。**
  另做逐样本 `engine.predict` 输出哈希比对（要求逐位一致）与一个**敏感性对照 C**（旁路对老卡强制开启，
  预期指标会动 —— 若 C 也不动，说明这个一致性检查是空的）。
- **P2（新模块可用）**：bypass 档 negation `exact_match` ≥ **0.87**（= 0.9667 × 90%），**两个 seed 都要过**。
- **P3（代价）**：报参数量增量、稳态每步耗时、峰值显存，与 F 档**配对**（同机、同 seed、同预算）。
- **空测试自检三条（缺一条结论作废）**：
  1. 打印**生效后**的可训参数量（bypass / frozen 两档分别打印 `requires_grad=True` 的实参量）；
  2. 旁路**初始为零**时：(a) 编码器输出与无旁路逐位一致（max|Δ| 必须 = 0）；(b) F 档（旁路在场但冻结在零）
     的 negation 指标 ≈ frozen 基线（文档 0.6383；在 A′ 基座上直接对，在 base_encoder 上以 F 自身为配对基线）；
  3. 训练后旁路 B（及 A）的范数/标准差必须明显偏离初值（B 初值 std = 0；若仍为 0 ⇒ 空壳，判失败）。
- **判定规则**：P1 ∧ P2（两 seed）∧ 三自检全过 ⇒ 假设成立；P2 单 seed 过 ⇒ 只报方向；
  P2 两 seed 都 < 0.87 ⇒ 假设在该配置下不成立（如实报，不改判据）。
- **失败后的探索臂（预登记，不参与判定）**：r=32 一档（seed 42）+ 加倍步数一档，只标 "exploratory"。

## 运行清单（跑前冻结）

| 臂 | 基座 | 模式 | seed | 用途 |
|---|---|---|---|---|
| B-base s42/s43 | base_encoder | bypass r16 | 42/43 | 主结果（P2） |
| F-base s42/s43 | base_encoder | frozen（旁路在场、零、不训） | 42/43 | 配对基线 + P3 对照 + 自检 2b |
| F-A′ s42 | A′ 基座 | frozen | 42 | 校验配方复现文档 0.6383 |
| B-A′ s42/s43 | A′ 基座 | bypass r16 | 42/43 | 与文档 frozen/joint 同基座口径的对照 |
| eval_old_cards | base_encoder + checkpoints/cards/ | A/B/C 三条件 | — | P1 + 逐位一致性 + 敏感性对照 |

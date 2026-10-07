# P16 — 端到端对话路径的 D1 实测（只诊断）REPORT

PREREG `experiments/e2e_d1/PREREG.md`（mtime **2026-10-07 13:48:00 +08**，早于首跑 13:49；判据 E0–E4、三选一跑前写死、跑完未改）。
只读 `src/`、`checkpoints/`、`experiments/`（除本目录）；只写 `experiments/e2e_d1/`、`logs/`；纯 CPU、不训练、不改卡、不换 `DEFAULT_ATTACH`/`DEFAULT_BASE`、不动 `src/`、无 git 写操作。
> **边界句（必读）**：**本单元只诊断、不修复。** **不回答「该不该换成真基座」** —— 决策留给 Lead。

**真实入口（写明）**：`dtseek.tasks.dialogue.respond(engine, text, type_="plain")`；其内部 `run_cards → engine.predict` 的**分段解码**（`_run_segment`）。**本单元不用 `evaluate_task`**，两口径不并表、不互换（E2）。

## 1. W1 打印结果（本次打印，未引报告）

| 对象 | 打印值 |
|---|---|
| 源码字面 | `engine.DEFAULT_BASE = checkpoints/base_encoder.pt`；`dialogue.DEFAULT_ATTACH = [experiments/compose_ops/artifacts/cards/negation.pt]` |
| 引擎 A（线上核）实际加载 | `base_path=checkpoints/base_encoder.pt`，`doc_sha256 = dc27db5337d16f18…`（**自校验 = CATALOG 值：True**），device=cpu |
| 引擎 B（来源核）实际加载 | `base_path=experiments/compose_ops/artifacts/base_encoder.pt`，`doc_sha256 = 801657fb40a803a2…` |
| 两引擎差异 | `A==B: False`（**只换 base_path**）；**卡文件完全相同**：negation `ba1a80f9…`、person `2364296b…`、pronoun `204b3fb2…`、relation `4ddb0f42…`、sentiment `2ae22bcd…` |
| Plan 是否含 negation | 初始状态 `admissible_actions = ['sentiment','negation']` ⇒ **在授权集内 = True**；`plan[]` 行字面含 "negation" = 0（`describe()` 只打**第一个**授权动作，不是"没调"） |
| 真实调用 | `cards_run` 含 negation = **128/128（1.0000）**，两引擎同 |

样例（A）：输入 `不要让孩子挣扎或摔倒` → `cards_run=['sentiment','negation']`、`terminal=generate`、`text=你这句话里的『要让孩子挣扎或摔倒』表达了愤怒；并且有否定成分『不要让』。`

## 2. W0 主表（端到端 respond；两引擎只换基座；Δ = A 线上 − B 来源核，负 = 线上核掉分）

**输入文件清单（与 prod_card_audit 同口径）**：`experiments/capability_map/cache/{pronoun,relation,sentiment,person,negation}_ordered_s{42,43}.pkl` 共 10 个 → `deepcopy` → `Random(S).shuffle` → `n_val=max(200,len//10)` → 前 `n_val` = eval_S；实际 n = pronoun 600、relation 720、sentiment 3200、person 600、negation 600（×2 seed，每引擎 11440 条）。

**2a 逐条相同比例（M1/M5，全部 5 能力）**

| cell | n | **same_record（整条记录全字段相同）** | same_text | same_kind | negAnchorSame | moodAnchorSame |
|---|---|---|---|---|---|---|
| pronoun s42 / s43 | 600 | **.0283±.0068 / .0300±.0070** | .0317/.0317 | .9250/.9133 | .5950/.5883 | .0683/.0833 |
| relation s42 / s43 | 720 | **.0333±.0067 / .0417±.0074** | .0333/.0417 | .9694/.9764 | .4167/.4583 | .0806/.0972 |
| sentiment s42 / s43 | 3200 | **.3103±.0082 / .3219±.0083** | .3119/.3237 | .9284/.9328 | .5506/.5606 | .4816/.4956 |
| person s42 / s43 | 600 | **.0167±.0052 / .0217±.0059** | .0167/.0217 | .9967/.9917 | .2767/.2800 | .0483/.0533 |
| **negation s42 / s43** | 600 | **.0383±.0078 / .0350±.0075** | .0433/.0350 | .9750/.9717 | **.3217/.3683** | .1150/.1083 |

⇒ **两基座下 68%–98% 的样本整条回复不同**（same 最低 person .017、最高 sentiment .31）；`kind`（text/reject）却有 .91–.99 相同 ⇒ **差异主要在"说了什么"，不在"开不开口"**。

**2b negation 专属指标（只读 negation 卡证据 ⇒ 隔离该卡；真值 = eval_S 首 span）**

| seed | 指标 | A 线上 | B 来源核 | **配对 Δ ± SE** | n | 判 |
|---|---|---|---|---|---|---|
| 42 | M_detect（有无否定证据 == 真值） | .4883 | .9950 | **−.5067 ± .0208** | 600 | SIG |
| 42 | M_cls（首发射类别，同 A0 `cls_acc` 分母） | **.2801** | 1.0000 | **−.7199 ± .0223** | 407 | SIG |
| 42 | M_span（逐字子串 == 真值子串） | .0393 | .9803 | **−.9410 ± .0117** | 407 | SIG |
| 43 | M_detect | .5167 | 1.0000 | **−.4833 ± .0204** | 600 | SIG |
| 43 | M_cls | **.2906** | 1.0000 | **−.7094 ± .0233** | 382 | SIG |
| 43 | M_span | .0576 | .9921 | **−.9346 ± .0127** | 382 | SIG |

**混淆拆解（把 Δ 拆成漏检 vs 误报）**：s42 A = tp114/fn293/fp14/tn179，B = tp407/fn0/fp3/tn190；s43 A = tp111/fn271/fp19/tn199，B = tp382/fn0/fp0/tn218。
⇒ **线上核的错几乎全是"漏检"**（s42 293/407 真值样本无 negation 证据；分歧方向 `real_B_has_A_not=293, real_A_has_B_not=0`），不是乱报。

**口径不混（E2，如实写）**：以上是**端到端口径**。`prod_card_audit` 的 `evaluate_task` exact **.9667→.3233** 属**另一口径**，本单元**未在端到端复算 exact**（`respond` 只暴露每卡**首发射**，拿不到整句切片集合）⇒ 该数字**未测、不移植**。可交叉印证的一点（实测）：端到端 M_cls = **.2801/.2906** 与 A0 `cls_acc` **数值逐位相同**（同 eval_S、同首切片判据），但**仍不并表**。

## 3. W3 影响面 + W4 确定性

- **W3**：`negation` 卡在默认路径（`example_dialogue` 建引擎 + `type_="plain"`）被调用 **22880/22880 = 1.0000**（5 能力 × 2 seed × 2 引擎，全部 `cards_run` 含 negation）—— **不是"极少"，是每条 plain 输入都调**（原因：`DIALOGUE_REQUIRED_BITS["plain"]=("mood","neg")`，`run_cards` 一次把两个授权位的卡都跑）。出口分布：A `text 9918 / reject 1522`（reject 13.30%），B `text 10175 / reject 1265`（11.06%）。
- **W4 确定性**：每 (能力, seed, 引擎) 前 64 条连跑两遍、整条记录逐字比对 ⇒ **不一致 0/1280 = 0**（硬门槛过）。

## 4. W0–W4 逐条实测 + 判定 + 机制

| # | 实测结果 | 证据强度 |
|---|---|---|
| **W1** | A=`dc27db53…` / B=`801657fb…`，卡文件逐字相同；negation 在授权集内且 **128/128 真被调用** | 打印，确定性 |
| **W0** | M_detect/cls/span 两 seed 全部 **同号负、\|Δ\|>2SE**（Δ cls −.7199±.0223 / −.7094±.0233）；整条回复相同率仅 .017–.32 | **配对 + 2 seed** |
| **W2** | 端到端**未出现**"不显著"分支；`evaluate_task` 与端到端**分表分口径**，exact 未在端到端复算（如实标未测） | 口径纪律 |
| **W3** | 触发 **22880/22880 = 1.0000** ⇒ 影响面 = 全部 plain 输入 | 5 能力 × 2 seed |
| **W4** | 不一致 **0/1280** | 确定性 |

**判定（三选一，PREREG 写死的分支）：`端到端也显著变差（影响真实路径成立）`** —— M_detect/M_cls/M_span 两 seed 全部显著变差，且 M5 显示 negation 卡首发射在两引擎下 **63.2%/67.8% 不一致**（negAnchorSame .3217/.3683 ⇒ 差异确实经过 negation 卡）。证据强度 = **配对 + 2 seed**。

**机制（实测支撑的最小解释）**：只换基座、卡一个不换，negation 卡的首切片分类从 1.0000 崩到 .2801，**错法是漏检**（fn 293/407，反向 `A_has_B_not=0`）⇒ 解码头读到的是另一套表征、直接判"无否定"，于是 `respond` 少拼一段"并且有否定成分『…』"，整条回复随之不同。
**归因边界（必须一起读）**：`same_record` 的差异**不只**来自 negation —— `moodAnchorSame` 只有 .048–.50，说明 **sentiment 卡在错配基座上也大量变**（与 A0 的反向正对照一致：四默认卡放外来核同样崩）；**M2–M4 只读 negation 卡证据**，因此"negation 卡掉多少"这一问是**隔离**的，而"整条回复差多少"是**全体卡共同效应**。
**最小复现（纯 CPU，~10 s，只打印）**：
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uv run python experiments/e2e_d1/w1_load.py`；`… uv run python experiments/e2e_d1/w0_e2e.py --caps negation --limit 32` → `M_cls A(线上)=0.2963 vs B(来源核)=1.0000`。

## 5. 原始命令 / 日志 / 产物 / 遗留

**命令**（逐个单跑，退出码 **0 / 0**）：`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uv run python experiments/e2e_d1/{w1_load,w0_e2e}.py`（W0 全量 185 s）。
**产物**：`experiments/e2e_d1/{PREREG.md,REPORT.md,common.py,w1_load.py,w0_e2e.py}`；`results/{w1_load,w0_e2e,w0_smoke}.json`；`logs/e2e_w1.log`、`logs/e2e_w0.log`（项目根 `logs/`）。

**实测（本目录跑出来的）**：§1–§4 全部数字；W3 = 1.0000；W4 = 0；混淆表与分歧方向。
**推断（未测，不得当结论）**
1. **端到端 exact（整句切片集合）未测** —— `respond` 只暴露首发射；A0 的 .9667→.3233 是 `evaluate_task` 口径，**没有**在端到端复现或推翻。
2. **其它四张卡在端到端的逐卡正确率未测**（只测了首发射两引擎一致性）；`same_record` 的归因只到"与 sentiment 卡也变"这一层，**未拆出各卡贡献占比**。
3. **换核的线上净收益 / OOD**：未测，本单元不回答。
4. 本单元**不回答「该不该换成真基座」**（决策留给 Lead）。

**遗留**：`respond` 只取每卡发射序第 1 个切片（模板单槽）⇒ 多标记句的第 2+ 个否定标记在回复层不可见，本单元未评估该损失；`type≠plain` 的其它声明类型本轮不支持、全拒答，未覆盖。

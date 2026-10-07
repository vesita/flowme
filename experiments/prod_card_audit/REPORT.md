# P15 — 在产 D1 缺陷实测 + 生产四卡免费规则对照（REPORT）

PREREG `experiments/prod_card_audit/PREREG.md`（mtime 2026-10-07 13:27:46 +08，**早于首跑 13:29**）；判据 F1–F5、三选一判定、A2 电池 8 条跑前写死，跑完未改。
只读 `src/`、`checkpoints/`、`experiments/`（除本目录）；只写 `experiments/prod_card_audit/` + `logs/`；纯 CPU、不训练、不改卡、不动 `src/`、无 git 写操作。

> **边界句（必读）**：**本单元只诊断，不修复。** 不改卡、不换卡、不改 `dialogue.py`/`engine.py`；
> **且不回答「该不该换成真基座」** —— 那是决策，留给 Lead。

## 1. A1：覆盖缺口钉成事实（全部本次打印，未引报告）

sha 方案 = state_dict **原始键序**拼接张量字节（自校验：线上核复现 `CATALOG.md:11` 的 `dc27db53…` ⇒ 过）。

| 对象 | 路径 | 文件 sha256（前 16） | **doc 张量 sha256（前 16）** |
|---|---|---|---|
| 线上默认 `engine.DEFAULT_BASE` | `checkpoints/base_encoder.pt` | `dcf237ae99cef7d3` | **`dc27db5337d16f18`** |
| **测试侧实际加载**（`tests/test_dialogue.py` fixture：`MultiTaskEngine()` + `attach(DEFAULT_ATTACH)`） | 同上（`base_path` 打印为 `checkpoints/base_encoder.pt`） | 同上 | **`dc27db5337d16f18`** |
| 线上默认 `dialogue.DEFAULT_ATTACH` 卡 | `experiments/compose_ops/artifacts/cards/negation.pt` | `ba1a80f9af81b1a2` | 无（v1 卡不带 encoder） |
| 该卡来源核（`train_args.split_from = checkpoints/arm_neg5_seed42.pt`） | `checkpoints/arm_neg5_seed42.pt` | — | **`801657fb40a803a2`** |
| 同源另一份 | `experiments/compose_ops/artifacts/base_encoder.pt` | — | **`801657fb40a803a2`** |
| 四默认卡来源核 | `checkpoints/multitask_v2_dtseek.pt` | — | **`dc27db5337d16f18`**（与线上 **neq=0 / max\|Δ\|=0.0**） |

**打印结论（用输出说话）**
- 测试侧基座 **==** 线上基座：`online_vs_test_same = true`（同文件、同 sha）。
- DEFAULT_ATTACH 卡的来源核 **≠** 线上基座：`card_provenance_vs_online_same = false`（`801657fb…` vs `dc27db53…`，26 张量全不同）。
- **覆盖缺口**：两侧基座同 sha 只说明「测试跑的是线上核」，**不说明「卡挂在线上核上不掉分」**；现有测试**从未把该卡放回自己的来源核上对照** ⇒ 后者 = 本单元 A0。

## 2. A0 主表（两组合 × 两基座 × 2 seed；Δ = **线上核 − 卡自带核**，负 = 在产掉分）

评测样本 = `experiments/capability_map/cache/{cap}_ordered_s{42,43}.pkl`（10 个文件，`eval_S` 与 `capability_map` 逐字同一公式）；
指标走 `runtime.evaluate_task`（**口径自检：与线上 `evaluate_task` 同 cell `cls_acc/exact/span_hit/bg_fp/n_cls/n_bg` 全等 = True**）。
分母：pronoun 409/424、relation 514/526、sentiment 2370/2373、person 417/410、negation 407/382（cls）；exact 为全量 600/720/3200/600/600。

| 组合 | 卡 | seed | cls 自带→线上 | **Δ cls ± SE** | exact 自带→线上 | **Δ exact ± SE** | 判（配对，两 seed 同号且 \|Δ\|>2SE） |
|---|---|---|---|---|---|---|---|
| 四卡+negation | pronoun | 42 | .8704 → .9878 | **+.0800 ± .0118** SIG | .6983 → .9217 | **+.2233 ± .0184** SIG | 对照：在外来核上显著更差 |
| 四卡+negation | pronoun | 43 | .8608 → .9929 | **+.0933 ± .0123** SIG | .6700 → .9600 | **+.2900 ± .0188** SIG | 同上 |
| 四卡+negation | relation | 42 | .3599 → .9611 | **+.4292 ± .0189** SIG | .1583 → .7306 | **+.5722 ± .0243** SIG | 同上 |
| 四卡+negation | relation | 43 | .3992 → .9658 | **+.4139 ± .0191** SIG | .1792 → .7417 | **+.5625 ± .0248** SIG | 同上 |
| 四卡+negation | sentiment | 42 | .4489 → .7662 | **+.2350 ± .0093** SIG | .4597 → .7528 | **+.2931 ± .0095** SIG | 同上 |
| 四卡+negation | sentiment | 43 | .4728 → .7729 | **+.2225 ± .0093** SIG | .4850 → .7716 | **+.2866 ± .0095** SIG | 同上 |
| 四卡+negation | person | 42 | 1.0000 → 1.0000 | +.0000 ± .0000 | .3050 → .3183 | **+.0133 ± .0047** SIG | cls **天花板**（首提及 id 恒 1）⇒ 检不出 |
| 四卡+negation | person | 43 | 1.0000 → 1.0000 | +.0000 ± .0000 | .3167 → .3217 | +.0050 ± .0029 | 同上 |
| **两组合（逐位相同）** | **negation** | 42 | **1.0000 → 0.2801** | **−.4883 ± .0204 SIG** | **.9667 → .3233** | **−.6433 ± .0203 SIG** | **显著变差** |
| **两组合（逐位相同）** | **negation** | 43 | **1.0000 → 0.2906** | **−.4517 ± .0203 SIG** | **.9833 → .3667** | **−.6167 ± .0199 SIG** | **显著变差** |

- **两组合对 negation 逐位相同**（`cls_ok/exact_ok/cls_real/指标` 全等，实测比对 = True）⇒ 掉分与「挂了几张卡」无关。
- **F2 检验力闸门：过（3/4）** —— 四默认卡在 E_own（外来核）下 pronoun/relation/sentiment **两 seed、两口径全部显著变差**；person 因 cls 天花板（1.0000/1.0000，首提及 id 恒 1）检不出 ⇒ 符合 PREREG §2 写死的「≥3/4」。

## 3. A2 — 生产四卡 `max_naive`（地板 8 条；`fit=train_S`、`lookup` 无 min_support、未见键回退 train 多数）

目标 = `labels[:,0]` 且只取 `t_labels>0`（与 `cls_acc` 同分母）；只用原始 `text`；卡指标 = A0 线上核口径。

| 卡 | seed | majority | len_bucket | first_char | last_char | punct_pattern | **lex_rule（词表字面）** | lex_first_key | **n_cue（计数，D2）** | **max_naive** | 卡 cls | **卡−地板** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pronoun | 42 | .3814 | .6406 | .7751 | .5795 | .6308 | **1.0000** | 1.0000 | .4817 | **1.0000** | .9878 | **−.0122** |
| pronoun | 43 | .3939 | .6486 | .7830 | .5920 | .6439 | **1.0000** | 1.0000 | .5024 | **1.0000** | .9929 | **−.0071** |
| relation | 42 | .6187 | .6187 | .6148 | .6187 | .6187 | **.9844** | .9844 | .6187 | **.9844** | .9611 | **−.0233** |
| relation | 43 | .6046 | .6046 | .6065 | .6046 | .5970 | **.9658** | .9658 | .6046 | **.9658** | .9658 | **−.000000**（−4.7e−7，实质并列） |
| sentiment | 42 | .3257 | .3473 | .4198 | .4612 | .3443 | **1.0000** | 1.0000 | .3553 | **1.0000** | .7662 | **−.2338** |
| sentiment | 43 | .3232 | .3460 | .4193 | .4501 | .3434 | **1.0000** | 1.0000 | .3485 | **1.0000** | .7729 | **−.2271** |
| person | 42 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | **0.0000** |
| person | 43 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | **0.0000** |

- 地板命中者：pronoun/sentiment 由 `lex_rule=1.0000`（词表＝标签定义来源，D4 构造性满分）、relation 由 `lex_rule`（.9844/.9658）、person 由 `majority=1.0000`（`label_dist_eval = {1: 417}`，首提及 id 恒 1）。
- cue 表来源（全部取自 `src/dtseek/tasks/builtin/*`，未新造）：pronoun=`PRONOUN_MAP`、sentiment=`EMOTION_KEYWORDS`、relation=`SYNONYM_PAIRS+ANTONYM_PAIRS`、person=`_PERSON_VOCAB`（类别按「首提及 id 恒 1」的结构事实出）；查表 `seen_rate` 均 ≥.97，`min_support=30` 变体另存 `disclosed_min_support30`（不计入地板）。

## 4. A0–A4 逐条实测 + 判定 + 机制

| # | 实测结果 | 证据强度 |
|---|---|---|
| **A1** | 测试侧基座 sha == 线上基座 sha（`dc27db53…`）；`DEFAULT_ATTACH` 卡来源核 `801657fb…` ≠ 线上 ⇒ **张量不一致（实测）** | 打印，确定性 |
| **A0** | negation：线上核 cls **1.0000 → .2801/.2906**、exact **.9667/.9833 → .3233/.3667**；Δ cls **−.4883±.0204 / −.4517±.0203**，两组合两 seed 全 SIG 同号 | **配对 + 2 seed** |
| **A2** | 四卡 `max_naive` = 1.0000 / .9844 / 1.0000 / 1.0000；卡−地板 **全部 ≤ 0** | 2 seed |
| **A3** | pronoun `−.0122/−.0071`、relation `−.0233/−0.0000`、sentiment `−.2338/−.2271`、person `0.0000/0.0000` ⇒ **四张卡全部「未超过免费规则」**（措辞未改写） | 2 seed |
| **A4** | 24 个 cell（2 组合 × 2 基座 × 5 卡 × 2 seed，各跑两遍）逐样本 `cls/real/exact` **不一致 = 0**，指标全等 | 确定性 |

**判定（三选一，PREREG 写死的分支）：`在产缺陷成立（显著变差）`** —— negation 在两组合 × 两 seed **全部**满足「两 seed 同号且 |Δ| > 2SE」，
且 F2 检验力闸门（3/4 对照显著）已过；证据强度 = **配对 2 seed**。

**机制（实测支撑的最小解释）**：该卡由可训核的联合臂 `arm_neg5_seed42` 拆出，**解码头与那份核共同适配**；
换到线上核（`dc27db53…`）后同一解码头读到的是另一套表征 ⇒ 首切片分类从 1.0000 崩到 .28。
**方向对称的正对照**：四默认卡放到外来核（E_own）上同样显著崩（relation .9611→.3599、sentiment .7662→.4489、pronoun .9878→.8704）
⇒ 现象是「核错配」本身、非单卡特例；person 因天花板检不出。**最小复现**（4 s、纯 CPU、只打印）：
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uv run python experiments/prod_card_audit/a0_bases.py --smoke` → `E_online negation s42 cls=0.2857` vs `E_own negation s42 cls=1.0000`。

## 5. 原始命令 / 日志 / 产物 / 遗留

**命令**（全部 `uv run python`、CPU，逐个单跑）：`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uv run python experiments/prod_card_audit/{a1_shas,a0_bases,a2_floor,a4_determinism}.py`
→ 四条退出码 **0 / 0 / 0 / 0**；A0 全量 29 s，A4 = 24 cell × 2 遍。

**产物**：`experiments/prod_card_audit/{PREREG.md,REPORT.md,a1_shas.py,a0_bases.py,a2_floor.py,a4_determinism.py,common.py}`；
`results/{a1_shas,a0_bases,a0_smoke,a2_floor,a4_determinism}.json`；`logs/{a0_bases,a2_floor,a4_determinism}.log`（A1 输出见 `results/a1_shas.json` 与 §1）。

**实测（本次跑出来的）**：§1–§4 全部数字；`combo_full` 与 `combo_neg` 的 negation 逐位相同；`evaluate_task` 口径自检全等；A4 不一致 = 0。

**推断（未测，不得当结论）**
1. 「换成真基座/换回来源核的线上净收益」——**未测**（本单元只测张量差与指标差）。
2. 「该卡在**端到端对话路径**（`engine.predict` 分段解码、`dialogue.respond`）上掉多少」——**未测**：本单元指标走 `evaluate_task`（整段截断口径，与 `capability_map` 对齐），**分段解码路径未覆盖**。
3. 「该卡在 OOD / 其它评测集上是否同样崩」——未测（只测 `eval_S` 两 seed）。
4. 本单元**不回答「该不该换成真基座」**（决策留给 Lead）。

**未实现的（明说）**
- A2 电池 8 条中 `fw_decision_list` / `tree_depth2` / `tree_depth4` **未实现**（需虚词决策表与句法树，本单元无句法分析器）⇒ `max_naive(8)` 是**完备地板的下界**：
  `卡 ≤ 下界` ⇒「未超过免费规则」**安全**；反向（卡 > 下界）**不足以**说超过完备地板 —— 本次四卡**没有出现**该情形。
- `lex_rule` 的 `min_support=30` 变体只作披露、未计入地板（PREREG 写死的查表口径是 `free_rule_floor::lookup`）。

**遗留**：`CATALOG.md` §3.4 剩余 95 个「未对照卡单元」本单元只补了生产四卡 + 在用 negation 路径；
`negation` 卡自身在四组合下的**免费规则对照**未做（其目标是闭集标记检测，`NEG_MARKERS` 字面规则未跑）——如实记为未做。
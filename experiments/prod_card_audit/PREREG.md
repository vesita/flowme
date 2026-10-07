# PREREG — P15 在产 D1 缺陷实测 + 生产四卡免费规则对照

> **本文件 mtime 必须早于本目录任何评测的首次执行**；判据 F0–F5 与三选一判定在看到本单元任何输出之前写死，跑完不改。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 本文件人为规定。

## 0. 问题与边界（写死）

两件事，都来自 `experiments/card_catalog/CATALOG.md` 的实测发现：

- **A0（主）**：线上默认挂载的 `dialogue.py:105 DEFAULT_ATTACH = experiments/compose_ops/artifacts/cards/negation.pt`
  的**来源核**（`split_from=checkpoints/arm_neg5_seed42.pt`，与 `experiments/compose_ops/artifacts/base_encoder.pt` 同一份）
  **≠** 线上基座 `engine.py:38 DEFAULT_BASE = checkpoints/base_encoder.pt`。
  **张量不一致是已实测事实；「掉不掉分」未测（推断）** ⇒ 本单元实测。
- **A2**：生产四卡（pronoun / relation / sentiment / person）在 `CATALOG.md` T2 里
  **免费规则对照 = 未验证**（从没跟平凡查表比过）⇒ 本单元补 `max_naive` 完整电池对照。

**边界句（写死，必须进报告）**：**本单元只诊断，不修复**；不改任何卡、不换卡、不动 `src/`
（`dialogue.py` / `engine.py` 一律只读）、不训练、不启 GPU；**且不回答「该不该换成真基座」**——那是决策，留给 Lead。

**硬约束**：只读 `src/`、`checkpoints/`、`experiments/`（除本目录）、`training/`、`dev-notes/`、`methodology/`；
只写 `experiments/prod_card_audit/`、`logs/`、`/tmp`；禁止 `git commit/stash/checkout/restore/clean`；
纯 CPU（`CUDA_VISIBLE_DEVICES=""`）；一律 `uv run python`。

## 1. A1：覆盖缺口先钉成事实（跑前只规定打印什么，不规定结论）

打印下面每一项的 **sha256（逐张量按 state_dict 原始键序拼接张量字节）**，以及**文件级 sha256**：

1. `dialogue.DEFAULT_ATTACH` 与 `engine.DEFAULT_BASE` 的字面值（源码打印，不引报告）；
2. 线上默认路径**实际加载**的基座（`MultiTaskEngine()` 的 `base_path` → `doc_encoder` 张量 sha）；
3. `DEFAULT_ATTACH` 那张卡：文件 sha、`train_args.split_from`、沿 split_from 解析出的**来源核**张量 sha、
   以及 `experiments/compose_ops/artifacts/base_encoder.pt` 的张量 sha；
4. `tests/test_dialogue.py` 的 engine fixture（`MultiTaskEngine()` + `attach(DEFAULT_ATTACH)`）**实际加载**的基座张量 sha。

**判定**：2 与 4 是否相同 —— **用打印结果说话**，不许只引 `CATALOG.md`。
（张量 sha 方案自校验：线上基座必须复现出 `CATALOG.md:11` 的 `dc27db5337d16f18…`，否则先修 sha 函数、不改判据。）

## 2. A0：两个引擎 × 两种卡组合 × 2 seed（主判据）

**两个引擎（写死）**

| 引擎 | `base_path` | 含义 |
|---|---|---|
| **E_own** | `experiments/compose_ops/artifacts/base_encoder.pt` | **卡自带基座**（生产 `DEFAULT_ATTACH` 那张卡的来源核；= `arm_neg5_seed42` 的核） |
| **E_online** | `checkpoints/base_encoder.pt` | **真线上基座**（`DEFAULT_BASE`） |

⚠️ **口径判断（跑前写清）**：生产四卡 `checkpoints/cards/*.pt` 的 `split_from = checkpoints/multitask_v2_dtseek.pt`，
其核与线上 `base_encoder.pt` **逐位相同**（`CATALOG.md` §5-4 已实测 neq=0，本单元再打印一次复核）。
因此对这四张卡而言 **E_online 就是它们的自带核**、**E_own 是外来核** ⇒ 这四张卡是**反向错配对照臂**（正对照）：
若检验对「核错配」敏感，它们在 E_own 下应显著变差。**negation 才是被检对象**（自带核 = E_own，线上 = E_online）。

**两种卡组合（写死，都跑）**

- `combo_full` = 默认四卡（`checkpoints/cards/{pronoun,relation,sentiment,person}.pt`）+ `DEFAULT_ATTACH` 的 negation；
- `combo_neg` = 只挂 negation（`experiments/compose_ops/artifacts/cards/negation.pt`）。
两组合都用 `MultiTaskEngine(base_path=…, auto_attach=…)` 的**真实加载路径**建引擎（A1 的打印即出自该路径）。

**评测样本（写死，与 `capability_map` / `core_probe` 同口径）**
`experiments/capability_map/cache/{cap}_ordered_s{S}.pkl` → `copy.deepcopy` → `random.Random(S).shuffle`
→ `n_val = max(200, len//10)` → 前 `n_val` = `eval_S`，其余 = `train_S`；
S ∈ {42, 43}（**两个不同评测子集 = 2 重复**）。文件清单：5 个能力 × 2 seed 共 10 个 pkl，路径全部打进产物。
指标口径走 `dtseek.tasks.runtime.evaluate_task`（与 `capability_map/eval_cards.py` 同一条）：
**`cls_acc`（分母 = 首切片标签 > 0，与 `real = t_labels > 0` 逐字相同）为主，`exact_match` 为辅**，另报 `span_hit` / `bg_fp`。

**配对 Δ ± SE（写死）**
逐样本 0/1 正确性配对：`d_i = correct(E_online) − correct(E_own)`，`Δ = mean(d)`，
`SE = std(d, ddof=1)/sqrt(n)`（配对口径，与 `free_rule_floor` F1 同一套）。
Δ < 0 ⇒ **在产（线上核）掉分**。`cls_acc` 在 `real` 子集上配对，`exact_match` 在全量样本上配对。

**F1 判据（写死，显著性门槛）**
`显著变差` ⇔ **两 seed 同号** 且 **|Δ| > 2×SE**（两个 seed 都要）。
`显著变好` ⇔ 两 seed 同号且 Δ > 2SE。其余（异号 / 落在 2SE 内 / 无法定号）= 未测出显著差异。

**F2 检验力闸门（写死，不许硬选）**
反向错配对照臂（生产四卡，在 E_own 下）**至少 3/4 张显著变差** ⇒ 检验对「核错配」敏感，F1 可判；
若对照臂 **4/4 都 |Δ| ≤ 2SE** ⇒ **检验力不可判 ⇒ 判定「证据不足」**（不许因为"没测出差异"就写"没有效果"）。

**三选一判定（写死）**
1. **在产缺陷成立（显著变差）**：negation 在 `combo_full` 与 `combo_neg` 两组合下、两 seed 均 `显著变差`，且 F2 闸门过；
2. **张量不一致但无显著影响**：F2 闸门过，且 negation 两组合两 seed 全部落在 `未测出显著差异`（**只许这么写**，不许写"没有效果"）；
3. **证据不足**：两 seed 异号 / 两组合结论不一致 / F2 闸门未过。

## 3. A2：生产四卡的 `max_naive` 完整电池（口径与 `free_rule_floor` 一致）

**目标（写死）**：与卡的 `cls_acc` **同分母**的目标 = `labels[:, 0]`，只取 `t_labels > 0` 的样本（与 `real` 逐字相同）。
**只用原始 `text`**：不读 `spans / label / category / pair_key` 等标注字段（防泄露，D5 同类）。

**查表口径（逐字复用 `experiments/free_rule_floor/rules.py::lookup`）**：`fit = train_S`、`eval = eval_S`、
**无 min_support、未见键回退 train 全局多数**、报 `seen_rate`。

**电池 8 条（计入 `max_naive`，跑前写死）**

| # | 组 | 规则 | 说明 |
|---|---|---|---|
| 1 | majority | `majority` | train 多数类 |
| 2–4 | 表面/字面 | `len_bucket`(`len(text)//6`) / `first_char` / `last_char` / `punct_pattern` | 逐字对齐 `two_channel_head::naive_skeleton` 的同名四条口径 |
| 5 | **词表/字面** | `lex_rule` | 任务自有闭集词表在 text 上**长词优先**取首命中 ⇒ 类别；无命中 ⇒ train 全局多数（纯字面，不拟合） |
| 6 | **词表/字面（查表）** | `lex_first_key` | 首命中词条字符串 → train 多数类（拟合式，报 seen_rate） |
| 7 | **计数类（D2）** | `n_cue` | text 中闭集 cue 出现**次数**（未分桶）→ train 多数类 |

四卡的闭集 cue 表（写死，全部取自 `src/dtseek/tasks/builtin/*/dataset.py` 的既有闭集，不新造）：
`pronoun` ← `PRONOUN_MAP`；`sentiment` ← `EMOTION_KEYWORDS`；`relation` ← `relation.lexicon` 的
`SYNONYM_PAIRS ∪ ANTONYM_PAIRS`（键 = 命中的那一对）；`person` ← `PRONOUN_MAP` ∪ 生成器称谓表（读不到称谓表就只用 PRONOUN_MAP 并如实记）。

**未实现的（明说，不许含糊）**：旧 8 条里的 `fw_decision_list` / `tree_depth2` / `tree_depth4` 需要虚词决策表与句法树，
本单元**没有句法分析器 ⇒ 未实现**。因此 `max_naive(7 条)` 是**完整地板的下界**：
`卡 < max_naive(7)` ⇒ 必然 `卡 < 完备地板` ⇒ 结论「未超过免费规则」**安全**；
`卡 > max_naive(7)` ⇒ **不足以**说超过完备地板（报告里必须这么写）。

**单列披露但不计入**：`lex_first_rule` 的「按定义满分」情形（标签本身由词表定义 ⇒ rule 按定义 1.0）
必须如实打印；`core_probe` 的 D4 纪律同样适用 —— 若 `lex_rule ≥ 卡`，写「**未超过免费规则**」，不改措辞。

**F3（写死）**：逐卡报 majority / `lex_rule` / `n_cue` / `max_naive(7)` 与 **卡 − 地板**，2 seed 都报；
任一卡 `卡 < 地板` ⇒ **如实写「未超过免费规则」**（A3，不得改写措辞）。

## 4. A4：确定性

同一输入（每个能力 `eval_S` 前 64 条 × 2 引擎 × 2 组合）**连跑两遍**，
逐样本 `cls` 预测与 `exact` 判定逐一比对，**不一致条数必须 = 0**（不一致 = 0 是硬门槛，≠ 0 则该臂作废）。

## 5. 运行

`CUDA_VISIBLE_DEVICES="" uv run python experiments/prod_card_audit/<script>.py`，CPU only，不训练、不启 GPU 任务、不杀他人进程。
**只写** `experiments/prod_card_audit/`、`logs/`、`/tmp`。

**停止边界**：做完 §验收口径 A0–A4 即停。**不要**：修卡 / 改线上配置 / 重训 / 改既有实验 / 动 `src/` /
写 `dev-notes/` / `methodology/` / 派子代理 / git 提交类操作。

## 6. 验收口径（与派单一致，写死）

| # | 判据 | 门槛 |
|---|---|---|
| A0（主） | 自带基座 vs 线上基座的指标差（默认四卡+negation / 仅 negation 两组合） | 报配对 Δ ± SE；显著变差 ⇒ 在产缺陷成立 |
| A1 | 覆盖缺口钉成事实 | 打印两侧实际加载的基座 sha256；相同/不同用输出说话 |
| A2 | 生产四卡的 `max_naive` | 逐卡 majority / 词表字面 / 计数类 / 完备地板 + 卡−地板 |
| A3 | 低于地板者 | 如实写「未超过免费规则」 |
| A4 | 确定性 | 不一致 = 0 |

**置信度**：区分实测/推断，标注证据强度（单 seed / 配对 / 2 seed）；
**不许**把"没测出差异"写成"没有效果"；**绝不允许空报告**（失败就贴失败输出尾部与退出码）。

# PREREG — P16 端到端对话路径的 D1 实测（只诊断）

> **本文件 mtime 必须早于本目录任何评测的首次执行**；判据 E0–E4 与三选一判定在看到本单元任何输出之前写死，跑完不改。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 本文件人为规定。

## 0. 问题与边界（写死）

`experiments/prod_card_audit/REPORT.md`（A0，已实测、**不重新论证**）：
在产 `negation` 卡（`dialogue.DEFAULT_ATTACH`，来源核 `801657fb…`）挂**线上核**（`dc27db53…`）
在 **`evaluate_task` 口径**（整段截断 + 逐样本 loader）下 exact **.9667 → .3233**、cls **1.0000 → .2801**，
两 seed 两组合全 SIG。**但那是 `evaluate_task` 口径，端到端对话路径未覆盖**（A0 报告 §5「推断/未测」第 2 条自己写了）。

本单元把这条**推到真实路径**：**`dialogue.respond`**（真实入口；其内部 `run_cards` → `engine.predict`
的**分段解码**，即 `_run_segment` 的逐步 argmax 解码 —— 与 `evaluate_task` 的 loader 整段口径**不同**）。
**回答：这个缺陷在真实对话里影响多大。**

**边界句（写死，必须进报告）**：**本单元只诊断、不修复**；不改卡、不换 `DEFAULT_ATTACH`/`DEFAULT_BASE`、
不动 `src/`、不改既有实验、不训练、不启 GPU、无 git 写操作；
**且不回答「该不该换成真基座」**——那是决策，留给 Lead。

**硬约束**：只读 `src/`、`checkpoints/`、`experiments/`（除本目录）、`training/`、`dev-notes/`、`methodology/`；
只写 `experiments/e2e_d1/`、`logs/`、`/tmp`；纯 CPU（`CUDA_VISIBLE_DEVICES=""` + 显式线程数）；一律 `uv run python`。

## 1. W1：端到端路径实际加载什么（跑前只规定打印什么）

用**与 `examples/example_dialogue.py` 默认路径逐字相同**的建引擎方式打印：

1. 源码字面值：`engine.DEFAULT_BASE`、`dialogue.DEFAULT_ATTACH`（import 打印，不引报告）；
2. 引擎 A（线上核）与引擎 B（来源核）**各自实际加载的基座**：
   `base_path` + 基座 `state_dict` **原始键序**张量 sha256（同 A1 方案，自校验 A 必须 = `dc27db53…`）；
3. 实际挂载的卡文件清单与**文件级 sha256**（四默认卡 + `DEFAULT_ATTACH` 的 negation）；
4. **negation 是否真的在 `Plan` 里 / 真的被调用**：对一批输入跑 `respond(type_="plain")`，
   打印第一条样本的 `plan[]` 全文与 `cards_run`，并报**触发比例** = `cards_run` 含 `negation` 的样本数 / 总数。

**判定用打印结果说话**（W3 会给全量触发比例）。

## 2. E0（主）：两种基座下跑端到端

**两个引擎（写死，卡集完全相同，只换基座）**

| 引擎 | `base_path` | 含义 |
|---|---|---|
| **A = E_online** | `checkpoints/base_encoder.pt`（= `DEFAULT_BASE`） | **线上核**（在产默认） |
| **B = E_own** | `experiments/compose_ops/artifacts/base_encoder.pt` | negation 卡的**来源核**（= `arm_neg5_seed42`） |

两引擎都走 `MultiTaskEngine(base_path=…)`（`auto_attach=True` ⇒ 四默认卡）+ `for p in DEFAULT_ATTACH: attach(p)`
（与 `examples/example_dialogue.py:54-56`、`tests/test_dialogue.py` fixture 同一写法）。
⚠️ 只换 `base_path`，**卡文件一个都不换**（这是本单元唯一自变量）。

**同一批输入（写死，与 `capability_map` / `prod_card_audit` 同口径，报文件清单）**
`experiments/capability_map/cache/{cap}_ordered_s{S}.pkl` → `copy.deepcopy` → `random.Random(S).shuffle`
→ `n_val = max(200, len//10)` → 前 `n_val` = `eval_S`（与 `capability_map/probe.py::split_of` 逐字相同）；
S ∈ {42, 43}；cap ∈ {pronoun, relation, sentiment, person, negation}（5 能力 × 2 seed = 10 个 pkl，路径打进产物）。
输入 = 样本原始 `text`；真值 = `sorted(spans, key=start)` 的首 span（与 `GenericTaskDataset.__getitem__` 同序）。

**真实入口（写死，选哪个写明）**：**`dtseek.tasks.dialogue.respond(engine, text, type_="plain")`**。
（它是端到端入口：声明 type → `turn_plan` 出计划 → 只调计划里的卡 → `run_cards` = `engine.predict`
**分段解码** → `compose_reply` 拼回复。本单元**不用** `evaluate_task`，两者口径不同、**不许混报**。）

**指标（复用什么写明，逐条口径）**

- **M1 逐条相同比例（主，全部 5 能力）**：两引擎整条 `respond` 记录 `==` 相等的样本比例
  （记录含 `kind/text/evidence/cards_run/plan/terminal/reason/type`，全字段逐字比较）；
  另拆 `same_text`（回复文本）与 `same_evidence`（evidence 列表）。
- **M2 `neg_detect`（negation 能力，全量分母）**：`ok = (回复里有 negation 证据) == (真值有否定切片)`。
- **M3 `neg_cls`（negation 能力，仅真值 `label>0` 子集，与 A0 `cls_acc` 同分母）**：
  取回复里 negation 卡的**首发射**切片 `class_name → class_id`（无证据 = 预测 0），与真值首 span 标签比。
- **M4 `neg_span`（同 M3 分母）**：该切片在文本中的**逐字子串** == 真值首 span 子串
  （用子串比而不是下标比，避开 `predict` 的 `strip()`/分段偏移）。
- **M5 逐卡证据一致性（归因用，5 能力）**：两引擎下 negation 卡 / sentiment 卡各自的首发射
  （子串 + class_name）相同的比例 —— 用来分辨差异来自哪张卡。

**配对 Δ ± SE（写死）**：逐样本 0/1，`d_i = ok(E_online) − ok(E_own)`，`Δ = mean(d)`，
`SE = std(d, ddof=1)/√n`（与 A0 / `free_rule_floor` F1 同一套）。**Δ < 0 ⇒ 线上核（在产）掉分**。
M1 是比例本身（Δ = same 比例 = 1 − 差异率），M2–M4 报 Δ ± SE。
**≥2 个评测子集（S=42, 43）**给出重复；S 之间 SE 独立报告，不合并。

**E1 显著性判据（写死）**
`显著变差` ⇔ **两 seed 同号（都 < 0）** 且 **|Δ| > 2×SE**（两个 seed 都要）。
`显著变好` ⇔ 两 seed 同号 > 0 且 > 2SE。其余 = **未测出显著差异**（**只许这么写**，不许写「没有效果」）。

**E2 口径不混（写死）**：本单元**只报端到端口径**；`evaluate_task` 的 .9667→.3233 只作为**引用的既有结论**，
不与本单元数字并列成同一张表、不互相替换。若端到端未显著 ⇒ 报告**必须**写
「`evaluate_task` 口径变差、端到端未复现」。

**三选一判定（写死，不许硬选）**
1. **端到端也显著变差（影响真实路径）**：M2/M3（negation 能力）两 seed 全部 `显著变差`，
   且 M5 显示 negation 卡的证据在两引擎下确实不一致（不一致率 > 0，证明差异真的经过 negation 卡）；
2. **端到端未复现（影响面小）**：M2/M3 两 seed 全部落在 `未测出显著差异`（**只许这么写**），
   且 W3 触发比例确实很小（< 10%）；若触发比例高但未显著 ⇒ **仍写 2，但必须同时写明「影响面不小、是没测出差异」**；
3. **证据不足**：两 seed 异号 / M2 与 M3 结论互相矛盾 / M5 不一致率 = 0（差异根本没经过 negation 卡，无法归因）/ W4 确定性不达标。

## 3. W3：影响面（写死）

报**默认路径里 `negation` 卡实际被调用的样本数 / 比例**（分母 = 全部 `respond(type_="plain")` 调用数，
按 能力 × seed × 引擎 分列）。默认路径 = `examples/example_dialogue.py` 的建引擎写法 + `type_="plain"`。
（`DIALOGUE_REQUIRED_BITS["plain"] = ("mood","neg")` ⇒ 推断上应当 100%，**必须打印实测值**。）

## 4. W4：确定性（写死）

每个 (能力, seed, 引擎) 取 `eval_S` **前 64 条**，同一输入连跑两遍，
整条 `respond` 记录逐字比对，**不一致条数必须 = 0**（≠ 0 则该臂作废 ⇒ 判「证据不足」）。

## 5. 运行

`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uv run python experiments/e2e_d1/<script>.py`，
CPU only、不训练、不启 GPU 任务、⚠️ 绝不杀他人进程。
**只写** `experiments/e2e_d1/`、`logs/`、`/tmp`。

**停止边界**：做完 §6 的 E0–E4 即停。**不要**：修复缺陷 / 改配置 / 重训 / 改既有实验 / 动 `src/` /
写 `dev-notes/`、`methodology/` / 派子代理 / git 提交类操作。

## 6. 验收口径（与派单一致，写死）

| # | 判据 | 门槛 |
|---|---|---|
| **E0（主）** | 端到端两基座下的输出/指标差 | 报**逐条相同比例** + **配对 Δ ± SE**；显著变差 ⇒ 影响真实路径成立 |
| **E1（W1）** | 端到端实际加载的基座/卡 sha + 是否真调用 negation | **打印结果说话** |
| **E2（W2）** | **口径不混** | 若端到端不显著 ⇒ 如实写「`evaluate_task` 口径变差、端到端未复现」 |
| **E3（W3）** | 影响面 | 报 negation 卡被调用的样本数 / 比例 |
| **E4（W4）** | 确定性 | 不一致 = 0 |

**置信度**：区分实测 / 推断，标注证据强度（单 seed / 配对 / 2 seed）；
**不许**把「没测出差异」写成「没有效果」；**绝不允许空报告**（失败就贴失败输出尾部与退出码）。

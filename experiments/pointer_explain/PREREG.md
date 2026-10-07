# PREREG —— P11 指针通道的结构化记录 + 可判解释

跑前写死（本文件 mtime 必须早于 `run_ptr.py` 首次执行；驱动里有 mtime 断言）。跑后不改一字。
口径：**实测** = 本目录跑出来的数字（`results.json` / `logs/run.log` 为准）；**推断** = 由数字推出；
**构造规定** = 我人为规定的规则。

## 0. 问题与边界（写死）

本单元只回答：**「指针输出能否有可判的结构性解释」**。
**不回答**「能否给出因果理由」—— 那需要干预实验（换候选、看输出变不变），本单元不做。
铁律：区分实测/推断；不许把「没测出差异」写成「没有效果」；**不许把「候选 id + 分数」包装成因果解释**。
纯 CPU（驱动与重放一律 `CUDA_VISIBLE_DEVICES=""`），不训练、不启 GPU 任务；只写 `experiments/pointer_explain/`（含 `logs/`）。

## 1. 受测记录集（只读）

| 项 | 来源（只读） | 构成 |
|---|---|---|
| 指针记录 | `experiments/gen_dispatch/results_all.json` | `random.pointer_examples` 6 + `enriched.pointer_examples` 6 = **12 条**（= P9 里指针覆盖 0/12 的那 12 条） |
| card_flow 指针记录 | `experiments/card_flow/results_real.json` | **0 条**（`batch_real.records` 只有 `status=ok/reject`，无指针通道）⇒ 受测集就是上面 12 条 |
| 既有产物 | `experiments/explain_card/`、`src/dtseek/tasks/render.py` | 只读 import，不改一字 |

**既有指针记录里有什么**（实测字段）：`input / text / evidence[{card,span,class_name}] / terminal / plan_step_id`。
**没有** `kind`、`instruction`、`ref_map`、候选集、分数、序 —— 这就是 P9 覆盖 0/12 的原因。

## 2. 结构化记录的字段（构造规定，写死）

```
{ "kind": "pointer", "rid", "input",
  "chosen":    [{card, span, span_text, score, score_provenance, rank, score_rank, eligible}, ...]  # 每个入选位一条
  "candidates":[{card, span, span_text, score, score_provenance, rank, score_rank, eligible,
                 ineligible_reason, emission_index}, ...]
  "scores_kind": "engine_confidence（原跑未持久化；重放取回。跨卡不可比 ⇒ 只作记录内一致性核对）",
  "ref_map":   [{unit_id, cls:"content", text, ref, span, out}],
               # 内容宇宙 = **全部候选 span**（每个都逐字回溯输入）；理由可以合法引用对照候选。
               # 额外断言（P9 回溯断言的加强版）：L1 解释的内容单元 ⊆ chosen。
  "evidence":  既有记录逐字原样（不改一个字）,
  "unavailable":[ ... ]   # 留空并标"不可得"的字段 + 原因
}
```

- **`chosen` 为列表**：一条指针记录可有多个入选位（每位一卡）；每个元素形如规格里的 `{card,span,score,rank}`。
- **`rank`（选择序）= 卡位序（`dialogue.DIALOGUE_BITS`：情绪→否定→人称→人物）→ 卡内发射序**；
  这是**真实入选规则**（`anchors[0]`，`dialogue.py` / `pipeline.py`）。**分数不参与入选**。
- **`score_rank`（分数序）= score 降序，tie-break (card, 发射序)** —— 记录内可核对的另一条序，**不是入选依据**。
- **候选池 = 本轮 propose 发射的全部切片**（6 张卡）；`eligible` = 该卡在 `cards_run` ∩ 位表内（真正有资格入选）。

**不可得字段清单（写死，留空 + 标"不可得"，不许编）**

| 字段 | 为什么不可得 |
|---|---|
| `chosen.score_original`（原跑分数） | 原记录未持久化 `confidence`；只能重放取回 ⇒ 原值不可得，重放值标 `provenance=replay` |
| `cards_run / plan`（原跑计划授权卡） | 原记录未持久化 ⇒ 重放取回，标 `provenance=replay` |
| `L2 理由`（为什么选它） | 见 §3 三条合格条件；不过 ⇒ 留空标"无法给出结构性理由" |
| `差异事实`（对照候选的分数/序差） | 同卡无对照候选 ⇒ 结构上不可得 |
| `negation 切片的 pos（类型）` | `pipeline.type_pos` 对 negation 返回 `None`（`card_flow.type_of` 给 `否` ∉ 官方 `POS_TYPES={名,动,形}`）⇒ 不入受约束槽 |

**不可得字段比例** = 不可得项数 ÷ 字段总数；字段表 = 14 项 × 12 条 = 168：
`kind / chosen.card / chosen.span / chosen.span_text / chosen.rank / chosen.score_replay /
chosen.score_original / candidates / scores_kind / ref_map / evidence / L1解释文本 / L2理由 / 差异事实`。

## 3. 两层解释（写死）

**L1 机械层（可判）**：候选集 + 两条序（选择序、分数序）+ 入选者 + 分数。
核对方式 = **独立重算**（纯函数重排，不碰引擎）与记录逐字段比对 ⇒ 不一致即 R2 违例。

**L2 语义层**：给出"为什么选它"的**结构性理由**，必须**三条同时成立**：

- **C1 零信息新增 + 可回溯**：理由文本过 P9 同一套谓词（6 谓词 + 回溯断言，见 §4）；
- **C2 判别性**：理由必须携带封闭差异字面 `DIFF_LIT = {比, 先, 第, 高, 低, 唯一, 最}` 之一，
  且该差异在本记录的 `candidates` 内**重算为真**、对入选者为真而对至少一个同卡对照候选为假；
- **C3 指名性**：理由文本的每个内容单元 ∈ 源 `ref_map` 且逐字 ∈ 输入。

三条全过 ⇒ **给出理由**；任一条不过 ⇒ **明确输出"无法给出结构性理由"（fail-closed）+ 失败在哪条 + 首条谓词报错原文**。

**L2 四个尝试（跑前写死）**

| # | 句式 | 目的 |
|---|---|---|
| A1 | `「X」{class_name}。` | 类别理由（如 `「满」积极。`） |
| A2 | `「X」比「Y」高。`（Y=同卡对照候选；无对照 ⇒ 记 N/A） | 分数/序差异理由 |
| A3 | `因为「X」。` | 因果理由 |
| A4 | `「X」。` | 纯复述（预期过 C1、**不过 C2**） |

## 4. 谓词复用（不另立一套）

底层 6 个谓词全部经 `experiments/explain_card/explain_card.py` 的包装调用（`_rule_a / _slot_schema /
_word_face / _item / _struct / _deref` ⇒ 即 `src/dtseek/tasks/render.py` 的谓词）；
L2 尝试与 G-free 臂直接走 `ec.x0_free`（P9 的自由臂检查器）。**本模块不实现任何谓词。**
组合函数 `x0_ptr` 与 `ec.x0_problems` 同构，唯一差别是骨架表参数换成下表；文本→骨架的合成与
`ec.free_record` 同法（分词器，不是谓词）。

**指针解释骨架表（封闭，跑中不改；门禁 = `rule_a` + `slot_schema`）

| id | 签名 | pattern |
|---|---|---|
| P01 | (形,) | `「[1]」。` |
| P02 | (形,形) | `「[1]」；「[2]」。` |
| P03 | (名,) | `「[1]」。` |

（`否` 槽不进表 —— `slot_schema` 判它不在 `POS_TYPES`，与 P9 的 A[71] 同因。）

## 5. 对照臂（写死）

- **G-structured**：带结构记录的指针解释（本臂）。
- **G-free**：**无结构、只有输出文本**。两个变体都必测：
  - `strict`：只有 `text`，无 `ref_map` ⇒ 直接走 `ec.free_record`；
  - `generous`：只从输出文本里做**词法**抽取（`『…』` 内的片段）当内容单元，仍不给候选/序/分数 ⇒ 走 `ec.explain_free` + `ec.free_record` + `ec.x0_free`。
  **R1 上必须崩（违例 > 0）；否则停下来查。**

## 6. 判据（写死）

| # | 判据 | 门槛 |
|---|---|---|
| **R0（主）** | 指针覆盖率 = 能产出**不可编且可回溯**解释的记录数 / 12 | 报**绝对数** + **不可得字段比例** |
| **R1** | 解释零信息新增（内容单元 100% 回溯输入） | **违例 = 0** |
| **R2** | 机械层可核对：独立重算的两条序 == 记录序，且 chosen == 重算入选者 | **不一致 = 0** |
| **R3** | 反例注入：① 改 chosen span ② 改分数使分数序矛盾 ③ 加输入没有的内容词 | **全部被抓，漏 0** |
| **R4** | 对照臂 G-free 在 R1 上崩 | **必报；违例 = 0 ⇒ 停下来查** |
| **R5** | 确定性（两遍一致）+ 平凡基线并列 | 不一致 = 0；基线必报 |
| **R6** | "给不出结构性理由"的比例 | 如实报（很可能是主要真结果） |

**判定三选一（阈值写死）**

- **A「指针输出可获得可判的结构性解释」**：R0 ≥ 1 ∧ R1=0 ∧ R2=0 ∧ R3 漏=0 ∧ R4 违例>0 ∧ **R6 ≤ 1/3**；
- **B「只能给机械层、给不出理由」**：R0 ≥ 1 ∧ R1=0 ∧ R2=0 ∧ R3 漏=0 ∧ R4 违例>0 ∧ **R6 ≥ 2/3**；
- **C「证据不足」**：上面基础条件任一不成立，或 **1/3 < R6 < 2/3**。

**平凡基线（必报，与 R0 同口径）**：`B-echo` = 只看"输出文本非空"就出解释（复读 `text`）——
覆盖率可到 12/12，但 R1 必须在**同一条回溯断言**下判它；本臂覆盖率必须与它并列报出。
（背景：免费规则账已 6 次，含 `n_slots` 查表 .8316、N1 .8833 ⇒ 覆盖率本身不构成证据，可核对性才构成。）

## 7. 跑前已做（披露，避免事后口径漂移）

PREREG 之前跑过 3 个**可行性探针**（`/tmp/p11/probe*.py`，非判据）：只测「CPU 重放能否逐字复现既有
`evidence/text/terminal/plan_step_id`」与 `type_pos` 取值。**结论：12/12 逐字复现（探针实测）**
⇒ 因此 R2 的"重放与既有证据一致"是**先验已知**，报告里按"已知前提"标注；
R1/R3/R4/R5/R6 的数字在 PREREG 前**没有**测过。

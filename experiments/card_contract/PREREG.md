# P13 卡片契约统一 —— 判据与等价性定义（跑前写死）

判据写死于本文件；`run_contract.py` 启动时断言本文件 mtime **早于** 首次代码改动与本目录首个结果产物，
也早于本驱动任何一次执行。跑后不改一字。

## 0. 范围与非目标

- **要做的**：四套东西合成一套卡片契约 —— ① 骨架表（`src/dtseek/tasks/render.py` 的 `SKELETONS`
  为唯一家）② 记录格式（一个 dataclass + 校验函数）③ 拒答语义（`text` 缺省 + `reason` 必填）
  ④ 机械层解释接口（只读复用 `explain_card` 与 `pointer_explain` 的谓词套件）。
- **不做的**：不训模型、不改既有实验目录（`experiments/` 中除 `card_contract/` 外一字不改）、
  不写 `dev-notes/` `methodology/`、不 commit/stash/checkout、不派子代理、不杀他人进程、纯 CPU。
- **允许写**：`experiments/card_contract/`、`logs/`、`/tmp`、以及**实际改动并证明等价**的 `src/` 文件。

## 1. 等价性 / 「不变」的定义（先写死，避免事后放宽）

1. **逐字不变（渲染）**：对同一 `(skeleton_id, assignment, bag, input_text, declared)` 输入，
   `render()` 返回记录里 `text` 字段的字符串**完全相等**（`==`，非相似度）。
   比对方式 = 快照 JSON `json.dumps(sort_keys=True)` 全等；差异条数 = 逐条数不相等的键数。
2. **既有记录重放**：从既有实验产物 JSON 读出**存档记录**，用**当前** `src` 层重新渲染，
   与存档 `text` 逐字比对。重放集在 §3 中逐条登记来源与条数（**跑前登记，跑后不得增删**）。
3. **「不一致」**：重放产出了 `kind="text"` 且 `text` ≠ 存档 `text`。
   **「不可重放」**（骨架不在当前表内 ⇒ 拒答）**单列计数**，不计入「不一致」，
   但必须**恰好等于跑前登记的废弃骨架记录数**，超出即判 V0 失败。
   这些记录的存档 `text` **不被改写**（既有实验产物一字不改）。
4. **默认行为不变**：`git diff -- src/` 中每一处改动，都必须同时给出
   (a) 既有记录重放不一致 = 0；(b) 全库 `pytest` 与基线**同为全绿**（条数、失败数逐字一致）。
   任一条不满足 ⇒ **回退该改动**并在报告里说明（不许"测试改一下"）。
5. **实测 / 推断**：本目录跑出来的数字 = 实测；由数字推出而未直接观测的 = 推断；
   我人为规定的规则 = 构造规定。三者在报告里分开标注。

## 2. 验收口径（V0–V5，门槛写死）

| # | 判据 | 门槛 |
|---|---|---|
| **V0（主）** | 旧骨架渲染逐字不变 | 对既有记录重放，**不一致条数 = 0**（不可重放按 §1.3 单列，须 = 登记数）|
| **V1** | 骨架表 id 语义唯一 | **同 id 不同签名 = 0**（签名 = `(pattern, tuple(slot.pos, slot.theta))`），或给出命名空间隔离证明 |
| **V2** | 拒答契约统一 | 归一后拒答记录**带 `text` 的条数 = 0**；`reason` 空 = 0 |
| **V3** | 默认行为不变（实测） | 既有 `pytest` 全绿（与基线同）+ **非拒答**记录重放逐字一致（不一致 = 0）|
| **V4** | 机械层解释接口统一 | 三类记录（render / gen_dispatch / pointer）都能产出可判解释；**指针覆盖率 > 0**（基线 0/12）|
| **V5** | 反例注入 | ① 篡改引用 ② 拒答带 text ③ 骨架字面含实义词 ⇒ **全被抓**（漏 = 0）|

**判定三选一**（规则先写死，不许事后硬选）：
- **契约统一成立** = V0–V5 **六条全过**；
- **不成立** = V0 或 V3 任一不过（= 默认行为/既有记录被改坏）；
- **部分成立** = 除上述之外有过有不过（报告须逐条点名哪条没过、为什么）。

## 3. 重放集（跑前登记；来源全部为既有只读产物）

| 集 | 来源（只读） | 登记条数（跑前） | 用途 |
|---|---|---|---|
| **F：官方骨架夹具** | 本目录 `fixtures`（16 条官方骨架 × 覆盖其槽位类型的袋） | 16 | V0 改前/改后快照全等 |
| **G：官方骨架既有记录** | `experiments/gen_dispatch/results_all.json` 的 `generate_samples` / `samples` 中带 `instruction` 且 `kind=text` 的记录 | 跑前数一次，写死在 `results.json.gate` | V0 + V3 |
| **A：card_flow 真卡记录** | `experiments/card_flow/results_real.json` 的 `batch_real.records` 中 `status=ok` | 36（其中骨架已废弃者 = 1，见 §4）| V0 迁移等价 + 覆盖率修复 |
| **P：指针记录** | `experiments/pointer_explain/results.json` 的 `struct_records` | 12 | V4（基线 0/12）|
| **D：gen_dispatch 记录** | `experiments/gen_dispatch/results_all.json` 的 `random/enriched` 记录与 `pointer_examples` | 跑前数一次 | V4 三类之一 |

## 4. 骨架迁移的**跑前裁定**（逐 id，跑后不改）

- **唯一家** = `src/dtseek/tasks/render.py` 的 `SKELETONS`（含手写 `_SKELETON_LIST`，import 时门禁）。
- **命名空间**：官方族 id = `S##`/`R##`（一字不改）；card_flow 迁移族 id = **`CF##`**（`##` = 原 card_flow 编号）。
  两族 id 字符串不相交 ⇒ 「同 id 不同签名」不可能发生；这是 V1 的隔离证明。
- **语义归并**（card_flow 与官方**签名完全相同**的 ⇒ 并进官方 id，不再另立）：
  card_flow `S13 → 官方 S01`、`S14 → 官方 S02`、`S12 → 官方 S04`。
- **废弃 + 原因**（`否` / `数` 槽不在官方 `POS_TYPES={名,动,形}` ⇒ 过不了 `slot_schema_problems`
  门禁，进不了表；扩展 `POS_TYPES` 会改变 `item_problems` 的默认判定域 ⇒ 视为改默认行为，不采纳）：
  card_flow `S04, S05, S10, S15, S16, S22, S26`（含 `否`）、`S23, S24`（含 `数`）—— 共 9 条。
- **迁移进表**（余 13 条 ⇒ `CF##`）：`S01, S02, S03, S06, S07, S08, S09, S11, S18, S19, S20, S21, S25`。
  槽位：`[n1] → [1]` 顺序重编号；**题元声明一律丢弃**并标 `direction_safe=True`
  （构造规定：实测 A 集 48/48 个候选 `theme=None`，真卡不产题元；`explain_card` 的同口径适配器已实测
  45/46 过官方谓词）。
- **表序**：CF 族**追加在官方 16 条之后**（`_SKELETON_LIST` 尾部）⇒ 穷举式调用方在"官方表本就能渲染"
  的输入上选中项不变。
- **登记的不可重放数**：A 集 36 条中骨架落在废弃集的 = **1 条**（骨架 `S04`，签名 `(名,否,形)`）⇒
  V0 的「不可重放」必须 = 1，不一致必须 = 0。

## 5. 记录契约（跑前写死的字段与语义）

- 字段（dataclass `CardRecord`）：`kind, text, evidence, plan_step_id, instruction, ref_map, reason, channel`
  （+ 可选 `plan`、`cards_run`、`type`、`terminal`、结构字段 `candidates/chosen/scores_kind`）。
- `kind ∈ {text, reject, pointer}`；**拒答（`kind=reject`）时 `text` 必须缺省（None 且不落地）**、
  `reason` 必填非空、`evidence` 必须为空。
- 内容单元必须**逐字回溯**：复用 `render.py` 的 `check_deref` / `item_problems` 谓词，不另立一套。
- 归一器 `to_contract(raw)` 是**唯一出口**：上游 `render.render()` 与 `dialogue._reject()` 的
  `（拒答）…` 占位串在归一器里并入 `reason` 并丢弃 `text`。

## 6. 解释接口（跑前写死）

- 三类记录 ⇒ 机械层可判解释：**只读 import** `experiments/explain_card/explain_card.py`（`explain` / `x0_problems`
  / `x2_problems`）与 `experiments/pointer_explain/ptr_explain.py`（`l1_card` / `x0_ptr`），
  谓词本体仍是 `render.py` 的 `rule_a/slot_schema/word_face/item/structure/deref`。
  **不另立一套谓词**（本目录若出现新的"判定谓词"，必须是对上述函数的转发）。
- 指针覆盖率基线 = `explain_card` X5 的 **0/12**；目标 = **> 0**（记录补 `kind/instruction/ref_map`
  结构字段 + `chosen/candidates/scores` 后走 `ptr_explain` 的受约束臂）。

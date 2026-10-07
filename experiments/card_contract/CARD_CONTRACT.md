# 卡片契约（P13）

一张卡从上游（`render` / `dialogue` / `gen_dispatch` / `pointer_explain`）产出来，
进系统时只认**一个**骨架表、**一个**记录形状、**一条**拒答语义、**一条**解释接口。
实现点：`src/dtseek/tasks/render.py`（骨架表）、`src/dtseek/tasks/card_contract.py`
（记录 schema + 归一 + 判据）；判据实测见 `experiments/card_contract/results.json`。

## 1. 骨架表：唯一一家

- 唯一的家是 `render.SKELETONS`（封闭、手写、可枚举），共 **29 条**：
  - **官方族 16 条** `S01..S13` + `R01..R03`，来自 `_SKELETON_LIST`，**一字未改**；
    视图 `render.SKELETON_OFFICIAL`。
  - **迁移族 13 条** `CF01..CF25`，来自 `_CF_SKELETON_LIST`（id 的 `##` = 原
    `card_flow` 编号），视图 `render.SKELETON_CF`。
- **两族 id 不相交**（前缀隔离）⇒ 结构上不存在「同 id 不同签名」。
- import 时跑 `validate_skeleton_table()`（fail-closed）：规模 ≤ `SKELETON_TABLE_LIMIT`、
  id 不重复、表 == 手写全集、两族不相交、逐条过规则 A 与槽位 schema。
- 迁移族的构造规定：`[n1]→[1]` 顺序重编号；**题元一律丢弃** ⇒ `direction_safe=True`。
- 穷举式调用方（`_SKELETON_LIST` 的只读遍历）看到的仍是官方族 16 条；
  要整张表就遍历 `SKELETONS.values()`。
- 逐 id 的 `旧 → 新` 与处置理由：`experiments/card_contract/migration.py::MIGRATION`。

## 2. 记录 schema：一个 dataclass

`card_contract.CardRecord`，字段顺序即文档顺序：

| 字段 | 类型 | 语义 | 必填 |
| --- | --- | --- | --- |
| `kind` | `text`/`reject`/`pointer` | 记录种类 | 是 |
| `text` | `str \| None` | 回复文本；**拒答时缺省**（`to_dict()` 不落地该键） | text/pointer 必填 |
| `evidence` | `list[dict]` | 证据（span 溯源单元）；拒答必须为空 | 是 |
| `plan_step_id` | `str` | `step:<i>` 或 `none` | 是 |
| `instruction` | `dict \| None` | `{skeleton_id, assignment}` 结构指令 | 生成型必填 |
| `ref_map` | `list[dict]` | 引用映射（内容/形式/逻辑 → out 区间） | 生成型必填 |
| `reason` | `str` | 拒答理由；拒答必填非空 | 拒答必填 |
| `channel` | `pointer`/`generate`/`reject` | 通道 | 是 |
| `plan` / `cards_run` | `list[str]` | 计划行 / 本轮跑过的卡 | 否 |
| `type` / `terminal` | `str \| None` | 对话声明类型 / 终点 | 否 |
| `chosen` / `candidates` / `scores_kind` | list / list / str | **指针结构字段**：入选者、候选集、分数口径 | 指针必填 |

通道由构造决定：`kind=reject → reject`；`kind=pointer → pointer`；`kind=text` 带
`instruction → generate`，否则 `pointer`。`plan_step_id` 缺失时按
`plan` 非空 ⇒ `step:{len(plan)-1}`，否则 `none` 派生。

## 3. 拒答语义：一个出口

- **上游记录不改**：`render.render()` 与 `dialogue._reject()` 仍产 `（拒答）…` 占位
  `text`（改源头会打破既有测试，见 §5）。
- **唯一出口 `card_contract.to_contract(raw)`**：`kind=reject` ⇒ 丢 `text`、
  占位串并入 `reason`（已有 `reason` 则保留）、清空 `evidence`、补 `channel=reject`。
- 归一后的拒答记录：**`text` 键不存在、`reason` 非空、`evidence` 为空**。
- 任何记录进系统前先过 `to_contract()`；`to_dict()` 是它落地成普通 dict 的形式。

## 4. 判据与解释可得性

`card_contract.contract_problems(rec, input_text=…)` 返回空 = 通过：

- **C1** `kind`/`evidence`/`plan_step_id`/`channel` 齐全且在取值域内；
- **C2 拒答**：`text` 缺省 + `reason` 非空 + `evidence` 空；
- **C3 文本**：`text` 非空、`evidence` 非空；生成型必须带 `instruction` + `ref_map`；
- **C4** `plan_step_id` 形制（`none` 或 `step:<i>`）与轮次对齐；
- **C5 内容逐字回溯**：复用 `render.item_problems`（span 落界、逐字、零信息新增）；
- **C6 指针结构**：必须带 `chosen`/`candidates`/`scores_kind`；
- **C7 生成覆盖**：`ref_map` 各 `out` 区间恰好铺满 `text` 且逐字相符。

解释接口按 `kind/channel` 路由到既有谓词套件（本契约不新写判定谓词）：

| 记录 | 解释套件 | 谓词 |
| --- | --- | --- |
| `reject` | `explain_card.explain` 拒答卡 | `x2_problems` |
| `text` + `instruction` | `explain_card.explain` E 表 | `x0_problems` |
| `pointer` | `pointer_explain.l1_card` P 表 | `x0_ptr` |

指针记录必须先补结构字段才能过 C6/C5（实测：缺结构 12/12 违例 → 补齐 0/12 违例，
可判解释从 0/12 → 12/12）。

## 5. 新卡接入检查清单

1. **骨架**：字面只许纯形式词 + 逻辑词（`rule_a_problems == []`）；槽位 `pos ∈ POS_TYPES={名,动,形}`；
   名槽要么带题元、要么 `direction_safe=True` 且指派随原文 span 升序；
   写进 `_CF_SKELETON_LIST`（官方族字面不动），跑 `validate_skeleton_table() == []`。
2. **记录**：上游构造 → `to_contract(raw)` → `to_contract` 后的
   `contract_problems(rec, input_text=原文) == []`。
3. **拒答**：若上游给了占位 `text`，只能由 `to_contract` 吃掉；判据用 C2，不要另立谓词。
4. **解释**：按 `kind` 走 §4 的三条路由之一，产出的 `problems` 必须为空；
   指针卡要带 `chosen`/`candidates`/`scores_kind`。
5. **回归**：`uv run python -m pytest -q` 全绿（基线 **258 passed**）；
   改表/改 schema 后重跑 `experiments/card_contract/run_contract.py --phase final`，
   V0 的官方族快照必须 0 差异。

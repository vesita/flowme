# P13 卡片契约统一 · 报告

**判定：契约统一成立**（V0–V5 六条判据全过，`results.json::verdict_inputs` 六 flag 均为 true）。
工作目录 `/home/vesita/coding/my/DTSeek`，纯 CPU、未训练、未提交、未派子代理。

## 1. 迁移映射表 + V1 证明

旧表 25 条（`experiments/card_flow/skeletons.py`，只读）→ **归并 3 / 迁移 13 / 废弃 9**。

| 处置 | 映射 | 条数 | 原因 |
| --- | --- | --- | --- |
| 归并 | S13→S01、S14→S02、S12→S04 | 3 | 语义与官方条目同签名（pattern + 槽位一致） |
| 迁移 | S01→CF01、S02→CF02、S03→CF03、S06→CF06、S07→CF07、S08→CF08、S09→CF09、S11→CF11、S18→CF18、S19→CF19、S20→CF20、S21→CF21、S25→CF25 | 13 | id 换到 CF 命名空间（`##` = 原编号），题元一律丢弃、`direction_safe=True` |
| 废弃 | 含 `否` 槽 7 条：S04、S05、S10、S15、S16、S22、S26；含 `数` 槽 2 条：S23、S24 | 9 | `否`/`数` ∉ `POS_TYPES={名,动,形}`，过不了 `slot_schema_problems` 门禁 |

记录级处置（36 条成功记录 100% 落位）：S11×25→CF11、S02×8→CF02、S06×2→CF06、S04×1→废弃（不进表）。

V1 实测：家里 29 条 = 官方 16 + CF 13；`home_dup_ids=[]`、`home_same_id_diff_sign=[]`、
`namespace_overlap=[]`（两族前缀隔离，且 `validate_skeleton_table()` 里加了相交即报错的门禁）、
`migrated_sig_mismatch=[]`（13 条迁移 + 3 条归并的签名逐字一致）、`unresolved=[]`、
`missing_in_home=[]`、`deprecated_sneaked_in_home=[]`、`home_pos_out_of_domain=[]`。
反事实（不改名会怎样）：**13 个 id（S01..S13）同 id 不同签名，覆盖 36/36 条记录**；改名后为 0。

## 2. 记录契约字段表 + V2/V3 实测

唯一形状 `dtseek.tasks.card_contract.CardRecord`（字段即顺序）：

| 字段 | 取值/类型 | 必填条件 |
| --- | --- | --- |
| `kind` | text / reject / pointer | 三者之一 |
| `text` | str，**拒答缺省** | kind=text、pointer |
| `evidence` | list[dict] | 必填；拒答必须为空 |
| `plan_step_id` | `step:<i>` 或 `none` | 必填（缺则按 plan 派生） |
| `instruction` | `{skeleton_id, assignment}` | channel=generate 必填 |
| `ref_map` | list[dict] | channel=generate 必填 |
| `reason` | str | kind=reject 必填非空 |
| `channel` | pointer / generate / reject | 必填（按 kind+instruction 推导） |
| `plan`、`cards_run` | list[str] | 可选 |
| `type`、`terminal` | str | 可选 |
| `chosen`、`candidates`、`scores_kind` | list / list / str | kind=pointer 必填 |

V2 实测（26 条上游原始记录 = render 9 + dialogue 9 + 存档 dispatch 片段 8）：
归一前**带 text 的 18 条**，本契约谓词违例 26/26、`gen_dispatch` 既有谓词违例 26/26、两者判定一致 **26/26**；
归一后 **`text` 键落地 0 条、`reason` 空 0 条、两套谓词违例 0 + 0、判定一致 26/26**。

**源头归一试验（改→实测→回退）**：让 `render._record` / `dialogue._record` 在 `kind=reject` 时不落地 `text`
⇒ 全库 pytest **7 failed, 251 passed**（3 条 `gen_dispatch/test_contract.py` 的"冲突证据"测试 + 4 条 `tests/test_dialogue.py`
断言拒答卡带 `text`）⇒ **该 src 改动不采纳**，从 `/tmp/p13bak` 回退，md5 复核一致，回退后 **258 passed**。
归一只落在契约层唯一出口 `to_contract()`。

V3 实测：pytest 四次全绿（基线 258 → 改后 258 → 回退后 258 → 最终 258，均 `EXIT=0`）；
非拒答记录逐字比对 **26 条（官方夹具 16 + G 集 10）不一致 0**；raw 拒答记录 18 条逐字差 **0**。

## 3. 文档章节索引 + 新卡接入检查清单

`experiments/card_contract/CARD_CONTRACT.md`：§1 骨架表唯一一家（29 条 = 官方 16 + CF 13、门禁、迁移规定）；
§2 记录 schema（上表 + 通道派生 + `plan_step_id` 派生）；§3 拒答语义（上游不改、`to_contract()` 唯一出口）；
§4 判据 C1–C7 与解释路由（reject→`x2`、text→`x0`、pointer→`x0_ptr`）；§5 新卡接入检查清单。

清单（文档 §5 原文的 5 条）：① 骨架过规则 A + 槽位 schema + 名槽题元/方向安全，写进 `_CF_SKELETON_LIST`，
`validate_skeleton_table()==[]`；② 上游构造 → `to_contract()` → `contract_problems(rec, input_text)==[]`；
③ 拒答占位 `text` 只能由 `to_contract` 吃掉，判据用 C2 不另立谓词；④ 按 kind 走三条解释路由，
`problems==[]`，指针卡带 `chosen/candidates/scores_kind`；⑤ 全库 pytest 全绿（基线 258）+ 重跑
`run_contract.py --phase final`，V0 官方族快照 0 差异。

## 4. V0–V5 逐条实测 + 判定

| 判据 | 实测 | 结论 |
| --- | --- | --- |
| V0 官方 16 条渲染逐字不变 | 官方夹具差异 0、官方表签名差异 0、G 重放 10/10 match（0 diff 0 reject）、A 重放 35 match + 1 deprecated（PREREG 登记数恰为 1）、raw 拒答 18 条差异 0、G→A 表新增仅 13 个 CF id | 过 |
| V1 同 id 不同签名 = 0 | 重复 id 0、同 id 异签名 0、两族相交 0、迁移签名不一致 0、未处置 0；反事实 36/36 撞 | 过 |
| V2 归一后拒答带 text = 0、reason 空 = 0 | 0 / 0，违例 0 + 0（两套谓词），判定一致 26/26 | 过 |
| V3 既有测试全绿 + 非拒答 text 逐字不变 | `258 passed, EXIT=0`；26 条非拒答记录不一致 0 | 过 |
| V4 三类记录可判解释 + 指针覆盖率 > 0 | render 35/35、gen_dispatch 30/30、pointer 12/12，`problems_total` 全 0；指针基线 **0/12 → 12/12**；指针契约违例 缺结构 12 → 补结构 0 | 过 |
| V5 三类反例全被抓 | ① 篡改引用 90/90；② 拒答带 text 解释侧 26/26 + 契约侧 26/26；③ 骨架字面含实义词 3/3（规则 A / 表门禁 / 渲染出口），`missed=0` | 过 |

**判定：契约统一成立**（不硬选：六条判据各有一条机器可复现的数字，全部达标）。

## 5. 改过的 `src/` 文件 + 等价性证明

- `src/dtseek/tasks/render.py`（+52 / −4）：新增 `_CF_SKELETON_LIST` 13 条、`_ALL_SKELETON_LIST`、
  `SKELETON_CF`/`SKELETON_OFFICIAL` 视图、`__all__` 两项；`SKELETONS` 由官方族改为全集；
  `validate_skeleton_table()` 的对照集合改为全集并新增两族相交门禁。
  **等价性**：官方 16 条的 pattern/槽位/顺序在 diff 里是上下文行（未被触碰），V0 官方表签名差异 0、
  官方夹具 16/16 逐字一致、G 集 10/10 逐字一致、`_SKELETON_LIST` 仍为官方 16 条（穷举式只读调用方看到的序列不变）。
- `src/dtseek/tasks/card_contract.py`（新增，303 行）：`CardRecord` / `to_contract` / `contract_problems` /
  `content_problems`。**等价性**：除本目录外无任何模块 import 它（默认行为不被触达）；
  该文件自身的修改时间晚于 PREREG，`gate(final=True)` 实测通过。
- **回退记录**：源头拒答归一试验的改动已从 `/tmp/p13bak` 精确回退，`dialogue.py` 全程未改
  （md5 `01593cfe…d7869` 不变），`render.py` 回退后 md5 与试验前一致（`c23fed45…1a66`）。

## 6. 命令日志/产物路径 + 遗留与不确定

**日志**：`logs/p13/commands.log`（原始命令与回显）、`baseline_pytest_all.log`、`after_render_pytest.log`、
`exp_source_normalize_pytest.log`（7 failed 现场）、`after_revert_pytest.log`、`final_pytest_all.log`（均 258 passed）。
**产物**：`experiments/card_contract/` 下 `PREREG.md`、`CARD_CONTRACT.md`、`report.md`（本文件）、
`fixtures.py`、`migration.py`、`replay.py`、`explain_ops.py`、`run_contract.py`、
`baseline_render.json`、`after_render.json`、`results.json`。

**遗留（实测现状，非失败）**：① `_SKELETON_LIST` 保持官方 16 条，只遍历它的调用方
（`gen_dispatch/pipeline.py::search_renderable`、`card_flow/crosscheck_official_render.py`）看不到 CF 族；
要用整表必须遍历 `SKELETONS` —— 保留 16 条正是为了让这两个只读实验的默认行为逐字不变，属已登记的取舍。
② A 集 1 条不可重放（S04 含 `否` 槽）⇒ 覆盖 35/36，是 PREREG 里写死的登记数。

**不确定（推断级，须与上面的实测区分）**：③ G 集袋的 `pos` 由槽位规格回填（原跑靠
`item.pos == slot.pos` 才出 `kind=text` ⇒ 回填值等于原跑值），这是**推断**；唯一可得的
`bag_pos` 是整袋 pos 直方图、不按 ref 对齐，只能做 ⊆ 不等式佐证（10/10 违例 0，弱证据）。
④ 迁移时"题元一律丢弃"依据 card_flow A 集 48/48 个候选 `theme=None`，是**在既有产物上实测**的，
但**未**在真卡袋上重跑全量生成流程复核。⑤ `plan_step_id` 缺失时按 `step:{len(plan)-1}` 派生是
**构造规定**（不是从既有产物归纳出来的），既有产物中该字段有值、未触发该分支。

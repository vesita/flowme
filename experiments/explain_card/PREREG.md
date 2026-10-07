# PREREG —— P9 受约束的解释卡：解释不可编

跑前写死（本文件 mtime 必须早于 `run_explain.py` 首次执行；驱动里有 mtime 断言）。
跑后不改一字。口径：**实测** = 本目录跑出来的数字（`results.json` / `logs/run.log` 为准）；
**推断** = 由数字推出；**构造规定** = 我人为规定的规则。

## 0. 问题与边界（写死）

本单元只回答：**「解释能否被约束为不可编 + 可回溯」**。
**不回答**「解释是否有用 / 易懂 / 正确」—— 那需要人评，本单元不做。
铁律：区分实测/推断；不许把「没测出差异」写成「没有效果」。
纯 CPU，不训练、不加载 GPU 任务；只用既有产物推理。只写 `experiments/explain_card/`（含 `logs/`）。

## 1. 受测记录集（只读复用）

| 集 | 来源（只读） | 构成 |
|---|---|---|
| **A** | `experiments/card_flow/results_real.json` `batch_real.records` | 120 条：`status=ok` 36 / `status=reject` 84（stage=compose 82、filter 2） |
| **B** | `experiments/gen_dispatch/results_all.json` | `generate_samples` 10（random 2 + enriched 8）、`reject_examples` 8、`pointer_examples` 12 |
| **B′** | `experiments/gen_dispatch/results_adv.json` | 递归抽出 `kind∈{text,reject}` 的记录 15 条（`render` 11 + A13 子记录 3 + A14 1） |

**格式差异（如实说明）**：A 集是 card_flow 自有最小版渲染的产物（`status/structure/mapping`，
**没有** `ref_map`（cls∈{content,formal,logic,align}）/`evidence(source=span|table)`/`kind="reject"+reason` 的官方形状）；
B/B′ 才是 `src/dtseek/tasks/render.py` 的原生形状，但 `generate_samples` 只留了 `ref_map_head`（前 3–4 条）。
**替代方案（构造规定）**：写一个**只读适配器**把 A 归一到官方形状，再用官方谓词复验；适配器只做
「换 id 空间 / 补 ref_map / 补 evidence」，不改任何文本与 span。适配规则：

1. 内容引用 `c1 → int`：按该记录 `mapping` 的**顺序**（= 槽位序 = span 升序）编 0..k-1；`candidate_id` 取原 cid。
2. `BagItem(pos=结构自带 type, theta=theme, screened=candidates[].valid, candidate_id=cid)`。
3. 骨架表：card_flow 25 条 → 官方 `render.Skeleton` 形状的**适配表** `CF_SKELETONS`（同 id、
   pattern 由 `[n1][a1]。` 翻成 `[1][2]。`、槽位类型同名、`direction_safe=True` 且名槽 `theta=None`
   —— 真卡不产题元，官方层同口径走「只复述不换方向」分支）。类型不在官方 `POS_TYPES={名,动,形}`
   的骨架（`否`/`数`）**适配期即失败 ⇒ 未覆盖**，如实计入 X5。
4. `B` 集 `ref_map`：由 `instruction + text + evidence + 官方骨架表` **确定性重放** `render()` 的读表逻辑重建，
   并断言 `重建[:len(ref_map_head)] == ref_map_head`（重建对不对的独立核对）。
5. `B` 集 `BagItem` 缺 `type` ⇒ `pos` 由**记录自带的** `instruction.assignment[k]` ↔ 官方骨架 `slots[k].pos`
   推出（**推断，非独立证据**）；`screened=True` 由「该记录当初通过了 `render()` 的 `item_problems`」推得（同为推断）。

## 2. 解释卡的输出形状（字段，构造规定）

```
{ "kind": "text" | "reject",
  "text":        "<解释文本>"        # 仅 kind=text；与 render.py 记录同形
  "reason":      "<逐字来自源记录>"   # 仅 kind=reject；非空
  "ref_map":     [{unit_id, cls∈{content,formal,logic,align}, text, ref, span, out}, ...]
  "evidence":    [{source:"span", ref, text, span, candidate_id} | {source:"table", unit_id, word, cls}]
  "instruction": {"skeleton_id", "assignment"}      # 解释指令（解释骨架 + 内容单元 ref）
  "source":      {"set","idx","kind","skeleton_id","assignment","text"/"reason"}  # 被解释记录的指纹
}
```
`kind="reject"` 的卡**不得有 `text` 键**（`reason` 必填非空）。

**解释文本怎么来**：走 `render.render(instruction, bag, input, skeletons=EXPL_SKELETONS)`——
即**复用官方渲染器本身**（它内部就是 `check_structure` → 渲染 → `check_deref`）。
槽位只填源记录 `ref_map` 的 content 单元（逐字），字面只来自下面这张**封闭解释骨架表**：

| id | 签名（槽位类型序） | pattern |
|---|---|---|
| E01 | (名,) | `「[1]」。` |
| E02 | (名,形) | `「[1]」；「[2]」。` |
| E03 | (名,名) | `「[1]」；「[2]」。` |
| E04 | (名,动) | `「[1]」；「[2]」。` |
| E05 | (名,动,形) | `「[1]」；「[2]」；「[3]」。` |

写死：签名不在表内 ⇒ 该记录**不可解释**，计入 X5 未覆盖。解释文本 = 逐字内容单元 + 封闭表标点，
**不含任何输入里没有的内容**（这正是「零信息新增」在解释上的落法）。
人要读的结构信息在 `ref_map` / `evidence` / `source` 字段里（字段名是代码常量，不是生成内容）。

**拒答的解释**：`reason` **逐字复制**源记录的 `reason`（不改写、不补写），`text` 键不建。

## 3. 谓词复用清单（逐谓词写死用在哪一步；不另立一套）

| `render.py` 谓词 | 用在哪一步 |
|---|---|
| `rule_a_problems(skeleton)` | ① 门禁 `CF_SKELETONS`（适配表）② 门禁 `EXPL_SKELETONS`（解释表）③ 对 **E-free** 解释文本合成的骨架跑（抓自由拼接的字面） |
| `word_face_problems(input, skeleton, declared)` | ① 源骨架（含逻辑词的 card_flow S18–S21）② 解释骨架 ③ E-free 合成骨架 |
| `item_problems(BagItem, input)` | ① 源袋每个块 ② **解释文本每个内容单元**（span 逐字 + `screened/candidate_id` = 零信息新增） |
| `check_structure(instruction, bag, input, skeletons=…)` | ① 源记录结构侧复验 ② 解释卡结构侧 |
| `check_deref(record, bag, input, skeletons=…)` | ① 源记录解引用侧复验 ② **解释卡整体**（ref_map 恰好铺满 + 内容条目回溯 + 功能词走表项 id） |

**X0（零信息新增谓词）** = 下列 5 条问题数之和，门槛 **0**：
`rule_a(解释骨架)` + `slot_schema(解释骨架)` + `word_face(解释骨架)` + `Σ item_problems(每个内容单元)`
+ `check_structure(解释卡)` + `check_deref(解释卡)`，
外加一条**回溯断言**：解释文本每个 `cls=content` 条目的 `(text, span)` 必须**存在于源记录 ref_map 的 content 条目**中。
（`slot_schema_problems` 是 `render.py` 的既有函数，一并跑。）

## 4. 反例注入（X1，三类，必须逐类报实际抓到的输出）

在**通过 X0 的解释卡**上做突变（每类在全部可解释卡上各跑一遍）：

- **① 加一个输入没有的内容词**：解释 `text` 末尾追加词典词 `方案`（先断言该词 ∉ 该条输入），
  `ref_map` 同步加一条 `cls=content, text=方案, ref=<在指派里的真实 ref>, span=<伪造区间>`，`out` 衔接保持铺满。
  期望被 `item_problems`（非输入逐字子串 / 未回溯到被筛候选）与 `check_deref`（同一袋块映射多次 / 指派不一一对应）
  及回溯断言抓到。
- **② 引用指向不存在的 ref/span**：(a) 把某条 content 条目的 `ref` 改成 `999`；(b) 把某条的 `span` 改成
  `[0, 999]`（越界）。期望 `check_deref` 抓（不在指派/不在袋、span 非逐字子串）。
- **③ 拒答解释偷偷带 `text`**：给 `kind=reject` 的解释卡塞 `text="（拒答）…"`。期望 X2 谓词抓。

判定：**逐类报实际抓到的条数与首条报错原文**；**漏一类即不通过**。

## 5. 对照臂

- **E-constrained**：§2 的受约束解释卡。
- **E-free**：**自由生成解释**——同一份记录用固定中文模板自由拼接
  （`f"这句话的意思是{input}，其中「{u}」作{role}，所以输出「{record.text}」。"` 逐单元循环，role 由 cls 推），
  无任何约束；再用**同一套**谓词（分词后合成骨架 → `rule_a`/`word_face`/`item_problems`/`check_structure`/`check_deref`）
  去咬它。
  **X4 = E-free 在 X0 上的违例数**，必报；**若 E-free 也不违规 ⇒ 说明约束没起作用，停下来查**。

## 6. 判据与门槛（写死）

| # | 判据 | 门槛 |
|---|---|---|
| **X0（主）** | E-constrained 解释文本过「零信息新增」谓词（内容单元 100% 回溯） | **违例 = 0** |
| **X1** | 三类反例注入全被抓 | 逐类报实际抓到的条数 + 首条报错原文；**漏一类即不通过** |
| **X2** | 拒答的解释含非空 `reason` 且 **`text` 键缺省** | **违例 = 0** |
| **X3** | 确定性：同一记录两次渲染逐字相同（整卡 `json.dumps(sort_keys=True)` 全等） | 不一致 = 0 |
| **X4** | E-free 在 X0 上的违例数 | **必报**；=0 ⇒ 停下来查约束是否真起作用 |
| **X5** | 覆盖率：能给出解释的记录比例（**成功 / 拒答分别报**）+ 未覆盖原因分布 | 必报，无门槛 |

**判定三选一（不许硬选）**：
**T1 解释可被约束为不可编** = X0∧X1∧X2∧X3 过 **且** X4 显示 E-free 违例数 > 0（对照确实更差）；
**T2 约束无效** = X0–X3 任一不过，或 X4 的 E-free 违例数 = 0；
**T3 证据不足** = 受测记录太少（可解释成功记录 < 10 条）或关键数据缺失导致任一判据跑不出来。

**边界句（必须写进报告）**：本单元只回答「解释能否被约束为不可编 + 可回溯」，
**不回答「解释是否有用 / 易懂 / 正确」**（那需要人评，本单元不做）。

## 7. 实现与命令（写死）

- `explain_card.py`：适配器 + 解释卡渲染 + 谓词套件 + E-free + 注入。
- `run_explain.py`：驱动（`uv run python experiments/explain_card/run_explain.py`），
  产出 `results.json`、`logs/run.log`；启动时断言本文件 mtime < 现在。
- 确定性：全集渲染两遍逐卡比对（X3）。
- 不训练、不碰 `src/`、不碰其他实验目录；禁止 `git commit/stash/checkout/restore/clean`。

# P9 受约束的解释卡 —— 解释能否被约束为「不可编 + 可回溯」

判据：`experiments/explain_card/PREREG.md`（mtime **2026-10-07 11:51:39 < 首跑 11:58:57**，驱动里有断言；跑后未改一字）。
口径：**实测** = `results.json` / `logs/run.log`；**推断** 与 **构造规定** 分开标注。纯 CPU，未训练、未启动任何 GPU 任务；
`git diff --stat -- src/` **空**（`src/` 只读 import），只写了 `experiments/explain_card/`。

> **边界句（先写）**：本单元只回答**「解释能否被约束为不可编 + 可回溯」**，
> **不回答「解释是否有用 / 易懂 / 正确」**（那需要人评，本单元不做）。

## 1. 解释卡的输出形状 + 谓词复用（逐谓词）

字段（与 `render.py` 记录同形）：`kind` / `text`（仅 text 卡）/ `reason`（仅 reject 卡、非空、**逐字**来自源记录）/
`ref_map[{unit_id, cls∈{content,formal,logic,align}, text, ref, span, out}]` / `evidence[{source:"span"|"table"}]` /
`instruction{skeleton_id, assignment}` / `source{set,rid,kind,skeleton_id,assignment,text|reason}`。**reject 卡不建 `text` 键。**

文本怎么来：`render.render(Instruction(E0x, assignment), bag, input, skeletons=EXPL_SKELETONS)` ——
**复用官方渲染器本身**（它内部就是 `check_structure` → 渲染 → `check_deref`）。
封闭解释骨架表 5 条（E01–E05，字面只有 `「」；。`），槽位只填源 `ref_map` 的 content 单元；签名不在表内 ⇒ 不可解释（进 X5）。

| `render.py` 谓词 | 用在哪一步 | 调用 / 返回非空（实测） |
|---|---|---|
| `rule_a_problems` | 门禁解释表 E01–E05；复验源骨架；咬 E-free 合成骨架 | 322 / **46**（45 条 E-free + 其样例重算 1；受约束臂与源骨架恒空） |
| `slot_schema_problems` | 同上三处的类型/槽位 schema | 322 / **1**（源记录 A[71] 的 `否` 槽） |
| `word_face_problems` | 源骨架 / 解释骨架 / E-free 骨架的逻辑词·数字逐字有据 | 317 / **45**（全部来自 E-free；受约束臂与源骨架恒空） |
| `item_problems` | **解释文本每个内容单元**（span 逐字 + `screened/candidate_id`；源袋逐块由 `check_structure` 内部同跑） | 592 / **45**（全部来自加测 ①′） |
| `check_structure` | 源记录结构侧复验 + 解释卡结构侧 | 317 / **47** |
| `check_deref` | 源记录 + **解释卡整体**（ref_map 恰好铺满 / 内容回溯 / 功能词走表项 id） | 317 / **226** |

**X0（零信息新增，PREREG §3）** = `rule_a + slot_schema + word_face(解释骨架)` + `Σ item_problems(每个内容单元)`
+ `check_structure(解释卡)` + `check_deref(解释卡)` + **回溯断言**（每个 `cls=content` 的 `(text,span)` 必须存在于源 `ref_map` 的 content 条目）。

## 2. 真实解释样例 8 条（原文照抄，含 3 条拒答）

| 输入 | 记录结构（源） | 解释卡 |
|---|---|---|
| 我让它发挥最大的价值 | `S02 [1,3]` text=`我让发挥最。`；units `我让[0,2]`+`发挥最[3,6]`+表`。` | text=`「我让」；「发挥最」。` `E02[1,3]`；units `「[0,1]`·`ref:1 我让[0,2]`·`」；「`·`ref:3 发挥最[3,6]`·`」。` |
| 你才得早睡呢 | `S02 [1,5]` text=`你睡。`；units `你[0,1]`+`睡[4,5]`+表`。` | text=`「你」；「睡」。` `E02[1,5]` |
| 死了记得把你家那个送我 | `S11 [4]` text=`那个送吗？`；units `那个送[7,10]`+表`吗`+`？` | text=`「那个送」。` `E01[4]` |
| ✨ 带一杯她喜欢的奶茶…（gen_dispatch 随机批） | `R02 [0,1]` text=`她喜欢。`；units `她[5,6]`+`喜欢[6,8]`+表`。` | text=`「她」；「喜欢」。` `E04[0,1]` |
| 但题目里有没有说明这一点呢 | `R02 [0,2]` text=`目说。`；units `目[2,3]`+`说[7,8]`+表`。` | text=`「目」；「说」。` `E04[0,2]` |
| **北京还是兰州呢**（拒答） | `status=reject stage=compose reason=type_inventory` | `kind=reject`，`reason=type_inventory`，**无 `text` 键** |
| **据我的经验学理工大多洗澡不勤**（拒答） | 同上 | `kind=reject`，`reason=type_inventory`，**无 `text` 键** |
| **淀粉酶增高程度往往与腮腺肿胀程度成正比**（拒答） | dispatch `R-CH5 终点=pointer 但指针无据：…；P=计划授权调用的卡没有发射任何切片` | `kind=reject`，`reason` 逐字同上，**无 `text` 键** |

（每条的完整 `ref_map`/`evidence` 在 `results.json` 的 `samples`；解释文本里的内容单元 **100% 逐字来自源 `ref_map`**，分隔符来自封闭表。）

## 3. X0–X5 逐条实测 + X1 三类注入的实际输出

| # | 实测 | 结果 |
|---|---|---|
| **X0（主）** | 45 张可解释 text 卡跑满 6 项谓词 + 回溯断言 | **违例卡 0 / 问题总数 0** ✅ |
| **X1** | 三类注入（＋1 个 PREREG 外加测） | **逐类全抓，漏 0**（下表）✅ |
| **X2** | 107 张 reject 解释卡 | `reason` 空 0、带 `text` 键 **0** ⇒ **违例 0** ✅ |
| **X3** | 全集 165 卡渲染两遍，`json.dumps(sort_keys)` 全等 | **不一致 0** ✅ |
| **X4** | E-free 在同 45 条上跑同一套谓词 | **违例卡 45/45，问题总数 4576**（`rule_a` 2131 / `structure` 2335 / `word_face` 65 / `deref` 45）✅ 对照确实更差 |
| **X5** | 见 §4 | 成功 **45/46 = 0.9783**；拒答 **107/107 = 1.0** |

**X1 逐类实际抓到的（首条报错原文）**

| 注入 | 注入/被抓/漏 | 类别 | 代表报错原文 |
|---|---|---|---|
| ① 加一个输入没有的内容词（`方案`） | 45 / 45 / **0** | 回溯断言 45、deref 135 | `回溯断言: 内容单元 '方案' span=(0, 2) 不在源记录 ref_map 的 content 条目里（解释引入了新内容）`；`I3 违规：ref_map[5] 文本 '方案' 与袋块 '我让' 不符` |
| ②a 引用指向不存在的 ref | 45 / 45 / **0** | 回溯 45、deref 90 | `回溯: 内容条目 ref=999 不在袋里`；`I3 违规：ref_map[1] 内容条目 ref=999 不在指令指派里（编的内容词）` |
| ②b 引用指向不存在的 span | 45 / 45 / **0** | 回溯断言 45、deref 45 | `I3 违规：ref_map[1] span (0, 999) 不是输入逐字子串` |
| ③ 拒答解释偷偷带 `text` | 107 / 107 / **0** | X2 107 | `X2 违规：kind=reject 的解释卡带了 text 键（拒答不许有文本）` |
| ①′（**PREREG 外加测**，不改判据）：再伪造一个袋块给它背书 | 45 / 45 / **0** | item 45、回溯断言 45、deref 90 | `evidence 不齐备：袋 9999 '方案' 不是输入逐字子串（span 给出的是 '我让'）` |

①②③ 三类**全部被抓、漏 0** ⇒ X1 过（①′ 是额外加强，不进判据）。

**源记录复验（适配器的独立核对）**：46 条 text 记录过官方五谓词，**违例 1 条 = A[71]**（骨架 `S04` 的 `否` 槽不在官方 `POS_TYPES={名,动,形}`），
其余 **45/46 过** ⇒ card_flow 自有形状的记录经只读适配后与官方谓词判定一致（`ref_map` 重放 vs `ref_map_head` 逐条相等）。

## 4. 覆盖率与未覆盖原因分布（X5）

| 桶 | n | 给出解释 | 比例 | 未覆盖原因（原文） |
|---|---|---|---|---|
| 成功 `kind=text` | 46 | **45** | **0.9783** | `签名 ('名','否','形') 不在封闭解释表（含 否/数 槽 ⇒ 官方 POS_TYPES 外）` ×1 |
| 拒答 `kind=reject` | 107 | **107** | **1.0000** | 无 |
| 指针通道（有 text 无 `instruction/ref_map`） | 12 | 0 | 0.0000 | `记录无 kind/instruction/ref_map（指针通道，非 render 层输出）` ×12 |
| **全体** | **165** | **152** | **0.9212** | 上两类合计 13 |

分集：A（card_flow 120）119/120=0.9917；B（gen_dispatch results_all 30）18/30=0.60（差额全是指针 12）；B′（results_adv 15）15/15=1.0。

## 5. 判定（三选一）+ 机制

**判定：T1 —— 解释可被约束为不可编（X0∧X1∧X2∧X3 全过，且 X4 的 E-free 违例 4576 ≫ 0）。**

机制（实测支撑）：解释文本不是"自由写出来再审"，而是**由封闭骨架表 + 源 `ref_map` 的逐字内容单元渲染出来的**，
再把**同一批 `render.py` 谓词**跑在它身上 —— 于是"编"只能发生在三个可核对的位置：
(1) 内容单元的 `(text,span)` 不在源 `ref_map`（回溯断言）、(2) `ref`/`span` 解析不回结构单元（`check_deref`）、
(3) 功能字面不是封闭表里的纯形式/逻辑词（`rule_a`/`word_face`）。三处都被实测的注入打穿 = 都会报错。
对照臂 E-free 之所以全崩，正是因为它必须用中文连接词/判断语（`这句…其实是说…作主语…通顺`），
这些字面既不在封闭表、也不在输入里 ⇒ `rule_a` 一条就报 2131 个问题。
**推断（非实测）**：把解释空间压到"逐字 + 封闭标点"是这套谓词的**必然代价**（见 §6）。

## 6. 命令 / 日志 / 产物 / 遗留与不确定

```
uv run python experiments/explain_card/run_explain.py      # 退出码 0（EXIT=0），wall 2.064 s（含 uv 启动），纯 CPU
git status --porcelain / git diff --stat -- src/           # src/ 空；只写 experiments/explain_card/
```
产物：`PREREG.md`、`explain_card.py`、`run_explain.py`、`results.json`、`logs/run.log`。

- **实测**：X0–X5 全部数字、注入报错原文、源复验 45/46、谓词调用/非空调用计数、覆盖率与未覆盖原因。
- **推断**：① "E-free 必然崩"只在**这一条固定模板**上实测（n=45、单模板，非多模板分布）；② "解释空间 = 逐字 + 封闭标点"由 `rule_a` 字面白名单推出，未做穷举证明。
- **未实现 / 明说**：① **人评（有用/易懂/正确）本单元不做**（边界句）；② 受约束臂上 `word_face_problems` 与 `item_problems` **恒空**（本集无逻辑词/数字骨架、内容单元构造上逐字）—— 它们"会咬人"只在 E-free 臂（`word_face` 45 次非空）与加测 ①′（`item` 45 次非空）里被观察到，**不能**据此说它们在受约束路径上有效；③ 解释卡的字段名是代码常量、不是生成内容，故"标签会不会编"**未被谓词覆盖**（未测）；④ `Declared`（意图卡显式声明）**未使用**；⑤ 指针通道 12 条**没有**解释（无 `ref_map/instruction`），不是"解释失败"而是"无结构可解释"；⑥ 对抗组 B′ 记录不带 `input`（只有 `id/expect`），其拒答解释只用 `reason`。

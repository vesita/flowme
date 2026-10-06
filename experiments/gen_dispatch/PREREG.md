# PREREG — 生成卡接入卡外手写调度层（`experiments/gen_dispatch/`）

> **本文在首次真卡运行之前写死**（mtime 早于 `results_*.json` / `logs/*.log`），跑完不改；
> 结果填进 `report.md`。
> 口径：「实测」= 本目录跑出来的数字；「推断」= 由数字推出；「构造规定」= 我这里人为规定的标签/规则。

## 0. 唯一问题

**把「生成卡」接进卡外手写调度层 `dispatch.py`**：调用方声明 `type` → `dispatch.plan()` 出
`Plan=[Step{...}]`，计划里**能出现生成卡这一步**；执行该步产出**统一格式记录**
`{kind, text, evidence, plan_step_id}`（生成型 `kind="text"`，**必须带 evidence**）。
指针 / 生成的切换是**手写规则**，绝不训 gate。

## 1. 组件（**全部只读 import，本实验不改任何一处**）

| 组件 | 用途 |
|---|---|
| `src/dtseek/tasks/dispatch.py` | 卡外手写调度层：`Signals/CallCard/Terminate/Step/Plan`、`INFO_BITS`、`REQUIRED_BITS`、`admissible_actions`、`decide`、`plan` |
| `src/dtseek/tasks/render.py` | 官方重整+解引用层：骨架表 16 条、规则 0/A/B、`check_structure` / `check_deref`、词面守卫、零信息新增、`render()` |
| `src/dtseek/tasks/dialogue.py` | 位表 `DIALOGUE_BITS/REQUIRED_BITS/BIT_KINDS`、`compose_reply`（指针通道复用）、`conf_of`、`observe` |
| `src/dtseek/tasks/corpus.py` | 语料入口（fail-closed，命中为空即抛） |
| `experiments/card_flow/{proposers,predicates,lexicon}.py` | 真卡候选提案、5 条筛选谓词、`type_of` 词典 |
| `experiments/two_channel_head/` | 只读引用其数字（骨架选择 acc 0.422@steps30 / 0.6396@steps1800） |
| 卡 | `checkpoints/cards/{person,pronoun,relation,sentiment}.pt` + `checkpoints/negation_accept_card.pt` + **只读复用** `experiments/cumulative_add/cards/ctrl_idiom_s42.pt`（成语卡，供给「动」槽） |

**卡清单依赖（fail-closed）**：声明的袋卡里任意一张挂载/调用失败 ⇒ **G 判假**
（不许少一张卡硬搜），`reason` 必须写明缺哪张卡；最终结局按 §2 的 R-CH 行序走
（R-CH2 降级指针 / R-CH3 拒答），两种结局都把缺卡写进 `reason`。

## 2. 开关规则原文（跑前写死；**不含任何学出来的量**）

```
规则 R-CH（指针 / 生成 / 拒答 的通道开关；手写 if-else，逐行照抄如下）
  输入：
    T = dispatch 规则表 decide()（R0–R6 那张 if-else 链）在本状态给出的**终止动作**
        ∈ {generate, pointer, reject}；T 的输入 = (声明的 type, 4 位三态, conf, steps)
    G = 「可合法渲染」谓词：官方骨架表里存在 (骨架, 指派) 使
        check_structure() == [] 且 render() 出 kind="text" 且 check_deref() == []
    P = 「指针有据」谓词：计划授权调用过的卡里，至少 1 张发射了 1 个合法切片
        （区间在原文内、逐字、类别非背景）
  行：
    R-CH0  T == reject                          ⇒ 通道 reject   （text 缺省，reason 必填）
    R-CH1  T == generate 且 G                   ⇒ 通道 generate （两道检查 0 问题才出 text）
    R-CH2  T == generate 且 ¬G 且 P             ⇒ 通道 pointer  （降级；reason 记「生成不可行」）
    R-CH3  T == generate 且 ¬G 且 ¬P            ⇒ 通道 reject
    R-CH4  T == pointer  且 P                   ⇒ 通道 pointer
    R-CH5  T == pointer  且 ¬P                  ⇒ 通道 reject   （指针无据不许编，G 也不许越权升级）
  行序固定 = 上表顺序；同一输入两次执行命中同一行。
```

**为什么不含学习量**：R-CH 读的三个量分别是 ① 手写规则表的输出（`dispatch.decide()` 是纯 if-else，
无权重、无分数、无阈值学习）、② 穷举**手写封闭骨架表** + 官方谓词（`check_structure`/`check_deref`）、
③ 卡切片的存在性（布尔）。R-CH 自身出现的常量只有 0（下标）与行序；
`dispatch`/`dialogue` 侧的 `CONF_MID=0.5 / CONF_HIGH=0.8` 是**手写常量**（构造规定，非拟合）。
**不引入**任何 `nn.Module` / 权重文件 / 学出来的分数参与通道判定；
执行 `rule_audit()` 会 `inspect.getsource` 读 R-CH 源码并断言不含 `weight|state_dict|model|score|predict(`
等字样，结果报进 H4。

**降级说明（构造规定）**：R-CH5 故意不许「指针无据就改走生成」—— 计划说的是"降级为指针"，
低置信状态下自己组织句子违反规则表的意图；这种输入按拒答处理，并单独报出条数。

## 3. 记录格式与拒答契约（跑前写死）

**统一记录**（核心 4 键，其余是溯源附加键）：

```
{ kind: "text" | "reject",
  text: 仅 kind=="text" 时存在（非空 str）；kind=="reject" 时**键缺省**（或 None）
  evidence: list（text 记录必须非空；reject 必须为空）
  plan_step_id: str —— "step:<i>"（i = 终止步在 Plan.steps 里的下标）或 "none"（拒答发生在出计划之前）
  + channel ∈ {pointer, generate, reject} / reason / plan[] / instruction / ref_map / type / cards_run }
```

**拒答契约 C1–C6**（`rules.contract_problems()` 逐条判；空 = 通过）：

- **C1** `kind ∈ {text, reject}`；`evidence`、`plan_step_id` 键必须存在。
- **C2（本任务的冲突裁决）** `kind=="reject"` ⇒ `text` **缺省或 None**，不许是任何字符串；
  `reason` 必填且非空白；`evidence == []`。
- **C3** `kind=="text"` ⇒ `text` 非空；`evidence` 非空；生成型还必须带 `instruction` 与非空 `ref_map`。
- **C4** `plan_step_id` 要么是 `"none"`（且 `plan == []`），要么 `== "step:i"` 且 `i < len(plan_steps)`。
- **C5** 指针型记录的每条 evidence span 必须在输入上逐字可解引用；生成型必须能被**独立重跑**
  `check_structure` / `check_deref` 判为 0 问题。
- **C6** 两边一致：同一个 `contract_problems()` 同时跑
  ① 本层出口记录、② `render.render()` 的**原始** reject 记录、③ `dialogue.respond()` 的**原始** reject 记录。
  ②③ 的原始记录**按本契约判必不通过**（`render.py` 的 reject 仍带 `（拒答）…` 占位串）——
  这是实测的冲突证据；经 `rules.seal()` 在**唯一出口**归一后必须通过。
  通过率与原始违例条数都报进 H2（**不改 `render.py`**）。

## 4. 抽样（跑前写死；不许 `fs[:N]`）

1. **语料**：`corpus.resolve_corpus_files()`（glob 空即抛）→ **逐文件自算汉字占比**
   （汉字数 / 非空白字符数，整文件流式）→ **只保留占比 ≥ 50%**（`dev-notes/19` 的阈值）。
   被过滤掉的文件名与占比**全量报出**；实际使用的文件清单**全量报出**（含行数）。
2. **句子**：去 `用户：/模型：/user:/assistant:` 前缀 → 按 `[。！？!?；]` 切 →
   保留 `6 ≤ len(s) ≤ 40` 且汉字数 ≥ 6（与 `card_flow.load_samples` 同一条规则）。
3. **随机样本 N = 1000**：全语料句子池 **reservoir sampling，seed = 42**（不依赖文件返回顺序）。
4. **富集样本 M = 400**：句中含 ≥1 个 `card_flow.lexicon.IDIOM_TYPE` 词条（手写词典谓词）——
   因为官方骨架表 16 条**全部需要动槽**，而真实卡库存里没有动词卡（`lexicon.py` 注释：夹具卡），
   「动」只能来自成语卡。**这是选择偏倚，必须单列口径**：
   随机样本负责报通道比例与覆盖率；富集样本只负责提供**生成样例**，两条口径不合并。
   不足 400 ⇒ 按 `M = 800 → 1600` 阶梯扩样，两次都报；仍不足则用实际条数。
5. **max_naive 假卡批**：随机样本按 seed 序取前 200 条（假卡 = `card_flow.fake_propose`，词典匹配，免费）。
6. 所有样本**固定 seed=42**、两集合互不重叠（富集集从随机集里剔除）。

## 5. 判据（跑前写死）

| # | 判据 | 门槛（可执行定义） |
|---|---|---|
| **H1** | 同一输入的计划逐字可复现 | (a) `dispatch.plan()` 在**全状态空间 4860 态**上连跑两遍，`Plan.as_tuple()` 全等；(b) 端到端样本 100 条连跑两遍，`plan[]` 逐字全等。任一不等 = 不过 |
| **H2** | 拒答显式且契约一致 | 全部记录过 C1–C6；**reject 的 `text` 缺省率 = 100% 且 `reason` 非空率 = 100%**；**「正例（该通道有合法输出）却不产出 ⇒ 误拒」条数 = 0**；**「无合法输出却出 text」条数 = 0**；`render`/`dialogue` 原始 reject 违例条数如实报（归一后必须 0） |
| **H3** | 生成输出过 render 两道检查 | 每条 `channel=generate ∧ kind=text` 的记录：**独立重跑** `check_structure` 与 `check_deref`，两者的**问题总数 = 0**（逐条报 n 与 max） |
| **H4** | 开关规则可判 | 报出 `generate / pointer / reject` 选中比例（随机批 + 富集批分别报）；`rule_audit()` 断言 R-CH 源码不含学习量（断言结果 + 源码原文进报告） |
| **H5** | 对抗组逐条实测 | §6 的 A1–A12 **每条**都报 `预期 / 实测 / pass`；全 pass 才算过 |
| **H6** | 老卡不受影响 | 本实验**不改 `src/`**（判定：`git status --porcelain src/` 为空 + `git diff` 为空）；全库 `pytest -q` 原始输出进报告；若实际改了 `src/` ⇒ 改判为逐卡 Δ 对比各自噪声带（pronoun 2.83 / sentiment 0.41 / relation 1.39 / person 0.33 pt） |

**判定**：H1–H5 全过 ⇒ **接入成立**；过 3–4 条 ⇒ **部分成立**；≤2 条 ⇒ **不成立**。
不许硬选；每条给实测数字。H6 不进三选一（纪律项），但**必须报**。

**样例条目**（交付物，非判据）：真实生成样例 **5–10 条**，每条给
输入 → 计划 → 结构指令 → 渲染文本 → 引用映射片段 → 通道；不足 5 条如实报并给原因。

## 6. 对抗组（跑前写死；夹具输入，逐条实测）

| id | 打什么 | 预期 |
|---|---|---|
| A1 | 题元互换（S01 施事/受事对调，`我打你`↔`你打我`） | `reject`，reason 含「题元方向冲突」 |
| A2 | 无据逻辑词（S13 `因为…要…`，输入不含 因为/要） | `check_structure` 非空 ⇒ `reject` |
| A3 | 有据逻辑词（输入逐字含 因为…要） | 不因词面守卫被拒（正对照） |
| A4 | 缺块（指派少一项） | `reject`，reason 含「槽位数不等 / 缺块」 |
| A5 | 类型不符（形块进动槽） | `reject`，reason 含「类型不匹配」 |
| A6 | 数字改动（含数字的袋块 text 被改） | `reject`（逐字/evidence 不齐备） |
| A7 | 否定改动（否定片段被改字） | `reject`（逐字/evidence 不齐备） |
| A8 | 专名改动（人名片段被改字） | `reject`（逐字/evidence 不齐备） |
| A9 | 越界 span | `reject` |
| A10 | 骨架 id 不存在（封闭表） | `reject`，reason 含「骨架 id 不存在」 |
| A11 | 同一块指派到两个槽 | `reject`，reason 含「被指派到多个槽」 |
| A12 | 只复述子集换方向（指派不随 span 升序） | `reject`，reason 含「不随原文 span 升序」 |
| A13 | 开关侧：`type` 不声明 / 不认识 | `reject`，`text` 缺省、`reason` 非空 |
| A14 | 开关侧：空输入 | `reject`，`text` 缺省、`reason` 非空 |

## 7. `max_naive`（**必报**：任何"生成卡有用"的说法都并列免费规则基线）

| 基线（全免费、不调卡/不训模型） | 报什么 | 与我方的对账口径 |
|---|---|---|
| **N1 全生成**（不管计划终点，一律走生成通道） | 合法率 `P(G)`、产出率、非法输出条数（`¬G` 却输出） | 我方规则非法输出必须 = 0 |
| **N2 全指针**（一律走指针通道） | 合法率 `P(P)`、非法输出条数（`¬P` 却输出） | 同上 |
| **N3 按 type 静态映射**（`plain → generate` / `plain → pointer` 两条常数规则） | 同 N1/N2（在本批上与 N1/N2 等价，仍单列报） | 同上 |
| **N4 假卡（词典）提议者** + 同一渲染层 | 同一批句子上产出的**过两道检查的生成样例条数** | 我方真卡样例条数若 ≤ 它 ⇒ 写明「真卡没打过词典规则」 |
| **N5 echo 复述基线**（原句原样输出） | 过 H3 两道检查的比例（预期 0，因为它没有 instruction/ref_map） | 说明"逐字可回溯"不等于"合法生成" |

`max_naive` = 上表中**最强免费基线**的对应数字，逐项与我方并列；
不并列 `max_naive` 的结论一律不采信（本项目已五次出现"卡没打过免费规则"：
锚定逐字 1.00 / 价值卡词表 1.00 / B 族表内记忆 / D4 `lex_member` 0.765；
唯一例外 `two_channel_head` 骨架 0.6396 > `max_naive`+2SE）。
**已知边界**：本实验的生成通道**本身也是一条免费规则**（穷举手写骨架表），
所以 N1/N2/N3/N5 是"我这条规则 vs 别的免费规则"，N4 是"真卡 vs 词典卡"——两组口径分开写，不许互相冒充。

## 8. 纪律

- **只写** `experiments/gen_dispatch/`、`logs/`、`/tmp`；**不改** `src/`、`card_flow/`、`two_channel_head/`、
  其它 `experiments/`、`training/`、`tests/`；不 `commit/stash/checkout/restore/clean`。
- 一律 `uv run python`；GPU 开跑前看 `ps` 与 `systemctl --user is-active 'dtseek*'`，忙就排队，绝不杀他人进程。
- 日志：`logs/gen_dispatch_<mode>_<时间>.log`；结果：`experiments/gen_dispatch/results_*.json`。
- 单跑要短：随机批 + 富集批 + 对抗组 + H1 各自独立成一次运行（各自可单独重跑）。

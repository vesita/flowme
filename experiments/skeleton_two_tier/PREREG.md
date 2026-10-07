# PREREG —— 两级骨架机制（Tier-1 闭合骨架表 + Tier-2 自由句式兜底）

**唯一问题**：两级机制能把真卡 120 句上的覆盖率从 **30.0%（36/120）** 抬到多少，
且四道保证（G1 拒答显式 / G2 零信息新增 / G3 引用映射铺满 / G4 逐字可复现）是否仍成立？

本文件 **mtime 必须早于首次运行**（驱动里有 `assert prereg_mtime < t0`）。
跑后不改判据；只许在 report 里补数字与遗留。

## 0. 范围与纪律

- **只写**：`experiments/skeleton_two_tier/`、`logs/`、`/tmp`。
- **只读 import**：`src/dtseek/tasks/render.py`、`src/dtseek/tasks/dialogue.py`（G1 归一演示）、
  `experiments/card_flow/`（Tier-1 五段流与骨架表）、`experiments/{gen_dispatch,two_channel_head,struct_supervision,training}/`、`tests/`、`dev-notes/`、`methodology/`。
- **不改 `src/`**（本轮零改动）；不 `git commit/stash/checkout/restore/clean`。
- 一律 `uv run python`；带 `PYTHONDONTWRITEBYTECODE=1`，避免往只读目录写 `__pycache__`。
- 长跑（若有）用 `systemd-run --user --unit=… --collect --setenv=PYTHONUNBUFFERED=1`；本轮预计**单跑 < 2 分钟**，前台短跑。

## 1. 语料与样本（跑前写死）

- 语料按 `dev-notes/19` **显式按文件名取**，**不许依赖 `fs` 返回顺序、不许 `fs[:N]`**。
- 本实验用文件（均在 dev-notes/19 的 ✅ 档，汉字占比 ≥ 50%）：
  - `lccc_dialogue.txt`（汉字占比 85.4%，dev-notes/19 实测）
  - `dailychat_dialogue.txt`（82.9%）
  - **驱动断言这两个文件存在**（缺一个 ⇒ fail-closed 报错退出，不走任何退化分支）。
  - ⛔ 明确不用：`gsm8k_cot_dialogue.txt`(5.9%)、`code_alpaca_dialogue.txt`(0.0%)。
- 抽样规则**与 `card_flow` 完全一致**（保证与 30.0% 基线是同一批句子）：
  去 `用户：/模型：/user:/assistant:` 前缀 → 按 `[。！？!?；]` 切 → 取 `6 ≤ len ≤ 40` 且汉字 ≥ 6
  → 每文件 `random.Random(42).sample(pool, 60)` ⇒ **120 句**。
- **一致性断言**：真卡跑完后，把本驱动的 120 句与 `experiments/card_flow/results_real.json`
  里每条记录的 `input` 逐句比对，**必须完全相同**（不同 ⇒ 报错退出，不许拿不同批次比覆盖率）。

## 2. Tier-1（闭合骨架）口径

- 复用 `experiments/card_flow/` 的五段流（只读 import）：`proposers.real_propose` →
  `predicates.filter_candidates` → `flow.compose`（**25 条骨架表穷举 + 槽位指派搜索**）→
  `render.render_structure` → `render.deref_verify`。
- **不用 `two_channel_head` 已训骨架头**，理由（跑前写死）：它的输出空间是 40 模板/30 有效类，
  槽位类型签名要求 `动`；`card_flow` 报告 X1/X2 实测**该表与本表 id 空间冲突且全部含动槽**
  （120 句覆盖 0/120、36/36 类型签名不命中）⇒ 用它当 Tier-1 选择器只会把 Tier-1 打成 0。
  本轮 Tier-1 = **确定性穷举**（与 card_flow 30.0% 同口径）。
- Tier-1 成功 ⇒ 产出定稿，**不再进 Tier-2**；Tier-1 拒答 ⇒ 进 Tier-2。

## 3. Tier-2（自由句式 = 自由的句法 + 受锁的词）跑前规格

### 3.1 自由度的来源（必须交代，跑前写死）

| 自由项 | 来源 | 锁 |
|---|---|---|
| **槽的类型序列与长度**（句式主体） | 输入句上「被筛为有效」候选的**类型库存**（任意长度、任意类型，不查闭合表） | 每槽类型必须 = `lexicon.type_of` 重算值；未知类型（成语）不许进槽 |
| **词序** | 输入原文 **span 升序**（复述） | **真卡无题元 ⇒ fail-closed 到「只复述、不换方向」子集**，本轮**词序自由度关闭**（不猜方向）；互换注入必拒 |
| **连接（间隙功能词）** | 封闭功能词表 ∪ 输入逐字实据 | 见 3.2 规则，逻辑词必须输入有据 |
| **句末式** | 输入末字符（是句末标点则逐字复用）否则 `。` | 封闭表 |

⇒ **自由句式 ≠ 自由生成**：自由的是「句式/槽序实例化/连接」，**词全部受锁**
（内容词逐个回溯到被筛为有效的候选；功能词来自封闭表或输入有据；输出可唯一解析回槽序+功能词序列）。

### 3.2 生成规则（确定性，跑前写死）

1. **片段选择**：候选池 = ②筛选通过且 `type_of() ∈ {名,动,形,否,数}` 的候选；
   按 `(s0, −长度, card, cid)` 排序、同 span 去重（留键序第一条）；
   **贪心 leftmost-longest 取互不重叠的升序序列** ⇒ 全部片段（**不截断、不挑肥拣瘦**）。
2. **间隙连接**（在第 i 与 i+1 片段之间，取输入里这两段之间的原文 gap）：
   - ① gap 内出现 `LOGIC_WORDS = {因为,所以,但,要,能,可以}`（最长优先、再按表序）⇒ 用它（**逻辑词，输入有据**）；
   - ② 否则 gap 内从左到右第一个 ∈ `{，、；：和与}` ⇒ 用它（**形式词/标点**）；
   - ③ 否则**不插连接**（两片段相邻）。
3. **句末**：输入末字符 ∈ `{。！？!?…}` ⇒ 逐字复用；否则 `。`。
4. **结构指令**：`{tier:2, skeleton:"T2", slots:{p_i: 类型}, assign:{p_i: cid}, refs, evidence, funcs:[f_1..f_n], order:"span_asc"}`；
   `f_i`（i<n）= 间隙连接（可为空串），`f_n` = 句末（必非空）。
5. **渲染** = 纯函数：`片段_1 + f_1 + … + 片段_n + f_n`，同时产出 `ref_map`
   （内容条目 `cls=content` 带 `cid/span`；功能条目 `cls=formal|logic` 带 `unit_id=t2#k`、`span=None`）。

### 3.3 两道检查（结构侧 / 解引用侧）+ **解析器**

- **结构侧** `t2_structure_problems`：schema（槽/功能词数量齐、无多余、ref 集合一致、无重复指派）/
  逐字（span 落在输入内且 `input[s0:e0]==surface`）/ 属于被筛有效集 / 类型重算一致 /
  **升序 + 互不重叠**（方向安全）/ 功能词 ∈ 封闭表 + 类别正确 + 逻辑词输入有据。任一不过 ⇒ 拒答。
- **渲染后**：`t2_deref_verify` 重走指令核对输出（丢块/改面 ⇒ `ref_render_mismatch`；
  多吐字符 ⇒ `unmapped_output`；映射集 ≠ 指派集 ⇒ `ref_set_mismatch`）。
- **解析器** `t2_parse(text_out, input_text, valid_index)`（**不读结构指令，独立实现**）：
  - 文法：`输出 := 内容 (间隙连接 内容)* 句末`，**必须以内容开头、以句末结尾、内容≥1**；
  - 内容 token 必须是某条**有效且已定型**候选的 surface，并记录其**输入 span**；
  - 全程强制 **span 严格升序 + 互不重叠**（与结构侧同一条方向约束）；
  - 间隙连接 ∈ 封闭表 ∪ {空}，逻辑词须在输入中逐字有据；句末 ∈ `{。！？!?…}`；
  - **唯一性**：枚举全部合法解析（上限计数到 2 即停）——
    **0 条 ⇒ `parse_no_way`；≥2 条 ⇒ `parse_ambiguous` ⇒ 该产出「解析不出 ⇒ 不计分」**（A 臂口径）。
  - 解析结果必须与渲染器的 `ref_map` **逐条相等**，否则拒答（`ref_render_mismatch`）。
- **G3 判定**因此仍可判：`ref_map` 由**独立解析器**复原，且各区间**恰好铺满**输出。

### 3.4 降级链（跑前写死；保证 N1 ≤ 两级）

首选句式 R0 = 3.2 的全片段版；若结构/解析不过 ⇒ **依次去掉末尾一个片段**重算连接与句末，
直到只剩**最左单片段**（R_{n-1}）。仍不过 ⇒ 拒答 `tier2_all_attempts_failed`。
- **C2 主数 = R0（首选句式）的可解析率**；另报降级深度分布与最终产出率（构造性 100%）。
- 每次尝试都独立过全部检查；**绝不产出解析不出的文本**。

### 3.5 拒答原因码（fail-closed，跑前写死）

`tier2_no_valid_candidate` / `tier2_type_inventory` / `tier2_schema_missing_slot` /
`tier2_schema_extra_slot` / `tier2_ref_set_mismatch` / `tier2_duplicate_ref` /
`tier2_span_not_verbatim` / `tier2_not_in_valid_set` / `tier2_type_mismatch` /
`tier2_reorder_unjustified` / `tier2_span_overlap` / `tier2_func_not_in_table` /
`tier2_logic_no_evidence` / `tier2_func_count_mismatch` / `tier2_unmapped_output` /
`tier2_ref_render_mismatch` / `tier2_parse_no_way` / `tier2_parse_ambiguous` /
`tier2_all_attempts_failed`。

## 4. 判据（跑前写死；三选一，不许硬选）

| # | 判据 | 门槛 |
|---|---|---|
| **C1** | **覆盖率**（两级 vs 仅 Tier-1，真卡 120 句，`n_ok/120`） | 报**绝对数**与前后对比（基线 30.0% = 36/120、拒答率 70%）；**不设假门槛，报数** |
| **C2** | **可解析率**（Tier-2 首选句式 R0） | **分母 = 构造出 R0 的句子**（进入 Tier-2 且有 ≥1 条已定型有效候选）；同时并报更严分母（= 进入 Tier-2 的全部句子）。解析不出（0/≥2 解析）**不计分**；另报降级链深度分布 |
| **C3** | **G1–G4 各 0 违例** | G1 拒答记录**无 `text` 字段**（官方 `render.py`/`dialogue.py` 原始带占位 ⇒ `contract.normalize` 统一契约归一后判；**非占位 text 不许被归一掉**）；G2 内容词 100% 回溯到被筛有效候选；G3 `ref_map` 恰好铺满 + 独立解析复原一致；G4 两遍完整流（含真卡重推理）逐字相同 |
| **C4** | **对抗组**（Tier-2 14 条 + 复跑 card_flow Tier-1 14 id） | 题元方向不许互换；无据逻辑词必拒；数字/否定/专名不许改；缺块/类型不符 fail-closed；**故意越界/歧义必须被抓**（逐条 reason 命中，**0 mismatch**） |
| **C5** | **`max_naive` 并列** | 本实验**无生成质量类指标**（G1–G4 是 0/1 违例结构判据）⇒ 形式上 N/A；**仍并列两条免费规则**：N0 = 直接复述原文、N1 = 单片段（最左有效片段）+ 同一句末规则（拒答码 `tier2_*` 同族，另记 `tier2_naive_structure`/`tier2_naive_deref`）。**低于就如实写「未超过免费规则」**；并注明 **N1 就是降级链末档 ⇒ 两级 ≥ N1 是构造性成立，不是实测优势** |
| **C6** | Tier-2 占比与被拒原因分布 | 报数 |

**判定三选一（可操作规则，跑前写死）**：

- **两级成立** = `两级 C1 − 仅 Tier-1 C1 ≥ 20pp` **∧** C3 四条各 0 违例 **∧** C4 全中（0 mismatch）
  **∧** 增量里 `n_frag=1`（单片段）的产出 **< 80%**（抬升不能全靠与 N1 同款的退化产出）。
- **部分成立** = 满足前三条，但增量里单片段产出 **≥ 80%**（⇒ 抬升主要来自「退化到免费规则同款」，如实写）。
- **不成立** = 抬升 < 20pp，**或** C3 任一条有违例，**或** C4 有 mismatch。

## 5. 对抗组（跑前写死，预期 reason 一一对应）

**Tier-2（14 条）**：
T01 题元方向互换（夹具有题元，指派倒序）⇒ `tier2_reorder_unjustified`；
T02 无据逻辑词（`funcs` 塞 `因为`，输入无）⇒ `tier2_logic_no_evidence`；
T03 数字改动（span 文本 `3` 声明 `4`）⇒ `tier2_span_not_verbatim`；
T04 专名改动（surface 改字）⇒ `tier2_span_not_verbatim`；
T05 否定块丢失（渲染时丢片段）⇒ `tier2_ref_render_mismatch`；
T06 缺块（`assign` 少一个槽）⇒ `tier2_schema_missing_slot`；
T07 类型不符（`slots` 声明 `形`、候选是 `名`）⇒ `tier2_type_mismatch`；
T08 越界 span（`e0=99`）⇒ `tier2_span_not_verbatim`；
T09 表外功能词（`funcs` 塞 `然后`）⇒ `tier2_func_not_in_table`；
T10 用未筛为有效的候选 ⇒ `tier2_not_in_valid_set`；
T11 未知类型进槽（成语 `type=None`）⇒ `tier2_type_mismatch`；
T12 渲染注入（输出尾追加字符）⇒ `tier2_unmapped_output`；
T13 **解析歧义**（故意造两解文本）⇒ `tier2_parse_ambiguous`（**解析不出 ⇒ 不计分**）；
T14 正例（必须渲染且逐字 = 期望）⇒ `ok`。
**Tier-1 复跑**：`card_flow.run_adversarial()` 的 14 id / 17 次执行，预期与 `card_flow/report.md` §4 一致（全中）。

## 6. 命令与产物（跑前写死）

```
uv run python experiments/skeleton_two_tier/run_two_tier.py --mode selfcheck   # 假卡 + 解析/审计自检 + 对抗组（无 GPU）
uv run python experiments/skeleton_two_tier/run_two_tier.py --mode adv         # 对抗组（Tier-2 + Tier-1 复跑）
uv run python experiments/skeleton_two_tier/run_two_tier.py --mode real        # 真卡 120 句 × 2 遍（G4 含重推理）
uv run python experiments/skeleton_two_tier/run_two_tier.py --mode official    # 官方层交叉核对（G1 归一 + 渲染一致性）
```
产物：`results_{selfcheck,adv,real,official}.json`、`report.md`、日志 `logs/two_tier_<mode>.log`。

## 7. 跑前已知边界（不许事后当发现）

- **未实现**：Tier-1 骨架头的模型选择（用穷举，见 §2）；题元标签在真卡上不存在（题元互换只能用夹具判）；
  数字卡/动词卡不存在（`数`/`动` 槽在真卡上不出现）；语义连贯性不在四保证内、不测。
- **口径判断**：Tier-2 输出允许**单片段句**（如 `糟。`）——它与 Tier-1 已被接受的 `我。`（S01）同构；
  因此 C1 的抬升必须与 C5 的 N1 并列读，报告里不许把「覆盖率抬升」单独当成绩。
- **推断待验**：解析唯一性可能因「同 surface 不同 span」而损失覆盖 —— 实测数见 C2。

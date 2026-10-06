# card_flow 实验报告 —— 端到端卡片流 + 文本生成（判据见 PREREG.md）

跑前判据：`experiments/card_flow/PREREG.md`（mtime 2026-10-07 01:23:43 < 首跑 01:27:55，驱动里有断言）。

## 1. 五段流的实现

| 段 | 职责 | 落点（函数） |
|---|---|---|
| ① 候选 | 指针卡在真句上抽切片（类别+span），内容词天生可回溯 | `proposers.real_propose`（真卡 `MultiTaskEngine.predict`）/ `proposers.fake_propose`（假卡：词典匹配，先跑通管线） |
| ② 筛选 | 5 条可核对谓词全过才算 valid | `predicates.filter_candidates`：F1 区间合法（含空区间 s0==e0）/ F2 非空 / F3 类别规范名且非背景 / F4 不跨标点 / F5 逐字回溯 |
| ③ 重整 | 骨架穷举 + 槽位指派 ⇒ **结构指令**（不含文本） | `flow.compose`（表序 × 候选池积，含题元/顺序/去重/逻辑词门禁） |
| ④ 渲染 | 确定性纯函数 | `render.render_structure` |
| ⑤ 解引用 | 结构侧 schema/evidence + 解引用侧覆盖，留存引用映射 | `predicates.structure_problems`（结构侧）+ `render.deref_verify`（文本每字符必须落在映射片段或骨架字面上） |

骨架表 **25 条**（上限 40，`skeletons.SKELETONS`，import 时跑 `audit()` 规则 A 审计：字面只许
标点 + 们/吗/和/与，逻辑词 因为/所以/但/要/能/可以 必须"原文有据"，含逻辑词的骨架 4 条：S18/S19/S20/S21）。
槽位类型 = `名/动/形`（§7.7 规定）+ 扩展 `否`（否定卡切片）+ 扩展 `数`（**现有卡无数字切片，仅夹具**）；
题元取值域 = `施事/受事/时`（槽位声明；真实卡不产题元 ⇒ 走"只复述不换方向"分支）。

`src/dtseek/tasks/render.py` 时间线：**01:03 全仓 find 不存在 ⇒ 我自己实现最小版**；他方 01:18 建成该文件；
首跑 01:27 时它已存在。判据跑用的是我的最小版；随后 **import 官方层做交叉核对**（`crosscheck_official_render.py`），
用到：`BagItem/Instruction/Declared/SKELETONS/render/check_structure/check_deref/word_face_problems/
rule_a_problems/slot_schema_problems/validate_skeleton_table`。

## 2. 真实生成样例（真卡 5 张，语料真句子，原文照抄）

| 输入 | 结构指令 | 渲染文本 | 引用映射 片段 ↦ ref ↦ span（卡/类别/类型） |
|---|---|---|---|
| 我让它发挥最大的价值 | S02 {n1:c1,a1:c3} | 我让发挥最。 | 我让↦c1↦[0,2](pronoun/第三人称/名)；发挥最↦c3↦[3,6](sentiment/愤怒/形) |
| 你才得早睡呢 | S02 {n1:c1,a1:c5} | 你睡。 | 你↦c1↦[0,1](person/人物2/名)；睡↦c5↦[4,5](sentiment/悲伤/形) |
| 我们这前两天沙尘暴，天气要多糟糕有多糟糕，真是烦人 | S02 {n1:c0,a1:c4} | 我糟。 | 我↦c0↦[0,1](pronoun/第一人称/名)；糟↦c4↦[18,19](sentiment/愤怒/形) |
| 死了记得把你家那个送我 | S11 {n1:c4} | 那个送吗？ | 那个送↦c4↦[7,10](person/人物2/名) |
| 明天给你们讲 | S06 {n1:c1,n2:c3} | 天和你。 | 天↦c1↦[1,2](person/人物1/名)；你↦c3↦[3,4](pronoun/第三人称/名) |
| 呵呵是的平均18度挺冷的 | S11 {n1:c1} | 18度挺吗？ | 18度挺↦c1↦[6,10](pronoun/第三人称/名) |
| 好吧，幻想还是可以的 | S11 {n1:c1} | 幻想还是吗？ | 幻想还是↦c1↦[3,7](pronoun/第三人称/名) |
| 好大家是不是都要上班啊 | S11 {n1:c4} | 要上吗？ | 要上↦c4↦[7,9](pronoun/第二人称/名) |

⚠️ 这些句子**语义不通**——但每个内容片段都逐字来自原文、都能回溯到被筛为有效的候选（G2/G3 过）。
四条不变量只管结构，**不管提案卡切片对不对**；见 §6。

## 3. G1–G4 逐条实测（真批 120 句 + 对抗组 17 次执行）

| # | 实测 | 反例证据 |
|---|---|---|
| G1 拒答显式 | 必拒 **13/13 = 100%**，reason 全部命中预期；正例 **4/4 无误拒**；真批 84 条 reject 中**带 `text` 字段的 = 0** | 缺块→`schema_missing_slot`；类型不符→`type_mismatch`；越界→`span_not_verbatim`；无据逻辑词→`logic_no_evidence` |
| G2 零信息新增 | 真批 36 条成功输出，独立事后复核 `audit_record` 违规 **0**（每片段 == 原文 span，且 cid ∈ 有效候选集） | **故意越界被抓**：A12 候选 e0=99 → 结构期 `span_not_verbatim` 拒；A6 数字 span 文本"3"但声明"4" → 拒 |
| G3 引用映射完整 | 36/36：映射区间 + 骨架字面**恰好铺满**输出（无洞无尾），映射数 == 槽位数，链 片段↦ref↦span 全通 | **故意注入被抓**：A13 毒渲染追加"因为" → `unmapped_output` 拒；A8 毒渲染丢掉否定块 → `ref_render_mismatch` 拒 |
| G4 逐字可复现 | 36/36 两次完整流（含真卡重推理）**逐字相同**，不一致 **0** | 假批 41/41 同样 0 |

假卡批（同一 120 句）：ok=41，G2/G3/G4 违规同为 0。

## 4. 对抗组逐条实测（14 个 id / 17 次执行）

| id | 预期 | 实测 |
|---|---|---|
| A1 题元方向（有标签，互换） | 拒 `theme_conflict` | ✅ 拒，reason 命中 |
| A2 题元方向（无标签，互换） | 拒 `reorder_unjustified` | ✅ 拒 |
| A3 你打我（两分支） | 互换必拒 + 正确必渲染 | ✅ 拒 / ✅ 渲染 `你打我。` |
| A4 无因果依据插 `因为` | 拒 `logic_no_evidence` | ✅ |
| A5 输入含 `但` 用 `但` 骨架 | 必须渲染（不误拒） | ✅ `我难过，但你高兴。` |
| A6 数字改动 | 拒 `span_not_verbatim` | ✅ |
| A7 专名改动 | 拒 `span_not_verbatim` | ✅ |
| A8 否定块丢失 | 拒 `ref_render_mismatch` | ✅ |
| A9 情态 `要` 无据 | 拒 `logic_no_evidence` | ✅ |
| A10 缺块 | 拒 `schema_missing_slot` | ✅ |
| A11 类型不符 | 拒 `type_mismatch` | ✅ |
| A12 越界 | 筛选期 F1 / 结构期拒 | ✅ 两处都抓 |
| A13 渲染注入 | 拒 `unmapped_output` | ✅ |
| A14 正例 | 必须渲染 | ✅ `我打你。` |

## 5. 骨架覆盖率 / 拒答率 / 前向时间

- **覆盖率（真卡，n=120）**：C1 = **0.3000**（36/120）；C2 = **0.3051**（36/118 有有效候选的句）。
  假卡：C1=0.3417、C2=0.5694。
- **拒答率（真卡）**：0.70（propose 0 / filter 2 / compose 82 / render 0 / deref 0）。
  82 条 compose 拒答的类型库存：`have:形`=51、`have:否,形`=15、`have:否`=6、`have:-`=10（有效但类型查不到=成语）。
  **含有效"名"候选的句子 = 36 = 全部成功句**（"有名候选却仍拒答" = 0）⇒ 名槽是唯一绑定约束。
- **骨架表的死区**：含 `动` 槽的骨架在真卡上**一次都没成功**（动候选 0 条）；含 `数` 槽的 2 条同样 0 条（无数字卡）。
- **前向时间（真卡 5 张 × 120 句，GPU，与他方训练作业并发，跑时 GPU 空闲 7.47/8.57 GB）**：
  negation 8.239 ms/句（p50 3.412，max 563 = 首次预热）、relation 4.847、pronoun 3.733、sentiment 3.325、
  person 3.321 ⇒ **合计 23.465 ms/句**。
  **逐步对账**：120 句 × 23.465 = 2.82 s ＋ 36 条 G4 重跑 × 23.465 = 0.85 s ⇒ 期望 **3.66 s**，实测 wall **3.53 s**（−3.5%）。
  每句 5 卡各 1 段（`mean_segments=1`），解码步预算 = 段数 × max_steps（person 16，其余 4）；
  含编码的"每预算步"耗时：person 0.208 / pronoun 0.933 / relation 1.212 / sentiment 0.831 / negation 2.060 ms。

## 6. 判定 + 卡在哪一段

**判据层面：G1–G4 全过 + 对抗组 14 个 id 全部符合预期 ⇒ 按 PREREG §8 判「卡片流端到端成立（在当前骨架表规模下）」。**

但**卡在①候选区（实测，非推断）**，机制如下：

1. **指针卡切片质量差**（`diag_card_quality.json`，后验诊断，不进判据）：通过②筛选的候选与本卡训练词典对齐率
   pronoun **5/34=14.7%**、negation 6/28=21.4%、person 1/5、relation **0/152**、sentiment **0/95**；
   pronoun 134 条 anchor 里 **92 条是空区间**（s0==e0，被 F1/F2 抓掉），relation 41 条跨标点（F4 抓掉）。
   ⇒ 通过筛选的切片"结构上全合法、语义上多为胡话"，于是生成的是**可回溯的胡话**（§2 的表）。
   **谓词不许判语义**（§7.2：能用可核对谓词判的才交给谓词）⇒ 这条不能靠放宽/加严谓词修，只能修卡。
2. **类型库存缺"动"与"数"**：现有 5 张卡不产动词/数字切片 ⇒ 骨架表 25 条里含动槽的 14 条、含数槽的 2 条
   **永远填不满**；覆盖率的分母里 82 句缺"名"（卡片没圈到人称/人物）。
3. **题元标签无来源**：5 张卡的 anchor 没有题元字段 ⇒ 真实流恒走"只复述、不换方向"分支；
   带题元的方向测试只能用夹具角色块（A1/A3/A14），且**官方层独立复现了同一结论**（见下）。

**官方 `src/dtseek/tasks/render.py` 交叉核对（只读 import）**：
- X1：官方骨架表 16 条（**全部含动槽**）在 120 句真卡候选袋上覆盖 **0/120**（18 句袋空、102 句填不满）；
- X2：我的 36 条成功输出，类型签名在官方表 **0/36 命中**，且 **36/36 是"同 id 不同签名"**（两表 id 空间冲突：我的 S02=`[名][形]。`，官方 S02=`[1][2][3]吗？`）⇒ **两张骨架表不相交，必须统一 id 与类型域**；
- X3：官方层题元方向 —— 正确指派出 `我打你。`/`你打我。`，互换一律 `题元方向冲突（规则 B）` 拒 ✅；
- X4：官方 `word_face_problems` —— `因为/要` 无据拒、有据不误拒 ✅；`validate_skeleton_table()` 无问题。
- **口径差异**：官方 `render()` 的 reject 记录**仍带 `text` 字段**（内容是"（拒答）…"占位串）⇒ 按我 PREREG G1 的
  字面判据（reject 不许有 text 字段）官方层**不满足**；需要上层约定"拒答时 text 为空/缺省"。

## 7. 原始命令 / 日志 / 产物 / 遗留

```
uv run python experiments/card_flow/run_card_flow.py --mode adv    # 17 次执行
uv run python experiments/card_flow/run_card_flow.py --mode fake   # 120 句，0.05 s
uv run python experiments/card_flow/run_card_flow.py --mode real   # 120 句，3.53 s
uv run python experiments/card_flow/crosscheck_official_render.py
uv run python experiments/card_flow/diag_card_quality.py
```
产物：`PREREG.md`、`results_adv.json`、`results_fake.json`、`results_real.json`、
`results_crosscheck_official.json`、`diag_card_quality.json`、日志 `logs/real_run.log`。
（**`logs/` 被他方会话独占 ⇒ 我的日志落在 `experiments/card_flow/logs/`**；只写 `experiments/card_flow/`，`src/` 只读。）

**遗留与不确定（实测/推断分开）**
- **实测**：G1–G4 数字、覆盖率、拒答率、前向时间、卡片切片质量、官方层交叉核对。
- **推断**："覆盖率瓶颈在候选类型库存而非骨架表"由 `have:` 分布 + "有名候选却拒答=0" 支撑，未做消融。
- **未实现（不假装）**：①"选骨架+填槽"的**提案模型未训练**（本轮由确定性穷举给出）⇒ §7.8 的 S1/S2/S3 **N/A**；
  ②**意图卡 `Declared` 未实现**（逻辑词只走"原文有据"这一条路）；③数字卡/动词卡不存在（`数`/`动` 的部分分支仅夹具可跑）；
  ④题元标签仅夹具具备；⑤语义连贯性**不在四不变量内、本轮未测**。

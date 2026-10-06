# 输出侧不变量实测报告：拒答出口 / 闭合性 / 可溯源

预注册 `experiments/out_invariants/PREREG.md`（跑前落盘）。全部**纯推理**，未训练、未改 `src/`。

## 1. 权重来源、卡清单、分组与规模

| 卡 | 权重 |
|---|---|
| pronoun / sentiment / relation / person | `checkpoints/cards/*.pt` |
| negation | `checkpoints/negation_accept_card_e12.pt`（`extra` 为空，无 branch，可独立挂载） |

基座 `checkpoints/base_encoder.pt`，引擎 `MultiTaskEngine`（`src/dtseek/tasks/engine.py`），设备 cuda/rocm6.4。
**未使用** `experiments/core_branch/cards/*`（分支实验档位，非稳定版）。实际挂载日志见 `logs/run_reject.log` 开头 5 行 `[engine]`。

- **组 A（分布内背景）**：各卡 `build_dataset(6000)` → `Random(42).shuffle` → 前 `max(200, n//10)` 作 val（沿用
  `experiments/core_branch/eval_old_cards_branch.py:val_split` 口径；**这 5 张卡的 `build_dataset` 没有 `split` 参数**，
  仅 cloze_fill / reply_pick 有）→ 取 `spans == []` 的背景样本。
  n：pronoun 191 / sentiment 149 / relation 200 / person 183 / negation 193。
- **组 B（分布外，三子集）**：B1 交叉背景（其余 4 卡背景各 50 = 200）；B2 `adversarial_routing/adversarial.jsonl` 中
  `true_label ≠ 该卡`（sentiment 卡只剩 135 条，其余 200）；B3 `input_type/adversarial.jsonl` 前 200。
  池化 B：600 / 535 / 600 / 600 / 600。
- **开火** = 该样本上该卡吐 ≥1 条锚点（`pred_cls == 0` 即背景类直接 break，`engine.py:164`）。总样本 6786 × 2 遍。

## 2. 主表（P1：`fire(B) ≤ fire(A) + 10pt`）

| 卡 | A 开火率(bg_fp) | B 开火率 | Δ | B1 | B2 | B3 | P1 | P3(≤0.85) |
|---|---|---|---|---|---|---|---|---|
| pronoun | 0.0052 | **0.9600** | **+0.9548** | 0.975 | 1.000 | 0.905 | ✗ | ✗ |
| sentiment | 0.0000 | **0.9589** | **+0.9589** | 0.965 | 0.970 | 0.945 | ✗ | ✗ |
| relation | **0.8300** | 0.7650 | −0.0650 | 0.630 | 0.960 | 0.705 | ✓※ | ✓※ |
| person | 0.0000 | 0.1033 | **+0.1033** | 0.015 | 0.215 | 0.080 | ✗(+0.33pt) | ✓ |
| negation | 0.0881 | 0.7250 | **+0.6369** | 0.570 | 0.835 | 0.770 | ✗ | ✓ |

- **P1 逐卡：4/5 不过**（person 只超线 0.33pt，但仍是不过）。**P3 逐卡：2/5 不过**（pronoun / sentiment）。
- **最差卡（架构下限，按 Δ）= sentiment，Δ = +0.9589**；pronoun 紧随 +0.9548。**下限（按背景误开火）= relation，bg_fp = 0.83**。
- **relation 的 P1/P3 "通过"无意义（※）**：它自己的分布内背景就 83% 开火，分母被自己的误报抬高，
  且其 200 条 A 样本**含词表词的 = 0 条**却开火 166 条（实测，非推断）⇒ 这张卡的"拒答"根本不存在。

## 3. 置信度分桶 + 已知答案对照

**分桶占组比（n=组内条数占比；`no_fire` = 未开火）**：

| 卡·组 | <0.5 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | no_fire |
|---|---|---|---|---|---|---|---|
| pronoun·A | 0 | 0 | 0 | 0 | 0 | 0.005 | **0.995** |
| pronoun·B | 0.048 | 0.070 | 0.070 | 0.085 | 0.142 | **0.545** | 0.040 |
| sentiment·A | 0 | 0 | 0 | 0 | 0 | 0 | **1.000** |
| sentiment·B | 0.097 | 0.146 | 0.168 | 0.153 | 0.144 | **0.251** | 0.041 |
| relation·A | 0 | 0.015 | 0 | 0.005 | 0.010 | **0.800** | 0.170 |
| relation·B | 0 | 0 | 0.003 | 0.002 | 0.003 | **0.757** | 0.235 |
| person·A | 0 | 0 | 0 | 0 | 0 | 0 | **1.000** |
| person·B | 0 | 0.005 | 0.008 | 0.008 | 0.012 | 0.070 | **0.897** |
| negation·A | 0 | 0.042 | 0.010 | 0.005 | 0.026 | 0.005 | **0.912** |
| negation·B | 0 | 0.028 | 0.058 | 0.075 | 0.137 | **0.427** | 0.275 |

**关键**：分布外开火**不是低置信度噪声** —— pronoun B 有 54.5%（占全组）落在 ≥0.9 桶、relation A/B 都约 76~80% 落在 ≥0.9，
且 relation A 与 B 的置信度分布几乎重合 ⇒ **置信度阈值区分不了"自己领域"与"领域外"**。
未预注册的阈值扫描（8 档 t=0..0.99）证实：**没有任何一个 t 能让 5 张卡同时过 P1∧P3**（t≥0.99 时 pronoun B 仍 0.323 > 0.105，negation 0.145 > 0.105）。
与 dev-notes/15 §7 输入类型实验的阈值扫描同一结论方向：**fail-closed 出口救不了它**。

**已知答案对照（PREREG §4）**
- L1 机械对照：`fire([])=False` / `fire([1锚点])=True` **2/2 过**；人工清点 3 条真实返回 × 5 卡 = **15/15 一致**（`manual_verify.py` PASS）。
- L2 模型已知答案 10 条：**"该开火"5/5 全中**；"不该开火"只有 person 1/5 命中，总分 **6/10 < 预注册的 8/10**。
  4 条失败全部是**模型真的开火且切片荒谬**（例：`会议定在周三上午九点，地点是三号会议室。` → pronoun 在 `点，地点是三号` 发"第二人称"、
  negation 在 `会` 发"否定"、sentiment 在 `点，地点是三号会议` 发"愤怒/不满"）。测量函数本身另有三重独立证据
  （L1 全过、人工清点 15/15、person 卡在 89.7% 样本上确实返回 0 锚点 ⇒ 不是恒 True）。

## 4. 闭合性（代码检查 + 真实组合调用，`closure.json`）

| 项 | 引擎（卡）产物 | 算子产物 |
|---|---|---|
| 键集 | `category class_id color confidence e0 local_e0 local_s0 next_action pair_index pair_side s0 step`（12） | `label start end score`（4） |
| 区间语义 | **闭区间**（`text[s0:e0+1]` 实测正确：`[9,10]`→`喜欢`） | **闭区间**（与引擎同口径，`gap`/`_slice` 用 `end+1`） |
| 监督真值（另一层） | dataset span = **半开**（`plugin.py:276`，实测 `start:end` 才对） | — |

真实组合回喂实测：
- `filter` 产物键 `[]`（该例无重叠）、`pair` 产物键 `label/start/end/score`；
- **引擎锚点直喂 `filter`/`pair` → `KeyError: 'start'`**（不同型）；
- **算子产物回喂 `to_items` → `KeyError: 'category'`；回喂 `engine.render` → `KeyError: 's0'`**；
- 算子之间（`filter`/`pair`/`decide`）互相回喂 **OK**（子集内自封闭）。
- **结论：闭合性不成立** —— 卡产物与组合产物是**两种记录**，只有一条**单向**适配器 `to_items`，组合后无法回到引擎/渲染/适配器。
  真实调用 3 例已落盘：跨分句（flip 被 clause 正确拦下）、同分句（正常）、
  **标签不同源**：`flip_rule()` 默认 `apply_to=("积极")`，而卡的 display 名是 `积极/喜悦` ⇒ **默认规则在真实调用上静默零效果**
  （实测：默认 label 不变，显式 `apply_to=("积极/喜悦",)` 或先 `norm()` 才翻转）。
  `experiments/compose_ops/run_eval.py:57-59` 的 `norm()` 正是这个补丁 ⇒ **闭合性目前靠调用方补丁维持**。
  另一实测细节：该翻转是靠 negation 卡吐的**假标记 `我一`**（0-1）配上对的，真标记 `不` 未被发射 —— 结果对、理由错。

## 5. 可溯源（`predict` 返回值清点）

返回值只有 3 个顶层键 `{'text','num_segments','tasks'}`；锚点 12 键见上表。对照 `(module_id, 输入, 参数)`：

| 要素 | 现状 |
|---|---|
| module_id | **部分**：只存在于 `result['tasks']` 的字典键上；单条锚点脱离该字典即丢失卡名 |
| 输入 | **部分**：`text` 有整段原文；**没有**该锚点来自哪个 segment、segment 原文与偏移（只有 `local_*` + `s0/e0`，分段信息本身不在产物里） |
| 参数 | **缺**：`tasks`（跑了哪些卡）、`max_chunk_len`、`segment_policy`、`max_steps/max_len`、`base_path`、`card_paths`（哪份权重）、spec 快照、阈值/拒答判据 —— 全部不进产物；前两者是 engine 属性，后者**概念上不存在** |
| 组合后 | `item={label,start,end,score}`：**来源卡、step、class_id、color 全部丢弃** |

**判定：可溯源未实现**（只有"卡名 + 原文"，无参数、无权重标识、无调用链；组合产物更进一步丢失来源）。

## 6. 判定

- **确定性（实测）**：6786 条样本全部跑第二遍，逐样本 `sha256(JSON)` **0 处不一致（100% 逐字一致）** ⇒ "显式决策"的可复现性前提成立。
- **P1**：**不过**（4/5 卡违反）。**P2**：分项已逐卡给出，下限 sentiment Δ=+0.9589（按背景误开火则 relation bg_fp=0.83）。**P3**：**不过**（pronoun 0.96 / sentiment 0.959 远高于 0.85）。
- **三选一结论：拒答不成立（分布外照样开火）。**
  - 如实标注：PREREG 的 L2 条款字面为 6/10 < 8/10 ⇒ 按字面应记"证据不足"。核查后该 4 条失败是**模型行为**（真的开火）而非测量错误，
    对照的本意（验证测量函数）已由 L1 2/2 + 人工清点 15/15 + person 卡 89.7% 真实不开火共同满足 ⇒ 我据此给出"不成立"，
    **这是对预注册条款的一处事后解释（判据偏离，明示在此）**。
- **推论（与 dev-notes/15 §9 铁律 1 同源）**：**拒答也必须显式化** —— 由调用方按声明的输入类型/模块边界决定"不适用"，
  或由确定性规则门禁判定；不能指望输出头自己在分布外闭嘴。第三次元决策实测（路由 9.19%、输入类型 29.78%、**拒答 Δ +0.955**）同病：
  开放输入空间 + 封闭分布训练 = 必然开火。

## 7. 命令 / 产物 / 遗留

```
uv run python experiments/out_invariants/run_reject.py        # 58.8s，2>&1 | tee logs/run_reject.log
uv run python experiments/out_invariants/manual_verify.py     # L1-3 PASS 15/15
uv run python experiments/out_invariants/closure_trace.py     # 闭合性+可溯源，tee logs/closure_trace.log
```
产物：`PREREG.md` / `results.json`（判据、校准、阈值扫描、已知答案）/ `closure.json`（键集、区间、3 例组合、溯源清点）/
`raw/rows.json`（7000+ 逐样本记录）/ `raw/manual_count3.json`（人工清点原始返回）/ `logs/*.log`。

**实测**：以上全部数字。**未测 / 推断**：① 组 A 用 val split，但这些卡**没有 eval 折**，`train∩val` 未查，bg_fp 可能有偏；
② B2 对抗集原为"路由"造，当作"卡的分布外"是语义借用（已排除 `true_label=该卡` 的 90 条）；
③ 只测了这 5 张卡 + `filter`/`pair` 两算子，其它卡/算子未覆盖；④ relation bg_fp=0.83 的成因（卡训坏 vs 背景定义）未解剖；
⑤ 长文本多段、`window` 策略下的拒答行为未单独测；⑥ 可溯源为代码清点，未做端到端调用链还原（字段本身缺失，无从还原）。

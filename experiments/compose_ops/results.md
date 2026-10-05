# 组合算子评测：词典外否定表达（experiments/compose_ops）

预注册见 `preregister.md`（跑前写死）。产物：`dataset.json`、`raw_results.json` /
`summary.json`（配置 A）、`raw_results_arm_neg5.json` / `summary_arm_neg5.json`（配置 B）。
数字全部**实测**：单次前向、argmax 解码、单 seed、单次运行，无采样。

## 1. 算子 API（`src/dtseek/tasks/compose.py`）

```
item = {"label": str, "start": int, "end": int, "score": float}   # 0-based 闭区间，同 s0/e0

to_items(anchors) -> [item]                      # 引擎锚点 → item
filter(a, b, *, mode="adjacent"|"overlap"|"inside", window=8,
       clause=False, text=None, require=None, require_side="b") -> [item]
       # 保留 b 中与某个 a 满足位置规则的项；clause=两跨度之间不得有分句标点
       # require=文本前置校验（字面闭集或正则），require_side 指定校验 a 还是 b
pair(a, b, rule, *, 同上一套位置参数) -> [item]
       # 给 a 的每一项在 b 里配最近项并套 rule；配不上/规则不适用 ⇒ 原样透传；长度恒等
flip_rule(table=None, *, apply_to=("积极",), default=None) -> dict
       # 声明式规则 {"kind":"flip","apply_to":(…),"table":{词:翻转后类别},"default":None}
decide(items, default="中性") -> str              # 句级结论 = score 最高锚点；基线与组合共用
```

规则是**数据不是代码**：`NEGATED_CLASS` 把 sentiment 卡自己的显式否定式约定
（`不高兴→悲伤`、`不满意→愤怒`，见 `sentiment/dataset.py` v3）从「词条」扩展到「词典外组合式」；
查表不中就**不翻**（`default=None`）—— 锚点圈的不是情绪词时绝不乱翻。

冻结管线（`preregister.md` §0，两套 ckpt 共用）：

```
kept = filter(sent, neg, mode="adjacent", window=8, clause=True, text,
              require=TRIGGERS, require_side="b")
comp = pair(sent, kept, flip_rule(), mode="adjacent", window=8, clause=True, text)
```

## 2. 评测集与已知答案对照

`make_dataset.py` 生成；`run_eval.py` 开跑前 **fail-closed** 校验两条硬约束
（cue 必在 199 词情绪词典内；否定式整体不得是词典词条），任何一条不成立即退出 —— 两套都通过。

| 集合 | n | 构成 | 标注分布 | 多数类基线 |
|---|---|---|---|---|
| negation | **60** | 词典外否定表达（cue 全在词典内） | 悲伤 30 / 愤怒 30 | 50.0% |
| control | **57** | 非否定情绪句 44 + 含否定但否定不作用于情绪词 13 | 积极 28 / 愤怒 14 / 悲伤 15 | 49.1% |
| stress | 10 | 含否定标记但**正确极性为正**（附加集，不入 P1–P3） | 积极 10 | 100% |
| sanity | 12 | 手挑、完全确信的已知答案（跨三集） | — | — |

否定式 label 沿用 sentiment 卡**自己的**显式否定式约定（高兴/开心→悲伤，满意/喜欢→愤怒），
不是外来标准。**已知答案对照先行**：12 条先过，标注与管线成立 —— 错的全是**模型错**
（如 `我非常高兴能来` 卡判 愤怒、`这件事太离谱了` 卡判 悲伤），与标注口径无关。

## 3. sentiment 单卡基线（先报这个）

| ckpt 配置 | 单卡端到端（negation 60） | 多数类基线 | 判成「积极」 |
|---|---|---|---|
| **A** `checkpoints/base_encoder.pt`（拆自 multitask_v2）+ `cards/sentiment.pt` + `negation_accept_card.pt` | **26.7%**（16/60） | 50.0% | **63.3%** |
| **B** `artifacts/`（拆自 `arm_neg5_seed42`，5 卡同一训练轮） | **40.0%**（24/60） | 50.0% | 41.7% |

两套都**低于**该集多数类基线 50.0%，也都把近半数以上词典外否定表达判成「积极」。
⇒ 单卡 26.7% / 40.0% ≪ 80%，**「本靶子无空间」熔断不触发**，继续做组合。

这与 dev-notes/05 §4 不矛盾：`不高兴→悲伤` 是**词条**，单卡早就答对；
`没有很开心 / 谈不上高兴 / 算不上喜欢` 是**组合式**，不在 199 词里。

## 4. 三项分开报（negation 60 条）+ 对照

| 指标 | A 单卡 | **A 组合** | B 单卡 | **B 组合** |
|---|---|---|---|---|
| 上游召回 R（锚点∩cue；严格整段覆盖） | — | **96.7% / 90.0%** | — | **96.7% / 90.0%** |
| 条件成功率 C（召回后判对） | 25.9%（15/58） | **75.9%**（44/58） | 39.7%（23/58） | **79.3%**（46/58） |
| **端到端 E** | **26.7%** | **75.0%** | **40.0%** | **78.3%** |
| Δ 端到端 | — | **+48.3pt** | — | **+38.3pt** |
| 多数类基线 | 50.0% | 50.0% | 50.0% | 50.0% |
| 极性口径（正/非正） | 36.7% | **85.0%** | 58.3% | **96.7%** |
| 判成「积极」的比例 | 63.3% | **15.0%** | 41.7% | **3.3%** |
| 回退（基线对→组合错） | — | **0 条** | — | **0 条** |
| 修好（基线错→组合对） | — | 29 条 | — | 23 条 |

恒等式核对（A）：75.0% = 96.7% × 75.9% + 3.3% × 50.0%。

**对照集 57 条**：A 71.9% → **71.9%**（Δ 0.0pt）；B 64.9% → **64.9%**（Δ 0.0pt）。
纯非否定 44 条：A 75.0→75.0，B 63.6→63.6；含否定但不作用于情绪词 13 条：A 61.5→61.5，B 69.2→69.2。

**附加压力集（含否定的正向句，10 条）**：A 30.0%→10.0%，B 20.0%→10.0% ——
A 里 2 条由对变错（`不愧是让人开心的一天`、`别提我多高兴了` 被 `不/别` 误翻）。
规则的已知代价，**不计入 P1–P3**，如实报告。

## 5. 判据判定（两套 ckpt 一致）

| 判据 | 阈值 | A | B | 结论 |
|---|---|---|---|---|
| 熔断 | 单卡 ≥80% ⇒ 无空间 | 26.7% | 40.0% | 不触发，靶子有效 |
| **P1** | Δ ≥ +15pt **且** E ≥ 80% | +48.3 ✅ / 75.0 ❌ | +38.3 ✅ / 78.3 ❌ | **两套均不通过**（差 5.0 / 1.7pt） |
| **P2** | 召回 ≥ 70% | 96.7% | 96.7% | **通过** |
| **P3** | 对照集不得变差 > 3pt | 0.0pt | 0.0pt | **通过（完全无变差）** |

P1 差的部分全部落在**上游类别桶**、不在规则：B 的 13 道错题里 **11 道是「极性已对、
悲伤↔愤怒分错桶」**（组合算子只翻积极锚点，结构上够不着），只剩 2 道仍判积极
（`没有多欢喜`、`算不上喜欢这个配色` —— 锚点文本没包住情绪词）；
A 的 15 道错题里 6 道是桶错 + 9 道仍判积极（其中 6 条 negation 卡没给出可用否定标记）。
⇒ **瓶颈不在规则**（P2 过、两套回退均为 0），在**否定标记定位**与**情绪锚点边界/类别桶**。

## 6. 消融（预声明，不参与 P1–P3）

| 变体 | A neg / ctrl | B neg / ctrl |
|---|---|---|
| 主口径：卡 + 触发闭集校验 | **75.0 / 71.9** | **78.3 / 64.9** |
| A1：卡 + 无触发校验 | 81.7 / 64.9（ctrl −7.0pt，P3 挂） | 78.3 / 63.2（ctrl −1.7pt） |
| A2：正则扫 TRIGGERS（不用卡） | **85.0** / 70.2（ctrl −1.7pt） | 78.3 / 64.9 |

实测：negation 卡吐出的跨度常**漂离真实标记字**（`这家书` / `我对` / `议顺利结束` / `满`），
不含标记字的假阳会把正向对照句翻掉，所以主口径必须加 `require=TRIGGERS`。
配置 A 下卡定位不准还要额外付 10.0pt（A2 85.0 vs 主口径 75.0）；配置 B 下卡与正则打平。
⇒ **在本评测上 negation 卡相对正则没有增量**（A −10.0pt / B 0.0pt，对照侧 A +1.7pt / B 0.0pt）。

## 7. 单测与命令

```
uv run python experiments/compose_ops/make_dataset.py     # {'negation':60,'control':57,'control_plain':44,'stress':10,'sanity':12}
uv run python experiments/compose_ops/run_eval.py          # 配置 A
uv run python experiments/compose_ops/run_eval.py \
  --base experiments/compose_ops/artifacts/base_encoder.pt \
  --sentiment experiments/compose_ops/artifacts/cards/sentiment.pt \
  --negation experiments/compose_ops/artifacts/cards/negation.pt --tag arm_neg5   # 配置 B
uv run python -m pytest tests/test_compose.py -q          # 16 passed in 0.80s
uv run ruff check src/dtseek/tasks/compose.py tests/test_compose.py experiments/compose_ops/   # All checks passed!
```

单测覆盖：空输入（a/b 任一为空）、无重叠、部分重叠（overlap/adjacent/inside 三态）、
跨分句阻断、require 命中与不命中、多对多就近配对、输出长度与键集恒等、非积极类不翻。

## 8. 遗留与不确定项

**实测：**
- 两套 ckpt 上组合分别把靶子从 26.7%→75.0%、40.0%→78.3%，**回退均为 0**；P2/P3 过，P1 差 5.0 / 1.7pt。
- 极性口径：A 36.7%→85.0%，B 58.3%→**96.7%**（B 下正向误判 41.7%→3.3%）。
- negation 卡相对正则无增量（见 §6）；含否定的正向句会被误翻（stress 两条集合均 20~30%→10%）。
- **配置 B 的 ckpt 来历**：`experiments/compose_ops/artifacts/` 在本会话进行中被外部放入
  （06:54:54，非本代理写入），是从 `checkpoints/arm_neg5_seed42.pt` 拆出的 5 卡一套。
  本代理先用 `checkpoints/*` 跑出配置 A，之后补跑该套为配置 B。
- **预注册未写死 ckpt**：`preregister.md` §0/§4 只冻结了规则与标注口径，没写 ckpt 路径 ——
  这是预注册的漏洞；两套结论一致（P1 不过 / P2、P3 过），故不改变判定，但应记为方法学缺陷。
- negation 卡另有 `checkpoints/negation_accept_card_e12.pt`，**未跑**（避免在评测集上挑模型）。

**推断（未单独验证）：**
- 配置 B 若上游把 11 道「悲伤↔愤怒」桶错修掉，E = 58/60 = 96.7% —— 算术外推，未做实验。
- 主口径与 A2 的差异同时含「卡 vs 正则」与「校验开销」两因素，未做解耦 2×2。

**方法学限制：** 单次前向、单 seed、单评测集、短句口语；结论不外推到长句/多从句语体。

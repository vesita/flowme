# 预注册：组合算子评测（跑前写死，事后不许改）

写入时间：在 `run_eval.py` 第一次跑评测集**之前**。数据集由 `make_dataset.py` 生成，
规则参数在下面冻结；跑完只允许如实报告，不允许回改本文件。

## 0. 冻结的管线

```
sent_items = to_items(res["tasks"]["sentiment"])          # 基线就用它
neg_items  = to_items(res["tasks"]["negation"])

kept  = filter(sent_items, neg_items,
               mode="adjacent", window=8, clause=True, text=text,
               require=TRIGGERS, require_side="b")        # 位置规则 + 触发闭集校验
comp  = pair(sent_items, kept, flip_rule(),
             mode="adjacent", window=8, clause=True, text=text)
label = decide(comp)                                      # score 最高的锚点；空 → 中性
```

- `window=8`：两跨度之间相隔 ≤8 个字符；`clause=True`：两跨度**之间**不得有分句标点
  （`，。！？；、,.!?;…`）。这条分句条件是主要的防误翻闸门：它把
  “这个方案不难，我很喜欢”里的 `不` 挡在 `喜欢` 之外，同时放行
  “没感到十分快乐”这种跨过动词的否定。
- `require=TRIGGERS`：配对的否定锚点文本必须命中 `不/没/别/未/莫/勿/难道/何必` 闭集。
  实测 negation 卡会吐出完全不含标记字的假阳跨度（“这家书”“我对”“议顺利结束”），
  不校验会把正向对照句一起翻掉。
- `flip_rule()`：只对 `label == 积极` 的锚点生效；在跨度文本里**最长命中** `NEGATED_CLASS`
  的词才翻转，查不到就原样不动（锚点圈的不是情绪词时绝不乱翻）。
- `decide()`：句级结论 = score 最高锚点的 label；无锚点 → 中性。**基线与组合共用这把尺子。**

## 1. 预先声明的消融（同一次运行一起报，不参与 P1–P3 判定）

- **A1** `require=None`（去掉触发闭集校验）—— 用来解释 P3 若变差，差在哪。
- **A2** 否定标记改用**正则扫 TRIGGERS**（完全不跑 negation 卡）—— 用来回答
  “negation 卡相对正则有没有增量”。

## 2. 指标定义（negation 60 条为主战场）

| 记号 | 定义 |
|---|---|
| 召回 R | 存在 sentiment 锚点与 gold cue 跨度**相交** 的比例 |
| 条件成功率 C | 在召回成立的样本上，组合结论 == 标注 的比例 |
| 端到端 E | 全部样本上，组合结论 == 标注 的比例 |
| 基线 E_base | 同一把 `decide()` 直接吃 sentiment 单卡锚点 |
| 多数类基线 | 该集合里出现最多的标注类别的占比 |

恒等式 `E = R·C + (1−R)·P(对 | 未召回)`，三项分开报，不合成一个数。

## 3. 判据

- **P1** 组合端到端 E ≥ E_base **+15pt** 且 E ≥ 80%；
- **P2** 上游召回 R ≥ 70%（不过 ⇒ 瓶颈在召回不在规则，如实报）；
- **P3** 对照集 57 条上 E_comp ≥ E_base − 3pt；
- **熔断** 若 sentiment 单卡在 negation 60 条上 E_base ≥ 80% ⇒ 判「本靶子无空间」并停止，
  不换靶子、不调样本；
- **对照先行**：12 条已知答案（`sanity=true`）先单独报；若标注/管线本身站不住，先修管线。

## 4. 标注口径（同样跑前冻结）

否定式的 label 沿用 sentiment 卡自己的显式否定式约定（`sentiment/dataset.py` v3）：
否定「高兴/开心/快乐/痛快/舒服/轻松…」→ **悲伤**；否定「满意/喜欢/认可/赞同/赞成/欣赏」→ **愤怒**。
cue 必须在 199 词情绪词典内，而「否定 + cue」整体**不得**是词典词条 —— `run_eval.py` 开跑前
fail-closed 校验这两条，任何一条不成立就退出。

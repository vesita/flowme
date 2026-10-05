# 选择卡族实测结果（reply_pick / cloze_fill）——单 seed=42

> 判据见 `PREREG.md`（**训练前冻结，跑完未改**）。除第 6 节标注「诊断」外，全部为实测；单 seed，只能作方向。

## 1. 两卡 spec 关键值
| 项 | reply_pick 择优回复 | cloze_fill 完形填空 |
|---|---|---|
| name / label | `reply_pick` / 择优回复 | `cloze_fill` / 完形填空 |
| classes | 无合适候选(0) + 候选1..4（`class_names` 取 `c.name`） | 同左 |
| max_steps / emission / segment_policy | 1 / single / window | 同左 |
| max_len（引擎整段阈值 = max_len−8） | 128（120） | 96（88） |
| 标记符号（词表内、确定性 id） | `\|`=124、`）`=156、`_`=95（`__`） | 同左（`__` 空位） |
| anchor | 被选候选内容，0-based 半开存储 | 空位处被选词，同口径 |

## 2. 数据构造（实测）
- 语料：`resolve_corpus_files()` 18 个 `*dialogue.txt`，轮转扫描 1,200,000 行。
- reply：原始 195,069 对 → 门禁通过 **63,552（32.6%）**；train 折 57,321 / eval 折 6,231。
  训练 6,000 = 有正解 5,100 + 无合适 900；位置 {1:1282,2:1259,3:1252,4:1307}；
  负例条数 {同块774, 同话题4410, 随机10116}，**纯高难 1,728/5,100=33.9%**。
  难度档：档1=跑题（随机回复、长度±8、问句二元组重合≤0.20）；档2=同话题不同轮次（同块优先）；档3 本轮不做。
- cloze：可用句子 **628,317**（train 折 565,858 / eval 折 62,459）；训练 6,000 = 5,100+900；
  位置 {1:1252,2:1292,3:1291,4:1265}；空位 句首765/句中3570/句末765=15/70/15；与标点相邻 8.1%（≤70%）。
  丢弃：唯一性不足 8,513 / 干扰项不足 134 / 无合适未达全弱共现 65,700 / 正解共现不足 185。
- fail-closed 检查点全部在构造期抛错：①标记词表校验；②内容含 `|` `）` `_` 或超长即抛；
  ③每样本≤1 切片、锚点逐字=被选候选、文本不重复；④按 md5 分折后训练/评测问句与句子**逐条不相交**；
  ⑤cloze `score(正解)≥4 且 ≥3×max(score(干扰))`、bg 4 项全 `score<4`、干扰同词类同字长频次∈[f/3,3f+2]；
  ⑥空位配额句首/句末≥15%、与标点相邻≤70%。**实测全部通过**（单测钉住，35 项）。

## 3. 预注册原始指标（engine 口径；批量前向对账 200/200=100%）
**reply_pick**（盲猜 5 类 =20%，限定四选一=25%）

| 集 | n(有解/bg) | cls_acc | 分位置(1..4) | max−min | P2a 误触发 | P2b 拒答召回 | anchor命中 | 联合 | em/prec/rec | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 打乱 | 1500(1275/225) | **0.2000** | .2342/.1316/.1826/.2500 | 11.84pt | 0.1969 | 0.2933 | 0.0196 | 0.0039 | .0607/.0211/.0196 | 6.6 |
| E2 固定1位 | 1500 | 0.2212 | .2212 | — | 0.1906 | 0.2933 | 0.0177 | 0.0039 | .0500/.0076/.0071 | 3.3 |
| E3 纯档2 | 800(680/120) | 0.1926 | .2102/.1684/.1768/.2189 | 5.05pt | 0.2206 | 0.2667 | 0.0229 | 0.0044 | .0550/.0194/.0176 | 3.4 |
| E4 纯档1 | 800(680/120) | 0.1706 | .1761/.1724/.1570/.1771 | 2.01pt | 0.2279 | 0.2750 | 0.0086 | 0.0015 | .0475/.0082/.0074 | 3.1 |

- 预测类别分布 E1：{无:317, 候1:311, 候2:266, 候3:245, 候4:361}（未恒选一槽，但偏 0/4 槽）。
- **P1b Δ = E2−E1 = +2.12pt**；**P3 换帧探针 n=32：class_acc 0.3125 / span_acc 0.0312**（分开报）；
  sanity 已知答案对照 3/8 对（`passed=False`）；trainer 内置 val：cls 0.1844 / span 0.0160 / bg_fp 0.7228 / em 0.0517 / prec 0.0064 / rec 0.0060。

**cloze_fill**

| 集 | n(有解/bg) | cls_acc | 分位置 | max−min | P2a | P2b | anchor | 联合 | em/prec/rec | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 打乱 | 1500(1275/225) | **0.1906** | .1857/.1461/.2030/.2376 | 9.15pt | 0.1804 | 0.4044 | 0.1235 | 0.0235 | .1380/.0984/.0910 | 22.7 |
| C2 固定1位 | 1500 | 0.1882 | .1882 | — | 0.1827 | 0.4044 | 0.0917 | 0.0173 | .1020/.0527/.0486 | 4.8 |

- 分空位类型 C1：句首 0.1937 / 句中 0.1859 / 句末 0.2094（max−min 2.35pt）。
- **P1b Δ = C2−C1 = −0.24pt**；**P3 探针 class_acc 0.2500 / span_acc 0.0000**，sanity `passed=False`；
  trainer val：cls 0.2209 / span 0.1203 / bg_fp 0.6452 / em 0.0867 / prec 0.0394 / rec 0.0375。

**判据逐条**：P1a reply **11.84pt FAIL** / cloze 9.15pt PASS；P1b 两卡 PASS（+2.12 / −0.24pt，但都贴盲猜，PASS 无信息量）；
P2a reply **19.69% FAIL**、cloze **18.04% FAIL**（阈值 5%）；P2b 29.33% / 40.44%（只报）；
P3 见上（只报）；P4 reply em 6.07%、cloze em 13.80%（只报）；P4b reply E3 19.26% vs E4 17.06%（高难负例更低）；
P5 anchor reply **1.96% FAIL**、cloze **12.35% FAIL**（阈值 95%）；P6 见第 4 节。

## 4. 逐字命令 / 日志 / 代价
```
systemd-run --user --unit=dtseek-reply-pick-s42 --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  /bin/bash -lc 'uv run python -u experiments/selection_cards/run_card.py --card reply_pick --epochs 12 --steps-per-epoch 150 --batch-size 64 --samples 6000 --seed 42 --out experiments/selection_cards/reply_pick.pt > experiments/selection_cards/reply_pick_train.log 2>&1'
systemd-run --user --unit=dtseek-cloze-fill-s42 --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  /bin/bash -lc 'uv run python -u experiments/selection_cards/run_card.py --card cloze_fill --epochs 12 --steps-per-epoch 150 --batch-size 64 --samples 6000 --seed 42 --out experiments/selection_cards/cloze_fill.pt > experiments/selection_cards/cloze_fill_train.log 2>&1'
uv run python experiments/selection_cards/eval_selection.py --card reply_pick --ckpt experiments/selection_cards/reply_pick.pt --sets E1,E2,E3,E4
uv run python experiments/selection_cards/eval_selection.py --card cloze_fill --ckpt experiments/selection_cards/cloze_fill.pt --sets C1,C2
uv run pytest -q && uv run ruff check
```
- `run_card.py` = 先 `importlib.import_module` 两张卡（`register(CARD)`），再 `runpy.run_path(training/train_task_card.py)`（绕开被锁的 `builtin/__init__.py`，协议未动）。
- 代价 P6（实测）：reply **train_sec 35.43 / peak_mem 151.46 MB / sec_per_step 0.0197**；cloze **45.74 / 128.43 / 0.0254**（1800 步，与他人作业共卡）。
- 日志：`experiments/selection_cards/{reply_pick,cloze_fill}_train.log`；结果 JSON：`results_{reply_pick,cloze_fill}.json`；权重 `.pt`（各自独立文件名）。

## 5. pytest / ruff 原始输出
- `uv run pytest -q` → **`177 passed in 18.57s`**（基线 81 + 他人 negation 61 + 本任务 35）；本任务 `tests/test_selection_cards.py` → `35 passed in 5.78s`。
- `uv run ruff check`（全仓）→ **`Found 258 errors`**；基线 226，**+32 全部来自他人新增实验脚本**（adversarial_routing / rank_budget / ndb_* / negation_card/run_5card，及他人写在本目录的 `run_easy_tier.py` 1 条）。
- 我的路径逐文件 → **`All checks passed!`**（reply_pick/、cloze_fill/、tests/test_selection_cards.py、PREREG.md、eval_selection.py、run_card.py）。

## 6. 结论：真的会选吗？——**不会（实测，单 seed）**
- 两卡类别准确率 **reply 20.00% / cloze 19.06%**，与盲猜 20%（四选一 25%）**不可区分**；anchor 命中 1.96% / 12.35%，拒答误触发 19.69% / 18.04% 均远超 5%。
- **诊断（非预注册，实测）**：①同配方 pronoun 对照 val cls **99.26%** → 训练管线、GPU、探针没问题；
  ②按 trainer 同口径复算：**训练集 100% / val 22.16%** → 模型**记住不泛化**；
  ③数据×5（30000）→ 18.80%；步数×4（48×150）→ 20.84%；头容量×4（L4/ff1024）+步数×4 → 23.05%：**三种放大都贴盲猜**；
  ④冻结基座池化特征线性探针 reply train 54.6% / val 23.3%（pronoun 98.5% / 95.8%）→ 表征里没有可迁移的选择信号；
  ⑤**合成显式关键词规则**（同框架、词面信号直接给出）：train 记住（loss 0.11）但 val 25.1% → 连字面匹配都不迁移。
- 判断：失败不在协议（128 装得下问句+4 候选，实测 p50≈90、硬上限 120），不在数据（锚点/位置/折不相交均单测通过），
  也不在步数与头容量；**指向「冻结基座 + 小头不具备比较候选的表征」**——与 dev-notes/12 §12.5 冻结 0.6383 vs 联合 0.9667 同向。
- `PREREG_JOINT_FIX.md`（联合训练臂，基座可训）与 `run_easy_tier.py` 是**他人**在同一目录预注册/开跑的后续臂，本轮不重复跑。

**多 seed 配对方法（只描述，未跑）**：固定数据折与评测集，取 seed∈{42,52,62} 各训一次，两臂（加卡前/后）逐 seed 配对，
看老卡指标差值的均值±sd；按 dev-notes/12 §11.2 同数据重跑噪声 ~2pt，差值 |Δ|>2×sd 才算退化；
同 seed 同数据的成对差值比两组独立均值更敏感，需同时报 3 个 seed 的原始值。

## 7. 遗留与不确定
- `src/dtseek/tasks/builtin/__init__.py` 的 3 行注册**被 negation 独占锁（c_63）挡住，4 次编辑被拒**，未绕过策略；训练/评测靠 `run_card.py` 显式 import 注册，**协议与他人代码均未改**。锁释放后需补 `cloze_fill,` / `reply_pick,` 两行（字典序）。
- 位置差 11.84pt 的置信区间较宽（cls 判对仅 255 条），单 seed 不宜当结论；只报原值。
- `bg_fp`（trainer 口径 72.28%）与 P2b 拒答召回（29.33%）是同一批 bg 样本的两面，勿重复计数。
- 证据强度：全部为**单 seed、单次**测量；「不会选」有 5 组独立实测互相佐证，但**换 seed 未复测**。

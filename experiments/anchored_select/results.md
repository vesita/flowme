# 锚定候选打标（experiments/anchored_select）—— 结果

> 判据见 `PREREG.md`（mtime **2026-10-06 14:58:14**，早于首次训练：smoke 15:13、主臂 15:15），跑后未改。
> 口径：「实测」= 本目录跑出来的数字；「推断」= 由数字推出；「口径判断」= 构造时规定的标签归属。

## 1. 数据集（`build_data.py`，fail-closed；独立复核 `inspect_data.py`）

语料只走 `dtseek.tasks.corpus.resolve_corpus_files(CORPUS_GLOB)`（glob 空即抛）；按 SPLIT_SEED=20240927
打乱文件序、每文件最多取 4000 句，共收 13053 句 → 按下标切成**互不相交的三个源句池** 9500/2100/950。
布局正负逐字相同：`原文：{context}|候选：{candidate}`（`|` 是词表内字符；实测全角 `｜` 是 `<unk>`，弃用）。
正例锚点 = 右侧候选字段的 0-based 半开区间；负例 = 背景（不发射）。

| 集合 | n | 正:负 | 正例 | 负例 | 多数类/盲猜 |
|---|---|---|---|---|---|
| train | 8000 | 4000:4000 | 逐字片段 4000 | 换词 1334 / 反义 1333 / 无关 1333 | 50.0% |
| test | 1600 | 800:800 | 逐字片段 800 | 换词 267 / 反义 267 / 无关 266 | 50.0% |
| adversarial | 600 | 300:300 | **同义改写 300** | **同词表换位置 300** | 50.0% |

- **两两不相交实测（`inspect_data.py` 独立重算，必须为 0）**：train∩test / train∩adv / test∩adv 的
  `text` / `context` / `candidate` 三口径重叠数**全部 = 0**；三集并集 10200 = 各自之和。
- 标签一致性：span 未指向候选 0 条；train/test 正例非逐字片段 0 条；负例却是逐字片段 0 条；
  adv `reorder` 300/300 能还原成"上下文某连续片段的同词表重排"；adv `paraphrase` 逐字出现在上下文里 0 条。
- 最长样本 70 字符 < `max_len=96` ⇒ **0 截断**。
- **表层捷径基线（零训练）**：`候选∈上下文` ⇒ train **100%** / test **100%** / adv **50%**
  （对抗集上它退化成恒判"不支持"，恰好好多数类）。
- 标注噪声人工抽检：`reorder` 抽 20 条逐条读，**0 条语义等价**（口径判断：忠实性而非世界真假；20 条样本小，上界粗）。

## 2. 训练口径（现有卡机制）

- **主臂 = 只训头**：`run_card.py` 注册本卡后**原样委托** `training/train_task_card.py`
  （冻结核 + `RobustARSliceDecoder`），12 ep × 150 步 = **1800 步**、batch 64、lr_head 1e-3、cosine、
  base=`checkpoints/base_encoder.pt`、**seed 42/43**。卡 `anchored_sel`：classes=(背景,支持)、
  `max_steps=1`、`max_len=96`、`cls_weight_bg=1.0`（PREREG 写死：max_steps=1 时背景类不被稀释）。
- **自检实测**：核 1,688,460 参数 **trainable=0（`requires_grad` 实测冻结）**；头 **629,764** 全可训；
  可训合计 **629,764**；训练样本 8000（trainer 内部 7200/800 切 val，指标按全量 8000 自评）。
- **P4 联合臂**（`train_joint.py`，克隆 `cumulative_add --mode control`）：16 ep × 84 = **1344 步**、
  lr_base 3e-4 / lr_head 1e-3；核可训 1,688,460 + 新头 629,764 = **2,318,224**；
  四张老卡头 `requires_grad=False` 实测 OK 且**不在优化器**（`SELFTEST_OPT` 断言）；
  **G0 起点门禁 PASS**（四卡起点 exact 与 `capability_map/summary.json` 逐位相同）。
- 随机标签对照真的做了：train 标签整体 randperm（保持 1:1），同配方同 seed 各训一次（`--shuf`）。

## 3. 主表（二分类准确率 %；exact / 锚点为更严附报，不用于过线）

| 集合 | n | 多数类=盲猜 | s42 | s43 | 极差 | δ | 门槛 | 判定 |
|---|---|---|---|---|---|---|---|---|
| train | 8000 | 50.0 | 68.94 | 69.19 | 0.25pt | 2.00pt | ≥52.00 | — |
| **test** | 1600 | 50.0 | **61.62** | **59.94** | 1.68pt | 2.00pt | ≥52.00 | **P1 PASS**（两 seed） |
| **adversarial** | 600 | 50.0 | **52.50** | **53.83** | 1.33pt | 2.00pt | ≥52.00 | **P2 PASS**（两 seed，余量 +0.50/+1.83pt） |

| 对照 | s42 | s43 | 门槛 | 判定 |
|---|---|---|---|---|
| 随机标签对照（test） | 52.69 | 51.88 | ≤53.0 | **P3 PASS**（且主臂 61.62/59.94 > 对照） |
| 盲猜 / 多数类（三集） | 50.0 | 50.0 | 报出 | 构建期断言 1:1 |
| 表层子串规则（零训练） | 100 / 100 / 50（train/test/adv） | 同左 | 报出 | 对抗集恰为多数类 |

附报（同两 seed）：test `exact` 30.75/29.63%、**正例锚点命中 0.63/0.75%**；adv `exact` 22.33/21.83%、锚点 0.33/0.67%。
逐规则 test：verbatim 62.38/61.38（正例召回）、word_swap 48.31/50.19、antonym 64.04/59.93、unrelated 70.30/65.41。
逐规则 adv：**paraphrase 60.67/64.67（判对）、reorder 44.33/43.00（低于 chance ⇒ 方向性答错）**。
**指针诊断**（`diagnose_span.py`）：冻结臂正例 607/800 与 599/800 把起点预测成 **0**（塌缩），start 命中 3.75/3.25%、双命中 0.75%。
**P4 联合臂**（Δ pt，两 seed）：pronoun **+1.83/+0.67**（带 −2.83）、sentiment **+12.53/+11.81**（带 −0.41）、
relation **+3.19/+3.19**（带 −1.39）、person **+0.50/+0.00**（带 −0.33）⇒ **全部 ≥ −带，P4 PASS（两 seed）**。
联合臂附带数字（非判据）：train 91.39/91.54（锚点 84.98/86.38%）、test 65.87/64.62（锚点 50.12/50.25%）、
**adv 51.50/50.33（< 52.00 门槛，塌回多数类）**；联合臂指针 start 命中 53.75%、双命中 50.75%，但仍有 369/800 起点=0。

## 4. δ 怎么算

`δ = max(2.0pt, 该判据所用集合上主臂两 seed 的极差 |s42−s43|)`，逐集合各算、不共用：
train 极差 **0.25pt** → δ=2.00；test 极差 **1.68pt** → δ=2.00；adv 极差 **1.33pt** → δ=2.00。
**三处 δ 全部由 2pt 下限决定**（极差都小于 2pt），门槛一律 50+2 = **52.00%**。

## 5. 判定：**① 能学（打标这一半）** —— P1∧P2∧P3 两 seed 全过，P4 过

**机制（实测）**：把"选最优"换成"逐候选判有无证据"，冻结核 + 一个 629,764 参数头就能把 test 从 50% 拉到
61.62/59.94%，且随机标签对照只有 52.69/51.88 ⇒ 增益来自标签结构、不是记忆。**与 `reply_pick` 的差别有三条，
按证据强度排序**：

1. **任务形态（实测支持）**：5-way 跨候选择优 → 单候选二分类。`reply_pick` 20.00% vs 盲猜 20%（Δ=0）；
   本实验 61.62/59.94 vs 50（**Δ=+11.6/+9.9pt**）。同核、同头配方、同评测管线。
2. **数据里有廉价可学的表层相关物（实测 + 推断）**：train/test 的正负完全由"候选是否逐字含于上下文"分开
   （子串规则 100%），`reply_pick` 的标签没有这种相关物（推断：dev-notes/13 §9 已排除数据量/步数/头容量，
   故这是最可能的主要差别；**未做消融，不能单凭本实验证明**）。注意模型只学到 61.6%、**没把该规则学满**。
3. **指针那半完全没变（实测）**：冻结臂锚点命中 0.63/0.75%、预测塌到 start=0（75%）—— 与 `reply_pick`
   span_hit 1.96% 同因。**差别只在"打标"，不在"锚定"**；把核放开（联合臂）才把锚点救到 50%，
   但代价是 adv 塌回 51.50/50.33（<52 门槛）：**核越学会表层规则，对抗集越差**。

⇒ 对投机式对话的含义：**"有证据支持吗"这一步在当前核上存在，但只解决了一半** —— 判定余量薄（adv +0.5pt）、
一半对抗规则方向性答错（reorder 43~44%）、且只训头口径下**锚点不学**。零幻觉保证需要"判定 + 锚点"两条，
现在只拿到前者；**据此不能说"选择问题解决了"，也不能直接开建生成卡**。

## 6. 原始命令 / unit / 日志 / 产物

```bash
uv run python experiments/anchored_select/build_data.py > logs/anchored_build.log 2>&1
uv run python experiments/anchored_select/inspect_data.py > logs/anchored_inspect.log
systemd-run --user --unit=dtseek-anchored-a --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -c 'bash experiments/anchored_select/run_stage_a.sh > logs/anchored_stageA.log 2>&1'
systemd-run --user --unit=dtseek-anchored-b3 ... run_stage_b.sh > logs/anchored_stageB.log   # P4 联合臂
uv run python experiments/anchored_select/summarize.py   # 主表 + δ + P1..P5 → results/summary.json
```

- 日志：`logs/anchored_{build,inspect,smoke,stageA,stageB,diag_span}.log`；hung 掉的那次留档 `logs/anchored_stageB.hang1.log`。
- 产物：`experiments/anchored_select/cards/{anchored,shuf}_s{42,43}.pt`、`joint_s{42,43}.pt`；
  `results/{main,shuf,joint_train,joint_eval}_s{42,43}.json`、`results/summary.json`；数据 `data/*.jsonl`、`data/stats.json`。

## 7. 遗留与不确定

- **实测**：以上全部数字（2 seed、判据跑前写死、随机标签对照真做、P4 用联合臂真实 Δ、对账 `reconcile_ok` 全过）。
- **实测（运维）**：第一次 stage B 在 15:24:40 后**空转 2.5h 无进度**（CPU 199%、log 不动，疑 ROCm/争用 hang），
  已 kill 自己的 unit `dtseek-anchored-b2` 重启为 `b3`；该次无产物，计时与指标全部来自重启后的跑。未杀任何他人进程。
- **推断（未消融）**：与 `reply_pick` 的主要差别可能来自"表层相关物"而非纯任务形态；要分开需再做一组
  "正负=语义等价判定、无字面相关"的消融。
- **推断（未做对照）**：adv `reorder` 是打乱语序的片段，模型可能用"通顺度"而非语义判；无 fluency 对照。
- **口径判断**：同义改写的语义等价由人工词表给定；`reorder` 的"不支持"按**文本忠实性**口径；抽检 20/20 无噪声（上界粗）。
- **未做**：第三 seed（PREREG 只要求 ≥2）；对抗集 600 条 ⇒ 单 seed 分辨率约 ±3.6pt，δ 已取 2pt 下限；
  联合臂在 test/adv 的数字是附带观测，**不参与 P1/P2 判定**（主臂 = 只训头，PREREG §3 写死）。

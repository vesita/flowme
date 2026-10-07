# P5 标签构念效度 —— REPORT

产物目录 `experiments/label_construct_validity/`；判据 `PREREG.md`（mtime 早于首跑 inventory/q3）。
**判定 = 不存在可用（非构造性）情绪/逻辑标签源**（三选一之一，不是硬选：见 §4 Q0 逐源实测）。

## 1. 候选标签源清单（Q0，逐源实测；`rule` = 用该标签自身的定义规则去预测它，行级 exact match）

| 源 | 定义方式 | n | 类别 | `rule` acc ± SE | 门槛 1−2SE | 可用？ |
|---|---|---|---|---|---|---|
| **emo** `core_keep/cache/sentiment_32000.pkl`（全量） | 词表生成器 `EMOTION_KEYWORDS` | 32000 | 4（8000×4） | **1.0000 ± 0.0000** | 1.0000 | **否** |
| **emo** core_probe test split（rng42 同口径） | 同上 | 4000 | 4（1000×4） | **1.0000 ± 0.0000** | 1.0000 | **否** |
| **logic**（无标签文件） | 标签**不落盘**：`two_channel_head/data/train.jsonl` 键 = sent/skel_id/skel/n_slots/bag/bag_span/assign/split，无 logic 字段；core_probe 加载时用 `LOGIC_WORDS`(19 词) 现算 | — | 2 | **1.0000（按定义）** | — | **否** |
| skel `two_channel_head/train.jsonl::skel_id` | 模板生成器回填 | 8000 | 40 | 0.5329 ± 0.0056（`stats.json::max_naive_train` 引用实测） | 0.9888 | 数值过线但**标签仍构造** |
| idr `person_6000.pkl` 提及级 `is_repeat` | 生成器+复核解析器 | 33139 | 2 | 0.7918 ± 0.0022（复测，与 core_probe 逐位同） | 0.9955 | 数值过线但**标签仍构造** |
| person-id `person_6000.pkl` 人物 id | 生成器（首现顺序） | 6000 | 7 | 0.3750 ± 0.0063（**本单元自实现扫描器**，非定义本身） | 0.9875 | 否（口径不同源，PREREG §2(c)） |
| pronoun / negation / ownership / idiom | 各自词表提取器 | 各 6000 | — | **1.0000 ± 0.0000** ×4 | 1.0000 | 否 |
| relation `relation_6000.pkl` 类别 | `PAIR_TABLES` 词表对 | 7200 | 3（3200/2000/2000） | **1.0000 ± 0.0000** | 1.0000 | 否 |
| **value_card `data/gold.jsonl`（仓内唯一 hand 标注）** | 外部代理手写 `rule=hand` | **120** | 3（40/40/40） | 0.4250 ± 0.0451 | 0.9097 | 数值过线、来源独立，但**构念=得体/冒犯，非情绪非逻辑** |
| nanoSeek `intent_probe.py::INTENTS` | 正则判据，**标签不落盘**（实测 `intent_mined.txt` 1601 行纯对话） | — | 14 | 不可计算 | — | 否（无实例级标签） |

覆盖（`logs/coverage.log`）：DTSeek 68 个 pkl/jsonl 全列出；nanoSeek `data/**` 无情绪/情感标注文件；
`/home/vesita/datasets/NLP/` 全是无标签语料（小说/c4/distill/lccc/sharegpt…）；`escov_zh.txt` 17325 行**无标签字段**。

## 2. Q1（若有源）—— **不适用**

按 PREREG 步骤 4：情绪/逻辑两族 `rule = 1.0000`（构造性满分）⇒ 无可用源 ⇒ **不跑 probe、不造标签、停**。
（Q1 表本单元留空是判据要求，不是遗漏。）

## 3. Q2 `max_naive` 并列（逐任务，majority / rule 两个数都给）

| 任务 | majority | rule | 本单元对 rule 的动作 |
|---|---|---|---|
| emo | 0.2500 | **1.0000** | 复测 32000/32000 与 test 4000/4000 全中 |
| skel | 0.4168 | 0.5329 | 引用 `stats.json` |
| logic | 0.7896 | **1.0000** | 标签即词表 ⇒ 按定义 |
| idr | 0.5279 | 0.7918 | 复测逐位一致 |

## 4. Q3 emo 跨 split 重复复核（实测）

- **行级重复**：emo **1108/4000 = 27.70%**（与 REPORT「27.7%」复核一致，相对差 <0.1pt）；
  skel **0/2500**、logic **0/2500**、idr **0/16551** = **0.00%**。唯一文本口径 emo 5.15%，其余 0%。
- **重复是类条件的（新发现）**：class0(中性) **1000/1000 = 100%** 重复，class1 30/1000=3.0%、
  class2 41/1000=4.1%、class3 37/1000=3.7%；train 内部也有重复（4000 行仅 2977 唯一）。
- **probe 分解**（复用 core_probe `build_sets/extract/fit_probe`，核冻结、5 层、2 seed、CPU）：
  b1 = **79.00 / 79.07**（REPORT 78.64，差 0.4pt，SE=0.65，同协议不同设备）；
  **同文子集 98.74 / 98.56（n=1108）**，**新文本子集 71.44 / 71.61（n=2892，SE=0.84pt）**；
  新文本子集**无 class0** ⇒ 3 类问题，majority = **0.3354**（train 多数类 0 在该子集 acc = **0.0000**）。
- **一句话结论（实测+推断分开）**：78.64% 的绝对水平里约 **7.5pt** 来自那 27.7% 与 train 同文的行
  （实测加权 0.277×98.6 + 0.723×71.5 = 79.0 复现总数）；**对没见过的文本应读作 71.5±0.8%**（推断：
  相对 33.5% 多数类仍大幅超出，但**中性类的新文本泛化本次 100% 泄漏、完全没测到** ⇒ 不能说
  「四类都能泛化」，也不能把 78.64 当作新文本上的绝对水平）。

## 5. Q0–Q3 逐条 + 判定

- **Q0（主）**：情绪 `rule = 1.0000`（两口径均 32000/32000、4000/4000 全中，SE=0 ⇒ 不 < 1−2SE）；
  逻辑**连标签文件都不存在**（加载时词表现算）。⇒ **情绪/逻辑无可用源**（实测，证据强度：全量枚举）。
- **Q1**：不适用（无源 ⇒ 不跑，按 PREREG 停）。
- **Q2**：四个任务的 majority 与 rule 已全部并列（§3）。
- **Q3**：27.70% 复核一致 + 类条件泄漏 + probe 分解（§4）。
- **判定：不存在可用（非构造性）情绪/逻辑标签源。** 旁证：全仓 9 个标签源里 7 个 `rule = 1.0000`、
  1 个引用值 0.5329、1 个手写 0.4250 —— **数值过线的三个要么标签仍是生成器构造（skel/idr/person），
  要么构念不是情绪/逻辑且 n=120（value_gold）**；`rule<1.0` 是必要条件，本单元实测它**不充分**。

## 6. 要造出「非构造性情绪/逻辑标签」需要什么（推断，标注成本为估算）

- **规模**：≥8000 实例（train/test 各 4000、类均衡；n=4000 时 SE≈0.008、门槛 0.985），
  真正目标是让 `rule` 明显低（**建议 <0.85**），否则 `probe − rule` 无分辨力。
- **形式**：句/轮级 4 类情绪（或 valence 连续值）+ 可选 span；**盲标**：标注者看不到词表、不接受词表预标，
  双人标注 + 仲裁，报 Kappa。**红线**：词表预标再人工「过一遍」⇒ `rule` 仍≈1，重复同一构念错误。
- **成本（推断）**：单人约 5 s/条 ⇒ 8000 条 ≈ 11–12 h；双标+仲裁约 25–30 h。
- **可复用来源**：① 仓内 hand 标注样板 `experiments/value_card/gold/gold_hand.txt`（120 行 `⟦⟧` span 格式、40/40/40，`build_data.py::load_gold` fail-closed 解析）—— 格式可直接照抄；② 语料底稿 `nanoSeek/data/chinese/clean_v3/*dialogue.txt`（1.12 GB / 1234 万行，dev-notes/18）；③ **最短路径（推断）**：`nanoSeek/data/chinese/new_sources/escov_zh.txt`（17325 行，情感支持对话译文，**标签已丢失**）—— 若找回上游英文 ESConv 的 emotion 标注并按句对齐，即得独立标签；④ 仓外数据集（**仓内没有，需下载+许可核对**）：ChnSentiCorp / NLPCC 情感·情绪 / 微博 6 类情绪 / GoEmotions；⑤ 逻辑标签要独立于词表，需**复句关系/篇章关系树库**类来源，仓内与 nanoSeek **均不存在**（实测）。

## 7. 原始命令 / 日志 / 产物

- `uv run python experiments/label_construct_validity/inventory.py` → `logs/inventory.log`（**exit 0**；
  首跑 **exit 1** `KeyError:'category'` 背景行无该键，已按「无键=背景0」修正，失败留痕在本行）。
- `uv run python experiments/label_construct_validity/q3_overlap.py` → `logs/q3.log`（**exit 0**；
  首跑 **exit 1** `TypeError` 聚合 float，已修）。
- `logs/coverage.log`（覆盖枚举 + `git status --porcelain` = 只有 `?? experiments/label_construct_validity/`）。
- 产物：`results/inventory.json`、`results/q3_overlap.json`；判据 `PREREG.md`；代码 `inventory.py`、`q3_overlap.py`。

## 8. 遗留与不确定（实测 / 推断分开）

1. **实测**：person-id 的 0.3750 是本单元**自实现**扫描器的结果，与生成器解析器不同源 ⇒ 只说明
   「我这条规则复现不全」，**不能**读成「该标签独立于规则」。
2. **实测**：`value_gold` 规则为类别级口径、n=120 ⇒ SE=4.5pt，量级粗。
3. **实测**：Q3 probe 跑在 CPU，b1 与 REPORT 差 0.4pt（同协议、不同设备/浮点）；分解结论不受影响。
4. **推断（强）**：nodup 的 71.5% 是 3 类问题上的数，**不能**与 78.64（4 类、含泄漏）直接比大小。
5. **未覆盖**：nanoSeek `_discarded/`、`local/`、`temp/`、`pre-research/` 与 `data/external/` 未逐一读
   （dev-notes/18 同样标注未深入）；本单元只覆盖 DTSeek + `nanoSeek/data` + `/home/vesita/datasets/NLP`。
6. **未做**：Q1 probe（判「无可用源」后按 PREREG 停）；未改任何既有文件（`git status` 已核）。

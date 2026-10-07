# NS1 · nanoSeek 侦察：数据集 / 评估标准 / 可复用资产（供 flowme 文本生成卡框架）

> 纯只读实测 2026-10-07；路径相对 `nanoSeek/`；`git status --porcelain` 空 = 一字节未动。**实测** = 现场读到；**未验证**见 §6。

## 1 数据集清单（**生成类** = 有显式「输入 → 目标文本」对才作候选；**非生成类** = 连续纯文本、无目标对）
全库 25 源逐文件「MB / blocks / 字符」构成：`data/chinese/DATASET_REPORT.md:138-164`（只给构成、不给均值）；当前各段构成与 `counts` 见各 `manifest_v3_*.json:39`；溯源 = 每源记 SHA-256+字节+时间戳、ckpt 关联哈希 `DATA_SOURCES.md:85`。

| 类 | 数据集（路径 · 字节 · 许可） | 任务 / 格式 |
| :-- | :-- | :-- |
| ✅**生成·首推** | `data/chinese/clean_v3/gsm8k_cot_dialogue.txt` 4,753,284 B / 7,477 blocks `DATASET_REPORT.md:152` · **MIT** `DATA_SOURCES.md:46` | 数学 CoT：`用户：<题>` + `模型：<逐步推理>` + `#### <答案>`（**实测** head） |
| ✅**生成** | `clean_v3/sharegpt_zh_38k.txt` 143,684,697 B `manifest_v3_dlg.json:90`、`belle_multiturn.txt` 71,430,422 B `:48`、`wildchat_zh.txt` 28,844,849 B `:96` · **Apache-2.0 / GPL-3.0 仅研究 / AI2 ImpACT-LR** `analysis/zh_dialogue_sourcing.md:188-190`、`analysis/belle_wildchat_import.md:33` | 多轮对话；v3_dlg 共 9 源、288,209 样本 / 117.6M 字符 `manifest_v3_dlg.json:39` |
| ✅**生成** | `stages/v3_intent5/{intent_mined,intent_synth,topic_switch}.txt` = 336,230 / 26,669 / 97,105 B `manifest_v3_intent5.json:48,54,72`（全段 6,580 样本 `:39`，**许可未登记→§6**）；`new_sources/escov_zh.txt` 1,344,651 B `manifest_v3_dlg.json:60`（机翻 ESConv 1,300 段 / 38,365 条，口径 `data/chinese/filter_mt_translation.py:5-35`） | 意图跟随：一句话输入 → 期望言语行为的回复；情感支持多轮对话。均为 `用户：/模型：` 逐行 |
| ✅**生成** | 其余对话/指令源 `deepseek_r1_distill`(573 MB)、`qwen3_235b_distill`(267 MB)、`coig_*`、`code_alpaca`(24.2 MB)、`zhihu_kol`(73 MB)、`lccc`/`glm`/`kdconv`/`qa_knowledge` —— 字节与 blocks 全在 `DATASET_REPORT.md:138-164`；许可分级 `DATA_SOURCES.md:41-78` | 指令 / 问答 / 对话续写 |
| ❌**非生成** | `clean_v3/{c4_zh(466 MB), wikipedia_cn(456 MB), classical_poetry(5.8 MB), 四大名著(1.7~2.5 MB)}.txt` `DATASET_REPORT.md:141-164` | 纯文本语言建模，无「输入→目标」对 ⇒ **不作接口候选** |

## 2 评估标准与评测脚本（它自己用什么判分）

| 指标 / 口径 | 评测脚本 | 出处 |
| :-- | :-- | :-- |
| 对话可读性：字符 n-gram 重复率 rep2/3/4、空白占比、distinct-1/2、轮次结构 | `inference/scripts/eval_dialogue.py` | 指标定义 `:83` `:97` `:106` `:115`；口径说明 `:252` |
| 多轮：**收尾率 / 自开轮次率 / 收不住率**；崩溃 = rep3>0.3 或 len<=2 或首轮打满上限 | `inference/scripts/eval_multiturn.py` | `:168-188`；崩溃判据 `:182` |
| 意图跟随 28 条：关键词判据 expect/forbid + good/bad **已知答案对照** + 逐条原文落盘 | `scripts/intent_probe.py` | `:1-33`（先 `--selftest`） |
| 逐源 CE(nats)，real / shuffled / unigram **三层已知答案链** | `scripts/per_source_ce_probe.py` | `:1-26` |
| 配对 CE（同批窗口消 eval 噪声）+ 块间 sd + 每 token CE 分位数；另 val 污染率（必须流式扫，建全量索引 OOM 过两次） | `scripts/ckpt_paired_eval.py`、`scripts/val_train_contamination_probe.py` | `:1-20`、路由 `AGENTS.md:248-262` |
| 答案精确匹配 EM（按位数分项）—— **算术子项目专用，非主模型口径** | `nano_arith/train.py` | `:5` `:37` |
| **无 BLEU / ROUGE / BERTScore 脚本**（全仓 grep 实测，仅第三方示例 `data/external/TheAlgorithms-Python/machine_learning/loss_functions.py:488`）；困惑度以 **CE(nats)**（每 token CE 分位数）表达 | — | `scripts/ckpt_paired_eval.py:4-16` |

**它写死的口径纪律（逐条）**：单点 val 不可信、只信配对 `AGENTS.md:139`；`best.pt` 是 22 个评估点里噪声选出的 `:144`；`results.csv` 不可跨口径比较 `:159`；rep3 对长度/结构敏感、跨类型必须长度匹配 `:168`；口径改动必须量「有效 / 可达 token 占比」`:176`；比数字前先核对源集合、分母、口径 `:181`；**分段验收必须报两把尺子 + 汇总会平均掉灾难、至少读完一个单源** `:186`；判据启动前写死、事后不许挑 `:196`；prompt 标签必须与训练语料一致，否则喂 OOD 产生虚假主场优势 `inference/scripts/eval_dialogue.py:32-37`；「EOS 率」曾把轮次截断当收尾的口径 bug `inference/scripts/eval_multiturn.py:13-19`。

## 3 相关经验（README / doc / DATA_SOURCES 的坑与结论，逐条 `文件:行`）
1. **许可·非商用**：`CodeExercise-Python-27k` 是 **CC BY-NC-SA 4.0**，商用必须剔除 `code_alpaca_dialogue.txt` `DATA_SOURCES.md:45-49`；`COIG-CQIA` README 标 "More Information Needed"，知乎子集版权复杂、仅研究用 `DATA_SOURCES.md:56-60`。
2. **许可·未核实**：C 类早期抓取（lccc / kdconv / zhihu_kol / multi_turn / glm / muice / dailychat）来源许可未完全核实，zhihu_kol 风险**高** `DATA_SOURCES.md:64-78`；商用前逐项核查或换干净源 `DATA_SOURCES.md:86-89`。
3. **许可·后加数据**：Belle = GPL-3.0 + 仅限研究、WildChat 中文子集 = AI2 ImpACT-LR、sharegpt = Apache-2.0、LCCC 标 MIT 但上游历史上「仅供研究」`[推断]` `analysis/belle_wildchat_import.md:221`、`analysis/zh_dialogue_sourcing.md:188-191`。
4. **采样/切分陷阱**：val 曾被「双标准切分」破坏 —— 4 个对话文件占 train 7.88% 却占 val 45.70%（放大 5.8×），两个 split 实际不是同一个任务 `DATASET_REPORT.md:15`、`:126`。
5. **采样陷阱**：loss masking 让 88% 语料白读、`pack_align=True` 让 67~73% token 永远采不到 ⇒ 必须 `use_doc_packing: true` + `pack_align: false` `AGENTS.md:176-180`。
6. **评测陷阱**：`DATASET_REPORT.md` 里「val CE 标准误 1e-3 nats」的说法**是错的、不许引用** `DATASET_REPORT.md:281` vs `AGENTS.md:139-143`。
7. **数据形态**：`v3_dlg` 是真多轮，终止符逐轮覆盖且**必须在 bin 上量**（文本层 grep 得 0）`AGENTS.md:203-204`；改了清洗/标注逻辑旧 bin 不会自己变对、必须重建 `AGENTS.md:276-277`。

## 4 可复用资产（路径 + 一句话作用 + 能否直接用于生成）

| 资产 | 作用 | 用于生成？ |
| :-- | :-- | :-- |
| `data/chinese/char_tokenizer.json` | char-level 词表 8192、`<eos>`=128、四区稀疏分区 v3 `DATA_SOURCES.md:11-12` | ✅ 直接可用，解码即文本 |
| `data/chinese/prepare.py` | txt → `train/val_char*.bin` + manifest、`--val-all` 统一切分 `:1-35`；配套清洗 `clean_corpus.py` / `filter_mt_translation.py` / `pii_scrub.py`（默认 dry-run、`--apply` 才写、拒绝原地覆盖 `AGENTS.md:265`） | ✅ 数据加载 / 切分 / 预处理（**本次纯只读，未跑**） |
| `training/dialogue_stream.py` + `training/masking.py` | 「按句滑窗 + `<resp>` 内联」，`prompt()` / `loss_token_spans()` 标出回复区间 `:1-45`；`resp_span`/`eos_line` mask 只对回复算 loss | ✅ **正是「输入 → 目标文本」的切分器与监督 mask** |
| `inference/scripts/sample_py.py` | 温度 → 重复惩罚 → top-k → 多项式采样的权威采样器 `:1-25` | ✅ 输出卡的解码入口 |
| `eval_dialogue.py` / `eval_multiturn.py` / `intent_probe.py` / `per_source_ce_probe.py` / `ckpt_paired_eval.py` | 见 §2 | ✅ 生成侧评测入口（**不要另写采样/评估脚本** `AGENTS.md:275`） |

## 5 ★ 接口点（给文本生成卡片框架选 1–2 个）
- **首选 `gsm8k_cot_dialogue.txt`（4.75 MB / 7,477 样本，MIT）+ 指标 = 配对 CE（`scripts/ckpt_paired_eval.py`）+ `####` 终局答案 exact match。** 理由：① 天然是「**输入卡**（题面→模因）→ **思维卡**（推理步→推理步）→ **输出卡**（模因→可读推理文本）」的三卡闭环，`用户：/模型：` 逐条可切成 (输入, 目标文本) 监督对，格式**实测**于该文件首块；② 许可 MIT 无商用风险 `DATA_SOURCES.md:46`；③ 目标文本**自带唯一正确答案** `#### 72` ⇒ 评测客观可判、不依赖人评（EM 口径现成于 `nano_arith/train.py:5,37`）；④ 4,166,488 字符 / 占 train 0.45% `DATASET_REPORT.md:152`，规模适中、适合小卡框架先跑通。
- **次选 `stages/v3_intent5/{intent_mined,intent_synth,topic_switch}.txt`（336,230 + 26,669 + 97,105 = 460,004 B）+ 指标 = `scripts/intent_probe.py` 28 条关键词判据。** 理由：① 单轮「一句话输入 → 期望言语行为的回复」是最短的输入卡/输出卡监督对，自研挖掘+合成、无需清洗；② 评测脚本现成且**带 good/bad 已知答案对照 + `--selftest`**、命中与失格原因逐条落原文可复核 `scripts/intent_probe.py:1-33`；③ 有留出探针原句不进训练 `AGENTS.md:210`，是干净 holdout。
- **不推荐**先用对话大盘（sharegpt / belle / wildchat）：规模大但指标只有 rep3 / distinct / 收尾率这类**代理指标**，不判「答没答对」`inference/scripts/eval_dialogue.py:8-13`；且三家许可分别为 Apache-2.0 / GPL-3.0 仅研究 / AI2 ImpACT-LR，商用需逐家核。

## 6 未验证清单（读不到或没实测的，点名）
- **未验证·许可**：`escov_zh.txt` 与 `stages/v3_intent5/*` 未在 `DATA_SOURCES.md`（最后更新 2026-09-06，`:5`）登记；`intent_*` 记为 MIT 属 `[推断]`；`sharegpt_zh_38k.txt` 归到 Apache-2.0 源是按 block 数 38,537（`analysis/belle_wildchat_import.md:177`）对 `analysis/zh_dialogue_sourcing.md:188`（38,557 段）的 **`[推断]`**，仓库内无直接导入记录。
- **未验证·脚本**：**没有**任何脚本解析 GSM8K 的 `####` 答案做 EM —— 全仓 grep `####` 仅命中 `training/rl/qual_probe.py:12` 的分隔符打印；接口点①的 EM 读数**需自行实现**。
- **未验证·可移植性**：`nano_arith` 的 EM 是独立 17-token 算术子项目口径 `nano_arith/README.md:13-18`，与主模型 char 词表不互通，能否直接移植**未测**。
- **未验证·效果**：本报告按停止边界**未跑任何模型 / 训练 / 采样**；「生成类可用」只核了**数据格式与脚本存在性**，未验证下游效果。
- **未验证·口径**：`DATASET_REPORT.md:138-164` 是 **v2 口径**构成表；v3 分段（lang/dlg/know/intent）的逐源字符占比需另读各 `manifest_v3_*.json` 的 `source_breakdown`，**未逐段抄录**；`out/` 与 `data/external/`（第三方克隆）未逐一核对，报告未引用其结论。

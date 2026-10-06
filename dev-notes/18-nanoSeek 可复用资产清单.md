# 18-nanoSeek 可复用资产清单（只读调研，2026-10-07）

问题：nanoSeek 里有哪些能被 DTSeek 直接复用的资产。glob / `torch.load` / tokenizer 加载均为**现场实测**，未验证项逐条标注。

## 一、资产表

| 资产 | 路径 | 规模 | 格式 | DTSeek 能否直接用 | 用途 |
|---|---|---|---|---|---|
| 清洗后对话语料（glob 已命中 18 个） | `nanoSeek/data/chinese/clean_v3/*dialogue.txt` | 1.12 GB / 12,347,659 行 | txt，`用户：`/`模型：` 块、空行分隔 | **是**（已覆盖） | reply_pick/完形/代词/情绪/人物/关系卡的挖掘底稿 |
| 同目录**对话型但未命中**的文件 | `clean_v3/{belle_multiturn,sharegpt_zh_38k,wildchat_zh,qa_knowledge}.txt` | 300 MB / 179 万行 | 同上 | **需转换**（名不含 `dialogue`，glob 命不中） | 同上，白捡的量 |
| 意图跟随语料 | `new_sources/intent_mined.txt` 336KB/1805 行、`intent_synth.txt` 27KB、`stages/v3_intent5/` 同套 | ~0.7 MB | 同上，**标签不落盘** | 需转换 | intent 卡训练候选 |
| 换话题语料 | `new_sources/topic_switch.txt`（400 处 `<topic>` 标） | 97 KB / 2,799 行 | 同上 + 轮首机制符 | 需转换 | 轮级结构监督（非槽位） |
| **意图标签集（唯一结构化标注）** | `scripts/intent_probe.py::INTENTS` | 14 意图 × (prompt, expect/forbid 正则, good/bad 已知答案) | py 常量 | **是** | intent 卡的评测判据与类别体系 |
| 意图评测历史结果 | `analysis/intent_probe_*.json`（12 个） | 各 6–15 KB | json：逐条原文+命中原因 | 是（作回归对照） | 评测基线 |
| 分词器（DTSeek 自带，非 nanoSeek） | `../nano-char-tokenizer/src/nano_char_tokenizer/tokenizer.json` | 158 KB，vocab=8192 | HF tokenizers | **是**（`engine.py:80` 在用） | — |
| nanoSeek 分词器 | `data/chinese/char_tokenizer.json` 8192 / `tokenizer.json` BPE 8000 | 126 KB / 593 KB | HF tokenizers，**sha 与上者不同** | 否（DTSeek 不加载它） | 只有 rebuild bin 时才需要 |
| 模型 ckpt | `nanoSeek/out/*/[last,best].pt`（`base_v3_intent5/last.pt` 等，各 ~635 MB） | 23 GB（整个 out/） | nanoGPT 式 `{model,optimizer,config,…}`，**无 `format` 键** | **否**（实测 `read_base` 抛 ArtifactError；`split_checkpoint.py` 也要求 `task_specs`/`decoders`，它没有） | 结构移植已由 dev-notes/04 完成，权重不可直接加载 |
| 意图语料生成器 | `data/chinese/build_intent_stage.py`（29KB，`--selftest`/`--dump`）、`build_topic_stage.py` | — | py | **是**（纯文本产线，零 GPU） | intent/slot 卡语料扩产的现成纪律样板 |
| 清洗/脱敏/切分产线 | `clean_corpus.py`（dry-run 默认）、`pii_scrub.py`、`import_external.py`、`build_stages.py` | — | py | 需转换（`build_stages` 产 nanoSeek bin） | 语料治理 |
| 对话评测 | `inference/scripts/eval_dialogue.py`、`eval_multiturn.py`（口径：收尾率/自开率/distinct-n） | — | py | 需转换（采样入口绑 `out_dir/best.pt`） | 生成卡评测的指标口径可抄 |
| 配对重评 | `scripts/ckpt_paired_eval.py` | — | py | 需转换（评 nanoSeek ckpt 非 DTSeek 卡） | "单点 val 不可信"的配对纪律 |
| 单流多轮格式 | `training/dialogue_stream.py`（`<resp>`id140 + `<eos>`，按句滑窗） | 21 KB | py + txt 约定 | 需转换 | 多轮卡的数据形态参考 |
| 主训练脚本 | `training/train.py` 99 KB + `configs/*.yaml` | — | py | 否（GPT-MoE 语言模型训练，与卡训练不同构） | 仅架构/稳态经验 |
| DSH 真实会话语料 | `data/dsh/dialogue.txt` | 894 KB / 9,832 行 | 同对话 txt 格式 | 需转换（不在 clean_v3） | 真实中文指令/对话 |
| 已弃合成指令语料 | `data/chinese/_discarded/{math_cot,math_cot_v2,code_algo_v2,agent_tools,baike_qa,code_syntax}_dialogue.txt` | ~125 MB | 对话 txt | 需转换（弃用原因**我未读**，未验证） | 指令类文本；生成器 `data/scripts/synthetic_generator.py` 等仍可再生 |
| 外部原始语料 | `data/external/`（chinese-poetry 等 889 MB）+ 仓库外 `/home/vesita/datasets/NLP/`（2.4 GB） | 889 MB / 2.4 GB | 仓库/数据集克隆 | 需转换 | 再生原料 |
| 许可/出处 | `nanoSeek/DATA_SOURCES.md`、`LICENSE`（MIT，继承 nanoGPT）、`THIRD_PARTY_NOTICES.md` | — | md | — | A/B/C 三类：C 类（lccc/kdconv/zhihu 等）**需查证**；`code_alpaca_dialogue.txt` 为 CC BY-NC-SA **非商用** |

## 二、三样最该先用的

1. **`clean_v3` 里 glob 已命中的 18 个 `*dialogue.txt`（1.12 GB / 1234 万行）** — 零改动，`corpus.py` fail-closed 已通过；对应**选择卡族（dev-notes/13 择优回复 reply_pick、完形 cloze_fill）**的问句-答句池与负例池（`reply_pick/dataset.py:131` 就是按 `用户：`/`模型：` 块取轮对）。
2. **`intent_probe.py::INTENTS` 14 条意图标签** — 全仓唯一现成的**结构化**标注（prompt + 期望言语行为 + expect/forbid 正则 + good/bad 已知答案），对应 **dispatch.py 待补的 `intent` 卡**（`INFO_BITS` 的 intent 位，`registry_problems` 卡着未注册就全线拒答）：先拿来当类别体系与评测口径，判据自带自测。
3. **`build_intent_stage.py`（挖掘+模板合成一体，`--selftest`/`--dump` 纪律）** — 对应 **`slot` 卡（time/place/object 槽位抽取，dispatch `INFO_BITS` 三位）**：nanoSeek 没有槽位标注，但它证明了"正则挖掘 + 受控合成 + 探针留出"能凭空造出密集监督语料，同一套做法是 slot 卡语料的最短路径。

## 三、明确**不存在**的

- **骨架级标注（骨架 id + 槽位指派）：没有。** 全仓无任何"句子骨架/模板 id"落盘数据；最接近的是三样**轮级/类级**东西：① DTSeek 自己的 `dialogue.py::CARD_TEMPLATES`（骨架在代码里，不是数据）；② `INTENTS` 的 14 个**意图类**标签（不是槽位）；③ `topic_switch.txt` 的 `<topic>` 与 `persona_identity.txt` 的 `<resp>…<eos>`（**轮**级标记）。
- **词组级（phrase/span）标注：没有。** 没有任何词或短语的 span 标注文件；最接近是 DTSeek dev-notes/07 自建的成语词典（来自 mapull，不在 nanoSeek）。
- **`time/place/object` 槽位数据：没有**（dispatch 的 `slot` 卡在 nanoSeek 侧零供给）。
- **结构化标注数据集：没有。** 全仓 `.jsonl` 只有 1 个软链（`raw_all/qa_knowledge_274k_zh.jsonl`，纯 QA 无标签）。
- **可加载的 DTSeek 产物：没有。** nanoSeek 无 `base_encoder.pt` / `cards/*.pt`；DTSeek 的 `checkpoints/` 是自产。

## 四、`corpus.py` 实测覆盖

代码（`src/dtseek/tasks/corpus.py:21-32`）：`CORPUS_DIR = $DTSEEK_CORPUS_DIR 或 /home/.../nanoSeek/data/chinese/clean_v3`，`CORPUS_PATTERN = "*dialogue.txt"`，命中 0 抛 `FileNotFoundError`。调用方：`reply_pick / cloze_fill / pronoun / sentiment / negation / ownership / person / relation` 共 8 个卡的 dataset。

```
$ python3 -c "...glob(str(Path(DIR)/'*dialogue.txt'))..."
PATTERN: /home/vesita/coding/my/nanoSeek/data/chinese/clean_v3/*dialogue.txt
HITS: 18
  code_alpaca_dialogue.txt 10532481  coig_code_dialogue.txt 2047984  coig_cqia_dialogue.txt 10939356
  coig_logic_dialogue.txt 1148570  coig_math_dialogue.txt 300953  coig_other_dialogue.txt 9429859
  coig_wiki_dialogue.txt 97155903  dailychat_dialogue.txt 109653  deepseek_r1_distill_dialogue.txt 601145745
  glm_dialogue.txt 13577151  gsm8k_cot_dialogue.txt 4753284  identity_dialogue.txt 1684
  kdconv_dialogue.txt 2605151  lccc_dialogue.txt 22659836  muice_dialogue.txt 576852
  multi_turn_dialogue.txt 600  qwen3_235b_distill_dialogue.txt 269449054  zhihu_kol_dialogue.txt 76877415
--- all files in clean_v3 not matched ---
 MISS belle_multiturn.txt 71430422  sharegpt_zh_38k.txt 143684697  wildchat_zh.txt 28844849
 MISS qa_knowledge.txt 55797724  c4_zh.txt 291919401  wikipedia_cn.txt 426578530  classical_poetry.txt 6058993
 MISS 三国演义/水浒传/红楼梦/西游记(.txt)  四大名著  CLEANING_REPORT.md  VERIFY_REPORT.md
```

**没覆盖的**：4 个对话型文件（belle/sharegpt/wildchat/qa_knowledge，300 MB——glob 想吃这批只需软链成 `*_dialogue.txt` 或扩 pattern）；`new_sources/`（intent/topic/escov）、`stages/v3_*`、`data/dsh/dialogue.txt` 全部不在 `CORPUS_DIR` 下。**长篇/百科/诗歌未命中是设计使然**（`corpus.py:28` 注释写明只取对话类）。

## 五、实际读过的文件

DTSeek：`src/dtseek/tasks/corpus.py`、`engine.py`、`artifacts.py`、`dispatch.py:100-189`、`builtin/reply_pick/dataset.py:100-159`、`scripts/split_checkpoint.py`、`pyproject.toml`/`uv.lock`（grep）、`dev-notes/04`、`dev-notes/07`（头部）。`../nano-char-tokenizer/`：`__init__.py`、`README.md`、`build_vocab.py`（头部）。
nanoSeek：`DATA_SOURCES.md`、`AGENTS.md`（注入）、`data/chinese/readme.md`、`build_intent_stage.py`/`build_stages.py`/`build_topic_stage.py`（docstring）、`scripts/intent_probe.py:1-122`、`training/dialogue_stream.py`（头部）、`cli.py`（头部）、`data/dsh/export.py`（头部）；目录 `ls/du`：`data/*`、`analysis/`、`out/`、`training/`、`scripts/`、`_discarded/`、`stages/*`；样本 `head`：identity/dailychat/muice/lccc/coig_math/sharegpt/belle/qa_knowledge/escov/intent_*、`topic_switch`、`persona_identity`（仓库外，仅开头）。
**实测**：glob 全量输出（上节）、`sha256` 三个 tokenizer、`Tokenizer.from_file` 三者 vocab、`torch.load(out/base_v3_intent5/last.pt)` 键列表 + `read_base` 拒绝。

**未验证/未读**：`_discarded/` 各文件的弃用原因；`out/` 23 GB 只 `ls` 未逐一核对；`intent_probe_*.json` 只 head 了 `v3intent2` 一份；`local/`、`temp/`、`pre-research/`、`nano_arith/` 未深入；未实际把 nanoSeek txt 喂进 DTSeek 的 dataset 跑一遍（只读了代码路径）；`_backup_*` 与 `raw_all` 未逐一读。

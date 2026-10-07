# P19 PREREG —— 让 negation 卡与线上核匹配（修复在产 D1 缺陷）

落盘时间：见本文件 mtime（**必须早于首次 `src/` 改动**，跑完判据一字未改）。
只写 `experiments/fix_negation/`、`logs/`、`/tmp`，以及 `src/dtseek/tasks/dialogue.py` 的 **`DEFAULT_ATTACH` 那一行**与必要的 `tests/` 更新；
不改 `render.py` / `card_contract.py` / 核 / 其它卡 / 既有实验目录 / `dev-notes/` / `methodology/`；无 `git commit/stash/checkout/restore/clean`。

> **边界句（必读）**：本单元只回答 **「能否把该卡修到与线上核匹配」**；
> **不回答「整条对话链质量提升多少」**（那是全体卡共同效应，见 `e2e_d1` 的归因边界）。

## 0. 已知事实（引 P16/P15/P13，不重新论证）

- `src/dtseek/tasks/dialogue.py:105` `DEFAULT_ATTACH = experiments/compose_ops/artifacts/cards/negation.pt`
  （`split_from=checkpoints/arm_neg5_seed42.pt` ⇒ 来源核 `801657fb…`）；`engine.py:38 DEFAULT_BASE = checkpoints/base_encoder.pt`
  ⇒ 线上核 `dc27db53…`。P16：只换基座、五卡逐字相同 ⇒ 线上核 M_cls .2801/.2906、来源核 1.0000。
- `DIALOGUE_REQUIRED_BITS["plain"]=("mood","neg")` ⇒ negation 触发率 1.0000（全部 plain 输入）。
- **回退路径（写死）**：`DEFAULT_ATTACH = ("experiments/compose_ops/artifacts/cards/negation.pt",)`
  （该文件 sha256 = `ba1a80f9af81b1a2…`，来源核 `801657fb…`）。回退 = 把这一行改回原值即可，无其它依赖。

## 1. 判据（跑前写死；F0–F5 全部实测，区分实测/推断）

| # | 判据 | 门槛（写死） |
|---|---|---|
| **F0** | sha 自校验：按 `prod_card_audit/a1_shas.py` 的方案（state_dict **原始键序**拼张量字节 ⇒ sha256） | 复现线上核 **`dc27db5337d16f18…`**，脚本 exit 0 |
| **F1（主）** | 端到端 `dialogue.respond` 口径（**不用 `evaluate_task`**）negation 首切片 `M_cls`，2 seed（42/43），改后 | 两 seed **均 ≥ .98**（来源核水平 ≈1.0000 的“附近”定义为 ≥.98），且 **Δ(改后−改前) 两 seed 同号、\|Δ\|>2SE** |
| **F2** | 免费规则地板（P15 同电池 8 条 + `max_naive`）对照选中卡 | 报 **卡 − 地板**（两 seed）；**≤ 0 ⇒ 如实写「未超过免费规则」**（含“并列=0”） |
| **F3** | 默认行为等价性：改前后同批输入重放 | **除 negation 卡证据及其派生文本子串外，不一致 = 0**（全 5 能力 × 2 seed） |
| **F4** | 全库 `pytest` | **全绿**；任何依赖旧卡的断言逐条说明改了什么、为什么 |
| **F5** | 老卡门禁 + 回退路径 | 四老卡 `evaluate_task`（combo_full、线上核）`exact` Δ ≥ **−各自噪声带**（pronoun **2.83** / sentiment **0.41** / relation **1.39** / person **0.33** pt，两 seed）；原 `DEFAULT_ATTACH` 值写进报告 |

**判定三选一（优先级写死，见下）**：
1. F1 未达标 ⇒ **无法修复**（给原因，不许硬选）；
2. F1 达标且 F2 **两 seed 均 ≤ 0** ⇒ **修复但收益被免费规则解释**（同时照实报 F3/F4/F5 结果）；
3. F1 达标且 F2 > 0 且 F3/F4/F5 全过 ⇒ **修复成立**。
   若 F1 达标但 **F3 或 F4 失败** ⇒ **不判「修复成立」**，如实判「修复不通过验收」并给原因。

## 2. 步骤（每步独立落盘；先 PREREG + 代码 + 冒烟，再全量）

1. **F0/扫描（`s1_shas.py`）**：先自校验 `dc27db53…`；再扫 `checkpoints/`、`experiments/`、`training/` 下**所有** `task=="negation"` 的 `.pt`，
   逐张报来源核 doc sha（`train_args.base` → 该文件 doc_sha；否则 `train_args.split_from` → 该 ckpt 的 `doc_encoder` doc_sha；都没有 ⇒ 报 `NONE`）。
2. **候选在线上核上的指标（`s2_e2e.py`，口径照 `e2e_d1`）**：每张候选卡挂到 `checkpoints/base_encoder.pt` + 默认四卡，
   跑 `respond(type_="plain")` 于 negation `eval_S`（`capability_map/cache/negation_ordered_s{42,43}.pkl`，公式与 `e2e_d1/common.py::split_of` 逐字相同），
   报 `M_detect / M_cls / M_span`。
   **选卡顺序（跑前写死，避免“在评测集上挑模型”）**：先评 `checkpoints/negation_accept_card.pt`（多处在用的线上实现，**主候选**）；
   **仅当主候选 F1 不达标（两 seed 有一 < .98）时**才评 `_e12`，并如实披露这一步是次候选。
   ⇒ **主候选达标 ⇒ 直接换卡（最小改动），跳到第 4 步**；两张都只有线上核来源但仍不达标 ⇒ 走第 3 步重训。
3. **重训（仅在无现成解时）**：核**冻结**（只训头）、`base = checkpoints/base_encoder.pt`、2 seed（42/43），
   数据 = negation 任务既有生成口径（报 `train_args` 与文件清单）；**同时报免费规则地板**。
4. **F2 免费规则对照（`s4_floor.py`）**：P15 电池 8 条 = `majority / len_bucket / first_char / last_char / punct_pattern / lex_rule(词表字面) / lex_first_key / n_cue(计数，D2)`，
   `fit = train_S`、`eval = eval_S`、`lookup` 逐字复用 `free_rule_floor/rules.py` 口径；目标 = `labels[:,0]` 且只取 `t_labels>0`（与 `cls_acc` 同分母）。
   **另加 detect 口径地板**（全 eval 样本、目标 = 有无否定证据）以暴露 cls 口径的退化，两个口径都报。
5. **改配置 + 验证（`s3_before.py` 先捕获 before，改行后再跑 `s3_after.py`）**：
   - 只改 `dialogue.py` 的 `DEFAULT_ATTACH` **一行**；
   - 端到端前后对比（`respond` 口径）`M_detect/M_cls/M_span` ± SE × 2 seed；
   - F3 等价性重放（投影比对，不一致必须 = 0）；
   - 全库 `pytest`；
   - F5 四老卡门禁（`evaluate_task`、combo_full、线上核，两 seed）。
6. **停止边界**：做完 1–5 就停。不改别的卡/别的默认值、不动核、不重训四张生产卡、不写 `dev-notes/`/`methodology/`、不派子代理。

## 3. F3 投影定义（写死，避免事后放宽）

对每条 `respond` 记录取投影 P：
`{type, kind, terminal, reason, plan, cards_run, evidence[card != "negation"], text_without_neg_clause}`，
其中 `text_without_neg_clause` = 去掉模板 `CARD_TEMPLATES["negation"]` 产生的子串（正则 `并且有否定成分『[^』]*』`）并把 `；。` 收干净。
同时另报一条**严格投影**（仅去 negation evidence，`text` 原样比）作为披露项。**两投影不一致数都必须 = 0**（严格投影若 ≠ 0 需逐条归因到 negation 模板子串）。

## 4. 命令与产物

- 扫描/评测：`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s1_shas.py` 等；
- 训练（仅第 3 步）：`systemd-run --user --unit=… --collect --property=WorkingDirectory=…/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 …`，**绝不用 `setsid nohup &`**；GPU 忙就排队，**不杀他人进程**。
- 产物：`experiments/fix_negation/{PREREG.md,REPORT.md,common.py,s1_shas.py,s2_e2e.py,s3_equiv.py,s4_floor.py,s5_gate.py}`、`results/*.json`、`logs/*.log`。

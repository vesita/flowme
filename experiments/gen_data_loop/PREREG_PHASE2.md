# Phase 2 预注册 —— D4（这套数据生成模式能不能产出**可用**数据）

> **落盘时间：2026-10-06 21:16 CST**，早于本目录的**首次训练**。
> 本文写死判据与配方；`REPORT_PHASE2.md` 只回填实测，**不改门槛**。
> 口径同 `PREREG.md`：「实测」= 本目录跑出来的数字；「推断」= 由数字推出；「引用」= 数据侧已核验的既有实测。

## 0. 范围与分工（写死）

- **数据侧不动**：`data/{train,test,adversarial,ctrl_*,natural}.jsonl` 与 `data/stats.json` 已构建并独立复核，
  本阶段**只读**。不重跑 `build_data.py`，不改 `data/`。
- 本阶段**只做 D4 族**（D4(a) / D4(b) / V-bonus）+ 引用 D1/D2/D3 + **D6′（口径订正，见 §5）**。
- **不走联合训练**（冻结核 + 只训头）⇒ **D5 不适用**，如实报「不适用」，不伪造数字。
- 写入范围：`experiments/gen_data_loop/`（新增脚本与结果、本文、`REPORT_PHASE2.md`）、`logs/`、`/tmp`。
  不改 `data/`、其它 `experiments/`、`src/`、`training/`、`tests/`、`dev-notes/`；
  不 `git commit/stash/checkout/restore/clean`。

## 1. 数据地基（引用，不重跑）

`data/stats.json` + `PREREG.md` §4.1 已核验六项：

| 项 | 实测（引用） |
|---|---|
| D1 标签可核对性 | 提案臂标签 **100%** 由 P2–P5 谓词判出；非谓词判定（natural 人工标签、配额下采样、注入臂 anchored_select 口径）单列 |
| D2 接受率 | train **0.2803** / test **0.2802** / adversarial **0.3677**，全在 **[0.15, 0.85]** |
| D3 无免费捷径 | 提案臂 train 最高朴素规则 `lex_member` **0.797 < 0.90**；对抗集所有朴素规则 ≈0.5（最高 `neg_char` 0.5408） |
| 交集 | text / input / full 两两不相交**全 0**（四池 + natural） |
| 截断 | 截断丢弃 `ctrl 0 / proposal 8`（train），**本阶段再加**编码期截断守卫（§4） |
| D6 数据侧 | 见 §5（原判据 + D6′ 两个数字都报） |

## 2. 两条臂与配方（跑前写死，**唯一变量 = 数据来源**）

| 臂 | 训练/测试数据 | 训练行数 |
|---|---|---|
| **P（提案臂）** | `data/train.jsonl` / `data/test.jsonl` | 8000 |
| **C（注入式对照臂）** | `data/ctrl_train.jsonl` / `data/ctrl_test.jsonl` | 8000 |

- 卡 `cand_validity`（候选有效性）：classes = (`无效`(背景/不发射), `有效`)、`max_steps=1`、`max_len=96`、
  `emission=single`、`segment_policy=sentence`、`cls_weight_bg=1.0`。
- 版式两臂逐字一致：`原文：{full}|候选：{candidate}|类：{category}`（数据侧已保证）。
- **只训头**：核 `checkpoints/base_encoder.pt` 冻结（`requires_grad` 实测全 0，否则该跑作废）+ `RobustARSliceDecoder`。
- 配方（**两臂逐字相同**）：epochs **12** × steps_per_epoch **150** = **1800 步**、batch **64**、lr_head **1e-3**、
  cosine、AdamW wd **1e-4**、grad clip **1.0**、seed **42 / 43**、`--samples 8000`（与文件条数不符直接拒绝）。
- 机制走**现有**入口：`cand_card.py` 注册本进程内任务卡 → `run_card.py` 自检 → 原样委托
  `training/train_task_card.py`（不改 `training/`）。
- 训练一律 `systemd-run --user --unit=<名> --collect --property=WorkingDirectory=... --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1`，
  **绝不用 `setsid nohup &`**；开训前查 `ps` 与 `systemctl --user is-active 'dtseek*'`，**绝不杀他人进程**。

### 2.1 空测试自检（每个跑必打印，缺任一项 = 该跑无效）

- `SELFTEST_1`：核参数总量 / **核可训参数量（必须 0）** / 头参数量 / 真实总可训参数量 / 设备。
- `SELFTEST_DATA`：训练集 n、正负条数、**多数类基线**、盲猜 0.5、shuf 标志。
- `SELFTEST_TRUNC`：逐样本**实际编码长度**（attention_mask 求和）≤ `max_len`，超长条数必须 **0**。
- **随机标签对照真执行**：`--shuf` 时标签整体 `randperm`（1:1 计数不变，shuf-seed = `seed×1000+7`），
  span 按新标签重算；`SELFTEST_DATA.shuf=1` 必须打印，且正负计数与未打乱逐字相同。

## 3. 评测集与指标（跑前写死）

**两条臂都评**同样三个集（主表 2×3×2 seed）：

| 集 | 角色 | n | 多数类 |
|---|---|---|---|
| `natural.jsonl` | **主** | 200 | 0.50 |
| `adversarial.jsonl` | 附（提案臂对抗） | 1200 | 0.50 |
| `ctrl_adversarial.jsonl` | 附（注入臂对抗） | 1200 | 0.50 |

外加 V-bonus 用的「自己 test」：P→`test.jsonl`，C→`ctrl_test.jsonl`（各 1600）。

- **acc（主指标）** = 逐样本**类**判对的比例 = 第 0 步 argmax 类别 == 真值类
  （判有效 ⇒ 恰发射一条且 label=1；判无效 ⇒ 不发射）。
  与 `evaluate_task` 的 `cls_acc` / `bg_fp` **逐数对账**（复算式
  `(cls_acc×n_pos + (1−bg_fp)×n_neg)/n`），不一致直接退出。
- **exact**（附报，不过线）= 类 + 区间逐位一致（无效要求不发射）。
- **SE**：单集 `SE = sqrt(p(1−p)/n)`，按实测 p 算，同时报 n=200 处的参考值。
- **max_naive / 卡−max_naive**：见 §6 硬要求 1。

## 4. 静默截断守卫（跑前写死）

`GenericTaskDataset` 会把 `start/end` 钳到 `max_len−1`、tokenizer 会静默截断超长文本 ——
两者都会把「漏一半」伪装成正常样本。故：

- 训练前对**训练集与每个评测集**逐行断言 `len(text) ≤ max_len` 且 `span.end ≤ max_len`，违者直接抛错；
- 同时用 `attention_mask` 求和复核**实际编码长度** ≤ `max_len`，报 `SELFTEST_TRUNC`，超长条数必须 0。

## 5. D6 / D6′ —— **判据订正（两个数字都报）**

**订正时间：2026-10-06 21:16 CST（本文件落盘时）；订正人：D4 实验子代理；**
**订正依据：`dev-notes/16 §7.6.1`（Lead 已订正），此处按其要求落进 PREREG 并注明理由与时间。**

**订正理由（口径错，不是数据塌缩）**：原判据拿「distinct 候选率 ≥ 0.9×注入臂」比的是
**闭集词表**（提案臂候选来自约 230 词的封闭词表、均长 2.21 —— 闭集正是「标签可被谓词判定」的前提）
对 **任意片段**（注入臂候选 distinct 率天然 1.0）⇒ 惩罚了让这条路能成立的那个性质，
两个清单**不可比**。

| 判据 | 口径 | 实测（引用 stats.json） | 过? |
|---|---|---|---|
| **D6（原，未订正）** | distinct 候选率 ≥ 0.9×注入臂 | 提案 0.238 vs 注入 1.0 → 门槛 0.9，**0.238 < 0.9** | ❌ **不过** |
| **D6′-1** | 采样后 distinct 候选率 ≥ **原始提议池** | 池 0.158 → 采样后 **0.238**（升 50%） | ✅ |
| **D6′-2** | 类别熵不低于注入臂 | **2.0201 vs 2.0201** | ✅ |
| **D6′-3** | distinct 输入率不低于注入臂 | **0.7406 vs 0.5** | ✅ |

⇒ **结论必须同时写出「原判据 D6 不过 + 新判据 D6′ 全过」；只报新判据 = 本条不通过。**

## 6. 判据（**跑前写死**，事后不改）

设 natural n = 200、多数类 M = 0.50，`SE_M = sqrt(0.5×0.5/200) = 0.035355 = 3.54pt`
⇒ **D4(a) 门槛 = M + 2×SE_M = 0.5707 ≈ 57.1%**。

| # | 判据 | 门槛（跑前写死） |
|---|---|---|
| **D4(a)** | **提案臂在 `natural` 上** | acc ≥ **0.5707**，**seed 42 / 43 各自满足**（2 seed 同号） |
| **D4(b)** | **提案臂优于注入式对照臂**（同一 natural、同 seed 配对） | `acc_P − acc_C > 2×SE_diff`，`SE_diff = sqrt(p_P(1−p_P)/n + p_C(1−p_C)/n)`（n=200），**2 seed 各自为正且过 2×SE_diff** |
| **D5** | 不伤老卡 | **本轮不走联合训练 ⇒ 不适用**；若将来走温启动联合，门槛 = Δ ≥ −各自噪声带（pronoun 2.83 / sentiment 0.41 / relation 1.39 / person 0.33 pt）。**本轮如实报「不适用」，不伪造数字** |
| **V-bonus** | 随机标签对照（**必做、真执行**） | shuf 臂在 `natural` 上 acc ≤ **0.5707**，**且**在自己 test 上 ≤ **0.53**（50% + 3pt），两 seed。**若 shuf 也过 D4 ⇒ 判据恒真、整轮作废** |

### 6.1 硬要求 1（追加）：**每一格都必须并列 `max_naive`**

`stats.json` 已给各集朴素规则读数，主表每一格必须并列：

| 集 | `max_naive`（规则） | 读数 |
|---|---|---|
| `natural` | `lex_member` | **0.765** |
| `adversarial` | `neg_char` | **0.5408** |
| `ctrl_adversarial` | `neg_char` | **0.5067** |
| `test`（P 自己） | `lex_member` | 0.7969 |
| `ctrl_test`（C 自己） | `substring` | 1.0000 |

并写出 **「卡 − max_naive」**。
**若卡 acc 低于该集的 `max_naive`，如实写「未超过免费规则」** —— 这是 `anchored_select` 的死法
（卡 0.69 vs 免费规则 1.00），**不许因为 D4(a) 过了就淡化它**。
评测脚本独立复算朴素规则（import `build_data.naive_battery`）与 `stats.json` 读数对账，不一致直接退出。

### 6.2 结论三分（跑前写死，**不许硬选**）

- **① 可用且优于注入式** = D1 ∧ D2 ∧ D3 ∧ **D4(a)** ∧ **D4(b)** ∧ **D6′** ∧ **V-bonus**（D6 原判据不过也必须如实写出）；
- **② 可用但不优于注入式** = D1 ∧ D2 ∧ D3 ∧ D4(a) ∧ D6′ ∧ V-bonus 过，但 **D4(b) 不过**；
- **③ 产不出可用数据** = **D4(a) 或 D3 不过**（或 D1/D2/D6′/V-bonus 任一不过）。

## 7. 结论必须写进去的两条限制（跑前写死）

1. **长度混杂**：natural 均长 **14.15** vs 提案臂 **25.62（−45%）** ⇒ D4(a)/(b) 的结论只能说
   「**在该长度分布下**」，**不能说「泛化到自然用法」**。
2. **样本量**：natural 仅 **200 行** ⇒ 门槛约 **57.1%**、把握度弱（SE 3.54pt），
   必须标 **「样本量限制」**；**边缘通过不得写成强证据**。

## 8. 运行清单（跑前写死）

| run | 臂 | shuf | seed | 产物 |
|---|---|---|---|---|
| 1–2 | P | 否 | 42 / 43 | `cards/cand_P_s{42,43}.pt` |
| 3–4 | C | 否 | 42 / 43 | `cards/cand_C_s{42,43}.pt` |
| 5–6 | P | 是 | 42 / 43 | `cards/cand_P_shuf_s{42,43}.pt` |
| 7–8 | C | 是 | 42 / 43 | `cards/cand_C_shuf_s{42,43}.pt` |

unit 名：`dtseek-genphase2`；日志 `logs/genphase2_<run>.log`；结果 `results/eval_<...>.json`。
**每步独立落盘**（一个 run 一个日志、一个结果文件），单跑 ≤ 几分钟量级。

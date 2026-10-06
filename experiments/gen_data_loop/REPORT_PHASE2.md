# REPORT Phase 2 —— D4（这套数据生成模式能不能产出可用数据）

> 判据与配方见 `PREREG_PHASE2.md`（mtime 2026-10-06 21:19:17 CST，**早于首次训练** 21:25:12）。
> 口径：**实测** = 本目录跑出来的数字；**引用** = 数据侧已核验的 `data/stats.json` / `PREREG.md`；**推断** = 由数字推出。
> 数据侧未改动（只读）；未 `git commit/stash/checkout/restore/clean`。

## 1. 两臂配置（唯一变量 = 数据来源）

| 项 | P（提案臂） | C（注入式对照臂） |
|---|---|---|
| 训练文件 | `data/train.jsonl` 8000（4000:4000） | `data/ctrl_train.jsonl` 8000（4000:4000） |
| 自己的 test | `data/test.jsonl` 1600 | `data/ctrl_test.jsonl` 1600 |
| 版式 | `原文：{full}\|候选：{cand}\|类：{cat}` | **逐字相同** |
| 卡 | `cand_validity`：classes 无效(背景)/有效、max_steps=1、max_len=96、cls_weight_bg=1.0 | 同一张卡 |
| 核 | `checkpoints/base_encoder.pt`，**core_total 1,688,460 / core_trainable = 0（冻结 ✅，实测 requires_grad）** | 同 |
| 头 | `RobustARSliceDecoder`，**head = 629,764（唯一可训参数）** | 同 |
| 配方 | 12 epoch × 150 step = **1800 步**、batch 64、lr 1e-3、cosine、wd 1e-4、clip 1.0、seed 42/43 | **逐字相同** |
| 训练耗时 | 30.8 / 31.4 s（0.0171 s/step，峰值显存 128 MB） | 30.9 / 30.9 s |
| shuf 耗时 | 30.6 / 30.8 s | 30.7 / 30.5 s |

空测试自检（每跑必打，实测全过）：
`SELFTEST_1` 核可训 = 0、头 = 629,764、总可训 = 629,764；
`SELFTEST_DATA` n=8000、正=4000、负=4000、多数类 **0.5000**、盲猜 0.5000；
shuf 臂 `shuf=1 / label_order_changed=true / 计数仍 4000:4000`，shuf-seed = 42007 / 43007（**真执行**）；
`SELFTEST_TRUNC` 超长条数 **0**（train text 最长 96、编码后 96、span.end 最大 56；natural 31 / adversarial 62 / ctrl_adversarial 78）。

## 2. 主表（acc；每格并列 多数类 / SE / 余量 / 余量-SE / max_naive / 卡−max_naive）

余量 = acc − 多数类；余量/SE = (acc − M)/SE；**shuf 列为该格随机标签对照的 natural 读数**（详见 §3 V-bonus）。

| 臂 | seed | 集 | n | acc | 多数类 | SE | 余量 | 余量/SE | max_naive | **卡−max_naive** |
|---|---|---|---|---|---|---|---|---|---|---|
| **P** | 42 | **natural** | 200 | **0.7050** | 0.5000 | 0.0322 | +0.2050 | +6.36 | 0.7650 `lex_member` | **−0.0600 未超过免费规则** |
| **P** | 43 | **natural** | 200 | **0.7150** | 0.5000 | 0.0319 | +0.2150 | +6.74 | 0.7650 `lex_member` | **−0.0500 未超过免费规则** |
| P | 42 | adversarial | 1200 | 0.7450 | 0.5000 | 0.0126 | +0.2450 | +19.47 | 0.5408 `neg_char` | +0.2042 |
| P | 43 | adversarial | 1200 | 0.7275 | 0.5000 | 0.0129 | +0.2275 | +17.70 | 0.5408 `neg_char` | +0.1867 |
| P | 42 | ctrl_adversarial | 1200 | 0.4733 | 0.5000 | 0.0144 | −0.0267 | −1.85 | 0.5067 `neg_char` | **−0.0334 未超过免费规则** |
| P | 43 | ctrl_adversarial | 1200 | 0.4908 | 0.5000 | 0.0144 | −0.0092 | −0.64 | 0.5067 `neg_char` | **−0.0159 未超过免费规则** |
| **C** | 42 | **natural** | 200 | 0.5050 | 0.5000 | 0.0354 | +0.0050 | +0.14 | 0.7650 `lex_member` | −0.2600 |
| **C** | 43 | **natural** | 200 | 0.5250 | 0.5000 | 0.0353 | +0.0250 | +0.71 | 0.7650 `lex_member` | −0.2400 |
| C | 42 | adversarial | 1200 | 0.4575 | 0.5000 | 0.0144 | −0.0425 | −2.96 | 0.5408 `neg_char` | −0.0833 |
| C | 43 | adversarial | 1200 | 0.4750 | 0.5000 | 0.0144 | −0.0250 | −1.73 | 0.5408 `neg_char` | −0.0658 |
| C | 42 | ctrl_adversarial | 1200 | 0.4850 | 0.5000 | 0.0144 | −0.0150 | −1.04 | 0.5067 `neg_char` | −0.0217 |
| C | 43 | ctrl_adversarial | 1200 | 0.5017 | 0.5000 | 0.0144 | +0.0017 | +0.12 | 0.5067 `neg_char` | −0.0050 |

附报 exact（类+区间逐位）与「自己 test」：

| 卡 | natural exact | 自己 test acc / exact | 跨臂 test acc |
|---|---|---|---|
| P_s42 / P_s43 | 0.6200 / 0.5850 | `test` **0.8469**/0.8094 · **0.8381**/0.8006 | →ctrl_test 0.4963 / 0.4913 |
| C_s42 / C_s43 | 0.1900 / 0.1850 | `ctrl_test` 0.6581/0.3025 · 0.6412/0.2950 | →test 0.5138 / 0.5038 |

> **max_naive 读数与 `build_data.naive_battery` 独立复算逐位一致**（0.765 / 0.5408 / 0.5067 / 0.7969 / 1.0），
> 不一致会直接退出；`卡−max_naive` 为负的格子一律标 **「未超过免费规则」**。

## 3. 判据逐条实测

| # | 实测 | 结果 |
|---|---|---|
| **D1**（引用） | 提案臂标签 100% 由 P2–P5 谓词判出；非谓词判定（natural 人工标签 / 配额下采样 / 注入臂口径）单列 | ✅ |
| **D2**（引用） | 接受率 0.2803 / 0.2802 / 0.3677 ∈ [0.15, 0.85] | ✅ |
| **D3**（引用） | train 最高朴素规则 `lex_member` **0.797 < 0.90**；对抗集全部 ≈0.5 | ✅ |
| **D4(a)** | P 在 natural：seed42 **0.7050**、seed43 **0.7150** ≥ 门槛 **0.5707**（余量 +0.1343/+0.1443，= +4.16 / +4.52 SE） | ✅ **2 seed 同号** |
| **D4(b)** | 同 seed 配对：42 → 0.7050−0.5050 = **+0.2000**，2×SE_diff = 0.0957（差/SE = +4.18）；43 → 0.7150−0.5250 = **+0.1900**，2×SE_diff = 0.0952（+3.99） | ✅ **2 seed 各自 > 2×SE_diff** |
| **D5** | **本轮不走联合训练**（冻结核 + 只训头）⇒ **不适用**，不伪造数字（若走温启动，门槛 = Δ ≥ −噪声带 pronoun 2.83 / sentiment 0.41 / relation 1.39 / person 0.33 pt） | **N/A** |
| **V-bonus** | shuf（真执行，计数仍 4000:4000）：natural = P 0.5050/0.5150、C 0.4750/0.4800，**全 ≤ 0.5707**；自己 test = P 0.4819/0.5056、C 0.4956/0.4963，**全 ≤ 0.53** | ✅ **2 臂 2 seed 全过** |
| **D6（原判据）** | distinct 候选率 0.238 vs 0.9×注入臂 0.9×1.0 = 0.9 → **0.238 < 0.9** | ❌ **不过**（如实报） |
| **D6′-1** | 采样后 distinct 候选率 **0.1583 → 0.2380**（≥ 原始提议池，升 50%） | ✅ |
| **D6′-2** | 类别熵 **2.0201 vs 2.0201**（不低于注入臂） | ✅ |
| **D6′-3** | distinct 输入率 **0.7406 vs 0.5000**（不低于注入臂） | ✅ |

**D6 订正的透明度**：原判据拿**闭集词表**（提案臂候选来自约 230 词闭集、均长 2.21 —— 闭集正是「标签可被谓词判定」的前提）
对**任意片段**（注入臂 distinct 率天然 1.0）⇒ 口径不可比、惩罚了让这条路成立的性质。
订正落盘于 `PREREG_PHASE2.md` §5，**时间 2026-10-06 21:19 CST、早于首次训练**，理由与依据（`dev-notes/16 §7.6.1`）已注明。
**两个数字都报：原判据 D6 不过 + D6′ 全过。**

### 判定（三选一，按 PREREG_PHASE2 §6.2，不硬选）

> **① 可用且优于注入式**
> = D1 ✅ ∧ D2 ✅ ∧ D3 ✅ ∧ **D4(a)** ✅ ∧ **D4(b)** ✅ ∧ **D6′** ✅ ∧ **V-bonus** ✅（**D6 原判据 ❌，如实保留**；D5 = N/A）

**机制（实测 + 推断分开）**
- **实测**：P 在自己 test 0.847/0.838、adversarial 0.745/0.728（max_naive 0.5408，**+0.20/+0.19**）；C 在自己 ctrl_test 仅 0.658/0.641，
  而该集的免费规则 `substring` = **1.0000** ⇒ 注入臂**连自己的免费规则都没学会**（−0.34），与 `anchored_select` 的老症状同族。
- **实测**：**跨臂泛化 ≈ 随机**（P→ctrl_test 0.496/0.491；C→test 0.514/0.504）⇒ 两臂学到的不是同一件事。
- **实测**：shuf 臂（标签打乱）在全部 5 个集上都掉回多数类附近 ⇒ **D4 的增益不是「版式/长度/词表」这类表面特征白送的**。
- **推断**：注入臂的标签 = 「候选是否逐字出现在原文」，该性质在 natural 上**恒真**（指针式候选必锚定）⇒
  对照臂在 natural 上没有可用信号；提案臂的标签绑定「候选与真值源是否一致」⇒ 真值源可迁移的那部分信号被学到了。
- **推断（受限）**：P 在 natural 仍**低于**免费规则 `lex_member` 0.765 ⇒ 该卡**尚未超过「查一张 230 词表」**；
  D4(a)/D4(b) 只说明「**相对多数类与相对注入臂**可用」，**不说明它比免费规则强**。

## 4. 两条限制（**显式声明，必须随结论一起读**）

1. **长度混杂**：natural 均长 **14.15** vs 提案臂 **25.62（−45%）**（引用 `stats.json`）⇒ 本报告所有 D4(a)/D4(b) 结论
   只能说「**在该长度分布下**」，**不能说「泛化到自然用法」**。
2. **样本量限制**：natural 仅 **200 行** ⇒ 门槛约 **57.1%**、SE = 3.54pt，把握度弱；
   D4(a) 余量 +4.16/+4.52 SE 尚可，但 **D4(b) 的差是 2 个模型在同一 200 行上的配对差**，
   **边缘通过不得写成强证据**；本报告任何 single-set 数字都带 ±3~4pt 的不确定性。

## 5. 命令、unit、日志与产物

**原始命令（唯一一次总启动）**
```bash
systemd-run --user --unit=dtseek-genphase2 --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  --property=StandardOutput=append:/home/vesita/coding/my/DTSeek/logs/genphase2_master.log \
  --property=StandardError=append:/home/vesita/coding/my/DTSeek/logs/genphase2_master.log \
  /usr/bin/bash experiments/gen_data_loop/run_all.sh
```
单跑（`run_all.sh` 内，8 train + 8 eval）：
```bash
.venv/bin/python -u experiments/gen_data_loop/run_card.py --arm {P,C} --seed {42,43} \
  [--shuf --shuf-seed $((seed*1000+7))] --out experiments/gen_data_loop/cards/cand_<...>.pt \
  --epochs 12 --steps-per-epoch 150 --batch-size 64 --lr-head 1e-3 --samples 8000
.venv/bin/python -u experiments/gen_data_loop/eval_card.py --ckpt <卡> --tag <tag> \
  --splits natural,adversarial,ctrl_adversarial,test,ctrl_test --out experiments/gen_data_loop/results/eval_<tag>.json
uv run python experiments/gen_data_loop/summarize.py
```
- **unit**：`dtseek-genphase2`（另有一次 `dtseek-genphase2-smoke` 空测，产物 `/tmp/cand_smoke.pt`）；开训前 `ps` + `systemctl --user list-units 'dtseek*'` 均空，**未与他人争卡、未杀任何进程**。
- **日志**：`logs/genphase2_master.log`（含 `ALL_DONE rc=0 21:34:14`）、`logs/genphase2_{train,eval}_<tag>.log`。
- **产物**：`experiments/gen_data_loop/cards/cand_{P,C}[_shuf]_s{42,43}.pt`（8 个）；
  `results/eval_<tag>.json`（8 个）、`results/summary.md`、`results/summary.json`。
- **新增脚本**：`cand_card.py`（本进程注册卡）、`run_card.py`（自检 + 原样委托 `training/train_task_card.py`，未改 `training/`）、
  `eval_card.py`、`summarize.py`、`run_all.sh`。

## 6. 遗留与不确定

| 项 | 性质 | 说明 |
|---|---|---|
| 卡在 natural 上**未超过免费规则**（−0.06 / −0.05） | **实测** | 不影响 ① 的判据，但**是这张卡当前的真实上限短板**；要超过需另开 PREREG 调配方 |
| P 在 `ctrl_adversarial` 0.473/0.491（< 多数类） | **实测** | P 未见过注入式对抗分布，属预期外溢出，仅附报、不过线 |
| D6 原判据不过 | **实测** | 按要求保留两个数字，不因 D6′ 过而抹去 |
| natural 200 行、均长差 −45% | **实测** | 见 §4，两条限制不可省略 |
| shuf 仅打乱**标签**（span 按新标签重算），未打乱版式 | **口径判断** | 这是 PREREG 写死的做法；若担心版式残留信息，需另开对照 |
| 2 seed 而非更多 seed | **推断** | 按 PREREG 写死 42/43；n=200 下多 seed 只会缩训练侧方差，不缩评测侧 3.54pt |
| 单次总启动、单机单卡 | **实测** | 全部 16 个子跑 rc=0，无 OOM、无重跑 |

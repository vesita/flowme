# P19 修复在产 D1 缺陷：negation 卡与线上核匹配 —— REPORT

判定：**修复但收益被免费规则解释**（F1 达标 ∧ F2 ≤ 0）。只回答"能否把该卡修到与线上核匹配"，不回答整条对话链质量（见 `experiments/e2e_d1/REPORT.md` 归因边界）。

## 1. 步骤2：现成卡扫描（F0 自校验 + 在线核口径 `dialogue.respond`，n=600/seed，seed=42/43）

| 卡（路径） | 来源核 doc_sha | 线上核? | M_detect s42/s43 | M_cls s42/s43 | M_span s42/s43 |
|---|---|---|---|---|---|
| experiments/compose_ops/artifacts/cards/negation.pt（原默认，split_from arm_neg5_seed42） | 801657fb | **否** | .4883/.5167 | .2801/.2906 | .0393/.0576 |
| checkpoints/negation_accept_card.pt（train_args.base=base_encoder.pt，ep8） | dc27db53 | 是 | .8883/.8983 | .8894/.8953 | .6216/.6466 |
| checkpoints/negation_accept_card_e12.pt（同上，ep12） | dc27db53 | 是 | .9133/.9283 | .9140/.9188 | .7101/.7094 |
| **experiments/fix_negation/cards/negation_r2_s42.pt（新训 R2，选中）** | dc27db53 | 是 | .9850/.9933 | .9877/.9948 | .9656/.9764 |
| （复现披露）fix_negation/cards/negation_r2_s43.pt | dc27db53 | 是 | .9883/.9933 | .9853/.9948 | .9631/.9791 |
| （复现披露）fix_negation/cards/negation_r1_s42 / _s43（R1，6000 步） | dc27db53 | 是 | .9750/.9833；.9867/.9817 | .9779/.9764；.9828/.9843 | .9459/.9450；.9435/.9581 |

F0（实测，`logs/s1.log`、`results/s1_shas.json`）：线上核 doc_sha 自校验复现 `dc27db5337d16f185dd08cc50505390e53d03606b9d307457828f4ecbfdf6b1a`；全树 negation 卡 **3 张、其中线上核来源 2 张**；原默认**非**线上核来源 ⇒ 结论：**没有"≈1.0000 现成可换"的卡（最高 .9188）⇒ 走步骤 3 重训**。
重训（`PREREG_TRAIN.md`→`PREREG_TRAIN_CLARIFY{,2}.md`，core 冻结只训头，`training/train_task_card.py --base checkpoints/base_encoder.pt --card negation --samples 8000`，数据=任务既有生成器 `build_negation_dataset(8000, DEFAULT_SEED=20240927)` 单一来源）：R1=40ep×150=6000 步 → M_cls .9779/.9764 **< .98** ⇒ 按停止规则触发 R2=80ep×150=**12000 步**（数据/头/lr/batch 逐字同 R1）→ .9877/.9948 **≥.98**；训练 seed 43 同配方复现 .9853/.9948（选卡规则=训练 seed42，写在见 r1_s43 结果之前，见 CLARIFY §1）。

## 2. 步骤5（前/后）+ F2 免费规则地板

| respond 口径 | s42 改前 | s42 改后 | Δ±SE | s43 改前 | s43 改后 | Δ±SE |
|---|---|---|---|---|---|---|
| M_detect | .4883±.0204 | .9850±.0050 | **+.4967±.0210*** | .5167±.0204 | .9933±.0033 | **+.4767±.0207*** |
| M_cls | .2801±.0223 | .9877±.0055 | **+.7076±.0228*** | .2906±.0232 | .9948±.0037 | **+.7042±.0234*** |
| M_span | .0393±.0096 | .9656±.0090 | **+.9263±.0130*** | .0576±.0119 | .9764±.0078 | **+.9188±.0140*** |

F2（8 条免费规则，fit=train_S / eval=eval_S，`results/s4_floor.json`、`logs/s4_after.log`）：
口径A cls：majority / len_bucket / first_char / last_char / punct_pattern / lex_rule / lex_first_key / n_cue **全 = 1.0000**（winner=majority；train 标签分布 {1:3493} 退化）⇒ max_naive **1.0000**；
口径B detect：majority .6783/.6367 | len .6783/.6367 | first .7217/.6817 | last .6817/.6517 | punct .6783/.6367 | lex_rule .6783/.6367 | **lex_first_key 1.0000/1.0000** | **n_cue 1.0000/1.0000** ⇒ max_naive **1.0000**（winner=lex_first_key）；披露项 lex_lit 1.0000 / lex_dataset 1.0000。
**卡 − 地板**：改前 cls **−.7199/−.7094**、detect **−.5117/−.4833** ⇒ 未超过免费规则；改后 cls **−.0123/−.0052**、detect **−.0150/−.0067** ⇒ **两 seed 仍 ≤ 0 ⇒ 未超过免费规则**（即：换不换都跑不过字面规则）。

## 3. 改动清单 + F3 等价性 + F4 pytest

- **src 仅 1 处**：`src/dtseek/tasks/dialogue.py:105` `DEFAULT_ATTACH = ("experiments/fix_negation/cards/negation_r2_s42.pt",)`；`git diff --stat` = **1 file, 1 insertion, 1 deletion**（其余未动：render.py / card_contract.py / 核 / 其它卡 / 其它实验目录 / dev-notes / methodology 均 0 改动）。
- **tests 0 改动**（`git status --porcelain tests/` 为空）：tests 只按符号引用 `DEFAULT_ATTACH`（`tests/test_dialogue.py:115` fixture），合成断言走 `_FakeEngine`（:302），**无断言写死旧卡文件名或旧卡输出** ⇒ 改前/改后都无需改测试（实测见 F4）。
- **F3**（11440 条记录，`results/s3_compare.json` + `s3_proj_prereg.json`）：
  按 `PREREG §3` **字面投影 relaxed=6974 / strict=7886 ⇒ ≠0（字面未达成，如实记）**；
  归因（实测）：6867 条是 negation 证据本身变了；另 **107 条 negation 证据相同、差异只有 plan 里的 conf 档 中→高**（掩掉 conf 后 plan 逐字相同），同样归因于该卡的 confidence；
  **非 negation 侧字段（type / cards_run / 其它 4 张卡的 evidence）逐条比对 11440/11440 不一致 = 0** ⇒ 实质判据"除该卡外默认行为逐字不变"**达成**，字面投影口径**未达成**（该投影未事先排除 conf 派生字段）。
- **F4**：改前 `258 passed in 25.79s`（`logs/pytest_before.log`）；改后 `uv run pytest -q` ⇒ **`258 passed in 23.29s`**（`logs/pytest_after.log`），全绿且**零测试改动**。

## 4. F0–F5 逐条 + 判定 + 机制

| 门 | 实测结果 | 判定 |
|---|---|---|
| F0 | 复现 `dc27db53…`；3 张 negation 卡 / 2 张线上核；原默认非线上核 | **过** |
| F1 | M_cls .2801/.2906 → **.9877/.9948（两 seed ≥ .98）**，Δ +.7076±.0228 / +.7042±.0234 同号且 >2SE；训练 seed43 同配方 .9853/.9948 | **过** |
| F2 | 卡 − 地板：cls −.0123/−.0052、detect −.0150/−.0067（地板两口径均 1.0000） | **≤0 ⇒ 未超过免费规则** |
| F3 | 实质 0 不一致（11440/11440 非派生字段）；字面投影 6974 ≠ 0 | **实质达成 / 字面未达成（并列如实报）** |
| F4 | 258 passed（前 258 / 后 258），tests 零改动 | **过** |
| F5 | 四老卡 Δ 全 = **0.000000** ≥ −带（pronoun −.0283 / sentiment −.0041 / relation −.0139 / person −.0033）；before 与 P15 `a0_bases` 逐位相同 8/8 | **过** |

**判定 = 修复但收益被免费规则解释**（预注册分支 2：F1 达标 ∧ F2 两 seed ≤0）。**机制**（推断，基于上述实测）：原卡的头是在**另一套核表征 801657fb** 上训出的，挂到线上核 dc27db53 后解码头读到的不是它认得的表征 ⇒ 首切片漏检 293/407；换成**同核训练**的头后漏检 5/407。免费规则地板恒为 1.0000，是因为该任务标签由字面 cue 完全决定（正例必含 11 个标记之一、背景零标记，生成器口径所致）⇒ 卡的收益上限被地板钉死，**换卡只把"远低于地板"修到"贴着地板"，并未超过地板**。

## 5. 命令 / unit / 日志 / 产物 / 回退 / 遗留

- 训练：`systemd-run --user --unit=dtseek-fixneg-r1 --collect --property=WorkingDirectory=…/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 /usr/bin/bash experiments/fix_negation/run_train.sh r1`（unit `dtseek-fixneg-r1`）；R2 同式 `--unit=dtseek-fixneg-r2 … run_train.sh r2`（`dtseek-fixneg-r2`）。GPU：ROCm，峰值 165.7 MB/卡，12000 步 295 s（0.0246 s/步）。
- 测量：`uv run python experiments/fix_negation/s1_shas.py | s2_e2e.py --cards … | s3_equiv.py --phase before|after| --compare | s4_floor.py | s5_gate.py --phase before|after| --compare | s6_holdout.py | d1_errors.py`；日志 `experiments/fix_negation/logs/*`（含 `pytest_before/after.log`、`after_run.log`、`train_r1/r2_*.log`、`overlap_check.log`），结果 `experiments/fix_negation/results/*.json`（14 个）。
- 产物：选中卡 `experiments/fix_negation/cards/negation_r2_s42.pt`（2481 KB，file sha256 `604571b1df578f1f…`，629,764 头参数，train_args `base=checkpoints/base_encoder.pt, samples=8000, epochs=80, lr_head=1e-3`）。
- **回退**：把 `dialogue.py:105` 改回 `DEFAULT_ATTACH: tuple[str, ...] = ("experiments/compose_ops/artifacts/cards/negation.pt",)`（该卡 file sha `ba1a80f9af81b1a2…`、来源核 `801657fb…`）；本单元未 commit/stash/checkout/restore/clean。
- **遗留（实测 / 推断分列）**：① 【实测】`eval_S` 文本 **100% 落在重训数据内**（600/600，`logs/overlap_check.log`）⇒ F1 是**同口径 in-sample 指标**，不是泛化指标（旧卡同样如此，口径未改）；② 【实测】换 seed=90210 现生成 2000 条，**0 条未见**（1659 与 train8000 重叠、349 与 eval_S 重叠）⇒ 该生成器是确定性语料扫描，**无法用换 seed 得到 held-out** ⇒ 本单元**没有泛化证据**；③ 【实测】余错集中在 `别/莫/勿` 同形虚化（漏检 5+2、误报 4+2，`logs/d1_r1s42.log` 同型）；④ 【推断】若要求"超过免费规则"，需要的不是更强的头，而是**标签口径改造**（背景里混入字面 cue），已超出本单元"只换卡"的边界；⑤ 【推断】F1 若按"训练 seed 也要都过"读，R1 只有 s43 过、R2 两 seed 都过（R1/R2 数字见 §1，未挑卡）。

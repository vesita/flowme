# core_select_only 结果报告 —— 那个增益是「select 梯度进核」带来的，还是「老任务锚定」参与后才可能的？

> 判据与配方见 `PREREG.md`（mtime **2026-10-06 23:29:29**，早于空测试首跑 23:34:32 与正式首跑 23:42:11，跑后**未改**）。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 构造时人为规定。
> 数据只读复用 `experiments/select_rerank/data/clean/`：跑前跑后 md5 一致（`6b1e6fb1…` / `d3d597af…` / `2a5ca5f4…`）。

## 1. 两臂配置（实测）+ `selonly` 未挂载老卡头的证据

| | **核对臂 `joint`** | **主臂 `selonly`** |
|---|---|---|
| 每步数据 | select batch 64 + **4 个老任务 batch**（`core_keep` J3 同批梯度） | **只有 select batch 64** |
| 核 `NanoDocEncoder` | 1,688,460 **全可训**，全程 `eval()`（dropout=0） | 同左 |
| 打分头（masked mean-pool，聚合参数 0） | 164,353 可训 | 同左 |
| 四张老卡头 | **挂载并冻结**：pronoun 630,278 / sentiment 630,278 / relation 630,021 / person 631,306，逐参数 `requires_grad=[false]`（实测打印） | **训练期间不挂载、不参与任何计算**；训练**结束后**仅 `no_grad` 评测加载 |
| **可训合计** | **1,852,813** | **1,852,813** |
| 配方（逐项同口径） | 1800 步 / batch 64 / lr_core **3e-4** / lr_head **1e-3** / cosine T=1800 / clip 1.0 / wd 1e-4 / seed **42,43** / 核 `eval()` / **live 编码** | 同左 —— **唯一变量 = 每步是否同时跑老任务批次** |
| **稳态步时** | 0.2450 / 0.2532 s | 0.0367 / 0.0374 s（**×6.7 快**） |
| **峰值显存** | 1978.8 / 1979.7 MB | 441.8 / 441.8 MB |
| 单跑墙钟 | 459.6 / 465.1 s | 82.0 / 82.7 s（rand 83.0 / 83.3 s） |
| step1 核梯度范数 / 老任务前向次数 | 84.6237 / 75.0101；**7200 次**（= 1800×4） | **0.1583992 / 0.1915；0 次** |
| 核漂移 `‖core−base‖/‖base‖` | 0.125432 / 0.126406 | 0.050032 / 0.048874（rand 0.135812 / 0.136801） |

**`selonly` 未挂载老卡头的实测证据**（每跑 `results/*.json::mount_evidence` + 日志 `MOUNT_EVIDENCE_PRE/POST` 行）：
`heads_keys_during_training=[]`、`old_loaders_during_training=[]`、`old_forward_count=0`、`recipe.old_tasks_in_batch=[]`、
`old_heads_mounted_during_training=false`、`model.named_modules()` 顶层**只有 `["encoder","head"]`**（45 个模块，无任何卡解码器）——
`selftest.py` 6 项断言全过（`SELFTEST_DONE`）。
**交叉验证**：`selonly` step1 核梯度 **0.1583992** = `core_grad_probe` 实测 `|g_sel| = 1.583992e-01`（同值）⇒ 它的核梯度就是**纯 select 梯度**；
`joint` step1 = 84.6237 vs probe `‖g_old‖ = 84.62594`（相对差 2.6e-5）。

**空测试抓到的真 bug（实测，已修）**：初版 `live = arm=="joint"` 让 `selonly` 走 token 缓存 ⇒ 核**收不到梯度**
（drift=0.000000、步时 0.0015s、显存 81MB、heldout 44.64）。改为两臂一律 live 编码，并加硬断言
（step1 `‖g_core‖ > 0`、`drift > 0`）后复测通过。
**同源对账（实测）**：本目录 `--arm joint --steps 100` vs 只读 import 的 `select_semantic_joint/train_sem.py` 同参跑
⇒ `loss_first` / `loss_select_first50` 差 **0.000e+00**、全部权重 **max|Δ| = 0.000e+00**（逐位相同）。

## 2. 主表（准确率 %；n=1250 ⇒ **SE = 1.414pt**、**2×SE = 2.828pt**；多数类 0.5016）

| 臂 | seed | `heldout` | 对50%/SE | **Δ vs 冻结核** | **Δ/SE** | `shifted` | test | train | adv |
|---|---|---|---|---|---|---|---|---|---|
| 冻结核基线 | 42 | 49.52 | −0.34 | — | — | 92.56 | 94.48 | 100.00 | 71.04 |
| 冻结核基线 | 43 | 51.04 | +0.73 | — | — | 92.48 | 94.48 | 100.00 | 71.76 |
| `joint` | 42 | **59.36** | +6.62 | +9.84 | +6.96 | 99.60 | 99.88 | 100.00 | 79.48 |
| `joint` | 43 | **63.20** | +9.33 | +12.16 | +8.60 | 99.60 | 99.84 | 100.00 | 81.40 |
| **`selonly`** | 42 | **73.60** | **+16.69** | **+24.08** | **+17.03** | **100.00** | 100.00 | 100.00 | 86.80 |
| **`selonly`** | 43 | **68.40** | **+13.01** | **+17.36** | **+12.28** | **100.00** | 99.96 | 100.00 | 84.20 |
| `selonly` rand | 42 | 45.68 | −3.06 | −3.84 | — | 49.44 | 49.52 | 49.68 | 47.56 |
| `selonly` rand | 43 | 51.60 | +1.13 | +0.56 | — | 46.88 | 50.12 | 49.80 | 49.24 |

- **`selonly` vs `joint`（T2）**：Δ = **+14.24 / +5.20 pt** = **+10.07 / +3.68 SE** ⇒ **超出 2×SE**，方向是 `selonly` **更高**（两 seed 同号）。
- **随机标签对照**：4 项 heldout/shifted 全 ≤ **52.83**；打乱真执行 4026 / 4016 条、train(对真标签) 49.68 / 49.80 ⇒ 空对照有效。

## 3. T5 老卡代价（`selonly` 训练后 = 共享漂移核 + 各自冻结头；exact，逐卡）

| seed | 卡 | step0 | post | **Δ** | 各自带 | 超带？ |
|---|---|---|---|---|---|---|
| 42 | pronoun | 0.9533 | 0.4917 | **−0.4617** | 0.0283 | **是（×16）** |
| 42 | sentiment | 0.7937 | 0.3081 | **−0.4856** | 0.0041 | **是（×118）** |
| 42 | relation | 0.9639 | 0.2847 | **−0.6792** | 0.0139 | **是（×49）** |
| 42 | person | 0.3100 | 0.3050 | −0.0050 | 0.0033 | **是（×1.5）** |
| 43 | pronoun | 0.9800 | 0.4650 | **−0.5150** | 0.0283 | **是（×18）** |
| 43 | sentiment | 0.8147 | 0.3125 | **−0.5022** | 0.0041 | **是（×122）** |
| 43 | relation | 0.9653 | 0.2708 | **−0.6944** | 0.0139 | **是（×50）** |
| 43 | person | 0.3233 | 0.3167 | −0.0067 | 0.0033 | **是（×2.0）** |

**8/8 全部退化且远超各自带 ⇒ 上游预注册的预测代价兑现**（老卡头全程冻结也照样崩 ⇒ 崩的是核）。
对照：本档 `joint` 8/8 **无一退化**（pronoun +0.0200/−0.0017、sentiment +0.1559/+0.1413、relation +0.0153/+0.0250、person +0.0083/−0.0017），与 `select_semantic_joint` R5 逐位一致。

## 4. T0–T6 逐条实测 + 判定 + 机制

| # | 实测 | 判定 |
|---|---|---|
| **T0（门禁）** | `joint` heldout **59.36 / 63.20**、shifted **99.60 / 99.60**、test 99.88 / 99.84 ⇒ 逐项 **Δ = 0.00**；另 drift 0.125432 / 0.126406（原 0.1254 / 0.1264）、`loss_first` 0.693002 / 0.692196、`loss_select_first50` 0.353859 / 0.296474 与原主臂及两份 probe 记录吻合 | **过 ✅（零误差）** |
| **T1（主）** | Δ = **+24.08 / +17.36pt** = **+17.03 / +12.28 SE**（门槛 2.828pt，余量 +21.25 / +14.53pt），两 seed 同号 | **过 ✅** |
| **T2** | Δ(selonly−joint) = **+14.24 / +5.20pt** = +10.07 / +3.68 SE，**超出** 2×SE，方向 `selonly` 更高 | **不过 ❌（见判定说明）** |
| **T3** | shifted **100.00 / 100.00** ≥ 89.73 / 89.65（门槛 = 冻结基线 −2.828），vs 基线 **+7.44 / +7.52pt** | **过 ✅（未拿表内记忆换）** |
| **T4（门禁）** | rand heldout/shifted = 45.68/49.44、51.60/46.88 全 ≤ 52.83；n_diff 4026 / 4016、计数仍 1:1 | **过 ✅** |
| **T5（代价，必报）** | 16 项齐全；`selonly` **8/8 超带退化**（见 §3），`joint` 8/8 无退化 | **报告完成（预期代价兑现）** |
| **T6** | 漂移 joint 0.125432/0.126406、selonly 0.050032/0.048874（rand 0.135812/0.136801）；步时 0.2450/0.2532 vs 0.0367/0.0374 s；峰值 1978.8/1979.7 vs 441.8/441.8 MB | 数值如上 |

**判定（PREREG §3 跑前写死三选一）= ③ 证据不足。**
原因逐字：门禁 T0/T4 过、**T1 过**，但 **T2 不过**（|Δ| > 2×SE）⇒ 按字面不落入 ①（①要求与 `joint` 差距在噪声内），也不落入 ②（`selonly` 不是 ≈50%）。
**如实列出的预注册不符处**：PREREG §3 ③ 的括注假定「若 T2 不过则 `joint` 更高」，**实测方向相反**（`selonly` 高出 +14.24 / +5.20pt）——门槛未改、按字面判 ③，不事后挪门槛。
**实质读数（与形式判定分开）**：对「唯一要回答的问题」，实测**明确排除 ②**、**支持 ① 的核心命题且更强** —— 只给 select 梯度就到 73.60 / 68.40（+17.0 / +12.3 SE），随机标签清零到 45.68 / 51.60 ⇒ **老任务锚定不是增益的必要条件**；老任务的净作用反而是**压低新任务上限**（joint 低 5–14pt）并**保住老卡**（T5）。

**机制（实测 → 推断分开）**：
- 【实测】**增益由真标签的 select 梯度驱动**：T1 大幅过 + T4 清零 + step1 核梯度 0.1583992 = probe 的纯 `|g_sel|`（同值）。
- 【实测】**漂移本身不产生增益**：selonly 真标签漂移仅 0.050 / 0.049（joint 的 40%），rand 漂移反而更大（0.136 / 0.137）却回 50%。
- 【实测】**老任务锚定压制新任务**：同配方、同数据、同 seed，去掉 4 个老任务批次 → heldout +14.24 / +5.20pt；与 `core_update_probe` 的 `cos(both,old)=0.970/0.996/1.000`（更新方向由老任务主导）方向一致。
- 【实测】**保老卡必须靠老任务每步参与**：头全程冻结，`selonly` 仍 8/8 崩（relation −67.9 / −69.4pt）⇒ 冻结头本身不保护任何东西。
- 【推断·设计结论】「让核学 select」与「保住老卡」是**同一组参数上的两个目标**：`selonly` 换来更高新任务分但老卡全崩，`joint` 牺牲 5–14pt 换老卡无退化 ⇒ **两者必须绑定**（这正是上游预告的代价），要解耦需要卡级私有容量 / 分离机制，**本档未测**。
- 【推断·置信度】T0/T1/T3/T4 与「老任务非必需」= **强**（2 seed + 随机对照 + 零误差门禁）；「老任务压制新任务」= **中-强**（2 seed 同号但幅度差 9pt）；「增益来自 select 梯度进核」= **强**（随机标签清零 + 梯度同源）。

## 5. 原始命令 / unit / 日志 / 产物 / 纪律 / 遗留

```bash
uv run python experiments/core_select_only/selftest.py            # 23:34:32 首跑，6 项全过（跑 3 次，v1 抓到缓存 bug）
systemd-run --user --unit=dtseek-cso-joint --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  bash experiments/core_select_only/run_all.sh joint      # 同式换 --unit=dtseek-cso-selonly + run_all.sh selonly
uv run python experiments/core_select_only/analyze.py             # T0–T6 + 判定 → results/summary.json
```

- **unit**（两者现均 inactive，`list-units 'dtseek*'` 为空）：`dtseek-cso-joint.service`（invocation `357838cbace44506bbf9173605c16981`，23:42:11→23:57:40 ALL DONE，墙 15min29.0s / CPU 24min39.6s / 内存峰值 2.2G）、
  `dtseek-cso-selonly.service`（invocation `2bc0034a60e04467805ff3798444058b`，23:57:57→00:03:37 ALL DONE，墙 5min39.7s / CPU 8min5.5s / 2.2G）；
  均 `--collect` + `PYTHONUNBUFFERED=1` + `PYTHONDONTWRITEBYTECODE=1`，**无 `setsid nohup`**；开训前查过 `ps` 与 `systemctl --user is-active 'dtseek*'`（空闲），**未杀任何进程**。
- **日志**：`logs/core_select_only_{joint,selonly}_s{42,43}[_rand].log`（6 个）+ `core_select_only_selftest{,_v2,_v3}.log` + `core_select_only_analyze.log`；单元 stdout 在 `journalctl --user -u dtseek-cso-*`。
- **产物**：`experiments/core_select_only/results/*.json`（6 主跑 + 2 探跑 + `selftest.json` + **`summary.json`（T0–T6 + 判定）**）、`weights/*.pt`（9）。
- **只读自证（实测）**：数据 md5 跑前跑后一致；`find -newermt PREREG` 对 `select_rerank/select_pool/select_semantic_joint/core_grad_probe/core_update_probe/src/training/tests` **为空**、对他人 `__pycache__` 无新增（`select_semantic_joint` 仍无 `__pycache__`）；
  `git status` 中本目录外**无条目**；本代理**未执行任何 git 写操作**（commit/stash/checkout/restore/clean 均未跑）。
- **非本代理改动（实测，供对账）**：`dev-notes/16-*.md` mtime 00:00:28 属另一会话（其持有 `methodology/`、`dev-notes/16` 占用声明），本代理只读未写；commit `ba1b9b9` 把本目录**早期快照**收进了仓库（非本代理执行）。

**遗留与不确定（实测 / 推断分开）**：
1. 【实测】`joint` 的随机标签对照本档**没跑**（T4 只要求 `selonly`）；`select_semantic_joint` 已有 joint rand（48.24 / 49.76）可引。
2. 【实测】**仅 2 seed**，`selonly` seed 间差 5.20pt（≈3.7 SE）；步数 / lr / 老任务权重**均未扫**（跑前定死，不调参刷过）。
3. 【实测】T2 的 SE 用**独立二项 SE（1.414pt）**，两臂在同一批 1250 条上评测（配对差），配对 SE 未算 ⇒ Δ/SE 是保守下界，方向不变。
4. 【实测】`selonly` 100 步探跑 heldout 已 67.68、1800 步 73.60 ⇒ 未见饱和；**未扫步数**。
5. 【推断】`selonly` 与 `joint` 的差异同时含「无老任务梯度」与「总梯度尺度 / clip 位置不同」两个成分；`core_update_probe` 已证 Adam+clip 对整体尺度不敏感，但**中间臂（老任务只前向不回传）未跑**。
6. 【实测】显存差 1979 − 442 ≈ 1.5GB 归因于老任务 rollout 计算图（结构性解释，**未逐层字节对账**）。
7. 【推断】本档结论只对 clean(N) 臂、mean-pool、1800 步成立；换数据集 / 聚合方式 / 更长训练的泛化性**未测**。

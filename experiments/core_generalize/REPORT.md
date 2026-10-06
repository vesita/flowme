# 核见过更多卡，能不能让一个「从不参与联合训练」的新能力学得更好？

`experiments/core_generalize/` · 预注册 `PREREG.md`（mtime 08:42:31，首次训练 08:45:04）· 汇总 `summary.json` · 脚本 `analyze.py`

## 1. hold-out 可学性预检（先过这一关才有资格当探针）

跑前写死：`exact(PRE) − floor ≥ δ_pre = max(2pt, PRE 两 seed 极差)`，两 seed 都要满足。
`floor` = val 中无切片样本占比（"永远预测空切片集"的 exact；能复现已知失败案例 `reply_pick` 联合 0.1717 vs 地板 0.1683）。

| 候选 | 联合训练 exact s42/s43 | floor s42/s43 | δ_pre | 结论 |
|---|---|---|---|---|
| **idiom** | **0.8883 / 0.8350** | 0.3567 / 0.3333 | 5.33pt | ✅ 入选（+53.2 / +50.2pt） |
| **ownership** | **0.9817 / 0.9917** | 0.3083 / 0.2933 | 2.00pt | ✅ 入选（+67.3 / +69.8pt） |
| reply_pick | 不重跑，联合 0.1717 | 0.1683 | — | ❌ 贴地板，引用 `checkpoints/reply_pick_joint_metrics.json` |
| cloze_fill | 不重跑，cls 0.19 ≈ 盲猜 0.20、exact 0.138 | 0.15 | — | ❌ 贴地板，引用 `experiments/selection_cards/results_cloze_fill.json` |

**可用 hold-out = 2 个（idiom、ownership），满足"至少 2 个"底线**；都从不参与任何核训练。

## 2. 四个核臂（全部从 `checkpoints/base_encoder.pt` 温启动，配方与 core_keep J0 同）

16ep×84=**1344 步**、batch 64、lr_base 3e-4、lr_head 1e-3、AdamW(wd 1e-4)、cosine；温启动头 ← `capability_map/cards/{task}_frozen_s{seed}`（negation 无冻结卡 ⇒ 随机头，与 core_keep 同）。

| 臂 | 训练任务 | 步数 | 可训参数（核 + 头） | 核漂移 ‖c−b‖F/‖b‖F s42/s43 |
|---|---|---|---|---|
| **C1** | sentiment | 1344 | 2,318,738（1,688,460 + 630,278） | 0.1110 / 0.1118 |
| **C3** | pronoun+sentiment+relation | 1344 | 3,579,037（1,688,460 + 1,890,577） | 0.1130 / 0.1136 |
| **C5** | +person+negation | 1344 | 4,840,107（1,688,460 + 3,151,647） | 0.1087 / 0.1094 |
| **C1x5** | sentiment，**步数×5** | **6720** | 2,318,738（同 C1） | **0.2361**（仅 s42） |

**健康检查（集内卡 exact，Δ = 对 core_keep J0 同 seed 同任务）**：

| 臂 | s42 | s43 | Δ vs J0（s42 / s43） |
|---|---|---|---|
| C1 | sentiment 0.9734 | 0.9753 | **+5.69 / +4.31** |
| C3 | pron 0.9817 · sent 0.9572 · rel 0.9986 | 0.9817 · 0.9631 · 1.0000 | pron +1.50/+0.17 · sent +4.06/+3.09 · rel +0.56/+0.69 |
| C5 | pron 0.9650 · sent 0.9137 · rel 0.9931 · person 0.3217 · neg 0.9683 | 0.9800 · 0.9241 · 0.9944 · 0.3233 · 0.9417 | pron −0.17/0.00 · sent −0.28/−0.81 · rel 0.00/+0.14 · person +0.17/−0.17 · **neg −1.17/−2.50** |
| C1x5 | sentiment 0.9928 | — | +7.63 |

作废线 = δ_global 7.67pt 且两 seed 同向 ⇒ **无臂作废**（最差 C5-negation −1.17/−2.50pt）。
`SELFTEST_init`：温启动头 step-0 exact 与 `capability_map/eval.json` 的 frozen **逐位一致 Δ=0.000e+00**（C1/C3/C5/C1x5 共 7 次全过；negation 是随机头不参与对账）；`ALIGN_CHECK eval==val` 5 任务 ×2 seed 全 True。

## 3. 主表（冻结核 + 同一头训练配方，唯一变量 = 核）

**探针预算自检**：16ep×84=**1344 步**、batch 64、lr 1e-3、`n_train=5400 / n_val=600`、样本抽取 **86,016**；核 `core_trainable=0`（1,688,460 params，`requires_grad` 全 False，eval+no_grad）；头参数 idiom 629,764 / ownership 630,278；随机头 step-0 = 0.26~0.35（idiom）/ 0.11~0.28（ownership），均低于地板 ⇒ 起跑线干净。

### idiom（floor 0.3567 / 0.3333，**δ = 7.67pt**）

| seed | B0（未训 base 核） | C1 | C3 | C5 | C1x5 |
|---|---|---|---|---|---|
| 42 | 0.8033 | 0.6817 | 0.8400 | **0.8100** | 0.5767 |
| 43 | 0.7267 | 0.6600 | 0.7783 | **0.7583** | — |

- **P1 ✅**：C5−C1 = **+12.83 / +9.83pt**，两 seed 都 ≥ δ。
- **P2 ✅（仅 δ 带内）**：C1 < C5 成立，但 **C3 反超 C5 −3.02 / −2.00pt**（7.67pt 带内）⇒ 不是严格 C1≤C3≤C5。
- **P3 ✅**：C5−C1x5 = **+23.33pt**（s42）≥ δ ⇒ 不是"数据/优化更多"。

### ownership（floor 0.3083 / 0.2933，**δ = 2.00pt**）

| seed | B0 | C1 | C3 | C5 | C1x5 |
|---|---|---|---|---|---|
| 42 | 0.9233 | 0.9167 | 0.9233 | 0.9167 | 0.9267 |
| 43 | 0.9133 | 0.9200 | 0.9217 | 0.9233 | — |

- **P1 ❌**：C5−C1 = **0.00 / +0.33pt** ≪ 2pt；**P3 ❌**：C5−C1x5 = **−1.00pt**（C1x5 反而最高）。
- **P2 ✅**（全部挤在 1pt 内，平凡成立）；所有格子 0.91~0.93 ≫ 地板 0.30 ⇒ 无"贴地板"格。

## 4. δ 是怎么算出来的（原始极差）

跑前写死：**δ(h) = max(2pt, B0（base 核）在该 hold-out 上两个头 seed 的 exact 极差)**。B0 的核恒定 ⇒ 极差量的是**纯头训练噪声**；同一 δ 用于该 hold-out 的所有比较，不逐臂调。

| hold-out | B0 s42 | B0 s43 | 原始极差 | δ |
|---|---|---|---|---|
| idiom | 0.8033 | 0.7267 | **0.0767** | **7.67pt** |
| ownership | 0.9233 | 0.9133 | **0.0100** | 2.00pt（被 2pt 下限兜住） |

`δ_global = 7.67pt`（健康检查作废线用同一个数）。敏感性（实测）：若 idiom 也按 2pt 算，P1（+12.83/+9.83）与 P3（+23.33）照样过 ⇒ 结论不依赖 δ 取值。

## 5. 判定：**证据不足**（预注册三选一，不硬选）

| hold-out | P1 | P2 | P3 | 方向 |
|---|---|---|---|---|
| idiom | ✅ | ✅（带内） | ✅ | 支持 a |
| ownership | ❌ | ✅（平凡） | ❌ | 支持 b |

- **a 要求两个 hold-out 全过 P1/P3 ⇒ ownership 挡掉 a**；**b 要求所有 hold-out 的 P1 都不过 ⇒ idiom 挡掉 b**。
- 「两 seed 同号」有 0.0000 读法歧义（ownership 的 C5−C1 s42 = **0.0000**，严格不同号）：**两种读法都算，判定都是"证据不足"**（`summary.json` 的 `verdict` 与 `verdict_alt_noconflict`）。

**机制（描述性锚点 = 未训练 base 核 B0，非预注册判据）**：

| idiom Δ vs B0（s42/s43） | C1 | C3 | C5 | C1x5 |
|---|---|---|---|---|
| | **−12.17 / −6.67** | +3.67 / +5.17 | **+0.67 / +3.17** | **−22.67** |

1. **C5 并没有比未训练的 base 更通用**（+0.67/+3.17pt，全在 δ 内）⇒ P1 在 idiom 上之所以过，是因为 **C1/C1x5 把自己的通用性训掉了**，不是 C5 长出新通用性。
2. C1 与 C5 **漂移量几乎一样**（0.1110 vs 0.1087）却差 12.8pt ⇒ 起作用的是**漂移方向**（单任务 vs 多任务），不是漂移大小。
3. C1x5 漂移翻倍（0.2361）⇒ idiom 掉到 0.5767（比 C1 再差 10.5pt）⇒ **更多数据/步数让单任务核更偏**。
4. ownership 上 base/C1/C3/C5/C1x5 全在 0.91~0.93 ⇒ **表示本来够用，多样性无增益**（P1/P3 双不过）。

⇒ **一句话**：多样性没让核"更通用"，但**单任务训练会让核"变窄"**；老卡提升（dev-notes/15 §15.2 机制 1）在"新任务能否被冻结核支撑"上**没有可测量的红利**。

## 6. 原始命令 / unit / 日志 / 产物

完整 `systemd-run` 原文见 `experiments/core_generalize/COMMANDS.md`。

| 阶段 | unit | 日志 | 产物 |
|---|---|---|---|
| 数据准备 | 前台 `uv run .../prepare.py` | — | `cache/`、`data_info.json` |
| 预检 | `dtseek-cg-pre` | `logs/PRE_{idiom,ownership}_s{42,43}.log` | `cores/PRE_*_metrics.json`、`precheck.json`、`holdouts.txt` |
| 核臂 | `dtseek-cg-cores`、`dtseek-cg-cores2` | `logs/{C1,C3,C5,C1x5}_*.log` | `cores/{C1,C3,C5,C1x5}_*_{,.}metrics.json` |
| 探针 | `dtseek-cg-probes` | `logs/{B0,C1_s42,…}_{idiom,ownership}_s{42,43}.log` | `probes/*.json`（18 个） |
| 4× 预算（探索） | `dtseek-cg-budget4x` | `logs/*_e64.log` | `probes_budget4x/*.json` |
| 汇总 | 前台 `analyze.py` | — | `summary.json` |

全部 `PYTHONUNBUFFERED=1` + `HSA_OVERRIDE_GFX_VERSION=10.3.0`；**未杀任何他人进程**，启停前后都查 `systemctl --user is-active 'dtseek-*'`；`logs/` 被另一代理独占 ⇒ 本实验日志全写在 `experiments/core_generalize/logs/`（已在 `collab_board` 通告）。

## 7. 遗留与不确定（实测 / 推断分开）

**实测**
1. **卡数与数据量仍未完全解耦**：C1x5 与 C5 总样本抽取数相同（6720×64 = 5×1344×64 = 430,176），但 C1x5 控制的是「核优化步数 ×5、任务 1 个」，C5 是「每步 5 任务梯度混合、核步数 1344」——**不是同一件事的两种剂量**；且 C1x5 **只有 seed 42**。
2. idiom 头训练噪声大（B0 两 seed 差 7.67pt）⇒ δ 被抬高；δ 降到 2pt 不改 P1/P3 结论（§4）。
3. ownership 全臂挤在 1pt 内 ⇒ 已补 4× 预算探针回答"是不是 1344 步天花板"（结果见下）。
4. 跨 seed 的 eval 集不同（沿用 capability_map / dev-notes/15 口径），跨 seed 只比"同 seed 内的配对差"。
5. C5 与 core_keep J0 同配方但非比特级复现（训练前多做 step-0 评测改变了 DataLoader RNG 流）⇒ 健康检查是"同配方对 J0"，不是"复现 J0"。

**推断（未测）**
6. hold-out 只有 2 个且都在 `tagging` 族；`selection` 族两张卡已知贴地板被排除 ⇒ **判据在 selection 族上无证据**。
7. 若预检放宽到更长预算，两候选大概率仍过关（余量 +50~70pt），但**没有实测**。
8. "C1 在 idiom 上掉下去"是表示受损还是学得慢 ⇒ 由 4× 预算探针回答（下表）：**一半是慢、一半补不回**；只有 seed 42。

### 4× 预算探针（探索性，5376 步，seed 42，**不进 P1/P2/P3**）

| hold-out | 核 | 1344 步 | 5376 步 | Δ |
|---|---|---|---|---|
| idiom | B0 | 0.8033 | 0.8100 | +0.67 |
| idiom | C1 | 0.6817 | **0.7467** | **+6.50** |
| idiom | C5 | 0.8100 | 0.8183 | +0.83 |
| ownership | B0 | 0.9233 | 0.9300 | +0.67 |
| ownership | C1 | 0.9167 | 0.9200 | +0.33 |
| ownership | C5 | 0.9167 | 0.9250 | +0.83 |

- **ownership 的平台不是预算天花板**：4× 预算三个核都只动 ≤0.8pt、仍挤在 0.92~0.93，且**各核同顶** ⇒ 与"表示本来够用、无多样性红利"一致（**实测**）。
- **idiom 上 C1 的落后只有一半是"学得慢"**：4× 追回 6.5pt，但仍比 C5 低 7.17pt、比 base 低 6.33pt ⇒ 补不回的部分（**推断**为表示受损）。
- ⚠️ 单 seed、事后追加、判据不使用。

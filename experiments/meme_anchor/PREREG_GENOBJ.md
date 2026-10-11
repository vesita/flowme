# PREREG_GENOBJ —— H2：**自由生成必须被当目标训练**（生成目标轴）

**本文件在跑之前写死；判据不因结果调整。**

## 0. 要回答什么

现状（v0 配置 = `REPORT_BRIDGE_ALIGN §4`：冻结 α0.5 输入卡 + 适配卡 mid=64 + 两语言输出卡 + 衔接处 `infonce λ=0.3`，8000 步）：

| 读数 | 值 |
|---|---|
| teacher-forced 重构（中→中 / 英→英） | **92.2% / 96.2%** |
| **自由生成 EM**（贪心自回归，中→中 / 英→英） | **65.0 / 70.5%**（s1234）· **64.5 / 63.5%**（s5678） |

**猜测（H2）**：这两行的差距**不是容量问题，是目标问题** —— 现有 4 项 CE **全部是 teacher forcing**，
**从未为「模型自己走错之后的恢复」付过费**。⇒ 加一项**自生成前缀的 CE**（rollout）应提升自由生成，且不伤重构。

**这不是新机制**：它是 exposure bias / scheduled sampling 的标准做法；在**本项目里从未被测过**。
外部参考（只作定位、不作判据）：LLM-JEPA（[arXiv:2509.14252](https://arxiv.org/abs/2509.14252)）的结论是"表征预测目标与生成损失互补"。

## 1. 单变量与两臂（写死）

| 臂 | 目标 | 其余 |
|---|---|---|
| **T（对照）** | `--obj tf`：现状 4 项 teacher-forced CE | 同下 |
| **R（处理）** | `--obj roll --roll-w 0.5`：`L = L_tf + 0.5 · mean(4 项 rollout CE)` | 同 T |

两臂共同（逐字）：
`--bridge parser --bridge-mid 64 --loss infonce --align-on bridge --maps AaBb,AbBa`
`--init-inputs logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s{S}.pt --freeze-inputs --init-outputs`
`--steps 8000 --eval-every 1000`；`seed ∈ {1234, 5678, 9012, 3456}`（**n=4**）。

★ **前置（不是本试验的一部分）**：v0 输入卡目前**只有 s1234 / s5678 两个**（`ls logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s*.pt`）。
n=4 需要**先补 s9012 / s3456** 两个同配方输入卡，配方逐字同 L3-m2 的 α=0.5 臂：
`--run 0.0 <S> --steps 8000 --eval-every 1000 --maps AaBb,AbBa --loss none --abba-weight 0.5`
（`logs/86g_alpha_a0.5_s1234.log` 的 `[M2]` 行可核对：`lam=0.0 abba_w=0.5 loss=none bs=32`，总参数 9,222,144）。
⇒ 两个补齐 run 完成、且 4 个 ckpt 齐了，才允许起 H2 的 8 个 run（跑批脚本的 fail-closed 预检卡这一条）。

**rollout 项的精确定义（实现必须逐字照做）**：

1. pass 1（`torch.no_grad()`）：`lg1 = model.logits(lang, meme, t_in)`；`pred = lg1.argmax(-1)`
2. 自生成输入：`r_in[:, 0] = BOS`；`r_in[:, i] = pred[:, i-1]`（i ≥ 1）
3. pass 2（带梯度）：`lg2 = model.logits(lang, meme, r_in)`；`roll = crit(lg2.reshape(-1, VOCAB), t_out.reshape(-1))`
4. 四项映射（zz / ee / ze / ez）各算一遍，roll 项取四项均值
5. ★ **pass 1 必须在 `model.eval()` + `torch.no_grad()` 下执行**（只取 argmax，不需要 dropout）；
   ★ **pass 2 必须包在 `torch.random.fork_rng()` 里**（dropout 照常生效，但**不消耗全局 RNG**）。
   ⇒ 两臂每步的**全局 RNG 消耗逐位一致**（R34；否则 R 臂多出的两次前向会让 dropout mask 从第 2 步起分叉）。

★ **这不是"算力更多所以更好"**：两臂的**梯度步数（8000）与数据量完全相同**，roll 臂多出的只是一次 `no_grad` 前向
（用来构造输入，**不产生额外更新**）⇒ 优化预算相同，差异只来自"输入 token 是模型自己的还是金标"。
（R30 的等算力对照在此表现为**等更新步数**。）
G1 的自由生成在**固定前 200 条 test** 上评（`gen_eval` 自带 `test_ids[:200]`），两臂**同一子集** ⇒ 配对可比。

## 2. 仪表自检（**跑前必过**；任一条不过 ⇒ 不许进正式跑）

| # | 检查 | 门槛 | 为什么 |
|---|---|---|---|
| **N1** | **惰性对照**：`--obj roll --roll-w 0` 与 `--obj tf` 在**同一批**上的 loss | 逐位相同（`max\|Δ\| ≤ 1e-6`） | 证明新通路在权重 0 时**真的不参与**（"改了 ≠ 改上了"） |
| **N2** | **非零性**：初始化时（step 1）roll 项 CE **严格大于** tf 项 CE，且四项都如此 | 四项全真 | 证明 rollout 用的是**模型自己的预测**（若误用 GT，两者相等） |
| **N3** | **随机流**：两臂同 seed 的数据序与 RNG 消耗逐位一致 | 数据序同（`randperm(seed)` 在臂内）；rollout **不许**引入新的 RNG 消耗 | R34：比路径前先对齐随机流 |
| **N4** | **输入契约**：单批断言 `(r_in[:,0] == BOS).all()` 且 `(r_in[:,1:] == pred[:,:-1]).all()` | 必须全真 | 抓"首列没接 BOS / 移位错" —— 这两种错 N1/N2 **都抓不住**（错位序列的 CE 照样比 tf 大） |
| **N5** | **梯度截断**：单批断言 `lg1.grad_fn is None` 且 `pred.requires_grad is False` | 必须全真 | 抓"pass 1 漏写 `no_grad`" —— 数值全同，N1/N2 抓不住，只有计算图驻留能暴露 |
| **N6** | **全局 RNG 不变式**：同 seed 下，`tf` 臂与 `roll` 臂各跑 **1 步**（同批）后 `torch.get_rng_state()` 逐位相同 | 必须逐位相同 | 抓 dropout 随机流分叉（`fork_rng` 若只包了一半、或 pass 1 忘了 `eval()`，这里必红） |

## 3. 判据（写死；配对 = 同 seed 的 R − T）

| # | 判据 | 门槛 |
|---|---|---|
| **G1（主）** | **自由生成 EM**：`gen_em_zz` / `gen_em_ee` **各自报**，主判据用**宏平均** `gen_em_mean = 0.5·(zz+ee)`；a **eval 样本量 `n_eval = 1000`**（`gen_eval(..., n=1000)`；固定前 1000 条 test，两臂同一子集 ⇒ 配对） | 配对 **Δ_mean ≥ +2×SE_paired** 且 **4/4 seed 同号** |
| **G1-ext（预注册扩展）** | 若 `\|Δ_mean\|` 落在 **`[1×SE_paired, 2×SE_paired)`** 带内 ⇒ **补 2 个 seed 扩到 n=6** 再判（只允许这一种追加，不许事后加别的臂） | 扩展后按同一门槛重判；仍不过 ⇒ 落"未测出" |
| **G2（不伤保真）** | teacher-forced 重构 `char_zz` / `char_ee` | 配对掉 ≤ **`max(1.0pp, 2×SE_pair)`**（不再拍死 2pp；依据 `measurement.md §1/§2` 的"禁止统一阈值"） |
| **G3（不破对齐）** | 衔接 Δ / AUROC | 配对掉 ≤ **`max(0.01, 2×SE_pair)`**（同上，不再拍死 0.02/0.03） |
| **G4（读数完备）** | 8/8 run 有 `[RUN-FINAL]` + 四项生成 EM + 四项 char | 缺任一 ⇒ **该臂读数不完备、判词作废** |
| **G5（Δ_min 声明）** | **用两臂配对差**算：`SE_paired = std(Δ_s, ddof=1)/sqrt(k)`，**Δ_min = 2×SE_paired** | 必须报；观测 `\|Δ\| < Δ_min` ⇒ **只许写"该口径下未测出差异"**。★ **不许拿单臂 SE 顶替配对 SE**（`measurement.md:76`），也不许用 T 臂方差反推（那会与 G1 循环） |
| **G6（饱和检查）** | 每 1000 步快照的末两格增量 | 未饱和 ⇒ 读数标【下界】（R35），G1 只判方向 |

**预测值（跑前登记）**：R 臂自由生成 EM **+3~10pp**；T 臂 Δ_min 预计 **≈3.5pp**（由 n=4 反推）。
⇒ **若真实效应 ≤3pp，本设计判不出来**，届时只能写"未测出 ≥Δ_min 的效应"。

## 4. 三分支判决（写死，不许事后改）

| 分支 | 条件 | 结论 |
|---|---|---|
| **一（成立）** | G1 过（`Δ_mean ≥ 2×SE_paired` 且 4/4 同号）**且** G2/G3 都过 | **"生成缺口是目标问题"成立** ⇒ rollout 项进 v0 配方；再进 H1/H6 |
| **二（代价过大）** | G1 过，但 **G2 或 G3 破非劣门** | **目标互斥**：rollout 提升自由生成但伤重构/对齐 ⇒ 记"该权重下互斥"，下一步只许调 roll 权重或换形态（须新预注册） |
| **三（未测出）** | G1 未过，且**未达显著为负**（含 `1SE < Δ < 2SE` 的带内情形，按 G1-ext 扩到 n=6 后仍不过） | **未测出显著收益** ⇒ 生成缺口不是（或不止是）exposure bias ⇒ **不纠缠**（`§0.2`），转 H6 |
| **四（有害）** | `Δ_mean ≤ −2×SE_paired` 且 4/4 为负 | **rollout 目标在本规模下有害** ⇒ 记结论并写进 `PROJECT_STATE.md` 的"已判死"（注明"该配置下"） |
| **五（作废）** | G4 读数不完备（任一 run 缺 `[RUN-FINAL]` 或指标） | **判词作废**，补跑后重判（不许用残缺读数出结论） |

## 5. 边界

8000 步**未饱和**（R35）· 单语料（Tatoeba 日常短句、繁简混排）· roll 权重只试 **0.5** 一档 ·
rollout 只试**一种形态**（全自生成输入，不做 scheduled sampling 的 ε 退火）· 结论**只在"输入卡冻结"的形状下**成立
（L3-m3d 已证输入卡可训会另加代价）· **不涉 `§10.1`**（无 T vs M 对照）· **不依赖假设 A**（本试验不主张模因的信息完整性，只比训练目标）。

## 6. 成本

8 run × ~11–15 min ≈ **90–120 min** 单卡（roll 臂每步多一次 `no_grad` 前向）。

## 7. 跑批与产物

- 脚本 `runs_genobj.sh`：**fail-closed 预检**（4 个 v0 ckpt、驱动、`--obj` 选项存在）
- 判据表 `stages/86m_genobj_verdict.py`（读 `logs/86g_result_*.json`）；★ **跑前必须用伪造结果逐分支驱动一遍**（R48）
- 日志 `logs/86g_gen_${obj}_l0.3_s${S}.log`
- ckpt 文件名必须含 **`o-{obj}`** 与**载入标识**（否则两臂撞名互相覆盖 —— 这条踩过）

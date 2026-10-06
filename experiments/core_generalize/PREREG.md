# PREREG — 核见过更多卡，能不能让一个「从不参与联合训练」的新能力学得更好？

> 目录 `experiments/core_generalize/`。**本文件在任何训练跑起来之前落盘**（时间戳见文件 mtime 与
> `logs/PREREG.ts`）。判据不得事后修改；跑完只许补「结果」，不许改「判据」。
> 只写 `experiments/core_generalize/`、`experiments/core_generalize/logs/`、`/tmp`。

## 0. 唯一要回答的问题

**一个见过更多卡的核，是否让一个从不参与任何核训练的新能力（hold-out）学得更好/更快？**

- 是 ⇒ **核在变通用**（假设 a）；
- 否 ⇒ **表示本来就够用**，老卡提升只是「核在旧任务上被多训了」（假设 b，dev-notes/15 §15.2 机制 1）。

## 1. 臂（核臂）

全部以 `checkpoints/base_encoder.pt` 为起点温启动（核 ← base；有冻结卡的任务头 ←
`experiments/capability_map/cards/{task}_frozen_s{seed}.pt`，与 `core_keep` 温启动同一写法），
配方与 `core_keep` J0 逐项一致：**16 epoch × 84 step = 1344 步、batch 64、lr_base 3e-4、
lr_head 1e-3、AdamW(wd 1e-4)、cosine、clip 1.0**。

| 臂 | 训练任务 | 步数 | 说明 |
|---|---|---|---|
| **C1** | `sentiment` | 1344 | 最少（1 张卡） |
| **C3** | `pronoun` + `sentiment` + `relation` | 1344 | 中（3 张） |
| **C5** | 上面 + `person` + `negation` | 1344 | 最多（5 张） |
| **C1x5** | `sentiment` | **6720**（80 ep × 84） | 数据量控制臂，**仅 seed 42** |
| **PRE-\<h\>** | hold-out h 单独联合训练 | 1344 | **可学性预检**（不是核臂） |

- 每步每任务各抽一个 batch ⇒ C5 每步样本抽取 = 5 × C1；
  **C1x5 与 C5 的总样本抽取数相同（6720×64 = 5×1344×64 = 430,176）**，差别在
  「核的优化步数 ×5」 vs 「每步任务数 ×5」。
- 核臂 seeds：C1/C3/C5 各 42、43；**C1x5 只有 42**（如实标注）。

## 2. hold-out（**绝不参与任何核训练**）

先验候选 `idiom`、`ownership`；已知贴地板的 `reply_pick`、`cloze_fill` 不选（见 §2.1）。
hold-out 数据 `6000` 条、`DEFAULT_SEED=20240927` 构建，val 划分 = `shuffle(seed)` 后前
`max(200, n//10) = 600` 条（与 `capability_map`/`train_multitask` 同口径）。

**盲猜地板 `floor` = val 中无切片样本占比**（「永远预测空切片集」的 exact_match）——
实测 `reply_pick` 联合档 0.1717 vs 地板 0.1683，正好贴地板，这个定义能复现已知的失败案例。

| hold-out | floor s42 | floor s43 |
|---|---|---|
| idiom | 0.3567 | 0.3333 |
| ownership | 0.3083 | 0.2933 |

### 2.1 可学性预检（**必须先过，才有资格当探针**）

对每个候选 h 跑 `PRE-h` × seed{42,43}（温启动 base + 随机 h 头，**同 1344 步**）：

> **h 有效 ⇔ 两个 seed 上 `exact(PRE-h) − floor ≥ δ_pre(h)`**，
> 其中 `δ_pre(h) = max(2pt, PRE-h 两 seed 极差)`。

不满足 ⇒ **排除并如实报告**（不能改预算重跑到它过关为止）。
已知 `reply_pick`（联合 exact 0.1717 / 地板 0.1683）与 `cloze_fill`
（`cls_acc` 0.19 ≈ 盲猜 0.20、exact 0.138 < 地板 0.15）属此类，**不重跑，直接引用**。

**底线**：可用 hold-out 必须 ≥ 2 个；若只有 1 个，如实报告并判「证据不足」。

## 3. 探针测量（唯一变量 = 核）

对每个核（**冻结，requires_grad=False，eval，前向 no_grad**；以及未动的 base 作参照臂 **B0**），
训同一个 hold-out 的**新头**：随机初始化、1344 步（16×84）、batch 64、lr 1e-3、AdamW(wd 1e-4)、
cosine、clip 1.0 —— 头、数据、预算全同，报 `exact_match`。

- 配对：C1/C3/C5 的核 seed 与探针头 seed **同号配对**（s42 核配 s42 头，s43 配 s43）
  ⇒ 表里每格 2 个数 = 两 seed；**C1x5 只有 seed-42 行**。
- **B0**（base 核，未参与任何训练）× 2 头 seed = δ 的来源 + 起点锚。
- 同一 seed 内，所有核评的是**同一条 eval 集**（配对）；跨 seed 的 eval 集不同
  （与 `capability_map`、dev-notes/15 同口径）。
- 探针运行数：B0 4 + C1 4 + C3 4 + C5 4 + C1x5 2 = **18**。

## 4. δ（**同一公式，同一用途，不许逐臂调**）

> **δ(h) = max(2pt, B0 在 hold-out h 上两个头 seed 的 exact 极差)**

δ 来自「hold-out 头训练 2 seed」且核恒定 ⇒ 量到的是纯头训练噪声；
**算出后对 h 的所有比较（P1/P2/P3）共用一个值**。
`δ_global = max(δ(idiom), δ(ownership))` 用于健康检查的作废线（也是一个数，跑前定）。

## 5. 预注册判据

- **P1（多样性有效）**：`exact(C5) ≥ exact(C1) + δ(h)` —— **两个 seed 都满足**才算过。
- **P2（单调）**：`C1 ≤ C3 ≤ C5`，允许 C3 落在 δ 带内：即
  `C3 ≥ C1 − δ` 且 `C5 ≥ C3 − δ` 且 `C5 ≥ C1`，**两个 seed 都满足**才算过。
- **P3（排除数据量解释）**：`exact(C5) ≥ exact(C1x5) + δ(h)`（**仅 seed 42**）。
  若 `|C5 − C1x5| < δ(h)` ⇒ **判「是数据/优化，不是多样性」**。
- **健康检查**：报每个核在**集内卡**上的 exact（每任务每 seed），对照
  ① step-0 冻结基线（`capability_map` eval.json 的 `frozen`）与
  ② `core_keep` J0（同温启动、同 1344 步、5 卡联合）。
  **作废线**：某臂任一集内任务的 exact 比 J0 同 seed 低 ≥ `δ_global` 且两 seed 同向 ⇒ 该臂作废。
- **≥2 seed**（C1x5 仅 seed 42，如实标注）；**两 seed 同号才下结论**。
- **贴地板**：任何格子的 exact ≤ `floor + δ(h)` ⇒ 标记「贴地板」，**不作结论**。

### 判定三选一（不许硬选）

| 判定 | 条件（两个有效 hold-out 都要方向一致） |
|---|---|
| **a 核在变通用** | P1 过 且 P3 过 且 P2 不反向（两 seed 同号） |
| **b 表示本来够用** | P1 不过（C5−C1 稳定 < δ），或 **P3 不过（C1x5 ≈ C5 ⇒ 数据/优化解释）** |
| **证据不足** | 两 seed 不同号 / 两个 hold-out 方向相反 / 只剩 1 个有效 hold-out / 有臂被健康检查作废 / 关键格贴地板 |

## 6. 空测试自检（跑前写死）

1. **SELFTEST_init**：每个核臂温启动后、**训练前**，在集内卡上算 step-0 exact，
   必须与 `capability_map` eval.json 的 `frozen`（同 seed 同 eval 集）**逐位一致**
   （|Δ| ≤ 1e-6，权重与数据都没被动过）⇒ 不一致则该臂作废。
   同时打印**真实可训参数量**（core 可训 / 每个头可训 / 合计，直接数 `requires_grad`）。
2. **SELFTEST_budget**：每个 hold-out 头打印**真实预算**（steps、epochs、batch、n_train、n_val、
   样本抽取总数）与 step-0 exact（随机头起跑线）。
3. **SELFTEST_floor**：所有格子标 `floor` 与 `floor+δ`；落在带内的写「贴地板」，不当结论。
4. **核漂移**：`‖core − base‖F / ‖base‖F` 每臂每 seed 打印。

## 7. 跑前就写明的局限（不许事后补救式解释）

- **卡数与数据量没有完全解耦**：C5 与 C1x5 的总样本抽取数相同，但
  「每步 5 个任务的梯度混合」≠「同一任务 5 倍步数」；C1x5 只有 seed 42。
- hold-out 只有 2 个，且都来自 `tagging` 族（`selection` 族的两张卡已知贴地板）。
- 跨 seed 的 eval 集不同（沿用既有口径），所以跨 seed 比的是「同 seed 内的配对差」。

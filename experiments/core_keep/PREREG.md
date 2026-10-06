# PREREG — core_keep：改核能否不伤老卡（跑前写死）

落盘时间：2026-10-06 04:17 (+08:00)，**训练开始之前**。
本文件一旦落盘即为判据来源；后续只允许在 `RESULTS.md` 里追加实测数字，**不许回头改这里的阈值**。

---

## 1. 唯一要回答的问题与假设

**问题**：在联合训练（允许改核）时，加上"保持老卡行为"的约束，
能否既把新能力 `negation` 做到 ~0.93+（P2 线 0.870/0.831），
又让四张老卡不退化（P1：|Δ| ≤ 各自噪声带）？

**假设**（H1）：伤老卡的原因不是"核动了"本身，而是联合训练**没有压力保持老卡行为**。
在老卡数据上加 **LwF 式输出蒸馏**（新核+新头的输出 vs 冻结旧核+冻结旧头的输出的 KL）
+ 老卡数据 replay，可以同时满足两端。

**反假设**（H0）：改核本身必然伤老卡 —— 任何保持性约束都会把 `negation` 压到 P2 线以下
（两端不可兼得），或者约束强度不足以把老卡拉回噪声带内。

---

## 2. 参照核（蒸馏的 teacher）

- **`checkpoints/base_encoder.pt` 的 `doc_encoder`**，与 `checkpoints/multitask_v2_dtseek.pt`
  的 `doc_encoder` **逐位相同**（实测 `torch.equal` 全部 26 个张量 = True，2026-10-06）。
  ⇒ 二选一无差别，本文档统称"**冻结旧核**"，实现里读 `base_encoder.pt`。
- 冻结旧核构造方式：`load_base_encoder("checkpoints/base_encoder.pt")` —— dropout=0、
  `.eval()`、`requires_grad=False`（这是 `src/dtseek/tasks/artifacts.py` 的既有行为）。
- 蒸馏的 teacher 头 = **`experiments/capability_map/cards/{cap}_frozen_s{S}.pt`**
  （四张老卡各一张，按 seed 取）。**这正是 P1 的参照基线本身** ——
  所以蒸馏目标与 P1 要保住的对象是同一份行为，不存在"保错了对象"。

---

## 3. 四档 + 一个保真对照（同一份数据、同一份步数预算）

统一口径（逐项对齐 `training/train_multitask.py` 的联合训练，**不修改该文件**）：
5 卡 `pronoun, sentiment, relation, person, negation`；
`task_samples = {pronoun:6000, sentiment:32000, relation:6000, person:6000, negation:6000}`；
`epochs=16 × steps_per_epoch=84 = 1344 步`；`batch_size=64`；
AdamW `weight_decay=1e-4`；`lr_base=3e-4`（核）/ `lr_head=1e-3`（头）；
`CosineAnnealingLR(T_max=1344)`；`clip_grad_norm_=1.0`；
每步 loss = 5 个任务的 `task_loss` **直接相加**（口径与 `train_multitask` 完全一致）；
数据划分 `random.Random(seed).shuffle(raw)` → 前 `max(200, n//10)` 为 val（= capability_map 的 `eval_S`）。

**共同温启动（J0/J1/J2/J3 四档完全一致，这是本实验的受控点）**：
- 核 ← `checkpoints/base_encoder.pt`（= 冻结旧核的权重）；
- 四张老卡头 ← `experiments/capability_map/cards/{cap}_frozen_s{S}.pt`；
- `negation` 头 ← **随机初始化**（按 seed、按与 `train_multitask` 相同的构造顺序，
  因此四档之间、以及与从头联合档之间，negation 头初值逐位相同）。
⇒ 四档在 step 0 的老卡行为**精确等于 P1 参照基线**（Δ=0），任何后续变化都是"漂移"。

| 档 | 定义 | 相对 J0 的唯一变量 |
|---|---|---|
| **J0** | 5 卡纯联合（无任何保持性约束） | — |
| **J1** | J0 + **LwF 蒸馏**：四张老卡的数据上加 KL | +KD |
| **J2** | J1 + **老卡 replay**：每步为四张老卡各再取一个 batch，CE × λ_rep | +KD +replay |
| **J3** | J0 + **冻结四张老卡头**（核可训、老头 `requires_grad=False`） | 冻结老头 |
| **F**（保真对照，仅 seed 42） | **随机初始化**的 5 卡纯联合，其余逐项同 J0 | 初值 |

### 3.1 蒸馏项的确切形式（J1/J2）

对四张老卡的每个 batch，在**教师强制**（与 `task_loss` 逐行同构的同一条轨迹）下，
逐 AR 步比较 student 与 teacher 的四个输出分布：

```
KL_s = Σ_{head∈{cls,start,end}} KL_i( T )           # 逐样本、逐头
       i ∈ {cls_logits, start_logits, end_logits, action_logits}
       log_softmax(student_i / T) vs softmax(teacher_i / T)，sum(-1)，乘 T²
KD   = Σ_{s=1..max_steps} (KL_s · step_mask_s).sum() / step_mask_s.sum()
```

- **温度 `T = 2.0`**（写死）；
- **权重 `λ_kd = 1.0`**（写死；KL 与 CE 同为"逐步 mean-over-batch 再沿步求和"，
  量纲可比，λ=1 即同权）；
- teacher 前向全程 `torch.no_grad()`，teacher 头与旧核都不进优化器；
- 只对四张老卡算 KD，**`negation` 不蒸馏**（新能力必须放开学）。
- 总损失：`L = Σ_{5 卡} CE + λ_kd · Σ_{4 老卡} KD`（J1）；
  `L = Σ_{5 卡} CE + λ_kd · Σ_{4 老卡} KD + λ_rep · Σ_{4 老卡} CE_replay`（J2）。

### 3.2 replay 的确切形式（J2）

- `λ_rep = 1.0`（写死）；每步对四张老卡各从**第二个独立 DataLoader**（同一 train 划分、
  独立 `generator=manual_seed(seed+777)` 的 shuffle）再取一个 batch，算 `task_loss` × λ_rep。
- 含义：老卡在总损失中的 CE 权重由 4/9 → 8/13（等价于"老卡数据按 1:1 混入"）。
- 主 batch 的取法、顺序、迭代器位置与 J0/J1 **完全一致**（replay 是额外的，不替换）。

### 3.3 J3 的确切形式

- 四张老卡头 `requires_grad_(False)`，**不进优化器**，但**仍参与老卡 CE**（梯度穿过头流到核）；
- 核可训（lr_base）、`negation` 头可训（lr_head）；
- **无蒸馏、无 replay** —— 它是用来分离"核漂移 vs 头漂移"的诊断档。

---

## 4. 预注册判据（跑前写死）

### P1（老卡不退化）—— 逐卡用**自己的**噪声带，禁止统一阈值

基线 = `experiments/capability_map/cards/{cap}_frozen_s{S}.pt` 在
**同一评测集** `eval_S`（= `capability_map/probe.split_of(cap, S)`，与训练 val 逐位同源）
上的 `exact_match`。**评测脚本会在同一次运行里现算基线**，并要求与
`capability_map/eval.json` 的记录值一致（自检 4）。

| 卡 | 基线 s42 | 基线 s43 | 噪声带（写死） |
|---|---|---|---|
| pronoun | 0.9533 | 0.9800 | **0.0283** |
| sentiment | 0.7938 | 0.8147 | **0.0041** |
| relation | 0.9639 | 0.9653 | **0.0139** |
| person | 0.3100 | 0.3233 | **0.0033** |

**判据**：四张老卡**全部**满足 `|exact − 基线| ≤ 该卡噪声带` ⇒ P1 过。
噪声带来源：dev-notes/12 §12.5 附（4 卡同配置同数据仅换 seed 42↔43 的逐卡差）。

### P2（新能力到位）

`negation` exact ≥ **0.870（s42）/ 0.831（s43）**。

### P3（代价与配平，必报不判过）

参数量（真实 `requires_grad` 计数）、稳态步时、峰值显存、
**蒸馏项占总损失的比例**（`mean(KD·λ_kd) / mean(L)`，按整个训练期与末 epoch 各报一次）。

### seed

**42 与 43 两个 seed**，**两 seed 都过才下结论**（单 seed 只报方向）。

### 空测试自检（缺一条结论作废）

1. **真实可训参数量**：直接数 `requires_grad=True` 的参数，按 核 / 逐头 分组打印，
   并与"档位应有值"对账（J3 的四张老头必须为 0）。
2. **蒸馏/正则项的实际数值**：打印第一步、每 epoch 均值、末值；
   J1/J2 若**恒为 0** ⇒ 判该档**失败**（约束没生效）。J0/J3 打印"本档无蒸馏项"。
3. **J3 老头 `requires_grad` 实况**：逐头逐参数打印，必须全 False，否则该档作废。

### 判定

- **J1 / J2 / J3 任一档同时过 P1 与 P2（两 seed）** ⇒ 找到"改核不伤老卡"的机制，
  并写清机制是什么（蒸馏 / replay / 冻核头 各自贡献）。
- **都不过** ⇒ 如实报，并指出**哪一端先崩**（老卡先退化，还是新能力先不够），
  给出蒸馏强度 × 老卡保持 / 新能力的取舍方向（若跑了 λ 扫描）。

### 4.1 λ_kd 升档阶梯（**训练开始前预注册**，用于画取舍曲线）

主档固定 `λ_kd = 1.0`（两 seed）。若 **J1 主档 P1 不过而 P2 过**，说明"约束方向对、强度不够"
⇒ 按 **`λ_kd ∈ {4, 16}`** 逐级升档，**先只跑 s42**（探索性，不用于下结论），
直到 P1 过 或 P2 掉线为止；每一级记录 `kd_share` / 四老卡 Δ / negation exact，
形成"蒸馏强度 × 老卡保持 × 新能力"三点取舍曲线。若 J1 主档 **P2 就不过**，
则**不升档**（约束已压过主损失），直接按判定规则报"新能力先崩"。
J2 的 `λ_rep` 与 J3 无 λ，不参与阶梯。
**阶梯结果一律标注为「探索性、单 seed」，不进主判定。**

### 4.2 追加档 J1c（**训练开始前预注册**）：从头联合 + 蒸馏

温启动分支问的是"**已经收敛的模型**加卡时能不能不退化"；从头分支问的是
"**联合训练落到哪里**"。两个问题都要有对照才算完整，所以追加一档：

- **J1c** = `training/train_multitask.py` 口径的 5 卡**随机初始化**联合 + 与 J1 完全相同的
  蒸馏项（`λ_kd=1.0, T=2.0`），seed 42/43；
- 它的对照**不重训**：直接用已有的 `checkpoints/arm_neg5_seed{42,43}.pt`
  （= 从头联合、无约束，同一 `eval_S`，`capability_map/eval_cards.py` 已 align 校验）。
- J1c 的判定与 J0~J3 **同 P1/P2**，但**单独成列**，不与温启动档混着下结论。
- 另有 **F 档**（= 从头联合，seed 42）作为**我自己的训练脚本的保真对照**：
  若 F 与 `arm_neg5_seed42` 的五卡 exact 偏差 > 2pt，则判"脚本不可信"，**全部结论作废**。

### 4.3 追加对照 J4（**04:57 追加，在 J4 启动之前写死**）：温启动 + 冻结核

动机：J0/J1 的老卡**向上**超带（sentiment +12pt），而温启动的老卡头是从 frozen 卡
**继续训了 1344 步**的 ⇒ 提升里"核变好"与"多训了一倍"两个来源混在一起。
补第四格做 **2×2 分解**（step 0 四格都在基线）：

| 档 | 核 | 老卡头 | 分离出的效应 |
|---|---|---|---|
| 基线（step 0） | 冻 | 冻 | — |
| **J3** | 训 | 冻 | **纯核漂移** |
| **J4**（新） | **冻** | 训 | **纯头训练（多训 1344 步）** |
| **J0** | 训 | 训 | 两者合计 |

- J4 配方：温启动、`freeze_base=True` 口径（核 `requires_grad=False` + `.eval()` 关 dropout，
  **核不进优化器**），五张头全可训，其余与 J0 逐项相同。
- **判定不改 P1/P2**：J4 只用于解释"老卡涨了是谁的功劳"，**不参与主判定**，
  结果一律标注「追加对照」。
- 时机声明：本节写于 J0_s42 / J1_s42 已出结果之后、**J4 尚未启动之前**，
  属"结果驱动的追加"，如实标注，不冒充预注册主档。
### 4.4 蒸馏目标的口径说明（写死，避免事后换口径）

蒸馏项用的是 **LwF 正统形式**：
`KL( 老卡头_学生(新核, x) ‖ 老卡头_教师(冻结旧核, x) )` —— 两侧都经过**该老卡的头**，
学生侧头可训、教师侧头冻结。
备选口径 `老卡头(新核,x) ‖ 老卡头(冻结旧核,x)`（同一个头、只换核，纯测核漂移）
**本次不用**：它约束不了头漂移，而 P1 测的是最终行为（头+核合起来的输出）。
头漂移与核漂移的分离由 **J3（冻老卡头）** 承担，不由蒸馏项承担。

### 附：从头联合的"伤老卡"锚点（复用已有产物，不再重训）

`checkpoints/arm_neg5_seed{42,43}.pt` = `training/train_multitask.py` 的 5 卡**随机初始化**联合，
其 `exact` 已由 `experiments/capability_map/eval_cards.py` 在**同一 `eval_S`** 上评过
（`align_all_ok=True`）：
pronoun 0.9333/0.9300、sentiment 0.7100/0.7034、relation 0.7986/0.9194、
person 0.3133/0.3217、negation 0.9667/0.9233。
它作为"**不加任何约束的联合训练**"的外部锚点；`F` 档则用来验证**我自己的训练脚本**
能否复现它（保真度自检，偏差 > 2pt 视为脚本不可信、结论作废）。

---

## 5. 允许写入的路径

只写 `experiments/core_keep/`、`logs/corekeep_*`、`/tmp`。
不改 `src/`、`training/`、`tests/`、`dev-notes/`、`experiments/{additivity,capability_map,core_branch}/`。
禁止 `git commit/stash/checkout/restore/clean`。

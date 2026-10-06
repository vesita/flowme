# PREREG — cumulative_add：连续加卡（每步温启动联合）能不能反复用（跑前写死）

落盘时间：2026-10-06 08:45 (+08:00)，**训练开始之前**。
本文件一旦落盘即为判据来源；后续只允许在 `RESULTS.md` 追加实测数字，**不许回头改这里的阈值**。
唯一允许在跑中补写的机器文件是 `bands.json`（见 §4.1，公式已在此写死，只有数值后填）。

---

## 1. 唯一要回答的问题

dev-notes/15 §15 已证明「**一步**温启动联合加卡」不伤老卡（8 格 × 2 seed，sentiment +12pt）。
但那只验证了「加一张卡」这一步。本实验问：

> **反复用这套流程（每步都在上一步产物上温启动联合训练），**
> **(1) 老卡会不会随步数累积退化？(2) 每步的代价与核漂移会不会失控？**

## 2. 流程与配置（唯一口径 = §15 的 J3：核可训、老头冻结）

起点：`checkpoints/base_encoder.pt` + 四张已收敛老卡
`experiments/capability_map/cards/{pronoun,sentiment,relation,person}_frozen_{seed}.pt`。

| 步 | 加什么 | 训练后评测对象 | 该步可训参数 |
|---|---|---|---|
| 0 | （不训，门禁） | 四张老卡基线 | 0 |
| 1 | `negation` | 4 老卡 + negation | 核 + negation 头 |
| 2 | `idiom` | 上一步全部 + idiom | 核 + idiom 头 |
| 3 | `ownership` | 上一步全部 + ownership | 核 + ownership 头 |

- **每步都在上一步的产物上温启动**：核 ← 上一步 ckpt 的 `doc_encoder`；
  已存在卡的头 ← 上一步 ckpt 的对应头，且 **`requires_grad=False`（冻结）**；
  新卡头 ← 随机初始化、可训。步 1 与对照档的起点 = `base_encoder` + 四张 frozen 卡（= §15 J3 的起点）。
- **训练口径与 `experiments/core_keep/train_core_keep.py` 的 J3 逐项一致**（不改该文件）：
  `epochs=16 × steps_per_epoch=84 = 1344 步`；`batch_size=64`；AdamW `weight_decay=1e-4`；
  `lr_base=3e-4`（核）/ `lr_head=1e-3`（头）；`CosineAnnealingLR(T_max=1344)`；
  `clip_grad_norm_=1.0`；每步 loss = 该步**在场的全部卡**的 `task_loss` 直接相加；
  无蒸馏、无 replay（J3 口径）。**每步都是满 1344 步，不累积步数预算。**
- 数据：老 5 卡与 §15 同源（`capability_map/cache/*_20240927.pkl` 只读复用，sentiment 4 分钟构建不重做）；
  `idiom=6000`、`ownership=6000`（新缓存 `experiments/cumulative_add/cache/`）。
  划分 `random.Random(seed).shuffle(raw)` → 前 `max(200, n//10)` 为 val，**同一卡跨步 val 必须逐条相同**
  （metrics 里记 val 文本 sha256，跨步不一致 ⇒ 该步作废）。
- 构造顺序固定 `CARD_ORDER = pronoun, sentiment, relation, person, negation, idiom, ownership`
  （决定新卡头的随机初值；步 1 因此与 §15 J3 的 negation 头初值同流）。

### 2.1 对照（P2 需要，必须自行跑）

「它单独温启动联合」= **从 `base_encoder` + 四张 frozen 老卡出发，只加这一张卡**，同口径 1344 步、同 2 seed：

| 对照 | 卡组 | 说明 |
|---|---|---|
| `ctrl_negation` | 4 老卡 + negation | **≡ §15 的 J3**，直接复用 `experiments/core_keep/cards/J3_s{42,43}.pt`，不重训 |
| `ctrl_idiom` | 4 老卡 + idiom | 本实验新跑 |
| `ctrl_ownership` | 4 老卡 + ownership | 本实验新跑 |

⇒ 对 negation，链上第 1 步与对照**同构**，P2 在该步退化为恒等式（如实标注），
真正的 P2 检验在 idiom / ownership 两步；同时链上第 1 步 vs `J3` 是一次**跨实现复刻自检**。

## 3. 起点门禁（**必须实测，不是声明**）

**「逐位相同」的操作定义**（跑前写死，两种观测都记）：
- **主判据（输出逐位）**：对同一 val 集做一次完整自回归发射，把**每个样本的切片序列**
  `[(label,start,end),...]` 全部序列化进 sha256 —— 摘要字符串必须**完全相等**。
- **补充观测（logits 逐位）**：首步 `cls/start/end logits` 的 float32 原始字节 sha256。
  两项都相等 ⇒ 最强结论；只有主判据相等 ⇒ 照过门禁，但如实报「logits 摘要不等」。
- 另外比 `exact_match` 等全部指标的 **`==`（不是 `<1e-6`）**。

三处门禁，任一失败 ⇒ **该步作废**（打印 `GATE_FAIL` 并非零退出）：

| 门禁 | 比什么 |
|---|---|
| **G0（步 0 起点）** | `base_encoder` + 四张 frozen 卡的四卡输出 vs **`capability_map/summary.json` 的 `frozen_exact`（各自单独跑的记录值）**，`==` 逐位 |
| **G_i（步 i 起点，i≥1）** | 载入步 i−1 产物后，**该步已存在全部卡**的输出 vs 步 i−1 结束时记录的输出，逐位 |
| **S（分割一致性）** | 每卡 val 文本 sha256 跨步不变 |

## 4. 预注册判据

### P1（不累积退化）—— 逐卡用**自己的**噪声带，**禁止统一阈值**

每一步结束后，**此前所有卡**的 `exact` 相对**该步起点**：`Δ = end − start ≥ −band(c)`（**单侧**，只惩罚退化；
双侧读法 `|Δ| ≤ band` 同时报，作为次读法 —— §14 的教训：提升不算违规）。

| 卡 | band（写死，单位 pt） | 来源 |
|---|---|---|
| pronoun | **2.83** | dev-notes/12 §12.5（同配置仅换 seed 42↔43 的逐卡差），与 core_keep 相同 |
| sentiment | **0.41** | 同上 |
| relation | **1.39** | 同上 |
| person | **0.33** | 同上 |
| negation | **1.83** | **2 seed 极差**：§15 J3 单独温启动联合 `0.9817 − 0.9633`（既有数据） |
| idiom | `bands.json` | **2 seed 极差** `|ctrl_idiom(s42) − ctrl_idiom(s43)|`，公式写死、数值由 `bands.py` 从**对照**算出 |
| ownership | `bands.json` | 同上，用 `ctrl_ownership` |

> `bands.json` 只含对照跑出来的数（对照先于链上跑），**不含链上任何数据**；
> 文件带生成时间戳，链上第 1 步在其之后启动 —— 这条先后顺序就是"没看着链上调阈值"的证据。

**P1 判定**：某一步中所有已存在卡全部 `Δ ≥ −band`，且 **seed 42 与 43 都成立** ⇒ 该步过。
另**报不算判**：相对**步 0 基线**的累计 `Δ_cum`（回答"累积"的字面问题）。

### P2（新卡到位）

每步新加入的卡 `exact(s) ≥ 0.90 × ctrl(s)`，**逐 seed**，两 seed 都过才判过。
对照按 §2.1；negation 步照算但标注"同构、恒等式"。

### P3（代价与漂移，逐步报、不判过，但要回答"线性还是加速"）

每步报：**稳态步时**（末 200 步中位数）、**总步时**、**峰值显存**、
**核漂移** `d_i = ‖core_i − core_0‖F / ‖core_0‖F`（`core_0` = `base_encoder`），
以及步间增量 `δ_i = d_i − d_{i-1}` 与 `‖core_i − core_{i−1}‖F/‖core_{i−1}‖F`。

- **线性 vs 加速**（写死）：`δ_2 < δ_3` 两 seed 同号 ⇒ **加速**；`δ_2 > δ_3` 两 seed 同号 ⇒ **减速/收敛**；
  否则 **混合**（如实写）。首段 `δ_1 = d_1 − 0` 单列（它对应 §15 已知的那一步）。
- **"失控"的操作定义**（描述性，不替代 P1）：
  代价失控 = 步时或峰值显存相对步 1 增长 >20%（两 seed 同号）；
  漂移失控 = `δ` 单调加速且 `d_3 ≥ 2 × d_1`（两 seed 同号）。

### seed 与判定

- **seed 42 与 43 两个 seed**；**两 seed 同号才下结论**，单 seed 只报方向。
- **判定三选一**：`流程可反复用`（P1、P2 全过且两 seed 一致）/ `不可反复用`
  （指出**在第几步、哪张卡先崩**、崩的是 P1 还是 P2）/ `证据不足`（任一 seed 缺失、
  或门禁失败导致某步作废、或两 seed 不同号）。

### 空测试自检（缺一条结论作废）

1. **真实可训参数量**：数 `requires_grad=True` 的参数，按核 / 逐头分组打印（`SELFTEST_1`），
   老头必须为 0、新卡头必须非 0。
2. **老头是否真的冻结**：逐头逐参数打印 `requires_grad` 实况（`SELFTEST_3`），全 False 才算；
   另打印优化器 param group 的实际张量数（确认冻结头不在优化器里）。
3. **起点门禁实测**（G0 与每个 G_i，`SELFTEST_GATE`），必须是跑出来的比较结果，不是声明。
4. **贴地板检查**：任一卡在任一步 `exact` 低于其**盲猜下界**（按类别数与背景占比估）或与背景占比同量级，
   如实列「贴地板」，不许当作"能力到位"。
5. val 文本 sha256 跨步一致（`SELFTEST_SPLIT`）。

## 5. 允许写入的路径

只写 `experiments/cumulative_add/`、`logs/cumadd_*`、`/tmp`。
不改 `src/`、`training/`、`tests/`、`dev-notes/`、
`experiments/{core_keep,capability_map,core_branch,additivity,out_invariants,input_type,core_generalize}/`。
禁止 `git commit/stash/checkout/restore/clean`。单卡 GPU：启动前检查 `systemctl --user is-active 'dtseek-*'`，
忙则排队，**绝不杀别人的进程**。

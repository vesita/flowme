# PREREG — free_rule_floor：完备免费规则地板 + `n_slots`-only 对照臂

**本文件 mtime 早于本单元首次运行**（先落 PREREG，再写代码、再开跑）。判据 F0–F5 与三选一判定在看到本单元任何输出之前写死，跑完不改。

---

## 0. 本单元的新臂是「预注册对照」，不是事后调参

- **是什么**：`bag_modules` 已把「骨架增益是否由袋项个数 `n_slots` 承载」压成**强推断**（4 条实测链，但**没有单独的「只给 `n_slots`」训练臂**，其 PREREG §7 明确写了"按 PREREG 不做事后补训"）。本单元由主 AI 在 `bag_modules` 结论落地后**新立项**，把这个推断**转成一次直接实验**，并同时定下**完备的免费规则电池**。
- **为什么算预注册**：
  1. 臂表（A/N/U/UP）、配方、seed、判据 F0–F5、电池清单、负对照门槛**全部在本单元首跑之前写死**，不随结果调整；
  2. 新臂 N 是**由 `bag_modules` 既有实测链先验指定的对照臂**（假设方向来自上一个已冻结的实验，不是来自本单元的任何输出）；
  3. 继承 `bag_modules` 的一切口径（模型、数据、切分、同步数、同 seed、同配方），**本单元不引入任何可调自由度**；若 F0 复现失败，**只修探针、不改判据**。
- **因此**：N 臂的结果可以与 A/U/UP 直接对账；它不是"看到结果后补的臂"，而是**跑前声明的对照臂**。

---

## 1. 臂（同模型、同数据、同步数、同 seed、同配方）

| 臂 | 骨架头额外输入 | 说明 |
|---|---|---|
| **A** | 无 | 复现 `bag_modules` A（= `struct_supervision` B，逐位） |
| **N（本单元新增）** | **只给 `n_slots`**（袋项个数）：一条只读 `mask` 的 aux 旁路 | **不含任何模块标签**（type/role/cls 全不进）、**不含位置桶**（pos_b 不进） |
| **U** | 模块标签序列（type/role/cls + pos_b + mask） | 复现 `bag_modules` U |
| **UP** | 标签 + 位置感知池化 | 复现 `bag_modules` UP |

**N 臂定义（写死）**：
- `skel_logits = skel_out(h) + aux(z)`，`z` 来自一个 GRU，**每一步输入都是同一个可学常向量**（内容张量不进）、只按 `mask` 做掩码均值 ⇒ GRU 的输出序列只依赖 `mask` 的前缀长度，即 **`z = f(n_slots)`**（`max_slots = 4`，`n_slots ∈ {1..4}`）。代码层面对 `logits(mask, type_t, role_t, cls_t, pos_b)` **只用 `mask`**，其余入参显式收下但丢弃（selfcheck 断言：把 aux 置为随机非零后，**改标签内容不改 N 的 logits**、**改 n_slots 会改**）。
- 构造顺序与 U **完全一致**（trunk → 占位 → gen → lab），trunk/gen 初值与 A/U/UP **逐位相同**；aux **零初始化** ⇒ 第 0 步 N 的 logits == A。
- 参数量（selfcheck 实测）：A = 186,088；N = A + 5,320（GRU 4,032 + aux 1,280 + 常向量 8）；U = A + 5,456（差 144 = 4 个标签嵌入）；UP = 193,088。
- **读法**：N = A + 「袋项个数」旁路；U = A + 「袋项个数 + 标签内容 + 位置」旁路 ⇒ **U − N = 标签内容与位置的净贡献**（位置与标签内容合并报告，不拆开判）。
- **技术勘误（跑前记录，判据一字未改）**：初稿写的是"GRU 输入张量恒为 0"——自检拦下（零输入 + PyTorch GRU 零隐态 ⇒ 隐态恒为 0，旁路彻底退化、N≡A）。改为**每步同一可学常向量**；这只改实现细节，"旁路只读 `mask`、`z = f(n_slots)`"的定义与 F0–F5 判据均不变。**发生时间早于本单元任何训练运行。**

**配方（写死，继承 `bag_modules`）**：1800 步、batch 64、AdamW lr 1e-3 / wd 1e-4、cosine、grad clip 1.0、seed ∈ {42, 43}、核 1,688,460 全程冻结 + `eval()`、`DataLoader(generator=manual_seed(seed))`、骨架 CE + 指派 CE（`L_gen`）、标签序列按 span 句序。
**数据（写死）**：train/test = `two_channel_head/data/*.jsonl` 原切分；adv1/adv2 = `struct_supervision/data/*.jsonl` 原切分；**不重新抽语料**；标签文件 = `bag_modules/data/labels_*.json` **原样复制**（不重跑标注器、不改标签）。

---

## 2. 判据（写死；判定三选一，不许硬选）

- **F0 复现（前置门）**：A / U / UP 的 test 与 adv2 骨架 acc 必须与 `bag_modules/REPORT.md` 报告值 **|Δ| ≤ 2×行级 SE 且同号**（SE test .0100 / adv2 .0158）。对不上 ⇒ **先修探针再继续**，不许带着错的复现往下跑。
- **F1（主判据）N vs U**：配对 Δ = U − N（逐行 0/1 差的配对 SE），2 seed。
  - |Δ| ≤ 2×SE 且两 seed 同号 ⇒ **「增益由袋项个数承载、模块内容贡献 ≈ 0」成为直接实验结论**；
  - |Δ| > 2×SE 且两 seed 同号 ⇒ **模块内容有独立贡献**；
  - 两 seed 异号 / 落在 2SE 区间内无法定号 ⇒ **证据不足**。
- **F2 逐臂 vs 完备 `max_naive`**：报 `卡 − max_naive`（新电池）与 `卡 − max_naive`（旧电池 .5329 / .182 / 0）**两个数字都留**。
- **F3 旁路探针**：逐臂 `true → shuffled` 骨架 acc；**N 臂报打乱 `mask` 后的掉落**；A 无标签通道 ⇒ 记「不适用」。U/UP 另报「只打乱标签内容、保留 mask」变体（把内容与计数在评测期拆开）。
- **F4**：2 seed 同号 + SE 必报；不许只报一个 seed。
- **F5**：电池逐规则清单与数值（含 seen_rate）。
- **负对照（硬门，必做）**：`U-rand`（训练标签跨行 randperm，**mask 不动**）× 2 seed 必须复现「增益不消失」：`Δ_test(U-rand − A) > 2×SE` 且两 seed 同号，量级 ≈ +0.20。**不复现 ⇒ 判「管线与 bag_modules 不同构」，本单元 F1 结果作废，只报失败。**
- **置信度**：区分**实测 / 推断**，标注证据强度（单 seed / 配对 / 2 seed）；**「没测出差异」不写成「没有效果」**。

---

## 3. 完备免费规则电池（写死）

**计入 `max_naive` 的 11 条**（`fit = train`，`eval = split`，未见键回退 train 全局多数）：

- **旧 8 条 —— 逐字复用 `two_channel_head/build_gen_data.py::naive_skeleton`（口径对齐）**：
  `majority` / `len_bucket`(`len//6`) / `first_char` / `last_char` / `punct_pattern` / `fw_decision_list` / `tree_depth2` / `tree_depth4`。
  **对齐门槛**：我自算的这 8 条必须**逐位等于** `two_channel_head/data/stats.json` 的 `naive_skeleton_test` 与 `struct_supervision/data/stats_adv.json` 的 `adv1_skel`/`adv2_skel`（即 .5208 / 0.0 / .182），train 口径须等于 .5329；对不上 ⇒ 口径未对齐，先修。
- **新增 3 条**：
  1. `n_slots → train 多数骨架`；
  2. `(n_slots, 标签序列)` → train 多数（标签 = 按 span 句序的 `(t, r, c)` 三元组序列）；
  3. `袋项类别多重集` → train 多数（**类别 = 该袋项的 `(t, r, c)` 三元组**，排序去序）。
- **单列披露但不计入 `max_naive`**：`bag_text_multiset`（袋项**字面文本**多重集 → train 多数）—— 与 `two_channel_head` 对 `annotation_inverse_pos_sort` 的处置同类（字面足以逼近标注器的分组，属"标注函数的逆"一类）。**此条处置在跑前写死，不由跑出的数值高低决定。**
- 逐规则报 test / adv1 / adv2 的 acc（+ 查表类的 seen_rate）+ `max_naive_new = max(11)`；并报与旧电池 `.5329 / .182 / 0` 的差。

---

## 4. 与既有结果的关系

- 旧 `max_naive`（test .5329、adv2 .182、adv1 0）**不含 `n_slots` 特征**，本单元如实并列两个口径，**不覆盖、不删改** `bag_modules/`、`struct_supervision/`、`two_channel_head/`、`src/`、`training/`、`dev-notes/`、`methodology/` 的任何文件。
- **只写** `experiments/free_rule_floor/`、`logs/`、`/tmp`；**禁止** git commit / stash / checkout / restore / clean。

## 5. 运行

`systemd-run --user --unit=dtseek-frf --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 ...`（分析/probe 用 `dtseek-frf2`）；一律 `uv run python`；GPU 忙则排队，不杀他人进程。

**停止边界**：做完 P0–P3 + 2 seed 即停；不改既有实验、不加新卡、不训核、不写 `dev-notes/`/`methodology/`、不派子代理、不动 `render.py`。

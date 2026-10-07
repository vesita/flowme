# PREREG —— P4 功能词最小对训练信号（`experiments/funcword_minpair/`）

> **本文在首次训练之前写死**（`mtime` 必须早于任何 train 进程启动时刻），跑完不改；结果填 `REPORT.md`。
> 口径：「实测」= 本目录跑出的数字；「推断」= 由数字推出；「口径判断」= 构造/定义时人为规定的归属。
> **只读**：`skeleton_leak/`、`bag_modules/`、`free_rule_floor/`、`struct_supervision/`、
> `two_channel_head/`、`card_flow/`、`src/`、`training/`、`dev-notes/`、`methodology/`。
> **只允许写**：`experiments/funcword_minpair/`、`logs/`、`/tmp`。禁止 git commit/stash/checkout/restore/clean。

## 0. 唯一问题（写死）

**同一模型、同一批训练行、同 batch 序、同步数、同 seed，只把「骨架监督形式」换掉
—— 常规多类交叉熵（M）vs 最小对判别式（MP）vs 两者相加（MP+），
谁能让模型在 **held-out 最小对** 上的 `pair_success` 显著超过机会水平 50%？**

- **背景（既有实测，本单元不重跑、不重新论证）**：`experiments/skeleton_leak/` 实测
  四臂在最小对上 `pair_success ≤ 2.14%`、对内预测 79~85% 不变、距成对机会 .5 差 t = −25~−31
  ⇒ **模型几乎不做功能词判别**。本单元的目的是**把这条诊断直接测掉**（换监督形式能不能救）。
- **边界（写死）**：本单元回答「**最小对判别式监督能否让模型读出功能词**」；
  **不回答**「换特征/换核能否提升」，**不回答**「功能词在核里有没有」（那是另一个问题，
  本单元只测**给定现有冻结核 + 只训头**时监督形式的效应）。
- **核 1,688,460 全程冻结 + `eval()`，只训头**（与 `struct_supervision` / `bag_modules` 同口径），
  除非本文写了别的 —— 本文**没写别的**。

---

## 1. 数据与最小对构造（跑前写死）

### 1.1 语料与既有数据（只读）

- 训练/测试**只读复用** `experiments/two_channel_head/data/{train,test}.jsonl`（8000 / 2500，不重建）；
  编码只读复用其 `cache/gen_*_L64_M4.pt`。
- 评测集**只读复用** `experiments/skeleton_leak/data/{a_bal,a_lit,b_pairs,c_pairs}.jsonl`
  与 `experiments/skeleton_leak/cache/enc_*_L64.pt`（不改、不重建、不往对方目录写）。
- 本单元**不新建语料扫描**（既有的 B1/B2 合成对已由 `skeleton_leak` 在同口径下构造并断言
  与 train/test/adv1/adv2 零重叠）⇒ `dev-notes/19` 的过滤由上游继承：
  文件级汉字占比 ≥0.6 保留 14/18（丢 code_alpaca .0004 / gsm8k .0585 / coig_math .5418 / qwen3 .5112），
  `SPLIT_SEED` 打乱文件序、**不写 `fs[:N]`**；句级汉字比 ≥.6、句长 14–44。
  文件清单 = `skeleton_leak/results/leak_stats.json::lang.files_kept`（本单元照抄并列出）。
- 骨架池 = train 出现过的 **30 个 id**；**训练零样本 id `{16,21,24,27,28,29,30,32,33,34}` 一律排除**
  （口径判断：零样本类三臂构造性为 0，混入只测「认没见过的类」）。
- ⚠️ 引擎锚点 `s0/e0` 是**闭区间**、`render` 的 span 是**半开** —— 本单元一律用
  `match_sentence` 返回的 `[start,end)` 半开 span，并断言 `sent[a:b] == bag[i]`。

### 1.2 训练最小对：三条**机械规则**（跑前写死，不事后增补）

对训练行 `r`（骨架 `s`，句序槽文本 `g1..gk`），候选目标骨架 `t ≠ s` 必须满足下列之一
（`parts(X)` = 模板按 `\[\d+\]` 切出的字面段；两骨架槽数 `k` 相同、段数相同）：

| 规则 | 定义 | 含义 |
|---|---|---|
| **R1 等长换词** | 恰有 **1 段**不同，两段**等长**，两段字面 ⊆ `ALLOWED_LIT_CHARS`，且两段**至少含 1 个非标点功能词** | 同槽位、只差一个功能词（**含标点位置随之移动**的情形，如 `[1]，因为[2]` ↔ `[1]的话，[2]`；与任务给的例 `[1]，然而[2]` vs `[1]之前，[2]` 同型） |
| **R2 功能词增删** | 恰有 1 段不同，**一段是另一段的连续子串**，增/删的子串**非空且不含标点** | 只差一个功能词（增/删），如 `A，B` ↔ `A，但是B` |
| **R3 槽序交换** | 两骨架**字面字符多重集相等**但**字面段序列不同**，且存在袋使两渲染句**槽序不同** | 同袋、同功能词集合（多重集）、只差槽序 |

**机械枚举结果（跑前实测，写死）**：
- **R1 = 62 对**、**R2 = 21 对**、**R3 = 1 对（#6 `因为[1]，[2]` ↔ #7 `[1]，因为[2]`）**。
- **MP-order 的可构造性结论（口径判断 + 机械证明）**：
  **「槽序变、功能词线性位置不变、骨架不同」在本骨架表下不可构造** —— 因为槽位编号被规范化为
  「按出现顺序连续编号」，把模板里的槽位重排后得到的仍是**同一个字符串** ⇒ 只能是同一个骨架。
  故 **唯一可构造的跨骨架槽序对 = R3（字面自身随之移位）**，全表只有 `#6 ↔ #7` 两个方向。
  同骨架的槽序交换（如 `A，B` ↔ `B，A`）**可构造但骨架两侧相同** ⇒ 对骨架 `pair_success` 无信息，
  **不进骨架判据、也不入训练**（只在 §6 作为未做项声明）。
- **覆盖（跑前实测）**：R1∪R2∪R3 覆盖 **24 / 30** 个训练骨架；
  按行计 **4274 / 8000 = 53.43%** 的行至少有 1 个候选目标；
  经 fail-closed 验证后 **4205 / 8000 = 52.56%** 的行拿到可用伙伴。
  **未覆盖骨架 = {#1,#2,#3,#4,#5,#31}（3726 行）**，逐个列数（实测）：
  #1 2087 / #2 1165 / #4 224 / #3 153 / #5 86 / #31 11。

### 1.3 伙伴选择（跑前写死，确定性、与 seed 无关）

- **每行至多 1 个 R1/R2 伙伴**：候选目标骨架按 `md5(f"P4:{行下标}:{t}")` 升序排列，
  取**第一个通过 fail-closed 验证**的；一个都不通过 ⇒ 该行**无伙伴**。
- **R3 追加**：骨架 ∈ {#6,#7} 的行**额外固定追加 1 个 R3 槽序伙伴**（若验证通过）。
  理由（跑前声明）：MP-order 是本任务指定的两型之一，若不显式追加，R1/R2 会把它完全遮掉。
- **fail-closed 验证（逐条断言）**：
  `ok_sentence(v)` ∧ `match_sentence(v) == (t, 与源行**同序同文**的槽文本)` ∧
  `v` 不与任何**已用句**（train ∪ test ∪ adv1 ∪ adv2 ∪ a_bal ∪ a_lit ∴ b_pairs ∪ c_pairs ∪ 本单元已产变体）重复。
  **未命中率 = 失败尝试数 / 总尝试数**，逐原因分类（`ok_sentence拒` / `错骨架#X` / `槽文本不一致` / `无匹配` / `重复`）。

### 1.4 held-out 最小对（**训练与评测严格分离**）

| 型 | held-out 来源（只读） | n（对） | 说明 |
|---|---|---:|---|
| **MP-word** | `skeleton_leak/data/b_pairs.jsonl` 中 `kind ∈ {b1_src,b1_variant}` | **520**（26 方向 × 20） | 同袋、同槽文本、只换功能词；两侧**同袋** ⇒ 无内容泄露 |
| **MP-order** | 同上 `kind ∈ {b2_src,b2_variant}` | **40**（2 方向 × 20） | `#7 ↔ #6` 连词位置 + 槽序同时交换（= R3 的自然句版本） |
| （诊断）自然句孪生 | `skeleton_leak/data/c_pairs.jsonl` | 13 组 × 20/side | **两侧袋不同 ⇒ 不是最小对**，只报骨架 acc，**不进 pair_success 判据** |

- **分离断言（fail-closed，跑前 + 跑后各跑一次，写死）**：
  ① 训练行句面 ∩ held-out 句面 = 0；
  ② **训练变体句面** ∩ held-out 句面 = 0；
  ③ held-out ∩ (train ∪ test ∪ adv1 ∪ adv2) = 0；
  ④ 两型 held-out 句级互斥（b_pairs 的 B1/B2 之间、与 c_pairs 之间）。
  任一非 0 ⇒ **fail-closed 停**，如实记入 REPORT。
- `pair` 的配对方式沿用 `skeleton_leak/eval_arms.py` 口径：`b_pairs` 行内相邻成对
  （`_src` 必与 `_variant` 相邻、袋相同、骨架不同）；`c_pairs` 按 `pair` 组内**下标配对**（仅诊断）。

---

## 2. 三臂与配方（**唯一变量 = 骨架监督形式**）

### 2.1 模型（同模型、同初始化）

- **复用 `experiments/struct_supervision/model.py::StructSupModel("B", seed)`（只读 import）**
  ⇒ 输入特征 `[v_sent; v_bag; v_sent⊙v_bag; |v_sent−v_bag|]`(512) → Trunk(512→256→128) →
  `GenHead`（骨架 `Linear(128→40)` + 逐槽指派头），**构造顺序与 RNG 与 struct B 逐位相同**
  （selfcheck 断言三臂初值 `torch.equal`，且 == struct B 同 seed 初值）。
- 三臂**每步输入同一批、同一 batch 序**（`DataLoader(shuffle=True, generator=manual_seed(seed))`，
  数据集长度同为 8000）；**三臂都前向「行 + 其伙伴变体」** ⇒ 输入序列完全相同，
  **唯一差异是 loss 里哪一项被计入**。
- 核冻结：`requires_grad=False` + `eval()`，selfcheck 断言可训参数只在 `trunk`+`gen`。

### 2.2 三臂 loss（λ 写死，不调）

记 `f(x) ∈ R^40` = 骨架头 logits；`CE` = 40 类交叉熵；
`CE_{{s,t}}(x, y)` = 把 logits **限制到该对的两个 gold** `{s,t}` 后的交叉熵（y ∈ {s,t}）。

| 臂 | loss（写死） | 说明 |
|---|---|---|
| **M** | `CE(f(x_row), y_row) + CE_assign(row)` | **= `struct_supervision` B 臂逐字同式**（既有口径） |
| **MP** | `mean_{有伙伴行} ( mean_{该行伙伴} ½[CE_{{s,t}}(x_row,y) + CE_{{s,t}}(x_var,y')] ) + CE_assign(row)` | **最小对判别式**：判「哪侧对应哪个骨架」；40 类头不变，只在**该对的两个 gold 上做 softmax** |
| **MP+** | `M + λ·MP_pair`，**λ = 1.0（写死）** | 两者相加 |

- **`CE_assign` 三臂逐字相同**（都只在**原始行**上算，伙伴变体不进指派 loss）
  ⇒ 指派监督不构成臂间变量。
- MP 臂**没有** 40 类全类 CE；无伙伴的行对骨架**零梯度**（但仍有指派 loss 经 trunk 回传）。
  这是「最小对监督」的定义性组成，**不是缺陷**，REPORT 必须写明其骨架标签覆盖只有 52.56%。

### 2.3 配方（写死）

- **步数 1800、batch 64、AdamW lr 1e-3 / wd 1e-4、cosine、grad clip 1.0、seed 42/43**
  （与 `two_channel_head` / `struct_supervision` / `bag_modules` 同配方）。
- 编码缓存只写 `experiments/funcword_minpair/cache/`；**不写**任何只读目录。
- 运行：短任务（有明确终点）直接 `uv run python …` 正常等待；
  需要 unit 时用
  `systemd-run --user --unit=dtseek-fwmp* --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 …`；
  **绝不用 `setsid nohup &`**；开跑前查 `ps` 与 `systemctl --user is-active 'dtseek*'`，**忙就排队、不杀他人进程**。
- **随机标签对照（P6，门禁）** `--randlabel`：
  train 骨架标签做**类级双射重标**（在 train 出现的 30 个 id 上 `randperm`，生成器 `seed*1000+13`），
  **行标签与伙伴的 pair gold 同步套用同一 π**（π 是双射 ⇒ 两侧标签仍不同、行标签多重集不变）；
  每行指派值**行内 randperm**（与既有口径同写法）；**评测仍用真标签**；跑前打印 randperm 前后计数。
  **真执行**，不写「计划做」。

---

## 3. 评测出口（**必须无计数泄露**）

### 3.1 集合与指标

| 集 | 用途 | 主指标 |
|---|---|---|
| **held-out 最小对 `b_pairs`** | **主出口（P1）** | **`pair_success`** |
| `a_bal`（1040，26 类，**已排除 L**） | P2 干净出口 | 骨架 acc + 卡−完备地板 |
| `a_lit`（1200 = a_bal + 160 行 L） | **P3 L/非 L 分列** | 骨架 acc（L 行 / 非 L 行分开报） |
| `test`（2500） | P0 复现对账 | 骨架 acc（并报 L/非 L 分列） |
| `c_pairs`（1040） | 诊断 | 骨架 acc（**不进 pair_success 判据**） |

**`pair_success` 定义（写死，= `skeleton_leak/eval_arms.py` 同口径）**：
对 held-out 对 `((x₁,s),(x₂,t))`，`pair_success = #{两侧非受限 40 类 argmax 都等于各自 gold} / n_pairs`。

- **机会水平 = 0.5（任务与 `skeleton_leak` 同口径，写死）**。
  **口径说明（推断，跑前写死）**：特征盲（对内给出相同预测）的模型在两 gold 不同时
  `pair_success = 0`、per-side acc ≤ 0.5；**0.5 因此是保守门槛**。
- **同时必报两个伴随口径**（不改判，只作机制证据）：
  ① `pair_2afc` = 把每侧 logits 限制到该对 `{s,t}` 后**两侧都选对**的比例（独立随机猜 = 0.25）；
  ② `per-side acc`（= `skeleton_leak` 的「成对机会 0.5」比较对象）与其 `t_vs_0.5`。
- **L 分列**：`L = {#35,#0,#1,#2}`（= `n_slots→train多数` 的全部输出，跑前由 train 桶多数复算并断言）。
  在 `test` / `a_lit` / `b_pairs` 上分别报 **L 组 vs 非 L 组**（`a_bal` 构造性不含 L，注明）。

### 3.2 完备免费地图（P5）

- 每格必报 **`max_naive_complete`**（`naive_skeleton` 电池 majority / len_bucket / first_char /
  last_char / punct_pattern / fw_decision_list / tree_depth2 / tree_depth4
  **+ `n_slots→train多数` + 本集多数类**，**全部 fit=train**）
  与 **卡 − 完备地板**；数值直接读 `skeleton_leak/results/leak_stats.json`（同口径复用，实测核对）。
- **另算 pair 地板**：把电池逐规则的**逐行预测**代入 `pair_success`（两侧都要对）取 max，
  记 `max_naive_pair_success`。实现时**断言该预测器给出的逐规则 acc == `naive_skeleton` 输出**（口径校验 fail-closed）。
  预期 `n_slots` 规则在 b_pairs 上两侧 n_slots 相同 ⇒ 同一预测 ⇒ **pair_success ≡ 0**。

### 3.3 机制探针（P4，每臂每 seed 必报）

本模型（= `struct_supervision` B 口径）**没有标签通道** ⇒ `bag_modules` 的
「标签通道打乱 / `aux_only`」原口径**结构上不适用**（如实写，不硬套）。同构替代如下：

| 探针 | 定义（写死） | 期望/用途 |
|---|---|---|
| **`shuffle_content` 只打乱内容** | 对每行取**同 `n_slots` 的另一行的句序槽文本**，用**本行骨架模板**重渲染成新句 ⇒ **保留骨架字面（功能词+标点）与 mask（=n_slots）、只换内容词**；fail-closed 验证 `match_sentence == 本行骨架`，**失败行剔除**，`true` 与 `shuf` **在同一批幸存行上配对比较** | 掉到地板 ⇒ 模型读**内容**；不掉 ⇒ 模型读**结构字面** |
| **`shuffle_all` 打乱全部输入** | 整行输入（`v_sent/v_items/item_mask`，**含 mask**）按 `randperm(seed*1000+99)` 跨行置换，gold 不动 | 应落回**多数类 + 2SE**；否则报「未落回」 |
| **`aux_only` 只看标签通道** | **替代口径**：只吃 `n_slots`（袋项个数）的查表读出（fit=train，≡ `n_slots→train多数` 免费规则）逐格报 | 本模型无标签通道，原 `aux_only` **不适用**；此列即「计数旁路」的量 |
| **`pair_shuffle_content`** | 对 held-out **对**取**同槽数的另一对**的槽文本，**两侧同用**重渲染 ⇒ 保留「只差一个功能词」的对结构，只换内容 | **本任务最关键的机制证据**：`pair_success` 在换内容后是否保持 |

### 3.4 统计口径（写死）

- 行级 acc SE = `sqrt(0.25/n)`；`pair_success` SE = `sqrt(0.25/n_pairs)`；
  槽位 SE = 逐行命中率样本 SE；**臂间差一律配对 SE** = 同一批样本逐 0/1 之差的
  `sd(d)/sqrt(n)`；t = Δ/SE；**2 seed = 42/43，必须报「是否同号」**。
- **不许**把「没测出差异」写成「没有效果」；实测/推断分开标，证据强度标（单 seed / 配对 / 2 seed）。

---

## 4. 验收口径（**跑前写死，判据不改**）

| # | 判据 | 门槛 |
|---|---|---|
| **P0** | 复现既有基线：臂 **M** 的 `test` 骨架 acc ≈ `struct_supervision` B / `bag_modules` A（**.6396 / .6424**），`a_bal` ≈ **.0635 / .0663** | 差值 **≤ 2×SE**（test SE .0100、a_bal SE .0155），两 seed 各自对上；否则 **先修探针**（不改配方去凑） |
| **P1（主）** | **held-out 最小对（`b_pairs` B1，n=520）`pair_success` > 0.5** | **> 0.5 + 2×SE**（SE = sqrt(.25/520) = .0219）**且两 seed 同号** |
| **P2** | `a_bal` 骨架 acc（无计数泄露出口） | 报数 + **卡 − 完备 `max_naive_complete`**（地板 .1404，fit=train） |
| **P3** | **L 组 vs 非 L 组分列**（`L={#35,#0,#1,#2}`） | **必报**：`test` / `a_lit` / `b_pairs` 三处都分列 |
| **P4** | 机制：`shuffle_content` / `shuffle_all` / `aux_only`（+ `pair_shuffle_content`） | **必报**，每臂每 seed |
| **P5** | `max_naive`（**完备**，含 `n_slots`）+ pair 地板 | **必报**，逐格 `卡 − 地板` |
| **P6** | 随机标签负对照（`--randlabel`，**真执行**并打印 randperm 前后计数） | `test` ≤ 多数类 + 2SE（.4368）、`a_bal` ≤ .0695；任一不过 ⇒ **证据不足** |

**判定三选一（写死，不许硬选；seed 只有 42/43、步数固定 1800、λ=1 不调）**：

1. **最小对判别有效（功能词已被表示）** = **P0 过 ∧ P1 过（两 seed 均 > 0.5+2SE 且同号）∧ P6 过**；
2. **无效（仍不读功能词）** = P0 过 ∧ P6 过 ∧ **两 seed 的 P1 主判均未过** ∧
   **per-side acc 也 ≤ 0.5 + 2SE（两 seed 均）** ∧ 两 seed 同判；
3. **证据不足** = 其余任何一种：P0 未过 / P6 未过 / 两 seed 不同号 / P1 一过一不过 /
   关键格子缺数 / **P1 未过但 per-side acc > 0.5+2SE**（存在部分判别信号、主判未达 ⇒ 不许硬判「无效」）。

- **机制解读（跑前写死，防事后找补）**：若 MP 臂 `pair_success` 提升但
  `pair_shuffle_content` **同步塌到地板** ⇒ 只能报「**依赖内容词**」，不得写成「读出功能词」；
  若 `pair_shuffle_content` **保持 > 0.5+2SE** 且 `shuffle_content` 下骨架 acc 不塌
  ⇒ 才支持「**结构字面（功能词）被表示并被使用**」。
- **范围限定（写死）**：任何结论只在「**核冻结、只训头、同一池化接口、1800 步、seed 42/43、
  λ=1**」口径下成立，不外推为「最小对监督在任意设定下有效/无效」。

---

## 5. 自检（开训前必须打印，`model.py --selfcheck`）

1. 核 `requires_grad` 全 False + `training=False`（实测打印），可训参数只有 `trunk`+`gen`；
2. **三臂初值逐位相同**（`torch.equal`）且 == `StructSupModel("B", seed)` 同 seed 初值；
3. **M 臂 loss == `struct_supervision` B 臂 `gen_loss` 逐位相等**（同输入同权重下 `max|Δ|=0`）；
4. **最小对构造自检**：R1/R2/R3 枚举数 = 62/21/1；覆盖行 4274、成功行 4205、失败分类合计一致；
5. **分离断言**：§1.4 的 ①②③④ 全 0（跑前 + 跑后各一次）；
6. `pair` 结构：`b_pairs` 相邻成对、袋相同、骨架不同、B1 槽文本同序同文、B2 槽序相反；
7. 随机标签对照真执行（打印 randperm 前后计数）；
8. 编码缓存路径只落在 `funcword_minpair/cache/`（断言不写只读目录）。

## 6. 未实现（跑前声明，不事后补）

- **同骨架槽序交换对**（`A，B` ↔ `B，A`，骨架两侧相同）**不构造、不训练、不评测** ——
  对骨架 `pair_success` 无信息（见 §1.2 的可构造性结论）；槽序/指派维度的最小对判别**本单元未测**。
- **跨语料新建 held-out 集**：不重扫语料，直接复用 `skeleton_leak` 已断言零重叠的 `b_pairs/c_pairs`
  ⇒ `b_pairs` 只有 17 个骨架、无 n=3/n=4 桶（**跨槽位数的最小对能力未测**，如实报）。
- **训核 / 换特征 / 换核**：一律不做。
- **`dev-notes/`、`methodology/`、`render.py`**：不写、不动。
- **`adv1`（训练零样本 10 类）不评测** —— 三臂构造性为 0，与既有一致。

# PREREG —— P20 核架构对照：从头训练，测「哪个改动真买到结构能力」（`experiments/core_arch/`）

> **本文件 mtime 必须早于本目录任何训练进程的首次启动时刻**；跑后不改判据。
> 改哪条都要在 `REPORT.md` 如实记（含「看到结果之后才改」的非盲修订）。
> 口径：**实测** = 本目录跑出来的数字；**推断** = 由数字推出；**口径判断** = 构造/定义时人为规定。
> **只读**：`experiments/{sentence_mode,entity_identity,skeleton_leak,two_channel_head,core_probe,free_rule_floor,select_pool}/`、`src/`、`checkpoints/`、`training/`、`dev-notes/`、`methodology/`。
> **只允许写**：`experiments/core_arch/`、`logs/`、`/tmp`。
> **禁止** `git commit` / `stash` / `checkout` / `restore` / `clean`；**不训产品、不碰 `src/`、不动 `checkpoints/` 与生产卡、不改既有实验**。

---

## 0. 唯一问题与边界（写死）

**改动核的哪一处，才能真正买到结构能力（顺序 / 功能词 / 槽序）？**

三臂**从头训练**（无预训练核），**唯一变量 = 核的「结构信息通路」**，规模刻意小（只回答方向对不对）。

**背景（既有实测，本单元不重新论证）**：
- 核心诊断五种独立观测（`dev-notes/21 §16.2`）：最小对上模型 **79~85% 不改变答案** ⇒ 读表层、不读结构。
- 机制（`experiments/two_channel_head/model.py:105` `skel_out(h)` 只吃池化特征）：**池化是置换不变的** ⇒ 语序/槽序/功能词位置丢失。
- `experiments/select_pool`：cross-attn 把 heldout 抬 **+5.6pt**，但 `shifted` **掉 25pt** ⇒ **必须双侧判据**。
- 唯一有真增量的一元：`idr` 线性读出 98.83% vs 免费规则 .7918（`core_probe`）。
- 免费规则账已 12 次（`n_slots` .8316 / N1 .8833 / 手写粒子表 .6807）⇒ **每项必须并列地板**。
- `layer_selective`：旧核上「选层/加小适配层」不行；**「换架构从头训」没测过** —— 本单元测它。

**⚠️ 边界句（写死，必须原样进报告）**：本单元只回答**「哪个架构改动买到结构能力」**，
**不回答「该不该换线上核」**（决策留 Lead）。也不回答因果、不回答线上口径迁移。

---

## 1. 三臂（唯一变量 = 核的「结构信息通路」；全部从头随机初始化）

**三臂完全相同的部分（写死）**：
- 字符级词表：`<pad>=0 / <unk>=1 / <m1>=2 / <m2>=3 / <ctx>=4` + 字符按出现序；
  **词表只由「训练池」构建**（见 §2），评测集 OOV 映射 `<unk>`，**报各评测集 OOV 率**。
- token 嵌入 `d=64`（`padding_idx=0`）；
- **双向自注意力编码器 = 2 层 `TransformerEncoderLayer`**（`d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, activation=gelu, batch_first=True`）+ key-padding mask；
  **PyTorch 内置层不带位置编码 ⇒ 位置编码是否加入 = 显式变量**；
- 读出后进同一 trunk：`Linear(in_dim → 64) → GELU → Linear(64 → 64)`；
- 两个任务头（S: 6 类，T: 2 类）；`max_len = 128`（超长截断，**报截断率**）。

| 臂 | 核改动（逐处） | 与 A 的差 |
|---|---|---|
| **A（现行基线）** | 字符级双向编码 → **掩码均值池化 `masked mean`** → trunk → 头；**不加任何位置编码** | — |
| **B（不池化）** | A + **① 位置编码**（`nn.Embedding(128,64)` 加到 token 嵌入）+ **② attention pooling**（单 query：`score_i = vᵀ tanh(W h_i)`，masked softmax，`z = Σ aᵢhᵢ`；`W:64→64, v:64`）→ 同一 trunk → 头 | +位置编码 +池化改 attention |
| **C（结构专用通道）** | B 的全部 + **③ 功能词通道**：固定功能词表 `FUNCWORD`（§1.1 写死）；`f_fw =` 功能词位置的 token 嵌入掩码均值；`f_fp =` **最后一个功能词位置**的位置编码（无功能词则全 0）；`z_c = Linear(128→64)(concat(f_fw, f_fp))`；trunk 输入 = `concat(z_B, z_c)`（128 维） | +B + 显式功能词身份/位置直给头 |

> **口径判断（必须如实写进报告）**：A **没有**位置编码 ⇒ 整核是**输入字符多重集**的函数，
> **严格置换不变**（这是把 `dev-notes/21 §16.2` 的「可证明地丢失」做成不变式的强形式）。
> **线上核 `nano_doc_encoder.py` 有 RoPE** ⇒ A 是「双向编码→池化→头」**拓扑**的复刻，
> **不是线上核的逐位复刻**。因此 **H1 若过，其「机制」部分是构造性的（推断），不是新发现**；
> 真正的经验问题是 **H1 的幅度、H2/H4/H5 的代价**。
> **A 与 B 同时改了「位置编码」与「池化方式」两处 ⇒ 两者各自贡献不可分离（遗留，推断留待后续拆解臂）。**

### 1.1 `FUNCWORD` 词表（跑前写死，不增不减）
```
因为 所以 但是 但 因此 于是 虽然 尽管 而且 并且 如果 要是 只要 除非 总之 可见 由此 结果
而 则 就 才 也 都 还 却 并 且 或 者 之 乎
不 没 别 非 未 无
吗 呢 吧 啊 呀 啦 嘛 哈 难道 岂 何尝
什么 啥 谁 哪 哪儿 哪里 哪些 哪个 怎么 咋 怎样 如何 为什么 为啥 为何 多少 几 何
请 别 不要 不准 不许 马上 赶紧 快点 记得 必须 注意 小心
的 地 得 了 着 过 把 被 从 到 对 和 与 在 给 让 使
```
（来源：`sentence_mode/rules.py::PARTICLE`、`sentence_mode/PREREG §2` 的 `QSET/IMP/RHO`、
`free_rule_floor/rules.py` 逻辑词表、常用虚词；**按字符逐一匹配**。）

---

## 2. 任务与数据（现成数据，不新造；两个任务同一次训练同时学）

**一次训练 = 两个任务的混合批**（三臂同数据、同步数、同优化器、同 seed）：

| 任务 | 角色 | 数据（只读复用） | 输入格式 | 标签 | n_train / n_test |
|---|---|---|---|---|---|
| **Task S（结构敏感，主靶子 H1）** | 句式模式，**遮蔽标点出口** | `experiments/sentence_mode/data/{train,test}.jsonl` | `exit["mask"]`（`render.PUNCT ∪ FINAL_PUNCT` 全遮蔽，逐行零标点断言） | `mode` ∈ 6 类 | **12000 / 6000** |
| **Task T（有真值，防换靶子 H2）** | 实体同一性 | `experiments/entity_identity/data/{train,test}.jsonl` | `<m1>s1<m2>s2<ctx>text` | `label` ∈ {SAME, DIFFERENT} | **4000 / 1200** |

- **词表训练池** = Task S train ∪ Task T train ∪ **`experiments/two_channel_head/data/train.jsonl` 的字符**。
  **口径判断**：第三个来源只用于**字符清单**（不使用其标签、不参与梯度），
  目的是让 H4 的结构 probe 所在域 OOV 可控；须报该来源贡献的字符数与各评测集 OOV 率。
- **Task T 输入格式是口径判断**（数据只给了 `s1/s2/text` 与 span）：把两个目标提及显式送给模型，
  否则模型无从知道要判哪两个提及。字面规则地板同样只看 `(s1,s2)` ⇒ 与 P12a 同口径可比。
- Task S 出口另有 `exit["keep"]`（标点在场）与 `exit["maskfinal"]`，**只删句末标点** —— 用于 H5/H4 辅证。
- **H5 的「shifted/表内」类指标（写死两个）**：① Task S **`keep` 出口** test acc（表层特征齐全、免费规则 = 1.0000）；
  ② 两任务 **train split acc**（表内记忆）。
- **H4 结构 probe 所在域（只读，不训练）**：`experiments/two_channel_head/data/{train,test}.jsonl`（8000/2500，`skel_id` 40 类、train 出现 30 类）与 `experiments/skeleton_leak/data/{b_pairs,c_pairs}.jsonl`（1120 行/560 对、1040 行）。

---

## 3. 配方（跑前写死；三臂逐项相同）

`steps=1200`，`batch=64`（每步 **32 行 Task S + 32 行 Task T**，无放回轮转 + 打乱），
`AdamW(lr=1e-3, wd=1e-4)`，`CosineAnnealingLR(T_max=1200)`，`grad clip=1.0`，`dropout=0.1`，
`loss = CE_S + CE_T`（等权），**seed ∈ {42, 43}**，device=cuda(HSA_OVERRIDE_GFX_VERSION=10.3.0)。

- **先落 PREREG + 代码 + 冒烟（steps ≤ 40）再全量**；每步独立落盘。
- **规模硬要求**：任一单跑 **> ~10 分钟 ⇒ 缩小模型/步数并在报告写明缩了什么**。
- 长跑用 `systemd-run --user --unit=<名> --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1`，**绝不用 `setsid nohup &`**；GPU 忙则排队，**绝不杀他人进程**；一律 `uv run python`。

---

## 4. 免费规则地板（**训练前先算**；门槛跑前写死）

**地板门槛（写死，引用 P12a 已有构造门口径）：`门槛 = 1 − 2SE`，`SE = sqrt(0.25/n_eval)`。**
- `地板 ≥ 门槛` ⇒ **规则近乎满分解题 ⇒ 该任务不可用，换任务或在报告说明**。
- 另并列**多数类基线**与 `多数类 + 2SE`（D2：地板含计数类），逐任务报出，供解释但不作门槛。

| 任务 | 完备免费规则电池（`fit=train`，逐键 → train 多数，未见键回退全局多数） | 门槛 1−2SE |
|---|---|---|
| **Task S / mask 出口** | R0 majority / R1-punct-simple / R2 particle(手写粒子表) / R3 final-char / R4 len-bucket / R5 first-char / **R6 labeldef（标签定义函数在该出口输入上的零拟合执行）**；`max_naive = max(R0..R6)` | n=6000 ⇒ **.9882** |
| **Task S / keep 出口** | 同上 | **1.0000** |
| **Task T** | majority / text-len-bucket / first-char / last-char / `s1==s2` / `s1∈s2` / `s2∈s1` / 编辑距离 ≤ 阈 / 字符 bigram Jaccard ≥ 阈 / LCS-maxlen ≥ 阈 / 共享字比 ≥ 阈（阈值网格 = train∪test 唯一值中点，**train 拟合**） / `variant`（构造产物，**单列披露不入地板**） | n=1200 ⇒ **.9711** |

**对账要求（实测）**：Task S 的 R0/R2/R3/R4/R5 必须与 `experiments/sentence_mode/results/battery.json`
**逐位相等**（同数据同拆同实现），R6 直接读该文件（只读导入 `sentence_mode/labels.py`）；
Task T 的四条字面规则须复现 P12a 公布的 **75.000 / 87.500 / 86.667 / 87.500 %**（容差 ±0.1pt，不一致先查口径）。

---

## 5. 判据（H0–H5，跑前写死；**不许硬选**）

| # | 判据 | 门槛 / 口径 |
|---|---|---|
| **H0** | 三臂**参数量 / 单步耗时 / 总 wall / 峰值显存** | **必报**；参数量相对差 > 10% 须说明来源（哪一层多的） |
| **H1（主）** | **Task S（mask 出口）test acc**：Δ = acc_B − acc_A、acc_C − acc_A，**逐行 0/1 配对** | **余量 > 2×SE_paired 且 2 seed 同号**；SE_paired = sqrt(var(diff)/n) |
| **H2（防换靶）** | **Task T test acc**：B/C ≥ A − 2×SE_paired | 必报；**退化 ⇒ 如实写「拿有真值能力换了结构能力」** |
| **H3** | 每任务**并列免费规则地板**（§4） | `地板 ≥ 1−2SE ⇒ 该任务不可用，须说明`；地板 < 门槛 ⇒ 该任务可用，但**地板仍必须逐格并列** |
| **H4（表示诊断）** | **线性 probe acc**（核冻结，简化 `core_probe` 口径：取臂的池化表示 `encode(ids)` → 按 probe-train 均值/方差标准化 → **L2 正则多项式逻辑回归**；环境无 `sklearn` ⇒ 用 `torch.optim.LBFGS`（lr 1.0, max_iter 200），目标 = mean CE + 0.5·λ·‖W‖²，`λ = 1/(C·n_probe_train)`，`C ∈ {0.01,0.1,1,10}` 按 probe-train 内部 80/20（seed 42）选，选完在全 probe-train 重拟合）。**结构信息** = ① `skel_id`（`two_channel_head` 拆 8000/2500）② **Task S `trig` 触发词起点分桶**（`∅`=0，否则 `1+min(7, start//8)`，9 类，**该标签从未参与训练**）；**有真值信息** = Task T SAME/DIFFERENT（4000/1200） | **逐臂报**；并列 `max_naive`（skel 完备电池 **.8316**、Task S trig 的多数类、Task T 地板 §4）；**这是「结构是否进了表示」的直接证据**；trig 位置对 A 是**置换不可学**的 ⇒ A 应落在多数类上 |
| **H4b** | probe 在 `skeleton_leak/b_pairs`（560 对）上的 **pair_success**（对内预测是否改变）+ 在 `c_pairs` 的 per-side acc | 必报；参照：**成对机会 .5**、置换不变规则 **0.0000**、字面骨架规则 **1.0000**（口径判断） |
| **H5** | **表内/表层**：Task S `keep` 出口 test acc + 两任务 train acc | 必报；**任一项 B/C 比 A 掉 > 2×SE_paired ⇒ 判「代价不可接受」的必要条件之一**（防 `select_pool` 的 +5.6/−25 输法） |
| **M（机制自检）** | 逐行**随机置换非 pad 前缀**的 token id（`seed*1000+11`，Task S mask 出口；**字符多重集逐位不变**）后重评 | **A 必须：预测一致率 ≥ 0.999 ∧ \|acc 差\| ≤ 0.002**（浮点求和次序容差；断言不过即停）；B/C 的 acc 差与一致率**报出不设门槛** —— 用于证明「唯一变量确实改变了结构通路」 |

**三选一判定（写死，不许硬选）**：
1. **改架构买到了结构能力** = H1 过 ∧ H2 不塌；
2. **没买到** = H1 不过；
3. **买到了但代价不可接受** = H1 过 ∧ H2 塌（或 H5 任一表内项掉 > 2SE）。

**铁律**：并列 `max_naive`；区分**实测/推断**；**不许**把「没测出差异」写成「没有效果」。

---

## 6. 产物与停止边界

产物：`PREREG.md`、`data.py / rules_floor.py / model_core.py / train.py / probe.py / summarize.py`、
`results/*.json`、`weights/*.pt`、`logs/*.log`、`REPORT.md`。

**做完 A/B/C 三臂 + 判据 + 表示诊断就停。**
**不要**：训产品级模型、改 `src/`、动生产卡、写 `dev-notes/` / `methodology/`、派子代理、git 提交类操作。

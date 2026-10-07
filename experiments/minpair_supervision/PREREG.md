# PREREG —— P21 最小对当监督 + 2AFC 口径（`experiments/minpair_supervision/`）

> **本文在任何训练进程启动之前写死**（mtime < 首个 train 启动时刻），跑完不改；结果填 `REPORT.md`。
> 「实测」= 本目录跑出的数字；「推断」= 由数字推出；「口径判断」= 定义时人为规定的归属。
> **只读**：`experiments/{skeleton_leak,funcword_minpair,core_arch,free_rule_floor,two_channel_head,struct_supervision}/`、`src/`、`training/`、`checkpoints/`、`dev-notes/`、`methodology/`。
> **只写**：`experiments/minpair_supervision/`、`logs/`、`/tmp`。禁 git commit/stash/checkout/restore/clean。不派子代理。

## 0. 唯一问题（写死）

**给定同一冻结核、只训头、同配方，把骨架监督从常规多类（M）换成配对判别（MP）、两者相加（MP+）
——能否让模型在 held-out 最小对的「配对口径」上显著超过机会？**

- **边界句（写死）**：本单元**只回答**「配对监督能否在配对口径上超机会」，
  **不回答**「该不该改线上核/监督」（决策留 Lead）；也不回答「功能词在核里有没有」（P20 已答）。
- 背景实测（不重论证）：P4 的 `pair_2afc` 报 MP .6212/.6096，但 Lead 抽查 `eval_MP+_s42.json` 得
  b1 .5404 / b2 .1250 / c_natural .6058 ⇒ **「机会 .25」的口径与实测零假设均未钉死** ⇒ 本单元先把口径钉死。
- **铁律**：并列报两个口径；区分实测/推断；不许把「没测出差异」写成「没有效果」。

## 1. 2AFC 定义（**跑前写死，原文照抄**）

对 held-out 对 `((x_A,s),(x_B,t))`，`l(x) ∈ R^40` = 骨架头 logits，`s ≠ t`（跑前断言）：

```
pair_2afc = #{ i : argmax(l_A[s], l_A[t]) = s  ∧  argmax(l_B[s], l_B[t]) = t } / n_pairs
```

- **两侧都判对（各自在该对两个 gold 的受限 2 类比较中选中自己的 gold）**；并列取靠前索引（torch argmax 语义）。
- **n** = held-out 对数（见 §2）；**配对方式** = `b_pairs.jsonl` 行内相邻成对（`kind` 以 `_src` 开头的行 i 与 i+1），
  **袋多重集相同（槽序可不同，b2 即槽序交换）**、骨架不同（跑前断言两侧袋多重集相同、`skel_id` 不同、两侧句面不同）。
- **与 40 类 argmax 口径并列**（M2，同一批对上同时算）：
  `pair_success_40 = #{ argmax_40(l_A)=s ∧ argmax_40(l_B)=t } / n`（= P4/`skeleton_leak` 主口径）；
  另报 `per_side_acc`（2n 侧）。
- **两侧泄漏防护（跑前写死，fail-closed）**：train / held-out **按对整体切**，绝不对侧切开；
  切分见 §2；任一断言失败即停并如实记 REPORT。

### 1.1 「机会」是什么（M0，**必须实测，不许直接引用 .25/.5**）

| 记号 | 定义（跑前写死） | 期望 |
|---|---|---|
| **机会_R 独立均匀随机猜** | 每对每侧独立抽 `c ~ Uniform{s,t}`，判两侧都中；**Monte-Carlo 2000 轮 × n 对，seed 20261007**，报 **均值 ± SE** | ≈ .25 |
| **机会_P 置换零模型** | 用**该臂本臂实测 logits**：对 pair 下标做随机置换 σ（**1000 次，seed 20261007**），用第 σ(i) 对的两侧 logits 去算第 i 对的 gold（同侧对同侧）⇒ 打断「预测↔gold」对齐但保留模型的**侧间相关性与类偏好**；报均值 ± SE | 通常 ≠ .25 |
| **盲/常值预测器** | 两侧同 logits（或同一常类）⇒ 受限 argmax 两侧同侧 ⇒ **恒 0**（用同一代码路径实测确认） | 0 |
| **M4 随机标签臂** | 训练期类级双射重标，**评测用真标签** ⇒ 实测 2AFC（见 §5） | ≈ 机会_P |

**M1 门槛（写死）**：`门槛 = max(机会_R + 2·SE_R, 机会_P + 2·SE_P)`，
对 held-out `b_all`（n=280）计算；**MP 或 MP+ 两 seed 均 > 门槛 且同号** 才算过。

## 2. 数据与切分（**只读复用 `skeleton_leak/data/b_pairs.jsonl`，不重造数据**）

- 全量 1120 行 = **560 对**（`b1` 520 对 / `b2` 40 对；28 个方向 × 20；n_slots ∈ {1,2}）；
  行内相邻成对；**1120 句面全唯一、560 对袋元组全唯一**（跑前断言，实测已核）。
- **分层切分（跑前写死、确定性、与 seed 无关）**：每个**方向**（`skel_id_A, skel_id_B, kind`）内 20 对
  按 `md5("P21:" + 源行 sent)` 升序，**前 10 → TRAIN，后 10 → HELDOUT**。
  ⇒ TRAIN 560 行 / 280 对；HELDOUT 560 行 / 280 对（b1 260 / b2 20），**28 个方向两侧都在**。
- **分离断言（跑前 + 跑后各一次，fail-closed）**：
  ① TRAIN 句面 ∩ HELDOUT 句面 = 0；② 每对两侧必同 split；③ TRAIN 袋元组 ∩ HELDOUT 袋元组 = 0；
  ④ 每方向 TRAIN=10、HELDOUT=10；⑤ TRAIN 与 HELDOUT 出现的 `skel_id` 集合相同。
- 编码**只读复用** `experiments/skeleton_leak/cache/enc_58b315d8f6ba_L64.pt`（n=1120，实测命中）；
  本目录 `cache/` 只放**探针新句**的编码。**不写任何只读目录。**
- 对账用全量 560 对（**含训练半，in-sample**）另报一行，**仅供与 P4 对账、不作判据**。

## 3. 三臂与配方（**唯一变量 = 骨架监督形式**）

- 模型 = 只读复用 `funcword_minpair/model.py::StructSupModel("B", seed, ...)`（= `struct_supervision` B 口径）；
  核 1,688,460 参数 `requires_grad=False` + `eval()`，**只训 trunk+gen**（selfcheck 打印）；
  三臂初值逐位 `torch.equal`（selfcheck 断言）。
- 记 `CE40` = 40 类交叉熵；`CE{s,t}` = logits 限制到该对两个 gold 后的 2 类交叉熵（P4 `pair_ce` 同式）；
  `CE_assign` = 指派项（与 `struct_supervision::gen_loss` 逐字同式），**三臂逐字相同**、不构成臂间变量。

| 臂 | loss（λ 写死） |
|---|---|
| **M** | `mean_row CE40 + CE_assign`（= 既有口径） |
| **MP** | `mean_pair ½[CE{s,t}(A) + CE{s,t}(B)] + CE_assign`（配对判别；**无 40 类全类 CE**） |
| **MP+** | `mean_row CE40 + 1.0 × mean_pair ½[CE{s,T}+CE{s,t}(B)] + CE_assign`，**λ = 1.0** |

- **batch = 32 对 = 64 行**（行数/步与既有口径一致）；`DataLoader` over pair 下标、
  `shuffle=True, generator=manual_seed(seed)` ⇒ 同 seed 三臂 batch 序逐位相同。
- **配方（沿用既有口径，写死）**：**1800 步**、lr 1e-3、AdamW wd 1e-4、cosine、grad clip 1.0、
  **seed 42/43**；训练行 560 ⇒ 约 206 epoch（数据极小，如实写明）。
- 运行：纯 CPU 优先（特征已缓存，只训头）；`uv run python`；长跑用
  `systemd-run --user --unit=dtseek-p21* --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 …`，
  **绝不用 `setsid nohup &`**；开跑前查 `systemctl --user is-active 'dtseek*'`，忙就排队、**不杀他人进程**。

## 4. 评测出口与地板（M2/M3）

- **主出口** = HELDOUT 对（b_all 280 / b1 260 / b2 20），**两个口径同批算**：
  2AFC（§1 主判）+ 40 类 argmax `pair_success`（M2 并列）。b2 n=20 只作诊断、不进判据。
- **地板（必报，逐格）**：
  1. **成对机会基线** = §1.1 的 **机会_R / 机会_P（实测）**；
  2. **`R-chain`（字面规则）** ⇒ 本任务对应物 = 功能词**字面决策链** `fw_decision_list`
     （P4 电池同式、**fit = `two_channel_head/data/train.jsonl` 既有口径**；
     `syllogism_card` 的字符串链 `R-chain` 属三段论任务、**本任务不适用**，如实注明）；
     同时报全电池 `max_naive_complete`（含 `n_slots`，与 P4 同口径对账）；
     **每个规则同时报 `pair_success_40` 与 2AFC**（2AFC 走同一代码路径：规则预测 → one-hot → 受限 argmax）；
  3. **`n_slots` 规则**（fit = `skeleton_leak/results/leak_stats.json::fit`：`{1:35,2:0,3:1,4:2}`，
     train 多数 = 0）在 held-out 对上：**L 组 / 非 L 组分列**，`L = {#35,#0,#1,#2}`（跑前断言 = n_slots 规则的全部输出）。
- **`aux_only`**：本模型（struct B 口径）**无标签通道 ⇒ 原口径不适用**（如实写）；替代 = 上面的 `n_slots` 查表列。

## 5. 机制探针（M5）与随机标签对照（M4）

| 探针 | 定义（跑前写死） |
|---|---|
| **`shuffle_content`（pair 版）** | held-out 对取**同 `n_slots` 的另一对**的句序槽文本，**两侧同用**灌进本对两侧骨架模板 ⇒ 保留「只差一个功能词」的对结构、只换内容；fail-closed（`ok_sentence` ∧ `match_sentence` 返回本骨架 ∧ 槽文本与 donor 一致），失败对剔除；**true 与 shuf 在同一批幸存对上配对比较**（2AFC 与 `pair_success_40` 都报） |
| **`shuffle_all`** | held-out 行输入（`v_sent/v_items/item_mask`）按 `randperm(seed*1000+99)` 跨行置换，gold 不动 ⇒ 应落回机会（否则报「未落回」） |
| **`aux_only`** | 不适用（见 §4）；替代 = `n_slots` 查表列 |
| **M4 `--randlabel`** | 类级双射重标（TRAIN 出现的骨架 id 上 `randperm`，生成器 `seed*1000+13`），**行 gold 与 pair gold 同步套用同一 π**（π 双射 ⇒ 两侧仍不同）；每行 assign 值行内 randperm；**评测用真标签**；**真执行**并打印 randperm 前后计数。**M/MP/MP+ × 两 seed 全跑** |

## 6. 验收口径（跑前写死，判据不改）

| # | 判据 | 门槛 |
|---|---|---|
| **M0** | 2AFC 的「机会」**实测值**（机会_R MC + 机会_P 置换 + 盲预测器） | **必须实测**，不许引用 .25/.5 |
| **M1（主）** | held-out 2AFC：**MP 或 MP+ > 门槛**（= max(机会_R+2SE_R, 机会_P+2SE_P)） | **两 seed 同号且均过** |
| **M2** | 40 类 argmax `pair_success_40` 并列（与 P4 对账，含 in-sample 全量行） | 必报；**若与 M1 结论相反 ⇒ 明确写出并解释，不许硬选** |
| **M3** | 地板：成对机会（实测）/ `R-chain` 对应物（字面决策链）/ `n_slots`（L 组·非 L 组分列） | 必报，逐格 `卡 − 地板` |
| **M4** | 随机标签对照（三臂 × 两 seed，真执行） | 2AFC ≤ 机会_R − 2SE_R，且**显著低于对应真臂**（配对 SE） |
| **M5** | 机制：`shuffle_content` / `shuffle_all`（+ `aux_only` 适用性） | 必报，每臂每 seed |

**判定三选一（写死，不许硬选）**：
1. **配对监督在配对口径上有效** = M0 有实测 ∧ **M1 过** ∧ M2/M3/M4/M5 全有数 ∧ M4 过；
2. **无效** = M0 有实测 ∧ M4 过 ∧ **MP 与 MP+ 两 seed 的 M1 均未过**（如实写「未过门槛」，不写「没有效果」）；
3. **两口径矛盾需澄清** = **M1 过而 M2 未过（或反之）** ⇒ 写明矛盾并解释可能机制，不硬选。

- **机制解读（跑前写死）**：若 MP/MP+ 的 2AFC 提升在 `shuffle_content` 下**同步塌到机会**⇒ 只能报「依赖内容词」；
  若 `shuffle_content` 后仍 > 门槛 ⇒ 支持「结构字面被表示并被使用」。
- **范围限定**：结论只在「核冻结、只训头、b_pairs 切半、1800 步、seed 42/43、λ=1」口径下成立，不外推。

## 7. 自检（开训前打印）

1. 核 `requires_grad` 全 False + `eval()`；可训参数只有 trunk+gen；
2. 三臂初值逐位 `torch.equal`（同 seed）；
3. §2 切分五断言全过（跑前 + 跑后各一次）；
4. 40 类 CE 与 P4/struct 同式（`pair_ce` 与 `funcword_minpair/model.py` 同函数复用）；
5. 随机标签对照真执行（打印 randperm 前后计数）；
6. 本目录只写 `cache/` `results/` `weights/` `logs/`，不写只读目录。

## 8. 未实现（跑前声明，不事后补）

- **不重造数据**、不新建语料扫描、不训核、不改 `src/`、不做免费规则穷举、不写 `dev-notes/`/`methodology/`。
- `b2` 仅 n=20 ⇒ **槽序方向的能力不作判据**（只报数）。
- `c_pairs`（两侧袋不同，非最小对）**不进任何判据**，只作诊断列（如报只报数字不参与判定）。

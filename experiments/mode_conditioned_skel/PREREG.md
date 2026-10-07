# PREREG —— P8 句式模式条件化能否救「选骨架」（`experiments/mode_conditioned_skel/`）

写死时间：2026-10-07 11:4x（`mtime` 早于**首次训练**（句式探针微调）；本文件跑前定稿，跑后不改，改哪条都要在 REPORT 如实记）。

**唯一问题**：先用「语句模式」（陈述/疑问/祈使/感叹/反问）把 40 类骨架缩到与句式一致的子集，
**能否显著抬高「选骨架」准确率**？唯一变量 = **推理时是否条件化**（同模型、同权重、同 seed）。

**⚠️ 边界（写死）**：本单元只回答「条件化在两个出口上抬多少、是否超过免费查表」；
**不动核、不改既有实验、不改 `src/`、不重训四臂**；不回答「模型为什么不读功能词」（既有实测已给）。

---

## 0. 背景（既有实测，本单元不重跑）

- `experiments/skeleton_leak/`（**无计数泄露出口 a_bal**，26 类，多数类 .0385，SE=.0155）：
  骨架 acc A **.0635/.0663**、U **.1308/.1298**、P **.0837/.0846**、UP **.1567/.1587**；
  污染出口 test：A .6396/.6424、U .8328/.8340、P .7708/.7764、UP .8416/.8440；adv2：.2310/.2390、.3490/.3540、.3160/.3100、.3800/.3760。
  最小对 `pair_success` ≤2.14%、对内预测 79~85% 不变 ⇒ 模型不读功能词；`L={#35,#0,#1,#2}` 仅凭计数可预测 ⇒ **评测必须 L 分列**。
- `free_rule_floor/`：完备免费地图 test **.8316** / adv2 **.336**（`n_slots→train多数`，零训练）；
  **免费规则账已 6 次** ⇒ 任何增益必须先与平凡查表比。
- `bag_modules/`：只打乱内容 U 仅掉 .0056~.018；随机标签 4/4 ≥ 真标签 ⇒ 残差非语义。
- `render.py` / `build_gen_data.py` 的骨架表**本身含句式**（`[1]，[2]吗` 疑问式等）⇒ 句式是骨架的天然划分维度。
- 本单元**只读** `experiments/{skeleton_leak,free_rule_floor,bag_modules,struct_supervision,two_channel_head,sentence_mode,additivity}/`、`src/`、`training/`、`checkpoints/`、`dev-notes/`、`methodology/`；
  **只写** `experiments/mode_conditioned_skel/`、`logs/`、`/tmp`；禁 git commit/stash/checkout/restore/clean。

## 1. 句式标签（⚠️ `experiments/sentence_mode/` 并行单元产物**尚未落盘** ⇒ 本单元**自建规则标签**，报告必须写明「规则标签」并**并联报它的 `rule`**）

**模式集合（写死）** `MODES = {陈述, 疑问, 祈使, 感叹, 反问}`。

### 1.1 映射 A（规则·骨架字面）—— 按骨架 pattern 字面，程序化分类，优先级自上而下
1. 字面含 `难道/岂/莫非/不是…吗` 任一反问标记 ⇒ **反问**；
2. 字面以 `！` 或 `!` 结尾（或含 `！`）⇒ **感叹**；
3. 字面含 `吗` 或 `么` ⇒ **疑问**；4. 字面含 `呢` ⇒ **疑问**；
5. 字面含 `吧` ⇒ **祈使**；6. 其余 ⇒ **陈述**。
**预期（表内实测事实，非结果）**：40 类表中无 `！` 骨架、无反问标记骨架 ⇒ **感叹/反问子集为空**；
真正可分的只有 **陈述 / 疑问 / 祈使** 三类 —— 这是**表结构事实**，必须在报告里写明。

### 1.2 行级句式标签 g(sent)（规则·输入侧，推理时可得）
取 `sent.rstrip()`，剥掉尾部句末标点/空白 `。．.…～~ ）)」』"”` 后看最后一个字：
`吗/么/呢` ⇒ **疑问**（若句中同时含 `难道/岂/莫非` ⇒ **反问**）；`吧` ⇒ **祈使**；`！/!` ⇒ **感叹**；否则 **陈述**。
**掩码等价类（写死）**：反问 **并入疑问**的子集；感叹子集为空 ⇒ **不掩码（回退 F）并计数报出**。
- **C-rule（oracle）** 的句式 = `A(gold 骨架)`（要 gold ⇒ 上界）；
- **C-pred（可部署）** 的句式 = `g(sent)`（**规则分类器**，本单元主 C-pred）；
- 并联 **C-pred-ML** = 线性探针（见 §3）；`g` 与探针的**准确率都必报（N4）**，对照金标 = `A(gold 骨架)`。

### 1.3 映射 B（学习/统计·train）—— 取覆盖 ≥ **X = 0.90** 的最小子集
在 **train** 上按 `g(sent)` 的掩码等价类分组，统计组内 **gold 骨架**分布，按频次降序（平手取 id 升序）
取最短前缀使其累计覆盖 **≥ 0.90** ⇒ `S^B_m`。
**必报**：逐骨架的 A/B 归属、A/B 一致率（B 未覆盖的骨架单列）、两套子集大小分布、B 在各出口的 **gold 覆盖率**。

## 2. 对照臂（唯一变量 = 是否条件化；同模型/同步数/同 seed 42/43）

| 臂 | 推理 |
|---|---|
| **F** | 全 40 类 argmax（= 既有口径，**必须复现既有数字**）|
| **C-rule** | 掩码到 `S_{A(gold)}`（**oracle 上界**，N3）|
| **C-pred** | 掩码到 `S_{g(sent)}`（规则分类器，主判据）|
| **C-pred-ML** | 掩码到 `S_{探针argmax}`（并联）|
| **C-soft** | `score[s] = logit[s] + λ·log p(A(s)\|x)`，**λ=1 写死**（探针后验，不硬切）；另报 **C-soft-lr**（减 train 先验的似然比版，λ=1）作诊断 |

- **主模型 = UP**（既有最高 F）；A/U/P 报 F（N0）与 a_bal 上的 C-rule/C-pred 摘要。
- 子集映射 **A 为主表**，**B 单独报**（分开报，不混）。
- **两个出口**：**污染出口 test / adv2**（对账用）与 **无泄露出口 a_bal（真判据）**；
  test/adv2 按 **L 组（gold ∈ {35,0,1,2}）/ 非 L 组**分列；a_bal **不含 L**（n_L=0 如实报）。

## 3. 句式探针（本单元唯一的「训练」，核冻结、只训头）

- 输入 = 冻结核 `v_sent`（与 eval 同编码口径，缓存只写本目录 `cache/`）；结构 = **`Linear(128→5)` 无隐藏层**；
  目标 = `A(gold 骨架)`（5 类）；**普通 CE、不加类权重**（写死：加类权重会系统性误伤占 95% 的陈述类，代价已知，如实报混淆矩阵）；
  Adam lr=1e-2、bs=256、≤60 epoch、train 内 10% 做 val、**早停 patience=10**；**init seed 42/43 两个**都训、都报；
  训练数据 = train 8000 行，**不碰 test/adv2/a_bal 的标签**（只在 eval 期用其标签算指标）。
- 报：acc、macro-F1、逐模式召回、混淆矩阵、**2 init seed 的均值±SE**。

## 4. `max_naive`（N2/N6 的关键对照，**必报**）

- **既有完备电池逐字复用** `two_channel_head/build_gen_data.py::naive_skeleton`（8 条，fit=train）
  + `n_slots→train多数` + **本集多数类** ⇒ `max_naive_complete`（与 skeleton_leak 同口径）。
- **新增条件化规则（计入 max_naive，跑前写死）**：
  **R1 = `(g(sent), n_slots) → train 多数骨架`（N2 主对照）**、R2 = `g(sent) → train 多数`、
  R3 = `(A(gold), n_slots)`（oracle 版，单列）、R4 = `S^B_{g}` 的 train 多数（映射 B 版，单列）。
  查表口径 = `free_rule_floor/rules.py::lookup`：**无 min_support**、未见键回退 train 全局多数、报 `seen_rate`。
- `max_naive_plus = max(完备电池 ∪ {R1, R2})`；**每格报 卡 − max_naive_plus**（铁律：并列 `max_naive`）。

## 5. 验收口径（**跑前写死，判据不改**）

SE 口径同 skeleton_leak：行级 acc SE = `sqrt(0.25/n)`；配对 Δ SE = 逐行 0/1 差的样本 SE；t = Δ/SE_pair；**2 seed = 42/43**。

| # | 判据 | 门槛 |
|---|---|---|
| **N0（前置）** | **F 复现既有数字**（A/U/P/UP × 42/43 × {test, adv2, a_bal}） | \|Δ\| ≤ 对应 SE，否则 **fail-closed 停，先修探针** |
| **N1（主）** | 无泄露出口 a_bal 上 **C-pred − F** | 配对 **> 2×SE_pair**，2 seed 同号，且 **C-pred-ML 同向**（C-rule 上界必报；污染出口分 L/非L 报） |
| **N2（关键对照）** | 卡 − **条件化免费查表 R1** | 逐出口报数；**≤0 ⇒ 收益被免费规则解释**；并报 Δ_model(=C−F) vs Δ_rule(=R1−n_slots规则) |
| **N3** | 子集大小（A/B）+ oracle 上界 C-rule | 必报「降维多少 + oracle 能到多少」（区分方法不行 vs 上界低） |
| **N4** | 句式分类器自身 acc | 规则 `g` 与探针**都报**（acc + macro-F1 + 混淆） |
| **N5** | 随机标签对照 + 老卡 Δ | 主门槛：**行级 mode randperm（seed*1000+99）后 acc ≤ F + 2SE_pair 且 ≤ C-rule − 2SE_pair**；**并列报** 是否 ≤ 多数类+2SE（不过则如实报"未落回"并诊断）；次级：骨架→模式随机置换（保子集大小）同报；**四张老卡 Δ ≥ −各自带**（pronoun 2.83 / sentiment 0.41 / relation 1.39 / person 0.33 pt） |
| **N6** | `max_naive`（**含条件化查表**） | 逐格必报 |

**判定（三选一，不许硬选）**
- **句式条件化有效** = N1 过 ∧ N2 > 0；
- **无效（或收益被免费规则解释）** = N1 未过，或 N1 过但 N2 ≤ 0；
- **证据不足** = N0 未过（探针没复现）∨ 两出口/两 seed 方向不一致 ∨ 分类器 acc 不可解释且 C-pred 与 C-rule 差距无诊断。
- **不许**把"没测出差异"写成"没有效果"；**实测/推断分开标**，证据强度（单次/配对/2 seed）标注。

## 6. 运行与写盘（写死）

- 全部 `uv run python`；有明确终点的一次性批处理按 `AGENTS.md` 正常等待；
  **探针训练**若超过 1 分钟改用
  `systemd-run --user --unit=dtseek-modskel* --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 ...`；
  **绝不用 `setsid nohup &`**；开跑前查 `systemctl --user list-units 'dtseek*'` 与 `ps`，**GPU 忙就排队、绝不杀他人进程**。
- 权重**只读** `experiments/bag_modules/weights/{A,U,P,UP}_s{42,43}.pt`（`strict=False`，断言缺键只许 `encoder.*`）；
  骨架编码优先**只读复用** `experiments/skeleton_leak/cache/enc_*.pt`（fp 同式），miss 则自编码写本目录 `cache/`。
- **老卡门禁（N5 下半）**：本单元**不写任何 checkpoint** ⇒ 老卡 Δ 的证据 = `checkpoints/base_encoder.pt` 与 `checkpoints/cards/*` 的 **sha256 训前/训后逐位相同** + 单列一次只读老卡评测（若时间允许，写入**本目录**，不写 `two_channel_head/`）。
- 冒烟：`--smoke`（train 200 行 / a_bal 100 行）先跑通，退出码 0 才跑全量。

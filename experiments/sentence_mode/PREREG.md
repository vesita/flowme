# PREREG —— P7 语句模式卡（输入侧）+ 标点遮蔽（`experiments/sentence_mode/`）

**本文件在任何数据构建 / 规则计算 / 训练之前写死**（`mtime` 早于首次训练进程启动时刻）。
跑完不改；改哪条都要在 `REPORT.md` 如实记。口径：**实测** = 本目录跑出来的数字；
**推断** = 由数字推出；**口径判断** = 构造时人为规定的标签/规则归属。

**修订记录（透明）**：
- **rev.A**（2026-10-07 03:5xZ，`sm7-build` 已启动但**训练未开始、本单元任何 `rule`/`acc` 数字都还没看到**）：
  M0 的幅度门槛由「`R-punct` 落差 ≥0.30」改为「**恒等式断言为主 + 幅度只报不判**」——
  原门槛把"遮蔽是否生效"（结构问题）与"词表规则是否仍强"（地板问题）混成了一个数，
  且对 `R-punct(simple)` 与 `R-punct(full)` 两个口径会给出相反结论。**判定逻辑未变**（M0 未过 ⇒ 停）。
- **rev.B**（2026-10-07 04:2xZ，**在看完 800 步全量结果之后、跑收敛档之前**写，**非盲**，如实登记）：
  800 步档的 `train_mode_acc_last100 = 0.6153/0.6061`（MASK 臂）⇒ **训练侧未收敛**（`dev-notes/21` D6：
  不得拿未收敛跑当结论）。**补一条收敛判据与敏感性档**：若 800 步档 train acc < 0.90，
  则补跑 `steps=8000` 的 **收敛档（敏感性，单列）**，用于区分「输入侧不可读」与「配方未收敛」。
  **主判据仍用预注册 800 步档**；收敛档**不替换**主表，只在 REPORT 里单列并说明它是否推翻主判定。
  本条是**看到 800 步 eval 数字后加的**，不是盲改，报告必须写明。

---

## 0. 唯一问题

**把句末标点遮掉之后，模型还能不能读出话轮的语句模式（陈述 / 是非疑问 / 特指疑问 / 祈使 / 感叹 / 反问）？
以及它是否比一张"平凡查表"更强？**

设计动机（既有实测，本单元不重跑，见 `dev-notes/19` / `dev-notes/21`）：
用户话轮 **n=300,004**、p50 **20 字**、**仅 22.6% 有句末标点**、含逗号 40.2%
⇒ 「看标点判句式」这条免费规则在 **77.4% 的真实输入上根本没有输入可看**
⇒ 主出口必须是**遮蔽标点**，模型只能读**词序 / 语气词 / 疑问词**。
免费规则吃掉增益已达 **6 次**（`dev-notes/21` §3）⇒ **任何新机制必须先与平凡查表比**。

**⚠️ 边界（写死）**：本单元回答「输入侧句式可读性 + 是否超地板」；
**不回答**该模式能否改善别的任务（骨架选择 / 回复），**不回答因果**。

---

## 1. 硬约束（写死）

1. **只读** `src/`（`render.py` 可 import）、`experiments/{card_flow,syllogism_card,free_rule_floor,
   skeleton_leak,core_probe,label_construct_validity}/`、`training/`、`checkpoints/`、`dev-notes/`、`methodology/`；
   **只允许写** `experiments/sentence_mode/`、`logs/`、`/tmp`。
2. **禁止** `git commit` / `stash` / `checkout` / `restore` / `clean`。
3. **不动共享核**：`checkpoints/base_encoder.pt` 与四张老卡**只读**；
   核 `requires_grad=False` + `eval()`，**只训新头**（本单元**不需要**动核 ⇒ 不申请动核豁免）。
4. **语料**：只走 `dtseek.tasks.corpus.resolve_corpus_files(CORPUS_GLOB)`；
   文件级汉字占比 ≥0.6 过滤、**打乱文件序**（SPLIT_SEED），**不依赖 `fs` 顺序、不写 `fs[:N]`**；
   报告必须列出**文件清单 + 汉字占比**（`dev-notes/19` 硬要求）。
5. 长跑用 `systemd-run --user --unit=<名> --collect --property=WorkingDirectory=<repo>
   --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1`；
   **不用 `setsid nohup &`**；GPU 忙则排队，**绝不杀他人进程**；一律 `uv run python`。
6. 先落 PREREG + 代码 + **冒烟**（steps ≤ 40）再跑全量；每步独立落盘。
7. 区分**实测 / 推断**；标证据强度；**不许**把"没测出差异"写成"没有效果"；**绝不允许空报告**。

---

## 2. 标签（六类）与来源（跑前写死）

**标签池 = 有句末标点的用户话轮**（`dev-notes/19` 的 22.6% 那部分）。
句末标点集 `FINAL_PUNCT = 。．.？?！!`（取**去尾空白后的末字**）；
**遮蔽集 `MASK_SET = render.PUNCT ∪ FINAL_PUNCT`**（`render.PUNCT` 未收 ASCII `.` 与 `．`，显式补上并登记）。

**标签函数 `label_full(s)` —— 按下面的优先级，逐条判定（顺序即优先级）**：

| 序 | 条件 | 类 | n 记号 |
|---:|---|---|---|
| 1 | 含 **反问标记** `RHO = {难道, 难道说, 谁说, 不是吗, 不是…吗(同句含"不是"且含"吗"), 岂, 何尝}` | **反问** | n_rho |
| 2 | 末字 ∈ {？?} 且 含 **疑问词** `QSET = {什么, 啥, 谁, 哪, 哪儿, 哪里, 哪些, 哪个, 怎么, 咋, 怎样, 如何, 为什么, 为啥, 为何, 多少, 几点, 几个, 几时, 多久, 何}` | **特指疑问** | n_spec |
| 3 | 末字 ∈ {？?} | **是非疑问** | n_yn |
| 4 | 末字 ∈ {！!} 且 含 **祈使标记** `IMP = {请, 别, 不要, 不准, 不许, 马上, 赶紧, 快点, 给我, 闭嘴, 停下, 等一下, 记得, 必须, 注意, 小心, 住手, 站起来, 走开, 滚, 加油, 试试}` | **祈使** | n_imp |
| 5 | 末字 ∈ {！!} | **感叹** | n_exc |
| 6 | 末字 ∈ {。．.} | **陈述** | n_dec |
| — | 否则 | **丢弃** | n_drop |

**丢弃率 = n_drop / 标签池候选数**，逐原因报（无句末标点 / 末字是逗号顿号等）。

**构造性 `rule`（逐类，跑前声明）**：
- **按定义 = 1.0000**：标签就是 `label_full` 的输出 ⇒ 在**原文**上 `rule=1.0` 是**定义**不是发现
  （`dev-notes/21` D4：这不构成任何构念证据）。
- **逐类来源独立性（更正 C 的"来源独立"一半）**：

| 类 | 顶部分支来源 | 细分来源 | 与 `R-particle` 同源？ |
|---|---|---|---|
| 陈述 / 是非疑问 / 感叹 | **句末标点**（原文有、遮蔽后无） | — | **否** |
| 特指疑问 | 句末标点 | **QSET 词表** | **是**（QSET ⊂ R-particle 的词表） |
| 祈使 / 感叹 | 句末标点 | **IMP 词表** | 否（IMP 不进 R-particle） |
| 反问 | **RHO 词表**（**不看标点**） | — | **是**（难道 ∈ R-particle） |

⇒ **构念效度的独立性只在"陈述/是非/感叹 vs 疑问"这条标点分界上完整成立**；
特指-是非 与 反问 两处的细分与 `R-particle` **同源**，报告必须明示，不得把整体 `rule<1−2SE` 说成"全部构念独立"。

**触发词（triggers）**：**封闭表 `TRIGGERS`**（模式 → 候选词，按表内优先级取**第一次出现**的命中；
无命中 ⇒ `∅`）。触发词只从 `QSET ∪ {吗, 是不是, 能不能, 会不会, 有没有, 对不对, 要不要, 行不去…} ∪ RHO ∪ IMP`
里取，**表跑前写死在 `labels.py`，不事后增补**。触发词的 span 必须满足 `input[s:e] == text`。

---

## 3. 三个出口（输入侧构造）

| 出口 | 构造 | 用途 |
|---|---|---|
| **E_keep 保留标点** | 输入 = 原文照抄（含句末标点） | **免费规则可解的上界**（对照，M2） |
| **E_maskfinal 只遮句末标点** | 删去末字若 ∈ `FINAL_PUNCT` | 敏感性（报数，不进主判） |
| **E_mask 遮蔽标点（主出口）** | 删去**全部** `render.PUNCT` 中的字符（**复用 `src/dtseek/tasks/render.py` 的 `PUNCT`**，与"标点是纯形式"同口径） | **主出口（M0/M1/M3）** |
| **E_colloq 口语出口** | 语料里**本就没有句末标点**的话轮；标签改由**无标点词表规则** `label_nopunct`（**五支顺序**：RHO→反问 / QSET→特指 / YN 吗系→是非 / IMP→祈使 / **EXC 感叹词→感叹** / 否则陈述；`EXC = {太, 真, 好, 棒, 绝了, 哇, 唉, 哎, 咦, 啊, 呀, 啦, 极了, 疯了, 厉害, 服了, 无语, 崩了, 恭喜}` 跑前写死） | 可构造性如实报；**该出口标签按定义 `rule=1.0` ⇒ 不参与 M3 构念判定**，只报数 |

- 三出口（keep / maskfinal / mask）**同一批 test 行**（逐行配对 ⇒ 可用配对 SE）。
- `E_mask` 必须**断言输入中零 `render.PUNCT` 字符**（遮蔽生效的机器可验前提）。
- `E_colloq` 有**独立的 train/test 划分**（池不同、标签源不同），规则也**在它自己的 train 上 fit**。

---

## 4. `max_naive` 电池（逐出口，`fit=train`，跑前写死）

| # | 规则 | 定义 |
|---|---|---|
| R0 | `majority` | train 多数类 |
| R1 | **`R-punct`** | **两口径都报**：`simple` = 键为**输入末字 ∈ `MASK_SET` 时的该末字**、否则 `∅`，查 train 多数、未见键回退全局多数；`full` = 末标点 ＋ 支内词表（= `label_full`，零拟合）。**在 `E_mask` 上 `simple` 恒等于 R0**（M0 的恒等式断言），`full` 退化为 R6 |
| R2 | **`R-particle`** | **固定顺序**决策表 `PARTICLE = 吗, 呢, 吧, 难道, 是不是, 能不能, 有没有, 什么, 啥, 谁, 哪, 怎么, 咋, 多少, 几, 为什么, 如何, 呀, 啊`：输入含的第一个粒子 ⇒ 该粒子在 train 上的多数类；全不含 ⇒ 全局多数 |
| R3 | **`R-final-char`** | 键 = 末字（**无条件**），查 train 多数，未见回退全局多数 |
| R4 | `R-len-bucket` | 长度桶 {≤10, ≤20, ≤30, >30} → train 多数 |
| R5 | `R-first-char` | 首字 → train 多数，未见回退 |
| R6 | **`R-labeldef`** | **标签定义函数去掉标点支**的版本（`label_nopunct`），零拟合；在 `E_keep` 上 = `label_full` ⇒ **按定义 1.0**，在 `E_mask`/`E_colloq` 上 <1 |
| **max_naive** | `max(R0…R6)` | **逐出口**报，另报**逐条**与**卡 − max_naive** |

口径纪律：`R6` **计入** `max_naive`（`dev-notes/21` §3 的第 6 次教训：标注函数的逆/定义规则必须先比）；
触发词通道另报 `R-trigger-majority`（train 最常见的 trigger span），**单列**、不计入 mode 的 `max_naive`。

---

## 5. 识别卡（输入侧，核冻结只训头）

- **核**：`checkpoints/base_encoder.pt`（`NanoDocEncoder`, hidden **128**, 3 层, `max_len=128`），
  `load_base_encoder` 已置 `requires_grad=False` + `eval()`；本单元**再钉一次并打印断言**。
  编码在 `torch.no_grad()` 下做（核不训 ⇒ 不需保留激活）。
- **头**：`H [B,L,128]` →
  - **模式头**：`mean(H)` → `MLP(128→256→128)` → `Linear(128→6)`；
  - **触发词头**：逐位置 `Linear(128→3)` 的 **BIO**（O/B/I）→ 解码为 span 或 `∅`；
  - **损失** `L = CE(mode) + λ·CE(BIO)`，**λ=1（跑前写死，不调）**。
- **输出 = 结构化指令**（`mode_id + triggers[]`），**复用 `render.py` 的谓词口径**：
  - 遮蔽字符集 = `render.PUNCT`；
  - 触发词校验走 `render.item_problems(render.BagItem(...))`（**evidence 齐备**：span 越界 / 逐字子串 /
    `screened`+`candidate_id` 零信息新增），不满足 ⇒ `kind="reject"`（fail-closed，不交给模型判）；
  - 自写 `check_mode_structure()`（模式 id ∈ 封闭表、触发词 ∈ `TRIGGERS`、span 逐字）与
    `check_mode_deref()`（`ref_map` 的 `out` 区间**恰好铺满**触发词文本、内容条目 `span` 可回溯原文）
    —— 语义逐条对齐 `render.check_structure` / `render.check_deref`；
  - 记录形状 `{kind, evidence, plan_step_id, instruction, ref_map, reason}` 与 `render._record` 同构。
- **不改 `src/`**：`render.py` 只 import，不写。

## 6. 臂与配方（写死）

- **臂 = 输入版本**：`KEEP`（全 E_keep）/ `MASK`（全 E_mask）/ `MIX`（逐样本以 `seed` 掷币 50/50）；
  **× seed {42, 43}** ⇒ 6 跑；**负对照 `MASK+rand`（训练标签跨行 randperm，输入不动）× 2 seed** ⇒ 共 8 跑。
- **配方**：steps **800**、batch **64**、AdamW lr **1e-3** / wd **1e-4**、cosine、grad clip **1.0**、
  `DataLoader(generator=manual_seed(seed))`；只训头（`requires_grad` 名单打印）。
- **规模**：标签池去重后 shuffle(SPLIT_SEED) ⇒ **train 12000 / test 6000**（自然比例，**不人为均衡**）；
  口语出口 **train 8000 / test 4000**（自然比例）。
- **冒烟**：steps ≤ 40 的一跑，只验 shape / 断言 / 落盘，**数字不进报告**（`dev-notes/20` 更正 4 的教训）。

---

## 7. §验收口径（跑前写死，判据不改）

SE：**行级二项 SE** `sqrt(p(1-p)/n)`；配对 Δ 的 SE = 逐行 0/1 差的样本 SE（三出口同行 ⇒ 可用）；
**2 seed = 42/43**。`卡 − max_naive` 的**主判据用 spec 字面口径**（`2×SE_card`），
**同时报配对 SE 版**（更严/更松都摆出来，不许只报对自己有利的那个）。

| # | 判据 | 门槛 |
|---|---|---|
| **M0（设计校验）** | **R-punct 在 `E_mask` 上崩溃** | **(a) 恒等式断言（主）**：`E_mask` 逐行零 `MASK_SET` 字符，且 `R-punct` 的键恒为 `∅` ⇒ **`R-punct(E_mask) ≡ majority`（偏差 ≤1e-9，代码断言）**；**(b) 幅度只报不判**：`R-punct` 两口径（`simple`＝只看末标点查表、`full`＝末标点＋支内词表＝`label_full`）在 keep/mask 两出口的 acc 与落差。**若 `R-punct(E_mask) > majority + 1e-9` ⇒ 遮蔽没生效，停下来修，不往下跑** |
| **M1（主）** | `E_mask` 上 卡 > `max_naive(E_mask) + 2×SE_card` | **两 seed 同号**才判过；同时报 `2×SE_paired` 版 |
| **M2** | `E_keep` 出口（对照） | 报 `max_naive(E_keep)` 与卡 − `max_naive`，用于量化"标点给了多少"（预期 `max_naive(E_keep) ≈ 1.0`） |
| **M3** | **构念效度** | 在 `E_mask` 上 `max_naive < 1 − 2×SE` **且** 逐类声明"来源是否独立于该规则"；不满足处如实写"标签仍由规则定义" |
| **M4** | 随机标签对照 | `MASK+rand` 的 mode acc ≤ `majority + 2×SE_rand`（掉回多数类） |
| **M5** | 老卡 Δ | 四卡 `Δ ≥ −各自噪声带`（pronoun **2.83** / sentiment **0.41** / relation **1.39** / person **0.33** pt），训前训后各跑一次；`base_encoder.pt` 与四卡 sha256 训前=训后 |
| **M6** | `max_naive` 全电池 | **逐出口**报 R0–R6 + `max_naive` + **卡 − `max_naive`**（含 R-particle / R-final-char 单列） |

**判定三选一（不许硬选）**：
1. **输入侧句式可读（且有构念效度）** = M0 过 ∧ M1 过 ∧ M3 过；
2. **只靠标点（遮蔽后崩）** = M0 过 ∧ M1 未过，且 `max_naive(E_keep) − max_naive(E_mask)` **≥ 0.30**
   （增益主要由标点解释）；
3. **证据不足** = M0 未过 ∨ M1 两 seed 异号 ∨ M4 未落回 ∨ M5 未过 ∨ 关键出口不可构造。

**不许**把"没测出差异"写成"没有效果"；实测/推断分开标，证据强度标（单 seed / 配对 / 2 seed）。

---

## 8. 运行清单（写死；unit 名先定，便于登记并跑）

| stage | unit | 命令 | 日志 |
|---|---|---|---|
| 数据 | `sm7-build` | `uv run python -u experiments/sentence_mode/build_data.py` | `logs/sm7_build.log` |
| 规则 | `sm7-rules` | `uv run python -u experiments/sentence_mode/rules.py` | `logs/sm7_rules.log` |
| 老卡 before | `sm7-old-before` | `uv run python -u experiments/sentence_mode/eval_old_cards.py --tag before` | `logs/sm7_old_before.log` |
| 训练（8 跑） | `sm7-train` | `bash experiments/sentence_mode/run_train.sh` | `logs/sm7_train.log` + 逐跑 `logs/sm7_train_<name>.log` |
| 老卡 after | `sm7-old-after` | `uv run python -u experiments/sentence_mode/eval_old_cards.py --tag after` | `logs/sm7_old_after.log` |
| 汇总 | `sm7-summary` | `uv run python -u experiments/sentence_mode/analyze.py` | `logs/sm7_analyze.log` |

启动模板：
`systemd-run --user --unit=<名> --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek
--setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 <脚本>`
启动前查 `systemctl --user is-active 'sm7-*' 'dtseek-*'`，**忙则排队，绝不杀他人进程**。

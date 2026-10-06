# layer_selective —— 把「共享参数集合」当可设计对象：只让新任务梯度进指定层，或插一个私有适配层

> **本文在首次训练之前写死**（mtime 必须早于任何 train 进程启动时刻），跑完不改；结果填进 `REPORT.md`。
> 口径：「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 构造时人为规定的标签归属。

## 0. 唯一问题

`experiments/core_select_only/REPORT.md` 已经把两极量完：
`selonly`（核全可训、只给 select 梯度）新能力 **73.60 / 68.40** 但**老卡 8/8 崩**（头冻结也崩 ⇒ 崩的是核）；
`joint`（核全可训 + 4 老任务同批）老卡 **8/8 无退化** 但 `heldout` 只 **59.36 / 63.20**。

> **本项目唯一要回答的问题：能不能「改核以拿到新能力」，同时「不崩老卡」？
> 落地操作 = 把「共享参数集合」当成可设计的东西** —— 只让新任务的梯度进**指定层**（EMB / TOP），
> 或者干脆不动核、插一个**私有适配层**（ADPT）。

五臂构成一个五点的「哪些参数可训」扫描：**F（不动核）→ EMB（最浅层）→ TOP（最深层）→ ADPT（核外私有容量）→ ALL（全动）**。
`methodology/measurement.md` 的结论（Adam 下范数比不能判「谁主导」，杠杆在**共享参数集合**与**方向**）正是本项目的出发点：
本项目不看范数比，直接**改集合**再测两端。

## 1. 数据（**只读复用**，一个字节都不写）

- 直接读 `experiments/select_rerank/data/` 的 **clean（N）臂** train/test/adv 三个 jsonl；
  行序逐行复刻 `select_rerank/train_rerank.py::load_rows`（train→test→adv），由只读 import 的
  `select_semantic_joint/common.py::load_rows` 保证。
- **md5 自证（跑前 + 跑后各一次，必须与下表一致）**（跑前实测 2026-10-07 01:06 已核）：

  | 文件 | md5 |
  |---|---|
  | `clean/train.jsonl` | `6b1e6fb1932e5c755cbbaca1c1181f5b` |
  | `clean/test.jsonl` | `d3d597afff387b0ca66ab58b838c1915` |
  | `clean/adv.jsonl` | `2a5ca5f4eaac570811eccd4f56a8d3be` |

- token 级编码缓存**只读复用** `experiments/select_pool/cache/clean_28988c64fa91_tok_ctx64_cand32.pt`
  （**核冻结臂 F / ADPT 读它** ⇒ 与 `select_pool` 臂 A 逐位同输入；**不写**该目录）。
- 老任务数据 / 老卡头只读复用 `experiments/capability_map/cache/*`、`capability_map/cards/*_frozen_s{S}.pt`、
  step-0 基线 `experiments/select_semantic_joint/results/r4_baseline_s{S}.json`。
- 只跑 clean 臂：`heldout_pair` / `shifted_pos` 各 **1250** 条。

## 2. 五臂（**唯一变量 = 哪些参数 `requires_grad=True`**）

模型构造顺序与 `select_pool` / `select_semantic_joint` / `core_select_only` **逐字一致**（`SemModel(spec, freeze_core=False)` → 之后才按臂施加冻结策略 / 建适配层），RNG 流同源；核**全程 `eval()`**（dropout=0）。

| 臂 | 可训参数（跑前预期，selftest 逐臂实测打印 `requires_grad=True` 的**名字+数量**核对） | 可训合计 | 每步数据 | 编码 |
|---|---|---|---|---|
| **F（基线）** | 只打分头 `head.*`（核 `encoder.*` 全 False） | **164,353** | select batch 64 | **读缓存**（= `select_pool` 臂 A） |
| **ALL**（= `selonly`） | 核 **1,688,460 全可训** + 头 164,353 | **1,852,813** | select batch 64 | live |
| **EMB** | 只 `encoder.embedding.weight` **1,048,576** + 头；`blocks.*`/`norm` 冻结 | **1,212,929** | select batch 64 | live |
| **TOP** | 只 `encoder.blocks.2.*` **213,252**（最深层）+ 头；`embedding`/`blocks.0/1`/`norm` 冻结 | **377,605** | select batch 64 | live |
| **ADPT** | 核 **全冻结（0 可训）** + **私有适配层 65,920** + 头 164,353 | **230,273** | select batch 64 | **读缓存**（核不动 ⇒ 与 F 逐位同输入） |

**ADPT 的私有适配层（跑前定死，跑后不改）**
- 位置：**核输出之后、mean-pool 之前**（token 级）：`x → x + fc2(GELU(fc1(x)))`，
  fc1 `Linear(128→256)`、fc2 `Linear(256→128)`，ctx 与 cand **共享同一份权重**。
- **残差 + fc2 权重与偏置零初始化** ⇒ **step-0 逐位等于 F**（自检断言 `max|adapter(x)−x| == 0`），
  只有训出来的变化才进核与头之间；初始化不改变核的任何表示。
- 参数量 **65,920 = 核的 3.90%**（≤ 5% 硬约束）；`semantics`：老卡走 `enc(inp)` **直连**，
  适配层不在老卡路径上 ⇒ 老卡影响只能来自核本身（核冻结 ⇒ 结构上不受影响，这条要如实报成「构造保证」而非「学出来的」）。

**优化器口径（跑前写死）**：AdamW、wd **1e-4**、cosine `T_max=steps`、grad clip **1.0**；
含预训练核参数的臂（ALL / EMB / TOP）两组：**核子集 lr 3e-4 / 头 lr 1e-3**；
不含的臂（F / ADPT）单组 **lr 1e-3**（随机初始化参数同打分头口径）。
**共享配方** = `select_semantic_joint`：**1800 步**、batch **64**、**seed 42 / 43**、核 `eval()`、
live 编码（核可训臂）/ 缓存（核冻结臂）、`n_train = 8000`。
**不许调参刷过**：步数 / lr / 聚合（mean-pool）/ 适配层宽度 / 位置全部跑前定死，跑后不改、不扫。

**随机标签对照（X4）**：只把 train 标签 `randperm`（`seed*1000+7`，保持 1:1），test/adv 仍用真标签；
每臂 2 seed。**资源降级顺序（跑前写死，避免事后挑）**：若 GPU 排队导致必须减跑，
顺序 = 先保真标签 10 跑 → F/ALL 的 rand（口径对照）→ **满足 X1 的臂的 rand**；仍缺则该臂 X4 标「未测」，
而 X1 已不过的臂不参与判定，故不影响三选一。

## 3. 判据（**跑前写死**；n=1250 ⇒ **SE = 1.414pt**、**2×SE = 2.828pt**；两臂在同一批 1250 条上评测）

| # | 判据 | 门槛（逐 seed） |
|---|---|---|
| **X0（门禁）** | **F 零误差复现 `select_pool` 臂 A**：`heldout` **49.52 / 51.04**、`shifted` **92.56 / 92.48**、`test` 94.48/94.48、`train` 100.00/100.00、`adv` 71.04/71.76、`loss_first` 0.693002/0.692196、核漂移 **= 0**；**ALL 零误差复现 `core_select_only::selonly`**：`heldout` **73.60 / 68.40**、`shifted` 100.00/100.00、`test` 100.00/99.96、`train` 100.00/100.00、`adv` 86.80/84.20、`loss_first` 0.693002/0.692196、`loss_select_first50` 0.153912/0.134212、核漂移 **0.05003196568723828 / 0.048874110162901115**、老卡 8 项 Δ（pronoun −0.461667/−0.515、sentiment −0.485625/−0.502188、relation −0.679167/−0.694444、person −0.005/−0.006667） | 逐项 **Δ = 0.00**（不是 ≤2×SE）；任一项非零 ⇒ **先查口径，不出结论** |
| **X1 新能力** | 臂的 `heldout` ≥ **ALL − 2×SE**（= **70.77 / 65.57**） | **2 seed 同号**（两 seed 都 ≥ 门槛） |
| **X2 老卡** | 四张老卡 Δ ≥ **−各自带**：pronoun **0.0283** / sentiment **0.0041** / relation **0.0139** / person **0.0033**（= `core_keep/eval_core_keep.py::BAND`）；基线 = 同进程 step-0 实算并与 `r4_baseline_s{S}.json` 对账 | **2 seed 全部 8 格**都过 |
| **X3** | `shifted` ≥ **F − 2×SE**（= 89.73 / 89.65） | 报告项，不改判定；不过则标「拿 shifted 塌陷换的」 |
| **X4（门禁）** | 随机标签对照 | 该臂 rand 的 `heldout` **与** `shifted` 均 **≤ 52.83pt**；否则**该臂判证据不足** |
| **X5（必报）** | 各臂**可训参数量（实测打印）**、**核漂移**、**步时（末 200 步中位）**、**峰值显存** | 数值齐全 |

- **判定三选一（跑前写死，不许硬选）**：
  - **① 解耦成功** = X0/X4 门禁过，且**存在至少一个臂同时满足 X1 ∧ X2**（两 seed 都满足）；
  - **② 解耦不成立** = X0/X4 门禁过，但**没有任何臂同时满足 X1 ∧ X2**；
  - **③ 证据不足** = X0 或 X4 门禁不过、只有单 seed、自检失败、或跑缺。
- **分两层读**：若成功臂是 **TOP / EMB** ⇒ 结论属「**层选择**」（梯度进哪一层就够了）；
  若成功臂是 **ADPT** ⇒ 结论属「**私有参数**」（不动核也能买新能力）；两者成功度不同则**按两层分别给结论**，不合并成一句。
- X3 / X5 是**报告项**，不参与三选一；≥2 seed 才下结论。

## 4. 空测试自检（`selftest.py` 跑前必过；训练脚本内还有硬断言）

1. **逐臂实测打印 `requires_grad=True` 的参数名与数量**（不是声明）：F 164,353 / ALL 1,852,813 /
   EMB 1,212,929 / TOP 377,605 / ADPT 230,273，且**逐臂打印冻结侧的实况**（哪些名字被冻住、数量）；
2. **ADPT 核确实冻结**：`sum(numel for p in encoder.parameters() if p.requires_grad) == 0`，
   且 step-1 反向后 `encoder` 全部参数 `p.grad is None`；EMB/TOP 同理（非目标层 `grad is None`）；
3. **step-1 梯度断言**：目标层梯度范数 > 0、非目标层梯度范数 == 0（逐臂打印）；
4. **适配层 step-0 恒等**：`max|adapter(x) − x| == 0`（零初始化生效）；
5. **多数类基线**：train/test/adv 三个 split `majority_baseline == 0.5000`（1:1 被破坏即炸）；
   两个 ctype 跑前实测 627/623 ⇒ 多数类 0.5016，n=1250 ⇒ SE 仍按 1.414pt 报；
6. **随机标签对照真执行**：打乱后 1:1 计数不变、与真标签不同条数 > 0（打印）；
7. **RNG/口径同源自检**：五个真标签臂 step-0 `loss_first` 必须都等于 **0.693002 / 0.692196**
   （同 seed 同头初始化 ⇒ 任何一臂不同即说明构造顺序被改过）；
8. **step-0 老卡对账**：四张老卡逐样本 pred sha256 与 `select_semantic_joint/results/r4_baseline_s{S}.json`
   **逐位相等**，不等 ⇒ 该 seed 作废。

## 5. 写入范围与纪律

只写 `experiments/layer_selective/`、`logs/layer_selective_*`、`/tmp`；
**不改** `experiments/select_rerank/`（含 `data/`）、`select_pool/`、`select_semantic_joint/`、`core_select_only/`、
`capability_map/`、`src/`、`training/`、`tests/`、`dev-notes/`（**全部只读**，`sys.dont_write_bytecode=True` 不落 `__pycache__`）；
**禁止 `git commit / stash / checkout / restore / clean`**；
长跑一律 `systemd-run --user --unit=<名> --collect --property=WorkingDirectory=... --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1`
（**绝不用 `setsid nohup &`**）；
**GPU 与他人共用**：开训前查 `ps` 与 `systemctl --user is-active 'dtseek*'`，忙则排队等，**绝不杀他人进程**；
先落盘 PREREG + 代码 + 冒烟结果，再跑全量；单跑目标 ≤ 2 分钟。

# select_semantic_joint —— 条件「乙」：更强表示下 `heldout_pair` 是否仍 ≈50%？

> **本文在首次训练之前写死**（mtime 必须早于任何 train 进程启动时刻），跑完不改；结果填进 `REPORT.md`。
> 口径：「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 构造时人为规定的标签归属。

## 0. 唯一问题（= `dev-notes/16` §5.5 写明的条件「乙」）

> **在「更强表示」下，同一套数据与同一套口径（mean-pool），`heldout_pair` 是否仍 ≈50%？**

- 前四条同类测量（`reply_pick` 20% / `cloze_fill` 19% / A 族对抗集 0.525 / B 族 `heldout_pair` 49.52~51.04）
  与 `select_pool` 四臂消融（判定 ③）都停在**核全程冻结**。
- **「更强表示」的落地 = 让核可训**（温启动：核 ← `checkpoints/base_encoder.pt`，
  四张老卡头 ← `capability_map/cards/*_frozen_s{S}.pt` **冻结**）。
  本项目已验证该参数配置（`core_keep` J3）能让老卡不退化（dev-notes/15 §15，8 格 × 2 seed Δ ≥ 0）。
- **R1 过 ⇒ 墙是「核从没学过这个任务」** ⇒ 路线变成「让核参与训练」；
  **R1 不过而 R2 过 ⇒ 墙在表示** ⇒ 这是第五条同类测量，且与前四条**同任务同数据**。
- **不许硬选、不许调参刷过**：判据跑前写死；换配方（步数 / lr / 聚合 / 是否加老任务数据）必须另开 PREREG。

## 1. 数据（**只读复用**，不改不重建）

- **直接读** `experiments/select_rerank/data/` 的 **clean（N）臂**，train/test/adv 三个 jsonl
  **一个字节都不写**；行序逐行复刻 `select_rerank/train_rerank.py::load_rows`（train→test→adv）。
- **口径自证（跑前 + 跑后各一次）**：三个文件的 md5 与 `find -newermt` 首训时刻前后各算一遍，
  必须与下表一致且 mtime 不变；结果写进 `REPORT.md`。

  | 文件 | md5 | mtime |
  |---|---|---|
  | `clean/train.jsonl` | `6b1e6fb1932e5c755cbbaca1c1181f5b` | 2026-10-06 20:10:26.515 |
  | `clean/test.jsonl` | `d3d597afff387b0ca66ab58b838c1915` | 2026-10-06 20:10:26.525 |
  | `clean/adv.jsonl` | `2a5ca5f4eaac570811eccd4f56a8d3be` | 2026-10-06 20:10:26.538 |

- **token 级编码缓存只读复用** `experiments/select_pool/cache/clean_28988c64fa91_tok_ctx64_cand32.pt`
  （不写该目录）——核对臂因此与 `select_pool` 臂 A **逐位同输入**。
- 只跑 **clean（N）臂**：`heldout_pair` / `shifted_pos` 各 **1250** 条（adv 的两个 ctype）。
- 老任务数据只读复用 `experiments/capability_map/cache/*_{n}_20240927.pkl`（不写该目录）。

## 2. 两臂（**唯一变量 = 核是否可训**）

| | **核对臂 `frozen`** | **主臂 `joint`** |
|---|---|---|
| 核 `NanoDocEncoder`（1,688,460） | `requires_grad` **全 False**、`eval()`、**不前向**（读缓存） | `requires_grad` **True**、**`eval()` 全程（dropout 关）**、每步前向（live） |
| 四张老卡头（pronoun/sentiment/relation/person） | 不参与（`r4` 时单独建） | 挂载并**冻结**（`requires_grad` 全 False） |
| 聚合 | masked **mean-pool**（= `select_pool` 臂 A，参数 0） | 同左 |
| 打分头 `MLP([ctx;cand;ctx⊙cand;\|ctx−cand\|])` | 164,353，可训 | 164,353，可训 |
| 每步训练数据 | select batch（读缓存） | **select batch（live 编码）+ 4 个老任务 batch**（= `core_keep` J3「旧任务+新任务同批梯度」口径） |
| 优化器 | AdamW 单组 **lr 1e-3**、wd 1e-4 | AdamW 两组：**核 lr 3e-4**（= `core_keep --lr-base`，§15 已验证，跑前定死不扫）／**头 lr 1e-3** |
| 调度 / clip | cosine T=1800 / 1.0 | 同左（clip 覆盖核+头的可训参数） |

- **两臂共享的 select 配方**（= `select_pool`/`select_rerank` 逐项相同）：
  **1800 步**、batch **64**、lr_head **1e-3**、AdamW(wd 1e-4)、cosine、grad clip 1.0、**seed 42 / 43**。
- **为什么核 `eval()`**：dropout 开/关会成为第二个变量 ⇒ 两臂**全程 eval**，唯一差别只有 `requires_grad`。
- **为什么核对臂不跑老任务批次**：核冻结 + 老头冻结 ⇒ 老任务损失对**任何可训参数的梯度恒为 0**
  （`selftest.py` 实测：joint 循环把核冻住跑 N 步，打分头权重与「只跑 select」**逐位相同**）
  ⇒ 核对臂跑它是纯浪费；该等价性是**「两臂只差核是否可训」这条的证据**，必须实测。
- 核对臂的代码路径 = `select_pool/train_pool.py` 逐行复刻（只改输出路径与缓存为**只读**）。

## 3. 判据（跑前写死；n=1250 ⇒ **SE = sqrt(0.25/1250) = 1.414pt**，**2×SE = 2.828pt**）

| # | 判据 | 门槛 |
|---|---|---|
| **R0（口径门禁）** | 核对臂 `frozen` **零误差复现** `select_pool` 臂 A | `heldout_pair` = **49.52** / **51.04**、`shifted_pos` = **92.56** / **92.48**，**逐 seed Δ = 0.00**；不为零 ⇒ **先查口径，不出结论、不开主臂** |
| **R1（主）** | 主臂 `heldout_pair` **> 核对臂**，逐 seed 算 Δ | **Δ > 2×SE = 2.83pt**，**两 seed 同号**；报 SE、Δ、Δ/SE |
| **R2（副）** | 主臂 `shifted_pos` 不显著低于核对臂 | 每 seed **≥ 核对臂 − 2.83pt**（防「拿表内记忆换」——`select_pool` 的 D 就是这么输的） |
| **R3（随机标签对照）** | train 标签 `randperm`（保持 1:1，`seed*1000+7`）后同配方重训，test/adv 仍用真标签 | 主臂 **与** 核对臂各 2 seed：`heldout_pair` **与** `shifted_pos` **均 ≤ 52.83pt**；否则 R1 作废 ⇒ 判「证据不足」 |
| **R4（门禁·硬）** | **step-0 逐位不变**：温启动起点上，四张老卡的**逐样本 pred（切片发射）+ exact** 与「**不接本任务**」时**逐位相同** | **sha256 相等**（两 seed 各一对）；不等 ⇒ 权重没接对 ⇒ 该 seed 作废 |
| **R5（报告）** | 主臂训练后老卡不退化 | 每卡每 seed **Δ ≥ −各自噪声带**：pronoun **0.0283** / sentiment **0.0041** / relation **0.0139** / person **0.0033**（= `core_keep/eval_core_keep.py::BAND`）；基线 = 同进程 step-0 实算（已由 R4 与 `capability_map/summary.json::frozen_exact` 对账） |
| **R6（必报）** | 核漂移与代价 | `‖core−base‖F / ‖base‖F`（相对 `checkpoints/base_encoder.pt`）+ 步时（steady 中位）+ 峰值显存 |

- **判定三选一**（R3/R4 是**门禁**：任一不过 ⇒ 直接判「证据不足」，不看 R1/R2）：
  - **① 墙在表示** = R1 **不过** 且 R2 **过**（且 R0/R3/R4 都过）；
  - **② 墙是「核没学过」** = R1 **过**（且 R0/R3/R4 都过；R2 只报不改判定）；
  - **③ 证据不足** = 其余（R0/R3/R4 不过、只在一个 seed 过、缺 seed、自检失败）。
  R5/R6 是**报告项，不参与三选一**（它们测「让核参与训练」这条路的**代价**，不测本次测量的有效性）；
  R5 不过会如实报，并在 `REPORT.md` 里写清它对路线的含义。
- **≥2 seed 才下结论；不调参刷过**：所有超参（步数、lr、聚合、老任务权重）跑前定死，跑后不改。

## 4. 空测试自检（训练脚本内断言，跑前必过）

1. 可训参数量逐臂报数：核对臂 = **164,353**；主臂 = 164,353 + **1,688,460** = **1,852,813**；
2. **冻结实况实测**（不声明）：主臂四张老卡头 `requires_grad` 全 False（逐参数打印）、
   核对臂核 `requires_grad` 全 False 且 `training == False`；
3. **多数类基线**：train/test/adv 及两个 ctype 的 `majority_baseline == 0.5000`（1:1 被破坏即炸）；
4. **随机标签对照真执行**：打乱后 1:1 计数不变、与真标签不同条数 > 0（并打印）；
5. **R4 step-0 sha256 实测**（不声明）：由 `r4_check.py` 在**两个独立配置**下各算一遍；
   主臂训练脚本在**任何 optimizer.step 之前**再算一遍并写进结果，三处必须相等；
6. **等价性实测**：joint 循环 + 冻核 跑 N 步 vs 只跑 select 跑 N 步 ⇒ 打分头权重 `torch.equal`；
7. **live 编码 vs 缓存**：step-0 对同一批文本，live（batch 64/256, eval, no_grad）与
   `select_pool` 缓存的 `max|Δ|`（只报不判，期望 ~1e-7 量级；核对臂始终用缓存）。

## 5. 写入范围与纪律

只写 `experiments/select_semantic_joint/`、`logs/select_semantic_joint_*`、`/tmp`；
**不改** `experiments/select_rerank/`（含 `data/` 与 `PREREG`，**只读**）、`experiments/select_pool/`、
`src/`、`training/`、`tests/`、`dev-notes/`、其它 `experiments/`；
不 `git commit/stash/checkout/restore/clean`；
长跑一律 `systemd-run --user --unit=... --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1`
（**绝不用 `setsid nohup &`**；`PYTHONDONTWRITEBYTECODE=1` 是为了不在只读目录里落 `__pycache__`）；
开训前查 `ps` 与 `systemctl --user is-active 'dtseek*'`，忙则排队，**绝不杀他人进程**。

# P19 训练附录 PREREG（重训配方写死，mtime 早于首次训练启动）

触发条件（`PREREG.md` §2 步骤 2 已写死）：主候选 `checkpoints/negation_accept_card.pt` 与次候选
`checkpoints/negation_accept_card_e12.pt` 在线上核 `respond` 口径下 M_cls 均 **< .98** ⇒ 走步骤 3 **重训**。
实测（`results/s2_e2e.json`，改前跑的）：主候选 **.8894 / .8953**、次候选 **.9140 / .9188**、现默认卡 **.2801 / .2906**（seed 42/43）。
⇒ 两张现成卡都不达标，**没有「在线上核上 ≈1.0000」的现成解**。

## 1. 重训硬条件（来自任务书，不改）

- **核冻结**（`train_task_card.py` 本来就 `load_base_encoder` 后不给核梯度）+ **`--base checkpoints/base_encoder.pt`（线上核）**；
- **2 seed = 42 / 43**（`--seed`，同时决定头初始化、训练集洗牌与 `build_dataset` 无关的随机性）；
- 数据 = **negation 任务既有口径**：`NegationCard.build_dataset(target_samples)` → `build_negation_dataset(target_samples, seed=DEFAULT_SEED)`
  （`src/dtseek/tasks/builtin/negation/dataset.py`，闭集 `NEG_MARKERS`、正例 0.65、同形/虚化/A-not-A 整句丢弃、`max_len=64`）；
- 产物只写 `experiments/fix_negation/cards/`（**不写 `checkpoints/`**）；
- **必须同时报免费规则地板**（`PREREG.md` F2，含 `n_cue` 计数类）。

## 2. 配方（**有界搜索：最多 2 个，按固定顺序**；两配方结果全部报，不隐藏）

| 配方 | 命令要点（`training/train_task_card.py`） | 意图 | 出口 |
|---|---|---|---|
| **R1（先跑）** | `--card negation --base checkpoints/base_encoder.pt --epochs 40 --steps-per-epoch 150 --samples 8000 --lr-head 1e-3 --batch-size 64 --seed {42,43}` | 与现有 accept 卡**同数据量（8000）、同头规格（2 层/128 维/4 头）、同 lr**，只把步数从 12 epoch(1800 步) 拉到 40 epoch(**6000 步**) ⇒ 排除「没训够」 | `experiments/fix_negation/cards/negation_r1_s{42,43}.pt` |
| **R2（仅当 R1 两 seed 有一 < .98 才跑）** | 同上但 `--samples 32000 --epochs 16 --steps-per-epoch 300`（4800 步，数据 4 倍） | 排除「数据不够多样」 | `experiments/fix_negation/cards/negation_r2_s{42,43}.pt` |

**停止规则（写死）**：R1 两 seed M_cls **均 ≥ .98 ⇒ 选 R1，不跑 R2**；否则跑 R2；
**R2 两 seed 仍有一 < .98 ⇒ F1 不达标 ⇒ 判定「无法修复」**（给原因），**不再加配方、不再调参**。

**披露（口径纪律）**：R1→R2 的触发用的就是 eval_S 的 M_cls，属**预注册的有界配方搜索（2 个）**，
不是无限调参；两个配方的全部数字都进 REPORT，**不按 eval 挑最好看的**，选卡按上面的固定顺序。

## 3. 评测与门禁（与 `PREREG.md` 一致，不重复改判据）

- 选中卡 ⇒ 改 `dialogue.py` 的 `DEFAULT_ATTACH` **一行** ⇒ 跑 F1（端到端前后对比）/ F2（免费规则地板）/
  F3（等价性重放，不一致 = 0）/ F4（全库 pytest）/ F5（四老卡门禁 + 回退路径）。
- 判定三选一与门槛沿用 `PREREG.md` §1，**一字不改**。

## 4. 启动方式

`systemd-run --user --unit=fixneg-train-<r>-s<seed> --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 …`
（**绝不用 `setsid nohup &`**；GPU 忙就排队，**不杀他人进程**）。日志 → `experiments/fix_negation/logs/train_*.log`。

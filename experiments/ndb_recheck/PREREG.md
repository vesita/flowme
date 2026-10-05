# NDB 重验配对实验 · 预注册判据（跑前写死，事后不许改）

时间：本文件在**任何一次训练启动之前**写入。对标历史结论 dev-notes/11
（旧语料基线上 `repeat_mention_acc` 0.375 → 0.843，3 seed Δ=+0.469，极差 0.011）。
语料入口已修复（person 真实语料占比 29.7%），历史结论**不能沿用**，必须在新语料上重验。

## 判据（原文照抄）

- **P1**：开 NDB 的 `repeat_mention_acc` 相对关 NDB 提升 **≥ +0.20**（对标历史 +0.469，允许语料变更导致的衰减）。
- **P2**：Δ 在 2 个 seed 上**同号**且极差 **≤ 0.05**。
- **P3**：`first_mention_acc` 下降 **≤ 0.05**（NDB 不应以牺牲首次提及为代价）。
- **P4（空壳检验）**：①记忆模块参数 `grad != 0`；②**旁路整个记忆模块后指标必须变化**。两条缺一不可
  （dev-notes/11：有臂梯度全非零，门控却自己关到 2.6e-3，是空壳）。
- **P5（代价）**：报参数增量、配对口径的每步耗时增量、峰值显存增量（用 `torch.cuda.max_memory_allocated`）。
- 若 P1 不过，**如实判失败**并给机制，不要沿用旧结论。

## 配对规格（跑前定死）

- 冻结基座：`checkpoints/base_encoder.pt`，由 `scripts/split_checkpoint.py --ckpt checkpoints/multitask_v2_dtseek.pt`
  从四卡基线一体 ckpt 拆出（meta.split_from 已核），训练中基座不参与更新（`torch.no_grad` 前向）。
- 驱动：`scripts/ab_test_ndb.py`（仓内现成配对 AB，不改任何 `src/` / `training/` / `scripts/` 文件）。
- 臂：`base`（不加 `--ndb`）vs `ndb`（加 `--ndb`）；seed 42、43 → 共 4 个训练进程。
- 逐项相同：`--card person --epochs 8 --steps-per-epoch 150 --batch-size 64 --num-layers 2`
  与同一 `--base`、同一数据（`build_dataset` 用卡内固定 seed，`val` 切分用 `random.Random(seed)`，
  两臂逐位同源）。
- NDB 臂内配置：`--ndb-read true`（**历史同款**：dev-notes/11 的 literal 臂即 `read_true=True`，
  见 `experiments/ndb_parametric/ab_parametric.py:68`）、`--ndb-levels 1,2 --ndb-slots 8192,4096`。
- 主判定只用这一套配置；`--ndb-read pred` 属于可选稳健性补充，**不参与 P1–P5 判定**。

## 已知配对瑕疵（跑前声明，不算事后找补）

- NDB 臂构造 `MentionNDB`（`train_task_card.py:108`）会消耗全局 torch RNG，
  因此两臂在**数据切分之后**的 DataLoader 批序与 dropout 掩码不再同轨迹。
  解码器参数初始化发生在 NDB 构造之前（`train_task_card.py:94`），两臂初始化逐位相同；
  数据集与 val 切分用局部 RNG，两臂逐位相同。历史 literal/base 配对同样带这条瑕疵。

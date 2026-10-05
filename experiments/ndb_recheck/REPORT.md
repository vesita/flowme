# NDB 开/关配对重验（新语料）报告

- unit：`ndb-recheck-ab`（`systemd-run --user --unit=ndb-recheck-ab --collect`，09:12 启动，跑完自动回收；`systemctl --user is-active` 由 active → inactive 判终）。总日志 `logs/ndb_recheck_ab.log`，逐跑日志 `logs/ndb_recheck_{base,ndb}_seed{42,43}.log`。
- 产物：`checkpoints/ndb_recheck_{base,ndb}_seed{42,43}.pt`（新文件名，未覆盖既有 ckpt）。
- 数据：`experiments/ndb_recheck/{PREREG.md, ab_summary.json, raw_metrics.json, probe_seed42.json, probe_seed43.json}`。

## 1. 预注册判据（跑前写死于 PREREG.md，事后未改）

- **P1**：开 NDB 的 `repeat_mention_acc` 相对关 NDB 提升 **≥ +0.20**（对标历史 +0.469，允许语料变更导致的衰减）。
- **P2**：Δ 在 2 个 seed 上**同号**且极差 **≤ 0.05**。
- **P3**：`first_mention_acc` 下降 **≤ 0.05**（NDB 不应以牺牲首次提及为代价）。
- **P4（空壳检验）**：①记忆模块参数 `grad != 0`；②**旁路整个记忆模块后指标必须变化**。两条缺一不可（dev-notes/11：有臂梯度全非零，门控却自己关到 2.6e-3，是空壳）。
- **P5（代价）**：报参数增量、配对口径的每步耗时增量、峰值显存增量（用 `torch.cuda.max_memory_allocated`）。
- 若 P1 不过，**如实判失败**并给机制，不要沿用旧结论。

## 2. 基座来源与训练命令（原文）

基座 = `checkpoints/base_encoder.pt`（hidden 128、冻结、`torch.no_grad` 前向）。它由 `scripts/split_checkpoint.py --ckpt checkpoints/multitask_v2_dtseek.pt` 从四卡基线一体 ckpt 拆出（ckpt `meta.split_from=checkpoints/multitask_v2_dtseek.pt`）。本次重拆到 /tmp 对比：`doc_encoder` 26 个张量**逐位相同** ✅。

```bash
systemd-run --user --unit=ndb-recheck-ab --collect --working-directory=/home/vesita/coding/my/DTSeek /bin/bash -c \
 'exec /usr/bin/uv run python -u scripts/ab_test_ndb.py --seeds 42,43 --epochs 8 --steps-per-epoch 150 \
  --batch-size 64 --num-layers 2 --ndb-read true --out-dir /tmp/ndb_recheck/run1 \
  --json-out experiments/ndb_recheck/ab_summary.json > logs/ndb_recheck_ab.log 2>&1'
```

驱动逐跑命令（`logs/ndb_recheck_ab.log` 的 `[run]` 行原文，2 臂 × 2 seed = 4 个独立进程，仅 seed / `--ndb` 有别）：

```
training/train_task_card.py --card person --base checkpoints/base_encoder.pt --out /tmp/ndb_recheck/run1/person_base_seed42.pt --epochs 8 --steps-per-epoch 150 --batch-size 64 --num-layers 2 --seed 42
training/train_task_card.py --card person --base checkpoints/base_encoder.pt --out /tmp/ndb_recheck/run1/person_ndb_seed42.pt  --epochs 8 --steps-per-epoch 150 --batch-size 64 --num-layers 2 --seed 42 --ndb --ndb-read true --ndb-levels 1,2 --ndb-slots 8192,4096
training/train_task_card.py --card person --base checkpoints/base_encoder.pt --out /tmp/ndb_recheck/run1/person_base_seed43.pt --epochs 8 --steps-per-epoch 150 --batch-size 64 --num-layers 2 --seed 43
training/train_task_card.py --card person --base checkpoints/base_encoder.pt --out /tmp/ndb_recheck/run1/person_ndb_seed43.pt  --epochs 8 --steps-per-epoch 150 --batch-size 64 --num-layers 2 --seed 43 --ndb --ndb-read true --ndb-levels 1,2 --ndb-slots 8192,4096
```

`--ndb-read true` = 历史同款（dev-notes/11 的 literal 臂即 `read_true=True`，见 `experiments/ndb_parametric/ab_parametric.py:68`）；NDB 臂内配置固定，不参与 P1–P5 口径。

## 3. 原始数字（2 seed × 2 臂）

| 臂 | seed | repeat | first | cluster_f1 | exact_match | id_acc | cls/span | 训练墙钟 | ms/步 | 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 42 | 0.3596 | 0.8671 | 0.3521 | 0.3233 | 0.6975 | 1.0000/1.0000 | 130.9s | 109.0 | 868MB |
| base | 43 | 0.3822 | 0.8837 | 0.3803 | 0.2967 | 0.7297 | 1.0000/1.0000 | 130.6s | 108.9 | 868MB |
| ndb | 42 | **0.8319** | 0.9906 | 0.7907 | 0.3222 | 0.9409 | 1.0000/1.0000 | 147.9s | 123.2 | 899MB |
| ndb | 43 | **0.8120** | 0.9969 | 0.7706 | 0.3022 | 0.9348 | 1.0000/1.0000 | 147.8s | 123.2 | 899MB |

配对 Δ（ndb − base，同 seed）：`repeat` **+0.4722 / +0.4298**（mean +0.4510，range 0.0425）；`first` +0.1235 / +0.1132；`cluster_f1` +0.4387 / +0.3903（range 0.0484）；`exact_match` −0.0011 / +0.0056；`id_acc` +0.2434 / +0.2051。

空壳检验（`probe_shell.py`，两次评估前都**逐位复现**训练 `AB_METRICS` → 探针同构 ✅）：

| seed | ①梯度（10 batch，5 个门控参数） | ②旁路 repeat（on → 旁路） | first | cluster | 门控压死 vs 整模块旁路 |
|---|---|---|---|---|---|
| 42 | 5/5 至少非零（4 个 10/10，`level_weight` 8/10，\|g\|max：read_gate.w 1.41） | 0.8319 → 0.2950（−0.5368） | 0.9906 → 0.8362 | 0.7907 → 0.3476 | 逐位相同 |
| 43 | 5/5 × 10/10（\|g\|max：read_gate.w 2.85） | 0.8120 → 0.2880（−0.5240） | 0.9969 → 0.8689 | 0.7706 → 0.3074 | 逐位相同 |

门控没有自己关死：`last_gate` 0.7394 / 0.4900（初值 ≈0.12），`read_gate_bias` −1.927 / −1.934（初值 −2.0）。

## 4. P1–P5 判定

- **P1 过**：Δ repeat +0.4722 / +0.4298，均值 +0.4510 ≥ +0.20（两个 seed 单独都 ≥ 0.43；历史 +0.469，新语料下几乎无衰减）。
- **P2 过（紧）**：两 seed 同为正号；极差 0.0425 ≤ 0.05 —— 过线但只剩 0.0075 余量，n=2 的极差本就不稳。
- **P3 过**：`first_mention_acc` 是**上升** +0.1235 / +0.1132（下降 = −0.12 ≤ 0.05）。
- **P4 过（两条齐）**：① 5 个门控参数 `grad` 均非零（seed43 全部 10/10；seed42 `level_weight` 8/10、其余 10/10）；② 两种旁路（模块不前向 / 读门控 −50）都让 `repeat` 掉 0.53 / 0.52、`cluster_f1` 掉 0.44 / 0.46，且两种旁路结果逐位相同。
- **P5 过（如实报）**：参数 631,306 → 631,306 + **264**；每步 109.0/108.9 → 123.2/123.2 ms，**Δ +14.2/+14.3 ms = +13.1%**（历史口径 +27.7%~34%，疑似 O(1) gather 优化已落地，未对账）；峰值显存 **+32MB**（表 64×9×12288×4B = 28.3MB = 0.026GB，与实测对上账）；墙钟 +17.0/+17.2s。
- **总判定：P1–P5 全过** —— 在新语料（person 真实语料 29.7%）上 NDB 的决定性结论**重验成立**，2 seed 配对。

## 5. NDB 写入语义的代码指认（只读，未改 `src/`）

- **评估/推理口径成立**：`src/dtseek/tasks/runtime.py:356-359`（`_rollout`）写入的 `s`/`cls` 来自 `start_logits.argmax` / `cls_logits.argmax`，即**模型自己发射的切片**；注释 `runtime.py:336-338`「记忆里存的全是模型自己发过的切片，没有任何真值」。推理引擎同语义：`src/dtseek/tasks/engine.py:187-193`（注释 `engine.py:188`）。
- **训练口径按字面不成立（但无泄漏）**：`runtime.py:143-146` 写的是**真值** `t_starts[:,s]`/`t_labels[:,s]`；读在写之前（`runtime.py:129-133`，注释 `:130`「此刻表里只有更早提及的绑定」），本步答案不喂给自己；写只在 `write_enabled()` 内生效（`mention_ndb.py:221-228`、`mention_ndb.py:287-288`），验证路径只 `reset` 从不写（`runtime.py:210-211`），`_rollout` 每次评估重建空表（`runtime.py:344-345`）→ 训练记忆绝不带入评估。
- 表是 no_grad 数据统计量、不进 `state_dict`（`mention_ndb.py:131-133`、`:289`）；可学习参数只有三组门控共 264 个（`mention_ndb.py:152-159`）。
- 结论：**「只回写模型自己发射的切片」在评估/推理路径逐字成立；训练路径写的是真值绑定（教师强制）**。若父任务把「绝不写标签真值」当全局不变式，当前实现不满足，需裁决改判据还是改设计。

## 6. 机制、遗留不确定项、证据强度

- 机制：base 臂 `repeat` 0.360/0.382 ≈ 无记忆水平（历史 0.375）→ 新语料没有改变「解码头无处存『我见过谁』」这一失败机制；NDB 用离散字面地址把文档局部绑定捡回来（`id_acc` +0.21~0.24、`cluster_f1` +0.39~0.44）。旁路后 NDB 训练的解码器比 base 臂还差（0.295/0.288 < 0.360/0.382）→ 解码器与记忆**协同适应**，记忆已成强依赖。
- **`exact_match` 不动**（−0.001/+0.006，两臂都 ≈0.30–0.32）：NDB 只修 id 指派，不修漏切片/位置（`slice_recall` 0.58→0.63）。对「整句完全对」的目标，这不是解药。
- 证据强度：**2 seed 配对**——数据集与 val 切分、解码器初始化两臂逐位同源；但 NDB 构造消耗全局 RNG（`train_task_card.py:108`），此后的批序/dropout 轨道两臂分歧（PREREG.md 跑前已声明，历史 literal 配对同瑕疵）。P2 极差由 2 个点算出，估计弱（历史 3 seed 极差 0.011 vs 本次 0.0425）。P4 探针两次都逐位复现训练指标，证据强。
- 遗留：① `--ndb-read pred`（部署口径）未跑，训练查真值起点、评估查预测起点的口径缝仍在；② 指标分母随预测变化（`n_measurable` 3246~3418 vs 3263~3378），Δ 严格说不是同一批提及子集；③ 步时代价与历史 +27.7% 的差异未归因到具体优化；④ 只有 2 seed。
- **顺带发现的 bug（只报告，未改）**：`TaskSpec.to_snapshot/from_snapshot`（`src/dtseek/tasks/plugin.py:219-230`、`:239-249`）不序列化 `identity_labels`/`annotate_all` → `build_card_decoder`（`src/dtseek/tasks/artifacts.py:104-115`）重建的 spec 跑 `evaluate_task` 会**静默丢掉全部身份指标**（repeat/first/cluster/id 变 nan/缺失），是「空指标」陷阱；`scripts/split_checkpoint.py:49-56` 已绕开它，本探针改用注册表 spec 才复现成功。

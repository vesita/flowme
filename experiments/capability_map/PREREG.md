# 预注册：能力可加性筛查 —— 冻结核线性探针能否 rank-order「冻结核 + 旁路卡」的达成率

写入时间：2026-10-06 02:45 +08，**在任何 probe / 训练 / 评测跑起来之前**。
跑完只允许如实报告，不许回改本文件。

## 0. 唯一待答问题

**有没有一个便宜的先验测试，能在训练之前判断「这个能力能不能作为一张卡加在冻结核上」？**

## 1. 假设（H）

**冻结认知核特征上的线性探针（held-out）能 rank-order 各能力在「冻结核 + 旁路卡」下
能达成的水平。**

## 2. 能力集合与三档测量

能力 5 个：`pronoun` / `sentiment` / `relation` / `person` / `negation`。
每能力测三件事，**全部在同一份 held-out 样本上**（见 §3）：

| 档 | 含义 | 来源 |
|---|---|---|
| **A 探针**（便宜先验） | 冻结核 `checkpoints/base_encoder.pt` 的 mean-pool 特征 → 零初始化单层逻辑回归 → `cls_acc` | 本实验跑 |
| **B 冻结核 + 旁路卡** | 复刻 `experiments/additivity/{bypass.py,train_bypass_card.py}` 的配方（r=16, α=16, 16ep×84 步, bs64, lr 1e-3, AdamW wd 1e-4, Cosine, clip 1.0） | 本实验跑（negation 复用已有 ckpt） |
| **C 联合上限** | `checkpoints/arm_neg5_seed{42,43}.pt`（5 卡联合、core 可训、同 task_samples） | **复用已有 ckpt**，只重跑评测以确认逐样本对齐 |

**冻结档（B 的配对下限）**：同配方 `--mode frozen`（旁路在场、冻结在零、只训头），本实验跑。
negation 的 frozen/bypass 两档**直接复用** `experiments/additivity/cards/negation_{frozen,bypass}_base_s{42,43}.pt`
（同配方、同 seed、同 6000 样本 ⇒ 与本实验重跑等价）。

## 3. held-out 评测集（逐样本对齐的唯一定义）

对能力 c、seed S ∈ {42, 43}：

```
raw   = card.build_dataset(task_samples[c], seed=DEFAULT_SEED)   # 确定性
data  = deepcopy(raw); random.Random(S).shuffle(data)
n_val = max(200, len(data) // 10)
eval_S(c)  = data[:n_val]          # ← 探针 / 冻结 / 旁路 / 联合 全部评这一份
train_S(c) = data[n_val:]          # ← 探针与卡的训练都只用这一份
```

这条与 `training/train_multitask.py:103-106` 和
`experiments/additivity/train_bypass_card.py:81-83` 的实现**逐字相同** ⇒
`arm_neg5_seedS` 的 `[eval]`、`train_bypass_card` 的 `[eval]`、本实验探针的评测
落在**同一批样本**上。`task_samples` = pronoun 6000 / sentiment 32000 / relation 6000 /
person 6000 / negation 6000（与 `arm_neg5` 的 `train_args.task_samples` 一致）。

**对齐自检**：`run_eval.py` 会用 arm_neg5 的权重在 `eval_S` 上重算 `cls_acc`，
必须与 `checkpoints/arm_neg5_seedS_metrics.json` **逐位相同**（打印比对结果）；
不同则判「对齐失败」，全部结论作废。
`train_bypass_card` 侧的对齐由 `train_cards.py` 反向重排数据实现（见 §6）。

## 4. 探针规格（跑前冻结）

**特征**：冻结 `checkpoints/base_encoder.pt` 的 `NanoDocEncoder`，`eval()` 前向，
按 `attention_mask` 做 mean-pool → x ∈ R^128。
（与 `experiments/adversarial_routing/probe_e4_style.py` 同一套口径。）

**优化器口径（照抄 E4 探针，不新发明）**：零初始化（weight/bias 全 0）单层逻辑回归，
AdamW lr=0.05、weight_decay=1e-4、200 epoch、batch 64、`torch.manual_seed(S)`。

**A 档 `probe_cls`（首切片类别口径）**
- 输入：样本级 mean-pool x；输出 `nn.Linear(128, C)`（C = 该卡 `spec.num_classes`，含背景类 0）；
- 标签：`GenericTaskDataset` 的 `labels[:, 0]`（首切片类别，背景句 = 0）；
- 训练集：`train_S` **全量**；
- 主报告口径 `probe_cls_acc`：**只在首切片标签 > 0 的样本上**算准确率 ——
  这与 `evaluate_task` 里 `cls_acc` 的分母（`real = t_labels > 0`）**完全一致**，故可比；
- 附报 `probe_acc_all`（含背景句的全量准确率）与 `probe_bg_acc`（背景句判成 0 的比例）。

**B 档 `probe_exact`（整句 exact 口径）**
- 逐位置线性探针：`nn.Linear(128, C+2)`，前 C 维 = 该位置的切片类别（0 = 不在任何切片内），
  后 2 维 = 「该位置是否切片起点」；
- 标签由 `GenericTaskDataset` 的 `labels/starts/ends/step_mask`（**clamped 后的**）铺成
  长度 max_len 的逐位置目标，与 `_truth_of` 用的是同一套 clamped 值；
- 损失 = 逐位置 CE(类别) + CE(起点)，两段各自按**训练集逆频率**加权（防止全 0 退化解）；
  只在 `attention_mask` 有效位上计损；
- 解码：`start=1 且 label>0` 开新跨度，同类别且 `start=0` 连续延长；
- 指标 `probe_exact`：预测跨度多重集与真值多重集完全相等的比例
  （与 `evaluate_task` 的 `exact_match` 同一条规则）；
- 训练集上限 **6000 样本**（逐位置特征是 N×64×128，内存所限；A 档不受此限）。

**两个口径不许混**：`probe_cls_acc` 对标卡的 `cls_acc`，`probe_exact` 对标卡的
`exact_match`；主表只用 `probe_cls_acc`。

**探针空测试自检（三条，缺一条结论作废）**
1. 打印探针**真实可训参数量**（直接数 `requires_grad` 的 numel），并与 `128×C+C` / `128×(C+2)+(C+2)` 的闭式值对账；
2. **零初始化 ⇒ 第 0 步的预测必须恒定**：训练前全部样本预测同一个类（零权重零偏置 ⇒ argmax 恒为 0），
   打印 `probe_untrained_pred_set`；
3. **训练后权重必须离开初值**：`|W|_max > 0` 且 `W` 的 std > 0，打印 `W_max/W_std` 与零初值对照。

## 5. 判据（跑前写死）

**P1（筛查可用）**
- 达成率 `gain = (cls_旁路 − cls_冻结) / (cls_联合 − cls_冻结)`；
- **分母排除规则**：`cls_联合 − cls_冻结 < 0.05` 或 `≤ 0` ⇒ 该能力**排除并说明**
  （0.05 ≈ 本项目实测最大跨 seed 噪声带 2.83pt 的 ~2 倍，dev-notes/12 §12.5）；
- 对保留下来的能力算 Spearman ρ(probe_cls_acc, gain)；
- **可用** ⇔ `|ρ| ≥ 0.8` **且** `ρ > 0`；
- **可评估性下限**：保留能力数 `n_valid ≥ 3` 才计算 ρ 并判定；
  `n_valid ≤ 2` ⇒ 判 **「P1 不可评估（样本量不足）」**，如实报，不得改判据凑结论。

**P2（多 seed）**：seed 42 / 43 各自独立算一遍；**两 seed 结论同号**才下定论，单 seed 只报方向。

**P3（OOD 必配）**：任何点估计必须配 OOD：
- **negation OOD（委托指定）**：`experiments/compose_ops/dataset.json` 里 `set=="negation"` 的
  **60 条**「词典外否定表达」。金标由 `negation/dataset.extract_negation_spans()` 产出；
  同时把这 60 条按其自带 `label`（悲伤=3 / 愤怒=2）与 `cue` 跨度交给 **sentiment** 档做 OOD。
- **全能力通用 OOD（bonus）**：各卡 `probe_units()` 的载体句
  （dev-notes/06 §7：探针载体与训练载体句式不重合）—— 对 pronoun/relation/person/sentiment/negation
  都能取到；金标 = `expected_class`（**只评 A 档口径，不评 exact**，因为 `ProbeUnit` 不带跨度）。
- 已知教训：分布内探针高分可能是**语体捷径**（dev-notes/14 §6：路由 98.83% → 对抗集 9.19%）。
- 判读规则（跑前写死）：OOD 相对分布内下降 **> 20pt** ⇒ 该探针点是语体捷径，**不可作为筛查依据**；
  下降 ≤ 10pt ⇒ OOD 成立；介于两者 ⇒ 证据不足。

**空测试自检三条（卡侧，复用 `train_bypass_card.py` 的 SELFTEST_1/2/3 原始打印）**
1. 生效后可训参数量（旁路 16,384 + 头）；
2. 旁路零初始化 ⇒ 编码器输出与无旁路逐位一致（`max|Δ| = 0`）；
3. 训练后旁路 B 离开零初值（`B_std_final > 0`，全部 4 个插入点）。

## 6. 运行清单（跑前冻结）

| 步 | 内容 | 产出 |
|---|---|---|
| 1 | `prepare.py`：构建/缓存 5 份数据集，导出 `splits.json` | `cache/*.pkl`、`splits.json` |
| 2 | `run_train.sh`：4 老卡 × {frozen, bypass} × {42,43} = **16 跑** | `cards/*.pt`、`logs/capmap_*.log` |
| 3 | `run_probe.py`：A/B 两档探针 × 5 能力 × 2 seed + OOD | `probe.json` |
| 4 | `run_eval.py`：冻结 / 旁路 / 联合 三档 × 5 能力 × 2 seed 的 `cls_acc` + 对齐自检 | `eval.json` |
| 5 | `summarize.py`：主表、Spearman、P1 判定 | `summary.json`、报告 |

**数据缓存**：sentiment `build_dataset` ~4 分钟，只构建一次并 pickle 到
`experiments/capability_map/cache/`；`train_cards.py` 通过给任务卡单例打 `build_dataset`
补丁复用缓存（**不改** `src/` / `training/` / `experiments/additivity/` 下任何文件），
并把 `raw` 重排成「`random.Random(S).shuffle` 之后 `data[:n_val] == eval_S`」的顺序，
使 `train_bypass_card.py` 内部那次 shuffle 后的 val 恰好等于 §3 定义的 `eval_S`。

**运行纪律**：全部用 `systemd-run --user --unit=dtseek-capability-map --collect
--property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0
--setenv=PYTHONUNBUFFERED=1` 启动；只写 `experiments/capability_map/`、`logs/`、`/tmp`；
不 `git commit/stash/checkout/restore/clean`；GPU 上排队等待，不杀他人进程。

## 7. 结果 → 结论的映射（跑前写死）

| 结果 | 结论 |
|---|---|
| P1 过（两 seed 同号） | 探针可当「加卡前筛查」：训练前用 `probe_cls_acc` 预筛能力 |
| P1 不过（ρ < 0.8 或反号） | **如实报「没有便宜筛查」**，并给机制（探针测的是「线性可读性」，旁路卡需要的是「低秩残差可重排性」，两者不等价） |
| P1 不可评估（n_valid ≤ 2） | 如实报「判据在此能力集上不可评估」，并说明分母塌缩本身就是一条结论：**老能力已被核掌握，新能力只有一个样本点** |
| P3 OOD 掉 > 20pt | 分布内点估计是语体捷径，**筛查不可信**（不论 P1 数值如何） |

# 能力可加性筛查：冻结核线性探针能不能在训练前预测「能不能加卡」

预注册 `experiments/capability_map/PREREG.md`（02:45，跑前冻结）；产物 `probe.json` / `eval.json` / `summary.json` / `exploratory.json`。

## 1. 探针口径

| | A 档 `probe_cls`（首切片类别） | B 档 `probe_exact`（整句 exact） |
|---|---|---|
| 特征 | 冻结核 `checkpoints/base_encoder.pt` 的 `doc_memory` 按 `attention_mask` mean-pool → R^128 | 同一个冻结核的**逐位置** `doc_memory[t]`（R^128 × L） |
| 分类器 | `Linear(128, C)`，weight/bias **零初始化**；AdamW lr=0.05 wd=1e-4，200 epoch，bs64（照抄 E4 探针口径） | `Linear(128, C+2)`：前 C 维=该位切片类别（0=不在任何跨度内），后 2 维=是否跨度起点；逐位置 CE + CE(起点)，逆频率加权 |
| 训练样本 | `train_S` **全量**（pronoun 5400 / sentiment 28800 / relation 6480 / person 5400 / negation 5400） | `train_S` 抽 **≤6000**（逐位置特征 N×64×128 的内存上限，PREREG §4） |
| 评测样本 | `eval_S`（见下） | 同一份 `eval_S` |
| 指标分母 | **只数首切片标签 > 0 的样本** —— 与 `runtime.evaluate_task` 里 `cls_acc` 的 `real = t_labels > 0` 逐字相同 | 整句跨度多重集与 `_truth_of` 完全相等（与 `exact_match` 同一条规则） |

**`eval_S` 的唯一定义（逐样本对齐的根）**：`card.build_dataset(task_samples, seed=DEFAULT_SEED)` → `random.Random(S).shuffle` → 前 `max(200, n//10)`。
这与 `training/train_multitask.py:103-106`、`experiments/additivity/train_bypass_card.py:81-83` 逐字相同。
**实测对齐证据**：10 组（5 能力 × 2 seed）的 `n_cls/n_bg` 与 `arm_neg5_seedS_metrics.json` **全部相同**（409/191、2370/830、514/206、417/183、407/193 / 424/176、2373/827、526/194、410/190、382/218），且 `eval_cards.py` 用 arm_neg5 权重重算后 `[ALIGN] all_ok=True`（`cls_acc/n_cls/n_bg/exact_match` 四项逐位相等）⇒ 探针与三档卡评在**同一批样本**上。

**为什么 A 档与卡的 `cls_acc` 可比**：卡的首步是「BOS 查询对全段做交叉注意力后 argmax 类别」，探针是「全段 mean-pool 后 argmax 类别」——两者都在**不给任何指针**的条件下回答同一个问题（这段话里第一个切片是哪一类），且分母同为「有切片的样本」。差别只在池化方式（可学习注意力 vs 均值），所以探针分数系统性低于卡（见主表）。

## 2. 主表（`cls_acc` 口径；2 seed）

| 能力 | seed | 探针 | 冻结 | 旁路 | 联合 | 达成率 | 分母(联合−冻结) | 判定 |
|---|---|---|---|---|---|---|---|---|
| pronoun | 42 | 0.8778 | 0.9927 | 0.9951 | 0.9878 | −0.5000 | −0.0049 | 排除(分母≤0) |
| pronoun | 43 | 0.9481 | 0.9929 | 0.9953 | 0.9811 | −0.2000 | −0.0118 | 排除(分母≤0) |
| sentiment | 42 | 0.6705 | 0.8122 | 0.8346 | 0.7093 | −0.2172 | −0.1030 | 排除(分母≤0) |
| sentiment | 43 | 0.6818 | 0.8230 | 0.8496 | 0.6764 | −0.1810 | −0.1466 | 排除(分母≤0) |
| relation | 42 | 0.8872 | 0.9533 | 0.9611 | 0.7665 | −0.0417 | −0.1868 | 排除(分母≤0) |
| relation | 43 | 0.9335 | 0.9525 | 0.9715 | 0.9087 | −0.4348 | −0.0437 | 排除(分母≤0) |
| person | 42 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | 0.0000 | 排除(分母<0.05) |
| person | 43 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | 0.0000 | 排除(分母<0.05) |
| **negation** | 42 | 0.8108 | 0.8870 | 0.8993 | 1.0000 | **0.1087** | **+0.1130** | 保留 |
| **negation** | 43 | 0.8770 | 0.8770 | 0.9058 | 0.9869 | **0.2619** | **+0.1099** | 保留 |

三档来源：探针本实验跑；冻结/旁路 = 4 老卡 × {frozen, bypass} × 2 seed 共 16 跑（`train_bypass_card.py` 同配方，只换 `build_dataset` 为逆置换排列）；negation 两档复用 `experiments/additivity/cards/negation_{frozen,bypass}_base_s{S}.pt`（其 `[eval]` 0.8870/0.8993、0.8770/0.9058 与本实验重算逐位相同）；联合 = **复用** `checkpoints/arm_neg5_seed{42,43}.pt`。

**口径 B（整句 `exact_match`，与口径 A 不同、不许混）**

| 能力 | seed | 探针 exact | 冻结 | 旁路 | 联合 |
|---|---|---|---|---|---|
| pronoun | 42/43 | 0.9033 / 0.8483 | 0.9533 / 0.9800 | 0.9717 / 0.9883 | 0.9333 / 0.9300 |
| sentiment | 42/43 | 0.2153 / 0.1584 | 0.7937 / 0.8147 | 0.8191 / 0.8403 | 0.7100 / 0.7034 |
| relation | 42/43 | 0.4375 / 0.3903 | 0.9639 / 0.9653 | 0.9694 / 0.9764 | 0.7986 / 0.9194 |
| person | 42/43 | 0.2950 / 0.3133 | 0.3100 / 0.3233 | 0.3117 / 0.3200 | 0.3133 / 0.3217 |
| negation | 42/43 | 0.0700 / 0.0550 | 0.6933 / 0.6500 | 0.7267 / 0.7433 | 0.9667 / 0.9233 |

两个口径差得很远（如 negation：`cls` 0.81 vs `exact` 0.07），原因是 A 档只考「是哪一类」，B 档还要求**逐位置定位全对**——卡的解码头是自回归+指针（能全局约束发射），逐位置线性探针一错就整句废。**B 档不可与卡的 `cls_acc` 对读。**

## 3. OOD（P3）

**① 各卡 `probe_units()` 载体（与训练载体句式不重合，只评口径 A）**

| 能力 | seed | 分布内 | OOD | Δpt | 判读(跑前规则: >20pt=语体捷径) |
|---|---|---|---|---|---|
| pronoun | 42 / 43 (n=36) | 0.8778 / 0.9481 | 0.6389 / 0.6111 | **−23.9 / −33.7** | **语体捷径** |
| sentiment | 42 / 43 (n=1194) | 0.6705 / 0.6818 | 0.8116 / 0.8174 | +14.1 / +13.6 | OOD 成立（OOD 反而更容易） |
| relation | 42 / 43 (n=2156) | 0.8872 / 0.9335 | 0.8228 / 0.8845 | −6.4 / −4.9 | OOD 成立 |
| negation | 42 / 43 (n=66) | 0.8108 / 0.8770 | 0.8939 / 0.8636 | +8.3 / −1.3 | OOD 成立 |
| person | — | — | — | — | **未构造**：person 卡 `probe_units()` 返回 0 个单元 |

**② compose_ops 的 60 条「词典外否定表达」（实取 58，2 条被 negation 卡三条 fail-closed 不变量丢弃）**

| 能力 | seed | 口径 | 分布内 | OOD | Δpt | 判读 |
|---|---|---|---|---|---|---|
| negation 探针 | 42/43 | cls | 0.8108 / 0.8770 | **1.0000 / 1.0000** | +18.9 / +12.3 | ⚠ **口径退化**：58/58 全含标记（n_real=58），`cls_acc` 只等于「探针开火率」，无判别力 |
| negation 探针 | 42/43 | exact | 0.0700 / 0.0550 | 0.0000 / 0.0000 | −7.0 / −5.5 | 绝对值 0：探针**定位**在 OOD 载体上全灭 |
| sentiment 探针 | 42/43 | cls | 0.6705 / 0.6818 | 0.3103 / 0.3276 | **−36.0 / −35.4** | **语体捷径** |
| sentiment 探针 | 42/43 | exact | 0.2153 / 0.1584 | 0.0172 / 0.0000 | −19.8 / −15.8 | 证据不足（绝对值近 0） |
| sentiment 卡 | 42 | cls/exact | 0.8122 / 0.7937 | frozen 0.2586/0.1207 · bypass 0.3621/0.1897 · joint 0.3793/0.2069 | — | 卡自己也崩到 0.26–0.38 ⇒ 情绪能力整体靠词典捷径 |
| sentiment 卡 | 43 | cls/exact | 0.8230 / 0.8147 | frozen 0.3276/0.1379 · bypass 0.3966/0.1724 · joint 0.1897/0.1552 | — | 同上，两 seed 同号 |
| negation 卡 | 42 | cls/exact | 0.8870 / 0.6933 | frozen 0.9828/0.5862 · bypass 0.9828/0.6379 · joint 1.0/0.9310 | — | cls 不掉（集合偏正例所致），exact 掉 10.7/8.9pt |
| negation 卡 | 43 | cls/exact | 0.8770 / 0.6500 | frozen 0.9828/0.4655 · bypass 0.9828/0.5172 · joint 1.0/0.9138 | — | 同上 |

## 4. P1 判定

- 保留能力数 `n_valid = 1`（**只有 negation** 分母 ≥ 0.05 且 > 0）；pronoun / sentiment / relation 两 seed 分母全为**负**，person 恰为 0。
- ⇒ **Spearman 无法计算**（seed 42 与 43 都是 `n_valid=1, rho=None`）。
- 按 PREREG §5「可评估性下限 `n_valid ≥ 3`」⇒ **P1 不可评估**；按 §7 映射，这不是「筛查可用」。
- 另有**独立的 P3 否决**：pronoun 探针点在 `probe_units` 上掉 23.9/33.7pt、sentiment 探针点在 compose60 上掉 36.0/35.4pt，均超 20pt 线 ⇒ **这两个探针点本身被判为语体捷径，不可作为筛查依据**（不论 P1 数值如何）。

> **结论：没有便宜筛查。** 在当前 5 个能力上，预注册判据不可评估；且探针点中 2/5 已被 OOD 规则独立否决。

**Post-hoc 探索（跑后追加，不参与 P1 判定，`exploratory.json`，n=5）**

| seed | ρ(探针, 冻结) | ρ(探针, 旁路) | ρ(探针, **旁路增量**) | ρ(探针, 联合) | ρ(探针exact, 冻结exact) |
|---|---|---|---|---|---|
| 42 | +0.90 | +0.90 | **−0.90** | +0.46 | +1.00 |
| 43 | +1.00 | +1.00 | **−0.90** | +0.70 | +1.00 |

## 5. 能力边界结论

**核已具备，可直接加卡（4 个，两 seed 同号）**：`pronoun` / `sentiment` / `relation` / `person`。
判据是**分母 ≤ 0**：冻结核 + 只训头**已经 ≥ 5 卡联合训练的上限**（如 sentiment 冻结 0.812/0.823 vs 联合 0.709/0.676；relation 0.953/0.953 vs 0.767/0.909）。这四张卡没有「可提升空间」需要旁路去填，旁路只额外给 +0.00~+2.7pt（pronoun +0.24、relation +0.78/+1.90、sentiment +2.24/+2.66、person 0.00）。**加卡零风险、也无需改核** —— 与 dev-notes/15 §8 的「老卡零损伤」互为印证。
⚠ 口径提醒：`person` 的 `cls_acc` 三档全 1.0（首提及 id 可靠出场顺序推出），但 `exact_match` 只有 0.31 ⇒ **首切片类别饱和 ≠ 该能力到位**，person 的瓶颈在跨度与 id，分类口径对它没有信号。

**核不具备，必须改核（1 个）**：`negation`。
`联合 − 冻结 = +11.30 / +10.99pt`；旁路只收回 +1.23 / +2.88pt ⇒ 达成率 **0.109 / 0.262**（两 seed < 0.27）。与 dev-notes/15 §8「rank 16→128 只 +1.2pt」一致：缺的不是适配容量，是核里没有该表征。

**机制（推断，非实测）**：探针对**绝对水平**的排序很稳（post-hoc ρ(探针, 冻结)=0.90/1.00），对**旁路增量**的排序却是**反的**（ρ=−0.90/−0.90）。原因：「探针分数高」与「核已具备该能力」是同一件事的两个投影 —— 它预测的是 `frozen`，而达成率的分母是 `joint − frozen`；当核已具备时分母塌到 0，探针再准也无从排序。**探针不是关于「旁路可补性」的独立信息。**

## 6. 空测试自检（原始输出）

**卡侧（`train_bypass_card.py` 的 SELFTEST，16 跑全过；抽 4 张 bypass 跑的 s42）**

```
SELFTEST_2a max|Δ|(bypass开@零 vs 关)=0.000e+00 max|Δ|(bypass开@零 vs 无旁路基座)=0.000e+00 B全零=True
SELFTEST_1 trainable_report(mode=bypass)={"base_core_total":1688460,"base_core_frozen":1688460,"base_core_trainable":0,"head_trainable":630278,"bypass_total":16384,"bypass_trainable":16384,"trainable_total":646662}   # relation 头 630021 / person 631306，旁路恒 16384
SELFTEST_3 bypass_drift blocks[0] B_std_init 0.0 → B_std_final 0.025990 (A_max|Δ|=0.11193) | blocks[1] 0.0→0.025436 | blocks[2] 0.0→0.022819 | norm 0.0→0.024856
SELFTEST_3_verdict B全部离开零初值=True
# frozen 档（旁路在场、冻结在零）：bypass_stats_final 四点 B_std 全 = 0.0，selftest_3_b_moved=False（预期）
```

**探针侧（`probe.py`，自检 1/2/3，s42 全 5 能力）**

```
A1 可训参数量 = 闭式值 match=True：pronoun 516=128×4+4 · sentiment 516 · relation 387=128×3+3 · person 1032=128×8+8 · negation 258=128×2+2
A2 训练前 untrained_pred_set=[0]（零权重零偏置 ⇒ argmax 恒 0）
A3 训练后 W_max/W_std：pronoun 8.2085e+00/2.3502e+00 · sentiment 8.9250e+00/1.6035e+00 · relation 1.3554e+01/3.2155e+00 · person 1.7643e+00/3.6824e-01 · negation 7.2617e-01/2.8453e-01（初值全 0）
B1 可训参数量 = 闭式值 match=True：pronoun 774=128×6+6 · sentiment 774 · relation 645 · person 1290=128×10+10 · negation 516=128×4+4
B2 训练前 untrained_lab_set=[0] untrained_start_set=[0] decoded_spans=0
B3 训练后 W_max/W_std：pronoun 1.9236e+01/3.8771e+00 · sentiment 1.4839e+00/3.2811e-01 · relation 2.5356e+00/5.7298e-01 · person 8.8653e+00/1.4443e+00 · negation 6.3456e-01/1.8650e-01
```

## 7. 命令 / unit / 日志 / 产物

全部 `systemd-run --user --unit=<名> --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 <脚本>`：

| unit | 脚本 | 结果 |
|---|---|---|
| `dtseek-capmap-prep` | `experiments/capability_map/run_prepare.sh`（→ `prepare.py`） | rc=0，3min55s，数据集缓存 `cache/*.pkl` + `splits.json` |
| `dtseek-capmap-train` | `run_train.sh`（→ `train_cards.py` → `train_bypass_card.py`） | rc=0，20min18s，16/16 卡 |
| `dtseek-capmap-probe` | `run_probe_eval.sh`（→ `probe.py` → `eval_cards.py` → `summarize.py`） | probe rc=0（5min06s）；eval/summarize 首轮因联合档 decoder 取法报错，修 `_dec_from_joint` 后**手工重跑 rc=0** |

日志：`logs/capmap_{pronoun,sentiment,relation,person}_{frozen,bypass}_s{42,43}.log`（16 条）、`logs/capmap_probe.log`、`logs/capmap_eval.log`、`logs/capmap_summary.log`；negation 复用档对应 `logs/additivity_{f,b}_base_s{42,43}.log`。
产物：`experiments/capability_map/{PREREG.md, splits.json, probe.json, eval.json, summary.json, exploratory.json, cards/*.pt(16), cache/}`。

## 8. 遗留与不确定

**实测**
1. `n_valid=1`：4/5 能力分母 ≤ 0 —— 不是判据写错，是**冻结核在这 4 个能力上已达/超联合上限**（2 seed、同 eval_S、align 全过）。
2. 探针便宜但**不是数量级便宜**：单 (能力, seed) 23–53s（sentiment 53s）vs 卡单跑平均 76s ≈ 1.5–3×；可训参数 258 vs 646,662（≈1/2500）。本项目基座仅 1.69M 参数，「训练卡很贵」这个前提在这里不成立。
3. negation 的 compose60 `cls` 口径**退化**（58/58 全正例），只能读作「探针开火率」；该集上的 `exact` 三档卡也都掉 9–21pt。
4. person 无 `probe_units` ⇒ 该能力**没有任何 OOD**，其探针点（1.0000，三档卡也 1.0）是无 OOD 保护的饱和点。
5. 联合档 `arm_neg5` 是**从随机初始化**的 5 卡联合，与冻结档（multitask_v2 拆分核）**不同谱系**（dev-notes/15 §8 已记录此口径差异，本实验沿用未改）。
**推断（未测）**
6. 若把联合档换成「从 `base_encoder` 热启动的 5 卡联合」，sentiment 的分母**可能**转正（当前 joint 0.709 < frozen 0.812 部分源于 5 卡联合损害 sentiment，dev-notes/13.1），但 pronoun/relation/person 三档均在 0.95–1.0，分母大概率仍 ≈0 ⇒ `n_valid` 仍 < 3；**要让 P1 可评估需要 ≥3 个「核里确实没有」的能力（目前只有 negation 一个）**——这是下一步的实验设计，不是本次结论。**均未跑，不能当结论。**

**边界**：本实验只写 `experiments/capability_map/`、`logs/`、`cache/`（在 `experiments/capability_map/` 下）；`git status --porcelain` 无任何已跟踪文件被修改；未执行任何 git 写操作。

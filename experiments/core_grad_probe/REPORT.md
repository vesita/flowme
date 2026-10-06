# core_grad_probe —— select CE 与 4 个老任务对**核**的梯度，谁主导？

> 补 `experiments/select_semantic_joint/REPORT.md` **遗留第 10 项**。判据与取值点见 `PREREG.md`
> （mtime **2026-10-06 22:40:14**，早于正式跑 **22:53:51**，跑后未改）。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出。seed **42**，单 seed。

## 1. 取值点（跑前写死）与同构性验证（实测）

取值点：**step 1 / 50 / 200**（1-based optimizer step；step1 = 主臂 `loss_first` 那一批、
step50 = 主臂「前 50 均值」窗口末点、step200 = 早期后再取一点）+ 一个**非轨迹点** `final`
= 主臂 `weights/joint_s42.pt`（1800 步终态）+ 本循环下一批，**不跑第 1800 步**。步数上限 **200**。

**同构怎么做的**：复用 `select_semantic_joint/common.py`（只读 import）的 `inputs_for / old_loss /
SemModel.forward_tokens` 与 `train_sem.IndexDataset`，**不另写前向**；`CosineAnnealingLR.T_max=1800`
（不是 200，否则前 200 步 LR 就不同）；探针只在 `total` 组装完、`opt.zero_grad()` **之前**插
`torch.autograd.grad(retain_graph=True)`（不写 `.grad`、不耗 RNG、不动 optimizer/scheduler）。

| 对账项 | 本探针 | 主臂 `joint_s42.json` | Δ |
|---|---|---|---|
| R4 step-0 四卡 sha256 | `d2ea73d8…f346` | `d2ea73d8…f346` | **逐位相同 ✅** |
| `loss_first` | 0.6930022239685059 | 0.693002 | **2.24e-07** |
| `loss_select_first50` | 0.35385904539376495 | 0.353859 | **4.54e-08** |
| `loss_total_first50` | 34.08201137542724 | 34.082011 | **3.75e-07** |
| 数据 md5（跑前） | 6b1e6fb1… / d3d597af… / 2a5ca5f4… | REPORT §6 同 | 相同 ✅ |
| 终态权重 `core_drift` | 0.125432 | 0.125432 | **0.00e+00** |

形状/精度（`[shapes]` 实测）：select ctx `[64,64,128]`、cand `[64,2,32,128]`、老任务 `input_ids [64,120]`、
batch **64**、`param/logit dtype = float32`、`device=cuda`、`lr_core=3e-4 / lr_head=1e-3`。
自检（已知答案）：cos(v,v)=1、cos(v,−v)=−1、cos(e1,e2)=0、‖(0,..,5,..)‖=5 全过；
**可加性残差 ‖g_total−(g_select+g_old)‖/‖g_total‖ = 2.7e-08 ~ 3.3e-08 ≤1e-4 ✅**（4 点全过）。
可复现性：冒烟跑（22:52，独立进程）与正式跑 step1 各量**逐位相同**（`|g_sel|=1.583992e-01`、`|g_old|=8.462594e+01`）。

## 2. 主表（θ_core = encoder 1,688,460 参数；2-范数，float64 累加）

| 点 | ‖g_select‖ | ‖g_old‖（合并） | g_pronoun | g_sentiment | g_relation | g_person | **cos(sel,old)** | ‖θ_core‖ | ‖g_sel‖/‖θ‖ | **比值 r** |
|---|---|---|---|---|---|---|---|---|---|---|
| step 1 | 1.584e-01 | 8.463e+01 | 1.30e+00 | 5.24e+01 | 1.31e+01 | 6.53e+01 | **−0.0148** | 56.17 | 2.82e-03 | **0.00187** |
| step 50 | 4.766e+00 | 3.545e+02 | 1.63e+01 | 2.23e+01 | 4.93e+01 | 3.50e+02 | **+0.0146** | 56.21 | 8.48e-02 | **0.01344** |
| step 200 | 3.327e-01 | 9.406e+01 | 4.10e+00 | 1.80e+01 | 6.50e+01 | 6.86e+01 | **+0.0044** | 56.33 | 5.91e-03 | **0.00354** |
| final（1800 步权重） | 6.326e-03 | 1.949e+01 | 1.60e-01 | 3.40e+00 | 4.33e-01 | 1.92e+01 | **−0.0008** | 56.81 | 1.11e-04 | **0.00032** |

- **分别求 vs 合并求（两个都报）**：`Σ‖g_old_i‖` = 1.321e+02 / 4.378e+02 / 1.558e+02 / 2.319e+01，
  合并 `‖g_old‖` 见表 ⇒ **抵消系数 0.64 / 0.81 / 0.60 / 0.84**（老任务之间大体同向，只抵消 16~40%）。
  用「分别求的和」当分母，r 更小：0.0012 / 0.0109 / 0.0021 / 0.00027 —— 结论不随口径变。
- **loss 值**（同批）：`L_old_sum` = 17.49 / 15.08 / 15.94 / 11.02，`L_select` = 0.693 / 0.0678 / 0.00193 / 1.3e-05。
- `‖g_total‖ ≈ ‖g_old‖`（select 占比 <1.4%）⇒ clip 1.0 作用在由老任务决定的总梯度上。
- 参考量：`‖g_old‖/‖θ_core‖` = 1.51 / 6.31 / 1.67 / 0.34（无 clip 时单是老任务就已是「大步」量级）。

## 3. 按层拆解（ratio = ‖g_select‖_layer / ‖g_old‖_layer）

| 层（参数量） | step1 | step50 | step200 | final | 层内 cos 范围 |
|---|---|---|---|---|---|
| embedding（1,048,576） | 0.00203 | **0.02011** | 0.00423 | 0.00033 | −0.002 ~ +0.016 |
| blocks.0（213,252） | 0.00183 | 0.01373 | 0.00337 | 0.00034 | −0.017 ~ +0.016 |
| blocks.1（213,252） | 0.00165 | 0.00756 | 0.00227 | 0.00030 | −0.034 ~ +0.031 |
| blocks.2（213,252） | 0.00148 | 0.00635 | 0.00168 | 0.00031 | −0.055 ~ +0.011 |
| norm（128，输出 RMSNorm） | 0.00249 | 0.01096 | 0.00278 | 0.00038 | −0.056 ~ **+0.189** |

- **没有哪一层由 select 主导**：5 组 × 4 点 = **20/20 全部 r < 0.03**，最接近的是 step50 的 embedding（2.0%）。
- 层间只差 ~3×：step50 起**浅层（embedding）相对最强、深层（blocks.2）最弱**，方向一致、没有例外层。
- 本核**没有独立输出投影**：`doc_memory` 出口只有 RMSNorm `norm`（128 参数），打分头不属核（已在 PREREG §2 写死）。
- 老任务里**谁在主导老梯度**（实测）：step1 = person 65.3 ≈ sentiment 52.4；step50 = **person 350（≈‖g_old‖ 354）**；
  step200 = person 68.6 ≈ relation 65.0；final = **person 19.2（占 98%）**。person 的 loss 一直 10.8~15.4，是老梯度的主要来源。

## 4. 结论（判据 = PREREG §4，跑前定死）

**① select 的梯度是「被淹没」。** r = ‖g_select‖/‖g_old‖ = **0.0019 / 0.0134 / 0.0035 / 0.0003**，
**4/4 点 < 0.1**（判据要求 ≥2/3）—— 老任务合并梯度大 **74× ~ 3081×**；分层 20/20 同样 <0.03。不需硬选。

**② 但「淹没」只在原始范数口径成立，两件事必须同时记下（都是实测）**：
- **cos ≈ 0（−0.015 ~ +0.015，分层最大 0.19）⇒ 两者几乎正交、不相互抵消** —— select 不是在跟老任务「对拉」，
  而是在老任务梯度张成的空间之外加一个很小的分量；老任务也不「反对」select。
- select loss 从 0.693 → 1.3e-05、train 100%（主臂实测）⇒ 小范数 ≠ 没起作用；优化器是 AdamW（逐坐标归一
  化 + clip 作用在总梯度上），一致的小分量在 `m/√v` 里不会被简单的量级比直接抹掉。**本测没有量 Adam 更新口径。**

**③ 对「机制归因 = 中」的处置：维持中**（不升级为强）。
- 升级的部分：遗留第 10 项**已用同构实测补上**，这条子结论本身证据是**强**的（同构对账 Δ≤4e-7、R4 逐位、
  自检与可加性残差全过、两独立进程复现、4 点一致）。
- 维持中的原因（推断）：结果**反而给原机制加了张力** —— 核上的原始梯度 98%+ 来自老任务，「增益由 select
  梯度直接驱动核」这条朴素读法不成立；而 R3（随机标签归零）说明老任务数据本身不产生增益。两者合起来是
  「select 信号必要但极小」，**关键缺口仍是遗留第 2 项（缺「只让核学 select、不给老任务」那一档）**，本测没跑那一档。

## 5. 原始命令 / 日志 / 产物 / 遗留

```bash
# 冒烟（22:52:26→22:52:40 exit=0）
systemd-run --user --unit=dtseek-coregrad-smoke --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  bash experiments/core_grad_probe/run_probe.sh smoke
# 正式（22:53:51→22:54:54 exit=0，墙 63.0s / CPU 97.0s / 内存峰值 2.2G，invocation 60ef7124c6e941b393ce25c4596d3280）
systemd-run --user --unit=dtseek-coregrad ... bash experiments/core_grad_probe/run_probe.sh
```
- 日志：`logs/core_grad_probe_s42.log`、`logs/core_grad_probe_smoke.log`；journal：`journalctl --user -u dtseek-coregrad`。
- 产物：`experiments/core_grad_probe/results/grad_probe_s42.json`（主）、`grad_probe_smoke_s42.json`、`PREREG.md`、`probe_grad.py`、`run_probe.sh`。
- 纪律（实测）：`git status` 对 `select_*` / `src/` / `training/` **无条目**，`find -newermt PREREG` 对只读目录**为空**；
  本目录之外零写入；两 unit 均已 inactive（`systemctl --user list-units 'dtseek*'` 为空）；未杀任何进程、无 git 写操作。
- 跑前查过 `ps` 与 `systemctl --user is-active 'dtseek*'`：**无他人任务**，未排队。

**遗留与不确定（实测 / 推断分开）**：
1. 【实测】**单 seed（42）**；seed 43 未测。
2. 【实测】轨迹点只到 **200 步**；200~1800 步之间只用终态权重补了 1 点，**中间过程未逐步量**。
3. 【实测】口径是 **∂L/∂θ 的原始 2-范数**；**Adam 预条件后的有效更新量（‖m/√v‖ 或 Δθ 分解）未测** ——
   这是「被淹没」能否外推到「实际更新也由老任务决定」的唯一缺口。
4. 【实测】终态点是「主臂权重 + 本探针下一批」的**非轨迹点**（数据不是主臂第 1801 批）。
5. 【推断】`‖g_select‖` 在 step1 很小（0.158）可能与打分头随机初始化、其对核特征的 Jacobian 尚弱有关 —— **未验证**。
6. 【推断】老梯度主要来自 person 卡（loss 10.8~15.4 远高于其余三张）⇒ 若要给老任务降权，杠杆在 person —— **未测**。

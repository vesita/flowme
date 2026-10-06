# core_update_probe —— 跑前写死的取值点与判据

> 回答**唯一**问题：在 AdamW 的**实际更新**下，核 θ_core 的一步位移由 select 还是老任务决定？
> 缺口来源：`experiments/core_grad_probe/REPORT.md` 遗留第 3 项（只量了 ∂L/∂θ 原始 2-范数）。
> 本文件 mtime 必须早于正式跑；跑后不改。

## 1. 取值点（跑前写死）

- 1-based optimizer step：**step 1 / 50 / 200**（与 core_grad_probe 同构同点），步数上限 **200**。
- **不取终态点**：主臂 `weights/joint_s42.pt` 只存权重不存优化器状态 ⇒ 1800 步的 (m, v) 不可复原，
  补一个"终态点"只能伪造状态 ⇒ 本测不做（写进遗留）。
- 每个取值点在**同一步、同一 (θ, 真实优化器状态)** 上跑 7 个臂（单步、互不污染、跑完把 θ 精确还原）：
  | 臂 | 梯度 | 优化器状态 | clip |
  |---|---|---|---|
  | `sel` | 只 select CE | 真实历史状态 | 配方 1.0 |
  | `old` | 只 Σ 老任务 | 真实历史状态 | 配方 1.0 |
  | `both` | total（= 主臂口径） | 真实历史状态 | 配方 1.0 |
  | `old_noclip` | 只老任务 | 真实历史状态 | 不 clip（对照） |
  | `sel_reset` / `old_reset` / `both_reset` | 同上三项 | **复位 m=v=0**（只反映本步梯度的 Adam 归一化） | 配方 1.0 |
- 控制（跑前写死，只在 step 1）：
  1. **零梯度步**：全部 grad=None + 真实状态 → AdamW 应**恰 0 位移**（已知答案）。
  2. **SGD 线性控制**：同 (θ, 三梯度)，SGD(lr 同、wd=0、momentum=0) → 可加性残差应 ≈ 0。
     它证明残差公式在"线性优化器"下退化为 0 ⇒ Adam 下非零残差是 Adam 非线性，不是公式写错。
     **阈值修正（2026-10-06 23:14，冒烟后、正式跑前；只改本控制的阈值，§3 判据不动）**：
     冒烟实测该残差 = **8.54e-5**，来源是 **fp32 参数空间量化**（`p − lr·g` 的舍入，
     理论下界 `2^-24·‖θ‖/‖Δθ_sgd‖` ≈ 1.3e-4），不是公式错。故：
     - `sgd_linear_additivity_resid < 1e-3`（且报 fp32 量化下界估计，两者同量级即通过）；
     - 新增更锋利的**梯度层 float64 控制** `grad_additivity_resid_f64 < 1e-5`
       （同一残差公式在梯度张量上直接算，已知答案 ≈0；core_grad_probe 同款量测过 2.7e-8）。
  3. **reset 臂的预条件比值**：Adam 归一化 ⇒ `m/√v = (1-β1)/√(1-β2) · sign(g)`（非零坐标上），
     故**预测** `‖m/√v_sel‖/‖m/√v_old‖ = sqrt(nnz_sel/nnz_old)`（nnz = 梯度非零坐标数，本步现算）。
  4. **clip 对照**：`old_noclip` vs `old`（Adam 尺度不变 ⇒ 两者 Δθ 应几乎相同，报 reldiff）。
  5. **both 臂 = 主臂步**：`both` 臂 Δθ 与紧随其后主循环真实 `opt.step()` 的 Δθ 比 reldiff（应 ≈0）。
  6. cos/‖·‖ 已知答案自检（cos(v,v)=1、cos(v,−v)=−1、cos(e1,e2)=0、‖(0,..,5,..)‖=5）。

## 2. 指标（每个取值点都报）

- **(a) 预条件范数**：真实状态下单步喂 select / 只喂 old 后，读各自 `exp_avg/exp_avg_sq` ⇒
  `‖m/√v_sel‖`、`‖m/√v_old‖`、比值、**分层**（embedding / blocks.0-2 / norm）比值；
  另报 `‖m̂/(√v̂+eps)‖`（= 有效更新，已含 bias correction 与 eps）与 reset 臂同款量。
- **(b) 真实位移**：`‖Δθ_sel‖`、`‖Δθ_old‖`、`‖Δθ_both‖`、`cos(Δθ_sel, Δθ_old)`、
  **可加性残差** `‖Δθ_both − (Δθ_sel+Δθ_old)‖ / ‖Δθ_both‖`（Adam 非线性 ⇒ 跑前预期**不**≈0，如实报）、
  `cos(Δθ_both, Δθ_old)`、`cos(Δθ_both, Δθ_sel)`、
  **净改变量** `rel(both, old) = ‖Δθ_both−Δθ_old‖/‖Δθ_both‖`（= 抹掉 select 当前梯度后实际更新变了多少）、
  `rel(both, sel)`（= 抹掉老任务当前梯度后变了多少）、分层比值、`‖θ_core‖`、clip 实际 scale、
  wd 项占比 `‖lr·wd·θ‖/‖Δθ‖`。
- 假想位移「把老任务权重置 0」= 臂 `sel`（真实状态）与 `sel_reset`（无历史），两个都报。

## 3. 判据（跑前写死）

**A. 「实际更新由谁决定」三选一**（在 ≥2/3 取值点同时成立才算）：
- **由老任务决定**：`cos(Δθ_both, Δθ_old) > cos(Δθ_both, Δθ_sel)` **且** `rel(both, old) < rel(both, sel)`。
- **由 select 决定**：两条同时反向。
- **同量级**：前两条判不出来（互有胜负 / 都不成立），且 `‖Δθ_sel‖/‖Δθ_old‖ ∈ [1/3, 3]`。
- 若出现「方向判老任务、但比值 ∈ [1/3,3]」这类交叉 → **如实报交叉**，不硬选，写明分歧来源。

**B. 「被淹没」能否外推到实际更新**（与 A 独立判）：
- 更新口径比值 `R = ‖Δθ_sel‖/‖Δθ_old‖`（真实状态）与 `R_reset`（复位状态）都报。
- 三点全部 `R < 0.1` ⇒ **能外推**；三点全部 `R ≥ 0.33` ⇒ **不能外推**（Adam 逐坐标归一化抹平量级）；
  其余 ⇒ **部分外推**，报具体数值。
- 注意：`R_reset` 跑前预期 ≈ `sqrt(nnz_sel/nnz_old)` ≈ 1（控制 3）—— 若成立，说明"量级被淹没"
  在 Adam 第一步口径下**原则上就不成立**，A 的方向判据必须靠 cos / 净改变量说话。

**C. 同构性门槛（不过则作废）**：R4 step-0 sha256 逐位相同；`loss_first` / `loss_select_first50` /
`loss_total_first50` vs 主臂 `joint_s42.json` |Δ| ≤ 1e-4；复现 `core_grad_probe` 的
`‖g_select‖/‖g_old‖`（step1/50/200）|Δ| ≤ 1e-3 相对误差。

## 4. 同构怎么保证

只读 import `select_semantic_joint/common.py`（`inputs_for / old_loss / SemModel.forward_tokens`）
与 `train_sem.IndexDataset`，不另写前向；`CosineAnnealingLR.T_max = STEPS = 1800`；`lr_core=3e-4 /
lr_head=1e-3 / wd=1e-4 / clip=1.0 / batch=64 / seed=42`。探针只在 `total` 组装完、`opt.zero_grad()`
**之前**插 `autograd.grad` + 克隆优化器单步（用完把 θ `copy_` 精确还原、`.grad` 清空），
主臂那一臂的 zero_grad/backward/clip/step/sched 与 `train_sem` 逐行同序。

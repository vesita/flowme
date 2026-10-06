# core_grad_probe 预注册（跑前写死）

> 目的：补 `select_semantic_joint/REPORT.md` **遗留第 10 项** —— select CE 与 4 个老任务 loss
> 对**核 θ_core**（encoder，1,688,460 参数）的梯度 2-范数各自多大、谁主导。
> 一次**纯测量**：不调参、不刷指标、不训练到底。

## 1. 取值点（跑前定死，1-based optimizer step 序号）

| tag | step | 为什么选它 |
|---|---|---|
| `step1` | **1** | 初始化点（主臂 `loss_first = 0.693002` 那一批，尚未任何 `opt.step()`） |
| `step50` | **50** | 主臂「前 50 步均值」窗口的末点 ⇒ 与 `loss_select_first50` 对账窗口对齐 |
| `step200` | **200** | 早期之后再取一点，看比值是否随训练漂移 |

另加**一个非轨迹点** `final`：载入主臂产物 `select_semantic_joint/weights/joint_s42.pt`
（1800 步终态）后，在**同一循环的第 201 批数据**上求梯度 —— 用来回答「后期谁主导」，
**不跑第 1800 步**（零额外训练）。

- seed = **42**（主臂两 seed 之一；单 seed，另一 seed 列为遗留）。
- 步数上限 **200**（远小于主臂 1800）；`CosineAnnealingLR` 的 `T_max` 必须用主臂的 **1800**，
  否则前 200 步 LR 曲线就不同 ⇒ 不同构。

## 2. 口径

- `g = ∂L/∂θ_core` 的 **2-范数**；**分别求**（`g_select`、`g_old_i` ×4）与**合并求**
  （`g_old = ∂ΣL_old_i/∂θ_core`）**两个都报**（方向不同 ⇒ 两者不同）。
- **方向余弦** `cos(g_select, g_old)`（合并口径）。
- 层拆分（跑前按 `NanoDocEncoder` 结构定）：`embedding` / `blocks.0` / `blocks.1` / `blocks.2` / `norm`。
  **本核没有独立的输出投影**：`doc_memory` 出口只有 RMSNorm `norm`（128 参数），打分头不属于核。
- 参考量：`‖θ_core‖`、`‖g_select‖/‖θ_core‖`；另报 `Σ‖g_old_i‖` vs `‖g_old‖`（抵消系数）。

## 3. 同构判据（跑前定死，跑后逐条对账）

1. **R4 step-0** 四卡逐样本 sha256 = `select_semantic_joint/results/r4_baseline_s42.json`（逐位相等）
   ⇒ 构造段（RNG 流、权重、老卡头）与主臂完全一致。
2. **loss 对账**：本探针的 `loss_first`、`loss_select_first50`、`loss_total_first50`
   与主臂 `results/joint_s42.json` 比，|Δ| ≤ **1e-4**（ROCm 有 1e-6 级不确定性）。
   对上 ⇒ 数据顺序、批量、精度、LR、优化器全程同构。
3. **测量函数自检**：`cos(v,v)=1`、`cos(v,−v)=−1`、已知向量范数；
   **可加性残差** `‖g_total − (g_select+g_old)‖ / ‖g_total‖ ≤ 1e-4`（total 与两项之和必须同向同模）。
4. 记录实际张量形状/dtype（select 批、老任务批、参数精度）随产物一并报出。

## 4. 结论判据（三选一，跑前定死）

记 `r = ‖g_select‖ / ‖g_old‖`（合并口径），按 3 个轨迹点判定：

| r（3/3 或 ≥2/3 点满足） | 判定 |
|---|---|
| `r ≥ 1` | select **主导** |
| `0.1 ≤ r < 1` | **同量级**（相差 < 10×） |
| `r < 0.1`（≥2/3 点） | **被淹没**（老任务梯度大 ≥10×） |

- 若点间不一致（跨档）⇒ 报「随训练演化」+ 逐点结论，**不硬选**。
- `cos(g_select, g_old)` 只作**补充**：明显 <0 ⇒ 相互抵消（比"谁大"更要紧）；不参与三选一。
- 对「机制归因 = 中」的处置：测量覆盖 3 点 + 终态、同构对账全过 ⇒ 可议升级；有缺口（单 seed、
  200 步以外未逐步量）⇒ 维持中。跑完按实测填，**不预设**。

## 5. 纪律

只写 `experiments/core_grad_probe/`、`logs/`、`/tmp`；`select_semantic_joint/` 等一律**只读**；
启动用 `systemd-run --user --unit=... --collect` + `PYTHONUNBUFFERED=1`；无 git 写操作；不杀他人进程。

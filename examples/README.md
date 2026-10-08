# examples：三类卡闭环 · 合成算术最小可跑例子

**这是什么**：三类卡闭环（**输入卡 → 模因 → 思维卡 → 模因 → 输出卡**）在合成算术上的**最小可跑例子**。
只**复用已入库的 ckpt 与脚本**（`stages/19_residual_scan.py`，exec 其前缀），在 **CPU** 上重算评估，**不训练**。

## 一键跑

```bash
bash examples/run_example.sh      # 退出码 0 = 成功；非 0 = 失败并打印原因
```

**ckpt 位置**：`logs/19_ckpt_<桶>_<res|nores>_seed<seed>.pt`（本例子用 seed1234 的 add_1d / add_3d）。

## ★ 能力边界表（严格 EM，n=800，seed1234，CPU 重算 = 已入库 GPU 值逐位相同）

| 桶 | 无残差 nores | 带残差 res | Δ |
|---|---|---|---|
| add_1d 一位数加 | 96.62% | 96.88% | +0.25pp（未过 2SE，不显著）|
| add_2d 两位数加 | 71.88% | 72.88% | +1.00pp（两 seed 异号）|
| **add_3d 三位数加** | **2.50%** | **37.25%** | **+34.75pp ★（2 seed 同号，Δ/SE=19.9）** |
| sub_2d 两位数减 | 61.38% | 61.00% | −0.38pp（两 seed 异号）|
| mul_2d 两位数乘 | 82.62% | 83.25% | +0.62pp（未过 2SE）|

**三位数加法这一行，残差贡献了 +34.75pp：2.50% → 37.25%**（另 seed：9.62% → 31.13%，同样 +21.50pp）。
其余四桶本来就好（61–98%），加残差**无影响甚至两 seed 异号** ⇒ 残差救的是「深层推理导致的表征/梯度崩溃」，
**不是普遍增益（五桶只有 1/5 显著）**。

## 它不做什么

- **不声称"会对话 / 会推理"**：这里只跑合成算术的严格 EM。
- **不声称"普遍增益"**：五桶里只有 add_3d 显著（1/5）。
- **GSM8K 上 EM 仍 ≤ 地板**（未过 L5 免费规则地板；见 `MEME_FRAMEWORK.md` §5 L5 / R7）。
- **不训练、不用 GPU**：只读 ckpt 做 CPU 评估。

## 门禁与自检（各在哪看）

| 项 | 是什么 | 在哪看 |
|---|---|---|
| **门 A（梯度经卡）** | 离散中介必须让上游梯度真的回传（R15 / STE 前门）| `stages/08_ste_mediation.py`、`logs/08_ste_mediation.log` |
| **门 B（不得退化成恒等）** | 恒等中介也能解题 ⇒ 卡的能力从未被提升（"在路径上" ≠ "不能退化成恒等"）| `MEME_FRAMEWORK.md` §12 / §13.2 |
| **门 C（地板）** | 先与 `max_naive` 免费规则（含计数类）比 | R7；`MEME_FRAMEWORK.md` §5 L5 |
| **R28 批内一致性** | 同一 prompt 在 batch=1 与 batch=16 下生成须一致（否则该口径作废）| 运行输出的 `[R28]` 行 |
| **R29 非零性冒烟** | 干预必须真的改变被测量：报 Δ(res−nores) 非零比例 | 运行输出的 `[R29]` 行 |

**R28 实测**：batch=1（主口径）与 batch=16 **仅 1–3/16 逐字一致**（已知 `batch>1` 列索缺陷）
⇒ **EM 全部走 batch=1**，batch=16 口径作废；详见 R28 / R29（`methodology/RULES.md`）。

## 输出样例与证据

- 应看到的输出：`examples/expected_output.md`
- 证据清单（逐条指向已入库文件/日志 + ckpt sha256）：`examples/CHECKLIST.md`

## ckpt sha256 前 16

| ckpt | sha256[:16] |
|---|---|
| `logs/19_ckpt_add_1d_nores_seed1234.pt` | `59284de1ca2364a3` |
| `logs/19_ckpt_add_1d_res_seed1234.pt` | `6cd0000899dd97f3` |
| `logs/19_ckpt_add_3d_nores_seed1234.pt` | `7b0228af3f7238d0` |
| `logs/19_ckpt_add_3d_res_seed1234.pt` | `854903674f9b952e` |

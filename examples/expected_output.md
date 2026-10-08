# 应看到的输出样例（样例，数值以实跑为准）

命令：`bash examples/run_example.sh`（CPU，不训练）。关键行：

```text
[EX] repo=/home/vesita/coding/my/flowme
[EX] device=cpu（CUDA_VISIBLE_DEVICES 已清空；不训练）
[EX] 已复用 19_residual_scan.py 前缀：Cards / DATA / greedy_gen / hits_from / r28_selfcheck
[ARM] add_1d  nores EM= 96.62% (n=800) sha16=59284de1ca2364a3
[ARM] add_1d  res   EM= 96.88% (n=800) sha16=6cd0000899dd97f3
[R29] add_1d  Δ(res−nores) 非零比例 = 18/800 = 2.2%
[R28] [add_1d] 批内一致性 K=16: batch=1 vs batch=16 逐字一致 3/16 ⇒ ★不一致 ⇒ 仅 batch=1 口径可用（本单元 EM 全部走 batch=1）
[ARM] add_3d  nores EM=  2.50% (n=800) sha16=7b0228af3f7238d0
[ARM] add_3d  res   EM= 37.25% (n=800) sha16=854903674f9b952e
[R29] add_3d  Δ(res−nores) 非零比例 = 292/800 = 36.5%
[R28] [add_3d] 批内一致性 K=16: batch=1 vs batch=16 逐字一致 1/16 ⇒ ★不一致 ⇒ 仅 batch=1 口径可用（本单元 EM 全部走 batch=1）
[CONCLUSION] 残差： 一位数加 96.62% → 96.88% | 三位数加 2.50% → 37.25%。★只有三位数加（深推理/进位链）显著受益；一位数加两臂都在天花板 ~97% ⇒ 不是普遍增益。
[EX] OK
```

**一句结论**：残差把三位数加法从 2.5% 救到 37.25%；一位数加法两臂都 ~97%。

R28 的 1–3/16 是**已知仪器缺陷**（`batch>1` 列索错位），不是模型失败 —— 主口径 batch=1 自洽，
故所有 EM 都走 batch=1。add_1d 的 R29 非零比例低，是因为一位数加已在天花板（干预没能改变结果），
而 add_3d 的 36.5% 证明该干预是活的、不是空操作。n=800 与入库一致；CPU 重算与已入库 GPU 值逐位相同。

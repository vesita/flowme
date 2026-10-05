# 预注册：NDB 读口径 true vs pred（部署口径）

## 动机（来自 dev-notes/11「写入语义订正」）
NDB 的**写**分两种口径：训练期教师强制写真值（`runtime.py:143`），评估/推理只写模型自己发射的切片（`:356`）；
**读**在写之前，因此不构成泄漏。但**读用什么定位**还有第二个轴：
`--ndb-read true` 用真值起点检索，`--ndb-read pred` 用预测起点检索（部署口径）。
已测 `true` 口径：Δrepeat **+0.4510**（2 seed，dev-notes/12 §10.7）。**`pred` 口径未跑**，这条缝没被量化。

## 设置
`scripts/ab_test_ndb.py --seeds 42,43 --epochs 8 --steps-per-epoch 150 --batch-size 64 --num-layers 2 --ndb-read pred`
基座与其它超参**逐项同** dev-notes/12 §10.7 的 `true` 口径，唯一变量是 `--ndb-read`。

## 判据（跑前写死）
- **P1**：`pred` 口径的 Δrepeat（相对同批 base）**≥ +0.20** ⇒ 结论稳健，只保留文档订正；
- **P2**：若 **< +0.20** ⇒ 把 dev-notes/11 的 `0.375→0.843` 结论**降级**为
  「该增益依赖训练期教师强制写入 + 真值定位读取」，并在 dev-notes/12 §10.7 加注；
- 两 seed 同号才算；报两 seed 原值与均值。
- **代价**：与 `true` 口径的每步耗时/峰值显存对比。

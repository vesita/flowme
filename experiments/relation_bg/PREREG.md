# 预注册：A′ 臂 = 新（已修复）relation 数据集上的 4 卡对照

## 为什么需要这一臂
relation 的背景池语体修复**改变了 relation 的数据集**。于是「5 卡（含否定卡）里的 relation 掉没掉」
若直接与旧 A 臂（`checkpoints/multitask_v2_metrics.json`，relation 用的是**被代码语料污染**的旧数据集）比较，
是**被混淆的**：分不清差异来自「多了一张卡」还是「数据集换了」。
A′ 臂 = 与旧 A 臂**完全相同**的配置（pronoun/sentiment/relation/person、seed 42、16 epoch），
只是 relation 用修复后的数据集。这样：
- A′ vs 旧 A：数据修复的净效果；
- 5 卡 vs A′：**加卡的净效果**（这才是基座通用性的判据）。

## 判据（跑前写死）
- A′ 的 pronoun / sentiment / person 三卡（数据集未变）`exact_match` 与旧 A 臂之差应 ≤ 1.0pt
  （只受 RNG 轨道影响）；若 > 1.0pt，说明单 seed 噪声比预期大，加卡的 2.0pt 阈值需要一并打折解释。
- A′ 的 relation `exact_match` 允许与旧 A 不同（数据集变了），以 relation 卡自己的重训报告为准。
